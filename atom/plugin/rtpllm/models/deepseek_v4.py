"""ATOM DeepSeek-V4 model adapter for rtp-llm plugin mode (ROCm).

Architecture:
- Inherits RTP-LLM's DeepSeekV4 for config parsing + weight loading
  (platform-independent, handles TP via parallelism_config)
- Overrides _create_python_model to use ATOM's ROCm-compatible V4 model
  instead of RTP-LLM's CUDA-only DeepSeekV4Model
- This is how ATOM enables RTP-LLM to run V4 on AMD MI308X/MI355X

CUDA Graph Support:
- Pre-allocates persistent V4 decode metadata buffers (indptrs, indices, etc.)
- prepare_cuda_graph() builds metadata from live attn_inputs BEFORE replay
  (CPU numpy + H2D to stable addresses)
- Inside graph: write_v4_paged_decode_indices Triton kernel fills SWA ring
  indices using pre-allocated buffers (captured once, replayed every step)
- _patched_v4_forward detects graph mode and skips CPU-heavy
  _build_v4_per_forward_metadata entirely

Reference: Qwen3.5 plugin (atom/plugin/rtpllm/models/qwen3_5.py)
           SGLang V4 bridge (atom/plugin/sglang/deepseek_v4_bridge.py)
"""

import logging
import os
from typing import Any

import numpy as np
import torch
from rtp_llm.models.deepseek_v4 import DeepSeekV4, DeepSeekV4Mtp
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

from atom.model_loader.loader import WeightsMapper

logger = logging.getLogger("atom.plugin.rtpllm.models.deepseek_v4")


class _NoopWeightManager:
    def update(self, req):
        return None


class _NoopModelWeightsLoader:
    _py_eplb = None

    def load_lora_weights(self, adapter_name, lora_path, device):
        return None


class _ATOMAttnPyObj:
    """Container returned by _ATOMDeepSeekV4Runtime.prepare_fmha_impl.

    RTP CudaGraphRunner caches this object once at initCapture and calls
    .prepare_cuda_graph(attn_inputs) on it before each replay.

    For DeepSeek-V4, prepare_cuda_graph performs ALL CPU-side metadata
    computation (indptrs, state_slot_mapping, n_committed, etc.) and H2D
    copies to pre-allocated buffers. This runs OUTSIDE the captured graph.
    Inside the graph, write_v4_paged_decode_indices Triton kernel reads
    these buffers to produce paged indices — no CPU sync required.

    Also exposes a .fmha_params attribute for type-compat with downstream
    code that may peek at the attribute.
    """

    def __init__(self, runtime: "_ATOMDeepSeekV4Runtime") -> None:
        self._runtime = runtime
        self.is_cuda_graph = False

    @property
    def fmha_params(self):
        return None

    def prepare_cuda_graph(self, attn_inputs) -> None:
        """Build V4 decode metadata from live attn_inputs (OUTSIDE graph).

        Computes indptrs, state_slot_mapping, n_committed on CPU and H2D
        copies to pre-allocated GPU buffers. The captured graph's Triton
        kernels will read from these stable addresses at replay time.
        """
        rt = self._runtime
        bufs = getattr(rt, "_cg_v4_bufs", None)
        if bufs is None:
            return  # Not prewarmed yet; graph capture hasn't happened

        device = rt._model_device
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        if is_prefill:
            return  # CUDA graph only captures decode path

        # --- Extract batch info from live attn_inputs ---
        input_lengths = getattr(attn_inputs, "input_lengths", None)
        if input_lengths is None or input_lengths.numel() == 0:
            return
        bs = int(input_lengths.numel())
        max_bs = int(bufs["indptr_swa"].shape[0]) - 1
        if bs > max_bs:
            bs = max_bs

        # Positions: sequence_lengths for decode (absolute position of new token)
        seq_lens = getattr(attn_inputs, "sequence_lengths", None)
        if seq_lens is not None and seq_lens.numel() >= bs:
            positions_np = seq_lens[:bs].detach().cpu().numpy().astype(np.int32)
        else:
            seq_lens_p1 = getattr(attn_inputs, "sequence_lengths_plus_1_d", None)
            if seq_lens_p1 is not None and seq_lens_p1.numel() >= bs:
                positions_np = (seq_lens_p1[:bs].detach().cpu().numpy() - 1).astype(np.int32)
            else:
                positions_np = np.zeros(bs, dtype=np.int32)

        # Block tables (for state_slot_mapping = SWA block_table[:, 0])
        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
            SWA_KV, HCA_KV,
            build_region_to_group_map, select_block_table_for_region,
        )
        region_to_group = build_region_to_group_map(rt.kv_cache)
        swa_bt = select_block_table_for_region(attn_inputs, SWA_KV, region_to_group)
        hca_bt = select_block_table_for_region(attn_inputs, HCA_KV, region_to_group)

        # state_slot_mapping: use COMPACT indices [0..bs-1] for compressor state.
        # The actual block IDs (block_table[:, 0]) are stored separately for
        # swa_kv gather/scatter (remapping pool ↔ compact buffer).
        if swa_bt is not None and swa_bt.numel() >= bs:
            block_ids_np = swa_bt[:bs, 0].detach().cpu().numpy().astype(np.int32)
        else:
            block_ids_np = np.arange(bs, dtype=np.int32)
        # Compact slot mapping for compressor state: always [0, 1, ..., bs-1]
        ssm_np = np.arange(bs, dtype=np.int32)

        # batch_id_per_token: for decode, token t belongs to seq t
        batch_id_np = np.full(max_bs, -1, dtype=np.int32)
        batch_id_np[:bs] = np.arange(bs, dtype=np.int32)

        # n_committed
        win = int(bufs["_win"])
        index_topk = int(bufs["_index_topk"])
        n_csa_np = ((positions_np + 1) // 4).astype(np.int32)
        n_hca_np = ((positions_np + 1) // 128).astype(np.int32)

        # --- Compute indptrs (ragged cumsums) ---
        actual_swa = np.minimum(positions_np + 1, win).astype(np.int32)
        csa_valid_k = np.minimum(
            np.minimum((positions_np + 1) // 4, n_csa_np), index_topk
        ).astype(np.int32)
        hca_valid = n_hca_np.astype(np.int32)

        swa_indptr_np = np.zeros(max_bs + 1, dtype=np.int32)
        csa_indptr_np = np.zeros(max_bs + 1, dtype=np.int32)
        hca_indptr_np = np.zeros(max_bs + 1, dtype=np.int32)
        if bs > 0:
            swa_indptr_np[1:bs + 1] = np.cumsum(actual_swa, dtype=np.int32)
            csa_indptr_np[1:bs + 1] = np.cumsum(actual_swa + csa_valid_k, dtype=np.int32)
            hca_indptr_np[1:bs + 1] = np.cumsum(actual_swa + hca_valid, dtype=np.int32)
        # Pad tail with last value (sentinel: kv_len=0 for padded slots)
        if bs < max_bs:
            swa_indptr_np[bs + 1:] = swa_indptr_np[bs]
            csa_indptr_np[bs + 1:] = csa_indptr_np[bs]
            hca_indptr_np[bs + 1:] = hca_indptr_np[bs]

        # --- H2D copy to pre-allocated buffers ---
        bufs["positions"][:bs].copy_(
            torch.from_numpy(positions_np).to(dtype=torch.int64), non_blocking=True
        )
        bufs["state_slot"][:bs].copy_(
            torch.from_numpy(ssm_np).to(dtype=torch.int32), non_blocking=True
        )
        # Store actual block IDs for swa_kv gather/scatter in graph mode
        _block_ids_buf = bufs.get("_block_ids")
        if _block_ids_buf is None:
            _block_ids_buf = torch.zeros(int(bufs["state_slot"].shape[0]), device=device, dtype=torch.int64)
            bufs["_block_ids"] = _block_ids_buf
        _block_ids_buf[:bs].copy_(
            torch.from_numpy(block_ids_np).to(dtype=torch.int64), non_blocking=True
        )
        bufs["batch_id"][:max_bs].copy_(
            torch.from_numpy(batch_id_np).to(dtype=torch.int32), non_blocking=True
        )
        bufs["n_csa"][:bs].copy_(
            torch.from_numpy(n_csa_np).to(dtype=torch.int32), non_blocking=True
        )
        bufs["n_hca"][:bs].copy_(
            torch.from_numpy(n_hca_np).to(dtype=torch.int32), non_blocking=True
        )
        bufs["indptr_swa"][:max_bs + 1].copy_(
            torch.from_numpy(swa_indptr_np).to(dtype=torch.int32), non_blocking=True
        )
        bufs["indptr_csa"][:max_bs + 1].copy_(
            torch.from_numpy(csa_indptr_np).to(dtype=torch.int32), non_blocking=True
        )
        bufs["indptr_hca"][:max_bs + 1].copy_(
            torch.from_numpy(hca_indptr_np).to(dtype=torch.int32), non_blocking=True
        )

        # HCA block_tables for write_v4_decode_hca_compress_tail kernel
        if hca_bt is not None and hca_bt.numel() >= bs:
            bt_gpu = bufs["block_tables_hca"]
            cols = min(int(hca_bt.shape[1]), int(bt_gpu.shape[1]))
            bt_gpu[:bs, :cols].copy_(hca_bt[:bs, :cols].to(torch.int32), non_blocking=True)

        # --- Build compress_plans using CpuGpuBuffer plan buffers ---
        # make_compress_plans runs on CPU numpy + H2D via CpuGpuBuffer.copy_to_gpu().
        # The returned CompressPlan objects reference the stable GPU buffers,
        # so forward_impl inside the captured graph reads from stable addresses.
        plan_buffers = bufs.get("_plan_buffers")
        decode_cap = bufs.get("_decode_compress_cap")
        if plan_buffers is not None:
            from atom.model_ops.v4_kernels.compress_plan import make_compress_plans
            extend_lens_cpu = np.ones(bs, dtype=np.int32)
            context_lens_cpu = (positions_np + 1).astype(np.int32)
            compress_plans = make_compress_plans(
                extend_lens_cpu,
                context_lens_cpu,
                [(4, True), (128, False)],
                plan_buffers=plan_buffers,
                decode_capacity_per_ratio=decode_cap,
            )
            bufs["_compress_plans"] = compress_plans

        # Store state_slot_mapping_cpu (numpy) for compressor internals
        bufs["_state_slot_mapping_cpu"] = ssm_np[:bs].copy()

        # Stash active batch size for the graph-mode forward path
        bufs["_active_bs"] = bs

        # --- Detect new request: reset compressor state for compact slots ---
        # Compact slots are reused across requests. If block_ids change, it's a
        # new request — state[0..bs-1] has stale data from previous request.
        # Reset to initial values (zeros for kv_state, -inf for score_state).
        # This runs OUTSIDE the graph (in prepare_cuda_graph), so it's safe.
        _prev_bids = bufs.get("_prev_block_ids_hash", None)
        _curr_hash = int(block_ids_np.sum()) if bs > 0 else 0
        if _prev_bids is None or _prev_bids != _curr_hash:
            bufs["_prev_block_ids_hash"] = _curr_hash
            # Reset compressor state on all attention layers
            try:
                for module in rt.model.modules():
                    if hasattr(module, 'compressor') and module.compressor is not None:
                        comp = module.compressor
                        if hasattr(comp, 'kv_state') and comp.kv_state.shape[0] >= bs:
                            comp.kv_state[:bs].zero_()
                            comp.score_state[:bs].fill_(float('-inf'))
                    # Also reset indexer's inner compressor
                    if hasattr(module, 'indexer') and module.indexer is not None:
                        idx_comp = getattr(module.indexer, 'compressor', None)
                        if idx_comp is not None and hasattr(idx_comp, 'kv_state') and idx_comp.kv_state.shape[0] >= bs:
                            idx_comp.kv_state[:bs].zero_()
                            idx_comp.score_state[:bs].fill_(float('-inf'))
            except Exception as e:
                logger.debug("State reset failed (non-fatal): %s", e)


class _ATOMDeepSeekV4Runtime(GptModelBase):
    """Runtime adapter backed by ATOM V4 model on ROCm."""

    def __init__(
        self,
        model_config,
        parallelism_config,
        weights,
        max_generate_batch_size,
        atom_model,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.model = atom_model
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            raise RuntimeError("ATOM V4 model has no parameters")
        self._model_device = first_param.device
        self._model_dtype = first_param.dtype

        from atom.plugin.rtpllm.utils.forward_context import RTPForwardContext
        self._rtp_layer_maps = RTPForwardContext.collect_layer_maps(model=self.model)
        self._rtp_kv_cache_data = None
        self._rtp_kv_cache_signature = None
        self._rtp_layer_group_map = None
        self._rtp_layer_group_map_signature = None
        # CUDA graph support fields
        self._atom_attn_pyobj: _ATOMAttnPyObj | None = None
        self._cg_layers_prewarmed: bool = False
        decode_caps = getattr(py_hw_kernel_config, "decode_capture_batch_sizes", None)
        if decode_caps:
            self._cg_max_num_tokens: int = min(
                int(max(decode_caps)), int(max_generate_batch_size)
            )
        else:
            self._cg_max_num_tokens: int = int(max_generate_batch_size)
        self._cg_max_seq_len: int = int(
            getattr(model_config, "max_seq_len", 0)
            or getattr(model_config, "max_position_embeddings", 0)
            or 32768
        )

    def load_weights(self):
        return None

    def _get_model_device(self):
        return self._model_device

    def _get_model_dtype(self):
        return self._model_dtype

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        """Return ATOM-aware attention container for RTP CUDA graph hooks."""
        if self._atom_attn_pyobj is None:
            self._atom_attn_pyobj = _ATOMAttnPyObj(self)
        self._atom_attn_pyobj.is_cuda_graph = bool(is_cuda_graph)
        # Keep eager/non-graph path untouched: only prewarm when graph path
        # explicitly asks for fmha_impl in cuda-graph mode.
        if bool(is_cuda_graph):
            inputs.attention_inputs.is_cuda_graph = True
            self._ensure_cuda_graph_prewarmed()
        return self._atom_attn_pyobj

    def _ensure_cuda_graph_prewarmed(self) -> None:
        if self._cg_layers_prewarmed:
            return
        max_num_tokens = int(self._cg_max_num_tokens)
        max_seq_len = int(self._cg_max_seq_len)
        if max_num_tokens <= 0 or max_seq_len <= 0:
            logger.warning(
                "ATOM V4 cuda-graph prewarm skipped: invalid budget "
                "(max_num_tokens=%d, max_seq_len=%d)",
                max_num_tokens,
                max_seq_len,
            )
            return
        device = self._get_model_device()

        # Pre-allocate metadata tensors consumed by _build_plugin_attention_metadata
        # during decode capture. RTP captures via cudaStreamBeginCapture (not
        # torch.cuda.graph()), so any tensor allocated during capture lives in the
        # regular pool and may be freed + reused after capture ends, causing replay
        # faults. Pre-allocating here keeps GPU addresses stable.
        kv_cache = getattr(self, "kv_cache", None)
        kernel_seq_size_per_block = (
            int(getattr(kv_cache, "kernel_seq_size_per_block", 0))
            or int(getattr(kv_cache, "seq_size_per_block", 0))
            or 1
        )
        max_bs = max_num_tokens
        max_blocks = (
            int(max_seq_len) + kernel_seq_size_per_block - 1
        ) // kernel_seq_size_per_block + 1

        self._cg_meta_bufs: dict = {
            "query_start_loc": torch.arange(
                0, max_bs + 1, device=device, dtype=torch.int32
            ),
            "seq_id": torch.arange(0, max_bs, device=device, dtype=torch.int64),
            "seq_id_i32": torch.arange(0, max_bs, device=device, dtype=torch.int32),
            "block_col": torch.empty(max_bs, device=device, dtype=torch.int32),
            "block_col_i64": torch.empty(max_bs, device=device, dtype=torch.int64),
            "slot_base": torch.empty(max_bs, device=device, dtype=torch.int32),
            "token_offset": torch.empty(max_bs, device=device, dtype=torch.int32),
            "slot_mapping": torch.empty(max_bs, device=device, dtype=torch.int64),
            "seq_lens_i32": torch.empty(max_bs, device=device, dtype=torch.int32),
            "block_table_i32": torch.empty(
                max_bs, max_blocks, device=device, dtype=torch.int32
            ),
        }
        # Pre-allocated int64 positions buffer for model forward (RoPE kernel
        # requires int64) while bind() needs int32. Graph-safe via copy_().
        self._cg_positions_i64 = torch.empty(max_num_tokens, device=device, dtype=torch.int64)

        # --- V4-specific decode graph buffers ---
        # These persistent buffers hold V4 attention metadata (ragged indices,
        # indptrs, state_slot_mapping, etc.). prepare_cuda_graph() writes them
        # before each replay; write_v4_paged_decode_indices Triton kernel reads
        # them inside the captured graph.
        model = self.model
        args = getattr(model, "args", None) or getattr(
            getattr(model, "model", None), "args", None
        )
        win = int(getattr(args, "window_size", 128)) if args else 128
        index_topk = int(getattr(args, "index_topk", 1024)) if args else 1024
        # max_committed_hca = worst case per-seq HCA entries
        max_committed_hca = max(1, max_seq_len // 128)

        from atom.utils import CpuGpuBuffer

        self._cg_v4_bufs: dict = {
            # Per-token / per-seq metadata (decode: 1 token/seq → max_bs tokens)
            "positions": torch.zeros(max_bs, device=device, dtype=torch.int64),
            "state_slot": torch.zeros(max_bs, device=device, dtype=torch.int32),
            "batch_id": torch.full((max_bs,), -1, device=device, dtype=torch.int32),
            "n_csa": torch.zeros(max_bs, device=device, dtype=torch.int32),
            "n_hca": torch.zeros(max_bs, device=device, dtype=torch.int32),
            # Ragged indptrs [max_bs + 1]
            "indptr_swa": torch.zeros(max_bs + 1, device=device, dtype=torch.int32),
            "indptr_csa": torch.zeros(max_bs + 1, device=device, dtype=torch.int32),
            "indptr_hca": torch.zeros(max_bs + 1, device=device, dtype=torch.int32),
            # Ragged index buffers (worst-case sizes)
            "idx_swa": torch.zeros(max_bs * win, device=device, dtype=torch.int32),
            "idx_csa": torch.zeros(
                max_bs * (win + index_topk), device=device, dtype=torch.int32
            ),
            "idx_hca": torch.zeros(
                max_bs * (win + max_committed_hca), device=device, dtype=torch.int32
            ),
            # HCA block_tables for write_v4_decode_hca_compress_tail
            "block_tables_hca": torch.zeros(
                max_bs, max_blocks, device=device, dtype=torch.int32
            ),
            # CpuGpuBuffer plan buffers for make_compress_plans (CUDA Graph path)
            # compress: at most bs compression boundaries per decode step
            # write: at most bs * K tokens in write window
            "_plan_buffers": {
                4: {
                    "compress": CpuGpuBuffer(max(1, max_bs), 4, dtype=torch.int32, device=device),
                    "write": CpuGpuBuffer(max(1, max_bs * 8), 4, dtype=torch.int32, device=device),
                },
                128: {
                    "compress": CpuGpuBuffer(max(1, max_bs), 4, dtype=torch.int32, device=device),
                    "write": CpuGpuBuffer(max(1, max_bs * 128), 4, dtype=torch.int32, device=device),
                },
            },
            "_decode_compress_cap": {4: max(1, max_bs), 128: max(1, max_bs)},
            # Config constants (stored for prepare_cuda_graph to read)
            "_win": win,
            "_index_topk": index_topk,
            "_max_committed_hca": max_committed_hca,
            "_active_bs": 0,
            # References for graph-capture fallback pool binding
            "_kv_cache_ref": kv_cache,
            "_kv_cache_ref_runtime": self,
        }

        # Initialize _compress_plans with valid empty plans so graph capture
        # doesn't KeyError even if prepare_cuda_graph returns early (warmup).
        from atom.model_ops.v4_kernels.compress_plan import make_compress_plans
        empty_extend = np.zeros(1, dtype=np.int32)
        empty_context = np.zeros(1, dtype=np.int32)
        self._cg_v4_bufs["_compress_plans"] = make_compress_plans(
            empty_extend,
            empty_context,
            [(4, True), (128, False)],
            plan_buffers=self._cg_v4_bufs["_plan_buffers"],
            decode_capacity_per_ratio=self._cg_v4_bufs["_decode_compress_cap"],
        )
        self._cg_v4_bufs["_state_slot_mapping_cpu"] = np.zeros(1, dtype=np.int32)

        # Pre-allocate the graph-mode cat buffer (monkey-patch working buffer).
        # Must be large enough for max(swa + csa_compress, swa + hca_compress)
        # so it never reallocates during capture (which would invalidate earlier
        # layers' captured addresses).
        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
            SWA_KV, CSA_KV, HCA_KV, get_pool_for_layer_region,
        )
        try:
            swa_pool = get_pool_for_layer_region(kv_cache, 0, SWA_KV)
            # Layer 0 is often dense (ratio=0) and has NO CSA/HCA pool.
            # Find actual CSA/HCA layers to query their pool sizes.
            compress_ratios = getattr(self, "_compress_ratios", None)
            if compress_ratios is None:
                model = self.model
                _args = getattr(model, "args", None) or getattr(
                    getattr(model, "model", None), "args", None
                )
                compress_ratios = list(getattr(_args, "compress_ratios", ())) if _args else []
            csa_layer_id = next((i for i, r in enumerate(compress_ratios) if r == 4), None)
            hca_layer_id = next((i for i, r in enumerate(compress_ratios) if r == 128), None)
            csa_pool = get_pool_for_layer_region(kv_cache, csa_layer_id, CSA_KV) if csa_layer_id is not None else None
            hca_pool = get_pool_for_layer_region(kv_cache, hca_layer_id, HCA_KV) if hca_layer_id is not None else None
            head_dim = int(getattr(args, "v_head_dim", 512)) if args else 512
            swa_pages = int(swa_pool.kv_cache_base.shape[0]) * win if swa_pool else 0
            csa_compress = (int(csa_pool.kv_cache_base.view(torch.bfloat16).numel()) // head_dim) if csa_pool else 0
            hca_compress = (int(hca_pool.kv_cache_base.view(torch.bfloat16).numel()) // head_dim) if hca_pool else 0
            max_unified = swa_pages + max(csa_compress, hca_compress)
            if max_unified > 0:
                import atom.plugin.rtpllm.attention_backend.rtp_v4_attention as _v4_attn
                _v4_attn._graph_cat_buf = torch.empty(
                    max_unified, head_dim, dtype=torch.bfloat16, device=device,
                )
                # Also initialize the module-level pool view caches so the
                # graph-capture fallback path can find them (there is NO eager
                # forward before graph capture in RTP-LLM).
                if swa_pool is not None:
                    _swa_raw = swa_pool.kv_cache_base
                    _v4_attn._SWA_FLAT_CACHE = _swa_raw.view(torch.bfloat16).reshape(-1, head_dim)
                if csa_pool is not None:
                    _v4_attn._CSA_COMPRESS_KV_CACHE = csa_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, head_dim)
                if hca_pool is not None:
                    _v4_attn._HCA_COMPRESS_KV_CACHE = hca_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, head_dim)
                logger.info("Pre-allocated graph cat buffer: [%d, %d] (swa=%d, csa_c=%d, hca_c=%d)",
                            max_unified, head_dim, swa_pages, csa_compress, hca_compress)
        except Exception as e:
            logger.warning("Failed to pre-allocate graph cat buffer: %s", e)

        self._cg_layers_prewarmed = True

        # Pre-build kv_cache_data so graph capture can bind all pool views.
        # Normally built lazily during first eager forward, but graph capture
        # happens BEFORE any eager call. Without this, fc.kv_cache_data=None
        # during capture → _bind_v4_kv_cache_views never runs → crash.
        if self._rtp_kv_cache_data is None:
            try:
                from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
                    build_v4_kv_cache_tensors, get_v4_compress_ratios,
                )
                _ratios = get_v4_compress_ratios(self)
                if _ratios:
                    self._rtp_kv_cache_data = build_v4_kv_cache_tensors(self, _ratios)
                    logger.info("Pre-built kv_cache_data for graph capture (%d layers)", len(_ratios))
            except Exception as e:
                logger.warning("Failed to pre-build kv_cache_data: %s", e)

        logger.info(
            "ATOM V4 cuda-graph prewarmed "
            "(max_num_tokens=%d, max_seq_len=%d, "
            "meta_bufs: query_start_loc[%d], slot_mapping[%d], block_table_i32[%dx%d], "
            "v4_bufs: idx_swa[%d], idx_csa[%d], idx_hca[%d])",
            max_num_tokens,
            max_seq_len,
            max_bs + 1,
            max_bs,
            max_bs,
            max_blocks,
            max_bs * win,
            max_bs * (win + index_topk),
            max_bs * (win + max_committed_hca),
        )

        # Warmup eager forward: force aiter to pre-allocate all MoE workspace
        # buffers BEFORE graph capture. Without this, aiter allocates workspace
        # during capture (via regular PyTorch allocator); the memory may be freed
        # after capture ends and reused, causing graph replay to hit stale addresses.
        try:
            dummy_bs = min(max_bs, 4)
            dummy_ids = torch.zeros(dummy_bs, dtype=torch.int64, device=device)
            dummy_pos = torch.zeros(dummy_bs, dtype=torch.int64, device=device)
            with torch.no_grad():
                self.model(input_ids=dummy_ids, positions=dummy_pos)
            logger.info("ATOM V4 warmup eager forward done (pre-allocate MoE workspace)")
        except Exception as e:
            logger.warning("ATOM V4 warmup eager forward failed (non-fatal): %s", e)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        try:
            return self._forward_impl(inputs, fmha_impl)
        except Exception as e:
            logger.error("ATOM V4 forward FATAL: %s", e, exc_info=True)
            raise

    def _forward_impl(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        model_device = self._model_device
        is_cuda_graph = bool(getattr(fmha_impl, "is_cuda_graph", False))

        input_ids = getattr(inputs, "input_ids", None)
        if input_ids is not None and input_ids.numel() > 0:
            input_ids = input_ids.to(device=model_device, non_blocking=True)

        attn_inputs = getattr(inputs, "attention_inputs", None)
        positions = getattr(attn_inputs, "position_ids", None) if attn_inputs else None
        if is_cuda_graph:
            inputs.attention_inputs.is_cuda_graph = True
        if positions is not None:
            positions = positions.to(device=model_device, dtype=torch.int32, non_blocking=True).contiguous()
        else:
            is_prefill = bool(getattr(attn_inputs, "is_prefill", True)) if attn_inputs else True
            if not is_prefill and attn_inputs is not None:
                # Decode: position = sequence_lengths (absolute position of new token)
                seq_lens = getattr(attn_inputs, "sequence_lengths", None)
                if seq_lens is not None and seq_lens.numel() > 0:
                    positions = seq_lens.to(device=model_device, dtype=torch.int32, non_blocking=True).contiguous()
                else:
                    num_tokens = input_ids.numel() if input_ids is not None else 1
                    positions = torch.zeros(num_tokens, dtype=torch.int32, device=model_device)
            else:
                # Prefill: construct sequential positions [0, 1, 2, ...]
                num_tokens = input_ids.numel() if input_ids is not None else 1
                positions = torch.arange(num_tokens, dtype=torch.int32, device=model_device)

        # Build int64 positions for model forward (RoPE kernel requires int64).
        # bind() needs int32 (slot_mapping). Graph mode uses pre-allocated buffer.
        if is_cuda_graph:
            pos_i64 = self._cg_positions_i64[:positions.shape[0]]
            pos_i64.copy_(positions)  # int32→int64 in-place, graph-safe
        else:
            pos_i64 = positions.to(dtype=torch.int64)

        from atom.plugin.rtpllm.utils.forward_context import RTPForwardContext

        with RTPForwardContext.bind(
            model=self.model,
            runtime=self,
            inputs=inputs,
            positions=positions,
            layer_maps=self._rtp_layer_maps,
            cg_max_seq_len=int(self._cg_max_seq_len),
            cg_bufs=getattr(self, "_cg_meta_bufs", None),
        ):
            # In CUDA Graph mode, run V4 index construction Triton kernels
            # INSIDE the captured graph block. These read from pre-allocated
            # buffers (updated by prepare_cuda_graph before replay).
            if is_cuda_graph:
                self._run_v4_graph_index_kernels()
            hidden_states_hc = self.model(input_ids=input_ids, positions=pos_i64)

        hidden_states = self.model.model.head.hc_head(
            hidden_states_hc,
            self.model.model.hc_head_fn,
            self.model.model.hc_head_scale,
            self.model.model.hc_head_base,
        )
        hidden_states = self.model.model.norm(hidden_states)
        return PyModelOutputs(hidden_states)

    def _run_v4_graph_index_kernels(self) -> None:
        """Run V4 index Triton kernels inside the captured graph.

        These kernels read from pre-allocated buffers (positions, state_slot,
        batch_id, indptrs) that prepare_cuda_graph() refreshes before each replay.
        Outputs (idx_swa, idx_csa, idx_hca) are also in pre-allocated buffers
        with stable addresses.

        Called from forward() INSIDE the graph capture block.
        """
        bufs = getattr(self, "_cg_v4_bufs", None)
        if bufs is None:
            return

        from atom.model_ops.v4_kernels import write_v4_paged_decode_indices
        from atom.plugin.vllm.deepseek_v4_ops import write_v4_decode_hca_compress_tail
        from atom.utils.forward_context import get_forward_context

        win = int(bufs["_win"])
        max_bs = int(bufs["indptr_swa"].shape[0]) - 1
        # Use max_bs as grid size (captured grid is fixed; sentinel batch_id=-1
        # causes the kernel to bail for inactive tokens).
        T = max_bs
        cs = win  # win_with_spec = window_size (no MTP spec steps in plugin mode)
        swa_pages_val = max_bs * cs  # upper bound; actual set via prepare_cuda_graph

        # Compute swa_pages from state_slot max + cs (approximate upper bound)
        # The exact value comes from the pool structure. For the captured graph,
        # use the runtime's kv_cache metadata.
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None:
            from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
                SWA_KV, get_pool_for_layer_region,
            )
            swa_pool = get_pool_for_layer_region(kv_cache, 0, SWA_KV)
            if swa_pool is not None:
                swa_num_blocks = int(swa_pool.kv_cache_base.shape[0])
                swa_pages_val = swa_num_blocks * win

        write_v4_paged_decode_indices(
            state_slot_per_seq=bufs["state_slot"],
            batch_id_per_token=bufs["batch_id"],
            positions=bufs["positions"],
            swa_indptr=bufs["indptr_swa"],
            csa_indptr=bufs["indptr_csa"],
            hca_indptr=bufs["indptr_hca"],
            swa_indices=bufs["idx_swa"],
            csa_indices=bufs["idx_csa"],
            hca_indices=bufs["idx_hca"],
            T=T,
            win=win,
            cs=cs,
        )

        write_v4_decode_hca_compress_tail(
            batch_id_per_token=bufs["batch_id"],
            positions=bufs["positions"],
            hca_indptr=bufs["indptr_hca"],
            n_committed_hca_per_seq=bufs["n_hca"],
            block_tables=bufs["block_tables_hca"],
            hca_indices=bufs["idx_hca"],
            T=T,
            win=win,
            swa_pages=swa_pages_val,
        )

        # Mark V4 metadata on forward_context so _patched_v4_forward skips
        # _build_v4_per_forward_metadata and reads from these buffers.
        fc = get_forward_context()
        attn_md = fc.attn_metadata
        attn_md._v4_cuda_graph_mode = True
        attn_md._v4_cg_bufs = bufs
        attn_md._v4_swa_pages = swa_pages_val


class ATOMDeepSeekV4(DeepSeekV4):
    """DeepSeek-V4 with ATOM ROCm backend.

    Inherits DeepSeekV4 for:
    - _create_config / _from_hf (config parsing, platform-independent)
    - Weight info declaration (get_weight_cls -> DeepSeekV4Weight)

    Overrides:
    - _create_python_model: uses ATOM's model instead of CUDA-only DeepSeekV4Model
    - load: external plugin mode with ATOM weight loading
    """

    @staticmethod
    def _is_external_plugin_mode():
        modules = os.getenv("RTP_LLM_EXTERNAL_MODEL_PACKAGES", "")
        return "atom.plugin.rtpllm.models" in modules

    def load(self, skip_python_model=False):
        if self._is_external_plugin_mode():
            self.device = self._get_device_str()
            self.weight = ModelWeights(
                num_layers=self.model_config.num_layers,
                device=self.device,
                dtype=self.model_config.compute_dtype,
            )
            self.model_weights_loader = _NoopModelWeightsLoader()
            self.py_eplb = self.model_weights_loader._py_eplb
            self.weight_manager = _NoopWeightManager()
            if skip_python_model:
                return
            self._create_python_model()
            logger.info("External plugin mode: ATOM V4 loading complete")
            return
        super().load(skip_python_model=skip_python_model)

    def _create_python_model(self):
        """Create ATOM V4 model for ROCm (instead of CUDA-only DeepSeekV4Model)."""
        from atom.model_loader.loader import load_model_in_plugin_mode
        from atom.plugin.prepare import prepare_model
        from atom.plugin.rtpllm.attention_backend import apply_attention_v4_rtpllm_patch

        target_device = torch.device(self.device if hasattr(self, "device") else "cuda")
        target_dtype = self.model_config.compute_dtype
        old_default_dtype = torch.get_default_dtype()
        try:
            old_default_device = torch.get_default_device()
        except Exception:
            old_default_device = None

        torch.set_default_device(target_device)
        if target_dtype in {torch.float16, torch.bfloat16, torch.float32}:
            torch.set_default_dtype(target_dtype)

        try:
            atom_model = prepare_model(config=self, engine="rtpllm")
            if atom_model is None:
                raise ValueError("ATOM failed to create V4 model")

            apply_attention_v4_rtpllm_patch()

            atom_model = atom_model.to(target_device)
            atom_config = getattr(atom_model, "atom_config", None)
            if atom_config is None:
                raise ValueError("Cannot get atom_config from V4 model")

            load_model_in_plugin_mode(
                model=atom_model,
                config=atom_config,
                prefix="model.",
                weights_mapper=WeightsMapper(
                    orig_to_new_prefix={
                        "embed.": "model.embed.",
                        "layers.": "model.layers.",
                        "norm.weight": "model.norm.weight",
                        "head.weight": "model.head.weight",
                        "hc_head_": "model.hc_head_",
                    }
                ),
            )

            self._inject_rtp_projection_weights(atom_model)

        finally:
            torch.set_default_dtype(old_default_dtype)
            if old_default_device is not None:
                torch.set_default_device(old_default_device)
            else:
                torch.set_default_device("cpu")

        self.py_model = _ATOMDeepSeekV4Runtime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info("Created ATOM DeepSeek-V4 runtime for ROCm")

    def _inject_rtp_projection_weights(self, atom_model):
        def _find(model, *names):
            for n in names:
                for pn, p in model.named_parameters(recurse=True):
                    if pn == n and p is not None:
                        return p
            return None

        lm = _find(atom_model, "model.head.weight", "head.weight")
        if lm is not None:
            self.weight.set_global_weight(W.lm_head, lm.detach())

        emb = _find(atom_model, "model.embed.weight", "embed.weight")
        if emb is not None:
            self.weight.set_global_weight(W.embedding, emb.detach())

        ln = _find(atom_model, "model.norm.weight", "norm.weight")
        if ln is not None:
            self.weight.set_global_weight(W.final_ln_gamma, ln.detach())


class ATOMDeepSeekV4Mtp(DeepSeekV4Mtp):
    """DeepSeek-V4 MTP draft model with ATOM ROCm backend."""

    def _create_python_model(self):
        logger.warning("ATOMDeepSeekV4Mtp: MTP not yet implemented")

