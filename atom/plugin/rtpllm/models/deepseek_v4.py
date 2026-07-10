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

import gzip
import logging
import os
import time
from typing import Any

import numpy as np
import torch
import torch.profiler as torch_profiler
from rtp_llm.models.deepseek_v4 import DeepSeekV4, DeepSeekV4Mtp
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

from atom.model_loader.loader import WeightsMapper
from atom.plugin.rtpllm.attention_backend.rtp_dsv4_spec import (
    DSV4_CSA_RATIO,
    DSV4_HCA_RATIO,
    DSV4_DEFAULT_WINDOW_SIZE,
    DSV4_DEFAULT_INDEX_TOPK,
    DSV4_COMPRESS_PLAN_WIDTH,
    DSV4_CSA_DECODE_WRITE_TOKENS,
    DSV4_HCA_DECODE_WRITE_TOKENS,
    DSV4_COMPRESS_RATIOS_WITH_OVERLAP,
)

logger = logging.getLogger("atom.plugin.rtpllm.models.deepseek_v4")

# ---------------------------------------------------------------------------
# Plugin-side torch.profiler (bypasses rtp-llm C++ StepWindowProfiler)
# Controlled by env vars:
#   ATOM_PLUGIN_PROFILE=1          — arm profiling (won't start until trigger)
#   ATOM_PLUGIN_PROFILE_DIR=<dir>  — output directory
# Trigger files (created by profile_plugin.sh):
#   touch $DIR/.start_profiling  → profiler starts on next forward with tokens
#   touch $DIR/.stop_profiling   → profiler stops and exports trace
# ---------------------------------------------------------------------------
_plugin_profiler = None
_plugin_profile_dir = None


def _start_plugin_profiler():
    global _plugin_profiler, _plugin_profile_dir
    _plugin_profile_dir = os.environ.get("ATOM_PLUGIN_PROFILE_DIR", "./plugin_traces")
    os.makedirs(_plugin_profile_dir, exist_ok=True)

    try:
        rank = torch.cuda.current_device()
    except Exception:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    def _on_trace_ready(prof):
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        ms = int((time.time() % 1) * 1000)
        gz_path = os.path.join(
            _plugin_profile_dir,
            f"plugin_rank{rank}_ts_{ts}_{ms:03d}.pt.trace.json.gz",
        )
        tmp_path = gz_path[:-3]  # .json without .gz
        try:
            t0 = time.monotonic()
            prof.export_chrome_trace(tmp_path)
            with open(tmp_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
                while True:
                    chunk = src.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            os.remove(tmp_path)
            sz = os.path.getsize(gz_path)
            logger.info(
                "Plugin profiler rank %d: trace exported to %s (%.1f MB, %.1fs)",
                rank,
                gz_path,
                sz / 1e6,
                time.monotonic() - t0,
            )
        except Exception:
            logger.exception("Plugin profiler rank %d: failed to export trace", rank)
            for p in (tmp_path, gz_path):
                if os.path.exists(p):
                    os.remove(p)

    _plugin_profiler = torch_profiler.profile(
        activities=[
            torch_profiler.ProfilerActivity.CPU,
            torch_profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        on_trace_ready=_on_trace_ready,
    )
    _plugin_profiler.__enter__()
    logger.info(
        "Plugin profiler rank %d started (trigger-based, dir=%s)",
        rank,
        _plugin_profile_dir,
    )


def _stop_plugin_profiler():
    global _plugin_profiler
    if _plugin_profiler is not None:
        try:
            _plugin_profiler.__exit__(None, None, None)
        except Exception:
            logger.exception("Plugin profiler stop failed")
        _plugin_profiler = None
        logger.info("Plugin profiler stopped and trace exported.")


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

    def _extract_graph_replay_metadata(self, attn_inputs, bufs):
        """Build CPU replay metadata and resolve live region block tables."""
        rt = self._runtime
        input_lengths = getattr(attn_inputs, "input_lengths", None)
        if input_lengths is None or input_lengths.numel() == 0:
            return None
        bs = int(input_lengths.numel())
        max_bs = int(bufs["indptr_swa"].shape[0]) - 1
        if bs > max_bs:
            raise ValueError(
                f"V4 graph replay batch size {bs} exceeds captured capacity {max_bs}"
            )

        seq_lens = getattr(attn_inputs, "sequence_lengths", None)
        if seq_lens is not None and seq_lens.numel() >= bs:
            positions_np = seq_lens[:bs].detach().cpu().numpy().astype(np.int32)
        else:
            seq_lens_p1 = getattr(attn_inputs, "sequence_lengths_plus_1_d", None)
            if seq_lens_p1 is None or seq_lens_p1.numel() < bs:
                raise ValueError("V4 graph replay requires sequence lengths")
            positions_np = (seq_lens_p1[:bs].detach().cpu().numpy() - 1).astype(
                np.int32
            )

        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
            SWA_KV,
            HCA_KV,
            build_region_to_group_map,
            select_block_table_for_region,
        )

        region_to_group = build_region_to_group_map(rt.kv_cache)
        swa_bt = select_block_table_for_region(attn_inputs, SWA_KV, region_to_group)
        hca_bt = select_block_table_for_region(attn_inputs, HCA_KV, region_to_group)
        if swa_bt is None or swa_bt.numel() < bs:
            raise ValueError("V4 graph replay requires an SWA_KV block table")

        block_ids_np = swa_bt[:bs, 0].detach().cpu().numpy().astype(np.int32)
        block_ids_np = np.maximum(block_ids_np, 0)
        state_slots_np = np.arange(bs, dtype=np.int32)
        batch_ids_np = np.full(max_bs, -1, dtype=np.int32)
        batch_ids_np[:bs] = np.arange(bs, dtype=np.int32)

        win = int(bufs["_win"])
        index_topk = int(bufs["_index_topk"])
        n_csa_np = ((positions_np + 1) // DSV4_CSA_RATIO).astype(np.int32)
        n_hca_np = ((positions_np + 1) // DSV4_HCA_RATIO).astype(np.int32)
        actual_swa = np.minimum(positions_np + 1, win).astype(np.int32)
        csa_valid_k = np.minimum(
            np.minimum((positions_np + 1) // DSV4_CSA_RATIO, n_csa_np),
            index_topk,
        ).astype(np.int32)

        indptr_swa = np.zeros(max_bs + 1, dtype=np.int32)
        indptr_csa = np.zeros(max_bs + 1, dtype=np.int32)
        indptr_hca = np.zeros(max_bs + 1, dtype=np.int32)
        indptr_swa[1 : bs + 1] = np.cumsum(actual_swa, dtype=np.int32)
        indptr_csa[1 : bs + 1] = np.cumsum(actual_swa + csa_valid_k, dtype=np.int32)
        indptr_hca[1 : bs + 1] = np.cumsum(actual_swa + n_hca_np, dtype=np.int32)
        if bs < max_bs:
            indptr_swa[bs + 1 :] = indptr_swa[bs]
            indptr_csa[bs + 1 :] = indptr_csa[bs]
            indptr_hca[bs + 1 :] = indptr_hca[bs]

        return {
            "bs": bs,
            "max_bs": max_bs,
            "positions": positions_np,
            "block_ids": block_ids_np,
            "state_slots": state_slots_np,
            "batch_ids": batch_ids_np,
            "n_csa": n_csa_np,
            "n_hca": n_hca_np,
            "indptr_swa": indptr_swa,
            "indptr_csa": indptr_csa,
            "indptr_hca": indptr_hca,
            "swa_bt": swa_bt,
            "hca_bt": hca_bt,
            "region_to_group": region_to_group,
            "win": win,
        }

    def _seed_graph_replay_kv_state(
        self,
        attn_inputs,
        bufs,
        positions_np,
        block_ids_np,
        block_ids_buf,
        swa_bt,
        region_to_group,
        bs,
        win,
    ) -> None:
        """Seed compact SWA/compressor state for new graph replay slots.

        Runs outside capture. Continuing requests retain their self-maintained
        compact state; only slots with discontinuous positions are reseeded.
        """
        rt = self._runtime
        current_bid_tuple = tuple(block_ids_np[:bs].tolist())
        previous_positions = bufs.get("_prev_seed_positions")
        current_positions = positions_np[:bs].tolist()
        if previous_positions is None:
            seed_mask = [True] * bs
        else:
            seed_mask = [
                (i >= len(previous_positions)) or (pos != previous_positions[i] + 1)
                for i, pos in enumerate(current_positions)
            ]
        bufs["_prev_seed_positions"] = current_positions
        if not any(seed_mask):
            return

        bufs["_prev_gather_block_ids"] = current_bid_tuple
        # Keep the device slice alive at the same point as before; graph forward
        # consumes the owning fixed buffer through bufs["_block_ids"].
        _ = block_ids_buf[:bs]
        swa_bt_cpu = swa_bt[:bs].detach().cpu().numpy().astype(np.int32)

        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
            CSA_STATE,
            HCA_STATE,
            select_block_table_for_region,
        )

        state_block_tables = {}
        for state_region in (CSA_STATE, HCA_STATE):
            state_bt = select_block_table_for_region(
                attn_inputs, state_region, region_to_group
            )
            if state_bt is not None:
                state_block_tables[state_region] = state_bt

        try:
            from atom.models.deepseek_v4 import DeepseekV4Attention

            kv_cache_data = getattr(rt, "_rtp_kv_cache_data", None)
            for module in rt.model.modules():
                if not isinstance(module, DeepseekV4Attention):
                    continue

                compact_swa = getattr(module, "_compact_swa_kv", None)
                pool_swa = getattr(module, "_rtp_pool_swa_kv", None)
                if compact_swa is not None and pool_swa is not None:
                    for slot in range(bs):
                        if not seed_mask[slot]:
                            continue
                        position = int(positions_np[slot])
                        ring_split = position % win
                        current_col = position // win
                        previous_col = current_col - 1 if current_col > 0 else 0
                        current_block = int(
                            swa_bt_cpu[slot, min(current_col, swa_bt_cpu.shape[1] - 1)]
                        )
                        current_block = max(current_block, 0)
                        if current_col > 0 and previous_col < swa_bt_cpu.shape[1]:
                            previous_block = int(
                                swa_bt_cpu[
                                    slot,
                                    min(previous_col, swa_bt_cpu.shape[1] - 1),
                                ]
                            )
                            previous_block = max(previous_block, 0)
                            if ring_split > 0 and current_block < pool_swa.shape[0]:
                                compact_swa[slot, :ring_split].copy_(
                                    pool_swa[current_block, :ring_split]
                                )
                            if previous_block < pool_swa.shape[0]:
                                compact_swa[slot, ring_split:].copy_(
                                    pool_swa[previous_block, ring_split:]
                                )
                        elif current_block < pool_swa.shape[0]:
                            compact_swa[slot].copy_(pool_swa[current_block])

                compressor = getattr(module, "compressor", None)
                ratio = getattr(module, "compress_ratio", 0)
                if compressor is None or ratio == 0 or kv_cache_data is None:
                    continue
                state_region = CSA_STATE if ratio == DSV4_CSA_RATIO else HCA_STATE
                state_bt = state_block_tables.get(state_region)
                layer_id = getattr(module, "layer_id", -1)
                cache_entry = kv_cache_data.get(f"layer_{layer_id}")
                pool_name = "CSA_STATE" if ratio == DSV4_CSA_RATIO else "HCA_STATE"
                state_pool = (
                    cache_entry.k_cache.get(pool_name)
                    if cache_entry and isinstance(cache_entry.k_cache, dict)
                    else None
                )
                if state_bt is None or state_pool is None:
                    continue
                pool_raw = state_pool.kv_cache_base.view(torch.float32)
                num_blocks = pool_raw.shape[0]
                elems = pool_raw.numel() // num_blocks
                pool_view = pool_raw.reshape(num_blocks, elems)
                state_block_ids = state_bt[:bs, 0].to(torch.int64)
                half = elems // 2
                ring = compressor.kv_state.shape[1]
                dim = compressor.kv_state.shape[2]
                if half != ring * dim:
                    continue
                gathered = pool_view[state_block_ids]
                kv_all = gathered[:, :half].reshape(bs, ring, dim)
                score_all = gathered[:, half:].reshape(bs, ring, dim)
                for slot in range(bs):
                    if seed_mask[slot]:
                        compressor.kv_state[slot] = kv_all[slot]
                        compressor.score_state[slot] = score_all[slot]
        except Exception as e:
            raise RuntimeError("V4 SWA/STATE seed failed before graph replay") from e

    def _copy_graph_replay_identity_buffers(self, bufs, replay, device):
        """Copy position/slot identity before graph replay state seeding."""
        bs = replay["bs"]
        bufs["positions"][:bs].copy_(
            torch.from_numpy(replay["positions"]).to(dtype=torch.int64),
            non_blocking=True,
        )
        bufs["state_slot"][:bs].copy_(
            torch.from_numpy(replay["state_slots"]).to(dtype=torch.int32),
            non_blocking=True,
        )
        block_ids_buf = bufs.get("_block_ids")
        if block_ids_buf is None:
            block_ids_buf = torch.zeros(
                int(bufs["state_slot"].shape[0]), device=device, dtype=torch.int64
            )
            bufs["_block_ids"] = block_ids_buf
        block_ids_buf[:bs].copy_(
            torch.from_numpy(replay["block_ids"]).to(dtype=torch.int64),
            non_blocking=True,
        )
        return block_ids_buf

    @staticmethod
    def _copy_graph_replay_attention_buffers(bufs, replay) -> None:
        """Copy ragged attention metadata after SWA/STATE seeding."""
        bs = replay["bs"]
        max_bs = replay["max_bs"]
        bufs["batch_id"][:max_bs].copy_(
            torch.from_numpy(replay["batch_ids"]).to(dtype=torch.int32),
            non_blocking=True,
        )
        for key in ("n_csa", "n_hca"):
            bufs[key][:bs].copy_(
                torch.from_numpy(replay[key]).to(dtype=torch.int32),
                non_blocking=True,
            )
        for key in ("indptr_swa", "indptr_csa", "indptr_hca"):
            bufs[key][: max_bs + 1].copy_(
                torch.from_numpy(replay[key]).to(dtype=torch.int32),
                non_blocking=True,
            )

        hca_bt = replay["hca_bt"]
        if hca_bt is not None and hca_bt.numel() >= bs:
            bt_gpu = bufs["block_tables_hca"]
            cols = min(int(hca_bt.shape[1]), int(bt_gpu.shape[1]))
            bt_gpu[:bs, :cols].copy_(
                hca_bt[:bs, :cols].to(torch.int32), non_blocking=True
            )

    @staticmethod
    def _update_graph_replay_compress_plans(bufs, replay) -> None:
        """Refresh graph-stable compression plans for the live decode step."""
        plan_buffers = bufs.get("_plan_buffers")
        if plan_buffers is None:
            return
        from atom.model_ops.v4_kernels.compress_plan import make_compress_plans

        bs = replay["bs"]
        extend_lens_cpu = np.ones(bs, dtype=np.int32)
        context_lens_cpu = (replay["positions"] + 1).astype(np.int32)
        bufs["_compress_plans"] = make_compress_plans(
            extend_lens_cpu,
            context_lens_cpu,
            DSV4_COMPRESS_RATIOS_WITH_OVERLAP,
            plan_buffers=plan_buffers,
            decode_capacity_per_ratio=bufs.get("_decode_compress_cap"),
        )

    def prepare_cuda_graph(self, attn_inputs) -> None:
        """Build V4 decode metadata from live attn_inputs (OUTSIDE graph).

        Computes indptrs, state_slot_mapping, n_committed on CPU and H2D
        copies to pre-allocated GPU buffers. The captured graph's Triton
        kernels will read from these stable addresses at replay time.
        """
        bufs = getattr(self._runtime, "_cg_v4_bufs", None)
        if bufs is None:
            return  # Not prewarmed yet; graph capture hasn't happened
        if bool(getattr(attn_inputs, "is_prefill", False)):
            return  # CUDA graph only captures decode path
        replay = self._extract_graph_replay_metadata(attn_inputs, bufs)
        if replay is None:
            return
        block_ids_buf = self._copy_graph_replay_identity_buffers(
            bufs, replay, self._runtime._model_device
        )
        self._seed_graph_replay_kv_state(
            attn_inputs,
            bufs,
            replay["positions"],
            replay["block_ids"],
            block_ids_buf,
            replay["swa_bt"],
            replay["region_to_group"],
            replay["bs"],
            replay["win"],
        )
        self._copy_graph_replay_attention_buffers(bufs, replay)
        self._update_graph_replay_compress_plans(bufs, replay)
        bufs["_state_slot_mapping_cpu"] = replay["state_slots"].copy()
        bufs["_active_bs"] = replay["bs"]


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
        self._cg_positions_i64 = torch.empty(
            max_num_tokens, device=device, dtype=torch.int64
        )

        # --- V4-specific decode graph buffers ---
        # These persistent buffers hold V4 attention metadata (ragged indices,
        # indptrs, state_slot_mapping, etc.). prepare_cuda_graph() writes them
        # before each replay; write_v4_paged_decode_indices Triton kernel reads
        # them inside the captured graph.
        model = self.model
        args = getattr(model, "args", None) or getattr(
            getattr(model, "model", None), "args", None
        )
        win = (
            int(getattr(args, "window_size", DSV4_DEFAULT_WINDOW_SIZE))
            if args
            else DSV4_DEFAULT_WINDOW_SIZE
        )
        index_topk = (
            int(getattr(args, "index_topk", DSV4_DEFAULT_INDEX_TOPK))
            if args
            else DSV4_DEFAULT_INDEX_TOPK
        )
        # max_committed_hca = worst case per-seq HCA entries
        max_committed_hca = max(1, max_seq_len // DSV4_HCA_RATIO)

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
                DSV4_CSA_RATIO: {
                    "compress": CpuGpuBuffer(
                        max(1, max_bs),
                        DSV4_COMPRESS_PLAN_WIDTH,
                        dtype=torch.int32,
                        device=device,
                    ),
                    "write": CpuGpuBuffer(
                        max(1, max_bs * DSV4_CSA_DECODE_WRITE_TOKENS),
                        DSV4_COMPRESS_PLAN_WIDTH,
                        dtype=torch.int32,
                        device=device,
                    ),
                },
                DSV4_HCA_RATIO: {
                    "compress": CpuGpuBuffer(
                        max(1, max_bs),
                        DSV4_COMPRESS_PLAN_WIDTH,
                        dtype=torch.int32,
                        device=device,
                    ),
                    "write": CpuGpuBuffer(
                        max(1, max_bs * DSV4_HCA_DECODE_WRITE_TOKENS),
                        DSV4_COMPRESS_PLAN_WIDTH,
                        dtype=torch.int32,
                        device=device,
                    ),
                },
            },
            "_decode_compress_cap": {
                DSV4_CSA_RATIO: max(1, max_bs),
                DSV4_HCA_RATIO: max(1, max_bs),
            },
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
            DSV4_COMPRESS_RATIOS_WITH_OVERLAP,
            plan_buffers=self._cg_v4_bufs["_plan_buffers"],
            decode_capacity_per_ratio=self._cg_v4_bufs["_decode_compress_cap"],
        )
        self._cg_v4_bufs["_state_slot_mapping_cpu"] = np.zeros(1, dtype=np.int32)

        # Initialize runtime-scoped pool views for graph-capture fallback. There
        # is no eager forward before capture, and process globals would retain
        # stale device pointers across model/KV-cache reloads.
        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
            SWA_KV,
            CSA_KV,
            HCA_KV,
            get_pool_for_layer_region,
        )

        try:
            swa_pool = get_pool_for_layer_region(kv_cache, 0, SWA_KV)
            compress_ratios = getattr(self, "_compress_ratios", None)
            if compress_ratios is None:
                model = self.model
                _args = getattr(model, "args", None) or getattr(
                    getattr(model, "model", None), "args", None
                )
                compress_ratios = (
                    list(getattr(_args, "compress_ratios", ())) if _args else []
                )
            csa_layer_id = next(
                (i for i, r in enumerate(compress_ratios) if r == DSV4_CSA_RATIO),
                None,
            )
            hca_layer_id = next(
                (i for i, r in enumerate(compress_ratios) if r == DSV4_HCA_RATIO),
                None,
            )
            csa_pool = (
                get_pool_for_layer_region(kv_cache, csa_layer_id, CSA_KV)
                if csa_layer_id is not None
                else None
            )
            hca_pool = (
                get_pool_for_layer_region(kv_cache, hca_layer_id, HCA_KV)
                if hca_layer_id is not None
                else None
            )
            head_dim = int(getattr(args, "v_head_dim", 512)) if args else 512

            pool_views = {}
            if swa_pool is not None:
                _swa_raw = swa_pool.kv_cache_base
                pool_views["swa"] = _swa_raw.view(torch.bfloat16).reshape(-1, head_dim)
            if csa_pool is not None:
                pool_views["csa"] = csa_pool.kv_cache_base.view(torch.bfloat16).reshape(
                    -1, head_dim
                )
            if hca_pool is not None:
                pool_views["hca"] = hca_pool.kv_cache_base.view(torch.bfloat16).reshape(
                    -1, head_dim
                )
            self._cg_v4_bufs["_pool_views"] = pool_views
            logger.info("Initialized pool view caches for graph capture fallback")
        except Exception as e:
            logger.warning("Failed to initialize pool view caches: %s", e)

        self._cg_layers_prewarmed = True

        # Pre-build kv_cache_data so graph capture can bind all pool views.
        # Normally built lazily during first eager forward, but graph capture
        # happens BEFORE any eager call. Without this, fc.kv_cache_data=None
        # during capture → _bind_v4_kv_cache_views never runs → crash.
        if self._rtp_kv_cache_data is None:
            try:
                from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
                    build_v4_kv_cache_tensors,
                    get_v4_compress_ratios,
                )

                _ratios = get_v4_compress_ratios(self)
                if _ratios:
                    self._rtp_kv_cache_data = build_v4_kv_cache_tensors(self, _ratios)
                    logger.info(
                        "Pre-built kv_cache_data for graph capture (%d layers)",
                        len(_ratios),
                    )
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
        #
        # DeepseekV4ForCausalLM.forward stashes input_ids on
        # forward_context.context for the V4 hash-MoE routing callback. That
        # requires forward_context.context to be a real Context — production
        # forwards get one via RTPForwardContext.bind(), but this standalone
        # warmup runs outside bind(), so we install a minimal dummy Context here
        # (and restore the previous one afterward). Without it the warmup crashes
        # at ForCausalLM.forward's `ctx.context.input_ids = ...` before any layer
        # runs, so no workspace is actually pre-allocated.
        from atom.utils.forward_context import Context, get_forward_context

        try:
            dummy_bs = min(max_bs, 4)
            dummy_ids = torch.zeros(dummy_bs, dtype=torch.int64, device=device)
            dummy_pos = torch.zeros(dummy_bs, dtype=torch.int64, device=device)
            _fctx = get_forward_context()
            _saved_context = _fctx.context
            _fctx.context = Context(
                positions=dummy_pos, input_ids=dummy_ids, is_dummy_run=True
            )
            try:
                with torch.no_grad():
                    self.model(input_ids=dummy_ids, positions=dummy_pos)
            finally:
                _fctx.context = _saved_context
            logger.info(
                "ATOM V4 warmup eager forward done (pre-allocate MoE workspace)"
            )
        except Exception as e:
            logger.warning("ATOM V4 warmup eager forward failed (non-fatal): %s", e)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        # Profiler trigger check (model-level, called once per forward)
        global _plugin_profiler, _plugin_profile_dir
        input_ids = getattr(inputs, "input_ids", None)
        if (
            os.environ.get("ATOM_PLUGIN_PROFILE") == "1"
            and input_ids is not None
            and input_ids.numel() > 0
        ):
            _profile_dir = _plugin_profile_dir or os.environ.get(
                "ATOM_PLUGIN_PROFILE_DIR", "./plugin_traces"
            )
            _start_trigger = os.path.join(_profile_dir, ".start_profiling")
            _stop_trigger = os.path.join(_profile_dir, ".stop_profiling")
            if _plugin_profiler is None and os.path.exists(_start_trigger):
                _start_plugin_profiler()
                try:
                    os.remove(_start_trigger)
                except OSError:
                    pass
            elif _plugin_profiler is not None and os.path.exists(_stop_trigger):
                _stop_plugin_profiler()
                try:
                    os.remove(_stop_trigger)
                except OSError:
                    pass
                os.environ["ATOM_PLUGIN_PROFILE"] = "done"

        try:
            return self._forward_impl(inputs, fmha_impl)
        except Exception as e:
            logger.error("ATOM V4 forward FATAL: %s", e, exc_info=True)
            raise

    def _forward_impl(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        model_device = self._model_device
        is_cuda_graph = bool(getattr(fmha_impl, "is_cuda_graph", False))

        input_ids = getattr(inputs, "input_ids", None)
        if input_ids is None or input_ids.numel() == 0:
            raise ValueError("ATOM V4 forward requires non-empty input_ids")
        input_ids = input_ids.to(device=model_device, non_blocking=True)

        attn_inputs = getattr(inputs, "attention_inputs", None)
        if attn_inputs is None:
            raise ValueError("ATOM V4 forward requires attention_inputs")
        positions = getattr(attn_inputs, "position_ids", None) if attn_inputs else None
        if is_cuda_graph:
            inputs.attention_inputs.is_cuda_graph = True
        if positions is not None:
            positions = positions.to(
                device=model_device, dtype=torch.int32, non_blocking=True
            ).contiguous()
        else:
            is_prefill = (
                bool(getattr(attn_inputs, "is_prefill", True)) if attn_inputs else True
            )
            if not is_prefill and attn_inputs is not None:
                # Decode: position = sequence_lengths (absolute position of new token)
                seq_lens = getattr(attn_inputs, "sequence_lengths", None)
                if seq_lens is not None and seq_lens.numel() > 0:
                    positions = seq_lens.to(
                        device=model_device, dtype=torch.int32, non_blocking=True
                    ).contiguous()
                else:
                    raise ValueError(
                        "ATOM V4 decode requires position_ids or sequence_lengths"
                    )
            else:
                # Prefill: construct per-sequence positions [0,..,L1-1, 0,..,L2-1, ...]
                # NOT cumulative [0,..,L1+L2-1] — SWA ring buffer needs per-seq positions.
                num_tokens = input_ids.numel() if input_ids is not None else 1
                _inp_lens = (
                    getattr(attn_inputs, "input_lengths", None) if attn_inputs else None
                )
                if _inp_lens is not None and _inp_lens.numel() > 1:
                    _lens_cpu = _inp_lens.cpu().tolist()
                    positions = torch.cat(
                        [
                            torch.arange(
                                int(seq_len), dtype=torch.int32, device=model_device
                            )
                            for seq_len in _lens_cpu
                        ]
                    )
                else:
                    positions = torch.arange(
                        num_tokens, dtype=torch.int32, device=model_device
                    )

        # Build int64 positions for model forward (RoPE kernel requires int64).
        # bind() needs int32 (slot_mapping). Graph mode uses pre-allocated buffer.
        if is_cuda_graph:
            v4_bufs = getattr(self, "_cg_v4_bufs", None)
            n_tokens = (
                input_ids.shape[0]
                if input_ids is not None and input_ids.numel() > 0
                else 1
            )
            if v4_bufs is not None and "positions" in v4_bufs:
                pos_i64 = self._cg_positions_i64[:n_tokens]
                pos_i64.copy_(v4_bufs["positions"][:n_tokens])
            else:
                pos_i64 = self._cg_positions_i64[:n_tokens]
                pos_i64.copy_(positions)
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
                SWA_KV,
                get_pool_for_layer_region,
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
