"""V4 attention adapter for rtp-llm plugin mode.

Monkey-patches DeepseekV4Attention.forward_impl to:
1. Bind RTP-LLM pool views to V4 attention module attributes
2. Construct V4-specific metadata (compress_plans, state_slot_mapping, etc.)
3. Delegate to original forward_impl with proper metadata
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
    SWA_KV,
    CSA_KV,
    HCA_KV,
    INDEXER_KV,
    CSA_STATE,
    HCA_STATE,
    INDEXER_STATE,
    _REGION_NAMES,
    select_block_table_for_region,
)

logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.rtp_v4_attention")

_PATCHED = False
_V4_META_BUILT_ATTR = "_rtp_v4_meta_built"
_V4_META_FAILED_ATTR = "_rtp_v4_meta_failed"
_V4_BUFFERS_ALLOCATED = "_rtp_v4_buffers_allocated"


def _ensure_v4_native_buffers(attn_module: Any, num_slots: int, device: torch.device) -> None:
    """Allocate ATOM-native KV buffers if not yet done.

    Resizes the 1-slot warmup placeholders to proper [num_slots, ...] shape.
    These shadow buffers are independent of RTP-LLM's paged pools — they let
    ATOM's V4 attention kernels (swa_write, compressor) work natively.
    """
    if getattr(attn_module, _V4_BUFFERS_ALLOCATED, False):
        return

    head_dim = attn_module.head_dim
    window_size = attn_module.window_size
    ratio = attn_module.compress_ratio

    # Resize swa_kv: [num_slots, window_size, head_dim]
    if attn_module.swa_kv.shape[0] < num_slots:
        attn_module.swa_kv = torch.zeros(
            num_slots, window_size, head_dim,
            dtype=torch.bfloat16, device=device,
        )

    # Resize compressor state buffers
    compressor = getattr(attn_module, "compressor", None)
    if compressor is not None and ratio > 0:
        overlap = 1 if ratio == 4 else 0
        coff = 1 + overlap
        state_dim0 = coff * ratio
        state_dim1 = coff * head_dim

        if compressor.kv_state.shape[0] < num_slots:
            compressor.kv_state = torch.zeros(
                num_slots, state_dim0, state_dim1,
                dtype=torch.float32, device=device,
            )
            compressor.score_state = torch.full(
                (num_slots, state_dim0, state_dim1),
                float("-inf"), dtype=torch.float32, device=device,
            )

        # compressor.kv_cache: will be bound from RTP-LLM pool in _bind step
        # (shadow fallback only if pool not available)
        if compressor.kv_cache is None:
            k_per_block = window_size // ratio
            compressor.kv_cache = torch.zeros(
                num_slots, k_per_block, head_dim,
                dtype=torch.bfloat16, device=device,
            )

    # Resize indexer state buffers (CSA layers only)
    indexer = getattr(attn_module, "indexer", None)
    if indexer is not None:
        idx_compressor = getattr(indexer, "compressor", None)
        idx_head_dim = getattr(indexer, "head_dim", head_dim)
        if idx_compressor is not None:
            csa_overlap = 1  # CSA always has overlap
            idx_coff = 1 + csa_overlap
            idx_state_dim0 = idx_coff * ratio
            idx_state_dim1 = idx_coff * idx_head_dim
            if idx_compressor.kv_state.shape[0] < num_slots:
                idx_compressor.kv_state = torch.zeros(
                    num_slots, idx_state_dim0, idx_state_dim1,
                    dtype=torch.float32, device=device,
                )
                idx_compressor.score_state = torch.full(
                    (num_slots, idx_state_dim0, idx_state_dim1),
                    float("-inf"), dtype=torch.float32, device=device,
                )
        if indexer.kv_cache is None:
            k_per_block = window_size // ratio
            aligned_dim = ((idx_head_dim + 4 + 15) // 16) * 16
            indexer.kv_cache = torch.zeros(
                num_slots, k_per_block, aligned_dim,
                dtype=torch.bfloat16, device=device,
            )

    # unified_kv: SWA-only view (compress region added via pool bind if available)
    swa_pages = num_slots * window_size
    attn_module.unified_kv = torch.zeros(
        swa_pages, head_dim,
        dtype=torch.bfloat16, device=device,
    )

    setattr(attn_module, _V4_BUFFERS_ALLOCATED, True)
    logger.debug("Allocated V4 native buffers for layer %d: swa_kv=[%d,%d,%d] ratio=%d",
                 attn_module.layer_id, num_slots, window_size, head_dim, ratio)


def _build_v4_per_forward_metadata(attn_md, attn_inputs, v4_ratios, v4_block_tables,
                                    region_to_group, device, window_size=128):
    """Construct V4-specific attention metadata from RTP-LLM inputs.

    Called once per forward (guarded by _V4_META_BUILT_ATTR flag on attn_md).
    Sets compress_plans, state_slot_mapping, state, cu_seqlens_q, etc.
    """
    from atom.utils.forward_context import AttnState

    is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
    raw_input_lengths = getattr(attn_inputs, "input_lengths", None)
    if raw_input_lengths is None or raw_input_lengths.numel() == 0:
        setattr(attn_md, _V4_META_BUILT_ATTR, True)
        setattr(attn_md, _V4_META_FAILED_ATTR, True)
        return
    bs = raw_input_lengths.shape[0]

    if is_prefill:
        # Prefill: input_lengths = number of new tokens per sequence
        input_lengths = raw_input_lengths
    else:
        # Decode: RTP-LLM sends total seq_len as input_lengths, not new token count.
        # New token count = 1 per sequence for decode.
        input_lengths = torch.ones(bs, dtype=torch.int32, device=device)

    total_check = int(input_lengths.sum().item())
    if bs == 0 or total_check == 0:
        setattr(attn_md, _V4_META_BUILT_ATTR, True)
        setattr(attn_md, _V4_META_FAILED_ATTR, True)
        return
    input_lens_cpu = input_lengths.cpu().numpy().astype(np.int32)

    # -- state (DECODE / PREFILL) --
    if is_prefill:
        attn_md.state = AttnState.PREFILL_NATIVE
    else:
        attn_md.state = AttnState.DECODE

    # -- state_slot_mapping from SWA_KV block table (fixed alloc, 1 block = 1 slot) --
    swa_bt = select_block_table_for_region(attn_inputs, SWA_KV, region_to_group)
    if swa_bt is not None and swa_bt.numel() >= bs:
        attn_md.state_slot_mapping = swa_bt[:bs, 0].to(dtype=torch.int32, device=device).contiguous()
        attn_md.state_slot_mapping_cpu = attn_md.state_slot_mapping.cpu().numpy().copy()
    else:
        attn_md.state_slot_mapping = torch.arange(bs, dtype=torch.int32, device=device)
        attn_md.state_slot_mapping_cpu = np.arange(bs, dtype=np.int32)

    # -- cu_seqlens_q (from corrected input_lengths, not raw attn_inputs.cu_seqlens) --
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    torch.cumsum(input_lengths.to(dtype=torch.int32, device=device), dim=0, out=cu[1:])
    attn_md.cu_seqlens_q = cu

    attn_md.max_seqlen_q = int(input_lens_cpu.max()) if bs > 0 else 1

    # -- batch_id_per_token --
    total_tokens = int(input_lens_cpu.sum())
    if total_tokens > 0:
        attn_md.batch_id_per_token = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32, device=device),
            input_lengths.to(dtype=torch.int32, device=device),
        )
    else:
        attn_md.batch_id_per_token = torch.zeros(1, dtype=torch.int32, device=device)

    # -- compress_plans --
    try:
        _build_compress_plans(attn_md, attn_inputs, v4_ratios, bs, device)
    except Exception as e:
        logger.warning("Failed to build compress_plans: %s — attention will use fallback", e)
        attn_md.compress_plans = {}

    # -- n_committed_csa_per_seq (how many CSA entries committed per seq) --
    seq_lens = getattr(attn_inputs, "sequence_lengths", None)
    if seq_lens is not None:
        seq_lens_cpu = seq_lens.cpu().numpy().astype(np.int32)
    else:
        prefix = getattr(attn_inputs, "prefix_lengths", None)
        prefix_cpu = prefix.cpu().numpy().astype(np.int32) if prefix is not None else np.zeros(bs, dtype=np.int32)
        seq_lens_cpu = (prefix_cpu + input_lens_cpu).astype(np.int32)

    unique_ratios = set(r for r in v4_ratios if r > 0)
    csa_ratio = 4  # V4 CSA always uses ratio=4
    for ratio_val in unique_ratios:
        committed = seq_lens_cpu // ratio_val
        key = f"n_committed_{_ratio_label(ratio_val)}_per_seq"
        setattr(attn_md, key, torch.from_numpy(committed.astype(np.int32)).to(device=device))
    attn_md.n_committed_csa_per_seq = torch.from_numpy(
        (seq_lens_cpu // csa_ratio).astype(np.int32)
    ).to(device=device) if csa_ratio in unique_ratios else torch.zeros(bs, dtype=torch.int32, device=device)

    # -- swa_pages (for unified_kv offset) --
    swa_bt = v4_block_tables.get(SWA_KV)
    attn_md.swa_pages = swa_bt.shape[1] if swa_bt is not None else 0

    # -- V4 sparse attention indices (ring buffer format) --
    # V4 ALWAYS uses sparse_attn_v4_paged_prefill/decode, never flash_attn.
    ssm = attn_md.state_slot_mapping
    win = window_size  # sliding window size from model config
    cs = win           # win_with_spec = window_size + max_spec_steps (0 when MTP off)
    total_tokens = int(input_lens_cpu.sum())

    # swa_pages: boundary in unified_kv between SWA [0,swa_pages) and compress [swa_pages,...)
    swa_bt = v4_block_tables.get(SWA_KV)
    num_swa_blocks = int(swa_bt.shape[0]) if swa_bt is not None else 6243
    attn_md.swa_pages = num_swa_blocks * cs

    # Get positions as numpy
    pos_tensor = getattr(attn_inputs, "position_ids", None)
    if pos_tensor is not None and pos_tensor.numel() > 0:
        positions_np = pos_tensor.cpu().numpy().astype(np.int32)
    elif not is_prefill:
        # Decode: position = sequence_lengths (absolute position of new token)
        seq_lens = getattr(attn_inputs, "sequence_lengths", None)
        if seq_lens is not None and seq_lens.numel() >= bs:
            positions_np = seq_lens.cpu().numpy().astype(np.int32)
        else:
            positions_np = np.zeros(bs, dtype=np.int32)
    else:
        # Prefill: sequential positions
        positions_np = np.concatenate([np.arange(l, dtype=np.int32) for l in input_lens_cpu])

    cu_q_cpu = np.zeros(bs + 1, dtype=np.int32)
    np.cumsum(input_lens_cpu, out=cu_q_cpu[1:])
    bid_per_tok = np.repeat(np.arange(bs, dtype=np.int32), input_lens_cpu)
    ssm_cpu = ssm.cpu().numpy().astype(np.int32)

    if not is_prefill:
        # === DECODE: per-token ring buffer SWA indices into unified_kv ===
        swa_indices = []
        swa_indptr = [0]
        for t in range(total_tokens):
            bid = int(bid_per_tok[t])
            slot = int(ssm_cpu[bid])
            pos = int(positions_np[t])
            n = min(pos + 1, win)
            for i in range(n):
                abs_pos = pos - n + 1 + i
                ring = abs_pos % cs
                paged = slot * cs + ring
                swa_indices.append(paged)
            swa_indptr.append(len(swa_indices))

        idx_t = torch.tensor(swa_indices, dtype=torch.int32, device=device) if swa_indices else torch.zeros(1, dtype=torch.int32, device=device)
        ptr_t = torch.tensor(swa_indptr, dtype=torch.int32, device=device)
        attn_md.kv_indices_swa = idx_t
        attn_md.kv_indptr_swa = ptr_t
        attn_md.kv_indices_csa = idx_t.clone()
        attn_md.kv_indptr_csa = ptr_t.clone()
        attn_md.kv_indices_hca = idx_t.clone()
        attn_md.kv_indptr_hca = ptr_t.clone()

    else:
        # === PREFILL_NATIVE: causal extend + empty prefix ===
        # Extend: causal mask indices into per-fwd kv tensor (row indices)
        ext_indices = []
        ext_indptr = [0]
        for t in range(total_tokens):
            bid = int(bid_per_tok[t])
            pos = int(positions_np[t])
            extend_count = min(pos + 1, win)
            cu_q = int(cu_q_cpu[bid])
            ext_start = cu_q + pos - extend_count + 1
            for k in range(extend_count):
                ext_indices.append(ext_start + k)
            ext_indptr.append(len(ext_indices))

        attn_md.kv_indices_extend = torch.tensor(ext_indices, dtype=torch.int32, device=device)
        attn_md.kv_indptr_extend = torch.tensor(ext_indptr, dtype=torch.int32, device=device)

        # Prefix: empty (PREFILL_NATIVE has no prior-chunk KV history)
        empty_idx = torch.zeros(0, dtype=torch.int32, device=device)
        empty_ptr = torch.zeros(total_tokens + 1, dtype=torch.int32, device=device)
        for name in ("swa", "csa", "hca"):
            setattr(attn_md, f"kv_indices_prefix_{name}", empty_idx)
            setattr(attn_md, f"kv_indptr_prefix_{name}", empty_ptr)

    attn_md.skip_prefix_len_csa = torch.zeros(total_tokens, dtype=torch.int32, device=device)

    # -- Indexer metadata (for CSA layers' topk selection) --
    n_committed = attn_md.n_committed_csa_per_seq  # [bs] int32
    n_committed_cpu = n_committed.cpu().numpy().astype(np.int32)
    attn_md.n_committed_csa_per_seq_cpu = n_committed_cpu

    if not is_prefill:
        # DECODE: only needs n_committed_per_seq_gpu
        attn_md.indexer_meta = {
            "n_committed_per_seq_gpu": n_committed,
        }
    else:
        # PREFILL: needs cu_committed, seq_base, visible_end, cu_ends
        cu_committed_cpu = np.concatenate([
            np.zeros(1, dtype=np.int32),
            np.cumsum(n_committed_cpu, dtype=np.int32),
        ])
        cu_committed_cpu[-1] = max(int(cu_committed_cpu[-1]), 1)
        total_committed = int(cu_committed_cpu[-1])

        cu_committed_gpu = torch.from_numpy(cu_committed_cpu).to(device=device)
        bid_per_tok = attn_md.batch_id_per_token[:total_tokens].long()

        seq_base = cu_committed_gpu[bid_per_tok].to(torch.int32)

        pos_gpu = torch.from_numpy(positions_np[:total_tokens]).to(device=device, dtype=torch.int64)
        # Guard: n_committed might be empty for warmup/edge cases
        if n_committed.numel() == 0:
            visible_end = torch.zeros(total_tokens, dtype=torch.int32, device=device)
        else:
            visible_end = torch.minimum(
                (pos_gpu + 1) // csa_ratio,
                n_committed[bid_per_tok].long(),
            ).to(torch.int32)
        cu_ends = seq_base + visible_end

        attn_md.indexer_meta = {
            "total_committed": total_committed,
            "cu_committed_gpu": cu_committed_gpu,
            "n_committed_per_seq_gpu": n_committed,
            "batch_id_per_token_gpu": bid_per_tok,
            "seq_base_per_token_gpu": seq_base,
            "cu_starts_gpu": seq_base,
            "cu_ends_gpu": cu_ends,
        }

    setattr(attn_md, _V4_META_BUILT_ATTR, True)


def _ratio_label(ratio):
    if ratio == 4:
        return "csa"
    elif ratio == 128:
        return "hca"
    return "swa"


def _build_compress_plans(attn_md, attn_inputs, v4_ratios, bs, device):
    """Build CompressPlan dict for each unique compress ratio."""
    from atom.model_ops.v4_kernels.compress_plan import make_compress_plans
    from atom.utils import CpuGpuBuffer

    is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
    input_lengths = attn_inputs.input_lengths
    input_lens_cpu = input_lengths.cpu().numpy().astype(np.int32)

    if is_prefill:
        extend_lens_cpu = input_lens_cpu.copy()
        prefix = getattr(attn_inputs, "prefix_lengths", None)
        prefix_cpu = prefix.cpu().numpy().astype(np.int32) if prefix is not None else np.zeros(bs, dtype=np.int32)
        context_lens_cpu = (prefix_cpu + input_lens_cpu).astype(np.int32)
    else:
        extend_lens_cpu = np.ones(bs, dtype=np.int32)
        seq_lens_p1 = getattr(attn_inputs, "sequence_lengths_plus_1_d", None)
        if seq_lens_p1 is not None:
            context_lens_cpu = seq_lens_p1.cpu().numpy().astype(np.int32)
        else:
            seq_lens = attn_inputs.sequence_lengths
            context_lens_cpu = (seq_lens.cpu().numpy() + 1).astype(np.int32)

    unique_ratios = sorted(set(r for r in v4_ratios if r > 0))
    unique_ratios_overlap = []
    for r in unique_ratios:
        is_overlap = (r == 4)
        unique_ratios_overlap.append((r, is_overlap))

    total = int(extend_lens_cpu.sum())
    plan_buffers = {}
    for ratio, is_overlap in unique_ratios_overlap:
        K = (2 if is_overlap else 1) * ratio
        max_compress = max(total // ratio + bs + 1, 1)
        max_write = max(min(total, bs * K) + 1, 1)
        plan_buffers[ratio] = {
            "compress": CpuGpuBuffer(max_compress, 4, dtype=torch.int32, device=device),
            "write": CpuGpuBuffer(max_write, 4, dtype=torch.int32, device=device),
        }

    attn_md.compress_plans = make_compress_plans(
        extend_lens_cpu,
        context_lens_cpu,
        unique_ratios_overlap,
        plan_buffers=plan_buffers,
        decode_capacity_per_ratio=None,
    )


def _bind_v4_kv_cache_views(
    attn_module: Any,
    layer_pools: Dict[str, Any],
) -> None:
    """Bind RTP-LLM pool tensors to V4 attention module attributes.

    RTP-LLM pool layout (per region, per layer):
      KV pools (uint8):  [num_blocks, entries_per_block * head_dim * 2]
      State pools (fp32): [num_blocks, entries_per_block * state_dim]

    ATOM expects:
      swa_kv:      [num_slots, window_size, head_dim] bf16
      unified_kv:  [swa_pages + compress_pages, head_dim] bf16
      compressor.kv_cache: [num_blocks, k_per_block, head_dim] bf16
    """
    head_dim = attn_module.head_dim
    ratio = attn_module.compress_ratio
    window_size = attn_module.window_size

    # SWA_KV: [B, 131072] uint8 → [B, 128, 512] bf16
    swa_pool = layer_pools.get("SWA_KV")
    if swa_pool is not None:
        raw = swa_pool.kv_cache_base  # [B, 128*512*2] uint8
        B = raw.shape[0]
        swa_bf16 = raw.view(torch.bfloat16)  # [B, 128*512]
        swa_kv = swa_bf16.reshape(B, window_size, head_dim)  # [B, 128, 512]
        attn_module.swa_kv = swa_kv

        # unified_kv = flat SWA + flat compress
        swa_flat = swa_bf16.reshape(-1, head_dim)  # [B*128, 512]
    else:
        device = next(attn_module.parameters()).device
        swa_flat = torch.zeros(1, head_dim, dtype=torch.bfloat16, device=device)

    # unified_kv: for PREFILL_NATIVE, only extend indices are used (into per-fwd kv),
    # unified_kv is NOT read (prefix is empty). So just set to swa_flat (zero-copy view).
    # For DECODE, unified_kv is read via kv_indices which point into SWA ring.
    # TODO: for decode with compressed entries, need to include compress region.
    attn_module.unified_kv = swa_flat


def _bind_v4_compressor_views(
    attn_module: Any,
    layer_pools: Dict[str, Any],
) -> None:
    """Bind compressor state + kv_cache from RTP-LLM pools."""
    ratio = attn_module.compress_ratio
    head_dim = attn_module.head_dim
    compressor = getattr(attn_module, "compressor", None)
    if compressor is None or ratio == 0:
        return

    # Compressor KV cache
    if ratio == 4:
        kv_pool = layer_pools.get("CSA_KV")
        state_pool = layer_pools.get("CSA_STATE")
        k_per_block = window_size // ratio  # entries per block
    elif ratio == 128:
        kv_pool = layer_pools.get("HCA_KV")
        state_pool = layer_pools.get("HCA_STATE")
        k_per_block = window_size // ratio
    else:
        return

    if kv_pool is not None:
        kv_raw = kv_pool.kv_cache_base.view(torch.bfloat16)
        compressor.kv_cache = kv_raw.reshape(-1, k_per_block, head_dim)

    # State: [B, entries * state_dim * 2] fp32 → split kv/score
    if state_pool is not None:
        state_base = state_pool.kv_cache_base  # already fp32
        B = state_base.shape[0]
        if ratio == 4:
            entries = (1 + (1 if ratio == 4 else 0)) * ratio  # state ring size
            state_dim = 2 * head_dim  # coff=2, 1024
        else:
            entries = (1 + (1 if ratio == 4 else 0)) * ratio  # state ring size
            state_dim = head_dim  # coff=1, 512
        # State is interleaved [kv, score] per entry. Full pool is huge (1.5GB for HCA).
        # Only extract slots used by current batch via state_slot_mapping.
        from atom.utils.forward_context import get_forward_context
        fc_inner = get_forward_context()
        ssm = getattr(fc_inner.attn_metadata, "state_slot_mapping", None)
        state_4d = state_base.reshape(B, entries, 2, state_dim)
        if ssm is not None and ssm.numel() > 0 and ssm.numel() < B:
            # Extract only needed slots (tiny copy)
            needed = state_4d[ssm.long()]  # [bs, entries, 2, state_dim]
            compressor.kv_state = needed[:, :, 0, :].contiguous()
            compressor.score_state = needed[:, :, 1, :].contiguous()
        else:
            # Fallback: full pool (may OOM for large pools)
            compressor.kv_state = state_4d[:, :, 0, :].contiguous()
            compressor.score_state = state_4d[:, :, 1, :].contiguous()


def _bind_v4_indexer_views(
    attn_module: Any,
    layer_pools: Dict[str, Any],
) -> None:
    """Bind indexer pools (CSA layers only, ratio=4)."""
    indexer = getattr(attn_module, "indexer", None)
    if indexer is None:
        return

    idx_head_dim = getattr(attn_module, "index_head_dim", 128)
    k1 = self.window_size // ratio  # entries per block

    kv_pool = layer_pools.get("INDEXER_KV")
    if kv_pool is not None:
        kv_raw = kv_pool.kv_cache_base.view(torch.bfloat16)  # [B, 32*128]
        indexer.kv_cache = kv_raw.reshape(-1, k1, idx_head_dim)

    state_pool = layer_pools.get("INDEXER_STATE")
    if state_pool is not None:
        state_base = state_pool.kv_cache_base  # fp32
        B = state_base.shape[0]
        entries = (1 + (1 if ratio == 4 else 0)) * ratio  # state ring size
        idx_state_dim = 2 * idx_head_dim  # 256
        idx_compressor = getattr(indexer, "compressor", None)
        if idx_compressor is not None:
            from atom.utils.forward_context import get_forward_context
            fc_inner = get_forward_context()
            ssm = getattr(fc_inner.attn_metadata, "state_slot_mapping", None)
            state_4d = state_base.reshape(B, entries, 2, idx_state_dim)
            if ssm is not None and ssm.numel() > 0 and ssm.numel() < B:
                needed = state_4d[ssm.long()]
                idx_compressor.kv_state = needed[:, :, 0, :].contiguous()
                idx_compressor.score_state = needed[:, :, 1, :].contiguous()
            else:
                idx_compressor.kv_state = state_4d[:, :, 0, :].contiguous()
                idx_compressor.score_state = state_4d[:, :, 1, :].contiguous()


def _patched_v4_forward(self, x, positions):
    """Patched forward for DeepseekV4Attention in rtp-llm plugin mode.

    Patches `forward` (not `forward_impl`) because `forward` delegates to a
    torch custom op (`v4_attention_with_output`) that captures a direct reference
    to `forward_impl`, bypassing Python method resolution.
    """
    from atom.utils.forward_context import get_forward_context

    fc = get_forward_context()
    attn_md = fc.attn_metadata
    v4_block_tables = getattr(attn_md, "v4_block_tables", None)

    if v4_block_tables is None:
        return _original_v4_forward(self, x, positions)

    # Build V4 metadata once per forward (first layer triggers)
    if getattr(attn_md, _V4_META_FAILED_ATTR, False):
        return torch.zeros_like(x)
    if not getattr(attn_md, _V4_META_BUILT_ATTR, False):
        try:
            rtp_attn_inputs = getattr(attn_md, "rtp_attn_inputs", None)
            if rtp_attn_inputs is None:
                rtp_attn_inputs = getattr(attn_md, "plugin_metadata", None)
                if hasattr(rtp_attn_inputs, "rtp_attn_inputs"):
                    rtp_attn_inputs = rtp_attn_inputs.rtp_attn_inputs
            v4_ratios = getattr(attn_md, "v4_compress_ratios", [])
            region_to_group = getattr(attn_md, "v4_region_to_group", {})
            _build_v4_per_forward_metadata(
                attn_md, rtp_attn_inputs, v4_ratios, v4_block_tables,
                region_to_group, x.device,
                window_size=self.window_size,
            )
        except Exception as e:
            logger.error("V4 metadata construction failed: %s — using zeros fallback", e, exc_info=True)
            setattr(attn_md, _V4_META_BUILT_ATTR, True)
            setattr(attn_md, _V4_META_FAILED_ATTR, True)
            return torch.zeros_like(x)

    # Determine layer type
    v4_ratios = getattr(attn_md, "v4_compress_ratios", [])
    ratio = v4_ratios[self.layer_id] if self.layer_id < len(v4_ratios) else 0

    # Determine layer type
    v4_ratios = getattr(attn_md, "v4_compress_ratios", [])
    ratio = v4_ratios[self.layer_id] if self.layer_id < len(v4_ratios) else 0

    # CSA layers: skip Indexer (csa_translate_pack needs unified_kv with compress region)
    if ratio == 4 and hasattr(self, "skip_topk"):
        self.skip_topk = True



    # 1. Ensure compressor/indexer have properly sized native buffers (shadow alloc, once)
    bs_for_alloc = getattr(attn_md, "state_slot_mapping", None)
    num_slots = max(int(bs_for_alloc.max()) + 1, 32) if bs_for_alloc is not None and bs_for_alloc.numel() > 0 else 32
    _ensure_v4_native_buffers(self, num_slots=num_slots, device=x.device)

    # 2. Bind KV cache from RTP-LLM pool (overrides shadow swa_kv/unified_kv + compressor.kv_cache)
    kv_cache_data = fc.kv_cache_data
    cache_entry = kv_cache_data.get(f"layer_{self.layer_id}") if kv_cache_data else None
    if cache_entry and isinstance(cache_entry.k_cache, dict) and cache_entry.k_cache:
        try:
            _bind_v4_kv_cache_views(self, cache_entry.k_cache)
            # Also bind compressor.kv_cache from RTP-LLM pool (correct size, avoids OOB)
            compressor = getattr(self, "compressor", None)
            if compressor is not None and ratio != 0:
                head_dim = self.head_dim
                if ratio == 4:
                    csa_pool = cache_entry.k_cache.get("CSA_KV")
                    if csa_pool is not None:
                        compressor.kv_cache = csa_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, 128 // ratio, head_dim)
                elif ratio == 128:
                    hca_pool = cache_entry.k_cache.get("HCA_KV")
                    if hca_pool is not None:
                        compressor.kv_cache = hca_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, 128 // ratio, head_dim)
        except Exception as e:
            if not getattr(attn_md, "_v4_bind_err", False):
                logger.error("V4 bind layer %d: %s", self.layer_id, e, exc_info=True)
                attn_md._v4_bind_err = True
            fc.context.is_dummy_run = True
            return self.forward_impl(x, positions)
    if ratio == 0:
        region_bt = v4_block_tables.get(SWA_KV)
    elif ratio == 4:
        region_bt = v4_block_tables.get(CSA_KV)
    else:
        region_bt = v4_block_tables.get(HCA_KV)
    if region_bt is not None:
        attn_md.block_tables = region_bt

    try:
        return self.forward_impl(x, positions)
    except Exception as e:
        if not getattr(attn_md, "_v4_fwd_err", False):
            logger.error("V4 fwd layer %d (ratio=%d): %s", self.layer_id, ratio, e, exc_info=True)
            attn_md._v4_fwd_err = True
        return torch.zeros_like(x)


_original_v4_forward = None


def apply_attention_v4_rtpllm_patch() -> None:
    """Monkey-patch DeepseekV4Attention.forward for rtp-llm plugin mode.

    Patches `forward` instead of `forward_impl` because `forward` delegates to
    `torch.ops.aiter.v4_attention_with_output` (a torch custom op) that captures
    a direct function reference to `forward_impl` via `static_forward_context`.
    Patching `forward_impl` on the class has no effect — the custom op's dispatch
    bypasses Python MRO.
    """
    global _PATCHED, _original_v4_forward
    if _PATCHED:
        return

    from atom.plugin.prepare import is_rtpllm

    if not is_rtpllm():
        return

    try:
        from atom.models.deepseek_v4 import DeepseekV4Attention
    except ImportError:
        logger.warning("Cannot import DeepseekV4Attention — V4 patch skipped")
        return

    _original_v4_forward = DeepseekV4Attention.forward
    DeepseekV4Attention.forward = _patched_v4_forward

    logger.info("Applied RTP-LLM V4 attention patch (forward level) for multi-region KV cache.")
    _PATCHED = True
