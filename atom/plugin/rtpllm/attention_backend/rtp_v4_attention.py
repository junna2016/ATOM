"""V4 attention adapter for rtp-llm plugin mode.

Monkey-patches DeepseekV4Attention.forward to:
1. Bind RTP-LLM pool views to V4 attention module attributes
2. Construct V4-specific metadata (compress_plans, state_slot_mapping, etc.)
3. Delegate to original forward_impl with proper metadata

Also monkey-patches sparse_attn_v4_paged_decode to support dual-ptr mode:
when compress_kv is set in forward context, the plugin's dual-ptr Triton kernel
reads SWA entries from swa_kv and compress entries from compress_kv, avoiding
any memory copy or new GPU allocation.
"""

import gzip
import logging
import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.profiler as torch_profiler
import triton
import triton.language as tl

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

LOG2E = math.log2(math.e)

# ---------------------------------------------------------------------------
# Dual-ptr paged decode kernel (plugin-only, does NOT modify ATOM native code)
# ---------------------------------------------------------------------------

@triton.jit
def _dual_ptr_paged_decode_fused_kernel(
    q_ptr,
    swa_kv_ptr,          # [swa_pages, D] bf16
    compress_kv_ptr,     # [compress_pages, D] bf16
    kv_indices_ptr,      # [total_indices] int32
    kv_indptr_ptr,       # [N+1] int32
    attn_sink_ptr,       # [H]
    out_ptr,             # [N, H, D]
    q_stride_t, q_stride_h, q_stride_d,
    swa_stride_n, swa_stride_d,
    compress_stride_n, compress_stride_d,
    out_stride_t, out_stride_h, out_stride_d,
    qk_scale,
    log2e,
    SWA_PAGES: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Dual-ptr fused decode: reads from swa_kv or compress_kv based on slot index."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr + t * q_stride_t + h_offs[:, None] * q_stride_h + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :], other=0.0,
    )

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    num_tiles = tl.cdiv(kv_len, BLOCK_K)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(0, num_tiles, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < kv_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=valid, other=0)

        is_swa = slot < SWA_PAGES
        swa_slot = slot
        compress_slot = tl.maximum(slot - SWA_PAGES, 0)

        swa_data = tl.load(
            swa_kv_ptr + swa_slot[:, None] * swa_stride_n + d_offs[None, :] * swa_stride_d,
            mask=valid[:, None] & d_mask[None, :] & is_swa[:, None], other=0.0,
        )
        compress_data = tl.load(
            compress_kv_ptr + compress_slot[:, None] * compress_stride_n + d_offs[None, :] * compress_stride_d,
            mask=valid[:, None] & d_mask[None, :] & (~is_swa)[:, None], other=0.0,
        )
        kv = swa_data + compress_data

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    sink_raw = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    sink = sink_raw * log2e
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha_kv[:, None]) / denom[:, None], 0.0)
    tl.store(
        out_ptr + t * out_stride_t + h_offs[:, None] * out_stride_h + d_offs[None, :] * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )


def _dual_ptr_paged_decode(
    q: torch.Tensor,
    swa_kv: torch.Tensor,
    compress_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    swa_pages: int,
) -> torch.Tensor:
    """Dual-ptr sparse decode: SWA and compress KV in separate tensors.

    kv_indices values < swa_pages → read from swa_kv
    kv_indices values >= swa_pages → read from compress_kv (offset by swa_pages)
    """
    T, H, D = q.shape
    out = torch.empty_like(q)

    block_h = triton.next_power_of_2(min(H, 64))
    block_h = max(block_h, 16)
    n_head_blocks = (H + block_h - 1) // block_h
    block_d = triton.next_power_of_2(D)
    block_k = 16

    qk_scale = float(softmax_scale) * LOG2E

    grid = (T, n_head_blocks)
    _dual_ptr_paged_decode_fused_kernel[grid](
        q,
        swa_kv,
        compress_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        swa_kv.stride(0), swa_kv.stride(1) if swa_kv.dim() > 1 else 1,
        compress_kv.stride(0), compress_kv.stride(1) if compress_kv.dim() > 1 else 1,
        out.stride(0), out.stride(1), out.stride(2),
        qk_scale,
        LOG2E,
        SWA_PAGES=swa_pages,
        H=H, D=D,
        BLOCK_H=block_h, BLOCK_D=block_d, BLOCK_K=block_k,
        num_warps=4, num_stages=2,
    )
    return out


# Saved references for monkey-patch
_original_paged_decode = None
_original_paged_prefill = None


# Pre-allocated working buffer for graph-mode cat (same as eager torch.cat output)
_graph_cat_buf = None

# Cached pool views for graph-mode fallback (populated during first eager bind
# or during _ensure_cuda_graph_prewarmed)
_SWA_FLAT_CACHE = None
_CSA_COMPRESS_KV_CACHE = None
_HCA_COMPRESS_KV_CACHE = None


def _patched_sparse_attn_v4_paged_decode(q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, kv_scales=None):
    """Monkey-patch wrapper: intercepts decode kernel calls in plugin mode.

    Uses the kernel's native dual-pointer (SPLIT_KV) mode to read from
    separate SWA and compress pools without any buffer copy or allocation.
    Falls back to torch.cat for eager non-graph mode (backward compat).
    """
    import torch as _t

    try:
        from atom.utils.forward_context import get_forward_context
        fc = get_forward_context()
        attn_md = getattr(fc, "attn_metadata", None)
        compress_kv = getattr(attn_md, "compress_kv", None) if attn_md else None
        swa_pages = getattr(attn_md, "swa_pages", 0) if attn_md else 0
    except Exception:
        compress_kv = None
        swa_pages = 0

    # --- Dual-pointer path (both graph and eager): zero-copy, zero-alloc ---
    if compress_kv is not None and compress_kv.numel() > 0 and swa_pages > 0:
        return _original_paged_decode(
            q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale,
            kv_scales=kv_scales,
            compress_kv=compress_kv,
            swa_pages=swa_pages,
        )

    # --- No compress needed (dense/SWA-only layers): pass through ---
    return _original_paged_decode(q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, kv_scales=kv_scales)

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
        # Already allocated. NEVER resize — buffer addresses must remain stable
        # for CUDA Graph replay. Prefill's large block_id indexing is handled
        # via STATE pool gather/scatter (not by growing these buffers).
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
    # Persist compact swa_kv reference (survives bind overrides).
    # IMPORTANT: Only set on FIRST allocation. If _compact_swa_kv already exists
    # (from graph warmup), do NOT overwrite — graph replay uses the original
    # address captured during graph capture. Overwriting with a resized tensor
    # would invalidate the captured address → precision errors.
    if not hasattr(attn_module, '_compact_swa_kv') or attn_module._compact_swa_kv is None:
        attn_module._compact_swa_kv = attn_module.swa_kv
    logger.debug("Allocated V4 native buffers for layer %d: swa_kv=[%d,%d,%d] ratio=%d",
                 attn_module.layer_id, num_slots, window_size, head_dim, ratio)


def _reset_v4_state_all(attn_module: Any) -> None:
    """Zero ALL V4 state caches for a new request (full cleanup).

    Called during prefill to prevent stale compressor/indexer state from
    a previous request from corrupting the softmax-pool computation.
    Zeros the entire buffer rather than selective slots because decode
    uses compact indices [0..bs-1] while prefill uses block_id indices —
    selective reset by block_id misses the stale compact-indexed state.
    """
    swa = getattr(attn_module, "swa_kv", None)
    if isinstance(swa, torch.Tensor):
        swa.zero_()
    compact_swa = getattr(attn_module, "_compact_swa_kv", None)
    if compact_swa is not None and compact_swa is not swa and isinstance(compact_swa, torch.Tensor):
        compact_swa.zero_()
    for compressor in (
        getattr(attn_module, "compressor", None),
        getattr(getattr(attn_module, "indexer", None), "compressor", None),
    ):
        if compressor is None:
            continue
        if isinstance(getattr(compressor, "kv_state", None), torch.Tensor):
            compressor.kv_state.zero_()
        if isinstance(getattr(compressor, "score_state", None), torch.Tensor):
            compressor.score_state.fill_(float("-inf"))


def _build_eager_decode_with_triton(attn_md, attn_inputs, v4_ratios, v4_block_tables,
                                     region_to_group, device, window_size=128,
                                     pool_swa_pages=0, index_topk=1024):
    """Build eager decode metadata using Triton kernels (same as CUDA Graph mode).

    This ensures eager decode uses EXACTLY the same index construction as graph
    mode, producing identical numerical results. Uses compact state_slot_mapping
    [0..bs-1], compact swa_kv buffer, and gather/scatter for pool sync.

    Does NOT modify any CUDA Graph state (_cg_v4_bufs etc.).
    Returns True on success, False if fallback to CPU metadata is needed.
    """
    from atom.model_ops.v4_kernels import write_v4_paged_decode_indices
    from atom.plugin.vllm.deepseek_v4_ops import write_v4_decode_hca_compress_tail
    from atom.model_ops.v4_kernels.compress_plan import make_compress_plans
    from atom.utils.forward_context import AttnState
    from atom.utils import CpuGpuBuffer
    from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import SWA_KV, HCA_KV

    input_lengths = getattr(attn_inputs, "input_lengths", None)
    if input_lengths is None or input_lengths.numel() == 0:
        return False
    bs = int(input_lengths.numel())
    win = window_size
    cs = win  # win_with_spec = window_size (no MTP spec steps in plugin mode)

    # --- Positions ---
    seq_lens = getattr(attn_inputs, "sequence_lengths", None)
    if seq_lens is not None and seq_lens.numel() >= bs:
        positions_np = seq_lens[:bs].detach().cpu().numpy().astype(np.int32)
    else:
        seq_lens_p1 = getattr(attn_inputs, "sequence_lengths_plus_1_d", None)
        if seq_lens_p1 is not None and seq_lens_p1.numel() >= bs:
            positions_np = (seq_lens_p1[:bs].detach().cpu().numpy() - 1).astype(np.int32)
        else:
            positions_np = np.zeros(bs, dtype=np.int32)

    # --- Block tables ---
    swa_bt = select_block_table_for_region(attn_inputs, SWA_KV, region_to_group)
    hca_bt = select_block_table_for_region(attn_inputs, HCA_KV, region_to_group)

    # --- Compact state_slot_mapping = [0..bs-1] ---
    ssm_np = np.arange(bs, dtype=np.int32)
    # Save original block IDs for gather/scatter
    if swa_bt is not None and swa_bt.numel() >= bs:
        block_ids_np = swa_bt[:bs, 0].detach().cpu().numpy().astype(np.int32)
        block_ids_np = np.maximum(block_ids_np, 0)  # guard -1
    else:
        block_ids_np = np.arange(bs, dtype=np.int32)

    # --- n_committed ---
    n_csa_np = ((positions_np + 1) // 4).astype(np.int32)
    n_hca_np = ((positions_np + 1) // 128).astype(np.int32)

    # --- Compute indptrs (ragged cumsums) ---
    actual_swa = np.minimum(positions_np + 1, win).astype(np.int32)
    csa_valid_k = np.minimum(
        np.minimum((positions_np + 1) // 4, n_csa_np), index_topk
    ).astype(np.int32)
    hca_valid = n_hca_np.astype(np.int32)

    swa_indptr_np = np.zeros(bs + 1, dtype=np.int32)
    csa_indptr_np = np.zeros(bs + 1, dtype=np.int32)
    hca_indptr_np = np.zeros(bs + 1, dtype=np.int32)
    if bs > 0:
        swa_indptr_np[1:bs + 1] = np.cumsum(actual_swa, dtype=np.int32)
        csa_indptr_np[1:bs + 1] = np.cumsum(actual_swa + csa_valid_k, dtype=np.int32)
        hca_indptr_np[1:bs + 1] = np.cumsum(actual_swa + hca_valid, dtype=np.int32)

    # --- Allocate GPU tensors ---
    positions_gpu = torch.from_numpy(positions_np).to(dtype=torch.int64, device=device)
    state_slot_gpu = torch.from_numpy(ssm_np).to(dtype=torch.int32, device=device)
    batch_id_gpu = torch.arange(bs, dtype=torch.int32, device=device)
    n_hca_gpu = torch.from_numpy(n_hca_np).to(dtype=torch.int32, device=device)

    indptr_swa_gpu = torch.from_numpy(swa_indptr_np).to(dtype=torch.int32, device=device)
    indptr_csa_gpu = torch.from_numpy(csa_indptr_np).to(dtype=torch.int32, device=device)
    indptr_hca_gpu = torch.from_numpy(hca_indptr_np).to(dtype=torch.int32, device=device)

    total_swa = int(swa_indptr_np[bs])
    total_csa = int(csa_indptr_np[bs])
    total_hca = int(hca_indptr_np[bs])
    idx_swa_gpu = torch.zeros(max(total_swa, 1), dtype=torch.int32, device=device)
    idx_csa_gpu = torch.zeros(max(total_csa, 1), dtype=torch.int32, device=device)
    idx_hca_gpu = torch.zeros(max(total_hca, 1), dtype=torch.int32, device=device)

    # --- Run Triton kernels (same as _run_v4_graph_index_kernels) ---
    T = bs  # decode: 1 token per sequence
    write_v4_paged_decode_indices(
        state_slot_per_seq=state_slot_gpu,
        batch_id_per_token=batch_id_gpu,
        positions=positions_gpu,
        swa_indptr=indptr_swa_gpu,
        csa_indptr=indptr_csa_gpu,
        hca_indptr=indptr_hca_gpu,
        swa_indices=idx_swa_gpu,
        csa_indices=idx_csa_gpu,
        hca_indices=idx_hca_gpu,
        T=T,
        win=win,
        cs=cs,
    )

    # HCA compress tail (swa_pages + block_tables[bid, j])
    swa_pages_val = pool_swa_pages if pool_swa_pages > 0 else bs * cs
    if hca_bt is not None and hca_bt.numel() >= bs:
        hca_bt_gpu = hca_bt[:bs].to(dtype=torch.int32, device=device)
        write_v4_decode_hca_compress_tail(
            batch_id_per_token=batch_id_gpu,
            positions=positions_gpu,
            hca_indptr=indptr_hca_gpu,
            n_committed_hca_per_seq=n_hca_gpu,
            block_tables=hca_bt_gpu,
            hca_indices=idx_hca_gpu,
            T=T,
            win=win,
            swa_pages=swa_pages_val,
        )

    # --- Set metadata on attn_md ---
    attn_md.state = AttnState.DECODE
    attn_md.state_slot_mapping = state_slot_gpu
    attn_md.state_slot_mapping_cpu = ssm_np.copy()
    attn_md.kv_indices_swa = idx_swa_gpu
    attn_md.kv_indptr_swa = indptr_swa_gpu
    attn_md.kv_indices_csa = idx_csa_gpu
    attn_md.kv_indptr_csa = indptr_csa_gpu
    attn_md.kv_indices_hca = idx_hca_gpu
    attn_md.kv_indptr_hca = indptr_hca_gpu
    attn_md.swa_pages = swa_pages_val
    attn_md.n_committed_csa_per_seq = torch.from_numpy(
        n_csa_np.astype(np.int32)).to(device=device)
    attn_md.n_committed_hca_per_seq = n_hca_gpu

    # cu_seqlens_q for decode: [0, 1, 2, ..., bs]
    attn_md.cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
    attn_md.max_seqlen_q = 1
    attn_md.batch_id_per_token = batch_id_gpu

    # --- indexer_meta (for CSA layers' topk selection in decode) ---
    attn_md.indexer_meta = {
        "n_committed_per_seq_gpu": torch.from_numpy(n_csa_np.astype(np.int32)).to(device=device),
    }
    attn_md.n_committed_csa_per_seq = attn_md.indexer_meta["n_committed_per_seq_gpu"]
    attn_md.n_committed_csa_per_seq_cpu = n_csa_np.copy()
    attn_md.skip_prefix_len_csa = torch.zeros(bs, dtype=torch.int32, device=device)

    # --- compress_plans ---
    try:
        extend_lens_cpu = np.ones(bs, dtype=np.int32)
        context_lens_cpu = (positions_np + 1).astype(np.int32)
        _plan_bufs = {
            4: {
                "compress": CpuGpuBuffer(max(1, bs), 4, dtype=torch.int32, device=device),
                "write": CpuGpuBuffer(max(1, bs * 8), 4, dtype=torch.int32, device=device),
            },
            128: {
                "compress": CpuGpuBuffer(max(1, bs), 4, dtype=torch.int32, device=device),
                "write": CpuGpuBuffer(max(1, bs * 128), 4, dtype=torch.int32, device=device),
            },
        }
        attn_md.compress_plans = make_compress_plans(
            extend_lens_cpu, context_lens_cpu, [(4, True), (128, False)],
            plan_buffers=_plan_bufs,
        )
    except Exception as e:
        logger.warning("Eager decode Triton: compress_plans failed: %s", e)
        attn_md.compress_plans = {}

    # --- Store block_ids and positions for gather/scatter + state reset ---
    attn_md._eager_triton_block_ids = torch.from_numpy(
        block_ids_np.astype(np.int64)).to(device=device)
    attn_md._eager_triton_active_bs = bs
    attn_md._eager_triton_swa_pages = swa_pages_val
    attn_md._eager_triton_positions = positions_gpu

    setattr(attn_md, _V4_META_BUILT_ATTR, True)
    return True


def _build_prefill_extend_indices_gpu(positions, cu_seqlens_q, bid_per_tok, win, total_tokens, device):
    """Build causal extend indices on GPU using vectorized torch ops."""
    extend_counts = torch.minimum(positions + 1, torch.tensor(win, dtype=torch.int32, device=device))
    indptr = torch.zeros(total_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(extend_counts, dim=0, out=indptr[1:])
    total_nnz = int(indptr[-1].item())

    if total_nnz == 0:
        return torch.zeros(1, dtype=torch.int32, device=device), indptr

    ext_starts = (cu_seqlens_q[bid_per_tok.long()] + positions - extend_counts + 1).to(torch.int32)
    ext_starts_expanded = torch.repeat_interleave(ext_starts, extend_counts)
    group_starts = torch.repeat_interleave(indptr[:-1], extend_counts)
    global_idx = torch.arange(total_nnz, device=device, dtype=torch.int32)
    indices = ext_starts_expanded + (global_idx - group_starts)
    return indices, indptr


def _build_hca_prefix_indices_gpu(positions, bid_per_tok, hca_bt, swa_pages, hca_k, total_tokens, device):
    """Build HCA prefix indices on GPU."""
    n_hca_per_tok = ((positions + 1) // 128).to(torch.int32)
    indptr = torch.zeros(total_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(n_hca_per_tok, dim=0, out=indptr[1:])
    total_nnz = int(indptr[-1].item())

    if total_nnz == 0 or hca_bt is None or swa_pages <= 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, indptr

    tok_ids = torch.repeat_interleave(torch.arange(total_tokens, device=device, dtype=torch.int64), n_hca_per_tok)
    group_starts_exp = torch.repeat_interleave(indptr[:-1].long(), n_hca_per_tok)
    ci = (torch.arange(total_nnz, device=device, dtype=torch.int64) - group_starts_exp).to(torch.int32)

    bid = bid_per_tok[tok_ids].long()
    lb = (ci // hca_k).long()
    sb = ci % hca_k
    max_blocks = hca_bt.shape[1]
    lb_clamped = torch.clamp(lb, 0, max_blocks - 1)
    pb = hca_bt[bid, lb_clamped].to(torch.int32)
    indices = (swa_pages + pb * hca_k + sb).to(torch.int32)
    return indices, indptr


def _build_csa_prefix_indices_gpu(positions, bid_per_tok, n_csa_per_seq, index_topk, total_tokens, device):
    """Build CSA prefix indices on GPU (zeros with computed indptr)."""
    n_csa_per_tok = torch.minimum(
        (positions + 1) // 4,
        torch.minimum(
            n_csa_per_seq[bid_per_tok.long()].to(torch.int32),
            torch.tensor(index_topk, dtype=torch.int32, device=device),
        ),
    ).to(torch.int32)
    indptr = torch.zeros(total_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(n_csa_per_tok, dim=0, out=indptr[1:])
    total_nnz = int(indptr[-1].item())
    indices = torch.zeros(max(total_nnz, 1), dtype=torch.int32, device=device)
    return indices, indptr


def _build_v4_per_forward_metadata(attn_md, attn_inputs, v4_ratios, v4_block_tables,
                                    region_to_group, device, window_size=128,
                                    pool_swa_pages=0):
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
        _ssm = swa_bt[:bs, 0].to(dtype=torch.int32, device=device)
        attn_md.state_slot_mapping = _ssm.as_strided(_ssm.shape, (1,) * _ssm.dim())
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

    # -- prefix / seq_lens (computed once, reused by compress_plans and n_committed) --
    prefix = getattr(attn_inputs, "prefix_lengths", None)
    prefix_cpu = prefix.cpu().numpy().astype(np.int32) if prefix is not None else np.zeros(bs, dtype=np.int32)
    if is_prefill:
        seq_lens_cpu = (prefix_cpu + input_lens_cpu).astype(np.int32)
    else:
        seq_lens = getattr(attn_inputs, "sequence_lengths", None)
        if seq_lens is not None:
            seq_lens_cpu = seq_lens.cpu().numpy().astype(np.int32)
        else:
            seq_lens_cpu = (prefix_cpu + input_lens_cpu).astype(np.int32)

    # -- compress_plans (reuse pre-computed CPU arrays) --
    try:
        _build_compress_plans(attn_md, attn_inputs, v4_ratios, bs, device,
                              input_lens_cpu=input_lens_cpu, prefix_cpu=prefix_cpu,
                              seq_lens_cpu=seq_lens_cpu)
    except Exception as e:
        logger.warning("Failed to build compress_plans: %s — attention will use fallback", e)
        attn_md.compress_plans = {}

    unique_ratios = set(r for r in v4_ratios if r > 0)
    csa_ratio = 4  # V4 CSA always uses ratio=4
    for ratio_val in unique_ratios:
        committed = seq_lens_cpu // ratio_val
        key = f"n_committed_{_ratio_label(ratio_val)}_per_seq"
        setattr(attn_md, key, torch.from_numpy(committed.astype(np.int32)).to(device=device))
    attn_md.n_committed_csa_per_seq = torch.from_numpy(
        (seq_lens_cpu // csa_ratio).astype(np.int32)
    ).to(device=device) if csa_ratio in unique_ratios else torch.zeros(bs, dtype=torch.int32, device=device)

    # -- swa_pages set below (from pool_swa_pages or block table fallback) --

    # -- V4 sparse attention indices (ring buffer format) --
    # V4 ALWAYS uses sparse_attn_v4_paged_prefill/decode, never flash_attn.
    ssm = attn_md.state_slot_mapping
    win = window_size  # sliding window size from model config
    cs = win           # win_with_spec = window_size + max_spec_steps (0 when MTP off)
    total_tokens = int(input_lens_cpu.sum())

    # swa_pages: boundary in unified_kv between SWA [0,swa_pages) and compress [swa_pages,...)
    # swa_pages from pool flat size (set by _bind_v4_kv_cache_views on first layer bind).
    # First forward (warmup) may not have it yet — use 0 as safe default (no compress
    # indices will be generated since n_committed=0 for warmup).
    attn_md.swa_pages = pool_swa_pages if pool_swa_pages > 0 else 0

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
        # Prefill: sequential positions (GPU, no Python loop)
        cu_q_gpu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        torch.cumsum(input_lengths.to(dtype=torch.int32, device=device), dim=0, out=cu_q_gpu[1:])
        offsets = torch.repeat_interleave(cu_q_gpu[:-1], input_lengths.to(dtype=torch.int32, device=device))
        positions_gpu = (torch.arange(total_tokens, device=device, dtype=torch.int32) - offsets)
        positions_np = positions_gpu.cpu().numpy().astype(np.int32)

    cu_q_cpu = np.zeros(bs + 1, dtype=np.int32)
    np.cumsum(input_lens_cpu, out=cu_q_cpu[1:])
    bid_per_tok = np.repeat(np.arange(bs, dtype=np.int32), input_lens_cpu)
    ssm_cpu = ssm.cpu().numpy().astype(np.int32)

    if not is_prefill:
        # === DECODE: per-ratio ragged indices [compress_HEAD, swa_TAIL] ===
        # SWA indices < swa_pages → dual-ptr kernel reads from swa_kv
        # Compress indices >= swa_pages → dual-ptr kernel reads from compress_kv
        swa_pages_val = attn_md.swa_pages  # consistent with decode kernel

        # Build SWA ring indices per token (shared TAIL for all buffers)
        swa_per_tok = []
        for t in range(total_tokens):
            bid = int(bid_per_tok[t])
            slot = int(ssm_cpu[bid])
            pos = int(positions_np[t])
            n = min(pos + 1, win)
            tok_swa = []
            for i in range(n):
                abs_pos = pos - n + 1 + i
                ring = abs_pos % cs
                tok_swa.append(slot * cs + ring)
            swa_per_tok.append(tok_swa)

        # SWA buffer (Dense layers): SWA-only
        swa_all = []
        swa_indptr = [0]
        for t in range(total_tokens):
            swa_all.extend(swa_per_tok[t])
            swa_indptr.append(len(swa_all))
        idx_swa = torch.tensor(swa_all, dtype=torch.int32, device=device) if swa_all else torch.zeros(1, dtype=torch.int32, device=device)
        ptr_swa = torch.tensor(swa_indptr, dtype=torch.int32, device=device)
        attn_md.kv_indices_swa = idx_swa
        attn_md.kv_indptr_swa = ptr_swa

        # HCA buffer: SWA-only for isolation test (disable compress entries)
        hca_k = win // 128  # k_per_block for HCA = block_size / ratio = 128/128 = 1
        hca_bt = v4_block_tables.get(HCA_KV)
        hca_bt_cpu = hca_bt.cpu().numpy() if hca_bt is not None else None
        hca_all = []
        hca_indptr = [0]
        for t in range(total_tokens):
            bid = int(bid_per_tok[t])
            pos = int(positions_np[t])
            n_committed = (pos + 1) // 128
            if hca_bt_cpu is not None and n_committed > 0:
                for ci in range(n_committed):
                    lb = ci // hca_k
                    sb = ci % hca_k
                    if lb < hca_bt_cpu.shape[1]:
                        pb = int(hca_bt_cpu[bid, lb])
                        if pb >= 0:
                            hca_all.append(swa_pages_val + pb * hca_k + sb)
            hca_all.extend(swa_per_tok[t])
            hca_indptr.append(len(hca_all))
        idx_hca = torch.tensor(hca_all, dtype=torch.int32, device=device) if hca_all else torch.zeros(1, dtype=torch.int32, device=device)
        ptr_hca = torch.tensor(hca_indptr, dtype=torch.int32, device=device)
        attn_md.kv_indices_hca = idx_hca
        attn_md.kv_indptr_hca = ptr_hca

        # CSA buffer: [topk_compress (HEAD, uninitialized)] [swa_ring (TAIL)]
        # HEAD section filled later by csa_translate_pack when Indexer runs.
        # Must pre-allocate space for both sections.
        index_topk = 1024  # DeepSeek-V4 default
        n_committed_csa_np = (seq_lens_cpu // 4).astype(np.int32)

        csa_all = []
        csa_indptr = [0]
        for t in range(total_tokens):
            bid = int(bid_per_tok[t])
            pos = int(positions_np[t])
            n_csa = int(min((pos + 1) // 4, int(n_committed_csa_np[bid]), index_topk))
            # HEAD: reserve n_csa slots (filled by csa_translate_pack), init to 0
            csa_all.extend([0] * n_csa)
            # TAIL: SWA ring entries
            csa_all.extend(swa_per_tok[t])
            csa_indptr.append(len(csa_all))

        idx_csa = torch.tensor(csa_all, dtype=torch.int32, device=device) if csa_all else torch.zeros(1, dtype=torch.int32, device=device)
        ptr_csa = torch.tensor(csa_indptr, dtype=torch.int32, device=device)
        attn_md.kv_indices_csa = idx_csa
        attn_md.kv_indptr_csa = ptr_csa

    else:
        # === PREFILL_NATIVE: causal extend + empty prefix ===
        # GPU-accelerated index construction (replaces Python for-loops)
        positions_gpu_i32 = torch.from_numpy(positions_np).to(device=device, dtype=torch.int32)
        bid_per_tok_gpu = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32, device=device),
            input_lengths.to(dtype=torch.int32, device=device),
        )
        cu_q_gpu = attn_md.cu_seqlens_q

        # Extend indices (causal mask into per-fwd kv)
        attn_md.kv_indices_extend, attn_md.kv_indptr_extend = \
            _build_prefill_extend_indices_gpu(positions_gpu_i32, cu_q_gpu, bid_per_tok_gpu, win, total_tokens, device)

        empty_idx = torch.zeros(0, dtype=torch.int32, device=device)
        empty_ptr = torch.zeros(total_tokens + 1, dtype=torch.int32, device=device)
        attn_md.kv_indices_prefix_swa = empty_idx
        attn_md.kv_indptr_prefix_swa = empty_ptr

        # HCA prefix indices
        swa_pages_pf = pool_swa_pages if pool_swa_pages > 0 else 0
        hca_bt = v4_block_tables.get(HCA_KV)
        hca_k_pf = win // 128
        attn_md.kv_indices_prefix_hca, attn_md.kv_indptr_prefix_hca = \
            _build_hca_prefix_indices_gpu(positions_gpu_i32, bid_per_tok_gpu, hca_bt, swa_pages_pf, hca_k_pf, total_tokens, device)

        # CSA prefix indices (zeros, filled by Indexer topk later)
        index_topk = 1024
        n_csa_per_seq_gpu = attn_md.n_committed_csa_per_seq
        attn_md.kv_indices_prefix_csa, attn_md.kv_indptr_prefix_csa = \
            _build_csa_prefix_indices_gpu(positions_gpu_i32, bid_per_tok_gpu, n_csa_per_seq_gpu, index_topk, total_tokens, device)

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


def _build_compress_plans(attn_md, attn_inputs, v4_ratios, bs, device,
                          input_lens_cpu=None, prefix_cpu=None, seq_lens_cpu=None):
    """Build CompressPlan dict for each unique compress ratio."""
    from atom.model_ops.v4_kernels.compress_plan import make_compress_plans
    from atom.utils import CpuGpuBuffer

    is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
    if input_lens_cpu is None:
        input_lens_cpu = attn_inputs.input_lengths.cpu().numpy().astype(np.int32)

    if is_prefill:
        extend_lens_cpu = input_lens_cpu.copy()
        if prefix_cpu is None:
            prefix = getattr(attn_inputs, "prefix_lengths", None)
            prefix_cpu = prefix.cpu().numpy().astype(np.int32) if prefix is not None else np.zeros(bs, dtype=np.int32)
        context_lens_cpu = (prefix_cpu + input_lens_cpu).astype(np.int32) if seq_lens_cpu is None else seq_lens_cpu
    else:
        extend_lens_cpu = np.ones(bs, dtype=np.int32)
        if seq_lens_cpu is not None:
            context_lens_cpu = seq_lens_cpu
        else:
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
        raw = swa_pool.kv_cache_base
        B = raw.shape[0]
        swa_bf16 = raw.view(torch.bfloat16)
        swa_kv = swa_bf16.reshape(B, window_size, head_dim)
        attn_module.swa_kv = swa_kv
        swa_flat = swa_bf16.reshape(-1, head_dim)
    else:
        device = next(attn_module.parameters()).device
        swa_flat = torch.zeros(1, head_dim, dtype=torch.bfloat16, device=device)

    attn_module.unified_kv = swa_flat
    # Save pool swa_kv view for graph-mode gather/scatter (3D: [num_blocks, win, head_dim])
    attn_module._rtp_pool_swa_kv = swa_kv if swa_pool is not None else None

    # Cache SWA flat view globally for graph-mode fallback
    global _SWA_FLAT_CACHE
    if _SWA_FLAT_CACHE is None and swa_flat.numel() > 1:
        _SWA_FLAT_CACHE = swa_flat

    # Compress KV: zero-copy view of CSA/HCA pool (for decode-time cat)
    compress_kv = None
    if ratio == 4:
        csa_pool = layer_pools.get("CSA_KV")
        if csa_pool is not None:
            compress_kv = csa_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, head_dim)
            # Cache for graph-mode fallback
            global _CSA_COMPRESS_KV_CACHE
            if _CSA_COMPRESS_KV_CACHE is None:
                _CSA_COMPRESS_KV_CACHE = compress_kv
    elif ratio == 128:
        hca_pool = layer_pools.get("HCA_KV")
        if hca_pool is not None:
            compress_kv = hca_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, head_dim)
            # Cache for graph-mode fallback
            global _HCA_COMPRESS_KV_CACHE
            if _HCA_COMPRESS_KV_CACHE is None:
                _HCA_COMPRESS_KV_CACHE = compress_kv

    attn_module._rtp_compress_kv = compress_kv
    attn_module._rtp_swa_pages = swa_flat.shape[0]


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

    win = attn_module.window_size  # 128

    # Compressor KV cache
    if ratio == 4:
        kv_pool = layer_pools.get("CSA_KV")
        state_pool = layer_pools.get("CSA_STATE")
        k_per_block = win // ratio  # entries per block = 128/4 = 32
    elif ratio == 128:
        kv_pool = layer_pools.get("HCA_KV")
        state_pool = layer_pools.get("HCA_STATE")
        k_per_block = win // ratio  # 128/128 = 1
    else:
        return

    if kv_pool is not None:
        kv_raw = kv_pool.kv_cache_base.view(torch.bfloat16)
        compressor.kv_cache = kv_raw.reshape(-1, k_per_block, head_dim)

    # State persistence: fused_compress_attn requires contiguous state, but pool
    # views are non-contiguous (interleaved [kv,score] layout). Shadow buffers
    # from _ensure_v4_native_buffers are persistent module attributes that
    # accumulate compressor state across decode steps. Do NOT overwrite them
    # from pool — .contiguous() creates a copy, so in-place updates during
    # forward_impl would be lost and never written back to pool.
    # Skip state bind entirely; shadow buffers handle state persistence.


def _bind_v4_indexer_views(
    attn_module: Any,
    layer_pools: Dict[str, Any],
) -> None:
    """Bind indexer kv_cache as FP8 shadow buffer.

    Indexer scoring kernel (top_k_per_row_decode → fp8_paged_mqa_logits)
    requires FP8 kv_cache regardless of RTP-LLM's kv_cache_dtype setting.
    We allocate a per-layer FP8 contiguous shadow buffer.

    Main Compressor (CSA/HCA) stays BF16 — only the Indexer's inner
    Compressor uses FP8 (is_quant=True in fused_compress_attn).

    State (kv_state/score_state): NOT touched — kept from _ensure_v4_native_buffers.
    """
    indexer = getattr(attn_module, "indexer", None)
    if indexer is None:
        return

    try:
        from aiter import dtypes
        fp8_dtype = dtypes.fp8
    except (ImportError, AttributeError):
        fp8_dtype = torch.float8_e4m3fnuz

    idx_head_dim = getattr(indexer, "head_dim", 128)
    ratio = attn_module.compress_ratio
    window_size = attn_module.window_size
    k1 = window_size // ratio  # 32
    aligned_dim = ((idx_head_dim + 4 + 15) // 16) * 16  # 144

    kv_pool = layer_pools.get("INDEXER_KV")
    if kv_pool is None:
        return

    NB = kv_pool.kv_cache_base.shape[0]

    # Use max of INDEXER_KV and CSA_KV block counts
    csa_pool = layer_pools.get("CSA_KV")
    if csa_pool is not None:
        NB = max(NB, csa_pool.kv_cache_base.shape[0])

    # Allocate FP8 shadow once per layer
    shadow_kv = getattr(indexer, "_rtp_idx_kv_shadow", None)
    if shadow_kv is None or shadow_kv.shape[0] < NB:
        shadow_kv = torch.zeros(NB, k1, aligned_dim, dtype=fp8_dtype,
                                device=kv_pool.kv_cache_base.device)
        indexer._rtp_idx_kv_shadow = shadow_kv

    # Bind kv_cache for both Indexer and its inner Compressor
    indexer.kv_cache = shadow_kv

    idx_compressor = getattr(indexer, "compressor", None)
    if idx_compressor is not None:
        idx_compressor.kv_cache = shadow_kv
        # FP8 cache_scale: strided fp32 view of the scale region within each block
        block_fp32_stride = (k1 * aligned_dim) // 4  # 1152
        scale_fp32_offset = (k1 * idx_head_dim) // 4  # 1024
        idx_compressor.cache_scale = (
            shadow_kv.view(torch.float32)
            .view(-1)
            .as_strided(
                size=(NB, k1),
                stride=(block_fp32_stride, 1),
                storage_offset=scale_fp32_offset,
            )
        )
    # State (kv_state/score_state): intentionally NOT modified here.
    # _ensure_v4_native_buffers allocates contiguous shadow buffers indexed
    # by state_slot_mapping (small values). fused_compress_attn requires
    # contiguous state — our shadow buffers satisfy this.


def _v4_forward_cuda_graph(self, x, positions, fc, attn_md):
    """CUDA Graph fast-path for V4 attention layers.

    All metadata (kv_indices, kv_indptr, state_slot_mapping, compress_plans)
    has been pre-built by prepare_cuda_graph + _run_v4_graph_index_kernels.
    This function only does:
    1. KV cache binding (stable pool addresses — safe for graph)
    2. Attaches pre-built metadata fields to attn_md
    3. Calls forward_impl
    """
    from atom.utils.forward_context import AttnState

    bufs = attn_md._v4_cg_bufs
    swa_pages_val = attn_md._v4_swa_pages
    v4_block_tables = getattr(attn_md, "v4_block_tables", {})
    v4_ratios = getattr(attn_md, "v4_compress_ratios", [])
    ratio = v4_ratios[self.layer_id] if self.layer_id < len(v4_ratios) else 0

    active_bs = int(bufs.get("_active_bs", 0)) or 1
    win = int(bufs["_win"])

    # 1. Ensure native buffers (one-time allocation, guarded by flag)
    # num_slots = max_bs (compact). state_slot_mapping is remapped to [0..bs-1]
    # by prepare_cuda_graph, so state buffers only need max_bs entries.
    # swa_kv is also compact [max_bs, win, head_dim] — gather/scatter handles
    # the mapping between compact slots and actual pool block positions.
    if not getattr(self, _V4_BUFFERS_ALLOCATED, False):
        max_bs = int(bufs["state_slot"].shape[0])
        _ensure_v4_native_buffers(self, num_slots=max(max_bs, 32), device=x.device)

    # 2. Bind KV cache from RTP-LLM pool (for compress_kv, compressor, pool_swa)
    kv_cache_data = fc.kv_cache_data
    if kv_cache_data is None:
        _rt = bufs.get("_kv_cache_ref_runtime")
        if _rt is not None:
            kv_cache_data = getattr(_rt, "_rtp_kv_cache_data", None)
    cache_entry = kv_cache_data.get(f"layer_{self.layer_id}") if kv_cache_data else None
    if cache_entry and isinstance(cache_entry.k_cache, dict) and cache_entry.k_cache:
        try:
            _bind_v4_kv_cache_views(self, cache_entry.k_cache)
            compressor = getattr(self, "compressor", None)
            if compressor is not None and ratio != 0:
                head_dim = self.head_dim
                if ratio == 4:
                    csa_pool = cache_entry.k_cache.get("CSA_KV")
                    if csa_pool is not None:
                        compressor.kv_cache = csa_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, self.window_size // ratio, head_dim)
                elif ratio == 128:
                    hca_pool = cache_entry.k_cache.get("HCA_KV")
                    if hca_pool is not None:
                        compressor.kv_cache = hca_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, self.window_size // ratio, head_dim)
            if ratio == 4:
                _bind_v4_indexer_views(self, cache_entry.k_cache)
        except Exception as e:
            logger.error("V4 graph bind layer %d: %s", self.layer_id, e, exc_info=True)
            return torch.zeros_like(x)

    # After bind: override swa_kv and unified_kv with compact buffer.
    # This ensures swa_write and paged_decode operate on the SAME independent
    # memory (not the pool). Pool data is synced via gather/scatter.
    _compact_swa = getattr(self, "_compact_swa_kv", None)
    if _compact_swa is not None:
        self.swa_kv = _compact_swa
        self.unified_kv = _compact_swa.view(-1, self.head_dim)

    if (cache_entry is None or not cache_entry.k_cache) and ratio != 0 and getattr(self, "_rtp_compress_kv", None) is None:
        # Fallback: fc.kv_cache_data is None during graph capture.
        # Use cached pool views from the module-level registry (populated
        # during _ensure_cuda_graph_prewarmed).
        head_dim = self.head_dim
        win = self.window_size
        if _SWA_FLAT_CACHE is not None:
            self.unified_kv = _SWA_FLAT_CACHE
            self.swa_kv = _SWA_FLAT_CACHE.view(-1, win, head_dim)
            self._rtp_swa_pages = _SWA_FLAT_CACHE.shape[0]
        if ratio == 4 and _CSA_COMPRESS_KV_CACHE is not None:
            self._rtp_compress_kv = _CSA_COMPRESS_KV_CACHE
            compressor = getattr(self, "compressor", None)
            if compressor is not None:
                k_per_block = win // ratio  # 32
                compressor.kv_cache = _CSA_COMPRESS_KV_CACHE.view(-1, k_per_block, head_dim)
        elif ratio == 128 and _HCA_COMPRESS_KV_CACHE is not None:
            self._rtp_compress_kv = _HCA_COMPRESS_KV_CACHE
            compressor = getattr(self, "compressor", None)
            if compressor is not None:
                k_per_block = win // ratio  # 1
                compressor.kv_cache = _HCA_COMPRESS_KV_CACHE.view(-1, k_per_block, head_dim)
        else:
            logger.warning("V4 graph fallback: no cached compress pool for layer %d ratio %d", self.layer_id, ratio)
    elif getattr(self, "swa_kv", None) is None or self.swa_kv.numel() <= 1:
        # Dense layers (ratio=0) also need swa_kv bound for swa_write.
        if _SWA_FLAT_CACHE is not None:
            head_dim = self.head_dim
            win = self.window_size
            self.unified_kv = _SWA_FLAT_CACHE
            self.swa_kv = _SWA_FLAT_CACHE.view(-1, win, head_dim)
            self._rtp_swa_pages = _SWA_FLAT_CACHE.shape[0]


    # 3. Set metadata fields from pre-allocated buffers
    attn_md.state = AttnState.DECODE
    attn_md.state_slot_mapping = bufs["state_slot"][:active_bs]
    # cu_seqlens_q for decode = arange(0..bs); reuse from _cg_meta_bufs
    # which was pre-allocated as arange in _ensure_cuda_graph_prewarmed.
    max_bs = int(bufs["indptr_swa"].shape[0]) - 1
    attn_md.cu_seqlens_q = bufs.get("_cu_seqlens_q", None)
    if attn_md.cu_seqlens_q is None:
        # Fallback: create once and cache (first capture call)
        attn_md.cu_seqlens_q = torch.arange(max_bs + 1, device=x.device, dtype=torch.int32)
        bufs["_cu_seqlens_q"] = attn_md.cu_seqlens_q
    attn_md.max_seqlen_q = 1
    attn_md.batch_id_per_token = bufs["batch_id"]  # int32 for forward_impl (qk_norm_rope + csa_translate_pack)
    attn_md.n_committed_csa_per_seq = bufs["n_csa"][:active_bs]
    attn_md.kv_indices_swa = bufs["idx_swa"]
    attn_md.kv_indices_csa = bufs["idx_csa"]
    attn_md.kv_indices_hca = bufs["idx_hca"]
    attn_md.kv_indptr_swa = bufs["indptr_swa"]
    attn_md.kv_indptr_csa = bufs["indptr_csa"]
    attn_md.kv_indptr_hca = bufs["indptr_hca"]
    attn_md.swa_pages = swa_pages_val
    attn_md.compress_kv = getattr(self, "_rtp_compress_kv", None)
    # skip_prefix_len_csa: pre-allocate and cache
    skip_buf = bufs.get("_skip_prefix_len_csa", None)
    if skip_buf is None:
        skip_buf = torch.zeros(max_bs, dtype=torch.int32, device=x.device)
        bufs["_skip_prefix_len_csa"] = skip_buf
    attn_md.skip_prefix_len_csa = skip_buf[:active_bs]

    # Indexer metadata for CSA layers
    if ratio == 4:
        attn_md.indexer_meta = {
            "n_committed_per_seq_gpu": bufs["n_csa"][:active_bs],
        }
        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import INDEXER_KV
        indexer_bt = v4_block_tables.get(INDEXER_KV)
        if indexer_bt is not None:
            attn_md._indexer_block_tables = indexer_bt

    # Set region block_table
    from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import SWA_KV, CSA_KV, HCA_KV
    if ratio == 0:
        region_bt = v4_block_tables.get(SWA_KV)
    elif ratio == 4:
        region_bt = v4_block_tables.get(CSA_KV)
    else:
        region_bt = v4_block_tables.get(HCA_KV)
    if region_bt is not None:
        attn_md.block_tables = region_bt

    # Compress plans built by prepare_cuda_graph via CpuGpuBuffer plan buffers.
    # CompressPlan objects reference stable GPU addresses (captured once, replayed).
    attn_md.compress_plans = bufs.get("_compress_plans", {})
    # state_slot_mapping_cpu (numpy) needed by compressor internals
    attn_md.state_slot_mapping_cpu = bufs.get("_state_slot_mapping_cpu")

    # --- Gather: pool[block_ids] → compact swa_kv[0..bs-1] ---
    _block_ids = bufs.get("_block_ids")
    _pool_swa = getattr(self, "_rtp_pool_swa_kv", None)
    if _block_ids is not None and _pool_swa is not None and active_bs > 0:
        _bid = _block_ids[:active_bs]  # [active_bs] int64
        # Gather SWA KV from pool to compact buffer
        self.swa_kv[:active_bs].copy_(
            _pool_swa.index_select(0, _bid)
        )

    # --- STATE pool gather: pool → compact kv_state/score_state ---
    from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import CSA_STATE, HCA_STATE
    _state_pool_view_g = None
    _state_block_ids_g = None
    compressor = getattr(self, "compressor", None)
    if compressor is not None and ratio != 0 and active_bs > 0:
        _state_region = CSA_STATE if ratio == 4 else HCA_STATE
        _state_bt = v4_block_tables.get(_state_region)
        if _state_bt is not None and cache_entry is not None and cache_entry.k_cache:
            _state_pool_name = "CSA_STATE" if ratio == 4 else "HCA_STATE"
            _sp = cache_entry.k_cache.get(_state_pool_name)
            if _sp is not None:
                _state_pool_raw = _sp.kv_cache_base.view(torch.float32)
                _n_blocks = _state_pool_raw.shape[0]
                _elems = _state_pool_raw.numel() // _n_blocks
                _state_pool_view_g = _state_pool_raw.reshape(_n_blocks, _elems)
                _state_block_ids_g = _state_bt[:active_bs, 0].to(torch.int64)
                _half = _elems // 2
                _ring = compressor.kv_state.shape[1]
                _dim = compressor.kv_state.shape[2]
                if _half == _ring * _dim:
                    _gathered = _state_pool_view_g[_state_block_ids_g]
                    compressor.kv_state[:active_bs] = _gathered[:, :_half].reshape(active_bs, _ring, _dim)
                    compressor.score_state[:active_bs] = _gathered[:, _half:].reshape(active_bs, _ring, _dim)

    try:
        result = self.forward_impl(x, positions)
    except Exception as e:
        logger.error("V4 graph fwd layer %d (ratio=%d): %s", self.layer_id, ratio, e, exc_info=True)
        return torch.zeros_like(x)

    # --- Scatter: compact → pool ---
    if _block_ids is not None and _pool_swa is not None and active_bs > 0:
        _bid = _block_ids[:active_bs]
        _pool_swa.index_copy_(0, _bid, self.swa_kv[:active_bs])
    # STATE scatter
    if _state_pool_view_g is not None and _state_block_ids_g is not None and compressor is not None:
        _ring = compressor.kv_state.shape[1]
        _dim = compressor.kv_state.shape[2]
        _kv_flat = compressor.kv_state[:active_bs].reshape(active_bs, -1)
        _sc_flat = compressor.score_state[:active_bs].reshape(active_bs, -1)
        _combined = torch.cat([_kv_flat, _sc_flat], dim=-1)
        _state_pool_view_g[_state_block_ids_g] = _combined

    return result


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
    _plugin_profile_dir = os.environ.get(
        "ATOM_PLUGIN_PROFILE_DIR", "./plugin_traces"
    )
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
                rank, gz_path, sz / 1e6, time.monotonic() - t0,
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
        rank, _plugin_profile_dir,
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


def _patched_v4_forward(self, x, positions):
    """Patched forward for DeepseekV4Attention in rtp-llm plugin mode.

    Patches `forward` (not `forward_impl`) because `forward` delegates to a
    torch custom op (`v4_attention_with_output`) that captures a direct reference
    to `forward_impl`, bypassing Python method resolution.

    CUDA Graph mode:
      When `attn_md._v4_cuda_graph_mode` is set (by _run_v4_graph_index_kernels),
      this function skips _build_v4_per_forward_metadata entirely. All V4
      metadata (kv_indices, kv_indptr, state_slot_mapping, etc.) has already
      been constructed by:
      - prepare_cuda_graph(): CPU computation + H2D to pre-allocated buffers
      - _run_v4_graph_index_kernels(): Triton kernels fill indices (captured)
    """
    global _plugin_profiler, _plugin_profile_dir
    if os.environ.get("ATOM_PLUGIN_PROFILE") == "1" and x.shape[0] > 0:
        _profile_dir = _plugin_profile_dir or os.environ.get("ATOM_PLUGIN_PROFILE_DIR", "./plugin_traces")
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

    from atom.utils.forward_context import get_forward_context

    fc = get_forward_context()
    attn_md = fc.attn_metadata
    v4_block_tables = getattr(attn_md, "v4_block_tables", None)

    if v4_block_tables is None:
        return _original_v4_forward(self, x, positions)

    # --- CUDA Graph fast path ---
    if getattr(attn_md, '_v4_cuda_graph_mode', False):
        return _v4_forward_cuda_graph(self, x, positions, fc, attn_md)

    # --- Eager (non-graph) path ---
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
            # Read index_topk from model args for CSA indices construction
            _m_args = getattr(self, "args", None)
            if _m_args is None:
                _m = getattr(self, "model", None)
                _m_args = getattr(_m, "args", None) if _m else None
            attn_md._index_topk = getattr(_m_args, "index_topk", 1024) if _m_args else 1024

            # DECODE: use Triton kernels for index construction (same as graph mode)
            _is_eager_prefill = bool(getattr(rtp_attn_inputs, "is_prefill", True))
            if not _is_eager_prefill:
                _triton_ok = _build_eager_decode_with_triton(
                    attn_md, rtp_attn_inputs, v4_ratios, v4_block_tables,
                    region_to_group, x.device,
                    window_size=self.window_size,
                    pool_swa_pages=getattr(self, "_rtp_swa_pages", 0),
                    index_topk=attn_md._index_topk,
                )
                if not _triton_ok:
                    setattr(attn_md, _V4_META_BUILT_ATTR, True)
                    setattr(attn_md, _V4_META_FAILED_ATTR, True)
                    return torch.zeros_like(x)
            else:
                # PREFILL: use original CPU metadata construction
                _build_v4_per_forward_metadata(
                    attn_md, rtp_attn_inputs, v4_ratios, v4_block_tables,
                    region_to_group, x.device,
                    window_size=self.window_size,
                    pool_swa_pages=getattr(self, "_rtp_swa_pages", 0),
                )
        except Exception as e:
            logger.error("V4 metadata construction failed: %s — using zeros fallback", e, exc_info=True)
            setattr(attn_md, _V4_META_BUILT_ATTR, True)
            setattr(attn_md, _V4_META_FAILED_ATTR, True)
            return torch.zeros_like(x)

    # Determine layer type
    v4_ratios = getattr(attn_md, "v4_compress_ratios", [])
    ratio = v4_ratios[self.layer_id] if self.layer_id < len(v4_ratios) else 0

    # --- EAGER DECODE (Triton path): identical flow to _v4_forward_cuda_graph ---
    _triton_block_ids = getattr(attn_md, "_eager_triton_block_ids", None)
    if _triton_block_ids is not None:
        active_bs = int(getattr(attn_md, "_eager_triton_active_bs", 0))
        swa_pages_val = int(getattr(attn_md, "_eager_triton_swa_pages", 0))

        # Ensure compact native buffers (same as graph: num_slots = max(bs, 32))
        if not getattr(self, _V4_BUFFERS_ALLOCATED, False):
            _ensure_v4_native_buffers(self, num_slots=max(active_bs, 32), device=x.device)

        # Bind KV cache from pool (compress_kv, compressor.kv_cache, pool_swa)
        kv_cache_data = fc.kv_cache_data
        cache_entry = kv_cache_data.get(f"layer_{self.layer_id}") if kv_cache_data else None
        if cache_entry and isinstance(cache_entry.k_cache, dict) and cache_entry.k_cache:
            try:
                _bind_v4_kv_cache_views(self, cache_entry.k_cache)
                compressor = getattr(self, "compressor", None)
                if compressor is not None and ratio != 0:
                    head_dim = self.head_dim
                    if ratio == 4:
                        csa_pool = cache_entry.k_cache.get("CSA_KV")
                        if csa_pool is not None:
                            compressor.kv_cache = csa_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, self.window_size // ratio, head_dim)
                    elif ratio == 128:
                        hca_pool = cache_entry.k_cache.get("HCA_KV")
                        if hca_pool is not None:
                            compressor.kv_cache = hca_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, self.window_size // ratio, head_dim)
                if ratio == 4:
                    _bind_v4_indexer_views(self, cache_entry.k_cache)
            except Exception as e:
                logger.error("V4 eager decode bind layer %d: %s", self.layer_id, e, exc_info=True)
                return torch.zeros_like(x)

        # Override swa_kv + unified_kv with compact buffer
        _compact_swa = getattr(self, "_compact_swa_kv", None)
        if _compact_swa is not None:
            self.swa_kv = _compact_swa
            self.unified_kv = _compact_swa.view(-1, self.head_dim)

        # Set per-layer metadata (compress_kv for dual-pointer, block_tables, swa_pages)
        attn_md.compress_kv = getattr(self, "_rtp_compress_kv", None)
        attn_md.swa_pages = swa_pages_val
        if ratio == 0:
            region_bt = v4_block_tables.get(SWA_KV)
        elif ratio == 4:
            region_bt = v4_block_tables.get(CSA_KV)
        else:
            region_bt = v4_block_tables.get(HCA_KV)
        if region_bt is not None:
            attn_md.block_tables = region_bt

        # Indexer block_table patch (CSA layers)
        if ratio == 4:
            from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import INDEXER_KV
            indexer_bt = v4_block_tables.get(INDEXER_KV)
            if indexer_bt is not None:
                attn_md._indexer_block_tables = indexer_bt
            if self.indexer is not None:
                idx_comp = getattr(self.indexer, "compressor", None)
                if idx_comp is not None and not getattr(idx_comp, "_rtp_bt_patched", False):
                    _orig_comp_fwd = idx_comp.forward
                    def _patched_comp_fwd(x, plan, state_slot_mapping, block_tables=None, _orig=_orig_comp_fwd):
                        from atom.utils.forward_context import get_forward_context
                        md = get_forward_context().attn_metadata
                        bt = getattr(md, "_indexer_block_tables", block_tables)
                        return _orig(x, plan=plan, state_slot_mapping=state_slot_mapping, block_tables=bt)
                    idx_comp.forward = _patched_comp_fwd
                    idx_comp._rtp_bt_patched = True
                if not getattr(self.indexer, "_rtp_score_bt_patched", False):
                    _orig_score = self.indexer.indexer_score_topk
                    def _patched_score(q_fp8, weights, topk, _orig=_orig_score):
                        from atom.utils.forward_context import get_forward_context
                        fc2 = get_forward_context()
                        md = fc2.attn_metadata
                        saved_bt = md.block_tables
                        idx_bt = getattr(md, "_indexer_block_tables", saved_bt)
                        md.block_tables = idx_bt
                        try:
                            return _orig(q_fp8, weights, topk)
                        finally:
                            md.block_tables = saved_bt
                    self.indexer.indexer_score_topk = _patched_score
                    self.indexer._rtp_score_bt_patched = True

        # --- STATE pool gather: pool → compact kv_state/score_state ---
        # RTP-LLM manages STATE pool lifecycle (alloc/free/zero-init).
        # We gather state from pool BEFORE forward_impl so compressor reads
        # correct accumulated state. After forward_impl, scatter back.
        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import CSA_STATE, HCA_STATE
        _state_pool_view = None
        _state_block_ids = None
        compressor = getattr(self, "compressor", None)
        if compressor is not None and ratio != 0 and active_bs > 0:
            _state_region = CSA_STATE if ratio == 4 else HCA_STATE
            _state_bt = v4_block_tables.get(_state_region)
            if _state_bt is not None and cache_entry is not None:
                _state_pool_name = "CSA_STATE" if ratio == 4 else "HCA_STATE"
                _sp = cache_entry.k_cache.get(_state_pool_name) if cache_entry.k_cache else None
                if _sp is not None:
                    _state_pool_raw = _sp.kv_cache_base.view(torch.float32)
                    _n_blocks = _state_pool_raw.shape[0]
                    _elems = _state_pool_raw.numel() // _n_blocks
                    _state_pool_view = _state_pool_raw.reshape(_n_blocks, _elems)
                    _state_block_ids = _state_bt[:active_bs, 0].to(torch.int64)
                    # Gather
                    _gathered = _state_pool_view[_state_block_ids]  # [bs, elems]
                    _half = _elems // 2
                    _ring = compressor.kv_state.shape[1]
                    _dim = compressor.kv_state.shape[2]
                    if _half == _ring * _dim:
                        compressor.kv_state[:active_bs] = _gathered[:, :_half].reshape(active_bs, _ring, _dim)
                        compressor.score_state[:active_bs] = _gathered[:, _half:].reshape(active_bs, _ring, _dim)
                        if self.layer_id == 2:
                            logger.debug("DECODE STATE GATHER layer=%d: state_bid=%s",
                                         self.layer_id, _state_block_ids.tolist())
                    else:
                        if self.layer_id == 2:
                            logger.debug("DECODE STATE GATHER SKIP layer=%d: half=%d ring*dim=%d",
                                         self.layer_id, _half, _ring * _dim)
            else:
                if self.layer_id == 2:
                    logger.debug("DECODE STATE: no state_bt or cache_entry. state_bt=%s cache_entry=%s",
                                 _state_bt is not None, cache_entry is not None)

        # Gather: pool[block_ids] → compact swa_kv[0..bs-1]
        _pool_swa = getattr(self, "_rtp_pool_swa_kv", None)
        if _pool_swa is not None and active_bs > 0:
            _bid = _triton_block_ids[:active_bs]
            self.swa_kv[:active_bs].copy_(_pool_swa.index_select(0, _bid))

        try:
            result = self.forward_impl(x, positions)
        except Exception as e:
            logger.error("V4 eager decode fwd layer %d (ratio=%d): %s", self.layer_id, ratio, e, exc_info=True)
            return torch.zeros_like(x)

        # --- Scatter: compact → pool ---
        # SWA scatter
        if _pool_swa is not None and active_bs > 0:
            _bid = _triton_block_ids[:active_bs]
            _pool_swa.index_copy_(0, _bid, self.swa_kv[:active_bs])
        # STATE scatter
        if _state_pool_view is not None and _state_block_ids is not None and compressor is not None:
            _ring = compressor.kv_state.shape[1]
            _dim = compressor.kv_state.shape[2]
            _kv_flat = compressor.kv_state[:active_bs].reshape(active_bs, -1)
            _sc_flat = compressor.score_state[:active_bs].reshape(active_bs, -1)
            _combined = torch.cat([_kv_flat, _sc_flat], dim=-1)
            _state_pool_view[_state_block_ids] = _combined

        return result

    # --- PREFILL path (unchanged) ---
    # Prefill uses original block_ids as state_slot_mapping (swa_kv = pool view).
    # No compact remap — swa_write needs real block_ids to write correct pool positions.
    ssm = getattr(attn_md, "state_slot_mapping", None)
    if ssm is not None and ssm.numel() > 0:
        num_slots = max(int(ssm.max()) + 1, 32)
    else:
        num_slots = 32
    _ensure_v4_native_buffers(self, num_slots=num_slots, device=x.device)

    # 2. Bind KV cache from RTP-LLM pool (for compress_kv, pool_swa, compressor.kv_cache)
    kv_cache_data = fc.kv_cache_data
    cache_entry = kv_cache_data.get(f"layer_{self.layer_id}") if kv_cache_data else None
    if cache_entry and isinstance(cache_entry.k_cache, dict) and cache_entry.k_cache:
        try:
            _bind_v4_kv_cache_views(self, cache_entry.k_cache)
            # Bind compressor.kv_cache from RTP-LLM pool
            compressor = getattr(self, "compressor", None)
            if compressor is not None and ratio != 0:
                head_dim = self.head_dim
                if ratio == 4:
                    csa_pool = cache_entry.k_cache.get("CSA_KV")
                    if csa_pool is not None:
                        compressor.kv_cache = csa_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, self.window_size // ratio, head_dim)
                elif ratio == 128:
                    hca_pool = cache_entry.k_cache.get("HCA_KV")
                    if hca_pool is not None:
                        compressor.kv_cache = hca_pool.kv_cache_base.view(torch.bfloat16).reshape(-1, self.window_size // ratio, head_dim)
            if ratio == 4:
                _bind_v4_indexer_views(self, cache_entry.k_cache)
        except Exception as e:
            if not getattr(attn_md, "_v4_bind_err", False):
                logger.error("V4 bind layer %d: %s", self.layer_id, e, exc_info=True)
                attn_md._v4_bind_err = True
            fc.context.is_dummy_run = True
            return self.forward_impl(x, positions)
    attn_md.compress_kv = getattr(self, "_rtp_compress_kv", None)
    attn_md.swa_pages = getattr(self, "_rtp_swa_pages", 0)

    if ratio == 0:
        region_bt = v4_block_tables.get(SWA_KV)
    elif ratio == 4:
        region_bt = v4_block_tables.get(CSA_KV)
    else:
        region_bt = v4_block_tables.get(HCA_KV)
    if region_bt is not None:
        attn_md.block_tables = region_bt

    # For CSA layers: store INDEXER_KV block_table for Indexer shadow buffer access.
    # ATOM assumes Main KV and Indexer KV share one block allocator (same block_table),
    # but RTP-LLM has separate pools. We store INDEXER_KV bt on attn_md each forward,
    # and monkey-patch Indexer to read it from live forward_context (not closure capture).
    if ratio == 4:
        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import INDEXER_KV
        indexer_bt = v4_block_tables.get(INDEXER_KV)
        if indexer_bt is not None:
            attn_md._indexer_block_tables = indexer_bt

        if self.indexer is not None:
            idx_comp = getattr(self.indexer, "compressor", None)
            if idx_comp is not None and not getattr(idx_comp, "_rtp_bt_patched", False):
                _orig_comp_fwd = idx_comp.forward
                def _patched_comp_fwd(x, plan, state_slot_mapping, block_tables=None, _orig=_orig_comp_fwd):
                    from atom.utils.forward_context import get_forward_context
                    md = get_forward_context().attn_metadata
                    bt = getattr(md, "_indexer_block_tables", block_tables)
                    return _orig(x, plan=plan, state_slot_mapping=state_slot_mapping, block_tables=bt)
                idx_comp.forward = _patched_comp_fwd
                idx_comp._rtp_bt_patched = True

            if not getattr(self.indexer, "_rtp_score_bt_patched", False):
                _orig_score = self.indexer.indexer_score_topk
                def _patched_score(q_fp8, weights, topk, _orig=_orig_score):
                    from atom.utils.forward_context import get_forward_context
                    fc = get_forward_context()
                    md = fc.attn_metadata
                    saved_bt = md.block_tables
                    idx_bt = getattr(md, "_indexer_block_tables", saved_bt)
                    md.block_tables = idx_bt
                    try:
                        return _orig(q_fp8, weights, topk)
                    finally:
                        md.block_tables = saved_bt
                self.indexer.indexer_score_topk = _patched_score
                self.indexer._rtp_score_bt_patched = True

    # State persistence disabled: fused_compress_attn requires contiguous state,
    # but pool views are non-contiguous (interleaved [kv,score] layout).
    # Shadow buffers from _ensure_v4_native_buffers are used instead (contiguous).
    # TODO: implement custom copy kernel for pool ↔ shadow buffer sync.

    # Guard: if state_slot_mapping contains -1, block not allocated (dummy/probe request)
    ssm = getattr(attn_md, "state_slot_mapping", None)
    if ssm is not None and ssm.numel() > 0 and int(ssm.min()) < 0:
        return torch.zeros_like(x)

    try:
        # Prefill: expand unified_kv with compress region so CSA/HCA prefill
        # attention can read compressed entries. Decode: handled by decode patch.
        _compress_kv = getattr(self, "_rtp_compress_kv", None)
        _do_cat = (fc.context.is_prefill and _compress_kv is not None
                   and _compress_kv.numel() > 0 and ratio != 0)
        _saved_unified = self.unified_kv
        _saved_swa_kv = self.swa_kv
        _saved_comp_kv = None
        if _do_cat:
            _sp = _saved_unified.shape[0]
            _full = torch.cat([_saved_unified, _compress_kv], dim=0)
            self.unified_kv = _full
            self.swa_kv = _full[:_sp].reshape(-1, self.window_size, self.head_dim)
            _comp = getattr(self, "compressor", None)
            if _comp is not None:
                _saved_comp_kv = _comp.kv_cache
                _comp.kv_cache = _full[_sp:].reshape(-1, self.window_size // ratio, self.head_dim)

        # Prefill: temporarily grow compressor state if needed (for block_id indexing).
        # Restored after forward_impl to preserve graph-stable addresses.
        _saved_kv_state = None
        _saved_score_state = None
        _pf_comp = getattr(self, "compressor", None)
        if _pf_comp is not None and ratio != 0:
            _pf_ssm = getattr(attn_md, "state_slot_mapping", None)
            _pf_max_slot = int(_pf_ssm.max()) + 1 if _pf_ssm is not None and _pf_ssm.numel() > 0 else 0
            if _pf_max_slot > _pf_comp.kv_state.shape[0]:
                _saved_kv_state = _pf_comp.kv_state
                _saved_score_state = _pf_comp.score_state
                _pf_comp.kv_state = torch.zeros(
                    _pf_max_slot, _pf_comp.kv_state.shape[1], _pf_comp.kv_state.shape[2],
                    dtype=torch.float32, device=x.device)
                _pf_comp.score_state = torch.full(
                    (_pf_max_slot, _pf_comp.score_state.shape[1], _pf_comp.score_state.shape[2]),
                    float('-inf'), dtype=torch.float32, device=x.device)
            else:
                # Buffer already large enough — zero ALL state to clear any
                # stale compressor state from a previous request. Full reset
                # is needed because decode uses compact indices [0..bs-1]
                # while prefill uses block_id indices — partial reset would
                # miss the stale compact-indexed state left by prior decode.
                _reset_v4_state_all(self)
            # Also check indexer compressor
            if ratio == 4:
                _pf_idx = getattr(self, "indexer", None)
                _pf_idx_comp = getattr(_pf_idx, "compressor", None) if _pf_idx else None
                if _pf_idx_comp is not None and _pf_max_slot > _pf_idx_comp.kv_state.shape[0]:
                    _pf_idx_comp._saved_kv = _pf_idx_comp.kv_state
                    _pf_idx_comp._saved_sc = _pf_idx_comp.score_state
                    _pf_idx_comp.kv_state = torch.zeros(
                        _pf_max_slot, _pf_idx_comp.kv_state.shape[1], _pf_idx_comp.kv_state.shape[2],
                        dtype=torch.float32, device=x.device)
                    _pf_idx_comp.score_state = torch.full(
                        (_pf_max_slot, _pf_idx_comp.score_state.shape[1], _pf_idx_comp.score_state.shape[2]),
                        float('-inf'), dtype=torch.float32, device=x.device)

        # --- Eager mode: use compact swa_kv + gather/scatter (same as graph) ---
        # NOTE: This block is now ONLY reached by PREFILL (decode goes through
        # the Triton path above). Prefill uses pool view directly — no compact
        # override or gather/scatter needed.

        result = self.forward_impl(x, positions)

        # --- Prefill STATE scatter: shadow kv_state → STATE pool ---
        # Prefill wrote compressor state to shadow buffer at kv_state[state_slot].
        # Scatter to STATE pool so decode's gather can read the correct state.
        if ratio != 0:
            from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import CSA_STATE, HCA_STATE
            _pf_compressor = getattr(self, "compressor", None)
            if _pf_compressor is not None:
                _pf_state_region = CSA_STATE if ratio == 4 else HCA_STATE
                _pf_state_bt = v4_block_tables.get(_pf_state_region)
                kv_cache_data_pf = fc.kv_cache_data
                _pf_cache_entry = kv_cache_data_pf.get(f"layer_{self.layer_id}") if kv_cache_data_pf else None
                if _pf_state_bt is not None and _pf_cache_entry is not None and _pf_cache_entry.k_cache:
                    _pf_pool_name = "CSA_STATE" if ratio == 4 else "HCA_STATE"
                    _pf_sp = _pf_cache_entry.k_cache.get(_pf_pool_name)
                    if _pf_sp is not None:
                        _pf_pool_raw = _pf_sp.kv_cache_base.view(torch.float32)
                        _pf_n_blocks = _pf_pool_raw.shape[0]
                        _pf_elems = _pf_pool_raw.numel() // _pf_n_blocks
                        _pf_pool_view = _pf_pool_raw.reshape(_pf_n_blocks, _pf_elems)
                        _pf_half = _pf_elems // 2
                        _pf_ring = _pf_compressor.kv_state.shape[1]
                        _pf_dim = _pf_compressor.kv_state.shape[2]
                        if _pf_half == _pf_ring * _pf_dim:
                            _pf_ssm = getattr(attn_md, "state_slot_mapping", None)
                            _pf_bs = int(_pf_ssm.numel()) if _pf_ssm is not None else 0
                            if _pf_bs > 0:
                                _pf_state_bids = _pf_state_bt[:_pf_bs, 0].to(torch.int64)
                                _pf_ssm_long = _pf_ssm[:_pf_bs].to(torch.int64)
                                _pf_kv = _pf_compressor.kv_state[_pf_ssm_long].reshape(_pf_bs, -1)
                                _pf_sc = _pf_compressor.score_state[_pf_ssm_long].reshape(_pf_bs, -1)
                                _pf_combined = torch.cat([_pf_kv, _pf_sc], dim=-1)
                                _pf_pool_view[_pf_state_bids] = _pf_combined

        if _do_cat:
            _compress_kv.copy_(_full[_sp:_sp + _compress_kv.shape[0]])
            _saved_unified.copy_(_full[:_sp])
            self.unified_kv = _saved_unified
            self.swa_kv = _saved_swa_kv
            if _saved_comp_kv is not None:
                getattr(self, "compressor").kv_cache = _saved_comp_kv

        # Restore graph-stable compressor state references
        if _saved_kv_state is not None:
            _pf_comp.kv_state = _saved_kv_state
            _pf_comp.score_state = _saved_score_state
        if ratio == 4:
            _pf_idx = getattr(self, "indexer", None)
            _pf_idx_comp = getattr(_pf_idx, "compressor", None) if _pf_idx else None
            if _pf_idx_comp is not None and hasattr(_pf_idx_comp, "_saved_kv"):
                _pf_idx_comp.kv_state = _pf_idx_comp._saved_kv
                _pf_idx_comp.score_state = _pf_idx_comp._saved_sc
                del _pf_idx_comp._saved_kv, _pf_idx_comp._saved_sc

        return result
    except Exception as e:
        logger.error("V4 fwd layer %d (ratio=%d): %s", self.layer_id, ratio, e, exc_info=True)
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
    global _PATCHED, _original_v4_forward, _original_paged_decode
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

    # Patch decode kernel for compress_kv cat support.
    # Must patch BOTH the kernel module AND the deepseek_v4 model module,
    # because deepseek_v4.py uses `from ... import sparse_attn_v4_paged_decode`
    # which creates a local reference that module-level patching doesn't update.
    try:
        import atom.model_ops.v4_kernels.paged_decode as _pd
        import atom.models.deepseek_v4 as _dsv4
        _original_paged_decode = _pd.sparse_attn_v4_paged_decode
        _pd.sparse_attn_v4_paged_decode = _patched_sparse_attn_v4_paged_decode
        _dsv4.sparse_attn_v4_paged_decode = _patched_sparse_attn_v4_paged_decode
        logger.info("Applied decode kernel patch (module + model refs).")
    except ImportError:
        logger.warning("Cannot import paged_decode — decode patch skipped")

    logger.info("Applied RTP-LLM V4 attention patch (forward level) for multi-region KV cache.")
    _PATCHED = True

