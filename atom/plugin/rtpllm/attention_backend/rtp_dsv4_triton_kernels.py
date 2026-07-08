"""Dual-ptr paged decode Triton kernels for the V4 plugin.

Self-contained leaf module: these kernels read SWA entries from swa_kv and
compress entries from compress_kv without any copy/allocation. Currently
implemented but not wired into the forward path (see status doc 9.x).
"""

import torch
import triton
import triton.language as tl

from atom.plugin.rtpllm.attention_backend.rtp_dsv4_constants import LOG2E

# ---------------------------------------------------------------------------
# Dual-ptr paged decode kernel (plugin-only, does NOT modify ATOM native code)
# ---------------------------------------------------------------------------


@triton.jit
def _dual_ptr_paged_decode_fused_kernel(
    q_ptr,
    swa_kv_ptr,  # [swa_pages, D] bf16
    compress_kv_ptr,  # [compress_pages, D] bf16
    kv_indices_ptr,  # [total_indices] int32
    kv_indptr_ptr,  # [N+1] int32
    attn_sink_ptr,  # [H]
    out_ptr,  # [N, H, D]
    q_stride_t,
    q_stride_h,
    q_stride_d,
    swa_stride_n,
    swa_stride_d,
    compress_stride_n,
    compress_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
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
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
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
            swa_kv_ptr
            + swa_slot[:, None] * swa_stride_n
            + d_offs[None, :] * swa_stride_d,
            mask=valid[:, None] & d_mask[None, :] & is_swa[:, None],
            other=0.0,
        )
        compress_data = tl.load(
            compress_kv_ptr
            + compress_slot[:, None] * compress_stride_n
            + d_offs[None, :] * compress_stride_d,
            mask=valid[:, None] & d_mask[None, :] & (~is_swa)[:, None],
            other=0.0,
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

    sink_raw = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(
        tl.float32
    )
    sink = sink_raw * log2e
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0, (acc * alpha_kv[:, None]) / denom[:, None], 0.0
    )
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
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
        q.stride(0),
        q.stride(1),
        q.stride(2),
        swa_kv.stride(0),
        swa_kv.stride(1) if swa_kv.dim() > 1 else 1,
        compress_kv.stride(0),
        compress_kv.stride(1) if compress_kv.dim() > 1 else 1,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        qk_scale,
        LOG2E,
        SWA_PAGES=swa_pages,
        H=H,
        D=D,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return out
