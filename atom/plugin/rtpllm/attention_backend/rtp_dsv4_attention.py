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

import logging
import os
from typing import Any, Dict

import torch

from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
    SWA_KV,
    CSA_KV,
    HCA_KV,
    INDEXER_KV,
    CSA_STATE,
    HCA_STATE,
)
from atom.plugin.rtpllm.utils.v4_observability import rate_limited_log
from atom.plugin.rtpllm.attention_backend.rtp_dsv4_constants import (
    _ATOM_INDEXER_FP8_ENTRY_BYTES,
    _RTP_INDEXER_BF16_ENTRY_BYTES,
    _RTP_INDEXER_FP8_ENTRY_BYTES,
    _V4_META_BUILT_ATTR,
    _V4_META_FAILED_ATTR,
    _V4_BUFFERS_ALLOCATED,
)
from atom.plugin.rtpllm.attention_backend.rtp_dsv4_metadata import (
    _build_eager_decode_with_triton,
    _build_v4_per_forward_metadata,
)
from atom.plugin.rtpllm.attention_backend.rtp_dsv4_spec import (
    DSV4_CSA_RATIO,
    DSV4_HCA_RATIO,
    DSV4_DEFAULT_INDEX_TOPK,
    DSV4_DEFAULT_INDEX_HEAD_DIM,
    DSV4_INDEX_SCALE_BYTES,
    DSV4_INDEX_ENTRY_ALIGNMENT,
    DSV4_MIN_NATIVE_STATE_SLOTS,
)

logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.rtp_dsv4_attention")


# One-time (per process) guard for the INDEXER_KV binding-mode banner, so an
# operator can confirm from the log whether the ROCm ATOM 144B direct-bind is
# active even with INFO/debug logs off — logged exactly once, not per CSA layer.
_INDEXER_BIND_LOGGED = False


def _indexer_fp8_144_kv_requested() -> bool:
    """True iff the ROCm ATOM 144B INDEXER_KV direct-bind was requested via env."""
    return os.environ.get("ROCM_ATOM_DSV4_INDEXER_FP8_KV_CACHE", "0") in (
        "1",
        "true",
        "True",
        "ON",
        "on",
    )


# Saved references for monkey-patch
_original_paged_decode = None
_original_paged_prefill = None


class V4AttentionRuntimeError(RuntimeError):
    """Fatal V4 plugin error for a real inference request."""


def _is_dummy_forward_context(fc: Any) -> bool:
    return bool(getattr(getattr(fc, "context", None), "is_dummy_run", False))


def _raise_or_zero(
    fc: Any,
    x: torch.Tensor,
    message: str,
    cause: Exception | None = None,
) -> torch.Tensor:
    """Only explicit dummy/warmup forwards may degrade to a zero tensor."""
    if _is_dummy_forward_context(fc):
        logger.warning("%s; returning zeros for dummy/warmup forward", message)
        return torch.zeros_like(x)
    error = V4AttentionRuntimeError(message)
    if cause is not None:
        raise error from cause
    raise error


def _patched_sparse_attn_v4_paged_decode(
    q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, kv_scales=None
):
    """Monkey-patch wrapper: intercepts decode kernel calls in plugin mode.

    Uses the kernel's native dual-pointer (SPLIT_KV) mode to read from
    separate SWA and compress pools without any buffer copy or allocation.
    Falls back to torch.cat for eager non-graph mode (backward compat).
    """

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
            q,
            unified_kv,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            kv_scales=kv_scales,
            compress_kv=compress_kv,
            swa_pages=swa_pages,
        )

    # --- No compress needed (dense/SWA-only layers): pass through ---
    return _original_paged_decode(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        softmax_scale,
        kv_scales=kv_scales,
    )


_PATCHED = False


def _ensure_v4_native_buffers(
    attn_module: Any, num_slots: int, device: torch.device
) -> None:
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
            num_slots,
            window_size,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )

    # Resize compressor state buffers
    compressor = getattr(attn_module, "compressor", None)
    if compressor is not None and ratio > 0:
        overlap = 1 if ratio == DSV4_CSA_RATIO else 0
        coff = 1 + overlap
        state_dim0 = coff * ratio
        state_dim1 = coff * head_dim

        if compressor.kv_state.shape[0] < num_slots:
            compressor.kv_state = torch.zeros(
                num_slots,
                state_dim0,
                state_dim1,
                dtype=torch.float32,
                device=device,
            )
            compressor.score_state = torch.full(
                (num_slots, state_dim0, state_dim1),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )

        # compressor.kv_cache: will be bound from RTP-LLM pool in _bind step
        # (shadow fallback only if pool not available)
        if compressor.kv_cache is None:
            k_per_block = window_size // ratio
            compressor.kv_cache = torch.zeros(
                num_slots,
                k_per_block,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
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
                    num_slots,
                    idx_state_dim0,
                    idx_state_dim1,
                    dtype=torch.float32,
                    device=device,
                )
                idx_compressor.score_state = torch.full(
                    (num_slots, idx_state_dim0, idx_state_dim1),
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )
        if indexer.kv_cache is None:
            k_per_block = window_size // ratio
            aligned_dim = (
                (idx_head_dim + DSV4_INDEX_SCALE_BYTES + DSV4_INDEX_ENTRY_ALIGNMENT - 1)
                // DSV4_INDEX_ENTRY_ALIGNMENT
            ) * DSV4_INDEX_ENTRY_ALIGNMENT
            indexer.kv_cache = torch.zeros(
                num_slots,
                k_per_block,
                aligned_dim,
                dtype=torch.bfloat16,
                device=device,
            )

    # unified_kv: SWA-only view (compress region added via pool bind if available)
    swa_pages = num_slots * window_size
    attn_module.unified_kv = torch.zeros(
        swa_pages,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    setattr(attn_module, _V4_BUFFERS_ALLOCATED, True)
    # Persist compact swa_kv reference (survives bind overrides).
    # IMPORTANT: Only set on FIRST allocation. If _compact_swa_kv already exists
    # (from graph warmup), do NOT overwrite — graph replay uses the original
    # address captured during graph capture. Overwriting with a resized tensor
    # would invalidate the captured address → precision errors.
    if (
        not hasattr(attn_module, "_compact_swa_kv")
        or attn_module._compact_swa_kv is None
    ):
        attn_module._compact_swa_kv = attn_module.swa_kv


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
    if (
        compact_swa is not None
        and compact_swa is not swa
        and isinstance(compact_swa, torch.Tensor)
    ):
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

    # Compress KV: zero-copy view of CSA/HCA pool (for decode-time cat)
    compress_kv = None
    if ratio == DSV4_CSA_RATIO:
        csa_pool = layer_pools.get("CSA_KV")
        if csa_pool is not None:
            compress_kv = csa_pool.kv_cache_base.view(torch.bfloat16).reshape(
                -1, head_dim
            )
            # Cache for graph-mode fallback
    elif ratio == DSV4_HCA_RATIO:
        hca_pool = layer_pools.get("HCA_KV")
        if hca_pool is not None:
            compress_kv = hca_pool.kv_cache_base.view(torch.bfloat16).reshape(
                -1, head_dim
            )
            # Cache for graph-mode fallback

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
    if ratio == DSV4_CSA_RATIO:
        kv_pool = layer_pools.get("CSA_KV")
        k_per_block = win // ratio  # entries per block = 128/4 = 32
    elif ratio == DSV4_HCA_RATIO:
        kv_pool = layer_pools.get("HCA_KV")
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


def _try_bind_v4_indexer_rtp_pool_144(
    attn_module: Any,
    indexer: Any,
    kv_pool: Any,
    fp8_dtype: torch.dtype,
    idx_head_dim: int,
    aligned_dim: int,
) -> bool:
    """Bind ROCm ATOM 144B INDEXER_KV pool directly when available."""
    if torch.version.hip is None:
        return False
    if not _indexer_fp8_144_kv_requested():
        return False

    base = getattr(kv_pool, "kv_cache_base", None)
    if base is None or base.dim() != 2 or base.numel() == 0:
        return False

    stride_bytes = int(base.shape[1]) * int(base.element_size())
    if stride_bytes % _ATOM_INDEXER_FP8_ENTRY_BYTES != 0:
        return False
    entries_per_block = stride_bytes // _ATOM_INDEXER_FP8_ENTRY_BYTES
    if entries_per_block <= 0 or aligned_dim != _ATOM_INDEXER_FP8_ENTRY_BYTES:
        return False

    raw_u8 = base.view(torch.uint8)
    if int(raw_u8.shape[1]) < stride_bytes:
        return False

    num_blocks = int(raw_u8.shape[0])
    kv_u8 = raw_u8.as_strided(
        size=(num_blocks, entries_per_block, aligned_dim),
        stride=(stride_bytes, aligned_dim, 1),
    )
    kv_cache = kv_u8.view(fp8_dtype)
    indexer.kv_cache = kv_cache
    indexer._rtp_idx_kv_pool_144 = kv_cache

    idx_compressor = getattr(indexer, "compressor", None)
    if idx_compressor is not None:
        idx_compressor.kv_cache = kv_cache
        block_fp32_stride = stride_bytes // 4
        scale_fp32_offset = (entries_per_block * idx_head_dim) // 4
        idx_compressor.cache_scale = (
            kv_cache.view(torch.float32)
            .view(-1)
            .as_strided(
                size=(num_blocks, entries_per_block),
                stride=(block_fp32_stride, 1),
                storage_offset=scale_fp32_offset,
            )
        )

    global _INDEXER_BIND_LOGGED
    if not _INDEXER_BIND_LOGGED:
        # Printed once per process at WARNING so it stays visible with INFO/debug
        # logs off — positive confirmation the 144B direct-bind is active (no
        # code path here is an error).
        logger.warning(
            "V4 INDEXER_KV 144B direct-bind ACTIVE (binding=rtp_pool_144) "
            "first_layer=%s blocks=%d entries_per_block=%d stride_bytes=%d",
            getattr(attn_module, "layer_id", "?"),
            num_blocks,
            entries_per_block,
            stride_bytes,
        )
        _INDEXER_BIND_LOGGED = True
    return True


def _infer_v4_indexer_entries_per_block(
    kv_pool: Any,
    aligned_dim: int,
    fallback_entries: int,
) -> int:
    base = getattr(kv_pool, "kv_cache_base", None)
    if base is None or base.dim() != 2 or base.numel() == 0:
        return fallback_entries

    stride_bytes = int(base.shape[1]) * int(base.element_size())
    for entry_bytes in (
        _ATOM_INDEXER_FP8_ENTRY_BYTES,
        _RTP_INDEXER_FP8_ENTRY_BYTES,
        _RTP_INDEXER_BF16_ENTRY_BYTES,
    ):
        if stride_bytes % entry_bytes == 0:
            entries_per_block = stride_bytes // entry_bytes
            if entries_per_block > 0:
                return entries_per_block

    if stride_bytes % aligned_dim == 0:
        entries_per_block = stride_bytes // aligned_dim
        if entries_per_block > 0:
            return entries_per_block

    logger.warning(
        "V4 INDEXER_KV could not infer entries_per_block from stride_bytes=%d; using fallback=%d",
        stride_bytes,
        fallback_entries,
    )
    return fallback_entries


def _bind_v4_indexer_views(
    attn_module: Any,
    layer_pools: Dict[str, Any],
) -> None:
    """Bind indexer kv_cache from RTP 144B pool or FP8 shadow buffer.

    Indexer scoring kernel (top_k_per_row_decode → fp8_paged_mqa_logits)
    requires FP8 kv_cache regardless of RTP-LLM's kv_cache_dtype setting.
    ROCm ATOM plugin can request a dedicated 144B INDEXER_KV pool and bind it
    zero-copy; otherwise we allocate a per-layer FP8 contiguous shadow buffer.

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

    idx_head_dim = getattr(indexer, "head_dim", DSV4_DEFAULT_INDEX_HEAD_DIM)
    ratio = attn_module.compress_ratio
    window_size = attn_module.window_size
    aligned_dim = (
        (idx_head_dim + DSV4_INDEX_SCALE_BYTES + DSV4_INDEX_ENTRY_ALIGNMENT - 1)
        // DSV4_INDEX_ENTRY_ALIGNMENT
    ) * DSV4_INDEX_ENTRY_ALIGNMENT

    kv_pool = layer_pools.get("INDEXER_KV")
    if kv_pool is None:
        return

    if _try_bind_v4_indexer_rtp_pool_144(
        attn_module,
        indexer,
        kv_pool,
        fp8_dtype,
        idx_head_dim,
        aligned_dim,
    ):
        return

    k1 = _infer_v4_indexer_entries_per_block(
        kv_pool,
        aligned_dim,
        window_size // ratio,
    )
    NB = kv_pool.kv_cache_base.shape[0]

    # Use max of INDEXER_KV and CSA_KV block counts
    csa_pool = layer_pools.get("CSA_KV")
    if csa_pool is not None:
        NB = max(NB, csa_pool.kv_cache_base.shape[0])

    # Allocate FP8 shadow once per layer
    shadow_kv = getattr(indexer, "_rtp_idx_kv_shadow", None)
    if shadow_kv is None or shadow_kv.shape[0] < NB:
        shadow_kv = torch.zeros(
            NB, k1, aligned_dim, dtype=fp8_dtype, device=kv_pool.kv_cache_base.device
        )
        indexer._rtp_idx_kv_shadow = shadow_kv

    # Bind kv_cache for both Indexer and its inner Compressor
    indexer.kv_cache = shadow_kv

    global _INDEXER_BIND_LOGGED
    if not _INDEXER_BIND_LOGGED and _indexer_fp8_144_kv_requested():
        # env requested the 144B direct-bind but we fell back to the shadow —
        # an unexpected degradation worth one WARNING. When the env is NOT set,
        # shadow is the intended path, so stay silent (no false alarm).
        logger.warning(
            "V4 INDEXER_KV requested 144B direct-bind but FELL BACK to shadow "
            "(binding=shadow_fallback) first_layer=%s blocks=%d entries_per_block=%d "
            "— check INDEXER_KV pool layout / global fp8_kv_cache",
            getattr(attn_module, "layer_id", "?"),
            NB,
            k1,
        )
        _INDEXER_BIND_LOGGED = True

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


def _bind_v4_layer_pools(attn_module, cache_entry, ratio):
    """Bind a layer's RTP pool views onto the ATOM V4 attention module.

    Binds the SWA/compress cache views, the main compressor's CSA/HCA kv_cache,
    and the indexer (144B direct-bind or FP8 shadow). Byte-identical across the
    graph, eager-decode and prefill forward paths; each caller wraps this in its
    own try/except with a path-specific fallback. Raises on any bind failure.
    """
    _bind_v4_kv_cache_views(attn_module, cache_entry.k_cache)
    compressor = getattr(attn_module, "compressor", None)
    if compressor is not None and ratio != 0:
        head_dim = attn_module.head_dim
        if ratio == DSV4_CSA_RATIO:
            csa_pool = cache_entry.k_cache.get("CSA_KV")
            if csa_pool is not None:
                compressor.kv_cache = csa_pool.kv_cache_base.view(
                    torch.bfloat16
                ).reshape(-1, attn_module.window_size // ratio, head_dim)
        elif ratio == DSV4_HCA_RATIO:
            hca_pool = cache_entry.k_cache.get("HCA_KV")
            if hca_pool is not None:
                compressor.kv_cache = hca_pool.kv_cache_base.view(
                    torch.bfloat16
                ).reshape(-1, attn_module.window_size // ratio, head_dim)
    if ratio == DSV4_CSA_RATIO:
        _bind_v4_indexer_views(attn_module, cache_entry.k_cache)


def _v4_decode_state_gather(
    attn_module, ratio, active_bs, v4_block_tables, cache_entry
):
    """Gather compressor kv_state/score_state from the RTP STATE pool into the
    compact per-slot buffers before forward_impl.

    Shared by the eager and CUDA-graph decode paths (identical logic). Returns
    ``(state_pool_view, state_block_ids)`` for the matching scatter, or
    ``(None, None)`` when there is nothing to sync (dense layer, empty batch,
    missing pool, or a state layout that does not match the compact buffers).
    """
    compressor = getattr(attn_module, "compressor", None)
    if compressor is None or ratio == 0 or active_bs <= 0:
        return None, None
    state_region = CSA_STATE if ratio == DSV4_CSA_RATIO else HCA_STATE
    state_bt = v4_block_tables.get(state_region)
    if state_bt is None or cache_entry is None or not cache_entry.k_cache:
        return None, None
    sp = cache_entry.k_cache.get(
        "CSA_STATE" if ratio == DSV4_CSA_RATIO else "HCA_STATE"
    )
    if sp is None:
        return None, None
    pool_raw = sp.kv_cache_base.view(torch.float32)
    n_blocks = pool_raw.shape[0]
    elems = pool_raw.numel() // n_blocks
    pool_view = pool_raw.reshape(n_blocks, elems)
    block_ids = state_bt[:active_bs, 0].to(torch.int64)
    half = elems // 2
    ring = compressor.kv_state.shape[1]
    dim = compressor.kv_state.shape[2]
    if half != ring * dim:
        return None, None
    gathered = pool_view[block_ids]
    compressor.kv_state[:active_bs] = gathered[:, :half].reshape(active_bs, ring, dim)
    compressor.score_state[:active_bs] = gathered[:, half:].reshape(
        active_bs, ring, dim
    )
    return pool_view, block_ids


def _v4_decode_state_scatter(attn_module, state_pool_view, state_block_ids, active_bs):
    """Scatter compact compressor kv_state/score_state back to the RTP STATE
    pool after forward_impl. No-op when the paired gather returned (None, None).
    """
    compressor = getattr(attn_module, "compressor", None)
    if state_pool_view is None or state_block_ids is None or compressor is None:
        return
    kv_flat = compressor.kv_state[:active_bs].reshape(active_bs, -1)
    sc_flat = compressor.score_state[:active_bs].reshape(active_bs, -1)
    state_pool_view[state_block_ids] = torch.cat([kv_flat, sc_flat], dim=-1)


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
        _ensure_v4_native_buffers(
            self, num_slots=max(max_bs, DSV4_MIN_NATIVE_STATE_SLOTS), device=x.device
        )

    # 2. Bind KV cache from RTP-LLM pool (for compress_kv, compressor, pool_swa)
    kv_cache_data = fc.kv_cache_data
    if kv_cache_data is None:
        _rt = bufs.get("_kv_cache_ref_runtime")
        if _rt is not None:
            kv_cache_data = getattr(_rt, "_rtp_kv_cache_data", None)
    cache_entry = kv_cache_data.get(f"layer_{self.layer_id}") if kv_cache_data else None
    if cache_entry and isinstance(cache_entry.k_cache, dict) and cache_entry.k_cache:
        try:
            _bind_v4_layer_pools(self, cache_entry, ratio)
        except Exception as e:
            rate_limited_log(
                f"v4_fallback:graph_bind:L{self.layer_id}",
                logging.ERROR,
                "graph bind failed — layer %d output ZEROED (garbage): %s",
                self.layer_id,
                e,
                exc_info_first=True,
            )
            return _raise_or_zero(
                fc,
                x,
                f"graph KV-pool bind failed for layer {self.layer_id}",
                e,
            )

    # After bind: override swa_kv and unified_kv with compact buffer.
    # This ensures swa_write and paged_decode operate on the SAME independent
    # memory (not the pool). Pool data is synced via gather/scatter.
    _compact_swa = getattr(self, "_compact_swa_kv", None)
    if _compact_swa is not None:
        self.swa_kv = _compact_swa
        self.unified_kv = _compact_swa.view(-1, self.head_dim)

    if (
        (cache_entry is None or not cache_entry.k_cache)
        and ratio != 0
        and getattr(self, "_rtp_compress_kv", None) is None
    ):
        # Capture may not have a ForwardContext cache entry yet. Pool views are
        # runtime-scoped so a model reload cannot reuse another runtime's memory.
        head_dim = self.head_dim
        win = self.window_size
        pool_views = bufs.get("_pool_views", {})
        swa_flat_cache = pool_views.get("swa")
        csa_compress_cache = pool_views.get("csa")
        hca_compress_cache = pool_views.get("hca")
        if swa_flat_cache is not None:
            self.unified_kv = swa_flat_cache
            self.swa_kv = swa_flat_cache.view(-1, win, head_dim)
            self._rtp_swa_pages = swa_flat_cache.shape[0]
        if ratio == DSV4_CSA_RATIO and csa_compress_cache is not None:
            self._rtp_compress_kv = csa_compress_cache
            compressor = getattr(self, "compressor", None)
            if compressor is not None:
                k_per_block = win // ratio  # 32
                compressor.kv_cache = csa_compress_cache.view(-1, k_per_block, head_dim)
        elif ratio == DSV4_HCA_RATIO and hca_compress_cache is not None:
            self._rtp_compress_kv = hca_compress_cache
            compressor = getattr(self, "compressor", None)
            if compressor is not None:
                k_per_block = win // ratio  # 1
                compressor.kv_cache = hca_compress_cache.view(-1, k_per_block, head_dim)
        else:
            return _raise_or_zero(
                fc,
                x,
                f"missing runtime-scoped graph compress pool for layer "
                f"{self.layer_id} ratio {ratio}",
            )
    elif getattr(self, "swa_kv", None) is None or self.swa_kv.numel() <= 1:
        # Dense layers (ratio=0) also need swa_kv bound for swa_write.
        swa_flat_cache = bufs.get("_pool_views", {}).get("swa")
        if swa_flat_cache is not None:
            head_dim = self.head_dim
            win = self.window_size
            self.unified_kv = swa_flat_cache
            self.swa_kv = swa_flat_cache.view(-1, win, head_dim)
            self._rtp_swa_pages = swa_flat_cache.shape[0]
        else:
            return _raise_or_zero(
                fc,
                x,
                f"missing runtime-scoped graph SWA pool for layer {self.layer_id}",
            )

    # 3. Set metadata fields from pre-allocated buffers
    attn_md.state = AttnState.DECODE
    attn_md.state_slot_mapping = bufs["state_slot"][:active_bs]
    # cu_seqlens_q for decode = arange(0..bs); reuse from _cg_meta_bufs
    # which was pre-allocated as arange in _ensure_cuda_graph_prewarmed.
    max_bs = int(bufs["indptr_swa"].shape[0]) - 1
    attn_md.cu_seqlens_q = bufs.get("_cu_seqlens_q", None)
    if attn_md.cu_seqlens_q is None:
        # Fallback: create once and cache (first capture call)
        attn_md.cu_seqlens_q = torch.arange(
            max_bs + 1, device=x.device, dtype=torch.int32
        )
        bufs["_cu_seqlens_q"] = attn_md.cu_seqlens_q
    attn_md.max_seqlen_q = 1
    attn_md.batch_id_per_token = bufs[
        "batch_id"
    ]  # int32 for forward_impl (qk_norm_rope + csa_translate_pack)
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
    if ratio == DSV4_CSA_RATIO:
        attn_md.indexer_meta = {
            "n_committed_per_seq_gpu": bufs["n_csa"][:active_bs],
        }
        indexer_bt = v4_block_tables.get(INDEXER_KV)
        if indexer_bt is not None:
            attn_md._indexer_block_tables = indexer_bt

    # Set region block_table
    if ratio == 0:
        region_bt = v4_block_tables.get(SWA_KV)
    elif ratio == DSV4_CSA_RATIO:
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
        self.swa_kv[:active_bs].copy_(_pool_swa.index_select(0, _bid))

    # --- STATE pool gather: pool → compact kv_state/score_state ---
    _state_pool_view_g, _state_block_ids_g = _v4_decode_state_gather(
        self, ratio, active_bs, v4_block_tables, cache_entry
    )

    try:
        result = self.forward_impl(x, positions)
    except Exception as e:
        rate_limited_log(
            f"v4_fallback:graph_fwd:L{self.layer_id}",
            logging.ERROR,
            "graph forward failed — layer %d (ratio=%d) output ZEROED (garbage): %s",
            self.layer_id,
            ratio,
            e,
            exc_info_first=True,
        )
        return _raise_or_zero(
            fc,
            x,
            f"graph attention forward failed for layer {self.layer_id} ratio {ratio}",
            e,
        )

    # --- Scatter: compact → pool ---
    if _block_ids is not None and _pool_swa is not None and active_bs > 0:
        _bid = _block_ids[:active_bs]
        _pool_swa.index_copy_(0, _bid, self.swa_kv[:active_bs])
    # STATE scatter
    _v4_decode_state_scatter(self, _state_pool_view_g, _state_block_ids_g, active_bs)

    return result


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
    from atom.utils.forward_context import get_forward_context

    fc = get_forward_context()
    attn_md = fc.attn_metadata
    v4_block_tables = getattr(attn_md, "v4_block_tables", None)

    if v4_block_tables is None:
        return _original_v4_forward(self, x, positions)

    # --- CUDA Graph fast path ---
    if getattr(attn_md, "_v4_cuda_graph_mode", False):
        return _v4_forward_cuda_graph(self, x, positions, fc, attn_md)

    # --- Eager (non-graph) path ---
    # Build V4 metadata once per forward (first layer triggers)
    if getattr(attn_md, _V4_META_FAILED_ATTR, False):
        return _raise_or_zero(
            fc,
            x,
            f"V4 metadata is invalid for layer {self.layer_id}",
        )
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
            attn_md._index_topk = (
                getattr(_m_args, "index_topk", DSV4_DEFAULT_INDEX_TOPK)
                if _m_args
                else DSV4_DEFAULT_INDEX_TOPK
            )

            # DECODE: use Triton kernels for index construction (same as graph mode)
            _is_eager_prefill = bool(getattr(rtp_attn_inputs, "is_prefill", True))
            if not _is_eager_prefill:
                _triton_ok = _build_eager_decode_with_triton(
                    attn_md,
                    rtp_attn_inputs,
                    v4_ratios,
                    v4_block_tables,
                    region_to_group,
                    x.device,
                    window_size=self.window_size,
                    pool_swa_pages=getattr(self, "_rtp_swa_pages", 0),
                    index_topk=attn_md._index_topk,
                )
                if not _triton_ok:
                    setattr(attn_md, _V4_META_BUILT_ATTR, True)
                    setattr(attn_md, _V4_META_FAILED_ATTR, True)
                    rate_limited_log(
                        f"v4_fallback:eager_meta_triton:L{self.layer_id}",
                        logging.ERROR,
                        "eager decode Triton metadata build returned False — "
                        "layer %d output ZEROED (garbage)",
                        self.layer_id,
                    )
                    return _raise_or_zero(
                        fc,
                        x,
                        f"eager decode metadata build failed for layer {self.layer_id}",
                    )
            else:
                # PREFILL: use original CPU metadata construction
                _build_v4_per_forward_metadata(
                    attn_md,
                    rtp_attn_inputs,
                    v4_ratios,
                    v4_block_tables,
                    region_to_group,
                    x.device,
                    window_size=self.window_size,
                    pool_swa_pages=getattr(self, "_rtp_swa_pages", 0),
                )
        except Exception as e:
            rate_limited_log(
                f"v4_fallback:eager_meta_build:L{self.layer_id}",
                logging.ERROR,
                "eager metadata construction failed — layer %d output ZEROED "
                "(garbage): %s",
                self.layer_id,
                e,
                exc_info_first=True,
            )
            setattr(attn_md, _V4_META_BUILT_ATTR, True)
            setattr(attn_md, _V4_META_FAILED_ATTR, True)
            return _raise_or_zero(
                fc,
                x,
                f"eager metadata construction failed for layer {self.layer_id}",
                e,
            )

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
            _ensure_v4_native_buffers(
                self,
                num_slots=max(active_bs, DSV4_MIN_NATIVE_STATE_SLOTS),
                device=x.device,
            )

        # Bind KV cache from pool (compress_kv, compressor.kv_cache, pool_swa)
        kv_cache_data = fc.kv_cache_data
        cache_entry = (
            kv_cache_data.get(f"layer_{self.layer_id}") if kv_cache_data else None
        )
        if (
            cache_entry
            and isinstance(cache_entry.k_cache, dict)
            and cache_entry.k_cache
        ):
            try:
                _bind_v4_layer_pools(self, cache_entry, ratio)
            except Exception as e:
                rate_limited_log(
                    f"v4_fallback:eager_bind:L{self.layer_id}",
                    logging.ERROR,
                    "eager decode bind failed — layer %d output ZEROED (garbage): %s",
                    self.layer_id,
                    e,
                    exc_info_first=True,
                )
                return _raise_or_zero(
                    fc,
                    x,
                    f"eager decode KV-pool bind failed for layer {self.layer_id}",
                    e,
                )

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
        elif ratio == DSV4_CSA_RATIO:
            region_bt = v4_block_tables.get(CSA_KV)
        else:
            region_bt = v4_block_tables.get(HCA_KV)
        if region_bt is not None:
            attn_md.block_tables = region_bt

        # Indexer block_table patch (CSA layers)
        if ratio == DSV4_CSA_RATIO:
            from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import INDEXER_KV

            indexer_bt = v4_block_tables.get(INDEXER_KV)
            if indexer_bt is not None:
                attn_md._indexer_block_tables = indexer_bt
            if self.indexer is not None:
                idx_comp = getattr(self.indexer, "compressor", None)
                if idx_comp is not None and not getattr(
                    idx_comp, "_rtp_bt_patched", False
                ):
                    _orig_comp_fwd = idx_comp.forward

                    def _patched_comp_fwd(
                        x,
                        plan,
                        state_slot_mapping,
                        block_tables=None,
                        _orig=_orig_comp_fwd,
                    ):
                        from atom.utils.forward_context import get_forward_context

                        md = get_forward_context().attn_metadata
                        bt = getattr(md, "_indexer_block_tables", block_tables)
                        return _orig(
                            x,
                            plan=plan,
                            state_slot_mapping=state_slot_mapping,
                            block_tables=bt,
                        )

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
        # RTP-LLM manages STATE pool lifecycle (alloc/free/zero-init). We gather
        # state BEFORE forward_impl so the compressor reads correct accumulated
        # state, then scatter back after. Shared with the CUDA-graph path.
        _state_pool_view, _state_block_ids = _v4_decode_state_gather(
            self, ratio, active_bs, v4_block_tables, cache_entry
        )

        _pool_swa = getattr(self, "_rtp_pool_swa_kv", None)

        # One-shot multi-block SWA seed at the first decode step of each request
        # (mirrors the CUDA-graph seed). REQUIRED for a prompt > win: the live
        # window spans cur_col/prev_col, but the per-step col0 gather below skips
        # freed-col0 rows (pos>=2*win) and reads the wrong block for pos in
        # [win,2*win). Without this a long-prompt slot's compact ring is never
        # seeded from the pool -> it serves stale / another slot's data in a bs>1
        # batch (cross-request contamination + degeneration). After seeding, the
        # ring self-maintains via swa_write. Per-slot position-discontinuity gate
        # so continuing requests keep their maintained ring.
        _bt_cpu = getattr(attn_md, "_eager_swa_bt_cpu", None)
        _pos_cpu = getattr(attn_md, "_eager_swa_positions_cpu", None)
        if _pool_swa is not None and _bt_cpu is not None and _pos_cpu is not None:
            _win = int(self.window_size)
            _prev = getattr(self, "_eager_prev_seed_pos", None)
            _cur = [int(p) for p in _pos_cpu]
            if _prev is None:
                _seed = [True] * len(_cur)
            else:
                _seed = [
                    (i >= len(_prev)) or (c != _prev[i] + 1) for i, c in enumerate(_cur)
                ]
            self._eager_prev_seed_pos = _cur
            _cols = _bt_cpu.shape[1]
            _npool = _pool_swa.shape[0]
            for _si in range(len(_cur)):
                if not _seed[_si]:
                    continue
                _p = _cur[_si]
                _split = _p % _win
                _cc = _p // _win
                _pc = _cc - 1
                _cb = max(int(_bt_cpu[_si, min(_cc, _cols - 1)]), 0)
                if _cc > 0:
                    _pb = max(int(_bt_cpu[_si, min(_pc, _cols - 1)]), 0)
                    # ring [0, split) from current block, [split, win) from prev
                    if _split > 0 and _cb < _npool:
                        self.swa_kv[_si, :_split].copy_(_pool_swa[_cb, :_split])
                    if _pb < _npool:
                        self.swa_kv[_si, _split:].copy_(_pool_swa[_pb, _split:])
                elif _cb < _npool:
                    # single block (pos < win): whole block is the window
                    self.swa_kv[_si].copy_(_pool_swa[_cb])

        # Per-step col0 ring gather (only pos<win rows now — see metadata).
        _swa_rows = getattr(attn_md, "_eager_swa_gather_rows", None)
        if _pool_swa is not None and _swa_rows is not None and _swa_rows.numel() > 0:
            _bid_valid = _triton_block_ids.index_select(0, _swa_rows)
            _gathered = _pool_swa.index_select(0, _bid_valid)
            self.swa_kv.index_copy_(0, _swa_rows, _gathered)

        try:
            result = self.forward_impl(x, positions)
        except Exception as e:
            rate_limited_log(
                f"v4_fallback:eager_fwd:L{self.layer_id}",
                logging.ERROR,
                "eager decode forward failed — layer %d (ratio=%d) output "
                "ZEROED (garbage): %s",
                self.layer_id,
                ratio,
                e,
                exc_info_first=True,
            )
            return _raise_or_zero(
                fc,
                x,
                f"eager attention forward failed for layer {self.layer_id} ratio {ratio}",
                e,
            )

        # --- Scatter: compact → pool, ONLY for valid-col0 rows ---
        # Skipping freed-col0 rows also avoids polluting physical block 0 (the
        # -1 clamp target), which may belong to another live request in a batch.
        if _pool_swa is not None and _swa_rows is not None and _swa_rows.numel() > 0:
            _bid_valid = _triton_block_ids.index_select(0, _swa_rows)
            _src = self.swa_kv.index_select(0, _swa_rows)
            _pool_swa.index_copy_(0, _bid_valid, _src)
        # STATE scatter
        _v4_decode_state_scatter(self, _state_pool_view, _state_block_ids, active_bs)

        return result

    # --- PREFILL path (unchanged) ---
    # Prefill uses original block_ids as state_slot_mapping (swa_kv = pool view).
    # No compact remap — swa_write needs real block_ids to write correct pool positions.
    ssm = getattr(attn_md, "state_slot_mapping", None)
    if ssm is not None and ssm.numel() > 0:
        num_slots = max(int(ssm.max()) + 1, DSV4_MIN_NATIVE_STATE_SLOTS)
    else:
        num_slots = DSV4_MIN_NATIVE_STATE_SLOTS
    _ensure_v4_native_buffers(self, num_slots=num_slots, device=x.device)

    # 2. Bind KV cache from RTP-LLM pool (for compress_kv, pool_swa, compressor.kv_cache)
    kv_cache_data = fc.kv_cache_data
    cache_entry = kv_cache_data.get(f"layer_{self.layer_id}") if kv_cache_data else None
    if cache_entry and isinstance(cache_entry.k_cache, dict) and cache_entry.k_cache:
        try:
            _bind_v4_layer_pools(self, cache_entry, ratio)
        except Exception as e:
            rate_limited_log(
                f"v4_fallback:prefill_bind:L{self.layer_id}",
                logging.ERROR,
                "prefill bind failed — layer %d falling back to dummy-run "
                "forward (degraded output): %s",
                self.layer_id,
                e,
                exc_info_first=True,
            )
            if _is_dummy_forward_context(fc):
                return self.forward_impl(x, positions)
            raise V4AttentionRuntimeError(
                f"prefill KV-pool bind failed for layer {self.layer_id}"
            ) from e
    attn_md.compress_kv = getattr(self, "_rtp_compress_kv", None)
    attn_md.swa_pages = getattr(self, "_rtp_swa_pages", 0)

    if ratio == 0:
        region_bt = v4_block_tables.get(SWA_KV)
    elif ratio == DSV4_CSA_RATIO:
        region_bt = v4_block_tables.get(CSA_KV)
    else:
        region_bt = v4_block_tables.get(HCA_KV)
    if region_bt is not None:
        attn_md.block_tables = region_bt

    # For CSA layers: store INDEXER_KV block_table for Indexer shadow buffer access.
    # ATOM assumes Main KV and Indexer KV share one block allocator (same block_table),
    # but RTP-LLM has separate pools. We store INDEXER_KV bt on attn_md each forward,
    # and monkey-patch Indexer to read it from live forward_context (not closure capture).
    if ratio == DSV4_CSA_RATIO:
        from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import INDEXER_KV

        indexer_bt = v4_block_tables.get(INDEXER_KV)
        if indexer_bt is not None:
            attn_md._indexer_block_tables = indexer_bt

        if self.indexer is not None:
            idx_comp = getattr(self.indexer, "compressor", None)
            if idx_comp is not None and not getattr(idx_comp, "_rtp_bt_patched", False):
                _orig_comp_fwd = idx_comp.forward

                def _patched_comp_fwd(
                    x, plan, state_slot_mapping, block_tables=None, _orig=_orig_comp_fwd
                ):
                    from atom.utils.forward_context import get_forward_context

                    md = get_forward_context().attn_metadata
                    bt = getattr(md, "_indexer_block_tables", block_tables)
                    return _orig(
                        x,
                        plan=plan,
                        state_slot_mapping=state_slot_mapping,
                        block_tables=bt,
                    )

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
        return _raise_or_zero(
            fc,
            x,
            f"invalid negative state slot for layer {self.layer_id}",
        )

    try:
        # Prefill: expand unified_kv with compress region so CSA/HCA prefill
        # attention can read compressed entries. Decode: handled by decode patch.
        _compress_kv = getattr(self, "_rtp_compress_kv", None)
        _do_cat = (
            fc.context.is_prefill
            and _compress_kv is not None
            and _compress_kv.numel() > 0
            and ratio != 0
        )
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
                _comp.kv_cache = _full[_sp:].reshape(
                    -1, self.window_size // ratio, self.head_dim
                )

        # Prefill: temporarily grow compressor state if needed (for block_id indexing).
        # Restored after forward_impl to preserve graph-stable addresses.
        _saved_kv_state = None
        _saved_score_state = None
        _pf_comp = getattr(self, "compressor", None)
        if _pf_comp is not None and ratio != 0:
            _pf_ssm = getattr(attn_md, "state_slot_mapping", None)
            _pf_max_slot = (
                int(_pf_ssm.max()) + 1
                if _pf_ssm is not None and _pf_ssm.numel() > 0
                else 0
            )
            if _pf_max_slot > _pf_comp.kv_state.shape[0]:
                _saved_kv_state = _pf_comp.kv_state
                _saved_score_state = _pf_comp.score_state
                _pf_comp.kv_state = torch.zeros(
                    _pf_max_slot,
                    _pf_comp.kv_state.shape[1],
                    _pf_comp.kv_state.shape[2],
                    dtype=torch.float32,
                    device=x.device,
                )
                _pf_comp.score_state = torch.full(
                    (
                        _pf_max_slot,
                        _pf_comp.score_state.shape[1],
                        _pf_comp.score_state.shape[2],
                    ),
                    float("-inf"),
                    dtype=torch.float32,
                    device=x.device,
                )
            else:
                # Buffer already large enough — zero ALL state to clear any
                # stale compressor state from a previous request. Full reset
                # is needed because decode uses compact indices [0..bs-1]
                # while prefill uses block_id indices — partial reset would
                # miss the stale compact-indexed state left by prior decode.
                _reset_v4_state_all(self)
            # Also check indexer compressor
            if ratio == DSV4_CSA_RATIO:
                _pf_idx = getattr(self, "indexer", None)
                _pf_idx_comp = getattr(_pf_idx, "compressor", None) if _pf_idx else None
                if (
                    _pf_idx_comp is not None
                    and _pf_max_slot > _pf_idx_comp.kv_state.shape[0]
                ):
                    _pf_idx_comp._saved_kv = _pf_idx_comp.kv_state
                    _pf_idx_comp._saved_sc = _pf_idx_comp.score_state
                    _pf_idx_comp.kv_state = torch.zeros(
                        _pf_max_slot,
                        _pf_idx_comp.kv_state.shape[1],
                        _pf_idx_comp.kv_state.shape[2],
                        dtype=torch.float32,
                        device=x.device,
                    )
                    _pf_idx_comp.score_state = torch.full(
                        (
                            _pf_max_slot,
                            _pf_idx_comp.score_state.shape[1],
                            _pf_idx_comp.score_state.shape[2],
                        ),
                        float("-inf"),
                        dtype=torch.float32,
                        device=x.device,
                    )

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
                _pf_state_region = CSA_STATE if ratio == DSV4_CSA_RATIO else HCA_STATE
                _pf_state_bt = v4_block_tables.get(_pf_state_region)
                kv_cache_data_pf = fc.kv_cache_data
                _pf_cache_entry = (
                    kv_cache_data_pf.get(f"layer_{self.layer_id}")
                    if kv_cache_data_pf
                    else None
                )
                if (
                    _pf_state_bt is not None
                    and _pf_cache_entry is not None
                    and _pf_cache_entry.k_cache
                ):
                    _pf_pool_name = (
                        "CSA_STATE" if ratio == DSV4_CSA_RATIO else "HCA_STATE"
                    )
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
                                _pf_state_bids = _pf_state_bt[:_pf_bs, 0].to(
                                    torch.int64
                                )
                                _pf_ssm_long = _pf_ssm[:_pf_bs].to(torch.int64)
                                _pf_kv = _pf_compressor.kv_state[_pf_ssm_long].reshape(
                                    _pf_bs, -1
                                )
                                _pf_sc = _pf_compressor.score_state[
                                    _pf_ssm_long
                                ].reshape(_pf_bs, -1)
                                _pf_combined = torch.cat([_pf_kv, _pf_sc], dim=-1)
                                _pf_pool_view[_pf_state_bids] = _pf_combined

        if _do_cat:
            _compress_kv.copy_(_full[_sp : _sp + _compress_kv.shape[0]])
            _saved_unified.copy_(_full[:_sp])
            self.unified_kv = _saved_unified
            self.swa_kv = _saved_swa_kv
            if _saved_comp_kv is not None:
                getattr(self, "compressor").kv_cache = _saved_comp_kv

        # Restore graph-stable compressor state references
        if _saved_kv_state is not None:
            _pf_comp.kv_state = _saved_kv_state
            _pf_comp.score_state = _saved_score_state
        if ratio == DSV4_CSA_RATIO:
            _pf_idx = getattr(self, "indexer", None)
            _pf_idx_comp = getattr(_pf_idx, "compressor", None) if _pf_idx else None
            if _pf_idx_comp is not None and hasattr(_pf_idx_comp, "_saved_kv"):
                _pf_idx_comp.kv_state = _pf_idx_comp._saved_kv
                _pf_idx_comp.score_state = _pf_idx_comp._saved_sc
                del _pf_idx_comp._saved_kv, _pf_idx_comp._saved_sc

        return result
    except Exception as e:
        rate_limited_log(
            f"v4_fallback:prefill_fwd:L{self.layer_id}",
            logging.ERROR,
            "prefill forward failed — layer %d (ratio=%d) output ZEROED (garbage): %s",
            self.layer_id,
            ratio,
            e,
            exc_info_first=True,
        )
        return _raise_or_zero(
            fc,
            x,
            f"prefill attention forward failed for layer {self.layer_id} ratio {ratio}",
            e,
        )


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

    logger.info(
        "Applied RTP-LLM V4 attention patch (forward level) for multi-region KV cache."
    )
    _PATCHED = True
