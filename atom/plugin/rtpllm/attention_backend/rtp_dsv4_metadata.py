"""Per-forward V4 metadata construction (eager decode + prefill).

Leaf module: builds kv_indices / kv_indptr / state_slot_mapping / compress_plans
for the ATOM V4 attention kernels. Does not call back into binding/forward, so
the main adapter imports from here (one-way, no cycle).
"""

import logging

import numpy as np
import torch

from atom.plugin.rtpllm.utils.v4_kv_cache_bridge import (
    SWA_KV,
    HCA_KV,
    select_block_table_for_region,
)
from atom.plugin.rtpllm.attention_backend.rtp_dsv4_constants import (
    _V4_META_BUILT_ATTR,
    _V4_META_FAILED_ATTR,
)

logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.rtp_dsv4_metadata")

def _build_eager_decode_with_triton(
    attn_md,
    attn_inputs,
    v4_ratios,
    v4_block_tables,
    region_to_group,
    device,
    window_size=128,
    pool_swa_pages=0,
    index_topk=1024,
):
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
            positions_np = (seq_lens_p1[:bs].detach().cpu().numpy() - 1).astype(
                np.int32
            )
        else:
            positions_np = np.zeros(bs, dtype=np.int32)

    # --- Block tables ---
    swa_bt = select_block_table_for_region(attn_inputs, SWA_KV, region_to_group)
    hca_bt = select_block_table_for_region(attn_inputs, HCA_KV, region_to_group)

    # --- Compact state_slot_mapping = [0..bs-1] ---
    ssm_np = np.arange(bs, dtype=np.int32)
    # Save original block IDs for gather/scatter. block_ids_raw_np keeps the
    # signed value: a freed SWA slot is -1, which we detect below to skip
    # pool<->ring sync for rows past their first window boundary.
    if swa_bt is not None and swa_bt.numel() >= bs:
        block_ids_raw_np = swa_bt[:bs, 0].detach().cpu().numpy().astype(np.int32)
        block_ids_np = np.maximum(block_ids_raw_np, 0)  # guard -1
    else:
        block_ids_raw_np = np.arange(bs, dtype=np.int32)
        block_ids_np = block_ids_raw_np

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
        swa_indptr_np[1 : bs + 1] = np.cumsum(actual_swa, dtype=np.int32)
        csa_indptr_np[1 : bs + 1] = np.cumsum(actual_swa + csa_valid_k, dtype=np.int32)
        hca_indptr_np[1 : bs + 1] = np.cumsum(actual_swa + hca_valid, dtype=np.int32)

    # --- Allocate GPU tensors ---
    positions_gpu = torch.from_numpy(positions_np).to(dtype=torch.int64, device=device)
    state_slot_gpu = torch.from_numpy(ssm_np).to(dtype=torch.int32, device=device)
    batch_id_gpu = torch.arange(bs, dtype=torch.int32, device=device)
    n_hca_gpu = torch.from_numpy(n_hca_np).to(dtype=torch.int32, device=device)

    indptr_swa_gpu = torch.from_numpy(swa_indptr_np).to(
        dtype=torch.int32, device=device
    )
    indptr_csa_gpu = torch.from_numpy(csa_indptr_np).to(
        dtype=torch.int32, device=device
    )
    indptr_hca_gpu = torch.from_numpy(hca_indptr_np).to(
        dtype=torch.int32, device=device
    )

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
    attn_md.n_committed_csa_per_seq = torch.from_numpy(n_csa_np.astype(np.int32)).to(
        device=device
    )
    attn_md.n_committed_hca_per_seq = n_hca_gpu

    # cu_seqlens_q for decode: [0, 1, 2, ..., bs]
    attn_md.cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
    attn_md.max_seqlen_q = 1
    attn_md.batch_id_per_token = batch_id_gpu

    # --- indexer_meta (for CSA layers' topk selection in decode) ---
    attn_md.indexer_meta = {
        "n_committed_per_seq_gpu": torch.from_numpy(n_csa_np.astype(np.int32)).to(
            device=device
        ),
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
                "compress": CpuGpuBuffer(
                    max(1, bs), 4, dtype=torch.int32, device=device
                ),
                "write": CpuGpuBuffer(
                    max(1, bs * 8), 4, dtype=torch.int32, device=device
                ),
            },
            128: {
                "compress": CpuGpuBuffer(
                    max(1, bs), 4, dtype=torch.int32, device=device
                ),
                "write": CpuGpuBuffer(
                    max(1, bs * 128), 4, dtype=torch.int32, device=device
                ),
            },
        }
        attn_md.compress_plans = make_compress_plans(
            extend_lens_cpu,
            context_lens_cpu,
            [(4, True), (128, False)],
            plan_buffers=_plan_bufs,
        )
    except Exception as e:
        logger.warning("Eager decode Triton: compress_plans failed: %s", e)
        attn_md.compress_plans = {}

    # --- Store block_ids and positions for gather/scatter + state reset ---
    attn_md._eager_triton_block_ids = torch.from_numpy(
        block_ids_np.astype(np.int64)
    ).to(device=device)
    attn_md._eager_triton_active_bs = bs

    # Rows eligible for the per-step col0 ring gather/scatter. Only correct when
    # the whole SWA window lies in col0, i.e. pos < win: then col0 holds exactly
    # the window. For pos >= win the window spans cur_col/prev_col (col0 may be
    # freed to -1 past 2*win), so col0-only sync would read the WRONG block; those
    # rows are seeded by the one-shot multi-block split seed in the eager forward
    # instead. (block_ids_raw_np < 0 = freed col0; the pool clamps -1 to block 0,
    # syncing which would corrupt the ring / pollute block 0 in a bs>1 batch.)
    _valid_rows_np = np.nonzero(
        (block_ids_raw_np >= 0) & (positions_np[:bs] < win)
    )[0].astype(np.int64)
    attn_md._eager_swa_gather_rows = torch.from_numpy(_valid_rows_np).to(device=device)
    attn_md._eager_triton_swa_pages = swa_pages_val
    attn_md._eager_triton_positions = positions_gpu

    # Full SWA block table + positions (CPU) for the eager one-shot multi-block
    # split seed: for pos >= win the live window spans cur_col/prev_col, which the
    # col0-only gather above skips. Without this a long-prompt slot's compact ring
    # is never seeded and serves stale / cross-request data (bs>1 contamination).
    if swa_bt is not None and swa_bt.numel() >= bs:
        attn_md._eager_swa_bt_cpu = (
            swa_bt[:bs].detach().cpu().numpy().astype(np.int32)
        )
    else:
        attn_md._eager_swa_bt_cpu = None
    attn_md._eager_swa_positions_cpu = positions_np[:bs].copy()

    setattr(attn_md, _V4_META_BUILT_ATTR, True)
    return True


def _build_prefill_extend_indices_gpu(
    positions, cu_seqlens_q, bid_per_tok, win, total_tokens, device
):
    """Build causal extend indices on GPU using vectorized torch ops."""
    extend_counts = torch.minimum(
        positions + 1, torch.tensor(win, dtype=torch.int32, device=device)
    )
    indptr = torch.zeros(total_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(extend_counts, dim=0, out=indptr[1:])
    total_nnz = int(indptr[-1].item())

    if total_nnz == 0:
        return torch.zeros(1, dtype=torch.int32, device=device), indptr

    ext_starts = (cu_seqlens_q[bid_per_tok.long()] + positions - extend_counts + 1).to(
        torch.int32
    )
    ext_starts_expanded = torch.repeat_interleave(ext_starts, extend_counts)
    group_starts = torch.repeat_interleave(indptr[:-1], extend_counts)
    global_idx = torch.arange(total_nnz, device=device, dtype=torch.int32)
    indices = ext_starts_expanded + (global_idx - group_starts)
    return indices, indptr


def _build_hca_prefix_indices_gpu(
    positions, bid_per_tok, hca_bt, swa_pages, hca_k, total_tokens, device
):
    """Build HCA prefix indices on GPU."""
    n_hca_per_tok = ((positions + 1) // 128).to(torch.int32)
    indptr = torch.zeros(total_tokens + 1, dtype=torch.int32, device=device)
    torch.cumsum(n_hca_per_tok, dim=0, out=indptr[1:])
    total_nnz = int(indptr[-1].item())

    if total_nnz == 0 or hca_bt is None or swa_pages <= 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, indptr

    tok_ids = torch.repeat_interleave(
        torch.arange(total_tokens, device=device, dtype=torch.int64), n_hca_per_tok
    )
    group_starts_exp = torch.repeat_interleave(indptr[:-1].long(), n_hca_per_tok)
    ci = (
        torch.arange(total_nnz, device=device, dtype=torch.int64) - group_starts_exp
    ).to(torch.int32)

    bid = bid_per_tok[tok_ids].long()
    lb = (ci // hca_k).long()
    sb = ci % hca_k
    max_blocks = hca_bt.shape[1]
    lb_clamped = torch.clamp(lb, 0, max_blocks - 1)
    pb = hca_bt[bid, lb_clamped].to(torch.int32)
    indices = (swa_pages + pb * hca_k + sb).to(torch.int32)
    return indices, indptr


def _build_csa_prefix_indices_gpu(
    positions, bid_per_tok, n_csa_per_seq, index_topk, total_tokens, device
):
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


def _build_v4_per_forward_metadata(
    attn_md,
    attn_inputs,
    v4_ratios,
    v4_block_tables,
    region_to_group,
    device,
    window_size=128,
    pool_swa_pages=0,
):
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
        # state_slot per sequence (indexes the compressor STATE shadow AND the
        # SWA write). Normally this is the SWA block-table col0. For a prompt
        # longer than 2*win, col0 is a FREED slot (-1); naively clamping every
        # such sequence to 0 makes them all share state slot 0, so:
        #   - sequentially, the Python STATE scatter's kv_state[-1] negative
        #     index (vs the kernel's -1->0 clamp) mismatched -> a prior request's
        #     state leaked into the STATE pool (cross-request contamination); and
        #   - concurrently, multiple >2*win prompts collide on shadow slot 0,
        #     corrupting each other's compressor state (degeneration).
        # Fix: keep col0 for live sequences (unchanged), but for a FREED col0 use
        # that sequence's CURRENT (newest, last non-negative column) SWA block —
        # a live, per-sequence-DISTINCT, valid block. Concurrent long prompts get
        # distinct physical blocks from RTP, so their state slots no longer clash.
        _raw = swa_bt[:bs]
        _cols = int(_raw.shape[1])
        _colidx = torch.arange(_cols, device=_raw.device, dtype=_raw.dtype).unsqueeze(0)
        _last_valid_col = (
            torch.where(_raw >= 0, _colidx, torch.full_like(_raw, -1))
            .max(dim=1)
            .values.clamp(min=0)
        )
        _cur_block = _raw.gather(1, _last_valid_col.long().unsqueeze(1)).squeeze(1)
        _col0 = swa_bt[:bs, 0]
        _ssm = (
            torch.where(_col0 >= 0, _col0, _cur_block)
            .clamp(min=0)
            .to(dtype=torch.int32, device=device)
        )
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
    prefix_cpu = (
        prefix.cpu().numpy().astype(np.int32)
        if prefix is not None
        else np.zeros(bs, dtype=np.int32)
    )
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
        _build_compress_plans(
            attn_md,
            attn_inputs,
            v4_ratios,
            bs,
            device,
            input_lens_cpu=input_lens_cpu,
            prefix_cpu=prefix_cpu,
            seq_lens_cpu=seq_lens_cpu,
        )
    except Exception as e:
        logger.warning(
            "Failed to build compress_plans: %s — attention will use fallback", e
        )
        attn_md.compress_plans = {}

    unique_ratios = set(r for r in v4_ratios if r > 0)
    csa_ratio = 4  # V4 CSA always uses ratio=4
    for ratio_val in unique_ratios:
        committed = seq_lens_cpu // ratio_val
        key = f"n_committed_{_ratio_label(ratio_val)}_per_seq"
        setattr(
            attn_md, key, torch.from_numpy(committed.astype(np.int32)).to(device=device)
        )
    attn_md.n_committed_csa_per_seq = (
        torch.from_numpy((seq_lens_cpu // csa_ratio).astype(np.int32)).to(device=device)
        if csa_ratio in unique_ratios
        else torch.zeros(bs, dtype=torch.int32, device=device)
    )

    # -- swa_pages set below (from pool_swa_pages or block table fallback) --

    # -- V4 sparse attention indices (ring buffer format) --
    # V4 ALWAYS uses sparse_attn_v4_paged_prefill/decode, never flash_attn.
    ssm = attn_md.state_slot_mapping
    win = window_size  # sliding window size from model config
    cs = win  # win_with_spec = window_size + max_spec_steps (0 when MTP off)
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
        torch.cumsum(
            input_lengths.to(dtype=torch.int32, device=device), dim=0, out=cu_q_gpu[1:]
        )
        offsets = torch.repeat_interleave(
            cu_q_gpu[:-1], input_lengths.to(dtype=torch.int32, device=device)
        )
        positions_gpu = (
            torch.arange(total_tokens, device=device, dtype=torch.int32) - offsets
        )
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
        idx_swa = (
            torch.tensor(swa_all, dtype=torch.int32, device=device)
            if swa_all
            else torch.zeros(1, dtype=torch.int32, device=device)
        )
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
        idx_hca = (
            torch.tensor(hca_all, dtype=torch.int32, device=device)
            if hca_all
            else torch.zeros(1, dtype=torch.int32, device=device)
        )
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

        idx_csa = (
            torch.tensor(csa_all, dtype=torch.int32, device=device)
            if csa_all
            else torch.zeros(1, dtype=torch.int32, device=device)
        )
        ptr_csa = torch.tensor(csa_indptr, dtype=torch.int32, device=device)
        attn_md.kv_indices_csa = idx_csa
        attn_md.kv_indptr_csa = ptr_csa

    else:
        # === PREFILL_NATIVE: causal extend + empty prefix ===
        # GPU-accelerated index construction (replaces Python for-loops)
        positions_gpu_i32 = torch.from_numpy(positions_np).to(
            device=device, dtype=torch.int32
        )
        bid_per_tok_gpu = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32, device=device),
            input_lengths.to(dtype=torch.int32, device=device),
        )
        cu_q_gpu = attn_md.cu_seqlens_q

        # Extend indices (causal mask into per-fwd kv)
        attn_md.kv_indices_extend, attn_md.kv_indptr_extend = (
            _build_prefill_extend_indices_gpu(
                positions_gpu_i32, cu_q_gpu, bid_per_tok_gpu, win, total_tokens, device
            )
        )

        empty_idx = torch.zeros(0, dtype=torch.int32, device=device)
        empty_ptr = torch.zeros(total_tokens + 1, dtype=torch.int32, device=device)
        attn_md.kv_indices_prefix_swa = empty_idx
        attn_md.kv_indptr_prefix_swa = empty_ptr

        # HCA prefix indices
        swa_pages_pf = pool_swa_pages if pool_swa_pages > 0 else 0
        hca_bt = v4_block_tables.get(HCA_KV)
        hca_k_pf = win // 128
        attn_md.kv_indices_prefix_hca, attn_md.kv_indptr_prefix_hca = (
            _build_hca_prefix_indices_gpu(
                positions_gpu_i32,
                bid_per_tok_gpu,
                hca_bt,
                swa_pages_pf,
                hca_k_pf,
                total_tokens,
                device,
            )
        )

        # CSA prefix indices (zeros, filled by Indexer topk later)
        index_topk = 1024
        n_csa_per_seq_gpu = attn_md.n_committed_csa_per_seq
        attn_md.kv_indices_prefix_csa, attn_md.kv_indptr_prefix_csa = (
            _build_csa_prefix_indices_gpu(
                positions_gpu_i32,
                bid_per_tok_gpu,
                n_csa_per_seq_gpu,
                index_topk,
                total_tokens,
                device,
            )
        )

    attn_md.skip_prefix_len_csa = torch.zeros(
        total_tokens, dtype=torch.int32, device=device
    )

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
        cu_committed_cpu = np.concatenate(
            [
                np.zeros(1, dtype=np.int32),
                np.cumsum(n_committed_cpu, dtype=np.int32),
            ]
        )
        cu_committed_cpu[-1] = max(int(cu_committed_cpu[-1]), 1)
        total_committed = int(cu_committed_cpu[-1])

        cu_committed_gpu = torch.from_numpy(cu_committed_cpu).to(device=device)
        bid_per_tok = attn_md.batch_id_per_token[:total_tokens].long()

        seq_base = cu_committed_gpu[bid_per_tok].to(torch.int32)

        pos_gpu = torch.from_numpy(positions_np[:total_tokens]).to(
            device=device, dtype=torch.int64
        )
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


def _build_compress_plans(
    attn_md,
    attn_inputs,
    v4_ratios,
    bs,
    device,
    input_lens_cpu=None,
    prefix_cpu=None,
    seq_lens_cpu=None,
):
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
            prefix_cpu = (
                prefix.cpu().numpy().astype(np.int32)
                if prefix is not None
                else np.zeros(bs, dtype=np.int32)
            )
        context_lens_cpu = (
            (prefix_cpu + input_lens_cpu).astype(np.int32)
            if seq_lens_cpu is None
            else seq_lens_cpu
        )
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
        is_overlap = r == 4
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
