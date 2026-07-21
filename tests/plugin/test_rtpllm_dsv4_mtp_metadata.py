import enum
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# The shared test conftest replaces atom.plugin.rtpllm.utils dependencies with
# lightweight stubs. Make the utils package itself a namespace so importing the
# leaf v4_kv_cache_bridge module does not execute utils/__init__.py and pull in
# the full RTP forward-context stack.
_utils_package = types.ModuleType("atom.plugin.rtpllm.utils")
_utils_package.__path__ = [
    str(Path(__file__).resolve().parents[2] / "atom" / "plugin" / "rtpllm" / "utils")
]


class _AttnState(enum.Enum):
    PREFILL_NATIVE = enum.auto()
    DECODE = enum.auto()


with mock.patch.dict(
    sys.modules, {"atom.plugin.rtpllm.utils": _utils_package}, clear=False
):
    metadata = importlib.import_module(
        "atom.plugin.rtpllm.attention_backend.rtp_dsv4_metadata"
    )
    attention = importlib.import_module(
        "atom.plugin.rtpllm.attention_backend.rtp_dsv4_attention"
    )
    SWA_KV = importlib.import_module(
        "atom.plugin.rtpllm.utils.v4_kv_cache_bridge"
    ).SWA_KV

# Importing a temporary namespace package may attach it to the real parent.
# Remove that attribute so other plugin tests can import the actual utils
# package and monkeypatch its forward_context module normally.
_rtpllm_package = sys.modules.get("atom.plugin.rtpllm")
if getattr(_rtpllm_package, "utils", None) is _utils_package:
    delattr(_rtpllm_package, "utils")


def _inputs(*, target_verify: bool, prefix: int):
    return SimpleNamespace(
        is_prefill=True,
        is_target_verify=target_verify,
        input_lengths=torch.tensor([2], dtype=torch.int32),
        prefix_lengths=torch.tensor([prefix], dtype=torch.int32),
        position_ids=torch.tensor([prefix, prefix + 1], dtype=torch.int32),
    )


def _build_metadata(inputs):
    attn_md = SimpleNamespace()
    swa_bt = torch.tensor([[7]], dtype=torch.int32)

    def select_table(_inputs, region, _region_to_group):
        return swa_bt if region == SWA_KV else None

    def build_plans(md, *_args, **_kwargs):
        md.compress_plans = {}

    with (
        mock.patch.object(
            sys.modules["atom.utils.forward_context"],
            "AttnState",
            _AttnState,
            create=True,
        ),
        mock.patch.object(
            metadata, "select_block_table_for_region", side_effect=select_table
        ),
        mock.patch.object(metadata, "_build_compress_plans", side_effect=build_plans),
    ):
        metadata._build_v4_per_forward_metadata(
            attn_md=attn_md,
            attn_inputs=inputs,
            v4_ratios=[0],
            v4_block_tables={SWA_KV: swa_bt},
            region_to_group={},
            device=torch.device("cpu"),
            window_size=128,
            swa_stride=130,
            pool_swa_pages=1024,
        )
    return attn_md


def test_target_verify_uses_compact_decode_storage():
    md = _build_metadata(_inputs(target_verify=True, prefix=11))
    assert md.state is _AttnState.DECODE
    assert md.state_slot_mapping.tolist() == [0]
    assert md._eager_triton_block_ids.tolist() == [7]
    assert md._eager_swa_positions_cpu.tolist() == [11]
    assert md.cu_seqlens_q.tolist() == [0, 2]


def test_draft_continuation_uses_compact_decode_storage():
    md = _build_metadata(_inputs(target_verify=False, prefix=17))
    assert md.state is _AttnState.DECODE
    assert md.state_slot_mapping.tolist() == [0]
    assert md._eager_triton_block_ids.tolist() == [7]
    # Indices address compact slot 0 with physical stride 130, not RTP
    # physical block 7 and not absolute rows in a two-token KV tensor.
    assert int(md.kv_indices_swa.max()) < 130


def test_fresh_prefill_keeps_physical_pool_slot():
    md = _build_metadata(_inputs(target_verify=False, prefix=0))
    assert md.state is _AttnState.PREFILL_NATIVE
    assert md.state_slot_mapping.tolist() == [7]
    assert not hasattr(md, "_eager_triton_block_ids")


def _call_paged_decode(*, num_tokens: int, indptr_capacity: int):
    q = torch.empty(num_tokens, 2, 8)
    unified_kv = torch.empty(64, 8)
    kv_indices = torch.empty(128, dtype=torch.int32)
    kv_indptr = torch.empty(indptr_capacity, dtype=torch.int32)
    attn_md = SimpleNamespace(compress_kv=None, swa_pages=0)
    forward_context = SimpleNamespace(attn_metadata=attn_md)
    sentinel = object()

    with (
        mock.patch.object(
            sys.modules["atom.utils.forward_context"],
            "get_forward_context",
            return_value=forward_context,
            create=True,
        ),
        mock.patch.object(attention, "_original_paged_decode", return_value=sentinel),
    ):
        result = attention._patched_sparse_attn_v4_paged_decode(
            q,
            unified_kv,
            kv_indices,
            kv_indptr,
            torch.empty(1),
            1.0,
        )
    return result, sentinel


def test_paged_decode_accepts_max_batch_graph_indptr_buffer():
    # A bs=24 graph bucket reuses metadata allocated for max_bs=32.
    result, sentinel = _call_paged_decode(num_tokens=24, indptr_capacity=33)
    assert result is sentinel


def test_paged_decode_rejects_undersized_indptr_buffer():
    with mock.patch.object(
        sys.modules["atom.utils.forward_context"],
        "get_forward_context",
        return_value=SimpleNamespace(
            attn_metadata=SimpleNamespace(compress_kv=None, swa_pages=0)
        ),
        create=True,
    ):
        with pytest.raises(
            attention.V4AttentionRuntimeError,
            match=r"required kv_indptr.numel\(\)>=25",
        ):
            attention._patched_sparse_attn_v4_paged_decode(
                torch.empty(24, 2, 8),
                torch.empty(64, 8),
                torch.empty(128, dtype=torch.int32),
                torch.empty(24, dtype=torch.int32),
                torch.empty(1),
                1.0,
            )


def test_eager_swa_token_scatter_persists_only_current_rows():
    module = SimpleNamespace(
        swa_kv=torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    )
    pool = torch.zeros(5, 4, 3)

    attention._v4_decode_swa_token_scatter(
        module,
        pool,
        block_ids=torch.tensor([3, 1], dtype=torch.int64),
        token_offsets=torch.tensor([2, 0], dtype=torch.int64),
        active_bs=2,
    )

    assert torch.equal(pool[3, 2], module.swa_kv[0, 2])
    assert torch.equal(pool[1, 0], module.swa_kv[1, 0])
    assert torch.count_nonzero(pool).item() == 6


def test_eager_swa_write_targets_use_current_tail_block_and_ring_offset():
    blocks, offsets = metadata._eager_swa_write_targets(
        swa_bt_cpu=torch.tensor(
            [[-1, 11, 12, 13], [-1, 21, 22, 23]], dtype=torch.int32
        ).numpy(),
        positions=torch.tensor([290, 383], dtype=torch.int32).numpy(),
        window_size=128,
        cache_size=130,
    )

    assert blocks.tolist() == [12, 22]
    assert offsets.tolist() == [30, 123]


def test_eager_seed_mask_detects_batch_slot_request_change():
    mask = attention._v4_eager_seed_mask(
        current_positions=[301, 301],
        current_blocks=[17, 29],
        previous_spans=[(300, 1), (300, 1)],
        previous_blocks=[17, 23],
    )

    assert mask == [False, True]
