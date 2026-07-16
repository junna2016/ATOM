import enum
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

# The shared test conftest replaces atom.plugin.rtpllm.utils dependencies with
# lightweight stubs. Make the utils package itself a namespace so importing the
# leaf v4_kv_cache_bridge module does not execute utils/__init__.py and pull in
# the full RTP forward-context stack.
_utils_package = types.ModuleType("atom.plugin.rtpllm.utils")
_utils_package.__path__ = [
    str(
        Path(__file__).resolve().parents[2]
        / "atom"
        / "plugin"
        / "rtpllm"
        / "utils"
    )
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
