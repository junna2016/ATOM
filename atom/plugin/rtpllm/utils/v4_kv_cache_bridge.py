"""V4 multi-region KV cache bridge: maps RTP-LLM's 7-pool architecture to ATOM views.

RTP-LLM allocates 7 independent BlockPools for V4 (CSA_KV, HCA_KV, INDEXER_KV,
INDEXER_STATE, CSA_STATE, HCA_STATE, SWA_KV). ATOM's V4 model expects per-module
tensor views bound as attributes (unified_kv, swa_kv, kv_state, score_state, etc.).

This module provides the mapping layer that:
1. Builds a region_name → group_id dispatch table from kv_cache.group_region_names
2. Fetches per-layer pool handles via kv_cache.get_layer_cache(layer_id, region)
3. Packages them into a KVCacheTensor dict for the forward_context
4. Provides per-region block_table selection
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from atom.config import KVCacheTensor

logger = logging.getLogger("atom.plugin.rtpllm.utils.v4_kv_cache_bridge")

# Region ids mirror RTP-LLM's KVCacheRegionName enum (CacheGroupType.h). We
# derive them from the pybind enum at import time so a value change on the RTP
# side propagates automatically — a stale hardcoded literal would silently point
# ATOM at the wrong pool (corrupting KV addressing) after a rebase. When the RTP
# ops module isn't importable (e.g. unit tests outside a server process) we fall
# back to the known literals so the module still imports.
_REGION_NAME_KEYS = (
    "SWA_KV",
    "CSA_KV",
    "HCA_KV",
    "INDEXER_KV",
    "INDEXER_STATE",
    "CSA_STATE",
    "HCA_STATE",
)
_FALLBACK_REGION_IDS = {
    "SWA_KV": 7,
    "CSA_KV": 1,
    "HCA_KV": 2,
    "INDEXER_KV": 3,
    "INDEXER_STATE": 4,
    "CSA_STATE": 5,
    "HCA_STATE": 6,
}

# {int region id -> pybind KVCacheRegionName member}; populated from the enum at
# import when available, so get_pool_for_layer_region needs no per-call import.
_REGION_TO_ENUM: Dict[int, Any] = {}


def _resolve_region_ids() -> Dict[str, int]:
    try:
        from rtp_llm.ops.compute_ops import KVCacheRegionName
    except Exception:
        return dict(_FALLBACK_REGION_IDS)
    ids: Dict[str, int] = {}
    for name in _REGION_NAME_KEYS:
        member = getattr(KVCacheRegionName, name, None)
        if member is None:
            # Member renamed/removed on the RTP side — a silent mismatch here
            # would corrupt KV addressing, so warn and fall back for this one.
            logger.warning(
                "KVCacheRegionName has no member %s; using fallback id %d",
                name,
                _FALLBACK_REGION_IDS[name],
            )
            ids[name] = _FALLBACK_REGION_IDS[name]
            continue
        rid = int(member)
        ids[name] = rid
        _REGION_TO_ENUM[rid] = member
        if rid != _FALLBACK_REGION_IDS[name]:
            # Real desync: RTP renumbered this region. Surface it loudly (ATOM
            # adopts the live RTP value, which is correct, but you want to know).
            logger.warning(
                "RTP KVCacheRegionName.%s=%d differs from ATOM's expected %d "
                "— adopting the RTP value",
                name,
                rid,
                _FALLBACK_REGION_IDS[name],
            )
    return ids


_REGION_IDS = _resolve_region_ids()
SWA_KV = _REGION_IDS["SWA_KV"]
CSA_KV = _REGION_IDS["CSA_KV"]
HCA_KV = _REGION_IDS["HCA_KV"]
INDEXER_KV = _REGION_IDS["INDEXER_KV"]
INDEXER_STATE = _REGION_IDS["INDEXER_STATE"]
CSA_STATE = _REGION_IDS["CSA_STATE"]
HCA_STATE = _REGION_IDS["HCA_STATE"]

# Human-readable names for logging
_REGION_NAMES = {
    SWA_KV: "SWA_KV",
    CSA_KV: "CSA_KV",
    HCA_KV: "HCA_KV",
    INDEXER_KV: "INDEXER_KV",
    INDEXER_STATE: "INDEXER_STATE",
    CSA_STATE: "CSA_STATE",
    HCA_STATE: "HCA_STATE",
}

# Which regions each layer type uses
_CSA_REGIONS = (SWA_KV, CSA_KV, INDEXER_KV, CSA_STATE, INDEXER_STATE)
_HCA_REGIONS = (SWA_KV, HCA_KV, HCA_STATE)
_DENSE_REGIONS = (SWA_KV,)


def build_region_to_group_map(kv_cache: Any) -> Dict[int, int]:
    """Build region_name(int) → group_id mapping from kv_cache.group_region_names.

    Returns:
        Dict mapping KVCacheRegionName enum → group index in by_group list.
        Empty dict if the kv_cache doesn't expose group_region_names (non-V4 model).
    """
    group_region_names = getattr(kv_cache, "group_region_names", None)
    if group_region_names is None or len(group_region_names) == 0:
        return {}
    mapping = {}
    for group_id, attn_type_enum in enumerate(group_region_names):
        region = int(attn_type_enum)
        mapping[region] = group_id
    return mapping


def select_block_table_for_region(
    attn_inputs: Any,
    region: int,
    region_to_group: Dict[int, int],
) -> Optional[torch.Tensor]:
    """Select the block table tensor for a specific cache region.

    Args:
        attn_inputs: RTP-LLM's PyAttentionInputs
        region: KVCacheRegionName enum value (e.g., SWA_KV=7, CSA_KV=1)
        region_to_group: mapping from build_region_to_group_map()

    Returns:
        Block table tensor [batch_size, max_blocks] or None
    """
    group_id = region_to_group.get(region)
    if group_id is None:
        return None
    by_group = getattr(attn_inputs, "kv_cache_kernel_block_id_device_by_group", None)
    if by_group is None or group_id >= len(by_group):
        return None
    return by_group[group_id]


def get_pool_for_layer_region(
    kv_cache: Any,
    layer_id: int,
    region: int,
) -> Optional[Any]:
    """Get the LayerKVCache for a specific (layer, region) pair.

    Returns:
        LayerKVCache object with .kv_cache_base attribute, or None
    """
    # _REGION_TO_ENUM is built once at import from the pybind enum (empty when
    # the RTP ops module is unavailable, in which case there is nothing to bind).
    region_enum = _REGION_TO_ENUM.get(region)
    if region_enum is None:
        return None
    try:
        return kv_cache.get_layer_cache(layer_id, region_enum)
    except Exception:
        # Expected control flow: a layer legitimately does not own every region
        # (e.g. dense layers own no CSA/HCA pool). Return None silently — this is
        # not an error, so do not log it.
        return None


def build_v4_kv_cache_tensors(
    runtime: Any,
    compress_ratios: list[int],
) -> Dict[str, KVCacheTensor]:
    """Build per-layer multi-region KV cache tensor mapping for V4.

    For each V4 layer, creates a KVCacheTensor where k_cache is a dict of
    {region_name_str: LayerKVCache} containing all the pool handles that
    layer needs. The V4 attention adapter (Phase 3) will read these to
    construct the correct views.

    Args:
        runtime: the _ATOMDeepSeekV4Runtime instance (has .kv_cache)
        compress_ratios: per-layer compress ratios [0, 0, 4, 128, 4, 128, ...]

    Returns:
        Dict[str, KVCacheTensor] keyed by "layer_{i}"
    """
    kv_cache = runtime.kv_cache
    if kv_cache is None:
        raise ValueError("V4 plugin requires initialized kv_cache.")

    region_to_group = build_region_to_group_map(kv_cache)

    if not region_to_group:
        logger.warning(
            "kv_cache has no group_region_names — falling back to single-pool mode. "
            "V4 multi-region KV cache will not work correctly."
        )

    cache_data: Dict[str, KVCacheTensor] = {}

    for layer_id, ratio in enumerate(compress_ratios):
        layer_pools: Dict[str, Any] = {}

        # Determine which regions this layer uses
        if ratio == 4:
            needed_regions = _CSA_REGIONS
        elif ratio == 128:
            needed_regions = _HCA_REGIONS
        else:
            needed_regions = _DENSE_REGIONS

        for region in needed_regions:
            group_id = region_to_group.get(region)
            if group_id is None:
                continue
            pool = get_pool_for_layer_region(kv_cache, layer_id, region)
            if pool is not None:
                region_name = _REGION_NAMES.get(region, f"REGION_{region}")
                layer_pools[region_name] = pool

        cache_data[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=layer_pools,
            v_cache=None,
            k_scale=None,
            v_scale=None,
        )

    _selfcheck_v4_pools(cache_data, compress_ratios)
    return cache_data


_SELFCHECK_DONE = False


def _pool_geometry(pool: Any):
    """(blocks, stride_bytes, dtype) from a LayerKVCache handle.

    Host-side metadata only — reads .shape/.element_size()/.dtype, never touches
    device memory, so it is free of both perf cost and GPU-fault risk.
    """
    base = getattr(pool, "kv_cache_base", None)
    if base is None or base.dim() < 1:
        return None
    blocks = int(base.shape[0])
    if base.dim() >= 2:
        stride_bytes = int(base.shape[1]) * int(base.element_size())
    else:
        stride_bytes = int(base.element_size())
    return blocks, stride_bytes, str(base.dtype)


def _selfcheck_v4_pools(
    cache_data: Dict[str, KVCacheTensor], compress_ratios: list
) -> None:
    """One-time (per process) health summary of the 7 V4 KV pools, logged at
    startup BEFORE any request. Aggregates each region across layers and flags
    any layer missing a region it should own, so a rebase that changes the pool
    layout surfaces here immediately instead of as wrong output later.

    Logging only; runs once; host-side metadata only (no inference-path cost).
    """
    global _SELFCHECK_DONE
    if _SELFCHECK_DONE:
        return
    _SELFCHECK_DONE = True
    try:
        n_csa = sum(1 for r in compress_ratios if r == 4)
        n_hca = sum(1 for r in compress_ratios if r == 128)
        n_dense = sum(1 for r in compress_ratios if r not in (4, 128))

        # region_name -> {expect, have, geom, missing_layers}
        agg: Dict[str, Dict[str, Any]] = {}
        # Layers that own NONE of their expected regions are non-KV layers (an
        # MTP/draft or extra trailing entry that RTP allocates no pool for —
        # compress_ratios can be longer than num_hidden_layers). Report these
        # separately instead of flagging every region as MISSING (false alarm).
        no_pool_layers = []
        for layer_id, ratio in enumerate(compress_ratios):
            if ratio == 4:
                expected = _CSA_REGIONS
            elif ratio == 128:
                expected = _HCA_REGIONS
            else:
                expected = _DENSE_REGIONS
            entry = cache_data.get(f"layer_{layer_id}")
            k_cache = getattr(entry, "k_cache", None) or {}
            if expected and not any(
                k_cache.get(_REGION_NAMES.get(r, f"REGION_{r}")) is not None
                for r in expected
            ):
                no_pool_layers.append(layer_id)
                continue
            for region in expected:
                name = _REGION_NAMES.get(region, f"REGION_{region}")
                a = agg.setdefault(
                    name, {"expect": 0, "have": 0, "geom": None, "missing": []}
                )
                a["expect"] += 1
                pool = k_cache.get(name)
                if pool is not None:
                    a["have"] += 1
                    if a["geom"] is None:
                        a["geom"] = _pool_geometry(pool)
                else:
                    a["missing"].append(layer_id)

        lines = [
            f"=== V4 KV pool self-check ({len(compress_ratios)} layers: "
            f"{n_csa} CSA, {n_hca} HCA, {n_dense} dense) ==="
        ]
        all_ok = True
        for name in (
            "SWA_KV",
            "CSA_KV",
            "HCA_KV",
            "INDEXER_KV",
            "CSA_STATE",
            "HCA_STATE",
            "INDEXER_STATE",
        ):
            a = agg.get(name)
            if a is None:
                continue
            geom = a["geom"]
            geom_s = (
                f"blocks={geom[0]} stride_bytes={geom[1]} dtype={geom[2]}"
                if geom
                else "geom=?"
            )
            ok = a["have"] == a["expect"]
            all_ok = all_ok and ok
            lines.append(
                f"  {name:<14} {a['have']}/{a['expect']} layers  {geom_s}  "
                f"[{'OK' if ok else 'MISSING'}]"
            )
            if a["missing"]:
                shown = a["missing"][:16]
                more = " ..." if len(a["missing"]) > 16 else ""
                lines.append(f"      missing in layers: {shown}{more}")
        lines.append(
            "  [OK] all required regions present"
            if all_ok
            else "  [WARN] required regions MISSING — KV addressing will be wrong"
        )
        if no_pool_layers:
            lines.append(
                f"  note: {len(no_pool_layers)} layer(s) own no KV pool "
                f"(MTP/non-attention, not checked): {no_pool_layers}"
            )
        # WARNING level so the banner stays visible with INFO/debug logs off.
        logger.warning("\n".join(lines))
    except Exception as e:  # never let a diagnostic break startup
        logger.warning("V4 KV pool self-check failed (non-fatal): %s", e)


def build_v4_block_tables(
    attn_inputs: Any,
    region_to_group: Dict[int, int],
) -> Dict[int, Optional[torch.Tensor]]:
    """Build per-region block table dict for V4 decode.

    Returns:
        Dict mapping region enum → block table tensor.
        Only includes regions that have an associated group in region_to_group.
    """
    block_tables: Dict[int, Optional[torch.Tensor]] = {}
    for region in (
        SWA_KV,
        CSA_KV,
        HCA_KV,
        INDEXER_KV,
        CSA_STATE,
        HCA_STATE,
        INDEXER_STATE,
    ):
        bt = select_block_table_for_region(attn_inputs, region, region_to_group)
        if bt is not None:
            block_tables[region] = bt
    return block_tables


def build_v4_state_block_tables(
    attn_inputs: Any,
    region_to_group: Dict[int, int],
) -> Dict[int, Optional[torch.Tensor]]:
    """Build per-region block table dict for V4 state pools.

    State pools (CSA_STATE, HCA_STATE, INDEXER_STATE) use fixed-size
    ring buffers, but still have block tables for slot addressing.
    """
    block_tables: Dict[int, Optional[torch.Tensor]] = {}
    for region in (CSA_STATE, HCA_STATE, INDEXER_STATE):
        bt = select_block_table_for_region(attn_inputs, region, region_to_group)
        if bt is not None:
            block_tables[region] = bt
    return block_tables


def v4_kv_cache_signature(
    runtime: Any,
    compress_ratios: list[int],
) -> Tuple[Any, ...]:
    """Compute a cache signature for V4 multi-region KV cache.

    Used to detect when pool pointers change and the cache mapping
    needs to be rebuilt (same pattern as _kv_cache_signature for Qwen3.5).
    """
    kv_cache = runtime.kv_cache
    if kv_cache is None:
        return ("no_kv_cache",)

    region_to_group = build_region_to_group_map(kv_cache)
    signature: list[Any] = [id(kv_cache), len(compress_ratios)]

    # Sample a few representative layers instead of all 43+
    sample_layers = [0, len(compress_ratios) // 2, len(compress_ratios) - 1]
    sample_regions = [SWA_KV, CSA_KV, HCA_KV]

    for layer_id in sample_layers:
        if layer_id >= len(compress_ratios):
            continue
        for region in sample_regions:
            if region not in region_to_group:
                continue
            pool = get_pool_for_layer_region(kv_cache, layer_id, region)
            if pool is not None:
                base = getattr(pool, "kv_cache_base", None)
                if base is not None:
                    signature.append(
                        (layer_id, region, int(base.data_ptr()), int(base.numel()))
                    )

    return tuple(signature)


def is_v4_model(runtime: Any) -> bool:
    """Detect if the runtime's model is a V4 model.

    Checks for V4-specific attributes set by ATOMDeepSeekV4.
    """
    model = getattr(runtime, "model", None)
    if model is None:
        return False
    # V4 model has DeepseekV4Args with compress_ratios
    args = getattr(model, "args", None)
    if args is None:
        args = getattr(getattr(model, "model", None), "args", None)
    if args is None:
        return False
    return (
        hasattr(args, "compress_ratios")
        and len(getattr(args, "compress_ratios", ())) > 0
    )


def get_v4_compress_ratios(runtime: Any) -> list[int]:
    """Extract compress_ratios from the ATOM V4 model."""
    model = getattr(runtime, "model", None)
    if model is None:
        return []
    args = getattr(model, "args", None)
    if args is None:
        args = getattr(getattr(model, "model", None), "args", None)
    if args is None:
        return []
    ratios = getattr(args, "compress_ratios", ())
    return list(ratios)
