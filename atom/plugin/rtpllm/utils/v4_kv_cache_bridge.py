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

# Mirror RTP-LLM's KVCacheRegionName enum (CacheGroupType.h)
SWA_KV = 7
CSA_KV = 1
HCA_KV = 2
INDEXER_KV = 3
INDEXER_STATE = 4
CSA_STATE = 5
HCA_STATE = 6

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
    try:
        from rtp_llm.ops.compute_ops import KVCacheRegionName
        # Map int region id to pybind11 KVCacheRegionName enum
        _REGION_TO_ENUM = {
            1: KVCacheRegionName.CSA_KV,
            2: KVCacheRegionName.HCA_KV,
            3: KVCacheRegionName.INDEXER_KV,
            4: KVCacheRegionName.INDEXER_STATE,
            5: KVCacheRegionName.CSA_STATE,
            6: KVCacheRegionName.HCA_STATE,
            7: KVCacheRegionName.SWA_KV,
        }
        region_enum = _REGION_TO_ENUM.get(region)
        if region_enum is not None:
            return kv_cache.get_layer_cache(layer_id, region_enum)
    except Exception as e:
        if layer_id == 0:
            logger.debug("get_pool_for_layer_region(%d, %d) failed: %s", layer_id, region, e)
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
    logger.debug("V4 KV cache: type=%s, group_region_names=%s, region_to_group=%s, "
                "attrs=%s",
                type(kv_cache).__name__,
                getattr(kv_cache, "group_region_names", "MISSING"),
                region_to_group,
                [a for a in dir(kv_cache) if not a.startswith('_') and ('cache' in a.lower() or 'group' in a.lower() or 'region' in a.lower() or 'layer' in a.lower())])
    # Log pool structure once for debugging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("V4 KV cache: %d groups, region_to_group=%s", len(region_to_group), region_to_group)

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

    return cache_data


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
    for region in (SWA_KV, CSA_KV, HCA_KV, INDEXER_KV):
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
    return hasattr(args, "compress_ratios") and len(getattr(args, "compress_ratios", ())) > 0


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
