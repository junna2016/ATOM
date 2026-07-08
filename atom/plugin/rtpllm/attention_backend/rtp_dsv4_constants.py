"""Shared constants for the RTP-LLM V4 attention plugin.

Leaf module (no intra-package imports) so kernels / metadata / the main adapter
can all import these without creating an import cycle.
"""

import math

LOG2E = math.log2(math.e)

# INDEXER_KV per-entry byte layouts (see _try_bind_v4_indexer_rtp_pool_144).
_ATOM_INDEXER_FP8_ENTRY_BYTES = 144
_RTP_INDEXER_BF16_ENTRY_BYTES = 256
_RTP_INDEXER_FP8_ENTRY_BYTES = 132

# attn_md attribute names used to memoize per-forward metadata state.
_V4_META_BUILT_ATTR = "_rtp_v4_meta_built"
_V4_META_FAILED_ATTR = "_rtp_v4_meta_failed"
_V4_BUFFERS_ALLOCATED = "_rtp_v4_buffers_allocated"
