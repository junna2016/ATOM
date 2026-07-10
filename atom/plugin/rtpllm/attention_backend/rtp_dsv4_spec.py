"""Authoritative DeepSeek-V4 attention specification for the RTP plugin."""

DSV4_DEFAULT_WINDOW_SIZE = 128
DSV4_CSA_RATIO = 4
DSV4_HCA_RATIO = 128
DSV4_DEFAULT_INDEX_TOPK = 1024
DSV4_DEFAULT_INDEX_HEAD_DIM = 128
DSV4_INDEX_SCALE_BYTES = 4
DSV4_INDEX_ENTRY_ALIGNMENT = 16

# Native compressor workspace constraint, not a serving batch-size default.
DSV4_MIN_NATIVE_STATE_SLOTS = 32

# A compress-plan row contains four int32 fields. CSA has an overlapping write
# span of 2 * ratio tokens; HCA is non-overlapping.
DSV4_COMPRESS_PLAN_WIDTH = 4
DSV4_CSA_DECODE_WRITE_TOKENS = 2 * DSV4_CSA_RATIO
DSV4_HCA_DECODE_WRITE_TOKENS = DSV4_HCA_RATIO
DSV4_COMPRESS_RATIOS_WITH_OVERLAP = (
    (DSV4_CSA_RATIO, True),
    (DSV4_HCA_RATIO, False),
)
