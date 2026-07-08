"""Observability helpers for the RTP-LLM V4 plugin.

Currently: rate-limited, keyed logging for runtime fallbacks. A layer that
errors on the online forward path returns ``torch.zeros_like(x)`` (fail-safe:
one bad request must not crash the server), but a zeroed layer output silently
corrupts the generation — sometimes obvious garbage, sometimes subtle
degradation you cannot catch by eyeballing. So the fallback must be VISIBLE,
without flooding (a layer failing every decode step would otherwise log once
per token per layer).

``rate_limited_log`` logs the 1st/2nd/4th/8th... occurrence of a key
(power-of-two backoff). The first occurrence carries full context (and the
exception traceback when ``exc_info_first``) to pin the root cause; later ones
are a one-line "still happening" with the running count.

Logging only — nothing here touches GPU tensor data, and these paths only fire
on the exception/fallback branch, so there is no inference-path cost.
"""

import logging

logger = logging.getLogger("atom.plugin.rtpllm.v4_observability")

# Per-key occurrence counts (module-global, per process, persists across
# requests so the backoff keeps widening rather than resetting each request).
_RATE_LIMIT_COUNTS: dict = {}


def rate_limited_log(key, level, fmt, *args, exc_info_first: bool = False) -> int:
    """Log ``fmt % args`` at ``level`` only on power-of-two occurrences of ``key``.

    Args:
        key: stable string identifying this event kind, e.g.
             ``"v4_zeros_fallback:decode:L2"``. Counts are tracked per key, so a
             noisy layer cannot suppress a different layer's first failure.
        level: logging level (e.g. ``logging.ERROR``).
        fmt, args: printf-style message.
        exc_info_first: attach the current exception traceback on the FIRST
            occurrence only (call from an ``except`` block).

    Returns the running count for ``key`` (1 on the first call).
    """
    n = _RATE_LIMIT_COUNTS.get(key, 0) + 1
    _RATE_LIMIT_COUNTS[key] = n
    if (n & (n - 1)) == 0:  # n in {1, 2, 4, 8, 16, ...}
        logger.log(
            level,
            "[%s x%d] " + fmt,
            key,
            n,
            *args,
            exc_info=(exc_info_first and n == 1),
        )
    return n


def reset_rate_limit_counts() -> None:
    """Clear all counters (test helper)."""
    _RATE_LIMIT_COUNTS.clear()
