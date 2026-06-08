"""ATOM wrappers for rtp-llm external model loading.

Loaded via:
    RTP_LLM_EXTERNAL_MODEL_PACKAGES=atom.plugin.rtpllm.models

This module intentionally keeps runtime behavior compatible with rtp-llm's
native implementations while providing a plugin entrypoint that can
be extended with ATOM-specific logic later.
"""

from rtp_llm.model_factory_register import (
    _hf_architecture_2_ft,
    _model_factory,
    register_model,
)

import logging

_logger = logging.getLogger("atom.plugin.rtpllm.models")


def _try_register(name, factory_key, hf_arch, module_path, class_name):
    """Register a model with graceful fallback on import errors."""
    try:
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        register_model(name, cls, [])
        _model_factory[factory_key] = cls
        _hf_architecture_2_ft[hf_arch] = factory_key
    except (ImportError, AttributeError) as e:
        _logger.warning("Skipping %s registration: %s", name, e)


_try_register(
    "atom_qwen35_moe", "qwen35_moe",
    "Qwen3_5MoeForConditionalGeneration",
    "atom.plugin.rtpllm.models.qwen3_5", "ATOMQwen35Moe",
)
_try_register(
    "atom_deepseek_v4", "deepseek_v4",
    "DeepseekV4ForCausalLM",
    "atom.plugin.rtpllm.models.deepseek_v4", "ATOMDeepSeekV4",
)
_try_register(
    "atom_deepseek_v4_mtp", "deepseek_v4_mtp",
    "DeepseekV4ForCausalLMNextN",
    "atom.plugin.rtpllm.models.deepseek_v4_mtp", "ATOMDeepSeekV4Mtp",
)
