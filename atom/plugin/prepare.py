import logging
from typing import Any

logger = logging.getLogger("atom")

# all of the supported frameworks, including server mode and plugin mode
_SUPPORTED_FRAMEWORKS = ["vllm", "sglang", "sgl", "atom", "rtpllm"]

# supported frameworks for plugin mode
_SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE = ["vllm", "sglang", "sgl", "rtpllm"]

# default is atom for server mode
_CURRENT_FRAMEWORK = "atom"


def is_sglang() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in ["sglang", "sgl"])


def is_vllm() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in ["vllm"])


def is_rtpllm() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in ["rtpllm"])


def is_plugin_mode() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in _SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE)


def _set_framework_backbone(framework: str) -> None:
    if framework.lower() not in _SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework {framework} for ATOM to plug in")
    global _CURRENT_FRAMEWORK
    _CURRENT_FRAMEWORK = framework


def prepare_model(config: Any, engine: str):
    """
    Prepare ATOM model for plugin mode upper frameworks.
    """
    logger.info(f"Prepare model for plugin mode, the upper engine is {engine}")

    _set_framework_backbone(engine)

    if not (is_sglang() or is_rtpllm()):
        raise ValueError(
            f"prepare_model does not support engine {engine!r} "
            f"with config type {type(config)}"
        )

    # import here to avoid partial initialization
    from .register import (
        _ATOM_SUPPORTED_MODELS,
        # register_ops_to_vllm,
        register_ops_to_sglang,
        init_aiter_dist,
        set_attn_cls,
    )

    from atom.plugin.config import generate_atom_config_for_plugin_mode

    atom_config = generate_atom_config_for_plugin_mode(config)

    if not hasattr(atom_config.hf_config, "architectures"):
        raise ValueError("Failed to parse model architectures from HF config")
    model_arch = atom_config.hf_config.architectures[0]

    if model_arch not in _ATOM_SUPPORTED_MODELS:
        supported_archs = list(_ATOM_SUPPORTED_MODELS.keys())
        raise ValueError(
            f"ATOM does not support the required model architecture: {model_arch}. "
            f"For now supported model architectures: {supported_archs}"
        )

    model_cls = _ATOM_SUPPORTED_MODELS[model_arch]
    logger.info(f"ATOM model class for {model_arch} is {model_cls}")

    if is_sglang() and model_arch in {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }:
        from atom.plugin.sglang.models.qwen3_5 import (
            apply_prepare_model_adaptations,
        )

        apply_prepare_model_adaptations(atom_config, model_arch)

    # rtp-llm plugin mode uses this entry point for direct model construction.
    # Ensure quant layer name remap/exclude processing is done BEFORE model init,
    # otherwise layer quant_type gets fixed with stale rules.
    if is_rtpllm():
        conv1d_exclude = "model.layers.*.linear_attn.conv1d"
        if conv1d_exclude not in atom_config.quant_config.exclude_layers:
            atom_config.quant_config.exclude_layers.append(conv1d_exclude)
            logger.info(
                "rtp-llm plugin: add quant exclude for incompatible layer pattern: %s",
                conv1d_exclude,
            )

        atom_config.quant_config.remap_layer_name(
            atom_config.hf_config,
            packed_modules_mapping=getattr(model_cls, "packed_modules_mapping", {}),
            quant_exclude_name_mapping=getattr(
                model_cls, "quant_exclude_name_mapping", {}
            ),
        )

    if is_sglang():
        # Qwen3-Next and Qwen3.5 series models keep the upstream attention backend path.
        if model_arch not in {
            "Qwen3NextForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        }:
            register_ops_to_sglang(atom_config=atom_config)
    set_attn_cls()

    # init aiter dist for using aiter custom collective ops
    if is_rtpllm():
        # RTP-LLM plugin: only rank 0 runs Python model init.
        # Mock aiter TP state so ATOM model shards weights correctly.
        # Real TP communication is handled by RTP-LLM's C++ backend.
        from aiter.dist import parallel_state as _ps

        tp = atom_config.tensor_parallel_size
        rank = atom_config.plugin_config.rank

        import torch
        import torch.distributed

        class _PluginTPGroup:
            def __init__(self, ws, r):
                self.world_size = ws
                self.rank_in_group = r
                self.rank = r
                self.local_rank = r
            def all_reduce(self, x, *args, **kw):
                if self.world_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
                return x
            def all_gather(self, x, *args, **kw):
                if self.world_size > 1 and torch.distributed.is_initialized():
                    world_size = self.world_size
                    output = torch.zeros(
                        [world_size * x.shape[0]] + list(x.shape[1:]),
                        device=x.device, dtype=x.dtype)
                    torch.distributed.all_gather_into_tensor(output, x)
                    return output
                return x
            def reduce_scatter_tensor(self, x, *args, **kw):
                if self.world_size > 1 and torch.distributed.is_initialized():
                    output = torch.zeros(
                        [x.shape[0] // self.world_size] + list(x.shape[1:]),
                        device=x.device, dtype=x.dtype)
                    torch.distributed.reduce_scatter_tensor(output, x)
                    return output
                return x
            def fused_allreduce_rmsnorm(self, x, residual, weight, eps, *args, **kw):
                if self.world_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
                from aiter import rms_norm
                out, residual_out = rms_norm(x + residual, weight, eps)
                return out, residual_out
            def fused_allreduce_rmsnorm_quant(self, x, residual, weight, eps, *args, **kw):
                if self.world_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
                from aiter import rms_norm
                out, residual_out = rms_norm(x + residual, weight, eps)
                return out, residual_out, None

        _ps._TP = _PluginTPGroup(tp, rank)
        _ps._DP = _PluginTPGroup(1, 0)
        _ps._WORLD = _PluginTPGroup(1, 0)
        _ps.get_tensor_model_parallel_world_size = lambda: tp
        _ps.get_tensor_model_parallel_rank = lambda: rank
        _ps.get_dp_group = lambda: _ps._DP
        _ps.get_tp_group = lambda: _ps._TP
        _ps.get_world_group = lambda: _ps._WORLD

        from aiter.dist import communication_op as _co
        _co.tensor_model_parallel_all_reduce = lambda x, *a, **kw: x

        logger.info("rtpllm plugin: set aiter TP=%d rank=%d for weight sharding", tp, rank)
    else:
        init_aiter_dist(config=atom_config)

    if is_sglang():
        # Patch SGLang graph_capture to also enter aiter's ca_comm.capture(),
        # avoiding hipMemcpyAsync in aiter collectives when model uses aiter's
        # custom all_reduce (same fix as atom/plugin/vllm/graph_capture_patch.py)
        from atom.plugin.sglang.graph_capture_patch import apply_graph_capture_patch

        apply_graph_capture_patch()

    try:
        model = model_cls(atom_config=atom_config)
    except TypeError as exc:
        # Some models (DeepseekV4, SGLang wrappers) use `config=...`
        # instead of `atom_config=...` as their constructor parameter name.
        if "atom_config" not in str(exc):
            raise
        model = model_cls(config=atom_config)
    if not hasattr(model, "atom_config"):
        model.atom_config = atom_config
    return model
