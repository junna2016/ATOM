"""ATOM DeepSeek-V4 model adapter for rtp-llm plugin mode (ROCm).

Architecture:
- Inherits RTP-LLM's DeepSeekV4 for config parsing + weight loading
  (platform-independent, handles TP via parallelism_config)
- Overrides _create_python_model to use ATOM's ROCm-compatible V4 model
  instead of RTP-LLM's CUDA-only DeepSeekV4Model
- This is how ATOM enables RTP-LLM to run V4 on AMD MI308X/MI355X

Reference: Qwen3.5 plugin (atom/plugin/rtpllm/models/qwen3_5.py)
"""

import logging
import os
from typing import Any

import torch
from rtp_llm.models.deepseek_v4 import DeepSeekV4, DeepSeekV4Mtp
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

from atom.model_loader.loader import WeightsMapper

logger = logging.getLogger("atom.plugin.rtpllm.models.deepseek_v4")


class _NoopWeightManager:
    def update(self, req):
        return None


class _NoopModelWeightsLoader:
    _py_eplb = None

    def load_lora_weights(self, adapter_name, lora_path, device):
        return None


class _ATOMDeepSeekV4Runtime(GptModelBase):
    """Runtime adapter backed by ATOM V4 model on ROCm."""

    def __init__(
        self,
        model_config,
        parallelism_config,
        weights,
        max_generate_batch_size,
        atom_model,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.model = atom_model
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            raise RuntimeError("ATOM V4 model has no parameters")
        self._model_device = first_param.device
        self._model_dtype = first_param.dtype

        from atom.plugin.rtpllm.utils.forward_context import RTPForwardContext
        self._rtp_layer_maps = RTPForwardContext.collect_layer_maps(model=self.model)
        self._rtp_kv_cache_data = None
        self._rtp_kv_cache_signature = None
        self._rtp_layer_group_map = None
        self._rtp_layer_group_map_signature = None

    def load_weights(self):
        return None

    def _get_model_device(self):
        return self._model_device

    def _get_model_dtype(self):
        return self._model_dtype

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        model_device = self._model_device

        input_ids = getattr(inputs, "input_ids", None)
        if input_ids is not None and input_ids.numel() > 0:
            input_ids = input_ids.to(device=model_device, non_blocking=True)

        attn_inputs = getattr(inputs, "attention_inputs", None)
        positions = getattr(attn_inputs, "position_ids", None) if attn_inputs else None
        if positions is not None:
            positions = positions.to(device=model_device, dtype=torch.int64)
        else:
            is_prefill = bool(getattr(attn_inputs, "is_prefill", True)) if attn_inputs else True
            if not is_prefill and attn_inputs is not None:
                # Decode: position = sequence_lengths (absolute position of new token)
                seq_lens = getattr(attn_inputs, "sequence_lengths", None)
                if seq_lens is not None and seq_lens.numel() > 0:
                    positions = seq_lens.to(device=model_device, dtype=torch.int64)
                else:
                    num_tokens = input_ids.numel() if input_ids is not None else 1
                    positions = torch.zeros(num_tokens, dtype=torch.int64, device=model_device)
            else:
                # Prefill: construct sequential positions [0, 1, 2, ...]
                num_tokens = input_ids.numel() if input_ids is not None else 1
                positions = torch.arange(num_tokens, dtype=torch.int64, device=model_device)

        from atom.plugin.rtpllm.utils.forward_context import RTPForwardContext

        with RTPForwardContext.bind(
            model=self.model,
            runtime=self,
            inputs=inputs,
            positions=positions,
            layer_maps=self._rtp_layer_maps,
        ):
            hidden_states_hc = self.model(input_ids=input_ids, positions=positions)

        hidden_states = self.model.model.head.hc_head(
            hidden_states_hc,
            self.model.model.hc_head_fn,
            self.model.model.hc_head_scale,
            self.model.model.hc_head_base,
        )
        hidden_states = self.model.model.norm(hidden_states)
        return PyModelOutputs(hidden_states)


class ATOMDeepSeekV4(DeepSeekV4):
    """DeepSeek-V4 with ATOM ROCm backend.

    Inherits DeepSeekV4 for:
    - _create_config / _from_hf (config parsing, platform-independent)
    - Weight info declaration (get_weight_cls -> DeepSeekV4Weight)

    Overrides:
    - _create_python_model: uses ATOM's model instead of CUDA-only DeepSeekV4Model
    - load: external plugin mode with ATOM weight loading
    """

    @staticmethod
    def _is_external_plugin_mode():
        modules = os.getenv("RTP_LLM_EXTERNAL_MODEL_PACKAGES", "")
        return "atom.plugin.rtpllm.models" in modules

    def load(self, skip_python_model=False):
        if self._is_external_plugin_mode():
            self.device = self._get_device_str()
            self.weight = ModelWeights(
                num_layers=self.model_config.num_layers,
                device=self.device,
                dtype=self.model_config.compute_dtype,
            )
            self.model_weights_loader = _NoopModelWeightsLoader()
            self.py_eplb = self.model_weights_loader._py_eplb
            self.weight_manager = _NoopWeightManager()
            if skip_python_model:
                return
            self._create_python_model()
            logger.info("External plugin mode: ATOM V4 loading complete")
            return
        super().load(skip_python_model=skip_python_model)

    def _create_python_model(self):
        """Create ATOM V4 model for ROCm (instead of CUDA-only DeepSeekV4Model)."""
        from atom.model_loader.loader import load_model_in_plugin_mode
        from atom.plugin.prepare import prepare_model
        from atom.plugin.rtpllm.attention_backend import apply_attention_v4_rtpllm_patch

        target_device = torch.device(self.device if hasattr(self, "device") else "cuda")
        target_dtype = self.model_config.compute_dtype
        old_default_dtype = torch.get_default_dtype()
        try:
            old_default_device = torch.get_default_device()
        except Exception:
            old_default_device = None

        torch.set_default_device(target_device)
        if target_dtype in {torch.float16, torch.bfloat16, torch.float32}:
            torch.set_default_dtype(target_dtype)

        try:
            atom_model = prepare_model(config=self, engine="rtpllm")
            if atom_model is None:
                raise ValueError("ATOM failed to create V4 model")

            apply_attention_v4_rtpllm_patch()

            atom_model = atom_model.to(target_device)
            atom_config = getattr(atom_model, "atom_config", None)
            if atom_config is None:
                raise ValueError("Cannot get atom_config from V4 model")

            load_model_in_plugin_mode(
                model=atom_model,
                config=atom_config,
                prefix="model.",
                weights_mapper=WeightsMapper(
                    orig_to_new_prefix={
                        "embed.": "model.embed.",
                        "layers.": "model.layers.",
                        "norm.weight": "model.norm.weight",
                        "head.weight": "model.head.weight",
                        "hc_head_": "model.hc_head_",
                    }
                ),
            )

            self._inject_rtp_projection_weights(atom_model)

        finally:
            torch.set_default_dtype(old_default_dtype)
            if old_default_device is not None:
                torch.set_default_device(old_default_device)
            else:
                torch.set_default_device("cpu")

        self.py_model = _ATOMDeepSeekV4Runtime(
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weights=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            atom_model=atom_model,
        )
        logger.info("Created ATOM DeepSeek-V4 runtime for ROCm")

    def _inject_rtp_projection_weights(self, atom_model):
        def _find(model, *names):
            for n in names:
                for pn, p in model.named_parameters(recurse=True):
                    if pn == n and p is not None:
                        return p
            return None

        lm = _find(atom_model, "model.head.weight", "head.weight")
        if lm is not None:
            self.weight.set_global_weight(W.lm_head, lm.detach())

        emb = _find(atom_model, "model.embed.weight", "embed.weight")
        if emb is not None:
            self.weight.set_global_weight(W.embedding, emb.detach())

        ln = _find(atom_model, "model.norm.weight", "norm.weight")
        if ln is not None:
            self.weight.set_global_weight(W.final_ln_gamma, ln.detach())


class ATOMDeepSeekV4Mtp(DeepSeekV4Mtp):
    """DeepSeek-V4 MTP draft model with ATOM ROCm backend."""

    def _create_python_model(self):
        logger.warning("ATOMDeepSeekV4Mtp: MTP not yet implemented")
