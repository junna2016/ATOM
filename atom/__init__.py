# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

__all__ = [
    "LLMEngine",
    "SamplingParams",
    "prepare_model_for_sglang",
]


def __getattr__(name):
    if name == "LLMEngine":
        from atom.model_engine.llm_engine import LLMEngine
        return LLMEngine
    if name == "SamplingParams":
        from atom.sampling_params import SamplingParams
        return SamplingParams
    if name == "prepare_model_for_sglang":
        from atom.plugin.sglang import prepare_model_for_sglang
        return prepare_model_for_sglang
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
