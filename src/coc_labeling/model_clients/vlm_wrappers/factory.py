# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Factory for constructing concrete VLM wrappers by model name."""

from typing import Optional, Type

from coc_labeling.model_clients.timeout import TIMEOUT_MAX
from coc_labeling.model_clients.vlm_wrappers.cloud import OpenAIWrapper
from coc_labeling.model_clients.vlm_wrappers.dummy import DummyWrapper
from coc_labeling.model_clients.vlm_wrappers.qwen import QwenWrapper
from coc_labeling.model_clients.vlm_wrappers.subscription_cli import (
    ClaudeCodeWrapper,
    CodexCliWrapper,
)

MODEL_WRAPPER_REGISTRY: dict[str, Type] = {
    "claude_code": ClaudeCodeWrapper,
    "codex": CodexCliWrapper,
    "dummy": DummyWrapper,
    "gpt5": OpenAIWrapper,
    "gpt5.5": OpenAIWrapper,
    "qwen3_vl_235b_awq": QwenWrapper,
    "qwen3.5_35b": QwenWrapper,
    "qwen3.5_397b_fp8": QwenWrapper,
}


def create_model_wrapper(
    model_name: str,
    init_model: bool = True,
    ip_addr: Optional[str] = None,
    timeout_sec: int = TIMEOUT_MAX,
):
    if model_name not in MODEL_WRAPPER_REGISTRY:
        supported_models = ", ".join(sorted(MODEL_WRAPPER_REGISTRY))
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Supported values: {supported_models}"
        )

    wrapper_cls = MODEL_WRAPPER_REGISTRY[model_name]
    return wrapper_cls(model_name, init_model, ip_addr, timeout_sec=timeout_sec)
