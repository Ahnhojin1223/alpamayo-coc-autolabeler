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

"""Vision-language model wrapper implementations and factory exports."""

from coc_labeling.model_clients.vlm_wrappers.cloud import OpenAIWrapper
from coc_labeling.model_clients.vlm_wrappers.common import BaseWrapper, encode_image
from coc_labeling.model_clients.vlm_wrappers.dummy import DummyWrapper
from coc_labeling.model_clients.vlm_wrappers.factory import (
    MODEL_WRAPPER_REGISTRY,
    create_model_wrapper,
)
from coc_labeling.model_clients.vlm_wrappers.qwen import QwenWrapper
from coc_labeling.model_clients.vlm_wrappers.subscription_cli import (
    ClaudeCodeWrapper,
    CodexCliWrapper,
)

__all__ = [
    "BaseWrapper",
    "ClaudeCodeWrapper",
    "CodexCliWrapper",
    "DummyWrapper",
    "OpenAIWrapper",
    "QwenWrapper",
    "MODEL_WRAPPER_REGISTRY",
    "create_model_wrapper",
    "encode_image",
]
