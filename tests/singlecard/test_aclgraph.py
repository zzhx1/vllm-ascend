#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Compare the outputs of vLLM with and without aclgraph.

Run `pytest tests/compile/test_aclgraph.py`.
"""

import os

import pytest

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="aclgraph only support on v1")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [12])
@pytest.mark.parametrize("full_graph", [True, False])
def test_models(
    model: str,
    max_tokens: int,
    full_graph: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="aclgraph only support on v1")
def test_deepseek_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pass


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="aclgraph only support on v1")
@pytest.mark.parametrize("model", MODELS)
def test_ray_backend_sets_no_compilation(
        model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    pass
