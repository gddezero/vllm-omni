# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.attention.backends import abstract
from vllm_omni.diffusion.attention.backends.abstract import AttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Platform:
    def __init__(self, kind: str):
        self.kind = kind

    def is_rocm(self):
        return self.kind == "rocm"

    def is_cuda(self):
        return self.kind == "cuda"

    def is_npu(self):
        return self.kind == "npu"

    def is_xpu(self):
        return self.kind == "xpu"

    def is_musa(self):
        return self.kind == "musa"

    def is_out_of_tree(self):
        return self.kind == "oot"

    def __repr__(self):
        return f"_Platform({self.kind})"


class _Impl(AttentionImpl):
    def __init__(self):
        pass

    def forward_cuda(self, query, key, value, attn_metadata=None):
        return "cuda"

    def forward_oot(self, query, key, value, attn_metadata=None):
        return "oot"


class _CudaOnlyImpl(AttentionImpl):
    def __init__(self):
        pass

    def forward_cuda(self, query, key, value, attn_metadata=None):
        return "cuda"


@pytest.mark.parametrize(("kind", "expected"), [("oot", "oot"), ("cuda", "cuda")])
def test_forward_dispatches_out_of_tree_platforms_to_forward_oot(monkeypatch, kind, expected):
    monkeypatch.setattr(abstract, "current_omni_platform", _Platform(kind))
    x = torch.zeros(1)
    assert _Impl().forward(x, x, x) == expected


def test_default_forward_oot_raises_with_implementation_name(monkeypatch):
    monkeypatch.setattr(abstract, "current_omni_platform", _Platform("oot"))
    x = torch.zeros(1)
    with pytest.raises(NotImplementedError, match="_CudaOnlyImpl has no forward implementation for out-of-tree"):
        _CudaOnlyImpl().forward(x, x, x)


def test_unknown_platform_still_raises(monkeypatch):
    monkeypatch.setattr(abstract, "current_omni_platform", _Platform("unspecified"))
    x = torch.zeros(1)
    with pytest.raises(NotImplementedError, match="No forward implementation for platform"):
        _Impl().forward(x, x, x)
