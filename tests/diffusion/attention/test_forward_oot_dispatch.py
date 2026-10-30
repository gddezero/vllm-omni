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
    with pytest.raises(
        NotImplementedError, match="_CudaOnlyImpl has no forward implementation for out-of-tree platform _Platform"
    ):
        _CudaOnlyImpl().forward(x, x, x)


def test_unknown_platform_still_raises(monkeypatch):
    monkeypatch.setattr(abstract, "current_omni_platform", _Platform("unspecified"))
    x = torch.zeros(1)
    with pytest.raises(NotImplementedError, match="No forward implementation for platform"):
        _Impl().forward(x, x, x)


def test_sdpa_impl_runs_on_out_of_tree_platforms(monkeypatch):
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

    monkeypatch.setattr(abstract, "current_omni_platform", _Platform("oot"))
    impl = SDPAImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5, causal=False)
    q, k, v = (torch.randn(1, 5, 2, 8) for _ in range(3))
    out = impl.forward(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=8**-0.5
    ).transpose(1, 2)
    torch.testing.assert_close(out, expected)
