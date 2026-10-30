# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.attention.ops import block_sparse
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd, resolve_block_sparse_provider
from vllm_omni.diffusion.data import AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Platform:
    def __init__(self, kernel):
        self.kernel = kernel
        self.queries = []

    def get_diffusion_block_sparse_attn(self, block_size, dtype, head_size):
        self.queries.append((block_size, dtype, head_size))
        return self.kernel


@pytest.fixture
def platform(monkeypatch):
    def install(kernel):
        stub = _Platform(kernel)
        monkeypatch.setattr(block_sparse, "current_omni_platform", stub)
        resolve_block_sparse_provider.cache_clear()
        block_sparse._platform_block_sparse_kernel.cache_clear()
        return stub

    yield install
    resolve_block_sparse_provider.cache_clear()
    block_sparse._platform_block_sparse_kernel.cache_clear()


@pytest.mark.parametrize("provider", ["auto", "platform"])
@pytest.mark.parametrize("block_size", [64, 128])
def test_platform_kernel_is_selected_on_any_device(platform, provider, block_size):
    stub = platform(lambda *args, **kwargs: None)
    resolved = resolve_block_sparse_provider(
        provider, "bf16", torch.device("cpu"), torch.bfloat16, 128, block_size=block_size
    )
    assert resolved == "platform"
    assert stub.queries == [(block_size, torch.bfloat16, 128)]


def test_platform_provider_rejects_non_bf16_precision(platform):
    platform(lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="requires BF16"):
        resolve_block_sparse_provider("platform", "sage", torch.device("cpu"), torch.bfloat16, 128, block_size=128)


def test_explicit_platform_provider_without_kernel_raises(platform):
    platform(None)
    with pytest.raises(ValueError, match="provides no block-sparse attention kernel"):
        resolve_block_sparse_provider("platform", "bf16", torch.device("cpu"), torch.bfloat16, 128, block_size=128)


def test_auto_with_sage_precision_keeps_the_cuda_selection(platform):
    platform(lambda *args, **kwargs: None)
    # The CUDA path rejects sage tile128 here on CPU; the point is that the
    # platform kernel is not picked for a non-BF16 precision.
    with pytest.raises(ValueError, match="tile128 VSA requires BF16"):
        resolve_block_sparse_provider("auto", "sage", torch.device("cpu"), torch.bfloat16, 128, block_size=128)


def test_auto_without_platform_kernel_keeps_cuda_requirement(platform):
    platform(None)
    with pytest.raises(ValueError, match="tile128 VSA requires CUDA"):
        resolve_block_sparse_provider("auto", "bf16", torch.device("cpu"), torch.bfloat16, 128, block_size=128)


def test_platform_provider_dispatches_to_platform_kernel(platform):
    calls = []

    def kernel(query, key, value, block_map, block_sizes, softmax_scale, *, block_size):
        calls.append((query, key, value, block_map, block_sizes, softmax_scale, block_size))
        return torch.full_like(query, 7.0)

    platform(kernel)
    q, k, v = (torch.randn(1, 256, 2, 128, dtype=torch.bfloat16) for _ in range(3))
    block_map = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    sizes = torch.tensor([128, 100], dtype=torch.int32)
    out = block_sparse_attn_bshd(q, k, v, block_map, sizes, 128**-0.5, provider="platform", block_size=128)
    assert torch.equal(out, torch.full_like(q, 7.0))
    assert len(calls) == 1
    assert calls[0][0] is q and calls[0][3] is block_map and calls[0][4] is sizes
    assert calls[0][5] == 128**-0.5 and calls[0][6] == 128


@pytest.mark.parametrize(
    ("block_map", "sizes", "match"),
    [
        (torch.ones(1, 2, 2, 2, dtype=torch.int32), torch.tensor([128, 100]), "boolean"),
        (torch.ones(1, 2, 3, 3, dtype=torch.bool), torch.tensor([128, 100]), "boolean"),
        (torch.ones(1, 2, 2, 2, dtype=torch.bool), None, "one size per tile"),
    ],
)
def test_platform_provider_validates_inputs(platform, block_map, sizes, match):
    platform(lambda *args, **kwargs: pytest.fail("kernel must not be called"))
    q, k, v = (torch.randn(1, 256, 2, 128, dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(ValueError, match=match):
        block_sparse_attn_bshd(q, k, v, block_map, sizes, 128**-0.5, provider="platform", block_size=128)


def test_platform_provider_rejects_untiled_inputs(platform):
    platform(lambda *args, **kwargs: pytest.fail("kernel must not be called"))
    q, k, v = (torch.randn(1, 200, 2, 128, dtype=torch.bfloat16) for _ in range(3))
    block_map = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="padded to whole tiles"):
        block_sparse_attn_bshd(
            q, k, v, block_map, torch.tensor([128, 72]), 128**-0.5, provider="platform", block_size=128
        )


def test_vsa_backend_is_available_with_only_a_platform_kernel(platform, monkeypatch):
    import importlib.util

    from vllm_omni.diffusion.attention.backends.fastvideo_vsa import FastVideoVSABackend

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None)
    platform(None)
    with pytest.raises(ImportError, match="fastvideo-kernel"):
        FastVideoVSABackend.validate_available()
    platform(lambda *args, **kwargs: None)
    FastVideoVSABackend.validate_available()


def test_attention_spec_accepts_platform_provider():
    spec = AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_provider="platform")
    assert spec.fastvideo_vsa_provider == "platform"
    with pytest.raises(ValueError, match="Sage VSA precision requires the FlashInfer provider"):
        AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_provider="platform", fastvideo_vsa_precision="sage")
