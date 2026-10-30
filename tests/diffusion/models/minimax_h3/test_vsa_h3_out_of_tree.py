# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastH3 VSA routing on an out-of-tree platform that supplies the block-sparse kernel."""

import sys
import types

import pytest
import torch

from vllm_omni.diffusion.attention.backends import abstract, fastvideo_vsa
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
    VideoTokenLayout,
    VideoTokenSpan,
)
from vllm_omni.diffusion.attention.ops import block_sparse
from vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3 import MiniMaxH3VSAImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _masked_reference_kernel(calls):
    """Platform kernel stand-in that honours the documented contract exactly."""

    def kernel(q, k, v, block_map, block_sizes, softmax_scale, *, block_size):
        calls.append({"shape": tuple(q.shape), "block_size": block_size, "sizes": block_sizes.tolist()})
        mask = block_map.repeat_interleave(block_size, -1).repeat_interleave(block_size, -2)
        valid = (torch.arange(block_size)[None] < block_sizes[:, None]).flatten()
        mask = mask & valid.view(1, 1, 1, -1)
        out = torch.nn.functional.scaled_dot_product_attention(
            *(x.transpose(1, 2).float() for x in (q, k, v)), attn_mask=mask, scale=softmax_scale
        ).transpose(1, 2)
        # Rows with no selected key are zero by contract (SDPA gives NaN there).
        return torch.nan_to_num(out, nan=0.0).to(q.dtype)

    return kernel


class _OutOfTreePlatform:
    def __init__(self, kernel):
        self.kernel = kernel

    def is_rocm(self):
        return False

    def is_cuda(self):
        return False

    def is_npu(self):
        return False

    def is_xpu(self):
        return False

    def is_musa(self):
        return False

    def is_out_of_tree(self):
        return True

    def get_diffusion_block_sparse_attn(self, block_size, dtype, head_size):
        return self.kernel


@pytest.fixture
def oot_platform(monkeypatch):
    def install(kernel):
        platform = _OutOfTreePlatform(kernel)
        for module in (abstract, fastvideo_vsa, block_sparse):
            monkeypatch.setattr(module, "current_omni_platform", platform)
        block_sparse.resolve_block_sparse_provider.cache_clear()
        block_sparse._platform_block_sparse_kernel.cache_clear()
        return platform

    yield install
    block_sparse.resolve_block_sparse_provider.cache_clear()
    block_sparse._platform_block_sparse_kernel.cache_clear()


def _omni_metadata(packed_padding=None):
    prefix, target = (5, 512, 3), (4, 4, 24)
    return AttentionMetadata(
        packed_padding=packed_padding,
        video_layout=VideoTokenLayout(
            used_len=904,
            video_spans=(
                VideoTokenSpan(start=5, latent_grid=(8, 4, 16), role="reference"),
                VideoTokenSpan(start=520, latent_grid=target, role="target"),
            ),
        ),
        extra={
            "vsa_h3_prefix_segments": prefix,
            "vsa_h3_tile_size": 128,
            "vsa_h3_ref_policy": "p2_multi_region",
            "vsa_h3_sparsity": 0.9,
            "vsa_h3_ref_keep_rate": 0.1,
        },
    )


def _impl():
    return MiniMaxH3VSAImpl(num_heads=2, head_size=8, softmax_scale=8**-0.5)


def _install_fake_native_tile128(monkeypatch):
    """The CUDA route's native kernel, faked with the same masked SDPA."""

    def kernel(q, k, v, indices, counts, sizes, **kwargs):
        mask = indices.bool().repeat_interleave(128, -1).repeat_interleave(128, -2)
        valid = (torch.arange(128)[None] < sizes[:, None]).reshape(1, 1, 1, -1)
        out = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask & valid)
        return out.to(q.dtype), None

    native = types.ModuleType("fastvideo_kernel.block_sparse_attn_sm100a")
    setattr(native, "is_supported", lambda *args: True)
    setattr(native, "block_sparse_attn_sm100a", kernel)
    package = types.ModuleType("fastvideo_kernel")
    setattr(package, "block_sparse_attn_sm100a", native)
    index_module = types.ModuleType("fastvideo_kernel.triton_kernels.index")
    setattr(index_module, "map_to_index", lambda mask: (mask, mask.sum(-1)))
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", package)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.triton_kernels", types.ModuleType("triton_kernels"))
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.triton_kernels.index", index_module)


def test_forward_routes_omni_tile128_through_the_platform_kernel(monkeypatch, oot_platform):
    calls = []
    oot_platform(_masked_reference_kernel(calls))
    torch.manual_seed(43)
    q, k, v = (torch.randn(1, 904, 2, 8, dtype=torch.bfloat16) for _ in range(3))
    impl = _impl()

    output = impl.forward(q, k, v, _omni_metadata())

    assert len(calls) == 1
    assert calls[0]["block_size"] == 128
    # Nine logical tiles: the dense prefix segments (5 and 3 rows), then four
    # reference-video and three target tiles. Unlike the native CUDA route,
    # nothing pads the tile count to an even number here.
    assert calls[0]["sizes"] == [5, 3, 128, 128, 128, 128, 128, 128, 128]
    assert output.shape == q.shape and torch.isfinite(output).all()

    # Same routing through the CUDA path with a faked native kernel.
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3.resolve_block_sparse_provider",
        lambda *args, **kwargs: "fastvideo",
    )
    _install_fake_native_tile128(monkeypatch)
    torch.testing.assert_close(output, impl.forward_cuda(q, k, v, _omni_metadata()))


def test_forward_restores_packed_padding_rows(oot_platform):
    oot_platform(_masked_reference_kernel([]))
    torch.manual_seed(5)
    padded = torch.randn(1, 904 + 40, 2, 8, dtype=torch.bfloat16)
    cu = torch.tensor([0, 904])
    metadata = _omni_metadata(PackedPaddingMetadata(q_length=904, kv_length=904, cu_seqlens_q=cu, cu_seqlens_k=cu))
    output = _impl().forward(padded, padded, padded, metadata)
    expected = _impl().forward(padded[:, :904], padded[:, :904], padded[:, :904], _omni_metadata())
    torch.testing.assert_close(output[:, :904], expected)
    assert not output[:, 904:].any()


def _sdpa(q, k, v):
    out = torch.nn.functional.scaled_dot_product_attention(
        *(x.transpose(1, 2).float() for x in (q, k, v)), scale=8**-0.5
    )
    return out.transpose(1, 2).to(q.dtype)


def test_missing_layout_raises_when_sparse_and_degrades_to_sdpa_otherwise(monkeypatch, oot_platform):
    from vllm_omni.diffusion.attention.backends import sdpa

    platform = oot_platform(_masked_reference_kernel([]))
    monkeypatch.setattr(sdpa, "current_omni_platform", platform, raising=False)
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 37, 2, 8, dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(ValueError, match="complete packed video layout"):
        _impl().forward(q, k, v, AttentionMetadata(extra={"vsa_h3_tile_size": 128}))
    # Same as CUDA without H3 layout metadata (e.g. warmup): dense SDPA, which
    # runs in BF16 there while the reference upcasts.
    torch.testing.assert_close(_impl().forward(q, k, v, None), _sdpa(q, k, v), atol=2e-3, rtol=2e-2)


def test_missing_platform_kernel_raises_for_sparse_requests(oot_platform):
    oot_platform(None)
    q = torch.randn(1, 904, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="needs the platform block-sparse kernel for tile128"):
        _impl().forward(q, q, q, _omni_metadata())


def test_packed_mask_free_on_out_of_tree_platforms(oot_platform):
    # forward_oot honours packed padding with or without a platform kernel
    # (H3 routing, or the packed-aware SDPA fallback).
    oot_platform(None)
    assert fastvideo_vsa.FastVideoVSABackend.supports_packed_mask_free()
