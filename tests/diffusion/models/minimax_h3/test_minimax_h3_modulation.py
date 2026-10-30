# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.layers.indexed_modulation import (
    _MAX_1D_GRID_SIZE,
    _iter_row_chunks,
    _launch_row_chunks,
    indexed_gate,
    indexed_gate_rms_norm_scale_shift,
    indexed_scale_shift_,
    rms_norm_indexed_scale_shift,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (0, []),
        (_MAX_1D_GRID_SIZE, [(0, _MAX_1D_GRID_SIZE)]),
        (_MAX_1D_GRID_SIZE + 1, [(0, _MAX_1D_GRID_SIZE), (_MAX_1D_GRID_SIZE, 1)]),
        (69888, [(0, _MAX_1D_GRID_SIZE), (_MAX_1D_GRID_SIZE, 4353)]),
    ],
)
def test_row_chunks_respect_ascend_grid_limit(rows, expected):
    assert list(_iter_row_chunks(rows)) == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("device_type", "expected_grids_and_offsets"),
    [
        ("npu", [(_MAX_1D_GRID_SIZE, 0), (4353, _MAX_1D_GRID_SIZE)]),
        ("cuda", [(69888, 0)]),
    ],
)
def test_launch_row_chunks_is_npu_only(device_type, expected_grids_and_offsets):
    class RecordingKernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return launch

    kernel = RecordingKernel()
    first_arg = object()
    second_arg = object()

    _launch_row_chunks(kernel, 69888, device_type, first_arg, second_arg, block_n=16)

    assert kernel.calls == [
        ((grid,), (first_arg, second_arg, row_offset), {"block_n": 16})
        for grid, row_offset in expected_grids_and_offsets
    ]


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    "device",
    [pytest.param("npu", marks=hardware_marks(res={"npu": "A3"}, num_cards=1))],
)
def test_modulation_wrappers_match_reference_above_grid_limit(device):
    rows = 69888
    hidden_size = 16
    num_conditions = 7
    eps = 1e-6
    dtype = torch.bfloat16

    torch.manual_seed(0)
    x = torch.randn(rows, hidden_size, dtype=dtype)
    residual = torch.randn_like(x)
    branch = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype)
    shift = torch.randn(num_conditions, hidden_size, dtype=dtype)
    scale = torch.randn_like(shift)
    gate = torch.randn_like(shift)
    indices = torch.arange(rows) % num_conditions

    indexed_shift = shift.index_select(0, indices)
    indexed_scale = scale.index_select(0, indices)
    indexed_gate_values = gate.index_select(0, indices)

    x_float = x.float()
    branch_float = branch.float()
    indexed_shift_float = indexed_shift.float()
    indexed_scale_float = indexed_scale.float()
    indexed_gate_float = indexed_gate_values.float()

    expected_scale_shift = (x_float * (1.0 + indexed_scale_float) + indexed_shift_float).to(dtype)
    expected_gate = (x_float + indexed_gate_float * branch_float).to(dtype)

    normalized = x_float
    normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    expected_rms = (normalized * (1.0 + indexed_scale_float) + indexed_shift_float).to(dtype)

    updated_residual = residual.float() + indexed_gate_float * branch_float
    expected_residual = updated_residual.to(dtype)
    normalized_residual = updated_residual
    normalized_residual = normalized_residual * torch.rsqrt(normalized_residual.pow(2).mean(-1, keepdim=True) + eps)
    normalized_residual = normalized_residual * weight.float()
    expected_modulated = (normalized_residual * (1.0 + indexed_scale_float) + indexed_shift_float).to(dtype)

    npu_x = x.to(device)
    npu_residual = residual.to(device)
    npu_branch = branch.to(device)
    npu_weight = weight.to(device)
    npu_shift = shift.to(device)
    npu_scale = scale.to(device)
    npu_gate = gate.to(device)
    npu_indices = indices.to(device)

    actual_scale_shift = indexed_scale_shift_(npu_x.clone(), npu_shift, npu_scale, npu_indices)
    actual_gate = indexed_gate(npu_x, npu_gate, npu_branch, npu_indices)
    actual_rms = rms_norm_indexed_scale_shift(
        npu_x,
        npu_weight,
        npu_shift,
        npu_scale,
        npu_indices,
        eps,
    )
    actual_residual, actual_modulated = indexed_gate_rms_norm_scale_shift(
        npu_residual,
        npu_gate,
        npu_branch,
        npu_weight,
        npu_shift,
        npu_scale,
        npu_indices,
        eps,
    )

    for actual, expected in (
        (actual_scale_shift, expected_scale_shift),
        (actual_gate, expected_gate),
        (actual_rms, expected_rms),
        (actual_residual, expected_residual),
        (actual_modulated, expected_modulated),
    ):
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_modulation_preserves_bf16_residual_boundary() -> None:
    torch.manual_seed(17)
    rows, hidden_size, conditions = 11, 3072, 4
    tensors = [
        torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device="cuda", dtype=torch.bfloat16),
    ]
    residual, gate, branch, weight, shift, scale = tensors
    indices = torch.arange(rows, device="cuda") % conditions
    eps = 1e-6

    residual_out, modulated_out = indexed_gate_rms_norm_scale_shift(
        residual,
        gate,
        branch,
        weight,
        shift,
        scale,
        indices,
        eps,
    )
    expected = rms_norm_indexed_scale_shift(
        residual_out,
        weight,
        shift,
        scale,
        indices,
        eps,
    )

    assert torch.equal(modulated_out, expected)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("fused_gate", [False, True])
def test_fused_modulation_keeps_fp32_affine_intermediates(fused_gate: bool) -> None:
    device = current_omni_platform.get_torch_device()
    rows, hidden_size = 2, 5376
    x = torch.ones(rows, hidden_size, device=device, dtype=torch.bfloat16)
    weight = torch.full((hidden_size,), 1.125, device=device, dtype=x.dtype)
    shift = torch.zeros(1, hidden_size, device=device, dtype=x.dtype)
    scale = torch.full_like(shift, 0.09375)
    indices = torch.zeros(rows, device=device, dtype=torch.int64)
    eps = 1e-6
    if fused_gate:
        residual, actual = indexed_gate_rms_norm_scale_shift(
            x, torch.zeros_like(shift), x, weight, shift, scale, indices, eps
        )
        assert torch.equal(residual, x)
    else:
        actual = rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)

    # Without the RMS epsilon, 1.125 * 1.09375 = 1.23046875 is exactly
    # halfway between two BF16 values. FP32 normalization keeps the epsilon
    # and rounds down to 1.2265625. Premature BF16 rounding drops the epsilon
    # and instead rounds the tie up to 1.234375, changing every output row.
    assert torch.equal(actual, torch.full_like(x, 1.2265625))


def _fp32_kernel_reference(x, residual, branch, weight, shift, scale, gate, indices, eps):
    """The Triton kernels' formulas: FP32 math, one rounding to the output dtype,
    and the fused kernel's BF16 rounding of the updated residual."""
    dtype = x.dtype
    s, sh, g = (t.index_select(0, indices).float() for t in (scale, shift, gate))

    def rms(v):
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * weight.float()

    updated = (residual.float() + g * branch.float()).to(torch.bfloat16)
    return {
        "scale_shift": (x.float() * (1.0 + s) + sh).to(dtype),
        "gate": (x.float() + g * branch.float()).to(dtype),
        "rms": (rms(x.float()) * (1.0 + s) + sh).to(dtype),
        "residual": updated.to(dtype),
        "modulated": (rms(updated.float()) * (1.0 + s) + sh).to(dtype),
    }


def _run_wrappers(x, residual, branch, weight, shift, scale, gate, indices, eps):
    residual_out, modulated = indexed_gate_rms_norm_scale_shift(
        residual, gate, branch, weight, shift, scale, indices, eps
    )
    return {
        "scale_shift": indexed_scale_shift_(x.clone(), shift, scale, indices),
        "gate": indexed_gate(x, gate, branch, indices),
        "rms": rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps),
        "residual": residual_out,
        "modulated": modulated,
    }


def _modulation_inputs(dtype, seed=3):
    torch.manual_seed(seed)
    rows, hidden_size, conditions = 37, 3072, 5
    x, residual, branch = (torch.randn(rows, hidden_size, dtype=dtype) for _ in range(3))
    weight = torch.randn(hidden_size, dtype=dtype)
    shift, scale, gate = (torch.randn(conditions, hidden_size, dtype=dtype) for _ in range(3))
    indices = torch.arange(rows) % conditions
    return (x, residual, branch, weight, shift, scale, gate, indices, 1e-6)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_path_follows_the_kernel_formulas(monkeypatch, dtype):
    """Pins the native path to the kernel formulas (not compared against Triton
    itself here: that needs a GPU)."""
    import vllm_omni.diffusion.layers.indexed_modulation as mod

    args = _modulation_inputs(dtype)
    weight, shift, scale, indices, eps = args[3], args[4], args[5], args[7], args[8]
    expected = _fp32_kernel_reference(*args)

    monkeypatch.setattr(mod, "_use_native", lambda x: True)
    actual = _run_wrappers(*args)
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype, name
        assert torch.equal(actual[name], value), name
    # The fused kernel normalizes the BF16-rounded residual it returns.
    assert torch.equal(
        actual["modulated"], rms_norm_indexed_scale_shift(actual["residual"], weight, shift, scale, indices, eps)
    )

    if dtype == torch.bfloat16:
        # Control: the CPU branch rounds the normalized value before the affine
        # transform, so the comparison above can tell the two numerics apart.
        monkeypatch.setattr(mod, "_use_native", lambda x: False)
        legacy = _run_wrappers(*args)
        assert not torch.equal(legacy["rms"], expected["rms"])


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("device_type", "has_triton", "expected"),
    [
        ("cpu", True, False),  # CPU keeps its own branch
        ("cuda", True, False),
        ("npu", True, False),
        ("tpu", True, True),  # out-of-tree device without a Triton backend
        ("cuda", False, True),
    ],
)
def test_use_native_only_where_triton_cannot_launch(monkeypatch, device_type, has_triton, expected):
    import types

    import vllm_omni.diffusion.layers.indexed_modulation as mod

    monkeypatch.setattr(mod, "HAS_TRITON", has_triton)
    tensor = types.SimpleNamespace(is_cpu=device_type == "cpu", device=types.SimpleNamespace(type=device_type))
    assert mod._use_native(tensor) is expected
