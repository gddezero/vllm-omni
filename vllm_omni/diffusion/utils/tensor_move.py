# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replace a parameter's or buffer's data in place, across device types."""

import torch
from torch import nn


def parameter_like(template: nn.Parameter, tensor: torch.Tensor) -> nn.Parameter:
    """A Parameter of the template's class and metadata wrapping ``tensor``."""
    if type(template) is nn.Parameter:
        replacement = nn.Parameter(tensor, requires_grad=template.requires_grad)
    else:
        replacement = template.__class__.__new__(template.__class__, tensor)
        replacement.requires_grad_(template.requires_grad)
    replacement.__dict__.update(getattr(template, "__dict__", {}))
    return replacement


def replace_tensor_data(tensor: torch.Tensor, value: torch.Tensor) -> None:
    """``tensor.data = value``, keeping the same Python object.

    Assigning ``.data`` needs shallow-copy compatible tensor types, which an
    out-of-tree device tensor and its CPU copy may not have (e.g. TPU). Swap
    the tensors in that case; the swap keeps the Parameter class and its
    attributes (e.g. ``weight_loader``).
    """
    if torch._has_compatible_shallow_copy_type(tensor, value):
        tensor.data = value
        return
    if isinstance(tensor, nn.Parameter):
        value = parameter_like(tensor, value)
    torch.utils.swap_tensors(tensor, value)
