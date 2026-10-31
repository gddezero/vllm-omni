# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replace a parameter's or buffer's data in place, across device types."""

import torch
from torch import nn
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


def parameter_like(template: nn.Parameter, tensor: torch.Tensor) -> nn.Parameter:
    """A Parameter of the template's class and metadata wrapping ``tensor``."""
    if is_traceable_wrapper_subclass(template):
        # e.g. DTensor parameters: nn.Parameter wraps the subclass itself, and
        # the template's __dict__ holds the wrapper's own state (not metadata).
        return nn.Parameter(tensor, requires_grad=template.requires_grad)
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
    the tensors in that case, keeping the tensor's class and attributes (e.g.
    a Parameter's ``weight_loader``). The swap needs the tensor to have no
    weakrefs and no other owner of its TensorImpl, as
    ``torch.utils.swap_tensors`` requires.
    """
    if torch._has_compatible_shallow_copy_type(tensor, value):
        tensor.data = value
        return
    if isinstance(tensor, nn.Parameter):
        value = parameter_like(tensor, value)
    elif not is_traceable_wrapper_subclass(tensor):
        # The swap exchanges __class__ and __dict__ too: carry the buffer's
        # tensor subclass and attributes over to the value first.
        if type(value) is not type(tensor):
            value = value.as_subclass(type(tensor))
        value.__dict__.update(getattr(tensor, "__dict__", {}))
    torch.utils.swap_tensors(tensor, value)
