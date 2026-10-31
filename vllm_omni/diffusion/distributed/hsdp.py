# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.utils.tensor_move import parameter_like
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def _unshardable_parameters(model: nn.Module) -> set[nn.Parameter]:
    """Return packed/integer or scalar parameters unsupported by FSDP2."""
    return {
        param
        for param in model.parameters()
        if param.ndim == 0 or not (param.is_floating_point() or param.is_complex())
    }


def _move_tensor_in_place(tensor: torch.Tensor, device: torch.device) -> None:
    """Move a parameter or buffer to ``device`` keeping the same Python object.

    ``tensor.data = ...`` needs shallow-copy compatible tensor types, which an
    out-of-tree device tensor and its CPU source may not have (e.g. TPU). Swap
    the tensors in that case, and always for traceable wrapper subclasses, as
    FSDP2 and ``nn.Module._apply`` do. The swap keeps the Parameter class and
    its attributes (e.g. ``weight_loader``).
    """
    with torch.no_grad():
        moved = tensor.to(device)
    if torch._has_compatible_shallow_copy_type(tensor, moved) and not is_traceable_wrapper_subclass(tensor):
        tensor.data = moved
        return
    if isinstance(tensor, nn.Parameter):
        moved = parameter_like(tensor, moved)
    torch.utils.swap_tensors(tensor, moved)


@functools.cache
def _fsdp_cannot_move_from_cpu(device_type: str) -> bool:
    """Whether FSDP2's ``tensor.data = tensor.to(device)`` fails for CPU tensors."""
    probe = torch.empty(0)
    return not torch._has_compatible_shallow_copy_type(probe, probe.to(device_type))


def _place_states_fsdp_cannot_move(
    module: nn.Module, mesh: DeviceMesh | None, ignored_params: set[nn.Parameter] | None = None
) -> None:
    """Pre-place the states fully_shard(module) would fail to move.

    FSDP2 moves plain parameters and buffers with ``tensor.data =
    tensor.to(device)``, which raises when the device tensor type is not
    shallow-copy compatible with the host one. On such devices, move the same
    states FSDP2 manages to the device it targets (its own helpers decide
    both), keeping Parameter identity. Other devices return immediately.
    """
    if mesh is None or not _fsdp_cannot_move_from_cpu(mesh.device_type):
        return
    # Private FSDP2 helpers, so the selection and target device match what
    # fully_shard does; only reached on devices that need this. The device is
    # resolved inside torch, where out-of-tree backends allow current_device.
    from torch.distributed.fsdp._fully_shard._fsdp_init import (
        _get_device_from_mesh,
        _get_managed_modules,
        _get_managed_states,
    )

    device = _get_device_from_mesh(mesh)
    params, buffers = _get_managed_states(_get_managed_modules((module,), ignored_params), ignored_params)
    for tensor in itertools.chain(params, buffers):
        if tensor.device == device or tensor.device.type == "meta" or isinstance(tensor, DTensor):
            continue
        _move_tensor_in_place(tensor, device)


@dataclass
class HSDPInferenceConfig:
    """Configuration for HSDP inference.

    This is a runtime config created from DiffusionParallelConfig's HSDP settings.
    """

    enabled: bool = False
    hsdp_replicate_size: int = 1
    hsdp_shard_size: int = -1  # -1 = auto (shard across entire world)
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype | None = None
    reshard_after_forward: bool = True


@dataclass
class HSDPShardContext:
    """Reusable state for incrementally sharding one HSDP model."""

    hsdp_kwargs: dict[str, Any]
    ignored_params: set[nn.Parameter] | None = None


def _create_hsdp_mesh(
    device_type: str,
    replicate_size: int,
    shard_size: int,
) -> DeviceMesh:
    """Create the FSDP2 DeviceMesh; it owns the HSDP process groups.

    Args:
        device_type: The device type (e.g., "cuda", "npu")
        replicate_size: Number of replica groups
        shard_size: Number of ranks in each FSDP shard group

    Returns:
        A 2D DeviceMesh with dimensions ("replicate", "shard")
    """
    world_size = replicate_size * shard_size
    actual_world_size = torch.distributed.get_world_size()
    if world_size != actual_world_size:
        raise ValueError(
            f"HSDP mesh dimensions ({replicate_size} x {shard_size} = {world_size}) "
            f"must equal WORLD size ({actual_world_size})"
        )

    # Build 2D mesh tensor: shape (replicate_size, shard_size)
    # Ranks are arranged so that each row is a shard group
    mesh_tensor = torch.arange(world_size).reshape(replicate_size, shard_size)

    device_mesh = init_device_mesh(
        device_type,
        mesh_shape=(replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )

    logger.debug(
        "Created HSDP mesh: replicate_size=%d, shard_size=%d, mesh=%s",
        replicate_size,
        shard_size,
        mesh_tensor.tolist(),
    )

    return device_mesh


def apply_hsdp_to_model(
    model: nn.Module,
    hsdp_config: HSDPInferenceConfig,
    target_device: torch.device | None = None,
) -> nn.Module:
    """
    Apply HSDP sharding to a model that already has weights loaded.

    This function redistributes the model's parameters across GPUs using HSDP.
    The model should already have its weights loaded via the standard load_weights method.

    Args:
        model: Model instance with weights already loaded
        hsdp_config: HSDP configuration with HSDP mesh dimensions
        target_device: Worker's execution device. When the model declares
            _hsdp_ignored_modules, those modules are excluded from FSDP's
            mesh-driven device placement, so the caller must specify where to
            put them. Optional only when there are no ignored modules.

    Returns:
        HSDP-wrapped model ready for inference
    """
    context = prepare_hsdp_shard_context(
        model,
        hsdp_config,
        target_device=target_device,
    )

    hsdp_shard_conditions = getattr(model, "_hsdp_shard_conditions", None)
    if not hsdp_shard_conditions:
        raise ValueError(f"Model {type(model).__name__} has no _hsdp_shard_conditions defined")

    # Apply HSDP sharding, this will automatically handle weight distribution
    shard_model(
        model,
        hsdp_shard_conditions=hsdp_shard_conditions,
        context=context,
    )

    logger.info("HSDP applied to model: %s", type(model).__name__)
    return model


def prepare_hsdp_shard_context(
    model: nn.Module,
    hsdp_config: HSDPInferenceConfig,
    target_device: torch.device | None = None,
) -> HSDPShardContext:
    """Prepare mesh, precision, and ignored-parameter state for HSDP.

    The returned context can be reused to shard completed child modules while
    checkpoint weights are still streaming, then to shard the root once all
    children have been loaded.
    """
    if not hsdp_config.enabled:
        raise ValueError("HSDP is not enabled in config")

    world_group = get_world_group()
    world_size = world_group.world_size
    rank = world_group.rank_in_group

    hsdp_replicate_size = hsdp_config.hsdp_replicate_size
    hsdp_shard_size = hsdp_config.hsdp_shard_size

    if hsdp_replicate_size * hsdp_shard_size != world_size:
        raise ValueError(
            f"HSDP mesh dimensions ({hsdp_replicate_size} x {hsdp_shard_size}) must equal WORLD size ({world_size})"
        )

    logger.info(
        "HSDP Inference (FSDP2): replicate_size=%d, shard_size=%d, world_size=%d, rank=%d",
        hsdp_replicate_size,
        hsdp_shard_size,
        world_size,
        rank,
    )

    # When the model contains FP8 parameters (online quantization), let FSDP
    # keep the original storage dtype on all-gather instead of casting to
    # hsdp_config.param_dtype (typically bfloat16). FP8 GEMM kernels expect
    # FP8 inputs; an implicit FP8 -> bf16 cast would silently break them.
    has_fp8_params = any(p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for p in model.parameters())
    preserve_parameter_dtypes = bool(getattr(model, "_hsdp_preserve_parameter_dtypes", False))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=None if has_fp8_params or preserve_parameter_dtypes else hsdp_config.param_dtype,
        reduce_dtype=hsdp_config.reduce_dtype,
        output_dtype=hsdp_config.output_dtype,
        cast_forward_inputs=False,
    )

    device_type = current_omni_platform.device_type

    # DeviceMesh is the single source of truth for FSDP2 shard/replicate groups
    # The mesh shape is (replicate, shard) where:
    # - replicate: groups of ranks that hold the same shard (for gradient all-reduce in training)
    # - shard: groups of ranks that each hold different shards (for parameter all-gather)
    device_mesh = _create_hsdp_mesh(
        device_type=device_type,
        replicate_size=hsdp_replicate_size,
        shard_size=hsdp_shard_size,
    )

    hsdp_shard_conditions = getattr(model, "_hsdp_shard_conditions", None)
    if not hsdp_shard_conditions:
        raise ValueError(f"Model {type(model).__name__} has no _hsdp_shard_conditions defined")

    # Collect parameters of any modules the model wants excluded from FSDP sharding.
    # See _hsdp_ignored_modules on each model class. These params keep their
    # original dtype/storage and are not subject to MixedPrecisionPolicy on the root
    # FSDP wrap. Useful for small auxiliary modules (e.g., timestep embedders) that
    # need to stay in a higher-precision dtype than the bulk of the model.
    ignored_module_names = getattr(model, "_hsdp_ignored_modules", []) or []
    ignored_params: set[nn.Parameter] = set()
    # FSDP wraps move sharded params to mesh devices automatically, but
    # ignored modules stay wherever load_weights left them (CPU under HSDP).
    # The caller (diffusers_loader / similar) is responsible for telling us
    # which device the worker should run on, matching the convention used
    # for VAEs / encoders / resident_modules.
    if ignored_module_names and target_device is None:
        raise ValueError(
            f"Model {type(model).__name__} declares _hsdp_ignored_modules="
            f"{ignored_module_names} but apply_hsdp_to_model was called "
            "without target_device. The caller must pass target_device so "
            "the ignored modules can be placed on the worker's execution device."
        )
    for mod_name in ignored_module_names:
        try:
            sub_mod = model.get_submodule(mod_name)
        except AttributeError:
            logger.warning("_hsdp_ignored_modules entry %r not found on model", mod_name)
            continue
        has_meta_tensor = any(tensor.device.type == "meta" for tensor in (*sub_mod.parameters(), *sub_mod.buffers()))
        if has_meta_tensor:
            sub_mod.to_empty(device=target_device)
        else:
            # Not sub_mod.to(): on devices whose tensors are not shallow-copy
            # compatible with CPU it replaces Parameters, breaking ties and
            # dropping loader metadata.
            for tensor in itertools.chain(sub_mod.parameters(), sub_mod.buffers()):
                if tensor.device != target_device:
                    _move_tensor_in_place(tensor, target_device)
        ignored_params.update(sub_mod.parameters())
    if ignored_params:
        logger.info(
            "HSDP excluding %d parameter tensors from sharding (modules: %s, moved to %s)",
            len(ignored_params),
            ignored_module_names,
            target_device,
        )

    # Serialized low-bit checkpoints may store packed weights in integer
    # Parameters (for example, ModelOpt NVFP4 uses uint8) and global scales as
    # scalar Parameters. FSDP2 cannot represent non-floating or zero-dimensional
    # sharded parameters. Keep those tensors resident and replicated; eligible
    # scales, biases, and other parameters remain HSDP-sharded.
    unshardable_params = _unshardable_parameters(model)
    if unshardable_params:
        if target_device is None:
            raise ValueError(
                f"Model {type(model).__name__} has parameters that HSDP must ignore, "
                "but apply_hsdp_to_model was called without target_device."
            )
        for param in unshardable_params:
            if param.device != target_device:
                _move_tensor_in_place(param, target_device)
        ignored_params.update(unshardable_params)
        logger.info(
            "HSDP excluding %d unshardable parameter tensors from sharding "
            "(non_floating=%d, scalar=%d, dtypes=%s, moved to %s)",
            len(unshardable_params),
            sum(not (param.is_floating_point() or param.is_complex()) for param in unshardable_params),
            sum(param.ndim == 0 for param in unshardable_params),
            sorted({str(param.dtype) for param in unshardable_params}),
            target_device,
        )

    return HSDPShardContext(
        hsdp_kwargs={
            "reshard_after_forward": hsdp_config.reshard_after_forward,
            "mesh": device_mesh,
            "mp_policy": mp_policy,
        },
        ignored_params=ignored_params or None,
    )


def shard_hsdp_module(module: nn.Module, context: HSDPShardContext) -> None:
    """Shard one completed child module with a prepared HSDP context."""
    module_kwargs = dict(context.hsdp_kwargs)
    if context.ignored_params:
        module_ignored_params = context.ignored_params.intersection(module.parameters())
        if module_ignored_params:
            module_kwargs["ignored_params"] = module_ignored_params
    _place_states_fsdp_cannot_move(module, module_kwargs.get("mesh"), module_kwargs.get("ignored_params"))
    fully_shard(module, **module_kwargs)
    for param in module.parameters():
        param.requires_grad = False


def finalize_hsdp_root(model: nn.Module, context: HSDPShardContext) -> None:
    """Shard the model root after all selected child modules are sharded."""
    root_kwargs = dict(context.hsdp_kwargs)
    if context.ignored_params:
        root_kwargs["ignored_params"] = context.ignored_params
    _place_states_fsdp_cannot_move(model, root_kwargs.get("mesh"), root_kwargs.get("ignored_params"))
    fully_shard(model, **root_kwargs)
    for param in model.parameters():
        param.requires_grad = False


def shard_model(
    model: nn.Module,
    *,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = None,
    mesh: DeviceMesh | None = None,
    hsdp_shard_conditions: list[Callable[[str, nn.Module], bool]],
    ignored_params: set[nn.Parameter] | None = None,
    context: HSDPShardContext | None = None,
) -> None:
    """Apply HSDP sharding to model modules based on shard conditions.

    ignored_params (if provided) are excluded from the root fully_shard
    wrap, so they are not collected into the root flat-parameter, are not
    subject to MixedPrecisionPolicy, and retain their original dtype.
    Each per-submodule wrap receives the subset of ignored_params that it owns.
    This is required for packed integer parameters inside sharded transformer
    blocks; the root wrap receives the full set for all remaining parameters.
    """
    if context is None:
        context = HSDPShardContext(
            hsdp_kwargs={
                "reshard_after_forward": reshard_after_forward,
                "mesh": mesh,
                "mp_policy": mp_policy,
            },
            ignored_params=ignored_params,
        )
    elif any(value is not None for value in (mp_policy, mesh, ignored_params)) or not reshard_after_forward:
        raise ValueError("Pass either context or individual HSDP shard options, not both")

    num_sharded = 0
    for name, module in reversed(list(model.named_modules())):
        if any(cond(name, module) for cond in hsdp_shard_conditions):
            shard_hsdp_module(module, context)
            num_sharded += 1

    if num_sharded == 0:
        raise ValueError("No modules were sharded")

    finalize_hsdp_root(model, context)
    logger.info(
        "Sharded %d modules + root (ignored_params=%d)",
        num_sharded,
        len(context.ignored_params) if context.ignored_params else 0,
    )
