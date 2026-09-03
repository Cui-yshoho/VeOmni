# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Optional HyperParallel adapters for VeOmni's FSDP2 execution path."""

import types
from functools import lru_cache
from typing import Any, Optional

import torch
import torch.distributed as dist


_HYPER_DTENSOR_TYPE: Optional[type] = None


def is_hyper_dtensor(tensor: Any) -> bool:
    """Check the exact public Hyper DTensor type after Hyper is selected."""
    return _HYPER_DTENSOR_TYPE is not None and isinstance(tensor, _HYPER_DTENSOR_TYPE)


def _missing_hyper_parallel_error() -> RuntimeError:
    return RuntimeError(
        "fsdp_backend='hyper' requires the optional 'hyper_parallel' package. "
        "Install HyperParallel in the active training environment first."
    )


@lru_cache(maxsize=1)
def get_hyper_fsdp2_api() -> dict[str, Any]:
    """Import HyperParallel lazily so the default Torch backend is unaffected."""
    global _HYPER_DTENSOR_TYPE
    try:
        from hyper_parallel import DeviceMesh, DTensor, HSDPModule, fully_shard
        from hyper_parallel.core.dtensor.dtensor import distribute_tensor
        from hyper_parallel.core.dtensor.placement_types import Shard
        from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
    except ImportError as exc:
        raise _missing_hyper_parallel_error() from exc

    _HYPER_DTENSOR_TYPE = DTensor
    return {
        "dtensor_type": DTensor,
        "module_type": HSDPModule,
        "fully_shard": fully_shard,
        "device_mesh": DeviceMesh,
        "distribute_tensor": distribute_tensor,
        "shard": Shard,
        "mixed_precision_policy": MixedPrecisionPolicy,
        "cpu_offload_policy": CPUOffloadPolicy,
    }


def to_hyper_device_mesh(torch_mesh, name: str):
    """Reuse VeOmni's existing process groups in a HyperParallel device mesh."""
    api = get_hyper_fsdp2_api()
    dim_names = tuple(torch_mesh.mesh_dim_names or ())
    if not dim_names:
        dim_names = tuple(f"{name}_{index}" for index in range(torch_mesh.ndim))
    groups = torch_mesh.get_all_groups()
    group = groups[0] if torch_mesh.ndim == 1 else groups
    return api["device_mesh"].from_group(
        group=group,
        device_type=torch_mesh.device_type,
        mesh=torch_mesh.mesh.tolist(),
        mesh_dim_names=dim_names,
    )


def extra_parallel_gradient_scaling_factor(fsdp_group_size: int, gradient_divide_factor: int) -> float:
    """Convert Torch's custom reduce-scatter divisor to Hyper's averaged reduction."""
    return fsdp_group_size / gradient_divide_factor


def refresh_materialized_shards(module) -> None:
    """Refresh Hyper's cached local-storage views after loading meta parameters."""
    seen = set()
    for submodule in module.modules():
        scheduler = getattr(submodule, "hsdp_scheduler", None)
        state = getattr(scheduler, "hsdp_state", None)
        if state is None:
            continue
        for hsdp_param in state.hsdp_params:
            if id(hsdp_param) in seen:
                continue
            seen.add(id(hsdp_param))
            hsdp_param.reset_sharded_param()


@torch.no_grad()
def rank0_broadcast_model_state(
    model: Any,
    state_metadata: list[tuple[str, torch.Size, torch.dtype, bool]],
    rank0_state: Optional[dict[str, torch.Tensor]],
    init_device: str,
    dtensor_factory: Any,
) -> None:
    """Broadcast a rank-0 initialized model into an already sharded Hyper model."""
    from ..models.module_utils import _dispatch_buffer, _dispatch_parameter, _get_communication_device
    from .parallel_plan import get_runtime_parallel_plan
    from .parallel_state import get_parallel_state

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("rank0_broadcast_model_state requires an initialized process group.")

    global_rank = get_parallel_state().global_rank
    if global_rank == 0 and rank0_state is None:
        raise ValueError("Rank 0 must provide the initialized model state.")

    parallel_plan = get_runtime_parallel_plan(model) if hasattr(model, "get_parallel_plan") else None
    communication_device = _get_communication_device(init_device)
    for name, shape, dtype, is_buffer in state_metadata:
        if global_rank == 0:
            assert rank0_state is not None
            tensor = rank0_state.pop(name).to(device=communication_device, dtype=dtype, non_blocking=True)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=communication_device)

        dist.broadcast(tensor, src=0)
        if is_buffer:
            _dispatch_buffer(model, name, tensor, dtensor_factory)
        else:
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan)
        del tensor


def wrap_optimizer(optimizer) -> None:
    """Run optimizer updates on local Hyper DTensor shards."""
    from hyper_parallel import SkipDTensorDispatch

    optimizers = list(getattr(optimizer, "optimizers_dict", {}).values()) + [optimizer]
    for current in optimizers:
        if getattr(current, "_hyper_step_wrapped", False):
            continue
        original_step = current.step

        def step(bound_optimizer, *args, _original_step=original_step, **kwargs):
            del bound_optimizer
            with SkipDTensorDispatch():
                return _original_step(*args, **kwargs)

        current.step = types.MethodType(step, current)
        current._hyper_step_wrapped = True
