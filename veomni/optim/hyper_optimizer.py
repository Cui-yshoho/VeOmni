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

"""Hyper DTensor layout and local-shard optimizer adapters."""

from typing import Any, Optional

from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard


class LocalShardOptimizer:
    """Compose an optimizer with Hyper's local-shard execution context.

    State and child optimizers remain owned by the wrapped implementation.
    Schedulers bind to that implementation so their step tracking remains intact.
    """

    def __init__(self, optimizer):
        self._scheduler_optimizer = optimizer

    def __getattr__(self, name):
        if name == "_scheduler_optimizer":
            raise AttributeError(name)
        return getattr(self._scheduler_optimizer, name)

    def step(self, *args, **kwargs):
        from hyper_parallel import SkipDTensorDispatch

        with SkipDTensorDispatch():
            return self._scheduler_optimizer.step(*args, **kwargs)


def wrap_optimizer(optimizer):
    """Adapt execution without replacing the optimizer's or its children's methods."""
    return optimizer if isinstance(optimizer, LocalShardOptimizer) else LocalShardOptimizer(optimizer)


def _is_hyper_dtensor_like(tensor: Tensor) -> bool:
    """Return whether ``tensor`` is an explicitly registered Hyper DTensor."""
    from ..distributed.hyper_parallelize import is_hyper_dtensor

    return is_hyper_dtensor(tensor)


def _is_shard_placement(placement: Any) -> bool:
    """Recognize both Torch and HyperParallel shard placements."""
    if isinstance(placement, Shard):
        return True
    is_shard = getattr(placement, "is_shard", None)
    return bool(is_shard()) if callable(is_shard) else False


def _is_replicate_placement(placement: Any) -> bool:
    """Recognize both Torch and HyperParallel replicate placements."""
    if isinstance(placement, Replicate):
        return True
    is_replicate = getattr(placement, "is_replicate", None)
    return bool(is_replicate()) if callable(is_replicate) else False


def _owner_path_update(update: Tensor, ref: Tensor) -> Optional[Tensor]:
    """Attach DTensor layout to a local update for the owner all-to-all path."""
    if isinstance(update, DTensor) or _is_hyper_dtensor_like(update):
        return update
    if not _is_hyper_dtensor_like(ref):
        return None

    from hyper_parallel.core.dtensor.dtensor import DTensor as HyperDTensor

    return HyperDTensor.from_local(
        update,
        device_mesh=ref.device_mesh,
        placements=ref.placements,
        shape=tuple(ref.shape),
    )


def hyper_full_grad(grad: Tensor, ref: Tensor) -> Tensor:
    """Gather a local update using its Hyper parameter's layout."""
    from hyper_parallel.core.dtensor.dtensor import DTensor as HyperDTensor

    return HyperDTensor.from_local(grad, device_mesh=ref.device_mesh, placements=ref.placements).full_tensor()


def hyper_shard_full(full: Tensor, ref: Tensor) -> Tensor:
    """Return the local shard of a replicated update in the parameter layout."""
    from hyper_parallel.core.dtensor.dtensor import DTensor as HyperDTensor
    from hyper_parallel.core.dtensor.placement_types import Replicate as HyperReplicate

    mesh = ref.device_mesh
    replicated = HyperDTensor.from_local(full, device_mesh=mesh, placements=[HyperReplicate()] * mesh.ndim)
    return replicated.redistribute(device_mesh=mesh, placements=ref.placements).to_local()
