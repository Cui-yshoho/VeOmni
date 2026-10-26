# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
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

"""VeOmni owns checkpoint lifecycle; Hyper persistence requires explicit opt-in."""

import copy
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from veomni.checkpoint.dcp_checkpointer import (
    DistributedCheckpointer,
    ModelState,
    OptimizerState,
    _ModelStrictLoadPlanner,
)
from veomni.checkpoint.layout import WEIGHTS_DIRNAME, model_dir, optimizer_dir, weights_dir
from veomni.models.checkpoint_manager import ModelCheckpointManager
from veomni.models.seed_omni.utils.checkpoint import OmniModuleCheckpointManager


def _torch_view(value: Any, reference: Any = None) -> Any:
    """Represent an existing local shard with Torch DCP's global metadata."""
    from hyper_parallel import DTensor as HyperDTensor

    if isinstance(value, DTensor):
        return value
    if isinstance(value, HyperDTensor):
        reference = value
    if not isinstance(reference, HyperDTensor):
        return value
    local = value.to_local() if isinstance(value, HyperDTensor) else value
    if tuple(local.shape) != tuple(reference.to_local().shape):
        return value
    source_mesh = reference.device_mesh
    groups = source_mesh.get_all_groups()
    mesh = DeviceMesh.from_group(
        groups[0] if len(groups) == 1 else groups,
        device_type=source_mesh.device_type,
        mesh=source_mesh.mesh.tolist(),
        mesh_dim_names=source_mesh.mesh_dim_names,
    )
    placements = []
    for placement in reference.placements:
        if type(placement).__name__ == "Shard":
            placements.append(Shard(placement.dim))
        elif type(placement).__name__ == "Replicate":
            placements.append(Replicate())
        else:
            raise NotImplementedError(f"Native DCP checkpoint view does not support {placement!r}.")
    shape = torch.Size(reference.shape)
    stride = torch.empty(shape, device="meta").stride()
    return DTensor.from_local(local.detach(), mesh, placements, shape=shape, stride=stride, run_check=False)


def _convert(value: Any, parameters: dict[str, Any], path: str = "") -> Any:
    if isinstance(value, dict):
        result = copy.copy(value)
        result.update({key: _convert(item, parameters, f"{path}.{key}".strip(".")) for key, item in value.items()})
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(_convert(item, parameters, path) for item in value)
    if not torch.is_tensor(value):
        return value
    reference = next((param for name, param in parameters.items() if path.startswith(f"state.{name}.")), None)
    return _torch_view(value, reference)


def _restore(value: Any, template: Any) -> Any:
    from hyper_parallel import DTensor as HyperDTensor

    if isinstance(value, dict):
        return {
            key: _restore(item, template.get(key) if isinstance(template, dict) else None)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            _restore(item, template[index] if template is not None else None) for index, item in enumerate(value)
        )
    if isinstance(value, DTensor):
        local = value.to_local()
        if isinstance(template, HyperDTensor):
            return HyperDTensor.from_local(
                local, template.device_mesh, template.placements, shape=tuple(template.shape)
            )
        return value if isinstance(template, DTensor) else local
    return value


class _NativeState:
    """Adapt one native Stateful payload without changing the training objects."""

    def __init__(self, state: Any):
        self.state = state
        self.template = None
        self.extra_parallel = state.should_extra_parallel_aware
        # These wrappers are checkpoint-local objects, not the model. Native EP
        # preprocessing requires Torch DTensors, so apply it after conversion.
        if self.extra_parallel:
            self.state.should_extra_parallel_aware = False

    def state_dict(self):
        parameters = {}
        if isinstance(self.state, OptimizerState):
            from hyper_parallel import SkipDTensorDispatch

            # Torch may initialize missing optimizer state with a zero-gradient
            # step; execute that step with the same local semantics as training.
            with SkipDTensorDispatch():
                self.template = self.state.state_dict()
            parameters = dict(self.state.model.named_parameters())
        else:
            self.template = self.state.state_dict()
        result = _convert(self.template, parameters)
        if self.extra_parallel:
            result = self.state.get_state_dict_with_extra_parallel_dim_preprocess(result, "restore")
        return result

    def load_state_dict(self, state_dict):
        if self.extra_parallel:
            state_dict = self.state.get_state_dict_with_extra_parallel_dim_preprocess(state_dict, "drop")
        restored = _restore(state_dict, self.template)
        if isinstance(self.state, OptimizerState):
            from hyper_parallel import SkipDTensorDispatch

            with SkipDTensorDispatch():
                self.state.load_state_dict(restored)
        else:
            self.state.load_state_dict(restored)


class NativeHyperCheckpointer(DistributedCheckpointer):
    """Reuse VeOmni DCP with checkpoint-only layout conversion for Hyper tensors."""

    @classmethod
    def execute_save(
        cls,
        save_state,
        storage_writer,
        save_async,
        save_to_lowest_rank=False,
        slot=WEIGHTS_DIRNAME,
        timeout_seconds=None,
    ):
        """Adapt checkpoint-local payloads at the existing native save boundary."""
        return super().execute_save(
            save_state={key: _NativeState(value) for key, value in save_state.items()},
            storage_writer=storage_writer,
            save_async=save_async,
            save_to_lowest_rank=save_to_lowest_rank,
            slot=slot,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def load(cls, path, state, process_group=None, module="", trainable_only=False, parallel_state=None):
        """Read native layouts with Hyper payload views; keep readers and planners native."""
        if state is None or "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")
        model = _NativeState(ModelState(state["model"], trainable_only=trainable_only, parallel_state=parallel_state))

        def optimizer_state():
            return _NativeState(
                OptimizerState(state["model"], state["optimizer"], parallel_state=parallel_state, load=True)
            )

        fused_dir = cls._legacy_fused_dir(path, module)
        payload = {"model": model}
        if fused_dir is not None and state.get("optimizer") is not None:
            payload["optimizer"] = optimizer_state()
        dcp.load(
            state_dict=payload,
            storage_reader=cls._create_storage_reader(
                fused_dir if fused_dir is not None else weights_dir(path, module)
            ),
            process_group=process_group,
            planner=_ModelStrictLoadPlanner(strict_model=not trainable_only),
        )
        if fused_dir is None and state.get("optimizer") is not None:
            dcp.load(
                state_dict={"optimizer": optimizer_state()},
                storage_reader=cls._create_storage_reader(optimizer_dir(path, module)),
                process_group=process_group,
                planner=_ModelStrictLoadPlanner(strict_model=False),
            )
        cls._load_extra_state(
            checkpoint_dir=fused_dir if fused_dir is not None else model_dir(path, module), state=state
        )
        return state


class _HyperCheckpointOwner:
    """Select Hyper persistence without modifying the native checkpoint manager."""

    def __init__(self, runtime: Any) -> None:
        """Initialize the adapter while retaining native object ownership."""
        self.runtime = runtime
        self.config = runtime.train_args.checkpoint
        self._last_saved_step = -1
        accelerator = getattr(runtime, "mesh_accelerator", runtime.args.accelerator)
        if accelerator.fsdp_config.fsdp_mode != "fsdp2":
            raise ValueError("Hyper checkpointing requires fsdp2.")
        if self.config.manager == "dcp":
            self.checkpointer = NativeHyperCheckpointer
        elif self.config.manager == "hyper":
            from hyper_parallel.integration.veomni.checkpoint import HyperParallelCheckpointer

            self.checkpointer = HyperParallelCheckpointer
        else:
            raise ValueError("Checkpoint manager must be 'dcp' (native) or 'hyper' for Hyper FSDP.")


class HyperModelCheckpointManager(_HyperCheckpointOwner, ModelCheckpointManager):
    """Keep native save/load lifecycle and paths, with Hyper persistence."""


class HyperOmniCheckpointManager(_HyperCheckpointOwner, OmniModuleCheckpointManager):
    """Keep native per-module paths and state, with Hyper persistence."""
