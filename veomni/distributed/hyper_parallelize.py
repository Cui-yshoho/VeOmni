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

"""HyperParallel execution policy for VeOmni's shared FSDP2 traversal."""

from functools import lru_cache
from typing import Any, Optional


_HYPER_DTENSOR_TYPE: Optional[type] = None


def is_hyper_dtensor(tensor: Any) -> bool:
    """Check the exact public Hyper DTensor type after Hyper is selected."""
    return _HYPER_DTENSOR_TYPE is not None and isinstance(tensor, _HYPER_DTENSOR_TYPE)


@lru_cache(maxsize=1)
def get_hyper_fsdp2_backend():
    """Import HyperParallel lazily so the default Torch backend is unaffected."""
    from .fsdp2.backend import FSDP2Backend

    global _HYPER_DTENSOR_TYPE
    try:
        from hyper_parallel import DTensor, HSDPModule, fully_shard
        from hyper_parallel.core.dtensor.dtensor import distribute_tensor
        from hyper_parallel.core.dtensor.placement_types import Shard
        from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, MixedPrecisionPolicy
    except ImportError as exc:
        raise RuntimeError(
            "fsdp_backend='hyper' requires the optional 'hyper_parallel' package. "
            "Install HyperParallel in the active training environment first."
        ) from exc

    _HYPER_DTENSOR_TYPE = DTensor
    return FSDP2Backend(
        "hyper",
        fully_shard,
        HSDPModule,
        MixedPrecisionPolicy,
        CPUOffloadPolicy,
        Shard,
        to_hyper_device_mesh,
        distribute_tensor,
    )


def to_hyper_device_mesh(torch_mesh, name: str):
    """Reuse VeOmni's existing process groups in a HyperParallel device mesh."""
    from hyper_parallel import DeviceMesh

    dim_names = tuple(torch_mesh.mesh_dim_names or ())
    if not dim_names:
        dim_names = tuple(f"{name}_{index}" for index in range(torch_mesh.ndim))
    groups = torch_mesh.get_all_groups()
    group = groups[0] if torch_mesh.ndim == 1 else groups
    return DeviceMesh.from_group(
        group=group,
        device_type=torch_mesh.device_type,
        mesh=torch_mesh.mesh.tolist(),
        mesh_dim_names=dim_names,
    )


def extra_parallel_gradient_scaling_factor(fsdp_group_size: int, gradient_divide_factor: int) -> float:
    """Convert Torch's custom reduce-scatter divisor to Hyper's averaged reduction."""
    return fsdp_group_size / gradient_divide_factor


_COMPUTE_BOUNDARIES = {
    "Qwen3MoeDecoderLayer": ("self_attn", "mlp"),
    "Qwen3VLMoeTextDecoderLayer": ("self_attn", "mlp"),
    "Qwen3VLMoeVisionBlock": ("attn", "mlp"),
}


def resolve_submodule_wrap_targets(target_modules, *, root_model, device_type, compiled=False):
    """Use validated CUDA boundaries; leave unknown or shared blocks intact."""
    if device_type != "cuda" or compiled:
        return {}
    parameter_names = {}
    for name, param in root_model.named_parameters(remove_duplicate=False):
        parameter_names.setdefault(id(param), []).append(name)
    resolved = {}
    for fqn, block in target_modules:
        paths = _COMPUTE_BOUNDARIES.get(type(block).__name__)
        if paths is None or any(not hasattr(block, path) for path in paths):
            continue
        children = [block.get_submodule(path) for path in paths]
        seen = set()
        for path, child in zip(paths, children):
            owned = {id(param) for param in child.parameters()}
            prefix = (fqn + "." if fqn else "") + path + "."
            if (
                not owned
                or seen & owned
                or any(not name.startswith(prefix) for param in owned for name in parameter_names[param])
            ):
                break
            seen.update(owned)
        else:
            resolved[block] = children
    return resolved


def build_submodule_prefetch_stages(block, children, existing_units):
    """Keep residual, EP and FP32 units with their corresponding compute stage."""
    stages = [[block]] + [[child] for child in children]
    for unit in existing_units:
        if unit is block or any(unit is child for child in children):
            continue
        stage = stages[0]
        for child, candidate in zip(children, stages[1:]):
            if any(unit is descendant for descendant in child.modules()):
                stage = candidate
                break
        stage.append(unit)
    return stages


def configure_submodule_prefetch(block_stages):
    """Prefetch two compute stages ahead; retain reverse-stage backward links."""
    entries = [(stage, index > 0 or len(block) == 1) for block in block_stages for index, stage in enumerate(block)]
    position = 0
    for index, stages in enumerate(block_stages):
        for stage_index, stage in enumerate(stages):
            if stage_index == 0 and len(stages) > 1:
                forward = stages[1]
            else:
                forward, remaining = [], 2
                for following, compute in entries[position + 1 :]:
                    forward.extend(following)
                    remaining -= int(compute)
                    if remaining == 0:
                        break
            if stage_index > 1:
                backward = stages[stage_index - 1]
            elif stage_index == 0 and len(stages) > 1:
                backward = stages[-1]
            elif index:
                preceding = block_stages[index - 1]
                backward = preceding[0] + (preceding[-1] if len(preceding) > 1 else [])
            else:
                backward = []
            for unit in stage:
                unit.set_modules_to_forward_prefetch(
                    [target for target in dict.fromkeys(forward) if target is not unit]
                )
                unit.set_modules_to_backward_prefetch(list(backward))
            position += 1


class HyperParallelizePolicy:
    """Keep Hyper-specific wrapping and scheduling outside the shared traversal."""

    def __init__(self, model, target_modules, backend, device_type, compiled=False):
        self.model = model
        self.target_modules = target_modules
        self.backend = backend
        self.wrap_targets = resolve_submodule_wrap_targets(
            target_modules, root_model=model, device_type=device_type, compiled=compiled
        )
        self.prefetch_stages = {}

    def exclude_persistent(self, params):
        self.wrap_targets = {
            block: children
            for block, children in self.wrap_targets.items()
            if not any(param in params for param in block.parameters())
        }

    def shard_placement(self, param):
        shard_dim = getattr(param, "_veomni_fsdp_shard_dim", None)
        return self.backend.shard(shard_dim) if shard_dim is not None else None

    def set_gradient_divide_factor(self, module, mesh, divisor):
        module.set_gradient_scaling_factor(extra_parallel_gradient_scaling_factor(mesh.size(), divisor))

    def wrap_submodules(self, block, fsdp_kwargs):
        children = self.wrap_targets.get(block)
        if children is None:
            return
        existing_units = list(block._fsdp_modules)
        for child in children:
            if not isinstance(child, self.backend.module_type):
                self.backend.fully_shard(child, **fsdp_kwargs)
                block._fsdp_modules.append(child)
        self.prefetch_stages[block] = build_submodule_prefetch_stages(block, children, existing_units)

    def set_reduce_comm_interval(self, interval):
        if interval == 1:
            return
        setter = getattr(self.model, "set_reduce_comm_interval", None)
        if setter is None:
            raise RuntimeError("This HyperParallel version does not support set_reduce_comm_interval.")
        setter(interval)

    def configure_prefetch(self):
        if not self.wrap_targets:
            return False
        from ..utils.logging import get_logger

        # Traversal orders blocks; do not reuse bottom-up wrapping order.
        block_stages = [
            self.prefetch_stages[block] if block in self.prefetch_stages else [list(reversed(block._fsdp_modules))]
            for _, block in self.target_modules
        ]
        configure_submodule_prefetch(block_stages)
        logger = get_logger(__name__)
        logger.info_rank0("[hyper-forward-prefetch] policy=stage depth=2")
        names = {id(module): name for name, module in self.model.named_modules()}
        logger.info_rank0(
            "[hyper-submodule-prefetch] stages="
            f"{[[[names[id(unit)] for unit in stage] for stage in stages] for stages in block_stages]}"
        )
        return True


def validate_hyper_checkpoint_config(args):
    """Reject unsupported Hyper checkpoint modes before model construction."""
    checkpoint = args.train.checkpoint
    if checkpoint.manager != "dcp":
        raise ValueError("fsdp_backend='hyper' requires train.checkpoint.manager='dcp'.")
    if checkpoint.stage_dir is not None:
        raise ValueError("fsdp_backend='hyper' does not support train.checkpoint.stage_dir.")
    if checkpoint.dcp_save_to_lowest_rank:
        raise ValueError("fsdp_backend='hyper' does not support train.checkpoint.dcp_save_to_lowest_rank.")
    if checkpoint.save_hf_weights:
        raise ValueError(
            "fsdp_backend='hyper' does not yet support Hugging Face export; "
            "set train.checkpoint.save_hf_weights=false."
        )
    if args.model.lora_config and (
        checkpoint.load_path is not None or checkpoint.save_steps or checkpoint.save_epochs
    ):
        raise ValueError(
            "fsdp_backend='hyper' does not yet support trainable-only LoRA checkpoints; "
            "disable checkpoint save/resume for this run."
        )


def hyper_clip_grad_norm(model, max_norm, norm_type, error_if_nonfinite, foreach):
    """Synchronize Hyper gradients before shared EP or backend norm clipping."""
    from hyper_parallel import hsdp_sync_stream

    from .fsdp2.clip_grad_norm import extra_parallel_fsdp2_clip_grad_norm

    hsdp_sync_stream()
    if hasattr(model, "_extra_parallel_param_groups"):
        return extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach
        )
    from hyper_parallel.core.utils import clip_grad_norm_

    return clip_grad_norm_(
        model, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach
    )
