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

"""Hyper tensor interoperability and weight loading using VeOmni's readers and converters."""

from __future__ import annotations

import itertools
import os
import time
from collections.abc import Mapping
from functools import partial, singledispatch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from safetensors import safe_open
from torch import distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Replicate
from tqdm import tqdm

from ..distributed.parallel_state import get_parallel_state
from ..distributed.torch_parallelize import _has_extra_parallel_plan
from ..distributed.torch_parallelize import (
    _materialize_and_load_weights as _native_materialize_and_load_weights,
)
from ..lora.state_dict import insert_adapter_name, load_adapter_state_dict
from ..models.checkpoint_tensor_loading import (
    checkpoint_converter_fused_expert_target,
    checkpoint_converter_is_dim0_zero_pad,
    checkpoint_converter_record_skip_without_loading,
    checkpoint_converter_should_skip_without_loading,
    get_checkpoint_tensor_converter,
    maybe_convert_checkpoint_tensor,
)
from ..models.module_utils import (
    BroadcastMetadata,
    _convert_weight_key,
    _ep_dim0_slice_meta,
    _find_submodule,
    _get_communication_device,
    _is_all_zero_tensor,
    _load_state_dict,
    _num_storage_bytes,
    _param_larger_than,
    _read_ep_dim0_slice,
    _requires_byte_broadcast,
    _resolve_safetensors_shards,
    _view_as_bytes,
    _view_from_bytes,
    post_process_after_weight_loading,
)
from ..models.module_utils import (
    _dispatch_parameter as _native_dispatch_parameter,
)
from ..utils import logging
from ..utils.device import get_device_type
from ..utils.helper import empty_cache


if TYPE_CHECKING:
    from hyper_parallel import DTensor
    from transformers import PreTrainedModel

    from ..distributed.parallel_plan import ParallelPlan

logger = logging.get_logger(__name__)


@singledispatch
def get_dtensor_adapter(tensor):
    """Return layout semantics registered within this Hyper integration."""
    return None


class DTensorAdapter:
    """Torch DTensor semantics; optional implementations register their own type."""

    local_updates = False

    def load_parameter(self, param, tensor, *, full_param_name, dtensor_factory, parallel_plan, dtensor_to_cpu):
        """Return False to retain the existing parameter loader unchanged."""
        return False

    def from_local(self, value, ref, *, placements=None, global_shape=False):
        options = {"shape": ref.shape, "stride": ref.stride()} if global_shape else {}
        return TorchDTensor.from_local(
            value,
            device_mesh=ref.device_mesh,
            placements=ref.placements if placements is None else placements,
            run_check=False,
            **options,
        )

    def replicated_placements(self, mesh):
        return [Replicate()] * mesh.ndim

    def full_gradient(self, value, ref):
        return self.from_local(value, ref).full_tensor()

    def shard_full(self, value, ref):
        mesh = ref.device_mesh
        replicated = self.from_local(value, ref, placements=self.replicated_placements(mesh))
        result = replicated.redistribute(device_mesh=mesh, placements=ref.placements)
        return result.to_local() if self.local_updates else result


_TORCH_ADAPTER = DTensorAdapter()


@get_dtensor_adapter.register(TorchDTensor)
def _torch_dtensor_adapter(tensor):
    return _TORCH_ADAPTER


def is_dtensor(tensor):
    return get_dtensor_adapter(tensor) is not None


def uses_local_updates(tensor):
    adapter = get_dtensor_adapter(tensor)
    return adapter is not None and adapter.local_updates


def is_shard_placement(placement):
    predicate = getattr(placement, "is_shard", None)
    return bool(predicate()) if callable(predicate) else False


def is_replicate_placement(placement):
    predicate = getattr(placement, "is_replicate", None)
    return bool(predicate()) if callable(predicate) else False


def owner_path_update(update, ref):
    """Preserve distributed updates or attach the parameter layout to local updates."""
    if is_dtensor(update):
        return update
    adapter = get_dtensor_adapter(ref)
    return (
        adapter.from_local(update, ref, global_shape=True) if adapter is not None and adapter.local_updates else None
    )


def full_gradient(update, ref=None):
    if is_dtensor(update):
        return update.full_tensor()
    adapter = get_dtensor_adapter(ref)
    return adapter.full_gradient(update, ref) if adapter is not None and adapter.local_updates else update


def shard_full(value, ref):
    adapter = get_dtensor_adapter(ref)
    return adapter.shard_full(value, ref) if adapter is not None else value


class HyperDTensorAdapter(DTensorAdapter):
    """Hyper exposes local data and applies optimizer math on local tensors."""

    local_updates = True

    def load_parameter(self, param, tensor, *, full_param_name, dtensor_factory, parallel_plan, dtensor_to_cpu):
        """Distribute a checkpoint tensor and update Hyper's local data in place."""
        if parallel_plan is not None:
            tensor = parallel_plan.shard_tensor(tensor, full_param_name, param.shape)
        if dtensor_factory is None:
            raise ValueError("dtensor parameter requires a dtensor_factory.")
        sharded = dtensor_factory(tensor.to(dtype=param.data.dtype), param.device_mesh, param.placements)
        if dtensor_to_cpu:
            sharded = sharded.to("cpu")
        param.data.copy_(sharded.to_local())
        return True

    def from_local(self, value, ref, *, placements=None, global_shape=False):
        from hyper_parallel.core.dtensor.dtensor import DTensor

        options = {"shape": tuple(ref.shape)} if global_shape else {}
        return DTensor.from_local(
            value,
            device_mesh=ref.device_mesh,
            placements=ref.placements if placements is None else placements,
            **options,
        )

    def replicated_placements(self, mesh):
        from hyper_parallel.core.dtensor.placement_types import Replicate

        return [Replicate()] * mesh.ndim


def register_dtensor_adapter(tensor_type):
    adapter = HyperDTensorAdapter()
    get_dtensor_adapter.register(tensor_type, lambda tensor: adapter)


def _dispatch_parameter(
    module,
    name,
    tensor,
    dtensor_factory=None,
    parallel_plan=None,
    dtensor_to_cpu=False,
    tensor_is_local_dtensor_shard=False,
):
    owner, local_name = _find_submodule(module, name)
    parameter = owner._parameters[local_name]
    adapter = get_dtensor_adapter(parameter)
    if adapter is None or not adapter.local_updates:
        return _native_dispatch_parameter(
            module, name, tensor, dtensor_factory, parallel_plan, dtensor_to_cpu, tensor_is_local_dtensor_shard
        )
    if tensor_is_local_dtensor_shard:
        target = parameter.to_local()
        if tuple(tensor.shape) != tuple(target.shape):
            raise RuntimeError(f"Local Hyper shard shape mismatch for {name}: {tensor.shape} != {target.shape}")
        target.copy_(tensor.to(target))
        return
    adapter.load_parameter(
        parameter,
        tensor,
        full_param_name=name,
        dtensor_factory=dtensor_factory,
        parallel_plan=parallel_plan,
        dtensor_to_cpu=dtensor_to_cpu,
    )


@torch.no_grad()
def load_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu", "mlu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    **kwargs,
) -> None:
    """
    Loads pre-trained model states in transformers' format.
    """
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}
    model.to_empty(device=init_device)
    dtensor_to_cpu = init_device == "cpu"

    # Get parallel plan if available -- via the runtime helper which
    # detects PEFT and prepends ``base_model.model.`` to every plan
    # pattern so EP-aware ``parallel_plan.shard_tensor`` matches the
    # full PEFT-namespaced ``full_param_name`` it'll see below.
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        from ..distributed.parallel_plan import get_runtime_parallel_plan

        parallel_plan = get_runtime_parallel_plan(model)

    # Remap bare base-model checkpoint keys to their live (PEFT-wrapped) FQNs.
    # Applied AFTER ``maybe_convert_checkpoint_tensor`` so converter-produced merged
    # keys (e.g. Qwen3-MoE per-expert -> fused ``...experts.gate_up_proj``) also flow
    # through the ``base_layer.weight`` rename. No-op when not PEFT.
    is_peft_model = kwargs.get("is_peft_model", False)
    adapter_path = kwargs.get("adapter_path", None)
    from ..lora.weight_loading import make_peft_key_mapper

    _apply_peft_override = make_peft_key_mapper(model, is_peft_model)

    converter = get_checkpoint_tensor_converter(model)
    if converter is None and is_peft_model:
        converter = get_checkpoint_tensor_converter(model.get_base_model())
    state_dict_iterators = _load_state_dict(weights_path)

    def _dispatch_kv(name: str, tensor: "torch.Tensor") -> None:
        if name in buffer_dict.keys():  # persistent buffers
            buffer_dict[name] = tensor.clone()
        elif name in parameter_names_to_load:
            parameter_names_to_load.remove(name)
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan, dtensor_to_cpu)
        else:
            logger.info_rank0(f"Unexpected key in state dict: {name}.")

    for state_dict_iterator in tqdm(
        state_dict_iterators, desc="Loading checkpoint shards", disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        for name, tensor in state_dict_iterator:
            name = _convert_weight_key(name, model)
            converted = maybe_convert_checkpoint_tensor(name, tensor, converter)
            if converted is None:
                continue
            _dispatch_kv(_apply_peft_override(converted.name), converted.tensor)

        del state_dict_iterator
        empty_cache()

    if converter is not None:
        for result in converter.finalize():
            _dispatch_kv(_apply_peft_override(result.name), result.tensor)

    if is_peft_model and adapter_path:
        # Load LoRA adapter weights when an adapter_path is provided; otherwise
        # they are initialised in post_process_after_weight_loading. The native
        # VeOmniLoraModel reads the PEFT-format file without importing peft.

        load_lora_weights(
            model,
            adapter_path,
            init_device,
            dtensor_factory,
            parameter_names_to_load=parameter_names_to_load,
            # EP-aware slicing: ``LoraIndependentExperts`` LoRA tensors are shrunk
            # from ``[E, ...]`` to ``[E_local, ...]`` inside ``_dispatch_parameter``
            # before the DTensor copy, or the propagation asserts on shape mismatch.
            parallel_plan=parallel_plan,
        )

    post_process_after_weight_loading(
        model, buffer_dict, parameter_names_to_load, dtensor_factory, dtensor_to_cpu=dtensor_to_cpu
    )

    fqn_to_index_mapping = kwargs.get("fqn_to_index_mapping")
    if fqn_to_index_mapping is not None:
        from ..models.checkpoint_tensor_loading import prepare_fqn_to_index_mapping_for_model

        prepare_fqn_to_index_mapping_for_model(model, fqn_to_index_mapping)


def _read_persistent_extra_parallel_slice(
    sl: Any,
    *,
    name: str,
    parameter: DTensor,
    zero_pad_dim0: bool,
    mesh_coordinate: Optional[Tuple[int, ...]] = None,
) -> "torch.Tensor":
    """Read one persistent ExtraParallel DTensor rectangle from safetensors.

    Unlike ordinary ExtraParallel+FSDP parameters, the model parameter already
    carries every shard placement. The loader must therefore slice once for
    each ``Shard`` placement and wrap the result as an already-local DTensor
    shard instead of feeding it through ``distribute_tensor`` again.
    """
    if parameter.device_mesh.ndim != 2 or parameter.ndim != 2:
        raise NotImplementedError(
            f"Persistent ExtraParallel streaming currently supports 2D tensors on 2D meshes; "
            f"got tensor ndim={parameter.ndim}, mesh ndim={parameter.device_mesh.ndim} for {name}."
        )

    placements = parameter.placements
    shard_dims = [placement.dim for placement in placements if placement.is_shard()]
    if len(shard_dims) != 2 or len(set(shard_dims)) != 2:
        raise NotImplementedError(
            f"Persistent ExtraParallel streaming requires two distinct Shard placements for {name}; got {placements}."
        )

    real_shape = tuple(sl.shape) if isinstance(sl, torch.Tensor) else tuple(sl.get_shape())
    global_shape = tuple(parameter.shape)
    local_shape = tuple(parameter.to_local().shape)
    if len(real_shape) != len(global_shape):
        raise RuntimeError(f"{name}: checkpoint shape {real_shape} does not match model rank {len(global_shape)}.")

    for dim, (real_dim, global_dim) in enumerate(zip(real_shape, global_shape)):
        if dim == 0 and zero_pad_dim0:
            if real_dim > global_dim:
                raise RuntimeError(f"{name}: checkpoint dim0={real_dim} exceeds model dim0={global_dim}.")
        elif real_dim != global_dim:
            raise RuntimeError(
                f"{name}: checkpoint dimension {dim}={real_dim} does not match model dimension {global_dim}."
            )

    read_slices = [slice(0, dim) for dim in real_shape]
    output_slices = [slice(0, dim) for dim in local_shape]
    for mesh_dim, placement in enumerate(placements):
        if not placement.is_shard():
            continue
        tensor_dim = placement.dim
        mesh_dim_name = parameter.device_mesh.mesh_dim_names[mesh_dim]
        mesh_rank = (
            parameter.device_mesh.get_local_rank(mesh_dim_name)
            if mesh_coordinate is None
            else mesh_coordinate[mesh_dim]
        )
        start = mesh_rank * local_shape[tensor_dim]
        end = start + local_shape[tensor_dim]
        read_start = min(start, real_shape[tensor_dim])
        read_end = min(end, real_shape[tensor_dim])
        read_slices[tensor_dim] = slice(read_start, read_end)
        output_slices[tensor_dim] = slice(0, max(read_end - read_start, 0))

    tensor = sl[tuple(read_slices)]
    if tuple(tensor.shape) == local_shape:
        return tensor

    if not zero_pad_dim0:
        raise RuntimeError(f"{name}: loaded local shape {tuple(tensor.shape)} does not match expected {local_shape}.")
    output = torch.zeros(local_shape, dtype=tensor.dtype)
    output[tuple(output_slices)] = tensor
    return output


@torch.no_grad()
def load_model_weights_ep_sharded(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    **kwargs,
) -> None:
    """Per-rank ExtraParallel-slice streaming loader (opt-in alternative to
    :func:`load_model_weights`).

    For parameters the model's ``get_parallel_plan`` marks ExtraParallel-sharded
    (e.g. MoE experts, ``Shard(0)`` over the ``ep`` mesh), this reads **only this
    rank's dim-0 slice** straight from the safetensors shard via
    ``safe_open(...).get_slice()[start:end]`` -- instead of reading the whole
    ``[E, ...]`` tensor and slicing it in host memory (what
    :func:`load_model_weights` does via ``ParallelPlan.shard_tensor``).

    Why it is faster / lighter for large MoE checkpoints:
      * per-rank disk/HDFS bytes for expert tensors drop from the whole set to
        ``1/ep`` of it (in aggregate the expert bytes are read ~once across the
        ``ep`` ranks, in parallel), vs every rank reading the full expert set and
        discarding ``(ep-1)/ep``;
      * peak host RAM drops accordingly -- the full ``[E, ...]`` tensor is never
        materialised, only the ``[E/ep, ...]`` local shard.

    Dense (non-ExtraParallel) params and buffers are read whole (small relative
    to experts) and dispatched exactly as in :func:`load_model_weights` (FSDP's
    ``distribute_tensor`` still shards them). The dispatch of a sliced expert is
    identical to the whole-tensor path's second half: pass the already-sliced
    ``[E/ep, ...]`` with ``parallel_plan=None`` so it is not sliced again, then
    the FSDP ``dtensor_factory`` shards it over the ``ep_fsdp`` sub-mesh.

    Raises ``NotImplementedError`` when the checkpoint/model is unsupported: no
    ExtraParallel ``get_parallel_plan``, or a checkpoint-tensor converter whose
    transform needs the whole tensor set. Two converter transforms stay streamable:

      * a pure dim-0 zero-pad -- each rank reads its real-row slice and zero-fills
        the tail -- opted into via ``CheckpointTensorConverter.is_dim0_zero_pad``;
      * per-expert -> fused stacking (e.g. an HF per-expert MoE checkpoint), opted
        into via ``fused_expert_target`` + ``for_expert_range``: each rank reads only
        the per-expert tensors in its ``Shard(0)`` expert range and stacks them into
        its local ``[E/ep, ...]`` slice; other ranks' experts are never read. Every
        rank raises ``RuntimeError`` if a built fused target that has per-expert keys
        does not have them for exactly experts ``0..E-1``, each with the same number
        of source keys (e.g. both ``gate_proj`` and ``up_proj``).

    PEFT is supported. Base-checkpoint keys are remapped to their PEFT
    ``base_layer`` FQNs (:func:`build_lora_key_overrides`) and streamed exactly
    like the non-PEFT base -- each rank keeps only its ``[E/ep, ...]`` expert
    slice (the fused MoE experts live under ``...experts.<spec>.base_layer.weight``
    and stay EP ``Shard(0)`` in the runtime plan). The LoRA adapter then streams via
    :func:`_stream_lora_adapter_ep_sharded` with that same runtime plan, which decides
    sharding per tensor: **independent** MoE LoRA tensors are 3-D ``Shard(0)`` over the
    ep group, so each rank reads only its own expert-row dim-0 slice from the
    safetensors adapter; **shared** MoE LoRA (2-D) and dense LoRA tensors are not in
    the plan, so every rank reads them whole (replicated). Un-loaded LoRA params -- a
    fresh build (no ``adapter_path``) or any params the adapter omits -- are
    kaiming-``A``/zero-``B`` initialised in ``post_process_after_weight_loading``.
    """
    is_peft_model = kwargs.get("is_peft_model", False)
    adapter_path = kwargs.get("adapter_path", None)

    # A checkpoint-tensor converter generally needs the whole tensor set, which
    # streaming can't provide. The streamable exceptions (dim-0 zero-pad, and
    # per-expert stacking restricted to this rank's experts) are decided per key
    # below. Under PEFT the converter lives on the *base* model (the wrapper has none).
    converter = get_checkpoint_tensor_converter(model)
    if converter is None and is_peft_model:
        converter = get_checkpoint_tensor_converter(model.get_base_model())

    # Runtime plan: PEFT-prefixes every pattern with ``base_model.model.`` and
    # registers the MoE-LoRA wrapper FQNs -- base experts under ``base_layer.weight``
    # stay EP ``Shard(0)``, and ``LoraIndependentExperts`` per-expert LoRA tensors
    # are added as ``Shard(0)`` (shared-LoRA tensors are left replicated). For a
    # non-PEFT / non-LoRA model this is a passthrough of ``get_parallel_plan()``.
    from ..distributed.parallel_plan import get_runtime_parallel_plan

    parallel_plan = get_runtime_parallel_plan(model)
    if parallel_plan is None or not getattr(parallel_plan, "extra_parallel_plan", None):
        raise NotImplementedError("ep_sharded_stream_load requires a model with an ExtraParallel parallel_plan.")

    # Base-checkpoint keys (bare base-model FQNs) -> their PEFT-wrapped destinations
    # (``base_model.model.<...>.base_layer.weight``). No-op when not PEFT.
    from ..lora.weight_loading import make_peft_key_mapper

    _apply_peft_override = make_peft_key_mapper(model, is_peft_model)

    # This streaming loader reads each rank's ExtraParallel slice with a dim-0
    # ``get_slice()[start:end]`` -- the same dim-0 assumption upstream's
    # ``ParallelPlan._slice_shard_tensor`` makes (every VeOmni ExtraParallel plan
    # today is ``Shard(0)``: MoE experts, embed parallel). If a plan ever shards a
    # non-zero dim, a blind dim-0 slice would silently corrupt weights, so bail to
    # the whole-tensor loader (which handles arbitrary ``Shard(dim)`` via DTensor).
    for _pname, _pplan in parallel_plan.extra_parallel_plan.items():
        for _fqn_pattern, _placement in _pplan.items():
            _dim = getattr(_placement, "dim", None)
            if _dim is not None and _dim != 0:
                raise NotImplementedError(
                    f"ep_sharded_stream_load only supports dim-0 ExtraParallel sharding (Shard(0)); "
                    f"'{_pname}' pattern '{_fqn_pattern}' uses Shard({_dim})."
                )

    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    params_by_name = dict(model.named_parameters())
    param_shapes = {name: tuple(p.shape) for name, p in params_by_name.items()}
    parameter_names_to_load = set(param_shapes.keys())

    parallel_state = get_parallel_state()
    key_to_file, file_to_path = _resolve_safetensors_shards(weights_path, **kwargs)

    def _local_expert_range(fused_name: str) -> Tuple[int, int, int]:
        """``(start, num_local, num_experts)`` of fused parameter ``fused_name`` on this rank."""
        target0 = param_shapes[fused_name][0]
        shard_group = parallel_plan._get_shard_parameter_groupname(fused_name)
        if shard_group is None:
            return 0, target0, target0
        spec_info = getattr(model, "_fqn2spec_info", {}).get(fused_name)
        if spec_info is not None and spec_info.persistent_fsdp_shard_dim is not None:
            raise NotImplementedError(
                f"ep_sharded_stream_load: converter target '{fused_name}' is a persistent ExtraParallel parameter."
            )
        _, para_rank, expected_full0 = _ep_dim0_slice_meta(parallel_state, shard_group, target0)
        return para_rank * target0, target0, expected_full0

    # Up-front scan of the *raw checkpoint keys* for a converter's non-dim0-zero-pad
    # transforms. It must run over raw keys rather than in the per-destination main loop:
    # a per-expert key (e.g. ``model.layers.0.mlp.experts.3.gate_proj.weight``) has no
    # model param of its own, so the loop's "unexpected key" skip would silently drop it
    # and leave the fused expert un-loaded.
    #
    # Per-expert -> fused stacking streams when the converter can name each key's fused
    # target and expert row (``fused_expert_target``) and build a converter restricted to
    # this rank's ``Shard(0)`` row range (``for_expert_range``): the rank then reads only
    # its own experts and stacks exactly its local slice. Any other transform (reshape,
    # dtype change, a converter without those capabilities) needs the whole tensor set,
    # so raise before materialising anything.
    expert_converter = None
    expert_range: Optional[Tuple[int, int]] = None
    # Per-expert keys whose fused target the model does not build (e.g. a checkpoint's MTP
    # experts when MTP is off): unexpected keys, skipped without reading.
    unbuilt_expert_keys = set()
    if converter is not None:
        expert_ranges: Dict[Tuple[int, int], str] = {}
        # Source-key count per expert of each fused target. A converter's ``finalize`` only
        # sees its own rank's range, so a gap there would fail one rank (or, for a wholly
        # absent range, none); checking here makes every rank reject the checkpoint.
        expert_key_counts: Dict[str, Dict[int, int]] = {}
        num_experts_by_target: Dict[str, int] = {}
        for raw_name in key_to_file:
            bare_name = _convert_weight_key(raw_name, model)
            if checkpoint_converter_should_skip_without_loading(converter, bare_name):
                continue
            if not converter.can_handle(bare_name) or checkpoint_converter_is_dim0_zero_pad(converter, bare_name):
                continue
            target = checkpoint_converter_fused_expert_target(converter, bare_name)
            if target is None:
                raise NotImplementedError(
                    f"ep_sharded_stream_load: checkpoint-tensor converter "
                    f"{type(converter).__name__} needs a non-dim0-zero-pad transform for key "
                    f"'{bare_name}' and cannot stream it per expert. Per-rank slice streaming "
                    f"cannot reconstruct the whole tensor set from a single shard, so this "
                    f"model/checkpoint combination is unsupported. Either save the checkpoint "
                    f"in the model's fused expert layout, or disable model.ep_sharded_stream_load "
                    f"to use the whole-tensor loader (broadcast or every-rank-read)."
                )
            fused_name = _apply_peft_override(target[0])
            if fused_name not in param_shapes:
                unbuilt_expert_keys.add(raw_name)
                continue
            if fused_name not in num_experts_by_target:
                start, num_local, num_experts_by_target[fused_name] = _local_expert_range(fused_name)
                expert_ranges.setdefault((start, num_local), fused_name)
            counts = expert_key_counts.setdefault(fused_name, {})
            counts[target[1]] = counts.get(target[1], 0) + 1
        for fused_name, counts in expert_key_counts.items():
            expected = set(range(num_experts_by_target[fused_name]))
            problems = []
            if expected - counts.keys():
                problems.append(f"missing per-expert keys for experts {sorted(expected - counts.keys())}")
            if counts.keys() - expected:
                problems.append(f"unexpected experts {sorted(counts.keys() - expected)}")
            # Several source keys can fuse into one target (gate_proj + up_proj -> gate_up_proj);
            # every expert must contribute as many as the most complete one.
            full = max(counts.values())
            short = sorted(e for e, n in counts.items() if n < full)
            if short:
                problems.append(f"experts {short} have fewer than {full} source keys")
            if problems:
                raise RuntimeError(
                    f"ep_sharded_stream_load: incomplete checkpoint detected for '{fused_name}' "
                    f"({len(expected)} experts): " + "; ".join(problems) + "."
                )
        if len(expert_ranges) > 1:
            raise NotImplementedError(
                f"ep_sharded_stream_load: per-expert checkpoint keys map to fused experts with different "
                f"local expert ranges {expert_ranges}; one converter range per rank is supported."
            )
        if expert_ranges:
            expert_range = next(iter(expert_ranges))
            expert_converter = converter.for_expert_range(*expert_range)

    model.to_empty(device=init_device)
    dtensor_to_cpu = init_device == "cpu"

    keys_by_file: Dict[str, List[str]] = {}
    for key, fname in key_to_file.items():
        keys_by_file.setdefault(fname, []).append(key)

    n_ep = n_dense = n_buf = n_skipped = n_expert_reads = n_remote_experts = 0
    for fname in tqdm(
        sorted(keys_by_file),
        desc="Streaming EP-sharded checkpoint",
        disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
    ):
        with safe_open(file_to_path[fname], framework="pt", device="cpu") as f:
            for raw_name in keys_by_file[fname]:
                # ``bare_name`` is the base-model FQN the converter is keyed on;
                # ``name`` is the live model destination (PEFT ``base_layer`` FQN
                # under PEFT, else identical to ``bare_name``).
                bare_name = _convert_weight_key(raw_name, model)
                name = _apply_peft_override(bare_name)
                if checkpoint_converter_should_skip_without_loading(converter, bare_name):
                    checkpoint_converter_record_skip_without_loading(converter, bare_name)
                    n_skipped += 1
                    continue
                if raw_name in unbuilt_expert_keys:
                    logger.info_rank0(f"Unexpected key in state dict: {name}.")
                    continue
                expert_target = (
                    checkpoint_converter_fused_expert_target(converter, bare_name)
                    if expert_converter is not None
                    else None
                )
                if expert_target is not None:
                    start, num_local = expert_range
                    if not start <= expert_target[1] < start + num_local:
                        n_remote_experts += 1
                        continue
                    converted = expert_converter.convert(bare_name, f.get_tensor(raw_name))
                    n_expert_reads += 1
                    if converted is not None:
                        fused_name = _apply_peft_override(converted.name)
                        if fused_name not in parameter_names_to_load:
                            raise RuntimeError(
                                f"ep_sharded_stream_load: converter emitted '{fused_name}', which is not a "
                                f"parameter left to load."
                            )
                        if tuple(converted.tensor.shape) != param_shapes[fused_name]:
                            raise RuntimeError(
                                f"ep_sharded_stream_load: converter emitted {fused_name} with shape "
                                f"{tuple(converted.tensor.shape)}, expected the local {param_shapes[fused_name]}."
                            )
                        # Already the local slice -> parallel_plan=None (do not slice again).
                        _dispatch_parameter(model, fused_name, converted.tensor, dtensor_factory, None, dtensor_to_cpu)
                        parameter_names_to_load.discard(fused_name)
                        n_ep += 1
                    continue
                if name in buffer_dict:  # persistent buffers: read whole
                    buffer_dict[name] = f.get_tensor(raw_name).clone()
                    n_buf += 1
                    continue
                if name not in parameter_names_to_load:
                    logger.info_rank0(f"Unexpected key in state dict: {name}.")
                    continue

                shard_group = parallel_plan._get_shard_parameter_groupname(name)
                if shard_group is not None:
                    # ExtraParallel (e.g. EP / embed) param -> read only this rank's
                    # dim-0 slice. ``param_shapes[name][0]`` is the per-rank chunk, so
                    # the model's full dim0 is ``expected_full0 = target0 * para_size``.
                    # If a converter declares this key a pure dim-0 zero-pad, the model
                    # may have more rows than the checkpoint (``real0``): read the
                    # overlap and zero-fill the tail. Otherwise the checkpoint must
                    # match the model exactly.
                    zero_pad = checkpoint_converter_is_dim0_zero_pad(converter, bare_name)
                    if converter is not None and converter.can_handle(bare_name) and not zero_pad:
                        # A converter that fuses/reshapes this key can't be streamed.
                        raise NotImplementedError(
                            f"ep_sharded_stream_load: converter applies a non-dim0-zero-pad "
                            f"transform to ExtraParallel key '{name}'."
                        )
                    spec_info = getattr(model, "_fqn2spec_info", {}).get(name)
                    if spec_info is not None and spec_info.persistent_fsdp_shard_dim is not None:
                        parameter = params_by_name[name]
                        if not is_dtensor(parameter):
                            raise RuntimeError(
                                f"Persistent ExtraParallel parameter {name!r} must remain a DTensor during loading."
                            )
                        tensor = _read_persistent_extra_parallel_slice(
                            f.get_slice(raw_name),
                            name=name,
                            parameter=parameter,
                            zero_pad_dim0=zero_pad,
                        )
                        _dispatch_parameter(
                            model,
                            name,
                            tensor,
                            dtensor_factory,
                            None,
                            dtensor_to_cpu,
                            tensor_is_local_dtensor_shard=True,
                        )
                        n_ep += 1
                        parameter_names_to_load.discard(name)
                        continue
                    target0 = param_shapes[name][0]
                    _, para_rank, expected_full0 = _ep_dim0_slice_meta(parallel_state, shard_group, target0)
                    tensor = _read_ep_dim0_slice(
                        f.get_slice(raw_name),
                        name=name,
                        target0=target0,
                        para_rank=para_rank,
                        expected_full0=expected_full0,
                        zero_pad=zero_pad,
                    )
                    # Already the local slice -> parallel_plan=None (do not slice
                    # again); dtensor_factory then shards over the ep_fsdp sub-mesh
                    # exactly as the whole-tensor path's post-shard_tensor half.
                    _dispatch_parameter(model, name, tensor, dtensor_factory, None, dtensor_to_cpu)
                    n_ep += 1
                else:
                    if converter is not None and converter.can_handle(bare_name):
                        # A streamable (dim-0 zero-pad) converter must only touch
                        # ExtraParallel tables; a dense key would need its conversion
                        # applied here, which this path does not do -> bail.
                        raise NotImplementedError(
                            f"ep_sharded_stream_load: converter handles non-ExtraParallel key '{name}'."
                        )
                    tensor = f.get_tensor(raw_name)
                    _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan, dtensor_to_cpu)
                    n_dense += 1
                parameter_names_to_load.discard(name)
        empty_cache()

    logger.info_rank0(
        f"ep_sharded_stream_load: read {n_ep} ExtraParallel-sliced, {n_dense} dense, {n_buf} buffer tensors/rank; "
        f"skipped {n_skipped} converter-declared tensors without reading; stacked {n_expert_reads} local "
        f"per-expert tensors and skipped {n_remote_experts} other ranks' without reading."
    )

    for active_converter in (converter, expert_converter):
        if active_converter is not None and active_converter.finalize():
            raise NotImplementedError(
                "ep_sharded_stream_load cannot dispatch converter outputs produced only at finalize time."
            )

    if is_peft_model and adapter_path:
        # Stream the LoRA adapter the same per-rank way as the base: independent MoE
        # LoRA (3-D ``Shard(0)`` in the runtime plan) is read one dim-0 expert slice
        # per rank; shared/dense LoRA (not in the plan) is read whole on every rank
        # (replicated). When ``adapter_path`` is None (fresh build) the LoRA params
        # stay un-loaded and are kaiming-A/zero-B init'd below.
        _stream_lora_adapter_ep_sharded(
            model,
            adapter_path,
            init_device,
            dtensor_factory,
            parallel_plan,
            parameter_names_to_load,
            param_shapes,
            parallel_state,
            dtensor_to_cpu,
        )

    post_process_after_weight_loading(
        model, buffer_dict, parameter_names_to_load, dtensor_factory, dtensor_to_cpu=dtensor_to_cpu
    )


@torch.no_grad()
def _stream_lora_adapter_ep_sharded(
    model: "nn.Module",
    adapter_path: str,
    init_device: str,
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]],
    parallel_plan: Any,
    parameter_names_to_load: set,
    param_shapes: Dict[str, tuple],
    parallel_state: Any,
    dtensor_to_cpu: bool,
    adapter_name: str = "default",
) -> None:
    """Per-rank ExtraParallel-slice streaming of a PEFT adapter.

    The adapter companion to :func:`load_model_weights_ep_sharded`. Reads the
    consolidated PEFT adapter (``adapter_model.safetensors``, full gathered
    ``[E, ...]`` tensors) and dispatches per tensor by the runtime ``parallel_plan``:

    * **independent** MoE LoRA tensors (3-D, ``Shard(0)`` over the ep group) are read
      one dim-0 expert slice per rank -- ``get_slice()[start:end]`` -- so each rank
      only ever materialises its ``[E/ep, ...]`` chunk (no whole-tensor host copy),
      then ``dtensor_factory`` shards it over the ``ep_fsdp`` sub-mesh (dispatch with
      ``parallel_plan=None`` since it's already the local slice);
    * **shared** MoE LoRA (2-D) and **dense** LoRA tensors are not in the plan, so
      every rank reads them whole (replicated) and dispatches through the plan (FSDP
      shards them normally) -- matching the user contract "shared 不用切分, each rank
      reads the full adapter".

    A pickled ``.bin`` adapter can't be partially read, so it falls back to the
    whole-file :func:`load_lora_weights` (correct, but independent LoRA is not
    streamed per-rank). Newly saved adapters are safetensors (see
    :func:`save_lora_adapter_with_dcp`).
    """
    from ..lora.state_dict import _find_adapter_file, insert_adapter_name

    # Fail closed: an explicitly configured adapter_path must resolve. Soft-skip
    # would let post_process fresh-init a random adapter and training would
    # proceed as if resume succeeded (other loaders propagate FileNotFoundError).
    file_path, is_safetensors = _find_adapter_file(adapter_path)

    if not is_safetensors:
        logger.warning_rank0(
            f"ep_sharded adapter: {adapter_path} is a pickled .bin (not sliceable); falling back to "
            "whole-file load_lora_weights (independent LoRA won't be streamed per-rank)."
        )
        load_lora_weights(
            model,
            adapter_path,
            init_device,
            dtensor_factory,
            parameter_names_to_load=parameter_names_to_load,
            parallel_plan=parallel_plan,
            adapter_name=adapter_name,
        )
        return

    n_ep = n_rep = 0
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for raw_key in f.keys():
            name = insert_adapter_name(raw_key, adapter_name)
            if name not in parameter_names_to_load:
                logger.info_rank0(f"ep_sharded adapter: unexpected key {name}.")
                continue
            shard_group = parallel_plan._get_shard_parameter_groupname(name)
            if shard_group is not None:
                # independent per-expert LoRA -> read only this rank's dim-0 slice.
                target0 = param_shapes[name][0]
                _, para_rank, expected_full0 = _ep_dim0_slice_meta(parallel_state, shard_group, target0)
                sl = f.get_slice(raw_key)
                real0 = sl.get_shape()[0]
                tensor = _read_ep_dim0_slice(
                    sl,
                    name=name,
                    target0=target0,
                    para_rank=para_rank,
                    expected_full0=expected_full0,
                    zero_pad=False,
                    strict_mismatch_message=(
                        f"ep_sharded adapter {name}: file dim0={real0} != model dim0={expected_full0} "
                        f"(independent LoRA expects the gathered [E, ...] tensor)."
                    ),
                )
                _dispatch_parameter(model, name, tensor, dtensor_factory, None, dtensor_to_cpu)
                n_ep += 1
            else:
                # shared / dense LoRA -> read whole (replicated); FSDP shards via plan.
                tensor = f.get_tensor(raw_key)
                _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan, dtensor_to_cpu)
                n_rep += 1
            parameter_names_to_load.discard(name)

    logger.info_rank0(
        f"ep_sharded adapter: streamed {n_ep} EP-sliced (independent) + {n_rep} whole "
        "(shared/dense) adapter tensors/rank."
    )


@torch.no_grad()
def rank0_load_and_broadcast_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda", "npu", "mlu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    cpu_load_param_name: List[str] = None,
    max_load_broadcast_size: float = 20.0,  # in GB
    **kwargs,
):
    """
    This functions serves as the same purpose as `load_model_weights`
    but reduces disk I/O by broadcasting weights from rank0.
    In comparison, `load_model_weights` would require every GPU to go through the entire model weights on disk.
    """
    if not dist.is_available() or not dist.is_initialized():
        logger.warning_once("Distributed environment not initialized, falling back to load_model_weights.")
        return load_model_weights(model, weights_path, init_device, dtensor_factory, **kwargs)

    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names_to_load = {name for name, _ in model.named_parameters()}
    model.to_empty(device=init_device)
    dtensor_to_cpu = init_device == "cpu"

    # Get parallel plan if available -- routed through the runtime helper
    # so PEFT-prefix bridging happens once per call (see ``load_model_weights``).
    parallel_plan = None
    if hasattr(model, "get_parallel_plan"):
        from ..distributed.parallel_plan import get_runtime_parallel_plan

        parallel_plan = get_runtime_parallel_plan(model)

    # Build LoRA key remapping when loading a base checkpoint into a PEFT-wrapped model.
    # non-lora-layer: xxx.xxx -> base_model.model.xxx.xxx
    # lora-layer: xxx.xxx.weight -> base_model.model.xxx.xxx.base_layer.weight
    is_peft_model = kwargs.get("is_peft_model", False)
    adapter_path = kwargs.get("adapter_path", None)
    from ..lora.weight_loading import make_peft_key_mapper

    _apply_peft_override = make_peft_key_mapper(model, is_peft_model)

    converter = get_checkpoint_tensor_converter(model)
    if converter is None and is_peft_model:
        converter = get_checkpoint_tensor_converter(model.get_base_model())
    global_rank = get_parallel_state().global_rank
    torch_device = _get_communication_device(init_device)

    def _broadcast_and_dispatch(name, shape, dtype, tensor):
        """Broadcast a single (name, tensor) from rank0 and dispatch it."""
        logger.info_rank0(f"rank0_load_and_broadcast_weights: broadcasting {name=}")
        broadcast_as_bytes = _requires_byte_broadcast(dtype)
        if global_rank == 0:
            tensor = tensor.to(torch_device, non_blocking=True)
            broadcast_tensor = _view_as_bytes(tensor) if broadcast_as_bytes else tensor
        else:
            if broadcast_as_bytes:
                broadcast_tensor = torch.empty(
                    _num_storage_bytes(shape, dtype), dtype=torch.uint8, device=torch_device
                )
            else:
                tensor = torch.empty(shape, dtype=dtype, device=torch_device)
                broadcast_tensor = tensor

        start_time = time.perf_counter()
        dist.broadcast(broadcast_tensor, src=0)
        logger.info_rank0(
            f"{name=}, {shape=}, {dtype=}, broadcast time (ms) spent: {1000 * (time.perf_counter() - start_time)}"
        )
        if broadcast_as_bytes:
            tensor = _view_from_bytes(broadcast_tensor, dtype, shape)

        if name in buffer_dict:
            buffer_dict[name] = tensor.detach().clone()
        elif name in parameter_names_to_load:
            parameter_names_to_load.discard(name)
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan, dtensor_to_cpu)
        else:
            if global_rank == 0:
                logger.info_rank0(f"Unexpected key in state dict: {name}.")
        del tensor

    # P2P chunk transfer parameter
    extra_parallel_shard_dst_cache: Dict[str, List[List[int]]] = {}
    p2p_chunk_tag_counter = 0

    def _next_chunk_p2p_tag() -> int:
        nonlocal p2p_chunk_tag_counter
        p2p_chunk_tag_counter += 1
        return p2p_chunk_tag_counter

    def _get_extra_parallel_shard_dst_ranks(
        parallel_state: Any,
        para_group_name: str,
        para_size: int,
        device: torch.device,
    ) -> List[List[int]]:
        """
        Build chunk_id -> [global ranks] table via one all_gather (cached for this load).

        Each global rank reports its extra_parallel_rank for `para_group_name`; ranks whose
        rank is in [0, para_size) are grouped by shard index.
        """
        if para_group_name in extra_parallel_shard_dst_cache:
            return extra_parallel_shard_dst_cache[para_group_name]
        if not dist.is_initialized():
            raise RuntimeError("_get_extra_parallel_shard_dst_ranks requires initialized process group")
        world_size = dist.get_world_size()
        mine = torch.tensor(
            [parallel_state.extra_parallel_rank(para_group_name)],
            dtype=torch.long,
            device=device,
        )

        gathered = [torch.empty_like(mine) for _ in range(world_size)]
        dist.all_gather(gathered, mine)
        table: List[List[int]] = [[] for _ in range(para_size)]
        for r in range(world_size):
            epr = int(gathered[r].item())
            assert 0 <= epr < para_size, (
                f"Global rank {r} reports extra_parallel_rank {epr} for parallel group {para_group_name}, which is out of range [0, {para_size})."
            )
            table[epr].append(r)
        extra_parallel_shard_dst_cache[para_group_name] = table
        return table

    def _chunk_and_broadcast_and_dispatch(name, shape, dtype, tensor):
        """Broadcast a single (name, tensor) from rank0 and dispatch it."""
        logger.info_rank0(f"rank0_load_and_broadcast_weights: chunking and broadcasting {name=}")

        assert name not in buffer_dict, f"Buffer {name} should not be chunked."
        assert name in parameter_names_to_load, f"Unexpected key in state dict: {name}."

        if global_rank == 0:
            assert tensor.device == torch.device("cpu"), "Large parameter should be loaded to CPU first."

        start_time = time.perf_counter()

        para_group_name = parallel_plan._get_shard_parameter_groupname(name)
        assert para_group_name is not None, (
            f"Parameter {name} can not be chunked as it is not part of any parallel group."
        )

        parallel_state = get_parallel_state()
        assert parallel_state.extra_parallel_enabled(para_group_name), (
            f"Parallel group {para_group_name} is not enabled for extra parallel."
        )

        module, local_name = _find_submodule(model, name)
        parameter = module._parameters[local_name]
        spec_info = getattr(model, "_fqn2spec_info", {}).get(name)
        if spec_info is not None and spec_info.persistent_fsdp_shard_dim is not None:
            # Persistent parameters retain both shard placements. Send each rank
            # its local rectangle rather than treating the global shape as an EP slice.
            mesh = parameter.device_mesh
            if sorted(mesh.mesh.reshape(-1).tolist()) != list(range(dist.get_world_size())):
                raise NotImplementedError("Persistent ExtraParallel chunk loading requires a world-complete mesh.")
            local_shape = tuple(parameter.to_local().shape)
            comm_device = torch.device(mesh.device_type)
            as_bytes = _requires_byte_broadcast(parameter.dtype)
            buffer = torch.empty(
                _num_storage_bytes(local_shape, parameter.dtype) if as_bytes else local_shape,
                dtype=torch.uint8 if as_bytes else parameter.dtype,
                device=comm_device,
            )
            para_size = parallel_state.extra_parallel_sizes[para_group_name]
            for coordinate in itertools.product(*(range(size) for size in mesh.mesh.shape)):
                dst = int(mesh.mesh[coordinate])
                tag = _next_chunk_p2p_tag()
                if global_rank == 0:
                    local = (
                        _read_persistent_extra_parallel_slice(
                            tensor,
                            name=name,
                            parameter=parameter,
                            zero_pad_dim0=shape[0] % para_size != 0,
                            mesh_coordinate=coordinate,
                        )
                        .to(device=comm_device, dtype=parameter.dtype)
                        .contiguous()
                    )
                    buffer.copy_(_view_as_bytes(local) if as_bytes else local)
                    if dst != 0:
                        dist.send(buffer, dst=dst, tag=tag)
                elif global_rank == dst:
                    dist.recv(buffer, src=0, tag=tag)
                if global_rank == dst:
                    local = _view_from_bytes(buffer, parameter.dtype, local_shape) if as_bytes else buffer
                    _dispatch_parameter(
                        model, name, local, dtensor_factory, None, dtensor_to_cpu, tensor_is_local_dtensor_shard=True
                    )
            parameter_names_to_load.discard(name)
            return

        shard_tensor = parameter.data
        # The EP slice is a logical parameter, before FSDP sharding.
        target_shape = parameter.shape

        para_size = parallel_state.extra_parallel_sizes[para_group_name]
        assert len(shape) >= 1, f"Original parameter {name} to be chunked has shape {shape}, which is 0-dim."
        assert len(target_shape) >= 1, f"Shard parameter {name} to get chunk has shape {target_shape}, which is 0-dim."

        if global_rank == 0:
            if shape[0] % para_size != 0:
                logger.info_rank0(
                    f"Parallel group {para_group_name} size {para_size} does not divide original parameter shape {shape} at dim 0."
                )

                target_size = list(target_shape)
                target_size[0] *= para_size
                target_size = torch.Size(target_size)
                loaded_size = tensor.size()
                pad_size = tuple(
                    (0, target_dim - loaded_dim) for target_dim, loaded_dim in zip(target_size, loaded_size)
                )
                pad_size = tuple(itertools.chain(*(pad_size[::-1])))

                logger.info_rank0(
                    f"Shard parameter shape = {target_shape}, Loaded parameter shape = {tensor.shape}, pad_size = {pad_size}"
                )

                tensor = torch.nn.functional.pad(tensor, pad_size, value=0.0)

            assert tensor.shape[0] // para_size == target_shape[0], (
                f"Parallel {para_group_name}: padded shape {shape} at dim 0 // {para_size} == {tensor.shape[0] // para_size}, not equal to {target_shape[0]}"
            )

            chunk_loaded_data = list(tensor.chunk(para_size, dim=0))

        is_shard_tensor_dtensor = hasattr(parameter, "device_mesh")
        if is_shard_tensor_dtensor:
            device_mesh = parameter.device_mesh
            shard_comm_device = torch.device(device_mesh.device_type)
        else:
            device_mesh = None
            shard_comm_device = _get_communication_device(init_device)

        chunk_broadcast_as_bytes = _requires_byte_broadcast(shard_tensor.dtype)
        if chunk_broadcast_as_bytes:
            broadcast_buffer = torch.empty(
                _num_storage_bytes(tuple(target_shape), shard_tensor.dtype),
                dtype=torch.uint8,
                device=shard_comm_device,
            )
        else:
            broadcast_buffer = torch.empty(
                target_shape,
                dtype=shard_tensor.dtype,
                device=shard_comm_device,
            )
        shard_dst_ranks = _get_extra_parallel_shard_dst_ranks(
            parallel_state, para_group_name, para_size, shard_comm_device
        )
        for chunk_id in range(para_size):
            # For example:
            #   if we have two params, ranks = [0, 1, 2, 3, 4, 5, 6, 7], then
            #   at extra_parallel_1 with para_size = 2,
            #     when chunk_id = 0, send_seq = [2, 4, 6], p2p tag = [1, 1, 1]
            #     when chunk_id = 1, send_seq = [1, 3, 5, 7], p2p tag = [2, 2, 2, 2]
            #   at extra_parallel_2 with para_size = 4,
            #     when chunk_id = 0, send_seq = [4], p2p tag = [3]
            #     when chunk_id = 1, send_seq = [1, 5], p2p tag = [4, 4]
            #     when chunk_id = 2, send_seq = [2, 6], p2p tag = [5, 5]
            #     when chunk_id = 3, send_seq = [3, 7], p2p tag = [6, 6]

            if dist.get_rank() == 0:
                if chunk_broadcast_as_bytes:
                    chunk_tensor = chunk_loaded_data[chunk_id].to(device=shard_comm_device, dtype=shard_tensor.dtype)
                    broadcast_buffer.copy_(_view_as_bytes(chunk_tensor))
                else:
                    broadcast_buffer.copy_(chunk_loaded_data[chunk_id].contiguous())
            dst_ranks = sorted(shard_dst_ranks[chunk_id])
            # One tag increment per chunk_id on every rank so the counter stays aligned.
            tag = _next_chunk_p2p_tag()
            send_seq = [d for d in dst_ranks if d != 0]
            extra_para_local_rank = parallel_state.extra_parallel_rank(para_group_name)

            if global_rank == 0:
                for dst in send_seq:
                    dist.send(broadcast_buffer, dst=dst, tag=tag)

                if global_rank in dst_ranks:
                    dispatch_buffer = (
                        _view_from_bytes(broadcast_buffer, shard_tensor.dtype, tuple(target_shape))
                        if chunk_broadcast_as_bytes
                        else broadcast_buffer
                    )
                    _dispatch_parameter(model, name, dispatch_buffer, dtensor_factory, None, dtensor_to_cpu)

            elif global_rank in send_seq:
                if is_shard_tensor_dtensor:
                    assert device_mesh.mesh.reshape(-1).tolist() == dst_ranks, (
                        f"Device mesh {device_mesh.mesh.tolist()} does not match dst ranks {dst_ranks}."
                    )

                assert extra_para_local_rank == chunk_id, f"Rank {global_rank} is not the shard {chunk_id} rank."
                dist.recv(broadcast_buffer, src=0, tag=tag)

                dispatch_buffer = (
                    _view_from_bytes(broadcast_buffer, shard_tensor.dtype, tuple(target_shape))
                    if chunk_broadcast_as_bytes
                    else broadcast_buffer
                )
                _dispatch_parameter(model, name, dispatch_buffer, dtensor_factory, None, dtensor_to_cpu)

        logger.info_rank0(
            f"{name=}, {shape=}, {dtype=}, chunk and broadcast time (ms) spent: {1000 * (time.perf_counter() - start_time)}"
        )

        parameter_names_to_load.discard(name)
        del tensor

    # --- Broadcast shard count ---
    state_dict_iterators = _load_state_dict(weights_path) if global_rank == 0 else None
    shard_count = len(state_dict_iterators) if global_rank == 0 else 0
    logger.info_rank0(f"rank0_load_and_broadcast_weights: {shard_count=} ")
    shard_count_tensor = torch.tensor(
        [shard_count],
        dtype=torch.int64,
        device=torch_device if torch_device.type != "cpu" else torch.device("cpu"),
    )
    dist.broadcast(shard_count_tensor, src=0)
    shard_count = int(shard_count_tensor.item())

    if global_rank == 0:
        shard_iterable = enumerate(
            tqdm(
                state_dict_iterators,
                desc="Loading checkpoint shards",
                disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
            )
        )
    else:
        shard_iterable = enumerate(range(shard_count))

    # iterate safetensor files; each file would have a iterator to read weight keys and tensors
    for _shard_idx, shard_payload in shard_iterable:
        state_dict_iterator = shard_payload if global_rank == 0 else None
        iterator = iter(state_dict_iterator) if global_rank == 0 else None

        while True:
            tensor: Optional["torch.Tensor"] = None

            if global_rank == 0:
                while True:
                    # Inner loop: rank0 keeps reading tensors until the converter
                    # produces a result or the shard is exhausted. The converter may
                    # return None to indicate "still accumulating" (e.g. collecting
                    # per-expert MoE tensors), so we continue without broadcasting.
                    try:
                        key, tensor = next(iterator)  # type: ignore[arg-type]
                        key = _convert_weight_key(key, model)
                        converted = maybe_convert_checkpoint_tensor(key, tensor, converter)
                        if converted is None:
                            continue
                        key, tensor = converted.name, converted.tensor
                        # PEFT override is applied AFTER the converter so that
                        # converter-produced merged keys (e.g. Qwen3-MoE
                        # per-expert -> ``...experts.gate_up_proj``) also get
                        # mapped to their PEFT-wrapped ``...base_layer.weight``
                        # destination when the experts module is wrapped by
                        # ``LoraSharedExperts`` / ``LoraIndependentExperts``.
                        # Bare key in (converter matches on base-model FQNs) ->
                        # live-model key out; no-op when not PEFT.
                        key = _apply_peft_override(key)
                        logger.info_rank0(f"loading {key=}")
                        if _is_all_zero_tensor(tensor):
                            logger.warning_rank0(
                                f"Detected tensor with all-zero values when reading safetensor: {key=}"
                            )
                        metadata = BroadcastMetadata(False, key, tensor.shape, tensor.dtype)
                        break
                    except StopIteration:
                        metadata = BroadcastMetadata(True, None, None, None)
                        break
            else:
                metadata = BroadcastMetadata(False, None, None, None)

            metadata_list = [metadata]
            dist.broadcast_object_list(metadata_list, src=0)
            metadata = metadata_list[0]

            if metadata.done:
                break

            name = metadata.name
            shape = metadata.shape
            dtype = metadata.dtype

            if name is None or shape is None or dtype is None:
                raise RuntimeError("Received incomplete broadcast metadata.")
            if (
                (
                    (cpu_load_param_name is not None and name in cpu_load_param_name)
                    or _param_larger_than(shape, dtype, max_load_broadcast_size=max_load_broadcast_size)
                )
                and name in parameter_names_to_load
                and (parallel_plan is not None and parallel_plan._get_shard_parameter_groupname(name) is not None)
            ):
                _chunk_and_broadcast_and_dispatch(name, shape, dtype, tensor)
            else:
                _broadcast_and_dispatch(name, shape, dtype, tensor)

        if global_rank == 0:
            del state_dict_iterator

        empty_cache()

    # --- Flush converter (broadcast any finalized tensors) ---
    if converter is not None:
        finalized = converter.finalize() if global_rank == 0 else []
        fin_count_tensor = torch.tensor(
            [len(finalized)],
            dtype=torch.int64,
            device=torch_device if torch_device.type != "cpu" else torch.device("cpu"),
        )
        dist.broadcast(fin_count_tensor, src=0)
        fin_count = int(fin_count_tensor.item())

        for i in range(fin_count):
            if global_rank == 0:
                result = finalized[i]
                # Same post-converter PEFT override as the streaming loop
                # above -- finalize() may emit merged keys after every shard
                # has been read, e.g. the last gate/up pair for the
                # Qwen3-MoE per-expert -> fused converter.
                fin_name = _apply_peft_override(result.name)
                metadata = BroadcastMetadata(False, fin_name, result.tensor.shape, result.tensor.dtype)
                tensor = result.tensor
            else:
                metadata = BroadcastMetadata(False, None, None, None)
                tensor = None

            metadata_list = [metadata]
            dist.broadcast_object_list(metadata_list, src=0)
            metadata = metadata_list[0]

            name = metadata.name
            shape = metadata.shape
            dtype = metadata.dtype
            if name is None or shape is None or dtype is None:
                raise RuntimeError("Received incomplete broadcast metadata from finalize.")
            _broadcast_and_dispatch(name, shape, dtype, tensor)

    if is_peft_model and adapter_path:
        # Rank-0 reads the PEFT-format adapter file (natively, no peft import)
        # and broadcasts each tensor to all ranks. The runtime plan is forwarded
        # so EP-sharded LoRA tensors (registered by
        # ``_extend_plan_for_moe_lora_independent`` for ``LoraIndependentExperts``)
        # get sliced from the disk-side ``[E, ...]`` shape down to the local
        # ``[E_local, ...]`` shape inside ``_dispatch_parameter`` before the
        # DTensor ``.copy_()`` -- without this the copy asserts on a global-shape
        # mismatch (the ep_size=2 + ``mode=="independent"`` failure mode).

        rank0_load_and_broadcast_lora_weights(
            model,
            adapter_path,
            init_device,
            dtensor_factory,
            parameter_names_to_load=parameter_names_to_load,
            parallel_plan=parallel_plan,
        )

    post_process_after_weight_loading(
        model, buffer_dict, parameter_names_to_load, dtensor_factory, dtensor_to_cpu=dtensor_to_cpu
    )

    fqn_to_index_mapping = kwargs.get("fqn_to_index_mapping")
    if fqn_to_index_mapping is not None:
        from ..models.checkpoint_tensor_loading import prepare_fqn_to_index_mapping_for_model

        prepare_fqn_to_index_mapping_for_model(model, fqn_to_index_mapping)


@torch.no_grad()
def load_lora_weights(
    model: nn.Module | PreTrainedModel,
    adapter_path: str,
    init_device: str | None = None,
    dtensor_factory: Callable[[torch.Tensor, Any, Any], torch.Tensor] | None = None,
    parameter_names_to_load: set | None = None,
    parallel_plan: ParallelPlan | None = None,
    adapter_name: str = "default",
) -> None:
    """Load adapter weights on every rank (shared-filesystem path).

    Reads the PEFT-format adapter file natively, remaps on-disk keys
    (``...lora_A.weight``) to live-model keys (``...lora_A.<adapter>.weight``),
    and dispatches tensors into the (possibly sharded) model.
    """
    init_device = init_device or get_device_type()
    raw_sd = load_adapter_state_dict(adapter_path, device=init_device)
    for name, tensor in raw_sd.items():
        name = insert_adapter_name(name, adapter_name)
        _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
        if parameter_names_to_load is not None:
            parameter_names_to_load.discard(name)


@torch.no_grad()
def rank0_load_and_broadcast_lora_weights(
    model: nn.Module | PreTrainedModel,
    adapter_path: str,
    init_device: str | None = None,
    dtensor_factory: Callable[[torch.Tensor, Any, Any], torch.Tensor] | None = None,
    parameter_names_to_load: set | None = None,
    parallel_plan: ParallelPlan | None = None,
    adapter_name: str = "default",
) -> None:
    """Rank-0 reads the adapter file, then broadcasts each tensor to all ranks."""
    init_device = init_device or get_device_type()
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    adapter_sd: dict[str, torch.Tensor] = {}
    if global_rank == 0:
        raw_sd = load_adapter_state_dict(adapter_path, device="cpu")
        adapter_sd = {insert_adapter_name(k, adapter_name): v for k, v in raw_sd.items()}
        if adapter_sd:
            first_raw = next(iter(raw_sd))
            first_remapped = next(iter(adapter_sd))
            logger.info_rank0(
                f"Loaded {len(adapter_sd)} adapter weight(s) from {adapter_path}, "
                f"key remap example: {first_raw} -> {first_remapped}"
            )

    if not dist.is_available() or not dist.is_initialized():
        for name, tensor in adapter_sd.items():
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
        return

    global_rank = get_parallel_state().global_rank
    torch_device = torch.device(init_device)

    count_tensor = torch.tensor(
        [len(adapter_sd)],
        dtype=torch.int64,
        device=torch_device if torch_device.type != "cpu" else torch.device("cpu"),
    )
    dist.broadcast(count_tensor, src=0)
    num_keys = int(count_tensor.item())
    if num_keys == 0:
        return

    sorted_keys = sorted(adapter_sd.keys()) if global_rank == 0 else [None] * num_keys

    for i in range(num_keys):
        if global_rank == 0:
            name = sorted_keys[i]
            tensor = adapter_sd[name].to(torch_device, non_blocking=True)
            metadata = BroadcastMetadata(False, name, tensor.shape, tensor.dtype)
        else:
            metadata = BroadcastMetadata(False, None, None, None)

        metadata_list = [metadata]
        dist.broadcast_object_list(metadata_list, src=0)
        metadata = metadata_list[0]

        name = metadata.name
        shape = metadata.shape
        dtype = metadata.dtype

        if global_rank != 0:
            tensor = torch.empty(shape, dtype=dtype, device=torch_device)

        start_time = time.perf_counter()
        dist.broadcast(tensor, src=0)
        logger.info_rank0(
            f"{name=}, {shape=}, {dtype=}, broadcast time (ms): {1000 * (time.perf_counter() - start_time)}"
        )
        _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
        if parameter_names_to_load is not None:
            parameter_names_to_load.discard(name)
        del tensor

    logger.info_rank0(f"rank0_load_and_broadcast_lora_weights: loaded {num_keys} adapter param(s)")


def _materialize_and_load_weights(
    model: nn.Module,
    weights_path: Optional[str | Mapping[str, str]],
    materialize_device: str,
    *,
    should_skip_hf_weight_load: bool,
    is_peft_model: bool,
    adapter_path: Optional[str],
    broadcast_from_rank0: bool,
    cpu_load_param_name: Optional[List[str]] = None,
    max_load_broadcast_size: float = 20.0,
    fqn_to_index_mapping: Optional[dict] = None,
    ep_sharded_stream_load: bool = False,
    module_skip_hf_weight_load: Optional[Mapping[str, bool]] = None,
    module_is_peft_model: Optional[Mapping[str, bool]] = None,
    module_adapter_path: Optional[Mapping[str, Optional[str]]] = None,
    dtensor_factory=None,
) -> None:
    """Move meta-initialized parameters onto a real device and fill them in.

    Used only after Hyper FSDP wrapping. File readers, conversion helpers and
    native-tensor dispatch are shared with VeOmni; Hyper distribution stays local.

    ``weights_path`` is ``None`` (random init), a single HF snapshot for the
    whole ``model``, or a ``{child_name: snapshot_path}`` mapping for a composed
    model whose children each have their own split-checkpoint subfolder (SeedOmni
    ``fsdp_scope='model'``). Per-child skip / LoRA flags override the scalars
    when the mapping form is used.
    """
    if isinstance(weights_path, Mapping):
        children = dict(model.named_children())
        missing = [name for name in weights_path if name not in children]
        if missing:
            raise KeyError(
                f"weights_path mapping has unknown child module(s) {missing}; model children are {sorted(children)}."
            )
        skip_map = dict(module_skip_hf_weight_load or {})
        peft_map = dict(module_is_peft_model or {})
        adapter_map = dict(module_adapter_path or {})
        for name, path in weights_path.items():
            _materialize_and_load_weights(
                children[name],
                path,
                materialize_device,
                should_skip_hf_weight_load=bool(skip_map.get(name, should_skip_hf_weight_load)),
                is_peft_model=bool(peft_map.get(name, is_peft_model)),
                adapter_path=adapter_map.get(name, adapter_path),
                broadcast_from_rank0=broadcast_from_rank0,
                cpu_load_param_name=cpu_load_param_name,
                max_load_broadcast_size=max_load_broadcast_size,
                fqn_to_index_mapping=fqn_to_index_mapping,
                ep_sharded_stream_load=ep_sharded_stream_load,
                dtensor_factory=dtensor_factory,
            )
        leftover_meta = [
            name
            for name, child in children.items()
            if name not in weights_path and any(param.is_meta for param in child.parameters())
        ]
        if leftover_meta:
            raise ValueError(
                "weights_path mapping left meta-initialized children unmaterialized: "
                f"{leftover_meta}; mapped={sorted(weights_path)}."
            )
        return
    if weights_path is None or should_skip_hf_weight_load:
        # This native branch returns before choosing a Torch weight loader.
        # It also preserves non-persistent buffers and rejects trainable-only
        # LoRA restores that would leave frozen base weights uninitialized.
        return _native_materialize_and_load_weights(
            model,
            weights_path,
            materialize_device,
            should_skip_hf_weight_load=should_skip_hf_weight_load,
            is_peft_model=is_peft_model,
            adapter_path=adapter_path,
            broadcast_from_rank0=broadcast_from_rank0,
        )

    if dtensor_factory is None:
        from .parallelize import get_hyper_fsdp2_api

        dtensor_factory = get_hyper_fsdp2_api().distribute_tensor

    logger.info_rank0(f"starting to load model weights from {weights_path}...")
    if is_peft_model:
        if adapter_path is not None:
            logger.info_rank0(f"also loading lora adapter weights from {adapter_path}...")
        else:
            logger.info_rank0("also init peft model lora weights...")

    if broadcast_from_rank0:
        logger.info_rank0("Loading model weights from disk on rank0 then broadcasting to other ranks...")
        rank0_load_and_broadcast_weights(
            model,
            weights_path,
            materialize_device,
            dtensor_factory=dtensor_factory,
            cpu_load_param_name=cpu_load_param_name,
            max_load_broadcast_size=max_load_broadcast_size,
            is_peft_model=is_peft_model,
            adapter_path=adapter_path,
            fqn_to_index_mapping=fqn_to_index_mapping,
        )
    else:
        _dt_local_split = partial(dtensor_factory, src_data_rank=None)
        if ep_sharded_stream_load and _has_extra_parallel_plan(model):
            # Opt-in fast/low-memory path for large MoE checkpoints: each rank
            # reads only its ExtraParallel dim-0 slice of the expert tensors
            # straight from the checkpoint (see ``load_model_weights_ep_sharded``).
            # Only valid on the every-rank-reads (non-broadcast) path. An
            # unsupported *checkpoint* still raises ``NotImplementedError``, which
            # we surface directly rather than silently falling back -- an opt-in
            # flag that quietly degrades is hard to reason about.
            logger.info_rank0(
                "Loading model weights via per-rank ExtraParallel-slice streaming (ep_sharded_stream_load)..."
            )
            load_model_weights_ep_sharded(
                model,
                weights_path,
                materialize_device,
                dtensor_factory=_dt_local_split,
                is_peft_model=is_peft_model,
                adapter_path=adapter_path,
                fqn_to_index_mapping=fqn_to_index_mapping,
            )
        else:
            if ep_sharded_stream_load:
                # The flag is set once for a whole run, but this helper runs once
                # per model -- once per OmniModule under SeedOmni V2 -- and a model
                # with no ExtraParallel plan (vision encoder, VAE, connector) has
                # no expert tensors to stream. Skipping is not a silent degrade:
                # there was never a fast path to take here, and refusing instead
                # would make the flag unusable for any heterogeneous model whose
                # MoE backbone wants it.
                logger.info_rank0("Ignoring ep_sharded_stream_load for a model with no ExtraParallel parallel_plan.")
            logger.info_rank0("Every rank would read weights from disk and expect this to be slow!")
            load_model_weights(
                model,
                weights_path,
                materialize_device,
                dtensor_factory=_dt_local_split,
                is_peft_model=is_peft_model,
                adapter_path=adapter_path,
                fqn_to_index_mapping=fqn_to_index_mapping,
            )
