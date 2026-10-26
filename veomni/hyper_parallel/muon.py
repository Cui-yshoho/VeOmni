# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Optional Hyper Muon construction using VeOmni parameter groups and NS kernels."""

import json
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..distributed.parallel_state import get_parallel_state
from ..optim.muon import (
    _KIND_FSDP_GATHER_2D,
    _KIND_LOCAL,
    _KIND_MOE_GATHER_3D,
    _KIND_MOE_LOCAL_3D,
    _fsdp_all2all_bucket_key,
    run_newton_schulz,
    split_muon_adamw_params,
)
from ..optim.muon import (
    DistributedMuon as NativeDistributedMuon,
)
from ..utils import logging
from ..utils.device import get_device_type
from .module_utils import (
    full_gradient,
    is_dtensor,
    shard_full,
    uses_local_updates,
)
from .module_utils import (
    is_replicate_placement as _is_replicate_placement,
)
from .module_utils import (
    is_shard_placement as _is_shard_placement,
)
from .module_utils import (
    owner_path_update as _owner_path_update,
)


logger = logging.get_logger(__name__)


def _run_hyper_muon_newton_schulz(grad: torch.Tensor, steps: int, **kwargs) -> torch.Tensor:
    """Adapt Hyper's owner-batched callback to VeOmni's NS dispatcher."""
    return run_newton_schulz(grad, ns_steps=steps, **kwargs)


def _guard_hyper_muon_callback_resume(optimizer, callback, ns_steps: int) -> None:
    """Persist callback settings and reject inconsistent optimizer restores."""
    if callback is None:

        def reject_callback_state(_optimizer, state_dict):
            if any("veomni_ns_callback" in group for group in state_dict["param_groups"]):
                raise ValueError("Hyper Muon Gram/QuACK checkpoint settings cannot be restored into std.")

        optimizer.register_load_state_dict_pre_hook(reject_callback_state)
        return
    signature = json.dumps(dict(callback.keywords, ns_steps=ns_steps), sort_keys=True)
    coefficients = [tuple(callback.keywords["ns_coefficients"])] * ns_steps

    def validate_state_dict(_optimizer, state_dict):
        for group in state_dict["param_groups"]:
            if (
                group.get("veomni_ns_callback") != signature
                or [tuple(row) for row in group.get("ns_coefficients", ())] != coefficients
                or group.get("ns_epsilon") != callback.keywords["eps"]
                or group.get("ns_steps") != ns_steps
            ):
                raise ValueError("Hyper Muon Gram/QuACK checkpoint settings do not match the configured NS callback.")

    for group in optimizer.param_groups:
        group["veomni_ns_callback"] = signature
    optimizer.register_load_state_dict_pre_hook(validate_state_dict)


def build_hyper_muon_with_adamw(
    model: "nn.Module",
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    fused: bool,
    no_decay_modules: Optional[List[str]],
    no_decay_params: Optional[List[str]],
    muon_kwargs: Optional[Dict[str, Any]],
):
    """Build HyperParallel's HSDP-aware Muon while retaining VeOmni parameter grouping."""
    from .optimizer import (
        _collect_ep_replicated_lora_param_ids,
        _is_extra_parallel_param,
        _make_param_groups_for_subset,
        _should_build_extra_parallel_aware,
    )

    muon_kwargs = dict(muon_kwargs or {})
    expert_zero_comm = bool(muon_kwargs.pop("expert_zero_comm", False))
    head_group_size = int(muon_kwargs.pop("head_group_size", 0) or 0)
    head_split_modules = tuple(muon_kwargs.pop("head_split_modules", None) or ())
    if head_group_size:
        raise ValueError(
            "use_hyper_optimizer=True does not yet support muon_head_group_size; "
            "set it to 0 instead of silently changing the orthogonalization layout."
        )
    if head_split_modules:
        raise ValueError(
            "use_hyper_optimizer=True does not yet support muon_head_split_modules; "
            "remove it instead of silently ignoring the selection."
        )

    adjust_lr_fn = muon_kwargs.get("adjust_lr_fn", "match_rms_adamw")
    if adjust_lr_fn != "match_rms_adamw":
        raise ValueError("use_hyper_optimizer=True currently supports only muon_adjust_lr_fn='match_rms_adamw'.")
    ns_implementation = muon_kwargs.get("ns_implementation", "gram_quack")
    if ns_implementation not in ("std", "gram", "gram_quack"):
        raise ValueError(f"Unknown Hyper Muon ns_implementation={ns_implementation!r}")
    if ns_implementation != "std" and get_device_type() != "cuda":
        raise ValueError(
            "Non-CUDA Hyper Muon requires muon_ns_implementation='std'; Gram/Quack callbacks are enabled only on CUDA."
        )
    muon_lr = muon_kwargs.get("lr")
    if muon_lr is None:
        muon_lr = lr

    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(
        model,
        no_decay_modules=no_decay_modules,
        no_decay_params=no_decay_params,
    )
    if not muon_params:
        raise ValueError("HyperParallel Muon was selected but the model has no eligible 2D/3D parameters.")

    muon_param_groups = [{"params": muon_params}]
    adamw_param_groups = _make_param_groups_for_subset(
        model, adamw_params, weight_decay, no_decay_modules, no_decay_params
    )
    ns_steps = int(muon_kwargs.get("ns_steps", 5))
    ns_epsilon = float(muon_kwargs.get("eps", 1e-7))
    ns_coefficients = tuple(float(value) for value in muon_kwargs.get("ns_coefficients", (3.4445, -4.775, 2.0315)))
    if len(ns_coefficients) != 3:
        raise ValueError("HyperParallel Muon requires exactly three muon_ns_coefficients values.")
    ns_callback_kwargs = {}
    if ns_implementation != "std":
        ns_callback_kwargs["muon_zeropower_fn"] = partial(
            _run_hyper_muon_newton_schulz,
            ns_coefficients=ns_coefficients,
            eps=ns_epsilon,
            ns_implementation=ns_implementation,
            gram_ns_reset_iterations=tuple(muon_kwargs.get("gram_ns_reset_iterations", (2,))),
        )
    try:
        from hyper_parallel.core.optimizer import get_hyper_optimizer
    except ImportError as exc:
        raise RuntimeError(
            "use_hyper_optimizer=True requires the optional 'hyper_parallel' package. "
            "Install HyperParallel in the active training environment first."
        ) from exc
    optimizer = get_hyper_optimizer(
        model=model,
        muon_params=muon_param_groups,
        adamw_params=adamw_param_groups,
        muon_kwargs={
            "muon_lr": float(muon_lr),
            "muon_weight_decay": float(muon_kwargs.get("weight_decay", 0.0)),
            "muon_momentum": muon_kwargs.get("momentum", 0.95),
            "muon_nesterov": muon_kwargs.get("nesterov", True),
            "muon_ns_steps": ns_steps,
            "muon_ns_variant": "custom",
            "muon_ns_coefficients": [ns_coefficients] * ns_steps,
            "muon_ns_epsilon": ns_epsilon,
            **ns_callback_kwargs,
        },
        adamw_kwargs={
            "adamw_lr": float(lr),
            "adamw_weight_decay": float(weight_decay),
            "adamw_betas": betas,
            "adamw_eps": eps,
            "fused": fused,
        },
    )
    _guard_hyper_muon_callback_resume(
        optimizer.optimizers_dict["muon"], ns_callback_kwargs.get("muon_zeropower_fn"), ns_steps
    )
    if _should_build_extra_parallel_aware(model):
        parallel_state = get_parallel_state()
        extra_names = list(parallel_state.extra_parallel_names)
        model._extra_parallel_param_groups = {name: [] for name in extra_names}
        model._extra_parallel_param_groups["non_extra_parallel"] = []
        for param in (*muon_params, *adamw_params):
            para = _is_extra_parallel_param(param, extra_names)
            key = para if para is not None else "non_extra_parallel"
            model._extra_parallel_param_groups[key].append(param)
        model._ep_replicated_lora_param_ids = _collect_ep_replicated_lora_param_ids(model)

    logger.info_rank0(
        "HyperParallel Muon optimizer: %d Muon params, %d AdamW params; first Muon=%s, first AdamW=%s; "
        "expert_zero_comm=%s.",
        len(muon_params),
        len(adamw_params),
        muon_names[:3],
        adamw_names[:3],
        expert_zero_comm,
    )
    return optimizer


def _shard_dims(p: Tensor) -> List[int]:
    """Return the list of tensor dims along which ``p`` is sharded."""
    return [placement.dim for placement in p.placements if _is_shard_placement(placement)]


def _classify_param(p: Tensor) -> str:
    """Return one of ``_KIND_*`` describing how Muon should treat ``p``."""
    if not is_dtensor(p):
        return _KIND_LOCAL

    shard_dims = _shard_dims(p)
    if not shard_dims:
        return _KIND_LOCAL

    if p.ndim == 2:
        return _KIND_FSDP_GATHER_2D

    if p.ndim == 3:
        if all(d == 0 for d in shard_dims):
            return _KIND_MOE_LOCAL_3D
        return _KIND_MOE_GATHER_3D

    raise ValueError(
        f"DistributedMuon got an unexpected param rank {p.ndim} "
        f"(shape={tuple(p.shape)}). Only 2D and 3D params are supported."
    )


@lru_cache(maxsize=None)
def _shard_submesh(mesh: Any, placements: Tuple[Any, ...]) -> Optional[Any]:
    """Mesh-topology half of :func:`_fsdp_all2all_submesh`, keyed by layout.

    Resolving the submesh takes ~90us and depends only on the mesh and the
    placement tuple, both fixed for the run, so the answer is cached instead of
    recomputed for every Muon parameter on every step.
    """
    shard_dims = [i for i, placement in enumerate(placements) if _is_shard_placement(placement)]
    if len(shard_dims) != 1:
        return None
    mesh_dim = shard_dims[0]
    if placements[mesh_dim].dim != 0:
        return None
    # Only replicas may sit on the remaining dims. A Partial() there carries a
    # pending reduction, which skipping the collective would silently drop.
    if any(i != mesh_dim and not _is_replicate_placement(pl) for i, pl in enumerate(placements)):
        return None

    if mesh.size(mesh_dim) <= 1:
        return None
    if mesh.ndim == 1:
        return mesh

    dim_names = mesh.mesh_dim_names
    if not dim_names:
        return None
    # Slice the parameter's own mesh, not the root: the sharded dim is a
    # first-class name here even when it is a flattened dim upstream, which
    # avoids both APIs torch deprecates for the root-slicing path.
    return mesh[dim_names[mesh_dim]]


def _fsdp_all2all_submesh(p: Tensor) -> Optional[Any]:
    """Return the 1D mesh the owner-based all-to-all path should run on.

    ``p`` qualifies when it carries exactly one ``Shard(0)`` placement -- the
    FSDP2 row sharding. Under HSDP the parameter mesh additionally has one or
    more ``Replicate()`` dims (e.g. ``(dp_replicate, dp_shard_sp)``). Those
    dims hold identical values on every replica once the backward pass has
    all-reduced the gradients, so they need no collective here: slice the
    sharded dim out of the parameter's mesh and let each replica reassemble the
    same rows on its own. That is the work split ``full_tensor()`` already
    performed, minus the redundant gather.

    Returns ``None`` for layouts the path cannot express (``Shard(d>0)``,
    several shard dims, a non-``Replicate()`` dim such as ``Partial()``,
    unnamed multi-dim meshes), which then fall back to the generic
    ``full_tensor()`` path. Empty tail-rank shards stay eligible, so a first
    dimension smaller than the mesh is fine.
    """
    return _shard_submesh(p.device_mesh, tuple(p.placements))


class DistributedMuon(NativeDistributedMuon):
    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coefficients = tuple(group["ns_coefficients"])
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            adjust_lr_fn = group["adjust_lr_fn"]
            ns_implementation = str(group.get("ns_implementation", "gram_quack"))
            gram_ns_reset_iterations = tuple(group.get("gram_ns_reset_iterations", (2,)))
            head_blocks = int(group.get("head_blocks", 1))
            # Scope settings travel inside ns_kwargs so the all-to-all owner path
            # reads them from one dict.
            ns_kwargs = {
                "ns_coefficients": ns_coefficients,
                "ns_steps": ns_steps,
                "eps": eps,
                "ns_implementation": ns_implementation,
                "gram_ns_reset_iterations": gram_ns_reset_iterations,
                "head_blocks": head_blocks,
            }

            # Keep at most one owner-assignment chunk live per mesh/dtype. This
            # bounds temporary updates and orthogonalized shards by world size
            # instead of retaining the entire optimizer group.
            a2a_buckets: Dict[Tuple[Any, torch.dtype], List[Tuple[Tensor, Tensor]]] = {}

            for p in group["params"]:
                if p.grad is None:
                    continue
                if torch.is_complex(p):
                    raise RuntimeError("DistributedMuon does not support complex parameters")
                if p.grad.is_sparse:
                    raise RuntimeError("DistributedMuon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                buf: Tensor = state["momentum_buffer"]

                grad = p.grad
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                kind = _classify_param(p)
                owner_update = _owner_path_update(update, p) if kind == _KIND_FSDP_GATHER_2D else None
                submesh = _fsdp_all2all_submesh(owner_update) if owner_update is not None else None
                if submesh is not None:
                    key = _fsdp_all2all_bucket_key(owner_update, submesh)
                    entries = a2a_buckets.setdefault(key, [])
                    entries.append((p, owner_update))
                    if len(entries) == submesh.size(0):
                        self._flush_fsdp_all2all_chunk(
                            entries,
                            submesh,
                            ns_kwargs,
                            lr=lr,
                            weight_decay=weight_decay,
                            adjust_lr_fn=adjust_lr_fn,
                        )
                else:
                    ortho = self._compute_ortho(update, kind, ref=p, **ns_kwargs)
                    self._apply_ortho(
                        p,
                        ortho,
                        is_local_shard=False,
                        lr=lr,
                        weight_decay=weight_decay,
                        adjust_lr_fn=adjust_lr_fn,
                        head_blocks=head_blocks,
                    )

            for (mesh, _local_dtype), entries in a2a_buckets.items():
                self._flush_fsdp_all2all_chunk(
                    entries,
                    mesh,
                    ns_kwargs,
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn=adjust_lr_fn,
                )

        return loss

    @staticmethod
    def _compute_ortho(
        update: Tensor,
        kind: str,
        ns_coefficients: Tuple[float, float, float],
        ns_steps: int,
        eps: float,
        ref: Optional[Tensor] = None,
        ns_implementation: str = "gram_quack",
        gram_ns_reset_iterations: Sequence[int] = (2,),
        head_blocks: int = 1,
    ) -> Tensor:
        """Adapt layouts while retaining native NS and head-block computation."""
        ns_kwargs = dict(
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
            ns_implementation=ns_implementation,
            gram_ns_reset_iterations=gram_ns_reset_iterations,
            head_blocks=head_blocks,
        )
        if kind in (_KIND_FSDP_GATHER_2D, _KIND_MOE_GATHER_3D):
            ortho = NativeDistributedMuon._compute_ortho(full_gradient(update, ref), _KIND_LOCAL, **ns_kwargs)
            return shard_full(ortho, ref) if ref is not None and uses_local_updates(ref) else ortho
        if kind == _KIND_MOE_LOCAL_3D and ref is not None and uses_local_updates(ref):
            kind = _KIND_LOCAL
        return NativeDistributedMuon._compute_ortho(update, kind, **ns_kwargs)
