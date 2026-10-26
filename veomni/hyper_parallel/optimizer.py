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

"""Hyper local-shard execution and DTensor semantics."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..distributed.parallel_state import get_parallel_state
from ..optim.muon import infer_head_block_counts, split_muon_adamw_params
from ..optim.optimizer import (
    AnyPrecisionAdamW,
    MultiOptimizer,
    _collect_muon_kwargs,
    _make_param_groups_for_subset,
    _muon_param_groups,
    _should_build_extra_parallel_aware,
    filter_empty_param_groups,
)
from ..optim.optimizer import build_optimizer as native_build_optimizer
from ..utils import logging
from .module_utils import is_dtensor, uses_local_updates
from .muon import DistributedMuon, build_hyper_muon_with_adamw


if TYPE_CHECKING:
    from ..arguments import OptimizerConfig

logger = logging.get_logger(__name__)


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

    def __iter__(self):
        """Expose leaf optimizers to Torch DCP without changing their state format."""
        optimizer = self._scheduler_optimizer
        return iter(getattr(optimizer, "chained_optimizers", (optimizer,)))

    def _sparse_state_dict(self):
        """Reuse VeOmni's sparse multi-optimizer checkpoint protocol."""
        from contextlib import nullcontext

        from ..optim.optimizer import MultiOptimizer

        optimizer = self._scheduler_optimizer
        for leaf in optimizer.chained_optimizers:
            synchronize_state = getattr(leaf, "_broadcast_state_fused_for_ckpt", None)
            if synchronize_state is not None:
                synchronize_state()
        checkpoint_view = MultiOptimizer(optimizer.model, optimizer.optimizers_dict, optimizer.optimizers_keys)
        parameter_view = getattr(optimizer, "_optimizer_parameter_view", nullcontext)
        with parameter_view():
            return checkpoint_view._sparse_state_dict()


def wrap_optimizer(optimizer):
    """Adapt execution without replacing the optimizer's or its children's methods."""
    return optimizer if isinstance(optimizer, LocalShardOptimizer) else LocalShardOptimizer(optimizer)


def _collect_ep_replicated_lora_param_ids(model: "nn.Module") -> set[int]:
    """Return ``id(p)`` of every LoRA Parameter that is *replicated* across the EP mesh.

    ``LoraSharedExperts`` (Mode 2) wraps a MoE experts module so that its
    extra LoRA tensors live alongside the EP-sharded base experts inside the
    same module. ``build_parallelize_model`` therefore pulls the LoRA
    parameters into the inner ``fully_shard(..., mesh=ep_fsdp, Shard(1))``
    wrap together with the base ``gate_up_proj`` / ``down_proj``. From the
    optimizer's point of view they look like "ep params" (their DTensor
    mesh has ``ep_fsdp`` as a dim) and the ExtraParallel-aware grad clipper
    would all-reduce their squared norms across the ``ep`` group — which
    is correct for the EP-sliced base experts but **double-counts** the
    shared LoRA whose values are identical on every EP rank (they are
    replicated by construction; ``LoraSharedExperts._ensure_ep_grad_sync_hooks``
    keeps their gradients in lock-step across the EP group).

    Keep them in their existing optimizer bucket. The clipper uses these
    identities to reduce their norms only across ``ep_fsdp``, skipping
    the replicated EP dimension without changing optimizer ownership.

    Detection by class name (``LoraSharedExperts``) so this stays free of
    a circular import (``veomni.lora.moe_layers`` already imports from
    ``veomni.distributed.parallel_state``).

    Call-order requirement
    ----------------------
    Must run after FSDP wrapping has converted params to DTensors, hence
    after :func:`build_parallelize_model`. Otherwise the snapshotted
    ``id(p)`` values point at the pre-wrap ``nn.Parameter`` objects, which
    get replaced by new DTensor-backed parameters when ``fully_shard``
    rewraps the wrapper module -- the snapshot then goes stale (Python may
    reuse those addresses for unrelated objects, causing false positives
    in :func:`extra_parallel_fsdp2_clip_grad_norm`).

    ``id(p)`` stability after this point is guaranteed because
    ``build_optimizer`` populates ``model._extra_parallel_param_groups``
    from the same post-wrap Parameter objects in the same call, and those
    lists hold long-lived references that prevent garbage collection.
    """
    out: set[int] = set()
    for mod in model.modules():
        # Walk the MRO so we still match after FSDP2's ``fully_shard`` rebases the
        # class to ``FSDPLoraSharedExperts(LoraSharedExperts, FSDPModule)``
        # (see torch's ``_fully_shard``); a bare ``__class__.__name__`` check
        # would silently skip every wrapper post-FSDP.
        if any(b.__name__ == "LoraSharedExperts" for b in type(mod).__mro__):
            for n, p in mod.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                # Catches wrong call-order misuse: if any matched LoRA param
                # is still a plain ``nn.Parameter`` (not a DTensor) at this
                # point, FSDP2 hasn't wrapped yet -- ``build_optimizer`` was
                # invoked before ``build_parallelize_model``. The snapshot
                # we are about to take would be stale immediately.
                assert is_dtensor(p), (
                    f"_collect_ep_replicated_lora_param_ids: LoRA param {n!r} on a "
                    f"LoraSharedExperts wrapper is not a DTensor -- call "
                    f"build_parallelize_model before build_optimizer so FSDP2 has "
                    f"converted these params to DTensors before we snapshot id(p)."
                )
                out.add(id(p))
    return out


def _is_extra_parallel_param(p: torch.nn.Parameter, extra_parallel_names: Sequence[str]) -> Optional[str]:
    """Return the ExtraParallel name for a DTensor param, or ``None``."""
    if not is_dtensor(p):
        return None
    mesh = getattr(p, "device_mesh", None)
    names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
    for para in extra_parallel_names:
        if f"{para}_fsdp" in names:
            return para
    return None


def _build_muon_with_adamw(
    model: "nn.Module",
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    fused: bool,
    no_decay_modules: Optional[List[str]],
    no_decay_params: Optional[List[str]],
    muon_kwargs: Optional[Dict[str, Any]],
) -> "MultiOptimizer":
    """Build a Muon (2D/3D hidden weights) + AdamW (everything else) MultiOptimizer.

    ExtraParallel params are split into separate sub-optimizers so DCP state
    dict keys and grad-clipping metadata stay keyed by mesh.
    """
    muon_kwargs = dict(muon_kwargs or {})
    # Summary-only keys; not DistributedMuon ctor args.
    expert_zero_comm = bool(muon_kwargs.pop("expert_zero_comm", False))
    adamw_lr = float(lr)
    head_group_size = int(muon_kwargs.pop("head_group_size", 0) or 0)
    head_split_modules = tuple(muon_kwargs.pop("head_split_modules", None) or ())
    if head_group_size < 0:
        raise ValueError(f"muon_head_group_size must be >= 0 (0 disables head splitting), got {head_group_size}")
    if head_group_size >= 1 and not head_split_modules:
        raise ValueError(
            f"muon_head_group_size={head_group_size} requires muon_head_split_modules: there is no "
            "default list, because which attention projections benefit from head splitting depends "
            "on the architecture. List the modules to split as leaf names or dotted path suffixes, "
            "e.g. ['self_attn.q_b_proj'] for DeepSeek V4/V3 MLA up-projections or "
            "['q_proj', 'k_proj', 'v_proj'] for GQA attention."
        )
    # The single owner of the Muon-vs-AdamW LR policy: an unset Muon LR either
    # inherits the AdamW LR (match_rms_adamw) or takes the Moonlight-style 25x.
    muon_lr_explicit = muon_kwargs.get("lr") is not None
    if not muon_lr_explicit:
        adjust_lr_fn = muon_kwargs.get("adjust_lr_fn", "match_rms_adamw")
        muon_kwargs["lr"] = adamw_lr if adjust_lr_fn == "match_rms_adamw" else 25.0 * adamw_lr
    muon_params, adamw_params, muon_names, adamw_names = split_muon_adamw_params(
        model,
        no_decay_modules=no_decay_modules,
        no_decay_params=no_decay_params,
    )
    if not muon_params:
        raise ValueError(
            "Muon optimizer was selected but the model has no eligible 2D/3D parameters. "
            "Falling back to AdamW would be silent so we raise instead."
        )

    head_blocks_by_param: Dict[int, int] = {}
    head_split_names: List[str] = []
    if head_group_size >= 1:
        blocks_by_fqn = infer_head_block_counts(model, head_group_size, head_split_modules)
        muon_name_set = set(muon_names)
        param_by_name = dict(model.named_parameters())
        for fqn, blocks in sorted(blocks_by_fqn.items()):
            if fqn not in muon_name_set:  # routed to AdamW, nothing to scope
                continue
            head_blocks_by_param[id(param_by_name[fqn])] = blocks
            head_split_names.append(fqn)
        if not head_split_names:
            logger.warning_rank0(
                f"[Muon] muon_head_group_size={head_group_size} matched no attention projection; "
                f"every param stays on full-matrix Muon. Check muon_head_split_modules="
                f"{list(head_split_modules)} against this architecture's attention module names."
            )

    extra_parallel_aware = _should_build_extra_parallel_aware(model)
    parallel_state = get_parallel_state() if extra_parallel_aware else None
    extra_parallel_names = list(parallel_state.extra_parallel_names) if extra_parallel_aware else []

    def _split_by_ep(
        params: List[torch.nn.Parameter],
    ) -> Tuple[Dict[str, List[torch.nn.Parameter]], List[torch.nn.Parameter]]:
        per_para: Dict[str, List[torch.nn.Parameter]] = {p: [] for p in extra_parallel_names}
        non_para: List[torch.nn.Parameter] = []
        if not extra_parallel_aware:
            return per_para, list(params)
        for p in params:
            para = _is_extra_parallel_param(p, extra_parallel_names)
            if para is not None:
                per_para[para].append(p)
            else:
                non_para.append(p)
        return per_para, non_para

    muon_per_para, muon_non_para = _split_by_ep(muon_params)
    adamw_per_para, adamw_non_para = _split_by_ep(adamw_params)

    muon_lr = float(muon_kwargs.get("lr", 2e-2))
    adjust_lr_fn = muon_kwargs.get("adjust_lr_fn", "match_rms_adamw")
    ns_implementation = muon_kwargs.get("ns_implementation", "gram_quack")
    ns_steps = int(muon_kwargs.get("ns_steps", 5))
    lr_source = (
        "explicit"
        if muon_lr_explicit
        else (
            "inherit optimizer.lr (match_rms_adamw)"
            if adjust_lr_fn == "match_rms_adamw"
            else "25x optimizer.lr (original)"
        )
    )
    logger.info_rank0(
        f"Muon optimizer: {len(muon_params)} param(s) on Muon, {len(adamw_params)} on AdamW. "
        f"First few Muon params: {muon_names[:5]}; first few AdamW params: {adamw_names[:5]}."
    )
    logger.info_rank0(
        "[Muon] "
        f"ns_implementation={ns_implementation}, ns_steps={ns_steps}, "
        f"adjust_lr_fn={adjust_lr_fn}, "
        f"muon_lr={muon_lr:g} ({lr_source}), adamw_lr={adamw_lr:g}, "
        f"muon_weight_decay={float(muon_kwargs.get('weight_decay', 0.0)):g}, "
        f"adamw_weight_decay={float(weight_decay):g}, "
        f"momentum={float(muon_kwargs.get('momentum', 0.95)):g}, "
        f"nesterov={bool(muon_kwargs.get('nesterov', True))}, "
        f"expert_zero_comm={expert_zero_comm} "
        "(applied at FSDP parallelize; see [muon_expert_zero_comm] logs)."
    )
    if head_split_names:
        blocks_hist: Dict[int, int] = {}
        for blocks in head_blocks_by_param.values():
            blocks_hist[blocks] = blocks_hist.get(blocks, 0) + 1
        logger.info_rank0(
            f"[Muon] head split: {len(head_split_names)} param(s) orthogonalized in blocks of "
            f"{head_group_size} head(s); {{blocks: params}}={dict(sorted(blocks_hist.items()))}; "
            f"first few: {head_split_names[:3]}."
        )
    if extra_parallel_aware:
        for para in extra_parallel_names:
            logger.info_rank0(
                f"Muon split for {para}: muon_{para}={len(muon_per_para[para])}, "
                f"adamw_{para}={len(adamw_per_para[para])}, "
                f"muon_non_extra_parallel={len(muon_non_para)}, adamw_non_extra_parallel={len(adamw_non_para)}"
            )

    def _make_muon(params: List[torch.nn.Parameter]) -> DistributedMuon:
        return DistributedMuon(
            _muon_param_groups(params, head_blocks_by_param),
            lr=muon_kwargs.get("lr", 2e-2),
            weight_decay=muon_kwargs.get("weight_decay", 0.0),
            momentum=muon_kwargs.get("momentum", 0.95),
            nesterov=muon_kwargs.get("nesterov", True),
            ns_coefficients=tuple(muon_kwargs.get("ns_coefficients", (3.4445, -4.7750, 2.0315))),
            eps=muon_kwargs.get("eps", 1e-7),
            ns_steps=muon_kwargs.get("ns_steps", 5),
            adjust_lr_fn=muon_kwargs.get("adjust_lr_fn", "match_rms_adamw"),
            ns_implementation=muon_kwargs.get("ns_implementation", "gram_quack"),
            gram_ns_reset_iterations=muon_kwargs.get("gram_ns_reset_iterations", (2,)),
        )

    def _make_adamw(params: List[torch.nn.Parameter]) -> AdamW:
        groups = _make_param_groups_for_subset(model, params, weight_decay, no_decay_modules, no_decay_params)
        foreach = not fused
        return AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=fused, foreach=foreach)

    optimizer_dict: Dict[str, Optimizer] = {}
    if extra_parallel_aware:
        for para in extra_parallel_names:
            if muon_per_para[para]:
                optimizer_dict[f"muon_{para}"] = _make_muon(muon_per_para[para])
        if muon_non_para:
            optimizer_dict["muon_non_extra_parallel"] = _make_muon(muon_non_para)
    else:
        optimizer_dict["muon"] = _make_muon(muon_non_para)

    if extra_parallel_aware:
        for para in extra_parallel_names:
            if adamw_per_para[para]:
                optimizer_dict[para] = _make_adamw(adamw_per_para[para])
        if adamw_non_para:
            optimizer_dict["non_extra_parallel"] = _make_adamw(adamw_non_para)
    elif adamw_params:
        optimizer_dict["adamw"] = _make_adamw(adamw_non_para)

    # Grad clipping groups by mesh, not by optimizer type.
    if extra_parallel_aware:
        model._extra_parallel_param_groups = {
            para: list(muon_per_para[para]) + list(adamw_per_para[para]) for para in extra_parallel_names
        }
        model._extra_parallel_param_groups["non_extra_parallel"] = list(muon_non_para) + list(adamw_non_para)
        model._ep_replicated_lora_param_ids = _collect_ep_replicated_lora_param_ids(model)

    return MultiOptimizer(model, optimizer_dict, key_names=list(optimizer_dict.keys()))


def build_extra_parallel_fsdp2_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[List[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
):
    """
    Build a MultiOptimizer instance when model is parallelized with ExtraParallel+FSDP2

    If param_groups provided, it can be a list of dicts with arbitrary parameter groups:
    - Example: [{"params": params1, "lr": lr1},
                {"params": params2, "lr": lr2},
                {"params": params3, "lr": lr3}]
    - Each group's params are automatically split into ExtraParallel1, ExtraParallel2, ... and non-ExtraParallel based on DTensor mesh
    - Custom learning rates and other optimizer settings are preserved per group
    """
    parallel_state = get_parallel_state()

    # Collect all ExtraParallel and non-ExtraParallel parameters across all groups
    extra_parallel_groups = {
        para: []  # List[Dict[str, Any]]
        for para in parallel_state.extra_parallel_names
    }
    non_extra_parallel_groups: List[Dict[str, Any]] = []

    # Process custom param_groups if provided
    if param_groups is not None:
        # Validate param_groups structure
        assert isinstance(param_groups, list), "param_groups must be a list"

        # Process each parameter group
        for group_config in param_groups:
            assert "params" in group_config, (
                f"Each group in param_groups must contain 'params' key, got: {group_config}"
            )

            # Extract group-specific settings
            group_lr = group_config.get("lr", lr)
            group_params = group_config["params"]

            # Split this group's params into ExtraParallel and non-ExtraParallel
            group_extra_parallel_params = {
                para: []  # List[torch.nn.Parameter]
                for para in parallel_state.extra_parallel_names
            }
            group_non_extra_parallel_params: List[torch.nn.Parameter] = []

            for p in group_params:
                if not p.requires_grad:
                    continue

                # Check if this parameter is part of ExtraParallel
                is_extra_parallel_params = False
                if is_dtensor(p):
                    mesh = getattr(p, "device_mesh", None)
                    names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                    for para in parallel_state.extra_parallel_names:
                        if f"{para}_fsdp" in names:
                            group_extra_parallel_params[para].append(p)
                            is_extra_parallel_params = True
                            break

                if not is_extra_parallel_params:
                    group_non_extra_parallel_params.append(p)

            # Create subgroups with weight decay handling
            for para in parallel_state.extra_parallel_names:
                if group_extra_parallel_params[para]:
                    group_para_subgroups = _make_param_groups_for_subset(
                        model, group_extra_parallel_params[para], weight_decay, no_decay_modules, no_decay_params
                    )
                    for subgroup in group_para_subgroups:
                        subgroup["lr"] = group_lr
                        # Preserve other custom settings from original group
                        for key, value in group_config.items():
                            if key not in ["params", "lr", "weight_decay"]:
                                subgroup[key] = value
                    extra_parallel_groups[para].extend(group_para_subgroups)

            if group_non_extra_parallel_params:
                group_non_extra_parallel_subgroups = _make_param_groups_for_subset(
                    model, group_non_extra_parallel_params, weight_decay, no_decay_modules, no_decay_params
                )
                for subgroup in group_non_extra_parallel_subgroups:
                    subgroup["lr"] = group_lr
                    # Preserve other custom settings from original group
                    for key, value in group_config.items():
                        if key not in ["params", "lr", "weight_decay"]:
                            subgroup[key] = value
                non_extra_parallel_groups.extend(group_non_extra_parallel_subgroups)
    else:
        # Default case (param_groups is None): all model parameters with uniform settings(lr)
        extra_parallel_params = {
            para: []  # List[torch.nn.Parameter]
            for para in parallel_state.extra_parallel_names
        }
        non_extra_parallel_params: List[torch.nn.Parameter] = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # Check if this parameter is part of ExtraParallel
            is_extra_parallel_params = False
            if is_dtensor(p):
                mesh = getattr(p, "device_mesh", None)
                names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                logger.debug_rank0(f"param {name} has device_mesh {mesh} with mesh dim names {names}")

                for para in parallel_state.extra_parallel_names:
                    if f"{para}_fsdp" in names:
                        logger.debug_rank0(f"Adding {name} to {para}_params in extra_parallel+fsdp2 optimizer")
                        extra_parallel_params[para].append(p)
                        is_extra_parallel_params = True
                        break

            if not is_extra_parallel_params:
                logger.debug_rank0(f"Adding {name} to non_extra_parallel_params in extra_parallel+fsdp2 optimizer")
                non_extra_parallel_params.append(p)

        # Build param groups with weight decay handling
        extra_parallel_groups = {
            para: _make_param_groups_for_subset(
                model, extra_parallel_params[para], weight_decay, no_decay_modules, no_decay_params
            )
            for para in parallel_state.extra_parallel_names
        }
        non_extra_parallel_groups = _make_param_groups_for_subset(
            model, non_extra_parallel_params, weight_decay, no_decay_modules, no_decay_params
        )

    def _build(groups: Sequence[Dict[str, Any]]) -> Optimizer:
        if optimizer_type == "adamw":
            nonlocal fused
            foreach = not fused
            _fused = fused
            return AdamW(groups, lr, betas, eps, weight_decay, fused=_fused, foreach=foreach)
        elif optimizer_type == "anyprecision_adamw":
            return AnyPrecisionAdamW(groups, lr, betas, eps, weight_decay)
        else:
            raise ValueError("Only adamw and anyprecision_adamw are supported as optimizers.")

    optimizer_dict: Dict[str, Optimizer] = {}
    for para in parallel_state.extra_parallel_names:
        if extra_parallel_groups[para]:
            optimizer_dict[para] = _build(extra_parallel_groups[para])
    if non_extra_parallel_groups:
        optimizer_dict["non_extra_parallel"] = _build(non_extra_parallel_groups)

    # cache for ExtraParallel-aware grad clipping helpers
    model._extra_parallel_param_groups = {
        para: [p for g in extra_parallel_groups[para] for p in g.get("params", [])]
        if extra_parallel_groups[para]
        else []
        for para in parallel_state.extra_parallel_names
    }
    model._extra_parallel_param_groups["non_extra_parallel"] = (
        [p for g in non_extra_parallel_groups for p in g.get("params", [])] if non_extra_parallel_groups else []
    )
    # Sidecar set for the grad clipper: ids of ``LoraSharedExperts`` LoRA params
    # that landed in some ``extra_parallel_params[para]`` bucket above. They live
    # on the inner ``ep_fsdp`` FSDP mesh (so they cannot be moved to the
    # ``non_extra_parallel`` optimizer — its DTensors are on a different mesh and
    # ``foreach_*`` ops reject mixed-mesh tensor lists), but unlike the EP-sliced
    # base experts and Mode-1 per-expert LoRA they are *replicated* across the
    # EP group (kept in lock-step by ``LoraSharedExperts._ensure_ep_grad_sync_hooks``).
    # ``extra_parallel_fsdp2_clip_grad_norm`` uses this set to skip the EP all-reduce
    # for these params; otherwise their squared norms are summed once per EP rank
    # and the global norm comes out larger than the EP=1 baseline, breaking
    # cross-EP grad-norm parity (and therefore any optimizer step that depends
    # on the clipped grad).
    model._ep_replicated_lora_param_ids = _collect_ep_replicated_lora_param_ids(model)

    key_names = list(optimizer_dict.keys())

    # Build MultiOptimizer and attach a pre-step hook to sanitize DTensor states
    multi_opt = MultiOptimizer(model, optimizer_dict, key_names=key_names)

    return multi_opt


def _construct_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    muon_kwargs: Optional[Dict[str, Any]] = None,
    optimizer_config: Optional["OptimizerConfig"] = None,
    *,
    use_hyper_optimizer=False,
) -> "torch.optim.Optimizer":
    """Build the optimizer.

    ``optimizer_config`` lets callers hand over the whole ``OptimizerConfig`` so
    that optimizer-specific knobs (currently Muon's ``muon_*`` fields) are read
    here rather than unpacked by every trainer. It does *not* supply ``lr``,
    ``weight_decay`` or the ``no_decay_*`` lists — those stay explicit arguments,
    because callers such as the VLM trainer drive the AdamW side from their own
    param groups. ``muon_kwargs`` overrides the config key by key, for callers
    that build Muon without an ``OptimizerConfig`` at all.
    """
    use_hyper_optimizer = use_hyper_optimizer or bool(
        optimizer_config is not None and optimizer_config.use_hyper_optimizer
    )
    if use_hyper_optimizer and optimizer_type != "muon":
        raise ValueError("use_hyper_optimizer=True requires optimizer_type='muon'.")
    native_kwargs = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        fused=fused,
        optimizer_type=optimizer_type,
        param_groups=param_groups,
        no_decay_modules=no_decay_modules,
        no_decay_params=no_decay_params,
        muon_kwargs=muon_kwargs,
        optimizer_config=optimizer_config,
    )
    has_hyper_params = any(uses_local_updates(param) for param in model.parameters())
    if not use_hyper_optimizer and (optimizer_type != "muon" or not has_hyper_params):
        if optimizer_type != "muon" and has_hyper_params and _should_build_extra_parallel_aware(model):
            if param_groups is not None:
                param_groups = filter_empty_param_groups(param_groups)
                if not param_groups:
                    raise ValueError("All optimizer param groups are empty; no trainable parameters to optimize.")
            return build_extra_parallel_fsdp2_optimizer(
                model,
                lr,
                betas,
                eps,
                weight_decay,
                fused,
                optimizer_type,
                param_groups,
                no_decay_modules,
                no_decay_params,
            )
        return native_build_optimizer(model, **native_kwargs)
    muon_factory = build_hyper_muon_with_adamw if use_hyper_optimizer else _build_muon_with_adamw
    if optimizer_config is not None:
        muon_kwargs = {**_collect_muon_kwargs(optimizer_config), **(muon_kwargs or {})}
    if param_groups is not None:
        logger.warning_rank0(
            "build_optimizer(optimizer_type='muon') ignores the provided "
            "param_groups argument; all parameters are re-split by "
            "split_muon_adamw_params. To control vit vs. backbone learning "
            "rates with Muon, file a follow-up or use AdamW for now."
        )
    return muon_factory(
        model,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        fused=fused,
        no_decay_modules=no_decay_modules,
        no_decay_params=no_decay_params,
        muon_kwargs=muon_kwargs,
    )


def build_optimizer(*args, **kwargs):
    """Build the selected optimizer and adapt only its local-shard execution."""
    return wrap_optimizer(_construct_optimizer(*args, **kwargs))


def build_lr_scheduler(optimizer, **kwargs):
    """Bind scheduler hooks to the actual leaf optimizers."""
    from ..optim.lr_scheduler import build_lr_scheduler as native_build

    owner = getattr(optimizer, "_scheduler_optimizer", optimizer)
    if hasattr(owner, "optimizers_dict") and not hasattr(owner, "key_names"):
        owner = MultiOptimizer(owner.model, owner.optimizers_dict, owner.optimizers_keys)
    return native_build(owner, **kwargs)
