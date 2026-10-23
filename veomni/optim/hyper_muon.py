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
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.device import get_device_type
from .muon import split_muon_adamw_params


logger = logging.get_logger(__name__)


def _run_hyper_muon_newton_schulz(grad: torch.Tensor, steps: int, **kwargs) -> torch.Tensor:
    """Adapt Hyper's owner-batched callback to VeOmni's NS dispatcher."""
    from .muon import run_newton_schulz

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
    from .optimizer import _is_extra_parallel_param, _make_param_groups_for_subset, _should_build_extra_parallel_aware

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
