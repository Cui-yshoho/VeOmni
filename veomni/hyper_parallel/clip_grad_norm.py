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
import itertools
import math
from typing import List

import torch
import torch.distributed as dist
from hyper_parallel import SkipDTensorDispatch, hsdp_sync_stream

from ..distributed.fsdp2.clip_grad_norm import (
    _finalize_total_norm,
    _fsdp_grad_norm_reduce_groups,
    _raise_if_nonfinite,
)
from ..distributed.parallel_state import get_parallel_state
from ..utils.device import get_device_type
from ..utils.logging import get_logger
from .module_utils import is_dtensor


logger = get_logger(__name__)


def clip_omni_module_grad_norm(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Gradient-norm clipping for a single OmniModule under its own parallelism.

    A Hyper ``OmniModule`` may be wrapped as FSDP2 or FSDP2 + ExtraParallel, so
    the world-complete sum of p-th powers is reduced over the right process group
    for this module's topology, finalized into the module norm, then used to clip
    this module's params:

    * FSDP2: local shard p-th-sum, all-reduce SUM over ``fsdp_group``.
    * FSDP2 + ExtraParallel: non-ExtraParallel params over ``fsdp_group``;
      ExtraParallel params over ``{ep}_fsdp`` then ``{ep}`` (mirrors
      ``extra_parallel_fsdp2_clip_grad_norm``).

    The reduced scalar is identical across ranks, so the returned norm is
    rank-consistent.
    """
    hsdp_sync_stream()
    norm_type = float(norm_type)
    ps = get_parallel_state()
    if ps.dp_mode != "fsdp2":
        raise ValueError("Hyper clipping requires fsdp2.")

    pth_sums: list[torch.Tensor] = []
    groups_to_clip: list[list[torch.nn.Parameter]] = []

    ep_param_groups = getattr(model, "_extra_parallel_param_groups", None)
    if ep_param_groups is not None and ps.any_extra_parallel_enabled:
        return extra_parallel_fsdp2_clip_grad_norm(model, max_norm, norm_type=norm_type).item()

    params = [p for p in model.parameters() if p.grad is not None]
    if params:
        # FSDP2 DTensor grads are sharded -> SUM local p-th powers over fsdp_group.
        # Plain Tensor grads are full replicas (e.g. Omni connector under a process
        # whose ParallelState is still fsdp2) — do NOT world-SUM those.
        dt_params = [p for p in params if is_dtensor(p)]
        dense_params = [p for p in params if not is_dtensor(p)]
        if dt_params:
            reduce_groups = _fsdp_grad_norm_reduce_groups(ps)
            pth_sums.append(_fsdp2_reduce_group(dt_params, norm_type, reduce_groups))
            groups_to_clip.append(dt_params)
        if dense_params:
            pth_sums.append(_fsdp2_reduce_group(dense_params, norm_type, []))
            groups_to_clip.append(dense_params)

    if not pth_sums:
        return 0.0

    if math.isinf(norm_type):
        total_norm = torch.stack(pth_sums).amax()
    else:
        total_norm = _finalize_total_norm(torch.stack(pth_sums).sum(), norm_type)

    with SkipDTensorDispatch():
        for params in groups_to_clip:
            torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm)

    return total_norm.item()


def clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    hsdp_sync_stream()
    if hasattr(model, "_extra_parallel_param_groups"):
        return extra_parallel_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    if getattr(model, "_fsdp_cpu_offload_enabled", False):
        params = [p for p in model.parameters() if p.grad is not None]
        total = _fsdp2_reduce_group(params, norm_type, _fsdp_grad_norm_reduce_groups(get_parallel_state()))
        total_norm = _finalize_total_norm(total, norm_type)
        _raise_if_nonfinite(total_norm, norm_type, error_if_nonfinite)
        # Reduce accelerator scalars, not CPU gradients, over NCCL/HCCL.
        # Only the final scalar returns to CPU for clipping offloaded shards.
        with SkipDTensorDispatch():
            torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm.cpu(), foreach=foreach)
        return total_norm

    from hyper_parallel.core.utils import clip_grad_norm_

    return clip_grad_norm_(
        model, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach
    )


@torch.no_grad()
def extra_parallel_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    ExtraParallel-aware gradient clipping for composable FSDP2:

    - Compute local norms for non-ExtraParallel and ExtraParallel parameter groups separately.
    - For finite p: sum p-th powers across the appropriate groups, then take 1/p.
      • non-ExtraParallel: all-reduce over FSDP group.
      • ExtraParallel: all-reduce over Para-FSDP (e.g. ep_fsdp, emb_fsdp) group, then over Para (e.g. ep, emb) group.
      • persistent 2D ExtraParallel: all-reduce once over the flattened Para-FSDP × Para mesh.
    - For inf-norm: take elementwise MAX with the same reduction groups (MAX).
    - Use a single global clip coefficient for both groups.
    """
    ps = get_parallel_state()
    extra_parallel_group = {
        para: ps.extra_parallel_group(para) if ps.extra_parallel_enabled(para) else None
        for para in ps.extra_parallel_names
    }
    # For Para (e.g. ep, emb) params sharded by FSDP2 along hidden dimension
    extra_parallel_fsdp_group = {
        para: ps.extra_parallel_fsdp_device_mesh[para][f"{para}_fsdp"].get_group()
        if ps.extra_parallel_enabled(para) and ps.extra_parallel_fsdp_device_mesh[para] is not None
        else None
        for para in ps.extra_parallel_names
    }

    # ``LoraSharedExperts`` Mode-2 LoRA params live on the same inner ``ep_fsdp``
    # FSDP mesh as the EP-sharded base experts (the optimizer cannot place them
    # in the ``non_extra_parallel`` bucket without crossing meshes — see the
    # comment near ``_ep_replicated_lora_param_ids`` in
    # ``veomni/optim/optimizer.py``), but they are *replicated* across the EP
    # group rather than EP-sliced. Including them in the EP-mesh all-reduce path
    # below would sum their squared norms once per EP rank, inflating the global
    # grad norm by ~``sqrt(ep_size)`` for the shared-LoRA bucket and breaking
    # EP=1/EP=N grad-norm parity. Split them off into a separate
    # ``extra_parallel_replicated_params`` bucket whose reduction skips the
    # ``ep`` group (only the ``ep_fsdp`` reduction stays — that one is a no-op
    # at ``ep_fsdp_size=1`` but is the right shape if it grows in the future).
    ep_replicated_ids: set[int] = getattr(model, "_ep_replicated_lora_param_ids", set())
    persistent_ids: set[int] = getattr(model, "_persistent_extra_parallel_param_ids", set())
    extra_parallel_params = {
        para: [
            p
            for p in model._extra_parallel_param_groups.get(para, [])
            if p.grad is not None and id(p) not in ep_replicated_ids and id(p) not in persistent_ids
        ]
        for para in ps.extra_parallel_names
    }
    extra_parallel_replicated_params = {
        para: [
            p
            for p in model._extra_parallel_param_groups.get(para, [])
            if p.grad is not None and id(p) in ep_replicated_ids and id(p) not in persistent_ids
        ]
        for para in ps.extra_parallel_names
    }
    persistent_extra_parallel_params = {
        para: [
            p
            for p in model._extra_parallel_param_groups.get(para, [])
            if p.grad is not None and id(p) in persistent_ids
        ]
        for para in ps.extra_parallel_names
    }
    non_extra_parallel_params: List[torch.nn.Parameter] = [
        p for p in model._extra_parallel_param_groups.get("non_extra_parallel", []) if p.grad is not None
    ]

    # Compute and reduce non-ExtraParallel
    non_extra_parallel_total = _fsdp2_reduce_group(
        params=non_extra_parallel_params,
        norm_type=norm_type,
        reduce_groups=_fsdp_grad_norm_reduce_groups(ps),
    )
    logger.debug_rank0(f"non_extra_parallel total grad norm: {non_extra_parallel_total}")

    for para in ps.extra_parallel_names:
        logger.debug_rank0(
            f"{para}_params reduces groups: {extra_parallel_fsdp_group[para]=}, {extra_parallel_group[para]=}"
        )

    # Compute and reduce ExtraParallel: first across para_fsdp (e.g. ep_fsdp, emb_fsdp), then across para (e.g. ep, emb)
    extra_parallel_total = {
        para: torch.tensor(0.0, device=torch.device(get_device_type()), dtype=torch.float32)
        for para in ps.extra_parallel_names
    }
    for para in ps.extra_parallel_names:
        if len(extra_parallel_params[para]) > 0:
            para_total = _fsdp2_reduce_group(
                params=extra_parallel_params[para],
                norm_type=norm_type,
                reduce_groups=[
                    (f"{para}_fsdp", extra_parallel_fsdp_group[para]),
                    (f"{para}", extra_parallel_group[para]),
                ],
            )
            extra_parallel_total[para] = para_total
            logger.debug_rank0(f"{para} total grad norm: {para_total}")

    # Replicated-across-Para subgroup (currently only Mode-2 shared LoRA on the
    # ``ep`` mesh): reduce ONLY across para_fsdp -- skipping the ``para`` (e.g.
    # ``ep``) reduction prevents double-counting since every Para rank holds the
    # same gradient.
    extra_parallel_replicated_total = {
        para: torch.tensor(0.0, device=torch.device(get_device_type()), dtype=torch.float32)
        for para in ps.extra_parallel_names
    }
    for para in ps.extra_parallel_names:
        if len(extra_parallel_replicated_params[para]) > 0:
            para_total = _fsdp2_reduce_group(
                params=extra_parallel_replicated_params[para],
                norm_type=norm_type,
                reduce_groups=[
                    (f"{para}_fsdp", extra_parallel_fsdp_group[para]),
                ],
            )
            extra_parallel_replicated_total[para] = para_total
            logger.debug_rank0(f"{para} replicated total grad norm: {para_total}")

    # Persistent 2D ExtraParallel parameters are not managed by FSDP. Their
    # local DTensor shards are unique across both mesh axes, so reduce their
    # norm statistic exactly once over the pre-created flattened mesh.
    persistent_extra_parallel_total = {
        para: torch.tensor(0.0, device=torch.device(get_device_type()), dtype=torch.float32)
        for para in ps.extra_parallel_names
    }
    for para in ps.extra_parallel_names:
        if len(persistent_extra_parallel_params[para]) > 0:
            para_total = _fsdp2_reduce_group(
                params=persistent_extra_parallel_params[para],
                norm_type=norm_type,
                reduce_groups=[(f"{para}_flat", ps.extra_parallel_flat_group(para))],
            )
            persistent_extra_parallel_total[para] = para_total
            logger.debug_rank0(f"{para} persistent 2D total grad norm: {para_total}")

    if math.isinf(norm_type):
        # ``torch.maximum`` is a 2-arg elementwise op, not variadic. Unpacking
        # ``*extra_parallel_total.values(), *extra_parallel_replicated_total.values()``
        # silently worked only when the combined dict size was 1 (single
        # ``extra_parallel_names`` entry + only one of the two buckets
        # populated). Reduce iteratively so the path holds for any number
        # of parallelism axes (``ep`` + ``emb`` + ...) and for the Mode-2
        # shared-LoRA case where both ``extra_parallel_total["ep"]`` and
        # ``extra_parallel_replicated_total["ep"]`` are populated.
        total_norm = non_extra_parallel_total
        for t in itertools.chain(
            extra_parallel_total.values(),
            extra_parallel_replicated_total.values(),
            persistent_extra_parallel_total.values(),
        ):
            total_norm = torch.maximum(total_norm, t)
    else:
        total_norm = _finalize_total_norm(
            non_extra_parallel_total
            + sum(extra_parallel_total.values())
            + sum(extra_parallel_replicated_total.values())
            + sum(persistent_extra_parallel_total.values()),
            norm_type,
        )

    _raise_if_nonfinite(total_norm, norm_type, error_if_nonfinite)

    # Apply the same clip coefficient to all groups
    with SkipDTensorDispatch():
        for para in ps.extra_parallel_names:
            torch.nn.utils.clip_grads_with_norm_(extra_parallel_params[para], max_norm, total_norm, foreach=foreach)
            torch.nn.utils.clip_grads_with_norm_(
                extra_parallel_replicated_params[para], max_norm, total_norm, foreach=foreach
            )
            torch.nn.utils.clip_grads_with_norm_(
                persistent_extra_parallel_params[para], max_norm, total_norm, foreach=foreach
            )
        torch.nn.utils.clip_grads_with_norm_(non_extra_parallel_params, max_norm, total_norm, foreach=foreach)

    return total_norm


def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    """Compute the local p-th norm sum without materializing fp32 gradients."""
    reduce_device = torch.device(get_device_type())
    res = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)

    with torch.no_grad():
        for param in params:
            g = param.grad
            if g is None:
                continue
            if is_dtensor(g):
                g = g.to_local()
            g = g.detach()
            # ``dtype`` controls accumulation inside the reduction kernel. In
            # contrast, ``g.to(float32)`` creates a full-size temporary, which
            # can exceed the remaining VRAM for multi-billion-element experts.
            if g.dtype in (torch.float16, torch.bfloat16, torch.float32):
                norm = torch.linalg.vector_norm(g, ord=p, dtype=torch.float32)
            else:
                # Preserve the previous implementation's behavior for wider
                # or uncommon dtypes. The memory-sensitive training dtypes
                # above use in-kernel FP32 accumulation without this copy.
                norm = torch.linalg.vector_norm(g.to(torch.float32), ord=p)
            res = res + norm.pow(p).to(device=reduce_device, dtype=torch.float32)
    return res


def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    reduce_device = torch.device(get_device_type())
    mx = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if is_dtensor(g):
            g_local = g.to_local()
        else:
            g_local = g
        if mx is None:
            mx = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn.to(reduce_device))
    if mx is None:
        mx = torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
    return mx


def _fsdp2_reduce_group(
    params: List[torch.nn.Parameter],
    norm_type: float,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
) -> torch.Tensor:
    """Compute local group statistic and reduce over provided groups.

    For finite p, returns the globally-reduced sum of p-th powers (not the final norm).
    For inf, returns the globally-reduced max.
    """
    if math.isinf(norm_type):
        val = _local_max(params)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        val = _local_pth_sum(params, p)
        logger.debug_rank0(f"local total grad norm: {val}. ProcessGroups to sum {reduce_groups}")
        for name, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
                logger.debug_rank0(f"After Sum of group {name} total grad norm is {val}")
        return val
