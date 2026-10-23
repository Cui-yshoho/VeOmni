from types import SimpleNamespace

import torch

from veomni.optim.lr_scheduler import MultiLRScheduler, build_lr_scheduler


def _optimizer():
    return torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=1e-3)


def test_build_scheduler_preserves_veomni_multi_optimizer_names():
    optimizer = SimpleNamespace(
        _is_multi_optimizer=True,
        key_names=["ep", "non_extra_parallel"],
        optimizers_dict={"ep": _optimizer(), "non_extra_parallel": _optimizer()},
    )

    scheduler = build_lr_scheduler(optimizer, train_steps=4, lr_decay_style="linear")

    assert isinstance(scheduler, MultiLRScheduler)
    assert list(scheduler) == optimizer.key_names


def test_build_scheduler_accepts_hyper_chained_optimizer_protocol():
    optimizer = SimpleNamespace(
        _is_multi_optimizer=True,
        optimizers_keys=["muon", "adamw"],
        optimizers_dict={"muon": _optimizer(), "adamw": _optimizer()},
    )

    scheduler = build_lr_scheduler(optimizer, train_steps=4, lr_decay_style="linear")

    assert isinstance(scheduler, MultiLRScheduler)
    assert list(scheduler) == optimizer.optimizers_keys


def test_build_scheduler_accepts_plain_optimizer_mapping():
    optimizers = {"left": _optimizer(), "right": _optimizer()}

    scheduler = build_lr_scheduler(optimizers, train_steps=4, lr_decay_style="linear")

    assert isinstance(scheduler, MultiLRScheduler)
    assert list(scheduler) == list(optimizers)
