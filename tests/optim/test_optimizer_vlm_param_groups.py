"""Regression tests for VLM-style optimizer param groups + DCP resume."""

import pytest
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from veomni.optim.optimizer import (
    build_optimizer,
    filter_empty_param_groups,
    restore_optimizer_param_group_defaults,
)


@pytest.mark.parametrize("fsdp_backend", ["torch", "hyper"])
@pytest.mark.parametrize("hyper_optimizer", [False, True])
def test_optimizer_and_fsdp_backends_remain_independent(monkeypatch, fsdp_backend, hyper_optimizer):
    from veomni.optim import hyper_muon as hyper_muon_module
    from veomni.optim import hyper_optimizer as hyper_optimizer_module
    from veomni.optim import optimizer as optimizer_module

    model = torch.nn.Linear(4, 4)
    model._veomni_fsdp_backend = fsdp_backend
    optimizer = object()
    selected, wrapped = [], []

    def native_factory(*args, **kwargs):
        selected.append("native")
        return optimizer

    def hyper_factory(*args, **kwargs):
        selected.append("hyper")
        return optimizer

    monkeypatch.setattr(optimizer_module, "_build_muon_with_adamw", native_factory)
    monkeypatch.setattr(hyper_muon_module, "build_hyper_muon_with_adamw", hyper_factory)
    monkeypatch.setattr(hyper_optimizer_module, "wrap_optimizer", lambda opt: (wrapped.append(opt), opt)[1])
    result = optimizer_module.build_optimizer(model, optimizer_type="muon", use_hyper_optimizer=hyper_optimizer)

    assert result is optimizer
    assert selected == ["hyper" if hyper_optimizer else "native"]
    assert wrapped == ([optimizer] if hyper_optimizer or fsdp_backend == "hyper" else [])


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_local_shard_execution_preserves_step_hooks_state_and_exception_cleanup(monkeypatch, fail, grouped):
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    from veomni.optim.hyper_optimizer import wrap_optimizer
    from veomni.optim.lr_scheduler import build_lr_scheduler

    events = []

    @contextmanager
    def dispatch_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setitem(sys.modules, "hyper_parallel", SimpleNamespace(SkipDTensorDispatch=dispatch_context))
    parameter = torch.nn.Parameter(torch.ones(2))
    leaf = torch.optim.AdamW([parameter], lr=1e-3)
    leaf.register_step_pre_hook(lambda *args: events.append("pre"))
    leaf.register_step_post_hook(lambda *args: events.append("post"))

    class Group:
        _is_multi_optimizer = True
        key_names = ["leaf"]

        def __init__(self):
            self.optimizers_dict = {"leaf": leaf}

        def step(self):
            return leaf.step()

        def state_dict(self):
            return {"leaf": leaf.state_dict()}

    owner = Group() if grouped else leaf
    original_step = owner.step.__func__
    adapter = wrap_optimizer(owner)
    assert wrap_optimizer(adapter) is adapter
    assert owner.step.__func__ is original_step
    scheduler = build_lr_scheduler(adapter, train_steps=4, lr=1e-3, lr_decay_style="linear")
    if fail:

        def reject_step(*args):
            raise RuntimeError("step failed")

        leaf.register_step_pre_hook(reject_step)
        with pytest.raises(RuntimeError, match="step failed"):
            adapter.step()
        assert events == ["enter", "pre", "exit"]
    else:
        parameter.grad = torch.ones_like(parameter)
        adapter.step()
        scheduler.step()
        assert events == ["enter", "pre", "post", "exit"]
        assert leaf.param_groups[0]["lr"] < 1e-3
        assert owner.step.__func__ is original_step if grouped else owner.step.__wrapped__ is original_step
        assert adapter.state_dict().keys() == owner.state_dict().keys()
        if not grouped:
            restored = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
            restored.load_state_dict(adapter.state_dict())
            assert restored.param_groups[0]["lr"] == leaf.param_groups[0]["lr"]
            assert restored.state_dict()["state"][0]["step"] == 1


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.backbone(x)


def test_filter_empty_param_groups_drops_frozen_vit_group():
    p = torch.nn.Parameter(torch.randn(2, 2))
    groups = [
        {"params": [], "lr": 1e-6},
        {"params": [p], "lr": 2e-5},
    ]
    filtered = filter_empty_param_groups(groups)
    assert len(filtered) == 1
    assert filtered[0]["params"] == [p]


def test_filter_empty_param_groups_drops_frozen_only_group():
    """A group whose params are all requires_grad=False has no optimizer state
    and must be dropped (otherwise the same KeyError surfaces on resume)."""
    frozen = torch.nn.Parameter(torch.randn(2, 2), requires_grad=False)
    trainable = torch.nn.Parameter(torch.randn(2, 2))
    groups = [
        {"params": [frozen], "lr": 1e-6},
        {"params": [frozen, trainable], "lr": 2e-5},
    ]
    filtered = filter_empty_param_groups(groups)
    assert len(filtered) == 1
    assert filtered[0]["params"] == [trainable]  # frozen dropped, trainable kept


def test_filter_empty_param_groups_handles_generators():
    """A generator params value must be materialized: an empty generator is
    dropped (bool(generator) is always True), and a non-empty one survives
    without being exhausted before optimizer construction."""
    p = torch.nn.Parameter(torch.randn(2, 2))
    groups = [
        {"params": (x for x in []), "lr": 1e-6},  # empty generator -> dropped
        {"params": (x for x in [p]), "lr": 2e-5},  # non-empty generator -> kept, materialized
    ]
    filtered = filter_empty_param_groups(groups)
    assert len(filtered) == 1
    assert filtered[0]["params"] == [p]  # materialized to a list, not exhausted


def test_restore_defaults_recurses_multi_optimizer():
    """restore_optimizer_param_group_defaults must reach sub-optimizers of a
    MultiOptimizer-like container (optimizers_dict) that has no top-level
    defaults/param_groups."""
    p1 = torch.nn.Parameter(torch.randn(2, 2))
    p2 = torch.nn.Parameter(torch.randn(2, 2))
    sub1 = torch.optim.AdamW([p1], lr=2e-5, betas=(0.9, 0.95))
    sub2 = torch.optim.AdamW([p2], lr=1e-6, betas=(0.9, 0.95))
    # Simulate the betas loss on both sub-optimizers.
    for opt in (sub1, sub2):
        opt.param_groups[0].pop("betas")

    class _FakeMultiOptimizer:
        def __init__(self, opts):
            self.optimizers_dict = opts

    restore_optimizer_param_group_defaults(_FakeMultiOptimizer({"a": sub1, "b": sub2}))
    assert "betas" in sub1.param_groups[0]
    assert "betas" in sub2.param_groups[0]


def test_restore_defaults_handles_dict_of_optimizers():
    """PyTorch DCP APIs accept a plain dict of optimizers; restore must recurse
    into dict values too."""
    p1 = torch.nn.Parameter(torch.randn(2, 2))
    p2 = torch.nn.Parameter(torch.randn(2, 2))
    sub1 = torch.optim.AdamW([p1], lr=2e-5, betas=(0.9, 0.95))
    sub2 = torch.optim.AdamW([p2], lr=1e-6, betas=(0.9, 0.95))
    for opt in (sub1, sub2):
        opt.param_groups[0].pop("betas")

    restore_optimizer_param_group_defaults({"a": sub1, "b": sub2})
    assert "betas" in sub1.param_groups[0]
    assert "betas" in sub2.param_groups[0]


def test_build_optimizer_skips_empty_vlm_param_groups():
    model = _TinyModel()
    opt = build_optimizer(
        model,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model.parameters()), "lr": 2e-5},
        ],
    )
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 2e-5


def test_vlm_trainer_style_param_groups_skip_empty_vit():
    """Mirrors VLMModelRuntime.build_optimizer: only append groups that have
    trainable params (freeze_vit -> vit_params == [] must not create a group).
    """
    model = _TinyModel()
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    param_groups = []
    if vit_params:
        param_groups.append({"params": vit_params, "lr": 1e-6})
    if other_params:
        param_groups.append({"params": other_params, "lr": 2e-5})

    assert vit_params == []
    assert len(param_groups) == 1
    opt = build_optimizer(model, lr=2e-5, param_groups=param_groups)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 2e-5


def test_vlm_style_optimizer_dcp_roundtrip_step_succeeds():
    """Reproduces production freeze_vit resume: save/load must keep betas."""
    model = _TinyModel()
    opt = build_optimizer(
        model,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model.parameters()), "lr": 2e-5},
        ],
    )

    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()

    optim_sd = get_optimizer_state_dict(
        model=model,
        optimizers=opt,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )

    model2 = _TinyModel()
    opt2 = build_optimizer(
        model2,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model2.parameters()), "lr": 2e-5},
        ],
    )
    set_optimizer_state_dict(
        model=model2,
        optimizers=opt2,
        optim_state_dict=optim_sd,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )
    restore_optimizer_param_group_defaults(opt2)

    for group in opt2.param_groups:
        assert "betas" in group

    loss2 = model2(torch.randn(2, 4)).sum()
    loss2.backward()
    opt2.step()


def test_restore_defaults_fixes_corrupted_empty_group():
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = torch.optim.AdamW(
        [
            {"params": [], "lr": 1e-6},
            {"params": [p], "lr": 2e-5},
        ],
        lr=2e-5,
        betas=(0.9, 0.95),
    )
    opt.param_groups[0] = {"params": [], "lr": 1e-6}
    restore_optimizer_param_group_defaults(opt)
    assert "betas" in opt.param_groups[0]

    p.grad = torch.randn_like(p)
    opt.step()
