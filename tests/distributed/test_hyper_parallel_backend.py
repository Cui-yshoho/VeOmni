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

import builtins
import importlib
import json
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor

from veomni.arguments import (
    AcceleratorConfig,
    CheckpointConfig,
    DataArguments,
    FSDPConfig,
    ModelArguments,
    OptimizerConfig,
    TrainingArguments,
    VeOmniArguments,
)
from veomni.checkpoint.checkpointer import build_checkpointer
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.checkpoint.hyper_parallel_checkpointer import (
    HyperParallelCheckpointer,
    _decode_optimizer_manifest,
    _encode_optimizer_manifest,
    _optimizer_tensor_placeholders,
    _restore_optimizer_state,
    _split_optimizer_state,
    _topology_signature,
    _validate_topology,
)
from veomni.distributed import hyper_fsdp2, torch_parallelize
from veomni.distributed.hyper_fsdp2 import extra_parallel_gradient_scaling_factor
from veomni.distributed.torch_parallelize import (
    _build_root_fsdp_kwargs,
    build_parallelize_model,
    parallelize_model_fsdp2,
)
from veomni.models.module_utils import _dispatch_parameter
from veomni.optim import muon as muon_module
from veomni.optim import optimizer as optimizer_module


def test_optional_backends_default_to_existing_implementations():
    assert FSDPConfig().fsdp_backend == "torch"
    assert OptimizerConfig(type="muon").use_hyper_optimizer is False
    assert build_checkpointer("dcp", "fsdp2") is DistributedCheckpointer


def test_optional_config_fields_preserve_existing_positional_order():
    optimizer_config = OptimizerConfig("adamw", 2e-4)
    fsdp_config = FSDPConfig("fsdp2", False)

    assert optimizer_config.lr == 2e-4
    assert optimizer_config.use_hyper_optimizer is False
    assert fsdp_config.reshard_after_forward is False
    assert fsdp_config.fsdp_backend == "torch"


def test_build_parallelize_model_omits_backend_kwarg_on_default_torch_route(monkeypatch):
    parallel_state = types.SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2")
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: parallel_state)
    calls = []

    def fake_parallelize_model_fsdp2(**kwargs):
        calls.append(kwargs)
        return kwargs["model"]

    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", fake_parallelize_model_fsdp2)
    model = nn.Linear(4, 4)
    mixed_precision = types.SimpleNamespace(enable=False)

    assert (
        build_parallelize_model(
            model,
            mixed_precision=mixed_precision,
            enable_gradient_checkpointing=False,
        )
        is model
    )
    assert "fsdp_backend" not in calls[-1]

    build_parallelize_model(
        model,
        mixed_precision=mixed_precision,
        enable_gradient_checkpointing=False,
        fsdp_backend="hyper",
    )
    assert calls[-1]["fsdp_backend"] == "hyper"


def test_default_optimizer_route_does_not_import_hyper(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("hyper_parallel"):
            raise AssertionError(f"Default Torch route imported optional dependency {name}.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    optimizer = optimizer_module.build_optimizer(nn.Linear(4, 4), optimizer_type="adamw", fused=False)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert build_checkpointer("dcp", "fsdp2") is DistributedCheckpointer


def test_gradient_stream_sync_is_lazy_and_hyper_only(monkeypatch):
    clip_grad_norm_module = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("hyper_parallel"):
            raise AssertionError(f"Default Torch route imported optional dependency {name}.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    torch_model = nn.Linear(4, 4)
    torch_model._extra_parallel_param_groups = {}
    monkeypatch.setattr(
        clip_grad_norm_module, "extra_parallel_fsdp2_clip_grad_norm", lambda *_args, **_kwargs: torch.tensor(1.0)
    )

    clip_grad_norm_module.clip_grad_norm(torch_model, 1.0)

    calls = []
    monkeypatch.setattr(builtins, "__import__", original_import)
    hyper_parallel = types.ModuleType("hyper_parallel")
    hyper_parallel.hsdp_sync_stream = lambda: calls.append("sync")
    monkeypatch.setitem(sys.modules, "hyper_parallel", hyper_parallel)
    hyper_model = nn.Linear(4, 4)
    hyper_model._veomni_fsdp_backend = "hyper"
    hyper_model._extra_parallel_param_groups = {}
    clip_grad_norm_module.clip_grad_norm(hyper_model, 1.0)

    assert calls == ["sync"]


def test_non_hyper_dtensor_duck_types_do_not_change_default_classification():
    fake_tensor = types.SimpleNamespace(device_mesh=object(), placements=())

    assert optimizer_module._is_fsdp_dtensor(fake_tensor) is False
    assert muon_module._is_hyper_dtensor_like(fake_tensor) is False


def test_hyper_dtensor_detection_uses_registered_type(monkeypatch):
    class FakeHyperDTensor:
        pass

    monkeypatch.setattr(hyper_fsdp2, "_HYPER_DTENSOR_TYPE", FakeHyperDTensor)

    assert hyper_fsdp2.is_hyper_dtensor(FakeHyperDTensor()) is True
    assert hyper_fsdp2.is_hyper_dtensor(types.SimpleNamespace()) is False


def test_dispatch_parameter_keeps_torch_dtensor_path(tmp_path):
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmp_path / 'torch_dtensor_rendezvous'}",
            world_size=1,
            rank=0,
        )

    try:
        mesh = DeviceMesh("cpu", [dist.get_rank()])
        source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        module = nn.Linear(4, 2, bias=False)
        module.weight = nn.Parameter(distribute_tensor(torch.zeros_like(source), mesh, [Shard(0)]))

        _dispatch_parameter(module, "weight", source, distribute_tensor)

        assert isinstance(module.weight, torch.distributed.tensor.DTensor)
        torch.testing.assert_close(module.weight.full_tensor(), source)
    finally:
        if initialized_here:
            dist.destroy_process_group()


def test_hyper_fsdp_backend_keeps_fsdp2_mode_and_selects_checkpointer():
    config = FSDPConfig(fsdp_backend="hyper")

    assert config.fsdp_mode == "fsdp2"
    assert build_checkpointer("dcp", "fsdp2", fsdp_backend="hyper") is HyperParallelCheckpointer


def _hyper_arguments(**checkpoint_kwargs):
    checkpoint_options = {"save_hf_weights": False}
    checkpoint_options.update(checkpoint_kwargs)
    return VeOmniArguments(
        model=ModelArguments(
            config_path="dummy-config",
            accelerator=AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_backend="hyper")),
        ),
        data=DataArguments(train_path="dummy.jsonl"),
        train=TrainingArguments(checkpoint=CheckpointConfig(**checkpoint_options)),
    )


def test_hyper_fsdp_accepts_supported_checkpoint_configuration():
    assert _hyper_arguments().model.accelerator.fsdp_config.fsdp_backend == "hyper"


@pytest.mark.parametrize(
    ("checkpoint_kwargs", "message"),
    [
        ({"manager": "invalid"}, "requires train.checkpoint.manager='dcp'"),
        ({"stage_dir": "/tmp/stage"}, "does not support train.checkpoint.stage_dir"),
        ({"dcp_save_to_lowest_rank": True}, "does not support train.checkpoint.dcp_save_to_lowest_rank"),
        ({"save_hf_weights": True}, "does not yet support Hugging Face export"),
    ],
)
def test_hyper_fsdp_rejects_unsupported_checkpoint_configuration(checkpoint_kwargs, message):
    with pytest.raises(ValueError, match=message):
        _hyper_arguments(**checkpoint_kwargs)


def test_hyper_checkpoint_preserves_explicit_step_zero(tmp_path, monkeypatch):
    calls = []

    def fake_save(state, checkpoint_id, use_collectives):
        calls.append((state, checkpoint_id, use_collectives))

    monkeypatch.setattr(
        "veomni.checkpoint.hyper_parallel_checkpointer._checkpoint_api",
        lambda: (fake_save, None),
    )
    HyperParallelCheckpointer.save(
        str(tmp_path),
        {"model": nn.Linear(2, 2)},
        global_steps=0,
    )

    assert calls[-1][1] == f"{tmp_path}/global_step_0/model"
    assert (tmp_path / "global_step_0" / ".hyper_complete").is_file()


def test_hyper_checkpoint_rejects_different_rank_local_topology(tmp_path):
    model = nn.Linear(2, 2)
    saved_state = types.SimpleNamespace(
        dp_replicate_size=1,
        dp_shard_size=2,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        ulysses_size=1,
        extra_parallel_sizes={"ep": 1},
        extra_parallel_names=("ep",),
        extra_parallel_fsdp_device_mesh={"ep": None},
        device_mesh=None,
    )
    current_state = types.SimpleNamespace(**{**saved_state.__dict__, "dp_replicate_size": 2, "dp_shard_size": 1})
    marker = tmp_path / ".hyper_complete"
    marker.write_text(json.dumps(_topology_signature(model, saved_state)), encoding="utf-8")

    _validate_topology(str(marker), model, saved_state)
    with pytest.raises(RuntimeError, match="same world size.*parallel topology"):
        _validate_topology(str(marker), model, current_state)


def test_root_reshard_policy_matches_torch_effective_behavior():
    child_kwargs = {"mesh": object(), "reshard_after_forward": True, "comm_fusion": True}

    torch_root_kwargs = _build_root_fsdp_kwargs(child_kwargs, use_hyper_fsdp2=False)
    hyper_root_kwargs = _build_root_fsdp_kwargs(child_kwargs, use_hyper_fsdp2=True)

    assert "reshard_after_forward" not in torch_root_kwargs
    assert hyper_root_kwargs["reshard_after_forward"] is False
    assert child_kwargs["reshard_after_forward"] is True


def test_hyper_optimizer_checkpoint_manifest_keeps_tensors_out_of_encoded_structure():
    state = {
        "state": {0: {"step": torch.tensor(2.0), "exp_avg": torch.arange(6).reshape(2, 3)}},
        "param_groups": [{"params": [0], "betas": (0.9, 0.95)}],
    }
    tensors = {}
    manifest = _split_optimizer_state(state, tensors)
    decoded = _decode_optimizer_manifest(_encode_optimizer_manifest(manifest))
    placeholders = {}
    _optimizer_tensor_placeholders(decoded, placeholders)

    assert tensors.keys() == placeholders.keys()
    assert all(tensor.device.type == "cpu" for tensor in placeholders.values())
    for key in tensors:
        placeholders[key].copy_(tensors[key])
    restored = _restore_optimizer_state(decoded, placeholders)
    torch.testing.assert_close(restored["state"][0]["step"], state["state"][0]["step"])
    torch.testing.assert_close(restored["state"][0]["exp_avg"], state["state"][0]["exp_avg"])
    assert restored["param_groups"] == state["param_groups"]


def test_hyper_fsdp_backend_rejects_invalid_modes_and_values():
    with pytest.raises(ValueError, match="requires fsdp_mode='fsdp2'"):
        FSDPConfig(fsdp_mode="ddp", fsdp_backend="hyper")
    with pytest.raises(ValueError, match="Unsupported fsdp_backend"):
        FSDPConfig(fsdp_backend="invalid")
    with pytest.raises(ValueError, match="Unsupported fsdp_backend"):
        parallelize_model_fsdp2(object(), fsdp_backend="invalid")


def test_hyper_optimizer_requires_muon():
    with pytest.raises(ValueError, match="requires optimizer type='muon'"):
        OptimizerConfig(type="adamw", use_hyper_optimizer=True)


def test_hyper_optimizer_rejects_unsupported_muon_layouts():
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="muon_ns_implementation='std'"):
        optimizer_module.build_optimizer(
            model,
            optimizer_type="muon",
            optimizer_config=OptimizerConfig(type="muon", use_hyper_optimizer=True),
        )


def test_hyper_optimizer_is_lazy_and_reuses_veomni_parameter_split(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_get_hyper_optimizer(**kwargs):
        captured.update(kwargs)
        return sentinel

    hyper_parallel = types.ModuleType("hyper_parallel")
    hyper_core = types.ModuleType("hyper_parallel.core")
    hyper_optimizer = types.ModuleType("hyper_parallel.core.optimizer")
    hyper_optimizer.get_hyper_optimizer = fake_get_hyper_optimizer
    monkeypatch.setitem(sys.modules, "hyper_parallel", hyper_parallel)
    monkeypatch.setitem(sys.modules, "hyper_parallel.core", hyper_core)
    monkeypatch.setitem(sys.modules, "hyper_parallel.core.optimizer", hyper_optimizer)
    monkeypatch.setattr(optimizer_module, "_should_build_extra_parallel_aware", lambda _model: False)
    monkeypatch.setattr(hyper_fsdp2, "wrap_optimizer", lambda _optimizer: None)

    model = nn.Linear(4, 4)
    config = OptimizerConfig(type="muon", use_hyper_optimizer=True, muon_ns_steps=3, muon_ns_implementation="std")
    result = optimizer_module.build_optimizer(
        model,
        lr=1e-3,
        optimizer_type="muon",
        optimizer_config=config,
    )

    assert result is sentinel
    assert captured["model"] is model
    assert captured["muon_params"][0]["params"] == [model.weight]
    assert captured["adamw_params"][0]["params"] == [model.bias]
    assert captured["muon_kwargs"]["muon_ns_steps"] == 3
    assert captured["muon_kwargs"]["muon_ns_variant"] == "custom"
    assert captured["muon_kwargs"]["muon_ns_coefficients"] == [(3.4445, -4.775, 2.0315)] * 3
    assert captured["muon_kwargs"]["muon_ns_epsilon"] == 1e-7


@pytest.mark.parametrize(
    ("fsdp_group_size", "gradient_divide_factor", "expected"),
    [(2, 2, 1.0), (4, 2, 2.0), (8, 16, 0.5)],
)
def test_extra_parallel_gradient_scaling_factor(fsdp_group_size, gradient_divide_factor, expected):
    assert extra_parallel_gradient_scaling_factor(fsdp_group_size, gradient_divide_factor) == expected
