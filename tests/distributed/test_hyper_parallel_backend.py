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
import copy
import importlib
import json
import sys
import types
from functools import partial

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
from veomni.distributed.fsdp2.submodule_prefetch import (
    build_submodule_prefetch_stages,
    configure_submodule_prefetch,
    resolve_submodule_wrap_targets,
)
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
    assert not hasattr(FSDPConfig(), "hyper_wrap_modules")
    assert OptimizerConfig(type="muon").use_hyper_optimizer is False
    assert build_checkpointer("dcp", "fsdp2") is DistributedCheckpointer


def test_optional_config_fields_preserve_existing_positional_order():
    optimizer_config = OptimizerConfig("adamw", 2e-4)
    fsdp_config = FSDPConfig("fsdp2", "module", False)

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
    assert "hyper_wrap_modules" not in calls[-1]

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

    assert calls[-1][1] == f"{tmp_path}/global_step_0/model/ckpt"
    assert (tmp_path / "global_step_0" / "model" / ".hyper_complete").is_file()


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


class PrefetchUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))

    def set_modules_to_forward_prefetch(self, modules):
        self.forward_configurations = getattr(self, "forward_configurations", 0) + 1
        self.forward_targets = modules

    def set_modules_to_backward_prefetch(self, modules):
        self.backward_targets = modules


class PrefetchBlock(PrefetchUnit):
    def __init__(self):
        super().__init__()
        self.second = PrefetchUnit()
        self.first = PrefetchUnit()


class Qwen3MoeDecoderLayer(PrefetchUnit):
    def __init__(self):
        super().__init__()
        # Registration order does not determine forward execution order.
        self.mlp = PrefetchUnit()
        self.self_attn = PrefetchUnit()


@pytest.mark.parametrize(
    "device, compiled, known",
    [
        ("cuda", False, True),
        ("npu", False, True),
        ("cpu", False, True),
        ("cuda", True, True),
        ("cuda", False, False),
    ],
)
def test_submodule_wrap_uses_safe_internal_defaults(device, compiled, known):
    root = nn.Module()
    root.block = Qwen3MoeDecoderLayer() if known else PrefetchBlock()
    result = resolve_submodule_wrap_targets(
        [("block", root.block)], root_model=root, device_type=device, compiled=compiled
    )
    expected = (
        {root.block: [root.block.self_attn, root.block.mlp]} if device == "cuda" and not compiled and known else {}
    )
    assert result == expected


@pytest.mark.parametrize("outside_kind", ["sibling", "other_block", "same_block", "missing_child", "empty_child"])
def test_submodule_wrap_falls_back_for_unsafe_boundaries(outside_kind):
    root = nn.Module()
    root.block = Qwen3MoeDecoderLayer()
    if outside_kind == "missing_child":
        del root.block.self_attn
    elif outside_kind == "empty_child":
        root.block.self_attn = nn.Identity()
    elif outside_kind == "same_block":
        root.block.mlp.weight = root.block.self_attn.weight
    else:
        root.outside = PrefetchUnit() if outside_kind == "sibling" else Qwen3MoeDecoderLayer()
        outside = root.outside if outside_kind == "sibling" else root.outside.self_attn
        outside.weight = root.block.self_attn.weight
    assert resolve_submodule_wrap_targets([("block", root.block)], root_model=root, device_type="cuda") == {}


@pytest.mark.parametrize("stage_counts", [(3, 3), (1, 3, 2)])
def test_submodule_forward_targets_are_set_once(stage_counts):
    stages = [[[PrefetchUnit(), PrefetchUnit()] for _ in range(count)] for count in stage_counts]
    units = [unit for block in stages for stage in block for unit in stage]
    configure_submodule_prefetch(stages)
    assert all(unit.forward_configurations == 1 for unit in units)


def test_submodule_prefetch_keeps_nested_expert_and_fp32_units():
    blocks = [PrefetchBlock(), PrefetchBlock()]
    stages = []
    for block in blocks:
        block.second.router = PrefetchUnit()
        block.second.experts = PrefetchUnit()
        existing = [block.second.experts, block.second.router]
        stages.append(build_submodule_prefetch_stages(block, [block.first, block.second], existing))
        assert stages[-1] == [[block], [block.first], [block.second, *existing]]
    configure_submodule_prefetch(stages)
    first, second = blocks
    assert first.forward_targets == [first.first]
    assert first.first.forward_targets == stages[0][2] + [second, second.first]
    assert first.second.forward_targets == [second, second.first, *stages[1][2]]
    assert first.second.experts.forward_targets == first.second.forward_targets
    assert second.backward_targets == stages[1][2]
    assert second.second.backward_targets == [second.first]
    assert second.first.backward_targets == [first, *stages[0][2]]
    assert first.first.backward_targets == []
    assert second.second.forward_targets == []


@pytest.mark.parametrize("backend, interval", [("hyper", 1), ("hyper", 2), ("torch", 1)])
@pytest.mark.parametrize(
    "device, known, prefetch",
    [
        ("cuda", True, True),
        ("cuda", True, False),
        ("npu", True, True),
        ("cuda", False, True),
    ],
)
def test_submodule_wrap_integration_uses_internal_device_policy(
    monkeypatch, device, known, prefetch, backend, interval
):
    model = nn.Module()
    block_type = Qwen3MoeDecoderLayer if known else PrefetchBlock
    model.layers = nn.ModuleList([block_type(), block_type()])
    model._no_split_modules = [block_type.__name__]
    state = types.SimpleNamespace(any_extra_parallel_enabled=False, extra_parallel_names=(), fsdp_mesh=object())
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: device)
    monkeypatch.setattr(hyper_fsdp2, "to_hyper_device_mesh", lambda *args: state.fsdp_mesh)
    monkeypatch.setattr(torch_parallelize, "_materialize_and_load_weights", lambda *args, **kwargs: None)
    calls, claimed = [], set()

    class WrappedUnit:
        def set_reduce_comm_interval(self, value):
            self.reduce_interval = value

    def fake_wrap(module, **options):
        owned = {id(param) for param in module.parameters()} - claimed
        claimed.update(owned)
        calls.append((module, options, owned))
        module.__class__ = type("Wrapped" + type(module).__name__, (type(module), WrappedUnit), {})

    monkeypatch.setattr(torch_parallelize, "fully_shard", fake_wrap)
    monkeypatch.setattr(torch_parallelize, "FSDPModule", WrappedUnit)
    if backend == "torch":

        def reject_hyper_planner(*args, **kwargs):
            pytest.fail("Torch-native must not enter the Hyper wrap planner")

        monkeypatch.setattr(
            sys.modules[resolve_submodule_wrap_targets.__module__],
            "resolve_submodule_wrap_targets",
            reject_hyper_planner,
        )
    monkeypatch.setattr(
        hyper_fsdp2,
        "get_hyper_fsdp2_api",
        lambda: {
            "fully_shard": fake_wrap,
            "module_type": WrappedUnit,
            "mixed_precision_policy": object,
            "cpu_offload_policy": object,
            "shard": Shard,
            "distribute_tensor": object,
        },
    )
    blocks = list(model.layers)
    parallelize_model_fsdp2(
        model,
        fsdp_backend=backend,
        mixed_precision=types.SimpleNamespace(enable=False),
        init_device="meta",
        enable_forward_prefetch=prefetch,
        hyper_reduce_comm_interval=interval,
    )
    if backend == "hyper" and device == "cuda" and known:
        assert [call[0] for call in calls] == [
            blocks[0].self_attn,
            blocks[0].mlp,
            blocks[0],
            blocks[1].self_attn,
            blocks[1].mlp,
            blocks[1],
            model,
        ]
        assert calls[2][2] == {id(blocks[0].weight)}
        if prefetch:
            assert blocks[0].self_attn.forward_targets == [blocks[0].mlp, blocks[1], blocks[1].self_attn]
            assert blocks[0].mlp.forward_targets == [blocks[1], blocks[1].self_attn, blocks[1].mlp]
        else:
            assert not hasattr(blocks[0].self_attn, "forward_targets")
    else:
        assert [call[0] for call in calls] == [*blocks, model]
        if backend == "hyper":
            assert blocks[0].forward_targets == [blocks[1]]
        else:
            assert not hasattr(blocks[0], "forward_targets")
    if backend == "hyper":
        assert calls[-1][1]["reshard_after_forward"] is False
    else:
        assert "reshard_after_forward" not in calls[-1][1]
    assert claimed == {id(param) for param in model.parameters()}
    assert getattr(model, "reduce_interval", None) == (interval if interval != 1 else None)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "2"])
def test_reduce_interval_rejects_invalid_values(interval):
    with pytest.raises(ValueError, match="positive integer"):
        FSDPConfig(fsdp_backend="hyper", hyper_reduce_comm_interval=interval)
    with pytest.raises(ValueError, match="positive integer"):
        parallelize_model_fsdp2(nn.Linear(2, 2), fsdp_backend="hyper", hyper_reduce_comm_interval=interval)


def test_reduce_interval_is_hyper_only():
    assert FSDPConfig().hyper_reduce_comm_interval == 1
    with pytest.raises(ValueError, match="requires fsdp_backend"):
        FSDPConfig(hyper_reduce_comm_interval=2)
    with pytest.raises(ValueError, match="requires fsdp_backend"):
        parallelize_model_fsdp2(nn.Linear(2, 2), hyper_reduce_comm_interval=2)


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_vl_moe"])
def test_submodule_declarations_resolve_real_qwen_meta_models(family):
    from transformers import (
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
        Qwen3VLMoeConfig,
        Qwen3VLMoeForConditionalGeneration,
    )

    text_options = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        head_dim=8,
    )
    if family == "qwen3_moe":
        config = Qwen3MoeConfig(**text_options)
        model_class = Qwen3MoeForCausalLM
        policy = {"Qwen3MoeDecoderLayer": ["self_attn", "mlp"]}
        expected_blocks = 2
    else:
        config = Qwen3VLMoeConfig(
            text_config=text_options,
            vision_config=dict(
                depth=1,
                hidden_size=16,
                intermediate_size=32,
                num_heads=2,
                out_hidden_size=16,
                deepstack_visual_indexes=[],
            ),
        )
        model_class = Qwen3VLMoeForConditionalGeneration
        policy = {
            "Qwen3VLMoeVisionBlock": ["attn", "mlp"],
            "Qwen3VLMoeTextDecoderLayer": ["self_attn", "mlp"],
        }
        expected_blocks = 3
    with torch.device("meta"):
        model = model_class(config)
    targets = [(name, module) for name, module in model.named_modules() if type(module).__name__ in policy]
    resolved = resolve_submodule_wrap_targets(targets, root_model=model, device_type="cuda")
    assert len(resolved) == expected_blocks
    assert all(len(children) == 2 for children in resolved.values())
    assert all(param.is_meta for param in model.parameters())


def test_submodule_prefetch_supports_three_arbitrary_stages():
    block = PrefetchBlock()
    block.third = PrefetchUnit()
    stages = build_submodule_prefetch_stages(block, [block.first, block.second, block.third], [])
    configure_submodule_prefetch([stages])
    assert block.first.forward_targets == [block.second, block.third]
    assert block.second.forward_targets == [block.third]
    assert block.third.backward_targets == [block.second]
    assert block.second.backward_targets == [block.first]


@pytest.mark.parametrize("device", ["cpu", "npu", "mlu"])
def test_hyper_optimizer_rejects_unsupported_muon_layouts(monkeypatch, device):
    monkeypatch.setattr(optimizer_module, "get_device_type", lambda: device)
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="muon_ns_implementation='std'"):
        optimizer_module.build_optimizer(
            model,
            optimizer_type="muon",
            optimizer_config=OptimizerConfig(type="muon", use_hyper_optimizer=True),
        )


@pytest.mark.parametrize(
    ("device", "implementation"),
    [("cpu", "std"), ("npu", "std"), ("cuda", "std"), ("cuda", "gram"), ("cuda", "gram_quack")],
)
def test_hyper_optimizer_is_lazy_and_reuses_veomni_parameter_split(monkeypatch, device, implementation):
    captured = {}
    sentinel = types.SimpleNamespace(optimizers_dict={})

    def fake_get_hyper_optimizer(**kwargs):
        captured.update(kwargs)
        sentinel.optimizers_dict["muon"] = torch.optim.SGD(kwargs["muon_params"], lr=1e-3)
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
    monkeypatch.setattr(optimizer_module, "get_device_type", lambda: device)

    model = nn.Linear(4, 4)
    config = OptimizerConfig(
        type="muon",
        use_hyper_optimizer=True,
        muon_ns_steps=3,
        muon_ns_implementation=implementation,
        muon_gram_ns_reset_iterations=[1],
    )
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
    callback = captured["muon_kwargs"].get("muon_zeropower_fn")
    if implementation == "std":
        assert callback is None
    else:
        calls = {}

        def fake_ns(grad, **kwargs):
            calls.update(kwargs)
            return grad + 1

        monkeypatch.setattr(muon_module, "run_newton_schulz", fake_ns)
        for shape in [(4, 3), (2, 4, 3)]:
            grad = torch.zeros(shape)
            torch.testing.assert_close(callback(grad, steps=3), grad + 1)
        assert calls == {
            "ns_steps": 3,
            "ns_coefficients": (3.4445, -4.775, 2.0315),
            "eps": 1e-7,
            "ns_implementation": implementation,
            "gram_ns_reset_iterations": (1,),
        }


@pytest.mark.parametrize("shape", [(8, 4), (2, 8, 4), (4, 8)])
def test_hyper_muon_gram_callback_matches_native_dispatch(shape):
    grad = torch.randn(shape, generator=torch.Generator().manual_seed(42), dtype=torch.bfloat16)
    kwargs = {"ns_implementation": "gram", "eps": 1e-7, "gram_ns_reset_iterations": (2,)}
    actual = optimizer_module._run_hyper_muon_newton_schulz(grad, steps=5, **kwargs)
    expected = muon_module.run_newton_schulz(grad, ns_steps=5, **kwargs)
    assert actual.shape == grad.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("changed", [None, "signature", "coefficients", "epsilon", "steps", "missing"])
def test_hyper_muon_callback_resume_checks_settings(changed):
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(4, 3))], lr=1e-3)
    callback = partial(
        optimizer_module._run_hyper_muon_newton_schulz,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_implementation="gram_quack",
        gram_ns_reset_iterations=(2,),
    )
    optimizer.param_groups[0].update(ns_coefficients=[(3.4445, -4.775, 2.0315)] * 5, ns_epsilon=1e-7, ns_steps=5)
    optimizer_module._guard_hyper_muon_callback_resume(optimizer, callback, 5)
    state = copy.deepcopy(optimizer.state_dict())
    group = state["param_groups"][0]
    if changed == "signature":
        group["veomni_ns_callback"] = group["veomni_ns_callback"].replace("gram_quack", "gram")
    elif changed == "coefficients":
        group["ns_coefficients"] = [(1.0, 2.0, 3.0)] * 5
    elif changed == "epsilon":
        group["ns_epsilon"] = 1e-6
    elif changed == "steps":
        group["ns_steps"] = 4
    elif changed == "missing":
        del group["veomni_ns_callback"]
    if changed is None:
        optimizer.load_state_dict(state)
        assert optimizer.param_groups[0]["veomni_ns_callback"] == group["veomni_ns_callback"]
    else:
        with pytest.raises(ValueError, match="checkpoint settings"):
            optimizer.load_state_dict(state)


@pytest.mark.parametrize("callback_state", [False, True])
def test_hyper_muon_std_rejects_callback_checkpoint(callback_state):
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(4, 3))], lr=1e-3)
    optimizer_module._guard_hyper_muon_callback_resume(optimizer, None, 5)
    state = copy.deepcopy(optimizer.state_dict())
    if callback_state:
        state["param_groups"][0]["veomni_ns_callback"] = "gram_quack"
        with pytest.raises(ValueError, match="cannot be restored into std"):
            optimizer.load_state_dict(state)
    else:
        optimizer.load_state_dict(state)


@pytest.mark.parametrize(
    ("fsdp_group_size", "gradient_divide_factor", "expected"),
    [(2, 2, 1.0), (4, 2, 2.0), (8, 16, 0.5)],
)
def test_extra_parallel_gradient_scaling_factor(fsdp_group_size, gradient_divide_factor, expected):
    assert extra_parallel_gradient_scaling_factor(fsdp_group_size, gradient_divide_factor) == expected
