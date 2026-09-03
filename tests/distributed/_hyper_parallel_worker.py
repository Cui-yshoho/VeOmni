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

"""NPU workers for VeOmni's optional HyperParallel integration tests."""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import Shard

from veomni.arguments import MixedPrecisionConfig, OptimizerConfig
from veomni.checkpoint.hyper_parallel_checkpointer import HyperParallelCheckpointer
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model, parallelize_model_fsdp2
from veomni.optim.optimizer import build_optimizer


class LinearToyBlock(nn.Linear):
    pass


class LinearToyModel(nn.Module):
    _no_split_modules = ["LinearToyBlock"]

    def __init__(self):
        super().__init__()
        self.block = LinearToyBlock(8, 8, device="meta")

    def forward(self, inputs):
        return self.block(inputs)

    def init_weights(self):
        for parameter in self.parameters():
            nn.init.constant_(parameter, 0.25)


class ExpertToyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, 16, 32))

    def forward(self):
        return self.weight.sum()


class ExpertToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Parameter(torch.ones(16, 16))
        self.experts = ExpertToyExperts()

    def forward(self, hidden_states):
        return hidden_states + self.dense.sum() + self.experts()


class ExpertToyModel(nn.Module):
    _no_split_modules = ["ExpertToyBlock"]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.block = ExpertToyBlock()

    def forward(self, hidden_states):
        return self.block(hidden_states).sum()

    def get_parallel_plan(self):
        plan = ParallelPlan(extra_parallel_plan={"ep": {"block.experts.weight": Shard(0)}})
        plan.extra_parallel_fsdp_no_shard_module = {"ep": {"block.experts"}}
        return plan


def _full_tensor(tensor):
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def _full_state(model):
    return {name: _full_tensor(param).detach().clone() for name, param in model.named_parameters()}


def _clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return value


def _assert_state_equal(actual, expected, path="optimizer"):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda message: f"{path}: {message}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: state keys differ: actual={actual.keys()}, expected={expected.keys()}.")
        for key, value in expected.items():
            _assert_state_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"{path}: optimizer state sequence differs.")
        for index, value in enumerate(expected):
            _assert_state_equal(actual[index], value, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: actual={actual!r}, expected={expected!r}.")


def _linear_model(*, skip_weight_load=False):
    return parallelize_model_fsdp2(
        LinearToyModel(),
        fsdp_backend="hyper",
        init_device="meta",
        should_skip_hf_weight_load=skip_weight_load,
        mixed_precision=MixedPrecisionConfig(enable=False),
    )


def _linear_step(model, optimizer, inputs):
    optimizer.zero_grad()
    loss = model(inputs).square().mean()
    loss.backward()
    model.clip_grad_norm_(1.0)
    optimizer.step()
    return loss.detach().clone()


def _optimizer_state_by_parameter(model, optimizer):
    result = {}
    for name, param in model.named_parameters():
        values = optimizer.state.get(param, {})
        result[name] = {
            key: value.detach().clone() if torch.is_tensor(value) else value for key, value in values.items()
        }
    return result


def _run_smoke(args, world_size):
    if world_size % args.replicate_size:
        raise ValueError("WORLD_SIZE must be divisible by --replicate-size")
    shard_size = world_size // args.replicate_size
    init_parallel_state(
        dp_size=world_size,
        dp_replicate_size=args.replicate_size,
        dp_shard_size=shard_size,
        device_type="npu",
        name="base",
    )
    model = parallelize_model_fsdp2(
        LinearToyModel(),
        fsdp_backend=args.fsdp_backend,
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(enable=False),
    )
    optimizer = build_optimizer(
        model,
        lr=1e-3,
        optimizer_type="muon",
        optimizer_config=OptimizerConfig(
            type="muon",
            use_hyper_optimizer=not args.native_muon,
            muon_ns_steps=3,
            muon_ns_implementation="std",
        ),
    )

    owner_path_calls = 0
    if args.native_muon:
        from veomni.optim.muon import DistributedMuon

        original_owner_path = DistributedMuon._ortho_fsdp_group_all2all

        def counted_owner_path(self, *owner_args, **owner_kwargs):
            nonlocal owner_path_calls
            owner_path_calls += 1
            return original_owner_path(self, *owner_args, **owner_kwargs)

        DistributedMuon._ortho_fsdp_group_all2all = counted_owner_path

    torch.manual_seed(7)
    inputs = torch.randn(2, 8, device="npu")
    loss = model(inputs).square().mean()
    loss.backward()
    grad_norm = model.clip_grad_norm_(1.0)
    optimizer.step()
    optimizer.zero_grad()

    if args.native_muon and owner_path_calls == 0:
        raise AssertionError("Hyper DTensor native Muon did not use the owner all-to-all path.")

    values = torch.tensor([loss.detach(), grad_norm.detach()], device="npu")
    gathered = [torch.empty_like(values) for _ in range(world_size)]
    dist.all_gather(gathered, values)
    for other in gathered[1:]:
        torch.testing.assert_close(other, gathered[0])

    if dist.get_rank() == 0:
        print(
            f"HyperParallel optimizer smoke passed: fsdp_backend={args.fsdp_backend}, world_size={world_size}, "
            f"replicate_size={args.replicate_size}, shard_size={shard_size}"
        )


def _run_checkpoint(args, world_size):
    init_parallel_state(dp_size=world_size, dp_mode="fsdp2", device_type="npu")
    inputs = torch.arange(16, device="npu", dtype=torch.float32).reshape(2, 8)
    model = _linear_model()
    optimizer = build_optimizer(model, optimizer_type="adamw", fused=False)
    _linear_step(model, optimizer, inputs)
    checkpoint_state = _full_state(model)
    checkpoint_optimizer_state = _optimizer_state_by_parameter(model, optimizer)
    HyperParallelCheckpointer.save(
        args.checkpoint_path,
        {"model": model, "optimizer": optimizer, "extra_state": {"global_step": 1}},
    )
    expected_loss = _linear_step(model, optimizer, inputs)
    expected_state = _full_state(model)

    resumed_model = _linear_model(skip_weight_load=True)
    resumed_optimizer = build_optimizer(resumed_model, optimizer_type="adamw", fused=False)
    resume_state = {"model": resumed_model, "optimizer": resumed_optimizer, "extra_state": {}}
    HyperParallelCheckpointer.load(args.checkpoint_path, resume_state)
    if resume_state["extra_state"]["global_step"] != 1:
        raise AssertionError("Checkpoint extra_state did not restore global_step.")
    _assert_state_equal(_full_state(resumed_model), checkpoint_state, "model")
    _assert_state_equal(_optimizer_state_by_parameter(resumed_model, resumed_optimizer), checkpoint_optimizer_state)

    actual_loss = _linear_step(resumed_model, resumed_optimizer, inputs)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    _assert_state_equal(_full_state(resumed_model), expected_state, "model_after_step")

    if dist.get_rank() == 0:
        print(f"HyperParallel checkpoint smoke passed: world_size={world_size}")


def _run_extra_parallel(args, world_size):
    if world_size % 2:
        raise ValueError("The ExtraParallel smoke test requires an even world size.")
    if world_size % args.replicate_size:
        raise ValueError("WORLD_SIZE must be divisible by --replicate-size")
    shard_size = world_size // args.replicate_size
    if dist.get_rank() == 0:
        os.makedirs(args.weights_path, exist_ok=False)
        torch.save(ExpertToyModel().state_dict(), os.path.join(args.weights_path, "pytorch_model.bin"))
    dist.barrier()
    init_parallel_state(
        dp_size=world_size,
        dp_replicate_size=args.replicate_size,
        dp_shard_size=shard_size,
        dp_mode="fsdp2",
        device_type="npu",
        extra_parallel_sizes=(2,),
        extra_parallel_names=("ep",),
        extra_parallel_placement_innermost=(False,),
    )

    def build(backend, *, skip_weight_load=False):
        with torch.device("meta"):
            model = ExpertToyModel()
        return build_parallelize_model(
            model,
            init_device="meta",
            weights_path=args.weights_path,
            fsdp_backend=backend,
            should_skip_hf_weight_load=skip_weight_load,
            broadcast_model_weights_from_rank0=True,
            enable_gradient_checkpointing=False,
        )

    inputs = torch.arange(16, device="npu", dtype=torch.float32)

    def train_step(model, optimizer=None):
        optimizer = optimizer or build_optimizer(model, optimizer_type="adamw", fused=False)
        optimizer.zero_grad()
        loss = model(inputs)
        loss.backward()
        grad_norm = model.clip_grad_norm_(1.0)
        optimizer.step()
        return loss.detach(), grad_norm.detach(), _full_state(model)

    torch_result = train_step(build("torch"))
    hyper_result = train_step(build("hyper"))
    torch.testing.assert_close(hyper_result[0], torch_result[0], rtol=0, atol=0)
    torch.testing.assert_close(hyper_result[1], torch_result[1], rtol=1e-6, atol=1e-6)
    for name, expected in torch_result[2].items():
        torch.testing.assert_close(hyper_result[2][name], expected, rtol=1e-5, atol=1e-6)

    if args.checkpoint_path:
        checkpoint_model = build("hyper")
        checkpoint_optimizer = build_optimizer(checkpoint_model, optimizer_type="adamw", fused=False)
        train_step(checkpoint_model, checkpoint_optimizer)
        checkpoint_optimizer_state = _clone_state(checkpoint_optimizer.state_dict())
        HyperParallelCheckpointer.save(
            args.checkpoint_path,
            {"model": checkpoint_model, "optimizer": checkpoint_optimizer, "extra_state": {"global_step": 1}},
        )
        expected_loss = train_step(checkpoint_model, checkpoint_optimizer)[0]
        expected_state = _full_state(checkpoint_model)

        resumed_model = build("hyper", skip_weight_load=True)
        resumed_optimizer = build_optimizer(resumed_model, optimizer_type="adamw", fused=False)
        resume_state = {"model": resumed_model, "optimizer": resumed_optimizer, "extra_state": {}}
        HyperParallelCheckpointer.load(args.checkpoint_path, resume_state)
        _assert_state_equal(resumed_optimizer.state_dict(), checkpoint_optimizer_state)
        actual_loss = train_step(resumed_model, resumed_optimizer)[0]
        if resume_state["extra_state"]["global_step"] != 1:
            raise AssertionError("Checkpoint extra_state did not restore global_step.")
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        _assert_state_equal(_full_state(resumed_model), expected_state, "model_after_step")

    if dist.get_rank() == 0:
        print(
            "HyperParallel ExtraParallel smoke passed: "
            f"world_size={world_size}, replicate_size={args.replicate_size}, "
            f"shard_size={shard_size}, ep_size=2"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("smoke", "checkpoint", "extra_parallel"))
    parser.add_argument("--replicate-size", type=int, default=1)
    parser.add_argument("--fsdp-backend", choices=("torch", "hyper"), default="hyper")
    parser.add_argument("--native-muon", action="store_true")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--weights-path")
    args = parser.parse_args()
    if args.scenario == "checkpoint" and not args.checkpoint_path:
        parser.error("checkpoint requires --checkpoint-path")
    if args.scenario == "extra_parallel" and not args.weights_path:
        parser.error("extra_parallel requires --weights-path")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    try:
        if args.scenario == "smoke":
            _run_smoke(args, world_size)
        elif args.scenario == "checkpoint":
            _run_checkpoint(args, world_size)
        else:
            _run_extra_parallel(args, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
