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

"""Exercise the VeOmni Hyper planner with real collectives on multiple GPUs."""

import os
import types

import torch
import torch.distributed as dist
from hyper_parallel import DTensor
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.checkpoint import checkpoint

from veomni.distributed import torch_parallelize
from veomni.ops.kernels import cross_entropy
from veomni.ops.kernels.cross_entropy import chunk_logprobs, fused_head
from veomni.ops.kernels.cross_entropy.liger import fused_liger_kernel_cross_entropy
from veomni.utils.model_outputs import FusedLinearAuxOutput, MoeCausalLMOutputWithLogProbs


cross_entropy.get_parallel_state = lambda: types.SimpleNamespace(sp_enabled=False)
chunk_logprobs.get_parallel_state = lambda: types.SimpleNamespace(sp_enabled=False)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(64, 64, bias=False)
        self.mlp = nn.Linear(64, 64, bias=False)

    def forward(self, hidden):
        return hidden + self.mlp(torch.tanh(self.self_attn(hidden)))


class Qwen3MoeForCausalLM(nn.Module):
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, mode, use_checkpoint, tied, frozen):
        super().__init__()
        self.embed_tokens = nn.Embedding(256, 64)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(), Qwen3MoeDecoderLayer()])
        self.norm = nn.LayerNorm(64)
        self.lm_head = nn.Linear(64, 256, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight
        if frozen:
            self.embed_tokens.weight.requires_grad_(False)
        self.mode = mode
        self.use_checkpoint = use_checkpoint
        self.weight_before_head = None
        self.weight_during_loss = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def loss_fn(self, **kwargs):
        weight = kwargs["weights"]
        self.weight_during_loss = (isinstance(weight, DTensor), tuple(weight.shape), weight.dtype)
        return cross_entropy.ForCausalLMLoss(
            **kwargs,
            return_log_probs=self.mode == "auxiliary",
            chunk_size=16,
            cross_entropy_fn=fused_liger_kernel_cross_entropy,
        )

    def forward(self, tokens, labels, inputs_embeds=None):
        self.weight_before_head = isinstance(self.lm_head.weight, DTensor)
        hidden = self.embed_tokens(tokens) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden = checkpoint(layer, hidden, use_reentrant=False) if self.use_checkpoint else layer(hidden)
        hidden = self.norm(hidden)
        if isinstance(self.lm_head, fused_head.FusedCrossEntropy):
            loss, _, *slots = self.lm_head(
                hidden,
                _veomni_loss_fn=self.loss_fn,
                _veomni_labels=labels,
                _veomni_vocab_size=256,
                _veomni_use_fused_loss=True,
            )
            aux = FusedLinearAuxOutput.from_loss_slots(*slots)
        else:
            loss, _, aux = self.loss_fn(
                logits=None, labels=labels, vocab_size=256, hidden_states=hidden, weights=self.lm_head.weight
            )
        return MoeCausalLMOutputWithLogProbs(loss=loss, fused_linear_aux=aux)


# The production planner deliberately limits head conversion to the patched
# Qwen3 MoE class; this toy mirrors that public identity and invocation shape.
Qwen3MoeForCausalLM.__module__ = "veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu"


def one_run(rank, candidate, mode, bypass_embedding, use_checkpoint, tied, frozen, no_sync, active_microsteps=(0, 1)):
    torch.manual_seed(1701)
    model = Qwen3MoeForCausalLM(mode, use_checkpoint, tied, frozen).cuda()
    if tied:
        model.lm_head.weight = model.embed_tokens.weight
    original_keys = set(model.state_dict())
    original_enable = fused_head.enable_qwen3_moe_fused_head
    if not candidate:
        fused_head.enable_qwen3_moe_fused_head = lambda model: None
    try:
        torch_parallelize.parallelize_model_fsdp2(
            model,
            fsdp_backend="hyper",
            mixed_precision=torch_parallelize.MixedPrecisionConfig(),
            init_device="meta",
            enable_forward_prefetch=True,
        )
    finally:
        fused_head.enable_qwen3_moe_fused_head = original_enable
    independent = candidate and not (tied or frozen)
    assert isinstance(model.lm_head, fused_head.FusedCrossEntropy) == independent
    assert set(model.state_dict()) == original_keys
    if tied:
        assert model.embed_tokens.weight is model.lm_head.weight
    if frozen:
        assert not model.embed_tokens.weight.requires_grad
    torch.manual_seed(1730 + rank)
    values = []
    for microstep in range(2):
        tokens = torch.randint(0, 256, (2, 16), device="cuda")
        labels = torch.randint(0, 256, (2, 16), device="cuda")
        inputs_embeds = torch.randn(2, 16, 64, device="cuda", requires_grad=True) if bypass_embedding else None
        if microstep not in active_microsteps:
            continue
        if no_sync:
            model.set_requires_gradient_sync(microstep == 1)
        output = model(tokens, labels, inputs_embeds=inputs_embeds)
        if tied:
            assert model.embed_tokens.weight is model.lm_head.weight
        assert model.weight_before_head == independent, (mode, model.weight_before_head, independent)
        assert model.weight_during_loss == (False, (256, 64), torch.bfloat16), model.weight_during_loss
        objective = output.loss if mode == "liger" else output.fused_linear_aux.log_probs.square().mean()
        assert objective is not None and bool(torch.isfinite(objective))
        values.append(objective.detach().float().cpu())
        (objective / 2).backward()
        if tied:
            assert model.embed_tokens.weight is model.lm_head.weight
    torch.cuda.synchronize()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            assert name == "embed_tokens.weight" and (bypass_embedding or frozen) and not tied, name
            gradients[name] = None
            continue
        local = getattr(parameter.grad, "_local_tensor", parameter.grad).detach()
        assert bool(torch.isfinite(local).all()), (mode, independent, name)
        gradients[name] = local.float().cpu().clone()
    result = (values, gradients)
    dist.barrier()
    del model, output, objective
    torch.cuda.empty_cache()
    return result


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
    state = types.SimpleNamespace(any_extra_parallel_enabled=False, extra_parallel_names=(), fsdp_mesh=mesh)
    torch_parallelize.get_parallel_state = lambda: state
    torch_parallelize._materialize_and_load_weights = lambda *args, **kwargs: None
    cases = [
        ("liger", False, False, False, False, False),
        ("liger", True, True, False, False, False),
        ("auxiliary", False, True, False, False, False),
        ("auxiliary", True, False, False, False, False),
        ("liger", False, True, True, False, False),
        ("liger", False, True, False, True, False),
        ("liger", False, False, False, False, True),
    ]
    for case in cases:
        base_values, base_grads = one_run(rank, False, *case)
        test_values, test_grads = one_run(rank, True, *case)
        for actual, expected in zip(test_values, base_values):
            torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.003)
        assert set(base_grads) == set(test_grads)
        for name, expected in base_grads.items():
            actual = test_grads[name]
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.0005)
        if case == ("liger", False, False, False, False, False):
            _, first_grads = one_run(rank, True, *case, active_microsteps=(0,))
            _, second_grads = one_run(rank, True, *case, active_microsteps=(1,))
            for name, actual in test_grads.items():
                if actual is not None:
                    torch.testing.assert_close(actual, first_grads[name] + second_grads[name], rtol=0.02, atol=0.001)
        print(f"PASS rank={rank} case={case}: loss, planner, lifecycle and local gradients", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
