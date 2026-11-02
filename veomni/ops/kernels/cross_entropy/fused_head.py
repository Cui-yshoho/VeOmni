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

"""FSDP-visible output projection and causal loss boundary."""

import torch.nn as nn


class FusedCrossEntropy(nn.Linear):
    """Run the existing causal loss inside the output projection's forward.

    The head continues to own ``weight`` at its original qualified name. The
    loss callable is supplied per invocation because OpSlots are shared by
    multiple model instances. Auxiliary tensors are returned as individual
    tuple elements so FSDP can register backward hooks on them.
    """

    def forward(
        self,
        input,
        *,
        _veomni_loss_fn=None,
        _veomni_labels=None,
        _veomni_vocab_size=None,
        _veomni_use_fused_loss=False,
        **kwargs,
    ):
        if _veomni_loss_fn is None:
            if kwargs:
                raise TypeError(f"Unexpected output projection arguments: {tuple(kwargs)}")
            return super().forward(input)

        logits = None if _veomni_use_fused_loss else super().forward(input)
        loss, loss_logits, aux = _veomni_loss_fn(
            logits=logits,
            labels=_veomni_labels,
            vocab_size=_veomni_vocab_size,
            hidden_states=input,
            weights=self.weight,
            **kwargs,
        )
        if not _veomni_use_fused_loss:
            loss_logits = None if aux is not None else logits
        aux_tensors = (
            (aux.log_probs, aux.entropy, aux.distillation_losses, aux.student_mass, aux.teacher_mass)
            if aux is not None
            else (None,) * 5
        )
        return loss, loss_logits, *aux_tensors


def enable_qwen3_moe_fused_head(model):
    """Enable the separately wrapped head only for the patched, unshared model."""
    model_type = type(model)
    if model_type.__name__ != "Qwen3MoeForCausalLM" or not model_type.__module__.startswith(
        "veomni.models.transformers.qwen3_moe.generated."
    ):
        return None
    head = model.get_output_embeddings()
    if type(head) is not nn.Linear or head.bias is not None or getattr(model, "lm_head", None) is not head:
        return None
    names = [name for name, param in model.named_parameters(remove_duplicate=False) if param is head.weight]
    if names != ["lm_head.weight"]:
        return None
    if any(
        getattr(head, name, None)
        for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
    ):
        return None
    # Preserve the original module, parameter, state-dict name and any external
    # references. FSDP subsequently adds its own dynamic subclass as usual.
    head.__class__ = FusedCrossEntropy
    return head
