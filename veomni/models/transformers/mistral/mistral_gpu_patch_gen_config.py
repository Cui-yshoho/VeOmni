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
"""
Patch configuration for Mistral GPU OpSlot-based kernel replacements.

Regen command:
patchgen veomni.models.transformers.mistral.mistral_gpu_patch_gen_config -o veomni/models/transformers/mistral/generated

Patches:
- OpSlot guards for RMSNorm, SwiGLU MLP, and RoPE. Each guard falls through to
  the original HF eager code when no fused kernel is bound, so the generated
  file is safe to import even when ``_bind_veomni_ops()`` does not run.

This file itself is not runnable — it is the declarative source of truth for
the runnable explicitly-patched modeling file
"generated/patched_modeling_mistral_gpu.py".
"""

import torch

from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.mistral.modeling_mistral",
    target_file="patched_modeling_mistral_gpu.py",
    description="Mistral with OpSlot-based GPU kernel replacements",
)


config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # These are bound at model-build time by _bind_veomni_ops().
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
    veomni_swiglu_mlp = OpSlot("swiglu_mlp", "standard")
    """
)


# ── RMSNorm (OpSlot guard, functional Liger kernel) ──────────────────────────


@config.override_method(
    "MistralRMSNorm.forward",
    description="OpSlot guard for Liger fused RMSNorm (standard formulation)",
)
def mistral_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Modification: OpSlot guard — use fused RMSNorm kernel when bound.
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)
    # Original HF code below, unchanged.
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


# ── SwiGLU MLP (OpSlot guard, functional Liger kernel) ───────────────────────


@config.override_method(
    "MistralMLP.forward",
    description="OpSlot guard for Liger fused SwiGLU MLP",
)
def mistral_mlp_forward_patched(self, x):
    # Modification: OpSlot guard — use fused SwiGLU kernel when bound.
    if veomni_swiglu_mlp.use_non_eager_impl:
        return veomni_swiglu_mlp(self, x)
    # Original HF code below, unchanged.
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


# ── Rotary Positional Embedding (OpSlot guard) ───────────────────────────────


@config.replace_function("apply_rotary_pos_emb", description="OpSlot guard for Liger fused RoPE")
def apply_rotary_pos_emb_patched(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Modification: OpSlot guard — use fused RoPE kernel when bound.
    if veomni_apply_rotary_pos_emb.use_non_eager_impl:
        return veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    # Original HF code below, unchanged.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
