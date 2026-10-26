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

"""Thin trainer subclasses; native initialization, data and loops are inherited."""

from . import model_runtime


def build_trainer(task, args):
    """Import only the selected modality and override its construction methods.

    Classes are local so selecting VLM does not import DiT's optional video
    stack. They are ordinary subclasses, not replacements of native classes.
    """
    if task == "vlm":
        from ..trainer.vlm_trainer import VLMTrainer as NativeVLMTrainer

        class VLMTrainer(NativeVLMTrainer):
            def _build_model_runtime(self):
                return model_runtime.vlm_runtime(self.base.args.model, "base", train=self.base.args.train)

        return VLMTrainer(args)

    if task == "dit":
        from ..trainer.dit_trainer import DiTTrainer as NativeDiTTrainer

        class DiTTrainer(NativeDiTTrainer):
            def _build_model_runtime(self):
                return model_runtime.dit_runtime(self.base.args.model, "base", train=self.base.args.train)

        return DiTTrainer(args)

    if task == "dpo":
        from ..trainer.text_dpo_trainer import TextDPOTrainer as NativeTextDPOTrainer

        class TextDPOTrainer(NativeTextDPOTrainer):
            def _build_policy_model_runtime(self):
                return model_runtime.text_runtime(self.base.args.model, "policy", train=self.base.args.train)

            def _build_reference_model_runtime(self):
                args = self.base.args
                return model_runtime.reference_runtime(
                    args.model, "reference", train=args.train, torch_dtype=args.dpo_config.refer_model_precision
                )

        return TextDPOTrainer(args)

    raise ValueError(f"Unknown Hyper task: {task!r}.")
