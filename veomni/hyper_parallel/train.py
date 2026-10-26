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

"""Explicit Hyper launcher; all training and inference loops remain native."""

import argparse
import sys
from importlib import import_module

from . import _uses_hyper


_TASKS = {
    "text": ("text_trainer", "TextTrainer", "VeOmniArguments"),
    "vlm": ("vlm_trainer", "VLMTrainer", "VeOmniVLMArguments"),
    "dit": ("dit_trainer", "DiTTrainer", "VeOmniDiTArguments"),
    "dpo": ("text_dpo_trainer", "TextDPOTrainer", "VeOmniDPOArguments"),
    "omni": ("omni.omni_trainer", "OmniTrainer", "OmniArguments"),
    "omni_infer": ("omni.omni_inferencer", "OmniInferencer", "OmniArguments"),
}


def build_trainer(task, args):
    """Choose a thin Hyper trainer only at the explicit entry."""
    if task in ("vlm", "dit", "dpo") and _uses_hyper(args.model):
        from .trainer import build_trainer as build_hyper_trainer

        return build_hyper_trainer(task, args)
    module_name, trainer_name, _ = _TASKS[task]
    trainer_type = getattr(import_module(f"..trainer.{module_name}", __package__), trainer_name)
    return trainer_type(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=_TASKS)
    task = parser.parse_args(sys.argv[1:2]).task
    # The existing argument parser consumes the config path and CLI overrides.
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if task in ("omni", "omni_infer"):
        from ..arguments.omni_arguments_types import OmniArguments
        from ..arguments.omni_parser import parse_omni_args

        args = parse_omni_args(OmniArguments, preload_path_fields=("model.model_config.modules",))
    else:
        from ..arguments import parse_args

        module_name, _, args_name = _TASKS[task]
        args_type = getattr(import_module(f"..trainer.{module_name}", __package__), args_name)
        args = parse_args(args_type)
    trainer = build_trainer(task, args)
    if task == "omni_infer":
        trainer.generate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
