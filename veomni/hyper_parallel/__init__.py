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

"""Optional HyperParallel integration; no import-time backend activation."""

HYPER_MARKER_FILENAME = ".hyper_complete"


def _hyper_checkpoint_path(output_dir):
    """Discover completed Hyper checkpoints without importing the execution backend."""
    import json
    import os

    from veomni.checkpoint.layout import discover_modules, model_dir, read_manifest
    from veomni.utils.checkpoint_utils import _GLOBAL_STEP_PREFIX

    root = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(root):
        return None
    completed = []
    for entry in os.listdir(root):
        if not entry.startswith(_GLOBAL_STEP_PREFIX) or not entry[len(_GLOBAL_STEP_PREFIX) :].isdigit():
            continue
        path = os.path.join(root, entry)
        if not os.path.isdir(path) or not isinstance(read_manifest(path), dict):
            continue
        for module in discover_modules(path):
            try:
                with open(os.path.join(model_dir(path, module), HYPER_MARKER_FILENAME)) as handle:
                    marker = json.load(handle)
                if not isinstance(marker, dict) or marker.get("format_version") != 1:
                    break
            except (OSError, ValueError):
                break
        else:
            completed.append((int(entry[len(_GLOBAL_STEP_PREFIX) :]), path))
    return max(completed)[1] if completed else None


def _uses_hyper(args, *, global_accelerator=None):
    accelerator = getattr(args, "accelerator", None)
    if global_accelerator is not None and global_accelerator.fsdp_config.fsdp_scope == "model":
        accelerator = global_accelerator
    fsdp = getattr(accelerator, "fsdp_config", None)
    return getattr(fsdp, "fsdp_backend", "torch") == "hyper" or bool(
        getattr(getattr(args, "optimizer", None), "use_hyper_optimizer", False)
    )
