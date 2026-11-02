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

"""Internal Hyper wrapping boundaries and two-stage prefetch schedule."""

_COMPUTE_BOUNDARIES = {
    "Qwen3MoeDecoderLayer": ("self_attn", "mlp"),
    "Qwen3VLMoeTextDecoderLayer": ("self_attn", "mlp"),
    "Qwen3VLMoeVisionBlock": ("attn", "mlp"),
}


def resolve_submodule_wrap_targets(target_modules, *, root_model, device_type, compiled=False):
    """Use validated compute boundaries; leave unknown or shared blocks intact."""
    if device_type not in ("cuda", "npu") or compiled:
        return {}
    parameter_names = {}
    for name, param in root_model.named_parameters(remove_duplicate=False):
        parameter_names.setdefault(id(param), []).append(name)
    resolved = {}
    for fqn, block in target_modules:
        paths = _COMPUTE_BOUNDARIES.get(type(block).__name__)
        if paths is None or any(not hasattr(block, path) for path in paths):
            continue
        children = [block.get_submodule(path) for path in paths]
        seen = set()
        for path, child in zip(paths, children):
            owned = {id(param) for param in child.parameters()}
            prefix = (fqn + "." if fqn else "") + path + "."
            if (
                not owned
                or seen & owned
                or any(not name.startswith(prefix) for param in owned for name in parameter_names[param])
            ):
                break
            seen.update(owned)
        else:
            resolved[block] = children
    return resolved


def build_submodule_prefetch_stages(block, children, existing_units):
    """Keep EP and FP32 units with their corresponding compute stage."""
    stages = [[child] for child in children]
    for unit in existing_units:
        if unit is block or any(unit is child for child in children):
            continue
        stage = stages[0]
        for child, candidate in zip(children, stages):
            if any(unit is descendant for descendant in child.modules()):
                stage = candidate
                break
        stage.append(unit)
    return stages


def configure_submodule_prefetch(block_stages):
    """Prefetch two stages ahead, including before whole-block backward replay."""
    entries = [stage for block in block_stages for stage in block]
    position = 0
    for index, stages in enumerate(block_stages):
        for stage_index, stage in enumerate(stages):
            forward = [unit for following in entries[position + 1 : position + 3] for unit in following]
            # Whole-block replay starts at the last stage's backward hook, so
            # launch the preceding block's AGs there before this block's RS.
            if stage_index and (not index or len(block_stages[index - 1]) == 1):
                backward = stages[stage_index - 1]
            elif index:
                preceding = block_stages[index - 1]
                backward = [unit for previous in preceding for unit in previous]
            else:
                backward = []
            for unit in stage:
                unit.set_modules_to_forward_prefetch(
                    [target for target in dict.fromkeys(forward) if target is not unit]
                )
                unit.set_modules_to_backward_prefetch(list(backward))
            position += 1
