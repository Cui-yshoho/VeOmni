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

"""Resolve FSDP APIs without duplicating VeOmni's model traversal."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FSDP2Backend:
    """Backend-specific primitives used by the shared FSDP2 builder."""

    name: str
    fully_shard: Callable
    module_type: type
    mixed_precision_policy: type
    cpu_offload_policy: type
    shard: type
    mesh_converter: Callable | None = None
    distribute_tensor: Callable | None = None

    def to_mesh(self, mesh, name):
        return self.mesh_converter(mesh, name) if self.mesh_converter is not None else mesh

    def root_kwargs(self, options):
        # Torch infers root no-reshard when omitted; Hyper needs explicit False.
        result = {key: value for key, value in options.items() if key != "reshard_after_forward"}
        if self.name == "hyper":
            result["reshard_after_forward"] = False
        return result

    def weight_load_kwargs(self):
        return {"dtensor_factory": self.distribute_tensor} if self.distribute_tensor is not None else {}


def get_fsdp2_backend(name: str) -> FSDP2Backend:
    """Load only the selected backend; Torch never imports HyperParallel."""
    if name == "torch":
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed._tensor import Shard
        from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule

        return FSDP2Backend("torch", fully_shard, FSDPModule, MixedPrecisionPolicy, CPUOffloadPolicy, Shard)
    if name == "hyper":
        from ..hyper_parallelize import get_hyper_fsdp2_backend

        return get_hyper_fsdp2_backend()
    raise ValueError(f"Unsupported fsdp_backend={name!r}; expected 'torch' or 'hyper'.")
