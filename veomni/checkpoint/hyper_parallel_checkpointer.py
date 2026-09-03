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

"""Checkpoint adapter for models sharded by the optional HyperParallel backend."""

import base64
import os
import pickle
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from ..distributed.hyper_fsdp2 import refresh_materialized_shards
from ..optim.optimizer import restore_optimizer_param_group_defaults
from ..utils.checkpoint_utils import _GLOBAL_STEP_PREFIX, _HYPER_COMPLETION_MARKER
from ..utils.logging import get_logger
from .dcp_checkpointer import DistributedCheckpointer


logger = get_logger(__name__)


def _checkpoint_api():
    try:
        from hyper_parallel.core.distributed_checkpoint import load, save
    except ImportError as exc:
        raise RuntimeError("Hyper FSDP2 checkpointing requires the optional 'hyper_parallel' package.") from exc
    return save, load


_TENSOR_REFERENCE_KEY = "__veomni_hyper_optimizer_tensor__"


def _split_optimizer_state(value, tensors, key_prefix="tensor"):
    """Replace tensors with small references while preserving optimizer state structure."""
    if torch.is_tensor(value):
        key = f"{key_prefix}_{len(tensors)}"
        tensors[key] = value
        return {
            _TENSOR_REFERENCE_KEY: key,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
        }
    if isinstance(value, dict):
        return {key: _split_optimizer_state(item, tensors, key_prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_split_optimizer_state(item, tensors, key_prefix) for item in value]
    if isinstance(value, tuple):
        return tuple(_split_optimizer_state(item, tensors, key_prefix) for item in value)
    return value


def _is_tensor_reference(value) -> bool:
    return isinstance(value, dict) and set(value) == {_TENSOR_REFERENCE_KEY, "shape", "dtype"}


def _optimizer_tensor_placeholders(value, tensors) -> None:
    """Create rank-local CPU load targets from the tensor metadata in a manifest."""
    if _is_tensor_reference(value):
        dtype_name = value["dtype"]
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported optimizer checkpoint dtype: {dtype_name}.")
        tensors[value[_TENSOR_REFERENCE_KEY]] = torch.empty(value["shape"], dtype=dtype, device="cpu")
        return
    if isinstance(value, dict):
        for item in value.values():
            _optimizer_tensor_placeholders(item, tensors)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _optimizer_tensor_placeholders(item, tensors)


def _restore_optimizer_state(value, tensors):
    if _is_tensor_reference(value):
        return tensors[value[_TENSOR_REFERENCE_KEY]]
    if isinstance(value, dict):
        return {key: _restore_optimizer_state(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_optimizer_state(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_optimizer_state(item, tensors) for item in value)
    return value


def _encode_optimizer_manifest(manifest) -> str:
    """Encode tensor-free rank-local structure; tensor payloads remain in Hyper DCP."""
    return base64.b64encode(pickle.dumps(manifest, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _decode_optimizer_manifest(payload: str):
    # Checkpoints are trusted inputs, matching torch.load and VeOmni's existing checkpoint contract.
    return pickle.loads(base64.b64decode(payload))  # noqa: S301


class HyperParallelCheckpointer(DistributedCheckpointer):
    """Implement VeOmni's checkpoint contract with HyperParallel DCP."""

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: Optional[bool] = False,
        global_steps: Optional[int] = None,
        trainable_only: bool = False,
        save_to_lowest_rank: bool = False,
        parallel_state=None,
    ) -> None:
        del parallel_state
        if trainable_only:
            raise NotImplementedError("Trainable-only checkpoints are not supported by Hyper FSDP2 yet.")
        if save_to_lowest_rank:
            raise NotImplementedError("dcp_save_to_lowest_rank is not supported by Hyper FSDP2 yet.")
        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        checkpoint_dir = f"{path}/{_GLOBAL_STEP_PREFIX}{global_steps}" if global_steps else path
        hp_save, _ = _checkpoint_api()
        cls._create_checkpoint_dir(checkpoint_dir)
        cls._save_extra_state(checkpoint_dir=checkpoint_dir, state=state)

        model = state["model"]
        if "optimizer" in state:
            optimizer_tensors = {}
            optimizer_manifest = _split_optimizer_state(state["optimizer"].state_dict(), optimizer_tensors)
            hp_save(
                {"manifest": _encode_optimizer_manifest(optimizer_manifest)},
                checkpoint_id=f"{checkpoint_dir}/optimizer_manifest",
                use_collectives=False,
            )
            if optimizer_tensors:
                hp_save(
                    {"tensors": optimizer_tensors},
                    checkpoint_id=f"{checkpoint_dir}/optimizer",
                    use_collectives=False,
                )

        if save_async:
            logger.warning_rank0("HyperParallel async checkpoint persistence is not NPU-safe; saving synchronously.")
        hp_save({"model": model.state_dict()}, checkpoint_id=f"{checkpoint_dir}/model", use_collectives=True)
        # Hyper stores DCP metadata below component directories instead of at
        # the VeOmni global-step root. Publish a completion marker only after
        # every component save succeeds so ``load_path=auto`` can select this
        # checkpoint without mistaking a partially written step for a resume.
        if not dist.is_initialized() or dist.get_rank() == 0:
            marker_path = os.path.join(checkpoint_dir, _HYPER_COMPLETION_MARKER)
            temporary_marker = f"{marker_path}.tmp"
            with open(temporary_marker, "w", encoding="utf-8") as marker:
                marker.write("complete\n")
            os.replace(temporary_marker, marker_path)
        if dist.is_initialized():
            dist.barrier()
        logger.info_rank0(f"Saved HyperParallel checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        trainable_only: bool = False,
        parallel_state=None,
    ) -> Dict[str, Any]:
        del parallel_state
        if trainable_only:
            raise NotImplementedError("Trainable-only checkpoints are not supported by Hyper FSDP2 yet.")
        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        _, hp_load = _checkpoint_api()
        model = state["model"]
        model_state = model.state_dict()
        hp_load({"model": model_state}, checkpoint_id=f"{path}/model", use_collectives=True)
        model.load_state_dict(model_state)
        refresh_materialized_shards(model)

        if "optimizer" in state:
            optimizer = state["optimizer"]
            manifest_payload = {"manifest": ""}
            hp_load(manifest_payload, checkpoint_id=f"{path}/optimizer_manifest", use_collectives=False)
            optimizer_manifest = _decode_optimizer_manifest(manifest_payload["manifest"])
            optimizer_tensors = {}
            _optimizer_tensor_placeholders(optimizer_manifest, optimizer_tensors)
            if optimizer_tensors:
                tensor_payload = {"tensors": optimizer_tensors}
                hp_load(tensor_payload, checkpoint_id=f"{path}/optimizer", use_collectives=False)
                optimizer_tensors = tensor_payload["tensors"]
            optimizer.load_state_dict(_restore_optimizer_state(optimizer_manifest, optimizer_tensors))
            restore_optimizer_param_group_defaults(optimizer)

        if "extra_state" in state:
            cls._load_extra_state(checkpoint_dir=path, state=state)
        logger.info_rank0(f"Loaded HyperParallel checkpoint from {path}")
        return state
