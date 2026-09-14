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

"""Guarded NPU integration tests for the optional HyperParallel backend."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


HAS_HYPER_PARALLEL = importlib.util.find_spec("hyper_parallel") is not None
NPU_COUNT = torch.npu.device_count() if hasattr(torch, "npu") and torch.npu.is_available() else 0


def _run_worker(worker, world_size, *args):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        worker,
        *map(str, args),
    ]
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (repo_root, env.get("PYTHONPATH"))))
    subprocess.run(command, check=True, env=env, timeout=300)


@pytest.mark.skipif(not HAS_HYPER_PARALLEL or NPU_COUNT < 2, reason="requires HyperParallel and 2 NPUs")
def test_hyper_fsdp2_and_muon_smoke():
    _run_worker("tests/distributed/_hyper_parallel_worker.py", 2, "smoke", "--replicate-size", 1)


@pytest.mark.skipif(not HAS_HYPER_PARALLEL or NPU_COUNT < 2, reason="requires HyperParallel and 2 NPUs")
def test_hyper_fsdp2_and_native_muon_owner_path_smoke():
    _run_worker(
        "tests/distributed/_hyper_parallel_worker.py",
        2,
        "smoke",
        "--replicate-size",
        1,
        "--native-muon",
    )


@pytest.mark.skipif(not HAS_HYPER_PARALLEL or NPU_COUNT < 4, reason="requires HyperParallel and 4 NPUs")
def test_hyper_hsdp_and_muon_smoke():
    _run_worker("tests/distributed/_hyper_parallel_worker.py", 4, "smoke", "--replicate-size", 2)


@pytest.mark.skipif(not HAS_HYPER_PARALLEL or NPU_COUNT < 2, reason="requires HyperParallel and 2 NPUs")
def test_hyper_fsdp2_checkpoint_resume(tmp_path):
    _run_worker(
        "tests/distributed/_hyper_parallel_worker.py", 2, "checkpoint", "--checkpoint-path", tmp_path / "checkpoint"
    )


@pytest.mark.skipif(not HAS_HYPER_PARALLEL or NPU_COUNT < 4, reason="requires HyperParallel and 4 NPUs")
def test_hyper_extra_parallel_parity_and_checkpoint_resume(tmp_path):
    _run_worker(
        "tests/distributed/_hyper_parallel_worker.py",
        4,
        "extra_parallel",
        "--weights-path",
        tmp_path / "weights",
        "--checkpoint-path",
        tmp_path / "checkpoint",
    )
