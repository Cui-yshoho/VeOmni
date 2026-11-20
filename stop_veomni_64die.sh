#!/usr/bin/env bash
# Run on master host 192.168.0.233. Uses the same hosts/containers as start_veomni_64die.sh.
# Usage: bash stop_veomni_64die.sh [--dry-run]
set -euo pipefail

HOSTS=(192.168.0.233 192.168.0.231 192.168.0.138 192.168.0.169)
DOCKER_HOSTS=(
  unix:///run/veomni-qwen38-candidates-20261115/docker.sock
  unix:///run/veomni-qwen38-candidates-20261115/docker.sock
  unix:///run/veomni-qwen38-128-20261114/docker.sock
  unix:///run/veomni-qwen38-128-20261114/docker.sock
)
CONTAINER_NAME=veomni-qwen38-hyper128-20261114
CODE_DIR=/workspace/envs/1116/VeOmni
SSH_USER=${SSH_USER:-root}
SSH_PORT=${SSH_PORT:-22}
GRACE_SECONDS=${GRACE_SECONDS:-10}
SSH_OPTIONS=(-p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

dry_run=0
case "${1:-}" in
  '') ;;
  --dry-run) dry_run=1 ;;
  -h|--help)
    echo "Usage: bash stop_veomni_64die.sh [--dry-run]"
    echo "Stop Qwen3.5 VeOmni VLM tasks in the four designated containers."
    exit 0
    ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if [[ "$#" -gt 1 ]]; then
  echo "Usage: bash stop_veomni_64die.sh [--dry-run]" >&2
  exit 2
fi

stop_node() {
  local rank=$1 remote_command
  local -a transport
  local -a arguments=(
    "${DOCKER_HOSTS[$rank]}" "$CONTAINER_NAME" "$CODE_DIR"
    "$dry_run" "$GRACE_SECONDS" "node_${rank} (${HOSTS[$rank]})"
  )
  if [[ "$rank" == 0 ]]; then
    transport=(bash -s -- "${arguments[@]}")
  else
    printf -v remote_command '%q ' bash -s -- "${arguments[@]}"
    transport=(ssh "${SSH_OPTIONS[@]}" "$SSH_USER@${HOSTS[$rank]}" "$remote_command")
  fi

  "${transport[@]}" <<'REMOTE'
set -euo pipefail
export DOCKER_HOST=$1
container=$2
running=$(docker inspect --format '{{.State.Running}}' "$container")
if [[ "$running" != true ]]; then
  echo "[$6] Container is not running; nothing to stop."
  exit 0
fi

docker exec -i "$container" bash -lc 'exec python - "$@"' -- "$3" "$4" "$5" "$6" <<'PY'
import math
import os
from pathlib import Path
import signal
import sys
import time

code_dir, dry_run, grace_seconds, label = sys.argv[1:]
code_dir = os.path.realpath(code_dir)
grace_seconds = float(grace_seconds)
if not math.isfinite(grace_seconds) or grace_seconds < 0:
    raise SystemExit("GRACE_SECONDS must be a finite non-negative number")
self_cgroup = Path("/proc/self/cgroup").read_bytes()


def announce(message):
    print(f"[{label}] {message}", flush=True)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return int(fields[1]), fields[19], fields[0]
    except (OSError, ValueError, IndexError):
        return None


def snapshot():
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if (entry / "cgroup").read_bytes() != self_cgroup:
                continue
            current = identity(pid)
            if current is None:
                continue
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
            cwd = os.path.realpath(entry / "cwd")
            result[pid] = {"identity": current, "argv": argv, "cwd": cwd}
        except OSError:
            continue
    return result


def is_training_task(pid, info):
    argv = info["argv"]
    in_code_dir = info["cwd"] == code_dir or info["cwd"].startswith(code_dir + "/")
    entry_names = {"train_vlm.py", "veomni_debug_dataloader.py"}
    explicit_entry = any(arg.rsplit("/", 1)[-1] in entry_names for arg in argv)
    qwen_config = any("qwen3_5_moe" in arg and arg.endswith((".yaml", ".yml")) for arg in argv)
    # Old training launchers may still run from an earlier checkout in this container.
    if explicit_entry and qwen_config:
        return True
    # Renamed/orphaned workers can retain the distributed job's environment.
    if in_code_dir:
        try:
            items = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            env = dict(item.split(b"=", 1) for item in items if b"=" in item)
            return (
                env.get(b"MASTER_ADDR") == b"192.168.0.233"
                and env.get(b"MASTER_PORT") == b"26600"
                and env.get(b"WORLD_SIZE") == b"64"
                and env.get(b"RANK", b"").isdigit()
                and int(env[b"RANK"]) < 64
            )
        except OSError:
            pass
    return False


processes = snapshot()
protected = {1, os.getpid()}
ancestor = os.getpid()
while ancestor in processes:
    ancestor = processes[ancestor]["identity"][0]
    if ancestor in protected:
        break
    protected.add(ancestor)

targets = {pid for pid, info in processes.items() if pid not in protected and is_training_task(pid, info)}
while True:
    children = {
        pid for pid, info in processes.items()
        if pid not in protected and info["identity"][0] in targets
    }
    expanded = targets | children
    if expanded == targets:
        break
    targets = expanded

if not targets:
    announce("No matching training processes found.")
    raise SystemExit(0)
for pid in sorted(targets):
    info = processes[pid]
    announce(f"PID={pid} PPID={info['identity'][0]} STAT={info['identity'][2]} {' '.join(info['argv'])[:400]}")
if dry_run == "1":
    announce(f"Dry run: {len(targets)} target processes; no signals sent.")
    raise SystemExit(0)


def survivors():
    remaining = {}
    for pid in targets:
        current = identity(pid)
        if current is not None and current[1] == processes[pid]["identity"][1]:
            remaining[pid] = current
    return remaining


def send_signal(pid, sig):
    current = identity(pid)
    if current is None or current[1] != processes[pid]["identity"][1] or current[2] == "Z":
        return
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        announce(f"Cannot signal PID={pid}: {error}")


announce(f"Sending TERM to {len(targets)} training processes.")
for pid in sorted(targets):
    send_signal(pid, signal.SIGTERM)
deadline = time.monotonic() + grace_seconds
while any(state[2] != "Z" for state in survivors().values()) and time.monotonic() < deadline:
    time.sleep(0.2)
remaining = survivors()
if any(state[2] != "Z" for state in remaining.values()):
    announce("Grace period expired; sending KILL to surviving target processes.")
    for pid in remaining:
        send_signal(pid, signal.SIGKILL)
    deadline = time.monotonic() + 3
    while any(state[2] != "Z" for state in survivors().values()) and time.monotonic() < deadline:
        time.sleep(0.2)
remaining = survivors()
active = {pid: state for pid, state in remaining.items() if state[2] != "Z"}
for pid, state in remaining.items():
    announce(f"Still present: PID={pid} STAT={state[2]}" + (" (exited; awaiting parent reap)" if state[2] == "Z" else ""))
if active:
    announce("Some target processes have not exited; inspect the listed states.")
    raise SystemExit(1)
announce("No live target processes remain. Device memory release must be checked with npu-smi info.")
PY
REMOTE
}

stop_pids=()
for rank in "${!HOSTS[@]}"; do
  stop_node "$rank" &
  stop_pids+=("$!")
done

failed=0
for rank in "${!HOSTS[@]}"; do
  if ! wait "${stop_pids[$rank]}"; then
    echo "node_${rank} (${HOSTS[$rank]}): stop/check failed; see errors above." >&2
    failed=1
  fi
done
exit "$failed"
