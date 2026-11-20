#!/usr/bin/env bash
# Run on any host in HOSTS, outside the training container.
# Requires key-based SSH from the launching host to the other three hosts (default root:22).
# Ensures /dev/shm is at least 256 GiB in all four containers before launching training.
# Usage: bash start_veomni_64die.sh
# Optional NPU port allocation: HCCL_NPU_SOCKET_PORT_RANGE=auto bash start_veomni_64die.sh
set -euo pipefail

HOSTS=(192.168.0.191 192.168.0.213 192.168.0.233 192.168.0.245)
MASTER_ADDR=${HOSTS[0]}
# Resolve Docker sockets by IP; other hosts use the default Docker daemon.
DOCKER_HOSTS=()
for host in "${HOSTS[@]}"; do
  case "$host" in
    192.168.0.30|192.168.0.138|192.168.0.169|192.168.0.172|192.168.0.191)
      DOCKER_HOSTS+=(unix:///run/veomni-qwen38-128-20261114/docker.sock)
      ;;
    192.168.0.233|192.168.0.213|192.168.0.231)
      DOCKER_HOSTS+=(unix:///run/veomni-qwen38-candidates-20261115/docker.sock)
      ;;
    *)
      DOCKER_HOSTS+=("")
      ;;
  esac
done
CONTAINER_NAME=veomni-qwen38-hyper128-20261114
CODE_DIR=/workspace/envs/1116/VeOmni
VIDEO_ENV=/workspace/envs/veomni-video-runtime/env.sh
# Change only this experiment name when switching YAML files in the same directory.
EXPERIMENT_NAME=junyuan_torch_fsdp2_adamw_64die_8k_deterministic
CONFIG="configs/multimodal/qwen3_5_moe/${EXPERIMENT_NAME}.yaml"
SSH_USER=${SSH_USER:-root}
SSH_PORT=${SSH_PORT:-22}
SSH_OPTIONS=(-p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

if ! command -v ssh >/dev/null; then
  echo "ERROR: SSH client is required on the launching host." >&2
  exit 1
fi

read -r -a local_ips <<< "$(hostname -I)"
local_rank=-1
for rank in "${!HOSTS[@]}"; do
  for address in "${local_ips[@]}"; do
    if [[ "$address" == "${HOSTS[$rank]}" ]]; then
      local_rank=$rank
      break 2
    fi
  done
done
if [[ "$local_rank" == -1 ]]; then
  echo "ERROR: this host has none of the configured IP addresses: ${HOSTS[*]}." >&2
  exit 1
fi
echo "Launching from node_${local_rank}: ${HOSTS[$local_rank]}"

run_node() {
  local phase=$1 rank=$2 remote_command
  local -a transport
  local -a arguments=(
    "$phase" "$rank" "${DOCKER_HOSTS[$rank]}" "$CONTAINER_NAME"
    "$CODE_DIR" "$VIDEO_ENV" "$CONFIG" "${HCCL_NPU_SOCKET_PORT_RANGE:-}" "$MASTER_ADDR"
  )
  if [[ "$rank" == "$local_rank" ]]; then
    transport=(bash -s -- "${arguments[@]}")
  else
    printf -v remote_command '%q ' bash -s -- "${arguments[@]}"
    transport=(ssh "${SSH_OPTIONS[@]}" "$SSH_USER@${HOSTS[$rank]}" "$remote_command")
  fi

  "${transport[@]}" <<'REMOTE'
set -euo pipefail
phase=$1
rank=$2
if [[ -n "$3" ]]; then
  export DOCKER_HOST=$3
else
  unset DOCKER_HOST
fi
container=$4
code_dir=$5
video_env=$6
config=$7
hccl_ports=$8
master_addr=$9

if ! command -v docker >/dev/null; then
  echo "ERROR: docker command is unavailable on host node_${rank}; run this script on a configured host, outside the container." >&2
  exit 1
fi

if [[ "$phase" == check ]]; then
  if ! running=$(docker inspect --format '{{.State.Running}}' "$container"); then
    echo "ERROR: cannot inspect $container using ${DOCKER_HOST:-default Docker daemon} on node_${rank}." >&2
    exit 1
  fi
  if [[ "$running" != true ]]; then
    echo "ERROR: training container is not running: $container (${DOCKER_HOST:-default Docker daemon})" >&2
    exit 1
  fi
  docker exec "$container" bash -lc '
    set -e
    cd "$1"
    test -f train.sh
    test -f tasks/train_vlm.py
    test -f "$3"
    source "$2"
    command -v python >/dev/null
    command -v torchrun >/dev/null

    shm_capacity_bytes() {
      local block_size block_count
      read -r block_size block_count < <(stat -f -c "%S %b" /dev/shm) || return 1
      printf "%s\n" "$((block_size * block_count))"
    }
    shm_target_bytes=$((256 * 1024 * 1024 * 1024))
    shm_bytes=$(shm_capacity_bytes)
    if (( shm_bytes < shm_target_bytes )); then
      echo "node_${4}: expanding /dev/shm to 256 GiB..."
      if ! mount -o remount,size=256G /dev/shm; then
        echo "ERROR: node_${4}: failed to resize /dev/shm; container mount privileges are required." >&2
        exit 1
      fi
      shm_bytes=$(shm_capacity_bytes)
    fi
    if (( shm_bytes < shm_target_bytes )); then
      echo "ERROR: node_${4}: /dev/shm capacity is still below 256 GiB ($shm_bytes bytes)." >&2
      exit 1
    fi
    echo "node_${4}: /dev/shm capacity verified (at least 256 GiB)."
    df -h /dev/shm
  ' -- "$code_dir" "$video_env" "$config" "$rank"
  exit 0
fi

docker exec -d "$container" bash -lc '
  set -e
  rank=$1
  code_dir=$2
  video_env=$3
  config=$4
  hccl_ports=$5
  master_addr=$6

  cd "$code_dir"
  experiment_name=${config##*/}
  log_dir="$PWD/outputs/${experiment_name%.yaml}/logs"
  mkdir -p "$log_dir"
  exec >"$log_dir/node_${rank}.launcher.log" 2>&1
  echo "Launching node_${rank} on $(hostname) at $(date -Is)"

  export CPU_AFFINITY_CONF=2
  export TASK_QUEUE_ENABLE=2
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

  export NODE_RANK="$rank"
  export TMPDIR="/tmp/veomni_tmp/node_${NODE_RANK}"
  export TEMP="$TMPDIR"
  export TMP="$TMPDIR"
  mkdir -p "$TMPDIR"

  export MASTER_ADDR="$master_addr"
  export MASTER_PORT=26600
  export NNODES=4
  export NPROC_PER_NODE=16

  source "$video_env"
  export LOG_FILE="$log_dir/node_${NODE_RANK}.log"
  if [[ -n "$hccl_ports" ]]; then
    export HCCL_NPU_SOCKET_PORT_RANGE="$hccl_ports"
  fi

  exec bash train.sh tasks/train_vlm.py "$config"
' -- "$rank" "$code_dir" "$video_env" "$config" "$hccl_ports" "$master_addr"
REMOTE
}

echo "Checking all four hosts and training containers..."
for rank in "${!HOSTS[@]}"; do
  echo "Checking node_${rank}: ${HOSTS[$rank]}"
  if ! run_node check "$rank"; then
    echo "Preflight failed on node_${rank}; no new training was launched." >&2
    echo "For SSH failures, configure key-based SSH from the launching host to the other three hosts." >&2
    exit 1
  fi
done

echo "Starting four nodes in parallel..."
launch_pids=()
for rank in "${!HOSTS[@]}"; do
  run_node start "$rank" &
  launch_pids+=("$!")
done

failed=0
for rank in "${!HOSTS[@]}"; do
  if wait "${launch_pids[$rank]}"; then
    echo "node_${rank} (${HOSTS[$rank]}): background launch request succeeded."
  else
    echo "node_${rank} (${HOSTS[$rank]}): launch failed; other nodes may already be running." >&2
    failed=1
  fi
done

if [[ "$failed" != 0 ]]; then
  exit 1
fi
echo "Training runs inside the containers after this script exits."
echo "Container log directory: $CODE_DIR/outputs/$EXPERIMENT_NAME/logs"
echo "Training logs: node_0.log through node_3.log (overwritten on launch)."
echo "Startup errors: node_0.launcher.log through node_3.launcher.log."
echo "Launch success does not indicate that distributed initialization or training has completed."
