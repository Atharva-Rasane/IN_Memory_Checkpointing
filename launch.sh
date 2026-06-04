#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLUSTER_ENV_FILE="${CLUSTER_ENV_FILE:-$SCRIPT_DIR/.cluster.env}"
if [[ -f "$CLUSTER_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$CLUSTER_ENV_FILE"
    set +a
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python was not found. Create .venv using the README instructions or set PYTHON." >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c "import torch, nvidia_resiliency_ext" >/dev/null 2>&1; then
    echo "Required Python packages are missing. Run: $PYTHON_BIN -m pip install -r requirements.txt" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "PyTorch cannot access an NVIDIA GPU. Check the VM GPU attachment and NVIDIA driver." >&2
    exit 1
fi

GPU_COUNT="$("$PYTHON_BIN" -c "import torch; print(torch.cuda.device_count())")"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$GPU_COUNT}"

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer; received: $value" >&2
        exit 1
    fi
}

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be a non-negative integer; received: $value" >&2
        exit 1
    fi
}

require_positive_integer "NNODES" "$NNODES"
require_positive_integer "NPROC_PER_NODE" "$NPROC_PER_NODE"
require_positive_integer "MASTER_PORT" "$MASTER_PORT"
require_nonnegative_integer "NODE_RANK" "$NODE_RANK"

if (( NODE_RANK >= NNODES )); then
    echo "NODE_RANK ($NODE_RANK) must be less than NNODES ($NNODES)." >&2
    exit 1
fi

if (( MASTER_PORT > 65535 )); then
    echo "MASTER_PORT must be at most 65535; received: $MASTER_PORT" >&2
    exit 1
fi

if (( NPROC_PER_NODE > GPU_COUNT )); then
    echo "NPROC_PER_NODE ($NPROC_PER_NODE) exceeds the visible GPU count ($GPU_COUNT)." >&2
    exit 1
fi

if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
    echo "Set MASTER_ADDR to node 0's reachable private IP when NNODES is greater than 1." >&2
    exit 1
fi

echo "Launching: nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NPROC_PER_NODE visible_gpus=$GPU_COUNT master=$MASTER_ADDR:$MASTER_PORT"

EXAMPLE_ARGS=("$@")
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    HAS_CHECKPOINT_ARG=false
    for arg in "${EXAMPLE_ARGS[@]}"; do
        if [[ "$arg" == "--ckpt_dir" || "$arg" == --ckpt_dir=* ]]; then
            HAS_CHECKPOINT_ARG=true
            break
        fi
    done

    if [[ "$HAS_CHECKPOINT_ARG" == false ]]; then
        EXAMPLE_ARGS=(--ckpt_dir "$CHECKPOINT_DIR" "${EXAMPLE_ARGS[@]}")
    fi
fi

exec "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    "$SCRIPT_DIR/BasicExample1.py" \
    "${EXAMPLE_ARGS[@]}"
