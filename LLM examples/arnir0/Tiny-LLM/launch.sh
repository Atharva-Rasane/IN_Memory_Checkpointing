#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CLUSTER_ENV_FILE="${CLUSTER_ENV_FILE:-$REPO_ROOT/.cluster.env}"

if [[ -f "$CLUSTER_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$CLUSTER_ENV_FILE"
    set +a
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python was not found. Run setup.sh or set PYTHON." >&2
    exit 1
fi

PACKAGE_CHECK="import torch, datasets, transformers, nvidia_resiliency_ext"
if ! "$PYTHON_BIN" -c "$PACKAGE_CHECK" >/dev/null 2>&1; then
    echo "Required packages are missing. Run: $PYTHON_BIN -m pip install -r requirements.txt" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "PyTorch cannot access an NVIDIA GPU." >&2
    exit 1
fi

GPU_COUNT="$("$PYTHON_BIN" -c "import torch; print(torch.cuda.device_count())")"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$GPU_COUNT}"
LLM_CHECKPOINT_DIR="${LLM_CHECKPOINT_DIR:-/mnt/checkpoint-ram/tiny-llm}"
LLM_METRICS_DIR="${LLM_METRICS_DIR:-$SCRIPT_DIR/logs/metrics}"

echo "Launching Tiny-LLM:"
echo "  nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NPROC_PER_NODE"
echo "  master=$MASTER_ADDR:$MASTER_PORT"

TRAIN_ARGS=("$@")
has_arg() {
    local target="$1"
    local arg
    for arg in "${TRAIN_ARGS[@]}"; do
        if [[ "$arg" == "$target" || "$arg" == "$target="* ]]; then
            return 0
        fi
    done
    return 1
}

if ! has_arg "--ckpt_dir"; then
    TRAIN_ARGS=(--ckpt_dir "$LLM_CHECKPOINT_DIR" "${TRAIN_ARGS[@]}")
fi
if ! has_arg "--metrics_dir"; then
    TRAIN_ARGS=(--metrics_dir "$LLM_METRICS_DIR" "${TRAIN_ARGS[@]}")
fi

exec "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    "$SCRIPT_DIR/train.py" \
    "${TRAIN_ARGS[@]}"
