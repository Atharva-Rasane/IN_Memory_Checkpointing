#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NNODES=2
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT=29500
NPROC_PER_NODE=1
CHECKPOINT_MOUNT="/mnt/checkpoint-ram"
CHECKPOINT_DIR="$CHECKPOINT_MOUNT/basic-example"
CHECKPOINT_SIZE="8G"
DRIVER_PACKAGE="nvidia-open"
NCCL_INTERFACE=""

usage() {
    cat <<'EOF'
Usage:
  bash setup.sh --node-rank RANK --master-addr NODE0_PRIVATE_IP [options]

Required:
  --node-rank RANK          This VM's zero-based rank (0 or 1 for a two-VM test)
  --master-addr ADDRESS     Node 0's reachable private IP or hostname

Options:
  --nnodes COUNT            Number of VMs (default: 2)
  --nproc-per-node COUNT    Worker processes/GPUs per VM (default: 1)
  --master-port PORT        PyTorch rendezvous TCP port (default: 29500)
  --checkpoint-size SIZE    tmpfs size limit (default: 8G)
  --nccl-interface NAME     Network interface NCCL should use, such as eth0
  --driver-package PACKAGE  nvidia-open, cuda-drivers, or skip (default: nvidia-open)
  -h, --help                Show this help
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-rank)
            require_value "$@"
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            require_value "$@"
            MASTER_ADDR="$2"
            shift 2
            ;;
        --nnodes)
            require_value "$@"
            NNODES="$2"
            shift 2
            ;;
        --nproc-per-node)
            require_value "$@"
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --master-port)
            require_value "$@"
            MASTER_PORT="$2"
            shift 2
            ;;
        --checkpoint-size)
            require_value "$@"
            CHECKPOINT_SIZE="$2"
            shift 2
            ;;
        --nccl-interface)
            require_value "$@"
            NCCL_INTERFACE="$2"
            shift 2
            ;;
        --driver-package)
            require_value "$@"
            DRIVER_PACKAGE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ $EUID -ne 0 ]] || die "Run this script as the normal VM login user, not root. It invokes sudo when needed."

[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer."
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "--node-rank is required and must be a non-negative integer."
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be a positive integer."
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || die "MASTER_PORT must be a positive integer."
(( NODE_RANK < NNODES )) || die "NODE_RANK must be less than NNODES."
(( MASTER_PORT <= 65535 )) || die "MASTER_PORT must be at most 65535."
[[ -n "$MASTER_ADDR" ]] || die "--master-addr is required."
[[ "$CHECKPOINT_SIZE" =~ ^[1-9][0-9]*[KMGT]?$ ]] || die "Checkpoint size must look like 8G or 512M."

if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
    die "Use node 0's reachable private IP for --master-addr."
fi

case "$DRIVER_PACKAGE" in
    nvidia-open|cuda-drivers|skip) ;;
    *) die "--driver-package must be nvidia-open, cuda-drivers, or skip." ;;
esac

[[ -r /etc/os-release ]] || die "Cannot identify the operating system."
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] ||
    die "This setup script supports Ubuntu 24.04. Detected: ${PRETTY_NAME:-unknown}."

case "$(uname -m)" in
    x86_64)
        CUDA_REPO_ARCH="x86_64"
        ;;
    aarch64|arm64)
        CUDA_REPO_ARCH="sbsa"
        ;;
    *)
        die "Unsupported architecture: $(uname -m)"
        ;;
esac

command -v sudo >/dev/null 2>&1 || die "sudo is required."
sudo -v

echo "Installing Ubuntu prerequisites..."
sudo env DEBIAN_FRONTEND=noninteractive apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    git \
    "linux-headers-$(uname -r)" \
    python3.12 \
    python3.12-venv \
    util-linux \
    wget

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
CUDA_KEYRING="$TEMP_DIR/cuda-keyring.deb"
CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/$CUDA_REPO_ARCH"

echo "Configuring NVIDIA's Ubuntu 24.04 package repository..."
wget -qO "$CUDA_KEYRING" "$CUDA_REPO/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i "$CUDA_KEYRING"
sudo env DEBIAN_FRONTEND=noninteractive apt-get update

NVIDIA_PACKAGES=(cuda-toolkit-12-8)
if [[ "$DRIVER_PACKAGE" != "skip" ]]; then
    NVIDIA_PACKAGES+=("$DRIVER_PACKAGE")
fi

echo "Installing NVIDIA packages: ${NVIDIA_PACKAGES[*]}"
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "${NVIDIA_PACKAGES[@]}"

echo "Creating Python virtual environment and installing dependencies..."
if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    python3.12 -m venv "$SCRIPT_DIR/.venv"
fi
"$SCRIPT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$SCRIPT_DIR/.venv/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Configuring local tmpfs checkpoint mount and persistent /etc/fstab entry..."
sudo mkdir -p "$CHECKPOINT_MOUNT"

if mountpoint -q "$CHECKPOINT_MOUNT"; then
    MOUNT_TYPE="$(findmnt -n -o FSTYPE --target "$CHECKPOINT_MOUNT")"
    [[ "$MOUNT_TYPE" == "tmpfs" ]] ||
        die "$CHECKPOINT_MOUNT is already mounted as $MOUNT_TYPE; expected tmpfs."
fi

USER_UID="$(id -u)"
USER_GID="$(id -g)"
FSTAB_MARKER="# inmemory-checkpointing local tmpfs"
FSTAB_LINE="tmpfs $CHECKPOINT_MOUNT tmpfs rw,nosuid,nodev,size=$CHECKPOINT_SIZE,mode=0770,uid=$USER_UID,gid=$USER_GID 0 0"
NEW_FSTAB="$TEMP_DIR/fstab"

awk -v marker="$FSTAB_MARKER" -v mount_path="$CHECKPOINT_MOUNT" -v line="$FSTAB_LINE" '
    skip_next { skip_next = 0; next }
    $0 == marker { skip_next = 1; next }
    $2 == mount_path && $3 == "tmpfs" { next }
    { print }
    END {
        print marker
        print line
    }
' /etc/fstab > "$NEW_FSTAB"

sudo install -m 0644 "$NEW_FSTAB" /etc/fstab
if mountpoint -q "$CHECKPOINT_MOUNT"; then
    sudo mount -o remount "$CHECKPOINT_MOUNT"
else
    sudo mount "$CHECKPOINT_MOUNT"
fi
sudo chown "$USER_UID:$USER_GID" "$CHECKPOINT_MOUNT"
mkdir -p "$CHECKPOINT_DIR"

echo "Writing this VM's cluster environment..."
{
    echo "# Generated by setup.sh. Run setup.sh again to change these values."
    printf 'export PYTHON=%q\n' "$SCRIPT_DIR/.venv/bin/python"
    printf 'export NNODES=%q\n' "$NNODES"
    printf 'export NODE_RANK=%q\n' "$NODE_RANK"
    printf 'export NPROC_PER_NODE=%q\n' "$NPROC_PER_NODE"
    printf 'export MASTER_ADDR=%q\n' "$MASTER_ADDR"
    printf 'export MASTER_PORT=%q\n' "$MASTER_PORT"
    printf 'export CHECKPOINT_DIR=%q\n' "$CHECKPOINT_DIR"
    printf 'export CUDA_HOME=%q\n' "/usr/local/cuda-12.8"
    printf 'export NCCL_DEBUG=%q\n' "WARN"
    if [[ -n "$NCCL_INTERFACE" ]]; then
        printf 'export NCCL_SOCKET_IFNAME=%q\n' "$NCCL_INTERFACE"
    fi
} > "$SCRIPT_DIR/.cluster.env"
chmod 0600 "$SCRIPT_DIR/.cluster.env"
chmod +x "$SCRIPT_DIR/launch.sh" "$SCRIPT_DIR/setup.sh"

echo "Verifying Python packages..."
"$SCRIPT_DIR/.venv/bin/python" - <<'PY'
from importlib.metadata import version
import torch

print("PyTorch:", torch.__version__)
print("NVRx:", version("nvidia-resiliency-ext"))
print("CUDA runtime:", torch.version.cuda)
print("NCCL:", torch.cuda.nccl.version())
PY

echo
echo "Cluster environment: $SCRIPT_DIR/.cluster.env"
echo "Checkpoint mount:    $CHECKPOINT_MOUNT"
echo "Checkpoint path:     $CHECKPOINT_DIR"
echo

if nvidia-smi >/dev/null 2>&1 &&
    "$SCRIPT_DIR/.venv/bin/python" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "Setup complete. Run the same launch command on both VMs:"
    echo "  ./launch.sh --replication --replication_jump 1 --replication_factor 2"
else
    echo "The packages are installed, but the GPU driver is not active yet." >&2
    echo "Reboot this VM, then rerun the same setup.sh command to verify the GPU." >&2
    exit 2
fi
