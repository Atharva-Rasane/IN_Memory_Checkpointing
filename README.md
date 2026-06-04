# In-Memory Checkpointing

This repository runs a small distributed training workload with PyTorch
DistributedDataParallel, NCCL, and NVIDIA Resiliency Extension local
checkpointing. The example requires NVIDIA GPUs and does not run on CPU-only
VMs.

The intended test environment is two Ubuntu 24.04 VMs with one NVIDIA GPU each.
`setup.sh` prepares each VM, and `launch.sh` starts the distributed test.

## Recommended OS and Versions

Use **Ubuntu 24.04**. It is the simplest supported choice because the published
`nvidia-resiliency-ext==0.6.0` wheel requires glibc 2.39, which Ubuntu 24.04
provides. NVIDIA also documents Ubuntu 22.04 support, but version 0.6.0 must be
built from source there.

The repository pins this compatible version set as of June 4, 2026:

| Component | Version |
| --- | --- |
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| NVIDIA driver | 570 or newer recommended |
| CUDA Toolkit | 12.8 |
| PyTorch | 2.11.0+cu128 |
| NCCL | 2.28.9, installed by the PyTorch wheel |
| NVIDIA Resiliency Extension | 0.6.0 |

## Two-VM Requirements

- Two Ubuntu 24.04 VMs with one NVIDIA GPU each.
- The same repository revision on both VMs.
- Private network connectivity between the VMs.
- TCP port `29500` allowed between the VMs for PyTorch rendezvous.
- Peer-to-peer traffic allowed between the VMs for NCCL. The simplest test
  security-group/firewall rule is to allow all private traffic between only
  these two VMs.
- SSH access configured separately by you or your cloud provider.

`setup.sh` does not configure SSH or cloud firewall/security-group rules.

## Clone or Update the Repository

Run on both VMs:

```bash
git clone https://github.com/Atharva-Rasane/IN_Memory_Checkpointing.git
cd IN_Memory_Checkpointing
```

For later updates:

```bash
git pull
```

## Prepare Both VMs

Choose node 0's reachable private IP. The examples below use `10.0.0.10`.

Run setup as the normal VM login user, not with `sudo`. The script invokes
`sudo` for system changes and performs these actions:

- Validates that the VM is running Ubuntu 24.04.
- Installs the NVIDIA open driver and CUDA Toolkit 12.8.
- Creates `.venv` and installs `requirements.txt`.
- Creates a local `tmpfs` mount at `/mnt/checkpoint-ram` and an `/etc/fstab`
  entry that recreates the mount after reboot.
- Uses `/mnt/checkpoint-ram/basic-example` as the checkpoint path.
- Writes this VM's rank and cluster values to `.cluster.env`.
- Verifies the Python packages and GPU.

On node 0:

```bash
bash setup.sh --node-rank 0 --master-addr 10.0.0.10
```

On node 1:

```bash
bash setup.sh --node-rank 1 --master-addr 10.0.0.10
```

The default setup assumes two VMs and one GPU/process per VM. If NCCL must use
a specific private network interface, add the same interface on both VMs:

```bash
bash setup.sh --node-rank 0 --master-addr 10.0.0.10 --nccl-interface eth0
```

The first run may report that the installed GPU driver is not active. If so,
reboot that VM and rerun the same `setup.sh` command:

```bash
sudo reboot
```

If the VM image already has a suitable NVIDIA driver, preserve it with:

```bash
bash setup.sh --node-rank 0 --master-addr 10.0.0.10 --driver-package skip
```

Use `--driver-package cuda-drivers` instead of the default `nvidia-open` only
when the GPU or cloud provider requires NVIDIA's proprietary kernel modules.

## Etcd and SSH

Etcd is intentionally not installed. This fixed two-VM test uses `torchrun`'s
static TCP rendezvous through node 0 at `MASTER_ADDR:MASTER_PORT`; it does not
use an etcd rendezvous backend. Adding etcd would create an unused service.

SSH is also separate from this repository. Configure SSH access manually, then
run the setup and launch commands from a terminal on each VM.

## Connect and Verify the Two VMs

After setup and any required reboot, connect to each VM and enter the
repository directory.

On both VMs, verify the local checkpoint mount and configured path:

```bash
findmnt /mnt/checkpoint-ram
df -h /mnt/checkpoint-ram
grep -E 'NODE_RANK|MASTER_ADDR|CHECKPOINT_DIR' .cluster.env
```

The checkpoint mount path is `/mnt/checkpoint-ram`, and the example checkpoint
directory is:

```text
/mnt/checkpoint-ram/basic-example
```

Each VM has its own separate local `tmpfs` at that path. Do not mount shared
NFS/network storage there for this local-checkpointing test. Checkpoint data in
`tmpfs` is intentionally volatile and is lost when the VM reboots.

`launch.sh` loads `.cluster.env` automatically, including the node rank,
master address, and checkpoint directory.

## Run the Two-VM Test

Start node 0 first:

```bash
./launch.sh --replication --replication_jump 1 --replication_factor 2
```

Then start node 1 with the same command:

```bash
./launch.sh --replication --replication_jump 1 --replication_factor 2
```

Node 0 waits until node 1 joins. Both nodes use
`/mnt/checkpoint-ram/basic-example` automatically.

A normal run starts from scratch and clears its configured local checkpoint
directory on both VMs before training. A run with `--resume` preserves the
directory and restores the latest checkpoint. The default run performs five
epochs with 20 distributed steps per epoch. Each rank processes a different
synthetic batch, DDP synchronizes gradients between both GPUs, and rank 0 logs
the global mean training loss.

At the end of every epoch, the example:

1. Saves the trained model and optimizer state to the RAM-backed checkpoint
   directory.
2. Replicates each checkpoint part between the two VMs.
3. Validates the new epoch checkpoint and removes the older checkpoint.

After the final epoch, the example loads the latest checkpoint into a new model
and optimizer and verifies that the restored model has the same validation
loss. NVIDIA's local checkpoint manager retains the latest valid epoch rather
than every historical epoch.

To make the checkpoint path explicit, the equivalent command on both VMs is:

```bash
./launch.sh \
  --ckpt_dir /mnt/checkpoint-ram/basic-example \
  --replication \
  --replication_jump 1 \
  --replication_factor 2
```

After every rank finishes loading, local rank 0 on each VM removes that VM's
checkpoint directory. Repeated test runs therefore start with an empty
checkpoint path.

Run a longer training test by passing the same arguments on both VMs:

```bash
./launch.sh \
  --epochs 20 \
  --steps_per_epoch 100 \
  --batch_size 1024 \
  --log_interval 25 \
  --replication \
  --replication_jump 1 \
  --replication_factor 2
```

## Measure Checkpoint Time and Test Recovery

The example measures checkpoint snapshot, save, finalize, and load operations
with CUDA synchronization so the reported wall-clock times include completed
GPU work. Rank 0 logs the global minimum, mean, and maximum duration across all
ranks. Every rank also appends structured metrics to:

```text
checkpoint_metrics/rank_<global-rank>.jsonl
```

These metrics are outside the RAM checkpoint directory, so they remain after a
successful run cleans up checkpoints and after an injected failure.

To simulate a failure after epoch 5 while training through epoch 10, run this
same command on both VMs:

```bash
./launch.sh \
  --epochs 10 \
  --steps_per_epoch 100 \
  --fail_epoch 5 \
  --failure_rank 0 \
  --keep_checkpoints \
  --replication \
  --replication_jump 1 \
  --replication_factor 2
```

The selected rank exits with status `42` only after every rank has finalized
the epoch-5 checkpoint. Peer ranks exit with status `43` at the same point so
no distributed workers remain blocked. The first launch is therefore expected
to fail.

Inspect the retained epoch-5 checkpoint and timing metrics on each VM:

```bash
find /mnt/checkpoint-ram/basic-example -type f -ls
cat checkpoint_metrics/rank_*.jsonl
```

Then run this same recovery command on both VMs:

```bash
./launch.sh \
  --resume \
  --epochs 10 \
  --steps_per_epoch 100 \
  --keep_checkpoints \
  --replication \
  --replication_jump 1 \
  --replication_factor 2
```

The recovery launch loads epoch 5, restores the model, optimizer, random-data
generator, and global step, then starts training at epoch 6. Do not reboot the
VMs between failure and recovery because the checkpoints are stored in
volatile `tmpfs`.

## Run on One VM

Prepare a single VM with:

```bash
bash setup.sh --nnodes 1 --node-rank 0 --master-addr 127.0.0.1
```

Run on every visible GPU configured by setup:

```bash
./launch.sh
```

Run the asynchronous-save path:

```bash
./launch.sh --async_save
```

With `--async_save`, each epoch's RAM checkpoint write overlaps the following
epoch's training. It is finalized before the next epoch checkpoint is
submitted.

## Tiny LLM Example

The `LLM examples/arnir0/Tiny-LLM` folder contains a separate Hugging Face
fine-tuning example for `arnir0/Tiny-LLM` on WikiText-2. It uses the same
two-VM `.cluster.env` values from `setup.sh`, but writes checkpoints to
`/mnt/checkpoint-ram/tiny-llm` by default.

Install the extra dependencies on both VMs after pulling this repository:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Run from `LLM examples/arnir0/Tiny-LLM` on both VMs:

```bash
./launch.sh \
  --epochs 10 \
  --steps_per_epoch 10 \
  --checkpoint_interval_steps 1 \
  --replication \
  --replication_jump 1 \
  --replication_factor 2
```

Run the supervised failure test with two injected failures, then automatic
resume attempts:

```bash
python3 health_worker.py -- \
  --keep_checkpoints \
  --replication \
  --replication_jump 1 \
  --replication_factor 2
```

The worker defaults to 10 epochs and fails after epochs 3 and 7. It restarts
with `--resume` after each failed attempt. Aggregate local rank metrics with:

```bash
python3 logs_collector.py
```

## Training Options

```text
--epochs COUNT             Distributed training epochs; default: 5
--steps_per_epoch COUNT    Distributed steps in each epoch; default: 20
--steps COUNT              Alias for --steps_per_epoch
--batch_size COUNT         Samples processed by each rank per step; default: 256
--learning_rate RATE       SGD learning rate; default: 0.05
--log_interval COUNT       Steps between global loss messages; default: 10
--seed VALUE               Base random seed; default: 1234
--async_save               Save each epoch checkpoint asynchronously
--keep_checkpoints         Do not remove checkpoint files after verification
--resume                   Resume from the latest checkpoint
--fail_epoch EPOCH         Fail after this epoch checkpoint is valid; 0 disables
--failure_rank RANK        Global rank that reports the primary failure
--metrics_dir DIRECTORY    Per-rank JSONL timing directory
```

The effective global batch size is `batch_size * total ranks`. Pass identical
training and checkpoint arguments on every VM.

`--keep_checkpoints` leaves the files available for inspection after the run.
Only the latest valid epoch is retained. The next normal run clears the
configured checkpoint directory before training; `--resume` preserves it.

## Setup Options

```text
--node-rank RANK          This VM's zero-based rank
--master-addr ADDRESS     Node 0's reachable private IP or hostname
--nnodes COUNT            Number of VMs; default: 2
--nproc-per-node COUNT    Worker processes/GPUs per VM; default: 1
--master-port PORT        PyTorch rendezvous TCP port; default: 29500
--checkpoint-size SIZE    tmpfs size limit; default: 8G
--nccl-interface NAME     Private network interface, such as eth0
--driver-package PACKAGE  nvidia-open, cuda-drivers, or skip
```

Rerunning `setup.sh` is safe. It updates packages, Python dependencies, the
checkpoint mount configuration, and `.cluster.env`.

## Launcher Environment Variables

`setup.sh` writes these values to `.cluster.env`, which is local to each VM and
ignored by Git:

| Variable | Purpose |
| --- | --- |
| `PYTHON` | Python interpreter in `.venv` |
| `NNODES` | Number of VMs |
| `NODE_RANK` | This VM's zero-based rank |
| `NPROC_PER_NODE` | Worker processes on this VM |
| `MASTER_ADDR` | Reachable private IP or hostname of node 0 |
| `MASTER_PORT` | PyTorch rendezvous TCP port |
| `CHECKPOINT_DIR` | Local checkpoint directory passed to the example |
| `NCCL_SOCKET_IFNAME` | Optional private network interface |

All extra arguments passed to `launch.sh` are forwarded to `BasicExample1.py`.
See available example arguments with:

```bash
./launch.sh --help
```

## Troubleshooting

- `PyTorch cannot access an NVIDIA GPU`: run `nvidia-smi`, reboot after driver
  installation, and rerun the same `setup.sh` command.
- `No matching distribution found for nvidia-resiliency-ext`: use Ubuntu 24.04
  with Python 3.10, 3.11, or 3.12.
- A multi-VM run hangs: verify node 0's private IP, TCP port `29500`, and
  peer-to-peer firewall rules.
- NCCL selects the wrong network interface: rerun setup on both VMs with
  `--nccl-interface`, such as `--nccl-interface eth0`.
- Replication setup fails: the replication factor cannot exceed the total
  number of ranks. This two-rank test uses jump `1` and factor `2`.

## Official References

- [NVIDIA Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext)
- [NVIDIA Resiliency Extension on PyPI](https://pypi.org/project/nvidia-resiliency-ext/)
- [PyTorch elastic run and rendezvous](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [PyTorch CUDA 12.8 installation command](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA 12.8 Linux installation guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)
- [NVIDIA Ubuntu driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html)
