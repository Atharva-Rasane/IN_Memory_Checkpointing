import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue
from nvidia_resiliency_ext.checkpointing.local.basic_state_dict import (
    BasicTensorAwareStateDict,
)
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)
from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
    CliqueReplicationStrategy,
)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def configure_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "local")
    log_path = Path(log_dir) / f"rank_{rank}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )


def init_distributed_backend(backend="nccl"):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend=backend, init_method="env://", device_id=device)
    logging.info(
        "Rank %s initialized with %s on %s.",
        dist.get_rank(),
        backend,
        device,
    )
    return device


def validate_replication_config(args):
    if not args.replication:
        return

    group_size = args.replication_jump * args.replication_factor
    world_size = dist.get_world_size()
    if world_size % group_size != 0:
        raise ValueError(
            "Replication requires WORLD_SIZE to be divisible by "
            "replication_jump * replication_factor; "
            f"received {world_size} % ({args.replication_jump} * "
            f"{args.replication_factor})."
        )


def validate_failure_config(args):
    if args.failure_rank >= dist.get_world_size():
        raise ValueError(
            f"failure_rank must be less than world size {dist.get_world_size()}; "
            f"received {args.failure_rank}."
        )


def create_checkpoint_manager(args):
    if args.replication:
        logging.info("Creating CliqueReplicationStrategy.")
        repl_strategy = CliqueReplicationStrategy.from_replication_params(
            args.replication_jump,
            args.replication_factor,
        )
    else:
        repl_strategy = None

    return LocalCheckpointManager(args.ckpt_dir, repl_strategy=repl_strategy)


def create_async_queue(args):
    return AsyncCallsQueue(persistent=False) if args.async_save else None


def reset_path_once(path):
    if dist.get_node_local_rank() == 0:
        logging.info("Resetting path: %s", path)
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()


class MetricsRecorder:
    def __init__(self, metrics_dir, run_label):
        self.path = Path(metrics_dir) / f"rank_{dist.get_rank()}.jsonl"
        self.run_label = run_label
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, payload):
        payload = {
            "timestamp_unix": time.time(),
            "run_label": self.run_label,
            "rank": dist.get_rank(),
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "world_size": dist.get_world_size(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")


def prepare_metrics(args):
    if not args.resume and dist.get_node_local_rank() == 0:
        shutil.rmtree(args.metrics_dir, ignore_errors=True)
    dist.barrier()
    return MetricsRecorder(args.metrics_dir, args.run_label)


def measure_duration(device, operation):
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - started_at


def record_timing(
    recorder,
    device,
    event,
    epoch,
    duration_seconds,
    global_step=None,
    checkpoint_bytes=None,
):
    local_duration = torch.tensor([duration_seconds], dtype=torch.float64, device=device)
    gathered = [torch.zeros_like(local_duration) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_duration)
    durations = [sample.item() for sample in gathered]

    payload = {
        "event": event,
        "epoch": epoch,
        "duration_seconds": duration_seconds,
        "global_min_seconds": min(durations),
        "global_mean_seconds": sum(durations) / len(durations),
        "global_max_seconds": max(durations),
    }
    if global_step is not None:
        payload["global_step"] = global_step
    if checkpoint_bytes is not None:
        checkpoint_megabytes = checkpoint_bytes / (1024**2)
        payload["checkpoint_bytes"] = checkpoint_bytes
        payload["checkpoint_megabytes"] = checkpoint_megabytes
        payload["effective_throughput_mb_per_second"] = (
            checkpoint_megabytes / max(durations) if max(durations) > 0 else None
        )

    recorder.record(payload)
    if dist.get_rank() == 0:
        size_message = (
            f" size={checkpoint_bytes / (1024**2):.2f}MiB"
            if checkpoint_bytes is not None
            else ""
        )
        logging.info(
            "TIMING event=%s epoch=%s step=%s min=%.6fs mean=%.6fs max=%.6fs%s",
            event,
            epoch,
            global_step,
            min(durations),
            sum(durations) / len(durations),
            max(durations),
            size_message,
        )


def record_event(recorder, event, epoch=None, global_step=None, **extra):
    payload = {"event": event}
    if epoch is not None:
        payload["epoch"] = epoch
    if global_step is not None:
        payload["global_step"] = global_step
    payload.update(extra)
    recorder.record(payload)
    if dist.get_rank() == 0:
        logging.info("EVENT %s %s", event, payload)


def checkpoint_size_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(checkpoint_size_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(checkpoint_size_bytes(item) for item in value)
    return 0


def distributed_mean(value):
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result /= dist.get_world_size()
    return result.item()


def parse_epoch_csv(value):
    if not value:
        return set()
    epochs = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        epoch = int(item)
        if epoch <= 0:
            raise argparse.ArgumentTypeError("fail epochs must be greater than zero")
        epochs.add(epoch)
    return epochs


def submit_checkpoint(args, ckpt_manager, async_queue, checkpoint, iteration):
    tensor_aware_state = BasicTensorAwareStateDict(checkpoint)
    if args.async_save:
        save_request = ckpt_manager.save(tensor_aware_state, iteration, is_async=True)
        async_queue.schedule_async_request(save_request)
    else:
        ckpt_manager.save(tensor_aware_state, iteration)


def finalize_async_checkpoint(async_queue):
    async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)


def load_latest_checkpoint(ckpt_manager):
    iteration = ckpt_manager.find_latest()
    if iteration == -1:
        raise RuntimeError("Local checkpoint has not been found")
    tensor_aware_state, checkpoint_part_id = ckpt_manager.load()
    logging.info(
        "Loaded checkpoint iteration %s from part %s.",
        iteration,
        checkpoint_part_id,
    )
    return tensor_aware_state.state_dict


def inject_epoch_failure(args, recorder, epoch, global_step):
    fail_epochs = parse_epoch_csv(args.fail_epochs)
    if epoch not in fail_epochs:
        return

    dist.barrier()
    rank = dist.get_rank()
    exit_code = 42 if rank == args.failure_rank else 43
    record_event(
        recorder,
        "failure_injected",
        epoch=epoch,
        global_step=global_step,
        failure_rank=args.failure_rank,
        exit_code=exit_code,
    )
    logging.error(
        "Stopping rank %d after requested failure at epoch %d.",
        rank,
        epoch,
    )
    logging.shutdown()
    os._exit(exit_code)


def cleanup_checkpoints(args):
    dist.barrier()
    if not args.keep_checkpoints and dist.get_node_local_rank() == 0:
        logging.info("Cleaning up local checkpoint directory: %s", args.ckpt_dir)
        shutil.rmtree(args.ckpt_dir, ignore_errors=True)
    dist.barrier()
