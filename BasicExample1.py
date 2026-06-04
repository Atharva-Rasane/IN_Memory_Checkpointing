import argparse
import copy
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue
from nvidia_resiliency_ext.checkpointing.local.basic_state_dict import BasicTensorAwareStateDict
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
    LocalCheckpointManager,
)
from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
    CliqueReplicationStrategy,
)

logging.basicConfig(level=logging.INFO)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed training with NVIDIA local checkpointing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_dir",
        default="/tmp/test_local_checkpointing/",
        help="Checkpoint directory for local checkpoints",
    )
    parser.add_argument(
        "--epochs",
        default=5,
        type=positive_int,
        help="Number of distributed training epochs",
    )
    parser.add_argument(
        "--steps_per_epoch",
        "--steps",
        dest="steps_per_epoch",
        default=20,
        type=positive_int,
        help="Distributed training steps in each epoch",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=positive_int,
        help="Training samples processed by each rank per step",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.05,
        type=positive_float,
        help="SGD learning rate",
    )
    parser.add_argument(
        "--log_interval",
        default=10,
        type=positive_int,
        help="Steps between global training-loss messages",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Base random seed",
    )
    parser.add_argument(
        "--async_save",
        action="store_true",
        help="Save each epoch checkpoint asynchronously",
    )
    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Keep checkpoint files after restore verification",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint instead of starting from scratch",
    )
    parser.add_argument(
        "--fail_epoch",
        default=0,
        type=nonnegative_int,
        help=(
            "Stop the job after this epoch checkpoint is valid; "
            "0 disables failure injection"
        ),
    )
    parser.add_argument(
        "--failure_rank",
        default=0,
        type=nonnegative_int,
        help="Global rank assigned the primary injected-failure exit status",
    )
    parser.add_argument(
        "--metrics_dir",
        default="checkpoint_metrics",
        help="Directory for per-rank JSONL timing metrics",
    )
    parser.add_argument(
        "--replication",
        action="store_true",
        help="Enable local-checkpoint replication on every rank",
    )
    parser.add_argument(
        "--replication_jump",
        default=4,
        type=positive_int,
        help="Spacing between ranks in a replication group",
    )
    parser.add_argument(
        "--replication_factor",
        default=2,
        type=positive_int,
        help="Number of ranks storing each checkpoint part",
    )
    return parser.parse_args()


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        return self.fc2(self.activation(self.fc1(inputs)))


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
    if args.fail_epoch > args.epochs:
        raise ValueError(
            f"fail_epoch must be no greater than epochs ({args.epochs}); "
            f"received {args.fail_epoch}."
        )


def create_checkpoint_manager(args):
    if args.replication:
        logging.info("Creating CliqueReplicationStrategy.")
        repl_strategy = CliqueReplicationStrategy.from_replication_params(
            args.replication_jump, args.replication_factor
        )
    else:
        repl_strategy = None

    return LocalCheckpointManager(args.ckpt_dir, repl_strategy=repl_strategy)


def reset_checkpoint_dir(args):
    if dist.get_node_local_rank() == 0:
        logging.info("Resetting local checkpoint directory: %s", args.ckpt_dir)
        shutil.rmtree(args.ckpt_dir, ignore_errors=True)
    dist.barrier()


class MetricsRecorder:
    def __init__(self, metrics_dir):
        self.path = Path(metrics_dir) / f"rank_{dist.get_rank()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, payload):
        payload = {
            "timestamp_unix": time.time(),
            "rank": dist.get_rank(),
            "local_rank": int(os.environ["LOCAL_RANK"]),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")


def prepare_metrics(args):
    if not args.resume and dist.get_node_local_rank() == 0:
        shutil.rmtree(args.metrics_dir, ignore_errors=True)
    dist.barrier()
    return MetricsRecorder(args.metrics_dir)


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
            "TIMING event=%s epoch=%s min=%.6fs mean=%.6fs max=%.6fs%s",
            event,
            epoch,
            min(durations),
            sum(durations) / len(durations),
            max(durations),
            size_message,
        )


def checkpoint_size_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(checkpoint_size_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(checkpoint_size_bytes(item) for item in value)
    return 0


def create_model_and_optimizer(args, device):
    model = SimpleModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    return model, optimizer


def make_targets(inputs):
    first = (
        0.8 * inputs[:, 0]
        - 0.4 * inputs[:, 1]
        + 0.2 * inputs[:, 2]
        + 0.1
    )
    second = (
        -0.3 * inputs[:, 3]
        + 0.7 * inputs[:, 4]
        - 0.2 * inputs[:, 5]
        - 0.1
    )
    return torch.stack((first, second), dim=1)


def distributed_mean(value):
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result /= dist.get_world_size()
    return result.item()


def train_epoch(args, epoch, model, optimizer, device, generator):
    loss_fn = nn.MSELoss()
    first_loss = None
    last_loss = None

    for step in range(1, args.steps_per_epoch + 1):
        inputs = torch.randn(
            args.batch_size,
            10,
            device=device,
            generator=generator,
        )
        targets = make_targets(inputs)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        if (
            step == 1
            or step % args.log_interval == 0
            or step == args.steps_per_epoch
        ):
            last_loss = distributed_mean(loss)
            if first_loss is None:
                first_loss = last_loss
            if dist.get_rank() == 0:
                logging.info(
                    "Epoch %d/%d, step %d/%d, global mean loss: %.6f",
                    epoch,
                    args.epochs,
                    step,
                    args.steps_per_epoch,
                    last_loss,
                )

    return first_loss, last_loss


@torch.no_grad()
def evaluate(model, batch_size, device):
    model.eval()
    inputs = torch.linspace(
        -2.0,
        2.0,
        steps=batch_size * 10,
        device=device,
    ).reshape(batch_size, 10)
    loss = nn.functional.mse_loss(model(inputs), make_targets(inputs))
    model.train()
    return distributed_mean(loss)


def build_checkpoint(
    args,
    model,
    optimizer,
    generator,
    epoch,
    global_step,
    validation_loss,
):
    return {
        "model": copy.deepcopy(model.module.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "generator_state": generator.get_state().to(model.module.fc1.weight.device),
        "epoch": epoch,
        "global_step": global_step,
        "validation_loss": validation_loss,
        "world_size": dist.get_world_size(),
        "steps_per_epoch": args.steps_per_epoch,
        "seed": args.seed,
    }


def save_checkpoint(args, ckpt_manager, async_queue, checkpoint, epoch):
    tensor_aware_state = BasicTensorAwareStateDict(checkpoint)

    if args.async_save:
        logging.info("Creating asynchronous RAM checkpoint for epoch %d.", epoch)
        save_request = ckpt_manager.save(tensor_aware_state, epoch, is_async=True)
        async_queue.schedule_async_request(save_request)
    else:
        logging.info("Saving RAM checkpoint for epoch %d.", epoch)
        ckpt_manager.save(tensor_aware_state, epoch)


def finalize_async_save(async_queue, epoch):
    logging.info("Finalizing asynchronous RAM checkpoint for epoch %d.", epoch)
    async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)


def load_checkpoint(ckpt_manager):
    logging.info("Loading latest training checkpoint.")
    epoch = ckpt_manager.find_latest()
    if epoch == -1:
        raise RuntimeError("Local checkpoint has not been found")

    tensor_aware_state, checkpoint_part_id = ckpt_manager.load()
    logging.info("Successfully loaded checkpoint part %s.", checkpoint_part_id)
    return tensor_aware_state.state_dict


def restore_training_state(args, checkpoint, model, optimizer, generator):
    if checkpoint["world_size"] != dist.get_world_size():
        raise RuntimeError(
            f"Checkpoint used world size {checkpoint['world_size']}, "
            f"but this run uses {dist.get_world_size()}."
        )
    if checkpoint.get("steps_per_epoch") != args.steps_per_epoch:
        raise RuntimeError(
            f"Checkpoint used {checkpoint.get('steps_per_epoch')} steps per epoch, "
            f"but this run requested {args.steps_per_epoch}."
        )
    if checkpoint.get("seed") != args.seed:
        raise RuntimeError(
            f"Checkpoint used seed {checkpoint.get('seed')}, but this run requested {args.seed}."
        )

    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    generator.set_state(checkpoint["generator_state"].cpu())
    return checkpoint["epoch"] + 1, checkpoint["global_step"]


def inject_failure(args, epoch):
    if args.fail_epoch != epoch:
        return

    dist.barrier()
    rank = dist.get_rank()
    if rank == args.failure_rank:
        logging.error(
            "Injecting requested failure on rank %d after epoch %d checkpoint finalized.",
            args.failure_rank,
            epoch,
        )
        exit_code = 42
    else:
        logging.error(
            "Stopping rank %d after rank %d's requested failure at epoch %d.",
            rank,
            args.failure_rank,
            epoch,
        )
        exit_code = 43

    logging.shutdown()
    os._exit(exit_code)


def verify_restored_checkpoint(args, checkpoint, expected_validation_loss, device):
    if checkpoint["epoch"] != args.epochs:
        raise RuntimeError(
            f"Expected checkpoint epoch {args.epochs}, loaded {checkpoint['epoch']}."
        )
    expected_global_step = args.epochs * args.steps_per_epoch
    if checkpoint["global_step"] != expected_global_step:
        raise RuntimeError(
            f"Expected global step {expected_global_step}, "
            f"loaded {checkpoint['global_step']}."
        )
    if checkpoint["world_size"] != dist.get_world_size():
        raise RuntimeError(
            f"Expected checkpoint world size {dist.get_world_size()}, "
            f"loaded {checkpoint['world_size']}."
        )
    if not math.isclose(
        checkpoint["validation_loss"],
        expected_validation_loss,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            "Checkpoint validation-loss metadata does not match the saved model: "
            f"{checkpoint['validation_loss']} != {expected_validation_loss}"
        )

    restored_model, restored_optimizer = create_model_and_optimizer(args, device)
    restored_model.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_model = DistributedDataParallel(
        restored_model,
        device_ids=[device.index],
        output_device=device.index,
    )

    restored_validation_loss = evaluate(restored_model, args.batch_size, device)
    if not math.isclose(
        restored_validation_loss,
        expected_validation_loss,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            "Restored model validation loss does not match the saved model: "
            f"{restored_validation_loss} != {expected_validation_loss}"
        )

    if dist.get_rank() == 0:
        logging.info(
            "Checkpoint restore verified at epoch %d, global step %d, "
            "with validation loss %.6f.",
            checkpoint["epoch"],
            checkpoint["global_step"],
            restored_validation_loss,
        )


def cleanup_checkpoints(args):
    dist.barrier()
    if not args.keep_checkpoints and dist.get_node_local_rank() == 0:
        logging.info("Cleaning up local checkpoint directory: %s", args.ckpt_dir)
        shutil.rmtree(args.ckpt_dir, ignore_errors=True)
    dist.barrier()


def finalize_pending_async(async_queue, pending, recorder, device):
    _, finalize_duration = measure_duration(
        device,
        lambda: finalize_async_save(async_queue, pending["epoch"]),
    )
    end_to_end_duration = time.perf_counter() - pending["started_at"]
    record_timing(
        recorder,
        device,
        "checkpoint_async_finalize",
        pending["epoch"],
        finalize_duration,
        pending["checkpoint_bytes"],
    )
    record_timing(
        recorder,
        device,
        "checkpoint_async_end_to_end",
        pending["epoch"],
        end_to_end_duration,
        pending["checkpoint_bytes"],
    )


def main():
    args = parse_args()
    logging.info("%s", args)

    device = init_distributed_backend()
    async_queue = None

    try:
        validate_replication_config(args)
        validate_failure_config(args)
        metrics = prepare_metrics(args)
        if args.resume:
            logging.info("Preserving checkpoint directory for resume: %s", args.ckpt_dir)
        else:
            reset_checkpoint_dir(args)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, optimizer = create_model_and_optimizer(args, device)
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )

        ckpt_manager = create_checkpoint_manager(args)
        if args.async_save:
            async_queue = AsyncCallsQueue(persistent=False)

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + dist.get_rank())
        start_epoch = 1
        global_step = 0
        if args.resume:
            restored_checkpoint, load_duration = measure_duration(
                device,
                lambda: load_checkpoint(ckpt_manager),
            )
            record_timing(
                metrics,
                device,
                "checkpoint_resume_load",
                restored_checkpoint["epoch"],
                load_duration,
                checkpoint_size_bytes(restored_checkpoint),
            )
            start_epoch, global_step = restore_training_state(
                args,
                restored_checkpoint,
                model,
                optimizer,
                generator,
            )
            if start_epoch > args.epochs:
                raise RuntimeError(
                    f"Checkpoint epoch {restored_checkpoint['epoch']} already reaches "
                    f"the requested {args.epochs} epochs."
                )
            if dist.get_rank() == 0:
                logging.info(
                    "Resuming from epoch %d at global step %d; next epoch is %d.",
                    restored_checkpoint["epoch"],
                    global_step,
                    start_epoch,
                )

        if dist.get_rank() == 0:
            logging.info(
                "Training through epoch %d with %d ranks and a global batch size of %d.",
                args.epochs,
                dist.get_world_size(),
                args.batch_size * dist.get_world_size(),
            )

        initial_loss = None
        final_loss = None
        validation_loss = None
        pending_async = None

        for epoch in range(start_epoch, args.epochs + 1):
            (epoch_initial_loss, final_loss), train_duration = measure_duration(
                device,
                lambda: train_epoch(
                    args,
                    epoch,
                    model,
                    optimizer,
                    device,
                    generator,
                ),
            )
            record_timing(metrics, device, "epoch_training", epoch, train_duration)
            if initial_loss is None:
                initial_loss = epoch_initial_loss

            validation_loss = evaluate(model, args.batch_size, device)
            if dist.get_rank() == 0:
                logging.info(
                    "Epoch %d/%d complete; validation loss: %.6f.",
                    epoch,
                    args.epochs,
                    validation_loss,
                )

            if pending_async is not None:
                finalize_pending_async(async_queue, pending_async, metrics, device)
                pending_async = None

            global_step += args.steps_per_epoch
            checkpoint, snapshot_duration = measure_duration(
                device,
                lambda: build_checkpoint(
                    args,
                    model,
                    optimizer,
                    generator,
                    epoch,
                    global_step,
                    validation_loss,
                ),
            )
            checkpoint_bytes = checkpoint_size_bytes(checkpoint)
            record_timing(
                metrics,
                device,
                "checkpoint_snapshot",
                epoch,
                snapshot_duration,
                checkpoint_bytes,
            )

            async_started_at = time.perf_counter()
            _, save_duration = measure_duration(
                device,
                lambda: save_checkpoint(
                    args,
                    ckpt_manager,
                    async_queue,
                    checkpoint,
                    epoch,
                ),
            )
            save_event = (
                "checkpoint_async_submit" if args.async_save else "checkpoint_save"
            )
            record_timing(
                metrics,
                device,
                save_event,
                epoch,
                save_duration,
                checkpoint_bytes,
            )
            if args.async_save:
                pending_async = {
                    "epoch": epoch,
                    "started_at": async_started_at,
                    "checkpoint_bytes": checkpoint_bytes,
                }

            if args.fail_epoch == epoch and pending_async is not None:
                finalize_pending_async(async_queue, pending_async, metrics, device)
                pending_async = None
            inject_failure(args, epoch)

        if pending_async is not None:
            finalize_pending_async(async_queue, pending_async, metrics, device)

        if dist.get_rank() == 0:
            logging.info(
                "Training complete. Loss changed from %.6f to %.6f; "
                "final validation loss: %.6f.",
                initial_loss,
                final_loss,
                validation_loss,
            )

        dist.barrier()
        restored_checkpoint, load_duration = measure_duration(
            device,
            lambda: load_checkpoint(ckpt_manager),
        )
        record_timing(
            metrics,
            device,
            "checkpoint_verification_load",
            restored_checkpoint["epoch"],
            load_duration,
            checkpoint_size_bytes(restored_checkpoint),
        )
        verify_restored_checkpoint(args, restored_checkpoint, validation_loss, device)
        cleanup_checkpoints(args)
    finally:
        try:
            if async_queue is not None:
                async_queue.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
