import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXAMPLE_ROOT))

from model import MODEL_ID, load_model_and_tokenizer
from utils import (
    checkpoint_size_bytes,
    cleanup_checkpoints,
    configure_logging,
    create_async_queue,
    create_checkpoint_manager,
    distributed_mean,
    finalize_async_checkpoint,
    init_distributed_backend,
    inject_epoch_failure,
    load_latest_checkpoint,
    measure_duration,
    nonnegative_int,
    positive_float,
    positive_int,
    prepare_metrics,
    record_event,
    record_timing,
    reset_path_once,
    submit_checkpoint,
    validate_failure_config,
    validate_replication_config,
)


class TokenBlockDataset(Dataset):
    def __init__(self, token_blocks):
        self.token_blocks = token_blocks

    def __len__(self):
        return len(self.token_blocks)

    def __getitem__(self, index):
        return torch.tensor(self.token_blocks[index], dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny LLM distributed training with RAM checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--dataset_name", default="Salesforce/wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--max_dataset_rows", type=positive_int, default=512)
    parser.add_argument("--seq_length", type=positive_int, default=128)
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--steps_per_epoch", type=positive_int, default=10)
    parser.add_argument("--batch_size", type=positive_int, default=1)
    parser.add_argument("--learning_rate", type=positive_float, default=5e-5)
    parser.add_argument("--log_interval", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ckpt_dir", default="/mnt/checkpoint-ram/tiny-llm")
    parser.add_argument("--metrics_dir", default="logs/metrics")
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--run_label", default="tiny_llm")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--async_save", action="store_true")
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoint_interval_steps",
        type=nonnegative_int,
        default=1,
        help="Save a RAM checkpoint every N iterations; 0 disables checkpointing",
    )
    parser.add_argument(
        "--fail_epochs",
        default="",
        help="Comma-separated epochs that should fail after their checkpoint is valid",
    )
    parser.add_argument("--failure_rank", type=nonnegative_int, default=0)
    parser.add_argument("--replication", action="store_true")
    parser.add_argument("--replication_jump", type=positive_int, default=4)
    parser.add_argument("--replication_factor", type=positive_int, default=2)
    return parser.parse_args()


def build_dataset(args, tokenizer):
    raw_dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
    )
    row_count = min(args.max_dataset_rows, len(raw_dataset))
    raw_dataset = raw_dataset.select(range(row_count))

    separator = tokenizer.eos_token or "\n\n"
    text = separator.join(row["text"] for row in raw_dataset if row["text"].strip())
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < args.seq_length:
        raise RuntimeError(
            f"Dataset produced {len(token_ids)} tokens, fewer than seq_length "
            f"{args.seq_length}."
        )

    blocks = []
    for offset in range(0, len(token_ids) - args.seq_length + 1, args.seq_length):
        blocks.append(token_ids[offset : offset + args.seq_length])

    return TokenBlockDataset(blocks)


def collate_token_blocks(batch):
    input_ids = torch.stack(batch, dim=0)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def move_batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def next_position(checkpoint):
    epoch = checkpoint["epoch"]
    step = checkpoint["step_in_epoch"] + 1
    if step > checkpoint["steps_per_epoch"]:
        return epoch + 1, 1
    return epoch, step


def build_training_checkpoint(
    args,
    model,
    optimizer,
    epoch,
    step_in_epoch,
    global_step,
    loss_value,
    device,
):
    return {
        "model": copy.deepcopy(model.module.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state().to(device),
        "cuda_rng_state": torch.cuda.get_rng_state(device).to(device),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "loss": loss_value,
        "world_size": dist.get_world_size(),
        "steps_per_epoch": args.steps_per_epoch,
        "seed": args.seed,
    }


def restore_training_state(args, checkpoint, model, optimizer, device):
    if checkpoint["world_size"] != dist.get_world_size():
        raise RuntimeError(
            f"Checkpoint world size {checkpoint['world_size']} does not match "
            f"current world size {dist.get_world_size()}."
        )
    if checkpoint["steps_per_epoch"] != args.steps_per_epoch:
        raise RuntimeError(
            f"Checkpoint steps_per_epoch {checkpoint['steps_per_epoch']} does "
            f"not match requested {args.steps_per_epoch}."
        )
    if checkpoint["seed"] != args.seed:
        raise RuntimeError(
            f"Checkpoint seed {checkpoint['seed']} does not match requested {args.seed}."
        )

    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device)
    return next_position(checkpoint), checkpoint["global_step"]


def save_training_checkpoint(
    args,
    model,
    optimizer,
    ckpt_manager,
    async_queue,
    recorder,
    device,
    epoch,
    step_in_epoch,
    global_step,
    loss_value,
):
    checkpoint, snapshot_duration = measure_duration(
        device,
        lambda: build_training_checkpoint(
            args,
            model,
            optimizer,
            epoch,
            step_in_epoch,
            global_step,
            loss_value,
            device,
        ),
    )
    checkpoint_bytes = checkpoint_size_bytes(checkpoint)
    record_timing(
        recorder,
        device,
        "checkpoint_snapshot",
        epoch,
        snapshot_duration,
        global_step=global_step,
        checkpoint_bytes=checkpoint_bytes,
    )

    _, save_duration = measure_duration(
        device,
        lambda: submit_checkpoint(
            args,
            ckpt_manager,
            async_queue,
            checkpoint,
            global_step,
        ),
    )
    event_name = "checkpoint_async_submit" if args.async_save else "checkpoint_save"
    record_timing(
        recorder,
        device,
        event_name,
        epoch,
        save_duration,
        global_step=global_step,
        checkpoint_bytes=checkpoint_bytes,
    )
    return {
        "epoch": epoch,
        "global_step": global_step,
        "checkpoint_bytes": checkpoint_bytes,
    }


def finalize_pending_async(args, async_queue, pending, recorder, device):
    if pending is None:
        return
    _, finalize_duration = measure_duration(
        device,
        lambda: finalize_async_checkpoint(async_queue),
    )
    record_timing(
        recorder,
        device,
        "checkpoint_async_finalize",
        pending["epoch"],
        finalize_duration,
        global_step=pending["global_step"],
        checkpoint_bytes=pending["checkpoint_bytes"],
    )


def main():
    args = parse_args()
    configure_logging(args.logs_dir)
    logging.info("%s", args)

    device = init_distributed_backend()
    async_queue = None

    try:
        validate_replication_config(args)
        validate_failure_config(args)
        metrics = prepare_metrics(args)
        record_event(metrics, "job_start", extra_args=vars(args))

        checkpointing_enabled = args.checkpoint_interval_steps > 0
        if args.resume and not checkpointing_enabled:
            raise RuntimeError("--resume requires checkpointing to be enabled")
        if not args.resume and checkpointing_enabled:
            reset_path_once(args.ckpt_dir)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        model, tokenizer = load_model_and_tokenizer(args.model_id, args.dtype)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )

        dataset = build_dataset(args, tokenizer)
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_token_blocks,
            pin_memory=True,
            drop_last=True,
        )

        ckpt_manager = create_checkpoint_manager(args) if checkpointing_enabled else None
        async_queue = create_async_queue(args) if checkpointing_enabled else None
        start_epoch = 1
        start_step = 1
        global_step = 0
        last_checkpoint_global_step = 0
        pending_async = None

        if args.resume:
            checkpoint, load_duration = measure_duration(
                device,
                lambda: load_latest_checkpoint(ckpt_manager),
            )
            record_timing(
                metrics,
                device,
                "checkpoint_resume_load",
                checkpoint["epoch"],
                load_duration,
                global_step=checkpoint["global_step"],
                checkpoint_bytes=checkpoint_size_bytes(checkpoint),
            )
            (start_epoch, start_step), global_step = restore_training_state(
                args,
                checkpoint,
                model,
                optimizer,
                device,
            )
            last_checkpoint_global_step = global_step
            record_event(
                metrics,
                "resume_loaded",
                epoch=checkpoint["epoch"],
                global_step=global_step,
                next_epoch=start_epoch,
                next_step=start_step,
            )
            if start_epoch > args.epochs:
                record_event(
                    metrics,
                    "job_already_complete",
                    epoch=checkpoint["epoch"],
                    global_step=global_step,
                )
                cleanup_checkpoints(args)
                return

        if dist.get_rank() == 0:
            logging.info(
                "Training %s through epoch %d with %d ranks.",
                args.model_id,
                args.epochs,
                dist.get_world_size(),
            )

        last_loss = None
        for epoch in range(start_epoch, args.epochs + 1):
            sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            first_step = start_step if epoch == start_epoch else 1

            for step_in_epoch in range(1, args.steps_per_epoch + 1):
                batch = next(data_iter, None)
                if batch is None:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                if step_in_epoch < first_step:
                    continue

                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output, iteration_duration = measure_duration(
                    device,
                    lambda: model(**batch),
                )
                loss = output.loss
                _, backward_duration = measure_duration(
                    device,
                    lambda: loss.backward(),
                )
                _, optimizer_duration = measure_duration(
                    device,
                    optimizer.step,
                )

                global_step += 1
                last_loss = distributed_mean(loss)
                record_timing(
                    metrics,
                    device,
                    "iteration_forward",
                    epoch,
                    iteration_duration,
                    global_step=global_step,
                )
                record_timing(
                    metrics,
                    device,
                    "iteration_backward",
                    epoch,
                    backward_duration,
                    global_step=global_step,
                )
                record_timing(
                    metrics,
                    device,
                    "iteration_optimizer",
                    epoch,
                    optimizer_duration,
                    global_step=global_step,
                )

                if dist.get_rank() == 0 and (
                    global_step == 1 or global_step % args.log_interval == 0
                ):
                    logging.info(
                        "epoch=%d/%d step=%d/%d global_step=%d loss=%.6f",
                        epoch,
                        args.epochs,
                        step_in_epoch,
                        args.steps_per_epoch,
                        global_step,
                        last_loss,
                    )

                if (
                    checkpointing_enabled
                    and global_step % args.checkpoint_interval_steps == 0
                ):
                    if args.async_save and pending_async is not None:
                        finalize_pending_async(
                            args,
                            async_queue,
                            pending_async,
                            metrics,
                            device,
                        )
                    saved_checkpoint = save_training_checkpoint(
                        args,
                        model,
                        optimizer,
                        ckpt_manager,
                        async_queue,
                        metrics,
                        device,
                        epoch,
                        step_in_epoch,
                        global_step,
                        last_loss,
                    )
                    last_checkpoint_global_step = global_step
                    pending_async = saved_checkpoint if args.async_save else None

            if checkpointing_enabled and args.async_save and pending_async is not None:
                finalize_pending_async(args, async_queue, pending_async, metrics, device)
                pending_async = None
            if checkpointing_enabled:
                if last_checkpoint_global_step != global_step:
                    saved_checkpoint = save_training_checkpoint(
                        args,
                        model,
                        optimizer,
                        ckpt_manager,
                        async_queue,
                        metrics,
                        device,
                        epoch,
                        args.steps_per_epoch,
                        global_step,
                        last_loss,
                    )
                    last_checkpoint_global_step = global_step
                    pending_async = saved_checkpoint if args.async_save else None
                    if args.async_save and pending_async is not None:
                        finalize_pending_async(
                            args,
                            async_queue,
                            pending_async,
                            metrics,
                            device,
                        )
                        pending_async = None
                inject_epoch_failure(args, metrics, epoch, global_step)
            start_step = 1

        if checkpointing_enabled and args.async_save and pending_async is not None:
            finalize_pending_async(args, async_queue, pending_async, metrics, device)

        if checkpointing_enabled:
            checkpoint, load_duration = measure_duration(
                device,
                lambda: load_latest_checkpoint(ckpt_manager),
            )
            record_timing(
                metrics,
                device,
                "checkpoint_verification_load",
                checkpoint["epoch"],
                load_duration,
                global_step=checkpoint["global_step"],
                checkpoint_bytes=checkpoint_size_bytes(checkpoint),
            )
            cleanup_checkpoints(args)

        record_event(metrics, "job_complete", epoch=args.epochs, global_step=global_step)
        if dist.get_rank() == 0:
            logging.info("Training complete. Final loss: %.6f.", last_loss)
    finally:
        try:
            if async_queue is not None:
                async_queue.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
