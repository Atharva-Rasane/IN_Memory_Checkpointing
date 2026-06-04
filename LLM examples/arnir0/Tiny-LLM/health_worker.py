import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restart the Tiny-LLM job after injected failures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--restart_delay_seconds", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fail_epochs", default="3,7")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def write_worker_event(log_path, payload):
    payload = {
        "timestamp_unix": time.time(),
        "node_rank": os.environ.get("NODE_RANK", "unknown"),
        **payload,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    log_path = script_dir / "logs" / "health_worker.jsonl"
    extra_args = args.training_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    for attempt in range(1, args.max_attempts + 1):
        command = [
            "bash",
            str(script_dir / "launch.sh"),
            "--epochs",
            str(args.epochs),
            "--fail_epochs",
            args.fail_epochs,
        ]
        if attempt > 1:
            command.append("--resume")
        command.extend(extra_args)

        write_worker_event(
            log_path,
            {
                "event": "attempt_start",
                "attempt": attempt,
                "command": command,
            },
        )
        started_at = time.monotonic()
        completed = subprocess.run(command, cwd=script_dir, check=False)
        duration = time.monotonic() - started_at
        write_worker_event(
            log_path,
            {
                "event": "attempt_end",
                "attempt": attempt,
                "returncode": completed.returncode,
                "duration_seconds": duration,
            },
        )

        if completed.returncode == 0:
            write_worker_event(
                log_path,
                {"event": "worker_complete", "attempt": attempt},
            )
            return

        if attempt == args.max_attempts:
            raise SystemExit(completed.returncode)

        time.sleep(args.restart_delay_seconds)


if __name__ == "__main__":
    main()
