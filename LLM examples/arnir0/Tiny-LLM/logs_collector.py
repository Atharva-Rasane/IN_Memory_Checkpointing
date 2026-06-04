import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Tiny-LLM training and checkpoint metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics_dir", default="logs/metrics")
    return parser.parse_args()


def read_metrics(metrics_dir):
    records = []
    for path in sorted(Path(metrics_dir).glob("rank_*.jsonl")):
        with path.open("r", encoding="utf-8") as metrics_file:
            for line in metrics_file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_line = "| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"

    print(border)
    print(header_line)
    print(border)
    for row in rows:
        print(
            "| "
            + " | ".join(
                str(value).ljust(widths[index]) for index, value in enumerate(row)
            )
            + " |"
        )
    print(border)


def duration_summary(records):
    by_event = defaultdict(list)
    for record in records:
        if "duration_seconds" in record:
            by_event[record["event"]].append(record["duration_seconds"])
    return by_event


def print_duration_table(by_event):
    print("Measured Timings")
    rows = []
    for event, values in sorted(by_event.items()):
        rows.append(
            [
                event,
                len(values),
                f"{min(values):.6f}",
                f"{median(values):.6f}",
                f"{mean(values):.6f}",
                f"{max(values):.6f}",
            ]
        )
    print_table(["event", "count", "min_s", "median_s", "mean_s", "max_s"], rows)


def sum_event(by_event, names):
    return sum(sum(by_event.get(name, [])) for name in names)


def count_event(records, name):
    return sum(1 for record in records if record.get("event") == name)


def failure_to_resume_gaps(records):
    failure_times = sorted(
        record["timestamp_unix"]
        for record in records
        if record.get("event") == "failure_injected"
    )
    resume_times = sorted(
        record["timestamp_unix"]
        for record in records
        if record.get("event") == "resume_loaded"
    )
    gaps = []
    for failure_time in failure_times:
        later_resumes = [value for value in resume_times if value > failure_time]
        if later_resumes:
            gaps.append(later_resumes[0] - failure_time)
    return gaps


def print_evaluation_mapping(records, by_event):
    iteration_time = sum_event(
        by_event,
        ["iteration_forward", "iteration_backward", "iteration_optimizer"],
    )
    checkpoint_time = sum_event(
        by_event,
        [
            "checkpoint_snapshot",
            "checkpoint_save",
            "checkpoint_async_submit",
            "checkpoint_async_finalize",
        ],
    )
    timestamps = [record["timestamp_unix"] for record in records]
    wall_time = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0
    effective_ratio = iteration_time / wall_time if wall_time > 0 else None
    checkpoint_count = sum(
        count_event(records, name)
        for name in ("checkpoint_save", "checkpoint_async_submit")
    )

    wasted_gaps = failure_to_resume_gaps(records)

    rows = [
        ["Iteration Time", f"{iteration_time:.6f}s total measured compute work"],
        ["Network Idle Time", "not measured; use Nsight Systems/NCCL traces"],
    ]
    if not wasted_gaps:
        rows.append(["Wasted Time After Failure", "not available in this log set"])
    else:
        rows.append(
            [
                "Wasted Time After Failure",
                f"{sum(wasted_gaps):.6f}s total failure-to-resume gap",
            ]
        )
    rows.extend(
        [
            ["Checkpoint Time", f"{checkpoint_time:.6f}s total measured checkpoint work"],
            ["Checkpoint Frequency", f"{checkpoint_count} checkpoint writes/submissions"],
            ["Injected Failures", f"{count_event(records, 'failure_injected')} events"],
            ["Failure Recovery Probability", "not measured; requires many failure trials"],
        ]
    )
    if effective_ratio is None:
        rows.append(["Effective Training Time Ratio", "not available"])
    else:
        rows.append(["Effective Training Time Ratio", f"{effective_ratio:.6f}"])
    rows.extend(
        [
            ["Scalability with Cluster Size", "not measured; compare cluster sizes"],
            ["Traffic Interleaving Effectiveness", "compare sync and --async_save runs"],
            ["Checkpoint Serialization Overhead", "checkpoint_snapshot approximates it"],
            ["Failure Detection Time", "not measured by this local worker"],
            ["Recovery Startup/Warmup Time", "use checkpoint_resume_load/resume_loaded"],
        ]
    )

    print()
    print("Evaluation Metric Mapping")
    print_table(["metric", "result"], rows)


def main():
    args = parse_args()
    records = read_metrics(args.metrics_dir)
    if not records:
        raise SystemExit(f"No rank_*.jsonl files found in {args.metrics_dir}")
    by_event = duration_summary(records)
    print_duration_table(by_event)
    print_evaluation_mapping(records, by_event)


if __name__ == "__main__":
    main()
