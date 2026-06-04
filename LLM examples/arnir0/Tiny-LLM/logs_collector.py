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


def duration_summary(records):
    by_event = defaultdict(list)
    for record in records:
        if "duration_seconds" in record:
            by_event[record["event"]].append(record["duration_seconds"])
    return by_event


def print_duration_table(by_event):
    print("Measured Timings")
    print("event,count,min_s,median_s,mean_s,max_s")
    for event, values in sorted(by_event.items()):
        print(
            f"{event},{len(values)},{min(values):.6f},"
            f"{median(values):.6f},{mean(values):.6f},{max(values):.6f}"
        )


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

    print()
    print("Evaluation Metric Mapping")
    print("metric,result")
    print(f"Iteration Time,{iteration_time:.6f}s total measured compute work")
    print("Network Idle Time,not measured; use Nsight Systems/NCCL traces")
    if not wasted_gaps:
        print("Wasted Time After Failure,not available in this log set")
    else:
        print(
            "Wasted Time After Failure,"
            f"{sum(wasted_gaps):.6f}s total failure-to-resume gap"
        )
    print(f"Checkpoint Time,{checkpoint_time:.6f}s total measured checkpoint work")
    print(f"Checkpoint Frequency,{checkpoint_count} checkpoint writes/submissions")
    print(f"Injected Failures,{count_event(records, 'failure_injected')} events")
    print("Failure Recovery Probability,not measured; requires many failure trials")
    if effective_ratio is None:
        print("Effective Training Time Ratio,not available")
    else:
        print(f"Effective Training Time Ratio,{effective_ratio:.6f}")
    print("Scalability with Cluster Size,not measured; compare multiple cluster sizes")
    print("Traffic Interleaving Effectiveness,compare sync and --async_save runs")
    print("Checkpoint Serialization Overhead,checkpoint_snapshot approximates it")
    print("Failure Detection Time,not measured by this local worker")
    print("Recovery Startup/Warmup Time,use checkpoint_resume_load and resume_loaded")


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
