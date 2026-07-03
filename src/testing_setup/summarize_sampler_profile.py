"""Summarize sampler profiling JSONL emitted by run_window_fusion_sampler."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize sampler_profile.jsonl timing records.")
    parser.add_argument("profile", type=Path, help="Path to sampler_profile.jsonl.")
    parser.add_argument("--top", type=int, default=12, help="Number of timing buckets to print.")
    parser.add_argument(
        "--phase",
        choices=("all", "window_setup", "epoch"),
        default="all",
        help="Restrict summary to a profile record phase.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = list(_records(args.profile, args.phase))
    if not records:
        raise SystemExit(f"no records found in {args.profile}")

    print(f"records={len(records)} phase={args.phase}")
    print(f"epochs={sum(1 for record in records if record.get('phase') == 'epoch')}")
    print(f"run_elapsed_seconds={max(float(record.get('run_elapsed_seconds', 0.0)) for record in records):.6f}")
    _print_section("python_delta_seconds", _sum_nested(records, "python_delta_seconds"), args.top)
    _print_section("native_delta_seconds", _sum_nested(records, "native_delta_seconds"), args.top)
    _print_section("native_delta_counts", _sum_nested(records, "native_delta_counts"), args.top)
    return 0


def _records(path: Path, phase: str) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as profile_file:
        for line in profile_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if phase != "all" and record.get("phase") != phase:
                continue
            yield record


def _sum_nested(records: Iterable[dict[str, object]], key: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for record in records:
        values = record.get(key, {})
        if not isinstance(values, dict):
            continue
        for label, value in values.items():
            totals[str(label)] += float(value)
    return dict(totals)


def _print_section(title: str, totals: dict[str, float], limit: int) -> None:
    print()
    print(title)
    for label, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]:
        print(f"  {label}: {value:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
