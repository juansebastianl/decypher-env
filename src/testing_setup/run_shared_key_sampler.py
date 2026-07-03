"""Run the shared-key AES-XTS sampler against the local testing fixture.

Example random-start run:
    python src/testing_setup/run_shared_key_sampler.py --samples-target 100 --epochs 10

Example fixture warm-start sanity check:
    python src/testing_setup/run_shared_key_sampler.py --warm-start-fixture
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_METADATA = Path(__file__).resolve().parent / "lorem_ipsum_aes_xts.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "sampler_runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the native shared-key sampler against src/testing_setup fixture data."
    )
    parser.add_argument("--metadata", type=Path, default=FIXTURE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--block-count", type=int, default=21)
    parser.add_argument("--samples-target", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0, help="OpenMP worker threads. 0 uses up to replica count.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sweeps-per-epoch", type=int, default=1)
    parser.add_argument("--t-min", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=20.0)
    parser.add_argument("--goal-ramp", type=float, default=0.1)
    parser.add_argument("--dual-eta", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warm-start-fixture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to starting from generated fixture plaintext/key metadata. This leaks the fixture key.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.metadata.exists():
        print(f"Fixture metadata does not exist: {args.metadata}", file=sys.stderr)
        return 1

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / timestamp
    samples_path = run_dir / "samples.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "src.decoder.legacy.sample_blocks",
        str(args.metadata),
        "--start-block",
        str(args.start_block),
        "--block-count",
        str(args.block_count),
        "--samples-target",
        str(args.samples_target),
        "--replicas",
        str(args.replicas),
        "--threads",
        str(args.threads),
        "--epochs",
        str(args.epochs),
        "--sweeps-per-epoch",
        str(args.sweeps_per_epoch),
        "--t-min",
        str(args.t_min),
        "--t-max",
        str(args.t_max),
        "--goal-ramp",
        str(args.goal_ramp),
        "--dual-eta",
        str(args.dual_eta),
        "--mu",
        str(args.mu),
        "--seed",
        str(args.seed),
        "--output",
        str(samples_path),
        "--metrics",
        str(metrics_path),
    ]
    if args.warm_start_fixture:
        command.append("--warm-start-fixture")

    print(f"Writing samples to: {samples_path}")
    print(f"Writing metrics to: {metrics_path}")
    return _run(command)


def _run(command: list[str]) -> int:
    import subprocess

    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
