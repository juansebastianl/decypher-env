"""Shared-key native PT sampler CLI for lowered AES-XTS block windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends.base import ParallelTemperingConfig
from .backends.native import NativeParallelTemperingEngine
from .sampler import SamplerConfig, XtsSampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample shared AES-XTS keys over a lowered block window.")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--block-count", type=int, default=21)
    parser.add_argument("--samples-target", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--t-min", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sweeps-per-epoch", type=int, default=1000)
    parser.add_argument("--goal-ramp", type=float, default=0.1)
    parser.add_argument("--dual-eta", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--warm-start-fixture",
        action="store_true",
        help="Initialize from fixture plaintext/key metadata when available.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sampler = XtsSampler(
        SamplerConfig(
            metadata_path=args.metadata,
            start_block=args.start_block,
            block_count=args.block_count,
            encoding="lowered",
            backend="native-pt",
            seed=args.seed,
        )
    )
    assignment = sampler.known_fixture_assignment() if args.warm_start_fixture else sampler.random_assignment()
    config = ParallelTemperingConfig(
        replicas=args.replicas,
        t_min=args.t_min,
        t_max=args.t_max,
        sweeps_per_epoch=args.sweeps_per_epoch,
        mu=args.mu,
        dual_eta=args.dual_eta,
        goal_weight=args.goal_ramp,
        seed=args.seed,
        threads=args.threads,
    )
    engine = NativeParallelTemperingEngine(sampler.circuit, assignment, config=config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a", encoding="utf-8") as sample_file, args.metrics.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(args.epochs):
            engine.run_epoch(args.sweeps_per_epoch)
            for sample in engine.drain_feasible(args.samples_target - written):
                sample_file.write(json.dumps(_sample_record(args, sample)) + "\n")
                written += 1
                if written >= args.samples_target:
                    break
            metrics = engine.metrics()
            metrics["epoch"] = epoch
            metrics["samples_written"] = written
            metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
            if written >= args.samples_target:
                break
    return 0


def _sample_record(args: argparse.Namespace, assignment) -> dict[str, object]:  # noqa: ANN001
    plaintext = bytes(assignment.plaintext)
    return {
        "start_block": args.start_block,
        "block_count": args.block_count,
        "key1_hex": bytes(assignment.key1).hex(),
        "key2_hex": bytes(assignment.key2).hex(),
        "plaintext_hex": plaintext.hex(),
        "plaintext_blocks_hex": [
            plaintext[offset : offset + 16].hex()
            for offset in range(0, len(plaintext), 16)
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
