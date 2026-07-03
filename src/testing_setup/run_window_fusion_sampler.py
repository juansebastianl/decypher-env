"""Sweep 21-block windows, collect shared-key samples, and fuse key marginals.

This script treats each ciphertext block as an independent factor on the shared
AES-XTS key. Samples from each 21-block window contribute key-bit marginals to
every block covered by that window; final key estimation fuses per-block
log-odds by summing them across blocks.

Example smoke run with random starts:
    python src/testing_setup/run_window_fusion_sampler.py --max-windows 2

Example fixture warm-start sanity check:
    python src/testing_setup/run_window_fusion_sampler.py --max-windows 2 --warm-start-fixture

Example larger run:
    python src/testing_setup/run_window_fusion_sampler.py \\
      --samples-per-window 100 --epochs 100 --sweeps-per-epoch 1000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decoder.legacy.backends.base import ParallelTemperingConfig
from src.decoder.legacy.backends.native import (
    NativeAlgebraicRelaxedEngine,
    NativeContinuousRelaxedEngine,
    NativeParallelTemperingEngine,
)
from src.decoder.circuit import ConstraintKind
from src.decoder.legacy.sampler import SamplerConfig, XtsSampler
from src.decoder.xts_model import load_fixture


FIXTURE_METADATA = Path(__file__).resolve().parent / "lorem_ipsum_aes_xts.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "fusion_runs"
KEY_BYTES = 64
KEY_BITS = KEY_BYTES * 8
BYTE_BITS = tuple(
    tuple((value >> shift) & 1 for shift in range(7, -1, -1))
    for value in range(256)
)


@dataclass
class BlockMarginal:
    samples: int
    ones: list[int]


@dataclass
class WindowProfiler:
    enabled: bool
    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def time(self, label: str):
        if not self.enabled:
            yield
            return
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[label] = self.seconds.get(label, 0.0) + (time.perf_counter() - started_at)
            self.counts[label] = self.counts.get(label, 0) + 1

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            "seconds": dict(self.seconds),
            "counts": dict(self.counts),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep shared-key sampler windows and fuse per-block key marginals."
    )
    parser.add_argument(
        "--benchmark-preset",
        choices=("micro", "single-window", "analytics-off"),
        default=None,
        help="Apply a repeatable benchmark configuration before explicit CLI overrides.",
    )
    parser.add_argument("--metadata", type=Path, default=FIXTURE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=21)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--samples-per-window", type=int, default=1)
    parser.add_argument(
        "--native-engine",
        choices=("native-pt", "native-continuous-relaxed", "native-algebraic-relaxed"),
        default="native-pt",
        help="Native engine implementation to use for each lowered fusion window.",
    )
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0, help="OpenMP worker threads. 0 uses up to replica count.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sweeps-per-epoch", type=int, default=1)
    parser.add_argument("--t-min", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=20.0)
    parser.add_argument("--goal-ramp", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--ascii-weight", type=float, default=1.0)
    parser.add_argument("--dual-eta", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument(
        "--lambda-scale-cold",
        type=float,
        default=1.0,
        help="Constraint-penalty multiplier applied to the coldest replica (>=1 enforces harder).",
    )
    parser.add_argument(
        "--lambda-scale-hot",
        type=float,
        default=1.0,
        help="Constraint-penalty multiplier applied to the hottest replica (<1 lets it explore freely).",
    )
    parser.add_argument(
        "--repair-move-prob",
        type=float,
        default=0.0,
        help=(
            "Probability per step of a constraint-repair move that sets a block's "
            "plaintext to the AES-XTS decryption of its target ciphertext under the "
            "current key, zeroing all goal constraints for that block at once."
        ),
    )
    parser.add_argument(
        "--key-gibbs-prob",
        type=float,
        default=0.0,
        help=(
            "Probability per step (native-continuous-relaxed only) of an exact key "
            "Gibbs move. For aes_rounds=1 this favours the joint (block) move over "
            "the S-box-coupled key1 byte pair (rk0[dst], rk1[src]) that co-determine "
            "a plaintext byte, which single-byte/bit moves cannot satisfy jointly; "
            "the remainder is a byte-wise Gibbs move (also covering key2). 0.0 "
            "reproduces the old single-flip engine. Use a multi-block, multi-sector "
            "window so the conditionals are identifiable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aes-rounds", type=int, default=14, help="AES rounds for lowered positive-control circuits.")
    parser.add_argument("--ladder-mode", choices=("geometric", "feedback"), default="geometric")
    parser.add_argument("--ladder-adapt-interval-epochs", type=int, default=5)
    parser.add_argument("--ladder-burn-in-epochs", type=int, default=5)
    parser.add_argument("--ladder-min-round-trips", type=int, default=1)
    parser.add_argument("--dual-mode", choices=("fixed", "scheduled"), default="fixed")
    parser.add_argument("--scheduled-rho-initial", type=float, default=0.1)
    parser.add_argument("--scheduled-eta-target", type=float, default=1e-3)
    parser.add_argument("--scheduled-rho-growth", type=float, default=2.0)
    parser.add_argument("--scheduled-violation-shrink", type=float, default=0.75)
    parser.add_argument("--algebra-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bp-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--alternative-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bp-iterations", type=int, default=20)
    parser.add_argument("--bp-damping", type=float, default=0.35)
    parser.add_argument("--bp-tolerance", type=float, default=1e-4)
    parser.add_argument("--bp-proposal-weight", type=float, default=0.25)
    parser.add_argument("--bethe-weight", type=float, default=0.0)
    parser.add_argument("--algebraic-newton-prob", type=float, default=0.0)
    parser.add_argument("--survey-restarts", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Beta prior smoothing for bit marginals.")
    parser.add_argument("--marginal-min-distinct-keys", type=int, default=25)
    parser.add_argument(
        "--positive-control-solve-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preset for reduced-round controls: one block, stronger goal pressure, and scheduled goal updates.",
    )
    parser.add_argument("--logit-clip", type=float, default=20.0)
    parser.add_argument(
        "--flush-every-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flush samples and metrics JSONL after each window for live debugging.",
    )
    parser.add_argument(
        "--epoch-log-interval",
        type=int,
        default=1,
        help="Write/print live epoch analytics every N epochs. Use 0 to disable epoch-level logging.",
    )
    parser.add_argument(
        "--epoch-log-mode",
        choices=("basic", "native-summary", "detailed"),
        default="basic",
        help="Control epoch JSONL detail. detailed exports full residuals to Python.",
    )
    parser.add_argument(
        "--warm-start-fixture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to starting each window from fixture plaintext/key metadata. This leaks the fixture key.",
    )
    parser.add_argument(
        "--fixed-key",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Seed each window's initial key from the fixture metadata. This leaks "
            "the true key and turns the run into a positive control. Off by default, "
            "so the key starts random and must be searched."
        ),
    )
    parser.add_argument(
        "--profile-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write Python/native timing deltas to sampler_profile.jsonl for optimization planning.",
    )
    parser.add_argument(
        "--validate-native-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug mode: sync current native assignment and compare epoch residuals with Python.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture = load_fixture(args.metadata)
    if fixture.key1 is None or fixture.key2 is None:
        raise ValueError("fixture metadata must include key_hex to compare true key")

    total_blocks = len(fixture.ciphertext) // 16
    starts = list(_window_starts(args.start_block, total_blocks, args.window_size, args.stride))
    if args.max_windows is not None:
        starts = starts[: args.max_windows]

    _warn_identifiability(args, starts)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "samples.jsonl"
    metrics_path = run_dir / "window_metrics.jsonl"
    epoch_metrics_path = run_dir / "epoch_metrics.jsonl"
    profile_path = run_dir / "sampler_profile.jsonl"
    marginals_path = run_dir / "per_block_marginals.jsonl"
    summary_path = run_dir / "fusion_summary.json"

    marginals = [BlockMarginal(samples=0, ones=[0] * KEY_BITS) for _ in range(total_blocks)]
    seen_sample_keys: set[str] = set()
    total_samples = 0
    accepted_state_marginal_fallback_windows = 0
    accepted_state_marginal_fallback_visits = 0
    latest_native_metrics: dict[str, object] = {}

    run_start = time.perf_counter()
    with (
        samples_path.open("w", encoding="utf-8") as samples_file,
        metrics_path.open("w", encoding="utf-8") as metrics_file,
        epoch_metrics_path.open("w", encoding="utf-8") as epoch_metrics_file,
        profile_path.open("w", encoding="utf-8") as profile_file,
    ):
        for window_number, start in enumerate(starts):
            window_start = time.perf_counter()
            seed = None if args.seed is None else args.seed + window_number
            samples, metrics = _run_window(
                args,
                start,
                seed,
                window_number=window_number,
                total_windows=len(starts),
                run_start=run_start,
                epoch_metrics_file=epoch_metrics_file,
                profile_file=profile_file,
            )
            window_elapsed = time.perf_counter() - window_start
            metrics["window_number"] = window_number
            metrics["start_block"] = start
            metrics["window_size"] = args.window_size
            metrics["samples_collected"] = len(samples)
            metrics["window_elapsed_seconds"] = window_elapsed
            metrics["run_elapsed_seconds"] = time.perf_counter() - run_start
            fallback_visits = 0
            # Only fold accepted-state (non-harvested) marginals into the fused key
            # when the window is trustworthy. marginal_trusted now requires that at
            # least one exact feasible solution was harvested, so a window that
            # merely wandered an infeasible-but-confident basin no longer injects a
            # confident-yet-wrong key into the fusion.
            if not samples and metrics.get("marginal_trusted"):
                fallback_visits = _add_native_key_marginals_to_blocks(
                    marginals,
                    start,
                    args.window_size,
                    metrics,
                )
                if fallback_visits:
                    accepted_state_marginal_fallback_windows += 1
                    accepted_state_marginal_fallback_visits += fallback_visits
            metrics["used_accepted_state_marginal_fallback"] = bool(fallback_visits)
            metrics["accepted_state_marginal_fallback_visits"] = fallback_visits
            latest_native_metrics = metrics
            metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")

            for sample in samples:
                key1 = bytes(sample.key1)
                key2 = bytes(sample.key2)
                key = key1 + key2
                key_hex = key.hex()
                sample_record = {
                    "window_number": window_number,
                    "start_block": start,
                    "window_size": args.window_size,
                    "key_hex": key_hex,
                    "key1_hex": key1.hex(),
                    "key2_hex": key2.hex(),
                    "plaintext_hex": bytes(sample.plaintext).hex(),
                    "is_new_key": key_hex not in seen_sample_keys,
                }
                samples_file.write(json.dumps(sample_record, sort_keys=True) + "\n")
                seen_sample_keys.add(key_hex)
                total_samples += 1
                _add_sample_to_block_marginals(marginals, start, args.window_size, key)

            if args.flush_every_window:
                samples_file.flush()
                metrics_file.flush()
                epoch_metrics_file.flush()
                profile_file.flush()

            residuals = metrics.get("residuals_by_class", {})
            proposals = metrics.get("proposal_attempts", {})
            accepts = metrics.get("proposal_accepts", {})
            print(
                f"window {window_number + 1}/{len(starts)} start={start} "
                f"samples={len(samples)} distinct_keys={len(seen_sample_keys)} "
                f"residuals={residuals} proposals={proposals} accepts={accepts} "
                f"elapsed={window_elapsed:.2f}s",
                flush=True,
            )

    fused_key, fused_log_odds = _fuse_key(marginals, alpha=args.alpha, logit_clip=args.logit_clip)
    true_key = fixture.key1 + fixture.key2
    summary = {
        "metadata": str(args.metadata),
        "windows_attempted": len(starts),
        "window_size": args.window_size,
        "stride": args.stride,
        "total_samples": total_samples,
        "distinct_sample_keys": len(seen_sample_keys),
        "blocks_with_samples": sum(1 for marginal in marginals if marginal.samples),
        "accepted_state_marginal_fallback_windows": accepted_state_marginal_fallback_windows,
        "accepted_state_marginal_fallback_visits": accepted_state_marginal_fallback_visits,
        "estimated_key_hex": fused_key.hex(),
        "true_key_hex": true_key.hex(),
        "matches_true_key": fused_key == true_key,
        "hamming_distance_to_true_key": _hamming(fused_key, true_key),
        "max_block_key_marginal_deviation": _max_block_key_marginal_deviation(marginals),
        "mean_block_key_marginal_deviation": _mean_block_key_marginal_deviation(marginals),
        "accepted_state_key_visits": latest_native_metrics.get("key_visit_count", 0),
        "accepted_state_key_distinct_count": latest_native_metrics.get("key_distinct_count", 0),
        "accepted_state_key_marginal_max_deviation": latest_native_metrics.get("key_marginal_max_deviation"),
        "accepted_state_key_information_bits": latest_native_metrics.get("key_information_bits"),
        "accepted_state_key_information_bits_raw": latest_native_metrics.get("key_information_bits_raw"),
        "accepted_state_key_information_null_bits": latest_native_metrics.get("key_information_null_bits"),
        "marginal_ess": latest_native_metrics.get("marginal_ess"),
        "marginal_rhat": latest_native_metrics.get("marginal_rhat"),
        "marginal_trusted": latest_native_metrics.get("marginal_trusted"),
        "marginal_diagnostic_note": latest_native_metrics.get("marginal_diagnostic_note"),
        "relative_log_z_estimate": latest_native_metrics.get("relative_log_z_estimate"),
        "log_z_note": latest_native_metrics.get("log_z_note"),
        "log_feasible_count_estimate": latest_native_metrics.get("log_feasible_count_estimate"),
        "samples_path": str(samples_path),
        "metrics_path": str(metrics_path),
        "epoch_metrics_path": str(epoch_metrics_path),
        "profile_enabled": args.profile_sampler,
        "profile_path": str(profile_path),
        "marginals_path": str(marginals_path),
    }

    with marginals_path.open("w", encoding="utf-8") as marginals_file:
        for block_index, marginal in enumerate(marginals):
            if not marginal.samples:
                continue
            marginals_file.write(
                json.dumps(
                    {
                        "block_index": block_index,
                        "samples": marginal.samples,
                        "log_odds": _block_log_odds(marginal, args.alpha, args.logit_clip),
                    }
                )
                + "\n"
            )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nFusion complete")
    print(f"Estimated key: {summary['estimated_key_hex']}")
    print(f"True key:      {summary['true_key_hex']}")
    print(f"Hamming distance: {summary['hamming_distance_to_true_key']}")
    print(f"Summary: {summary_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    # argparse falls back to sys.argv[1:] when argv is None, so explicit-option
    # detection must use the same source. Otherwise CLI invocations (argv is None)
    # see an empty "explicit" set and presets silently override user-provided flags
    # (e.g. --positive-control-solve-mode would force --window-size back to 1).
    explicit = _explicit_options(sys.argv[1:] if argv is None else argv)
    _apply_benchmark_preset(args, explicit)
    _apply_positive_control_preset(args, explicit)
    return args


def _explicit_options(argv: list[str]) -> set[str]:
    explicit: set[str] = set()
    for item in argv:
        if not item.startswith("--"):
            continue
        option = item.split("=", 1)[0]
        explicit.add(option)
        if option.startswith("--no-"):
            explicit.add("--" + option[5:])
    return explicit


def _set_if_implicit(args: argparse.Namespace, explicit: set[str], option: str, value) -> None:  # noqa: ANN001
    if f"--{option.replace('_', '-')}" not in explicit:
        setattr(args, option, value)


def _apply_benchmark_preset(args: argparse.Namespace, explicit: set[str]) -> None:
    if args.benchmark_preset == "micro":
        _set_if_implicit(args, explicit, "max_windows", 1)
        _set_if_implicit(args, explicit, "epochs", 1)
        _set_if_implicit(args, explicit, "sweeps_per_epoch", 1)
        _set_if_implicit(args, explicit, "replicas", 1)
        _set_if_implicit(args, explicit, "threads", 1)
        _set_if_implicit(args, explicit, "profile_sampler", True)
        _set_if_implicit(args, explicit, "epoch_log_interval", 1)
    elif args.benchmark_preset == "single-window":
        _set_if_implicit(args, explicit, "max_windows", 1)
        _set_if_implicit(args, explicit, "epochs", 5)
        _set_if_implicit(args, explicit, "sweeps_per_epoch", 1000)
        _set_if_implicit(args, explicit, "replicas", 16)
        _set_if_implicit(args, explicit, "threads", 16)
        _set_if_implicit(args, explicit, "profile_sampler", True)
    elif args.benchmark_preset == "analytics-off":
        _set_if_implicit(args, explicit, "epoch_log_interval", 0)
        _set_if_implicit(args, explicit, "profile_sampler", True)


def _apply_positive_control_preset(args: argparse.Namespace, explicit: set[str]) -> None:
    if not args.positive_control_solve_mode:
        return
    _set_if_implicit(args, explicit, "window_size", 1)
    _set_if_implicit(args, explicit, "samples_per_window", 1)
    _set_if_implicit(args, explicit, "goal_ramp", 5.0)
    _set_if_implicit(args, explicit, "mu", 1.0)
    _set_if_implicit(args, explicit, "dual_mode", "scheduled")
    _set_if_implicit(args, explicit, "scheduled_rho_initial", 1.0)
    _set_if_implicit(args, explicit, "scheduled_rho_growth", 2.0)
    _set_if_implicit(args, explicit, "t_min", 1.0)
    _set_if_implicit(args, explicit, "t_max", 50.0)
    _set_if_implicit(args, explicit, "lambda_scale_cold", 1.0)
    _set_if_implicit(args, explicit, "lambda_scale_hot", 0.1)
    _set_if_implicit(args, explicit, "repair_move_prob", 0.25)
    # Exact key Gibbs is essential: under repair the per-key energy is the ASCII
    # residual of the decryption, whose S-box avalanche makes single bit/byte
    # flips behave like a random walk. For one round this favours the joint
    # (rk0[dst], rk1[src]) block move that single-site moves cannot satisfy.
    _set_if_implicit(args, explicit, "key_gibbs_prob", 0.4)


def _run_window(
    args: argparse.Namespace,
    start_block: int,
    seed: int | None,
    *,
    window_number: int,
    total_windows: int,
    run_start: float,
    epoch_metrics_file,
    profile_file,
):
    profile = WindowProfiler(args.profile_sampler)
    engine_cls = _native_engine_class(args.native_engine)
    with profile.time("sampler_init"):
        sampler = XtsSampler(
            SamplerConfig(
                metadata_path=args.metadata,
                start_block=start_block,
                block_count=args.window_size,
                encoding="lowered",
                backend=args.native_engine,
                seed=seed,
                aes_rounds=args.aes_rounds,
                fixed_key=args.fixed_key,
            )
        )
    with profile.time("assignment_init"):
        assignment = sampler.known_fixture_assignment() if args.warm_start_fixture else sampler.random_assignment()
    with profile.time("config_init"):
        config = ParallelTemperingConfig(
            replicas=args.replicas,
            t_min=args.t_min,
            t_max=args.t_max,
            sweeps_per_epoch=args.sweeps_per_epoch,
            mu=args.mu,
            dual_eta=args.dual_eta,
            goal_weight=args.goal_ramp,
            consistency_weight=args.consistency_weight,
            ascii_weight=args.ascii_weight,
            seed=seed,
            threads=args.threads,
            profile=args.profile_sampler,
            validate_python_epoch=args.validate_native_epoch,
            ladder_mode=args.ladder_mode,
            ladder_adapt_interval_epochs=args.ladder_adapt_interval_epochs,
            ladder_burn_in_epochs=args.ladder_burn_in_epochs,
            ladder_min_round_trips=args.ladder_min_round_trips,
            dual_mode=args.dual_mode,
            scheduled_rho_initial=args.scheduled_rho_initial,
            scheduled_eta_target=args.scheduled_eta_target,
            scheduled_rho_growth=args.scheduled_rho_growth,
            scheduled_violation_shrink=args.scheduled_violation_shrink,
            algebra_diagnostics=args.algebra_diagnostics,
            bp_diagnostics=args.bp_diagnostics,
            alternative_diagnostics=args.alternative_diagnostics,
            marginal_alpha=args.alpha,
            marginal_min_distinct_keys=args.marginal_min_distinct_keys,
            lambda_scale_cold=args.lambda_scale_cold,
            lambda_scale_hot=args.lambda_scale_hot,
            aes_rounds=args.aes_rounds,
            repair_move_prob=args.repair_move_prob,
            key_gibbs_prob=args.key_gibbs_prob,
            bp_iterations=args.bp_iterations,
            bp_damping=args.bp_damping,
            bp_tolerance=args.bp_tolerance,
            bp_proposal_weight=args.bp_proposal_weight,
            bethe_weight=args.bethe_weight,
            algebraic_newton_prob=args.algebraic_newton_prob,
            survey_restarts=args.survey_restarts,
        )
    with profile.time("engine_init"):
        engine = engine_cls(sampler.circuit, assignment, config=config)
    samples = []
    window_start = time.perf_counter()
    with profile.time("initial_metrics"):
        latest_metrics = engine.metrics()
    native_profile = _native_profile(latest_metrics)
    if args.profile_sampler:
        setup_snapshot = profile.snapshot()
        _write_profile_record(
            profile_file,
            args=args,
            phase="window_setup",
            window_number=window_number,
            total_windows=total_windows,
            start_block=start_block,
            epoch=None,
            samples_collected=0,
            epoch_elapsed=0.0,
            window_elapsed=time.perf_counter() - window_start,
            run_elapsed=time.perf_counter() - run_start,
            python_delta_seconds=setup_snapshot["seconds"],
            python_delta_counts=setup_snapshot["counts"],
            python_cumulative_seconds=setup_snapshot["seconds"],
            python_cumulative_counts=setup_snapshot["counts"],
            native_delta=native_profile,
            native_cumulative=native_profile,
        )
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        python_before = profile.snapshot()
        native_before = native_profile
        with profile.time("run_epoch"):
            engine.run_epoch(args.sweeps_per_epoch)
        with profile.time("drain_feasible"):
            samples.extend(engine.drain_feasible(args.samples_per_window - len(samples)))
        with profile.time("metrics"):
            latest_metrics = engine.metrics()
        native_profile = _native_profile(latest_metrics)
        epoch_elapsed = time.perf_counter() - epoch_start
        if args.epoch_log_interval and (epoch + 1) % args.epoch_log_interval == 0:
            with profile.time("analytics_record"):
                record = _epoch_record(
                    args=args,
                    circuit=sampler.circuit,
                    engine=engine,
                    metrics=latest_metrics,
                    window_number=window_number,
                    total_windows=total_windows,
                    start_block=start_block,
                    epoch=epoch,
                    samples_collected=len(samples),
                    samples=samples,
                    epoch_elapsed=epoch_elapsed,
                    window_elapsed=time.perf_counter() - window_start,
                    run_elapsed=time.perf_counter() - run_start,
                )
            with profile.time("epoch_metrics_write"):
                epoch_metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                if args.flush_every_window:
                    epoch_metrics_file.flush()
            with profile.time("epoch_print"):
                print(_format_epoch_record(record), flush=True)
        if args.profile_sampler:
            python_after = profile.snapshot()
            _write_profile_record(
                profile_file,
                args=args,
                phase="epoch",
                window_number=window_number,
                total_windows=total_windows,
                start_block=start_block,
                epoch=epoch,
                samples_collected=len(samples),
                epoch_elapsed=epoch_elapsed,
                window_elapsed=time.perf_counter() - window_start,
                run_elapsed=time.perf_counter() - run_start,
                python_delta_seconds=_profile_delta(python_before["seconds"], python_after["seconds"]),
                python_delta_counts=_profile_delta(python_before["counts"], python_after["counts"]),
                python_cumulative_seconds=python_after["seconds"],
                python_cumulative_counts=python_after["counts"],
                native_delta=_profile_delta_nested(native_before, native_profile),
                native_cumulative=native_profile,
            )
            if args.flush_every_window:
                profile_file.flush()
        if len(samples) >= args.samples_per_window:
            break
    return samples, latest_metrics


def _native_engine_class(name: str):
    if name == "native-pt":
        return NativeParallelTemperingEngine
    if name == "native-continuous-relaxed":
        return NativeContinuousRelaxedEngine
    if name == "native-algebraic-relaxed":
        return NativeAlgebraicRelaxedEngine
    raise ValueError(f"unknown native engine: {name}")


def _native_profile(metrics: dict) -> dict[str, dict[str, float | int]]:
    return {
        "seconds": dict(metrics.get("profile_seconds", {})),
        "counts": dict(metrics.get("profile_counts", {})),
    }


def _profile_delta(
    before: dict[str, float | int],
    after: dict[str, float | int],
) -> dict[str, float | int]:
    deltas: dict[str, float | int] = {}
    for key in sorted(set(before) | set(after)):
        delta = after.get(key, 0) - before.get(key, 0)
        if delta:
            deltas[key] = delta
    return deltas


def _profile_delta_nested(
    before: dict[str, dict[str, float | int]],
    after: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    return {
        "seconds": _profile_delta(before.get("seconds", {}), after.get("seconds", {})),
        "counts": _profile_delta(before.get("counts", {}), after.get("counts", {})),
    }


def _write_profile_record(
    profile_file,
    *,
    args: argparse.Namespace,
    phase: str,
    window_number: int,
    total_windows: int,
    start_block: int,
    epoch: int | None,
    samples_collected: int,
    epoch_elapsed: float,
    window_elapsed: float,
    run_elapsed: float,
    python_delta_seconds: dict[str, float | int],
    python_delta_counts: dict[str, float | int],
    python_cumulative_seconds: dict[str, float | int],
    python_cumulative_counts: dict[str, float | int],
    native_delta: dict[str, dict[str, float | int]],
    native_cumulative: dict[str, dict[str, float | int]],
) -> None:
    record = {
        "phase": phase,
        "window_number": window_number,
        "total_windows": total_windows,
        "start_block": start_block,
        "window_size": args.window_size,
        "epoch": epoch,
        "epochs_requested": args.epochs,
        "samples_collected": samples_collected,
        "epoch_elapsed_seconds": epoch_elapsed,
        "window_elapsed_seconds": window_elapsed,
        "run_elapsed_seconds": run_elapsed,
        "sweeps_per_epoch": args.sweeps_per_epoch,
        "replicas": args.replicas,
        "threads": args.threads,
        "python_delta_seconds": python_delta_seconds,
        "python_delta_counts": python_delta_counts,
        "python_cumulative_seconds": python_cumulative_seconds,
        "python_cumulative_counts": python_cumulative_counts,
        "native_delta_seconds": native_delta.get("seconds", {}),
        "native_delta_counts": native_delta.get("counts", {}),
        "native_cumulative_seconds": native_cumulative.get("seconds", {}),
        "native_cumulative_counts": native_cumulative.get("counts", {}),
    }
    profile_file.write(json.dumps(record, sort_keys=True) + "\n")


def _epoch_record(
    *,
    args: argparse.Namespace,
    circuit,
    engine: NativeParallelTemperingEngine,
    metrics: dict,
    window_number: int,
    total_windows: int,
    start_block: int,
    epoch: int,
    samples_collected: int,
    samples: list,
    epoch_elapsed: float,
    window_elapsed: float,
    run_elapsed: float,
) -> dict[str, object]:
    if args.epoch_log_mode == "detailed":
        return _analytics_record(
            args=args,
            circuit=circuit,
            engine=engine,
            metrics=metrics,
            window_number=window_number,
            total_windows=total_windows,
            start_block=start_block,
            epoch=epoch,
            samples_collected=samples_collected,
            samples=samples,
            epoch_elapsed=epoch_elapsed,
            window_elapsed=window_elapsed,
            run_elapsed=run_elapsed,
        )
    proposal_attempts = metrics.get("proposal_attempts", {})
    proposal_accepts = metrics.get("proposal_accepts", {})
    residuals_by_class = metrics.get("residuals_by_class", {})
    total_residual = metrics.get("total_residual", sum(residuals_by_class.values()))
    record: dict[str, object] = {
        "window_number": window_number,
        "total_windows": total_windows,
        "start_block": start_block,
        "window_size": args.window_size,
        "epoch": epoch,
        "epochs_requested": args.epochs,
        "samples_collected": samples_collected,
        "epoch_elapsed_seconds": epoch_elapsed,
        "window_elapsed_seconds": window_elapsed,
        "run_elapsed_seconds": run_elapsed,
        "sweeps": metrics.get("sweeps", 0),
        "feasible_count_pending": metrics.get("feasible_count", 0),
        "total_residual": total_residual,
        "violations": metrics.get("violations", 0),
        "max_residual": metrics.get("max_residual", 0),
        "residuals_by_class": residuals_by_class,
        "multiplier_mean_by_class": metrics.get("multiplier_mean_by_class", {}),
        "multiplier_max_by_class": metrics.get("multiplier_max_by_class", {}),
        "proposal_attempts": proposal_attempts,
        "proposal_accepts": proposal_accepts,
        "proposal_acceptance_rates": _rates(proposal_attempts, proposal_accepts),
        "swap_attempts": metrics.get("swap_attempts", []),
        "swap_accepts": metrics.get("swap_accepts", []),
        "swap_acceptance_rates": _list_rates(metrics.get("swap_attempts", []), metrics.get("swap_accepts", [])),
        "sample_key_marginal_max_deviation": metrics.get(
            "key_marginal_max_deviation",
            _sample_key_marginal_max_deviation(samples),
        ),
        "key_visit_count": metrics.get("key_visit_count", 0),
        "key_distinct_count": metrics.get("key_distinct_count", 0),
        "key_information_bits": metrics.get("key_information_bits", 0.0),
        "key_information_bits_raw": metrics.get("key_information_bits_raw", 0.0),
        "key_information_null_bits": metrics.get("key_information_null_bits", 0.0),
        "marginal_ess": metrics.get("marginal_ess", 0.0),
        "marginal_rhat": metrics.get("marginal_rhat", 0.0),
        "marginal_trusted": metrics.get("marginal_trusted", False),
        "marginal_diagnostic_note": metrics.get("marginal_diagnostic_note"),
        "temperatures": metrics.get("temperatures", []),
        "energy_sample_counts": metrics.get("energy_sample_counts", []),
        "energy_mean_by_rung": metrics.get("energy_mean_by_rung", []),
        "energy_variance_by_rung": metrics.get("energy_variance_by_rung", []),
        "energy_temperatures_by_rung": metrics.get("energy_temperatures_by_rung", []),
        "total_round_trips": metrics.get("total_round_trips", 0),
        "relative_log_z_estimate": metrics.get("relative_log_z_estimate"),
        "log_z_note": metrics.get("log_z_note"),
        "log_feasible_count_estimate": metrics.get("log_feasible_count_estimate"),
        "rho_by_class": metrics.get("rho_by_class", {}),
        "rho_escalation_counts_by_class": metrics.get("rho_escalation_counts_by_class", {}),
        "infeasibility_suspected": metrics.get("infeasibility_suspected", False),
        "algebra_summary": metrics.get("algebra_summary", {}),
        "bp_available": metrics.get("bp_available", False),
        "bp_converged": metrics.get("bp_converged", False),
        "alternative_available": metrics.get("alternative_available", False),
        "alternative_log_z_estimates": metrics.get("alternative_log_z_estimates", []),
        "langevin_available": metrics.get("langevin_available", False),
        "langevin_seed_score": metrics.get("langevin_seed_score"),
    }
    if args.epoch_log_mode == "native-summary":
        record["native_summary"] = metrics.get("native_summary", {})
    return record


def _analytics_record(
    *,
    args: argparse.Namespace,
    circuit,
    engine: NativeParallelTemperingEngine,
    metrics: dict,
    window_number: int,
    total_windows: int,
    start_block: int,
    epoch: int,
    samples_collected: int,
    samples: list,
    epoch_elapsed: float,
    window_elapsed: float,
    run_elapsed: float,
) -> dict[str, object]:
    residuals = list(engine.residuals())
    per_block = _per_block_residuals(circuit, residuals, start_block, args.window_size)
    proposal_attempts = metrics.get("proposal_attempts", {})
    proposal_accepts = metrics.get("proposal_accepts", {})
    swap_attempts = metrics.get("swap_attempts", [])
    swap_accepts = metrics.get("swap_accepts", [])
    return {
        "window_number": window_number,
        "total_windows": total_windows,
        "start_block": start_block,
        "window_size": args.window_size,
        "epoch": epoch,
        "epochs_requested": args.epochs,
        "samples_collected": samples_collected,
        "epoch_elapsed_seconds": epoch_elapsed,
        "window_elapsed_seconds": window_elapsed,
        "run_elapsed_seconds": run_elapsed,
        "sweeps": metrics.get("sweeps", 0),
        "feasible_count_pending": metrics.get("feasible_count", 0),
        "total_residual": sum(residuals),
        "violations": sum(1 for residual in residuals if residual),
        "max_residual": max(residuals, default=0),
        "residuals_by_class": metrics.get("residuals_by_class", {}),
        "multiplier_mean_by_class": metrics.get("multiplier_mean_by_class", {}),
        "multiplier_max_by_class": metrics.get("multiplier_max_by_class", {}),
        "proposal_attempts": proposal_attempts,
        "proposal_accepts": proposal_accepts,
        "proposal_acceptance_rates": _rates(proposal_attempts, proposal_accepts),
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_accepts,
        "swap_acceptance_rates": _list_rates(swap_attempts, swap_accepts),
        "per_block_residuals": per_block,
        "top_blocks_by_total_residual": sorted(
            per_block,
            key=lambda item: item["total"],
            reverse=True,
        )[:5],
        "sample_key_marginal_max_deviation": metrics.get(
            "key_marginal_max_deviation",
            _sample_key_marginal_max_deviation(samples),
        ),
        "key_visit_count": metrics.get("key_visit_count", 0),
        "key_distinct_count": metrics.get("key_distinct_count", 0),
        "key_information_bits": metrics.get("key_information_bits", 0.0),
        "key_information_bits_raw": metrics.get("key_information_bits_raw", 0.0),
        "key_information_null_bits": metrics.get("key_information_null_bits", 0.0),
        "marginal_ess": metrics.get("marginal_ess", 0.0),
        "marginal_rhat": metrics.get("marginal_rhat", 0.0),
        "marginal_trusted": metrics.get("marginal_trusted", False),
        "marginal_diagnostic_note": metrics.get("marginal_diagnostic_note"),
        "temperatures": metrics.get("temperatures", []),
        "energy_sample_counts": metrics.get("energy_sample_counts", []),
        "energy_mean_by_rung": metrics.get("energy_mean_by_rung", []),
        "energy_variance_by_rung": metrics.get("energy_variance_by_rung", []),
        "energy_temperatures_by_rung": metrics.get("energy_temperatures_by_rung", []),
        "total_round_trips": metrics.get("total_round_trips", 0),
        "relative_log_z_estimate": metrics.get("relative_log_z_estimate"),
        "log_z_note": metrics.get("log_z_note"),
        "log_feasible_count_estimate": metrics.get("log_feasible_count_estimate"),
        "rho_by_class": metrics.get("rho_by_class", {}),
        "rho_escalation_counts_by_class": metrics.get("rho_escalation_counts_by_class", {}),
        "infeasibility_suspected": metrics.get("infeasibility_suspected", False),
        "algebra_summary": metrics.get("algebra_summary", {}),
        "bp_available": metrics.get("bp_available", False),
        "bp_converged": metrics.get("bp_converged", False),
        "alternative_available": metrics.get("alternative_available", False),
        "alternative_log_z_estimates": metrics.get("alternative_log_z_estimates", []),
        "langevin_available": metrics.get("langevin_available", False),
        "langevin_seed_score": metrics.get("langevin_seed_score"),
    }


def _format_epoch_record(record: dict[str, object]) -> str:
    residuals = record["residuals_by_class"]
    multipliers = record["multiplier_mean_by_class"]
    rates = record["proposal_acceptance_rates"]
    return (
        f"  epoch {record['epoch'] + 1}/{record['epochs_requested']} "
        f"window {record['window_number'] + 1}/{record['total_windows']} "
        f"samples={record['samples_collected']} "
        f"total_residual={record['total_residual']} violations={record['violations']} "
        f"class_residuals={residuals} multipliers={multipliers} proposal_rates={rates} "
        f"key_marginal_max_dev={record['sample_key_marginal_max_deviation']} "
        f"key_info_bits={record.get('key_information_bits')} distinct_keys={record.get('key_distinct_count')} "
        f"trusted={record.get('marginal_trusted')} "
        f"elapsed={record['epoch_elapsed_seconds']:.2f}s"
    )


def _window_starts(start_block: int, total_blocks: int, window_size: int, stride: int):
    if window_size < 1:
        raise ValueError("window size must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    last_start = total_blocks - window_size
    for start in range(start_block, last_start + 1, stride):
        yield start


def _warn_identifiability(args: argparse.Namespace, starts: list[int]) -> None:
    """Print up-front advisories about what the chosen windows can identify.

    Two structural limits are easy to trip and silently cap how much of the key
    can ever be recovered, regardless of sampler quality:

    * key2 / XTS tweak: within a single sector every block shares one initial
      tweak T0, and for one round any rk0_2 can be matched by an rk1_2 that
      reproduces T0 -- a 2**128 equivalence class. key2 is only pinned once the
      windows jointly span >= 2 sectors.
    * ASCII-printability is a weak prior: real text is low entropy and printable
      XOR-shifts overlap, so a key1 byte-pair position can have dozens of
      feasible values until many *distinct* plaintext blocks constrain it.
    """
    try:
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        sector_bytes = int(metadata.get("sector_bytes", 512))
    except (OSError, ValueError):
        sector_bytes = 512
    blocks_per_sector = max(1, sector_bytes // 16)

    covered_blocks = {b for start in starts for b in range(start, start + args.window_size)}
    per_window_sectors = [
        len({b // blocks_per_sector for b in range(start, start + args.window_size)}) for start in starts
    ]
    covered_sectors = {b // blocks_per_sector for b in covered_blocks}

    # Rough identifiability bound for 1-round AES-XTS. Free key bits the search
    # must pin: key1 = 256, plus the XTS tweak. A single sector only exposes the
    # 128-bit initial tweak T0 (key2 beyond T0 is a free 2**128 class); >= 2
    # sectors make all 256 bits of key2 relevant. Each block contributes 16
    # ASCII-printable bytes, each worth ~log2(256/98) ~ 1.38 bits, so ~22 bits of
    # constraint per *distinct, non-repeating* block. A unique true key needs the
    # constraint to exceed the free bits.
    free_key_bits = 256 + (256 if len(covered_sectors) >= 2 else 128)
    bits_per_block = 16 * 1.38
    min_blocks = math.ceil(free_key_bits / bits_per_block)
    distinct_blocks = len(covered_blocks)

    messages: list[str] = []
    if args.aes_rounds == 1 and len(covered_sectors) < 2:
        messages.append(
            f"key2 is UNRECOVERABLE here: all windows fall in a single sector "
            f"({blocks_per_sector} blocks/sector). Two compounding problems: (1) for "
            f"1-round AES-XTS key2 has a 2**128 equivalence class within one sector, "
            f"and (2) the key2/tweak profile-energy surface is a NEEDLE -- flat "
            f"everywhere except at the true tweak -- so PT/Gibbs has no gradient to "
            f"follow and the cold replica cannot descend (this is why even 16-31 "
            f"blocks never harvest). Use a window spanning >= 2 sectors, e.g. "
            f"--window-size {blocks_per_sector + 16} or start near a sector boundary; "
            f"only then does the surface acquire a usable gradient."
        )
    elif args.aes_rounds == 1 and max(per_window_sectors, default=1) < 2:
        messages.append(
            "key2 identifiability is split across windows: no single window spans 2 "
            "sectors, so each window individually sees the single-sector NEEDLE and "
            "cannot descend key2. Prefer a single window that crosses a sector "
            "boundary over many single-sector windows."
        )

    if args.aes_rounds == 1 and distinct_blocks < min_blocks:
        messages.append(
            f"under-determined: {distinct_blocks} distinct blocks give ~"
            f"{distinct_blocks * bits_per_block:.0f} bits of ASCII constraint vs "
            f"~{free_key_bits} free key bits, so feasible (all-printable) decryptions "
            f"exist for many WRONG keys. Expect spurious harvests -- a key that "
            f"decrypts to printable text but matches the true key in only a few bytes. "
            f"Use >= ~{min_blocks} distinct, non-repeating blocks across >= 2 sectors "
            f"for the true key to become the unique feasible solution."
        )
    elif args.aes_rounds == 1 and distinct_blocks < 2 * min_blocks:
        messages.append(
            f"thin margin: {distinct_blocks} blocks is near the ~{min_blocks}-block "
            f"identifiability floor. Low-entropy/repeated plaintext contributes less "
            f"than {bits_per_block:.0f} bits/block, so prefer a comfortable surplus "
            f"of distinct blocks to suppress spurious feasible keys."
        )

    if messages:
        print("[identifiability] " + "\n[identifiability] ".join(messages), file=sys.stderr)


def _per_block_residuals(circuit, residuals: list[int], start_block: int, window_size: int) -> list[dict[str, object]]:
    by_block = {
        block_index: {"block_index": block_index, "ascii": 0, "consistency": 0, "goal": 0, "total": 0}
        for block_index in range(start_block, start_block + window_size)
    }
    for constraint, residual in zip(circuit.constraints, residuals):
        if not residual:
            continue
        block_index = _constraint_block_index(constraint.label)
        if block_index is None or block_index not in by_block:
            continue
        if constraint.kind == ConstraintKind.ASCII_PRINTABLE:
            bucket = "ascii"
        elif constraint.kind == ConstraintKind.DEFINE8:
            bucket = "consistency"
        else:
            bucket = "goal"
        by_block[block_index][bucket] += residual
        by_block[block_index]["total"] += residual
    return list(by_block.values())


def _constraint_block_index(label: str) -> int | None:
    if not label:
        return None
    match = re.search(r"block[=_](\d+)", label)
    return int(match.group(1)) if match else None


def _rates(attempts: dict[str, int], accepts: dict[str, int]) -> dict[str, float | None]:
    return {
        key: (accepts.get(key, 0) / value if value else None)
        for key, value in attempts.items()
    }


def _list_rates(attempts: list[int], accepts: list[int]) -> list[float | None]:
    return [
        accept / attempt if attempt else None
        for attempt, accept in zip(attempts, accepts)
    ]


def _add_sample_to_block_marginals(
    marginals: list[BlockMarginal],
    start_block: int,
    window_size: int,
    key: bytes,
) -> None:
    bit_chunks = [BYTE_BITS[byte] for byte in key]
    for block_index in range(start_block, start_block + window_size):
        marginal = marginals[block_index]
        marginal.samples += 1
        bit_index = 0
        for bits in bit_chunks:
            for bit in bits:
                marginal.ones[bit_index] += bit
                bit_index += 1


def _add_native_key_marginals_to_blocks(
    marginals: list[BlockMarginal],
    start_block: int,
    window_size: int,
    metrics: dict[str, object],
) -> int:
    visits = int(metrics.get("key_visit_count") or 0)
    key_ones = metrics.get("key_ones") or []
    if visits <= 0 or not isinstance(key_ones, list) or len(key_ones) < KEY_BITS:
        return 0
    bounded_ones = [max(0, min(visits, int(value))) for value in key_ones[:KEY_BITS]]
    for block_index in range(start_block, start_block + window_size):
        marginal = marginals[block_index]
        marginal.samples += visits
        for bit_index, value in enumerate(bounded_ones):
            marginal.ones[bit_index] += value
    return visits


def _sample_key_marginal_max_deviation(samples: list) -> float | None:
    if not samples:
        return None
    ones = [0] * KEY_BITS
    for sample in samples:
        key = bytes(sample.key1) + bytes(sample.key2)
        bit_index = 0
        for byte in key:
            for bit in BYTE_BITS[byte]:
                ones[bit_index] += bit
                bit_index += 1
    count = len(samples)
    return max(abs((value / count) - 0.5) for value in ones)


def _max_block_key_marginal_deviation(marginals: list[BlockMarginal]) -> float | None:
    values = [
        max(abs((ones / marginal.samples) - 0.5) for ones in marginal.ones)
        for marginal in marginals
        if marginal.samples
    ]
    return max(values) if values else None


def _mean_block_key_marginal_deviation(marginals: list[BlockMarginal]) -> float | None:
    values = [
        abs((ones / marginal.samples) - 0.5)
        for marginal in marginals
        if marginal.samples
        for ones in marginal.ones
    ]
    return sum(values) / len(values) if values else None


def _fuse_key(
    marginals: list[BlockMarginal],
    *,
    alpha: float,
    logit_clip: float,
) -> tuple[bytes, list[float]]:
    fused_log_odds = [0.0] * KEY_BITS
    for marginal in marginals:
        if not marginal.samples:
            continue
        for bit_index, value in enumerate(_block_log_odds(marginal, alpha, logit_clip)):
            fused_log_odds[bit_index] += value
    return _bits_to_bytes([1 if value > 0 else 0 for value in fused_log_odds]), fused_log_odds


def _block_log_odds(marginal: BlockMarginal, alpha: float, logit_clip: float) -> list[float]:
    denominator = marginal.samples + 2.0 * alpha
    values = []
    for ones in marginal.ones:
        q = (ones + alpha) / denominator
        logit = math.log(q / (1.0 - q))
        values.append(max(-logit_clip, min(logit_clip, logit)))
    return values


def _bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def _bits_to_bytes(bits: list[int]) -> bytes:
    output = bytearray()
    for offset in range(0, len(bits), 8):
        value = 0
        for bit in bits[offset : offset + 8]:
            value = (value << 1) | bit
        output.append(value)
    return bytes(output)


def _hamming(left: bytes, right: bytes) -> int:
    return sum((a ^ b).bit_count() for a, b in zip(left, right))


if __name__ == "__main__":
    raise SystemExit(main())
