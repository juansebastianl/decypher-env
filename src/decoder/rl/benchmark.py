"""Reproducible benchmark suites for the solver-authoring environment.

Run with::

    python -m src.decoder.rl.benchmark --suite smoke
    python -m src.decoder.rl.benchmark --suite public --compare-baseline
    python -m src.decoder.rl.benchmark --suite smoke --write-baseline

Three fixed suites are defined:

* ``smoke``  -- one 1-round instance, tiny budget; runs in seconds for CI.
* ``public`` -- a small multi-round ladder; the standard evaluation set.
* ``hard``   -- more rounds / blocks; the intentionally-hard evaluation set.

Each suite pins its ``TaskSpec`` set, seeds, rounds ladders and budget
(``HarnessOptions``) so a run is reproducible. Baselines are produced by scoring
the bundled example solvers (``examples/solvers/``) and recording their reward
distributions, compile-failure / timeout / crash rates and ``pass@k`` under
``benchmarks/baselines/<suite>.json``; ``--compare-baseline`` flags regressions
outside tolerance.

The bundled solvers are reference baselines, not solutions: the full task is
intentionally hard, so low feasibility on ``public``/``hard`` is expected.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .curriculum import TaskSpec
from .harness import HarnessOptions, harness_available
from .solver_env import SolverAuthoringEnv

_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = _ROOT / "src" / "testing_setup" / "lorem_ipsum_aes_xts.json"
EXAMPLES_DIR = _ROOT / "examples" / "solvers"
BASELINE_DIR = _ROOT / "benchmarks" / "baselines"

# The bundled example solvers used as regression baselines.
BUNDLED_SOLVERS = (
    "baseline",
    "random_metropolis",
    "continuous_relaxed",
    "algebraic_relaxed",
    "parallel_tempering",
)


@dataclass
class Suite:
    name: str
    specs: list[TaskSpec]
    options: HarnessOptions
    solvers: tuple[str, ...] = BUNDLED_SOLVERS


def _fast_smoke_options() -> HarnessOptions:
    return HarnessOptions(run_timeout=15.0, compile_timeout=60.0, epochs=1, sweeps_per_epoch=300)


def _build_suites() -> dict[str, Suite]:
    smoke = Suite(
        name="smoke",
        specs=[TaskSpec(metadata_path=FIXTURE, block_count=1, rounds_ladder=(1,), seed=7)],
        options=_fast_smoke_options(),
        solvers=("baseline", "random_metropolis"),
    )
    public = Suite(
        name="public",
        specs=[
            TaskSpec(metadata_path=FIXTURE, block_count=1, rounds_ladder=(1, 2), seed=seed)
            for seed in (1, 7, 13)
        ],
        options=HarnessOptions(run_timeout=20.0, compile_timeout=60.0, epochs=1, sweeps_per_epoch=2000),
    )
    hard = Suite(
        name="hard",
        specs=[
            TaskSpec(metadata_path=FIXTURE, block_count=1, rounds_ladder=(1, 2, 3), seed=seed)
            for seed in (3, 11)
        ],
        options=HarnessOptions(run_timeout=30.0, compile_timeout=60.0, epochs=2, sweeps_per_epoch=4000),
    )
    return {"smoke": smoke, "public": public, "hard": hard}


SUITES = _build_suites()


def _example_source(name: str) -> str:
    return (EXAMPLES_DIR / name / "solver.cpp").read_text(encoding="utf-8")


def _pass_at_k(num_samples: int, num_feasible: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c, k)/C(n, k).

    Treats the bundled solvers as ``n`` samples per instance, ``c`` of which are
    feasible, and estimates the chance that at least one of ``k`` drawn samples
    is feasible.
    """
    if k <= 0 or num_samples <= 0:
        return 0.0
    if num_samples - num_feasible < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_feasible, k) / math.comb(num_samples, k)


@dataclass
class SolverReport:
    solver: str
    rewards: list[float] = field(default_factory=list)
    feasible: list[bool] = field(default_factory=list)
    compile_ok: list[bool] = field(default_factory=list)
    timed_out: list[bool] = field(default_factory=list)
    crashed: list[bool] = field(default_factory=list)

    def summary(self) -> dict:
        n = len(self.rewards)
        return {
            "solver": self.solver,
            "instances": n,
            "reward_mean": statistics.fmean(self.rewards) if self.rewards else 0.0,
            "reward_min": min(self.rewards) if self.rewards else 0.0,
            "reward_max": max(self.rewards) if self.rewards else 0.0,
            "reward_median": statistics.median(self.rewards) if self.rewards else 0.0,
            "compile_failure_rate": _rate(self.compile_ok, want=False),
            "timeout_rate": _rate(self.timed_out, want=True),
            "crash_rate": _rate(self.crashed, want=True),
            "feasible_rate": _rate(self.feasible, want=True),  # pass@1
        }


def _rate(flags: list[bool], want: bool) -> float:
    if not flags:
        return 0.0
    return sum(1 for f in flags if bool(f) == want) / len(flags)


def run_suite(suite: Suite, solvers: tuple[str, ...] | None = None) -> dict:
    """Score the given (or the suite's default) bundled solvers over the suite."""
    solvers = solvers or suite.solvers
    env = SolverAuthoringEnv(task_specs=suite.specs, options=suite.options)

    reports: list[SolverReport] = []
    # feasibility[instance] = number of solvers feasible on that instance.
    feasible_per_instance = [0] * len(suite.specs)
    for solver in solvers:
        source = _example_source(solver)
        report = SolverReport(solver=solver)
        results = env.score_batch([source] * len(suite.specs), suite.specs)
        for index, result in enumerate(results):
            report.rewards.append(result.reward)
            report.compile_ok.append(result.compile_ok)
            report.feasible.append(result.feasible)
            # Per-run timeout/crash: true if every rung timed out / crashed.
            report.timed_out.append(
                bool(result.per_task) and all(t["timed_out"] for t in result.per_task)
            )
            report.crashed.append(
                bool(result.per_task) and all(t["crashed"] or not t["ran"] for t in result.per_task)
            )
            if result.feasible:
                feasible_per_instance[index] += 1
        reports.append(report)

    n = len(solvers)
    pass_at_k = {
        f"pass@{k}": statistics.fmean(
            [_pass_at_k(n, c, k) for c in feasible_per_instance]
        ) if feasible_per_instance else 0.0
        for k in range(1, n + 1)
    }
    return {
        "suite": suite.name,
        "num_instances": len(suite.specs),
        "solvers": [r.summary() for r in reports],
        "pass_at_k": pass_at_k,
    }


def baseline_path(suite_name: str) -> Path:
    return BASELINE_DIR / f"{suite_name}.json"


def write_baseline(suite_name: str) -> Path:
    suite = SUITES[suite_name]
    report = run_suite(suite)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = baseline_path(suite_name)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def compare_to_baseline(report: dict, tolerance: float = 0.05) -> list[str]:
    """Return a list of regression messages (empty == within tolerance)."""
    path = baseline_path(report["suite"])
    if not path.exists():
        return [f"no baseline at {path}; run with --write-baseline first"]
    baseline = json.loads(path.read_text(encoding="utf-8"))
    regressions: list[str] = []
    base_by_solver = {s["solver"]: s for s in baseline["solvers"]}
    for solver in report["solvers"]:
        base = base_by_solver.get(solver["solver"])
        if base is None:
            continue
        delta = solver["reward_mean"] - base["reward_mean"]
        if delta < -tolerance:
            regressions.append(
                f"{solver['solver']}: reward_mean {solver['reward_mean']:.4f} "
                f"regressed {delta:+.4f} vs baseline {base['reward_mean']:.4f}"
            )
    return regressions


def _format_report(report: dict) -> str:
    lines = [f"suite={report['suite']} instances={report['num_instances']}"]
    for solver in report["solvers"]:
        lines.append(
            "  {solver:20s} reward mean={reward_mean:+.4f} "
            "min={reward_min:+.4f} max={reward_max:+.4f} "
            "compile_fail={compile_failure_rate:.2f} timeout={timeout_rate:.2f} "
            "crash={crash_rate:.2f} feasible={feasible_rate:.2f}".format(**solver)
        )
    passk = " ".join(f"{k}={v:.3f}" for k, v in report["pass_at_k"].items())
    lines.append(f"  pass@k: {passk}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the solver-authoring benchmark suites.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--write-baseline", action="store_true", help="record the current run as the baseline")
    parser.add_argument("--compare-baseline", action="store_true", help="flag regressions vs the committed baseline")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", help="print the raw JSON report")
    args = parser.parse_args(argv)

    if not harness_available():
        parser.error("C++ toolchain to build the harness is unavailable")

    if args.write_baseline:
        path = write_baseline(args.suite)
        print(f"wrote baseline to {path}")
        return 0

    report = run_suite(SUITES[args.suite])
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))

    if args.compare_baseline:
        regressions = compare_to_baseline(report, tolerance=args.tolerance)
        if regressions:
            print("\nREGRESSIONS:")
            for line in regressions:
                print(f"  - {line}")
            return 1
        print("\nno regressions vs baseline")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
