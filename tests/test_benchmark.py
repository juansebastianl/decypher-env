"""Benchmark regression test.

Runs the fast ``smoke`` suite and asserts the bundled solvers stay within
tolerance of the committed baseline (``benchmarks/baselines/smoke.json``). This
catches accidental reward/harness regressions in CI without needing the slower
``public`` / ``hard`` suites.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.decoder.rl.benchmark import (
    SUITES,
    _pass_at_k,
    baseline_path,
    compare_to_baseline,
    run_suite,
)
from src.decoder.rl.benchmark_viz import render_report_html, write_html
from src.decoder.rl.harness import harness_available

requires_toolchain = pytest.mark.skipif(
    not harness_available(), reason="C++ toolchain to build the harness is unavailable"
)


def test_pass_at_k_estimator() -> None:
    # No feasible samples -> 0; all feasible -> 1.
    assert _pass_at_k(5, 0, 1) == 0.0
    assert _pass_at_k(5, 5, 1) == 1.0
    # 1 of 4 feasible, k=1 -> 0.25 ; k=4 -> 1.0 (must draw the feasible one).
    assert _pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert _pass_at_k(4, 1, 4) == pytest.approx(1.0)
    # 2 of 4 feasible, k=2 -> 1 - C(2,2)/C(4,2) = 1 - 1/6.
    assert _pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)


@requires_toolchain
def test_smoke_suite_matches_baseline() -> None:
    report = run_suite(SUITES["smoke"])
    regressions = compare_to_baseline(report, tolerance=0.05)
    assert regressions == [], "\n".join(regressions)


@requires_toolchain
def test_smoke_suite_is_deterministic() -> None:
    first = run_suite(SUITES["smoke"])
    second = run_suite(SUITES["smoke"])
    first_rewards = [s["reward_mean"] for s in first["solvers"]]
    second_rewards = [s["reward_mean"] for s in second["solvers"]]
    assert first_rewards == second_rewards


def test_viz_renders_committed_baseline(tmp_path: Path) -> None:
    source = baseline_path("smoke")
    report = json.loads(source.read_text(encoding="utf-8"))

    page = render_report_html(report, source_name=source.name)
    assert "<svg" in page
    assert "Reward per solver" in page
    assert "pass@k" in page
    for solver in report["solvers"]:
        assert solver["solver"] in page

    output = write_html(source, tmp_path / "smoke.html")
    written = output.read_text(encoding="utf-8")
    assert "<svg" in written
    assert "Outcome rates" in written
    assert source.name in written
