"""Render benchmark reports (``benchmark.py`` JSON) as self-contained HTML.

Run with::

    python -m src.decoder.rl.benchmark_viz benchmarks/baselines/hard.json
    python -m src.decoder.rl.benchmark_viz report.json -o report.html
    python -m src.decoder.rl.benchmark_viz --all-baselines

The output is a single HTML file with no external dependencies (inline CSS +
SVG): a reward distribution chart per solver (mean bar, min--max whisker,
median tick), a failure/feasibility rate table with inline bars, the pass@k
curve, and the raw JSON for reference. ``--all-baselines`` renders a
``<suite>.html`` next to every ``<suite>.json`` under ``benchmarks/baselines/``.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = _ROOT / "benchmarks" / "baselines"

_PALETTE = ("#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#eeca3b")

_CSS = """
:root {
  --ink: #24292f; --muted: #57606a; --border: #d0d7de; --bg: #ffffff;
  --panel: #f6f8fa; --accent: #4c78a8; --good: #54a24b; --bad: #e45756;
}
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 880px; padding: 24px 16px 64px;
       font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--ink); background: var(--bg); }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 8px; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
.panel { background: var(--panel); border: 1px solid var(--border);
         border-radius: 8px; padding: 16px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
.rate-cell { min-width: 130px; }
.rate-bar { display: inline-block; height: 10px; border-radius: 2px;
            background: var(--accent); vertical-align: middle; margin-right: 6px; }
.rate-bar.bad { background: var(--bad); }
.rate-bar.good { background: var(--good); }
.legend { font-size: 13px; color: var(--muted); margin: 4px 0 0; }
.legend span { display: inline-block; margin-right: 16px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
          margin-right: 4px; vertical-align: baseline; }
details { margin-top: 32px; }
summary { cursor: pointer; color: var(--muted); }
pre { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
      padding: 12px; overflow-x: auto; font-size: 12px; }
svg text { font: 12px -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
           fill: var(--ink); }
svg .tick text { fill: var(--muted); }
"""


# ---------------------------------------------------------------------------
# Small SVG helpers
# ---------------------------------------------------------------------------


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick positions covering [lo, hi] with a 1/2/2.5/5 step."""
    if hi <= lo:
        hi = lo + 1.0
    raw_step = (hi - lo) / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if step >= raw_step:
            break
    first = math.floor(lo / step) * step
    ticks = []
    value = first
    while value <= hi + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt(value: float) -> str:
    return f"{value:+.3f}" if abs(value) < 100 else f"{value:+.1f}"


def _reward_chart(solvers: list[dict]) -> str:
    """Horizontal bars of reward_mean with min--max whiskers and median ticks."""
    left, right, row_h, axis_h = 170, 70, 34, 28
    plot_w = 620
    height = axis_h + row_h * len(solvers) + 8
    lo = min(0.0, *(s["reward_min"] for s in solvers))
    hi = max(0.0, *(s["reward_max"] for s in solvers))
    pad = (hi - lo) * 0.06 or 0.5
    lo, hi = lo - pad, hi + pad

    def x(value: float) -> float:
        return left + (value - lo) / (hi - lo) * plot_w

    parts = [
        f'<svg viewBox="0 0 {left + plot_w + right} {height}" role="img" '
        f'aria-label="Reward per solver">'
    ]
    for tick in _nice_ticks(lo, hi):
        if not lo <= tick <= hi:
            continue
        tx = x(tick)
        parts.append(
            f'<g class="tick"><line x1="{tx:.1f}" y1="{axis_h - 6}" x2="{tx:.1f}" '
            f'y2="{height - 8}" stroke="#d0d7de" stroke-width="1"/>'
            f'<text x="{tx:.1f}" y="{axis_h - 12}" text-anchor="middle">{tick:g}</text></g>'
        )
    zero_x = x(0.0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{axis_h - 6}" x2="{zero_x:.1f}" '
        f'y2="{height - 8}" stroke="#57606a" stroke-width="1"/>'
    )
    for row, solver in enumerate(solvers):
        y = axis_h + row * row_h
        mid = y + row_h / 2
        color = _PALETTE[row % len(_PALETTE)]
        name = html.escape(solver["solver"])
        mean, s_min, s_max = solver["reward_mean"], solver["reward_min"], solver["reward_max"]
        median = solver.get("reward_median", mean)
        bar_x0, bar_x1 = sorted((x(0.0), x(mean)))
        parts.append(f'<text x="{left - 10}" y="{mid + 4:.1f}" text-anchor="end">{name}</text>')
        parts.append(
            f'<rect x="{bar_x0:.1f}" y="{mid - 9:.1f}" width="{max(bar_x1 - bar_x0, 0.5):.1f}" '
            f'height="18" rx="2" fill="{color}" fill-opacity="0.85"/>'
        )
        parts.append(
            f'<line x1="{x(s_min):.1f}" y1="{mid:.1f}" x2="{x(s_max):.1f}" y2="{mid:.1f}" '
            f'stroke="#24292f" stroke-width="1.5"/>'
            f'<line x1="{x(s_min):.1f}" y1="{mid - 5:.1f}" x2="{x(s_min):.1f}" y2="{mid + 5:.1f}" '
            f'stroke="#24292f" stroke-width="1.5"/>'
            f'<line x1="{x(s_max):.1f}" y1="{mid - 5:.1f}" x2="{x(s_max):.1f}" y2="{mid + 5:.1f}" '
            f'stroke="#24292f" stroke-width="1.5"/>'
        )
        parts.append(
            f'<line x1="{x(median):.1f}" y1="{mid - 9:.1f}" x2="{x(median):.1f}" '
            f'y2="{mid + 9:.1f}" stroke="#ffffff" stroke-width="2"/>'
        )
        label_x = x(s_max) + 8
        parts.append(f'<text x="{label_x:.1f}" y="{mid + 4:.1f}">{_fmt(mean)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _pass_at_k_chart(pass_at_k: dict[str, float]) -> str:
    """Vertical bars of the pass@k estimate, k = 1..n."""
    entries = sorted(pass_at_k.items(), key=lambda kv: int(kv[0].split("@")[1]))
    left, bottom, top = 46, 30, 12
    bar_w, gap, plot_h = 56, 22, 160
    width = left + len(entries) * (bar_w + gap) + 12
    height = top + plot_h + bottom

    def y(value: float) -> float:
        return top + (1.0 - value) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="pass@k">']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        ty = y(tick)
        parts.append(
            f'<g class="tick"><line x1="{left}" y1="{ty:.1f}" x2="{width - 8}" y2="{ty:.1f}" '
            f'stroke="#d0d7de" stroke-width="1"/>'
            f'<text x="{left - 6}" y="{ty + 4:.1f}" text-anchor="end">{tick:g}</text></g>'
        )
    for index, (label, value) in enumerate(entries):
        bx = left + index * (bar_w + gap) + gap / 2
        by = y(value)
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" '
            f'height="{max(top + plot_h - by, 0.5):.1f}" rx="2" fill="#4c78a8"/>'
            f'<text x="{bx + bar_w / 2:.1f}" y="{by - 5:.1f}" text-anchor="middle">{value:.2f}</text>'
            f'<text x="{bx + bar_w / 2:.1f}" y="{height - 10}" text-anchor="middle">'
            f'{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _rate_cell(value: float, kind: str = "") -> str:
    width = round(max(0.0, min(1.0, value)) * 90)
    css = f"rate-bar {kind}".strip()
    return (
        f'<td class="rate-cell"><span class="{css}" style="width:{width}px"></span>'
        f"{value * 100:.0f}%</td>"
    )


def _solver_table(solvers: list[dict]) -> str:
    rows = []
    for solver in solvers:
        rows.append(
            "<tr>"
            f"<td>{html.escape(solver['solver'])}</td>"
            f"<td class=\"num\">{solver['reward_mean']:+.4f}</td>"
            f"<td class=\"num\">{solver['reward_median']:+.4f}</td>"
            f"<td class=\"num\">{solver['reward_min']:+.4f}</td>"
            f"<td class=\"num\">{solver['reward_max']:+.4f}</td>"
            + _rate_cell(solver["feasible_rate"], "good")
            + _rate_cell(solver["compile_failure_rate"], "bad")
            + _rate_cell(solver["timeout_rate"], "bad")
            + _rate_cell(solver["crash_rate"], "bad")
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>solver</th><th>mean</th><th>median</th><th>min</th><th>max</th>"
        "<th>feasible</th><th>compile fail</th><th>timeout</th><th>crash</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------


def render_report_html(report: dict, source_name: str = "") -> str:
    """Render one benchmark report dict into a standalone HTML page."""
    solvers = report.get("solvers", [])
    suite = html.escape(str(report.get("suite", "unknown")))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    source = f" &middot; source: <code>{html.escape(source_name)}</code>" if source_name else ""

    sections = [
        f"<h1>Benchmark report — <code>{suite}</code> suite</h1>",
        f'<div class="meta">{report.get("num_instances", 0)} instance(s) &middot; '
        f"{len(solvers)} bundled solver(s) &middot; generated {generated}{source}</div>",
    ]
    if solvers:
        sections += [
            "<h2>Reward per solver</h2>",
            '<div class="panel">',
            _reward_chart(solvers),
            '<p class="legend"><span><span class="swatch" style="background:#4c78a8"></span>'
            "bar = mean reward</span><span>whisker = min&ndash;max</span>"
            "<span>white tick = median</span></p>",
            "</div>",
            "<h2>Outcome rates</h2>",
            '<div class="panel">',
            _solver_table(solvers),
            "</div>",
        ]
    if report.get("pass_at_k"):
        sections += [
            "<h2>pass@k <span style='font-weight:400;color:var(--muted)'>"
            "(bundled solvers as k samples per instance)</span></h2>",
            '<div class="panel">',
            _pass_at_k_chart(report["pass_at_k"]),
            "</div>",
        ]
    sections += [
        "<details><summary>Raw report JSON</summary>",
        f"<pre>{html.escape(json.dumps(report, indent=2, sort_keys=True))}</pre>",
        "</details>",
    ]
    body = "\n".join(sections)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>\n'
        f"<title>Benchmark — {suite}</title>\n<style>{_CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def write_html(report_path: Path, output_path: Path | None = None) -> Path:
    """Render ``report_path`` (a benchmark JSON) to HTML next to it (or at *output_path*)."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_path = output_path or report_path.with_suffix(".html")
    output_path.write_text(render_report_html(report, source_name=report_path.name), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render benchmark report JSON as HTML.")
    parser.add_argument("reports", nargs="*", type=Path, help="benchmark report JSON file(s)")
    parser.add_argument("-o", "--output", type=Path, help="output HTML path (single report only)")
    parser.add_argument(
        "--all-baselines",
        action="store_true",
        help=f"render every committed baseline under {BASELINE_DIR}",
    )
    args = parser.parse_args(argv)

    reports = list(args.reports)
    if args.all_baselines:
        reports += sorted(BASELINE_DIR.glob("*.json"))
    if not reports:
        parser.error("no reports given (pass JSON paths or --all-baselines)")
    if args.output and len(reports) > 1:
        parser.error("-o/--output only makes sense with a single report")

    for report_path in reports:
        output = write_html(report_path, args.output)
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
