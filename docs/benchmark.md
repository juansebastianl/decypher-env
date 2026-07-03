# Benchmark & evaluation

The benchmark suite (`src/decoder/rl/benchmark.py`) gives reproducible,
committed baselines so you can tell real progress from noise and catch
regressions in CI.

## Suites

| suite    | instances                         | budget            | use |
|----------|-----------------------------------|-------------------|-----|
| `smoke`  | 1 × 1-round                       | tiny (seconds)    | CI regression, sanity |
| `public` | 3 × 1–2-round ladders (seeds 1/7/13) | medium          | standard evaluation |
| `hard`   | 2 × 1–3-round ladders (seeds 3/11) | large (2 epochs) | the intentionally-hard set |

Each suite pins its `TaskSpec` set, seeds, rounds ladders and `HarnessOptions`
budget, so a run is fully reproducible.

## Running

```bash
# Print a report for a suite:
python -m src.decoder.rl.benchmark --suite smoke
python -m src.decoder.rl.benchmark --suite public --json

# Compare against the committed baseline and flag regressions:
python -m src.decoder.rl.benchmark --suite smoke --compare-baseline

# Record the current run as the baseline (after an intentional change):
python -m src.decoder.rl.benchmark --suite smoke --write-baseline
```

## What it reports

For each bundled example solver (`examples/solvers/`), per suite:

- reward distribution (mean / min / max / median),
- compile-failure rate, timeout rate, crash rate,
- feasibility rate (`pass@1`),

and suite-level `pass@k` (the unbiased Chen-et-al. estimator over the bundled
solvers treated as `k` samples per instance).

Baselines are committed under `benchmarks/baselines/<suite>.json`. The
`--compare-baseline` mode flags any solver whose `reward_mean` regressed beyond
`--tolerance` (default `0.05`).

## Regression test

`tests/test_benchmark.py` runs the `smoke` suite and asserts the bundled solvers
stay within tolerance of the committed baseline, plus a determinism check
(two runs produce identical rewards). It skips cleanly without a C++ toolchain.

## Reading the numbers honestly

The bundled solvers are **reference baselines, not solutions**. Low feasibility
on `public`/`hard` is expected — see the "intentionally hard vs reduced-round
positive control vs progress" section of the README. Track the dense signals
(energy, round staircase) for progress on the hard rungs, and use the 1-round
rung as a positive control that your setup works at all.
