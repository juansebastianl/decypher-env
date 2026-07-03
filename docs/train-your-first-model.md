# Train your first model

This guide shows the framework-agnostic training loop first (so you understand
exactly what the environment gives you), then points at the per-framework
adapters. Everything below runs against the same core primitive:
`env.score_batch(sources, specs) -> list[RolloutResult]`.

## Prerequisites

- A C++17 compiler and `bwrap` (bubblewrap) on `PATH`.
- `pip install -r requirements-dev.txt` (plus your framework extra, below).

## The core primitive

```python
from pathlib import Path
from src.decoder.rl import SolverAuthoringEnv, HarnessOptions
from src.decoder.rl.curriculum import Curriculum

env = SolverAuthoringEnv.from_curriculum(
    Curriculum(Path("src/testing_setup/lorem_ipsum_aes_xts.json"), max_rounds=2),
    options=HarnessOptions(run_timeout=15.0, sweeps_per_epoch=2000),
)
spec = env.task_specs[0]

# A GRPO group: N completions for one prompt, scored in one batch.
completions = [sample_from_policy(env.prompts()[0]) for _ in range(8)]
results = env.score_batch(completions, spec)     # list[RolloutResult]
rewards = [r.reward for r in results]
```

Each `RolloutResult` carries the scalar `reward`, the per-instance `components`
and `per_task` metrics, the `compile_stderr`, a coarse `failure_kind`
(`ok`/`compile_error`/`timeout`/`crash`/`empty_source`), source-extraction
diagnostics, wall time, and (if `artifacts_dir` is set) on-disk artifact paths.

## A runnable mock GRPO loop (no framework required)

This uses a dummy policy that always emits the bundled baseline solver, so you
can confirm the full compile→run→score→advantage path end-to-end before wiring a
real trainer.

```python
from pathlib import Path
import statistics
from src.decoder.rl import SolverAuthoringEnv, HarnessOptions
from src.decoder.rl.curriculum import Curriculum

BASELINE = Path("examples/solvers/baseline/solver.cpp").read_text()

def sample_from_policy(prompt: str) -> str:
    # Replace with your model. Here: always the baseline solver.
    return f"```cpp\n{BASELINE}\n```"

env = SolverAuthoringEnv.from_curriculum(
    Curriculum(Path("src/testing_setup/lorem_ipsum_aes_xts.json"), max_rounds=2),
    options=HarnessOptions(run_timeout=15.0, sweeps_per_epoch=1000),
)

curriculum = Curriculum(Path("src/testing_setup/lorem_ipsum_aes_xts.json"), max_rounds=2)
group_size = 8
for stage in range(curriculum.num_stages()):
    spec = curriculum.task_for_stage(stage)
    prompt = env.prompts()[0]
    completions = [sample_from_policy(prompt) for _ in range(group_size)]
    results = env.score_batch(completions, spec)

    rewards = [r.reward for r in results]
    mean = statistics.fmean(rewards)
    advantages = [r - mean for r in rewards]           # GRPO baseline subtraction
    # ... feed (completion, advantage) pairs to your policy-gradient update ...

    # Method-of-multipliers: advance the curriculum using the residuals the best
    # rollout left, so the Lagrange multipliers grow where constraints stayed
    # violated (see curriculum.py).
    best = max(results, key=lambda r: r.reward)
    per_class = best.per_task[0]["per_class"] if best.per_task else [0, 0, 0]
    next_spec = curriculum.advance(stage, per_class)
    print(f"stage {stage}: mean reward {mean:.3f}, next consistency_weight "
          f"{next_spec.consistency_weight:.3f}")
```

## Then: wire your framework

Install the extra for your stack and load the adapter:

```python
from src.decoder.rl.adapters import load_environment
env = load_environment("verifiers")   # gem | skyrl | openenv | ors | nemo
```

Per-framework specifics (install extra, entry point, minimal train/eval snippet)
are in [`docs/adapters/`](adapters/):

- [Verifiers](adapters/verifiers.md) — `prime env` / prime-rl
- [GEM](adapters/gem.md) — `gem.make(...)`, oat/verl
- [SkyRL Gym](adapters/skyrl.md) — `skyrl_gym.register(...)`
- [OpenEnv](adapters/openenv.md) — MCP service + Docker
- [ORS](adapters/ors.md) — `Server([Env]).run()` + Docker
- [NeMo Gym](adapters/nemo.md) — FastAPI `/verify` + Docker

## Sanity checklist

1. The 1-round rung reaches feasibility with a competent solver (positive
   control). If not, fix your setup before trusting anything else.
2. `smoke` benchmark matches the committed baseline
   (`python -m src.decoder.rl.benchmark --suite smoke --compare-baseline`).
3. On harder rungs, watch the dense signals (energy down, round staircase up)
   rather than feasibility alone.
