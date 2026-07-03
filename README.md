# AES-XTS Solver-Authoring RL Environment

**A reinforcement-learning environment where the policy writes code.** The model
is handed a reduced-round AES-XTS inversion task and must emit a C++ *solver*;
the environment compiles it, runs it under a budget, and rewards it by how close
the solver drives the constraint system to a solution. It drops into GRPO-style
trainers unchanged via adapters to the major RL-environment frameworks
(Verifiers, GEM, SkyRL Gym, OpenEnv, ORS, NeMo Gym).

If you train models and want to make yours strong at this task, you only need
three things: **the task**, **the contract your model writes to**, and **the
reward**. They are below, then a 60-second quickstart.

---

## 1. The task

AES-XTS encryption is a fixed circuit of XORs, S-boxes, MixColumns and key
expansion. Inverting it (recovering key/plaintext from ciphertext) is the hard
problem. To make learning tractable we expose a **reduced-round curriculum**:
the same instance is scored at `aes_rounds = 1, 2, 3, …`, so a solver that only
cracks a 1-round instance still earns signal long before full 14-round AES is in
reach.

## 2. The action space — your model writes a solver

The policy's output is a **single C++ translation unit** implementing one
interface (`src/decoder/rl/native/include/solver_sdk.hpp`):

```cpp
#include "solver_sdk.hpp"
using namespace aes_xts_decoder::sdk;

class MySolver : public SolverBase {   // SolverBase gives a Metropolis loop for free
 public:
  explicit MySolver(const SolverContext& ctx) : SolverBase(ctx) {}
  // override RunEpoch / Step to implement your search
};

extern "C" ISolver* create_solver(const SolverContext& ctx) { return new MySolver(ctx); }
```

The harness constructs one solver per task with `SolverContext{circuit, config,
initial}`, calls `RunEpoch(sweeps)` repeatedly, then reads `CurrentAssignment()`
and `DrainFeasible()`. The SDK gives you scoring, the Lagrangian energy,
per-round residuals, wire derivation, an exact plaintext-repair helper, and a
seeded RNG — so a solver never touches AES internals. See
[`examples/README.md`](examples/README.md) for the full SDK surface and a menu
of strategies (baseline, Metropolis, parallel tempering, continuous relaxation,
algebraic/BP-guided).

## 3. The reward (verifiable, deterministic)

No learned reward model. For each `(instance, round)` the reward combines:

- **energy** — the augmented Lagrangian `Σ_i [ λ_i · r_i + ½ · ρ_i · r_i² ]` at
  the best state (lower is better), with per-class Lagrange multipliers `λ` and
  quadratic penalties `ρ`; consistency residuals are scaled by a
  `coupling_weight` that the curriculum ramps up over training, and the penalty
  `ρ` is ramped with a method-of-multipliers dual update (`λ ← λ + ρ·r`);
- **round staircase** — how many consecutive AES rounds the solver made
  internally consistent, the dense `1, 2, …, n` partial-credit signal;
- **feasibility** — a large bonus when a zero-residual (solved) state is found.

Failures are shaped, not fatal, and strictly ordered:

```
doesn't compile  <  compiles but crashes / times out  <  compiles and runs
```

Components are returned separately (see `RolloutResult`) for GRPO analysis,
along with compile logs, timeout/crash categories, source-extraction
diagnostics, per-task wall time and optional artifact paths.

### What is intentionally hard, and what counts as progress

This environment is **deliberately hard**: full 14-round AES-XTS inversion is not
expected to be solved, and a near-zero feasibility rate on the `public`/`hard`
benchmark suites is the *expected* baseline, not a bug. To keep the signal
honest we distinguish three things:

- **Intentionally hard** — the full-round task. Feasibility here would be a
  genuine cryptanalytic result; treat any claim of it with suspicion until the
  recovered key is verified end-to-end.
- **Reduced-round positive control** — the low rungs of the ladder
  (`aes_rounds = 1`, sometimes `2`). These *are* solvable and exist so you can
  confirm the environment, reward, and your trainer actually work: a competent
  solver should reach feasibility at 1 round. If it cannot, the problem is your
  setup, not the task.
- **Progress** — improvement in the *dense* signals (lower energy, more
  consecutive internally-consistent rounds on the staircase) even when
  feasibility is still zero. This is what a learning curve should track on the
  harder rungs.

---

## Quickstart

```python
from pathlib import Path
from src.decoder.rl import SolverAuthoringEnv

env = SolverAuthoringEnv.single(
    Path("src/testing_setup/lorem_ipsum_aes_xts.json"),
    block_count=1, rounds_ladder=(1, 2, 3),
)
spec = env.task_specs[0]

# A model completion is just text; extract + compile + score happens for you.
completion = Path("examples/solvers/random_metropolis/solver.cpp").read_text()
result = env.score_completion(completion, spec)   # -> RolloutResult

print("reward:", result.reward, "feasible:", result.feasible)
for t in result.to_dict()["per_task"]:
    print(t["aes_rounds"], t["reward"], t["energy"], t["round_staircase"])
```

### Plug into your training stack

Rather than a bespoke interface, the environment ships thin adapters to the
major RL-environment frameworks. Install the extra for your stack and load it:

```python
from src.decoder.rl.adapters import load_environment

vf_env = load_environment("verifiers")   # or: gem, skyrl, openenv, ors, nemo
```

The six adapters (Verifiers, GEM, SkyRL Gym, OpenEnv, ORS, NeMo Gym) map the
same `RolloutResult` into each framework's native reward/state/info shape. See
the per-framework guides under [`docs/adapters/`](docs/adapters/) and
[`docs/train-your-first-model.md`](docs/train-your-first-model.md) for a
runnable mock GRPO loop that calls `env.score_batch(...)` directly (no framework
required).

**Requirements:** a C++17 compiler and `bwrap` (bubblewrap) on `PATH` — the
harness compiles and runs untrusted solvers in a sandbox (see
[`docs/security-sandbox.md`](docs/security-sandbox.md)) — and
`pip install -r requirements-dev.txt`. Run the environment tests with
`pytest tests/`.

---

## Repo map — where to look, what to ignore

```
src/decoder/
  rl/                  ← THE ENVIRONMENT. Start here.
    solver_env.py        framework-neutral core: prompt + rollout + rubric
    reward.py            Lagrangian + staircase + feasibility; RolloutResult
    curriculum.py        reduced-round ladder + coupling/penalty schedule + dual update
    harness.py           flatten circuit, build + run the C++ harness (bwrap sandbox)
    benchmark.py         smoke|public|hard suites + baselines + regression compare
    adapters/            one thin adapter per RL framework (verifiers/gem/skyrl/…)
    native/              the solver SDK (include/) + compile-and-run harness/
  circuit.py, xts_model.py, aes_primitives.py, ascii_constraints.py,
  encodings/, constraints.py   ← task builder (the AES-XTS constraint model)
  legacy/              ← internals you can ignore: the old sampler CLIs and the
                          native reference engines (see legacy/README).
examples/              ← solver templates + strategy docs (start: examples/README.md)
docker/                ← Dockerfile + compose scaffolds for the HTTP adapters
benchmarks/baselines/  ← committed benchmark baselines (regression targets)
docs/                  ← ARCHITECTURE, adapters/, train-your-first-model,
                          benchmark, security-sandbox, troubleshooting
tests/
  test_solver_env.py     environment + reward correctness
  test_augmented_lagrangian.py   energy math + method-of-multipliers
  test_harness_hardening.py      malformed-payload + sandbox enforcement
  test_adapters.py       per-framework adapter smoke tests
  test_benchmark.py      benchmark regression
  test_decoder.py        legacy engine + task-builder tests
```

As an RL developer you live in **`src/decoder/rl/`** and **`examples/`**. The
`legacy/` tree and `docs/internals/` exist for depth and reference; you don't
need them to train a strong policy. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the three-layer picture.
