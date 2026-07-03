# Example solvers

This repository is, first and foremost, a **reinforcement-learning environment
for authoring AES-XTS solvers**. A policy (a language model, or you) emits C++
source implementing a small solver interface; the harness compiles it, runs it
against a curriculum of reduced-round AES-XTS instances, and returns a reward
derived from the augmented-Lagrangian energy of the constraint system.

Every solver in this folder is a **real, compilable plugin** (`solver.cpp`) that
the harness builds and runs exactly like a model's output — they are
documentation by example, not benchmarks. `parallel_tempering`,
`continuous_relaxed` and `algebraic_relaxed` are ports of the legacy native
engines to the plugin interface, with capability-parity tests in
`tests/test_example_parity.py`. None of them is tuned to invert full AES; that is
the hard open problem the environment is built to train against.

## The interface

Every solver — an example here, or a model's output during training — is a
single C++ translation unit that:

1. `#include "solver_sdk.hpp"`,
2. defines a class implementing `aes_xts_decoder::sdk::ISolver` (or derives from
   the convenience base `aes_xts_decoder::sdk::SolverBase`), and
3. exports `extern "C" ISolver* create_solver(const SolverContext&)`.

```cpp
class ISolver {
 public:
  virtual ~ISolver() = default;
  virtual ScoreData RunEpoch(std::size_t sweeps) = 0;      // advance the search
  virtual AssignmentState CurrentAssignment() const = 0;   // best state so far
  virtual std::vector<AssignmentState> DrainFeasible(std::size_t limit) = 0;
};

extern "C" ISolver* create_solver(const SolverContext&);   // dlopen entry point
```

`SolverContext` is a read-only view of the task:

```cpp
struct SolverContext {
  const SdkCircuit& circuit;   // the flattened AES-XTS constraint circuit
  const SolverConfig& config;  // seed, aes_rounds, per-class weights, coupling
  AssignmentState initial;     // a starting point (random ASCII + random keys)
};
```

## The SDK surface

`solver_sdk.hpp` gives you everything needed to write a search without touching
AES internals:

| Piece | What it does |
| --- | --- |
| `SdkCircuit::Score(assignment)` | exact residual score → `ScoreData` (`residuals`, `hamming_score`, `feasible()`) |
| `SdkCircuit::Energy(score, config)` | augmented-Lagrangian energy `Σ λ_class · r_i` (consistency scaled by `coupling_weight`) |
| `SdkCircuit::PerRoundResiduals(score)` | residual mass bucketed by AES data-path round |
| `SdkCircuit::RoundStaircase(score)` | count of consecutive internally-consistent rounds (+ partial credit) |
| `SdkCircuit::DeriveWires(&assignment)` | recompute internal wires from plaintext/keys |
| `SdkCircuit::RepairPlaintext(&assignment)` | exact block-Gibbs: rewrite plaintext by decrypting the target under the current keys (pass `derive_wires=false` to skip the wire rebuild in tight key-search loops) |
| `SdkCircuit::PlaintextAsciiPenalty(assignment)` | count of non-printable plaintext bytes — the cheap energy signal after a repair, used by key-conditional Gibbs/BP moves |
| `RngNext / RngUniform / RngRange / RandomTextAscii` | deterministic (seeded) RNG |
| `SolverBase` | default Metropolis loop + best-state tracking + feasible harvesting |

`AssignmentState` holds the mutable buffers you search over: `plaintext`,
`key1`, `key2`, and the internal `wires` (present for the lowered encoding this
environment uses).

## How reward works

For each `(instance, aes_rounds)` the harness reports:

- **energy** — the augmented-Lagrangian value at your best state (lower better),
- **round staircase** — how many consecutive AES rounds are internally
  consistent, the dense `1, 2, 3, …, n` signal that makes the otherwise
  exact-key-only reward tractable,
- **feasibility** — whether any harvested state has zero hard residual.

The Python rubric (`src/decoder/rl/reward.py`) turns these into a bounded reward
with the components exposed separately, and orders failures so that

```
doesn't compile  <  compiles but crashes/times out  <  compiles and runs
```

The reduced-round **curriculum** and the **coupling schedule** (which ramps how
hard rounds are forced to agree) live in `src/decoder/rl/curriculum.py`.

## A menu of solver ideas

The interface is deliberately open-ended. Strategies worth trying:

- **random / greedy baseline** — report the initial assignment; the reward floor
  (`baseline/`).
- **single-chain Metropolis** with a cooling schedule and occasional exact
  plaintext repair (`random_metropolis/`).
- **parallel tempering** — many replicas on a temperature ladder with swaps, to
  escape the rugged energy landscape (`parallel_tempering/`).
- **continuous relaxed** — parallel tempering plus exact key-conditional Gibbs
  moves that repair the plaintext for every candidate key byte
  (`continuous_relaxed/`); collapses easy instances that pure Metropolis cannot.
- **algebraic / belief-propagation guided** — seed the starting key from
  coordinate BP marginals over the key bytes, then search (`algebraic_relaxed/`).
- **key-Gibbs / repair moves** — alternate between fixing keys and repairing the
  plaintext exactly, so feasibility reduces to finding ASCII-decrypting keys.

## Build & run

You never invoke the compiler yourself — the harness does. From Python:

```python
from pathlib import Path
from src.decoder.rl import SolverAuthoringEnv

env = SolverAuthoringEnv.single(
    Path("src/testing_setup/lorem_ipsum_aes_xts.json"),
    block_count=1, rounds_ladder=(1, 2, 3),
)
source = Path("examples/solvers/baseline/solver.cpp").read_text()
breakdown = env.score_source(source, env.task_specs[0])
print(breakdown.reward, breakdown.to_dict()["per_task"])
```

To reproduce the same compile+run manually for debugging:

```bash
c++ -std=c++17 -O2 -fPIC -shared -fopenmp \
    -Isrc/decoder/rl/native/include \
    examples/solvers/baseline/solver.cpp \
    src/decoder/rl/native/sdk/solver_sdk.cpp \
    -o /tmp/baseline.so
```

The environment integrates with GRPO trainers via the `verifiers` shape
(`SolverAuthoringEnv.to_verifiers_env()`) and also offers a dependency-free
gym-like `reset()/step()` loop.
