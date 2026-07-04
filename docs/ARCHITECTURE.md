# Architecture

The repository is two layers. Only the top one matters if you are training a
model; the lower one is the substrate it stands on and can be treated as a
black box.

```
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 1 — THE ENVIRONMENT        src/decoder/rl/  +  examples/         │
│   The product. Turns "a model writes C++" into a scalar reward.        │
│   • solver_env.py   dataset + prompt + rollout + rubric               │
│   • reward.py       augmented-Lagrangian + round staircase + feasible │
│   • curriculum.py   reduced-round ladder + coupling schedule          │
│   • harness.py      circuit → wire format → build & run C++ harness   │
│   • native/         solver SDK (the ISolver contract) + the harness    │
│   examples/         solver templates + a menu of strategies            │
│                     (baseline, Metropolis, parallel tempering,         │
│                      continuous relaxation, algebraic/BP-guided)       │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │ builds tasks from
┌───────────────────────────────▼──────────────────────────────────────┐
│ Layer 2 — THE TASK BUILDER       src/decoder/{circuit,xts_model,       │
│                                   aes_primitives,ascii_constraints,     │
│                                   encodings,constraints}.py             │
│   The AES-XTS constraint model: a flat circuit IR of XOR/S-box/Mix/    │
│   key-expansion ops with per-round consistency (DEFINE8) constraints,  │
│   and reduced-round targets. Shared, stable, framework-agnostic.       │
└────────────────────────────────────────────────────────────────────────┘
```

## How a rollout flows

1. **Curriculum** (`rl/curriculum.py`) picks a fixture window and a rounds
   ladder, and builds one AES-XTS circuit per round count via the **task
   builder**.
2. **harness.py** flattens each circuit into a compact wire format, tagging each
   constraint with the AES data-path round it belongs to.
3. The model's completion is stripped to C++ source and handed to the **C++
   harness** (`rl/native/harness/`), which compiles it against the **solver SDK**
   into a shared object and runs `create_solver` in a forked, wall-clock-bounded
   child. A hang or crash becomes a shaped penalty, never a trainer outage.
4. The harness reports energy, per-round residuals, staircase and feasibility;
   **reward.py** turns them into a reward with components exposed.

## Why the augmented Lagrangian

AES inversion is an almost pure sparse-reward problem: only the exact key is a
"win". The circuit's consistency (`DEFINE8`) constraints let us measure *partial*
progress — how internally consistent each round is — and the augmented
Lagrangian turns that into a smooth energy. `SdkCircuit::Energy` computes,
per constraint `i` with residual `r_i`:

```
Energy = Σ_i [ λ_i · r_i  +  ½ · ρ_i · r_i² ]
```

where `λ_i` is the per-class Lagrange multiplier (`ascii_weight`,
`consistency_weight`, `goal_weight`) and `ρ_i` the per-class quadratic penalty
(`ascii_rho`, `consistency_rho`, `goal_rho`). The `coupling_weight` scales *both*
the multiplier and the penalty of the inter-round consistency class, so ramping
it is exactly "how hard rounds are forced to agree": from small (rounds may
drift independently and are each solvable alone) to large (rounds forced to
agree). A feasible (zero-residual) assignment has zero energy regardless of the
multipliers/penalties.

This is a genuine **method of multipliers**: `rl/curriculum.py` ramps `ρ` up
across stages (`penalty_schedule`) and, between stages, applies the dual update
`λ_i ← λ_i + ρ_i · r_i` (`Curriculum.advance` / `TaskSpec.dual_update`) using the
per-class residual mass the best assignment left, so multipliers grow on
constraints that stayed violated. The per-round staircase reward is read off the
same residuals.
