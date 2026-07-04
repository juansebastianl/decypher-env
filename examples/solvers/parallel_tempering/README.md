# parallel_tempering

A working `ISolver` plugin ([`solver.cpp`](solver.cpp)) implementing parallel
tempering against the plugin SDK. Parallel tempering (replica-exchange MCMC) is
the classic way to search a rugged energy landscape without getting stuck, and
it is the scaffold the other two reference strategies build on.

## Strategy

- Hold a population of **replicas** on a geometric temperature ladder
  (`kColdTemperature`..`kHotTemperature`). Hot replicas roam; cold replicas
  refine.
- Each replica does **augmented-Lagrangian Metropolis** using
  `SdkCircuit::Energy`, with a proposal mix of 65% wire bit-flip, 25%
  ASCII plaintext resample, otherwise a key move (half 4-byte word swap, half
  single-bit flip), plus an occasional exact XTS plaintext repair
  (`RepairPlaintext`).
- After every sweep, **adjacent replicas attempt a swap** with the standard
  replica-exchange rule `min(1, exp((β_i − β_j)(E_i − E_j)))`, so good structure
  found by hot chains flows down to the cold chain.
- Feasible (zero-residual) states are harvested and de-duplicated by key.

## How it maps to the `ISolver` contract

| Contract | This solver |
| --- | --- |
| `RunEpoch(sweeps)` | advance every replica by `sweeps` steps, swapping after each |
| `CurrentAssignment()` | the best (lowest hard residual) state seen on any replica |
| `DrainFeasible(limit)` | key-deduplicated feasible archive |

## Knobs

Compile-time constants at the top of `solver.cpp`: `kReplicas`,
`kColdTemperature` / `kHotTemperature`, `kRepairProb`. The per-class Lagrangian
weights and the inter-round `coupling_weight` come from `SolverConfig` (set by
the environment's curriculum), so acceptance automatically tracks the coupling
schedule.

## What to expect

Pure bit-flip Metropolis has no exact key knowledge, so from a random start it
**reduces** violations substantially (well below the no-search floor) but does
not by itself invert even a one-round instance. That is exactly the gap
`continuous_relaxed` and `algebraic_relaxed` close by adding exact
key-conditional moves on top of this scaffold. See `tests/test_example_parity.py`
for the capability checks.
