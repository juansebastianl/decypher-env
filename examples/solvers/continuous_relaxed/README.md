# continuous_relaxed

A working `ISolver` plugin ([`solver.cpp`](solver.cpp)) implementing the
"continuous relaxed" strategy against the plugin SDK.

## What "continuous relaxed" really means here

Despite the name, this solver carries no persistent 256-way byte distributions
or analytic gradients — its state is always **discrete bytes**. What makes it
strong is combining parallel tempering with **exact key-conditional Gibbs
moves**.

## Strategy

- The same parallel-tempering scaffold as `parallel_tempering/` (replica ladder,
  augmented-Lagrangian Metropolis, adjacent swaps, feasible harvest).
- Plus a **key-profile Gibbs move**: pick one key byte, sweep all 256 values,
  and for each value repair the plaintext exactly by decrypting the target
  ciphertext under the candidate keys (`RepairPlaintext`). After a repair the
  goal + consistency residuals are zero, so the only thing left to score is
  whether the decrypted plaintext is printable — read cheaply via
  `SdkCircuit::PlaintextAsciiPenalty` without a full circuit evaluation. The
  byte is then sampled from the Boltzmann distribution over those penalties.

That exact conditional is what lets the solver *collapse* an easy instance
rather than merely shrink it: on a one-round block it typically drives the hard
residual to zero (a feasible key/plaintext), which pure Metropolis cannot do.

## Cost control

Each Gibbs move costs ~256 plaintext repairs, so it is far more expensive than a
random step. `kGibbsBudget` caps the total number of Gibbs moves across the
whole rollout; once spent, the search continues with cheap Metropolis moves.
This keeps a rollout inside the harness wall-clock budget regardless of the
sweep count. `kKeyGibbsProb` sets how eagerly the budget is spent.

## How it maps to the `ISolver` contract

Identical to `parallel_tempering/`: `RunEpoch` advances the replicas (firing
Gibbs moves while budget remains), `CurrentAssignment` returns the best state,
`DrainFeasible` returns the key-deduplicated feasible archive.

## Knobs

`kReplicas`, `kColdTemperature` / `kHotTemperature`, `kRepairProb`,
`kKeyGibbsProb`, `kGibbsBudget` (compile-time constants); per-class weights and
`coupling_weight` from `SolverConfig`.
