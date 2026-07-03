# algebraic_relaxed

A working `ISolver` plugin ([`solver.cpp`](solver.cpp)) that ports the legacy
algebraic / belief-propagation-guided engine to the plugin SDK.

## What the legacy engine actually did

The legacy "algebraic" engine was a thin wrapper around the continuous-relaxed
search. Its "belief propagation" was **not** a factor graph over the circuit; it
was coordinate-wise belief updates over the **64 key bytes**: for each key byte,
sweep all 256 values, score each by repairing the plaintext (exact XTS decrypt)
and measuring the ASCII penalty, and form a damped-softmax marginal. Those
marginals were used **once**, to seed the starting key, before handing off to
the ordinary search. This port reproduces that behaviour.

## Strategy

1. **BP seed (once, at construction).** Run `kBpIterations` damped coordinate
   passes over the key bytes to build a marginal per byte, then sample a seed
   key from those marginals and repair the plaintext under it. The result is a
   start already biased toward keys that decrypt to printable text. Each
   candidate is scored with the cheap `PlaintextAsciiPenalty` (repair without
   full wire re-derivation), so the seed pass stays affordable.
2. **Search.** The identical continuous-relaxed engine: parallel tempering +
   exact key-profile Gibbs + repair moves.

On easy (few-round) instances the seed is a genuine head start; on hard ones it
degrades gracefully to the continuous engine, so this solver is never worse than
`continuous_relaxed`.

## How it maps to the `ISolver` contract

Same as `continuous_relaxed/`. The only addition is the one-time `BeliefSeed`
step inside the constructor, before the replicas are initialised.

## Knobs

`kBpIterations`, `kBpDamping` (the seed pass), plus everything
`continuous_relaxed` exposes (`kReplicas`, temperatures, `kRepairProb`,
`kKeyGibbsProb`, `kGibbsBudget`).

## Note on the aspirational design

A "true" algebraic solver — a factor graph with S-box/MixColumns/XOR factors and
real sum-product or max-product message passing, lifting the S-box to GF(2⁸)
constraints — is a legitimate strategy the interface supports, but it is **not**
what the legacy engine implemented and is not what this port does. It is left as
an exercise.
