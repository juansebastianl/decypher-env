# random_metropolis

A small but genuine solver built on `SolverBase`. It keeps the default
single-bit / ASCII-resample Metropolis step and adds two ideas worth reusing.

**Strategy**

1. **Geometric cooling.** The temperature is annealed from hot (`8.0`) to cold
   (`0.4`) across each epoch, so early sweeps explore and later sweeps exploit.
2. **Exact plaintext repair.** With small probability each sweep it calls
   `SdkCircuit::RepairPlaintext`, which rewrites every block's plaintext by
   decrypting the target ciphertext under the *current* keys. That zeroes all
   goal + consistency residuals in one shot, so feasibility collapses to the
   sub-problem "find keys whose decryption is ASCII" — a much denser target than
   raw key search.

**What it exercises in the interface**

`SolverBase` (Metropolis loop, best-state tracking, feasible harvesting),
`SdkCircuit::Score` / `Energy`, `RepairPlaintext`, `DeriveWires`, and the seeded
RNG (`RngUniform`).

**Knobs** (compile-time constants near the top of `solver.cpp`)

| Constant | Meaning |
| --- | --- |
| `kHotTemperature` / `kColdTemperature` | endpoints of the cooling schedule |
| `kRepairProb` | per-sweep probability of an exact plaintext repair |

**Framing**

This is an *illustration*, not a benchmark. It reliably solves 1–2 round
positive controls but is not expected to invert full-round AES; that difficulty
is exactly what the environment trains a policy to chip away at.
