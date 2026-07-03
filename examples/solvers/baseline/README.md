# baseline

The smallest possible solver plugin, implementing `ISolver` directly with no
search. It scores the initial assignment and reports it unchanged.

**What it illustrates**

- the exact shape of a conforming plugin: `#include "solver_sdk.hpp"`, a class
  deriving from `ISolver`, and an `extern "C" create_solver`;
- the minimum each method must do (`RunEpoch` returns the current score,
  `CurrentAssignment` returns a state, `DrainFeasible` hands back any solved
  state exactly once).

**What it exercises in the interface**

`SdkCircuit::Score` and `SdkCircuit::DeriveWires`. Nothing else.

**Knobs**

None — that is the point. Use it as the reward floor: any real solver should
score at least as well, and it is the reference template to copy when starting a
new solver. There are no tuning parameters.
