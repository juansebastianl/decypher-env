# Parallel Tempering Implementation

`parallel_tempering.cpp` is the high-level map for the native PT engine. The
implementation lives in normal C++ translation units so IDEs can index,
syntax-highlight, and navigate the code.

- `solver.hpp`: internal solver state and orchestration surface
- `circuit_model.hpp` / `circuit_model.cpp`: immutable circuit buffers and graph dependency indexes
- `byte_evaluator.hpp` / `byte_evaluator.cpp`: circuit value evaluation and wire derivation
- `hamming_scorer.hpp` / `hamming_scorer.cpp`: residual scoring over evaluated values
- `constructor.cpp`: model setup, replica initialization, profiling helpers
- `evaluation.cpp`: profiled evaluator/scorer wrappers and incremental updates
- `api.cpp`: public engine API, metrics, assignment accessors
- `proposals.hpp` / `proposals.cpp`: mutation families and XTS repair proposals
- `replica_set.hpp`: per-replica bookkeeping and swap ownership
- `moves.cpp`: replica step orchestration and swaps
- `schedule.hpp` / `schedule.cpp`: dual updates and feedback ladder adaptation
- `optimization.cpp`: energy stats, key stats, scheduling delegates
- `diagnostics.cpp`: marginal, algebra, and log-Z diagnostics
- `energy.cpp`: augmented-Lagrangian energy and initialization helpers
- `feasible_archive.hpp`: feasible sample deduplication
- `diagnostics_collector.hpp`: profile diagnostics helpers
- `helpers.cpp`: AES/XTS helpers, RNG helpers, ASCII/popcount utilities
- `internal.hpp`: shared private declarations for this module

Keep new hot-path behavior in the narrowest shard possible. If a new feature
needs a reusable abstraction, prefer adding it under `include/` with a concrete
implementation under `solvers/` instead of expanding the PT engine surface.
