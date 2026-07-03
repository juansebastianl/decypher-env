# legacy — the original solver research toolkit (internals)

This is Layer 3 in [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md): the
hand-written "author a solver, run a sampler" codebase that predates the RL
environment. **You do not need any of it to train a model.** It is kept here,
working and tested, for two reasons:

1. its native engines are the **reference strategies** documented under
   `examples/solvers/{parallel_tempering,continuous_relaxed,algebraic_relaxed}`,
   showing what a strong solver can look like; and
2. the sampler CLIs remain useful for exploring the constraint model directly.

## Contents

| Path | What it is |
| --- | --- |
| `backends/` | engine registry: the pure-Python reference engine and the native `_bridge` Cython extension (parallel tempering, continuous / algebraic relaxations) |
| `relaxations/` | discrete and continuous (byte-distribution) relaxations of the constraint residuals |
| `sampler.py` | `XtsSampler`: drives an engine against a fixture window |
| `sample_blocks.py`, `cli.py` | command-line entry points (`python -m src.decoder.legacy.cli …`) |

## Relationship to the environment

The environment (`src/decoder/rl/`) does **not** import this package. It ships
its own self-contained solver SDK and compile-and-run harness under
`src/decoder/rl/native/`. The only shared code is the framework-agnostic task
builder in `src/decoder/` (`circuit`, `xts_model`, `constraints`, …).

The native engines here are compiled into the `_bridge` extension by
`setup.py`. The environment, by contrast, compiles authored solvers **on
demand** — that is the whole point of the plugin path.
