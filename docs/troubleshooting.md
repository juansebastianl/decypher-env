# Troubleshooting

## Build / native extension

- **`HarnessUnavailable: C++ compiler 'c++' not found`** — install a C++17
  toolchain (`build-essential` on Debian/Ubuntu) and ensure `c++`/`g++` is on
  `PATH`. Override with the `CXX` env var (e.g. `CXX="ccache g++"`).
- **`failed to build harness binary`** — usually a missing OpenMP runtime. The
  harness compiles with `-fopenmp`; install `libomp-dev`/`libgomp1`.
- **`ccache` errors during `pip install -e .`** — the editable build honours
  `CXX`; if `ccache` isn't configured (no writable `CCACHE_DIR`), build with
  `CC=gcc CXX=g++ pip install -e . --no-build-isolation`.

## Sandbox (`bwrap`)

- **`SandboxUnavailable: bwrap ... is not installed`** — install bubblewrap
  (`apt-get install bubblewrap`). To run without isolation (unsafe; only in an
  already-isolated container) pass `Sandbox(allow_unsandboxed=True)`.
- **`bwrap: setting up uid map: Operation not permitted`** — the host disables
  unprivileged user namespaces. Enable them
  (`sysctl kernel.unprivileged_userns_clone=1` on some distros) or run the
  container with a policy that allows `CLONE_NEWUSER` (see
  `docker/docker-compose.yml`, which sets `security_opt: seccomp:unconfined`).
- **Solver always reports `crashed` with "killed by seccomp"** — the solver
  used a syscall outside the compute-only whitelist (network, `exec`, `fork`,
  `ptrace`). That is the sandbox working as intended; a legitimate solver should
  not need those.
- **Solver reports `timed_out` immediately** — the wall-clock budget
  (`HarnessOptions.run_timeout`) is too small for the sweeps requested; raise it
  or lower `sweeps_per_epoch`.

## Reward / parsing

- **`bad task magic`** — a stale harness binary read a newer payload (or vice
  versa). The wire magic is `AXH2`; delete the cached binary under
  `src/decoder/rl/native/harness/_build/` to force a rebuild.
- **Energy doesn't match my Python reference** — check you passed the same
  `TaskWeights` (including the `*_rho` penalties and `coupling_weight`) to both
  sides; `reference_energy(residuals, classes, weights)` mirrors the SDK.

## Per-framework adapters

- **`ImportError: ... pip install aes-xts-decoder[<extra>]`** — the adapter's
  framework isn't installed; install the named extra.
- **Verifiers**: needs `datasets`; component reward funcs read the memoised
  `RolloutResult` from `state`, so the solver is compiled once per rollout.
- **GEM**: package is `gem-llm` (imports as `gem`); `step()` returns the
  Gymnasium 5-tuple with `terminated=True` (single-turn).
- **SkyRL Gym**: package is `skyrl-gym` (imports as `skyrl_gym`); `step()`
  returns a `BaseTextEnvStepOutput` — keep its `metadata` dict typed as a plain
  dict (that's `RolloutResult.to_dict()`).
- **OpenEnv**: package is `openenv-core`; needs Python ≥ 3.11 and a FastMCP
  install. Only MCP-backed base classes expose `@mcp.tool`.
- **NeMo Gym**: package is `nemo-gym`; needs Python ≥ 3.12. A `/verify` `422`
  means the request body shape didn't match the expected schema — check the
  `response` field your agent sends.
- **ORS**: if the `ors` package isn't on your index, install it from its source
  repo per the guide; the adapter only needs `@tool` and `ToolOutput`.

## HTTP adapters in Docker

- Build the base first (`docker build -f docker/Dockerfile.base -t aes-solver-base .`)
  then the per-framework image. The image must contain the C++ toolchain, the
  SDK/harness sources, and `bwrap`.
- If `bwrap` fails inside the container, relax the container's seccomp/userns
  policy (see `docker/docker-compose.yml`) rather than disabling the in-process
  sandbox.
