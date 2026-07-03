# Open Reward Standard (ORS)

HTTP/server framework where tools are `@tool` functions returning a
`ToolOutput(blocks=..., reward=..., finished=...)`.

## Install

```bash
pip install "aes-xts-decoder[ors]"
```

If the `ors` package is not available on your index, install it from its source
repository (see the RL-environments guide); the adapter only needs the `@tool`
decorator and `ToolOutput`.

## Load / serve

```python
from src.decoder.rl.adapters.ors_env import load_environment, serve, list_tasks

env = load_environment(max_rounds=2)     # an ors.Environment
serve(max_rounds=2)                      # Server([env]).run()
list_tasks("smoke")                      # task ids backed by the benchmark suites
```

The environment exposes one tool, `submit_solver(source)`, which compiles+scores
the C++ in the sandbox and returns
`ToolOutput(blocks=[...], reward=result.reward, finished=True)`. `list_tasks`
splits are backed by the benchmark suites (`smoke`/`public`/`hard`).

## Deploy with Docker

```bash
docker build -f docker/Dockerfile.base -t aes-solver-base .
docker build -f docker/Dockerfile.ors  -t aes-solver-ors .
docker run --rm -p 8001:8001 --security-opt seccomp=unconfined aes-solver-ors
```

The image bundles the C++ toolchain, the SDK/harness sources, and `bwrap`.

## Framework-neutral scoring

`submit_solver_result(source, ...)` returns a `RolloutResult` directly and is
unit-testable without ORS installed.
