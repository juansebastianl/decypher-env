# OpenEnv (Meta PyTorch / Hugging Face)

HTTP/server framework using the MCP tool protocol. PyPI package is
**`openenv-core`** (Python >= 3.11); MCP-backed environments subclass
`MCPEnvironment` and register tools with `@mcp.tool` on a `FastMCP` server.

## Install

```bash
pip install "aes-xts-decoder[openenv]"   # openenv-core + uvicorn (+ FastMCP)
```

## Load / serve

```python
from src.decoder.rl.adapters.openenv_env import load_environment, create_app

env = load_environment(max_rounds=2)     # MCPEnvironment exposing submit_solver
app = create_app(max_rounds=2)           # ASGI app for uvicorn
```

The environment exposes a single MCP tool, `submit_solver(source)`, which
compiles+scores the C++ in the sandbox and returns the full
`RolloutResult.to_dict()` (reward, compile log, per-task metrics). The episode is
single-turn.

## Deploy with Docker

```bash
docker build -f docker/Dockerfile.base   -t aes-solver-base .
docker build -f docker/Dockerfile.openenv -t aes-solver-openenv .
docker run --rm -p 8000:8000 --security-opt seccomp=unconfined aes-solver-openenv
```

The image bundles the C++ toolchain, the SDK/harness sources, and `bwrap`.
Running as an isolated service is the recommended deployment shape for the
sandbox.

## Framework-neutral scoring

`submit_solver_result(source, ...)` is exported for unit tests and reuse without
OpenEnv installed; it returns a `RolloutResult` directly.
