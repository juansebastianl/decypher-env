# NeMo Gym (NVIDIA)

HTTP/server framework: three async FastAPI services (agent / model / resources).
Custom tasks live in a *resources server* subclassing `SimpleResourcesServer`
with a post-episode `/verify` endpoint. PyPI package is **`nemo-gym`**
(**Python >= 3.12**).

## Install

```bash
pip install "aes-xts-decoder[nemo]"   # nemo-gym + fastapi + uvicorn + pydantic
```

## Load / serve

```python
from src.decoder.rl.adapters.nemo_env import load_environment, create_app, run_webserver

ResourcesServer = load_environment(max_rounds=2)   # SimpleResourcesServer subclass
app = create_app(max_rounds=2)                     # plain FastAPI app (local testing)
run_webserver(port=8000, max_rounds=2)             # serve create_app() with uvicorn
```

The resources server registers `POST /submit_solver` (the model's C++ solver)
and implements `verify(body)` for the post-episode `POST /verify`, which
recomputes and returns the reward plus the full `RolloutResult` metrics. Both
route through the same sandboxed scoring core.

## Deploy with Docker

```bash
docker build -f docker/Dockerfile.base -t aes-solver-base .
docker build -f docker/Dockerfile.nemo -t aes-solver-nemo .
docker run --rm -p 8000:8000 --security-opt seccomp=unconfined aes-solver-nemo
```

The image bundles the C++ toolchain, the SDK/harness sources, and `bwrap`.

## Gotchas

- A `/verify` **422** means the request body didn't match the expected schema —
  check the `response` field your agent sends (the adapter extracts the C++ from
  it best-effort).
- NeMo Gym needs Python ≥ 3.12; the Docker base image is 3.12 to match.

## Framework-neutral scoring

`submit_solver_result(source, ...)` returns a `RolloutResult` directly and is
unit-testable without NeMo Gym installed.
