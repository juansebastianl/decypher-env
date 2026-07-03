# GEM (Axon-RL)

In-process framework following the Gymnasium API. PyPI package is **`gem-llm`**
and it imports as `gem`.

## Install

```bash
pip install "aes-xts-decoder[gem]"   # gem-llm
```

## Load

```python
from src.decoder.rl.adapters.gem_env import load_environment, register

env = load_environment(max_rounds=2)      # a gem.Env instance
# or register under an id so gem.make(...) works:
register("SolverAuthoring-v0")
import gem
env = gem.make("SolverAuthoring-v0")
```

## Interface

Single-turn Gymnasium env:

```python
obs, info = env.reset()                    # obs = the prompt
obs, reward, terminated, truncated, info = env.step(completion)
# terminated=True, truncated=False; info = RolloutResult.to_dict()
```

`step(completion)` compiles+scores the model's C++ (in the sandbox) and returns
the deterministic reward as the Gymnasium 5-tuple, with the full
`RolloutResult` as `info`.

## Minimal train

GEM ships single-file examples for `oat` and `verl`; point them at the
registered id. Each model response is one action; the reward is dense (energy +
round staircase) plus the feasibility bonus.
