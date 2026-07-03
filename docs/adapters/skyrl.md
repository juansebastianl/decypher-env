# SkyRL Gym (NovaSky / Berkeley)

In-process framework built on the Gymnasium API around `BaseTextEnv`. PyPI
package is **`skyrl-gym`** and it imports as `skyrl_gym`.

## Install

```bash
pip install "aes-xts-decoder[skyrl]"   # skyrl-gym
```

## Load

```python
from src.decoder.rl.adapters.skyrl_env import load_environment, register

env = load_environment(max_rounds=2)   # a BaseTextEnv subclass instance
register("SolverAuthoring-v0")         # register with skyrl_gym
```

## Interface

Single-turn text env:

```python
prompt, info = env.init()
step_output = env.step(completion)      # BaseTextEnvStepOutput
step_output.reward                      # deterministic RLVR reward
step_output.done                        # True (single-turn)
step_output.metadata                    # RolloutResult.to_dict()
```

## Notes

- Keep `metadata` typed as a plain `dict` (that's exactly
  `RolloutResult.to_dict()`); SkyRL's `step()` typing expects a dict there.
- The solver compiles+runs in the bwrap+seccomp sandbox (see
  [security-sandbox.md](../security-sandbox.md)).
