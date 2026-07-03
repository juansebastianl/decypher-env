# Verifiers (PrimeIntellect)

In-process framework. The environment is a `vf.SingleTurnEnv` with a `Dataset`
of prompts and a `vf.Rubric` of plain-Python reward functions.

## Install

```bash
pip install "aes-xts-decoder[verifiers]"   # verifiers + datasets, Python >= 3.11
```

## Load

```python
from src.decoder.rl.adapters.verifiers_env import load_environment

env = load_environment(max_rounds=2)   # -> vf.SingleTurnEnv
```

`load_environment()` is the Environments-Hub / prime-rl entry point. It builds
the dataset from the reduced-round curriculum and a rubric whose scalar reward
drives training while the energy / round-staircase / feasibility components are
attached as zero-weight rubric functions for inspection. The full
`RolloutResult.to_dict()` is stored on the verifiers `state` (key
`rollout_result`), so compile logs, failure categories and wall time survive.

The solver is compiled and run **once per rollout** (memoised on `state`), even
though several component functions read from it.

## Minimal eval

```python
env = load_environment(max_rounds=1)
# Drive it with your prime-rl / verifiers trainer, e.g.:
#   prime env eval  (or the verifiers Python API)
# The rubric returns the deterministic RLVR reward; no reward model is used.
```

## Notes

- Reward = augmented-Lagrangian energy + round staircase + feasibility bonus,
  with failures shaped (`compile < crash/timeout < runs`).
- See [security-sandbox.md](../security-sandbox.md): solvers compile+run in a
  bwrap+seccomp sandbox, so the training box only needs the C++ toolchain and
  `bwrap`.
