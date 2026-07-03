"""Framework adapters for the solver-authoring environment.

Rather than invent a bespoke environment contract, the environment ships one
thin adapter per major RL-environment framework. Each adapter maps the
framework-neutral :class:`~src.decoder.rl.reward.RolloutResult` into that
framework's native reward/state/info shape, and imports its framework dependency
lazily so importing this package never requires any of them installed.

Adapters and their install extras:

======================  ============================  ================
adapter module          framework                     install extra
======================  ============================  ================
``verifiers_env``       Verifiers (PrimeIntellect)    ``[verifiers]``
``gem_env``             GEM (Axon-RL)                 ``[gem]``
``skyrl_env``           SkyRL Gym (NovaSky)           ``[skyrl]``
``openenv_env``         OpenEnv (Meta PyTorch)        ``[openenv]``
``ors_env``             Open Reward Standard          ``[ors]``
``nemo_env``            NeMo Gym (NVIDIA)             ``[nemo]``
======================  ============================  ================

Each module exposes ``load_environment(**kwargs)``. Use :func:`load_environment`
here to dispatch by framework name.
"""

from __future__ import annotations

import importlib

# framework name -> submodule providing load_environment()
_ADAPTERS = {
    "verifiers": "verifiers_env",
    "gem": "gem_env",
    "skyrl": "skyrl_env",
    "openenv": "openenv_env",
    "ors": "ors_env",
    "nemo": "nemo_env",
}

FRAMEWORKS = tuple(_ADAPTERS)


def load_environment(framework: str, **kwargs):
    """Load the environment for ``framework`` (see :data:`FRAMEWORKS`)."""
    key = framework.lower()
    if key not in _ADAPTERS:
        raise ValueError(
            f"unknown framework {framework!r}; choose one of {', '.join(FRAMEWORKS)}"
        )
    module = importlib.import_module(f"{__name__}.{_ADAPTERS[key]}")
    return module.load_environment(**kwargs)


__all__ = ["FRAMEWORKS", "load_environment"]
