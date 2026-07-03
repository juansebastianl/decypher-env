"""Per-adapter smoke tests.

None of the six RL-environment frameworks are required to import the adapter
package, so these assert two things that must hold regardless of what is
installed:

* every adapter module imports without its framework present, and
  ``load_environment()`` either returns an environment (framework installed) or
  raises a clean, actionable ``ImportError`` telling you which extra to install;
* the framework-neutral scoring path the HTTP adapters wrap actually compiles
  and scores a solver (this one needs the C++ toolchain).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.decoder.rl.adapters import FRAMEWORKS, load_environment
from src.decoder.rl.adapters import _common
from src.decoder.rl.harness import HarnessOptions, harness_available
from src.decoder.rl.reward import RolloutResult

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "solvers"

requires_toolchain = pytest.mark.skipif(
    not harness_available(), reason="C++ toolchain to build the harness is unavailable"
)

_MODULES = {
    "verifiers": "verifiers_env",
    "gem": "gem_env",
    "skyrl": "skyrl_env",
    "openenv": "openenv_env",
    "ors": "ors_env",
    "nemo": "nemo_env",
}


@pytest.mark.parametrize("module_name", sorted(_MODULES.values()))
def test_adapter_module_imports_without_framework(module_name: str) -> None:
    module = importlib.import_module(f"src.decoder.rl.adapters.{module_name}")
    assert hasattr(module, "load_environment")


@pytest.mark.parametrize("framework", sorted(FRAMEWORKS))
def test_load_environment_returns_or_raises_clean_error(framework: str) -> None:
    try:
        env = load_environment(framework, max_rounds=1)
    except ImportError as exc:
        # Missing framework: the message must tell the user how to install it.
        assert "pip install" in str(exc)
    else:
        # Framework is installed: we at least got an object back.
        assert env is not None


def test_unknown_framework_raises_value_error() -> None:
    with pytest.raises(ValueError):
        load_environment("not-a-framework")


def test_completion_text_normalisation() -> None:
    assert _common.completion_text("raw source") == "raw source"
    assert _common.completion_text({"content": "x"}) == "x"
    assert _common.completion_text([{"role": "assistant", "content": "y"}]) == "y"
    assert _common.completion_text([]) == "[]"


@requires_toolchain
@pytest.mark.parametrize("module_name", ["openenv_env", "ors_env", "nemo_env"])
def test_http_adapter_scoring_core_compiles_and_scores(module_name: str) -> None:
    module = importlib.import_module(f"src.decoder.rl.adapters.{module_name}")
    source = (EXAMPLES / "baseline" / "solver.cpp").read_text(encoding="utf-8")
    options = HarnessOptions(run_timeout=15.0, epochs=1, sweeps_per_epoch=200)
    result = module.submit_solver_result(source, max_rounds=1, options=options)
    assert isinstance(result, RolloutResult)
    assert result.compile_ok, result.compile_stderr
    # The structured result round-trips to a plain dict for the transport layer.
    assert result.to_dict()["reward"] == result.reward
