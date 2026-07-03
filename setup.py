from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).parent

# setuptools requires extension sources/include dirs to be /-separated paths
# relative to setup.py (absolute paths break editable installs). Everything
# under _NATIVE is inside the project tree, so express it relative to ROOT.
_NATIVE = ROOT / "src" / "decoder" / "legacy" / "backends" / "native"
_SOLVERS = _NATIVE / "solvers"


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# The native `_bridge` extension powers the *legacy* reference solver engines
# (parallel tempering / continuous / algebraic relaxations). It is internals:
# the RL environment does not depend on it. The environment's own solver SDK and
# compile-and-run harness live under src/decoder/rl/native and are compiled on
# demand (see package_data below), not baked into this extension.
_bridge_sources = [
    _NATIVE / "_bridge.pyx",
    _SOLVERS / "baseline.cpp",
    _SOLVERS / "algebraic_relaxed.cpp",
    _SOLVERS / "algebraic_relaxed" / "algebraic_components.cpp",
    _SOLVERS / "algebraic_relaxed" / "belief_propagation.cpp",
    _SOLVERS / "continuous_relaxed.cpp",
    _SOLVERS / "parallel_tempering.cpp",
    _SOLVERS / "parallel_tempering" / "api.cpp",
    _SOLVERS / "parallel_tempering" / "byte_evaluator.cpp",
    _SOLVERS / "parallel_tempering" / "circuit_model.cpp",
    _SOLVERS / "parallel_tempering" / "constructor.cpp",
    _SOLVERS / "parallel_tempering" / "diagnostics.cpp",
    _SOLVERS / "parallel_tempering" / "energy.cpp",
    _SOLVERS / "parallel_tempering" / "evaluation.cpp",
    _SOLVERS / "parallel_tempering" / "hamming_scorer.cpp",
    _SOLVERS / "parallel_tempering" / "helpers.cpp",
    _SOLVERS / "parallel_tempering" / "moves.cpp",
    _SOLVERS / "parallel_tempering" / "optimization.cpp",
    _SOLVERS / "parallel_tempering" / "proposals.cpp",
    _SOLVERS / "parallel_tempering" / "schedule.cpp",
    _SOLVERS / "moves" / "mutate.cpp",
    _SOLVERS / "sampler" / "metropolis.cpp",
]
extensions = [
    Extension(
        "src.decoder.legacy.backends.native._bridge",
        [_rel(source) for source in _bridge_sources],
        include_dirs=[_rel(_NATIVE / "include")],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    )
]

# The RL environment compiles authored solvers on demand rather than baking
# them into the extension. The compile-and-run harness (a small standalone
# binary) and the solver SDK are built at runtime by src/decoder/rl/harness.py
# from these shipped sources, so they must travel with the package. The example
# solvers under examples/ deliberately do NOT ship as compiled units -- they are
# documentation-grade illustrations compiled through the same on-demand path a
# model-authored solver uses.
# Project metadata (name, version, dependencies, optional extras) lives in
# pyproject.toml (PEP 621). setup.py keeps only the imperative build config that
# pyproject cannot express: the Cython `_bridge` extension, package discovery,
# and the on-demand-compiled SDK/harness package_data.
setup(
    packages=find_packages(include=("src", "src.*")),
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    package_data={
        # The RL environment's solver SDK + compile-and-run harness travel with
        # the package so authored solvers can be built on demand.
        "src.decoder.rl": [
            "native/include/*.hpp",
            "native/sdk/*.cpp",
            "native/harness/*.hpp",
            "native/harness/*.cpp",
        ],
    },
    include_package_data=True,
)
