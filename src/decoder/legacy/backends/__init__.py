"""Sampler engine backend registry."""

from __future__ import annotations

from .base import SamplerEngine
from .native import (
    NativeAlgebraicRelaxedEngine,
    NativeContinuousRelaxedEngine,
    NativeParallelTemperingEngine,
    NativeSamplerEngine,
)
from .python import PythonSamplerEngine


def available_backends() -> tuple[str, ...]:
    names = ["python"]
    if NativeSamplerEngine.is_available():
        names.append("native")
    if NativeParallelTemperingEngine.is_available():
        names.append("native-pt")
    if NativeContinuousRelaxedEngine.is_available():
        names.append("native-continuous-relaxed")
    if NativeAlgebraicRelaxedEngine.is_available():
        names.append("native-algebraic-relaxed")
    return tuple(names)


def get_backend(name: str = "auto") -> type[SamplerEngine]:
    if name == "auto":
        if NativeSamplerEngine.is_available() and NativeSamplerEngine.supports_native_residuals:
            return NativeSamplerEngine
        return PythonSamplerEngine
    if name == "python":
        return PythonSamplerEngine
    if name == "native":
        if not NativeSamplerEngine.is_available():
            raise RuntimeError("native backend is not available; build the Cython extension first")
        return NativeSamplerEngine
    if name == "native-pt":
        if not NativeParallelTemperingEngine.is_available():
            raise RuntimeError("native PT backend is not available; build the Cython extension first")
        return NativeParallelTemperingEngine
    if name == "native-continuous-relaxed":
        if not NativeContinuousRelaxedEngine.is_available():
            raise RuntimeError("native continuous-relaxed backend is not available; build the Cython extension first")
        return NativeContinuousRelaxedEngine
    if name == "native-algebraic-relaxed":
        if not NativeAlgebraicRelaxedEngine.is_available():
            raise RuntimeError("native algebraic-relaxed backend is not available; build the Cython extension first")
        return NativeAlgebraicRelaxedEngine
    raise ValueError(f"unknown backend: {name}")
