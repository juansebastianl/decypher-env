"""Constraint relaxation strategies."""

from .base import ContinuousState, Relaxation
from .continuous import ByteDistributionRelaxation
from .discrete import DiscreteLocalRelaxation

__all__ = ["ByteDistributionRelaxation", "ContinuousState", "DiscreteLocalRelaxation", "Relaxation"]


def get_relaxation(name: str) -> Relaxation:
    if name == "discrete":
        return DiscreteLocalRelaxation()
    if name == "continuous":
        return ByteDistributionRelaxation()
    raise ValueError(f"unknown relaxation: {name}")
