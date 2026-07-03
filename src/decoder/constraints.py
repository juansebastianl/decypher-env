"""Constraint classification shared by the task builder and the RL environment.

This lives outside ``backends/`` so the environment (``rl/``) can classify
constraints without importing any of the legacy solver machinery.
"""

from __future__ import annotations

from enum import IntEnum

from .circuit import Circuit, ConstraintKind


class ConstraintClass(IntEnum):
    """Structural role of a constraint, used to weight the Lagrangian."""

    ASCII = 1
    CONSISTENCY = 2
    GOAL = 3


def classify_constraints(circuit: Circuit) -> tuple[ConstraintClass, ...]:
    """Label each constraint by structural role (ASCII / consistency / goal)."""

    classes: list[ConstraintClass] = []
    for constraint in circuit.constraints:
        if constraint.kind == ConstraintKind.ASCII_PRINTABLE:
            classes.append(ConstraintClass.ASCII)
        elif constraint.kind == ConstraintKind.DEFINE8:
            classes.append(ConstraintClass.CONSISTENCY)
        else:
            classes.append(ConstraintClass.GOAL)
    return tuple(classes)
