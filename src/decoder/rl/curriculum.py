"""Reduced-round curriculum and augmented-Lagrangian schedule.

The sparse "exact-key-only" reward is densified three ways, all expressed here:

1. A **rounds ladder**: a task is scored at several reduced round counts
   (``1, 2, ...``) so a solver that only cracks a 1-round instance still earns
   signal long before the full 14-round problem is within reach.
2. A **coupling schedule**: the multiplier on inter-round consistency residuals
   ramps from small (rounds drift independently and are each solvable alone) to
   large (rounds are forced to agree).
3. A **penalty (rho) schedule** and a **method-of-multipliers dual update**: the
   augmented-Lagrangian energy is ``sum_i [ lambda_i r_i + 1/2 rho_i r_i^2 ]``.
   The penalty ``rho`` is ramped up across stages, and after each stage the
   Lagrange multipliers are updated ``lambda_i <- lambda_i + rho_i * r_i`` using
   the residual mass the best assignment left, which is the classical method of
   multipliers rather than a hand-set weight schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from ..xts_model import (
    block_specs_for_fixture,
    block_specs_for_model,
    build_xts_block_collection,
    load_fixture,
)
from .harness import FlatCircuit, TaskWeights, flatten_circuit

AES_FULL_ROUNDS = 14


@dataclass(frozen=True)
class TaskSpec:
    """One scored problem: a fixture window evaluated across a rounds ladder."""

    metadata_path: Path
    start_block: int = 0
    block_count: int = 1
    rounds_ladder: tuple[int, ...] = (1, 2, 3)
    seed: int = 0
    coupling_weight: float = 1.0
    # Per-class Lagrange multipliers (linear term of the augmented Lagrangian).
    ascii_weight: float = 1.0
    consistency_weight: float = 1.0
    goal_weight: float = 1.0
    # Per-class quadratic penalty coefficients (rho). Default 0.0 keeps the pure
    # linear energy unless a curriculum ramps them.
    ascii_rho: float = 0.0
    consistency_rho: float = 0.0
    goal_rho: float = 0.0
    # Reward weight applied to each rung; defaults to emphasising the easier
    # (lower-round) rungs so signal appears early in training.
    rung_weights: tuple[float, ...] | None = None

    def weight_for_rounds(self, aes_rounds: int) -> float:
        if self.rung_weights is not None:
            index = self.rounds_ladder.index(aes_rounds)
            return self.rung_weights[index]
        # Geometrically decaying emphasis: 1.0, 0.5, 0.25, ... over the ladder.
        index = self.rounds_ladder.index(aes_rounds)
        return 0.5 ** index

    def build_circuits(self):
        fixture = load_fixture(self.metadata_path)
        specs = block_specs_for_fixture(
            fixture, start_block=self.start_block, block_count=self.block_count
        )
        circuits = []
        for aes_rounds in self.rounds_ladder:
            model_specs = block_specs_for_model(fixture, specs, aes_rounds=aes_rounds)
            circuit = build_xts_block_collection(
                model_specs, encoding="lowered", aes_rounds=aes_rounds
            )
            circuits.append((aes_rounds, circuit))
        return circuits

    def build_flat_tasks(self) -> list[tuple[FlatCircuit, TaskWeights]]:
        tasks: list[tuple[FlatCircuit, TaskWeights]] = []
        for aes_rounds, circuit in self.build_circuits():
            flat = flatten_circuit(circuit, aes_rounds)
            weights = TaskWeights(
                ascii_weight=self.ascii_weight,
                consistency_weight=self.consistency_weight,
                goal_weight=self.goal_weight,
                ascii_rho=self.ascii_rho,
                consistency_rho=self.consistency_rho,
                goal_rho=self.goal_rho,
                coupling_weight=self.coupling_weight,
                seed=self.seed,
            )
            tasks.append((flat, weights))
        return tasks

    def dual_update(self, per_class_residuals) -> "TaskSpec":
        """Return a new TaskSpec with multipliers advanced by one dual step.

        Implements the method-of-multipliers update
        ``lambda_i <- lambda_i + rho_i * r_i`` for each constraint class, where
        ``r_i`` is the residual mass that class still had in the scored rollout
        (``per_class_residuals`` is ``[ascii, consistency, goal]``, e.g. from
        :attr:`~src.decoder.rl.harness.TaskResult.per_class`). Classes with a
        zero penalty ``rho`` are unchanged, as are classes with no residual.
        """

        residuals = list(per_class_residuals) + [0.0, 0.0, 0.0]
        ascii_r, consistency_r, goal_r = residuals[0], residuals[1], residuals[2]
        # The consistency penalty is scaled by the coupling weight in the SDK,
        # so the effective rho used for its dual step is scaled the same way.
        return replace(
            self,
            ascii_weight=self.ascii_weight + self.ascii_rho * ascii_r,
            consistency_weight=self.consistency_weight
            + self.consistency_rho * self.coupling_weight * consistency_r,
            goal_weight=self.goal_weight + self.goal_rho * goal_r,
        )


@dataclass
class Curriculum:
    """A staged sequence of :class:`TaskSpec` with a coupling + penalty ramp.

    Each stage widens the rounds ladder, raises the ``coupling_weight`` on
    inter-round consistency, and raises the augmented-Lagrangian penalty
    ``rho``. Between stages, :meth:`advance` runs the method-of-multipliers dual
    update on the previous stage's residuals so the Lagrange multipliers grow
    where constraints stayed violated.
    """

    metadata_path: Path
    block_count: int = 1
    start_block: int = 0
    max_rounds: int = 3
    seed: int = 0
    coupling_schedule: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0)
    # Quadratic penalty (rho) ramp, applied to the consistency and goal classes.
    # Rises alongside the coupling schedule so the penalty tightens as rounds are
    # forced to agree. ASCII keeps rho=0 by default (it is a soft prior).
    penalty_schedule: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

    def num_stages(self) -> int:
        return max(self.max_rounds, len(self.coupling_schedule), len(self.penalty_schedule))

    def task_for_stage(self, stage: int) -> TaskSpec:
        rounds = min(stage + 1, self.max_rounds)
        ladder = tuple(range(1, rounds + 1))
        coupling = self.coupling_schedule[min(stage, len(self.coupling_schedule) - 1)]
        rho = self.penalty_schedule[min(stage, len(self.penalty_schedule) - 1)]
        return TaskSpec(
            metadata_path=self.metadata_path,
            start_block=self.start_block,
            block_count=self.block_count,
            rounds_ladder=ladder,
            seed=self.seed,
            coupling_weight=coupling,
            consistency_rho=rho,
            goal_rho=rho,
        )

    def stages(self):
        for stage in range(self.num_stages()):
            yield self.task_for_stage(stage)

    def advance(self, stage: int, per_class_residuals) -> TaskSpec:
        """Next stage's spec with the dual update applied to this stage's residuals.

        This is the driver a training loop calls between stages: take the
        per-class residual mass the just-scored rollout left
        (``[ascii, consistency, goal]``), roll the schedule forward one stage,
        and carry the method-of-multipliers-updated Lagrange multipliers over so
        classes that stayed violated get a larger multiplier next stage.
        """

        next_stage = min(stage + 1, self.num_stages() - 1)
        base = self.task_for_stage(stage)
        updated = base.dual_update(per_class_residuals)
        nxt = self.task_for_stage(next_stage)
        # Carry the updated multipliers into the next stage's schedule-set rho /
        # coupling / ladder.
        return replace(
            nxt,
            ascii_weight=updated.ascii_weight,
            consistency_weight=updated.consistency_weight,
            goal_weight=updated.goal_weight,
        )
