"""Reward rubric: augmented-Lagrangian energy + round staircase + feasibility.

The reward is deterministic and verifiable (RLVR): it is a function of the hard
residuals the harness measured, with no learned reward model. Components are
returned separately so a GRPO trainer can inspect and reweight them.

Ordering guarantees, by construction:

    doesn't compile  <  compiles but crashes/times out  <  compiles and runs

so the policy is always rewarded for producing a solver that at least builds and
executes, even before it solves anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Bumped when the reward shaping or the RolloutResult schema changes, so a
# trainer or logged artifact can tell which contract produced a score.
REWARD_VERSION = "1.0"


@dataclass
class RewardBreakdown:
    reward: float
    compile_ok: bool
    feasible: bool
    components: dict[str, float] = field(default_factory=dict)
    per_task: list[dict] = field(default_factory=list)
    compile_stderr: str = ""

    def to_dict(self) -> dict:
        return {
            "reward": self.reward,
            "compile_ok": self.compile_ok,
            "feasible": self.feasible,
            "components": dict(self.components),
            "per_task": list(self.per_task),
            "compile_stderr": self.compile_stderr,
        }


@dataclass
class RolloutResult:
    """The first-class, framework-agnostic result of scoring one solver rollout.

    This is the single structured object every framework adapter maps into its
    own native shape (Verifiers ``state``, a Gymnasium ``info`` dict, an ORS
    ``ToolOutput``, a NeMo ``/verify`` body, ...). It is a superset of
    :class:`RewardBreakdown`: it exposes the same ``reward`` / ``compile_ok`` /
    ``feasible`` / ``components`` / ``per_task`` / ``compile_stderr`` fields, and
    adds the diagnostics a trainer wants to inspect or log:

    * ``source_diagnostics`` -- did we find a fenced code block in the model
      completion, did it contain ``create_solver``, how long was the extracted
      source (source-extraction is a common silent failure mode),
    * ``wall_seconds`` -- total solver wall-clock across the ladder rungs,
    * ``failure_kind`` -- a single coarse category
      (``ok`` / ``compile_error`` / ``timeout`` / ``crash`` / ``empty_source``),
    * ``artifacts`` -- optional on-disk paths (saved solver source, compile log,
      rollout JSON) when artifact logging is enabled, and
    * ``reward_version`` -- the contract version that produced the score.
    """

    reward: float
    compile_ok: bool
    feasible: bool
    components: dict[str, float] = field(default_factory=dict)
    per_task: list[dict] = field(default_factory=list)
    compile_stderr: str = ""
    source_diagnostics: dict = field(default_factory=dict)
    wall_seconds: float = 0.0
    failure_kind: str = "ok"
    artifacts: dict[str, str] = field(default_factory=dict)
    reward_version: str = REWARD_VERSION

    @classmethod
    def from_breakdown(
        cls,
        breakdown: RewardBreakdown,
        *,
        source_diagnostics: dict | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> "RolloutResult":
        wall = sum(float(t.get("seconds", 0.0)) for t in breakdown.per_task)
        return cls(
            reward=breakdown.reward,
            compile_ok=breakdown.compile_ok,
            feasible=breakdown.feasible,
            components=dict(breakdown.components),
            per_task=list(breakdown.per_task),
            compile_stderr=breakdown.compile_stderr,
            source_diagnostics=dict(source_diagnostics or {}),
            wall_seconds=wall,
            failure_kind=_classify_failure(breakdown, source_diagnostics or {}),
            artifacts=dict(artifacts or {}),
        )

    def to_dict(self) -> dict:
        return {
            "reward": self.reward,
            "compile_ok": self.compile_ok,
            "feasible": self.feasible,
            "components": dict(self.components),
            "per_task": list(self.per_task),
            "compile_stderr": self.compile_stderr,
            "source_diagnostics": dict(self.source_diagnostics),
            "wall_seconds": self.wall_seconds,
            "failure_kind": self.failure_kind,
            "artifacts": dict(self.artifacts),
            "reward_version": self.reward_version,
        }


def _classify_failure(breakdown: RewardBreakdown, source_diagnostics: dict) -> str:
    """Coarse single-label category for quick filtering/telemetry."""
    if source_diagnostics.get("source_chars", 1) == 0:
        return "empty_source"
    if not breakdown.compile_ok:
        return "compile_error"
    if not breakdown.per_task:
        return "no_tasks"
    if all(t.get("timed_out") for t in breakdown.per_task):
        return "timeout"
    if all(t.get("crashed") or not t.get("ran") for t in breakdown.per_task):
        return "crash"
    return "ok"


@dataclass
class Rubric:
    """Weighted reward shaping over a harness run."""

    energy_weight: float = 1.0
    round_weight: float = 1.0
    feasible_weight: float = 2.0
    compile_penalty: float = -1.0
    timeout_penalty: float = -0.5
    crash_penalty: float = -0.5

    def _energy_reward(self, energy: float, constraint_count: int) -> float:
        # Bounded in (0, 1]; 1.0 at zero energy, decaying smoothly as the
        # weighted residual mass grows. Normalised by circuit size so the shape
        # is comparable across round counts.
        scale = float(max(1, constraint_count))
        return math.exp(-energy / scale)

    def score(self, run, task_spec) -> RewardBreakdown:
        if not run.compile_ok:
            return RewardBreakdown(
                reward=self.compile_penalty,
                compile_ok=False,
                feasible=False,
                components={"compile_penalty": self.compile_penalty},
                per_task=[],
                compile_stderr=run.compile_stderr,
            )

        total_weight = 0.0
        weighted_reward = 0.0
        any_feasible = False
        per_task: list[dict] = []
        agg = {"energy": 0.0, "round": 0.0, "feasible": 0.0, "penalty": 0.0}

        for task in run.tasks:
            rung_weight = task_spec.weight_for_rounds(task.aes_rounds)
            total_weight += rung_weight

            if task.timed_out:
                task_reward = self.timeout_penalty
                r_energy = r_round = r_feasible = 0.0
                agg["penalty"] += rung_weight * self.timeout_penalty
            elif task.crashed or not task.ran:
                task_reward = self.crash_penalty
                r_energy = r_round = r_feasible = 0.0
                agg["penalty"] += rung_weight * self.crash_penalty
            else:
                r_energy = self._energy_reward(task.energy, task.constraint_count)
                r_round = task.round_staircase / max(1, task.aes_rounds)
                r_feasible = 1.0 if task.feasible else 0.0
                any_feasible = any_feasible or task.feasible
                task_reward = (
                    self.energy_weight * r_energy
                    + self.round_weight * r_round
                    + self.feasible_weight * r_feasible
                )
                agg["energy"] += rung_weight * self.energy_weight * r_energy
                agg["round"] += rung_weight * self.round_weight * r_round
                agg["feasible"] += rung_weight * self.feasible_weight * r_feasible

            weighted_reward += rung_weight * task_reward
            per_task.append(
                {
                    "aes_rounds": task.aes_rounds,
                    "rung_weight": rung_weight,
                    "ran": task.ran,
                    "reward": task_reward,
                    "energy": task.energy,
                    "energy_reward": r_energy,
                    "round_staircase": task.round_staircase,
                    "round_reward": r_round,
                    "feasible": task.feasible,
                    "timed_out": task.timed_out,
                    "crashed": task.crashed,
                    "best_hamming": task.best_hamming,
                    "per_round": task.per_round,
                    "per_class": task.per_class,
                    "seconds": task.seconds,
                    "error": task.error,
                }
            )

        reward = weighted_reward / total_weight if total_weight > 0 else 0.0
        components = {key: (value / total_weight if total_weight > 0 else 0.0) for key, value in agg.items()}
        return RewardBreakdown(
            reward=reward,
            compile_ok=True,
            feasible=any_feasible,
            components=components,
            per_task=per_task,
            compile_stderr=run.compile_stderr,
        )
