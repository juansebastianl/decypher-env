"""The solver-authoring RL environment (framework-agnostic core).

Ties together the pieces every RL-environment framework needs:

- **dataset**: task specs drawn from a fixture window and a rounds curriculum,
- **prompt**: the task description + the ``ISolver`` contract + a baseline
  template + a pointer to the SDK surface,
- **rollout**: extract C++ from the model completion and compile-and-score it,
- **rubric**: the augmented-Lagrangian reward with components exposed.

This module is deliberately framework-neutral: it turns a model completion into
a structured :class:`~src.decoder.rl.reward.RolloutResult`. The thin per-
framework mappings (Verifiers, GEM, SkyRL Gym, OpenEnv, ORS, NeMo Gym) live in
:mod:`src.decoder.rl.adapters`; each wraps :meth:`SolverAuthoringEnv.score_source`
/ :meth:`score_completion` / :meth:`score_batch` and maps ``RolloutResult`` into
that framework's native reward/state/info shape.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .curriculum import Curriculum, TaskSpec
from .harness import HarnessOptions, evaluate_solver_source
from .reward import RolloutResult

_CODE_FENCE_RE = re.compile(r"```(?:c\+\+|cpp|cxx|c)?\s*\n(.*?)```", re.DOTALL)

CONTRACT = """\
You are authoring a C++ solver plugin for AES-XTS key/plaintext recovery.

Emit a single C++ translation unit that:
  * #include "solver_sdk.hpp"
  * defines a class deriving from aes_xts_decoder::sdk::ISolver (or the
    convenience base aes_xts_decoder::sdk::SolverBase), and
  * exports  extern "C" ISolver* create_solver(const SolverContext&).

The harness compiles your source against the SDK, constructs one solver per
task with SolverContext{circuit, config, initial}, then repeatedly calls
RunEpoch(sweeps); afterwards it reads CurrentAssignment(), DrainFeasible(...),
and scores the result. Reward = a bounded function of the augmented-Lagrangian
energy  Energy = sum_i [ lambda_i * r_i + 1/2 * rho_i * r_i^2 ]  (per-class
Lagrange multipliers lambda and quadratic penalties rho), plus a per-round
"staircase" for each consecutive AES round made internally consistent, plus a
large bonus for a feasible (zero-residual) assignment. Lower energy and more
solved rounds are always better; a solver that merely compiles and runs already
scores above one that fails to build.

SDK surface you may call (see solver_sdk.hpp):
  * SdkCircuit::Score(assignment) -> ScoreData (hard residuals, hamming_score)
  * SdkCircuit::Energy(score, config), PerRoundResiduals, RoundStaircase
  * SdkCircuit::DeriveWires(&assignment)      (recompute internal wires)
  * SdkCircuit::RepairPlaintext(&assignment)  (exact block-Gibbs from the keys)
  * RngNext/RngUniform/RngRange/RandomTextAscii (deterministic RNG)
Return only the C++ source, optionally in a ```cpp fenced block.
"""


def extract_source(completion: str) -> str:
    """Pull the C++ source out of a model completion (fenced or raw)."""

    matches = _CODE_FENCE_RE.findall(completion)
    if matches:
        # Prefer the block that looks like a solver.
        for block in matches:
            if "create_solver" in block:
                return block.strip()
        return matches[0].strip()
    return completion.strip()


def source_diagnostics(completion: str, source: str) -> dict:
    """Diagnostics about how source was recovered from a model completion.

    Source-extraction is a common silent failure mode (the model narrates
    instead of emitting a code block, or omits ``create_solver``); surfacing it
    lets a trainer tell "bad solver" apart from "no solver in the completion".
    """

    matches = _CODE_FENCE_RE.findall(completion)
    return {
        "found_fenced_block": bool(matches),
        "num_fenced_blocks": len(matches),
        "has_create_solver": "create_solver" in source,
        "source_chars": len(source),
        "extracted_from_raw": not bool(matches),
    }


def build_prompt(task_spec: TaskSpec) -> str:
    ladder = ", ".join(str(r) for r in task_spec.rounds_ladder)
    return (
        f"{CONTRACT}\n"
        f"Task: recover an ASCII plaintext / key for a {task_spec.block_count}-block "
        f"AES-XTS window. You are scored on reduced-round instances with "
        f"aes_rounds in [{ladder}] (coupling_weight={task_spec.coupling_weight}). "
        f"See examples/README.md and examples/solvers/baseline/ for a template."
    )


@dataclass
class SolverAuthoringEnv:
    """Environment over a list of :class:`TaskSpec` (or a :class:`Curriculum`).

    The core exposes three primitives the framework adapters build on:
    :meth:`prompts` (the dataset), :meth:`score_source` / :meth:`score_completion`
    (score one rollout), and :meth:`score_batch` (score many). Each scoring call
    returns a :class:`~src.decoder.rl.reward.RolloutResult`. When ``artifacts_dir``
    is set, each rollout also writes the extracted solver source, the compile log
    and a JSON of the result there, and records the paths on the result.
    """

    task_specs: list[TaskSpec] = field(default_factory=list)
    options: HarnessOptions = field(default_factory=HarnessOptions)
    artifacts_dir: Path | None = None

    @classmethod
    def from_curriculum(cls, curriculum: Curriculum, options: HarnessOptions | None = None) -> "SolverAuthoringEnv":
        return cls(task_specs=list(curriculum.stages()), options=options or HarnessOptions())

    @classmethod
    def single(
        cls,
        metadata_path: Path,
        *,
        block_count: int = 1,
        rounds_ladder: tuple[int, ...] = (1, 2, 3),
        seed: int = 0,
        coupling_weight: float = 1.0,
        options: HarnessOptions | None = None,
    ) -> "SolverAuthoringEnv":
        spec = TaskSpec(
            metadata_path=metadata_path,
            block_count=block_count,
            rounds_ladder=rounds_ladder,
            seed=seed,
            coupling_weight=coupling_weight,
        )
        return cls(task_specs=[spec], options=options or HarnessOptions())

    # -- dataset -------------------------------------------------------------
    def prompts(self) -> list[str]:
        return [build_prompt(spec) for spec in self.task_specs]

    # -- rollout + rubric ----------------------------------------------------
    def score_completion(self, completion: str, task_spec: TaskSpec) -> RolloutResult:
        """Extract C++ from a model completion, compile+score it, return the result."""
        source = extract_source(completion)
        diagnostics = source_diagnostics(completion, source)
        return self._score(source, task_spec, diagnostics)

    def score_source(self, source: str, task_spec: TaskSpec) -> RolloutResult:
        """Compile+score a raw solver source string (no completion parsing)."""
        diagnostics = source_diagnostics(source, source)
        return self._score(source, task_spec, diagnostics)

    def score_batch(
        self,
        sources: list[str],
        specs: TaskSpec | list[TaskSpec],
    ) -> list[RolloutResult]:
        """Score many solver sources (e.g. a GRPO group) reusing the cached harness.

        ``specs`` may be a single :class:`TaskSpec` applied to every source, or a
        list aligned with ``sources``.
        """
        if isinstance(specs, TaskSpec):
            specs = [specs] * len(sources)
        if len(specs) != len(sources):
            raise ValueError("sources and specs must be the same length")
        return [self.score_source(src, spec) for src, spec in zip(sources, specs)]

    # -- internals -----------------------------------------------------------
    def _score(self, source: str, task_spec: TaskSpec, diagnostics: dict) -> RolloutResult:
        breakdown = evaluate_solver_source(source, task_spec, self.options)
        artifacts = self._write_artifacts(source, breakdown) if self.artifacts_dir else None
        result = RolloutResult.from_breakdown(
            breakdown, source_diagnostics=diagnostics, artifacts=artifacts
        )
        if artifacts is not None:
            self._finalize_rollout_json(result, artifacts)
        return result

    def _write_artifacts(self, source: str, breakdown) -> dict[str, str]:
        base = Path(self.artifacts_dir)
        base.mkdir(parents=True, exist_ok=True)
        rollout_id = uuid.uuid4().hex[:12]
        source_path = base / f"solver_{rollout_id}.cpp"
        source_path.write_text(source, encoding="utf-8")
        artifacts = {"solver_source": str(source_path)}
        if breakdown.compile_stderr:
            log_path = base / f"compile_{rollout_id}.log"
            log_path.write_text(breakdown.compile_stderr, encoding="utf-8")
            artifacts["compile_log"] = str(log_path)
        artifacts["rollout_json"] = str(base / f"rollout_{rollout_id}.json")
        return artifacts

    @staticmethod
    def _finalize_rollout_json(result: RolloutResult, artifacts: dict[str, str]) -> None:
        path = artifacts.get("rollout_json")
        if path:
            Path(path).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
