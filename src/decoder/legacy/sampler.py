"""Sampler API for AES-XTS circuit candidates."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from ..ascii_constraints import TEXT_ASCII
from ..circuit import Assignment, Circuit, EvaluationResult
from .backends import get_backend
from .backends.base import SamplerEngine
from ..xts_model import (
    Fixture,
    assignment_from_fixture_plaintext,
    block_specs_for_fixture,
    block_specs_for_model,
    build_xts_block_collection,
    load_fixture,
)


@dataclass(frozen=True)
class SamplerConfig:
    metadata_path: Path
    start_block: int = 0
    block_count: int = 1
    fixed_key: bool = False
    seed: int | None = None
    backend: str = "auto"
    encoding: str = "opaque"
    relaxation: str = "discrete"
    aes_rounds: int = 14


@dataclass
class SampleState:
    assignment: Assignment
    result: EvaluationResult


class XtsSampler:
    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.fixture: Fixture = load_fixture(config.metadata_path)
        self.specs = block_specs_for_fixture(
            self.fixture,
            start_block=config.start_block,
            block_count=config.block_count,
        )
        model_specs = block_specs_for_model(
            self.fixture,
            self.specs,
            aes_rounds=config.aes_rounds,
        )
        self.circuit: Circuit = build_xts_block_collection(
            model_specs,
            encoding=config.encoding,
            relaxation=config.relaxation,
            aes_rounds=config.aes_rounds,
        )
        self.random = random.Random(config.seed)
        self.engine: SamplerEngine | None = None

    def known_fixture_assignment(self) -> Assignment:
        assignment = assignment_from_fixture_plaintext(self.fixture, self.specs)
        if self.circuit.wire_offsets:
            self.circuit.derive_wire_values(assignment)
        return assignment

    def random_assignment(self) -> Assignment:
        assignment = Assignment.empty(
            self.config.block_count * 16,
            wire_bytes=len(self.circuit.wire_offsets),
        )
        assignment.plaintext[:] = bytes(self.random.choice(TEXT_ASCII) for _ in assignment.plaintext)
        if assignment.wires is not None:
            assignment.wires[:] = self.random.randbytes(len(assignment.wires))
        if self.config.fixed_key:
            if self.fixture.key1 is None or self.fixture.key2 is None:
                raise ValueError("fixed-key sampling requires key metadata")
            assignment.key1[:] = self.fixture.key1
            assignment.key2[:] = self.fixture.key2
        else:
            assignment.key1[:] = self.random.randbytes(len(assignment.key1))
            assignment.key2[:] = self.random.randbytes(len(assignment.key2))
        return assignment

    def score(self, assignment: Assignment) -> EvaluationResult:
        if self.circuit.wire_offsets and assignment.wires is None:
            self.circuit.derive_wire_values(assignment)
        engine_cls = get_backend(self.config.backend)
        engine = engine_cls(
            self.circuit,
            assignment,
            fixed_key=self.config.fixed_key,
            seed=self.config.seed,
            relaxation=self.config.relaxation,
        )
        return engine.evaluate()

    def run(self, iterations: int) -> SampleState:
        assignment = self.random_assignment()
        engine_cls = get_backend(self.config.backend)
        self.engine = engine_cls(
            self.circuit,
            assignment,
            fixed_key=self.config.fixed_key,
            seed=self.config.seed,
            relaxation=self.config.relaxation,
        )
        result = self.engine.run_epoch(iterations)
        return SampleState(self.engine.assignment, result)
