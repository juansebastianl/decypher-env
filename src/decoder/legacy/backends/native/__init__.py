"""Native sampler engine wrapper."""

from __future__ import annotations

from ....circuit import Assignment, Circuit, EvaluationResult, OpCode
from ..base import ParallelTemperingConfig, classify_constraints
from ..python import PythonSamplerEngine

try:
    from . import _bridge
except ImportError:  # pragma: no cover - depends on optional compiled extension
    _bridge = None


class NativeSamplerEngine(PythonSamplerEngine):
    """Native-capable engine wrapper with Python reference semantics.

    The compiled extension currently owns the data export and lifecycle hooks;
    Python remains the correctness reference until native incremental search is
    implemented behind the same engine API.
    """

    supports_native_residuals = False

    def __init__(
        self,
        circuit: Circuit,
        assignment: Assignment,
        *,
        fixed_key: bool = True,
        seed: int | None = None,
        relaxation="discrete",
    ) -> None:
        if _bridge is None:
            raise RuntimeError("native backend is not available")
        self.native_engine = _bridge.NativeEngine(circuit)
        super().__init__(circuit, assignment, fixed_key=fixed_key, seed=seed, relaxation=relaxation)

    @classmethod
    def is_available(cls) -> bool:
        return _bridge is not None

    def evaluate(self) -> EvaluationResult:
        # Touch the native engine so wiring/build errors surface, while Python
        # remains the reference evaluator until the C++ solver owns residuals.
        self.native_engine.score(self.assignment)
        result = self.circuit.evaluate(self.assignment)
        self._last_result = result
        if result.satisfied:
            self._feasible.append(self._copy_assignment(self.assignment))
        return result


class NativeParallelTemperingEngine:
    """C++ parallel-tempering engine for lowered shared-key circuits."""

    supports_native_residuals = True

    def __init__(
        self,
        circuit: Circuit,
        assignment: Assignment,
        *,
        fixed_key: bool = True,
        seed: int | None = None,
        relaxation="discrete",
        config: ParallelTemperingConfig | None = None,
    ) -> None:
        if _bridge is None or not hasattr(_bridge, "ParallelTemperingEngine"):
            raise RuntimeError("native PT backend is not available")
        if config is None:
            config = ParallelTemperingConfig(seed=seed)
        elif seed is not None and config.seed is None:
            config = ParallelTemperingConfig(**{**config.__dict__, "seed": seed})
        self.circuit = circuit
        self.assignment = assignment
        self.fixed_key = fixed_key
        self.config = config
        self.constraint_classes = classify_constraints(circuit)
        self._native_score_supported = all(
            op.opcode not in (OpCode.AES_XTS_BLOCK, OpCode.AES256_BLOCK)
            for op in circuit.ops
        )
        self.native_engine = _bridge.ParallelTemperingEngine(
            circuit,
            assignment,
            config,
            [int(value) for value in self.constraint_classes],
        )
        self._last_result: EvaluationResult | None = None

    @classmethod
    def is_available(cls) -> bool:
        return _bridge is not None and hasattr(_bridge, "ParallelTemperingEngine")

    def configure_temperatures(self, temperatures) -> None:  # noqa: ANN001
        self.native_engine.set_temperatures(list(temperatures))

    def set_constraint_classes(self, classes, scales=None) -> None:  # noqa: ANN001
        self.constraint_classes = tuple(classes)
        self.constraint_scales = scales
        self.native_engine.set_constraint_classes([int(value) for value in self.constraint_classes])

    def set_multipliers(self, multipliers: list[float] | tuple[float, ...]) -> None:
        self.multipliers = list(multipliers)
        self.native_engine.set_multipliers(self.multipliers)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature
        self.native_engine.set_temperatures([float(temperature)])

    def evaluate(self) -> EvaluationResult:
        if not self._native_score_supported:
            result = self.circuit.evaluate(self.assignment)
            self._last_result = result
            return result
        score = self.native_engine.score_assignment(self.assignment)
        result = self.circuit.evaluate(self.assignment)
        if tuple(score["residuals"]) != tuple(result.residuals):
            raise RuntimeError("native PT score disagrees with Python residuals")
        self._last_result = result
        return result

    def residuals(self) -> tuple[float, ...]:
        return tuple(self.native_engine.residuals())

    def residuals_by_class(self):
        return self.native_engine.residuals_by_class()

    def swap_stats(self):
        metrics = self.native_engine.metrics()
        return {
            "attempts": metrics["swap_attempts"],
            "accepts": metrics["swap_accepts"],
        }

    def run_epoch(self, sweeps: int):
        score = self.native_engine.run_epoch(sweeps)
        if self.config.validate_python_epoch:
            self.sync_assignment()
            self._last_result = self.circuit.evaluate(self.assignment)
            if tuple(score["residuals"]) != tuple(self._last_result.residuals):
                raise RuntimeError("native PT epoch score disagrees with Python residuals")
            return self._last_result
        return score

    def sync_assignment(self) -> Assignment:
        self.assignment = self.native_engine.current_assignment()
        return self.assignment

    def drain_feasible(self, limit=None) -> list[Assignment]:  # noqa: ANN001
        return list(self.native_engine.drain_feasible(limit))

    def metrics(self):
        return self.native_engine.metrics()


class NativeContinuousRelaxedEngine(NativeParallelTemperingEngine):
    """Continuous-relaxed native engine with gradient-biased ensemble moves."""

    supports_native_residuals = True

    def __init__(
        self,
        circuit: Circuit,
        assignment: Assignment,
        *,
        fixed_key: bool = True,
        seed: int | None = None,
        relaxation="continuous",
        config: ParallelTemperingConfig | None = None,
    ) -> None:
        if _bridge is None or not hasattr(_bridge, "ContinuousRelaxedEngine"):
            raise RuntimeError("native continuous-relaxed backend is not available")
        if config is None:
            config = ParallelTemperingConfig(seed=seed)
        elif seed is not None and config.seed is None:
            config = ParallelTemperingConfig(**{**config.__dict__, "seed": seed})
        self.circuit = circuit
        self.assignment = assignment
        self.fixed_key = fixed_key
        self.relaxation = relaxation
        self.config = config
        self.constraint_classes = classify_constraints(circuit)
        self._native_score_supported = all(
            op.opcode not in (OpCode.AES_XTS_BLOCK, OpCode.AES256_BLOCK)
            for op in circuit.ops
        )
        self.native_engine = _bridge.ContinuousRelaxedEngine(
            circuit,
            assignment,
            config,
            [int(value) for value in self.constraint_classes],
        )
        self._last_result: EvaluationResult | None = None

    @classmethod
    def is_available(cls) -> bool:
        return _bridge is not None and hasattr(_bridge, "ContinuousRelaxedEngine")

    def run_epoch(self, sweeps: int):
        score = self.native_engine.run_epoch(sweeps)
        if self.config.validate_python_epoch:
            self.sync_assignment()
            native = self.native_engine.score_assignment(self.assignment)
            self._last_result = self.circuit.evaluate(self.assignment)
            if tuple(native["residuals"]) != tuple(self._last_result.residuals):
                raise RuntimeError("native continuous-relaxed score disagrees with Python residuals")
            return self._last_result
        return score


class NativeAlgebraicRelaxedEngine(NativeContinuousRelaxedEngine):
    """Algebraic/BP-guided native engine built on the continuous-relaxed runtime."""

    supports_native_residuals = True

    def __init__(
        self,
        circuit: Circuit,
        assignment: Assignment,
        *,
        fixed_key: bool = True,
        seed: int | None = None,
        relaxation="continuous",
        config: ParallelTemperingConfig | None = None,
    ) -> None:
        if _bridge is None or not hasattr(_bridge, "AlgebraicRelaxedEngine"):
            raise RuntimeError("native algebraic-relaxed backend is not available")
        if config is None:
            config = ParallelTemperingConfig(seed=seed)
        elif seed is not None and config.seed is None:
            config = ParallelTemperingConfig(**{**config.__dict__, "seed": seed})
        self.circuit = circuit
        self.assignment = assignment
        self.fixed_key = fixed_key
        self.relaxation = relaxation
        self.config = config
        self.constraint_classes = classify_constraints(circuit)
        self._native_score_supported = all(
            op.opcode not in (OpCode.AES_XTS_BLOCK, OpCode.AES256_BLOCK)
            for op in circuit.ops
        )
        self.native_engine = _bridge.AlgebraicRelaxedEngine(
            circuit,
            assignment,
            config,
            [int(value) for value in self.constraint_classes],
        )
        self._last_result: EvaluationResult | None = None

    @classmethod
    def is_available(cls) -> bool:
        return _bridge is not None and hasattr(_bridge, "AlgebraicRelaxedEngine")

    def run_epoch(self, sweeps: int):
        score = self.native_engine.run_epoch(sweeps)
        if self.config.validate_python_epoch:
            self.sync_assignment()
            native = self.native_engine.score_assignment(self.assignment)
            self._last_result = self.circuit.evaluate(self.assignment)
            if tuple(native["residuals"]) != tuple(self._last_result.residuals):
                raise RuntimeError("native algebraic-relaxed score disagrees with Python residuals")
            return self._last_result
        return score
