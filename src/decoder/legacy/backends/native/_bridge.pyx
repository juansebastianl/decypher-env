# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport int32_t, uint16_t, uint8_t, uint32_t, uint64_t
from libc.stddef cimport size_t
from cpython.bytearray cimport PyByteArray_AS_STRING


PROFILE_LABELS = (
    "run_epoch_total",
    "step_replicas_wall",
    "attempt_swaps",
    "dual_update",
    "step_replica_total",
    "mutate",
    "derive_wires",
    "score_delta",
    "score_full",
    "evaluate_values",
    "evaluate_values_derive",
    "residual_scan",
    "energy_accept",
    "restore_values",
    "harvest",
    "metrics",
    "drain_feasible",
    "current_assignment",
)

GRAPH_COUNT_LABELS = (
    "values",
    "ops",
    "constraints",
    "wire_bytes",
    "replicas",
    "threads",
)


cdef extern from "cpp_solver.hpp" namespace "aes_xts_decoder":
    cdef struct Int32View:
        const int32_t* data
        size_t size

    cdef struct UInt16View:
        const uint16_t* data
        size_t size

    cdef struct BytesView:
        uint8_t* data
        size_t size

    cdef struct CircuitView:
        Int32View op_rows
        Int32View constraint_rows
        UInt16View value_widths

    cdef struct AssignmentView:
        BytesView plaintext
        BytesView key1
        BytesView key2
        BytesView wires

    cdef struct Score:
        uint32_t violations
        uint32_t hamming_score
        const uint32_t* residuals
        size_t residual_count
        const uint32_t* failing_indices
        size_t failing_index_count

    Score score_assignment(const CircuitView& circuit, const AssignmentView& assignment)


cdef extern from "pt_engine.hpp" namespace "aes_xts_decoder":
    cdef cppclass PTConfig:
        size_t replicas
        double t_min
        double t_max
        size_t sweeps_per_epoch
        double mu
        double dual_eta
        double ascii_weight
        double consistency_weight
        double goal_weight
        uint64_t seed
        size_t threads
        bool profile
        int ladder_mode
        size_t ladder_adapt_interval_epochs
        size_t ladder_burn_in_epochs
        size_t ladder_min_round_trips
        int dual_mode
        double scheduled_rho_initial
        double scheduled_eta_tol
        double scheduled_eta_target
        double scheduled_rho_growth
        double scheduled_violation_shrink
        bool diagnostics
        bool algebra_diagnostics
        bool bp_diagnostics
        bool alternative_diagnostics
        size_t marginal_burn_in_epochs
        size_t marginal_window
        double marginal_alpha
        size_t marginal_min_distinct_keys
        double lambda_scale_cold
        double lambda_scale_hot
        int aes_rounds
        double repair_move_prob
        double key_gibbs_prob
        size_t bp_iterations
        double bp_damping
        double bp_tolerance
        double bp_proposal_weight
        double bethe_weight
        double algebraic_newton_prob
        size_t survey_restarts

    cdef cppclass AssignmentState:
        vector[uint8_t] plaintext
        vector[uint8_t] key1
        vector[uint8_t] key2
        vector[uint8_t] wires

    cdef cppclass ScoreData:
        uint32_t violations
        double hamming_score
        vector[uint32_t] residuals
        vector[uint32_t] failing_indices

    cdef cppclass PTDiagnostics:
        size_t epochs
        size_t sweeps
        size_t feasible_count
        double total_residual
        size_t violations
        uint32_t max_residual
        vector[size_t] swap_attempts
        vector[size_t] swap_accepts
        vector[size_t] proposal_attempts
        vector[size_t] proposal_accepts
        vector[double] residuals_by_class
        vector[double] multiplier_mean_by_class
        vector[double] multiplier_max_by_class
        vector[size_t] graph_counts
        vector[size_t] opcode_counts
        size_t key_visit_count
        size_t key_distinct_count
        vector[size_t] key_ones
        double key_marginal_max_deviation
        double key_information_bits
        double key_information_bits_raw
        double key_information_null_bits
        double marginal_ess
        double marginal_rhat
        bool marginal_trusted
        vector[double] temperatures
        vector[double] replica_lambda_scale
        vector[size_t] replica_round_trips
        vector[size_t] replica_up_counts
        vector[size_t] replica_down_counts
        size_t total_round_trips
        size_t last_ladder_adaptation_epoch
        vector[size_t] energy_sample_counts
        vector[double] energy_mean_by_rung
        vector[double] energy_variance_by_rung
        vector[double] energy_temperatures_by_rung
        double log_z_estimate
        double log_feasible_count_estimate
        bool log_z_estimate_available
        size_t log_z_state_bits
        vector[double] rho_by_class
        vector[size_t] lambda_update_counts_by_class
        vector[size_t] rho_escalation_counts_by_class
        bool infeasibility_suspected
        vector[size_t] algebra_counts
        vector[double] bp_key_marginals
        bool bp_converged
        bool algebra_exact
        bool bp_available
        bool alternative_available
        bool langevin_available
        vector[double] alternative_log_z_estimates
        double langevin_seed_score
        double bethe_free_energy
        double bp_residual
        double bp_entropy
        size_t survey_restarts
        double survey_entropy
        vector[double] profile_seconds
        vector[size_t] profile_counts

    cdef cppclass CppParallelTemperingEngine "aes_xts_decoder::ParallelTemperingEngine":
        CppParallelTemperingEngine(
            size_t value_count,
            vector[int32_t] opcodes,
            vector[int32_t] outputs,
            vector[int32_t] input_offsets,
            vector[int32_t] input_counts,
            vector[int32_t] inputs,
            vector[int32_t] immediate_offsets,
            vector[int32_t] immediate_counts,
            vector[int32_t] immediates,
            vector[int32_t] const_offsets,
            vector[int32_t] const_counts,
            vector[uint8_t] constants,
            vector[int32_t] constraint_kinds,
            vector[int32_t] constraint_left,
            vector[int32_t] constraint_right,
            vector[int32_t] wire_value_ids,
            vector[int32_t] wire_offsets,
            vector[uint16_t] value_widths,
            vector[int32_t] constraint_classes,
            size_t plaintext_start,
            size_t plaintext_count,
            size_t key1_start,
            size_t key1_count,
            size_t key2_start,
            size_t key2_count,
            vector[int32_t] xts_block_sectors,
            vector[int32_t] xts_block_indices,
            vector[uint8_t] xts_block_targets,
            AssignmentState initial,
            PTConfig config)
        ScoreData ScoreAssignment(AssignmentState assignment)
        ScoreData RunEpoch(size_t sweeps)
        vector[uint32_t] Residuals()
        PTDiagnostics Metrics()
        vector[AssignmentState] DrainFeasible(size_t limit)
        AssignmentState CurrentAssignment()
        AssignmentState DeriveAssignmentWires(AssignmentState assignment)
        void SetMultipliers(vector[double] multipliers)
        void SetTemperatures(vector[double] temperatures)
        void SetConstraintClasses(vector[int32_t] classes)


cdef extern from "continuous_relaxed_engine.hpp" namespace "aes_xts_decoder":
    cdef cppclass CppContinuousRelaxedEngine "aes_xts_decoder::ContinuousRelaxedEngine":
        CppContinuousRelaxedEngine(
            size_t value_count,
            vector[int32_t] opcodes,
            vector[int32_t] outputs,
            vector[int32_t] input_offsets,
            vector[int32_t] input_counts,
            vector[int32_t] inputs,
            vector[int32_t] immediate_offsets,
            vector[int32_t] immediate_counts,
            vector[int32_t] immediates,
            vector[int32_t] const_offsets,
            vector[int32_t] const_counts,
            vector[uint8_t] constants,
            vector[int32_t] constraint_kinds,
            vector[int32_t] constraint_left,
            vector[int32_t] constraint_right,
            vector[int32_t] wire_value_ids,
            vector[int32_t] wire_offsets,
            vector[uint16_t] value_widths,
            vector[int32_t] constraint_classes,
            size_t plaintext_start,
            size_t plaintext_count,
            size_t key1_start,
            size_t key1_count,
            size_t key2_start,
            size_t key2_count,
            vector[int32_t] xts_block_sectors,
            vector[int32_t] xts_block_indices,
            vector[uint8_t] xts_block_targets,
            AssignmentState initial,
            PTConfig config)
        ScoreData ScoreAssignment(AssignmentState assignment)
        ScoreData RunEpoch(size_t sweeps)
        vector[uint32_t] Residuals()
        PTDiagnostics Metrics()
        vector[AssignmentState] DrainFeasible(size_t limit)
        AssignmentState CurrentAssignment()
        AssignmentState DeriveAssignmentWires(AssignmentState assignment)
        void SetMultipliers(vector[double] multipliers)
        void SetTemperatures(vector[double] temperatures)
        void SetConstraintClasses(vector[int32_t] classes)


cdef extern from "algebraic_relaxed_engine.hpp" namespace "aes_xts_decoder":
    cdef cppclass CppAlgebraicRelaxedEngine "aes_xts_decoder::AlgebraicRelaxedEngine":
        CppAlgebraicRelaxedEngine(
            size_t value_count,
            vector[int32_t] opcodes,
            vector[int32_t] outputs,
            vector[int32_t] input_offsets,
            vector[int32_t] input_counts,
            vector[int32_t] inputs,
            vector[int32_t] immediate_offsets,
            vector[int32_t] immediate_counts,
            vector[int32_t] immediates,
            vector[int32_t] const_offsets,
            vector[int32_t] const_counts,
            vector[uint8_t] constants,
            vector[int32_t] constraint_kinds,
            vector[int32_t] constraint_left,
            vector[int32_t] constraint_right,
            vector[int32_t] wire_value_ids,
            vector[int32_t] wire_offsets,
            vector[uint16_t] value_widths,
            vector[int32_t] constraint_classes,
            size_t plaintext_start,
            size_t plaintext_count,
            size_t key1_start,
            size_t key1_count,
            size_t key2_start,
            size_t key2_count,
            vector[int32_t] xts_block_sectors,
            vector[int32_t] xts_block_indices,
            vector[uint8_t] xts_block_targets,
            AssignmentState initial,
            PTConfig config)
        ScoreData ScoreAssignment(AssignmentState assignment)
        ScoreData RunEpoch(size_t sweeps)
        vector[uint32_t] Residuals()
        PTDiagnostics Metrics()
        vector[AssignmentState] DrainFeasible(size_t limit)
        AssignmentState CurrentAssignment()
        AssignmentState DeriveAssignmentWires(AssignmentState assignment)
        void SetMultipliers(vector[double] multipliers)
        void SetTemperatures(vector[double] temperatures)
        void SetConstraintClasses(vector[int32_t] classes)


cdef class NativeEngine:
    cdef vector[int32_t] op_rows
    cdef vector[int32_t] constraint_rows
    cdef vector[uint16_t] widths

    def __cinit__(self, object circuit):
        cdef tuple row
        cdef int value
        cdef object circuit_value

        for row in circuit.op_table():
            for value in row:
                self.op_rows.push_back(<int32_t>value)

        for row in circuit.constraint_table():
            for value in row:
                self.constraint_rows.push_back(<int32_t>value)

        for circuit_value in circuit.values:
            self.widths.push_back(<uint16_t>circuit_value.width)

    def score(self, object assignment):
        cdef CircuitView circuit_view
        cdef AssignmentView assignment_view
        cdef Score score

        circuit_view.op_rows = Int32View(self.op_rows.data(), self.op_rows.size())
        circuit_view.constraint_rows = Int32View(self.constraint_rows.data(), self.constraint_rows.size())
        circuit_view.value_widths = UInt16View(self.widths.data(), self.widths.size())

        assignment_view.plaintext = BytesView(<uint8_t*>PyByteArray_AS_STRING(assignment.plaintext), len(assignment.plaintext))
        assignment_view.key1 = BytesView(<uint8_t*>PyByteArray_AS_STRING(assignment.key1), len(assignment.key1))
        assignment_view.key2 = BytesView(<uint8_t*>PyByteArray_AS_STRING(assignment.key2), len(assignment.key2))
        if assignment.wires is not None:
            assignment_view.wires = BytesView(<uint8_t*>PyByteArray_AS_STRING(assignment.wires), len(assignment.wires))
        else:
            assignment_view.wires = BytesView(NULL, 0)

        score = score_assignment(circuit_view, assignment_view)
        return {
            "violations": score.violations,
            "hamming_score": score.hamming_score,
            "residuals": [score.residuals[i] for i in range(score.residual_count)] if score.residuals != NULL else [],
            "failing_indices": [score.failing_indices[i] for i in range(score.failing_index_count)] if score.failing_indices != NULL else [],
        }


def flatten_op_rows(object circuit):
    return [value for row in circuit.op_table() for value in row]


def value_widths(object circuit):
    return [value.width for value in circuit.values]


cdef class ParallelTemperingEngine:
    cdef CppParallelTemperingEngine* engine
    cdef object assignment_type

    def __cinit__(self, object circuit, object assignment, object config=None, object constraint_classes=None):
        cdef vector[int32_t] opcodes
        cdef vector[int32_t] outputs
        cdef vector[int32_t] input_offsets
        cdef vector[int32_t] input_counts
        cdef vector[int32_t] inputs
        cdef vector[int32_t] immediate_offsets
        cdef vector[int32_t] immediate_counts
        cdef vector[int32_t] immediates
        cdef vector[int32_t] const_offsets
        cdef vector[int32_t] const_counts
        cdef vector[uint8_t] constants
        cdef vector[int32_t] constraint_kinds
        cdef vector[int32_t] constraint_left
        cdef vector[int32_t] constraint_right
        cdef vector[int32_t] wire_value_ids
        cdef vector[int32_t] wire_offsets
        cdef vector[uint16_t] value_widths
        cdef vector[int32_t] classes
        cdef vector[int32_t] xts_block_sectors
        cdef vector[int32_t] xts_block_indices
        cdef vector[uint8_t] xts_block_targets
        cdef object block_info
        cdef PTConfig cpp_config
        cdef AssignmentState initial
        cdef object op
        cdef object constraint
        cdef object value
        cdef int item
        cdef bytes const_bytes
        cdef tuple plaintext_range
        cdef tuple key1_range
        cdef tuple key2_range
        cdef bint profile_enabled
        cdef object ladder_mode
        cdef object dual_mode
        self.assignment_type = assignment.__class__

        for op in circuit.ops:
            opcodes.push_back(<int32_t>int(op.opcode))
            outputs.push_back(<int32_t>op.output)
            input_offsets.push_back(<int32_t>inputs.size())
            input_counts.push_back(<int32_t>len(op.inputs))
            for item in op.inputs:
                inputs.push_back(<int32_t>item)
            immediate_offsets.push_back(<int32_t>immediates.size())
            immediate_counts.push_back(<int32_t>len(op.immediates))
            for item in op.immediates:
                immediates.push_back(<int32_t>item)

        for const_bytes in circuit.constants:
            const_offsets.push_back(<int32_t>constants.size())
            const_counts.push_back(<int32_t>len(const_bytes))
            for value in const_bytes:
                constants.push_back(<uint8_t>value)

        for constraint in circuit.constraints:
            constraint_kinds.push_back(<int32_t>int(constraint.kind))
            constraint_left.push_back(<int32_t>constraint.left)
            constraint_right.push_back(<int32_t>(-1 if constraint.right is None else constraint.right))

        for value, item in circuit.wire_offsets.items():
            wire_value_ids.push_back(<int32_t>value)
            wire_offsets.push_back(<int32_t>item)

        for value in circuit.values:
            value_widths.push_back(<uint16_t>value.width)

        for item in constraint_classes or []:
            classes.push_back(<int32_t>item)

        for block_info in getattr(circuit, "xts_blocks", ()):
            xts_block_sectors.push_back(<int32_t>int(block_info.sector_number))
            xts_block_indices.push_back(<int32_t>int(block_info.block_index_in_sector))
            for value in block_info.ciphertext:
                xts_block_targets.push_back(<uint8_t>value)

        initial = _assignment_state(assignment)
        cpp_config.replicas = <size_t>int(getattr(config, "replicas", 16))
        cpp_config.t_min = <double>float(getattr(config, "t_min", 0.5))
        cpp_config.t_max = <double>float(getattr(config, "t_max", 20.0))
        cpp_config.sweeps_per_epoch = <size_t>int(getattr(config, "sweeps_per_epoch", 1000))
        cpp_config.mu = <double>float(getattr(config, "mu", 0.1))
        cpp_config.dual_eta = <double>float(getattr(config, "dual_eta", 0.01))
        cpp_config.ascii_weight = <double>float(getattr(config, "ascii_weight", 1.0))
        cpp_config.consistency_weight = <double>float(getattr(config, "consistency_weight", 1.0))
        cpp_config.goal_weight = <double>float(getattr(config, "goal_weight", 0.1))
        cpp_config.seed = <uint64_t>(0 if getattr(config, "seed", None) is None else int(getattr(config, "seed")))
        cpp_config.threads = <size_t>int(getattr(config, "threads", 0))
        profile_enabled = getattr(config, "profile", False)
        cpp_config.profile = <bool>profile_enabled
        ladder_mode = getattr(config, "ladder_mode", "geometric")
        cpp_config.ladder_mode = <int>(1 if ladder_mode == "feedback" else int(ladder_mode) if isinstance(ladder_mode, int) else 0)
        cpp_config.ladder_adapt_interval_epochs = <size_t>int(getattr(config, "ladder_adapt_interval_epochs", 5))
        cpp_config.ladder_burn_in_epochs = <size_t>int(getattr(config, "ladder_burn_in_epochs", 5))
        cpp_config.ladder_min_round_trips = <size_t>int(getattr(config, "ladder_min_round_trips", 1))
        dual_mode = getattr(config, "dual_mode", "fixed")
        cpp_config.dual_mode = <int>(1 if dual_mode == "scheduled" else int(dual_mode) if isinstance(dual_mode, int) else 0)
        cpp_config.scheduled_rho_initial = <double>float(getattr(config, "scheduled_rho_initial", 0.1))
        cpp_config.scheduled_eta_tol = <double>float(getattr(config, "scheduled_eta_tol", 0.5))
        cpp_config.scheduled_eta_target = <double>float(getattr(config, "scheduled_eta_target", 1e-3))
        cpp_config.scheduled_rho_growth = <double>float(getattr(config, "scheduled_rho_growth", 2.0))
        cpp_config.scheduled_violation_shrink = <double>float(getattr(config, "scheduled_violation_shrink", 0.75))
        cpp_config.diagnostics = <bool>getattr(config, "diagnostics", True)
        cpp_config.algebra_diagnostics = <bool>getattr(config, "algebra_diagnostics", False)
        cpp_config.bp_diagnostics = <bool>getattr(config, "bp_diagnostics", False)
        cpp_config.alternative_diagnostics = <bool>getattr(config, "alternative_diagnostics", False)
        cpp_config.marginal_burn_in_epochs = <size_t>int(getattr(config, "marginal_burn_in_epochs", 0))
        cpp_config.marginal_window = <size_t>int(getattr(config, "marginal_window", 4096))
        if cpp_config.marginal_window != 0 and cpp_config.marginal_window < 50:
            cpp_config.marginal_window = 50
        cpp_config.marginal_alpha = <double>float(getattr(config, "marginal_alpha", 0.5))
        cpp_config.marginal_min_distinct_keys = <size_t>int(getattr(config, "marginal_min_distinct_keys", 25))
        cpp_config.lambda_scale_cold = <double>float(getattr(config, "lambda_scale_cold", 1.0))
        cpp_config.lambda_scale_hot = <double>float(getattr(config, "lambda_scale_hot", 1.0))
        cpp_config.aes_rounds = <int>int(getattr(config, "aes_rounds", 14))
        cpp_config.repair_move_prob = <double>float(getattr(config, "repair_move_prob", 0.0))
        cpp_config.key_gibbs_prob = <double>float(getattr(config, "key_gibbs_prob", 0.0))
        cpp_config.bp_iterations = <size_t>int(getattr(config, "bp_iterations", 20))
        cpp_config.bp_damping = <double>float(getattr(config, "bp_damping", 0.35))
        cpp_config.bp_tolerance = <double>float(getattr(config, "bp_tolerance", 1e-4))
        cpp_config.bp_proposal_weight = <double>float(getattr(config, "bp_proposal_weight", 0.25))
        cpp_config.bethe_weight = <double>float(getattr(config, "bethe_weight", 0.0))
        cpp_config.algebraic_newton_prob = <double>float(getattr(config, "algebraic_newton_prob", 0.0))
        cpp_config.survey_restarts = <size_t>int(getattr(config, "survey_restarts", 0))

        plaintext_range = circuit.input_ranges["plaintext"]
        key1_range = circuit.input_ranges["key1"]
        key2_range = circuit.input_ranges["key2"]
        self.engine = new CppParallelTemperingEngine(
            <size_t>len(circuit.values),
            opcodes, outputs, input_offsets, input_counts, inputs,
            immediate_offsets, immediate_counts, immediates,
            const_offsets, const_counts, constants,
            constraint_kinds, constraint_left, constraint_right,
            wire_value_ids, wire_offsets, value_widths, classes,
            <size_t>plaintext_range[0], <size_t>plaintext_range[1],
            <size_t>key1_range[0], <size_t>key1_range[1],
            <size_t>key2_range[0], <size_t>key2_range[1],
            xts_block_sectors, xts_block_indices, xts_block_targets,
            initial, cpp_config)

    def __dealloc__(self):
        del self.engine

    def run_epoch(self, int sweeps):
        return _score_dict(self.engine.RunEpoch(<size_t>max(0, sweeps)))

    def residuals(self):
        cdef vector[uint32_t] residuals = self.engine.Residuals()
        return [residuals[i] for i in range(residuals.size())]

    def drain_feasible(self, limit=None):
        cdef vector[AssignmentState] samples
        cdef size_t actual_limit = 0 if limit is None else <size_t>int(limit)
        cdef size_t i
        samples = self.engine.DrainFeasible(actual_limit)
        return [_python_assignment(self.assignment_type, samples[i]) for i in range(samples.size())]

    def metrics(self):
        cdef PTDiagnostics metrics = self.engine.Metrics()
        cdef size_t profile_size = metrics.profile_seconds.size()
        cdef size_t profile_label_count = len(PROFILE_LABELS)
        cdef size_t graph_count_size = metrics.graph_counts.size()
        cdef size_t graph_label_count = len(GRAPH_COUNT_LABELS)
        if metrics.profile_counts.size() < profile_size:
            profile_size = metrics.profile_counts.size()
        if profile_label_count < profile_size:
            profile_size = profile_label_count
        if graph_label_count < graph_count_size:
            graph_count_size = graph_label_count
        return {
            "epochs": metrics.epochs,
            "sweeps": metrics.sweeps,
            "feasible_count": metrics.feasible_count,
            "total_residual": metrics.total_residual,
            "violations": metrics.violations,
            "max_residual": metrics.max_residual,
            "swap_attempts": [metrics.swap_attempts[i] for i in range(metrics.swap_attempts.size())],
            "swap_accepts": [metrics.swap_accepts[i] for i in range(metrics.swap_accepts.size())],
            "proposal_attempts": {
                "wire_flip": metrics.proposal_attempts[0] if metrics.proposal_attempts.size() > 0 else 0,
                "plaintext_ascii": metrics.proposal_attempts[1] if metrics.proposal_attempts.size() > 1 else 0,
                "key_bit_flip": metrics.proposal_attempts[2] if metrics.proposal_attempts.size() > 2 else 0,
                "key_word_swap": metrics.proposal_attempts[3] if metrics.proposal_attempts.size() > 3 else 0,
            },
            "proposal_accepts": {
                "wire_flip": metrics.proposal_accepts[0] if metrics.proposal_accepts.size() > 0 else 0,
                "plaintext_ascii": metrics.proposal_accepts[1] if metrics.proposal_accepts.size() > 1 else 0,
                "key_bit_flip": metrics.proposal_accepts[2] if metrics.proposal_accepts.size() > 2 else 0,
                "key_word_swap": metrics.proposal_accepts[3] if metrics.proposal_accepts.size() > 3 else 0,
            },
            "residuals_by_class": {
                "ascii": metrics.residuals_by_class[0] if metrics.residuals_by_class.size() > 0 else 0.0,
                "consistency": metrics.residuals_by_class[1] if metrics.residuals_by_class.size() > 1 else 0.0,
                "goal": metrics.residuals_by_class[2] if metrics.residuals_by_class.size() > 2 else 0.0,
            },
            "multiplier_mean_by_class": {
                "ascii": metrics.multiplier_mean_by_class[0] if metrics.multiplier_mean_by_class.size() > 0 else 0.0,
                "consistency": metrics.multiplier_mean_by_class[1] if metrics.multiplier_mean_by_class.size() > 1 else 0.0,
                "goal": metrics.multiplier_mean_by_class[2] if metrics.multiplier_mean_by_class.size() > 2 else 0.0,
            },
            "multiplier_max_by_class": {
                "ascii": metrics.multiplier_max_by_class[0] if metrics.multiplier_max_by_class.size() > 0 else 0.0,
                "consistency": metrics.multiplier_max_by_class[1] if metrics.multiplier_max_by_class.size() > 1 else 0.0,
                "goal": metrics.multiplier_max_by_class[2] if metrics.multiplier_max_by_class.size() > 2 else 0.0,
            },
            "native_summary": {
                "total_residual": metrics.total_residual,
                "violations": metrics.violations,
                "max_residual": metrics.max_residual,
                "graph_counts": {
                    GRAPH_COUNT_LABELS[i]: metrics.graph_counts[i]
                    for i in range(graph_count_size)
                },
                "opcode_counts": {
                    str(i): metrics.opcode_counts[i]
                    for i in range(metrics.opcode_counts.size())
                    if metrics.opcode_counts[i]
                },
            },
            "key_visit_count": metrics.key_visit_count,
            "key_distinct_count": metrics.key_distinct_count,
            "key_ones": [metrics.key_ones[i] for i in range(metrics.key_ones.size())],
            "key_marginal_max_deviation": metrics.key_marginal_max_deviation,
            "key_information_bits": metrics.key_information_bits,
            "key_information_bits_raw": metrics.key_information_bits_raw,
            "key_information_null_bits": metrics.key_information_null_bits,
            "marginal_ess": metrics.marginal_ess,
            "marginal_rhat": metrics.marginal_rhat,
            "marginal_trusted": True if metrics.marginal_trusted else False,
            "marginal_diagnostic_note": "ESS/R-hat are computed on cold full-key Hamming weight and gated by distinct cold keys",
            "temperatures": [metrics.temperatures[i] for i in range(metrics.temperatures.size())],
            "replica_lambda_scale": [metrics.replica_lambda_scale[i] for i in range(metrics.replica_lambda_scale.size())],
            "replica_round_trips": [metrics.replica_round_trips[i] for i in range(metrics.replica_round_trips.size())],
            "replica_up_counts": [metrics.replica_up_counts[i] for i in range(metrics.replica_up_counts.size())],
            "replica_down_counts": [metrics.replica_down_counts[i] for i in range(metrics.replica_down_counts.size())],
            "total_round_trips": metrics.total_round_trips,
            "last_ladder_adaptation_epoch": metrics.last_ladder_adaptation_epoch,
            "energy_sample_counts": [metrics.energy_sample_counts[i] for i in range(metrics.energy_sample_counts.size())],
            "energy_mean_by_rung": [metrics.energy_mean_by_rung[i] for i in range(metrics.energy_mean_by_rung.size())],
            "energy_variance_by_rung": [metrics.energy_variance_by_rung[i] for i in range(metrics.energy_variance_by_rung.size())],
            "energy_temperatures_by_rung": [metrics.energy_temperatures_by_rung[i] for i in range(metrics.energy_temperatures_by_rung.size())],
            "relative_log_z_estimate": metrics.log_z_estimate if metrics.log_z_estimate_available else None,
            "log_z_estimate": None,
            "log_feasible_count_estimate": None,
            "log_z_estimate_available": True if metrics.log_z_estimate_available else False,
            "log_z_state_bits": metrics.log_z_state_bits,
            "log_z_note": "relative-only thermodynamic diagnostic over plaintext+key bits; not an absolute feasible-count estimate",
            "rho_by_class": {
                "ascii": metrics.rho_by_class[0] if metrics.rho_by_class.size() > 0 else 0.0,
                "consistency": metrics.rho_by_class[1] if metrics.rho_by_class.size() > 1 else 0.0,
                "goal": metrics.rho_by_class[2] if metrics.rho_by_class.size() > 2 else 0.0,
            },
            "lambda_update_counts_by_class": {
                "ascii": metrics.lambda_update_counts_by_class[0] if metrics.lambda_update_counts_by_class.size() > 0 else 0,
                "consistency": metrics.lambda_update_counts_by_class[1] if metrics.lambda_update_counts_by_class.size() > 1 else 0,
                "goal": metrics.lambda_update_counts_by_class[2] if metrics.lambda_update_counts_by_class.size() > 2 else 0,
            },
            "rho_escalation_counts_by_class": {
                "ascii": metrics.rho_escalation_counts_by_class[0] if metrics.rho_escalation_counts_by_class.size() > 0 else 0,
                "consistency": metrics.rho_escalation_counts_by_class[1] if metrics.rho_escalation_counts_by_class.size() > 1 else 0,
                "goal": metrics.rho_escalation_counts_by_class[2] if metrics.rho_escalation_counts_by_class.size() > 2 else 0,
            },
            "infeasibility_suspected": True if metrics.infeasibility_suspected else False,
            "algebra_summary": {
                "variable_bits": metrics.algebra_counts[0] if metrics.algebra_counts.size() > 0 else 0,
                "linear_ops": metrics.algebra_counts[1] if metrics.algebra_counts.size() > 1 else 0,
                "nonlinear_ops": metrics.algebra_counts[2] if metrics.algebra_counts.size() > 2 else 0,
                "linear_constraints": metrics.algebra_counts[3] if metrics.algebra_counts.size() > 3 else 0,
                "rank_estimate": None,
                "free_bits_estimate": None,
                "gf2_elimination_implemented": True if metrics.algebra_exact else False,
                "factor_count": metrics.algebra_counts[4] if metrics.algebra_counts.size() > 4 else 0,
                "bp_requested": True if metrics.algebra_counts.size() > 5 and metrics.algebra_counts[5] else False,
                "algebra_requested": True if metrics.algebra_counts.size() > 6 and metrics.algebra_counts[6] else False,
            },
            "bp_available": True if metrics.bp_available else False,
            "bp_converged": True if metrics.bp_converged else False,
            "bp_key_marginals": [metrics.bp_key_marginals[i] for i in range(metrics.bp_key_marginals.size())],
            "alternative_available": True if metrics.alternative_available else False,
            "alternative_log_z_estimates": [metrics.alternative_log_z_estimates[i] for i in range(metrics.alternative_log_z_estimates.size())],
            "langevin_available": True if metrics.langevin_available else False,
            "langevin_seed_score": metrics.langevin_seed_score if metrics.langevin_available else None,
            "profile_seconds": {
                PROFILE_LABELS[i]: metrics.profile_seconds[i]
                for i in range(profile_size)
            },
            "profile_counts": {
                PROFILE_LABELS[i]: metrics.profile_counts[i]
                for i in range(profile_size)
            },
        }

    def residuals_by_class(self):
        return self.metrics()["residuals_by_class"]

    def score_assignment(self, object assignment):
        return _score_dict(self.engine.ScoreAssignment(_assignment_state(assignment)))

    def current_assignment(self):
        return _python_assignment(self.assignment_type, self.engine.CurrentAssignment())

    def derive_wires(self, object assignment):
        return _python_assignment(self.assignment_type, self.engine.DeriveAssignmentWires(_assignment_state(assignment)))

    def set_multipliers(self, object multipliers):
        cdef vector[double] values
        cdef object value
        for value in multipliers:
            values.push_back(<double>float(value))
        self.engine.SetMultipliers(values)

    def set_temperatures(self, object temperatures):
        cdef vector[double] values
        cdef object value
        for value in temperatures:
            values.push_back(<double>float(value))
        self.engine.SetTemperatures(values)

    def set_constraint_classes(self, object classes):
        cdef vector[int32_t] values
        cdef object value
        for value in classes:
            values.push_back(<int32_t>int(value))
        self.engine.SetConstraintClasses(values)


cdef class ContinuousRelaxedEngine:
    cdef CppContinuousRelaxedEngine* engine
    cdef object assignment_type

    def __cinit__(self, object circuit, object assignment, object config=None, object constraint_classes=None):
        cdef vector[int32_t] opcodes
        cdef vector[int32_t] outputs
        cdef vector[int32_t] input_offsets
        cdef vector[int32_t] input_counts
        cdef vector[int32_t] inputs
        cdef vector[int32_t] immediate_offsets
        cdef vector[int32_t] immediate_counts
        cdef vector[int32_t] immediates
        cdef vector[int32_t] const_offsets
        cdef vector[int32_t] const_counts
        cdef vector[uint8_t] constants
        cdef vector[int32_t] constraint_kinds
        cdef vector[int32_t] constraint_left
        cdef vector[int32_t] constraint_right
        cdef vector[int32_t] wire_value_ids
        cdef vector[int32_t] wire_offsets
        cdef vector[uint16_t] value_widths
        cdef vector[int32_t] classes
        cdef vector[int32_t] xts_block_sectors
        cdef vector[int32_t] xts_block_indices
        cdef vector[uint8_t] xts_block_targets
        cdef object block_info
        cdef PTConfig cpp_config
        cdef AssignmentState initial
        cdef object op
        cdef object constraint
        cdef object value
        cdef int item
        cdef bytes const_bytes
        cdef tuple plaintext_range
        cdef tuple key1_range
        cdef tuple key2_range
        cdef bint profile_enabled
        cdef object ladder_mode
        cdef object dual_mode
        self.assignment_type = assignment.__class__

        for op in circuit.ops:
            opcodes.push_back(<int32_t>int(op.opcode))
            outputs.push_back(<int32_t>op.output)
            input_offsets.push_back(<int32_t>inputs.size())
            input_counts.push_back(<int32_t>len(op.inputs))
            for item in op.inputs:
                inputs.push_back(<int32_t>item)
            immediate_offsets.push_back(<int32_t>immediates.size())
            immediate_counts.push_back(<int32_t>len(op.immediates))
            for item in op.immediates:
                immediates.push_back(<int32_t>item)

        for const_bytes in circuit.constants:
            const_offsets.push_back(<int32_t>constants.size())
            const_counts.push_back(<int32_t>len(const_bytes))
            for value in const_bytes:
                constants.push_back(<uint8_t>value)

        for constraint in circuit.constraints:
            constraint_kinds.push_back(<int32_t>int(constraint.kind))
            constraint_left.push_back(<int32_t>constraint.left)
            constraint_right.push_back(<int32_t>(-1 if constraint.right is None else constraint.right))

        for value, item in circuit.wire_offsets.items():
            wire_value_ids.push_back(<int32_t>value)
            wire_offsets.push_back(<int32_t>item)

        for value in circuit.values:
            value_widths.push_back(<uint16_t>value.width)

        for item in constraint_classes or []:
            classes.push_back(<int32_t>item)

        for block_info in getattr(circuit, "xts_blocks", ()):
            xts_block_sectors.push_back(<int32_t>int(block_info.sector_number))
            xts_block_indices.push_back(<int32_t>int(block_info.block_index_in_sector))
            for value in block_info.ciphertext:
                xts_block_targets.push_back(<uint8_t>value)

        initial = _assignment_state(assignment)
        cpp_config.replicas = <size_t>int(getattr(config, "replicas", 16))
        cpp_config.t_min = <double>float(getattr(config, "t_min", 0.5))
        cpp_config.t_max = <double>float(getattr(config, "t_max", 20.0))
        cpp_config.sweeps_per_epoch = <size_t>int(getattr(config, "sweeps_per_epoch", 1000))
        cpp_config.mu = <double>float(getattr(config, "mu", 0.1))
        cpp_config.dual_eta = <double>float(getattr(config, "dual_eta", 0.01))
        cpp_config.ascii_weight = <double>float(getattr(config, "ascii_weight", 1.0))
        cpp_config.consistency_weight = <double>float(getattr(config, "consistency_weight", 1.0))
        cpp_config.goal_weight = <double>float(getattr(config, "goal_weight", 0.1))
        cpp_config.seed = <uint64_t>(0 if getattr(config, "seed", None) is None else int(getattr(config, "seed")))
        cpp_config.threads = <size_t>int(getattr(config, "threads", 0))
        profile_enabled = getattr(config, "profile", False)
        cpp_config.profile = <bool>profile_enabled
        ladder_mode = getattr(config, "ladder_mode", "geometric")
        cpp_config.ladder_mode = <int>(1 if ladder_mode == "feedback" else int(ladder_mode) if isinstance(ladder_mode, int) else 0)
        cpp_config.ladder_adapt_interval_epochs = <size_t>int(getattr(config, "ladder_adapt_interval_epochs", 5))
        cpp_config.ladder_burn_in_epochs = <size_t>int(getattr(config, "ladder_burn_in_epochs", 5))
        cpp_config.ladder_min_round_trips = <size_t>int(getattr(config, "ladder_min_round_trips", 1))
        dual_mode = getattr(config, "dual_mode", "fixed")
        cpp_config.dual_mode = <int>(1 if dual_mode == "scheduled" else int(dual_mode) if isinstance(dual_mode, int) else 0)
        cpp_config.scheduled_rho_initial = <double>float(getattr(config, "scheduled_rho_initial", 0.1))
        cpp_config.scheduled_eta_tol = <double>float(getattr(config, "scheduled_eta_tol", 0.5))
        cpp_config.scheduled_eta_target = <double>float(getattr(config, "scheduled_eta_target", 1e-3))
        cpp_config.scheduled_rho_growth = <double>float(getattr(config, "scheduled_rho_growth", 2.0))
        cpp_config.scheduled_violation_shrink = <double>float(getattr(config, "scheduled_violation_shrink", 0.75))
        cpp_config.diagnostics = <bool>getattr(config, "diagnostics", True)
        cpp_config.algebra_diagnostics = <bool>getattr(config, "algebra_diagnostics", False)
        cpp_config.bp_diagnostics = <bool>getattr(config, "bp_diagnostics", False)
        cpp_config.alternative_diagnostics = <bool>getattr(config, "alternative_diagnostics", False)
        cpp_config.marginal_burn_in_epochs = <size_t>int(getattr(config, "marginal_burn_in_epochs", 0))
        cpp_config.marginal_window = <size_t>int(getattr(config, "marginal_window", 4096))
        if cpp_config.marginal_window != 0 and cpp_config.marginal_window < 50:
            cpp_config.marginal_window = 50
        cpp_config.marginal_alpha = <double>float(getattr(config, "marginal_alpha", 0.5))
        cpp_config.marginal_min_distinct_keys = <size_t>int(getattr(config, "marginal_min_distinct_keys", 25))
        cpp_config.lambda_scale_cold = <double>float(getattr(config, "lambda_scale_cold", 1.0))
        cpp_config.lambda_scale_hot = <double>float(getattr(config, "lambda_scale_hot", 1.0))
        cpp_config.aes_rounds = <int>int(getattr(config, "aes_rounds", 14))
        cpp_config.repair_move_prob = <double>float(getattr(config, "repair_move_prob", 0.0))
        cpp_config.key_gibbs_prob = <double>float(getattr(config, "key_gibbs_prob", 0.0))
        cpp_config.bp_iterations = <size_t>int(getattr(config, "bp_iterations", 20))
        cpp_config.bp_damping = <double>float(getattr(config, "bp_damping", 0.35))
        cpp_config.bp_tolerance = <double>float(getattr(config, "bp_tolerance", 1e-4))
        cpp_config.bp_proposal_weight = <double>float(getattr(config, "bp_proposal_weight", 0.25))
        cpp_config.bethe_weight = <double>float(getattr(config, "bethe_weight", 0.0))
        cpp_config.algebraic_newton_prob = <double>float(getattr(config, "algebraic_newton_prob", 0.0))
        cpp_config.survey_restarts = <size_t>int(getattr(config, "survey_restarts", 0))

        plaintext_range = circuit.input_ranges["plaintext"]
        key1_range = circuit.input_ranges["key1"]
        key2_range = circuit.input_ranges["key2"]
        self.engine = new CppContinuousRelaxedEngine(
            <size_t>len(circuit.values),
            opcodes, outputs, input_offsets, input_counts, inputs,
            immediate_offsets, immediate_counts, immediates,
            const_offsets, const_counts, constants,
            constraint_kinds, constraint_left, constraint_right,
            wire_value_ids, wire_offsets, value_widths, classes,
            <size_t>plaintext_range[0], <size_t>plaintext_range[1],
            <size_t>key1_range[0], <size_t>key1_range[1],
            <size_t>key2_range[0], <size_t>key2_range[1],
            xts_block_sectors, xts_block_indices, xts_block_targets,
            initial, cpp_config)

    def __dealloc__(self):
        del self.engine

    def run_epoch(self, int sweeps):
        return _score_dict(self.engine.RunEpoch(<size_t>max(0, sweeps)))

    def residuals(self):
        cdef vector[uint32_t] residuals = self.engine.Residuals()
        return [residuals[i] for i in range(residuals.size())]

    def drain_feasible(self, limit=None):
        cdef vector[AssignmentState] samples
        cdef size_t actual_limit = 0 if limit is None else <size_t>int(limit)
        cdef size_t i
        samples = self.engine.DrainFeasible(actual_limit)
        return [_python_assignment(self.assignment_type, samples[i]) for i in range(samples.size())]

    def metrics(self):
        return _metrics_dict(self.engine.Metrics())

    def residuals_by_class(self):
        return self.metrics()["residuals_by_class"]

    def score_assignment(self, object assignment):
        return _score_dict(self.engine.ScoreAssignment(_assignment_state(assignment)))

    def current_assignment(self):
        return _python_assignment(self.assignment_type, self.engine.CurrentAssignment())

    def derive_wires(self, object assignment):
        return _python_assignment(self.assignment_type, self.engine.DeriveAssignmentWires(_assignment_state(assignment)))

    def set_multipliers(self, object multipliers):
        cdef vector[double] values
        cdef object value
        for value in multipliers:
            values.push_back(<double>float(value))
        self.engine.SetMultipliers(values)

    def set_temperatures(self, object temperatures):
        cdef vector[double] values
        cdef object value
        for value in temperatures:
            values.push_back(<double>float(value))
        self.engine.SetTemperatures(values)

    def set_constraint_classes(self, object classes):
        cdef vector[int32_t] values
        cdef object value
        for value in classes:
            values.push_back(<int32_t>int(value))
        self.engine.SetConstraintClasses(values)


cdef class AlgebraicRelaxedEngine:
    cdef CppAlgebraicRelaxedEngine* engine
    cdef object assignment_type

    def __cinit__(self, object circuit, object assignment, object config=None, object constraint_classes=None):
        cdef vector[int32_t] opcodes
        cdef vector[int32_t] outputs
        cdef vector[int32_t] input_offsets
        cdef vector[int32_t] input_counts
        cdef vector[int32_t] inputs
        cdef vector[int32_t] immediate_offsets
        cdef vector[int32_t] immediate_counts
        cdef vector[int32_t] immediates
        cdef vector[int32_t] const_offsets
        cdef vector[int32_t] const_counts
        cdef vector[uint8_t] constants
        cdef vector[int32_t] constraint_kinds
        cdef vector[int32_t] constraint_left
        cdef vector[int32_t] constraint_right
        cdef vector[int32_t] wire_value_ids
        cdef vector[int32_t] wire_offsets
        cdef vector[uint16_t] value_widths
        cdef vector[int32_t] classes
        cdef vector[int32_t] xts_block_sectors
        cdef vector[int32_t] xts_block_indices
        cdef vector[uint8_t] xts_block_targets
        cdef object block_info
        cdef PTConfig cpp_config
        cdef AssignmentState initial
        cdef object op
        cdef object constraint
        cdef object value
        cdef int item
        cdef bytes const_bytes
        cdef tuple plaintext_range
        cdef tuple key1_range
        cdef tuple key2_range
        cdef bint profile_enabled
        cdef object ladder_mode
        cdef object dual_mode
        self.assignment_type = assignment.__class__

        for op in circuit.ops:
            opcodes.push_back(<int32_t>int(op.opcode))
            outputs.push_back(<int32_t>op.output)
            input_offsets.push_back(<int32_t>inputs.size())
            input_counts.push_back(<int32_t>len(op.inputs))
            for item in op.inputs:
                inputs.push_back(<int32_t>item)
            immediate_offsets.push_back(<int32_t>immediates.size())
            immediate_counts.push_back(<int32_t>len(op.immediates))
            for item in op.immediates:
                immediates.push_back(<int32_t>item)

        for const_bytes in circuit.constants:
            const_offsets.push_back(<int32_t>constants.size())
            const_counts.push_back(<int32_t>len(const_bytes))
            for value in const_bytes:
                constants.push_back(<uint8_t>value)

        for constraint in circuit.constraints:
            constraint_kinds.push_back(<int32_t>int(constraint.kind))
            constraint_left.push_back(<int32_t>constraint.left)
            constraint_right.push_back(<int32_t>(-1 if constraint.right is None else constraint.right))

        for value, item in circuit.wire_offsets.items():
            wire_value_ids.push_back(<int32_t>value)
            wire_offsets.push_back(<int32_t>item)

        for value in circuit.values:
            value_widths.push_back(<uint16_t>value.width)

        for item in constraint_classes or []:
            classes.push_back(<int32_t>item)

        for block_info in getattr(circuit, "xts_blocks", ()):
            xts_block_sectors.push_back(<int32_t>int(block_info.sector_number))
            xts_block_indices.push_back(<int32_t>int(block_info.block_index_in_sector))
            for value in block_info.ciphertext:
                xts_block_targets.push_back(<uint8_t>value)

        initial = _assignment_state(assignment)
        cpp_config.replicas = <size_t>int(getattr(config, "replicas", 16))
        cpp_config.t_min = <double>float(getattr(config, "t_min", 0.5))
        cpp_config.t_max = <double>float(getattr(config, "t_max", 20.0))
        cpp_config.sweeps_per_epoch = <size_t>int(getattr(config, "sweeps_per_epoch", 1000))
        cpp_config.mu = <double>float(getattr(config, "mu", 0.1))
        cpp_config.dual_eta = <double>float(getattr(config, "dual_eta", 0.01))
        cpp_config.ascii_weight = <double>float(getattr(config, "ascii_weight", 1.0))
        cpp_config.consistency_weight = <double>float(getattr(config, "consistency_weight", 1.0))
        cpp_config.goal_weight = <double>float(getattr(config, "goal_weight", 0.1))
        cpp_config.seed = <uint64_t>(0 if getattr(config, "seed", None) is None else int(getattr(config, "seed")))
        cpp_config.threads = <size_t>int(getattr(config, "threads", 0))
        profile_enabled = getattr(config, "profile", False)
        cpp_config.profile = <bool>profile_enabled
        ladder_mode = getattr(config, "ladder_mode", "geometric")
        cpp_config.ladder_mode = <int>(1 if ladder_mode == "feedback" else int(ladder_mode) if isinstance(ladder_mode, int) else 0)
        cpp_config.ladder_adapt_interval_epochs = <size_t>int(getattr(config, "ladder_adapt_interval_epochs", 5))
        cpp_config.ladder_burn_in_epochs = <size_t>int(getattr(config, "ladder_burn_in_epochs", 5))
        cpp_config.ladder_min_round_trips = <size_t>int(getattr(config, "ladder_min_round_trips", 1))
        dual_mode = getattr(config, "dual_mode", "fixed")
        cpp_config.dual_mode = <int>(1 if dual_mode == "scheduled" else int(dual_mode) if isinstance(dual_mode, int) else 0)
        cpp_config.scheduled_rho_initial = <double>float(getattr(config, "scheduled_rho_initial", 0.1))
        cpp_config.scheduled_eta_tol = <double>float(getattr(config, "scheduled_eta_tol", 0.5))
        cpp_config.scheduled_eta_target = <double>float(getattr(config, "scheduled_eta_target", 1e-3))
        cpp_config.scheduled_rho_growth = <double>float(getattr(config, "scheduled_rho_growth", 2.0))
        cpp_config.scheduled_violation_shrink = <double>float(getattr(config, "scheduled_violation_shrink", 0.75))
        cpp_config.diagnostics = <bool>getattr(config, "diagnostics", True)
        cpp_config.algebra_diagnostics = <bool>getattr(config, "algebra_diagnostics", False)
        cpp_config.bp_diagnostics = <bool>getattr(config, "bp_diagnostics", False)
        cpp_config.alternative_diagnostics = <bool>getattr(config, "alternative_diagnostics", False)
        cpp_config.marginal_burn_in_epochs = <size_t>int(getattr(config, "marginal_burn_in_epochs", 0))
        cpp_config.marginal_window = <size_t>int(getattr(config, "marginal_window", 4096))
        if cpp_config.marginal_window != 0 and cpp_config.marginal_window < 50:
            cpp_config.marginal_window = 50
        cpp_config.marginal_alpha = <double>float(getattr(config, "marginal_alpha", 0.5))
        cpp_config.marginal_min_distinct_keys = <size_t>int(getattr(config, "marginal_min_distinct_keys", 25))
        cpp_config.lambda_scale_cold = <double>float(getattr(config, "lambda_scale_cold", 1.0))
        cpp_config.lambda_scale_hot = <double>float(getattr(config, "lambda_scale_hot", 1.0))
        cpp_config.aes_rounds = <int>int(getattr(config, "aes_rounds", 14))
        cpp_config.repair_move_prob = <double>float(getattr(config, "repair_move_prob", 0.0))
        cpp_config.key_gibbs_prob = <double>float(getattr(config, "key_gibbs_prob", 0.0))
        cpp_config.bp_iterations = <size_t>int(getattr(config, "bp_iterations", 20))
        cpp_config.bp_damping = <double>float(getattr(config, "bp_damping", 0.35))
        cpp_config.bp_tolerance = <double>float(getattr(config, "bp_tolerance", 1e-4))
        cpp_config.bp_proposal_weight = <double>float(getattr(config, "bp_proposal_weight", 0.25))
        cpp_config.bethe_weight = <double>float(getattr(config, "bethe_weight", 0.0))
        cpp_config.algebraic_newton_prob = <double>float(getattr(config, "algebraic_newton_prob", 0.0))
        cpp_config.survey_restarts = <size_t>int(getattr(config, "survey_restarts", 0))

        plaintext_range = circuit.input_ranges["plaintext"]
        key1_range = circuit.input_ranges["key1"]
        key2_range = circuit.input_ranges["key2"]
        self.engine = new CppAlgebraicRelaxedEngine(
            <size_t>len(circuit.values),
            opcodes, outputs, input_offsets, input_counts, inputs,
            immediate_offsets, immediate_counts, immediates,
            const_offsets, const_counts, constants,
            constraint_kinds, constraint_left, constraint_right,
            wire_value_ids, wire_offsets, value_widths, classes,
            <size_t>plaintext_range[0], <size_t>plaintext_range[1],
            <size_t>key1_range[0], <size_t>key1_range[1],
            <size_t>key2_range[0], <size_t>key2_range[1],
            xts_block_sectors, xts_block_indices, xts_block_targets,
            initial, cpp_config)

    def __dealloc__(self):
        del self.engine

    def run_epoch(self, int sweeps):
        return _score_dict(self.engine.RunEpoch(<size_t>max(0, sweeps)))

    def residuals(self):
        cdef vector[uint32_t] residuals = self.engine.Residuals()
        return [residuals[i] for i in range(residuals.size())]

    def drain_feasible(self, limit=None):
        cdef vector[AssignmentState] samples
        cdef size_t actual_limit = 0 if limit is None else <size_t>int(limit)
        samples = self.engine.DrainFeasible(actual_limit)
        return [_python_assignment(self.assignment_type, samples[i]) for i in range(samples.size())]

    def metrics(self):
        return _metrics_dict(self.engine.Metrics())

    def residuals_by_class(self):
        return self.metrics()["residuals_by_class"]

    def score_assignment(self, object assignment):
        return _score_dict(self.engine.ScoreAssignment(_assignment_state(assignment)))

    def current_assignment(self):
        return _python_assignment(self.assignment_type, self.engine.CurrentAssignment())

    def derive_wires(self, object assignment):
        return _python_assignment(self.assignment_type, self.engine.DeriveAssignmentWires(_assignment_state(assignment)))

    def set_multipliers(self, object multipliers):
        cdef vector[double] values
        cdef object value
        for value in multipliers:
            values.push_back(<double>float(value))
        self.engine.SetMultipliers(values)

    def set_temperatures(self, object temperatures):
        cdef vector[double] values
        cdef object value
        for value in temperatures:
            values.push_back(<double>float(value))
        self.engine.SetTemperatures(values)

    def set_constraint_classes(self, object classes):
        cdef vector[int32_t] values
        cdef object value
        for value in classes:
            values.push_back(<int32_t>int(value))
        self.engine.SetConstraintClasses(values)


cdef AssignmentState _assignment_state(object assignment):
    cdef AssignmentState state
    cdef object value
    for value in assignment.plaintext:
        state.plaintext.push_back(<uint8_t>value)
    for value in assignment.key1:
        state.key1.push_back(<uint8_t>value)
    for value in assignment.key2:
        state.key2.push_back(<uint8_t>value)
    if assignment.wires is not None:
        for value in assignment.wires:
            state.wires.push_back(<uint8_t>value)
    return state


cdef object _python_assignment(object assignment_type, AssignmentState state):
    return assignment_type(
        bytearray([state.plaintext[i] for i in range(state.plaintext.size())]),
        bytearray([state.key1[i] for i in range(state.key1.size())]),
        bytearray([state.key2[i] for i in range(state.key2.size())]),
        bytearray([state.wires[i] for i in range(state.wires.size())]) if state.wires.size() else None,
    )


cdef object _score_dict(ScoreData score):
    return {
        "violations": score.violations,
        "hamming_score": score.hamming_score,
        "residuals": [score.residuals[i] for i in range(score.residuals.size())],
        "failing_indices": [score.failing_indices[i] for i in range(score.failing_indices.size())],
    }


cdef object _metrics_dict(PTDiagnostics metrics):
    cdef size_t profile_size = metrics.profile_seconds.size()
    cdef size_t profile_label_count = len(PROFILE_LABELS)
    cdef size_t graph_count_size = metrics.graph_counts.size()
    cdef size_t graph_label_count = len(GRAPH_COUNT_LABELS)
    if metrics.profile_counts.size() < profile_size:
        profile_size = metrics.profile_counts.size()
    if profile_label_count < profile_size:
        profile_size = profile_label_count
    if graph_label_count < graph_count_size:
        graph_count_size = graph_label_count
    return {
        "epochs": metrics.epochs,
        "sweeps": metrics.sweeps,
        "feasible_count": metrics.feasible_count,
        "total_residual": metrics.total_residual,
        "violations": metrics.violations,
        "max_residual": metrics.max_residual,
        "swap_attempts": [metrics.swap_attempts[i] for i in range(metrics.swap_attempts.size())],
        "swap_accepts": [metrics.swap_accepts[i] for i in range(metrics.swap_accepts.size())],
        "proposal_attempts": {
            "wire_flip": metrics.proposal_attempts[0] if metrics.proposal_attempts.size() > 0 else 0,
            "plaintext_bit_flip": metrics.proposal_attempts[1] if metrics.proposal_attempts.size() > 1 else 0,
            "key_bit_flip": metrics.proposal_attempts[2] if metrics.proposal_attempts.size() > 2 else 0,
            "sbox_input_bit_flip": metrics.proposal_attempts[3] if metrics.proposal_attempts.size() > 3 else 0,
            "key_schedule_chain": metrics.proposal_attempts[4] if metrics.proposal_attempts.size() > 4 else 0,
            "key_word_swap": metrics.proposal_attempts[5] if metrics.proposal_attempts.size() > 5 else 0,
            "key_tweak_profile_gibbs": metrics.proposal_attempts[6] if metrics.proposal_attempts.size() > 6 else 0,
            "bp_guided_key_byte": metrics.proposal_attempts[7] if metrics.proposal_attempts.size() > 7 else 0,
            "algebraic_sbox_repair": metrics.proposal_attempts[8] if metrics.proposal_attempts.size() > 8 else 0,
            "gf256_newton": metrics.proposal_attempts[9] if metrics.proposal_attempts.size() > 9 else 0,
        },
        "proposal_accepts": {
            "wire_flip": metrics.proposal_accepts[0] if metrics.proposal_accepts.size() > 0 else 0,
            "plaintext_bit_flip": metrics.proposal_accepts[1] if metrics.proposal_accepts.size() > 1 else 0,
            "key_bit_flip": metrics.proposal_accepts[2] if metrics.proposal_accepts.size() > 2 else 0,
            "sbox_input_bit_flip": metrics.proposal_accepts[3] if metrics.proposal_accepts.size() > 3 else 0,
            "key_schedule_chain": metrics.proposal_accepts[4] if metrics.proposal_accepts.size() > 4 else 0,
            "key_word_swap": metrics.proposal_accepts[5] if metrics.proposal_accepts.size() > 5 else 0,
            "key_tweak_profile_gibbs": metrics.proposal_accepts[6] if metrics.proposal_accepts.size() > 6 else 0,
            "bp_guided_key_byte": metrics.proposal_accepts[7] if metrics.proposal_accepts.size() > 7 else 0,
            "algebraic_sbox_repair": metrics.proposal_accepts[8] if metrics.proposal_accepts.size() > 8 else 0,
            "gf256_newton": metrics.proposal_accepts[9] if metrics.proposal_accepts.size() > 9 else 0,
        },
        "residuals_by_class": {
            "ascii": metrics.residuals_by_class[0] if metrics.residuals_by_class.size() > 0 else 0.0,
            "consistency": metrics.residuals_by_class[1] if metrics.residuals_by_class.size() > 1 else 0.0,
            "goal": metrics.residuals_by_class[2] if metrics.residuals_by_class.size() > 2 else 0.0,
        },
        "multiplier_mean_by_class": {
            "ascii": metrics.multiplier_mean_by_class[0] if metrics.multiplier_mean_by_class.size() > 0 else 0.0,
            "consistency": metrics.multiplier_mean_by_class[1] if metrics.multiplier_mean_by_class.size() > 1 else 0.0,
            "goal": metrics.multiplier_mean_by_class[2] if metrics.multiplier_mean_by_class.size() > 2 else 0.0,
        },
        "multiplier_max_by_class": {
            "ascii": metrics.multiplier_max_by_class[0] if metrics.multiplier_max_by_class.size() > 0 else 0.0,
            "consistency": metrics.multiplier_max_by_class[1] if metrics.multiplier_max_by_class.size() > 1 else 0.0,
            "goal": metrics.multiplier_max_by_class[2] if metrics.multiplier_max_by_class.size() > 2 else 0.0,
        },
        "native_summary": {
            "total_residual": metrics.total_residual,
            "violations": metrics.violations,
            "max_residual": metrics.max_residual,
            "graph_counts": {
                GRAPH_COUNT_LABELS[i]: metrics.graph_counts[i]
                for i in range(graph_count_size)
            },
            "opcode_counts": {
                str(i): metrics.opcode_counts[i]
                for i in range(metrics.opcode_counts.size())
                if metrics.opcode_counts[i]
            },
        },
        "key_visit_count": metrics.key_visit_count,
        "key_distinct_count": metrics.key_distinct_count,
        "key_ones": [metrics.key_ones[i] for i in range(metrics.key_ones.size())],
        "key_marginal_max_deviation": metrics.key_marginal_max_deviation,
        "key_information_bits": metrics.key_information_bits,
        "key_information_bits_raw": metrics.key_information_bits_raw,
        "key_information_null_bits": metrics.key_information_null_bits,
        "marginal_ess": metrics.marginal_ess,
        "marginal_rhat": metrics.marginal_rhat,
        "marginal_trusted": True if metrics.marginal_trusted else False,
        "marginal_diagnostic_note": "ESS/R-hat are computed on cold full-key Hamming weight and gated by distinct cold keys",
        "temperatures": [metrics.temperatures[i] for i in range(metrics.temperatures.size())],
        "replica_lambda_scale": [metrics.replica_lambda_scale[i] for i in range(metrics.replica_lambda_scale.size())],
        "replica_round_trips": [metrics.replica_round_trips[i] for i in range(metrics.replica_round_trips.size())],
        "replica_up_counts": [metrics.replica_up_counts[i] for i in range(metrics.replica_up_counts.size())],
        "replica_down_counts": [metrics.replica_down_counts[i] for i in range(metrics.replica_down_counts.size())],
        "total_round_trips": metrics.total_round_trips,
        "energy_sample_counts": [metrics.energy_sample_counts[i] for i in range(metrics.energy_sample_counts.size())],
        "energy_mean_by_rung": [metrics.energy_mean_by_rung[i] for i in range(metrics.energy_mean_by_rung.size())],
        "energy_variance_by_rung": [metrics.energy_variance_by_rung[i] for i in range(metrics.energy_variance_by_rung.size())],
        "energy_temperatures_by_rung": [metrics.energy_temperatures_by_rung[i] for i in range(metrics.energy_temperatures_by_rung.size())],
        "relative_log_z_estimate": metrics.log_z_estimate if metrics.log_z_estimate_available else None,
        "log_z_estimate": None,
        "log_feasible_count_estimate": None,
        "log_z_estimate_available": True if metrics.log_z_estimate_available else False,
        "log_z_state_bits": metrics.log_z_state_bits,
        "rho_by_class": {
            "ascii": metrics.rho_by_class[0] if metrics.rho_by_class.size() > 0 else 0.0,
            "consistency": metrics.rho_by_class[1] if metrics.rho_by_class.size() > 1 else 0.0,
            "goal": metrics.rho_by_class[2] if metrics.rho_by_class.size() > 2 else 0.0,
        },
        "lambda_update_counts_by_class": {
            "ascii": metrics.lambda_update_counts_by_class[0] if metrics.lambda_update_counts_by_class.size() > 0 else 0,
            "consistency": metrics.lambda_update_counts_by_class[1] if metrics.lambda_update_counts_by_class.size() > 1 else 0,
            "goal": metrics.lambda_update_counts_by_class[2] if metrics.lambda_update_counts_by_class.size() > 2 else 0,
        },
        "rho_escalation_counts_by_class": {
            "ascii": metrics.rho_escalation_counts_by_class[0] if metrics.rho_escalation_counts_by_class.size() > 0 else 0,
            "consistency": metrics.rho_escalation_counts_by_class[1] if metrics.rho_escalation_counts_by_class.size() > 1 else 0,
            "goal": metrics.rho_escalation_counts_by_class[2] if metrics.rho_escalation_counts_by_class.size() > 2 else 0,
        },
        "infeasibility_suspected": True if metrics.infeasibility_suspected else False,
        "algebra_summary": {
            "variable_bits": metrics.algebra_counts[0] if metrics.algebra_counts.size() > 0 else 0,
            "linear_ops": metrics.algebra_counts[1] if metrics.algebra_counts.size() > 1 else 0,
            "nonlinear_ops": metrics.algebra_counts[2] if metrics.algebra_counts.size() > 2 else 0,
            "linear_determined_wires": metrics.algebra_counts[3] if metrics.algebra_counts.size() > 3 else 0,
            "gf2_elimination_implemented": True if metrics.algebra_exact else False,
            "factor_count": metrics.algebra_counts[4] if metrics.algebra_counts.size() > 4 else 0,
            "bp_requested": True if metrics.algebra_counts.size() > 5 and metrics.algebra_counts[5] else False,
            "algebra_requested": True if metrics.algebra_counts.size() > 6 and metrics.algebra_counts[6] else False,
            "sbox_lift_count": metrics.algebra_counts[7] if metrics.algebra_counts.size() > 7 else 0,
            "gf256_newton_rank": metrics.algebra_counts[8] if metrics.algebra_counts.size() > 8 else 0,
            "survey_restarts": metrics.algebra_counts[9] if metrics.algebra_counts.size() > 9 else 0,
        },
        "bp_available": True if metrics.bp_available else False,
        "bp_converged": True if metrics.bp_converged else False,
        "bp_key_marginals": [metrics.bp_key_marginals[i] for i in range(metrics.bp_key_marginals.size())],
        "bp_residual": metrics.bp_residual,
        "bp_entropy": metrics.bp_entropy,
        "bethe_free_energy": metrics.bethe_free_energy,
        "survey_restarts": metrics.survey_restarts,
        "survey_entropy": metrics.survey_entropy,
        "alternative_available": True if metrics.alternative_available else False,
        "alternative_log_z_estimates": [metrics.alternative_log_z_estimates[i] for i in range(metrics.alternative_log_z_estimates.size())],
        "langevin_available": True if metrics.langevin_available else False,
        "langevin_seed_score": metrics.langevin_seed_score if metrics.langevin_available else None,
        "profile_seconds": {
            PROFILE_LABELS[i]: metrics.profile_seconds[i]
            for i in range(profile_size)
        },
        "profile_counts": {
            PROFILE_LABELS[i]: metrics.profile_counts[i]
            for i in range(profile_size)
        },
    }
