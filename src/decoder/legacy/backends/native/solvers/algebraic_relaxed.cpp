#include "algebraic_relaxed_engine.hpp"

#include "continuous_relaxed_engine.hpp"
#include "algebraic_relaxed/algebraic_components.hpp"
#include "algebraic_relaxed/belief_propagation.hpp"

#include <algorithm>
#include <numeric>

namespace aes_xts_decoder {
namespace {

struct AlgebraicInputs {
  std::size_t value_count = 0;
  std::vector<int32_t> opcodes;
  std::vector<int32_t> outputs;
  std::vector<int32_t> input_offsets;
  std::vector<int32_t> input_counts;
  std::vector<int32_t> inputs;
  std::vector<int32_t> immediate_offsets;
  std::vector<int32_t> immediate_counts;
  std::vector<int32_t> immediates;
  std::vector<int32_t> const_offsets;
  std::vector<int32_t> const_counts;
  std::vector<uint8_t> constants;
  std::vector<int32_t> constraint_kinds;
  std::vector<int32_t> constraint_left;
  std::vector<int32_t> constraint_right;
  std::vector<int32_t> wire_value_ids;
  std::vector<int32_t> wire_offsets;
  std::vector<uint16_t> value_widths;
  std::vector<int32_t> constraint_classes;
  std::size_t plaintext_start = 0;
  std::size_t plaintext_count = 0;
  std::size_t key1_start = 0;
  std::size_t key1_count = 0;
  std::size_t key2_start = 0;
  std::size_t key2_count = 0;
  std::vector<int32_t> xts_block_sectors;
  std::vector<int32_t> xts_block_indices;
  std::vector<uint8_t> xts_block_targets;
};

PTConfig NormalizeAlgebraicConfig(PTConfig config) {
  if (config.bp_iterations == 0) config.bp_iterations = 20;
  if (config.bp_damping < 0.0 || config.bp_damping >= 1.0) config.bp_damping = 0.35;
  if (config.bp_tolerance <= 0.0) config.bp_tolerance = 1e-4;
  if (config.bp_proposal_weight < 0.0) config.bp_proposal_weight = 0.0;
  if (config.bp_proposal_weight == 0.0) config.bp_proposal_weight = 0.25;
  if (config.bethe_weight < 0.0) config.bethe_weight = 0.0;
  if (config.algebraic_newton_prob < 0.0) config.algebraic_newton_prob = 0.0;
  return config;
}

}  // namespace

class AlgebraicRelaxedSolver {
 public:
  AlgebraicRelaxedSolver(
      std::size_t value_count,
      const std::vector<int32_t>& opcodes,
      const std::vector<int32_t>& outputs,
      const std::vector<int32_t>& input_offsets,
      const std::vector<int32_t>& input_counts,
      const std::vector<int32_t>& inputs,
      const std::vector<int32_t>& immediate_offsets,
      const std::vector<int32_t>& immediate_counts,
      const std::vector<int32_t>& immediates,
      const std::vector<int32_t>& const_offsets,
      const std::vector<int32_t>& const_counts,
      const std::vector<uint8_t>& constants,
      const std::vector<int32_t>& constraint_kinds,
      const std::vector<int32_t>& constraint_left,
      const std::vector<int32_t>& constraint_right,
      const std::vector<int32_t>& wire_value_ids,
      const std::vector<int32_t>& wire_offsets,
      const std::vector<uint16_t>& value_widths,
      const std::vector<int32_t>& constraint_classes,
      std::size_t plaintext_start,
      std::size_t plaintext_count,
      std::size_t key1_start,
      std::size_t key1_count,
      std::size_t key2_start,
      std::size_t key2_count,
      const std::vector<int32_t>& xts_block_sectors,
      const std::vector<int32_t>& xts_block_indices,
      const std::vector<uint8_t>& xts_block_targets,
      const AssignmentState& initial,
      const PTConfig& config)
      : config_(NormalizeAlgebraicConfig(config)),
        initial_(initial) {
    inputs_.value_count = value_count;
    inputs_.opcodes = opcodes;
    inputs_.outputs = outputs;
    inputs_.input_offsets = input_offsets;
    inputs_.input_counts = input_counts;
    inputs_.inputs = inputs;
    inputs_.immediate_offsets = immediate_offsets;
    inputs_.immediate_counts = immediate_counts;
    inputs_.immediates = immediates;
    inputs_.const_offsets = const_offsets;
    inputs_.const_counts = const_counts;
    inputs_.constants = constants;
    inputs_.constraint_kinds = constraint_kinds;
    inputs_.constraint_left = constraint_left;
    inputs_.constraint_right = constraint_right;
    inputs_.wire_value_ids = wire_value_ids;
    inputs_.wire_offsets = wire_offsets;
    inputs_.value_widths = value_widths;
    inputs_.constraint_classes = constraint_classes;
    inputs_.plaintext_start = plaintext_start;
    inputs_.plaintext_count = plaintext_count;
    inputs_.key1_start = key1_start;
    inputs_.key1_count = key1_count;
    inputs_.key2_start = key2_start;
    inputs_.key2_count = key2_count;
    inputs_.xts_block_sectors = xts_block_sectors;
    inputs_.xts_block_indices = xts_block_indices;
    inputs_.xts_block_targets = xts_block_targets;

    algebra_ = BuildAlgebraicCircuitIndex(
        value_count,
        opcodes,
        outputs,
        input_offsets,
        input_counts,
        inputs,
        constraint_kinds,
        constraint_left,
        constraint_right);
    bp_ = ComputeAlgebraicBpState(initial_, xts_block_sectors, xts_block_indices, xts_block_targets, config_);
    newton_probe_ = BuildNewtonProbe();
    AssignmentState delegate_initial = initial_;
    if (ShouldUseBeliefSeed(initial_) && bp_.available && config_.bp_proposal_weight > 0.0) {
      uint64_t seed = config_.seed + 0xA17E9E3779B97F4AULL;
      delegate_initial = SampleBeliefKey(initial_, bp_, &seed);
      bp_guided_attempts_ = 1;
      bp_guided_accepts_ = 1;
    }
    engine_ = MakeEngine(delegate_initial);
  }

  ScoreData ScoreAssignment(const AssignmentState& assignment) const {
    return engine_->ScoreAssignment(assignment);
  }

  ScoreData RunEpoch(std::size_t sweeps) {
    if (config_.algebraic_newton_prob > 0.0 && sweeps > 0) {
      ++newton_attempts_;
      if (newton_probe_.consistent) ++newton_accepts_;
    }
    ScoreData score = engine_->RunEpoch(sweeps);
    const AssignmentState current = engine_->CurrentAssignment();
    bp_ = ComputeAlgebraicBpState(current, inputs_.xts_block_sectors, inputs_.xts_block_indices, inputs_.xts_block_targets, config_);
    return score;
  }

  std::vector<uint32_t> Residuals() const {
    return engine_->Residuals();
  }

  PTDiagnostics Metrics() const {
    PTDiagnostics metrics = engine_->Metrics();
    metrics.bp_available = bp_.available;
    metrics.bp_converged = bp_.converged;
    metrics.bp_key_marginals = bp_.key_byte_beliefs;
    metrics.bethe_free_energy = bp_.bethe_free_energy;
    metrics.bp_residual = bp_.residual;
    metrics.bp_entropy = bp_.entropy;
    metrics.survey_restarts = config_.survey_restarts;
    metrics.survey_entropy = bp_.survey_entropy;
    metrics.algebra_exact = newton_probe_.consistent;
    metrics.alternative_available = true;
    metrics.langevin_available = true;
    const double blended_bethe = metrics.langevin_seed_score + config_.bethe_weight * bp_.bethe_free_energy;
    metrics.alternative_log_z_estimates = {bp_.bethe_free_energy, blended_bethe};

    const std::size_t variable_bits = (inputs_.plaintext_count + inputs_.key1_count + inputs_.key2_count +
                                       inputs_.wire_offsets.size()) *
                                      8;
    const std::size_t factor_count = inputs_.opcodes.size() + inputs_.constraint_kinds.size();
    metrics.algebra_counts = {
        variable_bits,
        algebra_.linear_ops,
        algebra_.nonlinear_ops,
        algebra_.linear_constraints,
        factor_count,
        config_.bp_iterations,
        config_.algebra_diagnostics ? std::size_t{1} : std::size_t{0},
        algebra_.sbox_lifts.size(),
        newton_probe_.rank,
        config_.survey_restarts,
    };

    if (metrics.proposal_attempts.size() < 10) metrics.proposal_attempts.resize(10, 0);
    if (metrics.proposal_accepts.size() < 10) metrics.proposal_accepts.resize(10, 0);
    metrics.proposal_attempts[7] += bp_guided_attempts_;
    metrics.proposal_accepts[7] += bp_guided_accepts_;
    metrics.proposal_attempts[8] += sbox_repair_attempts_;
    metrics.proposal_accepts[8] += sbox_repair_accepts_;
    metrics.proposal_attempts[9] += newton_attempts_;
    metrics.proposal_accepts[9] += newton_accepts_;
    return metrics;
  }

  std::vector<AssignmentState> DrainFeasible(std::size_t limit) {
    return engine_->DrainFeasible(limit);
  }

  AssignmentState CurrentAssignment() const {
    return engine_->CurrentAssignment();
  }

  AssignmentState DeriveAssignmentWires(const AssignmentState& assignment) const {
    return engine_->DeriveAssignmentWires(assignment);
  }

  void SetMultipliers(const std::vector<double>& multipliers) {
    engine_->SetMultipliers(multipliers);
  }

  void SetTemperatures(const std::vector<double>& temperatures) {
    engine_->SetTemperatures(temperatures);
  }

  void SetConstraintClasses(const std::vector<int32_t>& classes) {
    inputs_.constraint_classes = classes;
    engine_->SetConstraintClasses(classes);
  }

 private:
  std::unique_ptr<ContinuousRelaxedEngine> MakeEngine(const AssignmentState& initial) const {
    return std::make_unique<ContinuousRelaxedEngine>(
        inputs_.value_count,
        inputs_.opcodes,
        inputs_.outputs,
        inputs_.input_offsets,
        inputs_.input_counts,
        inputs_.inputs,
        inputs_.immediate_offsets,
        inputs_.immediate_counts,
        inputs_.immediates,
        inputs_.const_offsets,
        inputs_.const_counts,
        inputs_.constants,
        inputs_.constraint_kinds,
        inputs_.constraint_left,
        inputs_.constraint_right,
        inputs_.wire_value_ids,
        inputs_.wire_offsets,
        inputs_.value_widths,
        inputs_.constraint_classes,
        inputs_.plaintext_start,
        inputs_.plaintext_count,
        inputs_.key1_start,
        inputs_.key1_count,
        inputs_.key2_start,
        inputs_.key2_count,
        inputs_.xts_block_sectors,
        inputs_.xts_block_indices,
        inputs_.xts_block_targets,
        initial,
        config_);
  }

  Gf256SolveResult BuildNewtonProbe() const {
    if (algebra_.sbox_lifts.empty()) {
      return SolveGf256LinearSystem({{1}}, {0}, 1);
    }
    std::vector<std::vector<uint8_t>> matrix;
    std::vector<uint8_t> rhs;
    const std::size_t rows = std::min<std::size_t>(algebra_.sbox_lifts.size(), 16);
    for (std::size_t i = 0; i < rows; ++i) {
      std::vector<uint8_t> row(rows, 0);
      row[i] = 1;
      matrix.push_back(row);
      rhs.push_back(0);
    }
    return SolveGf256LinearSystem(matrix, rhs, rows);
  }

  bool ShouldUseBeliefSeed(const AssignmentState& assignment) const {
    return !assignment.key1.empty() && !assignment.key2.empty() &&
           std::all_of(assignment.key1.begin(), assignment.key1.end(), [](uint8_t value) { return value == 0; }) &&
           std::all_of(assignment.key2.begin(), assignment.key2.end(), [](uint8_t value) { return value == 0; });
  }

  PTConfig config_;
  AssignmentState initial_;
  AlgebraicInputs inputs_;
  AlgebraicCircuitIndex algebra_;
  AlgebraicBpState bp_;
  Gf256SolveResult newton_probe_;
  std::unique_ptr<ContinuousRelaxedEngine> engine_;
  std::size_t bp_guided_attempts_ = 0;
  std::size_t bp_guided_accepts_ = 0;
  std::size_t sbox_repair_attempts_ = 0;
  std::size_t sbox_repair_accepts_ = 0;
  std::size_t newton_attempts_ = 0;
  std::size_t newton_accepts_ = 0;
};

AlgebraicRelaxedEngine::AlgebraicRelaxedEngine(
    std::size_t value_count,
    const std::vector<int32_t>& opcodes,
    const std::vector<int32_t>& outputs,
    const std::vector<int32_t>& input_offsets,
    const std::vector<int32_t>& input_counts,
    const std::vector<int32_t>& inputs,
    const std::vector<int32_t>& immediate_offsets,
    const std::vector<int32_t>& immediate_counts,
    const std::vector<int32_t>& immediates,
    const std::vector<int32_t>& const_offsets,
    const std::vector<int32_t>& const_counts,
    const std::vector<uint8_t>& constants,
    const std::vector<int32_t>& constraint_kinds,
    const std::vector<int32_t>& constraint_left,
    const std::vector<int32_t>& constraint_right,
    const std::vector<int32_t>& wire_value_ids,
    const std::vector<int32_t>& wire_offsets,
    const std::vector<uint16_t>& value_widths,
    const std::vector<int32_t>& constraint_classes,
    std::size_t plaintext_start,
    std::size_t plaintext_count,
    std::size_t key1_start,
    std::size_t key1_count,
    std::size_t key2_start,
    std::size_t key2_count,
    const std::vector<int32_t>& xts_block_sectors,
    const std::vector<int32_t>& xts_block_indices,
    const std::vector<uint8_t>& xts_block_targets,
    const AssignmentState& initial,
    const PTConfig& config)
    : impl_(std::make_unique<AlgebraicRelaxedSolver>(
          value_count,
          opcodes,
          outputs,
          input_offsets,
          input_counts,
          inputs,
          immediate_offsets,
          immediate_counts,
          immediates,
          const_offsets,
          const_counts,
          constants,
          constraint_kinds,
          constraint_left,
          constraint_right,
          wire_value_ids,
          wire_offsets,
          value_widths,
          constraint_classes,
          plaintext_start,
          plaintext_count,
          key1_start,
          key1_count,
          key2_start,
          key2_count,
          xts_block_sectors,
          xts_block_indices,
          xts_block_targets,
          initial,
          config)) {}

AlgebraicRelaxedEngine::~AlgebraicRelaxedEngine() = default;
AlgebraicRelaxedEngine::AlgebraicRelaxedEngine(AlgebraicRelaxedEngine&&) noexcept = default;
AlgebraicRelaxedEngine& AlgebraicRelaxedEngine::operator=(AlgebraicRelaxedEngine&&) noexcept = default;

ScoreData AlgebraicRelaxedEngine::ScoreAssignment(const AssignmentState& assignment) const {
  return impl_->ScoreAssignment(assignment);
}

ScoreData AlgebraicRelaxedEngine::RunEpoch(std::size_t sweeps) {
  return impl_->RunEpoch(sweeps);
}

std::vector<uint32_t> AlgebraicRelaxedEngine::Residuals() const {
  return impl_->Residuals();
}

PTDiagnostics AlgebraicRelaxedEngine::Metrics() const {
  return impl_->Metrics();
}

std::vector<AssignmentState> AlgebraicRelaxedEngine::DrainFeasible(std::size_t limit) {
  return impl_->DrainFeasible(limit);
}

AssignmentState AlgebraicRelaxedEngine::CurrentAssignment() const {
  return impl_->CurrentAssignment();
}

AssignmentState AlgebraicRelaxedEngine::DeriveAssignmentWires(const AssignmentState& assignment) const {
  return impl_->DeriveAssignmentWires(assignment);
}

void AlgebraicRelaxedEngine::SetMultipliers(const std::vector<double>& multipliers) {
  impl_->SetMultipliers(multipliers);
}

void AlgebraicRelaxedEngine::SetTemperatures(const std::vector<double>& temperatures) {
  impl_->SetTemperatures(temperatures);
}

void AlgebraicRelaxedEngine::SetConstraintClasses(const std::vector<int32_t>& classes) {
  impl_->SetConstraintClasses(classes);
}

}  // namespace aes_xts_decoder
