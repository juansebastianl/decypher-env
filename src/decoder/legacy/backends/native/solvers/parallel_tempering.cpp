#include "parallel_tempering/solver.hpp"

#include <memory>
#include <utility>

namespace aes_xts_decoder {

// Parallel tempering native engine layout
// ---------------------------------------
// `ParallelTemperingEngine` is the Cython-facing facade. The implementation is
// owned by `ParallelTemperingSolver` and split into focused modules under
// `solvers/parallel_tempering/`:
//
//   solver.hpp       - internal solver state and orchestration surface
//   circuit_model.*  - immutable circuit buffers and graph dependency indexes
//   byte_evaluator.* - circuit value evaluation and wire derivation
//   hamming_scorer.* - residual scoring over evaluated values
//   constructor.cpp  - model setup, replica initialization, profiling hooks
//   evaluation.cpp   - profiled evaluator/scorer wrappers and incremental updates
//   api.cpp          - public solver API and metrics marshalling
//   proposals.*      - wire/plaintext/key mutations and XTS repair proposals
//   replica_set.hpp  - per-replica bookkeeping and swap ownership
//   moves.cpp        - replica step orchestration and swaps
//   schedule.*       - dual updates and feedback ladder adaptation
//   optimization.cpp - energy statistics, key stats, scheduling delegates
//   diagnostics.cpp  - key marginal and log-Z diagnostics
//   energy.cpp       - augmented-Lagrangian energy and initial weights
//   helpers.cpp      - AES/XTS helpers, RNG helpers, ASCII/popcount utilities

ParallelTemperingEngine::ParallelTemperingEngine(
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
    : impl_(std::make_unique<ParallelTemperingSolver>(
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

ParallelTemperingEngine::~ParallelTemperingEngine() = default;
ParallelTemperingEngine::ParallelTemperingEngine(ParallelTemperingEngine&&) noexcept = default;
ParallelTemperingEngine& ParallelTemperingEngine::operator=(ParallelTemperingEngine&&) noexcept = default;

ScoreData ParallelTemperingEngine::ScoreAssignment(const AssignmentState& assignment) const {
  return impl_->ScoreAssignment(assignment);
}

ScoreData ParallelTemperingEngine::RunEpoch(std::size_t sweeps) {
  return impl_->RunEpoch(sweeps);
}

std::vector<uint32_t> ParallelTemperingEngine::Residuals() const {
  return impl_->Residuals();
}

PTDiagnostics ParallelTemperingEngine::Metrics() const {
  return impl_->Metrics();
}

std::vector<AssignmentState> ParallelTemperingEngine::DrainFeasible(std::size_t limit) {
  return impl_->DrainFeasible(limit);
}

AssignmentState ParallelTemperingEngine::CurrentAssignment() const {
  return impl_->CurrentAssignment();
}

AssignmentState ParallelTemperingEngine::DeriveAssignmentWires(const AssignmentState& assignment) const {
  return impl_->DeriveAssignmentWires(assignment);
}

void ParallelTemperingEngine::SetMultipliers(const std::vector<double>& multipliers) {
  impl_->SetMultipliers(multipliers);
}

void ParallelTemperingEngine::SetTemperatures(const std::vector<double>& temperatures) {
  impl_->SetTemperatures(temperatures);
}

void ParallelTemperingEngine::SetConstraintClasses(const std::vector<int32_t>& classes) {
  impl_->SetConstraintClasses(classes);
}

}  // namespace aes_xts_decoder
