#include "solver.hpp"

namespace aes_xts_decoder {

ValueBuffer ParallelTemperingSolver::MakeValueBuffer() const {
  return evaluator_.MakeValueBuffer();
}

const uint8_t* ParallelTemperingSolver::ValuePtr(const ValueBuffer& values, int32_t value_id) const {
  return evaluator_.ValuePtr(values, value_id);
}

uint8_t* ParallelTemperingSolver::MutableValuePtr(ValueBuffer* values, int32_t value_id) const {
  return evaluator_.MutableValuePtr(values, value_id);
}

std::size_t ParallelTemperingSolver::ValueWidth(int32_t value_id) const {
  return evaluator_.ValueWidth(value_id);
}

uint8_t ParallelTemperingSolver::ValueU8(const ValueBuffer& values, int32_t value_id) const {
  return evaluator_.ValueU8(values, value_id);
}

void ParallelTemperingSolver::SetValueU8(ValueBuffer* values, int32_t value_id, uint8_t value) const {
  evaluator_.SetValueU8(values, value_id, value);
}

void ParallelTemperingSolver::SetValueBytes(ValueBuffer* values, int32_t value_id, const uint8_t* data, std::size_t size) const {
  evaluator_.SetValueBytes(values, value_id, data, size);
}

std::vector<uint8_t> ParallelTemperingSolver::CopyValueBytes(const ValueBuffer& values, int32_t value_id) const {
  return evaluator_.CopyValueBytes(values, value_id);
}

ValueBuffer ParallelTemperingSolver::EvaluateValues(const AssignmentState& assignment) const {
  const uint64_t profile_start = ProfileStart();
  auto values = evaluator_.EvaluateValues(assignment);
  ProfileAdd(PROFILE_EVALUATE_VALUES, profile_start);
  return values;
}

ValueBuffer ParallelTemperingSolver::EvaluateValuesAndDerive(AssignmentState* assignment) const {
  const uint64_t profile_start = ProfileStart();
  auto values = evaluator_.EvaluateValuesAndDerive(assignment);
  ProfileAdd(PROFILE_EVALUATE_VALUES_DERIVE, profile_start);
  return values;
}

void ParallelTemperingSolver::EvaluateOpInto(std::size_t op, const AssignmentState& assignment, ValueBuffer* values, KeyCache* cache) const {
  evaluator_.EvaluateOpInto(op, assignment, values, cache);
}

ValueBuffer ParallelTemperingSolver::EvaluateValuesInternal(const AssignmentState& assignment, AssignmentState* derive_target, KeyCache* cache) const {
  return evaluator_.EvaluateValuesInternal(assignment, derive_target, cache);
}

void ParallelTemperingSolver::DeriveWires(AssignmentState* assignment) const {
  const uint64_t profile_start = ProfileStart();
  evaluator_.DeriveWires(assignment);
  ProfileAdd(PROFILE_DERIVE_WIRES, profile_start);
}

ScoreData ParallelTemperingSolver::ScoreFromValues(const ValueBuffer& values) const {
  const uint64_t profile_start = ProfileStart();
  ScoreData score = scorer_.ScoreFromValues(values);
  ProfileAdd(PROFILE_RESIDUAL_SCAN, profile_start);
  return score;
}

ScoreData ParallelTemperingSolver::ScoreAssignment(const AssignmentState& assignment) const {
  const uint64_t profile_start = ProfileStart();
  auto score = scorer_.ScoreAssignment(assignment);
  ProfileAdd(PROFILE_SCORE_FULL, profile_start);
  return score;
}

uint32_t ParallelTemperingSolver::ResidualForConstraint(const ValueBuffer& values, std::size_t i) const {
  return scorer_.ResidualForConstraint(values, i);
}

ScoreData ParallelTemperingSolver::ScoreAssignmentAffected(
    const AssignmentState& assignment,
    const ScoreData& base,
    const std::vector<int32_t>& affected) const {
  ScoreData score = base;
  KeyCache cache;
  const auto values = EvaluateValuesInternal(assignment, nullptr, &cache);
  for (const int32_t index : affected) {
    if (index < 0 || static_cast<std::size_t>(index) >= score.residuals.size()) continue;
    const uint32_t old_residual = score.residuals[index];
    const uint32_t new_residual = ResidualForConstraint(values, index);
    if (old_residual == new_residual) continue;
    score.residuals[index] = new_residual;
    score.hamming_score += static_cast<double>(new_residual) - static_cast<double>(old_residual);
  }
  score.violations = 0;
  score.failing_indices.clear();
  for (std::size_t i = 0; i < score.residuals.size(); ++i) {
    if (score.residuals[i] != 0) {
      ++score.violations;
      score.failing_indices.push_back(static_cast<uint32_t>(i));
    }
  }
  return score;
}

ScoreData ParallelTemperingSolver::ScoreWireFlipAffected(
    std::size_t replica_index,
    const AssignmentState& assignment,
    const ScoreData& base,
    int32_t changed_value_id,
    std::vector<std::pair<int32_t, std::vector<uint8_t>>>* undo) {
  const uint64_t profile_start = ProfileStart();
  ScoreData score = base;
  auto& values = replica_values_[replica_index];
  auto remember = [&](int32_t value_id) {
    for (const auto& entry : *undo) {
      if (entry.first == value_id) return;
    }
    undo->push_back({value_id, CopyValueBytes(values, value_id)});
  };

  std::vector<int32_t> queue;
  std::vector<int32_t> affected_constraints;
  uint32_t& stamp = replica_mark_stamps_[replica_index];
  ++stamp;
  if (stamp == 0) {
    std::fill(replica_value_marks_[replica_index].begin(), replica_value_marks_[replica_index].end(), 0);
    std::fill(replica_constraint_marks_[replica_index].begin(), replica_constraint_marks_[replica_index].end(), 0);
    stamp = 1;
  }
  auto mark_value = [&](int32_t value_id) {
    if (value_id < 0 || static_cast<std::size_t>(value_id) >= replica_value_marks_[replica_index].size()) return false;
    if (replica_value_marks_[replica_index][value_id] == stamp) return false;
    replica_value_marks_[replica_index][value_id] = stamp;
    return true;
  };
  auto mark_constraints = [&](int32_t value_id) {
    if (value_id < 0 || static_cast<std::size_t>(value_id) >= circuit_.graph.constraints_by_value.size()) return;
    for (const int32_t constraint : circuit_.graph.constraints_by_value[value_id]) {
      if (constraint < 0 || static_cast<std::size_t>(constraint) >= replica_constraint_marks_[replica_index].size()) continue;
      if (replica_constraint_marks_[replica_index][constraint] == stamp) continue;
      replica_constraint_marks_[replica_index][constraint] = stamp;
      affected_constraints.push_back(constraint);
    }
  };

  remember(changed_value_id);
  const int32_t wire_offset = circuit_.graph.wire_offset_by_value[changed_value_id];
  SetValueU8(&values, changed_value_id, assignment.wires[wire_offset]);
  queue.push_back(changed_value_id);
  mark_value(changed_value_id);
  mark_constraints(changed_value_id);

  for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
    const int32_t value_id = queue[cursor];
    if (value_id < 0 || static_cast<std::size_t>(value_id) >= circuit_.graph.ops_by_input.size()) continue;
    for (const int32_t op_index : circuit_.graph.ops_by_input[value_id]) {
      const int32_t output = circuit_.outputs[op_index];
      remember(output);
      const auto old_value = CopyValueBytes(values, output);
      EvaluateOpInto(op_index, assignment, &values, &replica_key_caches_[replica_index]);
      mark_constraints(output);
      if (CopyValueBytes(values, output) != old_value && mark_value(output)) {
        queue.push_back(output);
      }
    }
  }

  for (const int32_t index : affected_constraints) {
    if (index < 0 || static_cast<std::size_t>(index) >= score.residuals.size()) continue;
    const uint32_t old_residual = score.residuals[index];
    const uint32_t new_residual = ResidualForConstraint(values, index);
    if (old_residual == new_residual) continue;
    score.residuals[index] = new_residual;
    score.hamming_score += static_cast<double>(new_residual) - static_cast<double>(old_residual);
    if (old_residual == 0 && new_residual != 0) {
      ++score.violations;
    } else if (old_residual != 0 && new_residual == 0 && score.violations > 0) {
      --score.violations;
    }
  }
  score.failing_indices.clear();
  ProfileAdd(PROFILE_SCORE_DELTA, profile_start);
  return score;
}

void ParallelTemperingSolver::RestoreValues(std::size_t replica_index, const std::vector<std::pair<int32_t, std::vector<uint8_t>>>& undo) {
  const uint64_t profile_start = ProfileStart();
  auto& values = replica_values_[replica_index];
  for (auto it = undo.rbegin(); it != undo.rend(); ++it) {
    SetValueBytes(&values, it->first, it->second.data(), it->second.size());
  }
  ProfileAdd(PROFILE_RESTORE_VALUES, profile_start);
}

}  // namespace aes_xts_decoder
