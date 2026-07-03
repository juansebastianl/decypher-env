#include "hamming_scorer.hpp"

#include <algorithm>

namespace aes_xts_decoder {

HammingScorer::HammingScorer(
    const CircuitModel& circuit,
    const ByteCircuitEvaluator& evaluator)
    : circuit_(circuit), evaluator_(evaluator) {}

ScoreData HammingScorer::ScoreFromValues(const ValueBuffer& values) const {
  ScoreData score{0, 0.0, {}, {}};
  score.residuals.reserve(circuit_.constraint_kinds.size());
  for (std::size_t i = 0; i < circuit_.constraint_kinds.size(); ++i) {
    const uint32_t residual = ResidualForConstraint(values, i);
    score.residuals.push_back(residual);
    if (residual != 0) {
      ++score.violations;
      score.hamming_score += residual;
      score.failing_indices.push_back(static_cast<uint32_t>(i));
    }
  }
  return score;
}

ScoreData HammingScorer::ScoreAssignment(const AssignmentState& assignment) const {
  return ScoreFromValues(evaluator_.EvaluateValues(assignment));
}

uint32_t HammingScorer::ResidualForConstraint(const ValueBuffer& values, std::size_t i) const {
  uint32_t residual = 0;
  const int32_t left_id = circuit_.constraint_left[i];
  if (circuit_.constraint_kinds[i] == CONSTRAINT_ASCII) {
    return IsTextAscii(evaluator_.ValueU8(values, left_id)) ? 0 : 1;
  }
  const int32_t right_id = circuit_.constraint_right[i];
  if (circuit_.constraint_kinds[i] == CONSTRAINT_DEFINE8 || circuit_.constraint_kinds[i] == CONSTRAINT_EQ8) {
    return Popcount(evaluator_.ValueU8(values, left_id) ^ evaluator_.ValueU8(values, right_id));
  }
  if (circuit_.constraint_kinds[i] == CONSTRAINT_EQ128) {
    const std::size_t count = std::min(evaluator_.ValueWidth(left_id), evaluator_.ValueWidth(right_id));
    const uint8_t* left = evaluator_.ValuePtr(values, left_id);
    const uint8_t* right = evaluator_.ValuePtr(values, right_id);
    for (std::size_t b = 0; b < count; ++b) residual += Popcount(left[b] ^ right[b]);
  }
  return residual;
}

}  // namespace aes_xts_decoder
