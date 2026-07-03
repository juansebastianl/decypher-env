#pragma once

#include "byte_evaluator.hpp"

namespace aes_xts_decoder {

class HammingScorer {
 public:
  HammingScorer(const CircuitModel& circuit, const ByteCircuitEvaluator& evaluator);

  ScoreData ScoreFromValues(const ValueBuffer& values) const;
  ScoreData ScoreAssignment(const AssignmentState& assignment) const;
  uint32_t ResidualForConstraint(const ValueBuffer& values, std::size_t constraint_index) const;

 private:
  const CircuitModel& circuit_;
  const ByteCircuitEvaluator& evaluator_;
};

}  // namespace aes_xts_decoder
