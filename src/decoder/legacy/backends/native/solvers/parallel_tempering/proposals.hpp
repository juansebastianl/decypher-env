#pragma once

#include "circuit_model.hpp"

namespace aes_xts_decoder {

struct ProposalResult {
  AssignmentState candidate;
  int family = 0;
  int32_t changed_value_id = -1;
};

class ProposalKernel {
 public:
  ProposalKernel(const CircuitModel& circuit, const PTConfig& config);

  ProposalResult Mutate(const AssignmentState& assignment, uint64_t* rng_state) const;
  bool RepairMove(const AssignmentState& assignment, AssignmentState* candidate, uint64_t* rng_state) const;

 private:
  bool KeyWordSwap(AssignmentState* candidate, uint64_t* rng_state) const;

  const CircuitModel& circuit_;
  const PTConfig& config_;
};

}  // namespace aes_xts_decoder
