#pragma once

#include "../../include/pt_engine.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aes_xts_decoder {

struct AlgebraicBpState {
  std::vector<double> key_byte_beliefs;
  bool available = false;
  bool converged = false;
  double residual = 0.0;
  double bethe_free_energy = 0.0;
  double entropy = 0.0;
  double survey_entropy = 0.0;
};

AlgebraicBpState ComputeAlgebraicBpState(
    const AssignmentState& assignment,
    const std::vector<int32_t>& xts_block_sectors,
    const std::vector<int32_t>& xts_block_indices,
    const std::vector<uint8_t>& xts_block_targets,
    const PTConfig& config);

AssignmentState SampleBeliefKey(
    const AssignmentState& base,
    const AlgebraicBpState& bp,
    uint64_t* rng_state);

}  // namespace aes_xts_decoder
