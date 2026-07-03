#pragma once

#include "../pt_engine.hpp"
#include "../state/state.hpp"

#include <cstdint>

namespace aes_xts_decoder {

// Scoring is intentionally independent from the sampler. The current native
// engine still owns circuit evaluation, but new engines can depend only on this
// interface and swap in alternate state representations or residual models.
class Scorer {
 public:
  virtual ~Scorer() = default;

  virtual ScoreData Score(const ByteStateView& state) const = 0;

  virtual ScoreData ScoreWireUpdate(
      const ByteStateView& state,
      int32_t changed_wire,
      uint8_t new_value,
      const ScoreData& current) const {
    (void)changed_wire;
    (void)new_value;
    (void)current;
    return Score(state);
  }
};

}  // namespace aes_xts_decoder
