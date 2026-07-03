#pragma once

#include "proposal.hpp"

#include <cstdint>

namespace aes_xts_decoder {

class WireBitFlipProposal final : public Proposal {
 public:
  WireBitFlipProposal(int32_t wire_index, uint8_t old_value, uint8_t new_value);

  void Apply(ByteState* state) const override;
  void Revert(ByteState* state) const override;
  int32_t ChangedWire() const override { return wire_index_; }
  uint8_t NewWireValue() const override { return new_value_; }

 private:
  int32_t wire_index_;
  uint8_t old_value_;
  uint8_t new_value_;
};

class RandomWireBitFlipGenerator final : public ProposalGenerator {
 public:
  std::unique_ptr<Proposal> Generate(
      const ByteState& state,
      std::mt19937_64* rng) override;
};

}  // namespace aes_xts_decoder
