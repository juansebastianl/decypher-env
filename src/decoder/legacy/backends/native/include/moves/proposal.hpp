#pragma once

#include "../state/byte_state.hpp"

#include <cstdint>
#include <memory>
#include <random>

namespace aes_xts_decoder {

class Proposal {
 public:
  virtual ~Proposal() = default;

  virtual void Apply(ByteState* state) const = 0;
  virtual void Revert(ByteState* state) const = 0;

  // Returns -1 unless the proposal updates exactly one internal wire byte.
  virtual int32_t ChangedWire() const { return -1; }
  virtual uint8_t NewWireValue() const { return 0; }
};

class ProposalGenerator {
 public:
  virtual ~ProposalGenerator() = default;

  virtual std::unique_ptr<Proposal> Generate(
      const ByteState& state,
      std::mt19937_64* rng) = 0;
};

}  // namespace aes_xts_decoder
