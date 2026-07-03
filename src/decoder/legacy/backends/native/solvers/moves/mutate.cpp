#include "../../include/moves/mutate.hpp"

#include <cstddef>
#include <random>

namespace aes_xts_decoder {

WireBitFlipProposal::WireBitFlipProposal(
    int32_t wire_index,
    uint8_t old_value,
    uint8_t new_value)
    : wire_index_(wire_index), old_value_(old_value), new_value_(new_value) {}

void WireBitFlipProposal::Apply(ByteState* state) const {
  state->wires[wire_index_] = new_value_;
}

void WireBitFlipProposal::Revert(ByteState* state) const {
  state->wires[wire_index_] = old_value_;
}

std::unique_ptr<Proposal> RandomWireBitFlipGenerator::Generate(
    const ByteState& state,
    std::mt19937_64* rng) {
  if (state.wires.empty()) return nullptr;

  std::uniform_int_distribution<std::size_t> wire_dist(0, state.wires.size() - 1);
  std::uniform_int_distribution<int> bit_dist(0, 7);
  const auto wire_index = static_cast<int32_t>(wire_dist(*rng));
  const auto bit = static_cast<uint8_t>(bit_dist(*rng));
  const uint8_t old_value = state.wires[wire_index];
  const uint8_t new_value = static_cast<uint8_t>(old_value ^ (1u << bit));
  return std::unique_ptr<Proposal>(
      new WireBitFlipProposal(wire_index, old_value, new_value));
}

}  // namespace aes_xts_decoder
