#include "proposals.hpp"

namespace aes_xts_decoder {

ProposalKernel::ProposalKernel(const CircuitModel& circuit, const PTConfig& config)
    : circuit_(circuit), config_(config) {}

ProposalResult ProposalKernel::Mutate(const AssignmentState& assignment, uint64_t* rng_state) const {
  ProposalResult result;
  result.candidate = assignment;
  const double roll = Uniform(rng_state);
  if (!result.candidate.wires.empty() && roll < 0.65) {
    result.family = 0;
    const std::size_t offset = RandRange(rng_state, result.candidate.wires.size());
    result.candidate.wires[offset] ^= static_cast<uint8_t>(1u << RandRange(rng_state, 8));
    if (offset < circuit_.graph.wire_value_by_offset.size()) {
      result.changed_value_id = circuit_.graph.wire_value_by_offset[offset];
    }
  } else if (roll < 0.9 && !result.candidate.plaintext.empty()) {
    result.family = 1;
    // This proposal repairs non-ASCII starts quickly; strict reversibility begins once the byte is in the ASCII support.
    result.candidate.plaintext[RandRange(rng_state, result.candidate.plaintext.size())] = TextAscii(rng_state);
  } else if ((result.candidate.key1.size() >= 32 || result.candidate.key2.size() >= 32) &&
             Uniform(rng_state) < 0.5 && KeyWordSwap(&result.candidate, rng_state)) {
    result.family = 3;
  } else {
    result.family = 2;
    auto& key = Uniform(rng_state) < 0.5 ? result.candidate.key1 : result.candidate.key2;
    key[RandRange(rng_state, key.size())] ^= static_cast<uint8_t>(1u << RandRange(rng_state, 8));
  }
  return result;
}

bool ProposalKernel::KeyWordSwap(AssignmentState* candidate, uint64_t* rng_state) const {
  if (candidate == nullptr || (candidate->key1.size() < 32 && candidate->key2.size() < 32)) return false;
  std::vector<uint8_t>& key =
      candidate->key1.size() >= 32 && (candidate->key2.size() < 32 || Uniform(rng_state) < 0.5)
          ? candidate->key1
          : candidate->key2;
  const std::size_t word_a = RandRange(rng_state, 8);
  std::size_t word_b = RandRange(rng_state, 7);
  if (word_b >= word_a) ++word_b;
  for (std::size_t byte = 0; byte < 4; ++byte) {
    std::swap(key[4u * word_a + byte], key[4u * word_b + byte]);
  }
  return true;
}

bool ProposalKernel::RepairMove(const AssignmentState& assignment, AssignmentState* candidate, uint64_t* rng_state) const {
  if (circuit_.xts_block_sectors.empty()) return false;
  const int rounds = (config_.aes_rounds >= 1 && config_.aes_rounds <= 14) ? config_.aes_rounds : 14;
  *candidate = assignment;
  if (candidate->key1.size() < 32 || candidate->key2.size() < 32) return false;

  const std::size_t block = RandRange(rng_state, circuit_.xts_block_sectors.size());
  const std::size_t pt_offset = block * 16;
  if (pt_offset + 16 > candidate->plaintext.size()) return false;

  const std::array<uint8_t, 240> expanded1 = ExpandKey(candidate->key1);
  const std::array<uint8_t, 240> expanded2 = ExpandKey(candidate->key2);

  std::array<uint8_t, 16> tweak = EncryptReducedRound(expanded2, SectorTweakInput(circuit_.xts_block_sectors[block]), rounds);
  const int32_t block_index = circuit_.xts_block_indices[block];
  for (int32_t k = 0; k < block_index; ++k) tweak = XtsMulX(tweak);

  const std::array<uint8_t, 16>& target = circuit_.xts_block_targets[block];
  std::array<uint8_t, 16> pre{};
  for (int i = 0; i < 16; ++i) pre[i] = static_cast<uint8_t>(target[i] ^ tweak[i]);
  const std::array<uint8_t, 16> dec = DecryptReducedRound(expanded1, pre, rounds);
  for (int i = 0; i < 16; ++i) {
    candidate->plaintext[pt_offset + i] = static_cast<uint8_t>(dec[i] ^ tweak[i]);
  }
  return true;
}

}  // namespace aes_xts_decoder
