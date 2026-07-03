#include "belief_propagation.hpp"

#include "../parallel_tempering/internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace aes_xts_decoder {
namespace {

constexpr std::size_t kKeyBytes = 64;
constexpr std::size_t kDomain = 256;

int TextAsciiDistanceLocal(uint8_t value) {
  if (IsTextAscii(value)) return 0;
  int best = 8;
  best = std::min(best, Popcount(static_cast<uint8_t>(value ^ 0x09)));
  best = std::min(best, Popcount(static_cast<uint8_t>(value ^ 0x0A)));
  best = std::min(best, Popcount(static_cast<uint8_t>(value ^ 0x0D)));
  for (int candidate = 0x20; candidate <= 0x7E; ++candidate) {
    best = std::min(best, Popcount(static_cast<uint8_t>(value ^ candidate)));
  }
  return best;
}

double AsciiByteEnergy(uint8_t value, const PTConfig& config) {
  const double residual = static_cast<double>(TextAsciiDistanceLocal(value)) / 8.0;
  const double penalty = std::max(config.mu, 1e-9);
  return config.ascii_weight * residual + 0.5 * penalty * residual * residual;
}

std::vector<std::array<uint8_t, 16>> BlockTargets(const std::vector<uint8_t>& flat) {
  std::vector<std::array<uint8_t, 16>> targets(flat.size() / 16);
  for (std::size_t block = 0; block < targets.size(); ++block) {
    for (std::size_t i = 0; i < 16; ++i) targets[block][i] = flat[16 * block + i];
  }
  return targets;
}

double RepairedAsciiEnergy(
    const AssignmentState& assignment,
    const std::vector<int32_t>& xts_block_sectors,
    const std::vector<int32_t>& xts_block_indices,
    const std::vector<std::array<uint8_t, 16>>& xts_block_targets,
    const PTConfig& config) {
  if (assignment.key1.size() < 32 || assignment.key2.size() < 32 || xts_block_targets.empty()) return 0.0;
  const std::array<uint8_t, 240> expanded1 = ExpandKey(assignment.key1);
  const std::array<uint8_t, 240> expanded2 = ExpandKey(assignment.key2);
  const int rounds = std::max(1, std::min(14, config.aes_rounds));
  double total = 0.0;
  const std::size_t block_count = std::min(xts_block_targets.size(), xts_block_sectors.size());
  for (std::size_t block = 0; block < block_count; ++block) {
    std::array<uint8_t, 16> tweak = EncryptReducedRound(expanded2, SectorTweakInput(xts_block_sectors[block]), rounds);
    const int32_t block_index = block < xts_block_indices.size() ? xts_block_indices[block] : 0;
    for (int32_t k = 0; k < block_index; ++k) tweak = XtsMulX(tweak);

    std::array<uint8_t, 16> pre{};
    for (std::size_t i = 0; i < 16; ++i) pre[i] = static_cast<uint8_t>(xts_block_targets[block][i] ^ tweak[i]);
    const std::array<uint8_t, 16> decrypted = DecryptReducedRound(expanded1, pre, rounds);
    for (std::size_t i = 0; i < 16; ++i) {
      total += AsciiByteEnergy(static_cast<uint8_t>(decrypted[i] ^ tweak[i]), config);
    }
  }
  return total;
}

void Normalize(std::array<double, kDomain>* probabilities) {
  double total = std::accumulate(probabilities->begin(), probabilities->end(), 0.0);
  if (total <= 0.0 || !std::isfinite(total)) {
    probabilities->fill(1.0 / static_cast<double>(kDomain));
    return;
  }
  for (double& value : *probabilities) value /= total;
}

double Entropy(const std::array<double, kDomain>& probabilities) {
  double entropy = 0.0;
  for (const double p : probabilities) {
    if (p > 0.0) entropy -= p * std::log(p);
  }
  return entropy;
}

}  // namespace

AlgebraicBpState ComputeAlgebraicBpState(
    const AssignmentState& assignment,
    const std::vector<int32_t>& xts_block_sectors,
    const std::vector<int32_t>& xts_block_indices,
    const std::vector<uint8_t>& xts_block_targets,
    const PTConfig& config) {
  AlgebraicBpState state;
  state.key_byte_beliefs.assign(kKeyBytes * kDomain, 1.0 / static_cast<double>(kDomain));
  if (assignment.key1.size() < 32 || assignment.key2.size() < 32) return state;

  const std::vector<std::array<uint8_t, 16>> targets = BlockTargets(xts_block_targets);
  if (targets.empty()) {
    state.available = true;
    state.converged = true;
    state.entropy = static_cast<double>(kKeyBytes) * std::log(static_cast<double>(kDomain));
    state.bethe_free_energy = -state.entropy;
    return state;
  }

  const std::size_t iterations = std::max<std::size_t>(1, config.bp_iterations ? config.bp_iterations : 20);
  const double damping = std::max(0.0, std::min(0.99, config.bp_damping));
  const double tolerance = config.bp_tolerance > 0.0 ? config.bp_tolerance : 1e-4;
  std::vector<double> previous = state.key_byte_beliefs;
  double max_delta = 0.0;

  for (std::size_t iter = 0; iter < iterations; ++iter) {
    max_delta = 0.0;
    for (std::size_t key_byte = 0; key_byte < kKeyBytes; ++key_byte) {
      AssignmentState candidate = assignment;
      std::vector<uint8_t>& key = key_byte < 32 ? candidate.key1 : candidate.key2;
      const std::size_t index = key_byte < 32 ? key_byte : key_byte - 32;
      std::array<double, kDomain> energies{};
      double min_energy = 0.0;
      for (std::size_t value = 0; value < kDomain; ++value) {
        key[index] = static_cast<uint8_t>(value);
        energies[value] = RepairedAsciiEnergy(candidate, xts_block_sectors, xts_block_indices, targets, config);
        if (value == 0 || energies[value] < min_energy) min_energy = energies[value];
      }

      std::array<double, kDomain> updated{};
      for (std::size_t value = 0; value < kDomain; ++value) {
        updated[value] = std::exp(std::max(-60.0, std::min(60.0, -(energies[value] - min_energy))));
      }
      Normalize(&updated);

      for (std::size_t value = 0; value < kDomain; ++value) {
        const std::size_t offset = key_byte * kDomain + value;
        const double damped = damping * previous[offset] + (1.0 - damping) * updated[value];
        max_delta = std::max(max_delta, std::abs(damped - previous[offset]));
        state.key_byte_beliefs[offset] = damped;
      }
    }
    previous = state.key_byte_beliefs;
    if (max_delta <= tolerance) break;
  }

  state.available = true;
  state.converged = max_delta <= tolerance;
  state.residual = max_delta;
  state.entropy = 0.0;
  for (std::size_t key_byte = 0; key_byte < kKeyBytes; ++key_byte) {
    std::array<double, kDomain> probabilities{};
    for (std::size_t value = 0; value < kDomain; ++value) {
      probabilities[value] = state.key_byte_beliefs[key_byte * kDomain + value];
    }
    Normalize(&probabilities);
    for (std::size_t value = 0; value < kDomain; ++value) {
      state.key_byte_beliefs[key_byte * kDomain + value] = probabilities[value];
    }
    state.entropy += Entropy(probabilities);
  }
  state.bethe_free_energy = RepairedAsciiEnergy(assignment, xts_block_sectors, xts_block_indices, targets, config) - state.entropy;
  const std::size_t restarts = config.survey_restarts;
  state.survey_entropy = restarts == 0 ? state.entropy : state.entropy / static_cast<double>(restarts + 1);
  return state;
}

AssignmentState SampleBeliefKey(
    const AssignmentState& base,
    const AlgebraicBpState& bp,
    uint64_t* rng_state) {
  AssignmentState sampled = base;
  if (bp.key_byte_beliefs.size() < kKeyBytes * kDomain || sampled.key1.size() < 32 || sampled.key2.size() < 32) {
    return sampled;
  }
  for (std::size_t key_byte = 0; key_byte < kKeyBytes; ++key_byte) {
    const double threshold = Uniform(rng_state);
    double cumulative = 0.0;
    uint8_t chosen = 255;
    for (std::size_t value = 0; value < kDomain; ++value) {
      cumulative += bp.key_byte_beliefs[key_byte * kDomain + value];
      if (cumulative >= threshold) {
        chosen = static_cast<uint8_t>(value);
        break;
      }
    }
    if (key_byte < 32) {
      sampled.key1[key_byte] = chosen;
    } else {
      sampled.key2[key_byte - 32] = chosen;
    }
  }
  return sampled;
}

}  // namespace aes_xts_decoder
