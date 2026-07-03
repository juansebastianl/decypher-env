#include "solver.hpp"

namespace aes_xts_decoder {

void ParallelTemperingSolver::DualUpdate() {
  dual_updater_.Update(*this);
}

void ParallelTemperingSolver::MaybeHarvest(const AssignmentState& assignment, const ScoreData& score) {
  const uint64_t profile_start = ProfileStart();
  feasible_archive_.MaybeInsert(assignment, score, &seen_keys_, &feasible_);
  ProfileAdd(PROFILE_HARVEST, profile_start);
}

void ParallelTemperingSolver::RecordAcceptedKeyStats(std::size_t replica_index) {
  if (replica_index != 0 || replicas_.empty()) return;
  if (sweeps_ == 0 || epochs_ < config_.marginal_burn_in_epochs) return;
  const AssignmentState& state = replicas_[0];
  std::size_t bit_index = 0;
  std::array<uint8_t, 512> bits{};
  auto count_key = [&](const std::vector<uint8_t>& key) {
    for (const uint8_t byte : key) {
      for (int shift = 7; shift >= 0 && bit_index < key_ones_.size(); --shift) {
        bits[bit_index] = static_cast<uint8_t>((byte >> shift) & 1u);
        key_ones_[bit_index] += bits[bit_index];
        ++bit_index;
      }
    }
  };
  count_key(state.key1);
  count_key(state.key2);
  ++key_visit_count_;
  key_bit_window_.push_back(bits);
  const std::string key = KeyString(state);
  key_string_window_.push_back(key);
  ++key_window_counts_[key];
  const std::size_t window = config_.marginal_window == 0 ? kMaxColdTrace : std::max<std::size_t>(50, config_.marginal_window);
  while (key_bit_window_.size() > window) {
    const auto& expired = key_bit_window_.front();
    for (std::size_t i = 0; i < key_ones_.size(); ++i) key_ones_[i] -= expired[i];
    key_bit_window_.pop_front();
    const std::string& expired_key = key_string_window_.front();
    auto found = key_window_counts_.find(expired_key);
    if (found != key_window_counts_.end()) {
      if (found->second <= 1) {
        key_window_counts_.erase(found);
      } else {
        --found->second;
      }
    }
    key_string_window_.pop_front();
    --key_visit_count_;
  }
  double hamming_weight = 0.0;
  for (const uint8_t bit : bits) hamming_weight += bit;
  cold_key_hamming_trace_.push_back(hamming_weight);
  if (cold_key_hamming_trace_.size() > kMaxColdTrace) cold_key_hamming_trace_.pop_front();
}

void ParallelTemperingSolver::RecordEnergyStats() {
  const std::size_t count = std::min(energies_.size(), energy_sample_counts_.size());
  for (std::size_t i = 0; i < count; ++i) {
    const double value = energies_[i];
    ++energy_sample_counts_[i];
    const double delta = value - energy_mean_by_rung_[i];
    energy_mean_by_rung_[i] += delta / static_cast<double>(energy_sample_counts_[i]);
    const double delta2 = value - energy_mean_by_rung_[i];
    energy_m2_by_rung_[i] += delta * delta2;
  }
}

void ParallelTemperingSolver::ResetEnergyStats() {
  std::fill(energy_sample_counts_.begin(), energy_sample_counts_.end(), 0);
  std::fill(energy_mean_by_rung_.begin(), energy_mean_by_rung_.end(), 0.0);
  std::fill(energy_m2_by_rung_.begin(), energy_m2_by_rung_.end(), 0.0);
  energy_temperatures_by_rung_ = EffectiveTemperatures();
}

void ParallelTemperingSolver::UpdateReplicaFlow() {
  const std::size_t count = replicas_.size();
  if (count == 0) return;
  const std::size_t top = count - 1;
  for (std::size_t rung = 0; rung < count; ++rung) {
    if (rung == 0) {
      if (replica_directions_[rung] == -1) ++replica_round_trips_[rung];
      replica_directions_[rung] = 1;
    } else if (rung == top) {
      if (replica_directions_[rung] == 1) ++replica_round_trips_[rung];
      replica_directions_[rung] = -1;
    }
    if (replica_directions_[rung] > 0) {
      ++replica_up_counts_[rung];
    } else if (replica_directions_[rung] < 0) {
      ++replica_down_counts_[rung];
    }
  }
}

void ParallelTemperingSolver::ResetFlowCounters() {
  std::fill(replica_up_counts_.begin(), replica_up_counts_.end(), 0);
  std::fill(replica_down_counts_.begin(), replica_down_counts_.end(), 0);
}

void ParallelTemperingSolver::OptimizeTemperatureLadder() {
  ladder_adapter_.MaybeAdapt(*this);
}

}  // namespace aes_xts_decoder
