#include "solver.hpp"

namespace aes_xts_decoder {

ParallelTemperingSolver::ParallelTemperingSolver(
    std::size_t value_count,
    const std::vector<int32_t>& opcodes,
    const std::vector<int32_t>& outputs,
    const std::vector<int32_t>& input_offsets,
    const std::vector<int32_t>& input_counts,
    const std::vector<int32_t>& inputs,
    const std::vector<int32_t>& immediate_offsets,
    const std::vector<int32_t>& immediate_counts,
    const std::vector<int32_t>& immediates,
    const std::vector<int32_t>& const_offsets,
    const std::vector<int32_t>& const_counts,
    const std::vector<uint8_t>& constants,
    const std::vector<int32_t>& constraint_kinds,
    const std::vector<int32_t>& constraint_left,
    const std::vector<int32_t>& constraint_right,
    const std::vector<int32_t>& wire_value_ids,
    const std::vector<int32_t>& wire_offsets,
    const std::vector<uint16_t>& value_widths,
    const std::vector<int32_t>& constraint_classes,
    std::size_t plaintext_start,
    std::size_t plaintext_count,
    std::size_t key1_start,
    std::size_t key1_count,
    std::size_t key2_start,
    std::size_t key2_count,
    const std::vector<int32_t>& xts_block_sectors,
    const std::vector<int32_t>& xts_block_indices,
    const std::vector<uint8_t>& xts_block_targets,
    const AssignmentState& initial,
    const PTConfig& config)
    : circuit_(
          value_count,
          opcodes,
          outputs,
          input_offsets,
          input_counts,
          inputs,
          immediate_offsets,
          immediate_counts,
          immediates,
          const_offsets,
          const_counts,
          constants,
          constraint_kinds,
          constraint_left,
          constraint_right,
          wire_value_ids,
          wire_offsets,
          constraint_classes,
          plaintext_start,
          plaintext_count,
          key1_start,
          key1_count,
          key2_start,
          key2_count,
          xts_block_sectors,
          xts_block_indices),
      config_(config),
      key_visit_count_(0),
      last_ladder_adaptation_epoch_(0),
      rho_by_class_{config.scheduled_rho_initial, config.scheduled_rho_initial, config.scheduled_rho_initial},
      previous_violation_by_class_{-1.0, -1.0, -1.0},
      lambda_update_counts_by_class_{0, 0, 0},
      rho_escalation_counts_by_class_{0, 0, 0},
      infeasibility_suspected_(false),
      bp_converged_(false),
      langevin_seed_score_(0.0),
      thread_count_(std::max<std::size_t>(1, config.threads ? config.threads : config.replicas)),
      epochs_(0),
      sweeps_(0),
      rng_state_(config.seed),
      evaluator_(circuit_),
      scorer_(circuit_, evaluator_),
      proposal_kernel_(circuit_, config_) {
  key_ones_.fill(0);
  const std::size_t xts_block_count = circuit_.xts_block_sectors.size();
  if (xts_block_targets.size() >= xts_block_count * 16) {
    circuit_.xts_block_targets.resize(xts_block_count);
    for (std::size_t b = 0; b < xts_block_count; ++b) {
      for (int i = 0; i < 16; ++i) {
        circuit_.xts_block_targets[b][i] = xts_block_targets[b * 16 + i];
      }
    }
  } else {
    // Mismatched metadata disables the repair move rather than risking bad reads.
    circuit_.xts_block_sectors.clear();
    circuit_.xts_block_indices.clear();
  }
  if (rho_by_class_[0] <= 0.0) rho_by_class_.fill(std::max(config.mu, 1e-9));
  circuit_.graph.value_offsets.assign(circuit_.value_count, 0);
  circuit_.graph.value_width_bytes.assign(circuit_.value_count, 1);
  for (std::size_t i = 0; i < circuit_.value_count; ++i) {
    const std::size_t bits = i < value_widths.size() ? value_widths[i] : 8;
    circuit_.graph.value_width_bytes[i] = std::max<std::size_t>(1, bits / 8);
    circuit_.graph.value_offsets[i] = circuit_.graph.value_storage_size;
    circuit_.graph.value_storage_size += circuit_.graph.value_width_bytes[i];
  }
  circuit_.graph.wire_offset_by_value.assign(circuit_.value_count, -1);
  circuit_.graph.wire_value_by_offset.assign(circuit_.wire_value_ids.size(), -1);
  for (std::size_t i = 0; i < circuit_.wire_value_ids.size(); ++i) {
    if (circuit_.wire_value_ids[i] >= 0 && static_cast<std::size_t>(circuit_.wire_value_ids[i]) < circuit_.graph.wire_offset_by_value.size()) {
      circuit_.graph.wire_offset_by_value[circuit_.wire_value_ids[i]] = circuit_.wire_offsets[i];
      if (circuit_.wire_offsets[i] >= 0 && static_cast<std::size_t>(circuit_.wire_offsets[i]) < circuit_.graph.wire_value_by_offset.size()) {
        circuit_.graph.wire_value_by_offset[circuit_.wire_offsets[i]] = circuit_.wire_value_ids[i];
      }
    }
  }
  circuit_.graph.definitions_by_right.assign(circuit_.value_count, {});
  for (std::size_t i = 0; i < circuit_.constraint_kinds.size(); ++i) {
    if (circuit_.constraint_kinds[i] == CONSTRAINT_DEFINE8 && circuit_.constraint_right[i] >= 0 &&
        static_cast<std::size_t>(circuit_.constraint_right[i]) < circuit_.graph.definitions_by_right.size()) {
      circuit_.graph.definitions_by_right[circuit_.constraint_right[i]].push_back(static_cast<int32_t>(i));
    }
  }
  std::vector<std::vector<int32_t>> producer_inputs(circuit_.value_count);
  circuit_.graph.ops_by_input.assign(circuit_.value_count, {});
  for (std::size_t op = 0; op < circuit_.opcodes.size(); ++op) {
    if (circuit_.outputs[op] < 0 || static_cast<std::size_t>(circuit_.outputs[op]) >= producer_inputs.size()) continue;
    for (int i = 0; i < circuit_.input_counts[op]; ++i) {
      const int32_t input = circuit_.inputs[circuit_.input_offsets[op] + i];
      producer_inputs[circuit_.outputs[op]].push_back(input);
      if (input >= 0 && static_cast<std::size_t>(input) < circuit_.graph.ops_by_input.size()) {
        circuit_.graph.ops_by_input[input].push_back(static_cast<int32_t>(op));
      }
    }
  }
  circuit_.graph.constraints_by_value.assign(circuit_.value_count, {});
  for (std::size_t c = 0; c < circuit_.constraint_kinds.size(); ++c) {
    std::vector<int32_t> touched;
    touched.push_back(circuit_.constraint_left[c]);
    if (circuit_.constraint_right[c] >= 0) touched.push_back(circuit_.constraint_right[c]);
    const std::size_t original_size = touched.size();
    for (std::size_t i = 0; i < original_size; ++i) {
      const int32_t value = touched[i];
      if (value >= 0 && static_cast<std::size_t>(value) < producer_inputs.size()) {
        touched.insert(touched.end(), producer_inputs[value].begin(), producer_inputs[value].end());
      }
    }
    for (const int32_t value : touched) {
      if (value >= 0 && static_cast<std::size_t>(value) < circuit_.graph.constraints_by_value.size()) {
        auto& entries = circuit_.graph.constraints_by_value[value];
        if (std::find(entries.begin(), entries.end(), static_cast<int32_t>(c)) == entries.end()) {
          entries.push_back(static_cast<int32_t>(c));
        }
      }
    }
  }
  const std::size_t replica_count = std::max<std::size_t>(1, config.replicas);
  temperatures_ = TemperatureLadder(replica_count, config.t_min, config.t_max);
  UpdateReplicaScales();
  multipliers_ = InitialMultipliers(config);
  UpdatePenaltyCache();
  swap_attempts_.assign(replica_count > 0 ? replica_count - 1 : 0, 0);
  swap_accepts_.assign(replica_count > 0 ? replica_count - 1 : 0, 0);
  proposal_attempts_.assign(4, 0);
  proposal_accepts_.assign(4, 0);
  replica_proposal_attempts_.assign(replica_count, {0, 0, 0, 0});
  replica_proposal_accepts_.assign(replica_count, {0, 0, 0, 0});
  profile_nanos_.assign(PROFILE_COUNTER_COUNT, 0);
  profile_counts_.assign(PROFILE_COUNTER_COUNT, 0);
  replica_rng_states_.assign(replica_count, 0);
  replica_key_caches_.resize(replica_count);
  replica_value_marks_.assign(replica_count, std::vector<uint32_t>(circuit_.value_count, 0));
  replica_constraint_marks_.assign(replica_count, std::vector<uint32_t>(circuit_.constraint_kinds.size(), 0));
  replica_mark_stamps_.assign(replica_count, 1);
  energy_sample_counts_.assign(replica_count, 0);
  energy_mean_by_rung_.assign(replica_count, 0.0);
  energy_m2_by_rung_.assign(replica_count, 0.0);
  energy_temperatures_by_rung_ = EffectiveTemperatures();
  replica_directions_.assign(replica_count, 0);
  replica_round_trips_.assign(replica_count, 0);
  replica_up_counts_.assign(replica_count, 0);
  replica_down_counts_.assign(replica_count, 0);
  algebra_counts_ = ComputeAlgebraCounts();
  bp_key_marginals_ = ComputeBpKeyMarginals();
  bp_converged_ = !bp_key_marginals_.empty();
  for (std::size_t i = 0; i < replica_count; ++i) {
    replica_rng_states_[i] = rng_state_ + 0x9E3779B97F4A7C15ULL * (i + 1);
    AssignmentState replica = initial;
    if (i != 0) {
      int family = 0;
      int32_t changed_value = -1;
      replica = Mutate(replica, &family, &changed_value, &replica_rng_states_[i]);
      if (family != 0) DeriveWires(&replica);
    }
    replicas_.push_back(replica);
    replica_values_.push_back(EvaluateValues(replica));
    scores_.push_back(ScoreFromValues(replica_values_.back()));
    energies_.push_back(Energy(scores_.back()));
    MaybeHarvest(replica, scores_.back());
  }
  RecordEnergyStats();
  UpdateReplicaFlow();
}

uint64_t ParallelTemperingSolver::ProfileStart() const {
  return config_.profile ? NowNanos() : 0;
}

void ParallelTemperingSolver::ProfileAdd(std::size_t counter, uint64_t started_at) const {
  if (!config_.profile || started_at == 0 || counter >= profile_nanos_.size()) return;
  const uint64_t elapsed = NowNanos() - started_at;
  #ifdef _OPENMP
  #pragma omp atomic update
  #endif
  profile_nanos_[counter] += elapsed;
  #ifdef _OPENMP
  #pragma omp atomic update
  #endif
  ++profile_counts_[counter];
}

std::vector<double> ParallelTemperingSolver::ProfileSeconds() const {
  return diagnostics_collector_.ProfileSeconds(profile_nanos_);
}

}  // namespace aes_xts_decoder
