#include "continuous_relaxed_engine.hpp"

#include "parallel_tempering/byte_evaluator.hpp"
#include "parallel_tempering/circuit_model.hpp"
#include "parallel_tempering/feasible_archive.hpp"
#include "parallel_tempering/hamming_scorer.hpp"
#include "parallel_tempering/internal.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace aes_xts_decoder {
namespace {

inline std::size_t BucketForClass(int cls) {
  return static_cast<std::size_t>(std::max(0, std::min(2, cls - 1)));
}

bool IsLinearOpcode(int opcode) {
  return opcode == OP_INPUT || opcode == OP_CONST || opcode == OP_XOR8 || opcode == OP_XOR128 ||
         opcode == OP_MIX_COLUMN_BYTE || opcode == OP_XTS_MUL_X_BYTE || opcode == OP_EXTRACT_BYTE;
}

double ClampExpArg(double value) {
  return std::max(-60.0, std::min(60.0, value));
}

int TextAsciiDistance(uint8_t value) {
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

}  // namespace

struct ContinuousRelaxedScore {
  ScoreData hard;
  std::vector<double> residuals;
  double smooth_sum = 0.0;
};

struct RelaxedAction {
  int family = 0;
  std::vector<std::pair<int, std::pair<std::size_t, uint8_t>>> flips;
  AssignmentState candidate;
  ValueBuffer candidate_values;
  ContinuousRelaxedScore candidate_score;
  KeyCache candidate_cache;
  double energy = 0.0;
  double weight = 0.0;
  bool candidate_ready = false;
  bool repair_valid = false;
};

struct KeyProfileCache {
  std::array<std::array<double, 256>, 32> energy{};
  std::array<uint8_t, 32> valid{};
  std::vector<uint8_t> cached_key1;
  std::vector<uint8_t> cached_key2;
};

constexpr std::size_t kSampledCandidatePool = 8;
constexpr std::size_t kSampledCandidateAttempts = 64;

uint64_t ActionKey(const RelaxedAction& action) {
  uint64_t key = static_cast<uint64_t>(static_cast<uint8_t>(action.family)) + 0x9E3779B97F4A7C15ULL;
  auto mix = [&](uint64_t value) {
    key ^= value + 0x9E3779B97F4A7C15ULL + (key << 6) + (key >> 2);
  };
  mix(action.flips.size());
  for (const auto& flip : action.flips) {
    mix(static_cast<uint64_t>(static_cast<uint8_t>(flip.first)));
    const std::size_t index = flip.second.first;
    mix(static_cast<uint64_t>(index));
    mix(static_cast<uint64_t>(flip.second.second));
  }
  return key;
}

class ContinuousRelaxedSolver {
 public:
  ContinuousRelaxedSolver(
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
        rng_state_(config.seed),
        evaluator_(circuit_),
        hard_scorer_(circuit_, evaluator_) {
    InitializeCircuit(value_widths, xts_block_targets);
    InitializeSboxTables();
    InitializeAsciiTables();
    const std::size_t replica_count = std::max<std::size_t>(1, config_.replicas);
    thread_count_ = std::max<std::size_t>(1, config_.threads ? config_.threads : config_.replicas);
    key_ones_.fill(0);
    temperatures_ = TemperatureLadder(replica_count, config_.t_min, config_.t_max);
    replica_scale_.assign(replica_count, 1.0);
    multipliers_ = InitialMultipliers();
    penalties_.assign(circuit_.constraint_kinds.size(), std::max(config_.mu, 1e-9));
    lambda_previous_ = multipliers_;
    dual_m_.assign(circuit_.constraint_kinds.size(), 0.0);
    dual_v_.assign(circuit_.constraint_kinds.size(), 0.0);
    dual_steps_.assign(circuit_.constraint_kinds.size(), 0);
    previous_violation_by_class_.fill(-1.0);
    rho_by_class_.fill(std::max(config_.scheduled_rho_initial > 0.0 ? config_.scheduled_rho_initial : config_.mu, 1e-9));
    lambda_update_counts_by_class_.fill(0);
    rho_escalation_counts_by_class_.fill(0);
    replica_proposal_attempts_.assign(replica_count, {0, 0, 0, 0, 0, 0, 0});
    replica_proposal_accepts_.assign(replica_count, {0, 0, 0, 0, 0, 0, 0});
    profile_nanos_.assign(PROFILE_COUNTER_COUNT, 0);
    profile_counts_.assign(PROFILE_COUNTER_COUNT, 0);
    swap_attempts_.assign(replica_count > 0 ? replica_count - 1 : 0, 0);
    swap_accepts_.assign(replica_count > 0 ? replica_count - 1 : 0, 0);
    energy_sample_counts_.assign(replica_count, 0);
    energy_mean_by_rung_.assign(replica_count, 0.0);
    energy_m2_by_rung_.assign(replica_count, 0.0);
    replica_round_trips_.assign(replica_count, 0);
    replica_up_counts_.assign(replica_count, 0);
    replica_down_counts_.assign(replica_count, 0);
    replica_directions_.assign(replica_count, 0);
    replica_rng_states_.assign(replica_count, 0);
    replica_key_caches_.resize(replica_count);
    replica_repair_valid_.assign(replica_count, false);
    replica_value_marks_.assign(replica_count, std::vector<uint32_t>(circuit_.value_count, 0));
    replica_constraint_marks_.assign(replica_count, std::vector<uint32_t>(circuit_.constraint_kinds.size(), 0));
    replica_mark_stamps_.assign(replica_count, 1);
    replica_pair_energy_.assign(replica_count, std::vector<double>(static_cast<std::size_t>(1) << 16, 0.0));
    replica_key1_profile_.resize(replica_count);

    for (std::size_t i = 0; i < replica_count; ++i) {
      replica_rng_states_[i] = rng_state_ + 0x9E3779B97F4A7C15ULL * (i + 1);
      AssignmentState replica = initial;
      if (i != 0) RandomizeSeed(&replica, &replica_rng_states_[i]);
      replica_repair_valid_[i] = ApplyAdmmRepair(&replica, &replica_key_caches_[i]);
      ApplyLinearPropagation(&replica);
      replicas_.push_back(replica);
      values_.push_back(EvaluateValues(replica, &replica_key_caches_[i]));
      relaxed_scores_.push_back(ScoreRelaxed(values_.back()));
      hard_scores_.push_back(relaxed_scores_.back().hard);
      energies_.push_back(Energy(relaxed_scores_.back()));
      if (hard_scores_.back().hamming_score == 0.0) MaybeHarvest(replica, hard_scores_.back());
    }
    RecordEnergyStats();
  }

  ScoreData ScoreAssignment(const AssignmentState& assignment) const {
    return hard_scorer_.ScoreAssignment(assignment);
  }

  ScoreData RunEpoch(std::size_t sweeps) {
    const uint64_t profile_start = ProfileStart();
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
      const uint64_t step_profile_start = ProfileStart();
      // Dynamic scheduling: per-replica step cost is highly uneven (cold replicas
      // run the ~16x-heavier key2 profile-refit while hot replicas take cheap
      // moves), so a static split leaves most threads idling on the slowest
      // replica. schedule(dynamic, 1) lets free threads grab the next replica.
      #ifdef _OPENMP
      #pragma omp parallel for num_threads(thread_count_) schedule(dynamic, 1)
      #endif
      for (std::int64_t replica = 0; replica < static_cast<std::int64_t>(replicas_.size()); ++replica) {
        const std::size_t replica_index = static_cast<std::size_t>(replica);
        RepairReplicaIfNeeded(replica_index);
        RefreshReplica(replica_index);
        StepReplica(replica_index);
      }
      ProfileAdd(PROFILE_STEP_REPLICAS_WALL, step_profile_start);
      AttemptSwaps();
      UpdateReplicaFlow();
      RecordEnergyStats();
      ++sweeps_;
    }
    ++epochs_;
    DualUpdate();
    RecomputeEnergies();
    PopulationAnneal();
    ProfileAdd(PROFILE_RUN_EPOCH_TOTAL, profile_start);
    return hard_scores_.empty() ? ScoreData{} : hard_scores_[0];
  }

  std::vector<uint32_t> Residuals() const {
    return hard_scores_.empty() ? std::vector<uint32_t>{} : hard_scores_[0].residuals;
  }

  PTDiagnostics Metrics() const {
    const uint64_t profile_start = ProfileStart();
    PTDiagnostics metrics{};
    metrics.epochs = epochs_;
    metrics.sweeps = sweeps_;
    metrics.feasible_count = feasible_.size();
    metrics.total_residual = hard_scores_.empty() ? 0.0 : hard_scores_[0].hamming_score;
    metrics.violations = hard_scores_.empty() ? 0 : hard_scores_[0].violations;
    metrics.max_residual = 0;
    metrics.swap_attempts = swap_attempts_;
    metrics.swap_accepts = swap_accepts_;
    metrics.proposal_attempts.assign(7, 0);
    metrics.proposal_accepts.assign(7, 0);
    for (const auto& attempts : replica_proposal_attempts_) {
      for (std::size_t i = 0; i < metrics.proposal_attempts.size(); ++i) metrics.proposal_attempts[i] += attempts[i];
    }
    for (const auto& accepts : replica_proposal_accepts_) {
      for (std::size_t i = 0; i < metrics.proposal_accepts.size(); ++i) metrics.proposal_accepts[i] += accepts[i];
    }
    metrics.residuals_by_class.assign(3, 0.0);
    metrics.multiplier_mean_by_class.assign(3, 0.0);
    metrics.multiplier_max_by_class.assign(3, 0.0);
    metrics.graph_counts = {
        circuit_.value_count,
        circuit_.opcodes.size(),
        circuit_.constraint_kinds.size(),
        circuit_.wire_offsets.size(),
        replicas_.size(),
        thread_count_,
    };
    metrics.opcode_counts.assign(15, 0);
    for (const int opcode : circuit_.opcodes) {
      if (opcode >= 0 && static_cast<std::size_t>(opcode) < metrics.opcode_counts.size()) {
        ++metrics.opcode_counts[opcode];
      }
    }
    std::vector<std::size_t> multiplier_counts(3, 0);
    if (!hard_scores_.empty()) {
      for (std::size_t i = 0; i < hard_scores_[0].residuals.size(); ++i) {
        metrics.max_residual = std::max(metrics.max_residual, hard_scores_[0].residuals[i]);
        const int cls = i < circuit_.constraint_classes.size() ? circuit_.constraint_classes[i] : 3;
        const std::size_t bucket = BucketForClass(cls);
        metrics.residuals_by_class[bucket] += hard_scores_[0].residuals[i];
        const double multiplier = i < multipliers_.size() ? multipliers_[i] : 1.0;
        metrics.multiplier_mean_by_class[bucket] += multiplier;
        metrics.multiplier_max_by_class[bucket] = std::max(metrics.multiplier_max_by_class[bucket], multiplier);
        ++multiplier_counts[bucket];
      }
    }
    for (std::size_t i = 0; i < multiplier_counts.size(); ++i) {
      if (multiplier_counts[i] != 0) {
        metrics.multiplier_mean_by_class[i] /= static_cast<double>(multiplier_counts[i]);
      }
    }
    metrics.key_visit_count = key_visit_count_;
    metrics.key_distinct_count = key_window_counts_.size();
    metrics.key_ones.assign(key_ones_.begin(), key_ones_.end());
    metrics.key_marginal_max_deviation = KeyMarginalMaxDeviation();
    metrics.marginal_ess = MarginalEss();
    metrics.marginal_rhat = MarginalRhat();
    const std::size_t min_distinct_keys = std::max<std::size_t>(2, config_.marginal_min_distinct_keys);
    // Mixing diagnostics (ESS/R-hat on the cold Hamming-weight trace) only detect
    // whether the chain moved, not whether it moved toward the truth: a replica
    // can be sharply (and wrongly) confident while never visiting a feasible key.
    // Require at least one exact feasible solution (zero hard residual) to have
    // been harvested before declaring the marginals trustworthy, so we no longer
    // report high "information" for a confident-but-infeasible basin.
    metrics.marginal_trusted = metrics.key_visit_count >= 50 && metrics.key_distinct_count >= min_distinct_keys &&
                               metrics.marginal_ess >= 25.0 && metrics.marginal_rhat > 0.0 && metrics.marginal_rhat <= 1.1 &&
                               total_harvested_ > 0;
    metrics.key_information_bits_raw = KeyInformationBits();
    metrics.key_information_null_bits = KeyInformationNullBaselineBits();
    metrics.key_information_bits =
        metrics.marginal_trusted ? std::max(0.0, metrics.key_information_bits_raw - metrics.key_information_null_bits) : 0.0;
    metrics.temperatures = temperatures_;
    metrics.replica_lambda_scale = replica_scale_;
    metrics.replica_round_trips = replica_round_trips_;
    metrics.replica_up_counts = replica_up_counts_;
    metrics.replica_down_counts = replica_down_counts_;
    metrics.total_round_trips = std::accumulate(replica_round_trips_.begin(), replica_round_trips_.end(), std::size_t{0});
    metrics.energy_sample_counts = energy_sample_counts_;
    metrics.energy_mean_by_rung = energy_mean_by_rung_;
    metrics.energy_variance_by_rung.assign(energy_m2_by_rung_.size(), 0.0);
    for (std::size_t i = 0; i < energy_m2_by_rung_.size(); ++i) {
      if (energy_sample_counts_[i] > 1) {
        metrics.energy_variance_by_rung[i] = energy_m2_by_rung_[i] / static_cast<double>(energy_sample_counts_[i] - 1);
      }
    }
    metrics.energy_temperatures_by_rung = temperatures_;
    metrics.rho_by_class.assign(rho_by_class_.begin(), rho_by_class_.end());
    metrics.lambda_update_counts_by_class.assign(lambda_update_counts_by_class_.begin(), lambda_update_counts_by_class_.end());
    metrics.rho_escalation_counts_by_class.assign(rho_escalation_counts_by_class_.begin(), rho_escalation_counts_by_class_.end());
    metrics.algebra_counts = AlgebraCounts();
    metrics.algebra_exact = true;
    metrics.bp_available = false;
    metrics.bp_converged = false;
    metrics.alternative_available = true;
    metrics.langevin_available = true;
    metrics.alternative_log_z_estimates = {LogZEstimate()};
    metrics.langevin_seed_score = relaxed_scores_.empty() ? 0.0 : relaxed_scores_[0].smooth_sum;
    metrics.log_z_estimate_available =
        std::any_of(energy_sample_counts_.begin(), energy_sample_counts_.end(), [](std::size_t count) { return count != 0; });
    metrics.log_z_state_bits = (circuit_.plaintext_count + circuit_.key1_count + circuit_.key2_count) * 8;
    metrics.log_z_estimate = metrics.log_z_estimate_available ? LogZEstimate() : 0.0;
    metrics.profile_seconds = ProfileSeconds();
    metrics.profile_counts = profile_counts_;
    ProfileAdd(PROFILE_METRICS, profile_start);
    return metrics;
  }

  std::vector<AssignmentState> DrainFeasible(std::size_t limit) {
    const uint64_t profile_start = ProfileStart();
    const std::size_t count = limit == 0 ? feasible_.size() : std::min(limit, feasible_.size());
    std::vector<AssignmentState> out(feasible_.begin(), feasible_.begin() + count);
    feasible_.erase(feasible_.begin(), feasible_.begin() + count);
    ProfileAdd(PROFILE_DRAIN_FEASIBLE, profile_start);
    return out;
  }

  AssignmentState CurrentAssignment() const {
    const uint64_t profile_start = ProfileStart();
    if (replicas_.empty()) return AssignmentState{};
    std::size_t best = 0;
    for (std::size_t i = 1; i < energies_.size(); ++i) {
      if (energies_[i] < energies_[best]) best = i;
    }
    AssignmentState assignment = replicas_[best];
    ProfileAdd(PROFILE_CURRENT_ASSIGNMENT, profile_start);
    return assignment;
  }

  AssignmentState DeriveAssignmentWires(const AssignmentState& assignment) const {
    AssignmentState derived = assignment;
    KeyCache cache;
    EvaluateValuesAndDerive(&derived, &cache);
    return derived;
  }

  void SetMultipliers(const std::vector<double>& multipliers) {
    multipliers_ = multipliers;
    if (multipliers_.size() < circuit_.constraint_kinds.size()) multipliers_.resize(circuit_.constraint_kinds.size(), 1.0);
    lambda_previous_ = multipliers_;
    dual_m_.assign(circuit_.constraint_kinds.size(), 0.0);
    dual_v_.assign(circuit_.constraint_kinds.size(), 0.0);
    dual_steps_.assign(circuit_.constraint_kinds.size(), 0);
    RecomputeEnergies();
  }

  void SetTemperatures(const std::vector<double>& temperatures) {
    if (!temperatures.empty()) temperatures_ = temperatures;
    replica_scale_.assign(temperatures_.size(), 1.0);
  }

  void SetConstraintClasses(const std::vector<int32_t>& classes) {
    circuit_.constraint_classes = classes;
    if (circuit_.constraint_classes.size() < circuit_.constraint_kinds.size()) {
      circuit_.constraint_classes.resize(circuit_.constraint_kinds.size(), 3);
    }
    multipliers_ = InitialMultipliers();
    lambda_previous_ = multipliers_;
    RecomputeEnergies();
  }

 private:
  void InitializeCircuit(const std::vector<uint16_t>& value_widths, const std::vector<uint8_t>& xts_block_targets) {
    const std::size_t xts_block_count = circuit_.xts_block_sectors.size();
    if (xts_block_targets.size() >= xts_block_count * 16) {
      circuit_.xts_block_targets.resize(xts_block_count);
      for (std::size_t b = 0; b < xts_block_count; ++b) {
        for (int i = 0; i < 16; ++i) {
          circuit_.xts_block_targets[b][i] = xts_block_targets[b * 16 + i];
        }
      }
    } else {
      circuit_.xts_block_sectors.clear();
      circuit_.xts_block_indices.clear();
    }
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
      const int32_t value = circuit_.wire_value_ids[i];
      const int32_t offset = circuit_.wire_offsets[i];
      if (value >= 0 && static_cast<std::size_t>(value) < circuit_.graph.wire_offset_by_value.size()) {
        circuit_.graph.wire_offset_by_value[value] = offset;
      }
      if (offset >= 0 && static_cast<std::size_t>(offset) < circuit_.graph.wire_value_by_offset.size()) {
        circuit_.graph.wire_value_by_offset[offset] = value;
      }
    }
    circuit_.graph.definitions_by_right.assign(circuit_.value_count, {});
    for (std::size_t i = 0; i < circuit_.constraint_kinds.size(); ++i) {
      if (circuit_.constraint_kinds[i] == CONSTRAINT_DEFINE8 && circuit_.constraint_right[i] >= 0 &&
          static_cast<std::size_t>(circuit_.constraint_right[i]) < circuit_.graph.definitions_by_right.size()) {
        circuit_.graph.definitions_by_right[circuit_.constraint_right[i]].push_back(static_cast<int32_t>(i));
      }
    }
    circuit_.graph.ops_by_input.assign(circuit_.value_count, {});
    std::vector<std::vector<int32_t>> producer_inputs(circuit_.value_count);
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
    linear_determined_wire_.assign(circuit_.wire_offsets.size(), false);
    sbox_input_value_.assign(circuit_.value_count, false);
    for (std::size_t op = 0; op < circuit_.opcodes.size(); ++op) {
      const int32_t output = circuit_.outputs[op];
      if (IsLinearOpcode(circuit_.opcodes[op]) && output >= 0 &&
          static_cast<std::size_t>(output) < circuit_.graph.wire_offset_by_value.size()) {
        const int32_t wire_offset = circuit_.graph.wire_offset_by_value[output];
        if (wire_offset >= 0 && static_cast<std::size_t>(wire_offset) < linear_determined_wire_.size()) {
          linear_determined_wire_[wire_offset] = true;
        }
      }
      if (circuit_.opcodes[op] == OP_SBOX8 && circuit_.input_counts[op] > 0) {
        const int32_t input = circuit_.inputs[circuit_.input_offsets[op]];
        if (input >= 0 && static_cast<std::size_t>(input) < sbox_input_value_.size()) {
          sbox_input_value_[input] = true;
        }
      }
    }
  }

  void InitializeSboxTables() {
    for (int input = 0; input < 256; ++input) {
      for (int target = 0; target < 256; ++target) {
        const int current = Popcount(static_cast<uint8_t>(SBOX[input] ^ target));
        for (int bit = 0; bit < 8; ++bit) {
          const int flipped = input ^ (1 << bit);
          sbox_flip_delta_[input][target][bit] =
              static_cast<int8_t>(Popcount(static_cast<uint8_t>(SBOX[flipped] ^ target)) - current);
        }
      }
    }
  }

  std::vector<double> InitialMultipliers() const {
    std::vector<double> values;
    values.reserve(circuit_.constraint_kinds.size());
    for (const int cls : circuit_.constraint_classes) {
      if (cls == CLASS_ASCII) values.push_back(config_.ascii_weight);
      else if (cls == CLASS_CONSISTENCY) values.push_back(config_.consistency_weight);
      else values.push_back(config_.goal_weight);
    }
    if (values.size() < circuit_.constraint_kinds.size()) values.resize(circuit_.constraint_kinds.size(), config_.goal_weight);
    return values;
  }

  std::vector<double> TemperatureLadder(std::size_t count, double t_min, double t_max) const {
    if (count <= 1) return {std::max(t_min, 1e-9)};
    if (t_min <= 0.0) t_min = 0.5;
    if (t_max < t_min) t_max = t_min;
    const double ratio = std::pow(t_max / t_min, 1.0 / static_cast<double>(count - 1));
    std::vector<double> temperatures(count);
    for (std::size_t i = 0; i < count; ++i) temperatures[i] = t_min * std::pow(ratio, static_cast<double>(i));
    return temperatures;
  }

  void RandomizeSeed(AssignmentState* replica, uint64_t* rng_state) const {
    for (uint8_t& value : replica->plaintext) value = TextAscii(rng_state);
    for (uint8_t& value : replica->key1) value ^= static_cast<uint8_t>(Next(rng_state) & 0xFFu);
    for (uint8_t& value : replica->key2) value ^= static_cast<uint8_t>(Next(rng_state) & 0xFFu);
    if (replica->wires.empty()) return;
    for (uint8_t& value : replica->wires) value = static_cast<uint8_t>(Next(rng_state) & 0xFFu);
  }

  ValueBuffer EvaluateValues(const AssignmentState& assignment, KeyCache* cache) const {
    const uint64_t profile_start = ProfileStart();
    ValueBuffer values = evaluator_.EvaluateValuesInternal(assignment, nullptr, cache);
    ProfileAdd(PROFILE_EVALUATE_VALUES, profile_start);
    return values;
  }

  ValueBuffer EvaluateValuesAndDerive(AssignmentState* assignment, KeyCache* cache) const {
    const uint64_t profile_start = ProfileStart();
    ValueBuffer values = evaluator_.EvaluateValuesInternal(*assignment, assignment, cache);
    ProfileAdd(PROFILE_EVALUATE_VALUES_DERIVE, profile_start);
    return values;
  }

  void EvaluateOpInto(std::size_t op, const AssignmentState& assignment, ValueBuffer* values, KeyCache* cache) const {
    evaluator_.EvaluateOpInto(op, assignment, values, cache);
  }

  void SetValueU8(ValueBuffer* values, int32_t value_id, uint8_t value) const {
    evaluator_.SetValueU8(values, value_id, value);
  }

  void SetValueBytes(ValueBuffer* values, int32_t value_id, const uint8_t* data, std::size_t size) const {
    evaluator_.SetValueBytes(values, value_id, data, size);
  }

  std::vector<uint8_t> CopyValueBytes(const ValueBuffer& values, int32_t value_id) const {
    return evaluator_.CopyValueBytes(values, value_id);
  }

  void ApplyLinearPropagation(AssignmentState* assignment) const {
    KeyCache cache;
    ApplyLinearPropagation(assignment, &cache);
  }

  void ApplyLinearPropagation(AssignmentState* assignment, KeyCache* cache) const {
    if (!assignment->wires.empty()) {
      const uint64_t profile_start = ProfileStart();
      evaluator_.EvaluateValuesInternal(*assignment, assignment, cache);
      ProfileAdd(PROFILE_DERIVE_WIRES, profile_start);
    }
  }

  const std::array<uint8_t, 240>& CachedExpandedKey(const std::vector<uint8_t>& key, int selector, KeyCache* cache) const {
    if (selector == 1) {
      if (cache->key1 != key) {
        cache->key1 = key;
        cache->expanded_key1 = ExpandKey(key);
      }
      return cache->expanded_key1;
    }
    if (cache->key2 != key) {
      cache->key2 = key;
      cache->expanded_key2 = ExpandKey(key);
    }
    return cache->expanded_key2;
  }

  bool RepairCacheMatches(const AssignmentState& assignment, const KeyCache& cache) const {
    return cache.key1 == assignment.key1 && cache.key2 == assignment.key2;
  }

  bool ApplyAdmmRepair(AssignmentState* assignment, KeyCache* cache) const {
    const uint64_t profile_start = ProfileStart();
    if (circuit_.xts_block_sectors.empty() || assignment->key1.size() < 32 || assignment->key2.size() < 32) return false;
    const int rounds = (config_.aes_rounds >= 1 && config_.aes_rounds <= 14) ? config_.aes_rounds : 14;
    const std::array<uint8_t, 240>& expanded1 = CachedExpandedKey(assignment->key1, 1, cache);
    const std::array<uint8_t, 240>& expanded2 = CachedExpandedKey(assignment->key2, 2, cache);
    for (std::size_t block = 0; block < circuit_.xts_block_sectors.size(); ++block) {
      const std::size_t pt_offset = block * 16;
      if (pt_offset + 16 > assignment->plaintext.size()) return false;
      std::array<uint8_t, 16> tweak = EncryptReducedRound(expanded2, SectorTweakInput(circuit_.xts_block_sectors[block]), rounds);
      for (int32_t k = 0; k < circuit_.xts_block_indices[block]; ++k) tweak = XtsMulX(tweak);
      std::array<uint8_t, 16> pre{};
      for (int i = 0; i < 16; ++i) pre[i] = static_cast<uint8_t>(circuit_.xts_block_targets[block][i] ^ tweak[i]);
      const std::array<uint8_t, 16> dec = DecryptReducedRound(expanded1, pre, rounds);
      for (int i = 0; i < 16; ++i) assignment->plaintext[pt_offset + i] = static_cast<uint8_t>(dec[i] ^ tweak[i]);
    }
    ProfileAdd(PROFILE_DERIVE_WIRES, profile_start);
    return true;
  }

  ContinuousRelaxedScore ScoreRelaxed(const ValueBuffer& values) const {
    const uint64_t profile_start = ProfileStart();
    ContinuousRelaxedScore score;
    score.hard = hard_scorer_.ScoreFromValues(values);
    score.residuals.reserve(circuit_.constraint_kinds.size());
    for (std::size_t i = 0; i < circuit_.constraint_kinds.size(); ++i) {
      const double residual = SmoothResidual(values, i);
      score.residuals.push_back(residual);
      score.smooth_sum += residual;
    }
    ProfileAdd(PROFILE_RESIDUAL_SCAN, profile_start);
    return score;
  }

  double SmoothResidual(const ValueBuffer& values, std::size_t i) const {
    const int32_t left_id = circuit_.constraint_left[i];
    if (circuit_.constraint_kinds[i] == CONSTRAINT_ASCII) {
      return static_cast<double>(TextAsciiDistance(evaluator_.ValueU8(values, left_id))) / 8.0;
    }
    const int32_t right_id = circuit_.constraint_right[i];
    if (right_id < 0) return 0.0;
    if (circuit_.constraint_kinds[i] == CONSTRAINT_DEFINE8 || circuit_.constraint_kinds[i] == CONSTRAINT_EQ8) {
      const uint8_t left = evaluator_.ValueU8(values, left_id);
      const uint8_t right = evaluator_.ValueU8(values, right_id);
      return static_cast<double>(Popcount(static_cast<uint8_t>(left ^ right))) / 8.0;
    }
    if (circuit_.constraint_kinds[i] == CONSTRAINT_EQ128) {
      const std::size_t count = std::min(evaluator_.ValueWidth(left_id), evaluator_.ValueWidth(right_id));
      const uint8_t* left = evaluator_.ValuePtr(values, left_id);
      const uint8_t* right = evaluator_.ValuePtr(values, right_id);
      double residual = 0.0;
      for (std::size_t b = 0; b < count; ++b) residual += Popcount(static_cast<uint8_t>(left[b] ^ right[b]));
      return residual / std::max(1.0, 8.0 * static_cast<double>(count));
    }
    return static_cast<double>(score_fallback_zero_);
  }

  double Energy(const ContinuousRelaxedScore& score) const {
    double total = 0.0;
    for (std::size_t i = 0; i < score.residuals.size(); ++i) {
      const double residual = score.residuals[i];
      const double multiplier = i < multipliers_.size() ? multipliers_[i] : 1.0;
      const double penalty = i < penalties_.size() ? penalties_[i] : config_.mu;
      total += multiplier * residual + 0.5 * penalty * residual * residual;
    }
    return total;
  }

  double EffectiveBeta(std::size_t replica_index) const {
    const double temperature = replica_index < temperatures_.size() ? std::max(temperatures_[replica_index], 1e-9) : 1.0;
    const double scale = replica_index < replica_scale_.size() ? replica_scale_[replica_index] : 1.0;
    return scale / temperature;
  }

  void RepairReplicaIfNeeded(std::size_t replica_index) {
    if (replica_index >= replicas_.size() || replica_index >= replica_key_caches_.size()) return;
    if (replica_index < replica_repair_valid_.size() && replica_repair_valid_[replica_index] &&
        RepairCacheMatches(replicas_[replica_index], replica_key_caches_[replica_index])) {
      return;
    }
    const bool repaired = ApplyAdmmRepair(&replicas_[replica_index], &replica_key_caches_[replica_index]);
    if (replica_index < replica_repair_valid_.size()) replica_repair_valid_[replica_index] = repaired;
  }

  void RefreshReplica(std::size_t replica_index) {
    values_[replica_index] = EvaluateValues(replicas_[replica_index], &replica_key_caches_[replica_index]);
    relaxed_scores_[replica_index] = ScoreRelaxed(values_[replica_index]);
    hard_scores_[replica_index] = relaxed_scores_[replica_index].hard;
    energies_[replica_index] = Energy(relaxed_scores_[replica_index]);
    if (hard_scores_[replica_index].hamming_score == 0.0) {
      #ifdef _OPENMP
      #pragma omp critical(feasible_harvest)
      #endif
      MaybeHarvest(replicas_[replica_index], hard_scores_[replica_index]);
    }
  }

  void InitializeAsciiTables() {
    for (int v = 0; v < 256; ++v) {
      inv_sbox_[SBOX[v]] = static_cast<uint8_t>(v);
      ascii_distance_[v] = TextAsciiDistance(static_cast<uint8_t>(v));
    }
    ascii_constraint_at_plain_.assign(circuit_.plaintext_count, -1);
    for (std::size_t ci = 0; ci < circuit_.constraint_kinds.size(); ++ci) {
      if (circuit_.constraint_kinds[ci] != CONSTRAINT_ASCII) continue;
      const long idx = static_cast<long>(circuit_.constraint_left[ci]) - static_cast<long>(circuit_.plaintext_start);
      if (idx >= 0 && static_cast<std::size_t>(idx) < circuit_.plaintext_count) {
        ascii_constraint_at_plain_[static_cast<std::size_t>(idx)] = static_cast<int32_t>(ci);
      }
    }
  }

  double AsciiByteEnergy(int32_t constraint_index, uint8_t value) const {
    const double r = static_cast<double>(ascii_distance_[value]) / 8.0;
    const double mult = (constraint_index >= 0 && static_cast<std::size_t>(constraint_index) < multipliers_.size())
                            ? multipliers_[constraint_index]
                            : config_.ascii_weight;
    const double pen = (constraint_index >= 0 && static_cast<std::size_t>(constraint_index) < penalties_.size())
                           ? penalties_[constraint_index]
                           : std::max(config_.mu, 1e-9);
    return mult * r + 0.5 * pen * r * r;
  }

  // After ApplyAdmmRepair the plaintext is the exact decryption of the current
  // key, so the goal and consistency residuals are zero and the full Lagrangian
  // energy equals the ASCII-printability energy. Reading it straight from the
  // plaintext avoids re-deriving every wire, which makes the 256-way key Gibbs
  // scans roughly an order of magnitude cheaper than a full circuit evaluation.
  double AsciiEnergyFromPlaintext(const AssignmentState& assignment) const {
    double total = 0.0;
    const std::size_t n = std::min(assignment.plaintext.size(), circuit_.plaintext_count);
    for (std::size_t idx = 0; idx < n; ++idx) {
      const int32_t ci = idx < ascii_constraint_at_plain_.size() ? ascii_constraint_at_plain_[idx] : -1;
      if (ci < 0) continue;
      total += AsciiByteEnergy(ci, assignment.plaintext[idx]);
    }
    return total;
  }

  double RepairedAsciiEnergy(AssignmentState* assignment, KeyCache* cache) const {
    ApplyAdmmRepair(assignment, cache);
    return AsciiEnergyFromPlaintext(*assignment);
  }

  void InvalidateKey1Profile(KeyProfileCache* profile) const {
    profile->valid.fill(0);
  }

  bool Key1ProfileUsable(const AssignmentState& assignment) const {
    return !circuit_.xts_block_sectors.empty() && assignment.key1.size() >= 32 && assignment.key2.size() >= 32;
  }

  void RefreshKey1ProfileRow(
      std::size_t replica_index,
      std::size_t row,
      AssignmentState* assignment,
      KeyProfileCache* profile) const {
    if (row >= 32 || !Key1ProfileUsable(*assignment)) return;
    if (profile->cached_key2 != assignment->key2 || profile->cached_key1.size() != assignment->key1.size()) {
      profile->cached_key1 = assignment->key1;
      profile->cached_key2 = assignment->key2;
      InvalidateKey1Profile(profile);
    }
    const uint8_t original = assignment->key1[row];
    KeyCache scratch = replica_index < replica_key_caches_.size() ? replica_key_caches_[replica_index] : KeyCache{};
    for (int value = 0; value < 256; ++value) {
      assignment->key1[row] = static_cast<uint8_t>(value);
      profile->energy[row][static_cast<std::size_t>(value)] = RepairedAsciiEnergy(assignment, &scratch);
    }
    assignment->key1[row] = original;
    profile->cached_key1 = assignment->key1;
    profile->cached_key2 = assignment->key2;
    profile->valid[row] = 1;
  }

  double ProfileKey1ForKey2MultiRound(
      AssignmentState* assignment,
      std::size_t replica_index,
      KeyProfileCache* profile) const {
    if (!Key1ProfileUsable(*assignment)) return std::numeric_limits<double>::infinity();
    static const int sweeps = []() {
      const char* e = std::getenv("CR_PROFILE_K1_SWEEPS");
      return std::max(1, e ? std::atoi(e) : 1);
    }();
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      for (std::size_t row = 0; row < 32; ++row) {
        RefreshKey1ProfileRow(replica_index, row, assignment, profile);
        const auto& row_energy = profile->energy[row];
        std::size_t best_value = 0;
        double best = row_energy[0];
        for (std::size_t value = 1; value < row_energy.size(); ++value) {
          if (row_energy[value] < best) {
            best = row_energy[value];
            best_value = value;
          }
        }
        const uint8_t chosen = static_cast<uint8_t>(best_value);
        if (assignment->key1[row] != chosen) {
          assignment->key1[row] = chosen;
          InvalidateKey1Profile(profile);
        }
      }
    }
    KeyCache scratch = replica_index < replica_key_caches_.size() ? replica_key_caches_[replica_index] : KeyCache{};
    const double energy = RepairedAsciiEnergy(assignment, &scratch);
    profile->cached_key1 = assignment->key1;
    profile->cached_key2 = assignment->key2;
    InvalidateKey1Profile(profile);
    return energy;
  }

  bool Key1ProfileGibbsStep(std::size_t replica_index) {
    if (replica_index >= replicas_.size() || replica_index >= replica_key1_profile_.size()) return false;
    AssignmentState& base = replicas_[replica_index];
    if (!Key1ProfileUsable(base)) return false;
    uint64_t* rng = &replica_rng_states_[replica_index];
    const std::size_t row = RandRange(rng, 32);
    KeyProfileCache& profile = replica_key1_profile_[replica_index];
    RefreshKey1ProfileRow(replica_index, row, &base, &profile);
    const auto& row_energy = profile.energy[row];
    const double min_energy = *std::min_element(row_energy.begin(), row_energy.end());
    const double beta = EffectiveBeta(replica_index);
    std::array<double, 256> weight{};
    double total = 0.0;
    for (int value = 0; value < 256; ++value) {
      weight[static_cast<std::size_t>(value)] =
          std::exp(ClampExpArg(-beta * (row_energy[static_cast<std::size_t>(value)] - min_energy)));
      total += weight[static_cast<std::size_t>(value)];
    }
    if (!(total > 0.0)) return false;
    const uint8_t original = base.key1[row];
    const double threshold = Uniform(rng) * total;
    double cumulative = 0.0;
    std::size_t chosen = 255;
    for (std::size_t value = 0; value < weight.size(); ++value) {
      cumulative += weight[value];
      if (cumulative >= threshold) {
        chosen = value;
        break;
      }
    }
    base.key1[row] = static_cast<uint8_t>(chosen);
    InvalidateKey1Profile(&profile);

    const bool repaired = ApplyAdmmRepair(&base, &replica_key_caches_[replica_index]);
    values_[replica_index] = EvaluateValuesAndDerive(&base, &replica_key_caches_[replica_index]);
    relaxed_scores_[replica_index] = ScoreRelaxed(values_[replica_index]);
    hard_scores_[replica_index] = relaxed_scores_[replica_index].hard;
    energies_[replica_index] = Energy(relaxed_scores_[replica_index]);
    if (replica_index < replica_repair_valid_.size()) replica_repair_valid_[replica_index] = repaired;

    ++replica_proposal_attempts_[replica_index][6];
    if (base.key1[row] != original) ++replica_proposal_accepts_[replica_index][6];
    RecordAcceptedKeyStats(replica_index);
    if (hard_scores_[replica_index].hamming_score == 0.0) {
      #ifdef _OPENMP
      #pragma omp critical(feasible_harvest)
      #endif
      MaybeHarvest(base, hard_scores_[replica_index]);
    }
    return true;
  }

  bool KeyTweakProfileGibbsStep(std::size_t replica_index) {
    if (replica_index >= replicas_.size()) return false;
    AssignmentState& base = replicas_[replica_index];
    if (!Key1ProfileUsable(base) || std::getenv("CR_FREEZE_KEY2") != nullptr) return false;
    uint64_t* rng = &replica_rng_states_[replica_index];
    const std::size_t index = RandRange(rng, 32);
    const uint8_t original = base.key2[index];
    std::array<double, 256> energy{};
    std::array<std::array<uint8_t, 32>, 256> best_key1{};
    for (int value = 0; value < 256; ++value) {
      AssignmentState candidate = base;
      candidate.key2[index] = static_cast<uint8_t>(value);
      KeyProfileCache profile;
      energy[static_cast<std::size_t>(value)] = ProfileKey1ForKey2MultiRound(&candidate, replica_index, &profile);
      for (std::size_t byte = 0; byte < 32; ++byte) {
        best_key1[static_cast<std::size_t>(value)][byte] = candidate.key1[byte];
      }
    }
    const double min_energy = *std::min_element(energy.begin(), energy.end());
    if (!std::isfinite(min_energy)) return false;
    const double beta = EffectiveBeta(replica_index);
    std::array<double, 256> weight{};
    double total = 0.0;
    for (int value = 0; value < 256; ++value) {
      weight[static_cast<std::size_t>(value)] =
          std::exp(ClampExpArg(-beta * (energy[static_cast<std::size_t>(value)] - min_energy)));
      total += weight[static_cast<std::size_t>(value)];
    }
    if (!(total > 0.0)) return false;
    const double threshold = Uniform(rng) * total;
    double cumulative = 0.0;
    std::size_t chosen = 255;
    for (std::size_t value = 0; value < weight.size(); ++value) {
      cumulative += weight[value];
      if (cumulative >= threshold) {
        chosen = value;
        break;
      }
    }
    for (std::size_t byte = 0; byte < 32; ++byte) base.key1[byte] = best_key1[chosen][byte];
    base.key2[index] = static_cast<uint8_t>(chosen);
    if (replica_index < replica_key1_profile_.size()) InvalidateKey1Profile(&replica_key1_profile_[replica_index]);

    const bool repaired = ApplyAdmmRepair(&base, &replica_key_caches_[replica_index]);
    values_[replica_index] = EvaluateValuesAndDerive(&base, &replica_key_caches_[replica_index]);
    relaxed_scores_[replica_index] = ScoreRelaxed(values_[replica_index]);
    hard_scores_[replica_index] = relaxed_scores_[replica_index].hard;
    energies_[replica_index] = Energy(relaxed_scores_[replica_index]);
    if (replica_index < replica_repair_valid_.size()) replica_repair_valid_[replica_index] = repaired;

    ++replica_proposal_attempts_[replica_index][6];
    if (base.key2[index] != original) ++replica_proposal_accepts_[replica_index][6];
    RecordAcceptedKeyStats(replica_index);
    if (hard_scores_[replica_index].hamming_score == 0.0) {
      #ifdef _OPENMP
      #pragma omp critical(feasible_harvest)
      #endif
      MaybeHarvest(base, hard_scores_[replica_index]);
    }
    return true;
  }

  // Joint (block) Gibbs update over the pair of key1 bytes that co-determine one
  // plaintext-byte position under 1-round AES-XTS decryption:
  //   P_b[dst] = InvSbox[ C_b[s] ^ T_b[s] ^ rk1[s] ] ^ rk0[dst] ^ T_b[dst],
  // where s = kInvShiftSrc[dst] and (for AES-256 round keys 0/1 are the raw key)
  // rk0[dst] = key1[dst], rk1[s] = key1[16+s]. Changing this pair only moves
  // P_b[dst] across blocks, so resampling it from the exact ASCII conditional is
  // an exact Gibbs step. Single-byte moves cannot satisfy the S-box-coupled pair
  // jointly (which is why bit/byte flips merely random-walk); this is the move
  // that actually descends to a feasible one-round key.
  bool KeyColumnPairBlockStep(std::size_t replica_index) {
    if (config_.aes_rounds != 1) return false;
    const std::size_t window = circuit_.xts_block_sectors.size();
    if (window == 0 || replica_index >= replicas_.size()) return false;
    AssignmentState& base = replicas_[replica_index];
    if (base.key1.size() < 32 || base.key2.size() < 32) return false;
    if (replica_index >= replica_pair_energy_.size() ||
        replica_pair_energy_[replica_index].size() < (static_cast<std::size_t>(1) << 16)) {
      return false;
    }
    uint64_t* rng = &replica_rng_states_[replica_index];

    static constexpr std::array<int, 16> kInvShiftSrc = {0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3};
    const int dst = static_cast<int>(RandRange(rng, 16));
    const int s = kInvShiftSrc[static_cast<std::size_t>(dst)];
    const std::size_t a_idx = static_cast<std::size_t>(dst);
    const std::size_t c_idx = 16u + static_cast<std::size_t>(s);

    const std::array<uint8_t, 240>& expanded2 = CachedExpandedKey(base.key2, 2, &replica_key_caches_[replica_index]);

    std::vector<uint8_t> pre_src(window);
    std::vector<uint8_t> tdst(window);
    std::vector<double> mult(window);
    std::vector<double> pen(window);
    for (std::size_t b = 0; b < window; ++b) {
      std::array<uint8_t, 16> tweak =
          EncryptReducedRound(expanded2, SectorTweakInput(circuit_.xts_block_sectors[b]), 1);
      for (int32_t k = 0; k < circuit_.xts_block_indices[b]; ++k) tweak = XtsMulX(tweak);
      pre_src[b] = static_cast<uint8_t>(circuit_.xts_block_targets[b][static_cast<std::size_t>(s)] ^
                                        tweak[static_cast<std::size_t>(s)]);
      tdst[b] = tweak[static_cast<std::size_t>(dst)];
      const std::size_t plain_idx = b * 16u + static_cast<std::size_t>(dst);
      const int32_t ci = plain_idx < ascii_constraint_at_plain_.size() ? ascii_constraint_at_plain_[plain_idx] : -1;
      mult[b] = (ci >= 0 && static_cast<std::size_t>(ci) < multipliers_.size()) ? multipliers_[ci] : config_.ascii_weight;
      pen[b] = (ci >= 0 && static_cast<std::size_t>(ci) < penalties_.size()) ? penalties_[ci] : std::max(config_.mu, 1e-9);
    }

    std::vector<double>& buf = replica_pair_energy_[replica_index];
    std::vector<uint8_t> col_base(window);
    double emin = std::numeric_limits<double>::infinity();
    for (int c = 0; c < 256; ++c) {
      for (std::size_t b = 0; b < window; ++b) {
        col_base[b] = static_cast<uint8_t>(inv_sbox_[static_cast<uint8_t>(pre_src[b] ^ c)] ^ tdst[b]);
      }
      double* row = &buf[static_cast<std::size_t>(c) << 8];
      for (int a = 0; a < 256; ++a) {
        double e = 0.0;
        for (std::size_t b = 0; b < window; ++b) {
          const double r = static_cast<double>(ascii_distance_[static_cast<uint8_t>(col_base[b] ^ a)]) / 8.0;
          e += mult[b] * r + 0.5 * pen[b] * r * r;
        }
        row[a] = e;
        if (e < emin) emin = e;
      }
    }

    const double beta = EffectiveBeta(replica_index);
    const std::size_t cells = static_cast<std::size_t>(1) << 16;
    // Greedy-polish fraction: with probability greedy_prob take the exact argmin
    // pair for this position instead of a Boltzmann draw. Pure Gibbs concentration
    // is capped by the cumulative weight of the many near-printable (but
    // infeasible) pairs, so even at large beta a position is left non-printable a
    // few % of the time; revisited 16x that leaves a residual ASCII plateau. The
    // 16 positions use disjoint key1 bytes, so greedily fixing each is the exact
    // conditional argmin and drives the key1-given-key2 sub-problem to zero.
    static const double greedy_prob = []() {
      const char* e = std::getenv("CR_BLOCK_GREEDY");
      return e ? std::atof(e) : 0.25;
    }();
    std::size_t chosen = cells - 1;
    if (Uniform(rng) < greedy_prob) {
      double best = std::numeric_limits<double>::infinity();
      for (std::size_t idx = 0; idx < cells; ++idx) {
        if (buf[idx] < best) { best = buf[idx]; chosen = idx; }
      }
    } else {
      double total = 0.0;
      for (std::size_t idx = 0; idx < cells; ++idx) {
        const double w = std::exp(ClampExpArg(-beta * (buf[idx] - emin)));
        buf[idx] = w;
        total += w;
      }
      if (!(total > 0.0)) return false;
      const double threshold = Uniform(rng) * total;
      double cumulative = 0.0;
      for (std::size_t idx = 0; idx < cells; ++idx) {
        cumulative += buf[idx];
        if (cumulative >= threshold) {
          chosen = idx;
          break;
        }
      }
    }
    base.key1[a_idx] = static_cast<uint8_t>(chosen & 0xFF);
    base.key1[c_idx] = static_cast<uint8_t>((chosen >> 8) & 0xFF);

    const bool repaired = ApplyAdmmRepair(&base, &replica_key_caches_[replica_index]);
    values_[replica_index] = EvaluateValuesAndDerive(&base, &replica_key_caches_[replica_index]);
    relaxed_scores_[replica_index] = ScoreRelaxed(values_[replica_index]);
    hard_scores_[replica_index] = relaxed_scores_[replica_index].hard;
    energies_[replica_index] = Energy(relaxed_scores_[replica_index]);
    if (replica_index < replica_repair_valid_.size()) replica_repair_valid_[replica_index] = repaired;

    ++replica_proposal_attempts_[replica_index][2];
    ++replica_proposal_accepts_[replica_index][2];
    RecordAcceptedKeyStats(replica_index);
    if (std::getenv("CR_DEBUG") != nullptr && replica_index == 0) {
      static thread_local long dbg_calls = 0;
      if ((dbg_calls++ % 400) == 0) {
        int n_ascii = 0, n_def = 0, n_eq128 = 0, n_eq8 = 0;
        for (uint32_t fi : hard_scores_[replica_index].failing_indices) {
          switch (circuit_.constraint_kinds[fi]) {
            case CONSTRAINT_ASCII: ++n_ascii; break;
            case CONSTRAINT_DEFINE8: ++n_def; break;
            case CONSTRAINT_EQ128: ++n_eq128; break;
            case CONSTRAINT_EQ8: ++n_eq8; break;
          }
        }
        std::fprintf(stderr, "[blkmove] r=0 beta=%.3f dst=%d emin=%.5f hard=%.1f viol=%u ascii=%d def=%d eq128=%d eq8=%d\n",
                     beta, dst, emin, hard_scores_[replica_index].hamming_score,
                     hard_scores_[replica_index].violations, n_ascii, n_def, n_eq128, n_eq8);
      }
    }
    if (hard_scores_[replica_index].hamming_score == 0.0) {
      #ifdef _OPENMP
      #pragma omp critical(feasible_harvest)
      #endif
      MaybeHarvest(base, hard_scores_[replica_index]);
    }
    return true;
  }

  // Joint (block) Gibbs update over the pair of key2 bytes that co-determine one
  // base-tweak byte under 1-round AES-XTS *encryption* -- the exact key2 mirror of
  // KeyColumnPairBlockStep:
  //   base_tweak_sec[dst2] = SBOX[ sector_input_sec[fs] ^ rk0_2[fs] ] ^ rk1_2[dst2],
  // with fs = kFwdShiftSrc[dst2], rk0_2[fs] = key2[fs], rk1_2[dst2] = key2[16+dst2].
  // The 16 dst2 positions use disjoint key2 bytes, just like key1. The twist is that
  // the controlled quantity is the per-sector base tweak, not a plaintext byte:
  // changing base_tweak[dst2] propagates through XtsMulX to every block of the sector
  // and through the key1 decryption to all 16 of that block's plaintext bytes. So we
  // first tabulate, per block, the ASCII energy as a function of that one tweak byte
  // v in 0..255 (key1 and the other tweak bytes fixed; mul_x is GF(2)-linear, so v's
  // image is T0 ^ a XOR of per-bit basis images), then scan the 256x256
  // (rk0_2[fs], rk1_2[dst2]) pair. Because the SAME key2 pair maps to a DIFFERENT
  // base-tweak byte in each sector (distinct sector_input), a multi-sector window
  // couples the 65536 cells and the move can discriminate the true key2; a
  // single-sector window is a pure XOR offset (degenerate with key1) and carries no
  // key2 information.
  bool KeyTweakColumnPairBlockStep(std::size_t replica_index) {
    if (config_.aes_rounds != 1) return false;
    const std::size_t window = circuit_.xts_block_sectors.size();
    if (window == 0 || replica_index >= replicas_.size()) return false;
    AssignmentState& base = replicas_[replica_index];
    if (base.key1.size() < 32 || base.key2.size() < 32) return false;
    if (replica_index >= replica_pair_energy_.size() ||
        replica_pair_energy_[replica_index].size() < (static_cast<std::size_t>(1) << 16)) {
      return false;
    }
    uint64_t* rng = &replica_rng_states_[replica_index];

    static constexpr std::array<int, 16> kInvShiftSrc = {0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3};
    static constexpr std::array<int, 16> kFwdShiftSrc = {0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11};
    const int dst2 = static_cast<int>(RandRange(rng, 16));
    const int fs = kFwdShiftSrc[static_cast<std::size_t>(dst2)];
    const std::size_t c_idx = static_cast<std::size_t>(fs);          // rk0_2[fs] = key2[fs]
    const std::size_t a_idx = 16u + static_cast<std::size_t>(dst2);  // rk1_2[dst2] = key2[16+dst2]

    const std::array<uint8_t, 240>& expanded1 = CachedExpandedKey(base.key1, 1, &replica_key_caches_[replica_index]);
    const std::array<uint8_t, 240>& expanded2 = CachedExpandedKey(base.key2, 2, &replica_key_caches_[replica_index]);
    const uint8_t* rk0_1 = expanded1.data();
    const uint8_t* rk1_1 = expanded1.data() + 16;

    // Distinct sectors in the window (typically 1-3); map each block to its sector slot.
    std::vector<int32_t> sec_ids;
    std::vector<std::size_t> block_sec(window);
    for (std::size_t b = 0; b < window; ++b) {
      const int32_t sec = circuit_.xts_block_sectors[b];
      std::size_t slot = 0;
      for (; slot < sec_ids.size(); ++slot) {
        if (sec_ids[slot] == sec) break;
      }
      if (slot == sec_ids.size()) sec_ids.push_back(sec);
      block_sec[b] = slot;
    }
    // Per-sector current base tweak and the forward-sbox term, so that
    // base_tweak_sec[dst2](c2,a2) = SBOX[sector_input_sec[fs] ^ c2] ^ a2.
    std::vector<std::array<uint8_t, 16>> base_tweak(sec_ids.size());
    std::vector<std::array<uint8_t, 256>> sbox_tbl(sec_ids.size());
    for (std::size_t si = 0; si < sec_ids.size(); ++si) {
      const std::array<uint8_t, 16> sector_input = SectorTweakInput(sec_ids[si]);
      base_tweak[si] = EncryptReducedRound(expanded2, sector_input, 1);
      const uint8_t input_fs = sector_input[static_cast<std::size_t>(fs)];
      for (int c2 = 0; c2 < 256; ++c2) {
        sbox_tbl[si][static_cast<std::size_t>(c2)] = SBOX[static_cast<uint8_t>(input_fs ^ c2)];
      }
    }

    // Per-block ASCII energy as a function of v = base_tweak_sector[dst2] in 0..255.
    std::vector<std::array<double, 256>> energy_by_block(window);
    for (std::size_t b = 0; b < window; ++b) {
      const int32_t bidx = circuit_.xts_block_indices[b];
      const std::array<uint8_t, 16>& tgt = circuit_.xts_block_targets[b];
      std::array<int32_t, 16> ci{};
      for (int dst = 0; dst < 16; ++dst) {
        const std::size_t plain_idx = b * 16u + static_cast<std::size_t>(dst);
        ci[static_cast<std::size_t>(dst)] =
            plain_idx < ascii_constraint_at_plain_.size() ? ascii_constraint_at_plain_[plain_idx] : -1;
      }
      // mul_x is GF(2)-linear, so T(v) = mul_x^bidx(base with byte dst2 = v)
      //   = mul_x^bidx(base|dst2=0) ^ (XOR over set bits i of v of mul_x^bidx(2^i at dst2)).
      std::array<uint8_t, 16> t0 = base_tweak[block_sec[b]];
      t0[static_cast<std::size_t>(dst2)] = 0;
      for (int32_t k = 0; k < bidx; ++k) t0 = XtsMulX(t0);
      std::array<std::array<uint8_t, 16>, 8> basis{};
      for (int bit = 0; bit < 8; ++bit) {
        std::array<uint8_t, 16> e{};
        e[static_cast<std::size_t>(dst2)] = static_cast<uint8_t>(1u << bit);
        for (int32_t k = 0; k < bidx; ++k) e = XtsMulX(e);
        basis[static_cast<std::size_t>(bit)] = e;
      }
      std::array<double, 256>& eb = energy_by_block[b];
      for (int v = 0; v < 256; ++v) {
        std::array<uint8_t, 16> T = t0;
        for (int bit = 0; bit < 8; ++bit) {
          if ((v >> bit) & 1) {
            const std::array<uint8_t, 16>& bvec = basis[static_cast<std::size_t>(bit)];
            for (int i = 0; i < 16; ++i) T[static_cast<std::size_t>(i)] ^= bvec[static_cast<std::size_t>(i)];
          }
        }
        double e = 0.0;
        for (int dst = 0; dst < 16; ++dst) {
          const int s = kInvShiftSrc[static_cast<std::size_t>(dst)];
          const uint8_t pt = static_cast<uint8_t>(
              inv_sbox_[static_cast<uint8_t>(tgt[static_cast<std::size_t>(s)] ^ T[static_cast<std::size_t>(s)] ^
                                             rk1_1[s])] ^
              rk0_1[dst] ^ T[static_cast<std::size_t>(dst)]);
          e += AsciiByteEnergy(ci[static_cast<std::size_t>(dst)], pt);
        }
        eb[static_cast<std::size_t>(v)] = e;
      }
    }

    // Scan the 256x256 key2 pair; energy(c2,a2) = sum_b energy_by_block[b][ v_b(c2,a2) ].
    std::vector<double>& buf = replica_pair_energy_[replica_index];
    double emin = std::numeric_limits<double>::infinity();
    for (int c2 = 0; c2 < 256; ++c2) {
      double* row = &buf[static_cast<std::size_t>(c2) << 8];
      for (int a2 = 0; a2 < 256; ++a2) {
        double e = 0.0;
        for (std::size_t b = 0; b < window; ++b) {
          const uint8_t v = static_cast<uint8_t>(sbox_tbl[block_sec[b]][static_cast<std::size_t>(c2)] ^ a2);
          e += energy_by_block[b][static_cast<std::size_t>(v)];
        }
        row[static_cast<std::size_t>(a2)] = e;
        if (e < emin) emin = e;
      }
    }

    const double beta = EffectiveBeta(replica_index);
    const std::size_t cells = static_cast<std::size_t>(1) << 16;
    static const double greedy_prob = []() {
      const char* e = std::getenv("CR_BLOCK_GREEDY");
      return e ? std::atof(e) : 0.25;
    }();
    std::size_t chosen = cells - 1;
    if (Uniform(rng) < greedy_prob) {
      double best = std::numeric_limits<double>::infinity();
      for (std::size_t idx = 0; idx < cells; ++idx) {
        if (buf[idx] < best) {
          best = buf[idx];
          chosen = idx;
        }
      }
    } else {
      double total = 0.0;
      for (std::size_t idx = 0; idx < cells; ++idx) {
        const double w = std::exp(ClampExpArg(-beta * (buf[idx] - emin)));
        buf[idx] = w;
        total += w;
      }
      if (!(total > 0.0)) return false;
      const double threshold = Uniform(rng) * total;
      double cumulative = 0.0;
      for (std::size_t idx = 0; idx < cells; ++idx) {
        cumulative += buf[idx];
        if (cumulative >= threshold) {
          chosen = idx;
          break;
        }
      }
    }
    base.key2[a_idx] = static_cast<uint8_t>(chosen & 0xFF);
    base.key2[c_idx] = static_cast<uint8_t>((chosen >> 8) & 0xFF);

    const bool repaired = ApplyAdmmRepair(&base, &replica_key_caches_[replica_index]);
    values_[replica_index] = EvaluateValuesAndDerive(&base, &replica_key_caches_[replica_index]);
    relaxed_scores_[replica_index] = ScoreRelaxed(values_[replica_index]);
    hard_scores_[replica_index] = relaxed_scores_[replica_index].hard;
    energies_[replica_index] = Energy(relaxed_scores_[replica_index]);
    if (replica_index < replica_repair_valid_.size()) replica_repair_valid_[replica_index] = repaired;

    ++replica_proposal_attempts_[replica_index][2];
    ++replica_proposal_accepts_[replica_index][2];
    RecordAcceptedKeyStats(replica_index);
    if (hard_scores_[replica_index].hamming_score == 0.0) {
      #ifdef _OPENMP
      #pragma omp critical(feasible_harvest)
      #endif
      MaybeHarvest(base, hard_scores_[replica_index]);
    }
    return true;
  }

  // Greedily set key1 to its per-position ASCII-optimal pair for the current
  // key2 (i.e. the conditional argmin of every (rk0[dst], rk1[src]) pair) and
  // return the resulting total ASCII energy. Because the 16 positions use
  // disjoint key1 bytes, independent per-position minimisation is the exact
  // global argmin over key1 for the given key2 -- this is the "profile energy"
  // E*(key2) = min_{key1} E(key1, key2).
  double GreedyKey1ForKey2(AssignmentState* base, std::size_t replica_index) {
    const std::size_t window = circuit_.xts_block_sectors.size();
    static constexpr std::array<int, 16> kInvShiftSrc = {0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3};
    const std::array<uint8_t, 240>& expanded2 = CachedExpandedKey(base->key2, 2, &replica_key_caches_[replica_index]);
    std::vector<std::array<uint8_t, 16>> tweaks(window);
    for (std::size_t b = 0; b < window; ++b) {
      std::array<uint8_t, 16> tweak =
          EncryptReducedRound(expanded2, SectorTweakInput(circuit_.xts_block_sectors[b]), 1);
      for (int32_t k = 0; k < circuit_.xts_block_indices[b]; ++k) tweak = XtsMulX(tweak);
      tweaks[b] = tweak;
    }
    std::vector<uint8_t> pre_src(window);
    std::vector<uint8_t> tdst(window);
    std::vector<double> mult(window);
    std::vector<double> pen(window);
    std::vector<uint8_t> col_base(window);
    double total_min = 0.0;
    for (int dst = 0; dst < 16; ++dst) {
      const int s = kInvShiftSrc[static_cast<std::size_t>(dst)];
      // pre_src and the per-block ASCII weights depend only on (block, dst), not on
      // the candidate key bytes (a, c). Hoisting them out of the 256x256 scan (and
      // inlining the energy that AsciiByteEnergy computes) removes ~window*65536
      // redundant index lookups, bound checks, and calls per position -- the bulk
      // of the refit cost -- while keeping the arithmetic bit-for-bit identical.
      for (std::size_t b = 0; b < window; ++b) {
        pre_src[b] = static_cast<uint8_t>(circuit_.xts_block_targets[b][static_cast<std::size_t>(s)] ^
                                          tweaks[b][static_cast<std::size_t>(s)]);
        tdst[b] = tweaks[b][static_cast<std::size_t>(dst)];
        const std::size_t plain_idx = b * 16u + static_cast<std::size_t>(dst);
        const int32_t ci = plain_idx < ascii_constraint_at_plain_.size() ? ascii_constraint_at_plain_[plain_idx] : -1;
        mult[b] = (ci >= 0 && static_cast<std::size_t>(ci) < multipliers_.size()) ? multipliers_[ci] : config_.ascii_weight;
        pen[b] = (ci >= 0 && static_cast<std::size_t>(ci) < penalties_.size()) ? penalties_[ci] : std::max(config_.mu, 1e-9);
      }
      double best = std::numeric_limits<double>::infinity();
      int best_a = base->key1[static_cast<std::size_t>(dst)];
      int best_c = base->key1[16u + static_cast<std::size_t>(s)];
      for (int c = 0; c < 256; ++c) {
        for (std::size_t b = 0; b < window; ++b) {
          col_base[b] = static_cast<uint8_t>(inv_sbox_[static_cast<uint8_t>(pre_src[b] ^ c)] ^ tdst[b]);
        }
        for (int a = 0; a < 256; ++a) {
          double e = 0.0;
          for (std::size_t b = 0; b < window; ++b) {
            const double r = static_cast<double>(ascii_distance_[static_cast<uint8_t>(col_base[b] ^ a)]) / 8.0;
            e += mult[b] * r + 0.5 * pen[b] * r * r;
          }
          if (e < best) {
            best = e;
            best_a = a;
            best_c = c;
          }
        }
      }
      base->key1[static_cast<std::size_t>(dst)] = static_cast<uint8_t>(best_a);
      base->key1[16u + static_cast<std::size_t>(s)] = static_cast<uint8_t>(best_c);
      total_min += best;
    }
    return total_min;
  }

  // Composite key2 move that breaks the key1<->key2 coupling. A bare key2-byte
  // Gibbs step is scored against a key1 that was fitted to the *old* tweaks, so
  // every key2 change looks catastrophic and is rejected -- the chain can never
  // explore key2. Here we perturb a key2 byte (which changes the per-sector
  // tweak T0 and hence every plaintext byte) and re-fit key1 to its optimum for
  // the proposed tweaks, then accept/reject on the profile energy E*(key2). This
  // is what lets the sampler discover a key2 whose tweaks admit an all-printable
  // decryption -- the prerequisite for harvesting a feasible one-round key.
  bool KeyTweakRefitStep(std::size_t replica_index) {
    if (config_.aes_rounds != 1) return false;
    if (circuit_.xts_block_sectors.empty() || replica_index >= replicas_.size()) return false;
    AssignmentState& base = replicas_[replica_index];
    if (base.key1.size() < 32 || base.key2.size() < 32) return false;
    if (std::getenv("CR_FREEZE_KEY2") != nullptr) return false;
    // The profile refit costs ~16x a block move (a full key1 re-fit per key2
    // proposal), so spend it only on the cold replicas, where feasible solutions
    // are actually harvested; hot replicas keep mixing with the cheap moves.
    static const double cold_ratio = []() {
      const char* e = std::getenv("CR_REFIT_COLD_RATIO");
      return e ? std::atof(e) : 4.0;
    }();
    const double temp = replica_index < temperatures_.size() ? temperatures_[replica_index] : 1.0;
    if (temp > config_.t_min * cold_ratio) return false;
    uint64_t* rng = &replica_rng_states_[replica_index];

    const double e_cur = GreedyKey1ForKey2(&base, replica_index);
    std::vector<uint8_t> saved_key1 = base.key1;
    std::vector<uint8_t> saved_key2 = base.key2;

    const std::size_t changes = 1 + RandRange(rng, 2);
    for (std::size_t k = 0; k < changes; ++k) {
      base.key2[RandRange(rng, base.key2.size())] = static_cast<uint8_t>(RandRange(rng, 256));
    }
    const double e_new = GreedyKey1ForKey2(&base, replica_index);
    const double beta = EffectiveBeta(replica_index);
    const bool accept = e_new <= e_cur || Uniform(rng) < std::exp(ClampExpArg(-beta * (e_new - e_cur)));
    if (!accept) {
      base.key1 = saved_key1;
      base.key2 = saved_key2;
    }

    const bool repaired = ApplyAdmmRepair(&base, &replica_key_caches_[replica_index]);
    values_[replica_index] = EvaluateValuesAndDerive(&base, &replica_key_caches_[replica_index]);
    relaxed_scores_[replica_index] = ScoreRelaxed(values_[replica_index]);
    hard_scores_[replica_index] = relaxed_scores_[replica_index].hard;
    energies_[replica_index] = Energy(relaxed_scores_[replica_index]);
    if (replica_index < replica_repair_valid_.size()) replica_repair_valid_[replica_index] = repaired;

    ++replica_proposal_attempts_[replica_index][2];
    if (accept) ++replica_proposal_accepts_[replica_index][2];
    RecordAcceptedKeyStats(replica_index);
    if (hard_scores_[replica_index].hamming_score == 0.0) {
      #ifdef _OPENMP
      #pragma omp critical(feasible_harvest)
      #endif
      MaybeHarvest(base, hard_scores_[replica_index]);
    }
    return true;
  }

  // Byte-wise Gibbs update on one key byte. Unlike single-bit key flips (whose
  // avalanche through the S-box makes the post-repair ASCII residual behave like
  // noise), this scores all 256 candidate values for the chosen byte and samples
  // from the exact conditional at the replica's temperature. For reduced rounds a
  // key byte only touches a few plaintext bytes, so the conditional is sharply
  // informative once several blocks share the key.
  bool KeyByteGibbsStep(std::size_t replica_index) {
    AssignmentState& base = replicas_[replica_index];
    const bool have_key1 = !base.key1.empty();
    const bool have_key2 = !base.key2.empty();
    if (!have_key1 && !have_key2) return false;
    uint64_t* rng = &replica_rng_states_[replica_index];
    static const bool freeze_key2 = std::getenv("CR_FREEZE_KEY2") != nullptr;
    const bool use_key1 = have_key1 && (!have_key2 || freeze_key2 || Uniform(rng) < 0.5);
    std::vector<uint8_t>& key = use_key1 ? base.key1 : base.key2;
    if (key.empty()) return false;
    const std::size_t index = RandRange(rng, key.size());
    const uint8_t original = key[index];

    const bool repair_active = !circuit_.xts_block_sectors.empty() &&
                               base.key1.size() >= 32 && base.key2.size() >= 32;
    KeyCache scratch = replica_key_caches_[replica_index];
    std::array<double, 256> energy{};
    for (int value = 0; value < 256; ++value) {
      key[index] = static_cast<uint8_t>(value);
      ApplyAdmmRepair(&base, &scratch);
      if (repair_active) {
        energy[value] = AsciiEnergyFromPlaintext(base);
      } else {
        const ValueBuffer values = EvaluateValuesAndDerive(&base, &scratch);
        energy[value] = Energy(ScoreRelaxed(values));
      }
    }

    const double min_energy = *std::min_element(energy.begin(), energy.end());
    const double beta = EffectiveBeta(replica_index);
    std::array<double, 256> weight{};
    double total = 0.0;
    for (int value = 0; value < 256; ++value) {
      weight[value] = std::exp(ClampExpArg(-beta * (energy[value] - min_energy)));
      total += weight[value];
    }
    int chosen = static_cast<int>(original);
    if (total > 0.0) {
      const double threshold = Uniform(rng) * total;
      double cumulative = 0.0;
      chosen = 255;
      for (int value = 0; value < 256; ++value) {
        cumulative += weight[value];
        if (cumulative >= threshold) {
          chosen = value;
          break;
        }
      }
    }

    key[index] = static_cast<uint8_t>(chosen);
    const bool repaired = ApplyAdmmRepair(&base, &replica_key_caches_[replica_index]);
    values_[replica_index] = EvaluateValuesAndDerive(&base, &replica_key_caches_[replica_index]);
    relaxed_scores_[replica_index] = ScoreRelaxed(values_[replica_index]);
    hard_scores_[replica_index] = relaxed_scores_[replica_index].hard;
    energies_[replica_index] = Energy(relaxed_scores_[replica_index]);
    if (replica_index < replica_repair_valid_.size()) replica_repair_valid_[replica_index] = repaired;

    ++replica_proposal_attempts_[replica_index][2];
    if (chosen != static_cast<int>(original)) ++replica_proposal_accepts_[replica_index][2];
    RecordAcceptedKeyStats(replica_index);
    if (hard_scores_[replica_index].hamming_score == 0.0) {
      #ifdef _OPENMP
      #pragma omp critical(feasible_harvest)
      #endif
      MaybeHarvest(base, hard_scores_[replica_index]);
    }
    return true;
  }

  void StepReplica(std::size_t replica_index) {
    const uint64_t profile_start = ProfileStart();
    if (config_.key_gibbs_prob > 0.0 &&
        Uniform(&replica_rng_states_[replica_index]) < config_.key_gibbs_prob) {
      bool moved = false;
      // For one-round AES the plaintext is governed by key1 (via the coupled
      // (rk0[dst], rk1[src]) pair) and key2 (via the per-sector tweak). The exact
      // joint block move makes key1 optimal for the current tweaks; the composite
      // tweak-refit move searches key2 with key1 re-optimised so the two stop
      // fighting each other. Single-byte Gibbs (also covering key2) is the
      // fallback and the only option for multi-round / non-XTS circuits.
      if (config_.aes_rounds == 1 && !replicas_[replica_index].key1.empty()) {
        // Move-mix is env-tunable for A/B sweeps: CR_REFIT_PROB selects the
        // expensive key2 profile-refit move, CR_BLOCK_PROB the exact key1 block
        // move, CR_KEY2BLOCK_PROB the exact key2 block move (the key2 mirror of the
        // key1 block move); the remainder falls through to single-byte Gibbs.
        static const double refit_prob = []() {
          const char* e = std::getenv("CR_REFIT_PROB");
          return e ? std::atof(e) : 0.0;
        }();
        static const double block_prob = []() {
          const char* e = std::getenv("CR_BLOCK_PROB");
          return e ? std::atof(e) : 0.6;
        }();
        static const double key2block_prob = []() {
          const char* e = std::getenv("CR_KEY2BLOCK_PROB");
          return e ? std::atof(e) : 0.0;
        }();
        const double u = Uniform(&replica_rng_states_[replica_index]);
        if (u < refit_prob) {
          moved = KeyTweakRefitStep(replica_index);
        } else if (u < refit_prob + block_prob) {
          moved = KeyColumnPairBlockStep(replica_index);
        } else if (u < refit_prob + block_prob + key2block_prob) {
          moved = KeyTweakColumnPairBlockStep(replica_index);
        }
      } else if (config_.aes_rounds >= 2 && Key1ProfileUsable(replicas_[replica_index])) {
        static const double profile_k2_prob = []() {
          const char* e = std::getenv("CR_PROFILE_K2_PROB");
          return e ? std::atof(e) : 0.5;
        }();
        if (Uniform(&replica_rng_states_[replica_index]) < profile_k2_prob) {
          moved = KeyTweakProfileGibbsStep(replica_index);
        } else {
          moved = Key1ProfileGibbsStep(replica_index);
        }
      }
      if (!moved) moved = KeyByteGibbsStep(replica_index);
      if (moved) {
        ProfileAdd(PROFILE_STEP_REPLICA_TOTAL, profile_start);
        return;
      }
    }
    std::vector<RelaxedAction> actions;
    const double forward_sum = BuildActionWeightsSampled(replica_index, &actions);
    if (actions.empty() || forward_sum <= 0.0) {
      ProfileAdd(PROFILE_STEP_REPLICA_TOTAL, profile_start);
      return;
    }
    const double threshold = Uniform(&replica_rng_states_[replica_index]) * forward_sum;
    double cumulative = 0.0;
    std::size_t selected_index = actions.size() - 1;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      cumulative += actions[i].weight;
      if (cumulative >= threshold) {
        selected_index = i;
        break;
      }
    }
    RelaxedAction selected = std::move(actions[selected_index]);
    const std::size_t family = static_cast<std::size_t>(std::max(0, std::min(6, selected.family)));
    ++replica_proposal_attempts_[replica_index][family];

    const double delta = selected.energy - energies_[replica_index];
    const double log_accept = -EffectiveBeta(replica_index) * delta;
    if (log_accept >= 0.0 || Uniform(&replica_rng_states_[replica_index]) < std::exp(ClampExpArg(log_accept))) {
      replicas_[replica_index] = std::move(selected.candidate);
      values_[replica_index] = std::move(selected.candidate_values);
      relaxed_scores_[replica_index] = std::move(selected.candidate_score);
      hard_scores_[replica_index] = relaxed_scores_[replica_index].hard;
      energies_[replica_index] = selected.energy;
      replica_key_caches_[replica_index] = std::move(selected.candidate_cache);
      if (replica_index < replica_repair_valid_.size()) replica_repair_valid_[replica_index] = selected.repair_valid;
      ++replica_proposal_accepts_[replica_index][family];
      RecordAcceptedKeyStats(replica_index);
      if (hard_scores_[replica_index].hamming_score == 0.0) {
        #ifdef _OPENMP
        #pragma omp critical(feasible_harvest)
        #endif
        MaybeHarvest(replicas_[replica_index], hard_scores_[replica_index]);
      }
    }
    ProfileAdd(PROFILE_STEP_REPLICA_TOTAL, profile_start);
  }

  double BuildActionWeightsSampled(
      std::size_t replica_index,
      std::vector<RelaxedAction>* actions) {
    actions->clear();
    const AssignmentState& base = replicas_[replica_index];
    const double base_energy = energies_[replica_index];
    const ValueBuffer& base_values = values_[replica_index];
    double total = 0.0;
    std::unordered_set<uint64_t> seen;
    auto score_action = [&](const RelaxedAction& action) {
      const uint64_t full_score_profile_start = ProfileStart();
      RelaxedAction weighted = action;
      weighted.candidate = base;
      weighted.candidate_cache = replica_key_caches_[replica_index];
      ApplyAction(&weighted.candidate, weighted);
      weighted.repair_valid =
          replica_index < replica_repair_valid_.size() ? replica_repair_valid_[replica_index] : false;
      if (ActionTouchesPlaintextOrKey(weighted)) weighted.repair_valid = false;
      if (weighted.family == 2 || weighted.family == 4 || weighted.family == 5) {
        weighted.repair_valid = ApplyAdmmRepair(&weighted.candidate, &weighted.candidate_cache);
      }
      if (!TryScoreActionDelta(replica_index, &weighted)) {
        weighted.candidate_values = EvaluateValuesAndDerive(&weighted.candidate, &weighted.candidate_cache);
        weighted.candidate_score = ScoreRelaxed(weighted.candidate_values);
      }
      weighted.energy = Energy(weighted.candidate_score);
      const double sbox_bias = SboxTableBias(base_values, action);
      weighted.weight = std::exp(ClampExpArg(0.5 * (base_energy - weighted.energy) - 0.25 * sbox_bias));
      weighted.candidate_ready = true;
      ProfileAdd(PROFILE_SCORE_FULL, full_score_profile_start);
      return weighted;
    };

    std::size_t attempts = 0;
    while (actions->size() < kSampledCandidatePool && attempts < kSampledCandidateAttempts) {
      ++attempts;
      RelaxedAction action;
      if (!SampleRandomAction(base, &replica_rng_states_[replica_index], &action)) continue;
      if (!seen.insert(ActionKey(action)).second) continue;
      RelaxedAction weighted = score_action(action);
      actions->push_back(weighted);
      total += weighted.weight;
    }
    return total;
  }

  bool SampleRandomAction(const AssignmentState& base, uint64_t* rng_state, RelaxedAction* action) const {
    enum class ActionSampleKind { Plaintext, KeyBit, Wire, KeyChain, KeyWordSwap };
    std::array<ActionSampleKind, 5> kinds{};
    std::size_t kind_count = 0;
    if (!base.plaintext.empty()) kinds[kind_count++] = ActionSampleKind::Plaintext;
    if (!base.key1.empty() || !base.key2.empty()) kinds[kind_count++] = ActionSampleKind::KeyBit;
    if (!base.wires.empty()) kinds[kind_count++] = ActionSampleKind::Wire;
    if (base.key1.size() >= 32 || base.key2.size() >= 32) kinds[kind_count++] = ActionSampleKind::KeyChain;
    if (base.key1.size() >= 32 || base.key2.size() >= 32) kinds[kind_count++] = ActionSampleKind::KeyWordSwap;
    if (kind_count == 0) return false;

    action->energy = 0.0;
    action->weight = 0.0;
    action->flips.clear();
    switch (kinds[RandRange(rng_state, kind_count)]) {
      case ActionSampleKind::Plaintext: {
        action->family = 1;
        const std::size_t index = RandRange(rng_state, base.plaintext.size());
        const int bit = static_cast<int>(RandRange(rng_state, 8));
        action->flips.push_back({0, {index, static_cast<uint8_t>(1u << bit)}});
        return true;
      }
      case ActionSampleKind::KeyBit: {
        action->family = 2;
        const bool use_key1 = !base.key1.empty() && (base.key2.empty() || Uniform(rng_state) < 0.5);
        const int region = use_key1 ? 1 : 2;
        const std::size_t key_size = use_key1 ? base.key1.size() : base.key2.size();
        const std::size_t index = RandRange(rng_state, key_size);
        const int bit = static_cast<int>(RandRange(rng_state, 8));
        action->flips.push_back({region, {index, static_cast<uint8_t>(1u << bit)}});
        return true;
      }
      case ActionSampleKind::Wire: {
        for (int attempt = 0; attempt < 8; ++attempt) {
          const std::size_t wire_offset = RandRange(rng_state, base.wires.size());
          if (wire_offset < linear_determined_wire_.size() && linear_determined_wire_[wire_offset]) continue;
          const int bit = static_cast<int>(RandRange(rng_state, 8));
          const int32_t value_id =
              wire_offset < circuit_.graph.wire_value_by_offset.size() ? circuit_.graph.wire_value_by_offset[wire_offset] : -1;
          action->family = (value_id >= 0 && static_cast<std::size_t>(value_id) < sbox_input_value_.size() &&
                            sbox_input_value_[value_id])
                               ? 3
                               : 0;
          action->flips.push_back({3, {wire_offset, static_cast<uint8_t>(1u << bit)}});
          return true;
        }
        return false;
      }
      case ActionSampleKind::KeyChain: {
        action->family = 4;
        const bool use_key1 = base.key1.size() >= 32 && (base.key2.size() < 32 || Uniform(rng_state) < 0.5);
        const int key_region = use_key1 ? 1 : 2;
        const std::size_t key_size = use_key1 ? base.key1.size() : base.key2.size();
        const std::size_t start = RandRange(rng_state, 8);
        const int bit = static_cast<int>(RandRange(rng_state, 8));
        for (std::size_t byte = start; byte < std::min<std::size_t>(key_size, 32); byte += 8) {
          action->flips.push_back({key_region, {byte, static_cast<uint8_t>(1u << bit)}});
        }
        return !action->flips.empty();
      }
      case ActionSampleKind::KeyWordSwap: {
        action->family = 5;
        const bool use_key1 = base.key1.size() >= 32 && (base.key2.size() < 32 || Uniform(rng_state) < 0.5);
        const int key_region = use_key1 ? 1 : 2;
        const std::vector<uint8_t>& key = use_key1 ? base.key1 : base.key2;
        const std::size_t word_a = RandRange(rng_state, 8);
        std::size_t word_b = RandRange(rng_state, 7);
        if (word_b >= word_a) ++word_b;
        for (std::size_t byte = 0; byte < 4; ++byte) {
          const std::size_t a_idx = 4u * word_a + byte;
          const std::size_t b_idx = 4u * word_b + byte;
          const uint8_t mask = static_cast<uint8_t>(key[a_idx] ^ key[b_idx]);
          if (mask == 0) continue;
          action->flips.push_back({key_region, {a_idx, mask}});
          action->flips.push_back({key_region, {b_idx, mask}});
        }
        return !action->flips.empty();
      }
    }
    return false;
  }

  double SboxTableBias(const ValueBuffer& base_values, const RelaxedAction& action) const {
    if (action.family != 3) return 0.0;
    double bias = 0.0;
    for (const auto& flip : action.flips) {
      if (flip.first != 3) continue;
      const std::size_t wire_offset = flip.second.first;
      const uint8_t mask = flip.second.second;
      const int bit = mask == 0 ? 0 : static_cast<int>(std::log2(static_cast<double>(mask)));
      if (wire_offset >= circuit_.graph.wire_value_by_offset.size()) continue;
      const int32_t value_id = circuit_.graph.wire_value_by_offset[wire_offset];
      if (value_id < 0 || static_cast<std::size_t>(value_id) >= sbox_input_value_.size() || !sbox_input_value_[value_id]) continue;
      if (static_cast<std::size_t>(value_id) >= circuit_.graph.ops_by_input.size()) continue;
      const uint8_t input = evaluator_.ValueU8(base_values, value_id);
      for (const int32_t op_index : circuit_.graph.ops_by_input[value_id]) {
        if (op_index < 0 || static_cast<std::size_t>(op_index) >= circuit_.opcodes.size()) continue;
        if (circuit_.opcodes[op_index] != OP_SBOX8) continue;
        const int32_t output = circuit_.outputs[op_index];
        if (output < 0 || static_cast<std::size_t>(output) >= circuit_.graph.definitions_by_right.size()) continue;
        for (const int32_t constraint : circuit_.graph.definitions_by_right[output]) {
          if (constraint < 0 || static_cast<std::size_t>(constraint) >= circuit_.constraint_left.size()) continue;
          const uint8_t target = evaluator_.ValueU8(base_values, circuit_.constraint_left[constraint]);
          bias += static_cast<double>(sbox_flip_delta_[input][target][bit]);
        }
      }
    }
    return bias;
  }

  void ApplyAction(AssignmentState* assignment, const RelaxedAction& action) const {
    for (const auto& flip : action.flips) {
      const int region = flip.first;
      const std::size_t index = flip.second.first;
      const uint8_t mask = flip.second.second;
      if (region == 0 && index < assignment->plaintext.size()) assignment->plaintext[index] ^= mask;
      else if (region == 1 && index < assignment->key1.size()) assignment->key1[index] ^= mask;
      else if (region == 2 && index < assignment->key2.size()) assignment->key2[index] ^= mask;
      else if (region == 3 && index < assignment->wires.size()) assignment->wires[index] ^= mask;
    }
  }

  bool ActionTouchesPlaintextOrKey(const RelaxedAction& action) const {
    for (const auto& flip : action.flips) {
      if (flip.first == 0 || flip.first == 1 || flip.first == 2) return true;
    }
    return false;
  }

  bool TryScoreActionDelta(std::size_t replica_index, RelaxedAction* action) {
    if (action->flips.size() != 1 || action->flips[0].first != 3) return false;
    if (action->family != 0 && action->family != 3) return false;
    const std::size_t wire_offset = action->flips[0].second.first;
    if (wire_offset >= circuit_.graph.wire_value_by_offset.size()) return false;
    const int32_t changed_value_id = circuit_.graph.wire_value_by_offset[wire_offset];
    if (changed_value_id < 0 || static_cast<std::size_t>(changed_value_id) >= circuit_.graph.constraints_by_value.size()) {
      return false;
    }
    std::vector<std::pair<int32_t, std::vector<uint8_t>>> undo;
    action->candidate_score = ScoreWireFlipAffected(replica_index, &action->candidate, relaxed_scores_[replica_index], changed_value_id, &undo);
    action->candidate_values = values_[replica_index];
    RestoreValues(replica_index, undo);
    action->candidate_cache = replica_key_caches_[replica_index];
    return true;
  }

  ContinuousRelaxedScore ScoreWireFlipAffected(
      std::size_t replica_index,
      AssignmentState* assignment,
      const ContinuousRelaxedScore& base,
      int32_t changed_value_id,
      std::vector<std::pair<int32_t, std::vector<uint8_t>>>* undo) {
    const uint64_t profile_start = ProfileStart();
    ContinuousRelaxedScore score = base;
    auto& values = values_[replica_index];
    auto remember = [&](int32_t value_id) {
      if (value_id < 0 || static_cast<std::size_t>(value_id) >= circuit_.value_count) return;
      for (const auto& entry : *undo) {
        if (entry.first == value_id) return;
      }
      undo->push_back({value_id, CopyValueBytes(values, value_id)});
    };

    std::vector<int32_t> queue;
    std::vector<int32_t> affected_constraints;
    uint32_t& stamp = replica_mark_stamps_[replica_index];
    ++stamp;
    if (stamp == 0) {
      std::fill(replica_value_marks_[replica_index].begin(), replica_value_marks_[replica_index].end(), 0);
      std::fill(replica_constraint_marks_[replica_index].begin(), replica_constraint_marks_[replica_index].end(), 0);
      stamp = 1;
    }
    auto mark_value = [&](int32_t value_id) {
      if (value_id < 0 || static_cast<std::size_t>(value_id) >= replica_value_marks_[replica_index].size()) return false;
      if (replica_value_marks_[replica_index][value_id] == stamp) return false;
      replica_value_marks_[replica_index][value_id] = stamp;
      return true;
    };
    auto mark_constraints = [&](int32_t value_id) {
      if (value_id < 0 || static_cast<std::size_t>(value_id) >= circuit_.graph.constraints_by_value.size()) return;
      for (const int32_t constraint : circuit_.graph.constraints_by_value[value_id]) {
        if (constraint < 0 || static_cast<std::size_t>(constraint) >= replica_constraint_marks_[replica_index].size()) continue;
        if (replica_constraint_marks_[replica_index][constraint] == stamp) continue;
        replica_constraint_marks_[replica_index][constraint] = stamp;
        affected_constraints.push_back(constraint);
      }
    };
    auto enqueue_if_changed = [&](int32_t value_id, const std::vector<uint8_t>& old_value) {
      if (CopyValueBytes(values, value_id) != old_value && mark_value(value_id)) queue.push_back(value_id);
    };
    std::function<void(int32_t, int32_t)> assign_wire = [&](int32_t value_id, int32_t source_value_id) {
      if (value_id < 0 || static_cast<std::size_t>(value_id) >= circuit_.graph.wire_offset_by_value.size()) return;
      const int32_t output_wire = circuit_.graph.wire_offset_by_value[value_id];
      if (output_wire < 0 || static_cast<std::size_t>(output_wire) >= assignment->wires.size()) return;
      remember(value_id);
      const std::vector<uint8_t> old_value = CopyValueBytes(values, value_id);
      const uint8_t value = evaluator_.ValueU8(values, source_value_id);
      assignment->wires[output_wire] = value;
      SetValueU8(&values, value_id, value);
      mark_constraints(value_id);
      enqueue_if_changed(value_id, old_value);
      for (const int32_t dependent_constraint : circuit_.graph.definitions_by_right[value_id]) {
        assign_wire(circuit_.constraint_left[dependent_constraint], value_id);
      }
    };

    remember(changed_value_id);
    const int32_t changed_wire_offset = circuit_.graph.wire_offset_by_value[changed_value_id];
    if (changed_wire_offset < 0 || static_cast<std::size_t>(changed_wire_offset) >= assignment->wires.size()) return score;
    SetValueU8(&values, changed_value_id, assignment->wires[changed_wire_offset]);
    queue.push_back(changed_value_id);
    mark_value(changed_value_id);
    mark_constraints(changed_value_id);

    for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
      const int32_t value_id = queue[cursor];
      if (value_id < 0 || static_cast<std::size_t>(value_id) >= circuit_.graph.ops_by_input.size()) continue;
      for (const int32_t op_index : circuit_.graph.ops_by_input[value_id]) {
        if (op_index < 0 || static_cast<std::size_t>(op_index) >= circuit_.opcodes.size()) continue;
        const int32_t output = circuit_.outputs[op_index];
        if (output < 0 || static_cast<std::size_t>(output) >= circuit_.value_count) continue;
        remember(output);
        const std::vector<uint8_t> old_value = CopyValueBytes(values, output);
        EvaluateOpInto(static_cast<std::size_t>(op_index), *assignment, &values, &replica_key_caches_[replica_index]);
        mark_constraints(output);
        enqueue_if_changed(output, old_value);
        if (static_cast<std::size_t>(output) < circuit_.graph.definitions_by_right.size()) {
          for (const int32_t constraint_index : circuit_.graph.definitions_by_right[output]) {
            if (constraint_index < 0 || static_cast<std::size_t>(constraint_index) >= circuit_.constraint_left.size()) continue;
            assign_wire(circuit_.constraint_left[constraint_index], output);
          }
        }
      }
    }

    for (const int32_t index : affected_constraints) {
      if (index < 0 || static_cast<std::size_t>(index) >= score.residuals.size()) continue;
      const std::size_t constraint_index = static_cast<std::size_t>(index);
      const double old_smooth = score.residuals[constraint_index];
      const double new_smooth = SmoothResidual(values, constraint_index);
      if (old_smooth != new_smooth) {
        score.residuals[constraint_index] = new_smooth;
        score.smooth_sum += new_smooth - old_smooth;
      }
      const uint32_t old_hard = score.hard.residuals[constraint_index];
      const uint32_t new_hard = hard_scorer_.ResidualForConstraint(values, constraint_index);
      if (old_hard == new_hard) continue;
      score.hard.residuals[constraint_index] = new_hard;
      score.hard.hamming_score += static_cast<double>(new_hard) - static_cast<double>(old_hard);
      if (old_hard == 0 && new_hard != 0) {
        ++score.hard.violations;
      } else if (old_hard != 0 && new_hard == 0 && score.hard.violations > 0) {
        --score.hard.violations;
      }
    }
    score.hard.failing_indices.clear();
    ProfileAdd(PROFILE_SCORE_DELTA, profile_start);
    return score;
  }

  void RestoreValues(std::size_t replica_index, const std::vector<std::pair<int32_t, std::vector<uint8_t>>>& undo) {
    const uint64_t profile_start = ProfileStart();
    auto& values = values_[replica_index];
    for (auto it = undo.rbegin(); it != undo.rend(); ++it) {
      SetValueBytes(&values, it->first, it->second.data(), it->second.size());
    }
    ProfileAdd(PROFILE_RESTORE_VALUES, profile_start);
  }

  void AttemptSwaps() {
    const uint64_t profile_start = ProfileStart();
    for (std::size_t i = 0; i + 1 < replicas_.size(); ++i) {
      ++swap_attempts_[i];
      const double left_beta = EffectiveBeta(i);
      const double right_beta = EffectiveBeta(i + 1);
      const double log_accept = (left_beta - right_beta) * (energies_[i] - energies_[i + 1]);
      if (log_accept >= 0.0 || Uniform(&rng_state_) < std::exp(ClampExpArg(log_accept))) {
        std::swap(replicas_[i], replicas_[i + 1]);
        std::swap(values_[i], values_[i + 1]);
        std::swap(relaxed_scores_[i], relaxed_scores_[i + 1]);
        std::swap(hard_scores_[i], hard_scores_[i + 1]);
        std::swap(energies_[i], energies_[i + 1]);
        std::swap(replica_key_caches_[i], replica_key_caches_[i + 1]);
        if (i + 1 < replica_repair_valid_.size()) {
          std::swap(replica_repair_valid_[i], replica_repair_valid_[i + 1]);
        }
        ++swap_accepts_[i];
      }
    }
    ProfileAdd(PROFILE_ATTEMPT_SWAPS, profile_start);
  }

  void DualUpdate() {
    const uint64_t profile_start = ProfileStart();
    if (relaxed_scores_.empty()) return;
    std::vector<double> means(relaxed_scores_[0].residuals.size(), 0.0);
    double weight_sum = 0.0;
    for (std::size_t r = 0; r < relaxed_scores_.size(); ++r) {
      const double weight = r < replica_scale_.size() ? replica_scale_[r] : 1.0;
      weight_sum += weight;
      for (std::size_t i = 0; i < means.size(); ++i) means[i] += weight * relaxed_scores_[r].residuals[i];
    }
    if (weight_sum <= 0.0) weight_sum = static_cast<double>(relaxed_scores_.size());
    std::array<double, 3> class_violation{0.0, 0.0, 0.0};
    std::array<std::size_t, 3> class_counts{0, 0, 0};
    for (std::size_t i = 0; i < means.size(); ++i) {
      means[i] /= weight_sum;
      const int cls = i < circuit_.constraint_classes.size() ? circuit_.constraint_classes[i] : 3;
      const std::size_t bucket = BucketForClass(cls);
      class_violation[bucket] += means[i];
      ++class_counts[bucket];
    }
    for (std::size_t bucket = 0; bucket < class_violation.size(); ++bucket) {
      if (class_counts[bucket] != 0) class_violation[bucket] /= static_cast<double>(class_counts[bucket]);
      const double previous = previous_violation_by_class_[bucket];
      const bool shrinking = previous < 0.0 || class_violation[bucket] <= config_.scheduled_violation_shrink * previous;
      if (!shrinking) {
        rho_by_class_[bucket] = std::max(rho_by_class_[bucket] * std::max(config_.scheduled_rho_growth, 1.0), rho_by_class_[bucket] + 1e-9);
        ++rho_escalation_counts_by_class_[bucket];
      }
      previous_violation_by_class_[bucket] = class_violation[bucket];
    }

    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    for (std::size_t i = 0; i < means.size(); ++i) {
      const int cls = i < circuit_.constraint_classes.size() ? circuit_.constraint_classes[i] : 3;
      if (cls != CLASS_ASCII && cls != CLASS_CONSISTENCY) continue;
      const double old_lambda = multipliers_[i];
      dual_m_[i] = beta1 * dual_m_[i] + (1.0 - beta1) * means[i];
      dual_v_[i] = beta2 * dual_v_[i] + (1.0 - beta2) * means[i] * means[i];
      ++dual_steps_[i];
      const double m_hat = dual_m_[i] / (1.0 - std::pow(beta1, static_cast<double>(dual_steps_[i])));
      const double v_hat = dual_v_[i] / (1.0 - std::pow(beta2, static_cast<double>(dual_steps_[i])));
      const double momentum = 0.9 * (multipliers_[i] - lambda_previous_[i]);
      lambda_previous_[i] = multipliers_[i];
      multipliers_[i] = std::max(0.0, multipliers_[i] + config_.dual_eta * m_hat / (std::sqrt(v_hat) + eps) + momentum);
      if (multipliers_[i] != old_lambda) ++lambda_update_counts_by_class_[BucketForClass(cls)];
    }
    for (std::size_t i = 0; i < penalties_.size(); ++i) {
      const int cls = i < circuit_.constraint_classes.size() ? circuit_.constraint_classes[i] : 3;
      penalties_[i] = rho_by_class_[BucketForClass(cls)];
    }
    ProfileAdd(PROFILE_DUAL_UPDATE, profile_start);
  }

  void PopulationAnneal() {
    if (replicas_.size() <= 1) return;
    const double beta = EffectiveBeta(0);
    const double best = *std::min_element(energies_.begin(), energies_.end());
    std::vector<double> cumulative(replicas_.size(), 0.0);
    double total = 0.0;
    for (std::size_t i = 0; i < energies_.size(); ++i) {
      total += std::exp(ClampExpArg(-beta * (energies_[i] - best)));
      cumulative[i] = total;
    }
    if (total <= 0.0) return;
    std::vector<AssignmentState> next_replicas;
    std::vector<ValueBuffer> next_values;
    std::vector<ContinuousRelaxedScore> next_relaxed;
    std::vector<ScoreData> next_hard;
    std::vector<double> next_energies;
    std::vector<KeyCache> next_caches;
    std::vector<uint8_t> next_repair_valid;
    next_replicas.reserve(replicas_.size());
    next_values.reserve(values_.size());
    next_relaxed.reserve(relaxed_scores_.size());
    next_hard.reserve(hard_scores_.size());
    next_energies.reserve(energies_.size());
    next_caches.reserve(replica_key_caches_.size());
    next_repair_valid.reserve(replica_repair_valid_.size());
    for (std::size_t i = 0; i < replicas_.size(); ++i) {
      const double draw = Uniform(&rng_state_) * total;
      const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), draw);
      const std::size_t picked = static_cast<std::size_t>(std::distance(cumulative.begin(), it));
      next_replicas.push_back(replicas_[picked]);
      next_values.push_back(values_[picked]);
      next_relaxed.push_back(relaxed_scores_[picked]);
      next_hard.push_back(hard_scores_[picked]);
      next_energies.push_back(energies_[picked]);
      next_caches.push_back(replica_key_caches_[picked]);
      next_repair_valid.push_back(picked < replica_repair_valid_.size() ? replica_repair_valid_[picked] : uint8_t{0});
    }
    replicas_.swap(next_replicas);
    values_.swap(next_values);
    relaxed_scores_.swap(next_relaxed);
    hard_scores_.swap(next_hard);
    energies_.swap(next_energies);
    replica_key_caches_.swap(next_caches);
    replica_repair_valid_.swap(next_repair_valid);
  }

  void RecomputeEnergies() {
    energies_.resize(relaxed_scores_.size());
    for (std::size_t i = 0; i < relaxed_scores_.size(); ++i) energies_[i] = Energy(relaxed_scores_[i]);
  }

  void MaybeHarvest(const AssignmentState& assignment, const ScoreData& score) {
    const uint64_t profile_start = ProfileStart();
    if (feasible_archive_.MaybeInsert(assignment, score, &seen_keys_, &feasible_)) ++total_harvested_;
    ProfileAdd(PROFILE_HARVEST, profile_start);
  }

  void RecordAcceptedKeyStats(std::size_t replica_index) {
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

  double KeyMarginalMaxDeviation() const {
    if (key_visit_count_ == 0) return 0.0;
    double max_deviation = 0.0;
    const double alpha = std::max(config_.marginal_alpha, 0.0);
    const double denominator = static_cast<double>(key_visit_count_) + 2.0 * alpha;
    for (const std::size_t ones : key_ones_) {
      const double p = denominator > 0.0
                           ? (static_cast<double>(ones) + alpha) / denominator
                           : static_cast<double>(ones) / static_cast<double>(key_visit_count_);
      max_deviation = std::max(max_deviation, std::abs(p - 0.5));
    }
    return max_deviation;
  }

  double KeyInformationBits() const {
    if (key_visit_count_ == 0) return 0.0;
    double information = 0.0;
    const double alpha = std::max(config_.marginal_alpha, 0.0);
    const double denominator = static_cast<double>(key_visit_count_) + 2.0 * alpha;
    for (const std::size_t ones : key_ones_) {
      double p = denominator > 0.0
                     ? (static_cast<double>(ones) + alpha) / denominator
                     : static_cast<double>(ones) / static_cast<double>(key_visit_count_);
      p = std::max(1e-12, std::min(1.0 - 1e-12, p));
      const double entropy = -(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p));
      information += 1.0 - entropy;
    }
    return information;
  }

  double KeyInformationNullBaselineBits() const {
    if (key_visit_count_ == 0) return 0.0;
    const double alpha = std::max(config_.marginal_alpha, 0.0);
    const double denominator = static_cast<double>(key_visit_count_) + 2.0 * alpha;
    double expected_per_bit = 0.0;
    const double log_denominator = std::lgamma(static_cast<double>(key_visit_count_) + 1.0) -
                                   static_cast<double>(key_visit_count_) * std::log(2.0);
    for (std::size_t ones = 0; ones <= key_visit_count_; ++ones) {
      const double log_probability = log_denominator - std::lgamma(static_cast<double>(ones) + 1.0) -
                                     std::lgamma(static_cast<double>(key_visit_count_ - ones) + 1.0);
      const double probability = std::exp(log_probability);
      double p = denominator > 0.0
                     ? (static_cast<double>(ones) + alpha) / denominator
                     : static_cast<double>(ones) / static_cast<double>(key_visit_count_);
      p = std::max(1e-12, std::min(1.0 - 1e-12, p));
      const double entropy = -(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p));
      expected_per_bit += probability * (1.0 - entropy);
    }
    return 512.0 * expected_per_bit;
  }

  double MarginalEss() const {
    const std::size_t n = cold_key_hamming_trace_.size();
    if (n < 4) return 0.0;
    double mean = 0.0;
    for (const double value : cold_key_hamming_trace_) mean += value;
    mean /= static_cast<double>(n);
    double variance = 0.0;
    for (const double value : cold_key_hamming_trace_) {
      const double centered = value - mean;
      variance += centered * centered;
    }
    if (variance <= 1e-12) return 0.0;
    const std::size_t max_lag = std::min<std::size_t>(n - 1, 512);
    double tau = 1.0;
    for (std::size_t lag = 1; lag <= max_lag; ++lag) {
      double covariance = 0.0;
      for (std::size_t i = 0; i + lag < n; ++i) {
        covariance += (cold_key_hamming_trace_[i] - mean) * (cold_key_hamming_trace_[i + lag] - mean);
      }
      const double rho = covariance / variance;
      if (rho <= 0.0) break;
      tau += 2.0 * rho;
      if (static_cast<double>(lag) >= 6.0 * tau) break;
    }
    return static_cast<double>(n) / std::max(2.0 * tau, 1.0);
  }

  double MarginalRhat() const {
    const std::size_t n = cold_key_hamming_trace_.size();
    if (n < 4) return 0.0;
    const std::size_t half = n / 2;
    auto chain_stats = [&](std::size_t begin, std::size_t end) {
      double mean = 0.0;
      for (std::size_t i = begin; i < end; ++i) mean += cold_key_hamming_trace_[i];
      mean /= static_cast<double>(end - begin);
      double variance = 0.0;
      for (std::size_t i = begin; i < end; ++i) {
        const double centered = cold_key_hamming_trace_[i] - mean;
        variance += centered * centered;
      }
      variance /= static_cast<double>(std::max<std::size_t>(1, end - begin - 1));
      return std::pair<double, double>{mean, variance};
    };
    const auto first = chain_stats(0, half);
    const auto second = chain_stats(half, half * 2);
    const double within = 0.5 * (first.second + second.second);
    if (within <= 1e-12) return 0.0;
    const double mean = 0.5 * (first.first + second.first);
    const double between = static_cast<double>(half) *
                           ((first.first - mean) * (first.first - mean) + (second.first - mean) * (second.first - mean));
    const double var_hat = ((static_cast<double>(half) - 1.0) / static_cast<double>(half)) * within +
                           between / static_cast<double>(half);
    return std::sqrt(std::max(var_hat / within, 0.0));
  }

  void RecordEnergyStats() {
    for (std::size_t i = 0; i < std::min(energies_.size(), energy_sample_counts_.size()); ++i) {
      const double value = energies_[i];
      ++energy_sample_counts_[i];
      const double delta = value - energy_mean_by_rung_[i];
      energy_mean_by_rung_[i] += delta / static_cast<double>(energy_sample_counts_[i]);
      const double delta2 = value - energy_mean_by_rung_[i];
      energy_m2_by_rung_[i] += delta * delta2;
    }
  }

  void UpdateReplicaFlow() {
    if (replicas_.empty()) return;
    const std::size_t top = replicas_.size() - 1;
    for (std::size_t rung = 0; rung < replicas_.size(); ++rung) {
      if (rung == 0) {
        if (replica_directions_[rung] == -1) ++replica_round_trips_[rung];
        replica_directions_[rung] = 1;
      } else if (rung == top) {
        if (replica_directions_[rung] == 1) ++replica_round_trips_[rung];
        replica_directions_[rung] = -1;
      }
      if (replica_directions_[rung] > 0) ++replica_up_counts_[rung];
      else if (replica_directions_[rung] < 0) ++replica_down_counts_[rung];
    }
  }

  std::vector<std::size_t> AlgebraCounts() const {
    std::size_t linear_ops = 0;
    std::size_t nonlinear_ops = 0;
    for (const int opcode : circuit_.opcodes) {
      if (IsLinearOpcode(opcode)) ++linear_ops;
      else ++nonlinear_ops;
    }
    std::size_t determined = 0;
    for (const bool value : linear_determined_wire_) {
      if (value) ++determined;
    }
    return {
        (circuit_.plaintext_count + circuit_.key1_count + circuit_.key2_count + circuit_.wire_offsets.size()) * 8,
        linear_ops,
        nonlinear_ops,
        determined,
        circuit_.opcodes.size() + circuit_.constraint_kinds.size(),
        std::size_t{0},
        std::size_t{1},
    };
  }

  double LogZEstimate() const {
    if (energy_mean_by_rung_.empty()) return 0.0;
    double total = 0.0;
    for (double value : energy_mean_by_rung_) total += value;
    return -total / static_cast<double>(energy_mean_by_rung_.size());
  }

  uint64_t ProfileStart() const {
    return config_.profile ? NowNanos() : 0;
  }

  void ProfileAdd(std::size_t counter, uint64_t started_at) const {
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

  std::vector<double> ProfileSeconds() const {
    std::vector<double> seconds(profile_nanos_.size(), 0.0);
    for (std::size_t i = 0; i < profile_nanos_.size(); ++i) {
      seconds[i] = static_cast<double>(profile_nanos_[i]) / 1e9;
    }
    return seconds;
  }

  CircuitModel circuit_;
  PTConfig config_;
  uint64_t rng_state_ = 0;
  ByteCircuitEvaluator evaluator_;
  HammingScorer hard_scorer_;
  FeasibleArchive feasible_archive_;
  std::vector<AssignmentState> replicas_;
  std::vector<ValueBuffer> values_;
  std::vector<ContinuousRelaxedScore> relaxed_scores_;
  std::vector<ScoreData> hard_scores_;
  std::vector<double> energies_;
  std::vector<double> temperatures_;
  std::vector<double> replica_scale_;
  std::vector<double> multipliers_;
  std::vector<double> penalties_;
  std::vector<double> lambda_previous_;
  std::vector<double> dual_m_;
  std::vector<double> dual_v_;
  std::vector<std::size_t> dual_steps_;
  std::array<double, 3> previous_violation_by_class_;
  std::array<double, 3> rho_by_class_;
  std::array<std::size_t, 3> lambda_update_counts_by_class_;
  std::array<std::size_t, 3> rho_escalation_counts_by_class_;
  std::array<std::size_t, 512> key_ones_{};
  std::size_t key_visit_count_ = 0;
  std::deque<std::array<uint8_t, 512>> key_bit_window_;
  std::deque<std::string> key_string_window_;
  std::unordered_map<std::string, std::size_t> key_window_counts_;
  std::deque<double> cold_key_hamming_trace_;
  static constexpr std::size_t kMaxColdTrace = 4096;
  std::vector<std::array<std::size_t, 7>> replica_proposal_attempts_;
  std::vector<std::array<std::size_t, 7>> replica_proposal_accepts_;
  std::vector<std::size_t> swap_attempts_;
  std::vector<std::size_t> swap_accepts_;
  std::vector<std::size_t> energy_sample_counts_;
  std::vector<double> energy_mean_by_rung_;
  std::vector<double> energy_m2_by_rung_;
  std::vector<std::size_t> replica_round_trips_;
  std::vector<std::size_t> replica_up_counts_;
  std::vector<std::size_t> replica_down_counts_;
  std::vector<int> replica_directions_;
  std::vector<uint64_t> replica_rng_states_;
  std::vector<KeyCache> replica_key_caches_;
  std::vector<uint8_t> replica_repair_valid_;
  std::vector<KeyProfileCache> replica_key1_profile_;
  std::vector<std::vector<uint32_t>> replica_value_marks_;
  std::vector<std::vector<uint32_t>> replica_constraint_marks_;
  std::vector<uint32_t> replica_mark_stamps_;
  std::vector<AssignmentState> feasible_;
  std::unordered_set<std::string> seen_keys_;
  std::vector<bool> linear_determined_wire_;
  std::vector<bool> sbox_input_value_;
  std::array<std::array<std::array<int8_t, 8>, 256>, 256> sbox_flip_delta_{};
  std::array<uint8_t, 256> inv_sbox_{};
  std::array<int, 256> ascii_distance_{};
  std::vector<int32_t> ascii_constraint_at_plain_;
  std::vector<std::vector<double>> replica_pair_energy_;
  std::size_t total_harvested_ = 0;
  const int score_fallback_zero_ = 0;
  std::size_t thread_count_ = 1;
  mutable std::vector<uint64_t> profile_nanos_;
  mutable std::vector<std::size_t> profile_counts_;
  std::size_t epochs_ = 0;
  std::size_t sweeps_ = 0;
};

ContinuousRelaxedEngine::ContinuousRelaxedEngine(
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
    : impl_(std::make_unique<ContinuousRelaxedSolver>(
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
          value_widths,
          constraint_classes,
          plaintext_start,
          plaintext_count,
          key1_start,
          key1_count,
          key2_start,
          key2_count,
          xts_block_sectors,
          xts_block_indices,
          xts_block_targets,
          initial,
          config)) {}

ContinuousRelaxedEngine::~ContinuousRelaxedEngine() = default;
ContinuousRelaxedEngine::ContinuousRelaxedEngine(ContinuousRelaxedEngine&&) noexcept = default;
ContinuousRelaxedEngine& ContinuousRelaxedEngine::operator=(ContinuousRelaxedEngine&&) noexcept = default;

ScoreData ContinuousRelaxedEngine::ScoreAssignment(const AssignmentState& assignment) const {
  return impl_->ScoreAssignment(assignment);
}

ScoreData ContinuousRelaxedEngine::RunEpoch(std::size_t sweeps) {
  return impl_->RunEpoch(sweeps);
}

std::vector<uint32_t> ContinuousRelaxedEngine::Residuals() const {
  return impl_->Residuals();
}

PTDiagnostics ContinuousRelaxedEngine::Metrics() const {
  return impl_->Metrics();
}

std::vector<AssignmentState> ContinuousRelaxedEngine::DrainFeasible(std::size_t limit) {
  return impl_->DrainFeasible(limit);
}

AssignmentState ContinuousRelaxedEngine::CurrentAssignment() const {
  return impl_->CurrentAssignment();
}

AssignmentState ContinuousRelaxedEngine::DeriveAssignmentWires(const AssignmentState& assignment) const {
  return impl_->DeriveAssignmentWires(assignment);
}

void ContinuousRelaxedEngine::SetMultipliers(const std::vector<double>& multipliers) {
  impl_->SetMultipliers(multipliers);
}

void ContinuousRelaxedEngine::SetTemperatures(const std::vector<double>& temperatures) {
  impl_->SetTemperatures(temperatures);
}

void ContinuousRelaxedEngine::SetConstraintClasses(const std::vector<int32_t>& classes) {
  impl_->SetConstraintClasses(classes);
}

}  // namespace aes_xts_decoder
