#include "solver.hpp"

namespace aes_xts_decoder {

ScoreData ParallelTemperingSolver::RunEpoch(std::size_t sweeps) {
  const uint64_t profile_start = ProfileStart();
  if (sweeps != 0) ResetEnergyStats();
  for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
    const uint64_t step_profile_start = ProfileStart();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(thread_count_) schedule(static)
    #endif
    for (std::int64_t replica = 0; replica < static_cast<std::int64_t>(replicas_.size()); ++replica) {
      StepReplica(static_cast<std::size_t>(replica));
    }
    ProfileAdd(PROFILE_STEP_REPLICAS_WALL, step_profile_start);
    AttemptSwaps();
    UpdateReplicaFlow();
    RecordEnergyStats();
    ++sweeps_;
  }
  ++epochs_;
  DualUpdate();
  OptimizeTemperatureLadder();
  ProfileAdd(PROFILE_RUN_EPOCH_TOTAL, profile_start);
  return scores_.empty() ? ScoreData{} : scores_[0];
}

std::vector<uint32_t> ParallelTemperingSolver::Residuals() const {
  return scores_.empty() ? std::vector<uint32_t>{} : scores_[0].residuals;
}

PTDiagnostics ParallelTemperingSolver::Metrics() const {
  const uint64_t profile_start = ProfileStart();
  PTDiagnostics metrics{};
  metrics.epochs = epochs_;
  metrics.sweeps = sweeps_;
  metrics.feasible_count = feasible_.size();
  metrics.total_residual = scores_.empty() ? 0.0 : scores_[0].hamming_score;
  metrics.violations = scores_.empty() ? 0 : scores_[0].violations;
  metrics.max_residual = 0;
  metrics.swap_attempts = swap_attempts_;
  metrics.swap_accepts = swap_accepts_;
  metrics.proposal_attempts.assign(4, 0);
  metrics.proposal_accepts.assign(4, 0);
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
  if (!scores_.empty()) {
    for (std::size_t i = 0; i < scores_[0].residuals.size(); ++i) {
      metrics.max_residual = std::max(metrics.max_residual, scores_[0].residuals[i]);
      int cls = i < circuit_.constraint_classes.size() ? circuit_.constraint_classes[i] : 3;
      const int bucket = std::max(0, std::min(2, cls - 1));
      metrics.residuals_by_class[bucket] += scores_[0].residuals[i];
      const double multiplier = i < multipliers_.size() ? multipliers_[i] : 1.0;
      metrics.multiplier_mean_by_class[bucket] += multiplier;
      metrics.multiplier_max_by_class[bucket] = std::max(metrics.multiplier_max_by_class[bucket], multiplier);
      ++multiplier_counts[bucket];
    }
  }
  for (std::size_t i = 0; i < metrics.multiplier_mean_by_class.size(); ++i) {
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
  metrics.marginal_trusted = metrics.key_visit_count >= 50 && metrics.key_distinct_count >= min_distinct_keys &&
                             metrics.marginal_ess >= 25.0 && metrics.marginal_rhat > 0.0 && metrics.marginal_rhat <= 1.1;
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
  metrics.last_ladder_adaptation_epoch = last_ladder_adaptation_epoch_;
  metrics.energy_sample_counts = energy_sample_counts_;
  metrics.energy_mean_by_rung = energy_mean_by_rung_;
  metrics.energy_temperatures_by_rung = energy_temperatures_by_rung_;
  metrics.energy_variance_by_rung.assign(energy_m2_by_rung_.size(), 0.0);
  for (std::size_t i = 0; i < energy_m2_by_rung_.size(); ++i) {
    if (energy_sample_counts_[i] > 1) {
      metrics.energy_variance_by_rung[i] = energy_m2_by_rung_[i] / static_cast<double>(energy_sample_counts_[i] - 1);
    }
  }
  metrics.log_z_estimate_available =
      std::any_of(energy_sample_counts_.begin(), energy_sample_counts_.end(), [](std::size_t count) { return count != 0; });
  metrics.log_z_state_bits = (circuit_.plaintext_count + circuit_.key1_count + circuit_.key2_count) * 8;
  metrics.log_z_estimate = metrics.log_z_estimate_available ? LogZEstimate() : 0.0;
  metrics.log_feasible_count_estimate = 0.0;
  metrics.rho_by_class.assign(rho_by_class_.begin(), rho_by_class_.end());
  metrics.lambda_update_counts_by_class.assign(lambda_update_counts_by_class_.begin(), lambda_update_counts_by_class_.end());
  metrics.rho_escalation_counts_by_class.assign(rho_escalation_counts_by_class_.begin(), rho_escalation_counts_by_class_.end());
  metrics.infeasibility_suspected = infeasibility_suspected_;
  metrics.algebra_counts = algebra_counts_;
  metrics.bp_key_marginals = bp_key_marginals_;
  metrics.bp_converged = bp_converged_;
  metrics.algebra_exact = false;
  metrics.bp_available = false;
  metrics.alternative_available = false;
  metrics.langevin_available = false;
  metrics.alternative_log_z_estimates = {};
  metrics.langevin_seed_score = 0.0;
  ProfileAdd(PROFILE_METRICS, profile_start);
  metrics.profile_seconds = ProfileSeconds();
  metrics.profile_counts = profile_counts_;
  return metrics;
}

std::vector<AssignmentState> ParallelTemperingSolver::DrainFeasible(std::size_t limit) {
  const uint64_t profile_start = ProfileStart();
  const std::size_t count = limit == 0 ? feasible_.size() : std::min(limit, feasible_.size());
  std::vector<AssignmentState> out(feasible_.begin(), feasible_.begin() + count);
  feasible_.erase(feasible_.begin(), feasible_.begin() + count);
  ProfileAdd(PROFILE_DRAIN_FEASIBLE, profile_start);
  return out;
}

AssignmentState ParallelTemperingSolver::CurrentAssignment() const {
  const uint64_t profile_start = ProfileStart();
  if (replicas_.empty()) {
    ProfileAdd(PROFILE_CURRENT_ASSIGNMENT, profile_start);
    return AssignmentState{};
  }
  std::size_t best = 0;
  double best_energy = energies_.empty() ? Energy(scores_[0]) : energies_[0];
  for (std::size_t i = 1; i < scores_.size(); ++i) {
    const double energy = i < energies_.size() ? energies_[i] : Energy(scores_[i]);
    if (energy < best_energy) {
      best = i;
      best_energy = energy;
    }
  }
  AssignmentState assignment = replicas_[best];
  ProfileAdd(PROFILE_CURRENT_ASSIGNMENT, profile_start);
  return assignment;
}

AssignmentState ParallelTemperingSolver::DeriveAssignmentWires(const AssignmentState& assignment) const {
  AssignmentState derived = assignment;
  DeriveWires(&derived);
  return derived;
}

void ParallelTemperingSolver::SetMultipliers(const std::vector<double>& multipliers) {
  multipliers_ = multipliers;
  if (multipliers_.size() < circuit_.constraint_kinds.size()) multipliers_.resize(circuit_.constraint_kinds.size(), 1.0);
  UpdatePenaltyCache();
  RecomputeEnergies();
  ResetEnergyStats();
}

void ParallelTemperingSolver::SetTemperatures(const std::vector<double>& temperatures) {
  if (!temperatures.empty()) temperatures_ = temperatures;
  UpdateReplicaScales();
  ResetEnergyStats();
}

void ParallelTemperingSolver::SetConstraintClasses(const std::vector<int32_t>& classes) {
  circuit_.constraint_classes = classes;
  if (circuit_.constraint_classes.size() < circuit_.constraint_kinds.size()) circuit_.constraint_classes.resize(circuit_.constraint_kinds.size(), 3);
  UpdatePenaltyCache();
  RecomputeEnergies();
  ResetEnergyStats();
}

}  // namespace aes_xts_decoder
