#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aes_xts_decoder {

struct PTConfig {
  std::size_t replicas;
  double t_min;
  double t_max;
  std::size_t sweeps_per_epoch;
  double mu;
  double dual_eta;
  double ascii_weight;
  double consistency_weight;
  double goal_weight;
  uint64_t seed;
  std::size_t threads;
  bool profile;
  int ladder_mode;
  std::size_t ladder_adapt_interval_epochs;
  std::size_t ladder_burn_in_epochs;
  std::size_t ladder_min_round_trips;
  int dual_mode;
  double scheduled_rho_initial;
  double scheduled_eta_tol;
  double scheduled_eta_target;
  double scheduled_rho_growth;
  double scheduled_violation_shrink;
  bool diagnostics;
  bool algebra_diagnostics;
  bool bp_diagnostics;
  bool alternative_diagnostics;
  std::size_t marginal_burn_in_epochs;
  std::size_t marginal_window;
  double marginal_alpha;
  std::size_t marginal_min_distinct_keys;
  double lambda_scale_cold;
  double lambda_scale_hot;
  int aes_rounds;
  double repair_move_prob;
  double key_gibbs_prob;
  std::size_t bp_iterations;
  double bp_damping;
  double bp_tolerance;
  double bp_proposal_weight;
  double bethe_weight;
  double algebraic_newton_prob;
  std::size_t survey_restarts;
};

struct PTDiagnostics {
  std::size_t epochs;
  std::size_t sweeps;
  std::size_t feasible_count;
  double total_residual;
  std::size_t violations;
  uint32_t max_residual;
  std::vector<std::size_t> swap_attempts;
  std::vector<std::size_t> swap_accepts;
  std::vector<std::size_t> proposal_attempts;
  std::vector<std::size_t> proposal_accepts;
  std::vector<double> residuals_by_class;
  std::vector<double> multiplier_mean_by_class;
  std::vector<double> multiplier_max_by_class;
  std::vector<std::size_t> graph_counts;
  std::vector<std::size_t> opcode_counts;
  std::size_t key_visit_count;
  std::size_t key_distinct_count;
  std::vector<std::size_t> key_ones;
  double key_marginal_max_deviation;
  double key_information_bits;
  double key_information_bits_raw;
  double key_information_null_bits;
  double marginal_ess;
  double marginal_rhat;
  bool marginal_trusted;
  std::vector<double> temperatures;
  std::vector<double> replica_lambda_scale;
  std::vector<std::size_t> replica_round_trips;
  std::vector<std::size_t> replica_up_counts;
  std::vector<std::size_t> replica_down_counts;
  std::size_t total_round_trips;
  std::size_t last_ladder_adaptation_epoch;
  std::vector<std::size_t> energy_sample_counts;
  std::vector<double> energy_mean_by_rung;
  std::vector<double> energy_variance_by_rung;
  std::vector<double> energy_temperatures_by_rung;
  double log_z_estimate;
  double log_feasible_count_estimate;
  bool log_z_estimate_available;
  std::size_t log_z_state_bits;
  std::vector<double> rho_by_class;
  std::vector<std::size_t> lambda_update_counts_by_class;
  std::vector<std::size_t> rho_escalation_counts_by_class;
  bool infeasibility_suspected;
  std::vector<std::size_t> algebra_counts;
  std::vector<double> bp_key_marginals;
  bool bp_converged;
  bool algebra_exact;
  bool bp_available;
  bool alternative_available;
  bool langevin_available;
  std::vector<double> alternative_log_z_estimates;
  double langevin_seed_score;
  double bethe_free_energy;
  double bp_residual;
  double bp_entropy;
  std::size_t survey_restarts;
  double survey_entropy;
  std::vector<double> profile_seconds;
  std::vector<std::size_t> profile_counts;
};

struct AssignmentState {
  std::vector<uint8_t> plaintext;
  std::vector<uint8_t> key1;
  std::vector<uint8_t> key2;
  std::vector<uint8_t> wires;
};

struct ScoreData {
  uint32_t violations;
  double hamming_score;
  std::vector<uint32_t> residuals;
  std::vector<uint32_t> failing_indices;
};

class ParallelTemperingSolver;

class ParallelTemperingEngine {
 public:
  ParallelTemperingEngine(
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
      const PTConfig& config);
  ~ParallelTemperingEngine();

  ParallelTemperingEngine(const ParallelTemperingEngine&) = delete;
  ParallelTemperingEngine& operator=(const ParallelTemperingEngine&) = delete;
  ParallelTemperingEngine(ParallelTemperingEngine&&) noexcept;
  ParallelTemperingEngine& operator=(ParallelTemperingEngine&&) noexcept;

  ScoreData ScoreAssignment(const AssignmentState& assignment) const;
  ScoreData RunEpoch(std::size_t sweeps);
  std::vector<uint32_t> Residuals() const;
  PTDiagnostics Metrics() const;
  std::vector<AssignmentState> DrainFeasible(std::size_t limit);
  AssignmentState CurrentAssignment() const;
  AssignmentState DeriveAssignmentWires(const AssignmentState& assignment) const;
  void SetMultipliers(const std::vector<double>& multipliers);
  void SetTemperatures(const std::vector<double>& temperatures);
  void SetConstraintClasses(const std::vector<int32_t>& classes);

 private:
  std::unique_ptr<ParallelTemperingSolver> impl_;
};

}  // namespace aes_xts_decoder
