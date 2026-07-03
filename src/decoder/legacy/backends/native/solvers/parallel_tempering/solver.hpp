#pragma once

#include "internal.hpp"
#include "circuit_model.hpp"
#include "byte_evaluator.hpp"
#include "hamming_scorer.hpp"
#include "replica_set.hpp"
#include "proposals.hpp"
#include "schedule.hpp"
#include "feasible_archive.hpp"
#include "diagnostics_collector.hpp"

#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace aes_xts_decoder {

class ParallelTemperingSolver {
 public:
  ParallelTemperingSolver(
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
  friend class DualUpdater;
  friend class LadderAdapter;

  ValueBuffer MakeValueBuffer() const;
  ValueBuffer EvaluateValues(const AssignmentState& assignment) const;
  ValueBuffer EvaluateValuesAndDerive(AssignmentState* assignment) const;
  ValueBuffer EvaluateValuesInternal(const AssignmentState& assignment, AssignmentState* derive_target, KeyCache* cache) const;
  void DeriveWires(AssignmentState* assignment) const;
  void EvaluateOpInto(std::size_t op_index, const AssignmentState& assignment, ValueBuffer* values, KeyCache* cache) const;
  ScoreData ScoreFromValues(const ValueBuffer& values) const;
  uint32_t ResidualForConstraint(const ValueBuffer& values, std::size_t constraint_index) const;
  ScoreData ScoreAssignmentAffected(const AssignmentState& assignment, const ScoreData& base, const std::vector<int32_t>& affected) const;
  ScoreData ScoreWireFlipAffected(std::size_t replica_index, const AssignmentState& assignment, const ScoreData& base, int32_t changed_value_id, std::vector<std::pair<int32_t, std::vector<uint8_t>>>* undo);
  void RestoreValues(std::size_t replica_index, const std::vector<std::pair<int32_t, std::vector<uint8_t>>>& undo);
  const uint8_t* ValuePtr(const ValueBuffer& values, int32_t value_id) const;
  uint8_t* MutableValuePtr(ValueBuffer* values, int32_t value_id) const;
  std::size_t ValueWidth(int32_t value_id) const;
  uint8_t ValueU8(const ValueBuffer& values, int32_t value_id) const;
  void SetValueU8(ValueBuffer* values, int32_t value_id, uint8_t value) const;
  void SetValueBytes(ValueBuffer* values, int32_t value_id, const uint8_t* data, std::size_t size) const;
  std::vector<uint8_t> CopyValueBytes(const ValueBuffer& values, int32_t value_id) const;
  AssignmentState Mutate(const AssignmentState& assignment, int* family, int32_t* changed_value_id, uint64_t* rng_state);
  bool RepairMove(const AssignmentState& assignment, AssignmentState* candidate, uint64_t* rng_state) const;
  void StepReplica(std::size_t replica_index);
  void AttemptSwaps();
  void DualUpdate();
  void MaybeHarvest(const AssignmentState& assignment, const ScoreData& score);
  void RecordAcceptedKeyStats(std::size_t replica_index);
  void RecordEnergyStats();
  void ResetEnergyStats();
  void UpdateReplicaFlow();
  void OptimizeTemperatureLadder();
  void ResetFlowCounters();
  std::vector<std::size_t> ComputeAlgebraCounts() const;
  std::vector<double> ComputeBpKeyMarginals() const;
  double PenaltyForConstraint(std::size_t constraint_index) const;
  void UpdatePenaltyCache();
  void UpdateReplicaScales();
  double EffectiveBeta(std::size_t replica_index) const;
  std::vector<double> EffectiveTemperatures() const;
  double KeyMarginalMaxDeviation() const;
  double KeyInformationBits() const;
  double KeyInformationNullBaselineBits() const;
  double MarginalEss() const;
  double MarginalRhat() const;
  double LogZEstimate() const;
  double Energy(const ScoreData& score) const;
  void RecomputeEnergies();
  std::vector<double> TemperatureLadder(std::size_t count, double t_min, double t_max) const;
  std::vector<double> InitialMultipliers(const PTConfig& config) const;
  uint64_t ProfileStart() const;
  void ProfileAdd(std::size_t counter, uint64_t started_at) const;
  std::vector<double> ProfileSeconds() const;

  CircuitModel circuit_;
  PTConfig config_;
  std::vector<AssignmentState> replicas_;
  std::vector<ScoreData> scores_;
  std::vector<double> energies_;
  std::vector<ValueBuffer> replica_values_;
  std::vector<double> temperatures_;
  std::vector<double> replica_scale_;
  std::vector<double> multipliers_;
  std::array<std::size_t, 512> key_ones_;
  std::size_t key_visit_count_;
  std::deque<std::array<uint8_t, 512>> key_bit_window_;
  std::deque<std::string> key_string_window_;
  std::unordered_map<std::string, std::size_t> key_window_counts_;
  std::deque<double> cold_key_hamming_trace_;
  static constexpr std::size_t kMaxColdTrace = 4096;
  std::vector<std::size_t> energy_sample_counts_;
  std::vector<double> energy_mean_by_rung_;
  std::vector<double> energy_m2_by_rung_;
  std::vector<double> energy_temperatures_by_rung_;
  std::vector<double> penalties_;
  std::vector<int> replica_directions_;
  std::vector<std::size_t> replica_round_trips_;
  std::vector<std::size_t> replica_up_counts_;
  std::vector<std::size_t> replica_down_counts_;
  std::size_t last_ladder_adaptation_epoch_;
  std::array<double, 3> rho_by_class_;
  std::array<double, 3> previous_violation_by_class_;
  std::array<std::size_t, 3> lambda_update_counts_by_class_;
  std::array<std::size_t, 3> rho_escalation_counts_by_class_;
  bool infeasibility_suspected_;
  std::vector<std::size_t> algebra_counts_;
  std::vector<double> bp_key_marginals_;
  bool bp_converged_;
  double langevin_seed_score_;
  std::vector<AssignmentState> feasible_;
  std::unordered_set<std::string> seen_keys_;
  std::vector<std::size_t> swap_attempts_;
  std::vector<std::size_t> swap_accepts_;
  std::vector<std::size_t> proposal_attempts_;
  std::vector<std::size_t> proposal_accepts_;
  std::vector<std::array<std::size_t, 4>> replica_proposal_attempts_;
  std::vector<std::array<std::size_t, 4>> replica_proposal_accepts_;
  std::vector<uint64_t> replica_rng_states_;
  std::vector<KeyCache> replica_key_caches_;
  std::vector<std::vector<uint32_t>> replica_value_marks_;
  std::vector<std::vector<uint32_t>> replica_constraint_marks_;
  std::vector<uint32_t> replica_mark_stamps_;
  mutable std::vector<uint64_t> profile_nanos_;
  mutable std::vector<std::size_t> profile_counts_;
  std::size_t thread_count_;
  std::size_t epochs_;
  std::size_t sweeps_;
  uint64_t rng_state_;
  ByteCircuitEvaluator evaluator_;
  HammingScorer scorer_;
  ProposalKernel proposal_kernel_;
  DualUpdater dual_updater_;
  LadderAdapter ladder_adapter_;
  FeasibleArchive feasible_archive_;
  DiagnosticsCollector diagnostics_collector_;
};

}  // namespace aes_xts_decoder
