#include "solver.hpp"

namespace aes_xts_decoder {

AssignmentState ParallelTemperingSolver::Mutate(const AssignmentState& assignment, int* family, int32_t* changed_value_id, uint64_t* rng_state) {
  const uint64_t profile_start = ProfileStart();
  ProposalResult proposal = proposal_kernel_.Mutate(assignment, rng_state);
  *family = proposal.family;
  *changed_value_id = proposal.changed_value_id;
  ProfileAdd(PROFILE_MUTATE, profile_start);
  return proposal.candidate;
}

// Constraint-repair (Gibbs-style) move: set one block's plaintext to the exact
// AES-XTS decryption of its target ciphertext under the replica's current key.
// This zeros every goal constraint for that block at once and works for any
// round count, so feasibility reduces to finding a key whose decryption is ASCII.
bool ParallelTemperingSolver::RepairMove(const AssignmentState& assignment, AssignmentState* candidate, uint64_t* rng_state) const {
  return proposal_kernel_.RepairMove(assignment, candidate, rng_state);
}

void ParallelTemperingSolver::StepReplica(std::size_t replica_index) {
  const uint64_t profile_start = ProfileStart();
  int family = 0;
  int32_t changed_value_id = -1;
  AssignmentState candidate;
  bool repaired = false;
  if (config_.repair_move_prob > 0.0 && !circuit_.xts_block_sectors.empty() &&
      Uniform(&replica_rng_states_[replica_index]) < config_.repair_move_prob &&
      RepairMove(replicas_[replica_index], &candidate, &replica_rng_states_[replica_index])) {
    family = 1;  // accounted with the plaintext proposal family
    repaired = true;
  } else {
    candidate = Mutate(replicas_[replica_index], &family, &changed_value_id, &replica_rng_states_[replica_index]);
  }
  family = std::max(0, std::min(3, family));
  ++replica_proposal_attempts_[replica_index][family];
  std::vector<std::pair<int32_t, std::vector<uint8_t>>> undo;
  const bool used_delta = !repaired && family == 0 && changed_value_id >= 0 && static_cast<std::size_t>(changed_value_id) < circuit_.graph.constraints_by_value.size();
  ValueBuffer candidate_values;
  bool candidate_values_ready = false;
  ScoreData candidate_score;
  if (used_delta) {
    candidate_score = ScoreWireFlipAffected(replica_index, candidate, scores_[replica_index], changed_value_id, &undo);
  } else {
    const uint64_t full_score_profile_start = ProfileStart();
    candidate_values = EvaluateValuesAndDerive(&candidate);
    candidate_score = ScoreFromValues(candidate_values);
    candidate_values_ready = true;
    ProfileAdd(PROFILE_SCORE_FULL, full_score_profile_start);
  }
  const uint64_t energy_profile_start = ProfileStart();
  const double current_energy = replica_index < energies_.size() ? energies_[replica_index] : Energy(scores_[replica_index]);
  const double candidate_energy = Energy(candidate_score);
  bool accept = candidate_energy <= current_energy;
  if (!accept) {
    const double effective_beta = EffectiveBeta(replica_index);
    accept = Uniform(&replica_rng_states_[replica_index]) < std::exp((current_energy - candidate_energy) * effective_beta);
  }
  ProfileAdd(PROFILE_ENERGY_ACCEPT, energy_profile_start);
  if (accept) {
    replicas_[replica_index] = std::move(candidate);
    scores_[replica_index] = candidate_score;
    if (replica_index < energies_.size()) energies_[replica_index] = candidate_energy;
    if (!used_delta && candidate_values_ready) {
      replica_values_[replica_index] = std::move(candidate_values);
    }
    ++replica_proposal_accepts_[replica_index][family];
    RecordAcceptedKeyStats(replica_index);
    if (scores_[replica_index].hamming_score == 0.0) {
      #ifdef _OPENMP
      #pragma omp critical(feasible_harvest)
      #endif
      MaybeHarvest(replicas_[replica_index], scores_[replica_index]);
    }
  } else if (used_delta) {
    RestoreValues(replica_index, undo);
  }
  ProfileAdd(PROFILE_STEP_REPLICA_TOTAL, profile_start);
}

void ParallelTemperingSolver::AttemptSwaps() {
  const uint64_t profile_start = ProfileStart();
  ReplicaSet replica_set(
      &replicas_,
      &scores_,
      &energies_,
      &replica_values_,
      &replica_key_caches_,
      &replica_directions_,
      &replica_round_trips_);
  for (std::size_t i = 0; i + 1 < replica_set.size(); ++i) {
    ++swap_attempts_[i];
    const double left_energy = i < energies_.size() ? energies_[i] : Energy(scores_[i]);
    const double right_energy = (i + 1) < energies_.size() ? energies_[i + 1] : Energy(scores_[i + 1]);
    const double left_beta = EffectiveBeta(i);
    const double right_beta = EffectiveBeta(i + 1);
    const double log_accept = (left_beta - right_beta) * (left_energy - right_energy);
    if (log_accept >= 0.0 || Uniform(&rng_state_) < std::exp(log_accept)) {
      replica_set.SwapAdjacent(i);
      ++swap_accepts_[i];
    }
  }
  ProfileAdd(PROFILE_ATTEMPT_SWAPS, profile_start);
}

}  // namespace aes_xts_decoder
