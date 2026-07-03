#include "../../include/sampler/metropolis.hpp"

#include <cmath>
#include <random>

namespace aes_xts_decoder {

MetropolisStepResult RunMetropolisStep(
    ByteState* state,
    ProposalGenerator* proposals,
    const Scorer& scorer,
    const AugmentedLagrangian& lagrangian,
    const ScoreData& current_score,
    double current_energy,
    double beta,
    std::mt19937_64* rng) {
  auto proposal = proposals->Generate(*state, rng);
  if (!proposal) {
    return MetropolisStepResult{false, current_score, current_energy};
  }

  proposal->Apply(state);
  const int32_t changed_wire = proposal->ChangedWire();
  const ScoreData candidate_score =
      changed_wire >= 0
          ? scorer.ScoreWireUpdate(
                state->view(), changed_wire, proposal->NewWireValue(), current_score)
          : scorer.Score(state->view());
  const double candidate_energy = lagrangian.Energy(candidate_score);
  const double log_accept = (current_energy - candidate_energy) * beta;

  bool accepted = log_accept >= 0.0;
  if (!accepted) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    accepted = std::log(uniform(*rng)) < log_accept;
  }

  if (accepted) {
    return MetropolisStepResult{true, candidate_score, candidate_energy};
  }

  proposal->Revert(state);
  return MetropolisStepResult{false, current_score, current_energy};
}

}  // namespace aes_xts_decoder
