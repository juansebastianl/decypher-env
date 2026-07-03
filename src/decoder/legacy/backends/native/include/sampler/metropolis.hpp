#pragma once

#include "../moves/proposal.hpp"
#include "../scoring/lagrangian.hpp"
#include "../scoring/scorer.hpp"
#include "../state/byte_state.hpp"

#include <random>

namespace aes_xts_decoder {

struct MetropolisStepResult {
  bool accepted;
  ScoreData score;
  double energy;
};

MetropolisStepResult RunMetropolisStep(
    ByteState* state,
    ProposalGenerator* proposals,
    const Scorer& scorer,
    const AugmentedLagrangian& lagrangian,
    const ScoreData& current_score,
    double current_energy,
    double beta,
    std::mt19937_64* rng);

}  // namespace aes_xts_decoder
