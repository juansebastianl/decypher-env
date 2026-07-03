// random_metropolis: a real (if modest) solver built on SolverBase.
//
// It keeps SolverBase's default single-bit / ASCII-resample Metropolis step but
// adds two ideas worth copying:
//   1. a short geometric temperature schedule that cools across the run, and
//   2. an occasional exact block-Gibbs "repair" of the plaintext from the
//      current keys, which zeroes every goal + consistency residual in one shot
//      (feasibility then reduces to finding keys whose decryption is ASCII).
//
// This is intentionally simple; see examples/README.md for the menu of much
// stronger strategies (parallel tempering, continuous relaxation, algebraic
// / belief-propagation guidance) the interface also supports.

#include "solver_sdk.hpp"

#include <cmath>

using namespace aes_xts_decoder::sdk;

namespace {

class RandomMetropolisSolver : public SolverBase {
 public:
  explicit RandomMetropolisSolver(const SolverContext& context) : SolverBase(context) {
    has_blocks_ = !circuit_.buffers().xts_block_sectors.empty();
    temperature_ = kHotTemperature;
  }

  ScoreData RunEpoch(std::size_t sweeps) override {
    for (std::size_t i = 0; i < sweeps; ++i) {
      // Geometric cooling from hot to cold across the epoch.
      double frac = sweeps > 1 ? static_cast<double>(i) / static_cast<double>(sweeps - 1) : 1.0;
      temperature_ = kHotTemperature * std::pow(kColdTemperature / kHotTemperature, frac);

      // Occasionally repair the plaintext from the current keys.
      if (has_blocks_ && RngUniform(&rng_state_) < kRepairProb) {
        AssignmentState candidate = current_;
        circuit_.RepairPlaintext(&candidate);
        ScoreData candidate_score = ScoreOf(candidate);
        if (EnergyOf(candidate_score) <= EnergyOf(current_score_)) {
          current_ = std::move(candidate);
          current_score_ = candidate_score;
          if (current_score_.hamming_score < best_score_.hamming_score) {
            best_ = current_;
            best_score_ = current_score_;
          }
          ConsiderFeasible(current_, current_score_);
          continue;
        }
      }
      Step();
    }
    return current_score_;
  }

 private:
  static constexpr double kHotTemperature = 8.0;
  static constexpr double kColdTemperature = 0.4;
  static constexpr double kRepairProb = 0.1;
  bool has_blocks_ = false;
};

}  // namespace

extern "C" ISolver* create_solver(const SolverContext& context) {
  return new RandomMetropolisSolver(context);
}
