// baseline: the smallest possible solver plugin.
//
// It implements the raw ISolver contract directly (no SolverBase helper) and
// performs no search at all -- it simply reports the initial assignment. Use it
// as the reference for the plugin ABI and as the reward floor: any real solver
// should score at least as well as this one.
//
// Build/run is handled by the harness (see examples/README.md). The only hard
// requirements are: include "solver_sdk.hpp" and export create_solver.

#include "solver_sdk.hpp"

using namespace aes_xts_decoder::sdk;

namespace {

class BaselineSolver : public ISolver {
 public:
  explicit BaselineSolver(const SolverContext& context)
      : circuit_(context.circuit), current_(context.initial) {
    if (circuit_.wire_count() > 0 && current_.wires.size() != circuit_.wire_count()) {
      circuit_.DeriveWires(&current_);
    }
    score_ = circuit_.Score(current_);
  }

  ScoreData RunEpoch(std::size_t /*sweeps*/) override { return score_; }

  AssignmentState CurrentAssignment() const override { return current_; }

  std::vector<AssignmentState> DrainFeasible(std::size_t /*limit*/) override {
    std::vector<AssignmentState> out;
    if (score_.feasible()) {
      out.push_back(current_);
      score_.hamming_score = 1.0;  // do not report the same feasible twice
    }
    return out;
  }

 private:
  const SdkCircuit& circuit_;
  AssignmentState current_;
  ScoreData score_;
};

}  // namespace

extern "C" ISolver* create_solver(const SolverContext& context) {
  return new BaselineSolver(context);
}
