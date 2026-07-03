// parallel_tempering: a port of the legacy native parallel-tempering engine to
// the ISolver plugin interface.
//
// It runs a population of replicas on a geometric temperature ladder. Each
// replica does augmented-Lagrangian Metropolis over the same four proposal
// families the legacy engine used (wire bit-flip, ASCII plaintext resample,
// key word-swap, key bit-flip), plus an occasional exact XTS plaintext repair.
// After every sweep, adjacent replicas attempt a replica-exchange swap so hot
// replicas (which explore) can feed good states down to cold replicas (which
// refine). Feasible states are harvested and de-duplicated by key.
//
// This is the faithful discrete engine; continuous_relaxed and algebraic_relaxed
// build on the same scaffold and add exact key-conditional (Gibbs) moves.

#include "solver_sdk.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

using namespace aes_xts_decoder::sdk;

namespace {

// Population size and temperature bounds mirror the legacy defaults, scaled down
// so a single rollout stays inside the harness wall-clock budget.
constexpr std::size_t kReplicas = 8;
constexpr double kColdTemperature = 0.5;
constexpr double kHotTemperature = 8.0;
// Probability of attempting an exact XTS block repair instead of a random move.
constexpr double kRepairProb = 0.15;

std::string KeyString(const AssignmentState& a) {
  std::string key;
  key.reserve(a.key1.size() + a.key2.size());
  key.append(reinterpret_cast<const char*>(a.key1.data()), a.key1.size());
  key.append(reinterpret_cast<const char*>(a.key2.data()), a.key2.size());
  return key;
}

class ParallelTemperingSolver : public ISolver {
 public:
  explicit ParallelTemperingSolver(const SolverContext& context)
      : circuit_(context.circuit), config_(context.config) {
    has_blocks_ = !circuit_.buffers().xts_block_sectors.empty();
    BuildLadder();
    InitReplicas(context.initial, context.config.seed);
  }

  ScoreData RunEpoch(std::size_t sweeps) override {
    for (std::size_t s = 0; s < sweeps; ++s) {
      for (std::size_t r = 0; r < replicas_.size(); ++r) StepReplica(r);
      AttemptSwaps();
    }
    return best_score_;
  }

  AssignmentState CurrentAssignment() const override { return best_; }

  std::vector<AssignmentState> DrainFeasible(std::size_t limit) override {
    if (limit == 0 || limit >= feasible_.size()) {
      std::vector<AssignmentState> out = std::move(feasible_);
      feasible_.clear();
      return out;
    }
    std::vector<AssignmentState> out(feasible_.begin(), feasible_.begin() + limit);
    feasible_.erase(feasible_.begin(), feasible_.begin() + limit);
    return out;
  }

 protected:
  double Energy(const ScoreData& score) const { return circuit_.Energy(score, config_); }

  void ConsiderBestAndFeasible(const AssignmentState& a, const ScoreData& score) {
    if (score.hamming_score < best_score_.hamming_score) {
      best_ = a;
      best_score_ = score;
    }
    if (score.feasible() && seen_keys_.insert(KeyString(a)).second) {
      feasible_.push_back(a);
    }
  }

  // One random local proposal on a replica, with Metropolis acceptance at the
  // replica's temperature. Overridden by the relaxed/algebraic variants to add
  // exact key-conditional moves.
  virtual void StepReplica(std::size_t r) {
    AssignmentState candidate = replicas_[r];
    uint64_t* rng = &rng_[r];

    if (has_blocks_ && RngUniform(rng) < kRepairProb) {
      circuit_.RepairPlaintext(&candidate);
    } else if (Mutate(&candidate, rng) && circuit_.wire_count() > 0) {
      // Only re-derive wires when plaintext/keys changed. A wire bit-flip is a
      // move in its own right (it perturbs an internal wire, changing that
      // DEFINE8 consistency residual); re-deriving would immediately undo it.
      circuit_.DeriveWires(&candidate);
    }

    ScoreData candidate_score = circuit_.Score(candidate);
    double delta = Energy(candidate_score) - Energy(scores_[r]);
    if (delta <= 0.0 || RngUniform(rng) < std::exp(-delta / temperatures_[r])) {
      replicas_[r] = std::move(candidate);
      scores_[r] = candidate_score;
      energies_[r] = Energy(candidate_score);
      ConsiderBestAndFeasible(replicas_[r], scores_[r]);
    }
  }

  // The legacy proposal mix: 65% wire bit-flip, 25% ASCII plaintext resample,
  // otherwise a key move (half word-swap, half single-bit flip). Returns true if
  // the plaintext or keys changed (so wires must be re-derived); false for a
  // wire bit-flip, which is itself the move and must not be re-derived away.
  bool Mutate(AssignmentState* a, uint64_t* rng) const {
    double roll = RngUniform(rng);
    if (!a->wires.empty() && roll < 0.65) {
      std::size_t off = RngRange(rng, a->wires.size());
      a->wires[off] ^= static_cast<uint8_t>(1u << RngRange(rng, 8));
      return false;
    } else if (roll < 0.90 && !a->plaintext.empty()) {
      a->plaintext[RngRange(rng, a->plaintext.size())] = RandomTextAscii(rng);
    } else {
      std::vector<uint8_t>& key = (RngUniform(rng) < 0.5) ? a->key1 : a->key2;
      if (key.size() >= 32 && RngUniform(rng) < 0.5) {
        std::size_t w0 = RngRange(rng, 8);
        std::size_t w1 = RngRange(rng, 8);
        if (w0 != w1)
          for (int b = 0; b < 4; ++b) std::swap(key[w0 * 4 + b], key[w1 * 4 + b]);
      } else if (!key.empty()) {
        std::size_t off = RngRange(rng, key.size());
        key[off] ^= static_cast<uint8_t>(1u << RngRange(rng, 8));
      }
    }
    return true;
  }

  // Replica exchange over adjacent pairs; the standard replica-exchange rule
  // with inverse temperatures beta_i = 1/T_i.
  void AttemptSwaps() {
    for (std::size_t i = 0; i + 1 < replicas_.size(); ++i) {
      double beta_i = 1.0 / temperatures_[i];
      double beta_j = 1.0 / temperatures_[i + 1];
      double log_accept = (beta_i - beta_j) * (energies_[i] - energies_[i + 1]);
      if (log_accept >= 0.0 || RngUniform(&swap_rng_) < std::exp(log_accept)) {
        std::swap(replicas_[i], replicas_[i + 1]);
        std::swap(scores_[i], scores_[i + 1]);
        std::swap(energies_[i], energies_[i + 1]);
      }
    }
  }

  void BuildLadder() {
    temperatures_.resize(kReplicas);
    if (kReplicas == 1) {
      temperatures_[0] = kColdTemperature;
      return;
    }
    double ratio = std::pow(kHotTemperature / kColdTemperature,
                            1.0 / static_cast<double>(kReplicas - 1));
    for (std::size_t i = 0; i < kReplicas; ++i)
      temperatures_[i] = kColdTemperature * std::pow(ratio, static_cast<double>(i));
  }

  void InitReplicas(const AssignmentState& initial, uint64_t seed) {
    replicas_.resize(kReplicas);
    scores_.resize(kReplicas);
    energies_.resize(kReplicas);
    rng_.resize(kReplicas);
    swap_rng_ = seed ? seed : 0x9E3779B97F4A7C15ULL;

    AssignmentState base = initial;
    if (circuit_.wire_count() > 0 && base.wires.size() != circuit_.wire_count())
      circuit_.DeriveWires(&base);
    best_ = base;
    best_score_ = circuit_.Score(base);

    for (std::size_t i = 0; i < kReplicas; ++i) {
      rng_[i] = (seed ? seed : 0x1234567890ABCDEFULL) + 0x9E3779B97F4A7C15ULL * (i + 1);
      replicas_[i] = base;
      // Perturb hotter replicas so the ladder starts diverse.
      for (std::size_t m = 0; m < i; ++m) Mutate(&replicas_[i], &rng_[i]);
      if (i > 0 && circuit_.wire_count() > 0) circuit_.DeriveWires(&replicas_[i]);
      scores_[i] = circuit_.Score(replicas_[i]);
      energies_[i] = Energy(scores_[i]);
      ConsiderBestAndFeasible(replicas_[i], scores_[i]);
    }
  }

  const SdkCircuit& circuit_;
  SolverConfig config_;
  bool has_blocks_ = false;

  std::vector<double> temperatures_;
  std::vector<AssignmentState> replicas_;
  std::vector<ScoreData> scores_;
  std::vector<double> energies_;
  std::vector<uint64_t> rng_;
  uint64_t swap_rng_ = 0;

  AssignmentState best_;
  ScoreData best_score_;
  std::vector<AssignmentState> feasible_;
  std::unordered_set<std::string> seen_keys_;
};

}  // namespace

extern "C" ISolver* create_solver(const SolverContext& context) {
  return new ParallelTemperingSolver(context);
}
