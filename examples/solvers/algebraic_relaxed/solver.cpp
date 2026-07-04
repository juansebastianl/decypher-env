// algebraic_relaxed: a BP-seeded variant of the continuous-relaxed search.
//
// A thin layer over the continuous-relaxed search: it runs a cheap
// belief-propagation pass over the 64 key bytes and uses the resulting
// marginals to *seed* the starting keys, then hands off to the same
// parallel-tempering + key-Gibbs search. The "BP" is coordinate-wise: for each
// key byte it sweeps all 256 values, scores each by repairing the plaintext
// (exact XTS decrypt) and measuring the ASCII energy, and forms a damped
// softmax marginal. Sampling those marginals gives a start already biased toward
// keys that decrypt to printable text.
//
// This plugin does exactly that: a BP seed at construction, then the identical
// continuous-relaxed search. On easy (few-round) instances the seed is a real
// head start; on hard ones it degrades gracefully to the continuous engine.

#include "solver_sdk.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

using namespace aes_xts_decoder::sdk;

namespace {

constexpr std::size_t kReplicas = 2;
constexpr double kColdTemperature = 0.5;
constexpr double kHotTemperature = 8.0;
constexpr double kRepairProb = 0.15;
constexpr double kKeyGibbsProb = 0.5;
// Total exact key-Gibbs moves across the whole rollout (see continuous_relaxed).
constexpr int kGibbsBudget = 96;

// Belief-propagation seeding parameters (iterations kept low since each
// iteration costs 256 repairs per key byte).
constexpr int kBpIterations = 2;
constexpr double kBpDamping = 0.35;

std::string KeyString(const AssignmentState& a) {
  std::string key;
  key.reserve(a.key1.size() + a.key2.size());
  key.append(reinterpret_cast<const char*>(a.key1.data()), a.key1.size());
  key.append(reinterpret_cast<const char*>(a.key2.data()), a.key2.size());
  return key;
}

class AlgebraicRelaxedSolver : public ISolver {
 public:
  explicit AlgebraicRelaxedSolver(const SolverContext& context)
      : circuit_(context.circuit), config_(context.config) {
    has_blocks_ = !circuit_.buffers().xts_block_sectors.empty();
    gibbs_budget_ = kGibbsBudget;
    BuildLadder();
    AssignmentState seed = context.initial;
    if (has_blocks_) BeliefSeed(&seed, context.config.seed);
    InitReplicas(seed, context.config.seed);
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

 private:
  double Energy(const ScoreData& score) const { return circuit_.Energy(score, config_); }

  // ASCII penalty of the plaintext that results from repairing under the given
  // keys. After a repair the goal + consistency residuals are zero, so this
  // count of non-printable bytes is the energy signal (no wire re-derivation or
  // full circuit score needed -- the expensive part of a repair is skipped).
  double RepairedPenalty(AssignmentState* a) const {
    circuit_.RepairPlaintext(a, /*derive_wires=*/false);
    return static_cast<double>(circuit_.PlaintextAsciiPenalty(*a));
  }

  // Coordinate belief propagation over the key bytes, then sample a seed key.
  void BeliefSeed(AssignmentState* seed, uint64_t seed_value) {
    const std::size_t n1 = seed->key1.size();
    const std::size_t n2 = seed->key2.size();
    const std::size_t n = n1 + n2;
    if (n == 0) return;

    std::vector<std::array<double, 256>> beliefs(n);
    for (auto& b : beliefs) b.fill(1.0 / 256.0);

    AssignmentState work = *seed;
    auto key_byte = [&](std::size_t i) -> uint8_t& {
      return (i < n1) ? work.key1[i] : work.key2[i - n1];
    };

    for (int iter = 0; iter < kBpIterations; ++iter) {
      for (std::size_t b = 0; b < n; ++b) {
        std::array<double, 256> penalty{};
        double min_penalty = 1e300;
        for (int v = 0; v < 256; ++v) {
          key_byte(b) = static_cast<uint8_t>(v);
          double e = RepairedPenalty(&work);
          penalty[v] = e;
          if (e < min_penalty) min_penalty = e;
        }
        std::array<double, 256> updated{};
        double total = 0.0;
        for (int v = 0; v < 256; ++v) {
          updated[v] = std::exp(-(penalty[v] - min_penalty));
          total += updated[v];
        }
        for (int v = 0; v < 256; ++v) {
          double normalized = total > 0.0 ? updated[v] / total : 1.0 / 256.0;
          beliefs[b][v] = kBpDamping * beliefs[b][v] + (1.0 - kBpDamping) * normalized;
        }
        // Fix this byte at its current MAP estimate for subsequent coordinates.
        int best_v = 0;
        double best_p = -1.0;
        for (int v = 0; v < 256; ++v)
          if (beliefs[b][v] > best_p) { best_p = beliefs[b][v]; best_v = v; }
        key_byte(b) = static_cast<uint8_t>(best_v);
      }
    }

    uint64_t rng = seed_value ? seed_value + 0xA17E9E3779B97F4AULL : 0xA17E9E3779B97F4AULL;
    for (std::size_t b = 0; b < n; ++b) {
      double total = 0.0;
      for (int v = 0; v < 256; ++v) total += beliefs[b][v];
      double u = RngUniform(&rng) * total;
      double acc = 0.0;
      int chosen = 255;
      for (int v = 0; v < 256; ++v) {
        acc += beliefs[b][v];
        if (u <= acc) { chosen = v; break; }
      }
      if (b < n1) seed->key1[b] = static_cast<uint8_t>(chosen);
      else seed->key2[b - n1] = static_cast<uint8_t>(chosen);
    }
    circuit_.RepairPlaintext(seed);
  }

  void Commit(std::size_t r, AssignmentState&& candidate, const ScoreData& score) {
    replicas_[r] = std::move(candidate);
    scores_[r] = score;
    energies_[r] = Energy(score);
    if (score.hamming_score < best_score_.hamming_score) {
      best_ = replicas_[r];
      best_score_ = score;
    }
    if (score.feasible() && seen_keys_.insert(KeyString(replicas_[r])).second)
      feasible_.push_back(replicas_[r]);
  }

  void StepReplica(std::size_t r) {
    uint64_t* rng = &rng_[r];
    if (has_blocks_ && gibbs_budget_ > 0 && RngUniform(rng) < kKeyGibbsProb) {
      --gibbs_budget_;
      KeyProfileGibbs(r);
      return;
    }
    MetropolisStep(r);
  }

  void MetropolisStep(std::size_t r) {
    AssignmentState candidate = replicas_[r];
    uint64_t* rng = &rng_[r];
    if (has_blocks_ && RngUniform(rng) < kRepairProb) {
      circuit_.RepairPlaintext(&candidate);
    } else if (Mutate(&candidate, rng) && circuit_.wire_count() > 0) {
      circuit_.DeriveWires(&candidate);
    }

    ScoreData candidate_score = circuit_.Score(candidate);
    double delta = Energy(candidate_score) - Energy(scores_[r]);
    if (delta <= 0.0 || RngUniform(rng) < std::exp(-delta / temperatures_[r]))
      Commit(r, std::move(candidate), candidate_score);
  }

  void KeyProfileGibbs(std::size_t r) {
    uint64_t* rng = &rng_[r];
    bool use_key1 = RngUniform(rng) < 0.5 || replicas_[r].key2.empty();
    std::vector<uint8_t>& key = use_key1 ? replicas_[r].key1 : replicas_[r].key2;
    if (key.empty()) { MetropolisStep(r); return; }
    std::size_t byte = RngRange(rng, key.size());

    AssignmentState candidate = replicas_[r];
    std::vector<uint8_t>& ckey = use_key1 ? candidate.key1 : candidate.key2;

    std::vector<double> penalty(256, 0.0);
    double min_penalty = 1e300;
    for (int v = 0; v < 256; ++v) {
      ckey[byte] = static_cast<uint8_t>(v);
      circuit_.RepairPlaintext(&candidate, /*derive_wires=*/false);
      double p = static_cast<double>(circuit_.PlaintextAsciiPenalty(candidate));
      penalty[v] = p;
      if (p < min_penalty) min_penalty = p;
    }
    double beta = config_.ascii_weight / temperatures_[r];
    double total = 0.0;
    std::vector<double> weight(256, 0.0);
    for (int v = 0; v < 256; ++v) {
      weight[v] = std::exp(-beta * (penalty[v] - min_penalty));
      total += weight[v];
    }
    int chosen = 0;
    double u = RngUniform(rng) * total;
    double acc = 0.0;
    for (int v = 0; v < 256; ++v) {
      acc += weight[v];
      if (u <= acc) { chosen = v; break; }
    }
    ckey[byte] = static_cast<uint8_t>(chosen);
    circuit_.RepairPlaintext(&candidate);
    Commit(r, std::move(candidate), circuit_.Score(candidate));
  }

  // Returns true if plaintext/keys changed (wires need re-deriving); false for a
  // wire bit-flip, which is the move itself and must not be re-derived away.
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
    if (kReplicas == 1) { temperatures_[0] = kColdTemperature; return; }
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
      for (std::size_t m = 0; m < i; ++m) Mutate(&replicas_[i], &rng_[i]);
      if (i > 0 && circuit_.wire_count() > 0) circuit_.DeriveWires(&replicas_[i]);
      scores_[i] = circuit_.Score(replicas_[i]);
      energies_[i] = Energy(scores_[i]);
      if (scores_[i].hamming_score < best_score_.hamming_score) {
        best_ = replicas_[i];
        best_score_ = scores_[i];
      }
      if (scores_[i].feasible() && seen_keys_.insert(KeyString(replicas_[i])).second)
        feasible_.push_back(replicas_[i]);
    }
  }

  const SdkCircuit& circuit_;
  SolverConfig config_;
  bool has_blocks_ = false;
  int gibbs_budget_ = 0;

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
  return new AlgebraicRelaxedSolver(context);
}
