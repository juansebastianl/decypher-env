// Solver SDK for the AES-XTS solver-authoring RL environment.
//
// This header is the stable public surface an authored solver compiles
// against. It is deliberately self-contained (only the C++ standard library)
// so a model-authored solver plugin can be compiled on demand by the harness
// with a single translation unit plus solver_sdk.cpp.
//
// Everything lives in namespace aes_xts_decoder::sdk so it never collides with
// the legacy in-tree engines that live in namespace aes_xts_decoder.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aes_xts_decoder {
namespace sdk {

// ---------------------------------------------------------------------------
// Search state and scores
// ---------------------------------------------------------------------------

// Mutable candidate buffers, mirroring the Python Assignment.
struct AssignmentState {
  std::vector<uint8_t> plaintext;
  std::vector<uint8_t> key1;
  std::vector<uint8_t> key2;
  std::vector<uint8_t> wires;  // internal lowered wires (may be empty)
};

// Hard (exact) residual scoring of an assignment. residuals[i] is the
// non-negative violation of constraint i in bits; feasible iff hamming==0.
struct ScoreData {
  uint32_t violations = 0;
  double hamming_score = 0.0;
  std::vector<uint32_t> residuals;
  std::vector<uint32_t> failing_indices;

  bool feasible() const { return hamming_score == 0.0; }
};

// ---------------------------------------------------------------------------
// Constraint / op opcodes (mirror circuit.py)
// ---------------------------------------------------------------------------

enum OpCode : int {
  OP_INPUT = 1,
  OP_CONST = 2,
  OP_XOR8 = 3,
  OP_XOR128 = 4,
  OP_AES_XTS_BLOCK = 5,
  OP_SBOX8 = 9,
  OP_MIX_COLUMN_BYTE = 10,
  OP_AES256_ROUND_KEY_BYTE = 11,
  OP_XTS_MUL_X_BYTE = 12,
  OP_AES256_BLOCK = 13,
  OP_EXTRACT_BYTE = 14,
};

enum ConstraintKind : int {
  CK_EQ8 = 1,
  CK_EQ128 = 2,
  CK_ASCII_PRINTABLE = 3,
  CK_DEFINE8 = 4,
};

enum ConstraintClass : int {
  CLASS_ASCII = 1,
  CLASS_CONSISTENCY = 2,
  CLASS_GOAL = 3,
};

// ---------------------------------------------------------------------------
// Flattened circuit (the same buffers the Cython bridge already exports)
// ---------------------------------------------------------------------------

struct CircuitBuffers {
  std::size_t value_count = 0;
  std::vector<int32_t> opcodes;
  std::vector<int32_t> outputs;
  std::vector<int32_t> input_offsets;
  std::vector<int32_t> input_counts;
  std::vector<int32_t> inputs;
  std::vector<int32_t> immediate_offsets;
  std::vector<int32_t> immediate_counts;
  std::vector<int32_t> immediates;
  std::vector<int32_t> const_offsets;
  std::vector<int32_t> const_counts;
  std::vector<uint8_t> constants;
  std::vector<int32_t> constraint_kinds;
  std::vector<int32_t> constraint_left;
  std::vector<int32_t> constraint_right;
  std::vector<int32_t> constraint_classes;
  std::vector<int32_t> constraint_rounds;  // per-constraint round index (-1 = prior)
  std::vector<int32_t> wire_value_ids;
  std::vector<int32_t> wire_offsets;
  std::vector<uint16_t> value_widths;
  std::size_t plaintext_start = 0;
  std::size_t plaintext_count = 0;
  std::size_t key1_start = 0;
  std::size_t key1_count = 0;
  std::size_t key2_start = 0;
  std::size_t key2_count = 0;
  std::vector<int32_t> xts_block_sectors;
  std::vector<int32_t> xts_block_indices;
  std::vector<int32_t> xts_block_offsets;  // plaintext offset per block
  std::vector<uint8_t> xts_block_targets;  // 16 bytes per block
  int aes_rounds = 14;
};

// ---------------------------------------------------------------------------
// Task configuration exposed to solvers
// ---------------------------------------------------------------------------

struct SolverConfig {
  uint64_t seed = 0;
  int aes_rounds = 14;
  // Per-class Lagrange multipliers (the linear term coefficients).
  double ascii_weight = 1.0;
  double consistency_weight = 1.0;
  double goal_weight = 1.0;
  // Per-class quadratic penalty coefficients (rho) of the augmented Lagrangian.
  // Energy = sum_i [ lambda_i * r_i + 1/2 * rho_i * r_i^2 ]. The environment
  // ramps these up across a curriculum (method of multipliers), tightening the
  // penalty on still-violated constraints as multipliers are updated.
  double ascii_rho = 0.0;
  double consistency_rho = 0.0;
  double goal_rho = 0.0;
  // Coupling multiplier applied to the inter-round consistency residuals. It
  // scales BOTH the consistency lambda and rho, so the ramp is exactly "how
  // hard rounds are forced to agree": from small (rounds drift independently
  // and are each solvable alone) to large (rounds forced to match).
  double coupling_weight = 1.0;
  std::size_t sweeps_budget = 0;  // 0 means "harness decides"
};

// ---------------------------------------------------------------------------
// SdkCircuit: evaluation, scoring, energy and repair helpers
// ---------------------------------------------------------------------------

class SdkCircuit {
 public:
  explicit SdkCircuit(CircuitBuffers buffers);

  const CircuitBuffers& buffers() const { return buffers_; }
  std::size_t constraint_count() const { return buffers_.constraint_kinds.size(); }
  std::size_t wire_count() const { return buffers_.wire_value_ids.size(); }
  int round_count() const { return round_count_; }

  // Populate assignment.wires from the current plaintext/keys by following the
  // DEFINE8 wire definitions (mirrors circuit.derive_wire_values).
  void DeriveWires(AssignmentState* assignment) const;

  // Exact residual score of an assignment.
  ScoreData Score(const AssignmentState& assignment) const;

  // Augmented-Lagrangian style energy: sum_i lambda_i r_i + 1/2 rho_i r_i^2,
  // with per-class weights and an elevated weight on inter-round consistency.
  double Energy(const ScoreData& score, const SolverConfig& config) const;

  // Sum of hard residual bits grouped by round index (index 0..round_count).
  // Bucket [round_count] collects prior/ungrouped constraints (e.g. ASCII).
  std::vector<double> PerRoundResiduals(const ScoreData& score) const;

  // Sum of hard residual bits grouped by constraint class, as a 3-element
  // vector indexed [ascii, consistency, goal] (i.e. CLASS_* minus one). This is
  // the per-class violation mass the environment's method-of-multipliers dual
  // update consumes to raise each class's Lagrange multiplier.
  std::vector<double> PerClassResiduals(const ScoreData& score) const;

  // Number of consecutive rounds (starting at round 1) whose consistency+goal
  // residuals are all zero, with a fractional term for the first unsolved
  // round. This is the dense "staircase" signal (1,2,3,...,n).
  double RoundStaircase(const ScoreData& score) const;

  // Exact block-Gibbs repair: rewrite each block's plaintext by decrypting the
  // target ciphertext under the assignment's keys, then re-derive wires. After
  // this the goal + consistency residuals for every block are zero.
  //
  // Pass derive_wires=false to skip the (relatively expensive) full wire
  // re-derivation. That is useful for exact key-conditional search: after a
  // repair the goal + consistency residuals are already zero, so the only thing
  // left to score is the ASCII penalty of the plaintext, which PlaintextAsciiPenalty
  // reads directly without touching the wires. Remember to repair once more with
  // derive_wires=true before committing a state you will Score().
  void RepairPlaintext(AssignmentState* assignment, bool derive_wires = true) const;

  // Number of plaintext bytes that are not printable text ASCII. After a repair
  // this equals the ASCII residual mass, so it is the cheap energy signal for
  // key-conditional (Gibbs/BP) moves that decrypt many candidate keys.
  std::size_t PlaintextAsciiPenalty(const AssignmentState& assignment) const;

  int ConstraintClassOf(std::size_t index) const;
  int ConstraintRoundOf(std::size_t index) const;

 private:
  void EvaluateValues(const AssignmentState& assignment,
                      std::vector<std::vector<uint8_t>>* values) const;
  std::vector<uint8_t> EvalOp(std::size_t op_index,
                              const AssignmentState& assignment,
                              const std::vector<std::vector<uint8_t>>& values) const;
  uint32_t ResidualForConstraint(std::size_t index,
                                 const std::vector<std::vector<uint8_t>>& values) const;
  const uint8_t* InputPtr(int32_t value_id, const AssignmentState& assignment) const;

  CircuitBuffers buffers_;
  int round_count_ = 0;
  // wire_offset_by_value[value_id] = offset into wires, or -1
  std::vector<int32_t> wire_offset_by_value_;
  // op_index_by_output[value_id] = producing op, or -1
  std::vector<int32_t> op_by_output_;
  // DEFINE8 constraints grouped by their right-hand producer value id.
  std::vector<std::vector<int32_t>> define_by_right_;
};

// ---------------------------------------------------------------------------
// Deterministic RNG utilities (splitmix64) for reproducible rollouts
// ---------------------------------------------------------------------------

uint64_t RngNext(uint64_t* state);
double RngUniform(uint64_t* state);
std::size_t RngRange(uint64_t* state, std::size_t limit);
uint8_t RandomTextAscii(uint64_t* state);
bool IsTextAscii(uint8_t value);

// ---------------------------------------------------------------------------
// The solver plugin ABI
// ---------------------------------------------------------------------------

// Read-only view of the task handed to a solver at construction time.
struct SolverContext {
  const SdkCircuit& circuit;
  const SolverConfig& config;
  AssignmentState initial;
};

// The interface an authored solver implements. The harness constructs one
// solver per rollout, advances it with RunEpoch, and reads results back.
class ISolver {
 public:
  virtual ~ISolver() = default;

  // Advance the search by `sweeps` local steps. Returns the score of the
  // solver's current (accepted) state.
  virtual ScoreData RunEpoch(std::size_t sweeps) = 0;

  // The best (lowest hard score) assignment discovered so far.
  virtual AssignmentState CurrentAssignment() const = 0;

  // Feasible (zero hard score) assignments discovered but not yet drained.
  virtual std::vector<AssignmentState> DrainFeasible(std::size_t limit) = 0;
};

// Optional convenience base class implementing the common bookkeeping
// (evaluation, augmented-Lagrangian energy, best-state tracking, feasible
// harvesting). Authored solvers typically override ProposeAndScore or Step.
class SolverBase : public ISolver {
 public:
  explicit SolverBase(const SolverContext& context);

  ScoreData RunEpoch(std::size_t sweeps) override;
  AssignmentState CurrentAssignment() const override { return best_; }
  std::vector<AssignmentState> DrainFeasible(std::size_t limit) override;

 protected:
  // Perform one search step, mutating current_ in place and returning true if
  // the step should be kept. Default implementation does a random local
  // mutation with a Metropolis acceptance test at temperature_.
  virtual bool Step();

  ScoreData ScoreOf(const AssignmentState& assignment) const;
  double EnergyOf(const ScoreData& score) const;
  void ConsiderFeasible(const AssignmentState& assignment, const ScoreData& score);

  const SdkCircuit& circuit_;
  SolverConfig config_;
  uint64_t rng_state_;
  double temperature_ = 1.0;

  AssignmentState current_;
  ScoreData current_score_;
  AssignmentState best_;
  ScoreData best_score_;
  std::vector<AssignmentState> feasible_;
};

}  // namespace sdk
}  // namespace aes_xts_decoder

// Every solver plugin exports this C entry point. The harness resolves it via
// dlsym after compiling the plugin into a shared object.
extern "C" aes_xts_decoder::sdk::ISolver* create_solver(
    const aes_xts_decoder::sdk::SolverContext& context);
