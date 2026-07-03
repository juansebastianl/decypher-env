// Implementation of the self-contained solver SDK. See solver_sdk.hpp.
//
// The circuit evaluation and residual scoring intentionally mirror the Python
// reference (src/decoder/circuit.py and aes_primitives.py) byte-for-byte so the
// harness reward is cross-checkable against circuit.evaluate().

#include "solver_sdk.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace aes_xts_decoder {
namespace sdk {

namespace {

constexpr int AES_BLOCK_BYTES = 16;

const std::array<uint8_t, 256> kSbox = {
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16};

const std::array<uint8_t, 10> kRcon = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};

std::array<uint8_t, 256> MakeInvSbox() {
  std::array<uint8_t, 256> inv{};
  for (int i = 0; i < 256; ++i) inv[kSbox[i]] = static_cast<uint8_t>(i);
  return inv;
}
const std::array<uint8_t, 256> kInvSbox = MakeInvSbox();

uint8_t GfMul(uint8_t a, uint8_t b) {
  uint8_t result = 0;
  for (int i = 0; i < 8; ++i) {
    if (b & 1) result ^= a;
    uint8_t high = a & 0x80;
    a = static_cast<uint8_t>(a << 1);
    if (high) a ^= 0x1B;
    b >>= 1;
  }
  return result;
}

uint32_t SubWord(uint32_t word) {
  return (static_cast<uint32_t>(kSbox[(word >> 24) & 0xFF]) << 24) |
         (static_cast<uint32_t>(kSbox[(word >> 16) & 0xFF]) << 16) |
         (static_cast<uint32_t>(kSbox[(word >> 8) & 0xFF]) << 8) |
         static_cast<uint32_t>(kSbox[word & 0xFF]);
}

uint32_t RotWord(uint32_t word) { return (word << 8) | (word >> 24); }

// 15 round keys, 16 bytes each.
std::vector<std::array<uint8_t, 16>> ExpandKey256(const std::vector<uint8_t>& key) {
  std::vector<uint32_t> words(60, 0);
  for (int i = 0; i < 8; ++i) {
    words[i] = (static_cast<uint32_t>(key[4 * i]) << 24) |
               (static_cast<uint32_t>(key[4 * i + 1]) << 16) |
               (static_cast<uint32_t>(key[4 * i + 2]) << 8) |
               static_cast<uint32_t>(key[4 * i + 3]);
  }
  for (int i = 8; i < 60; ++i) {
    uint32_t temp = words[i - 1];
    if (i % 8 == 0) {
      temp = SubWord(RotWord(temp)) ^ (static_cast<uint32_t>(kRcon[(i / 8) - 1]) << 24);
    } else if (i % 8 == 4) {
      temp = SubWord(temp);
    }
    words[i] = words[i - 8] ^ temp;
  }
  std::vector<std::array<uint8_t, 16>> round_keys(15);
  for (int rk = 0; rk < 15; ++rk) {
    for (int w = 0; w < 4; ++w) {
      uint32_t word = words[rk * 4 + w];
      round_keys[rk][w * 4 + 0] = static_cast<uint8_t>((word >> 24) & 0xFF);
      round_keys[rk][w * 4 + 1] = static_cast<uint8_t>((word >> 16) & 0xFF);
      round_keys[rk][w * 4 + 2] = static_cast<uint8_t>((word >> 8) & 0xFF);
      round_keys[rk][w * 4 + 3] = static_cast<uint8_t>(word & 0xFF);
    }
  }
  return round_keys;
}

void ShiftRows(std::array<uint8_t, 16>& state) {
  std::array<uint8_t, 16> out{};
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 4; ++col) out[row + 4 * col] = state[row + 4 * ((col + row) % 4)];
  state = out;
}

void InvShiftRows(std::array<uint8_t, 16>& state) {
  std::array<uint8_t, 16> out{};
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 4; ++col) out[row + 4 * ((col + row) % 4)] = state[row + 4 * col];
  state = out;
}

void SubBytes(std::array<uint8_t, 16>& state) {
  for (auto& b : state) b = kSbox[b];
}
void InvSubBytes(std::array<uint8_t, 16>& state) {
  for (auto& b : state) b = kInvSbox[b];
}

void MixColumns(std::array<uint8_t, 16>& state) {
  std::array<uint8_t, 16> out = state;
  for (int col = 0; col < 4; ++col) {
    int i = 4 * col;
    uint8_t a0 = state[i], a1 = state[i + 1], a2 = state[i + 2], a3 = state[i + 3];
    out[i] = GfMul(a0, 2) ^ GfMul(a1, 3) ^ a2 ^ a3;
    out[i + 1] = a0 ^ GfMul(a1, 2) ^ GfMul(a2, 3) ^ a3;
    out[i + 2] = a0 ^ a1 ^ GfMul(a2, 2) ^ GfMul(a3, 3);
    out[i + 3] = GfMul(a0, 3) ^ a1 ^ a2 ^ GfMul(a3, 2);
  }
  state = out;
}

void InvMixColumns(std::array<uint8_t, 16>& state) {
  std::array<uint8_t, 16> out = state;
  for (int col = 0; col < 4; ++col) {
    int i = 4 * col;
    uint8_t a0 = state[i], a1 = state[i + 1], a2 = state[i + 2], a3 = state[i + 3];
    out[i] = GfMul(a0, 14) ^ GfMul(a1, 11) ^ GfMul(a2, 13) ^ GfMul(a3, 9);
    out[i + 1] = GfMul(a0, 9) ^ GfMul(a1, 14) ^ GfMul(a2, 11) ^ GfMul(a3, 13);
    out[i + 2] = GfMul(a0, 13) ^ GfMul(a1, 9) ^ GfMul(a2, 14) ^ GfMul(a3, 11);
    out[i + 3] = GfMul(a0, 11) ^ GfMul(a1, 13) ^ GfMul(a2, 9) ^ GfMul(a3, 14);
  }
  state = out;
}

std::array<uint8_t, 16> EncryptReduced(const std::vector<std::array<uint8_t, 16>>& rk,
                                       std::array<uint8_t, 16> state, int rounds) {
  for (int i = 0; i < 16; ++i) state[i] ^= rk[0][i];
  for (int r = 1; r < rounds; ++r) {
    SubBytes(state);
    ShiftRows(state);
    MixColumns(state);
    for (int i = 0; i < 16; ++i) state[i] ^= rk[r][i];
  }
  SubBytes(state);
  ShiftRows(state);
  for (int i = 0; i < 16; ++i) state[i] ^= rk[rounds][i];
  return state;
}

std::array<uint8_t, 16> DecryptReduced(const std::vector<std::array<uint8_t, 16>>& rk,
                                       std::array<uint8_t, 16> state, int rounds) {
  for (int i = 0; i < 16; ++i) state[i] ^= rk[rounds][i];
  InvShiftRows(state);
  InvSubBytes(state);
  for (int r = rounds - 1; r >= 1; --r) {
    for (int i = 0; i < 16; ++i) state[i] ^= rk[r][i];
    InvMixColumns(state);
    InvShiftRows(state);
    InvSubBytes(state);
  }
  for (int i = 0; i < 16; ++i) state[i] ^= rk[0][i];
  return state;
}

std::array<uint8_t, 16> XtsMulX(const std::array<uint8_t, 16>& tweak) {
  std::array<uint8_t, 16> out{};
  int carry = 0;
  for (int i = 0; i < 16; ++i) {
    int next_carry = (tweak[i] & 0x80) ? 1 : 0;
    out[i] = static_cast<uint8_t>((tweak[i] << 1) | carry);
    carry = next_carry;
  }
  if (carry) out[0] ^= 0x87;
  return out;
}

std::array<uint8_t, 16> SectorTweakInput(int64_t sector_number) {
  std::array<uint8_t, 16> out{};
  uint64_t value = static_cast<uint64_t>(sector_number);
  for (int i = 0; i < 8; ++i) out[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xFF);
  return out;
}

int Popcount8(uint8_t v) { return __builtin_popcount(static_cast<unsigned>(v)); }

}  // namespace

// ---------------------------------------------------------------------------
// SdkCircuit
// ---------------------------------------------------------------------------

SdkCircuit::SdkCircuit(CircuitBuffers buffers) : buffers_(std::move(buffers)) {
  round_count_ = buffers_.aes_rounds;
  wire_offset_by_value_.assign(buffers_.value_count, -1);
  for (std::size_t i = 0; i < buffers_.wire_value_ids.size(); ++i) {
    wire_offset_by_value_[buffers_.wire_value_ids[i]] = buffers_.wire_offsets[i];
  }
  op_by_output_.assign(buffers_.value_count, -1);
  for (std::size_t i = 0; i < buffers_.opcodes.size(); ++i) {
    op_by_output_[buffers_.outputs[i]] = static_cast<int32_t>(i);
  }
  define_by_right_.assign(buffers_.value_count, {});
  for (std::size_t i = 0; i < buffers_.constraint_kinds.size(); ++i) {
    if (buffers_.constraint_kinds[i] == CK_DEFINE8) {
      int32_t left = buffers_.constraint_left[i];
      int32_t right = buffers_.constraint_right[i];
      if (right >= 0 && wire_offset_by_value_[left] >= 0) {
        define_by_right_[right].push_back(left);
      }
    }
  }
}

const uint8_t* SdkCircuit::InputPtr(int32_t value_id, const AssignmentState& a) const {
  int32_t wire_off = wire_offset_by_value_[value_id];
  if (wire_off >= 0) return &a.wires[wire_off];
  std::size_t vid = static_cast<std::size_t>(value_id);
  if (vid >= buffers_.plaintext_start && vid < buffers_.plaintext_start + buffers_.plaintext_count)
    return &a.plaintext[vid - buffers_.plaintext_start];
  if (vid >= buffers_.key1_start && vid < buffers_.key1_start + buffers_.key1_count)
    return &a.key1[vid - buffers_.key1_start];
  if (vid >= buffers_.key2_start && vid < buffers_.key2_start + buffers_.key2_count)
    return &a.key2[vid - buffers_.key2_start];
  return nullptr;
}

std::vector<uint8_t> SdkCircuit::EvalOp(std::size_t op,
                                        const AssignmentState& a,
                                        const std::vector<std::vector<uint8_t>>& values) const {
  int opcode = buffers_.opcodes[op];
  int32_t in_off = buffers_.input_offsets[op];
  int32_t in_cnt = buffers_.input_counts[op];
  int32_t imm_off = buffers_.immediate_offsets[op];
  int32_t imm_cnt = buffers_.immediate_counts[op];
  int32_t out = buffers_.outputs[op];
  std::size_t width_bytes = std::max<std::size_t>(1, buffers_.value_widths[out] / 8);

  auto input_value = [&](int k) -> const std::vector<uint8_t>& {
    return values[buffers_.inputs[in_off + k]];
  };

  switch (opcode) {
    case OP_INPUT: {
      const uint8_t* p = InputPtr(out, a);
      std::vector<uint8_t> r(width_bytes, 0);
      if (p) for (std::size_t i = 0; i < width_bytes; ++i) r[i] = p[i];
      return r;
    }
    case OP_CONST: {
      int32_t cid = buffers_.immediates[imm_off];
      int32_t coff = buffers_.const_offsets[cid];
      int32_t ccnt = buffers_.const_counts[cid];
      return std::vector<uint8_t>(buffers_.constants.begin() + coff,
                                  buffers_.constants.begin() + coff + ccnt);
    }
    case OP_XOR8:
    case OP_XOR128: {
      const auto& l = input_value(0);
      const auto& r = input_value(1);
      std::vector<uint8_t> o(l.size(), 0);
      for (std::size_t i = 0; i < l.size() && i < r.size(); ++i) o[i] = l[i] ^ r[i];
      return o;
    }
    case OP_SBOX8:
      return {kSbox[input_value(0)[0]]};
    case OP_MIX_COLUMN_BYTE: {
      uint8_t value = 0;
      for (int k = 0; k < in_cnt && k < imm_cnt; ++k)
        value ^= GfMul(input_value(k)[0], static_cast<uint8_t>(buffers_.immediates[imm_off + k]));
      return {value};
    }
    case OP_AES256_ROUND_KEY_BYTE: {
      int sel = buffers_.immediates[imm_off];
      int round = buffers_.immediates[imm_off + 1];
      int byte = buffers_.immediates[imm_off + 2];
      const std::vector<uint8_t>& key = (sel == 1) ? a.key1 : a.key2;
      auto rk = ExpandKey256(key);
      return {rk[round][byte]};
    }
    case OP_XTS_MUL_X_BYTE: {
      int byte = buffers_.immediates[imm_off];
      std::array<uint8_t, 16> tweak{};
      for (int k = 0; k < AES_BLOCK_BYTES && k < in_cnt; ++k) tweak[k] = input_value(k)[0];
      return {XtsMulX(tweak)[byte]};
    }
    case OP_EXTRACT_BYTE: {
      int byte = buffers_.immediates[imm_off];
      return {input_value(0)[byte]};
    }
    default:
      // Opaque AES ops are unused by the lowered environment; return zeros.
      return std::vector<uint8_t>(width_bytes, 0);
  }
}

void SdkCircuit::EvaluateValues(const AssignmentState& a,
                                std::vector<std::vector<uint8_t>>* values) const {
  values->assign(buffers_.value_count, {});
  for (std::size_t op = 0; op < buffers_.opcodes.size(); ++op) {
    int32_t out = buffers_.outputs[op];
    (*values)[out] = EvalOp(op, a, *values);
  }
}

void SdkCircuit::DeriveWires(AssignmentState* a) const {
  if (a->wires.size() != wire_count()) a->wires.assign(wire_count(), 0);
  std::vector<std::vector<uint8_t>> values(buffers_.value_count);

  std::function<void(int32_t, const std::vector<uint8_t>&)> assign_wire =
      [&](int32_t value_id, const std::vector<uint8_t>& bytes) {
        int32_t off = wire_offset_by_value_[value_id];
        if (off < 0) return;
        a->wires[off] = bytes.empty() ? 0 : bytes[0];
        values[value_id] = bytes;
        for (int32_t dep : define_by_right_[value_id]) assign_wire(dep, bytes);
      };

  for (std::size_t op = 0; op < buffers_.opcodes.size(); ++op) {
    int32_t out = buffers_.outputs[op];
    values[out] = EvalOp(op, *a, values);
    for (int32_t left : define_by_right_[out]) assign_wire(left, values[out]);
  }
}

uint32_t SdkCircuit::ResidualForConstraint(std::size_t index,
                                           const std::vector<std::vector<uint8_t>>& values) const {
  int kind = buffers_.constraint_kinds[index];
  const std::vector<uint8_t>& left = values[buffers_.constraint_left[index]];
  if (kind == CK_ASCII_PRINTABLE) {
    return IsTextAscii(left.empty() ? 0 : left[0]) ? 0u : 1u;
  }
  int32_t right_id = buffers_.constraint_right[index];
  const std::vector<uint8_t>& right = values[right_id];
  if (kind == CK_DEFINE8) {
    return static_cast<uint32_t>(Popcount8(left[0] ^ right[0]));
  }
  uint32_t score = 0;
  for (std::size_t i = 0; i < left.size() && i < right.size(); ++i)
    score += static_cast<uint32_t>(Popcount8(left[i] ^ right[i]));
  return score;
}

ScoreData SdkCircuit::Score(const AssignmentState& a) const {
  std::vector<std::vector<uint8_t>> values;
  EvaluateValues(a, &values);
  ScoreData score;
  score.residuals.resize(constraint_count());
  for (std::size_t i = 0; i < constraint_count(); ++i) {
    uint32_t r = ResidualForConstraint(i, values);
    score.residuals[i] = r;
    if (r > 0) {
      score.violations += 1;
      score.hamming_score += r;
      score.failing_indices.push_back(static_cast<uint32_t>(i));
    }
  }
  return score;
}

int SdkCircuit::ConstraintClassOf(std::size_t index) const {
  if (index < buffers_.constraint_classes.size()) return buffers_.constraint_classes[index];
  return CLASS_GOAL;
}

int SdkCircuit::ConstraintRoundOf(std::size_t index) const {
  if (index < buffers_.constraint_rounds.size()) return buffers_.constraint_rounds[index];
  return 0;
}

double SdkCircuit::Energy(const ScoreData& score, const SolverConfig& config) const {
  // Augmented Lagrangian: sum_i [ lambda_i * r_i + 1/2 * rho_i * r_i^2 ].
  // lambda_i is the per-class Lagrange multiplier and rho_i the per-class
  // quadratic penalty; the coupling weight scales both terms of the inter-round
  // consistency class so the curriculum ramp tightens exactly that coupling.
  // A zero residual contributes nothing, so a feasible assignment has zero
  // energy regardless of the multipliers/penalties.
  double energy = 0.0;
  for (std::size_t i = 0; i < score.residuals.size(); ++i) {
    double r = static_cast<double>(score.residuals[i]);
    if (r == 0.0) continue;
    double lambda;
    double rho;
    switch (ConstraintClassOf(i)) {
      case CLASS_ASCII:
        lambda = config.ascii_weight;
        rho = config.ascii_rho;
        break;
      case CLASS_CONSISTENCY:
        lambda = config.consistency_weight * config.coupling_weight;
        rho = config.consistency_rho * config.coupling_weight;
        break;
      default:
        lambda = config.goal_weight;
        rho = config.goal_rho;
        break;
    }
    energy += lambda * r + 0.5 * rho * r * r;
  }
  return energy;
}

std::vector<double> SdkCircuit::PerRoundResiduals(const ScoreData& score) const {
  std::vector<double> buckets(static_cast<std::size_t>(round_count_) + 1, 0.0);
  for (std::size_t i = 0; i < score.residuals.size(); ++i) {
    if (score.residuals[i] == 0) continue;
    int round = ConstraintRoundOf(i);
    if (round < 0) round = 0;
    if (round > round_count_) round = round_count_;
    buckets[static_cast<std::size_t>(round)] += score.residuals[i];
  }
  return buckets;
}

std::vector<double> SdkCircuit::PerClassResiduals(const ScoreData& score) const {
  std::vector<double> buckets(3, 0.0);  // [ascii, consistency, goal]
  for (std::size_t i = 0; i < score.residuals.size(); ++i) {
    if (score.residuals[i] == 0) continue;
    int cls = ConstraintClassOf(i);
    std::size_t bucket = (cls >= CLASS_ASCII && cls <= CLASS_GOAL)
                             ? static_cast<std::size_t>(cls - CLASS_ASCII)
                             : 2;  // default to goal bucket
    buckets[bucket] += score.residuals[i];
  }
  return buckets;
}

double SdkCircuit::RoundStaircase(const ScoreData& score) const {
  if (round_count_ <= 0) return score.feasible() ? 1.0 : 0.0;
  std::vector<double> buckets = PerRoundResiduals(score);
  int solved = 0;
  for (int r = 1; r <= round_count_; ++r) {
    if (buckets[static_cast<std::size_t>(r)] == 0.0)
      solved += 1;
    else
      break;
  }
  double fractional = 0.0;
  if (solved < round_count_) {
    // Diminishing partial credit in [0,1) for the first unsolved round:
    // fewer residual bits -> closer to 1.
    double residual = buckets[static_cast<std::size_t>(solved + 1)];
    fractional = 1.0 / (1.0 + residual);
  }
  return static_cast<double>(solved) + fractional;
}

void SdkCircuit::RepairPlaintext(AssignmentState* a, bool derive_wires) const {
  int rounds = buffers_.aes_rounds;
  auto rk1 = ExpandKey256(a->key1);
  auto rk2 = ExpandKey256(a->key2);
  for (std::size_t b = 0; b < buffers_.xts_block_sectors.size(); ++b) {
    int32_t sector = buffers_.xts_block_sectors[b];
    int32_t block_index = buffers_.xts_block_indices[b];
    int32_t offset = buffers_.xts_block_offsets[b];
    std::array<uint8_t, 16> tweak = EncryptReduced(rk2, SectorTweakInput(sector), rounds);
    for (int j = 0; j < block_index; ++j) tweak = XtsMulX(tweak);
    std::array<uint8_t, 16> ct{};
    for (int i = 0; i < 16; ++i) ct[i] = buffers_.xts_block_targets[b * 16 + i] ^ tweak[i];
    std::array<uint8_t, 16> pt = DecryptReduced(rk1, ct, rounds);
    for (int i = 0; i < 16; ++i)
      a->plaintext[static_cast<std::size_t>(offset) + i] = pt[i] ^ tweak[i];
  }
  if (derive_wires) DeriveWires(a);
}

std::size_t SdkCircuit::PlaintextAsciiPenalty(const AssignmentState& a) const {
  std::size_t penalty = 0;
  for (uint8_t byte : a.plaintext)
    if (!IsTextAscii(byte)) ++penalty;
  return penalty;
}

// ---------------------------------------------------------------------------
// RNG helpers (splitmix64)
// ---------------------------------------------------------------------------

uint64_t RngNext(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
double RngUniform(uint64_t* state) {
  return static_cast<double>(RngNext(state) >> 11) / static_cast<double>(1ULL << 53);
}
std::size_t RngRange(uint64_t* state, std::size_t limit) {
  if (limit == 0) return 0;
  return static_cast<std::size_t>(RngNext(state) % limit);
}
bool IsTextAscii(uint8_t v) {
  return v == 0x09 || v == 0x0A || v == 0x0D || (v >= 0x20 && v <= 0x7E);
}
uint8_t RandomTextAscii(uint64_t* state) {
  std::size_t choice = RngRange(state, 98);
  if (choice == 0) return 0x09;
  if (choice == 1) return 0x0A;
  if (choice == 2) return 0x0D;
  return static_cast<uint8_t>(0x20 + (choice - 3));
}

// ---------------------------------------------------------------------------
// SolverBase
// ---------------------------------------------------------------------------

SolverBase::SolverBase(const SolverContext& context)
    : circuit_(context.circuit),
      config_(context.config),
      rng_state_(context.config.seed ? context.config.seed : 0x9E3779B97F4A7C15ULL),
      current_(context.initial) {
  if (circuit_.wire_count() > 0 && current_.wires.size() != circuit_.wire_count()) {
    circuit_.DeriveWires(&current_);
  }
  current_score_ = ScoreOf(current_);
  best_ = current_;
  best_score_ = current_score_;
  ConsiderFeasible(current_, current_score_);
}

ScoreData SolverBase::ScoreOf(const AssignmentState& a) const { return circuit_.Score(a); }
double SolverBase::EnergyOf(const ScoreData& s) const { return circuit_.Energy(s, config_); }

void SolverBase::ConsiderFeasible(const AssignmentState& a, const ScoreData& score) {
  if (score.feasible()) feasible_.push_back(a);
}

bool SolverBase::Step() {
  AssignmentState candidate = current_;
  double u = RngUniform(&rng_state_);
  bool touched_non_wire = false;
  if (!candidate.wires.empty() && u < 0.5) {
    std::size_t off = RngRange(&rng_state_, candidate.wires.size());
    candidate.wires[off] ^= static_cast<uint8_t>(1u << RngRange(&rng_state_, 8));
  } else if (u < 0.75 && !candidate.plaintext.empty()) {
    std::size_t off = RngRange(&rng_state_, candidate.plaintext.size());
    candidate.plaintext[off] = RandomTextAscii(&rng_state_);
    touched_non_wire = true;
  } else {
    std::vector<uint8_t>& key = (RngUniform(&rng_state_) < 0.5) ? candidate.key1 : candidate.key2;
    if (!key.empty()) {
      std::size_t off = RngRange(&rng_state_, key.size());
      key[off] ^= static_cast<uint8_t>(1u << RngRange(&rng_state_, 8));
      touched_non_wire = true;
    }
  }
  if (touched_non_wire && circuit_.wire_count() > 0) circuit_.DeriveWires(&candidate);

  ScoreData candidate_score = ScoreOf(candidate);
  double delta = EnergyOf(candidate_score) - EnergyOf(current_score_);
  bool accept = delta <= 0.0 || RngUniform(&rng_state_) < std::exp(-delta / temperature_);
  if (accept) {
    current_ = std::move(candidate);
    current_score_ = candidate_score;
    if (current_score_.hamming_score < best_score_.hamming_score) {
      best_ = current_;
      best_score_ = current_score_;
    }
    ConsiderFeasible(current_, current_score_);
  }
  return accept;
}

ScoreData SolverBase::RunEpoch(std::size_t sweeps) {
  for (std::size_t i = 0; i < sweeps; ++i) Step();
  return current_score_;
}

std::vector<AssignmentState> SolverBase::DrainFeasible(std::size_t limit) {
  if (limit == 0 || limit >= feasible_.size()) {
    std::vector<AssignmentState> out = std::move(feasible_);
    feasible_.clear();
    return out;
  }
  std::vector<AssignmentState> out(feasible_.begin(), feasible_.begin() + limit);
  feasible_.erase(feasible_.begin(), feasible_.begin() + limit);
  return out;
}

}  // namespace sdk
}  // namespace aes_xts_decoder
