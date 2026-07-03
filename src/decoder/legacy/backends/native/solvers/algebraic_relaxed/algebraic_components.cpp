#include "algebraic_components.hpp"

#include <algorithm>
#include <unordered_map>

namespace aes_xts_decoder {
namespace {

bool IsGfLinearOpcode(int opcode) {
  return opcode == OP_INPUT || opcode == OP_CONST || opcode == OP_XOR8 || opcode == OP_XOR128 ||
         opcode == OP_MIX_COLUMN_BYTE || opcode == OP_XTS_MUL_X_BYTE || opcode == OP_EXTRACT_BYTE;
}

bool IsLinearConstraint(int kind) {
  return kind == CONSTRAINT_EQ8 || kind == CONSTRAINT_EQ128 || kind == CONSTRAINT_DEFINE8;
}

}  // namespace

AlgebraicCircuitIndex BuildAlgebraicCircuitIndex(
    std::size_t value_count,
    const std::vector<int32_t>& opcodes,
    const std::vector<int32_t>& outputs,
    const std::vector<int32_t>& input_offsets,
    const std::vector<int32_t>& input_counts,
    const std::vector<int32_t>& inputs,
    const std::vector<int32_t>& constraint_kinds,
    const std::vector<int32_t>& constraint_left,
    const std::vector<int32_t>& constraint_right) {
  AlgebraicCircuitIndex index;
  index.value_count = value_count;
  for (const int opcode : opcodes) {
    if (IsGfLinearOpcode(opcode)) {
      ++index.linear_ops;
    } else {
      ++index.nonlinear_ops;
    }
  }
  for (const int kind : constraint_kinds) {
    if (IsLinearConstraint(kind)) ++index.linear_constraints;
  }

  std::unordered_map<int32_t, std::pair<int32_t, int32_t>> define_by_right;
  for (std::size_t ci = 0; ci < constraint_kinds.size(); ++ci) {
    if (constraint_kinds[ci] != CONSTRAINT_DEFINE8 || ci >= constraint_right.size() || ci >= constraint_left.size()) {
      continue;
    }
    const int32_t right = constraint_right[ci];
    if (right < 0) continue;
    define_by_right.emplace(right, std::make_pair(constraint_left[ci], static_cast<int32_t>(ci)));
  }

  for (std::size_t op = 0; op < opcodes.size(); ++op) {
    if (opcodes[op] != OP_SBOX8 || op >= outputs.size() || op >= input_offsets.size() || op >= input_counts.size()) {
      continue;
    }
    if (input_counts[op] <= 0) continue;
    const std::size_t input_offset = static_cast<std::size_t>(std::max<int32_t>(0, input_offsets[op]));
    if (input_offset >= inputs.size()) continue;
    SboxLiftSite site;
    site.input_value = inputs[input_offset];
    site.sbox_value = outputs[op];
    const auto found = define_by_right.find(site.sbox_value);
    if (found != define_by_right.end()) {
      site.output_wire = found->second.first;
      site.constraint_index = found->second.second;
    }
    index.sbox_lifts.push_back(site);
  }
  return index;
}

uint8_t Gf256Mul(uint8_t a, uint8_t b) {
  uint8_t result = 0;
  for (int i = 0; i < 8; ++i) {
    if (b & 1) result ^= a;
    const bool high = (a & 0x80) != 0;
    a <<= 1;
    if (high) a ^= 0x1B;
    b >>= 1;
  }
  return result;
}

uint8_t Gf256Inverse(uint8_t value) {
  if (value == 0) return 0;
  for (int candidate = 1; candidate < 256; ++candidate) {
    if (Gf256Mul(value, static_cast<uint8_t>(candidate)) == 1) {
      return static_cast<uint8_t>(candidate);
    }
  }
  return 0;
}

uint8_t Gf256SboxAuxiliaryResidual(uint8_t x, uint8_t y) {
  if (x == 0 && y == 0) return 0;
  return static_cast<uint8_t>(Gf256Mul(x, y) ^ 0x01);
}

Gf256SolveResult SolveGf256LinearSystem(
    std::vector<std::vector<uint8_t>> matrix,
    std::vector<uint8_t> rhs,
    std::size_t columns) {
  Gf256SolveResult result;
  const std::size_t rows = matrix.size();
  if (rhs.size() != rows) return result;
  for (std::vector<uint8_t>& row : matrix) row.resize(columns, 0);

  std::vector<std::size_t> pivot_for_row;
  std::size_t row = 0;
  for (std::size_t col = 0; col < columns && row < rows; ++col) {
    std::size_t pivot = row;
    while (pivot < rows && matrix[pivot][col] == 0) ++pivot;
    if (pivot == rows) continue;
    if (pivot != row) {
      std::swap(matrix[pivot], matrix[row]);
      std::swap(rhs[pivot], rhs[row]);
    }

    const uint8_t inv = Gf256Inverse(matrix[row][col]);
    for (std::size_t c = col; c < columns; ++c) matrix[row][c] = Gf256Mul(matrix[row][c], inv);
    rhs[row] = Gf256Mul(rhs[row], inv);

    for (std::size_t other = 0; other < rows; ++other) {
      if (other == row) continue;
      const uint8_t factor = matrix[other][col];
      if (factor == 0) continue;
      for (std::size_t c = col; c < columns; ++c) {
        matrix[other][c] ^= Gf256Mul(factor, matrix[row][c]);
      }
      rhs[other] ^= Gf256Mul(factor, rhs[row]);
    }

    pivot_for_row.push_back(col);
    ++row;
  }

  for (std::size_t r = row; r < rows; ++r) {
    bool any = false;
    for (std::size_t c = 0; c < columns; ++c) any = any || matrix[r][c] != 0;
    if (!any && rhs[r] != 0) {
      result.consistent = false;
      result.rank = row;
      return result;
    }
  }

  result.solution.assign(columns, 0);
  for (std::size_t r = 0; r < pivot_for_row.size(); ++r) {
    result.solution[pivot_for_row[r]] = rhs[r];
  }
  result.rank = pivot_for_row.size();
  result.consistent = true;
  return result;
}

}  // namespace aes_xts_decoder
