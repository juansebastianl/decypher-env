#pragma once

#include "../parallel_tempering/internal.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aes_xts_decoder {

struct SboxLiftSite {
  int32_t input_value = -1;
  int32_t sbox_value = -1;
  int32_t output_wire = -1;
  int32_t constraint_index = -1;
};

struct AlgebraicCircuitIndex {
  std::size_t value_count = 0;
  std::size_t linear_ops = 0;
  std::size_t nonlinear_ops = 0;
  std::size_t linear_constraints = 0;
  std::vector<SboxLiftSite> sbox_lifts;
};

struct Gf256SolveResult {
  bool consistent = false;
  std::size_t rank = 0;
  std::vector<uint8_t> solution;
};

AlgebraicCircuitIndex BuildAlgebraicCircuitIndex(
    std::size_t value_count,
    const std::vector<int32_t>& opcodes,
    const std::vector<int32_t>& outputs,
    const std::vector<int32_t>& input_offsets,
    const std::vector<int32_t>& input_counts,
    const std::vector<int32_t>& inputs,
    const std::vector<int32_t>& constraint_kinds,
    const std::vector<int32_t>& constraint_left,
    const std::vector<int32_t>& constraint_right);

uint8_t Gf256Mul(uint8_t a, uint8_t b);
uint8_t Gf256Inverse(uint8_t value);
uint8_t Gf256SboxAuxiliaryResidual(uint8_t x, uint8_t y);
Gf256SolveResult SolveGf256LinearSystem(
    std::vector<std::vector<uint8_t>> matrix,
    std::vector<uint8_t> rhs,
    std::size_t columns);

}  // namespace aes_xts_decoder
