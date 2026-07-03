#pragma once

#include <cstddef>
#include <cstdint>

namespace aes_xts_decoder {

struct Int32View {
  const int32_t* data;
  std::size_t size;
};

struct UInt16View {
  const uint16_t* data;
  std::size_t size;
};

struct BytesView {
  uint8_t* data;
  std::size_t size;
};

struct CircuitView {
  Int32View op_rows;
  Int32View constraint_rows;
  UInt16View value_widths;
};

struct AssignmentView {
  BytesView plaintext;
  BytesView key1;
  BytesView key2;
  BytesView wires;
};

struct Score {
  uint32_t violations;
  uint32_t hamming_score;
  const uint32_t* residuals;
  std::size_t residual_count;
  const uint32_t* failing_indices;
  std::size_t failing_index_count;
};

Score score_assignment(const CircuitView& circuit, const AssignmentView& assignment);

}  // namespace aes_xts_decoder
