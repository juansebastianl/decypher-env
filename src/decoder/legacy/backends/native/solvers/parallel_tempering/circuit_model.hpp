#pragma once

#include "internal.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aes_xts_decoder {

struct GraphIndex {
  std::vector<std::size_t> value_offsets;
  std::vector<std::size_t> value_width_bytes;
  std::size_t value_storage_size = 0;
  std::vector<int32_t> wire_offset_by_value;
  std::vector<int32_t> wire_value_by_offset;
  std::vector<std::vector<int32_t>> definitions_by_right;
  std::vector<std::vector<int32_t>> constraints_by_value;
  std::vector<std::vector<int32_t>> ops_by_input;
};

struct CircuitModel {
  std::size_t value_count;
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
  std::vector<int32_t> wire_value_ids;
  std::vector<int32_t> wire_offsets;
  std::vector<int32_t> constraint_classes;
  std::size_t plaintext_start;
  std::size_t plaintext_count;
  std::size_t key1_start;
  std::size_t key1_count;
  std::size_t key2_start;
  std::size_t key2_count;
  std::vector<int32_t> xts_block_sectors;
  std::vector<int32_t> xts_block_indices;
  std::vector<std::array<uint8_t, 16>> xts_block_targets;
  GraphIndex graph;

  CircuitModel(
      std::size_t value_count,
      const std::vector<int32_t>& opcodes,
      const std::vector<int32_t>& outputs,
      const std::vector<int32_t>& input_offsets,
      const std::vector<int32_t>& input_counts,
      const std::vector<int32_t>& inputs,
      const std::vector<int32_t>& immediate_offsets,
      const std::vector<int32_t>& immediate_counts,
      const std::vector<int32_t>& immediates,
      const std::vector<int32_t>& const_offsets,
      const std::vector<int32_t>& const_counts,
      const std::vector<uint8_t>& constants,
      const std::vector<int32_t>& constraint_kinds,
      const std::vector<int32_t>& constraint_left,
      const std::vector<int32_t>& constraint_right,
      const std::vector<int32_t>& wire_value_ids,
      const std::vector<int32_t>& wire_offsets,
      const std::vector<int32_t>& constraint_classes,
      std::size_t plaintext_start,
      std::size_t plaintext_count,
      std::size_t key1_start,
      std::size_t key1_count,
      std::size_t key2_start,
      std::size_t key2_count,
      const std::vector<int32_t>& xts_block_sectors,
      const std::vector<int32_t>& xts_block_indices);
};

}  // namespace aes_xts_decoder
