#pragma once

#include "circuit_model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aes_xts_decoder {

struct ValueBuffer {
  std::vector<uint8_t> bytes;
};

struct KeyCache {
  std::vector<uint8_t> key1;
  std::vector<uint8_t> key2;
  std::array<uint8_t, 240> expanded_key1;
  std::array<uint8_t, 240> expanded_key2;
};

class ByteCircuitEvaluator {
 public:
  explicit ByteCircuitEvaluator(const CircuitModel& circuit);

  ValueBuffer MakeValueBuffer() const;
  ValueBuffer EvaluateValues(const AssignmentState& assignment) const;
  ValueBuffer EvaluateValuesAndDerive(AssignmentState* assignment) const;
  ValueBuffer EvaluateValuesInternal(
      const AssignmentState& assignment,
      AssignmentState* derive_target,
      KeyCache* cache) const;
  void DeriveWires(AssignmentState* assignment) const;
  void EvaluateOpInto(
      std::size_t op_index,
      const AssignmentState& assignment,
      ValueBuffer* values,
      KeyCache* cache) const;

  const uint8_t* ValuePtr(const ValueBuffer& values, int32_t value_id) const;
  uint8_t* MutableValuePtr(ValueBuffer* values, int32_t value_id) const;
  std::size_t ValueWidth(int32_t value_id) const;
  uint8_t ValueU8(const ValueBuffer& values, int32_t value_id) const;
  void SetValueU8(ValueBuffer* values, int32_t value_id, uint8_t value) const;
  void SetValueBytes(ValueBuffer* values, int32_t value_id, const uint8_t* data, std::size_t size) const;
  std::vector<uint8_t> CopyValueBytes(const ValueBuffer& values, int32_t value_id) const;

 private:
  const std::array<uint8_t, 240>& ExpandedKey(
      const std::vector<uint8_t>& key,
      int selector,
      KeyCache* cache) const;

  const CircuitModel& circuit_;
};

}  // namespace aes_xts_decoder
