#include "byte_evaluator.hpp"

#include <algorithm>
#include <functional>

namespace aes_xts_decoder {

ByteCircuitEvaluator::ByteCircuitEvaluator(const CircuitModel& circuit) : circuit_(circuit) {}

ValueBuffer ByteCircuitEvaluator::MakeValueBuffer() const {
  return ValueBuffer{std::vector<uint8_t>(circuit_.graph.value_storage_size, 0)};
}

const uint8_t* ByteCircuitEvaluator::ValuePtr(const ValueBuffer& values, int32_t value_id) const {
  return values.bytes.data() + circuit_.graph.value_offsets[value_id];
}

uint8_t* ByteCircuitEvaluator::MutableValuePtr(ValueBuffer* values, int32_t value_id) const {
  return values->bytes.data() + circuit_.graph.value_offsets[value_id];
}

std::size_t ByteCircuitEvaluator::ValueWidth(int32_t value_id) const {
  return circuit_.graph.value_width_bytes[value_id];
}

uint8_t ByteCircuitEvaluator::ValueU8(const ValueBuffer& values, int32_t value_id) const {
  return ValuePtr(values, value_id)[0];
}

void ByteCircuitEvaluator::SetValueU8(ValueBuffer* values, int32_t value_id, uint8_t value) const {
  MutableValuePtr(values, value_id)[0] = value;
}

void ByteCircuitEvaluator::SetValueBytes(ValueBuffer* values, int32_t value_id, const uint8_t* data, std::size_t size) const {
  const std::size_t width = std::min(size, ValueWidth(value_id));
  std::copy(data, data + width, MutableValuePtr(values, value_id));
}

std::vector<uint8_t> ByteCircuitEvaluator::CopyValueBytes(const ValueBuffer& values, int32_t value_id) const {
  const uint8_t* begin = ValuePtr(values, value_id);
  return std::vector<uint8_t>(begin, begin + ValueWidth(value_id));
}

ValueBuffer ByteCircuitEvaluator::EvaluateValues(const AssignmentState& assignment) const {
  KeyCache cache;
  return EvaluateValuesInternal(assignment, nullptr, &cache);
}

ValueBuffer ByteCircuitEvaluator::EvaluateValuesAndDerive(AssignmentState* assignment) const {
  KeyCache cache;
  return EvaluateValuesInternal(*assignment, assignment, &cache);
}

void ByteCircuitEvaluator::EvaluateOpInto(
    std::size_t op,
    const AssignmentState& assignment,
    ValueBuffer* values,
    KeyCache* cache) const {
  const int opcode = circuit_.opcodes[op];
  const int out = circuit_.outputs[op];
  const int input_offset = circuit_.input_offsets[op];
  const int input_count = circuit_.input_counts[op];
  const int imm_offset = circuit_.immediate_offsets[op];
  if (opcode == OP_INPUT) {
    if (out >= static_cast<int>(circuit_.plaintext_start) && out < static_cast<int>(circuit_.plaintext_start + circuit_.plaintext_count)) {
      SetValueU8(values, out, assignment.plaintext[out - circuit_.plaintext_start]);
    } else if (out >= static_cast<int>(circuit_.key1_start) && out < static_cast<int>(circuit_.key1_start + circuit_.key1_count)) {
      SetValueU8(values, out, assignment.key1[out - circuit_.key1_start]);
    } else if (out >= static_cast<int>(circuit_.key2_start) && out < static_cast<int>(circuit_.key2_start + circuit_.key2_count)) {
      SetValueU8(values, out, assignment.key2[out - circuit_.key2_start]);
    } else {
      const int32_t wire_offset = circuit_.graph.wire_offset_by_value[out];
      SetValueU8(values, out, assignment.wires[wire_offset]);
    }
  } else if (opcode == OP_CONST) {
    const int const_id = circuit_.immediates[imm_offset];
    const int offset = circuit_.const_offsets[const_id];
    const int count = circuit_.const_counts[const_id];
    SetValueBytes(values, out, circuit_.constants.data() + offset, static_cast<std::size_t>(count));
  } else if (opcode == OP_XOR8) {
    SetValueU8(values, out, static_cast<uint8_t>(ValueU8(*values, circuit_.inputs[input_offset]) ^ ValueU8(*values, circuit_.inputs[input_offset + 1])));
  } else if (opcode == OP_XOR128) {
    const uint8_t* left = ValuePtr(*values, circuit_.inputs[input_offset]);
    const uint8_t* right = ValuePtr(*values, circuit_.inputs[input_offset + 1]);
    uint8_t* out_value = MutableValuePtr(values, out);
    for (std::size_t i = 0; i < ValueWidth(out); ++i) {
      out_value[i] = left[i] ^ right[i];
    }
  } else if (opcode == OP_SBOX8) {
    SetValueU8(values, out, SBOX[ValueU8(*values, circuit_.inputs[input_offset])]);
  } else if (opcode == OP_MIX_COLUMN_BYTE) {
    uint8_t value = 0;
    for (int i = 0; i < input_count; ++i) {
      value ^= MixMul(ValueU8(*values, circuit_.inputs[input_offset + i]), circuit_.immediates[imm_offset + i]);
    }
    SetValueU8(values, out, value);
  } else if (opcode == OP_AES256_ROUND_KEY_BYTE) {
    const int key_selector = circuit_.immediates[imm_offset];
    const int round_index = circuit_.immediates[imm_offset + 1];
    const int byte_index = circuit_.immediates[imm_offset + 2];
    const auto& expanded = ExpandedKey(key_selector == 1 ? assignment.key1 : assignment.key2, key_selector, cache);
    SetValueU8(values, out, expanded[round_index * 16 + byte_index]);
  } else if (opcode == OP_XTS_MUL_X_BYTE) {
    std::array<uint8_t, 16> tweak{};
    for (int i = 0; i < 16; ++i) tweak[i] = ValueU8(*values, circuit_.inputs[input_offset + i]);
    SetValueU8(values, out, XtsMulX(tweak)[circuit_.immediates[imm_offset]]);
  } else if (opcode == OP_EXTRACT_BYTE) {
    SetValueU8(values, out, ValuePtr(*values, circuit_.inputs[input_offset])[circuit_.immediates[imm_offset]]);
  }
}

ValueBuffer ByteCircuitEvaluator::EvaluateValuesInternal(
    const AssignmentState& assignment,
    AssignmentState* derive_target,
    KeyCache* cache) const {
  ValueBuffer values = MakeValueBuffer();
  std::function<void(int, int)> assign_wire = [&](int value_id, int source_value_id) {
    if (derive_target == nullptr) return;
    if (value_id < 0 || static_cast<std::size_t>(value_id) >= circuit_.graph.wire_offset_by_value.size()) return;
    const int32_t wire_offset = circuit_.graph.wire_offset_by_value[value_id];
    if (wire_offset < 0) return;
    const uint8_t value = ValueU8(values, source_value_id);
    derive_target->wires[wire_offset] = value;
    SetValueU8(&values, value_id, value);
    for (const int32_t dependent_constraint : circuit_.graph.definitions_by_right[value_id]) {
      assign_wire(circuit_.constraint_left[dependent_constraint], value_id);
    }
  };
  for (std::size_t op = 0; op < circuit_.opcodes.size(); ++op) {
    const int out = circuit_.outputs[op];
    EvaluateOpInto(op, assignment, &values, cache);
    if (derive_target != nullptr) {
      for (const int32_t constraint_index : circuit_.graph.definitions_by_right[out]) {
        assign_wire(circuit_.constraint_left[constraint_index], out);
      }
    }
  }
  return values;
}

void ByteCircuitEvaluator::DeriveWires(AssignmentState* assignment) const {
  EvaluateValuesAndDerive(assignment);
}

const std::array<uint8_t, 240>& ByteCircuitEvaluator::ExpandedKey(
    const std::vector<uint8_t>& key,
    int selector,
    KeyCache* cache) const {
  if (selector == 1) {
    if (cache->key1 != key) {
      cache->key1 = key;
      cache->expanded_key1 = ExpandKey(key);
    }
    return cache->expanded_key1;
  }
  if (cache->key2 != key) {
    cache->key2 = key;
    cache->expanded_key2 = ExpandKey(key);
  }
  return cache->expanded_key2;
}

}  // namespace aes_xts_decoder
