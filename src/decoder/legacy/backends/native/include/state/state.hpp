#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aes_xts_decoder {

struct ByteSpan {
  const uint8_t* data;
  std::size_t size;

  const uint8_t& operator[](std::size_t index) const { return data[index]; }
  bool empty() const { return size == 0; }
};

struct MutableByteSpan {
  uint8_t* data;
  std::size_t size;

  uint8_t& operator[](std::size_t index) const { return data[index]; }
  bool empty() const { return size == 0; }
};

struct ByteStateView {
  ByteSpan plaintext;
  ByteSpan key1;
  ByteSpan key2;
  ByteSpan wires;
};

struct MutableByteStateView {
  MutableByteSpan plaintext;
  MutableByteSpan key1;
  MutableByteSpan key2;
  MutableByteSpan wires;
};

inline ByteSpan AsByteSpan(const std::vector<uint8_t>& values) {
  return ByteSpan{values.data(), values.size()};
}

inline MutableByteSpan AsMutableByteSpan(std::vector<uint8_t>* values) {
  return MutableByteSpan{values->data(), values->size()};
}

}  // namespace aes_xts_decoder
