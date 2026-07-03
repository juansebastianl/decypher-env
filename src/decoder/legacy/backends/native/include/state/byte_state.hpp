#pragma once

#include "state.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace aes_xts_decoder {

struct ByteState {
  std::vector<uint8_t> plaintext;
  std::vector<uint8_t> key1;
  std::vector<uint8_t> key2;
  std::vector<uint8_t> wires;

  ByteStateView view() const {
    return ByteStateView{
        AsByteSpan(plaintext),
        AsByteSpan(key1),
        AsByteSpan(key2),
        AsByteSpan(wires),
    };
  }

  MutableByteStateView mutable_view() {
    return MutableByteStateView{
        AsMutableByteSpan(&plaintext),
        AsMutableByteSpan(&key1),
        AsMutableByteSpan(&key2),
        AsMutableByteSpan(&wires),
    };
  }

  ByteState Clone() const { return *this; }

  void Swap(ByteState* other) {
    plaintext.swap(other->plaintext);
    key1.swap(other->key1);
    key2.swap(other->key2);
    wires.swap(other->wires);
  }
};

}  // namespace aes_xts_decoder
