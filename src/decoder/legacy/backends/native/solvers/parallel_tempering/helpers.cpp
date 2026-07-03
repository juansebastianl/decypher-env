#include "internal.hpp"

namespace aes_xts_decoder {

const std::array<uint8_t, 256> SBOX = {
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

uint8_t GfMul(uint8_t a, uint8_t b) {
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

std::array<uint8_t, 256> GfMulTable(uint8_t coefficient) {
  std::array<uint8_t, 256> table{};
  for (int value = 0; value < 256; ++value) {
    table[value] = GfMul(static_cast<uint8_t>(value), coefficient);
  }
  return table;
}

uint8_t MixMul(uint8_t value, int coefficient) {
  static const std::array<uint8_t, 256> kMul2 = GfMulTable(2);
  static const std::array<uint8_t, 256> kMul3 = GfMulTable(3);
  if (coefficient == 1) return value;
  if (coefficient == 2) return kMul2[value];
  if (coefficient == 3) return kMul3[value];
  return GfMul(value, static_cast<uint8_t>(coefficient));
}

std::array<uint8_t, 256> BuildInvSbox() {
  std::array<uint8_t, 256> inverse{};
  for (int value = 0; value < 256; ++value) {
    inverse[SBOX[value]] = static_cast<uint8_t>(value);
  }
  return inverse;
}

const std::array<uint8_t, 256>& InvSbox() {
  static const std::array<uint8_t, 256> kInvSbox = BuildInvSbox();
  return kInvSbox;
}

void ShiftRows(std::array<uint8_t, 16>* state) {
  std::array<uint8_t, 16> out{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      out[row + 4 * col] = (*state)[row + 4 * ((col + row) % 4)];
    }
  }
  *state = out;
}

void InvShiftRows(std::array<uint8_t, 16>* state) {
  std::array<uint8_t, 16> out{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      out[row + 4 * ((col + row) % 4)] = (*state)[row + 4 * col];
    }
  }
  *state = out;
}

void SubBytes(std::array<uint8_t, 16>* state) {
  for (uint8_t& value : *state) value = SBOX[value];
}

void InvSubBytes(std::array<uint8_t, 16>* state) {
  const auto& inv = InvSbox();
  for (uint8_t& value : *state) value = inv[value];
}

void MixColumns(std::array<uint8_t, 16>* state) {
  std::array<uint8_t, 16> out{};
  for (int col = 0; col < 4; ++col) {
    const int i = 4 * col;
    const uint8_t a0 = (*state)[i], a1 = (*state)[i + 1], a2 = (*state)[i + 2], a3 = (*state)[i + 3];
    out[i] = MixMul(a0, 2) ^ MixMul(a1, 3) ^ a2 ^ a3;
    out[i + 1] = a0 ^ MixMul(a1, 2) ^ MixMul(a2, 3) ^ a3;
    out[i + 2] = a0 ^ a1 ^ MixMul(a2, 2) ^ MixMul(a3, 3);
    out[i + 3] = MixMul(a0, 3) ^ a1 ^ a2 ^ MixMul(a3, 2);
  }
  *state = out;
}

void InvMixColumns(std::array<uint8_t, 16>* state) {
  std::array<uint8_t, 16> out{};
  for (int col = 0; col < 4; ++col) {
    const int i = 4 * col;
    const uint8_t a0 = (*state)[i], a1 = (*state)[i + 1], a2 = (*state)[i + 2], a3 = (*state)[i + 3];
    out[i] = GfMul(a0, 14) ^ GfMul(a1, 11) ^ GfMul(a2, 13) ^ GfMul(a3, 9);
    out[i + 1] = GfMul(a0, 9) ^ GfMul(a1, 14) ^ GfMul(a2, 11) ^ GfMul(a3, 13);
    out[i + 2] = GfMul(a0, 13) ^ GfMul(a1, 9) ^ GfMul(a2, 14) ^ GfMul(a3, 11);
    out[i + 3] = GfMul(a0, 11) ^ GfMul(a1, 13) ^ GfMul(a2, 9) ^ GfMul(a3, 14);
  }
  *state = out;
}

void AddRoundKey(std::array<uint8_t, 16>* state, const uint8_t* round_key) {
  for (int i = 0; i < 16; ++i) (*state)[i] ^= round_key[i];
}

// Encrypt one block with the lowered round structure (matches LoweredAesEncoding):
// AddRoundKey(rk0); (SubBytes, ShiftRows, MixColumns, AddRoundKey) x (rounds-1);
// SubBytes, ShiftRows, AddRoundKey(rk_rounds).
std::array<uint8_t, 16> EncryptReducedRound(const std::array<uint8_t, 240>& expanded,
                                            const std::array<uint8_t, 16>& block, int rounds) {
  std::array<uint8_t, 16> state = block;
  AddRoundKey(&state, expanded.data());
  for (int round = 1; round < rounds; ++round) {
    SubBytes(&state);
    ShiftRows(&state);
    MixColumns(&state);
    AddRoundKey(&state, expanded.data() + round * 16);
  }
  SubBytes(&state);
  ShiftRows(&state);
  AddRoundKey(&state, expanded.data() + rounds * 16);
  return state;
}

// Exact inverse of EncryptReducedRound for the same round count.
std::array<uint8_t, 16> DecryptReducedRound(const std::array<uint8_t, 240>& expanded,
                                            const std::array<uint8_t, 16>& block, int rounds) {
  std::array<uint8_t, 16> state = block;
  AddRoundKey(&state, expanded.data() + rounds * 16);
  InvShiftRows(&state);
  InvSubBytes(&state);
  for (int round = rounds - 1; round >= 1; --round) {
    AddRoundKey(&state, expanded.data() + round * 16);
    InvMixColumns(&state);
    InvShiftRows(&state);
    InvSubBytes(&state);
  }
  AddRoundKey(&state, expanded.data());
  return state;
}

std::array<uint8_t, 16> SectorTweakInput(int32_t sector_number) {
  std::array<uint8_t, 16> input{};
  uint64_t value = static_cast<uint64_t>(static_cast<uint32_t>(sector_number));
  for (int i = 0; i < 16 && value; ++i) {
    input[i] = static_cast<uint8_t>(value & 0xFF);
    value >>= 8;
  }
  return input;
}

uint64_t Next(uint64_t* state) {
  *state = (*state * 6364136223846793005ULL) + 1442695040888963407ULL;
  return *state;
}

double Uniform(uint64_t* state) {
  return static_cast<double>(Next(state) >> 11) * (1.0 / static_cast<double>(1ULL << 53));
}

std::size_t RandRange(uint64_t* state, std::size_t limit) {
  return limit == 0 ? 0 : static_cast<std::size_t>(Next(state) % limit);
}

uint8_t TextAscii(uint64_t* state) {
  static constexpr std::array<uint8_t, 98> kText = [] {
    std::array<uint8_t, 98> values{};
    values[0] = 0x09;
    values[1] = 0x0A;
    values[2] = 0x0D;
    for (int i = 0; i < 95; ++i) values[3 + i] = static_cast<uint8_t>(0x20 + i);
    return values;
  }();
  return kText[RandRange(state, kText.size())];
}

bool IsTextAscii(uint8_t value) {
  return value == 0x09 || value == 0x0A || value == 0x0D || (value >= 0x20 && value <= 0x7E);
}

int Popcount(uint8_t value) {
  int count = 0;
  while (value) {
    count += value & 1;
    value >>= 1;
  }
  return count;
}

uint32_t RotWord(uint32_t word) {
  return ((word << 8) & 0xFFFFFFFFu) | (word >> 24);
}

uint32_t SubWord(uint32_t word) {
  return (static_cast<uint32_t>(SBOX[(word >> 24) & 0xFF]) << 24) |
         (static_cast<uint32_t>(SBOX[(word >> 16) & 0xFF]) << 16) |
         (static_cast<uint32_t>(SBOX[(word >> 8) & 0xFF]) << 8) |
         static_cast<uint32_t>(SBOX[word & 0xFF]);
}

std::array<uint8_t, 240> ExpandKey(const std::vector<uint8_t>& key) {
  static constexpr std::array<uint8_t, 10> rcon = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};
  std::array<uint32_t, 60> words{};
  for (int i = 0; i < 8; ++i) {
    words[i] = (static_cast<uint32_t>(key[4 * i]) << 24) |
               (static_cast<uint32_t>(key[4 * i + 1]) << 16) |
               (static_cast<uint32_t>(key[4 * i + 2]) << 8) |
               static_cast<uint32_t>(key[4 * i + 3]);
  }
  for (int i = 8; i < 60; ++i) {
    uint32_t temp = words[i - 1];
    if (i % 8 == 0) {
      temp = SubWord(RotWord(temp)) ^ (static_cast<uint32_t>(rcon[(i / 8) - 1]) << 24);
    } else if (i % 8 == 4) {
      temp = SubWord(temp);
    }
    words[i] = words[i - 8] ^ temp;
  }
  std::array<uint8_t, 240> expanded{};
  for (int i = 0; i < 60; ++i) {
    expanded[4 * i] = static_cast<uint8_t>(words[i] >> 24);
    expanded[4 * i + 1] = static_cast<uint8_t>(words[i] >> 16);
    expanded[4 * i + 2] = static_cast<uint8_t>(words[i] >> 8);
    expanded[4 * i + 3] = static_cast<uint8_t>(words[i]);
  }
  return expanded;
}

std::array<uint8_t, 16> XtsMulX(const std::array<uint8_t, 16>& tweak) {
  std::array<uint8_t, 16> out{};
  uint8_t carry = 0;
  for (int i = 0; i < 16; ++i) {
    const uint8_t next_carry = (tweak[i] & 0x80) ? 1 : 0;
    out[i] = static_cast<uint8_t>((tweak[i] << 1) | carry);
    carry = next_carry;
  }
  if (carry) out[0] ^= 0x87;
  return out;
}

std::string KeyString(const AssignmentState& state) {
  return std::string(reinterpret_cast<const char*>(state.key1.data()), state.key1.size()) +
         std::string(reinterpret_cast<const char*>(state.key2.data()), state.key2.size());
}

uint64_t NowNanos() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

}  // namespace aes_xts_decoder
