#pragma once

#include "../../include/pt_engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace aes_xts_decoder {

inline constexpr int OP_INPUT = 1;
inline constexpr int OP_CONST = 2;
inline constexpr int OP_XOR8 = 3;
inline constexpr int OP_XOR128 = 4;
inline constexpr int OP_SBOX8 = 9;
inline constexpr int OP_MIX_COLUMN_BYTE = 10;
inline constexpr int OP_AES256_ROUND_KEY_BYTE = 11;
inline constexpr int OP_XTS_MUL_X_BYTE = 12;
inline constexpr int OP_EXTRACT_BYTE = 14;

inline constexpr int CONSTRAINT_EQ8 = 1;
inline constexpr int CONSTRAINT_EQ128 = 2;
inline constexpr int CONSTRAINT_ASCII = 3;
inline constexpr int CONSTRAINT_DEFINE8 = 4;

inline constexpr int CLASS_ASCII = 1;
inline constexpr int CLASS_CONSISTENCY = 2;

enum ProfileCounter : std::size_t {
  PROFILE_RUN_EPOCH_TOTAL = 0,
  PROFILE_STEP_REPLICAS_WALL = 1,
  PROFILE_ATTEMPT_SWAPS = 2,
  PROFILE_DUAL_UPDATE = 3,
  PROFILE_STEP_REPLICA_TOTAL = 4,
  PROFILE_MUTATE = 5,
  PROFILE_DERIVE_WIRES = 6,
  PROFILE_SCORE_DELTA = 7,
  PROFILE_SCORE_FULL = 8,
  PROFILE_EVALUATE_VALUES = 9,
  PROFILE_EVALUATE_VALUES_DERIVE = 10,
  PROFILE_RESIDUAL_SCAN = 11,
  PROFILE_ENERGY_ACCEPT = 12,
  PROFILE_RESTORE_VALUES = 13,
  PROFILE_HARVEST = 14,
  PROFILE_METRICS = 15,
  PROFILE_DRAIN_FEASIBLE = 16,
  PROFILE_CURRENT_ASSIGNMENT = 17,
  PROFILE_COUNTER_COUNT = 18,
};

extern const std::array<uint8_t, 256> SBOX;

uint8_t MixMul(uint8_t value, int coefficient);
std::array<uint8_t, 16> EncryptReducedRound(
    const std::array<uint8_t, 240>& expanded,
    const std::array<uint8_t, 16>& block,
    int rounds);
std::array<uint8_t, 16> DecryptReducedRound(
    const std::array<uint8_t, 240>& expanded,
    const std::array<uint8_t, 16>& block,
    int rounds);
std::array<uint8_t, 16> SectorTweakInput(int32_t sector_number);
uint64_t Next(uint64_t* state);
double Uniform(uint64_t* state);
std::size_t RandRange(uint64_t* state, std::size_t limit);
uint8_t TextAscii(uint64_t* state);
bool IsTextAscii(uint8_t value);
int Popcount(uint8_t value);
std::array<uint8_t, 240> ExpandKey(const std::vector<uint8_t>& key);
std::array<uint8_t, 16> XtsMulX(const std::array<uint8_t, 16>& tweak);
std::string KeyString(const AssignmentState& state);
uint64_t NowNanos();

}  // namespace aes_xts_decoder
