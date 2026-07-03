#pragma once

#include "internal.hpp"

#include <string>
#include <unordered_set>
#include <vector>

namespace aes_xts_decoder {

class FeasibleArchive {
 public:
  bool MaybeInsert(
      const AssignmentState& assignment,
      const ScoreData& score,
      std::unordered_set<std::string>* seen_keys,
      std::vector<AssignmentState>* feasible) const {
    if (score.hamming_score != 0.0) return false;
    const std::string key = KeyString(assignment);
    if (!seen_keys->insert(key).second) return false;
    feasible->push_back(assignment);
    return true;
  }
};

}  // namespace aes_xts_decoder
