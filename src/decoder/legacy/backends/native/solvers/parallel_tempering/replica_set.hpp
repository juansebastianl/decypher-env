#pragma once

#include "byte_evaluator.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace aes_xts_decoder {

struct ReplicaState {
  AssignmentState assignment;
  ScoreData score;
  double energy = 0.0;
  ValueBuffer values;
  uint64_t rng_state = 0;
  KeyCache key_cache;
  std::vector<uint32_t> value_marks;
  std::vector<uint32_t> constraint_marks;
  uint32_t mark_stamp = 1;
};

class ReplicaSet {
 public:
  ReplicaSet(
      std::vector<AssignmentState>* assignments,
      std::vector<ScoreData>* scores,
      std::vector<double>* energies,
      std::vector<ValueBuffer>* values,
      std::vector<KeyCache>* key_caches,
      std::vector<int>* directions,
      std::vector<std::size_t>* round_trips)
      : assignments_(assignments),
        scores_(scores),
        energies_(energies),
        values_(values),
        key_caches_(key_caches),
        directions_(directions),
        round_trips_(round_trips) {}

  std::size_t size() const { return assignments_->size(); }

  void SwapAdjacent(std::size_t left) {
    const std::size_t right = left + 1;
    std::swap((*assignments_)[left], (*assignments_)[right]);
    std::swap((*scores_)[left], (*scores_)[right]);
    std::swap((*energies_)[left], (*energies_)[right]);
    std::swap((*values_)[left], (*values_)[right]);
    std::swap((*key_caches_)[left], (*key_caches_)[right]);
    std::swap((*directions_)[left], (*directions_)[right]);
    std::swap((*round_trips_)[left], (*round_trips_)[right]);
  }

 private:
  std::vector<AssignmentState>* assignments_;
  std::vector<ScoreData>* scores_;
  std::vector<double>* energies_;
  std::vector<ValueBuffer>* values_;
  std::vector<KeyCache>* key_caches_;
  std::vector<int>* directions_;
  std::vector<std::size_t>* round_trips_;
};

}  // namespace aes_xts_decoder
