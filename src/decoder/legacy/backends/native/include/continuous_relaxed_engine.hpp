#pragma once

#include "pt_engine.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aes_xts_decoder {

class ContinuousRelaxedSolver;

class ContinuousRelaxedEngine {
 public:
  ContinuousRelaxedEngine(
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
      const std::vector<uint16_t>& value_widths,
      const std::vector<int32_t>& constraint_classes,
      std::size_t plaintext_start,
      std::size_t plaintext_count,
      std::size_t key1_start,
      std::size_t key1_count,
      std::size_t key2_start,
      std::size_t key2_count,
      const std::vector<int32_t>& xts_block_sectors,
      const std::vector<int32_t>& xts_block_indices,
      const std::vector<uint8_t>& xts_block_targets,
      const AssignmentState& initial,
      const PTConfig& config);
  ~ContinuousRelaxedEngine();

  ContinuousRelaxedEngine(const ContinuousRelaxedEngine&) = delete;
  ContinuousRelaxedEngine& operator=(const ContinuousRelaxedEngine&) = delete;
  ContinuousRelaxedEngine(ContinuousRelaxedEngine&&) noexcept;
  ContinuousRelaxedEngine& operator=(ContinuousRelaxedEngine&&) noexcept;

  ScoreData ScoreAssignment(const AssignmentState& assignment) const;
  ScoreData RunEpoch(std::size_t sweeps);
  std::vector<uint32_t> Residuals() const;
  PTDiagnostics Metrics() const;
  std::vector<AssignmentState> DrainFeasible(std::size_t limit);
  AssignmentState CurrentAssignment() const;
  AssignmentState DeriveAssignmentWires(const AssignmentState& assignment) const;
  void SetMultipliers(const std::vector<double>& multipliers);
  void SetTemperatures(const std::vector<double>& temperatures);
  void SetConstraintClasses(const std::vector<int32_t>& classes);

 private:
  std::unique_ptr<ContinuousRelaxedSolver> impl_;
};

}  // namespace aes_xts_decoder
