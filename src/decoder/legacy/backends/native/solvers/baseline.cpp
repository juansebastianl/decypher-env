#include "../include/cpp_solver.hpp"

namespace aes_xts_decoder {

Score score_assignment(const CircuitView& circuit, const AssignmentView& assignment) {
  (void)circuit;
  (void)assignment;
  return Score{0, 0, nullptr, 0, nullptr, 0};
}

}  // namespace aes_xts_decoder
