#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace aes_xts_decoder {

class DiagnosticsCollector {
 public:
  std::vector<double> ProfileSeconds(const std::vector<uint64_t>& profile_nanos) const {
    std::vector<double> seconds(profile_nanos.size(), 0.0);
    for (std::size_t i = 0; i < profile_nanos.size(); ++i) {
      seconds[i] = static_cast<double>(profile_nanos[i]) / 1e9;
    }
    return seconds;
  }
};

}  // namespace aes_xts_decoder
