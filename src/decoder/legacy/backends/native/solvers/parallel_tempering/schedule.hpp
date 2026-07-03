#pragma once

namespace aes_xts_decoder {

class ParallelTemperingSolver;

class DualUpdater {
 public:
  void Update(ParallelTemperingSolver& solver) const;
};

class LadderAdapter {
 public:
  void MaybeAdapt(ParallelTemperingSolver& solver) const;
};

}  // namespace aes_xts_decoder
