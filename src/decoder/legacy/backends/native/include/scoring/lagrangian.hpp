#pragma once

#include "../pt_engine.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace aes_xts_decoder {

struct LagrangianWeights {
  std::vector<double> lambda;
  std::vector<double> rho;
};

class AugmentedLagrangian {
 public:
  explicit AugmentedLagrangian(LagrangianWeights weights)
      : weights_(std::move(weights)) {}

  double Energy(const ScoreData& score) const {
    double energy = 0.0;
    for (std::size_t i = 0; i < score.residuals.size(); ++i) {
      const double residual = static_cast<double>(score.residuals[i]);
      const double lambda = i < weights_.lambda.size() ? weights_.lambda[i] : 1.0;
      const double rho = i < weights_.rho.size() ? weights_.rho[i] : 0.0;
      energy += lambda * residual + 0.5 * rho * residual * residual;
    }
    return energy;
  }

  const LagrangianWeights& weights() const { return weights_; }
  LagrangianWeights* mutable_weights() { return &weights_; }

 private:
  LagrangianWeights weights_;
};

}  // namespace aes_xts_decoder
