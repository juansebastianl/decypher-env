#include "solver.hpp"

namespace aes_xts_decoder {

double ParallelTemperingSolver::Energy(const ScoreData& score) const {
  double total = 0.0;
  for (std::size_t i = 0; i < score.residuals.size(); ++i) {
    const double residual = score.residuals[i];
    const double multiplier = i < multipliers_.size() ? multipliers_[i] : 1.0;
    const double penalty = i < penalties_.size() ? penalties_[i] : config_.mu;
    total += multiplier * residual + 0.5 * penalty * residual * residual;
  }
  return total;
}

void ParallelTemperingSolver::RecomputeEnergies() {
  energies_.resize(scores_.size());
  for (std::size_t i = 0; i < scores_.size(); ++i) {
    energies_[i] = Energy(scores_[i]);
  }
}

std::vector<double> ParallelTemperingSolver::TemperatureLadder(std::size_t count, double t_min, double t_max) const {
  if (count <= 1) return {t_min};
  if (t_min <= 0.0) t_min = 0.5;
  if (t_max < t_min) t_max = t_min;
  const double ratio = std::pow(t_max / t_min, 1.0 / static_cast<double>(count - 1));
  std::vector<double> temperatures(count);
  for (std::size_t i = 0; i < count; ++i) temperatures[i] = t_min * std::pow(ratio, static_cast<double>(i));
  return temperatures;
}

std::vector<double> ParallelTemperingSolver::InitialMultipliers(const PTConfig& config) const {
  std::vector<double> values;
  values.reserve(circuit_.constraint_classes.size());
  for (const int cls : circuit_.constraint_classes) {
    if (cls == CLASS_ASCII) values.push_back(config.ascii_weight);
    else if (cls == CLASS_CONSISTENCY) values.push_back(config.consistency_weight / 8.0);
    else values.push_back(config.goal_weight / 8.0);
  }
  return values;
}

}  // namespace aes_xts_decoder
