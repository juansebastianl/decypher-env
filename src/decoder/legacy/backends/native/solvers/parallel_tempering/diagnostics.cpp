#include "solver.hpp"

namespace aes_xts_decoder {

std::vector<std::size_t> ParallelTemperingSolver::ComputeAlgebraCounts() const {
  std::size_t linear_ops = 0;
  std::size_t nonlinear_ops = 0;
  for (const int opcode : circuit_.opcodes) {
    if (opcode == OP_XOR8 || opcode == OP_XOR128 || opcode == OP_MIX_COLUMN_BYTE || opcode == OP_XTS_MUL_X_BYTE ||
        opcode == OP_CONST || opcode == OP_INPUT || opcode == OP_EXTRACT_BYTE) {
      ++linear_ops;
    } else {
      ++nonlinear_ops;
    }
  }
  std::size_t linear_constraints = 0;
  for (const int kind : circuit_.constraint_kinds) {
    if (kind == CONSTRAINT_DEFINE8 || kind == CONSTRAINT_EQ8 || kind == CONSTRAINT_EQ128 || kind == CONSTRAINT_ASCII) {
      ++linear_constraints;
    }
  }
  const std::size_t variable_bits = (circuit_.plaintext_count + circuit_.key1_count + circuit_.key2_count + circuit_.wire_offsets.size()) * 8;
  const std::size_t factor_count = circuit_.opcodes.size() + circuit_.constraint_kinds.size();
  return {
      variable_bits,
      linear_ops,
      nonlinear_ops,
      linear_constraints,
      factor_count,
      config_.bp_diagnostics ? std::size_t{1} : std::size_t{0},
      config_.algebra_diagnostics ? std::size_t{1} : std::size_t{0},
  };
}

std::vector<double> ParallelTemperingSolver::ComputeBpKeyMarginals() const {
  return {};
}

double ParallelTemperingSolver::PenaltyForConstraint(std::size_t constraint_index) const {
  if (config_.dual_mode != 1) return config_.mu;
  const int cls = constraint_index < circuit_.constraint_classes.size() ? circuit_.constraint_classes[constraint_index] : 3;
  const std::size_t bucket = static_cast<std::size_t>(std::max(0, std::min(2, cls - 1)));
  return rho_by_class_[bucket];
}

void ParallelTemperingSolver::UpdatePenaltyCache() {
  penalties_.assign(circuit_.constraint_kinds.size(), config_.mu);
  if (config_.dual_mode != 1) return;
  for (std::size_t i = 0; i < penalties_.size(); ++i) {
    penalties_[i] = PenaltyForConstraint(i);
  }
}

void ParallelTemperingSolver::UpdateReplicaScales() {
  const std::size_t count = temperatures_.size();
  replica_scale_.assign(count, 1.0);
  if (count == 0) return;
  const double cold = config_.lambda_scale_cold > 0.0 ? config_.lambda_scale_cold : 1.0;
  const double hot = config_.lambda_scale_hot > 0.0 ? config_.lambda_scale_hot : 1.0;
  if (count == 1) {
    replica_scale_[0] = cold;
    return;
  }
  // Geometric grade from the coldest rung (index 0) to the hottest (index N-1)
  // so low-temperature replicas enforce constraints strongly while hot replicas
  // explore with weakened Lagrange pressure.
  const double ratio = std::pow(hot / cold, 1.0 / static_cast<double>(count - 1));
  for (std::size_t i = 0; i < count; ++i) {
    replica_scale_[i] = cold * std::pow(ratio, static_cast<double>(i));
  }
}

double ParallelTemperingSolver::EffectiveBeta(std::size_t replica_index) const {
  const double temperature = replica_index < temperatures_.size()
                                 ? std::max(temperatures_[replica_index], 1e-9)
                                 : 1e-9;
  const double scale = replica_index < replica_scale_.size() ? replica_scale_[replica_index] : 1.0;
  return scale / temperature;
}

std::vector<double> ParallelTemperingSolver::EffectiveTemperatures() const {
  std::vector<double> effective(temperatures_.size(), 0.0);
  for (std::size_t i = 0; i < temperatures_.size(); ++i) {
    const double beta = EffectiveBeta(i);
    effective[i] = beta > 0.0 ? 1.0 / beta : temperatures_[i];
  }
  return effective;
}

double ParallelTemperingSolver::KeyMarginalMaxDeviation() const {
  if (key_visit_count_ == 0) return 0.0;
  double max_deviation = 0.0;
  const double alpha = std::max(config_.marginal_alpha, 0.0);
  const double denominator = static_cast<double>(key_visit_count_) + 2.0 * alpha;
  for (const std::size_t ones : key_ones_) {
    const double p = denominator > 0.0
                         ? (static_cast<double>(ones) + alpha) / denominator
                         : static_cast<double>(ones) / static_cast<double>(key_visit_count_);
    max_deviation = std::max(max_deviation, std::abs(p - 0.5));
  }
  return max_deviation;
}

double ParallelTemperingSolver::KeyInformationBits() const {
  if (key_visit_count_ == 0) return 0.0;
  double information = 0.0;
  const double alpha = std::max(config_.marginal_alpha, 0.0);
  const double denominator = static_cast<double>(key_visit_count_) + 2.0 * alpha;
  for (const std::size_t ones : key_ones_) {
    double p = denominator > 0.0
                   ? (static_cast<double>(ones) + alpha) / denominator
                   : static_cast<double>(ones) / static_cast<double>(key_visit_count_);
    p = std::max(1e-12, std::min(1.0 - 1e-12, p));
    const double entropy = -(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p));
    information += 1.0 - entropy;
  }
  return information;
}

double ParallelTemperingSolver::KeyInformationNullBaselineBits() const {
  if (key_visit_count_ == 0) return 0.0;
  const double alpha = std::max(config_.marginal_alpha, 0.0);
  const double denominator = static_cast<double>(key_visit_count_) + 2.0 * alpha;
  double expected_per_bit = 0.0;
  const double log_denominator = std::lgamma(static_cast<double>(key_visit_count_) + 1.0) -
                                 static_cast<double>(key_visit_count_) * std::log(2.0);
  for (std::size_t ones = 0; ones <= key_visit_count_; ++ones) {
    const double log_probability = log_denominator - std::lgamma(static_cast<double>(ones) + 1.0) -
                                   std::lgamma(static_cast<double>(key_visit_count_ - ones) + 1.0);
    const double probability = std::exp(log_probability);
    double p = denominator > 0.0
                   ? (static_cast<double>(ones) + alpha) / denominator
                   : static_cast<double>(ones) / static_cast<double>(key_visit_count_);
    p = std::max(1e-12, std::min(1.0 - 1e-12, p));
    const double entropy = -(p * std::log2(p) + (1.0 - p) * std::log2(1.0 - p));
    expected_per_bit += probability * (1.0 - entropy);
  }
  return 512.0 * expected_per_bit;
}

double ParallelTemperingSolver::MarginalEss() const {
  const std::size_t n = cold_key_hamming_trace_.size();
  if (n < 4) return 0.0;
  double mean = 0.0;
  for (const double value : cold_key_hamming_trace_) mean += value;
  mean /= static_cast<double>(n);
  double variance = 0.0;
  for (const double value : cold_key_hamming_trace_) {
    const double centered = value - mean;
    variance += centered * centered;
  }
  if (variance <= 1e-12) return 0.0;
  const std::size_t max_lag = std::min<std::size_t>(n - 1, 512);
  double tau = 1.0;
  for (std::size_t lag = 1; lag <= max_lag; ++lag) {
    double covariance = 0.0;
    for (std::size_t i = 0; i + lag < n; ++i) {
      covariance += (cold_key_hamming_trace_[i] - mean) * (cold_key_hamming_trace_[i + lag] - mean);
    }
    const double rho = covariance / variance;
    if (rho <= 0.0) break;
    tau += 2.0 * rho;
    if (static_cast<double>(lag) >= 6.0 * tau) break;
  }
  return static_cast<double>(n) / std::max(2.0 * tau, 1.0);
}

double ParallelTemperingSolver::MarginalRhat() const {
  const std::size_t n = cold_key_hamming_trace_.size();
  if (n < 4) return 0.0;
  const std::size_t half = n / 2;
  auto chain_stats = [&](std::size_t begin, std::size_t end) {
    double mean = 0.0;
    for (std::size_t i = begin; i < end; ++i) mean += cold_key_hamming_trace_[i];
    mean /= static_cast<double>(end - begin);
    double variance = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      const double centered = cold_key_hamming_trace_[i] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<double>(std::max<std::size_t>(1, end - begin - 1));
    return std::pair<double, double>{mean, variance};
  };
  const auto first = chain_stats(0, half);
  const auto second = chain_stats(half, half * 2);
  const double within = 0.5 * (first.second + second.second);
  if (within <= 1e-12) return 0.0;
  const double mean = 0.5 * (first.first + second.first);
  const double between = static_cast<double>(half) *
                         ((first.first - mean) * (first.first - mean) + (second.first - mean) * (second.first - mean));
  const double var_hat = ((static_cast<double>(half) - 1.0) / static_cast<double>(half)) * within +
                         between / static_cast<double>(half);
  return std::sqrt(std::max(var_hat / within, 0.0));
}

double ParallelTemperingSolver::LogZEstimate() const {
  if (energy_temperatures_by_rung_.empty() || energy_mean_by_rung_.empty()) return 0.0;
  std::vector<std::pair<double, double>> beta_energy;
  const std::size_t count = std::min(energy_temperatures_by_rung_.size(), energy_mean_by_rung_.size());
  beta_energy.reserve(count + 1);
  for (std::size_t i = 0; i < count; ++i) {
    if (energy_sample_counts_[i] == 0) continue;
    beta_energy.push_back({1.0 / std::max(energy_temperatures_by_rung_[i], 1e-9), energy_mean_by_rung_[i]});
  }
  const double reference_log_states =
      static_cast<double>((circuit_.plaintext_count + circuit_.key1_count + circuit_.key2_count) * 8) * std::log(2.0);
  if (beta_energy.empty()) return reference_log_states;
  std::sort(beta_energy.begin(), beta_energy.end());
  double integral = 0.0;
  double prev_beta = 0.0;
  double prev_energy = beta_energy.front().second;
  for (const auto& item : beta_energy) {
    integral += 0.5 * (prev_energy + item.second) * (item.first - prev_beta);
    prev_beta = item.first;
    prev_energy = item.second;
  }
  return reference_log_states - integral;
}

}  // namespace aes_xts_decoder
