#include "solver.hpp"

namespace aes_xts_decoder {

void DualUpdater::Update(ParallelTemperingSolver& solver) const {
  const uint64_t profile_start = solver.ProfileStart();
  if (solver.scores_.empty()) {
    solver.ProfileAdd(PROFILE_DUAL_UPDATE, profile_start);
    return;
  }
  std::vector<double> means(solver.scores_[0].residuals.size(), 0.0);
  double weight_sum = 0.0;
  for (std::size_t r = 0; r < solver.scores_.size(); ++r) {
    const double weight = r < solver.replica_scale_.size() ? solver.replica_scale_[r] : 1.0;
    weight_sum += weight;
    for (std::size_t i = 0; i < solver.scores_[r].residuals.size(); ++i) {
      means[i] += weight * static_cast<double>(solver.scores_[r].residuals[i]);
    }
  }
  if (weight_sum <= 0.0) weight_sum = static_cast<double>(solver.scores_.size());
  if (solver.config_.dual_mode == 1) {
    std::array<double, 3> class_violation{0.0, 0.0, 0.0};
    std::array<std::size_t, 3> class_counts{0, 0, 0};
    for (std::size_t i = 0; i < means.size(); ++i) {
      means[i] /= weight_sum;
      const int cls = i < solver.circuit_.constraint_classes.size() ? solver.circuit_.constraint_classes[i] : 3;
      const std::size_t bucket = static_cast<std::size_t>(std::max(0, std::min(2, cls - 1)));
      class_violation[bucket] += means[i];
      ++class_counts[bucket];
    }
    for (std::size_t bucket = 0; bucket < class_violation.size(); ++bucket) {
      if (class_counts[bucket] != 0) class_violation[bucket] /= static_cast<double>(class_counts[bucket]);
      const double previous = solver.previous_violation_by_class_[bucket];
      const bool shrinking = previous < 0.0 || class_violation[bucket] <= solver.config_.scheduled_violation_shrink * previous;
      if (shrinking) {
        for (std::size_t i = 0; i < means.size(); ++i) {
          const int cls = i < solver.circuit_.constraint_classes.size() ? solver.circuit_.constraint_classes[i] : 3;
          const std::size_t constraint_bucket = static_cast<std::size_t>(std::max(0, std::min(2, cls - 1)));
          if (constraint_bucket == bucket && i < solver.multipliers_.size()) {
            solver.multipliers_[i] = std::max(0.0, solver.multipliers_[i] + solver.rho_by_class_[bucket] * means[i]);
          }
        }
        ++solver.lambda_update_counts_by_class_[bucket];
      } else {
        solver.rho_by_class_[bucket] = std::max(solver.rho_by_class_[bucket] * solver.config_.scheduled_rho_growth, solver.rho_by_class_[bucket] + 1e-9);
        ++solver.rho_escalation_counts_by_class_[bucket];
        if (solver.rho_escalation_counts_by_class_[bucket] >= 6 && class_violation[bucket] > solver.config_.scheduled_eta_target) {
          solver.infeasibility_suspected_ = true;
        }
      }
      solver.previous_violation_by_class_[bucket] = class_violation[bucket];
    }
    solver.UpdatePenaltyCache();
    solver.RecomputeEnergies();
    solver.ProfileAdd(PROFILE_DUAL_UPDATE, profile_start);
    return;
  }
  for (std::size_t i = 0; i < means.size(); ++i) {
    means[i] /= weight_sum;
    const int cls = i < solver.circuit_.constraint_classes.size() ? solver.circuit_.constraint_classes[i] : 3;
    if (cls == CLASS_ASCII || cls == CLASS_CONSISTENCY) {
      if (i < solver.multipliers_.size()) solver.multipliers_[i] = std::max(0.0, solver.multipliers_[i] + solver.config_.dual_eta * means[i]);
    }
  }
  solver.UpdatePenaltyCache();
  solver.RecomputeEnergies();
  solver.ProfileAdd(PROFILE_DUAL_UPDATE, profile_start);
}

void LadderAdapter::MaybeAdapt(ParallelTemperingSolver& solver) const {
  if (solver.config_.ladder_mode != 1 || solver.temperatures_.size() <= 2) return;
  if (solver.epochs_ <= solver.config_.ladder_burn_in_epochs) return;
  const std::size_t interval = std::max<std::size_t>(1, solver.config_.ladder_adapt_interval_epochs);
  if ((solver.epochs_ - solver.config_.ladder_burn_in_epochs) % interval != 0) return;
  const std::size_t total_round_trips = std::accumulate(solver.replica_round_trips_.begin(), solver.replica_round_trips_.end(), std::size_t{0});
  if (total_round_trips < solver.config_.ladder_min_round_trips) return;

  const std::size_t count = solver.temperatures_.size();
  std::vector<double> flow(count, 0.5);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t total = solver.replica_up_counts_[i] + solver.replica_down_counts_[i];
    if (total != 0) flow[i] = static_cast<double>(solver.replica_up_counts_[i]) / static_cast<double>(total);
  }

  std::vector<double> target(count, 0.0);
  double cumulative = 0.0;
  for (std::size_t i = 1; i < count; ++i) {
    const double df = std::abs(flow[i] - flow[i - 1]);
    cumulative += std::sqrt(df + 1e-9);
    target[i] = cumulative;
  }
  if (cumulative <= 0.0) return;
  for (double& value : target) value /= cumulative;

  const std::vector<double> old = solver.temperatures_;
  std::vector<double> next(count, old.front());
  next.front() = old.front();
  next.back() = old.back();
  for (std::size_t i = 1; i + 1 < count; ++i) {
    const double q = static_cast<double>(i) / static_cast<double>(count - 1);
    auto upper = std::lower_bound(target.begin(), target.end(), q);
    std::size_t hi = static_cast<std::size_t>(std::distance(target.begin(), upper));
    if (hi == 0) {
      next[i] = old.front();
    } else if (hi >= count) {
      next[i] = old.back();
    } else {
      const double lo_t = target[hi - 1];
      const double hi_t = target[hi];
      const double span = std::max(hi_t - lo_t, 1e-12);
      const double mix = (q - lo_t) / span;
      next[i] = old[hi - 1] + mix * (old[hi] - old[hi - 1]);
    }
    next[i] = std::max(next[i], next[i - 1] + 1e-9);
  }
  solver.temperatures_ = std::move(next);
  solver.UpdateReplicaScales();
  solver.last_ladder_adaptation_epoch_ = solver.epochs_;
  solver.ResetFlowCounters();
}

}  // namespace aes_xts_decoder
