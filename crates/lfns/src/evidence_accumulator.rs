//! Running moments of the evidence integrated by the dead particles.
//!
//! With `B_i` the prior volume after iteration `i` and
//! `E_i = sum_j eps_{i,j} D_j` the likelihood-weighted spacings of iteration
//! `i`, the dead-particle evidence obeys `Z_i = Z_{i-1} + B_{i-1} E_i` and
//! `B_i = B_{i-1} t_r`. Because `(E_i, t_r)` of one iteration is independent
//! of everything before it, the first and second moments of `(Z_i, B_i)`
//! follow a five-term recursion. Every term is a sum of positive numbers, so
//! all of them are stored as logarithms.

use crate::evidence_estimate::EvidenceEstimate;
use crate::live_set::LiveLikelihoodSummary;
use crate::log_space::{log_add_exp, log_sum_exp};
use crate::population::Population;
use crate::shrinkage_moments::ShrinkageMoments;

/// Moments of one iteration's likelihood-weighted spacings `E`.
struct BatchMoments {
    log_mean: f64,
    log_second_moment: f64,
    log_times_shrinkage: f64,
}

/// Accumulates `E[Z]`, `E[B]`, `E[Z^2]`, `E[Z B]`, and `E[B^2]` over the
/// iterations of a run.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EvidenceAccumulator {
    moments: ShrinkageMoments,
    log_dead_evidence: f64,
    log_volume: f64,
    log_dead_evidence_squared: f64,
    log_dead_evidence_times_volume: f64,
    log_volume_squared: f64,
}

impl EvidenceAccumulator {
    pub(crate) fn new(population: Population) -> Self {
        Self {
            moments: ShrinkageMoments::of(population),
            log_dead_evidence: f64::NEG_INFINITY,
            log_volume: 0.0,
            log_dead_evidence_squared: f64::NEG_INFINITY,
            log_dead_evidence_times_volume: f64::NEG_INFINITY,
            log_volume_squared: 0.0,
        }
    }

    /// `ln` of the expected prior volume enclosed by the live particles.
    pub(crate) fn log_volume(&self) -> f64 {
        self.log_volume
    }

    /// `ln` of the prior-volume weight each particle removed in the next
    /// iteration will carry.
    pub(crate) fn log_next_dead_weight(&self) -> f64 {
        self.log_volume + self.moments.log_spacing_mean()
    }

    /// `ln N`, for callers that average over the live particles.
    pub(crate) fn log_n_live(&self) -> f64 {
        self.moments.log_n_live()
    }

    /// Adds one iteration, given the log-likelihoods of its dead particles.
    pub(crate) fn record_iteration(&mut self, dead_log_likelihoods: &[f64]) {
        let batch = self.batch_moments(dead_log_likelihoods);
        let squared = log_sum_exp(
            [
                self.log_dead_evidence_squared,
                std::f64::consts::LN_2 + self.log_dead_evidence_times_volume + batch.log_mean,
                self.log_volume_squared + batch.log_second_moment,
            ]
            .into_iter(),
        );
        let times_volume = log_add_exp(
            self.log_dead_evidence_times_volume + self.moments.log_shrinkage_mean(),
            self.log_volume_squared + batch.log_times_shrinkage,
        );
        self.log_dead_evidence = log_add_exp(self.log_dead_evidence, self.log_volume + batch.log_mean);
        self.log_dead_evidence_squared = squared;
        self.log_dead_evidence_times_volume = times_volume;
        self.log_volume += self.moments.log_shrinkage_mean();
        self.log_volume_squared += self.moments.log_shrinkage_second_moment();
    }

    /// Combines the dead-particle moments with the live particles into an
    /// evidence estimate.
    pub(crate) fn estimate(&self, live: LiveLikelihoodSummary) -> EvidenceEstimate {
        let log_live_evidence = self.log_volume + live.log_mean;
        let log_evidence = log_add_exp(self.log_dead_evidence, log_live_evidence);
        if log_evidence == f64::NEG_INFINITY {
            return EvidenceEstimate::from_parts(self.log_dead_evidence, log_live_evidence, log_evidence, 0.0, 0.0);
        }
        let relative_floor = self.relative_floor_variance(live.log_mean, log_evidence);
        let relative_live = (live.log_variance_of_mean + self.log_volume_squared - 2.0 * log_evidence).exp();
        EvidenceEstimate::from_parts(
            self.log_dead_evidence,
            log_live_evidence,
            log_evidence,
            relative_floor,
            relative_live,
        )
    }

    /// `Var(Z_dead + L B) / Z^2` for a known live average `L`: the variance
    /// that remains when the live average carries no Monte Carlo error.
    fn relative_floor_variance(&self, log_live_mean: f64, log_evidence: f64) -> f64 {
        let log_second_moment = log_sum_exp(
            [
                self.log_dead_evidence_squared,
                std::f64::consts::LN_2 + log_live_mean + self.log_dead_evidence_times_volume,
                2.0 * log_live_mean + self.log_volume_squared,
            ]
            .into_iter(),
        );
        (log_second_moment - 2.0 * log_evidence).exp_m1().max(0.0)
    }

    fn batch_moments(&self, dead_log_likelihoods: &[f64]) -> BatchMoments {
        let log_sum = log_sum_exp(dead_log_likelihoods.iter().copied());
        let log_sum_of_squares = log_sum_exp(dead_log_likelihoods.iter().map(|value| 2.0 * value));
        BatchMoments {
            log_mean: log_sum + self.moments.log_spacing_mean(),
            log_second_moment: log_add_exp(log_sum_of_squares, 2.0 * log_sum) + self.moments.log_spacing_pair(),
            log_times_shrinkage: log_sum + self.moments.log_spacing_times_shrinkage(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-9;

    fn accumulator(n_live: u32, n_replaced: u32) -> EvidenceAccumulator {
        EvidenceAccumulator::new(Population::new(n_live, n_replaced).expect("valid test population"))
    }

    fn exact_live(log_mean: f64) -> LiveLikelihoodSummary {
        LiveLikelihoodSummary {
            log_mean,
            log_variance_of_mean: f64::NEG_INFINITY,
        }
    }

    #[test]
    fn single_removal_matches_the_variance_of_one_uniform_spacing() {
        // Z_dead = eps (1 - t_1), and Var(1 - t_1) = N / ((N + 1)^2 (N + 2)).
        let (n, eps) = (20.0_f64, 3.0_f64);
        let mut acc = accumulator(20, 1);
        acc.record_iteration(&[eps.ln()]);
        let estimate = acc.estimate(exact_live(f64::NEG_INFINITY));
        let mean = eps / (n + 1.0);
        let variance = eps * eps * n / ((n + 1.0) * (n + 1.0) * (n + 2.0));
        assert!((estimate.log_evidence().exp() - mean).abs() < TOLERANCE);
        assert!((estimate.relative_floor_std().powi(2) - variance / (mean * mean)).abs() < TOLERANCE);
    }

    #[test]
    fn constant_likelihood_has_no_volume_uncertainty() {
        // If every likelihood equals c then Z = c (1 - B) + c B = c exactly,
        // whatever the random volumes are, so the floor variance must vanish.
        let log_c = -700.0;
        let mut acc = accumulator(50, 10);
        for _ in 0..40 {
            acc.record_iteration(&[log_c; 10]);
        }
        let estimate = acc.estimate(exact_live(log_c));
        assert!((estimate.log_evidence() - log_c).abs() < TOLERANCE);
        assert!(estimate.relative_floor_std() < 1e-5);
    }

    #[test]
    fn dead_and_live_parts_partition_a_constant_likelihood() {
        let mut acc = accumulator(100, 25);
        for _ in 0..7 {
            acc.record_iteration(&[0.0; 25]);
        }
        let estimate = acc.estimate(exact_live(0.0));
        let volume = (76.0_f64 / 101.0).powi(7);
        assert!((estimate.log_live_evidence().exp() - volume).abs() < TOLERANCE);
        assert!((estimate.log_dead_evidence().exp() - (1.0 - volume)).abs() < TOLERANCE);
    }

    #[test]
    fn monte_carlo_error_of_the_live_average_adds_to_the_floor() {
        let mut acc = accumulator(50, 5);
        acc.record_iteration(&[-3.0, -2.5, -2.0, -1.5, -1.0]);
        let exact = acc.estimate(exact_live(-0.5));
        let noisy = acc.estimate(LiveLikelihoodSummary {
            log_mean: -0.5,
            log_variance_of_mean: -4.0,
        });
        assert!((exact.achievable_improvement()).abs() < TOLERANCE);
        assert!(noisy.relative_std() > noisy.relative_floor_std());
        assert!((noisy.relative_floor_std() - exact.relative_floor_std()).abs() < TOLERANCE);
    }

    #[test]
    fn no_likelihood_mass_yields_zero_evidence_without_nan() {
        let mut acc = accumulator(10, 2);
        acc.record_iteration(&[f64::NEG_INFINITY; 2]);
        let estimate = acc.estimate(exact_live(f64::NEG_INFINITY));
        assert_eq!(estimate.log_evidence(), f64::NEG_INFINITY);
        assert!(estimate.relative_std().abs() < TOLERANCE);
    }
}
