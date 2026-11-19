//! Moments of the prior-volume shrinkage factors of parallel nested sampling.
//!
//! Within one iteration the `r` lowest of `N` live particles are removed.
//! With `t_0 = 1` and `t_j` the j-th largest of `N` uniform numbers, the
//! spacings `D_j = t_{j-1} - t_j` are jointly Dirichlet(1, ..., 1) over
//! `N + 1` cells, which gives every moment below in closed form, including
//! the correlations between the `t_j` of one iteration.

use crate::population::Population;

/// Logarithms of the shrinkage moments for a fixed [`Population`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ShrinkageMoments {
    log_spacing_mean: f64,
    log_spacing_pair: f64,
    log_spacing_times_shrinkage: f64,
    log_shrinkage_mean: f64,
    log_shrinkage_second_moment: f64,
    log_n_live: f64,
}

impl ShrinkageMoments {
    pub(crate) fn of(population: Population) -> Self {
        let n_live = f64::from(population.n_live());
        let n_kept = n_live - f64::from(population.n_replaced());
        let log_pair_scale = -((n_live + 1.0) * (n_live + 2.0)).ln();
        Self {
            log_spacing_mean: -(n_live + 1.0).ln(),
            log_spacing_pair: log_pair_scale,
            log_spacing_times_shrinkage: (n_kept + 1.0).ln() + log_pair_scale,
            log_shrinkage_mean: ((n_kept + 1.0) / (n_live + 1.0)).ln(),
            log_shrinkage_second_moment: ((n_kept + 1.0) * (n_kept + 2.0)).ln() + log_pair_scale,
            log_n_live: n_live.ln(),
        }
    }

    /// `ln E[D_j] = -ln(N + 1)`, the same for every spacing.
    pub(crate) fn log_spacing_mean(&self) -> f64 {
        self.log_spacing_mean
    }

    /// `ln E[D_j D_k]` for `j != k`; `E[D_j^2]` is twice that value.
    pub(crate) fn log_spacing_pair(&self) -> f64 {
        self.log_spacing_pair
    }

    /// `ln E[D_j t_r]`, the same for every `j <= r`.
    pub(crate) fn log_spacing_times_shrinkage(&self) -> f64 {
        self.log_spacing_times_shrinkage
    }

    /// `ln E[t_r]`, the mean shrinkage of one full iteration.
    pub(crate) fn log_shrinkage_mean(&self) -> f64 {
        self.log_shrinkage_mean
    }

    /// `ln E[t_r^2]`.
    pub(crate) fn log_shrinkage_second_moment(&self) -> f64 {
        self.log_shrinkage_second_moment
    }

    /// `ln N`.
    pub(crate) fn log_n_live(&self) -> f64 {
        self.log_n_live
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-12;

    fn moments(n_live: u32, n_replaced: u32) -> ShrinkageMoments {
        ShrinkageMoments::of(Population::new(n_live, n_replaced).expect("valid test population"))
    }

    #[test]
    fn shrinkage_moments_match_the_beta_distribution_of_the_order_statistic() {
        // t_r ~ Beta(N - r + 1, r): mean a/(a+b), second moment a(a+1)/((a+b)(a+b+1)).
        let (a, b) = (8.0_f64, 3.0_f64);
        let m = moments(10, 3);
        assert!((m.log_shrinkage_mean().exp() - a / (a + b)).abs() < TOLERANCE);
        let second = a * (a + 1.0) / ((a + b) * (a + b + 1.0));
        assert!((m.log_shrinkage_second_moment().exp() - second).abs() < TOLERANCE);
    }

    #[test]
    fn spacing_moments_are_consistent_with_the_shrinkage_mean() {
        // t_r = 1 - sum of r spacings, so E[t_r] = 1 - r E[D].
        let m = moments(10, 3);
        let from_spacings = 1.0 - 3.0 * m.log_spacing_mean().exp();
        assert!((m.log_shrinkage_mean().exp() - from_spacings).abs() < TOLERANCE);
    }

    #[test]
    fn spacing_times_shrinkage_follows_from_the_pair_moments() {
        // E[D_j t_r] = E[D_j] - sum_k E[D_j D_k] = E[D] - (r + 1) E[D_j D_k].
        let m = moments(10, 3);
        let expected = m.log_spacing_mean().exp() - 4.0 * m.log_spacing_pair().exp();
        assert!((m.log_spacing_times_shrinkage().exp() - expected).abs() < TOLERANCE);
    }
}
