//! Univariate slice sampling (Neal 2003) with stepping out and shrinkage.

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Exp1};

/// Draws from a one-dimensional density known up to a constant, by one
/// slice-sampling transition from a current point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SliceSampler {
    initial_width: f64,
    max_expansions: u32,
}

impl SliceSampler {
    /// Width of the first bracket; hyperparameters are sampled on a log
    /// scale, where one unit is a natural step.
    const DEFAULT_INITIAL_WIDTH: f64 = 1.0;
    /// Bound on stepping out, so that a flat tail cannot stall the sampler.
    const DEFAULT_MAX_EXPANSIONS: u32 = 64;

    pub(crate) fn new() -> Self {
        Self {
            initial_width: Self::DEFAULT_INITIAL_WIDTH,
            max_expansions: Self::DEFAULT_MAX_EXPANSIONS,
        }
    }

    /// Performs one transition from `current` that leaves the density
    /// `exp(log_density)` invariant, and returns the new point.
    pub(crate) fn sample<R: Rng + ?Sized>(&self, log_density: impl Fn(f64) -> f64, current: f64, rng: &mut R) -> f64 {
        let exponential: f64 = Exp1.sample(rng);
        let log_height = log_density(current) - exponential;
        let (mut lower, mut upper) = self.step_out(&log_density, current, log_height, rng);
        loop {
            let candidate = lower + (upper - lower) * rng.random::<f64>();
            if log_density(candidate) > log_height || upper - lower <= f64::EPSILON * current.abs().max(1.0) {
                return candidate;
            }
            if candidate < current {
                lower = candidate;
            } else {
                upper = candidate;
            }
        }
    }

    /// Places a bracket of the initial width at random around `current`
    /// and widens it until both ends lie outside the slice.
    fn step_out<R: Rng + ?Sized>(
        &self,
        log_density: &impl Fn(f64) -> f64,
        current: f64,
        log_height: f64,
        rng: &mut R,
    ) -> (f64, f64) {
        let mut lower = current - self.initial_width * rng.random::<f64>();
        let mut upper = lower + self.initial_width;
        let mut expansions_left = self.max_expansions;
        while expansions_left > 0 && log_density(lower) > log_height {
            lower -= self.initial_width;
            expansions_left -= 1;
        }
        while expansions_left > 0 && log_density(upper) > log_height {
            upper += self.initial_width;
            expansions_left -= 1;
        }
        (lower, upper)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn chain_reproduces_mean_and_variance_of_a_shifted_normal() {
        // Target N(3, 0.5^2).
        let log_density = |x: f64| -0.5 * ((x - 3.0) / 0.5).powi(2);
        let sampler = SliceSampler::new();
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let n_draws = 40_000;
        let (mut current, mut sum, mut sum_of_squares) = (0.0, 0.0, 0.0);
        for _ in 0..n_draws {
            current = sampler.sample(log_density, current, &mut rng);
            sum += current;
            sum_of_squares += current * current;
        }
        let mean = sum / f64::from(n_draws);
        let variance = sum_of_squares / f64::from(n_draws) - mean * mean;
        assert!((mean - 3.0).abs() < 0.03, "mean {mean}");
        assert!((variance - 0.25).abs() < 0.02, "variance {variance}");
    }
}
