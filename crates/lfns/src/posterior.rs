//! The weighted posterior sample produced by a run.

use crate::log_space::log_sum_exp;
use crate::particle::Particle;

/// Dead and live particles with normalized posterior weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Posterior {
    particles: Vec<Particle>,
    log_weights: Vec<f64>,
}

impl Posterior {
    /// Pairs particles with their normalized log-weights; only a run state,
    /// which derives the weights from the run, may call this.
    pub(crate) fn from_weighted_particles(particles: Vec<Particle>, log_weights: Vec<f64>) -> Self {
        debug_assert_eq!(particles.len(), log_weights.len());
        Self { particles, log_weights }
    }

    /// Number of weighted particles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Whether the sample holds no particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Each particle with the logarithm of its posterior weight; the
    /// weights sum to one.
    pub fn weighted_particles(&self) -> impl Iterator<Item = (&Particle, f64)> {
        self.particles.iter().zip(self.log_weights.iter().copied())
    }

    /// Kish effective sample size `1 / sum_i w_i^2`: how many equally
    /// weighted draws the sample is worth.
    #[must_use]
    pub fn effective_sample_size(&self) -> f64 {
        (-log_sum_exp(self.log_weights.iter().map(|log_weight| 2.0 * log_weight))).exp()
    }
}
