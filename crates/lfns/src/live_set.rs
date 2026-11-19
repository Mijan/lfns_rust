//! The live particles of a nested sampling run.

use crate::lfns_error::LfnsError;
use crate::log_space::log_sum_exp;
use crate::particle::Particle;
use crate::point_matrix::PointMatrix;
use crate::population::Population;

/// Mean of the live likelihoods and the variance of that mean, in log space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LiveLikelihoodSummary {
    /// `ln` of the average likelihood over the live particles.
    pub(crate) log_mean: f64,
    /// `ln` of the estimated variance of that average.
    pub(crate) log_variance_of_mean: f64,
}

/// The live particles, kept sorted by ascending log-likelihood.
///
/// Between iterations the set holds exactly `N` particles. While an
/// iteration replaces the `r` lowest ones, it holds the `N - r` survivors.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveSet {
    population: Population,
    particles: Vec<Particle>,
}

impl LiveSet {
    /// Builds a full live set from `N` particles.
    pub(crate) fn from_particles(population: Population, mut particles: Vec<Particle>) -> Result<Self, LfnsError> {
        let expected = population.n_live() as usize;
        if particles.len() != expected {
            return Err(LfnsError::ParticleCountMismatch {
                context: "live set",
                expected,
                actual: particles.len(),
            });
        }
        Self::sort_ascending(&mut particles);
        Ok(Self { population, particles })
    }

    /// The live particles in ascending order of log-likelihood.
    #[must_use]
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Removes and returns the `r` particles of lowest likelihood, lowest
    /// first.
    pub(crate) fn remove_lowest(&mut self) -> Vec<Particle> {
        let n_replaced = self.population.n_replaced() as usize;
        debug_assert_eq!(self.particles.len(), self.population.n_live() as usize);
        self.particles.drain(..n_replaced).collect()
    }

    /// Refills the set with the replacements of the removed particles.
    pub(crate) fn add_replacements(&mut self, replacements: Vec<Particle>) {
        debug_assert_eq!(replacements.len(), self.population.n_replaced() as usize);
        self.particles.extend(replacements);
        Self::sort_ascending(&mut self.particles);
    }

    /// Coordinates of the current particles as one contiguous matrix.
    pub(crate) fn coordinates(&self, n_dimensions: usize) -> PointMatrix {
        PointMatrix::from_points(self.particles.iter().map(Particle::unit_coordinates), n_dimensions)
    }

    /// Estimates the mean live likelihood and the variance of that mean.
    pub(crate) fn summarize_likelihoods(&self, log_n_live: f64) -> LiveLikelihoodSummary {
        let log_likelihoods = self.particles.iter().map(Particle::log_likelihood);
        let log_mean = log_sum_exp(log_likelihoods) - log_n_live;
        let log_variance_of_mean = self.log_sample_variance(log_mean) - log_n_live;
        LiveLikelihoodSummary {
            log_mean,
            log_variance_of_mean,
        }
    }

    /// `ln` of the unbiased sample variance of the live likelihoods.
    fn log_sample_variance(&self, log_mean: f64) -> f64 {
        let Some(highest) = self.particles.last().map(Particle::log_likelihood) else {
            return f64::NEG_INFINITY;
        };
        if highest == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        let scaled_mean = (log_mean - highest).exp();
        let scaled_deviations = self
            .particles
            .iter()
            .map(|p| (p.log_likelihood() - highest).exp() - scaled_mean);
        let sum_of_squares: f64 = scaled_deviations.map(|deviation| deviation * deviation).sum();
        let degrees_of_freedom = f64::from(self.population.n_live()) - 1.0;
        (sum_of_squares / degrees_of_freedom).ln() + 2.0 * highest
    }

    fn sort_ascending(particles: &mut [Particle]) {
        particles.sort_by(|a, b| a.log_likelihood().total_cmp(&b.log_likelihood()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-12;

    fn live_set(log_likelihoods: &[f64], n_replaced: u32) -> Result<LiveSet, LfnsError> {
        let n_live = u32::try_from(log_likelihoods.len()).expect("small test set");
        let particles: Result<Vec<_>, _> = log_likelihoods.iter().map(|&l| Particle::new(vec![0.5], l)).collect();
        LiveSet::from_particles(Population::new(n_live, n_replaced)?, particles?)
    }

    #[test]
    fn removes_the_lowest_particles_lowest_first() -> Result<(), LfnsError> {
        let mut set = live_set(&[-1.0, -5.0, -3.0, -2.0], 2)?;
        let removed: Vec<f64> = set.remove_lowest().iter().map(Particle::log_likelihood).collect();
        assert_eq!(removed, vec![-5.0, -3.0]);
        Ok(())
    }

    #[test]
    fn summary_matches_linear_space_statistics_far_below_underflow() -> Result<(), LfnsError> {
        let offset = -3000.0;
        let likelihoods = [1.0_f64, 2.0, 3.0, 6.0];
        let logs: Vec<f64> = likelihoods.iter().map(|l| l.ln() + offset).collect();
        let summary = live_set(&logs, 1)?.summarize_likelihoods(4.0_f64.ln());
        // mean 3, sample variance 14/3, variance of the mean 14/12.
        assert!((summary.log_mean - (3.0_f64.ln() + offset)).abs() < TOLERANCE);
        assert!((summary.log_variance_of_mean - ((14.0_f64 / 12.0).ln() + 2.0 * offset)).abs() < TOLERANCE);
        Ok(())
    }

    #[test]
    fn rejects_a_set_of_the_wrong_size() {
        let population = Population::new(5, 1).expect("valid test population");
        assert!(LiveSet::from_particles(population, Vec::new()).is_err());
    }
}
