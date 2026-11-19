//! Analytical toy problems shared by the integration tests.

use std::convert::Infallible;

use lfns::LikelihoodEstimator;
use rand::Rng;

/// One-dimensional likelihood `l(u) = RATE * exp(-RATE * u)` on the unit
/// interval. The prior volume of a point equals its coordinate, the evidence
/// is `1 - exp(-RATE)`, and the posterior mean is very nearly `1 / RATE`.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialPeak;

impl ExponentialPeak {
    pub const RATE: f64 = 100.0;
}

impl LikelihoodEstimator for ExponentialPeak {
    type Failure = Infallible;

    fn n_parameters(&self) -> usize {
        1
    }

    fn estimate_log_likelihood<R: Rng + ?Sized>(&self, unit_point: &[f64], _rng: &mut R) -> Result<f64, Infallible> {
        Ok(Self::RATE.ln() - Self::RATE * unit_point[0])
    }
}
