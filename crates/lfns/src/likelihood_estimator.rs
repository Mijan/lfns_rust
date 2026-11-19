//! The interface between the sampler and a model.

use rand::Rng;

/// Produces one realization of an unbiased likelihood estimate for a point
/// of the unit cube.
///
/// The sampler works in prior-CDF space: every prior is uniform on
/// `[0, 1]^d` there. An implementation maps the point through the inverse
/// prior CDF to physical parameters before evaluating the model.
///
/// A deterministic model returns its exact log-likelihood and ignores `rng`.
/// A stochastic model returns the logarithm of an unbiased likelihood
/// estimate (for example from a particle filter) and draws all randomness
/// from `rng`, which makes a run reproducible from its seed.
pub trait LikelihoodEstimator: Sync {
    /// Why an evaluation could not be completed (for example a failed ODE
    /// solve). The sampler counts such points as having zero likelihood.
    type Failure: std::error::Error + Send + Sync + 'static;

    /// Number of estimated parameters `d`.
    fn n_parameters(&self) -> usize;

    /// Returns the logarithm of one likelihood realization at `unit_point`.
    ///
    /// The value must be finite or `f64::NEG_INFINITY` (zero likelihood).
    ///
    /// # Errors
    ///
    /// Returns `Self::Failure` when the likelihood cannot be computed.
    fn estimate_log_likelihood<R: Rng + ?Sized>(&self, unit_point: &[f64], rng: &mut R) -> Result<f64, Self::Failure>;
}
