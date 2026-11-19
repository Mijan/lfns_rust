//! Proposals drawn uniformly from the support of a DP-GMM density estimate.
//!
//! The live particles of an iteration are distributed like the constrained
//! prior, whose boundary is smooth when the likelihood is estimated with
//! noise. A Dirichlet process Gaussian mixture is fitted to them, and
//! candidates are drawn **uniformly** from the region where the fitted
//! density exceeds a floor: a draw `x` from the mixture is kept with
//! probability `floor / density(x)`, and discarded outright where the
//! density is below the floor or `x` lies outside the unit cube.

use dpgmm::{Dpgmm, DpgmmConfig, DpgmmError, GaussianMixture, Observations};
use rand::{Rng, RngExt};

use crate::point_matrix::PointMatrix;
use crate::proposal::{Proposal, ProposalFitter};

/// Why a DP-GMM proposal could not be built.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DpgmmProposalError {
    /// The quantile is not a probability strictly between 0 and 1.
    #[error(
        "rejection quantile {value} is not strictly between 0 and 1. Pass a small probability, \
         e.g. `with_rejection_quantile(0.001)`."
    )]
    InvalidQuantile {
        /// The rejected value.
        value: f64,
    },

    /// The mixture could not be fitted to the live particles.
    #[error("fitting the mixture to the live particles failed: {0}")]
    Fit(#[from] DpgmmError),

    /// Almost all of the fitted density lies outside the unit cube.
    #[error(
        "only {n_inside} of {n_draws} draws from the fitted mixture fell inside the unit cube, \
         too few to locate the density floor. The live particles are probably piled up at a \
         prior bound; widen that bound."
    )]
    MixtureOutsideUnitCube {
        /// Draws that fell inside the cube.
        n_inside: usize,
        /// Draws attempted.
        n_draws: usize,
    },
}

/// Settings of the DP-GMM proposal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpgmmProposalConfig {
    mixture: DpgmmConfig,
    rejection_quantile: f64,
}

impl DpgmmProposalConfig {
    /// Default fraction of the mixture's own draws whose density lies below
    /// the floor.
    pub const DEFAULT_REJECTION_QUANTILE: f64 = 0.001;

    /// Returns a copy that places the density floor at the given quantile
    /// of the density of the mixture's own draws. A smaller quantile covers
    /// more of the tails at the price of a lower acceptance rate.
    ///
    /// # Errors
    ///
    /// Returns [`DpgmmProposalError::InvalidQuantile`] unless
    /// `0 < quantile < 1`.
    pub fn with_rejection_quantile(self, quantile: f64) -> Result<Self, DpgmmProposalError> {
        if !(quantile > 0.0 && quantile < 1.0) {
            return Err(DpgmmProposalError::InvalidQuantile { value: quantile });
        }
        Ok(Self {
            rejection_quantile: quantile,
            ..self
        })
    }

    /// Returns a copy that fits the mixture with the given settings.
    #[must_use]
    pub fn with_mixture_config(self, mixture: DpgmmConfig) -> Self {
        Self { mixture, ..self }
    }

    /// Quantile at which the density floor is placed.
    #[must_use]
    pub fn rejection_quantile(&self) -> f64 {
        self.rejection_quantile
    }

    /// Settings of the mixture fit.
    #[must_use]
    pub fn mixture_config(&self) -> DpgmmConfig {
        self.mixture
    }
}

impl Default for DpgmmProposalConfig {
    fn default() -> Self {
        Self {
            mixture: DpgmmConfig::default(),
            rejection_quantile: Self::DEFAULT_REJECTION_QUANTILE,
        }
    }
}

/// Uniform proposal over the part of the unit cube where a fitted mixture
/// density exceeds a floor.
#[derive(Debug, Clone, PartialEq)]
pub struct DpgmmProposal {
    mixture: GaussianMixture,
    log_density_floor: f64,
}

impl DpgmmProposal {
    /// The fitted mixture.
    #[must_use]
    pub fn mixture(&self) -> &GaussianMixture {
        &self.mixture
    }

    /// Logarithm of the density floor that bounds the proposal region.
    #[must_use]
    pub fn log_density_floor(&self) -> f64 {
        self.log_density_floor
    }

    /// Draws from the mixture until a point lies inside the unit cube;
    /// returns its log-density.
    fn sample_inside_cube<R: Rng + ?Sized>(mixture: &GaussianMixture, rng: &mut R, unit_point: &mut [f64]) -> f64 {
        let unit_interval = 0.0..=1.0;
        loop {
            mixture.sample(rng, unit_point);
            if unit_point.iter().all(|coordinate| unit_interval.contains(coordinate)) {
                return mixture.log_density(unit_point);
            }
        }
    }
}

impl Proposal for DpgmmProposal {
    fn propose<R: Rng + ?Sized>(&self, rng: &mut R, unit_point: &mut [f64]) {
        loop {
            let log_density = Self::sample_inside_cube(&self.mixture, rng, unit_point);
            let keep_probability = (self.log_density_floor - log_density).exp();
            if log_density >= self.log_density_floor && rng.random::<f64>() < keep_probability {
                return;
            }
        }
    }
}

/// Fits a [`DpgmmProposal`] to the live particles of an iteration.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DpgmmProposalFitter {
    config: DpgmmProposalConfig,
}

impl DpgmmProposalFitter {
    /// Number of in-cube mixture draws from which the floor is estimated.
    const N_QUANTILE_DRAWS: usize = 1000;
    /// Mixture draws attempted before giving up on finding enough of them
    /// inside the unit cube.
    const MAX_QUANTILE_ATTEMPTS: usize = 1_000_000;

    /// Creates a fitter with the given settings.
    #[must_use]
    pub fn new(config: DpgmmProposalConfig) -> Self {
        Self { config }
    }

    /// Estimates the floor as a low quantile of the log-density of the
    /// mixture's own draws that fall inside the unit cube.
    fn estimate_log_density_floor<R: Rng + ?Sized>(
        &self,
        mixture: &GaussianMixture,
        rng: &mut R,
    ) -> Result<f64, DpgmmProposalError> {
        let unit_interval = 0.0..=1.0;
        let mut point = vec![0.0; mixture.n_dimensions()];
        let mut log_densities = Vec::with_capacity(Self::N_QUANTILE_DRAWS);
        for _ in 0..Self::MAX_QUANTILE_ATTEMPTS {
            mixture.sample(rng, &mut point);
            if point.iter().all(|coordinate| unit_interval.contains(coordinate)) {
                log_densities.push(mixture.log_density(&point));
            }
            if log_densities.len() == Self::N_QUANTILE_DRAWS {
                return Ok(self.select_quantile(&mut log_densities));
            }
        }
        Err(DpgmmProposalError::MixtureOutsideUnitCube {
            n_inside: log_densities.len(),
            n_draws: Self::MAX_QUANTILE_ATTEMPTS,
        })
    }

    fn select_quantile(&self, log_densities: &mut [f64]) -> f64 {
        // The product lies in [0, len), so the conversion neither truncates
        // meaningfully nor loses a sign.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let rank = (self.config.rejection_quantile * log_densities.len() as f64) as usize;
        *log_densities.select_nth_unstable_by(rank, f64::total_cmp).1
    }
}

impl ProposalFitter for DpgmmProposalFitter {
    type Proposal = DpgmmProposal;
    type Failure = DpgmmProposalError;

    fn fit<R: Rng + ?Sized>(
        &self,
        live_points: &PointMatrix,
        rng: &mut R,
    ) -> Result<DpgmmProposal, DpgmmProposalError> {
        let observations = Observations::from_rows(live_points.as_flat_slice(), live_points.n_dimensions())?;
        let mixture = Dpgmm::new(self.config.mixture).fit(&observations, rng)?;
        let log_density_floor = self.estimate_log_density_floor(&mixture, rng)?;
        Ok(DpgmmProposal {
            mixture,
            log_density_floor,
        })
    }
}
