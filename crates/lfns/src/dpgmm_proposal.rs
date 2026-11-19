//! Proposals drawn from the prior over the support of a DP-GMM density
//! estimate.
//!
//! The live particles of an iteration are distributed like the constrained
//! prior. To propose from the prior over the region they occupy, the
//! particles are first mapped from the unit cube to an unbounded *latent*
//! space, coordinate by coordinate, with the standard normal quantile
//! function. There the prior is a standard normal density `phi` and no
//! coordinate has an edge that a Gaussian could leak across, which matters
//! in many dimensions: a Gaussian fitted to points that fill a
//! 69-dimensional cube has almost all of its mass outside the cube.
//!
//! A Dirichlet process Gaussian mixture `m` is fitted to the latent points.
//! The proposal region is where the ratio `m / phi` (the density of the
//! live particles relative to the prior) exceeds a floor, and a candidate is
//! drawn from the prior restricted to that region by rejection: a draw `z`
//! from the mixture is kept with probability `floor * phi(z) / m(z)`, and
//! discarded outright where the ratio is below the floor. For a posterior
//! region that is small compared with the prior, `phi` is nearly constant
//! over it and this is a uniform draw over the support of the mixture, as
//! in the paper.

use dpgmm::{Dpgmm, DpgmmConfig, DpgmmError, GaussianMixture, Observations};
use rand::{Rng, RngExt};

use crate::point_matrix::PointMatrix;
use crate::proposal::{Proposal, ProposalFitter};
use crate::standard_normal::StandardNormal;

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
}

/// Settings of the DP-GMM proposal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpgmmProposalConfig {
    mixture: DpgmmConfig,
    rejection_quantile: f64,
}

impl DpgmmProposalConfig {
    /// Default fraction of the mixture's own draws that lie below the floor.
    pub const DEFAULT_REJECTION_QUANTILE: f64 = 0.001;

    /// Returns a copy that places the floor at the given quantile of the
    /// density ratio over the mixture's own draws. A smaller quantile covers
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

    /// Quantile at which the floor is placed.
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

/// The prior restricted to the region where a fitted mixture density,
/// relative to the prior, exceeds a floor.
#[derive(Debug, Clone, PartialEq)]
pub struct DpgmmProposal {
    latent_mixture: GaussianMixture,
    log_ratio_floor: f64,
}

impl DpgmmProposal {
    /// The mixture fitted to the live particles in latent space, where each
    /// unit-cube coordinate is mapped through the standard normal quantile
    /// function.
    #[must_use]
    pub fn latent_mixture(&self) -> &GaussianMixture {
        &self.latent_mixture
    }

    /// Logarithm of the floor on the mixture density relative to the prior
    /// that bounds the proposal region.
    #[must_use]
    pub fn log_ratio_floor(&self) -> f64 {
        self.log_ratio_floor
    }

    /// `ln` of the mixture density over the prior density at a latent point.
    fn log_ratio(mixture: &GaussianMixture, latent_point: &[f64]) -> f64 {
        let log_prior: f64 = latent_point.iter().map(|z| StandardNormal::log_density(*z)).sum();
        mixture.log_density(latent_point) - log_prior
    }
}

impl Proposal for DpgmmProposal {
    fn propose<R: Rng + ?Sized>(&self, rng: &mut R, unit_point: &mut [f64]) {
        loop {
            self.latent_mixture.sample(rng, unit_point);
            let log_ratio = Self::log_ratio(&self.latent_mixture, unit_point);
            let keep_probability = (self.log_ratio_floor - log_ratio).exp();
            if log_ratio >= self.log_ratio_floor && rng.random::<f64>() < keep_probability {
                break;
            }
        }
        for coordinate in unit_point.iter_mut() {
            *coordinate = StandardNormal::cdf(*coordinate);
        }
    }
}

/// Fits a [`DpgmmProposal`] to the live particles of an iteration.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DpgmmProposalFitter {
    config: DpgmmProposalConfig,
}

impl DpgmmProposalFitter {
    /// Number of mixture draws from which the floor is estimated.
    const N_QUANTILE_DRAWS: usize = 1000;

    /// Creates a fitter with the given settings.
    #[must_use]
    pub fn new(config: DpgmmProposalConfig) -> Self {
        Self { config }
    }

    /// Estimates the floor as a low quantile of the density ratio over the
    /// mixture's own draws.
    fn estimate_log_ratio_floor<R: Rng + ?Sized>(&self, mixture: &GaussianMixture, rng: &mut R) -> f64 {
        let mut point = vec![0.0; mixture.n_dimensions()];
        let mut log_ratios: Vec<f64> = (0..Self::N_QUANTILE_DRAWS)
            .map(|_| {
                mixture.sample(rng, &mut point);
                DpgmmProposal::log_ratio(mixture, &point)
            })
            .collect();
        // The product lies in [0, len), so the conversion neither truncates
        // meaningfully nor loses a sign.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let rank = (self.config.rejection_quantile * log_ratios.len() as f64) as usize;
        *log_ratios.select_nth_unstable_by(rank, f64::total_cmp).1
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
        let latent: Vec<f64> = live_points
            .as_flat_slice()
            .iter()
            .map(|u| StandardNormal::quantile(*u))
            .collect();
        let observations = Observations::from_rows(&latent, live_points.n_dimensions())?;
        let latent_mixture = Dpgmm::new(self.config.mixture).fit(&observations, rng)?;
        let log_ratio_floor = self.estimate_log_ratio_floor(&latent_mixture, rng);
        Ok(DpgmmProposal {
            latent_mixture,
            log_ratio_floor,
        })
    }
}
