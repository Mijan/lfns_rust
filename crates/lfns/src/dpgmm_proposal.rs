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
//!
//! By default the floor is placed on live particles kept out of the fit, so
//! that a stated share of the constrained prior lies below it (see
//! [`HeldOutFloor`]). Without a held-out floor it is a low quantile of the
//! ratio over the mixture's own draws, and a [`LowAcceptanceRule`] raises
//! that quantile for an iteration that follows one with a poor acceptance
//! rate.

use dpgmm::{Dpgmm, DpgmmConfig, DpgmmError, GaussianMixture, Observations};
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};

use crate::held_out_floor::HeldOutFloor;
use crate::low_acceptance_rule::LowAcceptanceRule;
use crate::point_matrix::PointMatrix;
use crate::proposal::{FitContext, Proposal, ProposalFitter};
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

    /// There are too few live particles to keep some out of the fit.
    #[error(
        "{n_points} live particle(s) cannot be split into particles for the mixture fit and particles \
         held out of it. Use at least two live particles, or switch the held-out floor off with \
         `without_held_out_floor()`."
    )]
    TooFewParticlesToHoldOut {
        /// Number of live particles handed to the fitter.
        n_points: usize,
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
    low_acceptance_rule: Option<LowAcceptanceRule>,
    held_out_floor: Option<HeldOutFloor>,
}

impl DpgmmProposalConfig {
    /// Default fraction of the mixture's own draws that lie below the floor.
    pub const DEFAULT_REJECTION_QUANTILE: f64 = 0.001;

    /// Returns a copy that places the floor on live particles kept out of
    /// the mixture fit. The rejection quantile and the low-acceptance rule
    /// are then not used.
    #[must_use]
    pub fn with_held_out_floor(self, floor: HeldOutFloor) -> Self {
        Self {
            held_out_floor: Some(floor),
            ..self
        }
    }

    /// Returns a copy that fits the mixture to every live particle and
    /// places the floor at [`Self::rejection_quantile`] of the mixture's
    /// own draws, subject to the low-acceptance rule.
    #[must_use]
    pub fn without_held_out_floor(self) -> Self {
        Self {
            held_out_floor: None,
            ..self
        }
    }

    /// The floor placed on held-out live particles, if one is used.
    #[must_use]
    pub fn held_out_floor(&self) -> Option<HeldOutFloor> {
        self.held_out_floor
    }

    /// Returns a copy that places the floor at the given quantile of the
    /// density ratio over the mixture's own draws, when no held-out floor
    /// is used. A smaller quantile covers more of the tails at the price of
    /// a lower acceptance rate.
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

    /// Returns a copy that places the floor at the quantile of `rule` for
    /// every iteration that follows one with a poor acceptance rate.
    #[must_use]
    pub fn with_low_acceptance_rule(self, rule: LowAcceptanceRule) -> Self {
        Self {
            low_acceptance_rule: Some(rule),
            ..self
        }
    }

    /// Returns a copy without a low-acceptance rule: the floor stays at
    /// [`Self::rejection_quantile`] whatever the acceptance rate.
    #[must_use]
    pub fn without_low_acceptance_rule(self) -> Self {
        Self {
            low_acceptance_rule: None,
            ..self
        }
    }

    /// Quantile at which the floor is placed, unless the low-acceptance
    /// rule applies.
    #[must_use]
    pub fn rejection_quantile(&self) -> f64 {
        self.rejection_quantile
    }

    /// The rule for iterations that follow a poor acceptance rate, if any.
    #[must_use]
    pub fn low_acceptance_rule(&self) -> Option<LowAcceptanceRule> {
        self.low_acceptance_rule
    }

    /// Quantile at which the floor of an iteration is placed, given the
    /// acceptance rate of the iteration before it.
    fn rejection_quantile_after(&self, acceptance_rates: &[f64]) -> f64 {
        match self.low_acceptance_rule {
            Some(rule) if rule.applies_after(acceptance_rates) => rule.rejection_quantile(),
            _ => self.rejection_quantile,
        }
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
            low_acceptance_rule: Some(LowAcceptanceRule::default()),
            held_out_floor: Some(HeldOutFloor::default()),
        }
    }
}

/// The prior restricted to the region where a fitted mixture density,
/// relative to the prior, exceeds a floor.
#[derive(Debug, Clone, PartialEq)]
pub struct DpgmmProposal {
    latent_mixture: GaussianMixture,
    log_ratio_floor: f64,
    rejection_quantile: f64,
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
    fn rejection_quantile(&self) -> Option<f64> {
        Some(self.rejection_quantile)
    }

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

    /// Estimates the floor as the given quantile of the density ratio over
    /// the mixture's own draws.
    fn estimate_log_ratio_floor<R: Rng + ?Sized>(
        mixture: &GaussianMixture,
        rejection_quantile: f64,
        rng: &mut R,
    ) -> f64 {
        let mut point = vec![0.0; mixture.n_dimensions()];
        let log_ratios: Vec<f64> = (0..Self::N_QUANTILE_DRAWS)
            .map(|_| {
                mixture.sample(rng, &mut point);
                DpgmmProposal::log_ratio(mixture, &point)
            })
            .collect();
        Self::quantile(log_ratios, rejection_quantile)
    }

    /// The value below which the given share of `values` lies.
    fn quantile(mut values: Vec<f64>, share: f64) -> f64 {
        // The product lies in [0, len), so the conversion neither truncates
        // meaningfully nor loses a sign.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let rank = ((share * values.len() as f64) as usize).min(values.len().saturating_sub(1));
        *values.select_nth_unstable_by(rank, f64::total_cmp).1
    }

    /// The live particles in latent space, point after point.
    fn latent_points(live_points: &PointMatrix) -> Vec<f64> {
        live_points
            .as_flat_slice()
            .iter()
            .map(|u| StandardNormal::quantile(*u))
            .collect()
    }

    /// Fits the mixture to every live particle and places the floor on the
    /// mixture's own draws.
    fn fit_on_all<R: Rng + ?Sized>(
        &self,
        live_points: &PointMatrix,
        context: FitContext<'_>,
        rng: &mut R,
    ) -> Result<DpgmmProposal, DpgmmProposalError> {
        let latent = Self::latent_points(live_points);
        let observations = Observations::from_rows(&latent, live_points.n_dimensions())?;
        let latent_mixture = Dpgmm::new(self.config.mixture).fit(&observations, rng)?;
        let rejection_quantile = self.config.rejection_quantile_after(context.acceptance_rates());
        let log_ratio_floor = Self::estimate_log_ratio_floor(&latent_mixture, rejection_quantile, rng);
        Ok(DpgmmProposal {
            latent_mixture,
            log_ratio_floor,
            rejection_quantile,
        })
    }

    /// Fits the mixture to a random part of the live particles and places
    /// the floor on the others.
    fn fit_with_held_out<R: Rng + ?Sized>(
        &self,
        live_points: &PointMatrix,
        floor: HeldOutFloor,
        rng: &mut R,
    ) -> Result<DpgmmProposal, DpgmmProposalError> {
        let (n_points, n_dimensions) = (live_points.n_points(), live_points.n_dimensions());
        if n_points < 2 {
            return Err(DpgmmProposalError::TooFewParticlesToHoldOut { n_points });
        }
        let latent = Self::latent_points(live_points);
        let mut order: Vec<usize> = (0..n_points).collect();
        order.shuffle(rng);
        let (held_out, fitted) = order.split_at(floor.n_held_out(n_points));
        let point = |index: &usize| &latent[index * n_dimensions..(index + 1) * n_dimensions];
        let fitted_points: Vec<f64> = fitted.iter().flat_map(|index| point(index).iter().copied()).collect();
        let observations = Observations::from_rows(&fitted_points, n_dimensions)?;
        let latent_mixture = Dpgmm::new(self.config.mixture).fit(&observations, rng)?;
        let held_out_ratios: Vec<f64> = held_out
            .iter()
            .map(|index| DpgmmProposal::log_ratio(&latent_mixture, point(index)))
            .collect();
        Ok(DpgmmProposal {
            log_ratio_floor: Self::quantile(held_out_ratios, floor.level_set_share()),
            latent_mixture,
            rejection_quantile: floor.level_set_share(),
        })
    }
}

impl ProposalFitter for DpgmmProposalFitter {
    type Proposal = DpgmmProposal;
    type Failure = DpgmmProposalError;

    fn fit<R: Rng + ?Sized>(
        &self,
        live_points: &PointMatrix,
        context: FitContext<'_>,
        rng: &mut R,
    ) -> Result<DpgmmProposal, DpgmmProposalError> {
        match self.config.held_out_floor {
            Some(floor) => self.fit_with_held_out(live_points, floor, rng),
            None => self.fit_on_all(live_points, context, rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// Two hundred points spread over the middle of the unit square.
    fn live_points() -> PointMatrix {
        square_points(200, 1)
    }

    /// Points uniform over the middle of the unit square.
    fn square_points(n_points: usize, seed: u64) -> PointMatrix {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let points: Vec<[f64; 2]> = (0..n_points)
            .map(|_| [0.3 + 0.4 * rng.random::<f64>(), 0.3 + 0.4 * rng.random::<f64>()])
            .collect();
        PointMatrix::from_points(points.iter().map(<[f64; 2]>::as_slice), 2)
    }

    #[test]
    fn held_out_floor_gives_up_about_its_share_of_the_region() -> Result<(), Box<dyn std::error::Error>> {
        let share = 0.1;
        let config = DpgmmProposalConfig::default().with_held_out_floor(HeldOutFloor::giving_up(share)?);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let proposal = DpgmmProposalFitter::new(config).fit(&square_points(2000, 1), FitContext::after(&[]), &mut rng)?;
        // Fresh points of the same region, which the fitter never saw.
        let fresh = square_points(4000, 2);
        let below = fresh
            .points()
            .filter(|point| {
                let latent: Vec<f64> = point.iter().map(|u| StandardNormal::quantile(*u)).collect();
                DpgmmProposal::log_ratio(proposal.latent_mixture(), &latent) < proposal.log_ratio_floor()
            })
            .count();
        #[allow(clippy::cast_precision_loss)]
        let lost = below as f64 / fresh.n_points() as f64;
        // 400 held-out points place the 10 % quantile within about 1.5 % (one standard error).
        assert!((0.04..0.16).contains(&lost), "lost {lost}, asked for {share}");
        assert_eq!(proposal.rejection_quantile(), Some(share));
        Ok(())
    }

    #[test]
    fn held_out_floor_needs_two_particles() {
        let fitter = DpgmmProposalFitter::default();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let outcome = fitter.fit(&square_points(1, 1), FitContext::after(&[]), &mut rng);
        assert!(matches!(outcome, Err(DpgmmProposalError::TooFewParticlesToHoldOut { n_points: 1 })));
    }

    #[test]
    fn low_acceptance_rule_tightens_the_region_only_after_a_poor_acceptance_rate()
    -> Result<(), Box<dyn std::error::Error>> {
        let rule = LowAcceptanceRule::below(0.001)?;
        let config = DpgmmProposalConfig::default().without_held_out_floor();
        let fitter = DpgmmProposalFitter::new(config.with_low_acceptance_rule(rule));
        // The same random stream gives the same mixture, so the floors
        // differ only through the quantile.
        let floor_after = |rates: &[f64]| -> Result<f64, DpgmmProposalError> {
            let mut rng = ChaCha8Rng::seed_from_u64(3);
            let proposal = fitter.fit(&live_points(), FitContext::after(rates), &mut rng)?;
            Ok(proposal.log_ratio_floor())
        };
        let (first, healthy, poor) = (floor_after(&[])?, floor_after(&[0.4])?, floor_after(&[0.4, 0.0002])?);
        assert!((first - healthy).abs() < 1e-12, "{first} vs {healthy}");
        assert!(poor > healthy, "floor {poor} after a poor rate is not above {healthy}");
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let tightened = fitter.fit(&live_points(), FitContext::after(&[0.0002]), &mut rng)?;
        assert_eq!(tightened.rejection_quantile(), Some(rule.rejection_quantile()));
        Ok(())
    }

    #[test]
    fn without_a_rule_the_acceptance_rate_does_not_matter() -> Result<(), DpgmmProposalError> {
        let config = DpgmmProposalConfig::default().without_held_out_floor();
        let fitter = DpgmmProposalFitter::new(config.without_low_acceptance_rule());
        let fit_after = |rates: &[f64]| {
            let mut rng = ChaCha8Rng::seed_from_u64(3);
            fitter.fit(&live_points(), FitContext::after(rates), &mut rng)
        };
        assert_eq!(fit_after(&[1e-9])?, fit_after(&[])?);
        Ok(())
    }
}
