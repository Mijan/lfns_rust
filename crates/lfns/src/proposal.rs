//! Interfaces for proposing new live particles.

use rand::Rng;

use crate::point_matrix::PointMatrix;

/// Draws candidate points for the constrained prior.
///
/// For the evidence estimate to be valid, a proposal must draw **uniformly**
/// from a region of the unit cube that contains every point whose likelihood
/// realization can exceed the current threshold.
pub trait Proposal: Sync {
    /// Writes one candidate into `unit_point`; every coordinate must lie in
    /// `[0, 1]`.
    fn propose<R: Rng + ?Sized>(&self, rng: &mut R, unit_point: &mut [f64]);

    /// Fraction of the density the proposal was fitted with that lies
    /// outside the region it draws from, recorded in the iteration report;
    /// `None` for a proposal that cuts nothing off.
    fn rejection_quantile(&self) -> Option<f64>;
}

/// What the sampler tells a fitter about the run, besides the live
/// particles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitContext {
    previous_acceptance_rate: Option<f64>,
}

impl FitContext {
    /// The context of an iteration that follows one with the given
    /// acceptance rate; `None` for the first iteration, which follows none.
    #[must_use]
    pub fn after(previous_acceptance_rate: Option<f64>) -> Self {
        Self {
            previous_acceptance_rate,
        }
    }

    /// Fraction of the previous iteration's proposals that were accepted;
    /// `None` in the first iteration.
    #[must_use]
    pub fn previous_acceptance_rate(&self) -> Option<f64> {
        self.previous_acceptance_rate
    }
}

/// Builds a [`Proposal`] from the live particles of the current iteration.
///
/// A fitter keeps no state between iterations: the proposal of an iteration
/// depends only on the live particles, on the [`FitContext`] the sampler
/// derives from the run state, and on `rng`, so a resumed run reproduces an
/// uninterrupted one.
pub trait ProposalFitter {
    /// The proposal this fitter produces.
    type Proposal: Proposal;
    /// Why a fit could not be completed.
    type Failure: std::error::Error + Send + Sync + 'static;

    /// Fits a proposal to the surviving live particles.
    ///
    /// # Errors
    ///
    /// Returns `Self::Failure` when no proposal can be built from
    /// `live_points`.
    fn fit<R: Rng + ?Sized>(
        &self,
        live_points: &PointMatrix,
        context: FitContext,
        rng: &mut R,
    ) -> Result<Self::Proposal, Self::Failure>;
}
