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
}

/// Builds a [`Proposal`] from the live particles of the current iteration.
///
/// A fitter keeps no state between iterations: the proposal of an iteration
/// depends only on the live particles and on `rng`, so a resumed run
/// reproduces an uninterrupted one.
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
    fn fit<R: Rng + ?Sized>(&self, live_points: &PointMatrix, rng: &mut R) -> Result<Self::Proposal, Self::Failure>;
}
