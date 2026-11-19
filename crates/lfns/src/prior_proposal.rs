//! The trivial proposal: draw from the whole prior.

use std::convert::Infallible;

use rand::{Rng, RngExt};

use crate::point_matrix::PointMatrix;
use crate::proposal::{FitContext, Proposal, ProposalFitter};

/// Proposes uniformly from the whole unit cube.
///
/// Always valid, because the cube contains every possible point, but the
/// acceptance rate falls in proportion to the remaining prior volume. Used
/// for the initial population and as a reference in tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PriorProposal;

impl Proposal for PriorProposal {
    fn propose<R: Rng + ?Sized>(&self, rng: &mut R, unit_point: &mut [f64]) {
        for coordinate in unit_point.iter_mut() {
            *coordinate = rng.random::<f64>();
        }
    }
}

/// Fitter that ignores the live particles and returns [`PriorProposal`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PriorProposalFitter;

impl ProposalFitter for PriorProposalFitter {
    type Proposal = PriorProposal;
    type Failure = Infallible;

    fn fit<R: Rng + ?Sized>(
        &self,
        _live_points: &PointMatrix,
        _context: FitContext,
        _rng: &mut R,
    ) -> Result<PriorProposal, Infallible> {
        Ok(PriorProposal)
    }
}
