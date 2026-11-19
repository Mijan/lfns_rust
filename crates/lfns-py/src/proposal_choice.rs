//! The proposal scheme selected by name from Python.

use ::dpgmm::DpgmmConfig;
use ::lfns::{
    DpgmmProposal, DpgmmProposalConfig, DpgmmProposalError, DpgmmProposalFitter, PointMatrix, PriorProposal, Proposal,
    ProposalFitter,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;

/// A proposal scheme: fits a [`ChosenProposal`] each iteration.
pub(crate) enum ProposalChoice {
    /// Draw from the whole prior.
    Prior,
    /// Draw uniformly from the support of a DP-GMM fit.
    Dpgmm(DpgmmProposalFitter),
}

impl ProposalChoice {
    const PRIOR_NAME: &'static str = "prior";
    const DPGMM_NAME: &'static str = "dpgmm";

    /// Turns the name given in Python into a proposal scheme.
    pub(crate) fn parse(name: &str, rejection_quantile: f64, n_sweeps: u32) -> PyResult<Self> {
        match name {
            Self::PRIOR_NAME => Ok(Self::Prior),
            Self::DPGMM_NAME => {
                let mixture = DpgmmConfig::new(n_sweeps).map_err(|error| PyValueError::new_err(error.to_string()))?;
                let config = DpgmmProposalConfig::default()
                    .with_mixture_config(mixture)
                    .with_rejection_quantile(rejection_quantile)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                Ok(Self::Dpgmm(DpgmmProposalFitter::new(config)))
            }
            other => Err(PyValueError::new_err(format!(
                "unknown proposal \"{other}\"; pass proposal=\"{}\" or proposal=\"{}\"",
                Self::DPGMM_NAME,
                Self::PRIOR_NAME
            ))),
        }
    }
}

/// The proposal of one iteration.
pub(crate) enum ChosenProposal {
    Prior(PriorProposal),
    Dpgmm(DpgmmProposal),
}

impl Proposal for ChosenProposal {
    fn propose<R: Rng + ?Sized>(&self, rng: &mut R, unit_point: &mut [f64]) {
        match self {
            Self::Prior(proposal) => proposal.propose(rng, unit_point),
            Self::Dpgmm(proposal) => proposal.propose(rng, unit_point),
        }
    }
}

impl ProposalFitter for ProposalChoice {
    type Proposal = ChosenProposal;
    type Failure = DpgmmProposalError;

    fn fit<R: Rng + ?Sized>(
        &self,
        live_points: &PointMatrix,
        rng: &mut R,
    ) -> Result<ChosenProposal, DpgmmProposalError> {
        match self {
            Self::Prior => Ok(ChosenProposal::Prior(PriorProposal)),
            Self::Dpgmm(fitter) => fitter.fit(live_points, rng).map(ChosenProposal::Dpgmm),
        }
    }
}
