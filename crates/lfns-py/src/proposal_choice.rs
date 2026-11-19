//! The proposal scheme selected by name from Python.

use ::dpgmm::DpgmmConfig;
use ::lfns::{
    DpgmmProposal, DpgmmProposalConfig, DpgmmProposalError, DpgmmProposalFitter, FitContext, LowAcceptanceRule,
    PointMatrix, PriorProposal, Proposal, ProposalFitter,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;

/// The proposal settings given in Python as keyword arguments.
pub(crate) struct ProposalSettings<'a> {
    /// Name of the proposal scheme.
    pub(crate) name: &'a str,
    /// Fraction of the mixture's own draws below the floor of the region.
    pub(crate) rejection_quantile: f64,
    /// Gibbs sweeps of the mixture fit.
    pub(crate) n_sweeps: u32,
    /// Acceptance rate below which the next iteration uses
    /// `low_acceptance_quantile`; `None` switches the rule off.
    pub(crate) low_acceptance_threshold: Option<f64>,
    /// Fraction of the mixture's own draws below the floor after a poor
    /// acceptance rate.
    pub(crate) low_acceptance_quantile: f64,
}

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

    /// Turns the settings given in Python into a proposal scheme.
    pub(crate) fn parse(settings: &ProposalSettings<'_>) -> PyResult<Self> {
        match settings.name {
            Self::PRIOR_NAME => Ok(Self::Prior),
            Self::DPGMM_NAME => Ok(Self::Dpgmm(DpgmmProposalFitter::new(Self::dpgmm_config(settings)?))),
            other => Err(PyValueError::new_err(format!(
                "unknown proposal \"{other}\"; pass proposal=\"{}\" or proposal=\"{}\"",
                Self::DPGMM_NAME,
                Self::PRIOR_NAME
            ))),
        }
    }

    /// The settings of the DP-GMM proposal, with the low-acceptance rule if
    /// a threshold was given.
    fn dpgmm_config(settings: &ProposalSettings<'_>) -> PyResult<DpgmmProposalConfig> {
        let mixture = DpgmmConfig::new(settings.n_sweeps).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let config = DpgmmProposalConfig::default()
            .with_mixture_config(mixture)
            .with_rejection_quantile(settings.rejection_quantile)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let Some(threshold) = settings.low_acceptance_threshold else {
            return Ok(config);
        };
        let rule = LowAcceptanceRule::below(threshold)
            .and_then(|rule| rule.with_rejection_quantile(settings.low_acceptance_quantile))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(config.with_low_acceptance_rule(rule))
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
        context: FitContext,
        rng: &mut R,
    ) -> Result<ChosenProposal, DpgmmProposalError> {
        match self {
            Self::Prior => Ok(ChosenProposal::Prior(PriorProposal)),
            Self::Dpgmm(fitter) => fitter.fit(live_points, context, rng).map(ChosenProposal::Dpgmm),
        }
    }
}
