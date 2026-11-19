//! Failed and invalid likelihood evaluations are handled explicitly.

use lfns::{LfnsError, LikelihoodEstimator, NestedSampler, Population, PriorProposalFitter, RunConfig};
use rand::Rng;

#[derive(Debug, thiserror::Error)]
#[error("the solver did not converge")]
struct SolverFailure;

/// Fails on the upper half of the unit interval, constant below.
struct FailsOnUpperHalf;

impl LikelihoodEstimator for FailsOnUpperHalf {
    type Failure = SolverFailure;

    fn n_parameters(&self) -> usize {
        1
    }

    fn estimate_log_likelihood<R: Rng + ?Sized>(&self, unit_point: &[f64], _rng: &mut R) -> Result<f64, SolverFailure> {
        if unit_point[0] > 0.5 {
            Err(SolverFailure)
        } else {
            Ok(0.0)
        }
    }
}

/// Violates the estimator contract by returning `NaN`.
struct ReturnsNan;

impl LikelihoodEstimator for ReturnsNan {
    type Failure = SolverFailure;

    fn n_parameters(&self) -> usize {
        1
    }

    fn estimate_log_likelihood<R: Rng + ?Sized>(
        &self,
        _unit_point: &[f64],
        _rng: &mut R,
    ) -> Result<f64, SolverFailure> {
        Ok(f64::NAN)
    }
}

fn config() -> Result<RunConfig, LfnsError> {
    Ok(RunConfig::new(Population::new(400, 40)?, 11))
}

#[test]
fn failed_evaluations_count_as_zero_likelihood() -> Result<(), LfnsError> {
    let state = NestedSampler::new(config()?, FailsOnUpperHalf, PriorProposalFitter).initialize()?;
    let n_zero = state
        .live()
        .particles()
        .iter()
        .filter(|p| p.log_likelihood() == f64::NEG_INFINITY)
        .count();
    // Half the prior fails; 400 draws put the count within 200 +- 5 sigma (50).
    assert!(
        (150..250).contains(&n_zero),
        "{n_zero} of 400 particles have zero likelihood"
    );
    // The evidence of a likelihood that is 1 on half the prior is 1/2.
    let evidence = state.estimate_evidence().log_evidence().exp();
    assert!((evidence - 0.5).abs() < 0.125, "evidence {evidence}");
    Ok(())
}

#[test]
fn nan_log_likelihood_is_an_error_not_a_rejection() -> Result<(), LfnsError> {
    let result = NestedSampler::new(config()?, ReturnsNan, PriorProposalFitter).initialize();
    assert!(matches!(result, Err(LfnsError::InvalidLogLikelihood { .. })));
    Ok(())
}
