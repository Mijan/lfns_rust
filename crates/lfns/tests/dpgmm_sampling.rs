//! End-to-end runs with the DP-GMM proposal on problems with known evidence.

use std::convert::Infallible;

use dpgmm::DpgmmConfig;
use lfns::{
    DpgmmProposalConfig, DpgmmProposalFitter, IterationReport, LikelihoodEstimator, NestedSampler, Population,
    RunConfig, RunRecorder, RunState, StopReason,
};
use rand::Rng;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Isotropic Gaussian likelihood centred in the unit square. Its width is so
/// small that practically all of its mass lies inside, so the evidence is 1.
struct NarrowGaussian;

impl NarrowGaussian {
    const WIDTH: f64 = 0.02;
}

impl LikelihoodEstimator for NarrowGaussian {
    type Failure = Infallible;

    fn n_parameters(&self) -> usize {
        2
    }

    fn estimate_log_likelihood<R: Rng + ?Sized>(&self, unit_point: &[f64], _rng: &mut R) -> Result<f64, Infallible> {
        let squared_distance: f64 = unit_point.iter().map(|u| (u - 0.5).powi(2)).sum();
        let variance = Self::WIDTH * Self::WIDTH;
        Ok(-(std::f64::consts::TAU * variance).ln() - 0.5 * squared_distance / variance)
    }
}

/// Keeps the reports of a run in memory.
#[derive(Default)]
struct ReportLog {
    reports: Vec<IterationReport>,
}

impl RunRecorder for ReportLog {
    fn record_initial_state(&mut self, _state: &RunState) -> Result<(), lfns::ComponentFailure> {
        Ok(())
    }

    fn record_iteration(&mut self, _state: &RunState, report: &IterationReport) -> Result<(), lfns::ComponentFailure> {
        self.reports.push(*report);
        Ok(())
    }
}

#[test]
fn narrow_peak_is_integrated_with_a_high_acceptance_rate() -> TestResult {
    let config = RunConfig::new(Population::new(120, 20)?, 31).with_max_iterations(200);
    let proposal_config = DpgmmProposalConfig::default().with_mixture_config(DpgmmConfig::new(10)?);
    let sampler = NestedSampler::new(config, NarrowGaussian, DpgmmProposalFitter::new(proposal_config));
    let mut log = ReportLog::default();
    let outcome = sampler.run(sampler.initialize()?, &mut log)?;

    let estimate = outcome.state().estimate_evidence();
    let n_proposals: u64 = log.reports.iter().map(IterationReport::n_proposals).sum();
    let last_acceptance = log.reports.last().ok_or("the run made no iteration")?.acceptance_rate();
    println!(
        "{} iterations, {n_proposals} proposals, ln Z = {:.3} +- {:.3}, last acceptance {last_acceptance:.2}",
        log.reports.len(),
        estimate.log_evidence(),
        estimate.relative_std()
    );

    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    // True ln Z = 0; allow four predicted standard deviations plus the O(1/N) bias.
    assert!(estimate.log_evidence().abs() < 4.0 * estimate.relative_std() + 0.1);
    // Prior proposals would be accepted at the rate of the remaining prior
    // volume, below 1e-3 at the end of this run.
    assert!(last_acceptance > 0.1, "acceptance rate {last_acceptance}");
    Ok(())
}
