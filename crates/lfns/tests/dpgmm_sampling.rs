//! End-to-end runs with the DP-GMM proposal on problems with known evidence.

use std::convert::Infallible;

use dpgmm::DpgmmConfig;
use lfns::{
    CheckpointStore, DpgmmProposalConfig, DpgmmProposalFitter, IterationReport, LikelihoodEstimator, LowAcceptanceRule,
    NestedSampler, NoRecorder, Population, RunConfig, RunRecorder, RunState, StopReason,
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

/// A fitter whose low-acceptance rule is switched on and off by this
/// problem's acceptance rates, with a recovery threshold above the acceptance
/// threshold, so which region an iteration uses depends on the history.
fn fitter_with_history_dependent_rule() -> Result<DpgmmProposalFitter, Box<dyn std::error::Error>> {
    let rule = LowAcceptanceRule::below(0.6)?
        .with_rejection_quantile(0.3)?
        .with_recovery_threshold(0.75)?;
    let config = DpgmmProposalConfig::default()
        .with_mixture_config(DpgmmConfig::new(10)?)
        .with_low_acceptance_rule(rule);
    Ok(DpgmmProposalFitter::new(config))
}

#[test]
fn resumed_run_with_the_low_acceptance_rule_equals_an_uninterrupted_run() -> TestResult {
    let population = Population::new(120, 20)?;
    let config = |max_iterations: u32| RunConfig::new(population, 31).with_max_iterations(max_iterations);
    let sampler = |max_iterations: u32| -> Result<_, Box<dyn std::error::Error>> {
        Ok(NestedSampler::new(
            config(max_iterations),
            NarrowGaussian,
            fitter_with_history_dependent_rule()?,
        ))
    };
    let uninterrupted = sampler(8)?
        .run(sampler(8)?.initialize()?, &mut NoRecorder)?
        .into_state();

    let directory = tempfile::tempdir()?;
    let mut store = CheckpointStore::create(directory.path(), config(3))?;
    sampler(3)?.run(sampler(3)?.initialize()?, &mut store)?;
    let mut reopened = CheckpointStore::open(directory.path())?;
    let restored = reopened.load_state()?;
    assert!(restored.latest_acceptance_rate().is_some());
    let resumed = sampler(8)?.run(restored, &mut reopened)?.into_state();

    assert_eq!(resumed, uninterrupted);
    Ok(())
}
