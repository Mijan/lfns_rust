//! The evidence estimate and its predicted error agree with the spread of
//! repeated runs on a problem with a known evidence.

mod common;

use common::ExponentialPeak;
use lfns::{LfnsError, NestedSampler, NoRecorder, Population, PriorProposalFitter, RunConfig};

const N_RUNS: u64 = 150;
const N_ITERATIONS: u32 = 40;

/// Exact evidence of [`ExponentialPeak`]: the integral of its likelihood over the unit interval.
fn exact_evidence() -> f64 {
    1.0 - (-ExponentialPeak::RATE).exp()
}

struct RunSummary {
    evidence: f64,
    predicted_relative_std: f64,
    posterior_mean: f64,
}

fn run_once(seed: u64) -> Result<RunSummary, LfnsError> {
    // A tolerance this small never triggers, so every run stops at the limit.
    let config = RunConfig::new(Population::new(50, 5)?, seed)
        .with_termination_tolerance(1e-12)?
        .with_max_iterations(N_ITERATIONS);
    let sampler = NestedSampler::new(config, ExponentialPeak, PriorProposalFitter);
    let outcome = sampler.run(sampler.initialize()?, &mut NoRecorder)?;
    let estimate = outcome.state().estimate_evidence();
    let posterior = outcome.state().assemble_posterior()?;
    let posterior_mean = posterior
        .weighted_particles()
        .map(|(p, w)| p.unit_coordinates()[0] * w.exp())
        .sum();
    Ok(RunSummary {
        evidence: estimate.log_evidence().exp(),
        predicted_relative_std: estimate.relative_std(),
        posterior_mean,
    })
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0.0), |(sum, count), value| (sum + value, count + 1.0));
    sum / count
}

#[test]
fn evidence_and_predicted_error_match_repeated_runs() -> Result<(), LfnsError> {
    let runs: Vec<RunSummary> = (0..N_RUNS).map(run_once).collect::<Result<_, _>>()?;
    let mean_evidence = mean(runs.iter().map(|run| run.evidence));
    let variance = mean(runs.iter().map(|run| (run.evidence - mean_evidence).powi(2)));
    let empirical_relative_std = variance.sqrt() / mean_evidence;
    let predicted_relative_std = mean(runs.iter().map(|run| run.predicted_relative_std));
    let mean_posterior_mean = mean(runs.iter().map(|run| run.posterior_mean));
    println!(
        "mean Z = {mean_evidence:.4} (exact {:.4}), relative std: empirical {empirical_relative_std:.4}, \
         predicted {predicted_relative_std:.4}, posterior mean {mean_posterior_mean:.5}",
        exact_evidence()
    );

    // The estimator carries an O(1/N) upward bias, far below its spread.
    let relative_bias = mean_evidence / exact_evidence() - 1.0;
    assert!((-0.05..0.20).contains(&relative_bias), "relative bias {relative_bias}");
    let error_ratio = predicted_relative_std / empirical_relative_std;
    assert!(
        (0.7..1.4).contains(&error_ratio),
        "predicted / empirical relative std = {error_ratio}"
    );
    let posterior_mean_ratio = mean_posterior_mean * ExponentialPeak::RATE;
    assert!(
        (0.85..1.15).contains(&posterior_mean_ratio),
        "posterior mean ratio {posterior_mean_ratio}"
    );
    Ok(())
}
