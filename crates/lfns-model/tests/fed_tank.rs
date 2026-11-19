//! The fed-tank example: rate rules, assignment rules, timed events, and
//! per-run conditions, checked against the closed-form solution.

use std::path::PathBuf;

use lfns::LikelihoodEstimator;
use lfns_model::{LikelihoodMethod, ModelError, OdeSettings, ParticleFilter, PetabLikelihood, PetabProblem};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const VOLUME: f64 = 2.0;
const NOISE_STD: f64 = 0.2;
/// Times at which the feed rate changes.
const INTERVAL_STARTS: [f64; 3] = [0.0, 4.0, 8.0];

/// One run of the example: its initial substrate and its feed rates per
/// interval, as written in `conditions.tsv`.
struct Run {
    initial_substrate: f64,
    feeds: [f64; 3],
}

const RUNS: [Run; 2] = [
    Run {
        initial_substrate: 2.0,
        feeds: [0.2, 1.0, 0.0],
    },
    Run {
        initial_substrate: 0.5,
        feeds: [0.6, 0.0, 0.4],
    },
];

fn load_problem() -> Result<PetabProblem, ModelError> {
    PetabProblem::load(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/fed_tank/problem.yaml"))
}

/// Substrate at `time`: within an interval of dilution `d` it relaxes
/// exponentially at rate `d + k` towards `d S_feed / (d + k)`.
fn closed_form_substrate(run: &Run, (k_use, s_feed): (f64, f64), time: f64) -> f64 {
    let mut substrate = run.initial_substrate;
    for (index, start) in INTERVAL_STARTS.iter().enumerate() {
        let end = INTERVAL_STARTS
            .get(index + 1)
            .copied()
            .unwrap_or(f64::INFINITY)
            .min(time);
        let dilution = run.feeds[index] / VOLUME;
        let steady = dilution * s_feed / (dilution + k_use);
        substrate = steady + (substrate - steady) * (-(dilution + k_use) * (end - start)).exp();
        if end >= time {
            break;
        }
    }
    substrate
}

#[test]
fn each_run_follows_its_own_schedule_and_initial_state() -> TestResult {
    let problem = load_problem()?;
    assert_eq!(
        problem
            .conditions()
            .iter()
            .map(lfns_model::ExperimentalCondition::id)
            .collect::<Vec<_>>(),
        ["run_a", "run_b"]
    );
    let estimated = (0.45, 7.0);
    let parameters = problem
        .parameter_space()
        .model_parameters_from_values(&[estimated.0, estimated.1]);
    // Includes the instants at which the feed changes.
    let times = [0.0, 1.0, 3.5, 4.0, 5.25, 8.0, 9.0, 12.0];
    for (condition_index, run) in RUNS.iter().enumerate() {
        let simulated = problem.simulate_deterministic(&parameters, condition_index, &times, OdeSettings::default())?;
        for (time, state) in times.iter().zip(&simulated.species_amounts) {
            let exact = closed_form_substrate(run, estimated, *time);
            assert!(
                (state[0] - exact).abs() < 1e-6,
                "run {condition_index}, t = {time}: {} vs {exact}",
                state[0]
            );
        }
    }
    Ok(())
}

#[test]
fn log_likelihood_matches_the_closed_form() -> TestResult {
    let problem = load_problem()?;
    let likelihood = PetabLikelihood::new(problem, LikelihoodMethod::Deterministic(OdeSettings::default()))?;
    let unit_point = [0.7, 0.3];
    let values = likelihood.problem().parameter_space().estimated_values(&unit_point);
    let mut expected = 0.0;
    for trajectory in likelihood.problem().trajectories() {
        for (time, readings) in trajectory.time_points() {
            let mean = closed_form_substrate(&RUNS[trajectory.condition_index()], (values[0], values[1]), *time);
            let standardized = (readings[0].measurement - mean) / NOISE_STD;
            expected += -0.5 * standardized * standardized - NOISE_STD.ln() - 0.5 * std::f64::consts::TAU.ln();
        }
    }
    let computed = likelihood.estimate_log_likelihood(&unit_point, &mut ChaCha8Rng::seed_from_u64(0))?;
    assert!((computed - expected).abs() < 1e-4, "{computed} vs {expected}");
    Ok(())
}

#[test]
fn likelihood_is_highest_near_the_parameters_that_generated_the_data() -> TestResult {
    let problem = load_problem()?;
    let likelihood = PetabLikelihood::new(problem, LikelihoodMethod::Deterministic(OdeSettings::default()))?;
    let space = likelihood.problem().parameter_space();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut at = |k_use: f64, s_feed: f64| {
        likelihood.estimate_for_parameters(&space.model_parameters_from_values(&[k_use, s_feed]), &mut rng)
    };
    let at_truth = at(0.3, 10.0)?;
    assert!(at_truth > at(0.6, 10.0)? && at_truth > at(0.3, 5.0)? && at_truth > at(0.1, 15.0)?);
    Ok(())
}

#[test]
fn a_model_with_rules_and_events_cannot_be_filtered_stochastically() -> TestResult {
    let outcome = PetabLikelihood::new(
        load_problem()?,
        LikelihoodMethod::ParticleFilter(ParticleFilter::new(10)?),
    );
    assert!(matches!(outcome, Err(ModelError::UnsupportedSbml { .. })));
    Ok(())
}
