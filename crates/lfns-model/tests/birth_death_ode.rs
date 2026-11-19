//! The deterministic birth-death example, checked against the closed-form
//! solution of its differential equation.

use std::path::PathBuf;

use lfns::LikelihoodEstimator;
use lfns_model::{LikelihoodMethod, OdeSettings, PetabLikelihood, PetabProblem};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn load_likelihood() -> Result<PetabLikelihood, lfns_model::ModelError> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/birth_death_ode/problem.yaml");
    PetabLikelihood::new(
        PetabProblem::load(path)?,
        LikelihoodMethod::Deterministic(OdeSettings::default()),
    )
}

/// Log-likelihood from `m(t) = k/g (1 - exp(-g t))` and normal noise.
fn closed_form_log_likelihood(likelihood: &PetabLikelihood, (k, g, sigma): (f64, f64, f64)) -> f64 {
    let readings = likelihood.problem().trajectories()[0].time_points();
    let log_density = |time: f64, measurement: f64| {
        let standardized = (measurement - k / g * (1.0 - (-g * time).exp())) / sigma;
        -0.5 * standardized * standardized - sigma.ln() - 0.5 * std::f64::consts::TAU.ln()
    };
    readings
        .iter()
        .map(|(time, group)| log_density(*time, group[0].measurement))
        .sum()
}

#[test]
fn all_three_parameters_are_estimated_including_the_noise_parameter() -> TestResult {
    let likelihood = load_likelihood()?;
    let ids: Vec<&str> = likelihood
        .problem()
        .parameter_space()
        .estimated()
        .iter()
        .map(lfns_model::EstimatedParameter::id)
        .collect();
    assert_eq!(ids, ["k", "gamma_mRNA", "sigma"]);
    assert_eq!(likelihood.n_parameters(), 3);
    Ok(())
}

#[test]
fn log_likelihood_matches_the_closed_form_across_the_prior() -> TestResult {
    let likelihood = load_likelihood()?;
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for unit_point in [[0.5, 0.5, 0.5], [0.1, 0.9, 0.3], [0.95, 0.05, 0.7], [0.3, 0.2, 0.99]] {
        let values = likelihood.problem().parameter_space().estimated_values(&unit_point);
        let expected = closed_form_log_likelihood(&likelihood, (values[0], values[1], values[2]));
        let computed = likelihood.estimate_log_likelihood(&unit_point, &mut rng)?;
        assert!(
            (computed - expected).abs() < 1e-5 * expected.abs().max(1.0),
            "{computed} vs {expected} at {values:?}"
        );
    }
    Ok(())
}

#[test]
fn deterministic_likelihood_does_not_depend_on_the_random_stream() -> TestResult {
    let likelihood = load_likelihood()?;
    let point = [0.4, 0.6, 0.5];
    let first = likelihood.estimate_log_likelihood(&point, &mut ChaCha8Rng::seed_from_u64(1))?;
    let second = likelihood.estimate_log_likelihood(&point, &mut ChaCha8Rng::seed_from_u64(2))?;
    assert_eq!(first.to_bits(), second.to_bits());
    Ok(())
}
