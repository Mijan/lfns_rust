//! The birth-death example of the repository, checked against closed-form
//! results for the birth-death process.

use std::path::PathBuf;

use lfns_model::{
    LikelihoodMethod, ParticleFilter, PetabLikelihood, PetabProblem, SimulationScratch, StochasticSimulator,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const TRANSCRIPTION_RATE: f64 = 1.0;
const DEGRADATION_RATE: f64 = 0.1;
const NOISE_STD: f64 = 2.0;
/// Largest molecule count carried by the exact filter; the probability of
/// more than 80 molecules at a stationary mean of 10 is negligible.
const MAX_COUNT: usize = 80;

fn load_problem() -> Result<PetabProblem, lfns_model::ModelError> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/birth_death/problem.yaml");
    PetabProblem::load(path)
}

/// Poisson probabilities `P(n)` for `n = 0..=MAX_COUNT`.
fn poisson_pmf(mean: f64) -> Vec<f64> {
    let mut pmf = vec![(-mean).exp()];
    for n in 1..=MAX_COUNT {
        pmf.push(pmf[n - 1] * mean / f64::from(u32::try_from(n).expect("small count")));
    }
    pmf
}

/// Binomial probabilities `P(j)` for `j = 0..=n` with success probability `p`.
fn binomial_pmf(n: usize, p: f64) -> Vec<f64> {
    let mut pmf = vec![(1.0 - p).powi(i32::try_from(n).expect("small count"))];
    for j in 1..=n {
        let ratio = f64::from(u32::try_from(n - j + 1).expect("small count"))
            / f64::from(u32::try_from(j).expect("small count"));
        pmf.push(pmf[j - 1] * ratio * p / (1.0 - p));
    }
    pmf
}

/// One step of the exact filter. Over a time `interval` each molecule
/// survives with probability `exp(-gamma * interval)` and a Poisson number
/// of new molecules is still present at the end.
fn propagate(distribution: &[f64], interval: f64) -> Vec<f64> {
    if interval <= 0.0 {
        return distribution.to_vec();
    }
    let survival = (-DEGRADATION_RATE * interval).exp();
    let births = poisson_pmf(TRANSCRIPTION_RATE / DEGRADATION_RATE * (1.0 - survival));
    let mut next = vec![0.0; MAX_COUNT + 1];
    for (count, probability) in distribution.iter().enumerate() {
        for (survivors, survive_probability) in binomial_pmf(count, survival).iter().enumerate() {
            for (born, birth_probability) in births.iter().enumerate().take(MAX_COUNT + 1 - survivors) {
                next[survivors + born] += probability * survive_probability * birth_probability;
            }
        }
    }
    next
}

fn normal_density(measurement: f64, count: usize) -> f64 {
    let standardized = (measurement - f64::from(u32::try_from(count).expect("small count"))) / NOISE_STD;
    (-0.5 * standardized * standardized).exp() / (NOISE_STD * std::f64::consts::TAU.sqrt())
}

/// Exact log-likelihood of the example data by forward filtering over the
/// molecule count.
fn exact_log_likelihood(problem: &PetabProblem) -> f64 {
    let mut distribution = vec![0.0; MAX_COUNT + 1];
    distribution[0] = 1.0;
    let (mut time, mut log_likelihood) = (0.0, 0.0);
    for (next_time, readings) in problem.trajectories()[0].time_points() {
        distribution = propagate(&distribution, next_time - time);
        for (count, probability) in distribution.iter_mut().enumerate() {
            *probability *= normal_density(readings[0].measurement, count);
        }
        let evidence: f64 = distribution.iter().sum();
        for probability in &mut distribution {
            *probability /= evidence;
        }
        log_likelihood += evidence.ln();
        time = *next_time;
    }
    log_likelihood
}

#[test]
fn example_problem_is_loaded_as_declared() -> TestResult {
    let problem = load_problem()?;
    assert_eq!(problem.species_ids(), ["mRNA"]);
    assert_eq!(problem.observable_ids(), ["mRNA_read"]);
    assert_eq!(problem.trajectories().len(), 1);
    assert_eq!(problem.trajectories()[0].time_points().len(), 21);
    let estimated: Vec<&str> = problem
        .parameter_space()
        .estimated()
        .iter()
        .map(lfns_model::EstimatedParameter::id)
        .collect();
    assert_eq!(estimated, ["k"]);
    // The prior is uniform in log10(k), so its midpoint is the geometric mean of the bounds.
    let midpoint = problem.parameter_space().estimated_values(&[0.5])[0];
    assert!((midpoint - (0.6310_f64 * 2.5119).sqrt()).abs() < 1e-9);
    Ok(())
}

#[test]
fn simulated_counts_follow_the_poisson_law_of_the_birth_death_process() -> TestResult {
    // Started empty, the count at time t is Poisson with mean k/gamma (1 - exp(-gamma t)).
    let problem = load_problem()?;
    let model = problem.model();
    let simulator = StochasticSimulator::new(model);
    let (mut rng, mut scratch) = (ChaCha8Rng::seed_from_u64(1), SimulationScratch::default());
    let (n_runs, end_time) = (4000_u32, 30.0);
    let (mut sum, mut sum_of_squares) = (0.0, 0.0);
    for _ in 0..n_runs {
        let mut values = vec![0.0; model.value_width()];
        model.write_initial_values(model.default_parameters(), &mut values);
        simulator.advance(&mut values, (0.0, end_time), &mut scratch, &mut rng)?;
        sum += values[0];
        sum_of_squares += values[0] * values[0];
    }
    let mean = sum / f64::from(n_runs);
    let variance = sum_of_squares / f64::from(n_runs) - mean * mean;
    let expected = TRANSCRIPTION_RATE / DEGRADATION_RATE * (1.0 - (-DEGRADATION_RATE * end_time).exp());
    assert!((mean - expected).abs() < 0.25, "mean {mean}, expected {expected}");
    assert!(
        (variance - expected).abs() < 1.0,
        "variance {variance}, expected {expected}"
    );
    Ok(())
}

#[test]
fn particle_filter_estimate_is_unbiased_for_the_exact_likelihood() -> TestResult {
    let problem = load_problem()?;
    let exact = exact_log_likelihood(&problem);
    let parameters = problem.model().default_parameters().to_vec();
    let likelihood = PetabLikelihood::new(problem, LikelihoodMethod::ParticleFilter(ParticleFilter::new(30)?));
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let n_runs = 300_u32;
    let mut ratio_sum = 0.0;
    for _ in 0..n_runs {
        ratio_sum += (likelihood.estimate_for_parameters(&parameters, &mut rng)? - exact).exp();
    }
    let mean_ratio = ratio_sum / f64::from(n_runs);
    println!("exact ln L = {exact:.3}, mean of estimate / exact = {mean_ratio:.3}");
    assert!(
        (mean_ratio - 1.0).abs() < 0.2,
        "mean estimate / exact likelihood = {mean_ratio}"
    );
    Ok(())
}

#[test]
fn a_single_particle_gives_a_finite_estimate() -> TestResult {
    let problem = load_problem()?;
    let parameters = problem.model().default_parameters().to_vec();
    let likelihood = PetabLikelihood::new(problem, LikelihoodMethod::ParticleFilter(ParticleFilter::new(1)?));
    let estimate = likelihood.estimate_for_parameters(&parameters, &mut ChaCha8Rng::seed_from_u64(3))?;
    assert!(estimate.is_finite());
    Ok(())
}
