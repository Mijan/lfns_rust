//! A run is a function of its seed: it does not depend on the number of
//! threads, and a resumed run equals an uninterrupted one.

mod common;

use common::ExponentialPeak;
use lfns::{CheckpointStore, NestedSampler, NoRecorder, Population, PriorProposalFitter, RunConfig, RunState};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const SEED: u64 = 2020;

fn config(max_iterations: u32) -> Result<RunConfig, lfns::LfnsError> {
    Ok(RunConfig::new(Population::new(40, 8)?, SEED)
        .with_termination_tolerance(1e-12)?
        .with_max_iterations(max_iterations))
}

fn sampler(max_iterations: u32) -> Result<NestedSampler<ExponentialPeak, PriorProposalFitter>, lfns::LfnsError> {
    Ok(NestedSampler::new(
        config(max_iterations)?,
        ExponentialPeak,
        PriorProposalFitter,
    ))
}

fn run_with_threads(n_threads: usize) -> Result<RunState, Box<dyn std::error::Error>> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build()?;
    let sampler = sampler(15)?;
    let outcome = pool.install(|| sampler.run(sampler.initialize()?, &mut NoRecorder))?;
    Ok(outcome.into_state())
}

#[test]
fn result_does_not_depend_on_the_number_of_threads() -> TestResult {
    assert_eq!(run_with_threads(1)?, run_with_threads(4)?);
    Ok(())
}

#[test]
fn resumed_run_equals_an_uninterrupted_run() -> TestResult {
    let uninterrupted = sampler(12)?
        .run(sampler(12)?.initialize()?, &mut NoRecorder)?
        .into_state();

    let directory = tempfile::tempdir()?;
    let mut store = CheckpointStore::create(directory.path(), config(5)?)?;
    sampler(5)?.run(sampler(5)?.initialize()?, &mut store)?;

    let mut reopened = CheckpointStore::open(directory.path())?;
    assert_eq!(reopened.config(), config(5)?);
    let restored = reopened.load_state()?;
    assert_eq!(restored.iteration(), 5);
    let resumed = sampler(12)?.run(restored, &mut reopened)?.into_state();

    assert_eq!(resumed, uninterrupted);
    assert_eq!(reopened.load_reports()?.len(), 12);
    assert_eq!(CheckpointStore::open(directory.path())?.load_state()?, uninterrupted);
    Ok(())
}

#[test]
fn records_of_an_interrupted_iteration_are_dropped_on_load() -> TestResult {
    let directory = tempfile::tempdir()?;
    let mut store = CheckpointStore::create(directory.path(), config(3)?)?;
    let finished = sampler(3)?.run(sampler(3)?.initialize()?, &mut store)?.into_state();

    // Simulate a crash after the dead batch of iteration 4 was appended but
    // before live.json was replaced.
    let dead_path = directory.path().join("dead.jsonl");
    let mut contents = std::fs::read_to_string(&dead_path)?;
    let last_line = contents
        .lines()
        .last()
        .ok_or("dead.jsonl is empty")?
        .replace("\"iteration\":3", "\"iteration\":4");
    contents.push_str(&last_line);
    contents.push('\n');
    std::fs::write(&dead_path, contents)?;

    assert_eq!(CheckpointStore::open(directory.path())?.load_state()?, finished);
    assert_eq!(std::fs::read_to_string(&dead_path)?.lines().count(), 3);
    Ok(())
}

#[test]
fn a_run_directory_cannot_be_reused_for_a_new_run() -> TestResult {
    let directory = tempfile::tempdir()?;
    CheckpointStore::create(directory.path(), config(1)?)?;
    assert!(CheckpointStore::create(directory.path(), config(1)?).is_err());
    Ok(())
}
