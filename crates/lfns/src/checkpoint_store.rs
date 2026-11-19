//! Persistence of a run in a directory, for inspection and resuming.
//!
//! A run directory holds four files:
//!
//! - `config.json`: the [`RunConfig`] of the run;
//! - `live.json`: the current live particles, replaced atomically after
//!   every iteration;
//! - `dead.jsonl`: one line per iteration with the particles it removed;
//! - `reports.jsonl`: one line per iteration with its [`IterationReport`].
//!
//! `live.json` is written last, so it names the last iteration that was
//! recorded completely. Lines of a later, interrupted iteration are dropped
//! when the run is loaded, and the sampler repeats that iteration exactly,
//! because every random number derives from the seed and the proposal index.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::iteration_report::IterationReport;
use crate::json_lines_file::JsonLinesFile;
use crate::lfns_error::{ComponentFailure, LfnsError};
use crate::particle::Particle;
use crate::run_config::RunConfig;
use crate::run_recorder::RunRecorder;
use crate::run_state::RunState;

/// Contents of `live.json`.
#[derive(Serialize, Deserialize)]
struct LiveRecord {
    iteration: u32,
    n_proposals: u64,
    particles: Vec<Particle>,
}

/// One line of `dead.jsonl`.
#[derive(Serialize, Deserialize)]
struct DeadBatchRecord {
    iteration: u32,
    particles: Vec<Particle>,
}

/// A run directory that records every iteration and can restore the run.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointStore {
    directory: PathBuf,
    config: RunConfig,
    dead_batches: JsonLinesFile,
    reports: JsonLinesFile,
}

impl CheckpointStore {
    const CONFIG_FILE: &'static str = "config.json";
    const LIVE_FILE: &'static str = "live.json";
    const LIVE_STAGING_FILE: &'static str = "live.json.tmp";
    const DEAD_FILE: &'static str = "dead.jsonl";
    const REPORTS_FILE: &'static str = "reports.jsonl";

    /// Creates the run directory of a new run and stores `config` in it.
    ///
    /// # Errors
    ///
    /// Returns [`LfnsError::RunDirectoryInUse`] if the directory already
    /// holds a run, and [`LfnsError::CheckpointIo`] if it cannot be written.
    pub fn create(directory: impl Into<PathBuf>, config: RunConfig) -> Result<Self, LfnsError> {
        let store = Self::in_directory(directory.into(), config);
        std::fs::create_dir_all(&store.directory).map_err(|e| io_error(&store.directory, e))?;
        let config_path = store.directory.join(Self::CONFIG_FILE);
        if config_path.exists() {
            return Err(LfnsError::RunDirectoryInUse { path: store.directory });
        }
        write_json(&config_path, &config)?;
        Ok(store)
    }

    /// Opens the run directory of an existing run.
    ///
    /// # Errors
    ///
    /// Returns [`LfnsError::CheckpointIo`] or [`LfnsError::CheckpointFormat`]
    /// if `config.json` is missing or malformed.
    pub fn open(directory: impl Into<PathBuf>) -> Result<Self, LfnsError> {
        let directory = directory.into();
        let config = read_json(&directory.join(Self::CONFIG_FILE))?;
        Ok(Self::in_directory(directory, config))
    }

    /// The configuration the run was started with.
    #[must_use]
    pub fn config(&self) -> RunConfig {
        self.config
    }

    /// Restores the state after the last completely recorded iteration and
    /// drops the records of an interrupted one.
    ///
    /// # Errors
    ///
    /// Returns a checkpoint error if a file is missing, malformed, or
    /// inconsistent with the others.
    pub fn load_state(&self) -> Result<RunState, LfnsError> {
        let live: LiveRecord = read_json(&self.directory.join(Self::LIVE_FILE))?;
        let n_iterations = live.iteration as usize;
        let batches: Vec<DeadBatchRecord> = Self::read_completed(&self.dead_batches, n_iterations)?;
        self.check_batches_are_consecutive(&batches)?;
        let _reports: Vec<IterationReport> = Self::read_completed(&self.reports, n_iterations)?;
        let dead: Vec<Particle> = batches.into_iter().flat_map(|batch| batch.particles).collect();
        RunState::restore(self.config.population(), live.particles, &dead, live.n_proposals)
    }

    /// The reports of all completely recorded iterations.
    ///
    /// # Errors
    ///
    /// Returns a checkpoint error if `reports.jsonl` cannot be read.
    pub fn load_reports(&self) -> Result<Vec<IterationReport>, LfnsError> {
        self.reports.read_records()
    }

    fn in_directory(directory: PathBuf, config: RunConfig) -> Self {
        let dead_batches = JsonLinesFile::at(directory.join(Self::DEAD_FILE));
        let reports = JsonLinesFile::at(directory.join(Self::REPORTS_FILE));
        Self {
            directory,
            config,
            dead_batches,
            reports,
        }
    }

    /// Reads the first `n_iterations` records of `file` and removes any
    /// records of a later, interrupted iteration from it.
    fn read_completed<T: Serialize + serde::de::DeserializeOwned>(
        file: &JsonLinesFile,
        n_iterations: usize,
    ) -> Result<Vec<T>, LfnsError> {
        let mut records: Vec<T> = file.read_records()?;
        if records.len() < n_iterations {
            return Err(LfnsError::CheckpointInconsistent {
                path: file.path().to_path_buf(),
                reason: format!(
                    "it holds {} iterations but live.json is at iteration {n_iterations}",
                    records.len()
                ),
            });
        }
        if records.len() > n_iterations {
            records.truncate(n_iterations);
            file.write_records(&records)?;
        }
        Ok(records)
    }

    fn check_batches_are_consecutive(&self, batches: &[DeadBatchRecord]) -> Result<(), LfnsError> {
        let misplaced = (1_u32..)
            .zip(batches)
            .find(|(expected, batch)| batch.iteration != *expected);
        match misplaced {
            None => Ok(()),
            Some((expected, batch)) => Err(LfnsError::CheckpointInconsistent {
                path: self.dead_batches.path().to_path_buf(),
                reason: format!("line {expected} holds iteration {}", batch.iteration),
            }),
        }
    }

    /// Replaces `live.json` atomically with the live particles of `state`.
    fn write_live(&self, state: &RunState) -> Result<(), LfnsError> {
        let record = LiveRecord {
            iteration: state.iteration(),
            n_proposals: state.n_proposals(),
            particles: state.live().particles().to_vec(),
        };
        let staging_path = self.directory.join(Self::LIVE_STAGING_FILE);
        let live_path = self.directory.join(Self::LIVE_FILE);
        write_json(&staging_path, &record)?;
        std::fs::rename(&staging_path, &live_path).map_err(|e| io_error(&live_path, e))
    }
}

impl RunRecorder for CheckpointStore {
    fn record_initial_state(&mut self, state: &RunState) -> Result<(), ComponentFailure> {
        Ok(self.write_live(state)?)
    }

    fn record_iteration(&mut self, state: &RunState, report: &IterationReport) -> Result<(), ComponentFailure> {
        let batch = DeadBatchRecord {
            iteration: state.iteration(),
            particles: state.latest_dead_batch().to_vec(),
        };
        self.dead_batches.append_record(&batch)?;
        self.reports.append_record(report)?;
        Ok(self.write_live(state)?)
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), LfnsError> {
    let contents = serde_json::to_string(value).map_err(|source| LfnsError::CheckpointFormat {
        path: path.to_path_buf(),
        source,
    })?;
    std::fs::write(path, contents).map_err(|e| io_error(path, e))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LfnsError> {
    let contents = std::fs::read_to_string(path).map_err(|e| io_error(path, e))?;
    serde_json::from_str(&contents).map_err(|source| LfnsError::CheckpointFormat {
        path: path.to_path_buf(),
        source,
    })
}

fn io_error(path: &Path, source: std::io::Error) -> LfnsError {
    LfnsError::CheckpointIo {
        path: path.to_path_buf(),
        source,
    }
}
