//! The settings of a run assembled from Python keyword arguments.

use std::path::PathBuf;

use ::lfns::{CheckpointStore, Population, RunConfig};
use pyo3::prelude::*;

use crate::proposal_choice::ProposalChoice;
use crate::report_recorder::ReportRecorder;
use crate::to_value_error;

/// Configuration, proposal scheme, and recorder of one run.
pub(crate) struct RunSettings {
    config: RunConfig,
    fitter: ProposalChoice,
    recorder: ReportRecorder,
}

impl RunSettings {
    /// Starts from the population `(n_live, n_replaced)` and the stopping
    /// rules, with the prior proposal and no checkpointing.
    pub(crate) fn new(
        population: (u32, u32),
        seed: u64,
        termination_tolerance: f64,
        max_iterations: Option<u32>,
    ) -> PyResult<Self> {
        let population = Population::new(population.0, population.1).map_err(to_value_error)?;
        let config = RunConfig::new(population, seed)
            .with_termination_tolerance(termination_tolerance)
            .map_err(to_value_error)?;
        let config = max_iterations.map_or(config, |limit| config.with_max_iterations(limit));
        Ok(Self {
            config,
            fitter: ProposalChoice::Prior,
            recorder: ReportRecorder::new(None),
        })
    }

    /// Returns the settings with the proposal scheme named `proposal`.
    pub(crate) fn with_proposal(self, proposal: &str, rejection_quantile: f64, n_sweeps: u32) -> PyResult<Self> {
        Ok(Self {
            fitter: ProposalChoice::parse(proposal, rejection_quantile, n_sweeps)?,
            ..self
        })
    }

    /// Returns the settings with checkpointing into `run_directory`, if one
    /// is given.
    pub(crate) fn with_run_directory(self, run_directory: Option<PathBuf>) -> PyResult<Self> {
        let store = run_directory
            .map(|directory| CheckpointStore::create(directory, self.config))
            .transpose();
        Ok(Self {
            recorder: ReportRecorder::new(store.map_err(to_value_error)?),
            ..self
        })
    }

    /// Releases the configuration, the proposal scheme, and the recorder.
    pub(crate) fn into_parts(self) -> (RunConfig, ProposalChoice, ReportRecorder) {
        (self.config, self.fitter, self.recorder)
    }
}
