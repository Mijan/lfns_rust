//! The settings of a run assembled from Python keyword arguments.

use std::path::PathBuf;

use ::lfns::{CheckpointStore, LikelihoodEstimator, NestedSampler, Population, RunConfig, RunOutcome, RunState};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::proposal_choice::{ProposalChoice, ProposalSettings};
use crate::report_recorder::ReportRecorder;
use crate::run_result::PyRunResult;
use crate::to_value_error;

/// Where a run is checkpointed and whether it continues an earlier run.
pub(crate) struct Checkpointing {
    /// The run directory, if the run is checkpointed.
    pub(crate) run_directory: Option<PathBuf>,
    /// Whether to continue the run already stored in the directory.
    pub(crate) resume: bool,
    /// Whether to print one line per iteration.
    pub(crate) progress: bool,
}

/// Configuration, proposal scheme, recorder, and, for a resumed run, the
/// state to continue from.
pub(crate) struct RunSettings {
    config: RunConfig,
    fitter: ProposalChoice,
    recorder: ReportRecorder,
    resumed_state: Option<RunState>,
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
            recorder: ReportRecorder::new(None, false),
            resumed_state: None,
        })
    }

    /// Returns the settings with the proposal scheme described by
    /// `proposal`.
    pub(crate) fn with_proposal(self, proposal: &ProposalSettings<'_>) -> PyResult<Self> {
        Ok(Self {
            fitter: ProposalChoice::parse(proposal)?,
            ..self
        })
    }

    /// Returns the settings with the given checkpointing: a new run
    /// directory is created, an existing one is continued if `resume` is
    /// set.
    pub(crate) fn with_checkpointing(self, checkpointing: Checkpointing) -> PyResult<Self> {
        let Checkpointing {
            run_directory,
            resume,
            progress,
        } = checkpointing;
        let Some(directory) = run_directory else {
            return Ok(Self {
                recorder: ReportRecorder::new(None, progress),
                ..self
            });
        };
        if !resume {
            let store = CheckpointStore::create(directory, self.config).map_err(to_value_error)?;
            return Ok(Self {
                recorder: ReportRecorder::new(Some(store), progress),
                ..self
            });
        }
        let store = CheckpointStore::open(directory).map_err(to_value_error)?;
        let stored = store.config();
        if (stored.population(), stored.seed()) != (self.config.population(), self.config.seed()) {
            return Err(PyValueError::new_err(format!(
                "the run directory holds a run with {:?} and seed {}, but this call asks for {:?} and seed {}; \
                 resume with the same n_live, n_replaced, and seed",
                stored.population(),
                stored.seed(),
                self.config.population(),
                self.config.seed()
            )));
        }
        let resumed_state = Some(store.load_state().map_err(to_value_error)?);
        Ok(Self {
            recorder: ReportRecorder::new(Some(store), progress),
            resumed_state,
            ..self
        })
    }

    /// Runs nested sampling on `estimator` with these settings. The
    /// interpreter is released for the duration of the run, so worker
    /// threads can call back into Python.
    pub(crate) fn run_sampler<L: LikelihoodEstimator + Send>(
        self,
        py: Python<'_>,
        estimator: L,
    ) -> PyResult<PyRunResult> {
        let Self {
            config,
            fitter,
            mut recorder,
            resumed_state,
        } = self;
        let sampler = NestedSampler::new(config, estimator, fitter);
        let outcome: Result<RunOutcome, _> = py.detach(|| {
            let state = match resumed_state {
                Some(state) => state,
                None => sampler.initialize()?,
            };
            sampler.run(state, &mut recorder)
        });
        let outcome = outcome.map_err(to_value_error)?;
        let stop_reason = outcome.stop_reason();
        let reports = recorder.into_reports().map_err(to_value_error)?;
        Ok(PyRunResult::new(outcome.into_state(), reports, stop_reason))
    }
}
