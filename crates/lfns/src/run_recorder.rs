//! The interface through which a run persists or reports its progress.

use crate::iteration_report::IterationReport;
use crate::lfns_error::ComponentFailure;
use crate::run_state::RunState;

/// Receives the state of a run after the initial population and after every
/// iteration.
pub trait RunRecorder {
    /// Records the initial population, before the first iteration.
    ///
    /// # Errors
    ///
    /// Returns the recorder's own error if the state cannot be recorded.
    fn record_initial_state(&mut self, state: &RunState) -> Result<(), ComponentFailure>;

    /// Records the state after the iteration described by `report`.
    ///
    /// # Errors
    ///
    /// Returns the recorder's own error if the iteration cannot be recorded.
    fn record_iteration(&mut self, state: &RunState, report: &IterationReport) -> Result<(), ComponentFailure>;
}

/// Recorder that keeps nothing; for runs that need no checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoRecorder;

impl RunRecorder for NoRecorder {
    fn record_initial_state(&mut self, _state: &RunState) -> Result<(), ComponentFailure> {
        Ok(())
    }

    fn record_iteration(&mut self, _state: &RunState, _report: &IterationReport) -> Result<(), ComponentFailure> {
        Ok(())
    }
}
