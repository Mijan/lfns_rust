//! Collects the reports of a run and optionally checkpoints it.

use ::lfns::{CheckpointStore, ComponentFailure, IterationReport, RunRecorder, RunState};

/// Keeps every iteration report in memory and forwards the run to a
/// checkpoint store if one was requested.
pub(crate) struct ReportRecorder {
    reports: Vec<IterationReport>,
    store: Option<CheckpointStore>,
}

impl ReportRecorder {
    pub(crate) fn new(store: Option<CheckpointStore>) -> Self {
        Self {
            reports: Vec::new(),
            store,
        }
    }

    /// Releases the collected reports.
    pub(crate) fn into_reports(self) -> Vec<IterationReport> {
        self.reports
    }
}

impl RunRecorder for ReportRecorder {
    fn record_initial_state(&mut self, state: &RunState) -> Result<(), ComponentFailure> {
        self.store
            .as_mut()
            .map_or(Ok(()), |store| store.record_initial_state(state))
    }

    fn record_iteration(&mut self, state: &RunState, report: &IterationReport) -> Result<(), ComponentFailure> {
        self.reports.push(*report);
        self.store
            .as_mut()
            .map_or(Ok(()), |store| store.record_iteration(state, report))
    }
}
