//! Collects the reports of a run, optionally checkpoints it and prints its
//! progress.

use ::lfns::{CheckpointStore, ComponentFailure, IterationReport, LfnsError, RunRecorder, RunState};

/// Keeps every iteration report in memory, forwards the run to a checkpoint
/// store if one was requested, and prints one line per iteration if asked.
pub(crate) struct ReportRecorder {
    reports: Vec<IterationReport>,
    store: Option<CheckpointStore>,
    prints_progress: bool,
}

impl ReportRecorder {
    pub(crate) fn new(store: Option<CheckpointStore>, prints_progress: bool) -> Self {
        Self {
            reports: Vec::new(),
            store,
            prints_progress,
        }
    }

    /// The reports of the whole run: with a checkpoint store, those of all
    /// recorded iterations, including the ones before a resume.
    pub(crate) fn into_reports(self) -> Result<Vec<IterationReport>, LfnsError> {
        match self.store {
            Some(store) => store.load_reports(),
            None => Ok(self.reports),
        }
    }

    fn print_progress(report: &IterationReport) {
        let evidence = report.evidence();
        eprintln!(
            "iteration {:>5}: ln Z = {:>12.4} +- {:.4}, to improve {:.4}, acceptance {:.3}, {} evaluations, {} failed",
            report.iteration(),
            evidence.log_evidence(),
            evidence.relative_std(),
            evidence.achievable_improvement(),
            report.acceptance_rate(),
            report.n_proposals(),
            report.n_failures(),
        );
    }
}

impl RunRecorder for ReportRecorder {
    fn record_initial_state(&mut self, state: &RunState) -> Result<(), ComponentFailure> {
        if self.prints_progress {
            eprintln!(
                "initial population of {} points evaluated",
                state.live().particles().len()
            );
        }
        self.store
            .as_mut()
            .map_or(Ok(()), |store| store.record_initial_state(state))
    }

    fn record_iteration(&mut self, state: &RunState, report: &IterationReport) -> Result<(), ComponentFailure> {
        if self.prints_progress {
            Self::print_progress(report);
        }
        self.reports.push(*report);
        self.store
            .as_mut()
            .map_or(Ok(()), |store| store.record_iteration(state, report))
    }
}
