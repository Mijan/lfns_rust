//! Collects the reports of a run, optionally checkpoints it and prints its
//! progress.

use ::lfns::{CheckpointStore, ComponentFailure, IterationReport, LfnsError, RunRecorder, RunState};

/// Worst, median, and best log-likelihood of the live particles: where the
/// population stands, which the evidence alone does not show.
struct LiveLikelihoodSpread {
    worst: f64,
    median: f64,
    best: f64,
}

impl LiveLikelihoodSpread {
    /// The spread of the live particles of `state`; `None` if it has none.
    fn of(state: &RunState) -> Option<Self> {
        let mut log_likelihoods: Vec<f64> = state
            .live()
            .particles()
            .iter()
            .map(::lfns::Particle::log_likelihood)
            .collect();
        log_likelihoods.sort_unstable_by(f64::total_cmp);
        Some(Self {
            worst: *log_likelihoods.first()?,
            median: log_likelihoods[log_likelihoods.len() / 2],
            best: *log_likelihoods.last()?,
        })
    }
}

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

    /// One line per iteration: the evidence, what the iteration cost, and
    /// the log-likelihoods of the live particles after it.
    fn print_progress(state: &RunState, report: &IterationReport) {
        let evidence = report.evidence();
        let live = LiveLikelihoodSpread::of(state).map_or_else(String::new, |spread| {
            format!(
                ", live ln L: worst {:.1}, median {:.1}, best {:.1}",
                spread.worst, spread.median, spread.best
            )
        });
        eprintln!(
            "iteration {:>5}: ln Z = {:>12.4} +- {:.4}, to improve {:.4}, acceptance {:.3}, {} evaluations, \
             {} failed{live}",
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
            Self::print_progress(state, report);
        }
        self.reports.push(*report);
        self.store
            .as_mut()
            .map_or(Ok(()), |store| store.record_iteration(state, report))
    }
}
