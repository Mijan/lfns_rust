//! The likelihood-free nested sampling loop.

use crate::iteration_report::IterationReport;
use crate::lfns_error::LfnsError;
use crate::likelihood_estimator::LikelihoodEstimator;
use crate::particle_collector::{AcceptanceRule, Collection, ParticleCollector};
use crate::prior_proposal::PriorProposal;
use crate::proposal::{FitContext, Proposal, ProposalFitter};
use crate::rng_streams::RngStreams;
use crate::run_config::RunConfig;
use crate::run_recorder::RunRecorder;
use crate::run_state::RunState;

/// Why a run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Running further cannot improve the relative standard deviation of
    /// the evidence by more than the termination tolerance.
    Converged,
    /// The configured iteration limit was reached first.
    IterationLimit,
}

/// The final state of a run and the reason it stopped.
#[derive(Debug, Clone, PartialEq)]
pub struct RunOutcome {
    state: RunState,
    stop_reason: StopReason,
}

impl RunOutcome {
    /// The state after the last iteration.
    #[must_use]
    pub fn state(&self) -> &RunState {
        &self.state
    }

    /// Why the run stopped.
    #[must_use]
    pub fn stop_reason(&self) -> StopReason {
        self.stop_reason
    }

    /// Releases the final state, for example to resume it with a different
    /// iteration limit.
    #[must_use]
    pub fn into_state(self) -> RunState {
        self.state
    }
}

/// Parallel likelihood-free nested sampling over the unit cube.
///
/// Each iteration removes the `r` live particles of lowest likelihood, fits
/// a proposal to the survivors, and draws `r` replacements whose likelihood
/// realization exceeds that of the last particle removed.
#[derive(Debug, Clone)]
pub struct NestedSampler<L, F> {
    config: RunConfig,
    estimator: L,
    fitter: F,
}

impl<L: LikelihoodEstimator, F: ProposalFitter> NestedSampler<L, F> {
    /// Creates a sampler for `estimator` that proposes with `fitter`.
    #[must_use]
    pub fn new(config: RunConfig, estimator: L, fitter: F) -> Self {
        Self {
            config,
            estimator,
            fitter,
        }
    }

    /// Draws the initial population from the prior.
    ///
    /// # Errors
    ///
    /// Returns [`LfnsError::InvalidLogLikelihood`] if the estimator returns
    /// `NaN` or `+inf`.
    pub fn initialize(&self) -> Result<RunState, LfnsError> {
        let n_live = self.config.population().n_live();
        let initial = self.collect(&PriorProposal, AcceptanceRule::Everything, n_live, (0, 0))?;
        RunState::begin(self.config.population(), initial)
    }

    /// Advances `state` until the run converges or reaches the iteration
    /// limit, recording every iteration with `recorder`.
    ///
    /// # Errors
    ///
    /// Returns an [`LfnsError`] if the state belongs to a different run, a
    /// proposal cannot be fitted, the acceptance rate collapses, or the
    /// recorder fails.
    pub fn run(&self, mut state: RunState, recorder: &mut impl RunRecorder) -> Result<RunOutcome, LfnsError> {
        self.check_state_belongs_to_this_run(&state)?;
        if state.iteration() == 0 {
            recorder
                .record_initial_state(&state)
                .map_err(|source| LfnsError::Recorder { iteration: 0, source })?;
        }
        loop {
            if let Some(stop_reason) = self.stop_reason(&state) {
                return Ok(RunOutcome { state, stop_reason });
            }
            let report = self.advance(&mut state)?;
            let iteration = report.iteration();
            recorder
                .record_iteration(&state, &report)
                .map_err(|source| LfnsError::Recorder { iteration, source })?;
        }
    }

    /// Performs one iteration.
    fn advance(&self, state: &mut RunState) -> Result<IterationReport, LfnsError> {
        let context = FitContext::after(state.latest_acceptance_rate());
        let log_threshold = state.retire_lowest();
        let iteration = state.iteration();
        let mut fit_rng = RngStreams::new(self.config.seed()).fit_rng(iteration);
        let proposal = self
            .fitter
            .fit(&state.live_coordinates(), context, &mut fit_rng)
            .map_err(|failure| LfnsError::ProposalFit {
                iteration,
                source: Box::new(failure),
            })?;
        let rule = AcceptanceRule::Above(log_threshold);
        let n_replaced = self.config.population().n_replaced();
        let replacements = self.collect(&proposal, rule, n_replaced, (iteration, state.n_proposals()))?;
        Ok(state.admit_replacements((log_threshold, proposal.rejection_quantile()), replacements))
    }

    fn collect<P: Proposal>(
        &self,
        proposal: &P,
        rule: AcceptanceRule,
        n_needed: u32,
        (iteration, first_index): (u32, u64),
    ) -> Result<Collection, LfnsError> {
        let collector = ParticleCollector {
            estimator: &self.estimator,
            proposal,
            streams: RngStreams::new(self.config.seed()),
            rule,
            max_proposals: self.config.max_proposals_per_iteration(),
            iteration,
        };
        collector.collect(n_needed, first_index)
    }

    fn stop_reason(&self, state: &RunState) -> Option<StopReason> {
        if state.estimate_evidence().achievable_improvement() < self.config.termination_tolerance() {
            return Some(StopReason::Converged);
        }
        let limit = self.config.max_iterations()?;
        (state.iteration() >= limit).then_some(StopReason::IterationLimit)
    }

    fn check_state_belongs_to_this_run(&self, state: &RunState) -> Result<(), LfnsError> {
        if state.population() != self.config.population() {
            return Err(LfnsError::IncompatibleState {
                expected: format!("{:?}", self.config.population()),
                found: format!("{:?}", state.population()),
            });
        }
        if state.n_dimensions() != self.estimator.n_parameters() {
            return Err(LfnsError::DimensionMismatch {
                context: "run state handed to the sampler",
                expected: self.estimator.n_parameters(),
                actual: state.n_dimensions(),
            });
        }
        Ok(())
    }
}
