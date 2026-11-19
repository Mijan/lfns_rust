//! The error type shared by the whole crate.

use std::path::PathBuf;

/// Boxed error of a user-supplied component (proposal fitter or recorder).
pub type ComponentFailure = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Everything that can go wrong while configuring, running, or resuming a
/// nested sampling run.
#[derive(Debug, thiserror::Error)]
pub enum LfnsError {
    /// The live-population sizes are inconsistent.
    #[error(
        "invalid population: n_live = {n_live}, n_replaced = {n_replaced}. \
         Choose 1 <= n_replaced < n_live, e.g. `Population::new(1000, 100)`."
    )]
    InvalidPopulation {
        /// Requested number of live particles.
        n_live: u32,
        /// Requested number of particles replaced per iteration.
        n_replaced: u32,
    },

    /// The termination tolerance is not a positive finite number.
    #[error(
        "invalid termination tolerance {value}. Pass a positive finite number, \
         e.g. `config.with_termination_tolerance(0.01)`."
    )]
    InvalidTolerance {
        /// The rejected tolerance.
        value: f64,
    },

    /// The per-iteration proposal budget is zero.
    #[error(
        "the proposal budget per iteration is 0, so no iteration could ever finish. \
         Pass a positive number to `with_max_proposals_per_iteration`."
    )]
    EmptyProposalBudget,

    /// A particle coordinate lies outside the unit cube.
    #[error(
        "coordinate {index} of a particle is {value}, outside the unit cube [0, 1]. \
         Particles live in prior-CDF space; map physical parameters through the prior CDF first."
    )]
    CoordinateOutsideUnitCube {
        /// Position of the offending coordinate.
        index: usize,
        /// The offending value.
        value: f64,
    },

    /// A log-likelihood is `NaN` or `+inf`.
    #[error(
        "log-likelihood {value} is not a valid logarithm of a finite likelihood. \
         Return a finite value or `f64::NEG_INFINITY` from `estimate_log_likelihood`, \
         and return `Err(..)` when the likelihood cannot be computed."
    )]
    InvalidLogLikelihood {
        /// The rejected value.
        value: f64,
    },

    /// Two components disagree on the number of parameters.
    #[error(
        "{context}: expected {expected} parameters but found {actual}. \
         Use the same parameter dimension for the estimator, the particles, and the checkpoint."
    )]
    DimensionMismatch {
        /// Where the mismatch was detected.
        context: &'static str,
        /// Dimension required by the sampler.
        expected: usize,
        /// Dimension that was supplied.
        actual: usize,
    },

    /// A set of particles has the wrong size.
    #[error(
        "{context}: expected {expected} particles but found {actual}. \
         The run directory does not match this configuration; resume with the configuration \
         stored in the run directory or start a new run in an empty directory."
    )]
    ParticleCountMismatch {
        /// Where the mismatch was detected.
        context: &'static str,
        /// Number of particles required.
        expected: usize,
        /// Number of particles supplied.
        actual: usize,
    },

    /// Too few proposals were accepted within the proposal budget.
    #[error(
        "iteration {iteration}: only {n_accepted} of {n_needed} replacements were accepted after \
         {n_proposals} proposals. Lower the variance of the likelihood estimate (more filter \
         particles), loosen the termination tolerance, or raise the budget with \
         `with_max_proposals_per_iteration`."
    )]
    AcceptanceCollapsed {
        /// Iteration that could not be completed.
        iteration: u32,
        /// Proposals evaluated in that iteration.
        n_proposals: u64,
        /// Replacements accepted so far.
        n_accepted: usize,
        /// Replacements required.
        n_needed: usize,
    },

    /// Every particle of the run has zero likelihood.
    #[error(
        "every particle has zero likelihood, so there is no posterior to assemble. Check that \
         the model can reproduce the data somewhere inside the prior bounds, and inspect the \
         failure counts of the iteration reports."
    )]
    NoLikelihoodMass,

    /// The run state does not belong to this sampler.
    #[error(
        "the run state was produced with {found}, but this sampler is configured with {expected}. \
         Resume with the configuration stored in the run directory."
    )]
    IncompatibleState {
        /// Property required by the sampler.
        expected: String,
        /// Property found in the state.
        found: String,
    },

    /// The proposal could not be fitted to the live particles.
    #[error("iteration {iteration}: fitting the proposal to the live particles failed: {source}")]
    ProposalFit {
        /// Iteration in which the fit failed.
        iteration: u32,
        /// The fitter's own error.
        source: ComponentFailure,
    },

    /// The run recorder failed to persist an iteration.
    #[error("recording iteration {iteration} failed: {source}")]
    Recorder {
        /// Iteration that could not be recorded.
        iteration: u32,
        /// The recorder's own error.
        source: ComponentFailure,
    },

    /// A run directory already holds a run.
    #[error(
        "{path} already contains a run. Resume it with `CheckpointStore::open`, \
         or choose an empty directory for a new run."
    )]
    RunDirectoryInUse {
        /// The occupied directory.
        path: PathBuf,
    },

    /// Reading or writing a checkpoint file failed.
    #[error("checkpoint file {path}: {source}")]
    CheckpointIo {
        /// File that could not be read or written.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// A checkpoint file is not valid JSON of the expected shape.
    #[error("checkpoint file {path} is malformed: {source}")]
    CheckpointFormat {
        /// File that could not be parsed.
        path: PathBuf,
        /// Underlying parse error.
        source: serde_json::Error,
    },

    /// The files of a run directory contradict each other.
    #[error(
        "checkpoint file {path} is inconsistent: {reason}. The run directory was modified or \
         belongs to a different run; restore it from a backup or start a new run."
    )]
    CheckpointInconsistent {
        /// File in which the inconsistency was found.
        path: PathBuf,
        /// What does not match.
        reason: String,
    },
}
