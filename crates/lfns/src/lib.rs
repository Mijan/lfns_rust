//! Likelihood-free nested sampling (LF-NS).
//!
//! LF-NS runs nested sampling on the joint space of parameters and
//! likelihood realizations, so it needs only an unbiased estimate of the
//! likelihood (for example from a particle filter) rather than the
//! likelihood itself. For a deterministic model with an exact likelihood it
//! reduces to parallel nested sampling.
//!
//! This crate holds the model-agnostic algorithm: the sampling loop
//! ([`NestedSampler`]), the evidence estimate with its variance and
//! termination criterion ([`EvidenceEstimate`]), reproducible parallel
//! proposal evaluation, and checkpointing ([`CheckpointStore`]). A model
//! enters through [`LikelihoodEstimator`], a proposal scheme through
//! [`ProposalFitter`]; [`DpgmmProposalFitter`] is the scheme of the paper.
//!
//! Reference: Mikelson and Khammash, "Likelihood-free nested sampling for
//! parameter inference of biochemical reaction networks", `PLoS` Comput Biol
//! 16(10): e1008264 (2020).

mod checkpoint_store;
mod dpgmm_proposal;
mod evidence_accumulator;
mod evidence_estimate;
mod iteration_report;
mod json_lines_file;
mod lfns_error;
mod likelihood_estimator;
mod live_set;
mod log_space;
mod nested_sampler;
mod particle;
mod particle_collector;
mod point_matrix;
mod population;
mod posterior;
mod prior_proposal;
mod proposal;
mod rng_streams;
mod run_config;
mod run_recorder;
mod run_state;
mod shrinkage_moments;

pub use checkpoint_store::CheckpointStore;
pub use dpgmm_proposal::{DpgmmProposal, DpgmmProposalConfig, DpgmmProposalError, DpgmmProposalFitter};
pub use evidence_estimate::EvidenceEstimate;
pub use iteration_report::IterationReport;
pub use lfns_error::{ComponentFailure, LfnsError};
pub use likelihood_estimator::LikelihoodEstimator;
pub use live_set::LiveSet;
pub use nested_sampler::{NestedSampler, RunOutcome, StopReason};
pub use particle::Particle;
pub use point_matrix::PointMatrix;
pub use population::Population;
pub use posterior::Posterior;
pub use prior_proposal::{PriorProposal, PriorProposalFitter};
pub use proposal::{Proposal, ProposalFitter};
pub use run_config::RunConfig;
pub use run_recorder::{NoRecorder, RunRecorder};
pub use run_state::RunState;
