//! Parallel rejection sampling whose result does not depend on timing.
//!
//! Proposals carry a global index and each owns the random stream of that
//! index. Batches of consecutive indices are evaluated in parallel, but
//! their outcomes are consumed strictly in index order and consumption
//! stops at the last particle needed. Which particles are accepted is
//! therefore a function of the seed alone: it depends neither on the number
//! of threads nor on how long individual evaluations take, so slow
//! parameter regions are not under-represented.

use rayon::prelude::*;

use crate::lfns_error::LfnsError;
use crate::likelihood_estimator::LikelihoodEstimator;
use crate::particle::Particle;
use crate::proposal::Proposal;
use crate::rng_streams::RngStreams;

/// Which evaluated particles join the live set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum AcceptanceRule {
    /// Initial population: every draw from the prior is kept.
    Everything,
    /// Constrained prior: keep draws whose log-likelihood exceeds the value.
    Above(f64),
}

impl AcceptanceRule {
    fn accepts(self, log_likelihood: f64) -> bool {
        match self {
            Self::Everything => true,
            Self::Above(log_threshold) => log_likelihood > log_threshold,
        }
    }
}

/// The accepted particles and what it cost to obtain them.
#[derive(Debug, Default)]
pub(crate) struct Collection {
    pub(crate) particles: Vec<Particle>,
    pub(crate) n_proposals: u64,
    pub(crate) n_failures: u64,
}

/// One evaluated proposal.
struct Evaluation {
    particle: Particle,
    failed: bool,
}

/// Collects a fixed number of accepted particles from a proposal.
pub(crate) struct ParticleCollector<'a, L, P> {
    pub(crate) estimator: &'a L,
    pub(crate) proposal: &'a P,
    pub(crate) streams: RngStreams,
    pub(crate) rule: AcceptanceRule,
    pub(crate) max_proposals: u64,
    pub(crate) iteration: u32,
}

impl<L: LikelihoodEstimator, P: Proposal> ParticleCollector<'_, L, P> {
    /// Largest number of proposals evaluated before outcomes are consumed.
    const MAX_BATCH_SIZE: u64 = 1 << 16;

    /// Evaluates proposals starting at global index `first_index` until
    /// `n_needed` have been accepted.
    pub(crate) fn collect(&self, n_needed: u32, first_index: u64) -> Result<Collection, LfnsError> {
        let n_needed = n_needed as usize;
        let mut collection = Collection::default();
        while collection.particles.len() < n_needed {
            let batch_size = self.next_batch_size(&collection, n_needed)?;
            let batch_start = first_index + collection.n_proposals;
            let evaluations: Vec<Result<Evaluation, LfnsError>> = (batch_start..batch_start + batch_size)
                .into_par_iter()
                .map(|index| self.evaluate(index))
                .collect();
            self.consume_in_order(evaluations, n_needed, &mut collection)?;
        }
        Ok(collection)
    }

    /// Consumes outcomes in index order and stops at the last one needed;
    /// later outcomes of the batch are discarded unseen.
    fn consume_in_order(
        &self,
        evaluations: Vec<Result<Evaluation, LfnsError>>,
        n_needed: usize,
        collection: &mut Collection,
    ) -> Result<(), LfnsError> {
        for evaluation in evaluations {
            let evaluation = evaluation?;
            collection.n_proposals += 1;
            collection.n_failures += u64::from(evaluation.failed);
            if self.rule.accepts(evaluation.particle.log_likelihood()) {
                collection.particles.push(evaluation.particle);
            }
            if collection.particles.len() == n_needed {
                break;
            }
        }
        Ok(())
    }

    /// Sizes the next batch to the number of proposals expected to be
    /// needed, so that few evaluated outcomes are discarded.
    fn next_batch_size(&self, collection: &Collection, n_needed: usize) -> Result<u64, LfnsError> {
        let budget_left = self.max_proposals.saturating_sub(collection.n_proposals);
        if budget_left == 0 {
            return Err(LfnsError::AcceptanceCollapsed {
                iteration: self.iteration,
                n_proposals: collection.n_proposals,
                n_accepted: collection.particles.len(),
                n_needed,
            });
        }
        let n_accepted = collection.particles.len() as u64;
        let n_missing = n_needed as u64 - n_accepted;
        let expected = n_missing.saturating_mul(collection.n_proposals + 1) / (n_accepted + 1);
        let n_threads = rayon::current_num_threads() as u64;
        Ok(expected.clamp(n_threads, Self::MAX_BATCH_SIZE).min(budget_left))
    }

    /// Draws and evaluates the proposal that owns stream `index`. A failed
    /// likelihood evaluation counts as zero likelihood.
    fn evaluate(&self, index: u64) -> Result<Evaluation, LfnsError> {
        let mut rng = self.streams.proposal_rng(index);
        let mut unit_point = vec![0.0; self.estimator.n_parameters()];
        self.proposal.propose(&mut rng, &mut unit_point);
        let (log_likelihood, failed) = match self.estimator.estimate_log_likelihood(&unit_point, &mut rng) {
            Ok(log_likelihood) => (log_likelihood, false),
            Err(_) => (f64::NEG_INFINITY, true),
        };
        Ok(Evaluation {
            particle: Particle::new(unit_point, log_likelihood)?,
            failed,
        })
    }
}
