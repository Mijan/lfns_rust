//! Diagnostics of one completed iteration.

use serde::{Deserialize, Serialize};

use crate::evidence_estimate::EvidenceEstimate;

/// What one iteration cost and where the evidence estimate stands after it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IterationReport {
    iteration: u32,
    #[serde(with = "crate::log_space::log_value")]
    log_threshold: f64,
    n_accepted: u32,
    n_proposals: u64,
    n_failures: u64,
    evidence: EvidenceEstimate,
}

impl IterationReport {
    /// Assembles a report; only a run state, which knows the true counts,
    /// may call this.
    pub(crate) fn of_iteration(
        iteration: u32,
        log_threshold: f64,
        n_accepted: u32,
        (n_proposals, n_failures): (u64, u64),
        evidence: EvidenceEstimate,
    ) -> Self {
        Self {
            iteration,
            log_threshold,
            n_accepted,
            n_proposals,
            n_failures,
            evidence,
        }
    }

    /// Number of the iteration, starting at 1.
    #[must_use]
    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    /// Log-likelihood threshold new particles had to exceed.
    #[must_use]
    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    /// Proposals evaluated to fill the live set.
    #[must_use]
    pub fn n_proposals(&self) -> u64 {
        self.n_proposals
    }

    /// Proposals whose likelihood could not be computed.
    #[must_use]
    pub fn n_failures(&self) -> u64 {
        self.n_failures
    }

    /// Fraction of proposals that were accepted.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        // Counts stay far below 2^53, where the conversion is exact.
        #[allow(clippy::cast_precision_loss)]
        let n_proposals = self.n_proposals as f64;
        f64::from(self.n_accepted) / n_proposals
    }

    /// Evidence estimate after the iteration.
    #[must_use]
    pub fn evidence(&self) -> &EvidenceEstimate {
        &self.evidence
    }
}
