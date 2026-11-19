//! The evidence estimate available after any iteration.

use serde::{Deserialize, Serialize};

/// Estimate of the Bayesian evidence `Z` and of its uncertainty.
///
/// `Z` is split into the part integrated by the dead particles and the part
/// still enclosed by the live particles. Variances are reported relative to
/// `Z^2`, so they stay representable however small `Z` is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvidenceEstimate {
    #[serde(with = "crate::log_space::log_value")]
    log_dead_evidence: f64,
    #[serde(with = "crate::log_space::log_value")]
    log_live_evidence: f64,
    #[serde(with = "crate::log_space::log_value")]
    log_evidence: f64,
    relative_floor_variance: f64,
    relative_live_variance: f64,
}

impl EvidenceEstimate {
    /// Assembles an estimate from its parts; only the accumulator, which
    /// derives them consistently, may call this.
    pub(crate) fn from_parts(
        log_dead_evidence: f64,
        log_live_evidence: f64,
        log_evidence: f64,
        relative_floor_variance: f64,
        relative_live_variance: f64,
    ) -> Self {
        Self {
            log_dead_evidence,
            log_live_evidence,
            log_evidence,
            relative_floor_variance,
            relative_live_variance,
        }
    }

    /// `ln Z`, using dead and live particles.
    #[must_use]
    pub fn log_evidence(&self) -> f64 {
        self.log_evidence
    }

    /// `ln` of the contribution of the dead particles.
    #[must_use]
    pub fn log_dead_evidence(&self) -> f64 {
        self.log_dead_evidence
    }

    /// `ln` of the contribution of the live particles.
    #[must_use]
    pub fn log_live_evidence(&self) -> f64 {
        self.log_live_evidence
    }

    /// Standard deviation of the estimate divided by `Z`.
    #[must_use]
    pub fn relative_std(&self) -> f64 {
        (self.relative_floor_variance + self.relative_live_variance).sqrt()
    }

    /// Lower bound on the relative standard deviation reachable by running
    /// further: the current one with the Monte Carlo error of the live
    /// average removed.
    #[must_use]
    pub fn relative_floor_std(&self) -> f64 {
        self.relative_floor_variance.sqrt()
    }

    /// The most the relative standard deviation can still improve; the run
    /// terminates once this drops below the tolerance.
    #[must_use]
    pub fn achievable_improvement(&self) -> f64 {
        self.relative_std() - self.relative_floor_std()
    }
}
