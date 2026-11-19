//! User-chosen settings of a nested sampling run.

use serde::{Deserialize, Serialize};

use crate::lfns_error::LfnsError;
use crate::population::Population;

/// Settings that define a run: population, seed, and stopping rules.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RunConfigRecord", into = "RunConfigRecord")]
pub struct RunConfig {
    population: Population,
    seed: u64,
    termination_tolerance: f64,
    max_iterations: Option<u32>,
    max_proposals_per_iteration: u64,
}

/// Unvalidated serialized form of [`RunConfig`].
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunConfigRecord {
    population: Population,
    seed: u64,
    termination_tolerance: f64,
    max_iterations: Option<u32>,
    max_proposals_per_iteration: u64,
}

impl RunConfig {
    /// Default bound on the achievable improvement of the relative standard
    /// deviation below which a run stops.
    pub const DEFAULT_TERMINATION_TOLERANCE: f64 = 0.01;
    /// Default number of proposals after which an iteration is abandoned.
    pub const DEFAULT_MAX_PROPOSALS_PER_ITERATION: u64 = 100_000_000;

    /// Creates a configuration with default stopping rules.
    #[must_use]
    pub fn new(population: Population, seed: u64) -> Self {
        Self {
            population,
            seed,
            termination_tolerance: Self::DEFAULT_TERMINATION_TOLERANCE,
            max_iterations: None,
            max_proposals_per_iteration: Self::DEFAULT_MAX_PROPOSALS_PER_ITERATION,
        }
    }

    /// Returns a copy that stops once the achievable improvement of the
    /// relative standard deviation is below `tolerance`.
    ///
    /// # Errors
    ///
    /// Returns [`LfnsError::InvalidTolerance`] unless `tolerance` is positive
    /// and finite.
    pub fn with_termination_tolerance(self, tolerance: f64) -> Result<Self, LfnsError> {
        if !(tolerance.is_finite() && tolerance > 0.0) {
            return Err(LfnsError::InvalidTolerance { value: tolerance });
        }
        Ok(Self {
            termination_tolerance: tolerance,
            ..self
        })
    }

    /// Returns a copy that also stops after `max_iterations` iterations.
    #[must_use]
    pub fn with_max_iterations(self, max_iterations: u32) -> Self {
        Self {
            max_iterations: Some(max_iterations),
            ..self
        }
    }

    /// Returns a copy that abandons an iteration after `max_proposals`
    /// proposals.
    ///
    /// # Errors
    ///
    /// Returns [`LfnsError::EmptyProposalBudget`] if `max_proposals` is zero.
    pub fn with_max_proposals_per_iteration(self, max_proposals: u64) -> Result<Self, LfnsError> {
        if max_proposals == 0 {
            return Err(LfnsError::EmptyProposalBudget);
        }
        Ok(Self {
            max_proposals_per_iteration: max_proposals,
            ..self
        })
    }

    /// Live population sizes.
    #[must_use]
    pub fn population(&self) -> Population {
        self.population
    }

    /// Seed from which every random number of the run derives.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Termination tolerance on the achievable improvement.
    #[must_use]
    pub fn termination_tolerance(&self) -> f64 {
        self.termination_tolerance
    }

    /// Iteration limit, if any.
    #[must_use]
    pub fn max_iterations(&self) -> Option<u32> {
        self.max_iterations
    }

    /// Proposal budget of one iteration.
    #[must_use]
    pub fn max_proposals_per_iteration(&self) -> u64 {
        self.max_proposals_per_iteration
    }
}

impl TryFrom<RunConfigRecord> for RunConfig {
    type Error = LfnsError;

    fn try_from(record: RunConfigRecord) -> Result<Self, Self::Error> {
        let config = Self::new(record.population, record.seed)
            .with_termination_tolerance(record.termination_tolerance)?
            .with_max_proposals_per_iteration(record.max_proposals_per_iteration)?;
        Ok(Self {
            max_iterations: record.max_iterations,
            ..config
        })
    }
}

impl From<RunConfig> for RunConfigRecord {
    fn from(config: RunConfig) -> Self {
        Self {
            population: config.population,
            seed: config.seed,
            termination_tolerance: config.termination_tolerance,
            max_iterations: config.max_iterations,
            max_proposals_per_iteration: config.max_proposals_per_iteration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RunConfig {
        RunConfig::new(Population::new(10, 2).expect("valid test population"), 7)
    }

    #[test]
    fn rejects_non_positive_tolerance() {
        assert!(config().with_termination_tolerance(0.0).is_err());
        assert!(config().with_termination_tolerance(f64::NAN).is_err());
    }

    #[test]
    fn rejects_an_empty_proposal_budget() {
        assert!(config().with_max_proposals_per_iteration(0).is_err());
    }

    #[test]
    fn deserialization_enforces_the_tolerance_invariant() {
        let json = r#"{"population":{"n_live":10,"n_replaced":2},"seed":1,
            "termination_tolerance":-1.0,"max_iterations":null,"max_proposals_per_iteration":10}"#;
        assert!(serde_json::from_str::<RunConfig>(json).is_err());
    }
}
