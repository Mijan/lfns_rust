//! Size of the live population and of the batch replaced per iteration.

use serde::{Deserialize, Serialize};

use crate::lfns_error::LfnsError;

/// Number of live particles `N` and number `r` replaced in each iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "PopulationRecord", into = "PopulationRecord")]
pub struct Population {
    n_live: u32,
    n_replaced: u32,
}

/// Unvalidated serialized form of [`Population`].
#[derive(Serialize, Deserialize)]
struct PopulationRecord {
    n_live: u32,
    n_replaced: u32,
}

impl Population {
    /// Creates a population of `n_live` particles of which `n_replaced` are
    /// replaced per iteration.
    ///
    /// # Errors
    ///
    /// Returns [`LfnsError::InvalidPopulation`] unless
    /// `1 <= n_replaced < n_live`.
    pub fn new(n_live: u32, n_replaced: u32) -> Result<Self, LfnsError> {
        if n_replaced == 0 || n_replaced >= n_live {
            return Err(LfnsError::InvalidPopulation { n_live, n_replaced });
        }
        Ok(Self { n_live, n_replaced })
    }

    /// Number of live particles `N`.
    #[must_use]
    pub fn n_live(self) -> u32 {
        self.n_live
    }

    /// Number of particles `r` replaced per iteration.
    #[must_use]
    pub fn n_replaced(self) -> u32 {
        self.n_replaced
    }
}

impl TryFrom<PopulationRecord> for Population {
    type Error = LfnsError;

    fn try_from(record: PopulationRecord) -> Result<Self, Self::Error> {
        Self::new(record.n_live, record.n_replaced)
    }
}

impl From<Population> for PopulationRecord {
    fn from(population: Population) -> Self {
        Self {
            n_live: population.n_live,
            n_replaced: population.n_replaced,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_replacing_the_whole_population() {
        assert!(Population::new(10, 10).is_err());
    }

    #[test]
    fn rejects_replacing_nothing() {
        assert!(Population::new(10, 0).is_err());
    }

    #[test]
    fn deserialization_enforces_the_same_invariant() {
        let parsed: Result<Population, _> = serde_json::from_str(r#"{"n_live":5,"n_replaced":9}"#);
        assert!(parsed.is_err());
    }
}
