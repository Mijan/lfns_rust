//! A tighter proposal region for iterations that follow a poor acceptance
//! rate.
//!
//! The floor of the DP-GMM proposal region is a low quantile of the density
//! ratio, so that the region covers almost all of the fitted mixture. Where
//! the mixture describes the constrained prior badly, most of that region
//! lies below the likelihood threshold and almost every proposal is
//! rejected. The rule trades coverage for acceptance only then: after an
//! iteration whose acceptance rate fell below a threshold, the next
//! proposal region is cut at a higher quantile. It switches back as soon as
//! the acceptance rate recovers.

/// Why a low-acceptance rule could not be built.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LowAcceptanceRuleError {
    /// The threshold is not a rate strictly between 0 and 1.
    #[error(
        "acceptance threshold {value} is not strictly between 0 and 1. Pass the acceptance rate \
         below which the tighter region is used, e.g. `LowAcceptanceRule::below(0.001)`."
    )]
    InvalidThreshold {
        /// The rejected value.
        value: f64,
    },
    /// The quantile is not a probability strictly between 0 and 1.
    #[error(
        "rejection quantile {value} is not strictly between 0 and 1. Pass the fraction of the \
         mixture to cut off after a poor acceptance rate, e.g. `with_rejection_quantile(0.2)`."
    )]
    InvalidQuantile {
        /// The rejected value.
        value: f64,
    },
}

/// The rejection quantile to use after an iteration with a poor acceptance
/// rate, and what counts as poor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowAcceptanceRule {
    acceptance_threshold: f64,
    rejection_quantile: f64,
}

impl LowAcceptanceRule {
    /// Default fraction of the mixture's own draws cut off after a poor
    /// acceptance rate.
    pub const DEFAULT_REJECTION_QUANTILE: f64 = 0.2;

    /// Creates a rule that applies after an iteration whose acceptance rate
    /// was below `acceptance_threshold`, with the default quantile.
    ///
    /// # Errors
    ///
    /// Returns [`LowAcceptanceRuleError::InvalidThreshold`] unless
    /// `0 < acceptance_threshold < 1`.
    pub fn below(acceptance_threshold: f64) -> Result<Self, LowAcceptanceRuleError> {
        if !(acceptance_threshold > 0.0 && acceptance_threshold < 1.0) {
            return Err(LowAcceptanceRuleError::InvalidThreshold {
                value: acceptance_threshold,
            });
        }
        Ok(Self {
            acceptance_threshold,
            rejection_quantile: Self::DEFAULT_REJECTION_QUANTILE,
        })
    }

    /// Returns a copy that cuts the region at the given quantile of the
    /// density ratio over the mixture's own draws.
    ///
    /// # Errors
    ///
    /// Returns [`LowAcceptanceRuleError::InvalidQuantile`] unless
    /// `0 < quantile < 1`.
    pub fn with_rejection_quantile(self, quantile: f64) -> Result<Self, LowAcceptanceRuleError> {
        if !(quantile > 0.0 && quantile < 1.0) {
            return Err(LowAcceptanceRuleError::InvalidQuantile { value: quantile });
        }
        Ok(Self {
            rejection_quantile: quantile,
            ..self
        })
    }

    /// Acceptance rate below which the rule applies to the next iteration.
    #[must_use]
    pub fn acceptance_threshold(&self) -> f64 {
        self.acceptance_threshold
    }

    /// Quantile at which the floor is placed when the rule applies.
    #[must_use]
    pub fn rejection_quantile(&self) -> f64 {
        self.rejection_quantile
    }

    /// Whether the rule applies to an iteration that follows one with the
    /// given acceptance rate; the first iteration follows none.
    #[must_use]
    pub fn applies_after(&self, previous_acceptance_rate: Option<f64>) -> bool {
        previous_acceptance_rate.is_some_and(|rate| rate < self.acceptance_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_only_below_the_threshold() -> Result<(), LowAcceptanceRuleError> {
        let rule = LowAcceptanceRule::below(0.001)?;
        assert!(rule.applies_after(Some(0.0005)));
        assert!(!rule.applies_after(Some(0.001)));
        assert!(!rule.applies_after(Some(0.3)));
        Ok(())
    }

    #[test]
    fn never_applies_to_the_first_iteration() -> Result<(), LowAcceptanceRuleError> {
        assert!(!LowAcceptanceRule::below(0.5)?.applies_after(None));
        Ok(())
    }

    #[test]
    fn rejects_thresholds_and_quantiles_that_are_not_probabilities() -> Result<(), LowAcceptanceRuleError> {
        assert!(LowAcceptanceRule::below(0.0).is_err());
        assert!(LowAcceptanceRule::below(1.0).is_err());
        assert!(LowAcceptanceRule::below(f64::NAN).is_err());
        let rule = LowAcceptanceRule::below(0.01)?;
        assert!(rule.with_rejection_quantile(0.0).is_err());
        assert!(rule.with_rejection_quantile(1.5).is_err());
        Ok(())
    }
}
