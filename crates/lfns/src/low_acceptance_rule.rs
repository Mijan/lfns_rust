//! A tighter proposal region for iterations that follow a poor acceptance
//! rate.
//!
//! The floor of the DP-GMM proposal region is a low quantile of the density
//! ratio, so that the region covers almost all of the fitted mixture. Where
//! the mixture describes the constrained prior badly, most of that region
//! lies below the likelihood threshold and almost every proposal is
//! rejected. The rule trades coverage for acceptance only then: after an
//! iteration whose acceptance rate fell below a threshold, the next
//! proposal region is cut at a higher quantile.
//!
//! The tighter region stays in use until an iteration reaches the recovery
//! threshold, and the wide one is used again from the next iteration on.
//! With the recovery threshold equal to the acceptance threshold (the
//! default) the rule looks at the previous iteration alone and switches
//! back as soon as it was not poor; a higher recovery threshold keeps the
//! tighter region through the acceptance rates in between, so the regions
//! do not alternate from one iteration to the next. Which region applies is
//! a function of the acceptance rates of all earlier iterations, so a
//! resumed run decides as an uninterrupted one does.

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
    /// The recovery threshold is below the acceptance threshold, or not a
    /// rate below 1.
    #[error(
        "recovery threshold {value} is not between the acceptance threshold {acceptance_threshold} \
         and 1. Pass the acceptance rate from which the wide region is used again, at least the \
         acceptance threshold, e.g. `with_recovery_threshold(0.2)`."
    )]
    InvalidRecoveryThreshold {
        /// The rejected value.
        value: f64,
        /// The acceptance threshold of the rule.
        acceptance_threshold: f64,
    },
}

/// The rejection quantile to use after an iteration with a poor acceptance
/// rate, and what counts as poor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowAcceptanceRule {
    acceptance_threshold: f64,
    recovery_threshold: f64,
    rejection_quantile: f64,
}

impl LowAcceptanceRule {
    /// Default acceptance rate below which the rule applies: one accepted
    /// proposal in a thousand.
    pub const DEFAULT_ACCEPTANCE_THRESHOLD: f64 = 0.001;

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
            recovery_threshold: acceptance_threshold,
            rejection_quantile: Self::DEFAULT_REJECTION_QUANTILE,
        })
    }

    /// Returns a copy that keeps the tighter region until an iteration's
    /// acceptance rate reaches `recovery_threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`LowAcceptanceRuleError::InvalidRecoveryThreshold`] unless
    /// `acceptance_threshold <= recovery_threshold < 1`.
    pub fn with_recovery_threshold(self, recovery_threshold: f64) -> Result<Self, LowAcceptanceRuleError> {
        if !(recovery_threshold >= self.acceptance_threshold && recovery_threshold < 1.0) {
            return Err(LowAcceptanceRuleError::InvalidRecoveryThreshold {
                value: recovery_threshold,
                acceptance_threshold: self.acceptance_threshold,
            });
        }
        Ok(Self {
            recovery_threshold,
            ..self
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

    /// Acceptance rate from which the rule stops applying to the next
    /// iteration.
    #[must_use]
    pub fn recovery_threshold(&self) -> f64 {
        self.recovery_threshold
    }

    /// Quantile at which the floor is placed when the rule applies.
    #[must_use]
    pub fn rejection_quantile(&self) -> f64 {
        self.rejection_quantile
    }

    /// Whether the rule applies to the iteration that follows those with the
    /// given acceptance rates, oldest first; the first iteration follows
    /// none. An acceptance rate below the acceptance threshold switches the
    /// rule on, one at or above the recovery threshold switches it off, and
    /// one in between leaves it as it was.
    #[must_use]
    pub fn applies_after(&self, acceptance_rates: &[f64]) -> bool {
        acceptance_rates.iter().fold(false, |applies, rate| {
            if *rate < self.acceptance_threshold {
                true
            } else if *rate >= self.recovery_threshold {
                false
            } else {
                applies
            }
        })
    }
}

impl Default for LowAcceptanceRule {
    /// The rule with [`Self::DEFAULT_ACCEPTANCE_THRESHOLD`] and
    /// [`Self::DEFAULT_REJECTION_QUANTILE`].
    fn default() -> Self {
        Self {
            acceptance_threshold: Self::DEFAULT_ACCEPTANCE_THRESHOLD,
            recovery_threshold: Self::DEFAULT_ACCEPTANCE_THRESHOLD,
            rejection_quantile: Self::DEFAULT_REJECTION_QUANTILE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_only_after_a_rate_below_the_threshold() -> Result<(), LowAcceptanceRuleError> {
        let rule = LowAcceptanceRule::below(0.001)?;
        assert!(rule.applies_after(&[0.3, 0.0005]));
        assert!(!rule.applies_after(&[0.3, 0.001]));
        assert!(!rule.applies_after(&[0.3]));
        Ok(())
    }

    #[test]
    fn never_applies_to_the_first_iteration() -> Result<(), LowAcceptanceRuleError> {
        assert!(!LowAcceptanceRule::below(0.5)?.applies_after(&[]));
        Ok(())
    }

    #[test]
    fn without_a_recovery_threshold_only_the_previous_iteration_counts() -> Result<(), LowAcceptanceRuleError> {
        let rule = LowAcceptanceRule::below(0.01)?;
        assert!(!rule.applies_after(&[0.005, 0.05]));
        assert!(rule.applies_after(&[0.05, 0.005]));
        Ok(())
    }

    #[test]
    fn stays_on_between_the_two_thresholds_until_the_rate_recovers() -> Result<(), LowAcceptanceRuleError> {
        let rule = LowAcceptanceRule::below(0.01)?.with_recovery_threshold(0.2)?;
        // Switched on by 0.005, kept on by 0.05 and 0.15, switched off by 0.2.
        assert!(rule.applies_after(&[0.3, 0.005]));
        assert!(rule.applies_after(&[0.3, 0.005, 0.05, 0.15]));
        assert!(!rule.applies_after(&[0.3, 0.005, 0.05, 0.2]));
        // Between the thresholds without ever having been poor: stays off.
        assert!(!rule.applies_after(&[0.3, 0.05, 0.15]));
        // Off again, then on again.
        assert!(rule.applies_after(&[0.005, 0.25, 0.1, 0.002, 0.1]));
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

    #[test]
    fn rejects_a_recovery_threshold_below_the_acceptance_threshold() -> Result<(), LowAcceptanceRuleError> {
        let rule = LowAcceptanceRule::below(0.01)?;
        assert!(rule.with_recovery_threshold(0.005).is_err());
        assert!(rule.with_recovery_threshold(1.0).is_err());
        assert!(rule.with_recovery_threshold(f64::NAN).is_err());
        assert!(rule.with_recovery_threshold(0.01).is_ok());
        Ok(())
    }
}
