//! A floor of the DP-GMM proposal region placed on live particles the
//! mixture was not fitted to.
//!
//! The proposal region is where the mixture density relative to the prior
//! exceeds a floor, and whatever part of the constrained prior lies below
//! the floor is never proposed: the run's volume bookkeeping then counts a
//! region that was not sampled. A quantile of the ratio over the mixture's
//! own draws says how much of the *mixture* is cut off, which is a
//! different share of the constrained prior in every dimension (a Gaussian
//! and a uniform region put their mass at different radii, and the gap
//! grows with the dimension).
//!
//! The live particles are themselves draws from the constrained prior. Set
//! some of them aside, fit the mixture to the others, and place the floor
//! at a quantile of the ratio over the particles set aside: that quantile
//! is, up to sampling error, the share of the constrained prior given up,
//! whatever the dimension and however well the mixture fits. The price of
//! a poor fit is then a low acceptance rate, which is visible, in place of
//! a lost region, which is not.

/// Why a held-out floor could not be built.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HeldOutFloorError {
    /// The share given up is not strictly between 0 and 1.
    #[error(
        "level-set share {value} is not strictly between 0 and 1. Pass the share of the constrained \
         prior the proposal may give up, e.g. `HeldOutFloor::giving_up(0.01)`."
    )]
    InvalidLevelSetShare {
        /// The rejected value.
        value: f64,
    },
    /// The fraction set aside is not strictly between 0 and 1.
    #[error(
        "held-out fraction {value} is not strictly between 0 and 1. Pass the fraction of the live \
         particles kept out of the mixture fit, e.g. `with_held_out_fraction(0.2)`."
    )]
    InvalidHeldOutFraction {
        /// The rejected value.
        value: f64,
    },
}

/// How much of the constrained prior the proposal region may give up, and
/// how many live particles are kept out of the fit to measure it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeldOutFloor {
    level_set_share: f64,
    held_out_fraction: f64,
}

impl HeldOutFloor {
    /// Default share of the constrained prior given up per iteration.
    pub const DEFAULT_LEVEL_SET_SHARE: f64 = 0.01;
    /// Default fraction of the live particles kept out of the mixture fit.
    pub const DEFAULT_HELD_OUT_FRACTION: f64 = 0.2;

    /// A floor below which the given share of the held-out particles lies.
    ///
    /// # Errors
    ///
    /// Returns [`HeldOutFloorError::InvalidLevelSetShare`] unless
    /// `0 < level_set_share < 1`.
    pub fn giving_up(level_set_share: f64) -> Result<Self, HeldOutFloorError> {
        if !(level_set_share > 0.0 && level_set_share < 1.0) {
            return Err(HeldOutFloorError::InvalidLevelSetShare { value: level_set_share });
        }
        Ok(Self {
            level_set_share,
            held_out_fraction: Self::DEFAULT_HELD_OUT_FRACTION,
        })
    }

    /// Returns a copy that keeps the given fraction of the live particles
    /// out of the mixture fit. More held-out particles place the floor more
    /// precisely and leave fewer for the fit.
    ///
    /// # Errors
    ///
    /// Returns [`HeldOutFloorError::InvalidHeldOutFraction`] unless
    /// `0 < fraction < 1`.
    pub fn with_held_out_fraction(self, fraction: f64) -> Result<Self, HeldOutFloorError> {
        if !(fraction > 0.0 && fraction < 1.0) {
            return Err(HeldOutFloorError::InvalidHeldOutFraction { value: fraction });
        }
        Ok(Self {
            held_out_fraction: fraction,
            ..self
        })
    }

    /// Share of the constrained prior the proposal region gives up.
    #[must_use]
    pub fn level_set_share(&self) -> f64 {
        self.level_set_share
    }

    /// Fraction of the live particles kept out of the mixture fit.
    #[must_use]
    pub fn held_out_fraction(&self) -> f64 {
        self.held_out_fraction
    }

    /// How many of `n_points` live particles are kept out of the fit: at
    /// least one, and never all of them.
    pub(crate) fn n_held_out(&self, n_points: usize) -> usize {
        // The product lies in [0, n_points], so the conversion neither
        // truncates meaningfully nor loses a sign.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let n_held_out = (self.held_out_fraction * n_points as f64).round() as usize;
        n_held_out.clamp(1, n_points.saturating_sub(1).max(1))
    }
}

impl Default for HeldOutFloor {
    fn default() -> Self {
        Self {
            level_set_share: Self::DEFAULT_LEVEL_SET_SHARE,
            held_out_fraction: Self::DEFAULT_HELD_OUT_FRACTION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_outside_the_unit_interval_are_rejected() {
        for value in [0.0, 1.0, -0.1, f64::NAN] {
            assert!(matches!(
                HeldOutFloor::giving_up(value),
                Err(HeldOutFloorError::InvalidLevelSetShare { .. })
            ));
            assert!(matches!(
                HeldOutFloor::default().with_held_out_fraction(value),
                Err(HeldOutFloorError::InvalidHeldOutFraction { .. })
            ));
        }
    }

    #[test]
    fn some_particles_are_held_out_and_some_are_left_for_the_fit() -> Result<(), HeldOutFloorError> {
        let floor = HeldOutFloor::giving_up(0.01)?.with_held_out_fraction(0.2)?;
        assert_eq!(floor.n_held_out(2000), 400);
        assert_eq!(floor.n_held_out(3), 1);
        assert_eq!(HeldOutFloor::default().with_held_out_fraction(0.99)?.n_held_out(10), 9);
        Ok(())
    }
}
