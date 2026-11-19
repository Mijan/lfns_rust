//! The Normal-Wishart distribution over a Gaussian's mean and precision.

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::dpgmm_error::DpgmmError;
use crate::wishart::{Wishart, WishartDraw};

/// A mean and a precision matrix drawn from a [`NormalWishart`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GaussianDraw {
    pub(crate) mean: DVector<f64>,
    pub(crate) precision: WishartDraw,
}

/// Whether an observation joins or leaves the data a posterior is
/// conditioned on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObservationChange {
    /// One more observation.
    Absorbed,
    /// One observation fewer.
    Released,
}

impl ObservationChange {
    /// `+1` for an observation gained, `-1` for one lost: the change of the
    /// observation count and the sign of the rank-one term of the scale.
    fn sign(self) -> f64 {
        match self {
            Self::Absorbed => 1.0,
            Self::Released => -1.0,
        }
    }
}

/// The distribution `S ~ Wishart(dof, inverse_scale^{-1})`,
/// `mu | S ~ N(mean, (mean_strength * S)^{-1})`.
///
/// It is the conjugate prior of a Gaussian with unknown mean and precision,
/// so both the prior of a mixture component and its posterior given data
/// are of this form.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NormalWishart {
    mean: DVector<f64>,
    mean_strength: f64,
    dof: f64,
    inverse_scale_cholesky: DMatrix<f64>,
    log_predictive_normalizer: f64,
}

impl NormalWishart {
    /// Creates the distribution; `inverse_scale` must be positive definite.
    pub(crate) fn new(
        mean: DVector<f64>,
        mean_strength: f64,
        dof: f64,
        inverse_scale: DMatrix<f64>,
    ) -> Result<Self, DpgmmError> {
        const STEP: DpgmmError = DpgmmError::NumericalBreakdown {
            step: "factorizing a component's scatter matrix",
        };
        let inverse_scale_cholesky = inverse_scale.cholesky().ok_or(STEP)?.unpack();
        let log_predictive_normalizer =
            Self::log_predictive_normalizer(mean.len(), mean_strength, dof, &inverse_scale_cholesky);
        if !log_predictive_normalizer.is_finite() {
            return Err(STEP);
        }
        Ok(Self {
            mean,
            mean_strength,
            dof,
            inverse_scale_cholesky,
            log_predictive_normalizer,
        })
    }

    /// Turns the distribution, in place, into its posterior after one more
    /// observation `point` of the Gaussian it describes.
    ///
    /// One observation changes the inverse scale by a rank-one matrix, so
    /// its Cholesky factor is updated in `O(d^2)` rather than recomputed in
    /// `O(d^3)`; the Gibbs sampler does this twice per point and sweep.
    /// `scratch` must have the length of `point`.
    pub(crate) fn absorb_observation(&mut self, point: &[f64], scratch: &mut [f64]) -> Result<(), DpgmmError> {
        self.shift_by_observation(point, scratch, ObservationChange::Absorbed)
    }

    /// Turns the distribution, in place, into the posterior without `point`,
    /// which must be one of the observations it is conditioned on. The
    /// counterpart of [`Self::absorb_observation`], at the same cost.
    ///
    /// On an error the distribution is left in an unspecified state and must
    /// be derived afresh from the remaining observations.
    pub(crate) fn release_observation(&mut self, point: &[f64], scratch: &mut [f64]) -> Result<(), DpgmmError> {
        self.shift_by_observation(point, scratch, ObservationChange::Released)
    }

    /// With `kappa` the mean strength before the change and `kappa'` after
    /// it, the inverse scale changes by
    /// `+- (kappa / kappa') (x - mean)(x - mean)^T` and the mean becomes
    /// `(kappa mean +- x) / kappa'`.
    fn shift_by_observation(
        &mut self,
        point: &[f64],
        scratch: &mut [f64],
        change: ObservationChange,
    ) -> Result<(), DpgmmError> {
        const STEP: DpgmmError = DpgmmError::NumericalBreakdown {
            step: "updating a component's scatter factor by one point",
        };
        debug_assert_eq!(point.len(), self.mean.len(), "point of the wrong dimension");
        debug_assert_eq!(scratch.len(), point.len(), "scratch of the wrong length");
        let sign = change.sign();
        let strength_after = self.mean_strength + sign;
        let direction_scale = (self.mean_strength / strength_after).sqrt();
        for ((direction, coordinate), mean) in scratch.iter_mut().zip(point).zip(self.mean.iter()) {
            *direction = direction_scale * (coordinate - mean);
        }
        Self::update_factor_by_rank_one(&mut self.inverse_scale_cholesky, scratch, sign).ok_or(STEP)?;
        for (mean, coordinate) in self.mean.iter_mut().zip(point) {
            *mean = (self.mean_strength * *mean + sign * coordinate) / strength_after;
        }
        self.mean_strength = strength_after;
        self.dof += sign;
        self.log_predictive_normalizer = Self::log_predictive_normalizer(
            self.mean.len(),
            self.mean_strength,
            self.dof,
            &self.inverse_scale_cholesky,
        );
        if self.log_predictive_normalizer.is_finite() {
            Ok(())
        } else {
            Err(STEP)
        }
    }

    /// Replaces the lower Cholesky factor of `A` by that of
    /// `A + sign * v v^T`, column by column with one plane rotation each.
    /// `direction` holds `v` on entry and is used up. Returns `None` if the
    /// result is not positive definite, which only a downdate can cause.
    fn update_factor_by_rank_one(lower: &mut DMatrix<f64>, direction: &mut [f64], sign: f64) -> Option<()> {
        let dimension = direction.len();
        for column in 0..dimension {
            let diagonal = lower[(column, column)];
            let squared_diagonal = diagonal * diagonal + sign * direction[column] * direction[column];
            if squared_diagonal.is_nan() || squared_diagonal <= 0.0 {
                return None;
            }
            let rotated_diagonal = squared_diagonal.sqrt();
            let cosine = rotated_diagonal / diagonal;
            let sine = direction[column] / diagonal;
            lower[(column, column)] = rotated_diagonal;
            for row in column + 1..dimension {
                let entry = (lower[(row, column)] + sign * sine * direction[row]) / cosine;
                lower[(row, column)] = entry;
                direction[row] = cosine * direction[row] - sine * entry;
            }
        }
        Some(())
    }

    /// Log-density at `point` of a new observation from a Gaussian whose
    /// mean and precision follow this distribution: a multivariate Student
    /// t. `scratch` must have the length of `point`.
    pub(crate) fn log_predictive(&self, point: &[f64], scratch: &mut [f64]) -> f64 {
        let squared_distance = self.whitened_squared_distance(point, scratch);
        let shrink = self.mean_strength / (self.mean_strength + 1.0);
        self.log_predictive_normalizer - 0.5 * (self.dof + 1.0) * (shrink * squared_distance).ln_1p()
    }

    /// Draws a mean and a precision matrix.
    pub(crate) fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<GaussianDraw, DpgmmError> {
        let precision = Wishart::new(self.dof, &self.inverse_scale_cholesky).sample(rng)?;
        let standard_normal = DVector::from_fn(self.mean.len(), |_, _| StandardNormal.sample(rng));
        let mean = &self.mean + (&precision.inverse_factor * standard_normal) / self.mean_strength.sqrt();
        Ok(GaussianDraw { mean, precision })
    }

    /// `(x - mean)^T inverse_scale^{-1} (x - mean)` by forward substitution
    /// with the Cholesky factor, without allocating.
    fn whitened_squared_distance(&self, point: &[f64], scratch: &mut [f64]) -> f64 {
        let lower = &self.inverse_scale_cholesky;
        let mut squared_distance = 0.0;
        for row in 0..point.len() {
            let solved: f64 = (0..row).map(|column| lower[(row, column)] * scratch[column]).sum();
            scratch[row] = (point[row] - self.mean[row] - solved) / lower[(row, row)];
            squared_distance += scratch[row] * scratch[row];
        }
        squared_distance
    }

    /// Constant of the Student t predictive with `dof - d + 1` degrees of
    /// freedom.
    fn log_predictive_normalizer(dimension: usize, mean_strength: f64, dof: f64, lower: &DMatrix<f64>) -> f64 {
        // Dimensions are far below 2^53, where the conversion is exact.
        #[allow(clippy::cast_precision_loss)]
        let dimension = dimension as f64;
        let log_determinant_half: f64 = lower.diagonal().iter().map(|value| value.ln()).sum();
        libm::lgamma(0.5 * (dof + 1.0))
            - libm::lgamma(0.5 * (dof + 1.0 - dimension))
            - 0.5 * dimension * std::f64::consts::PI.ln()
            + 0.5 * dimension * (mean_strength / (mean_strength + 1.0)).ln()
            - log_determinant_half
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn distribution() -> NormalWishart {
        let mean = DVector::from_vec(vec![0.5, -1.0]);
        let inverse_scale = DMatrix::from_row_slice(2, 2, &[3.0, 0.6, 0.6, 2.0]);
        NormalWishart::new(mean, 1.5, 6.0, inverse_scale).expect("valid test distribution")
    }

    /// Density of `N(mean, precision^{-1})` at `point`.
    fn gaussian_density(point: &DVector<f64>, draw: &GaussianDraw) -> f64 {
        let residual = point - &draw.mean;
        let quadratic = (residual.transpose() * &draw.precision.matrix * &residual)[(0, 0)];
        (0.5 * draw.precision.log_determinant - 0.5 * quadratic).exp() / (2.0 * std::f64::consts::PI)
    }

    #[test]
    fn predictive_equals_the_gaussian_density_averaged_over_draws() -> Result<(), DpgmmError> {
        // Two independent code paths: the closed-form Student t and Monte
        // Carlo integration over sampled means and precisions.
        let distribution = distribution();
        let point = DVector::from_vec(vec![1.0, -0.2]);
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let n_draws = 40_000;
        let mut sum = 0.0;
        for _ in 0..n_draws {
            sum += gaussian_density(&point, &distribution.sample(&mut rng)?);
        }
        let monte_carlo = sum / f64::from(n_draws);
        let closed_form = distribution.log_predictive(point.as_slice(), &mut [0.0; 2]).exp();
        assert!(
            (monte_carlo / closed_form - 1.0).abs() < 0.03,
            "{monte_carlo} vs {closed_form}"
        );
        Ok(())
    }

    #[test]
    fn one_dimensional_predictive_integrates_to_one() -> Result<(), DpgmmError> {
        let distribution = NormalWishart::new(DVector::zeros(1), 0.7, 3.0, DMatrix::from_element(1, 1, 2.0))?;
        let step = 0.01;
        let integral: f64 = (-20_000..20_000)
            .map(|i| distribution.log_predictive(&[f64::from(i) * step], &mut [0.0]).exp())
            .sum();
        assert!((integral * step - 1.0).abs() < 1e-3);
        Ok(())
    }

    /// The posterior of `distribution()` given `points`, computed from
    /// scratch with the textbook formula.
    fn posterior_from_scratch(points: &[[f64; 2]]) -> NormalWishart {
        let (prior_mean, strength, dof) = (DVector::from_vec(vec![0.5, -1.0]), 1.5, 6.0);
        let mut inverse_scale = DMatrix::from_row_slice(2, 2, &[3.0, 0.6, 0.6, 2.0]);
        let mut sum = DVector::zeros(2);
        for point in points {
            let point = DVector::from_row_slice(point);
            inverse_scale += &point * point.transpose();
            sum += point;
        }
        // Counts here are tiny; the conversion is exact.
        #[allow(clippy::cast_precision_loss)]
        let strength_after = strength + points.len() as f64;
        let mean = (strength * &prior_mean + sum) / strength_after;
        inverse_scale += strength * &prior_mean * prior_mean.transpose();
        inverse_scale -= strength_after * &mean * mean.transpose();
        #[allow(clippy::cast_precision_loss)]
        let dof_after = dof + points.len() as f64;
        NormalWishart::new(mean, strength_after, dof_after, inverse_scale).expect("valid posterior")
    }

    fn assert_same_distribution(actual: &NormalWishart, expected: &NormalWishart) {
        let factor_gap = (&actual.inverse_scale_cholesky - &expected.inverse_scale_cholesky)
            .abs()
            .max();
        let mean_gap = (&actual.mean - &expected.mean).abs().max();
        assert!(factor_gap < 1e-10, "Cholesky factors differ by {factor_gap}");
        assert!(mean_gap < 1e-12, "means differ by {mean_gap}");
        assert!((actual.mean_strength - expected.mean_strength).abs() < 1e-12);
        assert!((actual.dof - expected.dof).abs() < 1e-12);
        assert!((actual.log_predictive_normalizer - expected.log_predictive_normalizer).abs() < 1e-10);
    }

    #[test]
    fn absorbing_observations_one_by_one_equals_the_posterior_from_scratch() -> Result<(), DpgmmError> {
        let points = [[0.5, 1.0], [-1.0, 0.25], [2.0, -0.5], [0.1, 0.7]];
        let mut updated = distribution();
        for point in &points {
            updated.absorb_observation(point, &mut [0.0; 2])?;
        }
        assert_same_distribution(&updated, &posterior_from_scratch(&points));
        Ok(())
    }

    #[test]
    fn releasing_an_observation_equals_the_posterior_without_it() -> Result<(), DpgmmError> {
        let points = [[0.5, 1.0], [-1.0, 0.25], [2.0, -0.5], [0.1, 0.7]];
        let mut updated = posterior_from_scratch(&points);
        updated.release_observation(&points[1], &mut [0.0; 2])?;
        assert_same_distribution(&updated, &posterior_from_scratch(&[points[0], points[2], points[3]]));
        Ok(())
    }

    #[test]
    fn many_exchanges_of_observations_do_not_drift() -> Result<(), DpgmmError> {
        // A Gibbs sweep removes and re-adds every point; the factor must
        // still be the exact one after many such exchanges.
        let points = [[0.5, 1.0], [-1.0, 0.25], [2.0, -0.5], [0.1, 0.7]];
        let mut updated = posterior_from_scratch(&points);
        for _ in 0..10_000 {
            for point in &points {
                updated.release_observation(point, &mut [0.0; 2])?;
                updated.absorb_observation(point, &mut [0.0; 2])?;
            }
        }
        assert_same_distribution(&updated, &posterior_from_scratch(&points));
        Ok(())
    }

    #[test]
    fn releasing_what_was_never_observed_is_reported() {
        // Far from every observation, the downdate leaves no positive
        // definite scale.
        let mut posterior = posterior_from_scratch(&[[0.5, 1.0], [-1.0, 0.25]]);
        assert!(posterior.release_observation(&[50.0, -80.0], &mut [0.0; 2]).is_err());
    }

    #[test]
    fn indefinite_inverse_scale_is_reported() {
        let indefinite = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(NormalWishart::new(DVector::zeros(2), 1.0, 3.0, indefinite).is_err());
    }
}
