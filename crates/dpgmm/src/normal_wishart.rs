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

    #[test]
    fn indefinite_inverse_scale_is_reported() {
        let indefinite = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(NormalWishart::new(DVector::zeros(2), 1.0, 3.0, indefinite).is_err());
    }
}
