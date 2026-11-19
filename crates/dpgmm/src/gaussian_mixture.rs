//! A finite mixture of multivariate Gaussians.

use nalgebra::{DMatrix, DVector};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal};

use crate::dpgmm_error::DpgmmError;

/// One weighted Gaussian of a [`GaussianMixture`].
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianComponent {
    weight: f64,
    mean: Vec<f64>,
    covariance: Vec<f64>,
    /// Lower Cholesky factor of the covariance, row after row.
    cholesky: Vec<f64>,
    /// Inverse of that factor, row after row.
    inverse_cholesky: Vec<f64>,
    log_normalizer: f64,
}

impl GaussianComponent {
    /// Creates a component from its weight, mean, and covariance (given row
    /// after row).
    ///
    /// # Errors
    ///
    /// Returns [`DpgmmError::InvalidMixture`] if the weight is not in
    /// `(0, 1]`, the covariance does not match the mean in size, or the
    /// covariance is not positive definite.
    pub fn new(weight: f64, mean: Vec<f64>, covariance: &[f64]) -> Result<Self, DpgmmError> {
        let dimension = mean.len();
        if covariance.len() != dimension * dimension {
            let reason = format!(
                "a covariance of {} values does not match a mean of {dimension} dimensions; \
                 pass the {dimension}x{dimension} matrix row after row",
                covariance.len()
            );
            return Err(DpgmmError::InvalidMixture { reason });
        }
        let covariance = DMatrix::from_row_slice(dimension, dimension, covariance);
        Self::from_moments(weight, &DVector::from_vec(mean), &covariance)
    }

    /// Creates a component from linear-algebra types.
    pub(crate) fn from_moments(
        weight: f64,
        mean: &DVector<f64>,
        covariance: &DMatrix<f64>,
    ) -> Result<Self, DpgmmError> {
        if !(weight > 0.0 && weight <= 1.0) {
            let reason = format!("component weight {weight} is not in (0, 1]; weights are probabilities");
            return Err(DpgmmError::InvalidMixture { reason });
        }
        let not_positive_definite = || DpgmmError::InvalidMixture {
            reason: "a component covariance is not positive definite; pass a symmetric matrix with \
                     positive eigenvalues"
                .to_owned(),
        };
        let cholesky = covariance
            .clone()
            .cholesky()
            .ok_or_else(not_positive_definite)?
            .unpack();
        let identity = DMatrix::identity(mean.len(), mean.len());
        let inverse_cholesky = cholesky
            .solve_lower_triangular(&identity)
            .ok_or_else(not_positive_definite)?;
        let log_determinant_half: f64 = cholesky.diagonal().iter().map(|value| value.ln()).sum();
        // Dimensions are far below 2^53, where the conversion is exact.
        #[allow(clippy::cast_precision_loss)]
        let log_normalizer = -0.5 * mean.len() as f64 * std::f64::consts::TAU.ln() - log_determinant_half;
        Ok(Self {
            weight,
            mean: mean.as_slice().to_vec(),
            covariance: covariance.transpose().as_slice().to_vec(),
            cholesky: cholesky.transpose().as_slice().to_vec(),
            inverse_cholesky: inverse_cholesky.transpose().as_slice().to_vec(),
            log_normalizer,
        })
    }

    /// Mixing weight.
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Mean vector.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Covariance matrix, row after row.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Log-density of this Gaussian (without its weight) at `point`.
    #[must_use]
    pub fn log_density(&self, point: &[f64]) -> f64 {
        let dimension = self.mean.len();
        let mut squared_distance = 0.0;
        for row in 0..dimension {
            let factor_row = &self.inverse_cholesky[row * dimension..=row * dimension + row];
            let residuals = point.iter().zip(&self.mean).map(|(value, mean)| value - mean);
            let whitened: f64 = factor_row
                .iter()
                .zip(residuals)
                .map(|(factor, residual)| factor * residual)
                .sum();
            squared_distance += whitened * whitened;
        }
        self.log_normalizer - 0.5 * squared_distance
    }

    /// Writes one draw into `point`, in place and without allocating.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, point: &mut [f64]) {
        let dimension = self.mean.len();
        for value in point.iter_mut() {
            *value = StandardNormal.sample(rng);
        }
        // Row `row` of `mean + L z` needs z_0..=z_row only, so going from the
        // last row to the first overwrites no value that is still needed.
        for row in (0..dimension).rev() {
            let factor_row = &self.cholesky[row * dimension..=row * dimension + row];
            let correlated: f64 = factor_row
                .iter()
                .zip(point.iter())
                .map(|(factor, normal)| factor * normal)
                .sum();
            point[row] = self.mean[row] + correlated;
        }
    }
}

/// A probability density `sum_k w_k N(mean_k, covariance_k)`.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianMixture {
    components: Vec<GaussianComponent>,
}

impl GaussianMixture {
    /// Largest deviation of the weight sum from one that is still accepted.
    const WEIGHT_SUM_TOLERANCE: f64 = 1e-9;

    /// Creates a mixture from its components.
    ///
    /// # Errors
    ///
    /// Returns [`DpgmmError::InvalidMixture`] if there is no component, the
    /// components differ in dimension, or the weights do not sum to one.
    pub fn new(components: Vec<GaussianComponent>) -> Result<Self, DpgmmError> {
        let Some(first) = components.first() else {
            let reason = "a mixture needs at least one component".to_owned();
            return Err(DpgmmError::InvalidMixture { reason });
        };
        if components
            .iter()
            .any(|component| component.mean.len() != first.mean.len())
        {
            let reason = "the components differ in dimension; give every component the same dimension".to_owned();
            return Err(DpgmmError::InvalidMixture { reason });
        }
        let weight_sum: f64 = components.iter().map(GaussianComponent::weight).sum();
        if (weight_sum - 1.0).abs() > Self::WEIGHT_SUM_TOLERANCE {
            let reason = format!("the weights sum to {weight_sum}; divide each weight by that sum");
            return Err(DpgmmError::InvalidMixture { reason });
        }
        Ok(Self { components })
    }

    /// The components of the mixture.
    #[must_use]
    pub fn components(&self) -> &[GaussianComponent] {
        &self.components
    }

    /// Number of coordinates of a point.
    #[must_use]
    pub fn n_dimensions(&self) -> usize {
        self.components.first().map_or(0, |component| component.mean.len())
    }

    /// Log-density of the mixture at `point`.
    ///
    /// # Panics
    ///
    /// Panics if `point` does not have [`Self::n_dimensions`] coordinates.
    #[must_use]
    pub fn log_density(&self, point: &[f64]) -> f64 {
        assert_eq!(
            point.len(),
            self.n_dimensions(),
            "point and mixture differ in dimension"
        );
        let terms = self
            .components
            .iter()
            .map(|component| component.weight.ln() + component.log_density(point));
        terms.fold(f64::NEG_INFINITY, |sum, term| {
            let (larger, smaller) = if sum >= term { (sum, term) } else { (term, sum) };
            if larger == f64::NEG_INFINITY {
                larger
            } else {
                larger + (smaller - larger).exp().ln_1p()
            }
        })
    }

    /// Writes one draw from the mixture into `point`.
    ///
    /// # Panics
    ///
    /// Panics if `point` does not have [`Self::n_dimensions`] coordinates.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R, point: &mut [f64]) {
        assert_eq!(
            point.len(),
            self.n_dimensions(),
            "point and mixture differ in dimension"
        );
        self.choose_component(rng).sample(rng, point);
    }

    fn choose_component<R: Rng + ?Sized>(&self, rng: &mut R) -> &GaussianComponent {
        let mut remaining = rng.random::<f64>();
        for component in &self.components {
            if remaining < component.weight {
                return component;
            }
            remaining -= component.weight;
        }
        // Rounding can leave a sliver beyond the last weight.
        &self.components[self.components.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    const TOLERANCE: f64 = 1e-12;

    fn correlated_component(weight: f64) -> Result<GaussianComponent, DpgmmError> {
        GaussianComponent::new(weight, vec![1.0, -2.0], &[2.0, 0.6, 0.6, 0.5])
    }

    #[test]
    fn density_matches_the_bivariate_normal_formula() -> Result<(), DpgmmError> {
        let mixture = GaussianMixture::new(vec![correlated_component(1.0)?])?;
        // det = 0.64; inverse = [[0.5, -0.6], [-0.6, 2.0]] / 0.64; residual (0.5, 0.5).
        let quadratic = (0.5 * 0.25 - 2.0 * 0.6 * 0.25 + 2.0 * 0.25) / 0.64;
        let expected = -std::f64::consts::TAU.ln() - 0.5 * 0.64_f64.ln() - 0.5 * quadratic;
        assert!((mixture.log_density(&[1.5, -1.5]) - expected).abs() < TOLERANCE);
        Ok(())
    }

    #[test]
    fn samples_reproduce_mean_and_covariance() -> Result<(), DpgmmError> {
        let mixture = GaussianMixture::new(vec![correlated_component(1.0)?])?;
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let n_draws = 50_000;
        let (mut sum, mut cross) = ([0.0; 2], 0.0);
        let mut point = [0.0; 2];
        for _ in 0..n_draws {
            mixture.sample(&mut rng, &mut point);
            sum[0] += point[0];
            sum[1] += point[1];
            cross += (point[0] - 1.0) * (point[1] + 2.0);
        }
        let n = f64::from(n_draws);
        assert!((sum[0] / n - 1.0).abs() < 0.03);
        assert!((sum[1] / n + 2.0).abs() < 0.02);
        assert!((cross / n - 0.6).abs() < 0.03);
        Ok(())
    }

    #[test]
    fn mixture_density_is_the_weighted_sum_of_its_components() -> Result<(), DpgmmError> {
        let narrow = GaussianComponent::new(0.25, vec![0.0, 0.0], &[0.1, 0.0, 0.0, 0.1])?;
        let mixture = GaussianMixture::new(vec![correlated_component(0.75)?, narrow.clone()])?;
        let point = [0.2, -0.4];
        let expected =
            0.75 * correlated_component(0.75)?.log_density(&point).exp() + 0.25 * narrow.log_density(&point).exp();
        assert!((mixture.log_density(&point).exp() - expected).abs() < TOLERANCE);
        Ok(())
    }

    #[test]
    fn weights_must_sum_to_one() -> Result<(), DpgmmError> {
        let result = GaussianMixture::new(vec![correlated_component(0.4)?, correlated_component(0.4)?]);
        assert!(matches!(result, Err(DpgmmError::InvalidMixture { .. })));
        Ok(())
    }

    #[test]
    fn covariance_must_be_positive_definite() {
        let result = GaussianComponent::new(1.0, vec![0.0, 0.0], &[1.0, 2.0, 2.0, 1.0]);
        assert!(matches!(result, Err(DpgmmError::InvalidMixture { .. })));
    }
}
