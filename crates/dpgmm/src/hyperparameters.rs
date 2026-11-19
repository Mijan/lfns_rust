//! Hyperparameters of the mixture and their Gibbs updates.
//!
//! The model follows Görür and Rasmussen (2010), conjugate case, for data
//! whitened to zero mean and identity covariance:
//!
//! - component precision `S_k ~ Wishart(beta, (beta W)^{-1})`,
//!   mean `mu_k | S_k ~ N(xi, (rho S_k)^{-1})`;
//! - `xi ~ N(0, I)`, `W ~ Wishart(d, I / d)`, `rho ~ Gamma(1/4, rate 1/2)`;
//! - `1 / (beta - d + 1) ~ Gamma(1/2, rate d/2)`;
//! - concentration `1 / alpha ~ Gamma(1/2, rate 1/2)`.
//!
//! `xi`, `W`, and `rho` have conjugate conditionals. `alpha` and `beta` do
//! not; they are updated by slice sampling on a log scale.

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Gamma, StandardNormal};

use crate::dpgmm_error::DpgmmError;
use crate::normal_wishart::{GaussianDraw, NormalWishart};
use crate::slice_sampler::SliceSampler;
use crate::wishart::Wishart;

/// The concentration of the Dirichlet process and the parameters of the
/// Normal-Wishart prior shared by all components.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Hyperparameters {
    concentration: f64,
    dof: f64,
    mean_strength: f64,
    mean: DVector<f64>,
    scale: DMatrix<f64>,
}

impl Hyperparameters {
    /// Shape of the Gamma prior of `rho`.
    const MEAN_STRENGTH_PRIOR_SHAPE: f64 = 0.25;
    /// Rate of the Gamma prior of `rho`.
    const MEAN_STRENGTH_PRIOR_RATE: f64 = 0.5;
    /// `beta - d + 1` of the starting point.
    const INITIAL_EXCESS_DOF: f64 = 2.0;

    /// A central starting point for whitened data: unit concentration and
    /// mean strength, a prior centred at the origin with identity scale.
    pub(crate) fn initial(dimension: usize) -> Self {
        let mut hyperparameters = Self {
            concentration: 1.0,
            dof: 0.0,
            mean_strength: 1.0,
            mean: DVector::zeros(dimension),
            scale: DMatrix::identity(dimension, dimension),
        };
        hyperparameters.dof = hyperparameters.dimension() - 1.0 + Self::INITIAL_EXCESS_DOF;
        hyperparameters
    }

    /// Concentration `alpha` of the Dirichlet process.
    pub(crate) fn concentration(&self) -> f64 {
        self.concentration
    }

    /// Degrees of freedom `beta` of the precision prior.
    pub(crate) fn dof(&self) -> f64 {
        self.dof
    }

    /// Strength `rho` with which a component mean is tied to `xi`.
    pub(crate) fn mean_strength(&self) -> f64 {
        self.mean_strength
    }

    /// Prior mean `xi` of the component means.
    pub(crate) fn mean(&self) -> &DVector<f64> {
        &self.mean
    }

    /// Scale `W`; the prior mean of a component precision is `W^{-1}`.
    pub(crate) fn scale(&self) -> &DMatrix<f64> {
        &self.scale
    }

    /// The Normal-Wishart prior of a component's mean and precision.
    pub(crate) fn derive_prior(&self) -> Result<NormalWishart, DpgmmError> {
        NormalWishart::new(self.mean.clone(), self.mean_strength, self.dof, self.dof * &self.scale)
    }

    /// Redraws `alpha` given the number of observations and of components.
    pub(crate) fn resample_concentration<R: Rng + ?Sized>(
        &mut self,
        n_observations: f64,
        n_components: f64,
        rng: &mut R,
    ) {
        let log_density = |log_alpha: f64| {
            let alpha = log_alpha.exp();
            let value = libm::lgamma(alpha) - libm::lgamma(n_observations + alpha) + (n_components - 0.5) * log_alpha
                - 0.5 / alpha;
            if value.is_finite() { value } else { f64::NEG_INFINITY }
        };
        self.concentration = SliceSampler::new()
            .sample(log_density, self.concentration.ln(), rng)
            .exp();
    }

    /// Redraws `rho`, `W`, `xi`, and `beta` given the components' means and
    /// precisions.
    pub(crate) fn resample_component_prior<R: Rng + ?Sized>(
        &mut self,
        components: &[GaussianDraw],
        rng: &mut R,
    ) -> Result<(), DpgmmError> {
        self.resample_mean_strength(components, rng)?;
        self.resample_scale(components, rng)?;
        self.resample_mean(components, rng)?;
        self.resample_dof(components, rng);
        Ok(())
    }

    /// `rho | ... ~ Gamma(1/4 + K d / 2, rate 1/2 + sum_k q_k / 2)` with
    /// `q_k = (mu_k - xi)^T S_k (mu_k - xi)`.
    fn resample_mean_strength<R: Rng + ?Sized>(
        &mut self,
        components: &[GaussianDraw],
        rng: &mut R,
    ) -> Result<(), DpgmmError> {
        let squared_distances = components.iter().map(|component| {
            let offset = &component.mean - &self.mean;
            (offset.transpose() * &component.precision.matrix * offset)[(0, 0)]
        });
        let shape = Self::MEAN_STRENGTH_PRIOR_SHAPE + 0.5 * Self::count(components) * self.dimension();
        let rate = Self::MEAN_STRENGTH_PRIOR_RATE + 0.5 * squared_distances.sum::<f64>();
        let gamma = Gamma::new(shape, rate.recip()).map_err(|_| DpgmmError::NumericalBreakdown {
            step: "redrawing the mean strength",
        })?;
        self.mean_strength = gamma.sample(rng);
        Ok(())
    }

    /// `W | ... ~ Wishart(K beta + d, (d I + beta sum_k S_k)^{-1})`.
    fn resample_scale<R: Rng + ?Sized>(&mut self, components: &[GaussianDraw], rng: &mut R) -> Result<(), DpgmmError> {
        const STEP: DpgmmError = DpgmmError::NumericalBreakdown {
            step: "redrawing the prior scale matrix",
        };
        let dimension = self.mean.len();
        let inverse_scale = DMatrix::identity(dimension, dimension) * self.dimension()
            + self.dof * Self::sum_of_precisions(components, dimension);
        let cholesky = inverse_scale.cholesky().ok_or(STEP)?.unpack();
        let dof = Self::count(components) * self.dof + self.dimension();
        self.scale = Wishart::new(dof, &cholesky).sample(rng)?.matrix;
        Ok(())
    }

    /// `xi | ... ~ N(P^{-1} rho sum_k S_k mu_k, P^{-1})` with
    /// `P = I + rho sum_k S_k`.
    fn resample_mean<R: Rng + ?Sized>(&mut self, components: &[GaussianDraw], rng: &mut R) -> Result<(), DpgmmError> {
        const STEP: DpgmmError = DpgmmError::NumericalBreakdown {
            step: "redrawing the prior mean",
        };
        let dimension = self.mean.len();
        let precision = DMatrix::identity(dimension, dimension)
            + self.mean_strength * Self::sum_of_precisions(components, dimension);
        let weighted_means: DVector<f64> = components.iter().map(|c| &c.precision.matrix * &c.mean).sum();
        let cholesky = precision.cholesky().ok_or(STEP)?;
        let conditional_mean = cholesky.solve(&(self.mean_strength * weighted_means));
        let standard_normal = DVector::from_fn(dimension, |_, _| StandardNormal.sample(rng));
        let noise = cholesky.l().tr_solve_lower_triangular(&standard_normal).ok_or(STEP)?;
        self.mean = conditional_mean + noise;
        Ok(())
    }

    /// Redraws `beta` by slice sampling `ln(beta - d + 1)`.
    fn resample_dof<R: Rng + ?Sized>(&mut self, components: &[GaussianDraw], rng: &mut R) {
        let dimension = self.dimension();
        let n_components = Self::count(components);
        let log_scale_determinant = self.scale.determinant().ln();
        let fit: f64 = components
            .iter()
            .map(|c| {
                log_scale_determinant + c.precision.log_determinant
                    - self.scale.component_mul(&c.precision.matrix).sum()
            })
            .sum();
        let log_density = |log_excess: f64| {
            let excess = log_excess.exp();
            let dof = excess + dimension - 1.0;
            let log_multivariate_gamma: f64 = (0..self.mean.len())
                .map(|i| libm::lgamma(0.5 * (dof - Self::as_float(i))))
                .sum();
            let value = -0.5 * dimension / excess - 0.5 * log_excess
                + 0.5 * n_components * dof * dimension * (0.5 * dof).ln()
                + 0.5 * dof * fit
                - n_components * log_multivariate_gamma;
            if value.is_finite() { value } else { f64::NEG_INFINITY }
        };
        let current = (self.dof - dimension + 1.0).ln();
        self.dof = SliceSampler::new().sample(log_density, current, rng).exp() + dimension - 1.0;
    }

    fn sum_of_precisions(components: &[GaussianDraw], dimension: usize) -> DMatrix<f64> {
        let zero = DMatrix::zeros(dimension, dimension);
        components
            .iter()
            .fold(zero, |sum, component| sum + &component.precision.matrix)
    }

    fn dimension(&self) -> f64 {
        Self::as_float(self.mean.len())
    }

    fn count(components: &[GaussianDraw]) -> f64 {
        Self::as_float(components.len())
    }

    /// Converts a small count; counts here are far below 2^53, where the
    /// conversion is exact.
    #[allow(clippy::cast_precision_loss)]
    fn as_float(count: usize) -> f64 {
        count as f64
    }
}
