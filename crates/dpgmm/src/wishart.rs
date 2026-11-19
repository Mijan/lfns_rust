//! Sampling from the Wishart distribution.

use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::{ChiSquared, Distribution, StandardNormal};

use crate::dpgmm_error::DpgmmError;

/// One draw `S` from a Wishart distribution, with the by-products of the
/// Bartlett decomposition that produced it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WishartDraw {
    /// The drawn matrix `S`.
    pub(crate) matrix: DMatrix<f64>,
    /// `ln det S`.
    pub(crate) log_determinant: f64,
    /// A square matrix `G` with `G G^T = S^{-1}`.
    pub(crate) inverse_factor: DMatrix<f64>,
}

/// The Wishart distribution with `dof` degrees of freedom and scale matrix
/// `(L L^T)^{-1}`, parameterized by the Cholesky factor `L` of the inverse
/// scale. Its mean is `dof * (L L^T)^{-1}`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Wishart<'a> {
    dof: f64,
    inverse_scale_cholesky: &'a DMatrix<f64>,
}

impl<'a> Wishart<'a> {
    pub(crate) fn new(dof: f64, inverse_scale_cholesky: &'a DMatrix<f64>) -> Self {
        Self {
            dof,
            inverse_scale_cholesky,
        }
    }

    /// Draws `S = F F^T` with `F = L^{-T} B`, where `B` is the lower
    /// triangular Bartlett factor.
    pub(crate) fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<WishartDraw, DpgmmError> {
        const STEP: DpgmmError = DpgmmError::NumericalBreakdown {
            step: "drawing from a Wishart distribution",
        };
        let bartlett = self.sample_bartlett_factor(rng).ok_or(STEP)?;
        let lower = self.inverse_scale_cholesky;
        let factor = lower.tr_solve_lower_triangular(&bartlett).ok_or(STEP)?;
        let identity = DMatrix::identity(lower.nrows(), lower.nrows());
        let inverse_factor = lower * bartlett.tr_solve_lower_triangular(&identity).ok_or(STEP)?;
        let log_determinant = 2.0 * (Self::log_diagonal_sum(&bartlett) - Self::log_diagonal_sum(lower));
        Ok(WishartDraw {
            matrix: &factor * factor.transpose(),
            log_determinant,
            inverse_factor,
        })
    }

    /// Lower triangular `B` with `B_ii^2 ~ chi^2(dof - i)` and standard
    /// normal entries below the diagonal; `None` if `dof <= dimension - 1`.
    fn sample_bartlett_factor<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<DMatrix<f64>> {
        let dimension = self.inverse_scale_cholesky.nrows();
        let mut bartlett = DMatrix::zeros(dimension, dimension);
        let mut diagonal_dof = self.dof;
        for row in 0..dimension {
            for column in 0..row {
                bartlett[(row, column)] = StandardNormal.sample(rng);
            }
            bartlett[(row, row)] = ChiSquared::new(diagonal_dof).ok()?.sample(rng).sqrt();
            diagonal_dof -= 1.0;
        }
        Some(bartlett)
    }

    fn log_diagonal_sum(triangular: &DMatrix<f64>) -> f64 {
        triangular.diagonal().iter().map(|value| value.ln()).sum()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn sample_mean_matches_dof_times_scale() -> Result<(), DpgmmError> {
        // Inverse scale diag(4, 1): scale diag(1/4, 1), mean dof * scale.
        let lower = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let wishart = Wishart::new(5.0, &lower);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let n_draws = 20_000;
        let mut sum = DMatrix::zeros(2, 2);
        for _ in 0..n_draws {
            sum += wishart.sample(&mut rng)?.matrix;
        }
        let mean = sum / f64::from(n_draws);
        assert!((mean[(0, 0)] - 1.25).abs() < 0.05, "{mean}");
        assert!((mean[(1, 1)] - 5.0).abs() < 0.15, "{mean}");
        assert!(mean[(0, 1)].abs() < 0.05, "{mean}");
        Ok(())
    }

    #[test]
    fn by_products_are_consistent_with_the_drawn_matrix() -> Result<(), DpgmmError> {
        let lower = DMatrix::from_row_slice(2, 2, &[1.5, 0.0, 0.4, 0.8]);
        let draw = Wishart::new(4.5, &lower).sample(&mut ChaCha8Rng::seed_from_u64(1))?;
        let inverse = &draw.inverse_factor * draw.inverse_factor.transpose();
        assert!((&draw.matrix * inverse - DMatrix::identity(2, 2)).norm() < 1e-10);
        assert!((draw.matrix.determinant().ln() - draw.log_determinant).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn too_few_degrees_of_freedom_are_reported() {
        let lower = DMatrix::identity(3, 3);
        let result = Wishart::new(1.5, &lower).sample(&mut ChaCha8Rng::seed_from_u64(1));
        assert!(matches!(result, Err(DpgmmError::NumericalBreakdown { .. })));
    }
}
