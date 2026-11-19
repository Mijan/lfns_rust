//! Affine map to zero mean and identity covariance.
//!
//! The mixture is fitted to whitened data. Its data-dependent priors are
//! then centred at the origin with unit scale, every matrix the sampler
//! factorizes is well conditioned, and the fit is equivariant under affine
//! transformations of the observations.

use nalgebra::{DMatrix, DVector};

use crate::dpgmm_error::DpgmmError;
use crate::observations::Observations;

/// The map `z = L^{-1} (x - m)` with `m` the sample mean and `L L^T` the
/// sample covariance.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Whitening {
    mean: DVector<f64>,
    scale: DMatrix<f64>,
}

impl Whitening {
    /// Derives the whitening map of `observations`.
    pub(crate) fn of(observations: &Observations) -> Result<Self, DpgmmError> {
        let points = observations.points();
        let mean = points.row_mean().transpose();
        let centered = Self::center(points, &mean);
        // The count of observations is far below 2^53, where the conversion is exact.
        #[allow(clippy::cast_precision_loss)]
        let degrees_of_freedom = (points.nrows() - 1) as f64;
        let covariance = (centered.transpose() * &centered) / degrees_of_freedom;
        let cholesky = covariance.cholesky().ok_or(DpgmmError::DegenerateObservations)?;
        Ok(Self {
            mean,
            scale: cholesky.unpack(),
        })
    }

    /// Whitens the observations; one whitened point per **column**.
    pub(crate) fn whiten(&self, observations: &Observations) -> Result<DMatrix<f64>, DpgmmError> {
        let centered = Self::center(observations.points(), &self.mean);
        self.scale
            .solve_lower_triangular(&centered.transpose())
            .ok_or(DpgmmError::NumericalBreakdown {
                step: "whitening the observations",
            })
    }

    /// Maps a mean of whitened space back to the space of the observations.
    pub(crate) fn restore_mean(&self, whitened_mean: &DVector<f64>) -> DVector<f64> {
        &self.mean + &self.scale * whitened_mean
    }

    /// Maps a covariance of whitened space back to the space of the
    /// observations.
    pub(crate) fn restore_covariance(&self, whitened_covariance: &DMatrix<f64>) -> DMatrix<f64> {
        &self.scale * whitened_covariance * self.scale.transpose()
    }

    fn center(points: &DMatrix<f64>, mean: &DVector<f64>) -> DMatrix<f64> {
        let mut centered = points.clone();
        for mut row in centered.row_iter_mut() {
            row -= mean.transpose();
        }
        centered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    fn skewed_observations() -> Observations {
        let values = [0.0, 0.0, 4.0, 1.0, 1.0, 3.0, 5.0, 5.0, 2.0, -1.0, 7.0, 2.5];
        Observations::from_rows(&values, 2).expect("valid test observations")
    }

    #[test]
    fn whitened_points_have_zero_mean_and_identity_covariance() -> Result<(), DpgmmError> {
        let observations = skewed_observations();
        let whitened = Whitening::of(&observations)?.whiten(&observations)?;
        let n = 6.0;
        assert!(whitened.column_mean().norm() < TOLERANCE);
        let covariance = (&whitened * whitened.transpose()) / (n - 1.0);
        assert!((covariance - DMatrix::identity(2, 2)).norm() < TOLERANCE);
        Ok(())
    }

    #[test]
    fn restoring_inverts_whitening_for_means_and_covariances() -> Result<(), DpgmmError> {
        let observations = skewed_observations();
        let whitening = Whitening::of(&observations)?;
        let restored_mean = whitening.restore_mean(&DVector::zeros(2));
        assert!((restored_mean - observations.points().row_mean().transpose()).norm() < TOLERANCE);
        // The identity in whitened space is the sample covariance in data space.
        let restored = whitening.restore_covariance(&DMatrix::identity(2, 2));
        assert!((restored[(0, 0)] - 6.966_666_666_666_667).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn points_on_a_line_are_degenerate() {
        let observations = Observations::from_rows(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 2);
        let whitening = Whitening::of(&observations.expect("valid test observations"));
        assert_eq!(whitening, Err(DpgmmError::DegenerateObservations));
    }
}
