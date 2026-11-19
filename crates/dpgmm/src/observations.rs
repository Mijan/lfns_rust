//! The data a mixture is fitted to.

use nalgebra::DMatrix;

use crate::dpgmm_error::DpgmmError;

/// A set of finite points of equal dimension, enough of them to estimate a
/// covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct Observations {
    /// One point per row.
    points: DMatrix<f64>,
}

impl Observations {
    /// Builds observations from a flat buffer holding one point after the
    /// other.
    ///
    /// # Errors
    ///
    /// Returns a [`DpgmmError`] if `n_dimensions` is zero, the buffer is not
    /// a whole number of points, a value is not finite, or there are fewer
    /// than `n_dimensions + 1` points.
    pub fn from_rows(values: &[f64], n_dimensions: usize) -> Result<Self, DpgmmError> {
        if n_dimensions == 0 {
            return Err(DpgmmError::NoDimensions);
        }
        if values.len() % n_dimensions != 0 {
            return Err(DpgmmError::RaggedObservations {
                n_values: values.len(),
                n_dimensions,
            });
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(DpgmmError::NonFiniteObservation { index });
        }
        let n_observations = values.len() / n_dimensions;
        let n_required = n_dimensions + 1;
        if n_observations < n_required {
            return Err(DpgmmError::TooFewObservations {
                n_observations,
                n_dimensions,
                n_required,
            });
        }
        Ok(Self {
            points: DMatrix::from_row_slice(n_observations, n_dimensions, values),
        })
    }

    /// Number of points.
    #[must_use]
    pub fn n_observations(&self) -> usize {
        self.points.nrows()
    }

    /// Number of coordinates per point.
    #[must_use]
    pub fn n_dimensions(&self) -> usize {
        self.points.ncols()
    }

    /// The points, one per row.
    pub(crate) fn points(&self) -> &DMatrix<f64> {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_buffer_that_is_not_a_whole_number_of_points() {
        let result = Observations::from_rows(&[0.0; 7], 2);
        assert!(matches!(result, Err(DpgmmError::RaggedObservations { .. })));
    }

    #[test]
    fn rejects_non_finite_values() {
        let result = Observations::from_rows(&[0.0, f64::NAN, 1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(result, Err(DpgmmError::NonFiniteObservation { index: 1 }));
    }

    #[test]
    fn rejects_too_few_points_for_a_covariance() {
        let result = Observations::from_rows(&[0.0; 4], 2);
        assert!(matches!(
            result,
            Err(DpgmmError::TooFewObservations { n_required: 3, .. })
        ));
    }
}
