//! Python bindings of the `dpgmm` crate, for notebooks and analysis scripts.
//!
//! The module exposes one function, `fit`, and the fitted `GaussianMixture`.
//! Arrays cross the boundary as `NumPy` arrays with one point per row.

// PyO3 hands extracted arguments to a function by value.
#![allow(clippy::needless_pass_by_value)]

use ::dpgmm::{Dpgmm, DpgmmConfig, GaussianComponent, GaussianMixture, Observations};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A fitted mixture of Gaussians.
#[pyclass(name = "GaussianMixture", frozen)]
struct PyGaussianMixture {
    mixture: GaussianMixture,
}

#[pymethods]
impl PyGaussianMixture {
    /// Number of mixture components.
    #[getter]
    fn n_components(&self) -> usize {
        self.mixture.components().len()
    }

    /// Number of coordinates of a point.
    #[getter]
    fn n_dimensions(&self) -> usize {
        self.mixture.n_dimensions()
    }

    /// Mixing weights, shape `(n_components,)`.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let weights: Array1<f64> = self
            .mixture
            .components()
            .iter()
            .map(GaussianComponent::weight)
            .collect();
        weights.into_pyarray(py)
    }

    /// Component means, shape `(n_components, n_dimensions)`.
    #[getter]
    fn means<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = self
            .mixture
            .components()
            .iter()
            .flat_map(|component| component.mean().iter().copied());
        let shape = (self.n_components(), self.n_dimensions());
        let means = Array2::from_shape_vec(shape, values.collect()).map_err(to_value_error)?;
        Ok(means.into_pyarray(py))
    }

    /// Component covariances, shape `(n_components, n_dimensions, n_dimensions)`.
    #[getter]
    fn covariances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let values = self
            .mixture
            .components()
            .iter()
            .flat_map(|component| component.covariance().iter().copied());
        let shape = (self.n_components(), self.n_dimensions(), self.n_dimensions());
        let covariances = Array3::from_shape_vec(shape, values.collect()).map_err(to_value_error)?;
        Ok(covariances.into_pyarray(py))
    }

    /// Log-density of the mixture at each row of `points`.
    fn log_density<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let points = points.as_array();
        if points.ncols() != self.n_dimensions() {
            let message = format!(
                "points have {} columns but the mixture has {} dimensions; pass an array of shape (n, {})",
                points.ncols(),
                self.n_dimensions(),
                self.n_dimensions()
            );
            return Err(PyValueError::new_err(message));
        }
        let log_densities: Array1<f64> = points
            .rows()
            .into_iter()
            .map(|row| self.mixture.log_density(&row.to_vec()))
            .collect();
        Ok(log_densities.into_pyarray(py))
    }

    /// Draws `n_points` points from the mixture, one per row.
    fn sample<'py>(&self, py: Python<'py>, n_points: usize, seed: u64) -> Bound<'py, PyArray2<f64>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut points = Array2::zeros((n_points, self.n_dimensions()));
        let mut point = vec![0.0; self.n_dimensions()];
        for mut row in points.rows_mut() {
            self.mixture.sample(&mut rng, &mut point);
            row.iter_mut().zip(&point).for_each(|(target, value)| *target = *value);
        }
        points.into_pyarray(py)
    }
}

/// Fits a Dirichlet process Gaussian mixture to `points` (one point per
/// row) with `n_sweeps` Gibbs sweeps; the result is a function of `seed`.
#[pyfunction]
#[pyo3(signature = (points, *, n_sweeps = DpgmmConfig::DEFAULT_N_SWEEPS, seed = 0))]
fn fit(points: PyReadonlyArray2<'_, f64>, n_sweeps: u32, seed: u64) -> PyResult<PyGaussianMixture> {
    let points = points.as_array();
    let values: Vec<f64> = points.iter().copied().collect();
    let observations = Observations::from_rows(&values, points.ncols()).map_err(to_value_error)?;
    let estimator = Dpgmm::new(DpgmmConfig::new(n_sweeps).map_err(to_value_error)?);
    let mixture = estimator
        .fit(&observations, &mut ChaCha8Rng::seed_from_u64(seed))
        .map_err(to_value_error)?;
    Ok(PyGaussianMixture { mixture })
}

/// Turns a Rust error into a Python `ValueError` carrying its message.
fn to_value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Dirichlet process Gaussian mixture density estimation.
#[pymodule]
fn dpgmm(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGaussianMixture>()?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    Ok(())
}
