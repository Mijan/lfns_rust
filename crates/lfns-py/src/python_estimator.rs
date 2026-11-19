//! A likelihood estimator implemented by a Python callable.

use std::sync::{Arc, Mutex, PoisonError};

use ::lfns::LikelihoodEstimator;
use numpy::PyArray1;
use pyo3::prelude::*;
use rand::Rng;

/// The Python likelihood returned `None` to report a failed evaluation.
#[derive(Debug, thiserror::Error)]
#[error("the Python likelihood reported a failed evaluation by returning None")]
pub(crate) struct ReportedFailure;

/// Where the first exception raised by the Python callable is kept until the
/// run has stopped.
pub(crate) type ExceptionSlot = Arc<Mutex<Option<PyErr>>>;

/// Calls `log_likelihood(unit_point, seed)` for every evaluation.
///
/// The callable returns the log-likelihood as a float, or `None` if the
/// evaluation failed (counted as zero likelihood). An exception is a bug in
/// the callable, not a failed evaluation: it stops the run and is re-raised.
pub(crate) struct PythonEstimator {
    log_likelihood: Py<PyAny>,
    n_parameters: usize,
    first_exception: ExceptionSlot,
}

impl PythonEstimator {
    pub(crate) fn new(log_likelihood: Py<PyAny>, n_parameters: usize) -> (Self, ExceptionSlot) {
        let first_exception = ExceptionSlot::default();
        let estimator = Self {
            log_likelihood,
            n_parameters,
            first_exception: Arc::clone(&first_exception),
        };
        (estimator, first_exception)
    }

    fn call(&self, py: Python<'_>, unit_point: &[f64], seed: u64) -> PyResult<Option<f64>> {
        let point = PyArray1::from_slice(py, unit_point);
        self.log_likelihood.call1(py, (point, seed))?.extract(py)
    }

    fn keep_first_exception(&self, exception: PyErr) {
        let mut slot = self.first_exception.lock().unwrap_or_else(PoisonError::into_inner);
        slot.get_or_insert(exception);
    }
}

impl LikelihoodEstimator for PythonEstimator {
    type Failure = ReportedFailure;

    fn n_parameters(&self) -> usize {
        self.n_parameters
    }

    fn estimate_log_likelihood<R: Rng + ?Sized>(
        &self,
        unit_point: &[f64],
        rng: &mut R,
    ) -> Result<f64, ReportedFailure> {
        let seed = rng.next_u64();
        match Python::attach(|py| self.call(py, unit_point, seed)) {
            Ok(Some(log_likelihood)) => Ok(log_likelihood),
            Ok(None) => Err(ReportedFailure),
            Err(exception) => {
                self.keep_first_exception(exception);
                // NaN is rejected by the sampler as an invalid value, which
                // stops the run at once; the caller then re-raises the
                // exception kept above.
                Ok(f64::NAN)
            }
        }
    }
}
