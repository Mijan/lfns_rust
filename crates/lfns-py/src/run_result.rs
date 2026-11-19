//! The outcome of a run as seen from Python.

use ::lfns::{EvidenceEstimate, IterationReport, Particle, RunState, StopReason};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Points (one per row) and one value per point.
type PointsAndValues<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);

/// Final state, per-iteration reports, and stop reason of a run.
#[pyclass(name = "RunResult", frozen)]
pub(crate) struct PyRunResult {
    state: RunState,
    reports: Vec<IterationReport>,
    stop_reason: StopReason,
}

impl PyRunResult {
    pub(crate) fn new(state: RunState, reports: Vec<IterationReport>, stop_reason: StopReason) -> Self {
        Self {
            state,
            reports,
            stop_reason,
        }
    }

    fn estimate(&self) -> EvidenceEstimate {
        self.state.estimate_evidence()
    }

    /// Collects one float per report into a `NumPy` array.
    fn report_column<'py>(
        &self,
        py: Python<'py>,
        value: impl Fn(&IterationReport) -> f64,
    ) -> Bound<'py, PyArray1<f64>> {
        self.reports.iter().map(value).collect::<Array1<f64>>().into_pyarray(py)
    }

    /// Splits particles into a coordinate matrix and their log-likelihoods.
    fn export_particles<'py>(&self, py: Python<'py>, particles: &[Particle]) -> PyResult<PointsAndValues<'py>> {
        let coordinates: Vec<f64> = particles
            .iter()
            .flat_map(|p| p.unit_coordinates().iter().copied())
            .collect();
        let shape = (particles.len(), self.state.n_dimensions());
        let points = Array2::from_shape_vec(shape, coordinates).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let log_likelihoods: Array1<f64> = particles.iter().map(Particle::log_likelihood).collect();
        Ok((points.into_pyarray(py), log_likelihoods.into_pyarray(py)))
    }
}

#[pymethods]
impl PyRunResult {
    /// Natural logarithm of the evidence estimate.
    #[getter]
    fn log_evidence(&self) -> f64 {
        self.estimate().log_evidence()
    }

    /// Logarithm of the part of the evidence integrated by the dead particles.
    #[getter]
    fn log_dead_evidence(&self) -> f64 {
        self.estimate().log_dead_evidence()
    }

    /// Logarithm of the part of the evidence held by the live particles.
    #[getter]
    fn log_live_evidence(&self) -> f64 {
        self.estimate().log_live_evidence()
    }

    /// Standard deviation of the evidence estimate divided by the evidence.
    #[getter]
    fn relative_std(&self) -> f64 {
        self.estimate().relative_std()
    }

    /// Lower bound on the relative standard deviation reachable by running on.
    #[getter]
    fn relative_floor_std(&self) -> f64 {
        self.estimate().relative_floor_std()
    }

    /// Number of completed iterations.
    #[getter]
    fn n_iterations(&self) -> u32 {
        self.state.iteration()
    }

    /// Number of likelihood evaluations, including the initial population.
    #[getter]
    fn n_proposals(&self) -> u64 {
        self.state.n_proposals()
    }

    /// `"converged"` or `"iteration_limit"`.
    #[getter]
    fn stop_reason(&self) -> &'static str {
        match self.stop_reason {
            StopReason::Converged => "converged",
            StopReason::IterationLimit => "iteration_limit",
        }
    }

    /// Per-iteration diagnostics as a dict of arrays, one entry per iteration.
    fn reports<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        // Counts stay far below 2^53, where the conversion to float is exact.
        #[allow(clippy::cast_precision_loss)]
        let n_proposals = |report: &IterationReport| report.n_proposals() as f64;
        #[allow(clippy::cast_precision_loss)]
        let n_failures = |report: &IterationReport| report.n_failures() as f64;
        let columns = PyDict::new(py);
        columns.set_item("iteration", self.report_column(py, |r| f64::from(r.iteration())))?;
        columns.set_item("log_threshold", self.report_column(py, IterationReport::log_threshold))?;
        columns.set_item("n_proposals", self.report_column(py, n_proposals))?;
        columns.set_item("n_failures", self.report_column(py, n_failures))?;
        columns.set_item(
            "acceptance_rate",
            self.report_column(py, IterationReport::acceptance_rate),
        )?;
        columns.set_item("log_evidence", self.report_column(py, |r| r.evidence().log_evidence()))?;
        columns.set_item(
            "log_dead_evidence",
            self.report_column(py, |r| r.evidence().log_dead_evidence()),
        )?;
        columns.set_item(
            "log_live_evidence",
            self.report_column(py, |r| r.evidence().log_live_evidence()),
        )?;
        columns.set_item("relative_std", self.report_column(py, |r| r.evidence().relative_std()))?;
        columns.set_item(
            "relative_floor_std",
            self.report_column(py, |r| r.evidence().relative_floor_std()),
        )?;
        columns.set_item(
            "achievable_improvement",
            self.report_column(py, |r| r.evidence().achievable_improvement()),
        )?;
        Ok(columns)
    }

    /// Weighted posterior sample: unit-cube points and normalized log-weights.
    fn posterior<'py>(&self, py: Python<'py>) -> PyResult<PointsAndValues<'py>> {
        let posterior = self
            .state
            .assemble_posterior()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (particles, log_weights): (Vec<Particle>, Vec<f64>) = posterior
            .weighted_particles()
            .map(|(particle, log_weight)| (particle.clone(), log_weight))
            .unzip();
        let (points, _) = self.export_particles(py, &particles)?;
        Ok((points, Array1::from_vec(log_weights).into_pyarray(py)))
    }

    /// Kish effective sample size of the posterior sample.
    fn effective_sample_size(&self) -> PyResult<f64> {
        let posterior = self
            .state
            .assemble_posterior()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(posterior.effective_sample_size())
    }

    /// Live particles: unit-cube points and their log-likelihoods.
    fn live_points<'py>(&self, py: Python<'py>) -> PyResult<PointsAndValues<'py>> {
        self.export_particles(py, self.state.live().particles())
    }

    /// Dead particles in order of removal: unit-cube points and log-likelihoods.
    fn dead_points<'py>(&self, py: Python<'py>) -> PyResult<PointsAndValues<'py>> {
        self.export_particles(py, self.state.dead())
    }
}
