//! A PEtab problem as seen from Python.

use std::path::PathBuf;

use ::lfns::{LowAcceptanceRule, RunConfig};
use ::lfns_model::{LikelihoodMethod, OdeSettings, ParticleFilter, PetabLikelihood, PetabProblem};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::proposal_choice::ProposalSettings;
use crate::run_result::PyRunResult;
use crate::run_settings::{Checkpointing, RunSettings};
use crate::to_value_error;

/// Measurement times, observable indices, and measured values of one
/// trajectory, one entry per reading.
type Readings<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);

/// Species amounts and simulated measurements, each with one row per time.
type SimulatedArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// Flattens rows of equal length into a two-dimensional array.
fn to_matrix<'py>(py: Python<'py>, rows: &[Vec<f64>], n_columns: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let values: Vec<f64> = rows.iter().flatten().copied().collect();
    let matrix = Array2::from_shape_vec((rows.len(), n_columns), values).map_err(to_value_error)?;
    Ok(matrix.into_pyarray(py))
}

/// Turns the `dynamics` name and particle count given in Python into a
/// likelihood method.
fn parse_method(dynamics: &str, n_particles: Option<usize>) -> PyResult<LikelihoodMethod> {
    match (dynamics, n_particles) {
        ("deterministic", None) => Ok(LikelihoodMethod::Deterministic(OdeSettings::default())),
        ("deterministic", Some(_)) => Err(PyValueError::new_err(
            "n_particles applies to stochastic dynamics only; drop it or pass dynamics=\"stochastic\"",
        )),
        ("stochastic", Some(count)) => Ok(LikelihoodMethod::ParticleFilter(
            ParticleFilter::new(count).map_err(to_value_error)?,
        )),
        ("stochastic", None) => Err(PyValueError::new_err(
            "stochastic dynamics need the number of filter particles; pass n_particles=...",
        )),
        (other, _) => Err(PyValueError::new_err(format!(
            "unknown dynamics \"{other}\"; pass dynamics=\"stochastic\" or dynamics=\"deterministic\""
        ))),
    }
}

/// A parameter estimation problem loaded from a PEtab YAML file. The model
/// is simulated in Rust; Python only passes parameter values and settings.
#[pyclass(name = "PetabProblem", frozen)]
pub(crate) struct PyPetabProblem {
    problem: PetabProblem,
}

impl PyPetabProblem {
    /// Values of all model parameters when the estimated ones take the
    /// given values.
    fn model_parameters(&self, estimated_values: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
        let space = self.problem.parameter_space();
        let values = estimated_values.as_array().to_vec();
        if values.len() != space.estimated().len() {
            return Err(PyValueError::new_err(format!(
                "got {} parameter values but the problem estimates {}; pass one value per entry of `parameter_ids`",
                values.len(),
                space.estimated().len()
            )));
        }
        Ok(space.model_parameters_from_values(&values))
    }

    /// Position of the condition named `condition`; the first condition if
    /// none is named.
    fn condition_index(&self, condition: Option<&str>) -> PyResult<usize> {
        let Some(id) = condition else { return Ok(0) };
        self.problem
            .conditions()
            .iter()
            .position(|each| each.id() == id)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown condition \"{id}\"; the problem defines {:?}",
                    self.condition_ids()
                ))
            })
    }
}

#[pymethods]
impl PyPetabProblem {
    /// Loads the problem described by the PEtab YAML file at `path`.
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            problem: PetabProblem::load(path).map_err(to_value_error)?,
        })
    }

    /// Identifiers of the estimated parameters, in the order used by every
    /// array of parameter values.
    #[getter]
    fn parameter_ids(&self) -> Vec<String> {
        self.problem
            .parameter_space()
            .estimated()
            .iter()
            .map(|parameter| parameter.id().to_owned())
            .collect()
    }

    /// Identifiers of the model species.
    #[getter]
    fn species_ids(&self) -> Vec<String> {
        self.problem.species_ids().to_vec()
    }

    /// Identifiers of the observables.
    #[getter]
    fn observable_ids(&self) -> Vec<String> {
        self.problem.observable_ids().to_vec()
    }

    /// Identifiers of the experimental conditions.
    #[getter]
    fn condition_ids(&self) -> Vec<String> {
        self.problem
            .conditions()
            .iter()
            .map(|condition| condition.id().to_owned())
            .collect()
    }

    /// The condition of each measured trajectory, in the order of
    /// `measurements()`.
    #[getter]
    fn trajectory_conditions(&self) -> Vec<String> {
        let conditions = self.problem.conditions();
        self.problem
            .trajectories()
            .iter()
            .map(|trajectory| conditions[trajectory.condition_index()].id().to_owned())
            .collect()
    }

    /// Nominal values of the estimated parameters, from the parameter table.
    #[getter]
    fn nominal_parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from_vec(self.problem.parameter_space().estimated_nominal_values()).into_pyarray(py)
    }

    /// The measured trajectories, each as `(times, observable_indices,
    /// measurements)` with one entry per reading.
    fn measurements<'py>(&self, py: Python<'py>) -> Vec<Readings<'py>> {
        let export = |trajectory: &::lfns_model::Trajectory| {
            let readings = trajectory
                .time_points()
                .iter()
                .flat_map(|(time, group)| group.iter().map(move |r| (*time, r)));
            let times: Array1<f64> = readings.clone().map(|(time, _)| time).collect();
            let indices: Array1<usize> = readings.clone().map(|(_, reading)| reading.observable_index).collect();
            let values: Array1<f64> = readings.map(|(_, reading)| reading.measurement).collect();
            (
                times.into_pyarray(py),
                indices.into_pyarray(py),
                values.into_pyarray(py),
            )
        };
        self.problem.trajectories().iter().map(export).collect()
    }

    /// Maps points of the unit cube (one per row) to values of the estimated
    /// parameters through the priors of the problem.
    fn to_parameters<'py>(
        &self,
        py: Python<'py>,
        unit_points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let space = self.problem.parameter_space();
        let rows: Vec<Vec<f64>> = unit_points
            .as_array()
            .rows()
            .into_iter()
            .map(|row| space.estimated_values(&row.to_vec()))
            .collect();
        to_matrix(py, &rows, space.estimated().len())
    }

    /// Simulates the model once with the given values of the estimated
    /// parameters under `condition` (default: the first condition) and
    /// returns two arrays with one row per time: the species amounts and
    /// the observables.
    ///
    /// With `dynamics="stochastic"` the model is simulated with Gillespie's
    /// algorithm from `seed` and the observables carry measurement noise.
    /// With `dynamics="deterministic"` the ODE is solved and the observables
    /// are their noise-free expected values.
    #[pyo3(signature = (parameters, times, seed = 0, *, dynamics = "stochastic", condition = None))]
    // Besides `self` and the interpreter token, the arguments are the Python
    // signature of the method.
    #[allow(clippy::too_many_arguments)]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        parameters: PyReadonlyArray1<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
        seed: u64,
        dynamics: &str,
        condition: Option<&str>,
    ) -> PyResult<SimulatedArrays<'py>> {
        let model_parameters = self.model_parameters(&parameters)?;
        let condition_index = self.condition_index(condition)?;
        let times = times.as_array().to_vec();
        let trajectory = match parse_method(dynamics, (dynamics == "stochastic").then_some(1))? {
            LikelihoodMethod::ParticleFilter(_) => self.problem.simulate(
                &model_parameters,
                condition_index,
                &times,
                &mut ChaCha8Rng::seed_from_u64(seed),
            ),
            LikelihoodMethod::Deterministic(settings) => {
                self.problem
                    .simulate_deterministic(&model_parameters, condition_index, &times, settings)
            }
        };
        let trajectory = trajectory.map_err(to_value_error)?;
        let species = to_matrix(py, &trajectory.species_amounts, self.problem.species_ids().len())?;
        let observations = to_matrix(py, &trajectory.observations, self.problem.observable_ids().len())?;
        Ok((species, observations))
    }

    /// Returns the log-likelihood at the given values of the estimated
    /// parameters: exact for `dynamics="deterministic"`, one particle-filter
    /// estimate with `n_particles` particles for `dynamics="stochastic"`.
    #[pyo3(signature = (parameters, n_particles = None, seed = 0, *, dynamics = "stochastic"))]
    fn log_likelihood(
        &self,
        parameters: PyReadonlyArray1<'_, f64>,
        n_particles: Option<usize>,
        seed: u64,
        dynamics: &str,
    ) -> PyResult<f64> {
        let model_parameters = self
            .problem
            .parameter_space()
            .model_parameters_from_values(&parameters.as_array().to_vec());
        let likelihood =
            PetabLikelihood::new(self.problem.clone(), parse_method(dynamics, n_particles)?).map_err(to_value_error)?;
        likelihood
            .estimate_for_parameters(&model_parameters, &mut ChaCha8Rng::seed_from_u64(seed))
            .map_err(to_value_error)
    }

    /// Runs likelihood-free nested sampling on the problem. With
    /// `dynamics="stochastic"` the likelihood is estimated with
    /// `n_particles` filter particles; with `dynamics="deterministic"` the
    /// ODE is solved and the likelihood is exact. See `lfns.run` for the
    /// remaining settings, including checkpointing and resuming.
    #[pyo3(signature = (
        *, n_live, n_replaced, dynamics = "stochastic", n_particles = None, seed = 0,
        termination_tolerance = RunConfig::DEFAULT_TERMINATION_TOLERANCE, max_iterations = None,
        proposal = "dpgmm", rejection_quantile = ::lfns::DpgmmProposalConfig::DEFAULT_REJECTION_QUANTILE,
        n_sweeps = ::dpgmm::DpgmmConfig::DEFAULT_N_SWEEPS, low_acceptance_threshold = None,
        low_acceptance_quantile = LowAcceptanceRule::DEFAULT_REJECTION_QUANTILE,
        run_directory = None, resume = false, progress = false,
    ))]
    // The many arguments are keyword-only in Python, where they form the
    // public configuration surface of a run.
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn run(
        &self,
        py: Python<'_>,
        n_live: u32,
        n_replaced: u32,
        dynamics: &str,
        n_particles: Option<usize>,
        seed: u64,
        termination_tolerance: f64,
        max_iterations: Option<u32>,
        proposal: &str,
        rejection_quantile: f64,
        n_sweeps: u32,
        low_acceptance_threshold: Option<f64>,
        low_acceptance_quantile: f64,
        run_directory: Option<PathBuf>,
        resume: bool,
        progress: bool,
    ) -> PyResult<PyRunResult> {
        let settings = RunSettings::new((n_live, n_replaced), seed, termination_tolerance, max_iterations)?
            .with_proposal(&ProposalSettings {
                name: proposal,
                rejection_quantile,
                n_sweeps,
                low_acceptance_threshold,
                low_acceptance_quantile,
            })?
            .with_checkpointing(Checkpointing {
                run_directory,
                resume,
                progress,
            })?;
        let likelihood =
            PetabLikelihood::new(self.problem.clone(), parse_method(dynamics, n_particles)?).map_err(to_value_error)?;
        settings.run_sampler(py, likelihood)
    }
}
