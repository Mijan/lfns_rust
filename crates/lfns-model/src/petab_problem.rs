//! Loading a parameter estimation problem in PEtab format (version 1).
//!
//! Supported subset: one SBML model; one or more measurement, observable,
//! and condition tables; normally distributed measurement noise;
//! parameters on a linear or logarithmic scale with the default prior
//! (uniform on the parameter scale between the bounds). Measurements are
//! grouped into trajectories by simulation condition and `replicateId`.
//! Anything else is rejected with an error that names it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::deterministic_simulator::{DeterministicSimulator, OdeSettings};
use crate::formula_parser::FormulaParser;
use crate::model_error::ModelError;
use crate::parameter_space::{EstimatedParameter, ParameterScale, ParameterSpace};
use crate::reaction_network::ReactionNetwork;
use crate::simulation_failure::SimulationFailure;
use crate::simulation_model::{ObservableDefinition, SimulationModel};
use crate::stochastic_simulator::{SimulationScratch, StochasticSimulator};
use crate::trajectory::{Reading, Trajectory};

/// Contents of the PEtab YAML file.
#[derive(Deserialize)]
struct ProblemFile {
    format_version: serde_yaml_ng::Value,
    parameter_file: PathBuf,
    problems: Vec<ProblemEntry>,
}

/// One entry of `problems` in the YAML file.
#[derive(Deserialize)]
struct ProblemEntry {
    sbml_files: Vec<PathBuf>,
    measurement_files: Vec<PathBuf>,
    condition_files: Vec<PathBuf>,
    observable_files: Vec<PathBuf>,
}

/// One row of a parameter table.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ParameterRow {
    parameter_id: String,
    parameter_scale: String,
    lower_bound: f64,
    upper_bound: f64,
    nominal_value: f64,
    estimate: u8,
    #[serde(default)]
    objective_prior_type: String,
}

/// One row of an observable table.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ObservableRow {
    observable_id: String,
    observable_formula: String,
    noise_formula: String,
    #[serde(default)]
    noise_distribution: String,
    #[serde(default)]
    observable_transformation: String,
}

/// One row of a measurement table.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeasurementRow {
    observable_id: String,
    simulation_condition_id: String,
    measurement: f64,
    time: f64,
    #[serde(default)]
    replicate_id: String,
    #[serde(default)]
    preequilibration_condition_id: String,
    #[serde(default)]
    observable_parameters: String,
    #[serde(default)]
    noise_parameters: String,
}

/// One simulated trajectory: per requested time, the species amounts and a
/// noisy measurement of every observable.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulatedTrajectory {
    /// Species amounts at each time, in the order of the model species.
    pub species_amounts: Vec<Vec<f64>>,
    /// One simulated measurement per observable at each time.
    pub observations: Vec<Vec<f64>>,
}

/// A parameter estimation problem: a model, its parameters with their
/// priors, and the measured trajectories.
#[derive(Debug, Clone, PartialEq)]
pub struct PetabProblem {
    model: SimulationModel,
    parameter_space: ParameterSpace,
    trajectories: Vec<Trajectory>,
    species_ids: Vec<String>,
    observable_ids: Vec<String>,
}

impl PetabProblem {
    const DEFAULT_PRIOR: &'static str = "parameterScaleUniform";
    const NORMAL_NOISE: &'static str = "normal";
    const NO_TRANSFORMATION: &'static str = "lin";

    /// Loads the problem described by the PEtab YAML file at `yaml_path`.
    ///
    /// # Errors
    ///
    /// Returns a [`ModelError`] if a file cannot be read, is malformed, or
    /// uses a part of SBML or PEtab outside the supported subset.
    pub fn load(yaml_path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let yaml_path = yaml_path.as_ref();
        let directory = yaml_path.parent().unwrap_or(Path::new("."));
        let file: ProblemFile = serde_yaml_ng::from_str(&read_text(yaml_path)?).map_err(|e| invalid(yaml_path, &e))?;
        let entry = Self::single_problem(yaml_path, &file)?;
        let mut network = ReactionNetwork::from_sbml(&read_text(&directory.join(&entry.sbml_files[0]))?)?;
        let parameter_rows: Vec<ParameterRow> = read_table(&directory.join(&file.parameter_file))?;
        for row in &parameter_rows {
            network.set_parameter(&row.parameter_id, row.nominal_value);
        }
        let observable_rows: Vec<ObservableRow> = read_tables(directory, &entry.observable_files)?;
        let measurement_rows: Vec<MeasurementRow> = read_tables(directory, &entry.measurement_files)?;
        Self::check_conditions(directory, &entry.condition_files)?;
        let observables = Self::read_observables(&observable_rows)?;
        Ok(Self {
            model: SimulationModel::new(&network, &observables)?,
            parameter_space: Self::read_parameter_space(
                &network,
                &parameter_rows,
                &directory.join(&file.parameter_file),
            )?,
            trajectories: Self::read_trajectories(&measurement_rows, &observables, directory)?,
            species_ids: network.species().iter().map(|species| species.id.clone()).collect(),
            observable_ids: observables.into_iter().map(|observable| observable.id).collect(),
        })
    }

    /// The compiled model.
    #[must_use]
    pub fn model(&self) -> &SimulationModel {
        &self.model
    }

    /// The parameters and their priors.
    #[must_use]
    pub fn parameter_space(&self) -> &ParameterSpace {
        &self.parameter_space
    }

    /// The measured trajectories.
    #[must_use]
    pub fn trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }

    /// Identifiers of the model species, in the order of the state.
    #[must_use]
    pub fn species_ids(&self) -> &[String] {
        &self.species_ids
    }

    /// Identifiers of the observables.
    #[must_use]
    pub fn observable_ids(&self) -> &[String] {
        &self.observable_ids
    }

    /// Simulates the model stochastically with the given values of all
    /// model parameters and records it at `times` (increasing, from 0).
    ///
    /// # Errors
    ///
    /// Returns a [`SimulationFailure`] if the model cannot be simulated.
    pub fn simulate<R: rand::Rng + ?Sized>(
        &self,
        parameters: &[f64],
        times: &[f64],
        rng: &mut R,
    ) -> Result<SimulatedTrajectory, SimulationFailure> {
        let simulator = StochasticSimulator::new(&self.model);
        let mut scratch = SimulationScratch::default();
        let mut values = vec![0.0; self.model.value_width()];
        self.model.write_initial_values(parameters, &mut values);
        let mut trajectory = SimulatedTrajectory {
            species_amounts: Vec::new(),
            observations: Vec::new(),
        };
        let mut time = 0.0;
        for &next_time in times {
            simulator.advance(&mut values, (time, next_time), &mut scratch, rng)?;
            let observe = |index| self.model.sample_observation(index, &values, &mut Vec::new(), rng);
            trajectory
                .observations
                .push((0..self.model.n_observables()).map(observe).collect::<Result<_, _>>()?);
            trajectory
                .species_amounts
                .push(values[..self.model.n_species()].to_vec());
            time = next_time;
        }
        Ok(trajectory)
    }

    /// Integrates the model as an ODE with the given values of all model
    /// parameters and records it at `times` (increasing, from 0). The
    /// observations are the expected values of the observables, without
    /// measurement noise.
    ///
    /// # Errors
    ///
    /// Returns a [`SimulationFailure`] if the solver fails.
    pub fn simulate_deterministic(
        &self,
        parameters: &[f64],
        times: &[f64],
        settings: OdeSettings,
    ) -> Result<SimulatedTrajectory, SimulationFailure> {
        let species_amounts = DeterministicSimulator::new(&self.model, settings).solve(parameters, times)?;
        let (mut values, mut stack) = (vec![0.0; self.model.value_width()], Vec::new());
        self.model.write_initial_values(parameters, &mut values);
        let mut observations = Vec::with_capacity(times.len());
        for state in &species_amounts {
            values[..state.len()].copy_from_slice(state);
            let mean = |index| self.model.observation_mean(index, &values, &mut stack);
            observations.push((0..self.model.n_observables()).map(mean).collect());
        }
        Ok(SimulatedTrajectory {
            species_amounts,
            observations,
        })
    }

    fn single_problem<'a>(yaml_path: &Path, file: &'a ProblemFile) -> Result<&'a ProblemEntry, ModelError> {
        if file.format_version.as_u64() != Some(1) && file.format_version.as_str() != Some("1") {
            return Err(unsupported(
                "a format_version other than 1",
                "Convert the problem to PEtab version 1.",
            ));
        }
        match file.problems.as_slice() {
            [entry] if entry.sbml_files.len() == 1 => Ok(entry),
            _ => Err(ModelError::Petab {
                path: yaml_path.to_path_buf(),
                reason: "expected exactly one entry under `problems` with exactly one SBML file".to_owned(),
            }),
        }
    }

    /// Rejects condition tables that override model quantities.
    fn check_conditions(directory: &Path, files: &[PathBuf]) -> Result<(), ModelError> {
        for file in files {
            let path = directory.join(file);
            let text = read_text(&path)?;
            let header = text.lines().next().unwrap_or_default();
            if header
                .split('\t')
                .any(|column| !matches!(column.trim(), "conditionId" | "conditionName"))
            {
                let remedy = "Keep only the conditionId and conditionName columns and encode the values in the model.";
                return Err(unsupported("condition-specific overrides of model quantities", remedy));
            }
        }
        Ok(())
    }

    fn read_observables(rows: &[ObservableRow]) -> Result<Vec<ObservableDefinition>, ModelError> {
        rows.iter().map(Self::read_observable).collect()
    }

    fn read_observable(row: &ObservableRow) -> Result<ObservableDefinition, ModelError> {
        if !matches!(row.noise_distribution.as_str(), "" | Self::NORMAL_NOISE) {
            let feature = format!("the noise distribution `{}`", row.noise_distribution);
            return Err(unsupported(&feature, "Use `normal` noise."));
        }
        if !matches!(row.observable_transformation.as_str(), "" | Self::NO_TRANSFORMATION) {
            let feature = format!("the observable transformation `{}`", row.observable_transformation);
            return Err(unsupported(
                &feature,
                "Use `lin` and put the transformation into the observable formula.",
            ));
        }
        Ok(ObservableDefinition {
            id: row.observable_id.clone(),
            formula: FormulaParser::parse(&row.observable_formula)?,
            noise_formula: FormulaParser::parse(&row.noise_formula)?,
        })
    }

    fn read_parameter_space(
        network: &ReactionNetwork,
        rows: &[ParameterRow],
        path: &Path,
    ) -> Result<ParameterSpace, ModelError> {
        let mut estimated = Vec::new();
        for row in rows.iter().filter(|row| row.estimate == 1) {
            if !matches!(row.objective_prior_type.as_str(), "" | Self::DEFAULT_PRIOR) {
                let feature = format!(
                    "the prior `{}` of parameter `{}`",
                    row.objective_prior_type, row.parameter_id
                );
                return Err(unsupported(
                    &feature,
                    "Leave objectivePriorType empty to use the uniform default.",
                ));
            }
            let petab_error = |reason: String| ModelError::Petab {
                path: path.to_path_buf(),
                reason,
            };
            let scale = ParameterScale::parse(&row.parameter_scale).map_err(petab_error)?;
            let index = network.parameter_index(&row.parameter_id).unwrap_or_default();
            let bounds = (row.lower_bound, row.upper_bound);
            estimated.push(EstimatedParameter::new(&row.parameter_id, index, scale, bounds).map_err(petab_error)?);
        }
        let nominal_values = network.parameters().iter().map(|parameter| parameter.value).collect();
        ParameterSpace::new(nominal_values, estimated)
    }

    /// Groups the measurements into one trajectory per simulation condition
    /// and replicate, in order of first appearance.
    fn read_trajectories(
        rows: &[MeasurementRow],
        observables: &[ObservableDefinition],
        directory: &Path,
    ) -> Result<Vec<Trajectory>, ModelError> {
        let mut order: Vec<(&str, &str)> = Vec::new();
        let mut readings: HashMap<(&str, &str), Vec<(f64, Reading)>> = HashMap::new();
        for row in rows {
            Self::check_measurement(row)?;
            let observable_index = observables
                .iter()
                .position(|observable| observable.id == row.observable_id)
                .ok_or_else(|| ModelError::Petab {
                    path: directory.to_path_buf(),
                    reason: format!(
                        "a measurement refers to the undefined observable `{}`",
                        row.observable_id
                    ),
                })?;
            let key = (row.simulation_condition_id.as_str(), row.replicate_id.as_str());
            if !readings.contains_key(&key) {
                order.push(key);
            }
            readings.entry(key).or_default().push((
                row.time,
                Reading {
                    observable_index,
                    measurement: row.measurement,
                },
            ));
        }
        Ok(order
            .into_iter()
            .filter_map(|key| readings.remove(&key))
            .map(Trajectory::from_readings)
            .collect())
    }

    fn check_measurement(row: &MeasurementRow) -> Result<(), ModelError> {
        if !row.preequilibration_condition_id.is_empty() {
            return Err(unsupported(
                "pre-equilibration",
                "Start every simulation from the model's initial state.",
            ));
        }
        if !(row.observable_parameters.is_empty() && row.noise_parameters.is_empty()) {
            let remedy = "Define one observable per parameter set in the observable table.";
            return Err(unsupported(
                "measurement-specific observable or noise parameters",
                remedy,
            ));
        }
        if !(row.time.is_finite() && row.time >= 0.0) {
            return Err(unsupported(
                "measurements at negative or infinite times",
                "Use finite times of at least 0.",
            ));
        }
        Ok(())
    }
}

fn unsupported(feature: &str, remedy: &str) -> ModelError {
    ModelError::UnsupportedPetab {
        feature: feature.to_owned(),
        remedy: remedy.to_owned(),
    }
}

fn invalid(path: &Path, error: &dyn std::fmt::Display) -> ModelError {
    ModelError::Petab {
        path: path.to_path_buf(),
        reason: error.to_string(),
    }
}

fn read_text(path: &Path) -> Result<String, ModelError> {
    std::fs::read_to_string(path).map_err(|source| ModelError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Reads all rows of a tab-separated table with a header line.
fn read_table<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, ModelError> {
    let text = read_text(path)?;
    let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_reader(text.as_bytes());
    reader
        .deserialize()
        .collect::<Result<_, _>>()
        .map_err(|error| invalid(path, &error))
}

/// Reads and concatenates several tables of the same kind.
fn read_tables<T: DeserializeOwned>(directory: &Path, files: &[PathBuf]) -> Result<Vec<T>, ModelError> {
    let mut rows = Vec::new();
    for file in files {
        rows.extend(read_table(&directory.join(file))?);
    }
    Ok(rows)
}
