//! Loading a parameter estimation problem in PEtab format (version 1).
//!
//! Supported subset: one SBML model; one or more measurement, observable,
//! and condition tables; conditions that set initial species values and
//! parameters to numbers; normally distributed measurement noise;
//! parameters on a linear or logarithmic scale with the default prior
//! (uniform on the parameter scale between the bounds). Measurements are
//! grouped into trajectories by simulation condition and `replicateId`.
//! Anything else is rejected with an error that names it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rand::Rng;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::deterministic_simulator::{DeterministicSimulator, OdeSettings};
use crate::experimental_condition::ExperimentalCondition;
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

/// One simulated trajectory: per requested time, the species readings and
/// the observables.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulatedTrajectory {
    /// Species amounts at each time, in the order of the model species.
    pub species_amounts: Vec<Vec<f64>>,
    /// One value per observable at each time.
    pub observations: Vec<Vec<f64>>,
}

/// A parameter estimation problem: a model, its parameters with their
/// priors, the experimental conditions, and the measured trajectories.
#[derive(Debug, Clone, PartialEq)]
pub struct PetabProblem {
    model: SimulationModel,
    parameter_space: ParameterSpace,
    conditions: Vec<ExperimentalCondition>,
    trajectories: Vec<Trajectory>,
    species_ids: Vec<String>,
    observable_ids: Vec<String>,
}

impl PetabProblem {
    const DEFAULT_PRIOR: &'static str = "parameterScaleUniform";
    const NORMAL_NOISE: &'static str = "normal";
    const NO_TRANSFORMATION: &'static str = "lin";
    const CONDITION_ID_COLUMN: &'static str = "conditionId";
    const CONDITION_NAME_COLUMN: &'static str = "conditionName";

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
        let parameter_path = directory.join(&file.parameter_file);
        let parameter_rows: Vec<ParameterRow> = read_table(&parameter_path)?;
        for row in &parameter_rows {
            network.set_parameter(&row.parameter_id, row.nominal_value);
        }
        let observable_rows: Vec<ObservableRow> = read_tables(directory, &entry.observable_files)?;
        let measurement_rows: Vec<MeasurementRow> = read_tables(directory, &entry.measurement_files)?;
        let observables = observable_rows
            .iter()
            .map(Self::read_observable)
            .collect::<Result<Vec<_>, _>>()?;
        let parameter_space = Self::read_parameter_space(&network, &parameter_rows, &parameter_path)?;
        let conditions = Self::read_conditions(&network, &parameter_space, directory, &entry.condition_files)?;
        Ok(Self {
            model: SimulationModel::new(&network, &observables)?,
            trajectories: Self::read_trajectories(&measurement_rows, &observables, &conditions, directory)?,
            parameter_space,
            conditions,
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

    /// The experimental conditions.
    #[must_use]
    pub fn conditions(&self) -> &[ExperimentalCondition] {
        &self.conditions
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

    /// The value buffer at time zero for the given values of all model
    /// parameters under the condition at `condition_index`.
    #[must_use]
    pub fn initial_values(&self, parameters: &[f64], condition_index: usize) -> Vec<f64> {
        let mut values = vec![0.0; self.model.value_width()];
        self.model.write_initial_values(parameters, &mut values);
        self.conditions[condition_index].apply_to(&mut values);
        values
    }

    /// Simulates the model stochastically and records it at `times`
    /// (increasing, from 0); the observables carry measurement noise.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationFailure::NotAReactionNetwork`] if the model has
    /// rules or events, and another [`SimulationFailure`] if it cannot be
    /// simulated.
    pub fn simulate<R: Rng + ?Sized>(
        &self,
        parameters: &[f64],
        condition_index: usize,
        times: &[f64],
        rng: &mut R,
    ) -> Result<SimulatedTrajectory, SimulationFailure> {
        if !self.model.is_pure_reaction_network() {
            return Err(SimulationFailure::NotAReactionNetwork);
        }
        let simulator = StochasticSimulator::new(&self.model);
        let mut scratch = SimulationScratch::default();
        let mut values = self.initial_values(parameters, condition_index);
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

    /// Integrates the model as an ODE and records it at `times`
    /// (increasing, from 0). The observables are their expected values,
    /// without measurement noise.
    ///
    /// # Errors
    ///
    /// Returns a [`SimulationFailure`] if the solver fails.
    pub fn simulate_deterministic(
        &self,
        parameters: &[f64],
        condition_index: usize,
        times: &[f64],
        settings: OdeSettings,
    ) -> Result<SimulatedTrajectory, SimulationFailure> {
        let initial_values = self.initial_values(parameters, condition_index);
        let buffers = DeterministicSimulator::new(&self.model, settings).solve(&initial_values, times)?;
        let mut stack = Vec::new();
        let mut trajectory = SimulatedTrajectory {
            species_amounts: Vec::new(),
            observations: Vec::new(),
        };
        for values in &buffers {
            let mean = |index| self.model.observation_mean(index, values, &mut stack);
            trajectory
                .observations
                .push((0..self.model.n_observables()).map(mean).collect());
            trajectory
                .species_amounts
                .push(values[..self.model.n_species()].to_vec());
        }
        Ok(trajectory)
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

    /// Reads the condition tables: one condition per row, every further
    /// column naming a species (initial value) or a parameter it sets.
    fn read_conditions(
        network: &ReactionNetwork,
        parameter_space: &ParameterSpace,
        directory: &Path,
        files: &[PathBuf],
    ) -> Result<Vec<ExperimentalCondition>, ModelError> {
        let mut conditions = Vec::new();
        for file in files {
            let path = directory.join(file);
            let text = read_text(&path)?;
            let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_reader(text.as_bytes());
            let headers = reader.headers().map_err(|error| invalid(&path, &error))?.clone();
            for record in reader.records() {
                let record = record.map_err(|error| invalid(&path, &error))?;
                let cells = headers.iter().zip(record.iter());
                conditions.push(Self::read_condition(network, parameter_space, cells, &path)?);
            }
        }
        Ok(conditions)
    }

    fn read_condition<'a>(
        network: &ReactionNetwork,
        parameter_space: &ParameterSpace,
        cells: impl Iterator<Item = (&'a str, &'a str)>,
        path: &Path,
    ) -> Result<ExperimentalCondition, ModelError> {
        let (mut id, mut overrides) = ("", Vec::new());
        for (column, cell) in cells {
            match column {
                Self::CONDITION_ID_COLUMN => id = cell,
                Self::CONDITION_NAME_COLUMN => {}
                _ if cell.trim().is_empty() => {}
                target => {
                    let value: f64 = cell.trim().parse().map_err(|_| {
                        let feature = format!("the non-numeric value `{cell}` of `{target}` in the condition table");
                        unsupported(
                            &feature,
                            "Write a number; parameter names as condition values are not supported.",
                        )
                    })?;
                    if !value.is_nan() {
                        overrides.push(Self::condition_override(network, parameter_space, target, value, path)?);
                    }
                }
            }
        }
        Ok(ExperimentalCondition::new(id, overrides))
    }

    /// The value-buffer slot and value a condition column sets. A species
    /// column gives the initial value in the units formulas read it in.
    fn condition_override(
        network: &ReactionNetwork,
        parameter_space: &ParameterSpace,
        target: &str,
        value: f64,
        path: &Path,
    ) -> Result<(usize, f64), ModelError> {
        if let Some(index) = network.species_index(target) {
            return Ok((index, value * network.species()[index].amount_per_reading()));
        }
        let reason = |problem: &str| ModelError::Petab {
            path: path.to_path_buf(),
            reason: format!("the condition column `{target}` {problem}"),
        };
        let index = network
            .parameter_index(target)
            .ok_or_else(|| reason("names neither a species nor a parameter of the model"))?;
        if parameter_space
            .estimated()
            .iter()
            .any(|parameter| parameter.id() == target)
        {
            return Err(reason(
                "sets an estimated parameter; set `estimate` to 0 for it or remove the column",
            ));
        }
        Ok((network.species().len() + index, value))
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
        conditions: &[ExperimentalCondition],
        directory: &Path,
    ) -> Result<Vec<Trajectory>, ModelError> {
        let invalid_row = |reason: String| ModelError::Petab {
            path: directory.to_path_buf(),
            reason,
        };
        let mut order: Vec<(usize, &str)> = Vec::new();
        let mut readings: HashMap<(usize, &str), Vec<(f64, Reading)>> = HashMap::new();
        for row in rows {
            Self::check_measurement(row)?;
            let observable_index = observables
                .iter()
                .position(|observable| observable.id == row.observable_id)
                .ok_or_else(|| {
                    invalid_row(format!(
                        "a measurement refers to the undefined observable `{}`",
                        row.observable_id
                    ))
                })?;
            let condition_index = conditions
                .iter()
                .position(|condition| condition.id() == row.simulation_condition_id)
                .ok_or_else(|| {
                    invalid_row(format!(
                        "a measurement refers to the undefined condition `{}`",
                        row.simulation_condition_id
                    ))
                })?;
            let key = (condition_index, row.replicate_id.as_str());
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
        let mut trajectories = Vec::with_capacity(order.len());
        for key in order {
            if let Some(group) = readings.remove(&key) {
                trajectories.push(Trajectory::from_readings(key.0, group));
            }
        }
        Ok(trajectories)
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
        if !row.measurement.is_finite() {
            let feature = format!(
                "the non-finite measurement of `{}` at time {}",
                row.observable_id, row.time
            );
            return Err(unsupported(
                &feature,
                "Remove rows without a measured value from the measurement table.",
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
