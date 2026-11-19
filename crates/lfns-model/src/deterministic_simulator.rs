//! Deterministic simulation of a reaction network as an ordinary
//! differential equation.
//!
//! The species amounts follow `dx/dt = sum_r change_r * rate_r(x, p)`. The
//! equation is integrated with the variable-order BDF method of `diffsol`,
//! which is suited to the stiff systems typical of reaction networks. The
//! Jacobian comes from the symbolic derivatives of the rate laws.

use std::cell::RefCell;

use diffsol::{NalgebraLU, NalgebraMat, OdeBuilder, OdeSolverMethod};

use crate::model_error::ModelError;
use crate::simulation_failure::SimulationFailure;
use crate::simulation_model::SimulationModel;

/// Accuracy and effort limits of the ODE solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdeSettings {
    relative_tolerance: f64,
    absolute_tolerance: f64,
    max_steps: usize,
}

impl OdeSettings {
    /// Default relative tolerance on the species amounts.
    pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1e-8;
    /// Default absolute tolerance on the species amounts.
    pub const DEFAULT_ABSOLUTE_TOLERANCE: f64 = 1e-10;
    /// Default bound on the number of solver steps of one integration.
    pub const DEFAULT_MAX_STEPS: usize = 100_000;

    /// Creates settings with the given tolerances and the default step limit.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnsupportedPetab`] unless both tolerances are
    /// positive and finite.
    pub fn new(relative_tolerance: f64, absolute_tolerance: f64) -> Result<Self, ModelError> {
        let is_valid = |tolerance: f64| tolerance.is_finite() && tolerance > 0.0;
        if !(is_valid(relative_tolerance) && is_valid(absolute_tolerance)) {
            return Err(ModelError::UnsupportedPetab {
                feature: format!(
                    "ODE tolerances of {relative_tolerance} (relative) and {absolute_tolerance} (absolute)"
                ),
                remedy: "Use positive finite tolerances, e.g. 1e-8 and 1e-10.".to_owned(),
            });
        }
        Ok(Self {
            relative_tolerance,
            absolute_tolerance,
            max_steps: Self::DEFAULT_MAX_STEPS,
        })
    }

    /// Returns a copy that gives up after `max_steps` solver steps.
    #[must_use]
    pub fn with_max_steps(self, max_steps: usize) -> Self {
        Self { max_steps, ..self }
    }
}

impl Default for OdeSettings {
    fn default() -> Self {
        Self {
            relative_tolerance: Self::DEFAULT_RELATIVE_TOLERANCE,
            absolute_tolerance: Self::DEFAULT_ABSOLUTE_TOLERANCE,
            max_steps: Self::DEFAULT_MAX_STEPS,
        }
    }
}

/// Buffers shared by the callbacks the solver invokes.
#[derive(Default)]
struct OdeScratch {
    /// Species amounts followed by parameters, as the formulas expect.
    values: Vec<f64>,
    stack: Vec<f64>,
}

/// Integrates the ODE of a model from its initial state.
#[derive(Debug, Clone, Copy)]
pub struct DeterministicSimulator<'a> {
    model: &'a SimulationModel,
    settings: OdeSettings,
}

impl<'a> DeterministicSimulator<'a> {
    /// Creates a simulator for `model`.
    #[must_use]
    pub fn new(model: &'a SimulationModel, settings: OdeSettings) -> Self {
        Self { model, settings }
    }

    /// Integrates from time 0 with the given values of all model parameters
    /// and returns the species amounts at each of `times` (increasing), one
    /// row per time.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationFailure::SolverFailed`] if the solver cannot
    /// reach the last time within its limits.
    pub fn solve(&self, parameters: &[f64], times: &[f64]) -> Result<Vec<Vec<f64>>, SimulationFailure> {
        let n_species = self.model.n_species();
        let mut initial_values = vec![0.0; self.model.value_width()];
        self.model.write_initial_values(parameters, &mut initial_values);
        let scratch = RefCell::new(OdeScratch::default());
        let load = |amounts: &[f64], parameters: &[f64]| {
            let mut scratch = scratch.borrow_mut();
            scratch.values.clear();
            scratch.values.extend_from_slice(amounts);
            scratch.values.extend_from_slice(parameters);
            scratch
        };
        let mut problem = OdeBuilder::<NalgebraMat<f64>>::new()
            .rtol(self.settings.relative_tolerance)
            .atol([self.settings.absolute_tolerance])
            .p(parameters.iter().copied())
            .rhs_implicit(
                |amounts, parameters, _time, derivative| {
                    let scratch = &mut *load(amounts, parameters);
                    self.model
                        .write_time_derivative(&scratch.values, &mut scratch.stack, derivative);
                },
                |amounts, parameters, _time, direction, product| {
                    let scratch = &mut *load(amounts, parameters);
                    self.model
                        .write_jacobian_product(&scratch.values, direction, &mut scratch.stack, product);
                },
            )
            .init(
                |_parameters, _time, amounts| amounts.copy_from_slice(&initial_values[..n_species]),
                n_species,
            )
            .build()
            .map_err(Self::failure)?;
        problem.ode_options.max_number_timesteps = self.settings.max_steps;
        let mut solver = problem.bdf::<NalgebraLU<f64>>().map_err(Self::failure)?;
        let (solution, _stop_reason) = solver.solve_dense(times).map_err(Self::failure)?;
        Ok((0..times.len())
            .map(|column| (0..n_species).map(|row| solution[(row, column)]).collect())
            .collect())
    }

    fn failure(error: diffsol::DiffsolError) -> SimulationFailure {
        SimulationFailure::SolverFailed {
            message: error.to_string(),
        }
    }
}
