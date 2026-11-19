//! Deterministic simulation of a model as an ordinary differential equation.
//!
//! The equation is integrated with the variable-order BDF method of
//! `diffsol`, which is suited to the stiff systems typical of reaction
//! networks. The Jacobian comes from forward-mode differentiation of the
//! model's formulas. Timed assignments split the integration into segments
//! within which the parameters are constant.

use std::cell::RefCell;

use diffsol::{NalgebraLU, NalgebraMat, OdeBuilder, OdeSolverMethod};

use crate::model_error::ModelError;
use crate::simulation_failure::SimulationFailure;
use crate::simulation_model::{DualScratch, SimulationModel};

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
    /// Default bound on the number of solver steps between two timed
    /// assignments.
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

    /// Returns a copy that gives up after `max_steps` solver steps between
    /// two timed assignments.
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
    values: Vec<f64>,
    stack: Vec<f64>,
    dual: DualScratch,
}

/// Integrates the ODE of a model.
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

    /// Integrates from time 0, starting from the species amounts and
    /// parameters in `initial_values`, and returns the full value buffer at
    /// each of `times` (non-negative and increasing), with its derived
    /// values up to date.
    ///
    /// A buffer recorded exactly at the time of a timed assignment reflects
    /// the parameters after the assignment.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationFailure::SolverFailed`] if the solver cannot
    /// complete a segment within its limits.
    pub fn solve(&self, initial_values: &[f64], times: &[f64]) -> Result<Vec<Vec<f64>>, SimulationFailure> {
        let assignment_times = self.model.assignment_times();
        let last_time = times.last().copied().unwrap_or(0.0);
        let (mut values, mut stack) = (initial_values.to_vec(), Vec::new());
        let (mut time, mut recorded) = (0.0, Vec::with_capacity(times.len()));
        loop {
            self.model.apply_assignments_at(time, &mut values, &mut stack);
            self.model.refresh_derived_values(&mut values, time, &mut stack);
            let n_due = times[recorded.len()..]
                .iter()
                .take_while(|output| **output <= time)
                .count();
            recorded.extend(std::iter::repeat_n(values.clone(), n_due));
            if recorded.len() == times.len() {
                return Ok(recorded);
            }
            let next_assignment = assignment_times.iter().copied().find(|assignment| *assignment > time);
            let segment_end = next_assignment.map_or(last_time, |assignment| assignment.min(last_time));
            let n_inside = times[recorded.len()..]
                .iter()
                .take_while(|output| **output < segment_end)
                .count();
            let mut outputs = times[recorded.len()..recorded.len() + n_inside].to_vec();
            outputs.push(segment_end);
            let states = self.integrate_segment(&values, time, &outputs)?;
            for (state, output) in states.iter().zip(&outputs).take(n_inside) {
                recorded.push(self.buffer_with_state(&values, state, *output, &mut stack));
            }
            values[..self.model.n_species()].copy_from_slice(&states[n_inside]);
            time = segment_end;
        }
    }

    /// A copy of `values` holding `state` at `time`, derived values
    /// refreshed.
    fn buffer_with_state(&self, values: &[f64], state: &[f64], time: f64, stack: &mut Vec<f64>) -> Vec<f64> {
        let mut buffer = values.to_vec();
        buffer[..state.len()].copy_from_slice(state);
        self.model.refresh_derived_values(&mut buffer, time, stack);
        buffer
    }

    /// Integrates with constant parameters from `start` and returns the
    /// species amounts at each of `outputs` (increasing, after `start`).
    fn integrate_segment(
        &self,
        values: &[f64],
        start: f64,
        outputs: &[f64],
    ) -> Result<Vec<Vec<f64>>, SimulationFailure> {
        let n_species = self.model.n_species();
        let scratch = RefCell::new(OdeScratch {
            values: values.to_vec(),
            ..OdeScratch::default()
        });
        let load = |amounts: &[f64], time: f64| {
            let mut scratch = scratch.borrow_mut();
            let OdeScratch { values, stack, .. } = &mut *scratch;
            values[..n_species].copy_from_slice(amounts);
            self.model.refresh_derived_values(values, time, stack);
            scratch
        };
        let mut problem = OdeBuilder::<NalgebraMat<f64>>::new()
            .t0(start)
            .rtol(self.settings.relative_tolerance)
            .atol([self.settings.absolute_tolerance])
            .rhs_implicit(
                |amounts, _parameters, time, derivative| {
                    let scratch = &mut *load(amounts, time);
                    self.model
                        .write_time_derivative(&scratch.values, &mut scratch.stack, derivative);
                },
                |amounts, _parameters, time, direction, product| {
                    let scratch = &mut *load(amounts, time);
                    self.model
                        .write_jacobian_product(&scratch.values, direction, &mut scratch.dual, product);
                },
            )
            .init(
                |_parameters, _time, amounts| amounts.copy_from_slice(&values[..n_species]),
                n_species,
            )
            .build()
            .map_err(|error| Self::failure(&error))?;
        problem.ode_options.max_number_timesteps = self.settings.max_steps;
        let mut solver = problem
            .bdf::<NalgebraLU<f64>>()
            .map_err(|error| Self::failure(&error))?;
        let (solution, _stop_reason) = solver.solve_dense(outputs).map_err(|error| Self::failure(&error))?;
        Ok((0..outputs.len())
            .map(|column| (0..n_species).map(|row| solution[(row, column)]).collect())
            .collect())
    }

    fn failure(error: &diffsol::DiffsolError) -> SimulationFailure {
        SimulationFailure::SolverFailed {
            message: error.to_string(),
        }
    }
}
