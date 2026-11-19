//! A model and its observables, compiled for simulation.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::compiled_expression::{CompiledExpression, Dual};
use crate::expression::Expression;
use crate::model_error::ModelError;
use crate::reaction_network::ReactionNetwork;
use crate::simulation_failure::SimulationFailure;

/// An observable as defined in a PEtab observable table: its expected value
/// and the standard deviation of its normally distributed noise.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservableDefinition {
    /// Identifier of the observable.
    pub id: String,
    /// Expected measurement as a function of the model quantities.
    pub formula: Expression,
    /// Standard deviation of the measurement noise.
    pub noise_formula: Expression,
}

/// A compiled observable.
#[derive(Debug, Clone, PartialEq)]
struct Observable {
    mean: CompiledExpression,
    noise_std: CompiledExpression,
}

/// A compiled rate rule: the species it drives, the amount per unit of the
/// quantity the rule is written in, and the rate.
#[derive(Debug, Clone, PartialEq)]
struct CompiledRateRule {
    species_index: usize,
    amount_per_reading: f64,
    rate: CompiledExpression,
}

/// A compiled timed assignment.
#[derive(Debug, Clone, PartialEq)]
struct CompiledTimedAssignment {
    time: f64,
    parameter_slot: usize,
    value: CompiledExpression,
}

/// Buffers for evaluating derivatives along a direction.
#[derive(Debug, Clone, Default)]
pub struct DualScratch {
    tangents: Vec<f64>,
    stack: Vec<Dual>,
}

/// A model ready to be simulated.
///
/// All formulas are evaluated over a *value buffer*: the species amounts,
/// the parameters, the time, and the assignment-rule quantities. The last
/// two are derived; [`SimulationModel::refresh_derived_values`] brings them
/// up to date, and every method that reads a buffer expects that.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationModel {
    initial_amounts: Vec<f64>,
    default_parameters: Vec<f64>,
    rates: Vec<CompiledExpression>,
    net_changes: Vec<Vec<(usize, f64)>>,
    rate_rules: Vec<CompiledRateRule>,
    assignment_rules: Vec<CompiledExpression>,
    timed_assignments: Vec<CompiledTimedAssignment>,
    observables: Vec<Observable>,
}

impl SimulationModel {
    /// Compiles a network and its observables.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownSymbol`] if a formula refers to a name
    /// the network does not define.
    pub fn new(network: &ReactionNetwork, observables: &[ObservableDefinition]) -> Result<Self, ModelError> {
        let compile = |expression: &Expression, context: String| network.compile_expression(expression, &context);
        Ok(Self {
            initial_amounts: network.species().iter().map(|species| species.initial_amount).collect(),
            default_parameters: network.parameters().iter().map(|parameter| parameter.value).collect(),
            rates: Self::compile_each(network.reactions(), |r| {
                compile(&r.rate, format!("the kinetic law of reaction `{}`", r.id))
            })?,
            net_changes: network
                .reactions()
                .iter()
                .map(|reaction| reaction.net_changes.clone())
                .collect(),
            rate_rules: Self::compile_each(network.rate_rules(), |rule| {
                let species = &network.species()[rule.species_index];
                let rate = compile(&rule.rate, format!("the rate rule of `{}`", species.id))?;
                Ok(CompiledRateRule {
                    species_index: rule.species_index,
                    amount_per_reading: species.amount_per_reading(),
                    rate,
                })
            })?,
            assignment_rules: Self::compile_each(network.assignment_rules(), |rule| {
                compile(&rule.expression, format!("the assignment rule of `{}`", rule.id))
            })?,
            timed_assignments: Self::compile_each(network.timed_assignments(), |assignment| {
                let value = compile(
                    &assignment.value,
                    format!("an event assignment of `{}`", assignment.parameter),
                )?;
                let parameter_slot = network.slot_of(&assignment.parameter).unwrap_or_default();
                Ok(CompiledTimedAssignment {
                    time: assignment.time,
                    parameter_slot,
                    value,
                })
            })?,
            observables: Self::compile_each(observables, |definition| {
                let mean = compile(
                    &definition.formula,
                    format!("the formula of observable `{}`", definition.id),
                )?;
                let noise_std = compile(
                    &definition.noise_formula,
                    format!("the noise formula of observable `{}`", definition.id),
                )?;
                Ok(Observable { mean, noise_std })
            })?,
        })
    }

    fn compile_each<I, O>(items: &[I], compile: impl Fn(&I) -> Result<O, ModelError>) -> Result<Vec<O>, ModelError> {
        items.iter().map(compile).collect()
    }

    /// Number of species.
    #[must_use]
    pub fn n_species(&self) -> usize {
        self.initial_amounts.len()
    }

    /// Number of observables.
    #[must_use]
    pub fn n_observables(&self) -> usize {
        self.observables.len()
    }

    /// Length of a value buffer.
    #[must_use]
    pub fn value_width(&self) -> usize {
        self.time_slot() + 1 + self.assignment_rules.len()
    }

    /// Default values of all parameters, in the order of the network.
    #[must_use]
    pub fn default_parameters(&self) -> &[f64] {
        &self.default_parameters
    }

    /// Whether the model consists of reactions only, so that it can be
    /// simulated as a stochastic jump process.
    #[must_use]
    pub fn is_pure_reaction_network(&self) -> bool {
        self.rate_rules.is_empty() && self.assignment_rules.is_empty() && self.timed_assignments.is_empty()
    }

    /// The distinct times at which parameters are assigned, in order.
    #[must_use]
    pub fn assignment_times(&self) -> Vec<f64> {
        let mut times: Vec<f64> = self
            .timed_assignments
            .iter()
            .map(|assignment| assignment.time)
            .collect();
        times.dedup();
        times
    }

    /// The parameters held in a value buffer.
    #[must_use]
    pub fn parameters_of<'a>(&self, values: &'a [f64]) -> &'a [f64] {
        &values[self.n_species()..self.time_slot()]
    }

    /// Writes the initial species amounts and `parameters` into `values`.
    /// The derived values still need refreshing.
    pub fn write_initial_values(&self, parameters: &[f64], values: &mut [f64]) {
        let n_species = self.n_species();
        values[..n_species].copy_from_slice(&self.initial_amounts);
        values[n_species..n_species + parameters.len()].copy_from_slice(parameters);
    }

    /// Sets the time in `values` and recomputes the assignment-rule
    /// quantities from the species amounts and parameters.
    pub fn refresh_derived_values(&self, values: &mut [f64], time: f64, stack: &mut Vec<f64>) {
        let time_slot = self.time_slot();
        values[time_slot] = time;
        for (index, rule) in self.assignment_rules.iter().enumerate() {
            values[time_slot + 1 + index] = rule.evaluate(values, stack);
        }
    }

    /// Applies every timed assignment scheduled for exactly `time`.
    pub fn apply_assignments_at(&self, time: f64, values: &mut [f64], stack: &mut Vec<f64>) {
        for assignment in self
            .timed_assignments
            .iter()
            .filter(|assignment| assignment.time.total_cmp(&time).is_eq())
        {
            self.refresh_derived_values(values, time, stack);
            values[assignment.parameter_slot] = assignment.value.evaluate(values, stack);
        }
    }

    /// Writes the time derivative of the species amounts into `derivative`.
    pub fn write_time_derivative(&self, values: &[f64], stack: &mut Vec<f64>, derivative: &mut [f64]) {
        derivative.fill(0.0);
        for (rate, changes) in self.rates.iter().zip(&self.net_changes) {
            let flux = rate.evaluate(values, stack);
            for &(species, change) in changes {
                derivative[species] += change * flux;
            }
        }
        for rule in &self.rate_rules {
            derivative[rule.species_index] += rule.amount_per_reading * rule.rate.evaluate(values, stack);
        }
    }

    /// Writes the product of the Jacobian of the time derivative (with
    /// respect to the species amounts) and `direction` into `product`, by
    /// forward-mode differentiation through the assignment rules.
    pub fn write_jacobian_product(
        &self,
        values: &[f64],
        direction: &[f64],
        scratch: &mut DualScratch,
        product: &mut [f64],
    ) {
        let DualScratch { tangents, stack } = scratch;
        tangents.clear();
        tangents.resize(values.len(), 0.0);
        tangents[..direction.len()].copy_from_slice(direction);
        for (index, rule) in self.assignment_rules.iter().enumerate() {
            tangents[self.time_slot() + 1 + index] = rule.evaluate_dual(values, tangents, stack).tangent;
        }
        product.fill(0.0);
        for (rate, changes) in self.rates.iter().zip(&self.net_changes) {
            let flux_tangent = rate.evaluate_dual(values, tangents, stack).tangent;
            for &(species, change) in changes {
                product[species] += change * flux_tangent;
            }
        }
        for rule in &self.rate_rules {
            product[rule.species_index] +=
                rule.amount_per_reading * rule.rate.evaluate_dual(values, tangents, stack).tangent;
        }
    }

    /// The compiled rate laws, one per reaction.
    pub(crate) fn rates(&self) -> &[CompiledExpression] {
        &self.rates
    }

    /// Net species changes of reaction `reaction_index`.
    pub(crate) fn net_changes(&self, reaction_index: usize) -> &[(usize, f64)] {
        &self.net_changes[reaction_index]
    }

    /// Expected value of an observable given the state in `values`.
    #[must_use]
    pub fn observation_mean(&self, observable_index: usize, values: &[f64], stack: &mut Vec<f64>) -> f64 {
        self.observables[observable_index].mean.evaluate(values, stack)
    }

    /// Log-density of `measurement` of an observable given the state in
    /// `values`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationFailure::InvalidNoise`] if the noise formula does
    /// not evaluate to a positive number.
    pub fn log_observation_density(
        &self,
        observable_index: usize,
        measurement: f64,
        values: &[f64],
        stack: &mut Vec<f64>,
    ) -> Result<f64, SimulationFailure> {
        let (mean, noise_std) = self.observation_moments(observable_index, values, stack)?;
        let standardized = (measurement - mean) / noise_std;
        Ok(-0.5 * standardized * standardized - noise_std.ln() - 0.5 * std::f64::consts::TAU.ln())
    }

    /// Draws a measurement of an observable given the state in `values`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationFailure::InvalidNoise`] if the noise formula does
    /// not evaluate to a positive number.
    pub fn sample_observation<R: Rng + ?Sized>(
        &self,
        observable_index: usize,
        values: &[f64],
        stack: &mut Vec<f64>,
        rng: &mut R,
    ) -> Result<f64, SimulationFailure> {
        let (mean, noise_std) = self.observation_moments(observable_index, values, stack)?;
        let standard_normal: f64 = StandardNormal.sample(rng);
        Ok(mean + noise_std * standard_normal)
    }

    fn time_slot(&self) -> usize {
        self.initial_amounts.len() + self.default_parameters.len()
    }

    fn observation_moments(
        &self,
        observable_index: usize,
        values: &[f64],
        stack: &mut Vec<f64>,
    ) -> Result<(f64, f64), SimulationFailure> {
        let observable = &self.observables[observable_index];
        let noise_std = observable.noise_std.evaluate(values, stack);
        if !(noise_std > 0.0 && noise_std.is_finite()) {
            return Err(SimulationFailure::InvalidNoise {
                observable_index,
                value: noise_std,
            });
        }
        Ok((observable.mean.evaluate(values, stack), noise_std))
    }
}
