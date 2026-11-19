//! A reaction network and its observables, compiled for simulation.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::compiled_expression::CompiledExpression;
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
    /// Expected measurement as a function of species and parameters.
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

/// A model ready to be simulated.
///
/// All formulas are evaluated over a *value buffer*: the species amounts
/// followed by the parameters. A simulation advances the species part of
/// such a buffer in place.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationModel {
    initial_amounts: Vec<f64>,
    default_parameters: Vec<f64>,
    rates: Vec<CompiledExpression>,
    /// Per reaction, the non-zero derivatives of its rate with respect to
    /// the species amounts, as `(species index, derivative)`.
    rate_derivatives: Vec<Vec<(usize, CompiledExpression)>>,
    net_changes: Vec<Vec<(usize, f64)>>,
    observables: Vec<Observable>,
}

impl SimulationModel {
    /// Compiles a network and its observables.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownSymbol`] if a formula refers to a name
    /// that is neither a species nor a parameter of `network`.
    pub fn new(network: &ReactionNetwork, observables: &[ObservableDefinition]) -> Result<Self, ModelError> {
        let compile_observable = |definition: &ObservableDefinition| -> Result<Observable, ModelError> {
            let id = &definition.id;
            let mean = network.compile_expression(&definition.formula, &format!("the formula of observable `{id}`"))?;
            let noise_context = format!("the noise formula of observable `{id}`");
            let noise_std = network.compile_expression(&definition.noise_formula, &noise_context)?;
            Ok(Observable { mean, noise_std })
        };
        Ok(Self {
            initial_amounts: network.species().iter().map(|species| species.value).collect(),
            default_parameters: network.parameters().iter().map(|parameter| parameter.value).collect(),
            rates: network.compile_rates()?,
            rate_derivatives: Self::compile_rate_derivatives(network)?,
            net_changes: network
                .reactions()
                .iter()
                .map(|reaction| reaction.net_changes.clone())
                .collect(),
            observables: observables.iter().map(compile_observable).collect::<Result<_, _>>()?,
        })
    }

    /// Differentiates every rate law with respect to every species and
    /// keeps the derivatives that do not vanish identically.
    fn compile_rate_derivatives(
        network: &ReactionNetwork,
    ) -> Result<Vec<Vec<(usize, CompiledExpression)>>, ModelError> {
        let mut all = Vec::new();
        for reaction in network.reactions() {
            let mut derivatives = Vec::new();
            for (index, species) in network.species().iter().enumerate() {
                let derivative = reaction.rate.differentiate(&species.id);
                if !derivative.is_zero() {
                    let context = format!("the derivative of the kinetic law of reaction `{}`", reaction.id);
                    derivatives.push((index, network.compile_expression(&derivative, &context)?));
                }
            }
            all.push(derivatives);
        }
        Ok(all)
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

    /// Length of a value buffer: species plus parameters.
    #[must_use]
    pub fn value_width(&self) -> usize {
        self.initial_amounts.len() + self.default_parameters.len()
    }

    /// Default values of all parameters, in the order of the network.
    #[must_use]
    pub fn default_parameters(&self) -> &[f64] {
        &self.default_parameters
    }

    /// Writes the initial species amounts and `parameters` into `values`.
    pub fn write_initial_values(&self, parameters: &[f64], values: &mut [f64]) {
        let (amounts, parameter_slots) = values.split_at_mut(self.initial_amounts.len());
        amounts.copy_from_slice(&self.initial_amounts);
        parameter_slots.copy_from_slice(parameters);
    }

    /// The compiled rate laws, one per reaction.
    pub(crate) fn rates(&self) -> &[CompiledExpression] {
        &self.rates
    }

    /// Net species changes of reaction `reaction_index`.
    pub(crate) fn net_changes(&self, reaction_index: usize) -> &[(usize, f64)] {
        &self.net_changes[reaction_index]
    }

    /// Writes the time derivative of the species amounts in `values` into
    /// `derivative`: each reaction's rate times its net changes.
    pub fn write_time_derivative(&self, values: &[f64], stack: &mut Vec<f64>, derivative: &mut [f64]) {
        derivative.fill(0.0);
        for (rate, changes) in self.rates.iter().zip(&self.net_changes) {
            let flux = rate.evaluate(values, stack);
            for &(species, change) in changes {
                derivative[species] += change * flux;
            }
        }
    }

    /// Writes the product of the Jacobian of the time derivative (with
    /// respect to the species amounts) and `direction` into `product`.
    pub fn write_jacobian_product(&self, values: &[f64], direction: &[f64], stack: &mut Vec<f64>, product: &mut [f64]) {
        product.fill(0.0);
        for (derivatives, changes) in self.rate_derivatives.iter().zip(&self.net_changes) {
            let flux_change: f64 = derivatives
                .iter()
                .map(|(species, d)| d.evaluate(values, stack) * direction[*species])
                .sum();
            for &(species, change) in changes {
                product[species] += change * flux_change;
            }
        }
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
