//! Exact stochastic simulation of a reaction network (Gillespie's direct
//! method).

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Exp1};

use crate::simulation_failure::SimulationFailure;
use crate::simulation_model::SimulationModel;

/// Buffers a simulation reuses between steps.
#[derive(Debug, Clone, Default)]
pub struct SimulationScratch {
    propensities: Vec<f64>,
    stack: Vec<f64>,
}

impl SimulationScratch {
    /// The operand stack, for evaluating other formulas of the same model.
    pub fn stack(&mut self) -> &mut Vec<f64> {
        &mut self.stack
    }
}

/// Simulates the Markov jump process defined by a model's rate laws.
#[derive(Debug, Clone, Copy)]
pub struct StochasticSimulator<'a> {
    model: &'a SimulationModel,
    max_events: u64,
}

impl<'a> StochasticSimulator<'a> {
    /// Default bound on the reaction events between two time points.
    pub const DEFAULT_MAX_EVENTS: u64 = 10_000_000;

    /// Creates a simulator for `model`.
    #[must_use]
    pub fn new(model: &'a SimulationModel) -> Self {
        Self {
            model,
            max_events: Self::DEFAULT_MAX_EVENTS,
        }
    }

    /// The model this simulator advances.
    #[must_use]
    pub fn model(&self) -> &'a SimulationModel {
        self.model
    }

    /// Returns a copy that gives up after `max_events` reaction events
    /// between two time points.
    #[must_use]
    pub fn with_max_events(self, max_events: u64) -> Self {
        Self { max_events, ..self }
    }

    /// Advances the species amounts in `values` from time `from` to time
    /// `to`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationFailure::InvalidRate`] if a rate law is negative
    /// or `NaN`, and [`SimulationFailure::EventLimitExceeded`] if the event
    /// limit is reached before `to`.
    pub fn advance<R: Rng + ?Sized>(
        &self,
        values: &mut [f64],
        (from, to): (f64, f64),
        scratch: &mut SimulationScratch,
        rng: &mut R,
    ) -> Result<(), SimulationFailure> {
        let mut time = from;
        for _ in 0..self.max_events {
            let total = self.compute_propensities(values, scratch)?;
            if total <= 0.0 {
                return Ok(());
            }
            let waiting_time: f64 = Exp1.sample(rng);
            time += waiting_time / total;
            if time >= to {
                return Ok(());
            }
            let reaction_index = Self::choose_reaction(&scratch.propensities, total, rng);
            for &(species, change) in self.model.net_changes(reaction_index) {
                values[species] += change;
            }
        }
        Err(SimulationFailure::EventLimitExceeded { limit: self.max_events })
    }

    /// Evaluates every propensity into the scratch buffer and returns their
    /// sum.
    fn compute_propensities(&self, values: &[f64], scratch: &mut SimulationScratch) -> Result<f64, SimulationFailure> {
        scratch.propensities.clear();
        let mut total = 0.0;
        for (reaction_index, rate) in self.model.rates().iter().enumerate() {
            let value = rate.evaluate(values, &mut scratch.stack);
            if value.is_nan() || value < 0.0 {
                return Err(SimulationFailure::InvalidRate { reaction_index, value });
            }
            scratch.propensities.push(value);
            total += value;
        }
        Ok(total)
    }

    /// Picks a reaction with probability proportional to its propensity.
    fn choose_reaction<R: Rng + ?Sized>(propensities: &[f64], total: f64, rng: &mut R) -> usize {
        let mut remaining = rng.random::<f64>() * total;
        for (index, propensity) in propensities.iter().enumerate() {
            if remaining < *propensity {
                return index;
            }
            remaining -= propensity;
        }
        // Rounding can leave a sliver beyond the last propensity; give it to
        // the last reaction that can fire.
        propensities
            .iter()
            .rposition(|propensity| *propensity > 0.0)
            .unwrap_or(0)
    }
}
