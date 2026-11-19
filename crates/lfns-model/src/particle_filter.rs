//! Bootstrap particle filter estimate of a trajectory's likelihood.
//!
//! `H` particles are simulated from one measurement time to the next. At
//! each time the average over particles of the measurement density
//! estimates `p(y_t | y_1..t-1)`, and the particles are resampled in
//! proportion to that density. The product of the averages is an unbiased
//! estimate of the likelihood of the whole trajectory.

use rand::{Rng, RngExt};

use crate::model_error::ModelError;
use crate::simulation_failure::SimulationFailure;
use crate::simulation_model::SimulationModel;
use crate::stochastic_simulator::{SimulationScratch, StochasticSimulator};
use crate::trajectory::{Reading, Trajectory};

/// The particles of one filter run: `n_particles` value buffers laid end to
/// end, plus a second set used as the target of resampling.
struct ParticleCloud {
    width: usize,
    values: Vec<f64>,
    resampled: Vec<f64>,
    log_weights: Vec<f64>,
    cumulative_weights: Vec<f64>,
    scratch: SimulationScratch,
}

/// A bootstrap particle filter with a fixed number of particles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticleFilter {
    n_particles: usize,
}

impl ParticleFilter {
    /// Creates a filter with `n_particles` particles.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnsupportedPetab`] if `n_particles` is zero.
    pub fn new(n_particles: usize) -> Result<Self, ModelError> {
        if n_particles == 0 {
            return Err(ModelError::UnsupportedPetab {
                feature: "a particle filter with zero particles".to_owned(),
                remedy: "Use at least one particle.".to_owned(),
            });
        }
        Ok(Self { n_particles })
    }

    /// Number of particles.
    #[must_use]
    pub fn n_particles(&self) -> usize {
        self.n_particles
    }

    /// Returns the logarithm of one unbiased estimate of the likelihood of
    /// `trajectory` under `model`, every particle starting from the species
    /// amounts and parameters in `initial_values`. The model must be a pure
    /// reaction network.
    ///
    /// # Errors
    ///
    /// Returns a [`SimulationFailure`] if the model cannot be simulated or
    /// an observation density cannot be evaluated.
    pub fn estimate_log_likelihood<R: Rng + ?Sized>(
        &self,
        model: &SimulationModel,
        initial_values: &[f64],
        trajectory: &Trajectory,
        rng: &mut R,
    ) -> Result<f64, SimulationFailure> {
        let simulator = StochasticSimulator::new(model);
        let mut cloud = ParticleCloud::new(initial_values, self.n_particles);
        let (mut time, mut log_likelihood) = (0.0, 0.0);
        for (index, (next_time, readings)) in trajectory.time_points().iter().enumerate() {
            cloud.advance_and_weigh(&simulator, (time, *next_time), readings, rng)?;
            log_likelihood += cloud.log_mean_weight();
            if log_likelihood == f64::NEG_INFINITY {
                return Ok(log_likelihood);
            }
            let is_last = index + 1 == trajectory.time_points().len();
            if self.n_particles > 1 && !is_last {
                cloud.resample(rng);
            }
            time = *next_time;
        }
        Ok(log_likelihood)
    }
}

impl ParticleCloud {
    fn new(initial_values: &[f64], n_particles: usize) -> Self {
        let values = initial_values.repeat(n_particles);
        Self {
            width: initial_values.len(),
            resampled: values.clone(),
            values,
            log_weights: vec![0.0; n_particles],
            cumulative_weights: vec![0.0; n_particles],
            scratch: SimulationScratch::default(),
        }
    }

    /// Simulates every particle over `interval` and sets its log-weight to
    /// the log-density of `readings` given its state.
    fn advance_and_weigh<R: Rng + ?Sized>(
        &mut self,
        simulator: &StochasticSimulator,
        interval: (f64, f64),
        readings: &[Reading],
        rng: &mut R,
    ) -> Result<(), SimulationFailure> {
        for (particle, log_weight) in self.values.chunks_exact_mut(self.width).zip(&mut self.log_weights) {
            simulator.advance(particle, interval, &mut self.scratch, rng)?;
            *log_weight = 0.0;
            for reading in readings {
                let stack = self.scratch.stack();
                *log_weight += simulator.model().log_observation_density(
                    reading.observable_index,
                    reading.measurement,
                    particle,
                    stack,
                )?;
            }
        }
        Ok(())
    }

    /// `ln` of the average weight over the particles.
    fn log_mean_weight(&self) -> f64 {
        let largest = self.log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if largest == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        let scaled_sum: f64 = self
            .log_weights
            .iter()
            .map(|log_weight| (log_weight - largest).exp())
            .sum();
        // The particle count is far below 2^53, where the conversion is exact.
        #[allow(clippy::cast_precision_loss)]
        let n_particles = self.log_weights.len() as f64;
        largest + (scaled_sum / n_particles).ln()
    }

    /// Replaces the particles by an equally sized multinomial sample of
    /// them, drawn in proportion to their weights.
    fn resample<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let largest = self.log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for (cumulative, log_weight) in self.cumulative_weights.iter_mut().zip(&self.log_weights) {
            total += (log_weight - largest).exp();
            *cumulative = total;
        }
        let last = self.log_weights.len() - 1;
        for target in self.resampled.chunks_exact_mut(self.width) {
            let draw = rng.random::<f64>() * total;
            let ancestor = self
                .cumulative_weights
                .partition_point(|cumulative| *cumulative <= draw)
                .min(last);
            target.copy_from_slice(&self.values[ancestor * self.width..(ancestor + 1) * self.width]);
        }
        std::mem::swap(&mut self.values, &mut self.resampled);
    }
}
