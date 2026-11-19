//! The likelihood of a PEtab problem as seen by the nested sampler.

use lfns::LikelihoodEstimator;
use rand::Rng;

use crate::deterministic_simulator::{DeterministicSimulator, OdeSettings};
use crate::particle_filter::ParticleFilter;
use crate::petab_problem::PetabProblem;
use crate::simulation_failure::SimulationFailure;
use crate::trajectory::Trajectory;

/// How the model is simulated and its likelihood obtained.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LikelihoodMethod {
    /// Stochastic simulation; the likelihood is estimated by a particle
    /// filter and differs from call to call.
    ParticleFilter(ParticleFilter),
    /// Deterministic simulation of the ODE; the likelihood is exact.
    Deterministic(OdeSettings),
}

/// The likelihood of a [`PetabProblem`].
///
/// A point of the unit cube is mapped to parameter values through the
/// priors of the problem. The log-likelihoods of independent trajectories
/// add up.
#[derive(Debug, Clone, PartialEq)]
pub struct PetabLikelihood {
    problem: PetabProblem,
    method: LikelihoodMethod,
}

impl PetabLikelihood {
    /// Creates the likelihood of `problem`, obtained with `method`.
    #[must_use]
    pub fn new(problem: PetabProblem, method: LikelihoodMethod) -> Self {
        Self { problem, method }
    }

    /// The problem whose likelihood this is.
    #[must_use]
    pub fn problem(&self) -> &PetabProblem {
        &self.problem
    }

    /// Returns the log-likelihood (deterministic model) or the logarithm of
    /// one likelihood estimate (stochastic model) for the given values of
    /// all model parameters.
    ///
    /// # Errors
    ///
    /// Returns a [`SimulationFailure`] if the model cannot be simulated.
    pub fn estimate_for_parameters<R: Rng + ?Sized>(
        &self,
        parameters: &[f64],
        rng: &mut R,
    ) -> Result<f64, SimulationFailure> {
        let mut log_likelihood = 0.0;
        for trajectory in self.problem.trajectories() {
            log_likelihood += match self.method {
                LikelihoodMethod::ParticleFilter(filter) => {
                    filter.estimate_log_likelihood(self.problem.model(), parameters, trajectory, rng)?
                }
                LikelihoodMethod::Deterministic(settings) => {
                    self.deterministic_log_likelihood(parameters, trajectory, settings)?
                }
            };
        }
        Ok(log_likelihood)
    }

    /// Solves the ODE at the measurement times of `trajectory` and adds up
    /// the log-densities of its readings.
    fn deterministic_log_likelihood(
        &self,
        parameters: &[f64],
        trajectory: &Trajectory,
        settings: OdeSettings,
    ) -> Result<f64, SimulationFailure> {
        let model = self.problem.model();
        let times: Vec<f64> = trajectory.time_points().iter().map(|(time, _)| *time).collect();
        let states = DeterministicSimulator::new(model, settings).solve(parameters, &times)?;
        let (mut values, mut stack) = (vec![0.0; model.value_width()], Vec::new());
        model.write_initial_values(parameters, &mut values);
        let mut log_likelihood = 0.0;
        for (state, (_, readings)) in states.iter().zip(trajectory.time_points()) {
            values[..state.len()].copy_from_slice(state);
            for reading in readings {
                log_likelihood += model.log_observation_density(
                    reading.observable_index,
                    reading.measurement,
                    &values,
                    &mut stack,
                )?;
            }
        }
        Ok(log_likelihood)
    }
}

impl LikelihoodEstimator for PetabLikelihood {
    type Failure = SimulationFailure;

    fn n_parameters(&self) -> usize {
        self.problem.parameter_space().estimated().len()
    }

    fn estimate_log_likelihood<R: Rng + ?Sized>(
        &self,
        unit_point: &[f64],
        rng: &mut R,
    ) -> Result<f64, SimulationFailure> {
        let parameters = self.problem.parameter_space().model_parameters(unit_point);
        self.estimate_for_parameters(&parameters, rng)
    }
}
