//! The state of the Gibbs sampler: which point belongs to which component.
//!
//! Assignments are updated by collapsed Gibbs sampling (Neal 2000,
//! Algorithm 3): a component's mean and precision are integrated out, so a
//! point joins an existing component with probability proportional to the
//! component's size times its Student t predictive density, and opens a new
//! component with probability proportional to the concentration times the
//! prior predictive density.

use nalgebra::DMatrix;
use rand::{Rng, RngExt};

use crate::component_statistics::ComponentStatistics;
use crate::dpgmm_error::DpgmmError;
use crate::hyperparameters::Hyperparameters;
use crate::normal_wishart::{GaussianDraw, NormalWishart};

/// A component: its points' statistics and the posterior they imply.
#[derive(Debug, Clone, PartialEq)]
struct Cluster {
    statistics: ComponentStatistics,
    posterior: NormalWishart,
}

/// Assignments, components, and hyperparameters of one Markov chain.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GibbsState {
    /// Whitened points, one after the other.
    coordinates: Vec<f64>,
    dimension: usize,
    assignments: Vec<Option<usize>>,
    clusters: Vec<Cluster>,
    hyperparameters: Hyperparameters,
    prior: NormalWishart,
    log_weights: Vec<f64>,
    scratch: Vec<f64>,
}

impl GibbsState {
    /// Starts a chain with every point unassigned; `whitened_points` holds
    /// one point per column.
    pub(crate) fn new(whitened_points: &DMatrix<f64>) -> Result<Self, DpgmmError> {
        let dimension = whitened_points.nrows();
        let hyperparameters = Hyperparameters::initial(dimension);
        let prior = hyperparameters.derive_prior()?;
        Ok(Self {
            coordinates: whitened_points.as_slice().to_vec(),
            dimension,
            assignments: vec![None; whitened_points.ncols()],
            clusters: Vec::new(),
            hyperparameters,
            prior,
            log_weights: Vec::new(),
            scratch: vec![0.0; dimension],
        })
    }

    /// Fraction of the points held by each component.
    pub(crate) fn component_weights(&self) -> Vec<f64> {
        let n_points = Self::as_float(self.assignments.len());
        self.clusters
            .iter()
            .map(|cluster| f64::from(cluster.statistics.count()) / n_points)
            .collect()
    }

    /// Reassigns every point in turn.
    pub(crate) fn resample_assignments<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(), DpgmmError> {
        for point_index in 0..self.assignments.len() {
            self.detach_point(point_index)?;
            self.compute_assignment_log_weights(point_index);
            let choice = self.draw_assignment(rng);
            self.attach_point(point_index, choice)?;
        }
        Ok(())
    }

    /// Draws a mean and a precision for every component from its posterior.
    pub(crate) fn draw_components<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<GaussianDraw>, DpgmmError> {
        self.clusters
            .iter()
            .map(|cluster| cluster.posterior.sample(rng))
            .collect()
    }

    /// Redraws the hyperparameters given the component draws and updates
    /// every posterior to the new prior.
    pub(crate) fn resample_hyperparameters<R: Rng + ?Sized>(
        &mut self,
        components: &[GaussianDraw],
        rng: &mut R,
    ) -> Result<(), DpgmmError> {
        let n_points = Self::as_float(self.assignments.len());
        let n_components = Self::as_float(self.clusters.len());
        self.hyperparameters.resample_concentration(n_points, n_components, rng);
        self.hyperparameters.resample_component_prior(components, rng)?;
        self.prior = self.hyperparameters.derive_prior()?;
        for cluster in &mut self.clusters {
            cluster.posterior = cluster.statistics.derive_posterior(&self.hyperparameters)?;
        }
        Ok(())
    }

    /// Removes a point from its component, and the component if it is left
    /// empty.
    fn detach_point(&mut self, point_index: usize) -> Result<(), DpgmmError> {
        let Some(cluster_index) = self.assignments[point_index].take() else {
            return Ok(());
        };
        let point = &self.coordinates[point_index * self.dimension..(point_index + 1) * self.dimension];
        let cluster = &mut self.clusters[cluster_index];
        cluster.statistics.remove_point(point);
        if cluster.statistics.count() > 0 {
            cluster.posterior = cluster.statistics.derive_posterior(&self.hyperparameters)?;
            return Ok(());
        }
        self.clusters.remove(cluster_index);
        for assigned in self
            .assignments
            .iter_mut()
            .flatten()
            .filter(|assigned| **assigned > cluster_index)
        {
            *assigned -= 1;
        }
        Ok(())
    }

    /// Adds a point to component `cluster_index`; an index one past the
    /// last component opens a new one.
    fn attach_point(&mut self, point_index: usize, cluster_index: usize) -> Result<(), DpgmmError> {
        let point = &self.coordinates[point_index * self.dimension..(point_index + 1) * self.dimension];
        if cluster_index == self.clusters.len() {
            let statistics = ComponentStatistics::empty(self.dimension);
            self.clusters.push(Cluster {
                statistics,
                posterior: self.prior.clone(),
            });
        }
        let cluster = &mut self.clusters[cluster_index];
        cluster.statistics.add_point(point);
        cluster.posterior = cluster.statistics.derive_posterior(&self.hyperparameters)?;
        self.assignments[point_index] = Some(cluster_index);
        Ok(())
    }

    /// Fills `log_weights` with the unnormalized log-probabilities of
    /// joining each component and, last, of opening a new one.
    fn compute_assignment_log_weights(&mut self, point_index: usize) {
        let point = &self.coordinates[point_index * self.dimension..(point_index + 1) * self.dimension];
        self.log_weights.clear();
        for cluster in &self.clusters {
            let log_size = f64::from(cluster.statistics.count()).ln();
            self.log_weights
                .push(log_size + cluster.posterior.log_predictive(point, &mut self.scratch));
        }
        let log_concentration = self.hyperparameters.concentration().ln();
        self.log_weights
            .push(log_concentration + self.prior.log_predictive(point, &mut self.scratch));
    }

    /// Draws an index with probability proportional to `exp(log_weights)`.
    fn draw_assignment<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let largest = self.log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = self
            .log_weights
            .iter()
            .map(|log_weight| (log_weight - largest).exp())
            .sum();
        let mut remaining = rng.random::<f64>() * total;
        for (index, log_weight) in self.log_weights.iter().enumerate() {
            let weight = (log_weight - largest).exp();
            if remaining < weight {
                return index;
            }
            remaining -= weight;
        }
        // Rounding can leave a sliver beyond the last weight.
        self.log_weights.len() - 1
    }

    /// Converts a small count; counts here are far below 2^53, where the
    /// conversion is exact.
    #[allow(clippy::cast_precision_loss)]
    fn as_float(count: usize) -> f64 {
        count as f64
    }
}
