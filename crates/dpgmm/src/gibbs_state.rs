//! The state of the Gibbs sampler: which point belongs to which component.
//!
//! Assignments are updated by collapsed Gibbs sampling (Neal 2000,
//! Algorithm 3): a component's mean and precision are integrated out, so a
//! point joins an existing component with probability proportional to the
//! component's size times its Student t predictive density, and opens a new
//! component with probability proportional to the concentration times the
//! prior predictive density.
//!
//! Moving one point changes its component's posterior by a rank-one term.
//! Two consequences keep a sweep cheap:
//!
//! - the density of a point under its own component *without it* follows
//!   from the component as it is, so a point that stays where it was (the
//!   usual case once the chain has settled) costs no update at all;
//! - when a point does move, the two posteriors involved are updated in
//!   place in `O(d^2)`.
//!
//! The posteriors are derived afresh from the components' statistics once
//! per sweep, when the hyperparameters are redrawn, which also keeps
//! rounding from accumulating.

use nalgebra::DMatrix;
use rand::{Rng, RngExt};

use crate::component_statistics::ComponentStatistics;
use crate::dpgmm_error::DpgmmError;
use crate::gamma_ratio_table::GammaRatioTable;
use crate::hyperparameters::Hyperparameters;
use crate::normal_wishart::{GaussianDraw, NormalWishart};

/// What a component's posterior update needs besides the point: the prior
/// to fall back on, the log-gamma table, and scratch space.
struct UpdateContext<'a> {
    hyperparameters: &'a Hyperparameters,
    gamma_ratios: &'a mut GammaRatioTable,
    scratch: &'a mut [f64],
}

/// A component: its points' statistics and the posterior they imply.
#[derive(Debug, Clone, PartialEq)]
struct Cluster {
    statistics: ComponentStatistics,
    posterior: NormalWishart,
}

impl Cluster {
    /// Adds `point` to the component and conditions its posterior on it,
    /// in `O(d^2)`.
    fn admit(&mut self, point: &[f64], context: &mut UpdateContext) -> Result<(), DpgmmError> {
        self.statistics.add_point(point);
        let ratio = context.gamma_ratios.for_count(self.statistics.count());
        let update = self.posterior.absorb_observation(point, context.scratch, ratio);
        self.settle_posterior(update, context.hyperparameters)
    }

    /// Removes `point`, one of the component's points, and its influence
    /// on the posterior, in `O(d^2)`. The component must keep a point.
    fn dismiss(&mut self, point: &[f64], context: &mut UpdateContext) -> Result<(), DpgmmError> {
        self.statistics.remove_point(point);
        let ratio = context.gamma_ratios.for_count(self.statistics.count());
        let update = self.posterior.release_observation(point, context.scratch, ratio);
        self.settle_posterior(update, context.hyperparameters)
    }

    /// Log-weight of `point`, one of the component's points, rejoining the
    /// component: the logarithm of the number of its other points (read from
    /// `log_counts`) plus the predictive density of the posterior without
    /// the point. `None` for a component that consists of the point alone.
    fn log_weight_of_rejoining(
        &self,
        point: &[f64],
        log_counts: &[f64],
        context: &mut UpdateContext,
    ) -> Result<Option<f64>, DpgmmError> {
        let n_others = self.statistics.count() - 1;
        if n_others == 0 {
            return Ok(None);
        }
        let ratio = context.gamma_ratios.for_count(n_others);
        let log_density = match self.posterior.log_predictive_without(point, context.scratch, ratio) {
            Some(log_density) => log_density,
            None => self.log_predictive_after_removal(point, context)?,
        };
        Ok(Some(log_counts[n_others as usize] + log_density))
    }

    /// The same density from the reduced posterior derived afresh, for the
    /// rare case that rounding defeats the shortcut.
    fn log_predictive_after_removal(&self, point: &[f64], context: &mut UpdateContext) -> Result<f64, DpgmmError> {
        let mut reduced = self.statistics.clone();
        reduced.remove_point(point);
        Ok(reduced
            .derive_posterior(context.hyperparameters)?
            .log_predictive(point, context.scratch))
    }

    /// Accepts an in-place update of the posterior, or, where rounding made
    /// it break down, derives the posterior afresh from the statistics,
    /// which define it exactly.
    fn settle_posterior(
        &mut self,
        update: Result<(), DpgmmError>,
        hyperparameters: &Hyperparameters,
    ) -> Result<(), DpgmmError> {
        match update {
            Err(DpgmmError::NumericalBreakdown { .. }) => {
                self.posterior = self.statistics.derive_posterior(hyperparameters)?;
                Ok(())
            }
            other => other,
        }
    }
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
    gamma_ratios: GammaRatioTable,
    /// `ln k` for every possible component size `k`.
    log_counts: Vec<f64>,
    /// Scratch: the assignment weights of the point being reassigned.
    weights: Vec<f64>,
    scratch: Vec<f64>,
}

impl GibbsState {
    /// Starts a chain with every point unassigned; `whitened_points` holds
    /// one point per column.
    pub(crate) fn new(whitened_points: &DMatrix<f64>) -> Result<Self, DpgmmError> {
        let (dimension, n_points) = (whitened_points.nrows(), whitened_points.ncols());
        let hyperparameters = Hyperparameters::initial(dimension);
        let prior = hyperparameters.derive_prior()?;
        Ok(Self {
            coordinates: whitened_points.as_slice().to_vec(),
            dimension,
            assignments: vec![None; n_points],
            clusters: Vec::new(),
            gamma_ratios: GammaRatioTable::new(dimension, hyperparameters.dof(), n_points),
            hyperparameters,
            prior,
            log_counts: (0..=n_points).map(|count| Self::as_float(count).ln()).collect(),
            weights: Vec::new(),
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
            self.compute_assignment_log_weights(point_index)?;
            let choice = self.draw_assignment(rng);
            self.move_point(point_index, choice)?;
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
        self.gamma_ratios = GammaRatioTable::new(self.dimension, self.hyperparameters.dof(), self.assignments.len());
        for cluster in &mut self.clusters {
            cluster.posterior = cluster.statistics.derive_posterior(&self.hyperparameters)?;
        }
        Ok(())
    }

    /// Fills `weights` with the unnormalized log-probabilities of the
    /// point joining each component and, last, of opening a new one. The
    /// point's own component is weighed as if the point had been taken out
    /// of it; if the point is its only member, it cannot be rejoined and
    /// gets the weight zero.
    fn compute_assignment_log_weights(&mut self, point_index: usize) -> Result<(), DpgmmError> {
        let point = &self.coordinates[point_index * self.dimension..(point_index + 1) * self.dimension];
        let own_cluster = self.assignments[point_index];
        let mut context = UpdateContext {
            hyperparameters: &self.hyperparameters,
            gamma_ratios: &mut self.gamma_ratios,
            scratch: &mut self.scratch,
        };
        self.weights.clear();
        for (cluster_index, cluster) in self.clusters.iter().enumerate() {
            let log_weight = if own_cluster == Some(cluster_index) {
                let rejoining = cluster.log_weight_of_rejoining(point, &self.log_counts, &mut context)?;
                rejoining.unwrap_or(f64::NEG_INFINITY)
            } else {
                let log_count = self.log_counts[cluster.statistics.count() as usize];
                log_count + cluster.posterior.log_predictive(point, context.scratch)
            };
            self.weights.push(log_weight);
        }
        let log_concentration = self.hyperparameters.concentration().ln();
        let log_new_component = log_concentration + self.prior.log_predictive(point, context.scratch);
        self.weights.push(log_new_component);
        Ok(())
    }

    /// Draws an index with probability proportional to the exponential of
    /// the log-weights in `weights`, which it overwrites with the weights
    /// relative to the largest.
    fn draw_assignment<R: Rng + ?Sized>(&mut self, rng: &mut R) -> usize {
        let largest = self.weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for weight in &mut self.weights {
            *weight = (*weight - largest).exp();
            total += *weight;
        }
        let mut remaining = rng.random::<f64>() * total;
        for (index, weight) in self.weights.iter().enumerate() {
            if remaining < *weight {
                return index;
            }
            remaining -= weight;
        }
        // Rounding can leave a sliver beyond the last weight.
        self.weights.len() - 1
    }

    /// Carries out the drawn assignment: `choice` is the index of a
    /// component, or one past the last for a new one. Nothing changes if
    /// the point stays in its component, or if a point that is alone in
    /// its component draws a new one, which would be the same component.
    fn move_point(&mut self, point_index: usize, choice: usize) -> Result<(), DpgmmError> {
        let Some(own_cluster) = self.assignments[point_index] else {
            return self.attach_point(point_index, choice);
        };
        let is_alone = self.clusters[own_cluster].statistics.count() == 1;
        if choice == own_cluster || (is_alone && choice == self.clusters.len()) {
            return Ok(());
        }
        if is_alone {
            self.remove_cluster(own_cluster);
            return self.attach_point(point_index, if choice > own_cluster { choice - 1 } else { choice });
        }
        self.detach_point(point_index, own_cluster)?;
        self.attach_point(point_index, choice)
    }

    /// Removes a component and renumbers the assignments to those after it.
    fn remove_cluster(&mut self, cluster_index: usize) {
        self.clusters.remove(cluster_index);
        for assigned in self
            .assignments
            .iter_mut()
            .flatten()
            .filter(|assigned| **assigned > cluster_index)
        {
            *assigned -= 1;
        }
    }

    /// Takes a point out of a component that keeps other points.
    fn detach_point(&mut self, point_index: usize, cluster_index: usize) -> Result<(), DpgmmError> {
        let point = &self.coordinates[point_index * self.dimension..(point_index + 1) * self.dimension];
        let mut context = UpdateContext {
            hyperparameters: &self.hyperparameters,
            gamma_ratios: &mut self.gamma_ratios,
            scratch: &mut self.scratch,
        };
        self.clusters[cluster_index].dismiss(point, &mut context)
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
        let mut context = UpdateContext {
            hyperparameters: &self.hyperparameters,
            gamma_ratios: &mut self.gamma_ratios,
            scratch: &mut self.scratch,
        };
        self.clusters[cluster_index].admit(point, &mut context)?;
        self.assignments[point_index] = Some(cluster_index);
        Ok(())
    }

    /// Converts a small count; counts here are far below 2^53, where the
    /// conversion is exact.
    #[allow(clippy::cast_precision_loss)]
    fn as_float(count: usize) -> f64 {
        count as f64
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    use super::*;

    /// Two clusters of whitened-scale points in three dimensions.
    fn state_after_sweeps(n_sweeps: usize) -> Result<GibbsState, DpgmmError> {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (n_points, dimension) = (120, 3);
        let points = DMatrix::from_fn(dimension, n_points, |_, column| {
            let offset = if column % 3 == 0 { 2.5 } else { -1.0 };
            let noise: f64 = StandardNormal.sample(&mut rng);
            offset + 0.4 * noise
        });
        let mut state = GibbsState::new(&points)?;
        state.resample_assignments(&mut rng)?;
        for _ in 1..n_sweeps {
            let components = state.draw_components(&mut rng)?;
            state.resample_hyperparameters(&components, &mut rng)?;
            state.resample_assignments(&mut rng)?;
        }
        Ok(state)
    }

    #[test]
    fn posteriors_updated_point_by_point_equal_those_derived_from_the_statistics() -> Result<(), DpgmmError> {
        // After a sweep of in-place updates and skipped updates, every
        // component's posterior must still be the one its points define.
        let state = state_after_sweeps(8)?;
        let probe = [0.3, -0.7, 1.1];
        for cluster in &state.clusters {
            let exact = cluster.statistics.derive_posterior(&state.hyperparameters)?;
            let kept = cluster.posterior.log_predictive(&probe, &mut [0.0; 3]);
            assert!((kept - exact.log_predictive(&probe, &mut [0.0; 3])).abs() < 1e-9);
        }
        Ok(())
    }

    #[test]
    fn every_point_is_counted_in_exactly_the_component_it_is_assigned_to() -> Result<(), DpgmmError> {
        let state = state_after_sweeps(8)?;
        let mut counts = vec![0_u32; state.clusters.len()];
        for assignment in &state.assignments {
            counts[assignment.expect("every point is assigned after a sweep")] += 1;
        }
        let recorded: Vec<u32> = state
            .clusters
            .iter()
            .map(|cluster| cluster.statistics.count())
            .collect();
        assert_eq!(counts, recorded);
        assert!(recorded.iter().all(|count| *count > 0), "no component is left empty");
        Ok(())
    }
}
