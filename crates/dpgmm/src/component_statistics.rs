//! Sufficient statistics of the points assigned to one mixture component.

use nalgebra::{DMatrix, DVector, DVectorView};

use crate::dpgmm_error::DpgmmError;
use crate::hyperparameters::Hyperparameters;
use crate::normal_wishart::NormalWishart;

/// Count, sum, and sum of outer products of a set of points.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ComponentStatistics {
    count: u32,
    sum: DVector<f64>,
    sum_of_outer_products: DMatrix<f64>,
}

impl ComponentStatistics {
    pub(crate) fn empty(dimension: usize) -> Self {
        Self {
            count: 0,
            sum: DVector::zeros(dimension),
            sum_of_outer_products: DMatrix::zeros(dimension, dimension),
        }
    }

    /// Number of points in the set.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    pub(crate) fn add_point(&mut self, point: &[f64]) {
        let point = DVectorView::from_slice(point, point.len());
        self.count += 1;
        self.sum += point;
        self.sum_of_outer_products.ger(1.0, &point, &point, 1.0);
    }

    pub(crate) fn remove_point(&mut self, point: &[f64]) {
        let point = DVectorView::from_slice(point, point.len());
        self.count -= 1;
        self.sum -= point;
        self.sum_of_outer_products.ger(-1.0, &point, &point, 1.0);
    }

    /// Posterior over the component's mean and precision given its points
    /// and the Normal-Wishart prior defined by `hyperparameters`.
    pub(crate) fn derive_posterior(&self, hyperparameters: &Hyperparameters) -> Result<NormalWishart, DpgmmError> {
        if self.count == 0 {
            return hyperparameters.derive_prior();
        }
        let count = f64::from(self.count);
        let strength = hyperparameters.mean_strength();
        let sample_mean = &self.sum / count;
        let scatter = &self.sum_of_outer_products - count * &sample_mean * sample_mean.transpose();
        let offset = &sample_mean - hyperparameters.mean();
        let offset_weight = strength * count / (strength + count);
        let inverse_scale =
            hyperparameters.dof() * hyperparameters.scale() + scatter + offset_weight * &offset * offset.transpose();
        let mean = (strength * hyperparameters.mean() + &self.sum) / (strength + count);
        NormalWishart::new(mean, strength + count, hyperparameters.dof() + count, inverse_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_point_undoes_adding_it() {
        let mut statistics = ComponentStatistics::empty(2);
        statistics.add_point(&[1.0, 2.0]);
        let before = statistics.clone();
        statistics.add_point(&[-3.0, 0.5]);
        statistics.remove_point(&[-3.0, 0.5]);
        assert_eq!(statistics, before);
    }

    #[test]
    fn posterior_of_no_points_is_the_prior() -> Result<(), DpgmmError> {
        let hyperparameters = Hyperparameters::initial(3);
        let posterior = ComponentStatistics::empty(3).derive_posterior(&hyperparameters)?;
        assert_eq!(posterior, hyperparameters.derive_prior()?);
        Ok(())
    }

    #[test]
    fn posterior_does_not_depend_on_the_order_of_the_points() -> Result<(), DpgmmError> {
        let hyperparameters = Hyperparameters::initial(2);
        let points = [[0.5, 1.0], [-1.0, 0.25], [2.0, -0.5]];
        let (mut forward, mut backward) = (ComponentStatistics::empty(2), ComponentStatistics::empty(2));
        for point in &points {
            forward.add_point(point);
        }
        for point in points.iter().rev() {
            backward.add_point(point);
        }
        let probe = [0.3, 0.3];
        let forward_density = forward
            .derive_posterior(&hyperparameters)?
            .log_predictive(&probe, &mut [0.0; 2]);
        let backward_density = backward
            .derive_posterior(&hyperparameters)?
            .log_predictive(&probe, &mut [0.0; 2]);
        assert!((forward_density - backward_density).abs() < 1e-12);
        Ok(())
    }
}
