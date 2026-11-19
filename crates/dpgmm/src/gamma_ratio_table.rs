//! Cached log-gamma terms of the predictive density.

use crate::normal_wishart::NormalWishart;

/// The log-gamma ratio in the normalizer of a component's predictive
/// density, tabulated by the number of points of the component.
///
/// The ratio depends on the component only through its degrees of freedom,
/// which are the prior's plus the number of its points. The Gibbs sampler
/// needs it for the same few counts over and over, and evaluating the
/// log-gamma function is slow next to the arithmetic around it, so each
/// value is computed once per sweep.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GammaRatioTable {
    dimension: usize,
    prior_dof: f64,
    /// Ratio by count; `NaN` where it has not been computed yet.
    ratios: Vec<f64>,
}

impl GammaRatioTable {
    /// Creates an empty table for components of at most `max_count` points
    /// under a prior with `prior_dof` degrees of freedom.
    pub(crate) fn new(dimension: usize, prior_dof: f64, max_count: usize) -> Self {
        Self {
            dimension,
            prior_dof,
            ratios: vec![f64::NAN; max_count + 1],
        }
    }

    /// The ratio for a component of `count` points.
    pub(crate) fn for_count(&mut self, count: u32) -> f64 {
        let entry = &mut self.ratios[count as usize];
        if entry.is_nan() {
            *entry = NormalWishart::log_gamma_ratio(self.dimension, self.prior_dof + f64::from(count));
        }
        *entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabulated_ratios_equal_direct_evaluation() {
        let mut table = GammaRatioTable::new(3, 4.5, 10);
        for count in [0, 7, 7, 10] {
            let direct = NormalWishart::log_gamma_ratio(3, 4.5 + f64::from(count));
            assert_eq!(table.for_count(count).to_bits(), direct.to_bits());
        }
    }
}
