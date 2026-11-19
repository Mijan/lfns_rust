//! Fitting a Dirichlet process Gaussian mixture to observations.

use rand::Rng;

use crate::dpgmm_error::DpgmmError;
use crate::gaussian_mixture::{GaussianComponent, GaussianMixture};
use crate::gibbs_state::GibbsState;
use crate::normal_wishart::GaussianDraw;
use crate::observations::Observations;
use crate::whitening::Whitening;

/// Settings of a fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DpgmmConfig {
    n_sweeps: u32,
}

impl DpgmmConfig {
    /// Default number of Gibbs sweeps.
    pub const DEFAULT_N_SWEEPS: u32 = 200;

    /// Creates a configuration that runs `n_sweeps` Gibbs sweeps.
    ///
    /// # Errors
    ///
    /// Returns [`DpgmmError::NoSweeps`] if `n_sweeps` is zero.
    pub fn new(n_sweeps: u32) -> Result<Self, DpgmmError> {
        if n_sweeps == 0 {
            return Err(DpgmmError::NoSweeps);
        }
        Ok(Self { n_sweeps })
    }

    /// Number of Gibbs sweeps of a fit.
    #[must_use]
    pub fn n_sweeps(&self) -> u32 {
        self.n_sweeps
    }
}

impl Default for DpgmmConfig {
    fn default() -> Self {
        Self {
            n_sweeps: Self::DEFAULT_N_SWEEPS,
        }
    }
}

/// Density estimator based on a Dirichlet process mixture of Gaussians.
///
/// The number of components is not fixed in advance: it is inferred together
/// with the components' means and covariances by Gibbs sampling. A fit
/// returns the mixture of the last sweep, that is one draw from the
/// posterior over mixtures.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dpgmm {
    config: DpgmmConfig,
}

impl Dpgmm {
    /// Creates an estimator with the given settings.
    #[must_use]
    pub fn new(config: DpgmmConfig) -> Self {
        Self { config }
    }

    /// Fits a mixture to `observations`, drawing all randomness from `rng`.
    ///
    /// # Errors
    ///
    /// Returns [`DpgmmError::DegenerateObservations`] if the points lie in a
    /// subspace, and [`DpgmmError::NumericalBreakdown`] if a matrix loses
    /// positive definiteness during sampling.
    pub fn fit<R: Rng + ?Sized>(
        &self,
        observations: &Observations,
        rng: &mut R,
    ) -> Result<GaussianMixture, DpgmmError> {
        let whitening = Whitening::of(observations)?;
        let mut state = GibbsState::new(&whitening.whiten(observations)?)?;
        state.resample_assignments(rng)?;
        let mut components = state.draw_components(rng)?;
        for _ in 1..self.config.n_sweeps {
            state.resample_hyperparameters(&components, rng)?;
            state.resample_assignments(rng)?;
            components = state.draw_components(rng)?;
        }
        Self::assemble_mixture(&whitening, &state.component_weights(), &components)
    }

    /// Maps the component draws from whitened space back to the space of
    /// the observations.
    fn assemble_mixture(
        whitening: &Whitening,
        weights: &[f64],
        components: &[GaussianDraw],
    ) -> Result<GaussianMixture, DpgmmError> {
        const STEP: DpgmmError = DpgmmError::NumericalBreakdown {
            step: "assembling the fitted mixture",
        };
        let restored = weights.iter().zip(components).map(|(weight, component)| {
            let covariance = whitening.restore_covariance(&component.precision.inverse()?);
            let mean = whitening.restore_mean(&component.mean);
            GaussianComponent::from_moments(*weight, &mean, &covariance).map_err(|_| STEP)
        });
        GaussianMixture::new(restored.collect::<Result<_, _>>()?)
    }
}
