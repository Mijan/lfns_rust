//! Density estimation with Dirichlet process Gaussian mixture models
//! (DP-GMM).
//!
//! A DP-GMM describes data as a mixture of Gaussians without fixing the
//! number of components: the number is inferred from the data together with
//! the means and covariances. [`Dpgmm::fit`] runs a Gibbs sampler over
//! [`Observations`] and returns a [`GaussianMixture`] that can be evaluated
//! and sampled.
//!
//! The model is the conjugate one of Görür and Rasmussen, "Dirichlet process
//! Gaussian mixture models: choice of the base distribution", Journal of
//! Computer Science and Technology 25(4): 653-664 (2010).

mod component_statistics;
mod dpgmm;
mod dpgmm_error;
mod gaussian_mixture;
mod gibbs_state;
mod hyperparameters;
mod normal_wishart;
mod observations;
mod slice_sampler;
mod whitening;
mod wishart;

pub use dpgmm::{Dpgmm, DpgmmConfig};
pub use dpgmm_error::DpgmmError;
pub use gaussian_mixture::{GaussianComponent, GaussianMixture};
pub use observations::Observations;
