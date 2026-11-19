//! A fit recovers the structure of data drawn from known mixtures.

use dpgmm::{Dpgmm, DpgmmConfig, DpgmmError, GaussianComponent, GaussianMixture, Observations};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const N_DIMENSIONS: usize = 2;

/// Two well-separated, differently shaped clusters with weights 0.7 / 0.3.
fn two_cluster_truth() -> Result<GaussianMixture, DpgmmError> {
    let broad = GaussianComponent::new(0.7, vec![0.0, 0.0], &[1.0, 0.8, 0.8, 1.0])?;
    let narrow = GaussianComponent::new(0.3, vec![8.0, -6.0], &[0.2, 0.0, 0.0, 0.05])?;
    GaussianMixture::new(vec![broad, narrow])
}

/// Draws `n_points` points from `truth`, one after the other in a flat buffer.
fn draw_values(truth: &GaussianMixture, n_points: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut values = vec![0.0; n_points * N_DIMENSIONS];
    for point in values.chunks_exact_mut(N_DIMENSIONS) {
        truth.sample(&mut rng, point);
    }
    values
}

fn draw_observations(truth: &GaussianMixture, n_points: usize, seed: u64) -> Result<Observations, DpgmmError> {
    Observations::from_rows(&draw_values(truth, n_points, seed), N_DIMENSIONS)
}

fn fit(observations: &Observations, seed: u64) -> Result<GaussianMixture, DpgmmError> {
    Dpgmm::new(DpgmmConfig::new(30)?).fit(observations, &mut ChaCha8Rng::seed_from_u64(seed))
}

#[test]
fn separated_clusters_are_found_with_their_weights() -> Result<(), DpgmmError> {
    let fitted = fit(&draw_observations(&two_cluster_truth()?, 400, 1)?, 2)?;
    // Weight of the fitted components whose mean lies near the narrow cluster.
    let near_narrow = |c: &&GaussianComponent| (c.mean()[0] - 8.0).abs() < 1.0 && (c.mean()[1] + 6.0).abs() < 1.0;
    let narrow_weight: f64 = fitted
        .components()
        .iter()
        .filter(near_narrow)
        .map(GaussianComponent::weight)
        .sum();
    assert!(
        (narrow_weight - 0.3).abs() < 0.08,
        "weight near the narrow cluster: {narrow_weight}"
    );
    Ok(())
}

#[test]
fn fitted_density_tracks_the_true_density() -> Result<(), DpgmmError> {
    let truth = two_cluster_truth()?;
    let fitted = fit(&draw_observations(&truth, 400, 3)?, 4)?;
    for probe in [[0.0, 0.0], [0.7, 0.7], [8.0, -6.0], [8.3, -6.1]] {
        let difference = fitted.log_density(&probe) - truth.log_density(&probe);
        assert!(difference.abs() < 0.7, "log-density off by {difference} at {probe:?}");
    }
    // Between the clusters the true density is negligible; so must the fit be.
    let gap = [4.0, -3.0];
    assert!(fitted.log_density(&gap) < fitted.log_density(&[0.0, 0.0]) - 8.0);
    Ok(())
}

#[test]
fn a_fit_is_a_function_of_the_seed() -> Result<(), DpgmmError> {
    let observations = draw_observations(&two_cluster_truth()?, 150, 5)?;
    assert_eq!(fit(&observations, 6)?, fit(&observations, 6)?);
    Ok(())
}

#[test]
fn fit_is_equivariant_under_affine_maps_of_the_data() -> Result<(), DpgmmError> {
    // Whitening makes the sampler see the same numbers after x -> 1000 x + 5,
    // so the fitted mixture must be the mapped mixture.
    let values = draw_values(&two_cluster_truth()?, 150, 7);
    let mapped_values: Vec<f64> = values.iter().map(|value| 1000.0 * value + 5.0).collect();
    let observations = Observations::from_rows(&values, N_DIMENSIONS)?;
    let mapped = Observations::from_rows(&mapped_values, N_DIMENSIONS)?;
    let (fitted, fitted_mapped) = (fit(&observations, 8)?, fit(&mapped, 8)?);
    assert_eq!(fitted.components().len(), fitted_mapped.components().len());
    for (original, mapped) in fitted.components().iter().zip(fitted_mapped.components()) {
        assert!((mapped.mean()[0] - (1000.0 * original.mean()[0] + 5.0)).abs() < 1e-5);
        assert!((mapped.covariance()[0] / (1e6 * original.covariance()[0]) - 1.0).abs() < 1e-7);
    }
    Ok(())
}
