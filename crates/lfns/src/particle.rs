//! A parameter point together with one realization of its likelihood.

use serde::{Deserialize, Serialize};

use crate::lfns_error::LfnsError;

/// A point of the unit cube (prior-CDF space) and one log-likelihood value
/// obtained for it.
///
/// For stochastic models the log-likelihood is a single realization of an
/// unbiased estimator and is never re-estimated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ParticleRecord", into = "ParticleRecord")]
pub struct Particle {
    unit_coordinates: Box<[f64]>,
    log_likelihood: f64,
}

/// Unvalidated serialized form of [`Particle`].
#[derive(Serialize, Deserialize)]
struct ParticleRecord {
    u: Vec<f64>,
    #[serde(with = "crate::log_space::log_value")]
    log_l: f64,
}

impl Particle {
    /// Creates a particle.
    ///
    /// # Errors
    ///
    /// Returns [`LfnsError::CoordinateOutsideUnitCube`] if a coordinate is
    /// not in `[0, 1]`, and [`LfnsError::InvalidLogLikelihood`] if the
    /// log-likelihood is `NaN` or `+inf`.
    pub fn new(unit_coordinates: Vec<f64>, log_likelihood: f64) -> Result<Self, LfnsError> {
        if let Some((index, &value)) = Self::first_coordinate_outside_cube(&unit_coordinates) {
            return Err(LfnsError::CoordinateOutsideUnitCube { index, value });
        }
        if log_likelihood.is_nan() || log_likelihood == f64::INFINITY {
            return Err(LfnsError::InvalidLogLikelihood { value: log_likelihood });
        }
        Ok(Self {
            unit_coordinates: unit_coordinates.into_boxed_slice(),
            log_likelihood,
        })
    }

    /// Coordinates in the unit cube.
    #[must_use]
    pub fn unit_coordinates(&self) -> &[f64] {
        &self.unit_coordinates
    }

    /// Log-likelihood realization; `-inf` denotes zero likelihood.
    #[must_use]
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    fn first_coordinate_outside_cube(coordinates: &[f64]) -> Option<(usize, &f64)> {
        let unit_interval = 0.0..=1.0;
        coordinates
            .iter()
            .enumerate()
            .find(|(_, value)| !unit_interval.contains(*value))
    }
}

impl TryFrom<ParticleRecord> for Particle {
    type Error = LfnsError;

    fn try_from(record: ParticleRecord) -> Result<Self, Self::Error> {
        Self::new(record.u, record.log_l)
    }
}

impl From<Particle> for ParticleRecord {
    fn from(particle: Particle) -> Self {
        Self {
            u: particle.unit_coordinates.into_vec(),
            log_l: particle.log_likelihood,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_coordinates_outside_the_unit_cube() {
        assert!(Particle::new(vec![0.5, 1.5], 0.0).is_err());
        assert!(Particle::new(vec![f64::NAN], 0.0).is_err());
    }

    #[test]
    fn rejects_log_likelihoods_that_are_not_logarithms() {
        assert!(Particle::new(vec![0.5], f64::NAN).is_err());
        assert!(Particle::new(vec![0.5], f64::INFINITY).is_err());
    }

    #[test]
    fn zero_likelihood_survives_a_json_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let particle = Particle::new(vec![0.25, 0.75], f64::NEG_INFINITY)?;
        let restored: Particle = serde_json::from_str(&serde_json::to_string(&particle)?)?;
        assert_eq!(restored, particle);
        Ok(())
    }

    #[test]
    fn finite_values_survive_a_json_round_trip_bit_exactly() -> Result<(), Box<dyn std::error::Error>> {
        let particle = Particle::new(vec![0.1 + 0.2, 1.0 / 3.0], -1_234.567_890_123_456_7)?;
        let restored: Particle = serde_json::from_str(&serde_json::to_string(&particle)?)?;
        assert_eq!(restored, particle);
        Ok(())
    }
}
