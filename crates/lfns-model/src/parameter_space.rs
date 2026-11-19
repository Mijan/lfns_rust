//! The estimated parameters, their scales and bounds, and the map from the
//! unit cube to parameter values.

use crate::model_error::ModelError;

/// The scale on which a parameter is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterScale {
    /// The parameter itself.
    Linear,
    /// Its natural logarithm.
    Log,
    /// Its base-10 logarithm.
    Log10,
}

impl ParameterScale {
    /// Reads the `parameterScale` entry of a PEtab parameter table.
    ///
    /// # Errors
    ///
    /// Returns a description of the valid entries if `name` is not one.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "lin" => Ok(Self::Linear),
            "log" => Ok(Self::Log),
            "log10" => Ok(Self::Log10),
            other => Err(format!(
                "unknown parameterScale `{other}`; use `lin`, `log`, or `log10`"
            )),
        }
    }

    /// Maps a parameter value onto this scale.
    #[must_use]
    pub fn scale(self, value: f64) -> f64 {
        match self {
            Self::Linear => value,
            Self::Log => value.ln(),
            Self::Log10 => value.log10(),
        }
    }

    /// Maps a value on this scale back to the parameter.
    #[must_use]
    pub fn unscale(self, scaled: f64) -> f64 {
        match self {
            Self::Linear => scaled,
            Self::Log => scaled.exp(),
            Self::Log10 => 10.0_f64.powf(scaled),
        }
    }
}

/// A parameter that is estimated, with a prior that is uniform on its scale
/// between two bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct EstimatedParameter {
    id: String,
    /// Position among the parameters of the model.
    index: usize,
    scale: ParameterScale,
    scaled_lower: f64,
    scaled_upper: f64,
}

impl EstimatedParameter {
    /// Creates an estimated parameter with bounds given on the linear scale.
    ///
    /// # Errors
    ///
    /// Returns a description of the problem unless the bounds are finite on
    /// the chosen scale and `lower < upper`.
    pub fn new(id: &str, index: usize, scale: ParameterScale, (lower, upper): (f64, f64)) -> Result<Self, String> {
        let (scaled_lower, scaled_upper) = (scale.scale(lower), scale.scale(upper));
        if !(scaled_lower.is_finite() && scaled_upper.is_finite() && scaled_lower < scaled_upper) {
            return Err(format!(
                "parameter `{id}` has bounds [{lower}, {upper}], which are not a finite, non-empty interval on \
                 its scale; give lowerBound < upperBound, both positive for a logarithmic scale"
            ));
        }
        Ok(Self {
            id: id.to_owned(),
            index,
            scale,
            scaled_lower,
            scaled_upper,
        })
    }

    /// Identifier of the parameter.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Value of the parameter at the unit-interval coordinate `unit`.
    fn value_at(&self, unit: f64) -> f64 {
        self.scale
            .unscale(self.scaled_lower + unit * (self.scaled_upper - self.scaled_lower))
    }
}

/// The parameters of a model: fixed values for all of them and the prior of
/// those that are estimated.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSpace {
    nominal_values: Vec<f64>,
    estimated: Vec<EstimatedParameter>,
}

impl ParameterSpace {
    /// Creates a parameter space from the nominal values of all model
    /// parameters and the estimated ones among them.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnsupportedPetab`] if no parameter is estimated.
    pub fn new(nominal_values: Vec<f64>, estimated: Vec<EstimatedParameter>) -> Result<Self, ModelError> {
        if estimated.is_empty() {
            return Err(ModelError::UnsupportedPetab {
                feature: "a problem without estimated parameters".to_owned(),
                remedy: "Set `estimate` to 1 for at least one row of the parameter table.".to_owned(),
            });
        }
        Ok(Self {
            nominal_values,
            estimated,
        })
    }

    /// The estimated parameters, in the order of the unit-cube coordinates.
    #[must_use]
    pub fn estimated(&self) -> &[EstimatedParameter] {
        &self.estimated
    }

    /// Values of the estimated parameters at `unit_point`.
    #[must_use]
    pub fn estimated_values(&self, unit_point: &[f64]) -> Vec<f64> {
        self.estimated
            .iter()
            .zip(unit_point)
            .map(|(parameter, unit)| parameter.value_at(*unit))
            .collect()
    }

    /// Values of all model parameters at `unit_point`: the estimated ones
    /// from the prior transform, the others at their nominal values.
    #[must_use]
    pub fn model_parameters(&self, unit_point: &[f64]) -> Vec<f64> {
        let mut values = self.nominal_values.clone();
        for (parameter, unit) in self.estimated.iter().zip(unit_point) {
            values[parameter.index] = parameter.value_at(*unit);
        }
        values
    }

    /// Values of all model parameters when the estimated ones take
    /// `estimated_values`.
    #[must_use]
    pub fn model_parameters_from_values(&self, estimated_values: &[f64]) -> Vec<f64> {
        let mut values = self.nominal_values.clone();
        for (parameter, value) in self.estimated.iter().zip(estimated_values) {
            values[parameter.index] = *value;
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_interval_maps_uniformly_onto_the_scaled_bounds() -> Result<(), String> {
        let parameter = EstimatedParameter::new("k", 0, ParameterScale::Log10, (0.01, 100.0))?;
        assert!((parameter.value_at(0.0) - 0.01).abs() < 1e-12);
        assert!((parameter.value_at(0.5) - 1.0).abs() < 1e-12);
        assert!((parameter.value_at(1.0) - 100.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn logarithmic_scale_rejects_non_positive_bounds() {
        assert!(EstimatedParameter::new("k", 0, ParameterScale::Log, (0.0, 1.0)).is_err());
        assert!(EstimatedParameter::new("k", 0, ParameterScale::Linear, (2.0, 1.0)).is_err());
    }

    #[test]
    fn fixed_parameters_keep_their_nominal_values() -> Result<(), Box<dyn std::error::Error>> {
        let estimated = EstimatedParameter::new("b", 1, ParameterScale::Linear, (0.0, 10.0))?;
        let space = ParameterSpace::new(vec![7.0, 1.0, 3.0], vec![estimated])?;
        assert_eq!(space.model_parameters(&[0.25]), vec![7.0, 2.5, 3.0]);
        assert_eq!(space.estimated_values(&[0.25]), vec![2.5]);
        Ok(())
    }
}
