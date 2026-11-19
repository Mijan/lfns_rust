//! The experiment-specific part of a simulation.

/// One experimental condition: the initial amounts and parameter values
/// that differ from the model's defaults for one experiment, for example a
/// run's inoculum and its feed schedule.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentalCondition {
    id: String,
    /// `(slot in a value buffer, value)` for every overridden species amount
    /// and parameter.
    overrides: Vec<(usize, f64)>,
}

impl ExperimentalCondition {
    /// Creates a condition from the values it sets, given as
    /// `(slot in a value buffer, value)`.
    #[must_use]
    pub fn new(id: &str, overrides: Vec<(usize, f64)>) -> Self {
        Self {
            id: id.to_owned(),
            overrides,
        }
    }

    /// Identifier of the condition.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The slots this condition sets.
    pub fn overridden_slots(&self) -> impl Iterator<Item = usize> {
        self.overrides.iter().map(|(slot, _)| *slot)
    }

    /// Writes the condition's values into a value buffer.
    pub fn apply_to(&self, values: &mut [f64]) {
        for &(slot, value) in &self.overrides {
            values[slot] = value;
        }
    }
}
