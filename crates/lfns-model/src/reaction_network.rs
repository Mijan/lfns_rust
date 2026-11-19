//! A dynamical model: species, parameters, and what changes them.
//!
//! Species change through reactions (a rate law times net changes) and
//! through rate rules (an explicit time derivative). Assignment rules define
//! named intermediate quantities. Timed assignments change a parameter at a
//! fixed time, which is how piecewise-constant inputs such as feed rates
//! enter a model.

use crate::compiled_expression::CompiledExpression;
use crate::expression::{BinaryOperator, Expression};
use crate::model_error::ModelError;

/// A species. Its state is always an amount; if `volume` is set, formulas
/// that name the species see its concentration, the amount over that volume.
#[derive(Debug, Clone, PartialEq)]
pub struct Species {
    /// Identifier of the species.
    pub id: String,
    /// Amount at time zero.
    pub initial_amount: f64,
    /// Volume of the compartment if formulas read a concentration.
    pub volume: Option<f64>,
}

impl Species {
    /// A species that formulas read as an amount.
    #[must_use]
    pub fn amount(id: &str, initial_amount: f64) -> Self {
        Self {
            id: id.to_owned(),
            initial_amount,
            volume: None,
        }
    }

    /// A species that formulas read as a concentration in a compartment of
    /// the given volume.
    #[must_use]
    pub fn concentration(id: &str, initial_concentration: f64, volume: f64) -> Self {
        Self {
            id: id.to_owned(),
            initial_amount: initial_concentration * volume,
            volume: Some(volume),
        }
    }

    /// Amount per unit of the quantity formulas read: the volume for a
    /// concentration species, one otherwise.
    #[must_use]
    pub fn amount_per_reading(&self) -> f64 {
        self.volume.unwrap_or(1.0)
    }
}

/// A parameter with its default value.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    /// Identifier of the parameter.
    pub id: String,
    /// Default value.
    pub value: f64,
}

/// A reaction: how it changes the species amounts and how fast it fires.
#[derive(Debug, Clone, PartialEq)]
pub struct Reaction {
    /// Identifier of the reaction.
    pub id: String,
    /// Net change of each affected species per firing, as
    /// `(species index, change)`.
    pub net_changes: Vec<(usize, f64)>,
    /// Rate in amount per time. In the stochastic model it is the propensity
    /// of the reaction.
    pub rate: Expression,
}

/// An explicit time derivative of a species, in the units formulas read it
/// in (concentration per time for a concentration species).
#[derive(Debug, Clone, PartialEq)]
pub struct RateRule {
    /// Position of the species among the species of the network.
    pub species_index: usize,
    /// The time derivative.
    pub rate: Expression,
}

/// A named quantity defined by a formula at all times.
#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentRule {
    /// Identifier of the quantity.
    pub id: String,
    /// Its defining formula.
    pub expression: Expression,
}

/// A change of a parameter at a fixed time.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedAssignment {
    /// Time at which the assignment takes effect.
    pub time: f64,
    /// Identifier of the parameter that is assigned.
    pub parameter: String,
    /// The value it takes, evaluated at that time.
    pub value: Expression,
}

/// A dynamical model in amounts.
///
/// All formulas are evaluated over a *value buffer* laid out as: species
/// amounts, parameters, the time, and the assignment-rule quantities in an
/// order in which each depends only on earlier ones.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactionNetwork {
    species: Vec<Species>,
    parameters: Vec<Quantity>,
    reactions: Vec<Reaction>,
    rate_rules: Vec<RateRule>,
    assignment_rules: Vec<AssignmentRule>,
    timed_assignments: Vec<TimedAssignment>,
}

impl ReactionNetwork {
    /// Creates a network of species, parameters, and reactions.
    #[must_use]
    pub fn new(species: Vec<Species>, parameters: Vec<Quantity>, reactions: Vec<Reaction>) -> Self {
        Self {
            species,
            parameters,
            reactions,
            rate_rules: Vec::new(),
            assignment_rules: Vec::new(),
            timed_assignments: Vec::new(),
        }
    }

    /// Returns the network with the given rate rules.
    #[must_use]
    pub fn with_rate_rules(self, rate_rules: Vec<RateRule>) -> Self {
        Self { rate_rules, ..self }
    }

    /// Returns the network with the given assignment rules, ordered so that
    /// each rule depends only on rules before it.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Sbml`] if the rules depend on each other in a
    /// cycle.
    pub fn with_assignment_rules(self, rules: Vec<AssignmentRule>) -> Result<Self, ModelError> {
        Ok(Self {
            assignment_rules: Self::order_by_dependency(rules)?,
            ..self
        })
    }

    /// Returns the network with the given timed assignments, ordered by
    /// time; assignments at the same time keep their given order.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Sbml`] if an assignment targets an unknown
    /// parameter or has a negative or non-finite time.
    pub fn with_timed_assignments(self, mut assignments: Vec<TimedAssignment>) -> Result<Self, ModelError> {
        for assignment in &assignments {
            if self.parameter_index(&assignment.parameter).is_none() {
                let reason = format!(
                    "an event assigns `{}`, which is not a parameter of the model",
                    assignment.parameter
                );
                return Err(ModelError::Sbml { reason });
            }
            if !(assignment.time.is_finite() && assignment.time >= 0.0) {
                let reason = format!(
                    "an event of `{}` fires at the invalid time {}",
                    assignment.parameter, assignment.time
                );
                return Err(ModelError::Sbml { reason });
            }
        }
        assignments.sort_by(|a, b| a.time.total_cmp(&b.time));
        Ok(Self {
            timed_assignments: assignments,
            ..self
        })
    }

    /// The species.
    #[must_use]
    pub fn species(&self) -> &[Species] {
        &self.species
    }

    /// The parameters with their default values.
    #[must_use]
    pub fn parameters(&self) -> &[Quantity] {
        &self.parameters
    }

    /// The reactions.
    #[must_use]
    pub fn reactions(&self) -> &[Reaction] {
        &self.reactions
    }

    /// The rate rules.
    #[must_use]
    pub fn rate_rules(&self) -> &[RateRule] {
        &self.rate_rules
    }

    /// The assignment rules, each depending only on earlier ones.
    #[must_use]
    pub fn assignment_rules(&self) -> &[AssignmentRule] {
        &self.assignment_rules
    }

    /// The timed assignments in order of time.
    #[must_use]
    pub fn timed_assignments(&self) -> &[TimedAssignment] {
        &self.timed_assignments
    }

    /// Whether the model is a pure reaction network, which is the only kind
    /// that can be simulated as a stochastic jump process.
    #[must_use]
    pub fn is_pure_reaction_network(&self) -> bool {
        self.rate_rules.is_empty() && self.assignment_rules.is_empty() && self.timed_assignments.is_empty()
    }

    /// Position of the species `id` among the species.
    #[must_use]
    pub fn species_index(&self, id: &str) -> Option<usize> {
        self.species.iter().position(|species| species.id == id)
    }

    /// Position of the parameter `id` among the parameters.
    #[must_use]
    pub fn parameter_index(&self, id: &str) -> Option<usize> {
        self.parameters.iter().position(|parameter| parameter.id == id)
    }

    /// Sets the default value of parameter `id`, adding the parameter if the
    /// network does not have it yet (for example a noise parameter that only
    /// observable formulas use).
    pub fn set_parameter(&mut self, id: &str, value: f64) {
        match self.parameter_index(id) {
            Some(index) => self.parameters[index].value = value,
            None => self.parameters.push(Quantity {
                id: id.to_owned(),
                value,
            }),
        }
    }

    /// Slot of the time in a value buffer.
    #[must_use]
    pub fn time_slot(&self) -> usize {
        self.species.len() + self.parameters.len()
    }

    /// Length of a value buffer.
    #[must_use]
    pub fn value_width(&self) -> usize {
        self.time_slot() + 1 + self.assignment_rules.len()
    }

    /// Slot of the species, parameter, rule quantity, or time named `id` in
    /// a value buffer.
    #[must_use]
    pub fn slot_of(&self, id: &str) -> Option<usize> {
        if id == Expression::TIME {
            return Some(self.time_slot());
        }
        let rule_slot = || {
            self.assignment_rules
                .iter()
                .position(|rule| rule.id == id)
                .map(|i| self.time_slot() + 1 + i)
        };
        let parameter_slot = || self.parameter_index(id).map(|index| self.species.len() + index);
        self.species_index(id).or_else(parameter_slot).or_else(rule_slot)
    }

    /// Compiles `expression` against the value buffer of this network. A
    /// concentration species is read as its amount over its volume.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownSymbol`] if the expression refers to a
    /// name the network does not define.
    pub fn compile_expression(&self, expression: &Expression, context: &str) -> Result<CompiledExpression, ModelError> {
        let as_reading = |id: &str| {
            let volume = self.species.iter().find(|species| species.id == id)?.volume?;
            let amount = Expression::Symbol(id.to_owned());
            Some(Expression::binary(
                BinaryOperator::Divide,
                amount,
                Expression::Number(volume),
            ))
        };
        CompiledExpression::compile(&expression.substitute(&as_reading), &|id| self.slot_of(id), context)
    }

    /// Orders rules so that each depends only on rules placed before it.
    fn order_by_dependency(mut pending: Vec<AssignmentRule>) -> Result<Vec<AssignmentRule>, ModelError> {
        let mut ordered: Vec<AssignmentRule> = Vec::with_capacity(pending.len());
        while !pending.is_empty() {
            let is_ready = |rule: &AssignmentRule| {
                let mut symbols = Vec::new();
                rule.expression.collect_symbols(&mut symbols);
                !symbols
                    .iter()
                    .any(|symbol| pending.iter().any(|other| other.id == *symbol))
            };
            let Some(position) = pending.iter().position(is_ready) else {
                let ids: Vec<&str> = pending.iter().map(|rule| rule.id.as_str()).collect();
                let reason = format!("the assignment rules of {ids:?} depend on each other in a cycle");
                return Err(ModelError::Sbml { reason });
            };
            ordered.push(pending.remove(position));
        }
        Ok(ordered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(id: &str, formula: &str) -> AssignmentRule {
        AssignmentRule {
            id: id.to_owned(),
            expression: Expression::parse(formula).expect("valid test formula"),
        }
    }

    #[test]
    fn assignment_rules_are_ordered_by_dependency() -> Result<(), ModelError> {
        let network = ReactionNetwork::new(vec![Species::amount("s", 1.0)], Vec::new(), Vec::new())
            .with_assignment_rules(vec![rule("c", "a + b"), rule("b", "2 * a"), rule("a", "s + 1")])?;
        let order: Vec<&str> = network.assignment_rules().iter().map(|rule| rule.id.as_str()).collect();
        assert_eq!(order, ["a", "b", "c"]);
        Ok(())
    }

    #[test]
    fn cyclic_assignment_rules_are_rejected() {
        let network = ReactionNetwork::new(Vec::new(), Vec::new(), Vec::new());
        assert!(
            network
                .with_assignment_rules(vec![rule("a", "b"), rule("b", "a")])
                .is_err()
        );
    }

    #[test]
    fn concentration_species_are_read_as_amount_over_volume() -> Result<(), ModelError> {
        let network = ReactionNetwork::new(vec![Species::concentration("c", 3.0, 2.0)], Vec::new(), Vec::new());
        assert!((network.species()[0].initial_amount - 6.0).abs() < 1e-12);
        let reading = network.compile_expression(&Expression::parse("c")?, "test")?;
        assert!((reading.evaluate(&[6.0, 0.0], &mut Vec::new()) - 3.0).abs() < 1e-12);
        Ok(())
    }
}
