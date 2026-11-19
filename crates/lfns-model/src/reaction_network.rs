//! A reaction network: species, parameters, and reactions with rate laws.

use crate::compiled_expression::CompiledExpression;
use crate::expression::Expression;
use crate::model_error::ModelError;

/// A reaction: how it changes the species amounts and how fast it fires.
#[derive(Debug, Clone, PartialEq)]
pub struct Reaction {
    /// Identifier of the reaction.
    pub id: String,
    /// Net change of each affected species per firing, as
    /// `(species index, change)`.
    pub net_changes: Vec<(usize, f64)>,
    /// Rate in amount per time, over species amounts and parameters. In the
    /// stochastic model it is the propensity of the reaction.
    pub rate: Expression,
}

/// A named quantity with a value: a species with its initial amount, or a
/// parameter with its default value.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    /// Identifier of the quantity.
    pub id: String,
    /// Initial amount or default value.
    pub value: f64,
}

/// A reaction network in amounts (molecule counts for stochastic models).
#[derive(Debug, Clone, PartialEq)]
pub struct ReactionNetwork {
    species: Vec<Quantity>,
    parameters: Vec<Quantity>,
    reactions: Vec<Reaction>,
}

impl ReactionNetwork {
    /// Creates a network from its parts.
    #[must_use]
    pub fn new(species: Vec<Quantity>, parameters: Vec<Quantity>, reactions: Vec<Reaction>) -> Self {
        Self {
            species,
            parameters,
            reactions,
        }
    }

    /// The species with their initial amounts.
    #[must_use]
    pub fn species(&self) -> &[Quantity] {
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

    /// Position of the species or parameter `id` in a value buffer laid out
    /// as all species amounts followed by all parameters.
    #[must_use]
    pub fn slot_of(&self, id: &str) -> Option<usize> {
        let species_slot = self.species.iter().position(|species| species.id == id);
        species_slot.or_else(|| self.parameter_index(id).map(|index| self.species.len() + index))
    }

    /// Compiles `expression` against the value buffer of this network.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownSymbol`] if the expression refers to a
    /// name that is neither a species nor a parameter.
    pub fn compile_expression(&self, expression: &Expression, context: &str) -> Result<CompiledExpression, ModelError> {
        CompiledExpression::compile(expression, &|id| self.slot_of(id), context)
    }

    /// Compiles the rate law of every reaction.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownSymbol`] if a rate law refers to an
    /// unknown name.
    pub fn compile_rates(&self) -> Result<Vec<CompiledExpression>, ModelError> {
        let compile = |reaction: &Reaction| {
            self.compile_expression(
                &reaction.rate,
                &format!("the kinetic law of reaction `{}`", reaction.id),
            )
        };
        self.reactions.iter().map(compile).collect()
    }
}
