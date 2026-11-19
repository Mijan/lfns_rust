//! Reader for the reaction-network subset of SBML.
//!
//! Supported: compartments with a size, species with an initial amount or
//! concentration, global and reaction-local parameters, and reactions with
//! constant stoichiometries and a MathML kinetic law. Rules, events,
//! function definitions, initial assignments, and constraints are rejected
//! with an error that names them.
//!
//! The network is expressed in amounts. A species that SBML measures as a
//! concentration (no `hasOnlySubstanceUnits`) is replaced in every kinetic
//! law by its amount divided by the compartment size.

use std::collections::HashMap;

use roxmltree::{Document, Node};

use crate::expression::{BinaryOperator, Expression};
use crate::mathml_reader::MathmlReader;
use crate::model_error::ModelError;
use crate::reaction_network::{Quantity, Reaction, ReactionNetwork};

/// A species as declared in SBML.
struct SbmlSpecies {
    id: String,
    initial_amount: f64,
    /// Size of the compartment if kinetic laws see a concentration.
    concentration_volume: Option<f64>,
    /// Whether reactions leave the amount unchanged.
    is_fixed: bool,
}

/// Reads one SBML model.
pub(crate) struct SbmlReader<'a, 'input> {
    model: Node<'a, 'input>,
    compartment_sizes: HashMap<String, f64>,
    species: Vec<SbmlSpecies>,
}

impl ReactionNetwork {
    /// Reads a reaction network from the text of an SBML document.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Sbml`] if the document is malformed and
    /// [`ModelError::UnsupportedSbml`] if it uses constructs outside the
    /// supported subset.
    pub fn from_sbml(text: &str) -> Result<Self, ModelError> {
        let document = Document::parse(text).map_err(|error| ModelError::Sbml {
            reason: error.to_string(),
        })?;
        let model = SbmlReader::child(document.root_element(), "model").ok_or_else(|| ModelError::Sbml {
            reason: "the document has no <model> element".to_owned(),
        })?;
        SbmlReader::new(model)?.read_network()
    }
}

impl<'a, 'input> SbmlReader<'a, 'input> {
    const UNSUPPORTED_LISTS: [(&'static str, &'static str); 5] = [
        ("listOfRules", "rules"),
        ("listOfEvents", "events"),
        ("listOfFunctionDefinitions", "function definitions"),
        ("listOfInitialAssignments", "initial assignments"),
        ("listOfConstraints", "constraints"),
    ];

    fn new(model: Node<'a, 'input>) -> Result<Self, ModelError> {
        for (list, feature) in Self::UNSUPPORTED_LISTS {
            if Self::items(model, list).next().is_some() {
                return Err(ModelError::UnsupportedSbml {
                    feature: feature.to_owned(),
                });
            }
        }
        let mut reader = Self {
            model,
            compartment_sizes: HashMap::new(),
            species: Vec::new(),
        };
        for compartment in Self::items(model, "listOfCompartments") {
            let size = Self::number(compartment, "size")?.unwrap_or(1.0);
            reader.compartment_sizes.insert(Self::id(compartment)?, size);
        }
        reader.species = Self::items(model, "listOfSpecies")
            .map(|node| reader.read_species(node))
            .collect::<Result<_, _>>()?;
        Ok(reader)
    }

    fn read_network(&self) -> Result<ReactionNetwork, ModelError> {
        let species = self
            .species
            .iter()
            .map(|s| Quantity {
                id: s.id.clone(),
                value: s.initial_amount,
            })
            .collect();
        let parameters = Self::items(self.model, "listOfParameters")
            .map(Self::read_parameter)
            .collect::<Result<_, _>>()?;
        let reactions = Self::items(self.model, "listOfReactions")
            .map(|node| self.read_reaction(node))
            .collect::<Result<_, _>>()?;
        Ok(ReactionNetwork::new(species, parameters, reactions))
    }

    fn read_species(&self, node: Node) -> Result<SbmlSpecies, ModelError> {
        let id = Self::id(node)?;
        let volume = node
            .attribute("compartment")
            .and_then(|c| self.compartment_sizes.get(c))
            .copied()
            .unwrap_or(1.0);
        let initial_amount = match (
            Self::number(node, "initialAmount")?,
            Self::number(node, "initialConcentration")?,
        ) {
            (Some(amount), _) => amount,
            (None, Some(concentration)) => concentration * volume,
            (None, None) => {
                let reason = format!("species `{id}` has neither initialAmount nor initialConcentration");
                return Err(ModelError::Sbml { reason });
            }
        };
        let substance_only = node.attribute("hasOnlySubstanceUnits") == Some("true");
        let is_fixed =
            node.attribute("boundaryCondition") == Some("true") || node.attribute("constant") == Some("true");
        Ok(SbmlSpecies {
            id,
            initial_amount,
            concentration_volume: (!substance_only).then_some(volume),
            is_fixed,
        })
    }

    fn read_parameter(node: Node) -> Result<Quantity, ModelError> {
        let id = Self::id(node)?;
        let value = Self::number(node, "value")?.ok_or_else(|| ModelError::Sbml {
            reason: format!("parameter `{id}` has no value"),
        })?;
        Ok(Quantity { id, value })
    }

    fn read_reaction(&self, node: Node) -> Result<Reaction, ModelError> {
        let id = Self::id(node)?;
        let law = Self::child(node, "kineticLaw").ok_or_else(|| ModelError::Sbml {
            reason: format!("reaction `{id}` has no kinetic law"),
        })?;
        let math = Self::child(law, "math").ok_or_else(|| ModelError::Sbml {
            reason: format!("the kinetic law of reaction `{id}` has no <math>"),
        })?;
        let local_values = Self::read_local_parameters(law)?;
        let rate = MathmlReader::read_math(math)?.substitute(&|name| self.resolve_symbol(name, &local_values));
        let mut net_changes = Vec::new();
        self.add_changes(node, "listOfReactants", -1.0, &mut net_changes)?;
        self.add_changes(node, "listOfProducts", 1.0, &mut net_changes)?;
        Ok(Reaction { id, net_changes, rate })
    }

    /// Replaces reaction-local parameters and compartments by their values
    /// and concentration species by amount over volume.
    fn resolve_symbol(&self, name: &str, local_values: &HashMap<String, f64>) -> Option<Expression> {
        if let Some(value) = local_values.get(name).or_else(|| self.compartment_sizes.get(name)) {
            return Some(Expression::Number(*value));
        }
        let volume = self
            .species
            .iter()
            .find(|species| species.id == name)?
            .concentration_volume?;
        let amount = Expression::Symbol(name.to_owned());
        Some(Expression::binary(
            BinaryOperator::Divide,
            amount,
            Expression::Number(volume),
        ))
    }

    fn read_local_parameters(law: Node) -> Result<HashMap<String, f64>, ModelError> {
        let locals = Self::items(law, "listOfLocalParameters").chain(Self::items(law, "listOfParameters"));
        locals
            .map(|node| Self::read_parameter(node).map(|parameter| (parameter.id, parameter.value)))
            .collect()
    }

    /// Adds `sign * stoichiometry` for every species reference in `list`.
    fn add_changes(
        &self,
        reaction: Node,
        list: &str,
        sign: f64,
        net_changes: &mut Vec<(usize, f64)>,
    ) -> Result<(), ModelError> {
        for reference in Self::items(reaction, list) {
            let name = reference.attribute("species").unwrap_or_default();
            let index = self
                .species
                .iter()
                .position(|species| species.id == name)
                .ok_or_else(|| ModelError::Sbml {
                    reason: format!("a reaction refers to the undeclared species `{name}`"),
                })?;
            if self.species[index].is_fixed {
                continue;
            }
            let change = sign * Self::number(reference, "stoichiometry")?.unwrap_or(1.0);
            match net_changes.iter_mut().find(|(species, _)| *species == index) {
                Some((_, total)) => *total += change,
                None => net_changes.push((index, change)),
            }
        }
        Ok(())
    }

    fn child<'n, 'i>(node: Node<'n, 'i>, name: &str) -> Option<Node<'n, 'i>> {
        node.children()
            .find(|child| child.is_element() && child.tag_name().name() == name)
    }

    /// The element children of the list element `list` of `node`.
    fn items<'n, 'i>(node: Node<'n, 'i>, list: &str) -> impl Iterator<Item = Node<'n, 'i>> {
        Self::child(node, list)
            .into_iter()
            .flat_map(|list| list.children().filter(Node::is_element))
    }

    fn id(node: Node) -> Result<String, ModelError> {
        let missing = || ModelError::Sbml {
            reason: format!("a <{}> element has no id", node.tag_name().name()),
        };
        node.attribute("id").map(str::to_owned).ok_or_else(missing)
    }

    fn number(node: Node, attribute: &str) -> Result<Option<f64>, ModelError> {
        let parse = |text: &str| {
            text.parse::<f64>().map_err(|_| ModelError::Sbml {
                reason: format!("`{attribute}=\"{text}\"` is not a number"),
            })
        };
        node.attribute(attribute).map(parse).transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dimerisation in a compartment of size 2 with a concentration species
    /// `A`, an amount species `D`, and a reaction-local rate constant.
    const DIMERISATION: &str = r#"<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="m">
        <listOfCompartments><compartment id="cell" size="2"/></listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="cell" initialConcentration="5"/>
          <species id="D" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="true"/>
        </listOfSpecies>
        <listOfParameters><parameter id="scale" value="3"/></listOfParameters>
        <listOfReactions>
          <reaction id="dimerise">
            <listOfReactants><speciesReference species="A" stoichiometry="2"/></listOfReactants>
            <listOfProducts><speciesReference species="D"/></listOfProducts>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply><times/><ci>cell</ci><ci>kf</ci><ci>scale</ci><apply><power/><ci>A</ci><cn>2</cn></apply></apply>
              </math>
              <listOfLocalParameters><localParameter id="kf" value="0.5"/></listOfLocalParameters>
            </kineticLaw>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>"#;

    #[test]
    fn concentrations_become_amounts_and_local_parameters_become_numbers() -> Result<(), ModelError> {
        let network = ReactionNetwork::from_sbml(DIMERISATION)?;
        // Initial concentration 5 in volume 2 is an amount of 10.
        assert!((network.species()[0].value - 10.0).abs() < 1e-12);
        assert_eq!(network.reactions()[0].net_changes, vec![(0, -2.0), (1, 1.0)]);
        // Rate = cell * kf * scale * (A / cell)^2 = 2 * 0.5 * 3 * (10 / 2)^2 = 75.
        let rate = network.compile_rates()?.remove(0);
        assert!((rate.evaluate(&[10.0, 1.0, 3.0], &mut Vec::new()) - 75.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn rules_are_rejected_by_name() {
        let with_rule = DIMERISATION.replace(
            "<listOfReactions>",
            r#"<listOfRules><assignmentRule variable="scale"><math/></assignmentRule></listOfRules><listOfReactions>"#,
        );
        let error = ReactionNetwork::from_sbml(&with_rule).expect_err("rules are unsupported");
        assert!(
            matches!(error, ModelError::UnsupportedSbml { ref feature } if feature == "rules"),
            "{error}"
        );
    }

    #[test]
    fn exponent_notation_numbers_are_read() -> Result<(), ModelError> {
        let scaled = DIMERISATION.replace("<cn>2</cn>", r#"<cn type="e-notation">2<sep/>0</cn>"#);
        let rate = ReactionNetwork::from_sbml(&scaled)?.compile_rates()?.remove(0);
        assert!((rate.evaluate(&[10.0, 1.0, 3.0], &mut Vec::new()) - 75.0).abs() < 1e-12);
        Ok(())
    }
}
