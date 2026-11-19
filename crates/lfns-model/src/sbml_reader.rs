//! Reader for the supported subset of SBML.
//!
//! Supported: compartments with a constant size, species with an initial
//! amount or concentration, global and reaction-local parameters, reactions
//! with constant stoichiometries and a kinetic law, rate rules on species,
//! assignment rules on parameters, and events that assign parameters at a
//! fixed time (trigger `time >= constant`). Function definitions, initial
//! assignments, constraints, algebraic rules, and state-triggered events
//! are rejected with an error that names them.

use std::collections::HashMap;

use roxmltree::{Document, Node};

use crate::expression::{BinaryOperator, Expression};
use crate::mathml_reader::MathmlReader;
use crate::model_error::ModelError;
use crate::reaction_network::{
    AssignmentRule, Quantity, RateRule, Reaction, ReactionNetwork, Species, TimedAssignment,
};

/// A species as declared in SBML.
struct SbmlSpecies {
    species: Species,
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
    /// Reads a model from the text of an SBML document.
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
    const UNSUPPORTED_LISTS: [(&'static str, &'static str); 3] = [
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
        let (assignment_rules, rate_rules) = self.read_rules()?;
        let is_rule_target = |node: &Node| {
            assignment_rules
                .iter()
                .any(|rule| Some(rule.id.as_str()) == node.attribute("id"))
        };
        let parameters = Self::items(self.model, "listOfParameters")
            .filter(|node| !is_rule_target(node))
            .map(Self::read_parameter)
            .collect::<Result<_, _>>()?;
        let reactions = Self::items(self.model, "listOfReactions")
            .map(|node| self.read_reaction(node))
            .collect::<Result<_, _>>()?;
        let species = self.species.iter().map(|declared| declared.species.clone()).collect();
        ReactionNetwork::new(species, parameters, reactions)
            .with_rate_rules(rate_rules)
            .with_assignment_rules(assignment_rules)?
            .with_timed_assignments(self.read_events()?)
    }

    fn read_species(&self, node: Node) -> Result<SbmlSpecies, ModelError> {
        let id = Self::id(node)?;
        let volume = node
            .attribute("compartment")
            .and_then(|c| self.compartment_sizes.get(c))
            .copied()
            .unwrap_or(1.0);
        let substance_only = node.attribute("hasOnlySubstanceUnits") == Some("true");
        let species = match (
            Self::number(node, "initialAmount")?,
            Self::number(node, "initialConcentration")?,
        ) {
            (Some(amount), _) => Species {
                id,
                initial_amount: amount,
                volume: (!substance_only).then_some(volume),
            },
            (None, Some(concentration)) if substance_only => Species::amount(&id, concentration * volume),
            (None, Some(concentration)) => Species::concentration(&id, concentration, volume),
            (None, None) => {
                let reason = format!("species `{id}` has neither initialAmount nor initialConcentration");
                return Err(ModelError::Sbml { reason });
            }
        };
        let is_fixed =
            node.attribute("boundaryCondition") == Some("true") || node.attribute("constant") == Some("true");
        Ok(SbmlSpecies { species, is_fixed })
    }

    fn read_parameter(node: Node) -> Result<Quantity, ModelError> {
        let id = Self::id(node)?;
        let value = Self::number(node, "value")?.ok_or_else(|| ModelError::Sbml {
            reason: format!("parameter `{id}` has no value"),
        })?;
        Ok(Quantity { id, value })
    }

    /// Reads the rules: assignment rules on parameters and rate rules on
    /// species.
    fn read_rules(&self) -> Result<(Vec<AssignmentRule>, Vec<RateRule>), ModelError> {
        let (mut assignment_rules, mut rate_rules) = (Vec::new(), Vec::new());
        for rule in Self::items(self.model, "listOfRules") {
            let variable = rule.attribute("variable").unwrap_or_default();
            let species_index = self.species.iter().position(|declared| declared.species.id == variable);
            let expression = self.read_math(rule, &format!("the rule of `{variable}`"))?;
            match (rule.tag_name().name(), species_index) {
                ("assignmentRule", None) => assignment_rules.push(AssignmentRule {
                    id: variable.to_owned(),
                    expression,
                }),
                ("rateRule", Some(species_index)) => rate_rules.push(RateRule {
                    species_index,
                    rate: expression,
                }),
                ("assignmentRule", Some(_)) => {
                    return Err(Self::unsupported(&format!(
                        "an assignment rule on the species `{variable}`"
                    )));
                }
                ("rateRule", None) => {
                    return Err(Self::unsupported(&format!(
                        "a rate rule on `{variable}`, which is not a species"
                    )));
                }
                (other, _) => return Err(Self::unsupported(&format!("rules of kind <{other}>"))),
            }
        }
        Ok((assignment_rules, rate_rules))
    }

    /// Reads the events as timed assignments. An event whose trigger is
    /// already true at time zero fires there only if the trigger declares
    /// `initialValue="false"`, as SBML prescribes.
    fn read_events(&self) -> Result<Vec<TimedAssignment>, ModelError> {
        let mut assignments = Vec::new();
        for event in Self::items(self.model, "listOfEvents") {
            let id = event.attribute("id").unwrap_or("<unnamed>");
            if Self::child(event, "delay").is_some() || Self::child(event, "priority").is_some() {
                return Err(Self::unsupported(&format!("the delay or priority of event `{id}`")));
            }
            let trigger = Self::child(event, "trigger").ok_or_else(|| ModelError::Sbml {
                reason: format!("event `{id}` has no trigger"),
            })?;
            let time = Self::trigger_time(&self.read_math(trigger, &format!("the trigger of event `{id}`"))?, id)?;
            if time <= 0.0 && trigger.attribute("initialValue") != Some("false") {
                continue;
            }
            for assignment in Self::items(event, "listOfEventAssignments") {
                let parameter = assignment.attribute("variable").unwrap_or_default().to_owned();
                let value = self.read_math(assignment, &format!("an assignment of event `{id}`"))?;
                assignments.push(TimedAssignment { time, parameter, value });
            }
        }
        Ok(assignments)
    }

    /// The firing time of a trigger of the form `time >= constant`.
    fn trigger_time(trigger: &Expression, event_id: &str) -> Result<f64, ModelError> {
        match trigger {
            Expression::Binary(BinaryOperator::GreaterEqual | BinaryOperator::Greater, left, right) => {
                match (left.as_ref(), right.as_ref()) {
                    (Expression::Symbol(symbol), Expression::Number(time)) if symbol == Expression::TIME => Ok(*time),
                    _ => Err(Self::unsupported(&format!(
                        "the trigger of event `{event_id}`, which is not `time >= constant`"
                    ))),
                }
            }
            _ => Err(Self::unsupported(&format!(
                "the trigger of event `{event_id}`, which is not `time >= constant`"
            ))),
        }
    }

    fn read_reaction(&self, node: Node) -> Result<Reaction, ModelError> {
        let id = Self::id(node)?;
        let law = Self::child(node, "kineticLaw").ok_or_else(|| ModelError::Sbml {
            reason: format!("reaction `{id}` has no kinetic law"),
        })?;
        let local_values = Self::read_local_parameters(law)?;
        let rate = self.read_math(law, &format!("the kinetic law of reaction `{id}`"))?;
        let rate = rate.substitute(&|name| local_values.get(name).map(|value| Expression::Number(*value)));
        let mut net_changes = Vec::new();
        self.add_changes(node, "listOfReactants", -1.0, &mut net_changes)?;
        self.add_changes(node, "listOfProducts", 1.0, &mut net_changes)?;
        Ok(Reaction { id, net_changes, rate })
    }

    /// Reads the `<math>` child of `parent`, with compartments replaced by
    /// their sizes.
    fn read_math(&self, parent: Node, context: &str) -> Result<Expression, ModelError> {
        let math = Self::child(parent, "math").ok_or_else(|| ModelError::Sbml {
            reason: format!("{context} has no <math>"),
        })?;
        let compartment_size = |name: &str| self.compartment_sizes.get(name).map(|size| Expression::Number(*size));
        Ok(MathmlReader::read_math(math)?.substitute(&compartment_size))
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
                .position(|declared| declared.species.id == name)
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

    fn unsupported(feature: &str) -> ModelError {
        ModelError::UnsupportedSbml {
            feature: feature.to_owned(),
        }
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

    /// Value buffer of the dimerisation model: amounts of `A` and `D`, the
    /// parameter `scale`, and the time.
    const VALUES: [f64; 4] = [10.0, 1.0, 3.0, 0.0];

    fn rate_of_first_reaction(network: &ReactionNetwork) -> Result<f64, ModelError> {
        let rate = network.compile_expression(&network.reactions()[0].rate, "test")?;
        Ok(rate.evaluate(&VALUES, &mut Vec::new()))
    }

    #[test]
    fn concentrations_become_amounts_and_local_parameters_become_numbers() -> Result<(), ModelError> {
        let network = ReactionNetwork::from_sbml(DIMERISATION)?;
        // Initial concentration 5 in volume 2 is an amount of 10.
        assert!((network.species()[0].initial_amount - 10.0).abs() < 1e-12);
        assert_eq!(network.reactions()[0].net_changes, vec![(0, -2.0), (1, 1.0)]);
        // Rate = cell * kf * scale * (A / cell)^2 = 2 * 0.5 * 3 * (10 / 2)^2 = 75.
        assert!((rate_of_first_reaction(&network)? - 75.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn exponent_notation_numbers_are_read() -> Result<(), ModelError> {
        let scaled = DIMERISATION.replace("<cn>2</cn>", r#"<cn type="e-notation">2<sep/>0</cn>"#);
        assert!((rate_of_first_reaction(&ReactionNetwork::from_sbml(&scaled)?)? - 75.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn rules_and_timed_events_are_read() -> Result<(), ModelError> {
        let math = r#"<math xmlns="http://www.w3.org/1998/Math/MathML">"#;
        let time = r#"<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time"> time </csymbol>"#;
        let additions = format!(
            r#"<listOfRules>
                 <assignmentRule variable="twice">{math}<apply><times/><cn>2</cn><ci>scale</ci></apply></math></assignmentRule>
                 <rateRule variable="D">{math}<apply><max/><ci>twice</ci><cn>0</cn></apply></math></rateRule>
               </listOfRules>
               <listOfEvents>
                 <event id="late"><trigger initialValue="false">{math}<apply><geq/>{time}<cn>5</cn></apply></math></trigger>
                   <listOfEventAssignments><eventAssignment variable="scale">{math}<cn>7</cn></math></eventAssignment></listOfEventAssignments>
                 </event>
                 <event id="at_start"><trigger initialValue="false">{math}<apply><geq/>{time}<cn>0</cn></apply></math></trigger>
                   <listOfEventAssignments><eventAssignment variable="scale">{math}<cn>4</cn></math></eventAssignment></listOfEventAssignments>
                 </event>
               </listOfEvents>
               <listOfReactions>"#
        );
        let with_rules = DIMERISATION.replace("<listOfReactions>", &additions).replace(
            r#"<parameter id="scale" value="3"/>"#,
            r#"<parameter id="scale" value="3"/><parameter id="twice"/>"#,
        );
        let network = ReactionNetwork::from_sbml(&with_rules)?;
        assert_eq!(network.assignment_rules()[0].id, "twice");
        assert_eq!(network.rate_rules()[0].species_index, 1);
        assert_eq!(network.parameters().len(), 1, "a rule target is not a parameter");
        let times: Vec<f64> = network
            .timed_assignments()
            .iter()
            .map(|assignment| assignment.time)
            .collect();
        assert_eq!(times, vec![0.0, 5.0]);
        assert!(!network.is_pure_reaction_network());
        Ok(())
    }

    #[test]
    fn state_triggered_events_are_rejected_by_name() {
        let math = r#"<math xmlns="http://www.w3.org/1998/Math/MathML">"#;
        let event = format!(
            r#"<listOfEvents><event id="threshold"><trigger>{math}<apply><gt/><ci>A</ci><cn>1</cn></apply></math></trigger>
               </event></listOfEvents><listOfReactions>"#
        );
        let error = ReactionNetwork::from_sbml(&DIMERISATION.replace("<listOfReactions>", &event))
            .expect_err("unsupported trigger");
        assert!(error.to_string().contains("threshold"), "{error}");
    }

    #[test]
    fn function_definitions_are_rejected_by_name() {
        let with_function = DIMERISATION.replace(
            "<listOfReactions>",
            r#"<listOfFunctionDefinitions><functionDefinition id="f"/></listOfFunctionDefinitions><listOfReactions>"#,
        );
        let error = ReactionNetwork::from_sbml(&with_function).expect_err("function definitions are unsupported");
        assert!(
            matches!(error, ModelError::UnsupportedSbml { ref feature } if feature == "function definitions"),
            "{error}"
        );
    }
}
