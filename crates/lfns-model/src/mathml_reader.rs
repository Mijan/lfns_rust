//! Reader for the subset of MathML used in SBML kinetic laws.

use roxmltree::Node;

use crate::expression::{BinaryOperator, Expression, UnaryFunction};
use crate::model_error::ModelError;

/// Converts MathML elements into [`Expression`]s.
#[derive(Debug)]
pub(crate) struct MathmlReader;

impl MathmlReader {
    /// Reads the single expression inside a `<math>` element.
    pub(crate) fn read_math(math: Node) -> Result<Expression, ModelError> {
        let content = Self::element_children(math)
            .next()
            .ok_or_else(|| Self::invalid("an empty <math> element"))?;
        Self::read_element(content)
    }

    fn read_element(element: Node) -> Result<Expression, ModelError> {
        match element.tag_name().name() {
            "ci" => Ok(Expression::Symbol(Self::text_of(element)?.to_owned())),
            "cn" => Self::read_number(element),
            "apply" => Self::read_apply(element),
            other => Err(ModelError::UnsupportedSbml {
                feature: format!("the MathML element <{other}>"),
            }),
        }
    }

    /// Reads `<cn>`, including the `e-notation` and `rational` forms that
    /// split the number at a `<sep/>` element.
    fn read_number(element: Node) -> Result<Expression, ModelError> {
        let parts: Vec<&str> = element
            .children()
            .filter_map(|child| child.text())
            .map(str::trim)
            .collect();
        let parse = |text: &str| {
            text.parse::<f64>()
                .map_err(|_| Self::invalid(&format!("the number `{text}`")))
        };
        let value = match (element.attribute("type"), parts.as_slice()) {
            (Some("e-notation"), [mantissa, exponent]) => parse(mantissa)? * 10.0_f64.powf(parse(exponent)?),
            (Some("rational"), [numerator, denominator]) => parse(numerator)? / parse(denominator)?,
            (_, [single]) => parse(single)?,
            _ => return Err(Self::invalid("a malformed <cn> element")),
        };
        Ok(Expression::Number(value))
    }

    fn read_apply(apply: Node) -> Result<Expression, ModelError> {
        let mut children = Self::element_children(apply);
        let operator = children
            .next()
            .ok_or_else(|| Self::invalid("an empty <apply> element"))?;
        let arguments: Vec<Expression> = children.map(Self::read_element).collect::<Result<_, _>>()?;
        let name = operator.tag_name().name();
        match (name, arguments.len()) {
            ("plus", 1..) => Ok(Self::fold(BinaryOperator::Add, arguments)),
            ("times", 1..) => Ok(Self::fold(BinaryOperator::Multiply, arguments)),
            ("minus", 2) => Ok(Self::fold(BinaryOperator::Subtract, arguments)),
            ("divide", 2) => Ok(Self::fold(BinaryOperator::Divide, arguments)),
            ("power", 2) => Ok(Self::fold(BinaryOperator::Power, arguments)),
            (_, 1) => Self::read_unary(name, arguments),
            _ => Err(ModelError::UnsupportedSbml {
                feature: format!("the MathML operator <{name}> with {} arguments", arguments.len()),
            }),
        }
    }

    fn read_unary(name: &str, mut arguments: Vec<Expression>) -> Result<Expression, ModelError> {
        let function = match name {
            "minus" => UnaryFunction::Negate,
            "exp" => UnaryFunction::Exp,
            "ln" => UnaryFunction::Ln,
            "log" => UnaryFunction::Log10,
            "root" => UnaryFunction::Sqrt,
            "abs" => UnaryFunction::Abs,
            other => {
                return Err(ModelError::UnsupportedSbml {
                    feature: format!("the MathML operator <{other}>"),
                });
            }
        };
        Ok(Expression::unary(function, arguments.remove(0)))
    }

    /// Combines `a, b, c` into `(a op b) op c`.
    fn fold(operator: BinaryOperator, arguments: Vec<Expression>) -> Expression {
        let mut arguments = arguments.into_iter();
        let first = arguments.next().unwrap_or(Expression::Number(f64::NAN));
        arguments.fold(first, |left, right| Expression::binary(operator, left, right))
    }

    fn element_children<'a, 'input>(node: Node<'a, 'input>) -> impl Iterator<Item = Node<'a, 'input>> {
        node.children().filter(Node::is_element)
    }

    fn text_of<'a>(element: Node<'a, '_>) -> Result<&'a str, ModelError> {
        element
            .text()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .ok_or_else(|| Self::invalid("an empty <ci> element"))
    }

    fn invalid(what: &str) -> ModelError {
        ModelError::Sbml {
            reason: format!("a kinetic law contains {what}"),
        }
    }
}
