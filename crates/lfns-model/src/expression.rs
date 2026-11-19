//! Symbolic arithmetic expressions over named quantities.

/// A binary arithmetic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    /// `left + right`
    Add,
    /// `left - right`
    Subtract,
    /// `left * right`
    Multiply,
    /// `left / right`
    Divide,
    /// `left ^ right`
    Power,
}

/// A function of one argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFunction {
    /// Arithmetic negation.
    Negate,
    /// Natural exponential.
    Exp,
    /// Natural logarithm.
    Ln,
    /// Base-10 logarithm.
    Log10,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
}

/// An arithmetic expression such as a kinetic law or an observable formula.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// A literal number.
    Number(f64),
    /// A named quantity: a species, a parameter, or a compartment.
    Symbol(String),
    /// A function applied to one argument.
    Unary(UnaryFunction, Box<Expression>),
    /// An operator applied to two arguments.
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
}

impl Expression {
    /// Parses an infix formula such as `k * A / (Km + A)`. Supported are
    /// `+ - * / ^`, parentheses, and the functions `exp`, `log` (natural),
    /// `log10`, `sqrt`, `abs`, and `pow(base, exponent)`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Formula`] with the position of the first
    /// character that cannot be parsed.
    pub fn parse(formula: &str) -> Result<Self, crate::model_error::ModelError> {
        crate::formula_parser::FormulaParser::parse(formula)
    }

    /// Builds `left <operator> right`.
    #[must_use]
    pub fn binary(operator: BinaryOperator, left: Expression, right: Expression) -> Self {
        Self::Binary(operator, Box::new(left), Box::new(right))
    }

    /// Builds `function(argument)`.
    #[must_use]
    pub fn unary(function: UnaryFunction, argument: Expression) -> Self {
        Self::Unary(function, Box::new(argument))
    }

    /// Returns a copy in which every symbol for which `replacement` yields
    /// an expression is replaced by it.
    #[must_use]
    pub fn substitute(&self, replacement: &impl Fn(&str) -> Option<Expression>) -> Self {
        match self {
            Self::Number(value) => Self::Number(*value),
            Self::Symbol(name) => replacement(name).unwrap_or_else(|| self.clone()),
            Self::Unary(function, argument) => Self::unary(*function, argument.substitute(replacement)),
            Self::Binary(operator, left, right) => {
                Self::binary(*operator, left.substitute(replacement), right.substitute(replacement))
            }
        }
    }

    /// Returns the partial derivative with respect to `symbol`, simplified
    /// so that terms that vanish identically disappear.
    #[must_use]
    pub fn differentiate(&self, symbol: &str) -> Self {
        match self {
            Self::Number(_) => Self::Number(0.0),
            Self::Symbol(name) => Self::Number(if name == symbol { 1.0 } else { 0.0 }),
            Self::Unary(function, argument) => Self::differentiate_unary(*function, argument, symbol),
            Self::Binary(operator, left, right) => Self::differentiate_binary(*operator, left, right, symbol),
        }
    }

    /// Whether the expression is the literal zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Number(value) if *value == 0.0)
    }

    /// Chain rule for `function(argument)`.
    fn differentiate_unary(function: UnaryFunction, argument: &Self, symbol: &str) -> Self {
        let inner = argument.differentiate(symbol);
        let argument = || argument.clone();
        let outer = match function {
            UnaryFunction::Negate => Self::Number(-1.0),
            UnaryFunction::Exp => Self::unary(UnaryFunction::Exp, argument()),
            UnaryFunction::Ln => Self::quotient(Self::Number(1.0), argument()),
            UnaryFunction::Log10 => Self::quotient(Self::Number(std::f64::consts::LOG10_E), argument()),
            UnaryFunction::Sqrt => Self::quotient(Self::Number(0.5), Self::unary(UnaryFunction::Sqrt, argument())),
            UnaryFunction::Abs => Self::quotient(argument(), Self::unary(UnaryFunction::Abs, argument())),
        };
        Self::product(outer, inner)
    }

    /// Sum, product, quotient, and power rules.
    fn differentiate_binary(operator: BinaryOperator, left: &Self, right: &Self, symbol: &str) -> Self {
        let (d_left, d_right) = (left.differentiate(symbol), right.differentiate(symbol));
        let (left, right) = (|| left.clone(), || right.clone());
        match operator {
            BinaryOperator::Add => Self::sum(d_left, d_right),
            BinaryOperator::Subtract => Self::sum(d_left, Self::product(Self::Number(-1.0), d_right)),
            BinaryOperator::Multiply => Self::sum(Self::product(d_left, right()), Self::product(left(), d_right)),
            BinaryOperator::Divide => {
                let from_denominator = Self::quotient(Self::product(left(), d_right), Self::product(right(), right()));
                Self::sum(
                    Self::quotient(d_left, right()),
                    Self::product(Self::Number(-1.0), from_denominator),
                )
            }
            BinaryOperator::Power if d_right.is_zero() => {
                let reduced = Self::binary(BinaryOperator::Subtract, right(), Self::Number(1.0));
                Self::product(
                    Self::product(right(), Self::binary(BinaryOperator::Power, left(), reduced)),
                    d_left,
                )
            }
            BinaryOperator::Power => {
                let through_exponent = Self::product(d_right, Self::unary(UnaryFunction::Ln, left()));
                let through_base = Self::quotient(Self::product(right(), d_left), left());
                Self::product(
                    Self::binary(BinaryOperator::Power, left(), right()),
                    Self::sum(through_exponent, through_base),
                )
            }
        }
    }

    /// `left + right`, dropping a zero summand.
    fn sum(left: Self, right: Self) -> Self {
        match (left.is_zero(), right.is_zero()) {
            (true, _) => right,
            (_, true) => left,
            _ => Self::binary(BinaryOperator::Add, left, right),
        }
    }

    /// `left * right`, collapsing to zero if a factor is zero and dropping
    /// a factor of one.
    fn product(left: Self, right: Self) -> Self {
        let is_one = |expression: &Self| matches!(expression, Self::Number(value) if (*value - 1.0).abs() == 0.0);
        if left.is_zero() || right.is_zero() {
            Self::Number(0.0)
        } else if is_one(&left) {
            right
        } else if is_one(&right) {
            left
        } else {
            Self::binary(BinaryOperator::Multiply, left, right)
        }
    }

    /// `numerator / denominator`, collapsing to zero for a zero numerator.
    fn quotient(numerator: Self, denominator: Self) -> Self {
        if numerator.is_zero() {
            numerator
        } else {
            Self::binary(BinaryOperator::Divide, numerator, denominator)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::compiled_expression::CompiledExpression;
    use crate::formula_parser::FormulaParser;
    use crate::model_error::ModelError;

    /// Evaluates `formula` and its derivatives with respect to `x` and `y`
    /// at the given point.
    fn value_and_gradient(formula: &str, point: [f64; 2]) -> Result<(f64, [f64; 2]), ModelError> {
        let expression = FormulaParser::parse(formula)?;
        let slot = |name: &str| ["x", "y"].iter().position(|known| *known == name);
        let evaluate = |e: &super::Expression| -> Result<f64, ModelError> {
            Ok(CompiledExpression::compile(e, &slot, "test")?.evaluate(&point, &mut Vec::new()))
        };
        let gradient = [
            evaluate(&expression.differentiate("x"))?,
            evaluate(&expression.differentiate("y"))?,
        ];
        Ok((evaluate(&expression)?, gradient))
    }

    #[test]
    fn derivatives_match_central_differences() -> Result<(), ModelError> {
        let formulas = [
            "x * y / (2 + x)",
            "x ^ 3 - y ^ 2 * x",
            "exp(-x * y) + log(x + y)",
            "sqrt(x) * log10(y) + abs(x - y)",
            "x ^ y",
            "pow(x, 2) / (1 + pow(y, 2))",
        ];
        let (point, step) = ([1.3, 0.7], 1e-6);
        for formula in formulas {
            let (_, gradient) = value_and_gradient(formula, point)?;
            for dimension in 0..2 {
                let (mut above, mut below) = (point, point);
                above[dimension] += step;
                below[dimension] -= step;
                let difference =
                    (value_and_gradient(formula, above)?.0 - value_and_gradient(formula, below)?.0) / (2.0 * step);
                assert!(
                    (gradient[dimension] - difference).abs() < 1e-6,
                    "{formula}, dimension {dimension}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn derivative_with_respect_to_an_absent_symbol_is_zero() -> Result<(), ModelError> {
        let expression = FormulaParser::parse("3 * x ^ 2 + exp(x)")?;
        assert!(expression.differentiate("y").is_zero());
        Ok(())
    }
}
