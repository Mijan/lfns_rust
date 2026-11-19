//! Symbolic arithmetic expressions over named quantities.

use crate::model_error::ModelError;

/// A binary operator. Comparisons evaluate to 1 (true) or 0 (false).
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
    /// The larger of the two.
    Max,
    /// The smaller of the two.
    Min,
    /// `left > right`
    Greater,
    /// `left >= right`
    GreaterEqual,
    /// `left < right`
    Less,
    /// `left <= right`
    LessEqual,
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
    /// A named quantity: a species, a parameter, a rule variable, or
    /// [`Expression::TIME`].
    Symbol(String),
    /// A function applied to one argument.
    Unary(UnaryFunction, Box<Expression>),
    /// An operator applied to two arguments.
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
    /// `if_true` where `condition` is non-zero, `if_false` elsewhere.
    Conditional {
        /// The condition; non-zero means true.
        condition: Box<Expression>,
        /// Value where the condition holds.
        if_true: Box<Expression>,
        /// Value where it does not.
        if_false: Box<Expression>,
    },
}

impl Expression {
    /// The symbol that stands for the simulation time.
    pub const TIME: &'static str = "time";

    /// Parses an infix formula such as `k * A / (Km + A)`. Supported are
    /// `+ - * / ^`, parentheses, and the functions `exp`, `log` (natural),
    /// `log10`, `sqrt`, `abs`, `pow(base, exponent)`, `max(a, b)`, and
    /// `min(a, b)`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Formula`] with the position of the first
    /// character that cannot be parsed.
    pub fn parse(formula: &str) -> Result<Self, ModelError> {
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

    /// Builds the expression that is `if_true` where `condition` holds and
    /// `if_false` elsewhere.
    #[must_use]
    pub fn conditional(condition: Expression, if_true: Expression, if_false: Expression) -> Self {
        Self::Conditional {
            condition: Box::new(condition),
            if_true: Box::new(if_true),
            if_false: Box::new(if_false),
        }
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
            Self::Conditional {
                condition,
                if_true,
                if_false,
            } => Self::conditional(
                condition.substitute(replacement),
                if_true.substitute(replacement),
                if_false.substitute(replacement),
            ),
        }
    }

    /// Appends the names of all symbols the expression refers to.
    pub fn collect_symbols<'a>(&'a self, symbols: &mut Vec<&'a str>) {
        match self {
            Self::Number(_) => {}
            Self::Symbol(name) => symbols.push(name),
            Self::Unary(_, argument) => argument.collect_symbols(symbols),
            Self::Binary(_, left, right) => {
                left.collect_symbols(symbols);
                right.collect_symbols(symbols);
            }
            Self::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                condition.collect_symbols(symbols);
                if_true.collect_symbols(symbols);
                if_false.collect_symbols(symbols);
            }
        }
    }
}
