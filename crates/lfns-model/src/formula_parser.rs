//! Parser for infix formulas as written in PEtab tables.
//!
//! Grammar, lowest precedence first:
//!
//! ```text
//! sum      := product (('+' | '-') product)*
//! product  := signed (('*' | '/') signed)*
//! signed   := '-' signed | power
//! power    := atom (('^' | '**') signed)?
//! atom     := number | name | name '(' sum (',' sum)* ')' | '(' sum ')'
//! ```

use crate::expression::{BinaryOperator, Expression, UnaryFunction};
use crate::model_error::ModelError;

/// Recursive-descent parser over the characters of one formula.
#[derive(Debug)]
pub(crate) struct FormulaParser<'a> {
    formula: &'a str,
    position: usize,
}

impl<'a> FormulaParser<'a> {
    /// Parses `formula` into an expression.
    pub(crate) fn parse(formula: &'a str) -> Result<Expression, ModelError> {
        let mut parser = Self { formula, position: 0 };
        let expression = parser.parse_sum()?;
        parser.skip_whitespace();
        if parser.position < formula.len() {
            return Err(parser.error("expected an operator or the end of the formula"));
        }
        Ok(expression)
    }

    fn parse_sum(&mut self) -> Result<Expression, ModelError> {
        let mut left = self.parse_product()?;
        loop {
            let operator = if self.consume("+") {
                BinaryOperator::Add
            } else if self.consume("-") {
                BinaryOperator::Subtract
            } else {
                return Ok(left);
            };
            left = Expression::binary(operator, left, self.parse_product()?);
        }
    }

    fn parse_product(&mut self) -> Result<Expression, ModelError> {
        let mut left = self.parse_signed()?;
        loop {
            let operator = if self.peek_is("**") {
                return Ok(left);
            } else if self.consume("*") {
                BinaryOperator::Multiply
            } else if self.consume("/") {
                BinaryOperator::Divide
            } else {
                return Ok(left);
            };
            left = Expression::binary(operator, left, self.parse_signed()?);
        }
    }

    fn parse_signed(&mut self) -> Result<Expression, ModelError> {
        if self.consume("-") {
            return Ok(Expression::unary(UnaryFunction::Negate, self.parse_signed()?));
        }
        self.parse_power()
    }

    fn parse_power(&mut self) -> Result<Expression, ModelError> {
        let base = self.parse_atom()?;
        if self.consume("**") || self.consume("^") {
            return Ok(Expression::binary(BinaryOperator::Power, base, self.parse_signed()?));
        }
        Ok(base)
    }

    fn parse_atom(&mut self) -> Result<Expression, ModelError> {
        if self.consume("(") {
            let inner = self.parse_sum()?;
            self.expect(")")?;
            return Ok(inner);
        }
        if let Some(number) = self.take_number() {
            return Ok(Expression::Number(number));
        }
        let name = self
            .take_name()
            .ok_or_else(|| self.error("expected a number, a name, or `(`"))?;
        if self.consume("(") {
            return self.parse_call(name);
        }
        Ok(Expression::Symbol(name.to_owned()))
    }

    /// Parses the arguments of `name(` and builds the call.
    fn parse_call(&mut self, name: &str) -> Result<Expression, ModelError> {
        let mut arguments = vec![self.parse_sum()?];
        while self.consume(",") {
            arguments.push(self.parse_sum()?);
        }
        self.expect(")")?;
        let function = match (name, arguments.len()) {
            ("exp", 1) => UnaryFunction::Exp,
            ("log" | "ln", 1) => UnaryFunction::Ln,
            ("log10", 1) => UnaryFunction::Log10,
            ("sqrt", 1) => UnaryFunction::Sqrt,
            ("abs", 1) => UnaryFunction::Abs,
            ("pow", 2) => {
                let exponent = arguments.remove(1);
                return Ok(Expression::binary(BinaryOperator::Power, arguments.remove(0), exponent));
            }
            _ => return Err(self.error("unknown function; use exp, log, log10, sqrt, abs, or pow(base, exponent)")),
        };
        Ok(Expression::unary(function, arguments.remove(0)))
    }

    /// Takes the longest prefix that is a valid floating-point literal.
    fn take_number(&mut self) -> Option<f64> {
        self.skip_whitespace();
        let rest = &self.formula[self.position..];
        if !rest.starts_with(|c: char| c.is_ascii_digit() || c == '.') {
            return None;
        }
        let is_literal_char = |(index, c): (usize, char)| {
            let after_exponent = index > 0 && matches!(rest.as_bytes()[index - 1], b'e' | b'E');
            c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || (after_exponent && (c == '+' || c == '-'))
        };
        let length = rest.char_indices().take_while(|&item| is_literal_char(item)).count();
        let (value, consumed) = (1..=length)
            .rev()
            .find_map(|n| rest[..n].parse::<f64>().ok().map(|v| (v, n)))?;
        self.position += consumed;
        Some(value)
    }

    fn take_name(&mut self) -> Option<&'a str> {
        self.skip_whitespace();
        let rest = &self.formula[self.position..];
        if !rest.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
            return None;
        }
        let length = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .count();
        self.position += length;
        Some(&rest[..length])
    }

    fn peek_is(&mut self, token: &str) -> bool {
        self.skip_whitespace();
        self.formula[self.position..].starts_with(token)
    }

    fn consume(&mut self, token: &str) -> bool {
        let found = self.peek_is(token);
        if found {
            self.position += token.len();
        }
        found
    }

    fn expect(&mut self, token: &str) -> Result<(), ModelError> {
        if self.consume(token) {
            Ok(())
        } else {
            Err(self.error(&format!("expected `{token}`")))
        }
    }

    fn skip_whitespace(&mut self) {
        let rest = &self.formula[self.position..];
        self.position += rest.len() - rest.trim_start().len();
    }

    fn error(&self, message: &str) -> ModelError {
        ModelError::Formula {
            formula: self.formula.to_owned(),
            position: self.position,
            message: message.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled_expression::CompiledExpression;

    /// Parses and evaluates a formula with `x = 3` and `y = 4`.
    fn evaluate(formula: &str) -> Result<f64, ModelError> {
        let expression = FormulaParser::parse(formula)?;
        let slot = |name: &str| ["x", "y"].iter().position(|known| *known == name);
        let compiled = CompiledExpression::compile(&expression, &slot, "test formula")?;
        Ok(compiled.evaluate(&[3.0, 4.0], &mut Vec::new()))
    }

    #[test]
    fn operators_follow_arithmetic_precedence() -> Result<(), ModelError> {
        assert!((evaluate("1 + x * y - 6 / x")? - 11.0).abs() < 1e-12);
        assert!((evaluate("-x ^ 2")? + 9.0).abs() < 1e-12);
        assert!((evaluate("2 ** x ** 2")? - 512.0).abs() < 1e-12);
        assert!((evaluate("(1 + x) * y")? - 16.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn numbers_may_use_exponent_notation() -> Result<(), ModelError> {
        assert!((evaluate("1.5e-2 * x")? - 0.045).abs() < 1e-12);
        assert!((evaluate("2e1")? - 20.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn functions_are_applied() -> Result<(), ModelError> {
        assert!((evaluate("sqrt(x * x + y * y)")? - 5.0).abs() < 1e-12);
        assert!((evaluate("log(exp(y))")? - 4.0).abs() < 1e-12);
        assert!((evaluate("pow(x, 2) + log10(100)")? - 11.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn malformed_formulas_are_rejected_with_a_position() {
        let error = FormulaParser::parse("x + * y").expect_err("the formula is malformed");
        assert!(matches!(error, ModelError::Formula { position: 4, .. }), "{error}");
        assert!(FormulaParser::parse("x y").is_err());
        assert!(FormulaParser::parse("unknown(x)").is_err());
    }

    #[test]
    fn unknown_symbols_are_reported_when_compiling() {
        assert!(matches!(evaluate("x + z"), Err(ModelError::UnknownSymbol { .. })));
    }
}
