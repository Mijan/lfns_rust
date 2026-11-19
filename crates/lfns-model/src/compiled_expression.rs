//! Expressions compiled to a postfix program for repeated evaluation.

use crate::expression::{BinaryOperator, Expression, UnaryFunction};
use crate::model_error::ModelError;

/// One step of a postfix program.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Instruction {
    /// Pushes a constant.
    Constant(f64),
    /// Pushes the value in the given slot.
    Load(usize),
    /// Replaces the top value by a function of it.
    Unary(UnaryFunction),
    /// Replaces the top two values by the result of an operator.
    Binary(BinaryOperator),
    /// Replaces `condition, if_true, if_false` on top of the stack by the
    /// selected branch.
    Select,
}

/// A value together with its directional derivative.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Dual {
    /// The value.
    pub value: f64,
    /// The derivative along the chosen direction.
    pub tangent: f64,
}

impl Dual {
    fn constant(value: f64) -> Self {
        Self { value, tangent: 0.0 }
    }

    /// A factor times a tangent; zero when the tangent is zero even if the
    /// factor is infinite (the derivative of `x^p` at 0 for `p < 1`).
    fn scaled_tangent(factor: f64, tangent: f64) -> f64 {
        if tangent == 0.0 { 0.0 } else { factor * tangent }
    }
}

/// An expression whose symbols are resolved to positions in a value buffer.
///
/// Evaluation walks a flat instruction list with an operand stack supplied
/// by the caller, so it neither allocates nor looks up names.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledExpression {
    instructions: Vec<Instruction>,
}

impl CompiledExpression {
    /// Compiles `expression`, resolving each symbol to a slot with `slot_of`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnknownSymbol`] if `slot_of` knows no slot for a
    /// symbol; `context` names the formula in the message.
    pub fn compile(
        expression: &Expression,
        slot_of: &impl Fn(&str) -> Option<usize>,
        context: &str,
    ) -> Result<Self, ModelError> {
        let mut instructions = Vec::new();
        Self::emit(expression, slot_of, context, &mut instructions)?;
        Ok(Self { instructions })
    }

    /// Evaluates the expression over `values`. `stack` is scratch space; it
    /// is cleared first and may be reused between calls.
    #[must_use]
    pub fn evaluate(&self, values: &[f64], stack: &mut Vec<f64>) -> f64 {
        stack.clear();
        for instruction in &self.instructions {
            match *instruction {
                Instruction::Constant(value) => stack.push(value),
                Instruction::Load(slot) => stack.push(values[slot]),
                Instruction::Unary(function) => {
                    let argument = stack.pop().unwrap_or(f64::NAN);
                    stack.push(Self::apply_unary(function, argument));
                }
                Instruction::Binary(operator) => {
                    let right = stack.pop().unwrap_or(f64::NAN);
                    let left = stack.pop().unwrap_or(f64::NAN);
                    stack.push(Self::apply_binary(operator, left, right));
                }
                Instruction::Select => {
                    let if_false = stack.pop().unwrap_or(f64::NAN);
                    let if_true = stack.pop().unwrap_or(f64::NAN);
                    let condition = stack.pop().unwrap_or(f64::NAN);
                    stack.push(if condition == 0.0 { if_false } else { if_true });
                }
            }
        }
        stack.pop().unwrap_or(f64::NAN)
    }

    /// Evaluates the expression and its derivative along a direction:
    /// `tangents` holds the derivative of every slot of `values` along that
    /// direction (forward-mode automatic differentiation).
    #[must_use]
    pub fn evaluate_dual(&self, values: &[f64], tangents: &[f64], stack: &mut Vec<Dual>) -> Dual {
        stack.clear();
        for instruction in &self.instructions {
            match *instruction {
                Instruction::Constant(value) => stack.push(Dual::constant(value)),
                Instruction::Load(slot) => stack.push(Dual {
                    value: values[slot],
                    tangent: tangents[slot],
                }),
                Instruction::Unary(function) => {
                    let argument = stack.pop().unwrap_or_default();
                    stack.push(Self::apply_unary_dual(function, argument));
                }
                Instruction::Binary(operator) => {
                    let right = stack.pop().unwrap_or_default();
                    let left = stack.pop().unwrap_or_default();
                    stack.push(Self::apply_binary_dual(operator, left, right));
                }
                Instruction::Select => {
                    let if_false = stack.pop().unwrap_or_default();
                    let if_true = stack.pop().unwrap_or_default();
                    let condition = stack.pop().unwrap_or_default();
                    stack.push(if condition.value == 0.0 { if_false } else { if_true });
                }
            }
        }
        stack.pop().unwrap_or_default()
    }

    /// Appends the postfix form of `expression`: operands first, then the
    /// operation.
    fn emit(
        expression: &Expression,
        slot_of: &impl Fn(&str) -> Option<usize>,
        context: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), ModelError> {
        match expression {
            Expression::Number(value) => instructions.push(Instruction::Constant(*value)),
            Expression::Symbol(name) => {
                let unknown = || ModelError::UnknownSymbol {
                    context: context.to_owned(),
                    symbol: name.clone(),
                };
                instructions.push(Instruction::Load(slot_of(name).ok_or_else(unknown)?));
            }
            Expression::Unary(function, argument) => {
                Self::emit(argument, slot_of, context, instructions)?;
                instructions.push(Instruction::Unary(*function));
            }
            Expression::Binary(operator, left, right) => {
                Self::emit(left, slot_of, context, instructions)?;
                Self::emit(right, slot_of, context, instructions)?;
                instructions.push(Instruction::Binary(*operator));
            }
            Expression::Conditional {
                condition,
                if_true,
                if_false,
            } => {
                for part in [condition, if_true, if_false] {
                    Self::emit(part, slot_of, context, instructions)?;
                }
                instructions.push(Instruction::Select);
            }
        }
        Ok(())
    }

    fn apply_unary(function: UnaryFunction, argument: f64) -> f64 {
        match function {
            UnaryFunction::Negate => -argument,
            UnaryFunction::Exp => argument.exp(),
            UnaryFunction::Ln => argument.ln(),
            UnaryFunction::Log10 => argument.log10(),
            UnaryFunction::Sqrt => argument.sqrt(),
            UnaryFunction::Abs => argument.abs(),
        }
    }

    fn apply_binary(operator: BinaryOperator, left: f64, right: f64) -> f64 {
        let truth = |holds: bool| if holds { 1.0 } else { 0.0 };
        match operator {
            BinaryOperator::Add => left + right,
            BinaryOperator::Subtract => left - right,
            BinaryOperator::Multiply => left * right,
            BinaryOperator::Divide => left / right,
            BinaryOperator::Power => left.powf(right),
            BinaryOperator::Max => left.max(right),
            BinaryOperator::Min => left.min(right),
            BinaryOperator::Greater => truth(left > right),
            BinaryOperator::GreaterEqual => truth(left >= right),
            BinaryOperator::Less => truth(left < right),
            BinaryOperator::LessEqual => truth(left <= right),
        }
    }

    fn apply_unary_dual(function: UnaryFunction, argument: Dual) -> Dual {
        let value = Self::apply_unary(function, argument.value);
        let factor = match function {
            UnaryFunction::Negate => -1.0,
            UnaryFunction::Exp => value,
            UnaryFunction::Ln => argument.value.recip(),
            UnaryFunction::Log10 => std::f64::consts::LOG10_E / argument.value,
            UnaryFunction::Sqrt => 0.5 / value,
            UnaryFunction::Abs => argument.value.signum(),
        };
        Dual {
            value,
            tangent: Dual::scaled_tangent(factor, argument.tangent),
        }
    }

    fn apply_binary_dual(operator: BinaryOperator, left: Dual, right: Dual) -> Dual {
        let value = Self::apply_binary(operator, left.value, right.value);
        let tangent = match operator {
            BinaryOperator::Add => left.tangent + right.tangent,
            BinaryOperator::Subtract => left.tangent - right.tangent,
            BinaryOperator::Multiply => left.tangent * right.value + left.value * right.tangent,
            BinaryOperator::Divide => (left.tangent - value * right.tangent) / right.value,
            BinaryOperator::Power => Self::power_tangent(left, right, value),
            BinaryOperator::Max => {
                if left.value >= right.value {
                    left.tangent
                } else {
                    right.tangent
                }
            }
            BinaryOperator::Min => {
                if left.value <= right.value {
                    left.tangent
                } else {
                    right.tangent
                }
            }
            BinaryOperator::Greater
            | BinaryOperator::GreaterEqual
            | BinaryOperator::Less
            | BinaryOperator::LessEqual => 0.0,
        };
        Dual { value, tangent }
    }

    /// Derivative of `base ^ exponent`: through the base, and, when the
    /// exponent varies, through the exponent.
    fn power_tangent(base: Dual, exponent: Dual, value: f64) -> f64 {
        let through_base = Dual::scaled_tangent(exponent.value * base.value.powf(exponent.value - 1.0), base.tangent);
        through_base + Dual::scaled_tangent(value * base.value.ln(), exponent.tangent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: [&str; 2] = ["x", "y"];

    fn compile(formula: &str) -> Result<CompiledExpression, ModelError> {
        let slot = |name: &str| NAMES.iter().position(|known| *known == name);
        CompiledExpression::compile(&Expression::parse(formula)?, &slot, "test formula")
    }

    #[test]
    fn directional_derivatives_match_central_differences() -> Result<(), ModelError> {
        let formulas = [
            "x * y / (2 + x)",
            "x ^ 3 - y ^ 2 * x",
            "exp(-x * y) + log(x + y)",
            "sqrt(x) * log10(y) + abs(x - y)",
            "x ^ y",
            "max(x, y) * min(x * x, y) + pow(x, 2) / (1 + pow(y, 2))",
        ];
        let (point, direction, step) = ([1.3, 0.7], [0.6, -0.8], 1e-6);
        for formula in formulas {
            let compiled = compile(formula)?;
            let shifted = |sign: f64| {
                [
                    point[0] + sign * step * direction[0],
                    point[1] + sign * step * direction[1],
                ]
            };
            let difference = (compiled.evaluate(&shifted(1.0), &mut Vec::new())
                - compiled.evaluate(&shifted(-1.0), &mut Vec::new()))
                / (2.0 * step);
            let dual = compiled.evaluate_dual(&point, &direction, &mut Vec::new());
            assert!(
                (dual.value - compiled.evaluate(&point, &mut Vec::new())).abs() < 1e-14,
                "{formula}"
            );
            assert!(
                (dual.tangent - difference).abs() < 1e-6,
                "{formula}: {} vs {difference}",
                dual.tangent
            );
        }
        Ok(())
    }

    #[test]
    fn power_at_a_zero_base_has_a_zero_derivative_when_the_base_does_not_move() -> Result<(), ModelError> {
        // d/dy of x^0.5 at x = 0 is 0, although d/dx is infinite there.
        let dual = compile("x ^ 0.5 + y")?.evaluate_dual(&[0.0, 2.0], &[0.0, 1.0], &mut Vec::new());
        assert!((dual.tangent - 1.0).abs() < 1e-14, "{}", dual.tangent);
        Ok(())
    }

    #[test]
    fn conditional_selects_the_branch_and_its_derivative() {
        let condition = Expression::binary(
            BinaryOperator::Greater,
            Expression::Symbol("x".into()),
            Expression::Number(1.0),
        );
        let doubled = Expression::binary(
            BinaryOperator::Multiply,
            Expression::Number(2.0),
            Expression::Symbol("x".into()),
        );
        let expression = Expression::conditional(condition, doubled, Expression::Symbol("y".into()));
        let slot = |name: &str| NAMES.iter().position(|known| *known == name);
        let compiled = CompiledExpression::compile(&expression, &slot, "test").expect("known symbols");
        let above = compiled.evaluate_dual(&[3.0, 10.0], &[1.0, 0.0], &mut Vec::new());
        let below = compiled.evaluate_dual(&[0.5, 10.0], &[1.0, 0.0], &mut Vec::new());
        assert_eq!((above.value, above.tangent), (6.0, 2.0));
        assert_eq!((below.value, below.tangent), (10.0, 0.0));
    }
}
