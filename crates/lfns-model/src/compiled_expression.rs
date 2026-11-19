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
            }
        }
        stack.pop().unwrap_or(f64::NAN)
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
        match operator {
            BinaryOperator::Add => left + right,
            BinaryOperator::Subtract => left - right,
            BinaryOperator::Multiply => left * right,
            BinaryOperator::Divide => left / right,
            BinaryOperator::Power => left.powf(right),
        }
    }
}
