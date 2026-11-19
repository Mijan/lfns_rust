//! Errors raised while loading a model or a PEtab problem.

use std::path::PathBuf;

/// Why a model or problem could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// A file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// A formula is not a valid expression.
    #[error("cannot parse the formula `{formula}` at character {position}: {message}")]
    Formula {
        /// The formula text.
        formula: String,
        /// Zero-based position of the problem.
        position: usize,
        /// What was expected.
        message: String,
    },

    /// A formula refers to a name that is neither a species nor a parameter.
    #[error(
        "{context} refers to `{symbol}`, which is neither a species nor a parameter of the model. \
         Declare it in the SBML model or in the PEtab parameter table."
    )]
    UnknownSymbol {
        /// Where the symbol was used.
        context: String,
        /// The unresolved name.
        symbol: String,
    },

    /// The SBML document is not well-formed or lacks a required part.
    #[error("invalid SBML: {reason}")]
    Sbml {
        /// What is wrong.
        reason: String,
    },

    /// The model uses a part of SBML that is not implemented.
    #[error(
        "the SBML model uses {feature}, which is not supported. Supported are compartments, \
         species, parameters, reactions with kinetic laws, rate rules on species, assignment \
         rules on parameters, and events that assign parameters at a fixed time."
    )]
    UnsupportedSbml {
        /// The unsupported construct.
        feature: String,
    },

    /// A PEtab file is malformed.
    #[error("invalid PEtab file {path}: {reason}")]
    Petab {
        /// The file.
        path: PathBuf,
        /// What is wrong and how to fix it.
        reason: String,
    },

    /// The problem uses a part of PEtab that is not implemented.
    #[error("the PEtab problem uses {feature}, which is not supported yet. {remedy}")]
    UnsupportedPetab {
        /// The unsupported construct.
        feature: String,
        /// How to express the problem within the supported subset.
        remedy: String,
    },
}
