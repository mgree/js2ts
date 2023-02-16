#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// Implements utility functions for transforming the ast used by [`swc_ecma_parser`] to a custom ast that allows for type annotations.
/// This custom ast desugars certain statements and expressions (for example, `switch` statements into if statements) to
/// make type migration easier. Not all statements (for example, `with` statements) are supported.
pub mod parse;

/// Implements functions for constraint generation and generating a model for the resulting constraints for the types using [`z3`].
pub mod check;

/// Private crate for testing utilities.
#[cfg(test)]
pub(crate) mod testing;

/// Implements functions for validating the migrated [`parse::Ast`].
pub mod typecheck;
