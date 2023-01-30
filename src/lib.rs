#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// Implements utility functions for transforming the ast used by [`swc_ecma_parser`] to a custom ast that allows for type annotations.
/// This custom ast desugars certain statements and expressions (for example, `switch` statements into if statements) to
/// make type migration easier. Not all statements (for example, `with` statements) are supported.
pub mod parse;
