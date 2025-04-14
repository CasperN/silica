#![allow(dead_code)]

/*
TODO:
- High priority
    - Linear types
    - Lifetimes and references
    - Effects system

- Low priority
    - Custom Debug formatting to make things more succinct
    - Remove `find_duplicate_assignments` and `check_errors` as these
      are redundant with type checking and future error checking mechanisms.
    - Method for "normalizing" SSA variables, such that they are monotonically increasing
      adjacent integers.
*/

mod ast;
mod parse;
mod ssa;

use tree_sitter::{Language, Parser};

// Reference the generated language function (name depends on your grammar name)
extern "C" {
    fn tree_sitter_silica() -> Language;
}

pub fn parse(source_code: &str) -> Option<tree_sitter::Tree> {
    let language = unsafe { tree_sitter_silica() };
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .expect("Error loading Silica grammar");
    parser.parse(source_code, None) // Returns Option<Tree>
}
