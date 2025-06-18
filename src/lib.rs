#![allow(dead_code)] // TODO.

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
mod union_find;

pub use parse::parse;
