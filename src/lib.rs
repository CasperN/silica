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
#[allow(dead_code)]
mod ssa;
mod union_find;

pub fn compile(source: &str) {
    let mut errors = vec![];
    let program = parse::parse_ast_program(source, &mut errors);
    if !errors.is_empty() {
        for e in errors {
            println!("ParseError: {e:?}");
        }
        return;
    }
    match ast::typecheck_program(&mut program.unwrap()) {
        Ok(()) => println!("it typechecks!"),
        Err(e) => println!("Type error: {e:?}"),
    }
}
