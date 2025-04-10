#![allow(dead_code)]
use std::collections::{HashMap, HashSet};

/*
TODO:
- High priority
    - Lifetimes and references
    - Effects system

- Low priority
    - Custom Debug formatting to make things more succinct
    - Remove `find_duplicate_assignments` and `check_errors` as these
      are redundant with type checking and future error checking mechanisms.
    - Method for "normalizing" SSA variables, such that they are monotonically increasing
      adjacent integers.
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SsaVar(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BlockId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FnId(u32);

#[derive(Debug, Clone, PartialEq)]
enum Type {
    Bool,
    Int,
    Float,
    Fn { args: Vec<Type>, ret: Box<Type> },
}
impl Type {
    fn is_fn(&self) -> bool {
        matches!(self, Self::Fn { .. })
    }
    fn unwrap_fn_arg_types(&self) -> &[Type] {
        if let Self::Fn { args, .. } = self {
            args
        } else {
            panic!()
        }
    }
    fn unwrap_fn_return_type(&self) -> &Type {
        if let Self::Fn { ret, .. } = self {
            ret
        } else {
            panic!()
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Instruction {
    AssignBool(SsaVar, bool),
    AssignI64(SsaVar, i64),
    AssignF64(SsaVar, f64),
    AssignVar(SsaVar, SsaVar),
    AssignFn(SsaVar, FnId),
    // assignment, fn register, args...
    Call(SsaVar, SsaVar, Vec<SsaVar>),
    // Phi(SsaVar, HashMap<BlockId, SsaVar>),
}
impl Instruction {
    fn assigned_ssa_var(&self) -> SsaVar {
        match self {
            Instruction::AssignBool(s, _)
            | Instruction::AssignI64(s, _)
            | Instruction::AssignF64(s, _)
            | Instruction::AssignFn(s, _)
            | Instruction::AssignVar(s, _)
            | Instruction::Call(s, _, _) => *s,
        }
    }
}

// Terminators end a basic block
#[derive(Clone, Debug, PartialEq)]
enum Terminator {
    Goto(BlockId),
    CondBranch {
        on: SsaVar,
        true_target: BlockId,
        false_target: BlockId,
    },
    Return(SsaVar),
}

#[derive(Clone, Debug, PartialEq)]
struct BasicBlock {
    id: BlockId,
    instructions: Vec<Instruction>,
    terminator: Terminator,
}

#[derive(Clone, Debug, PartialEq)]
struct FnDef {
    name: String,
    arg_types: Vec<Type>,
    return_type: Type,
}
impl FnDef {
    fn as_type(&self) -> Type {
        Type::Fn {
            args: self.arg_types.clone(),
            ret: Box::new(self.return_type.clone()),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
struct Context {
    fns: HashMap<FnId, FnDef>,
}
impl Context {
    fn new() -> Self {
        Self::default()
    }
    fn add_fn(&mut self, f: FnDef) -> FnId {
        let id = FnId(self.fns.len() as u32);
        self.fns.insert(id, f);
        id
    }
    fn get_fn(&mut self, id: FnId) -> Option<&FnDef> {
        self.fns.get(&id)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct FnCode {
    args: Vec<(SsaVar, Type)>,
    blocks: Vec<BasicBlock>,
    // return type is None if it never returns.
    return_type: Option<Type>,
}

#[derive(Clone, Debug, PartialEq)]
enum InvalidTerminatorError {
    MissingSsaVar,
    MissingBlockId,
}

// TODO: This seems redundant with type checking.
#[derive(Clone, Debug, PartialEq)]
struct InvalidFnCodeErrors {
    duplicate_block_ids: HashSet<BlockId>,
    duplicate_assignments: HashSet<SsaVar>,
    invalid_terminators: Vec<(BlockId, InvalidTerminatorError)>,
}

impl FnCode {
    fn find_duplicate_block_ids(&self) -> HashSet<BlockId> {
        let mut seen = HashSet::new();
        let mut duplicates = HashSet::new();
        for block in self.blocks.iter() {
            if !seen.insert(block.id) {
                duplicates.insert(block.id);
            }
        }
        duplicates
    }
    fn find_duplicate_assignments(&self) -> HashSet<SsaVar> {
        let mut seen = HashSet::new();
        let mut duplicates = HashSet::new();
        for block in self.blocks.iter() {
            for instruction in block.instructions.iter() {
                if !seen.insert(instruction.assigned_ssa_var()) {
                    duplicates.insert(instruction.assigned_ssa_var());
                }
            }
        }
        duplicates
    }
    fn has_block(&self, id: BlockId) -> bool {
        self.blocks.iter().any(|block| block.id == id)
    }
    fn has_assigned_var(&self, v: SsaVar) -> bool {
        for block in self.blocks.iter() {
            for instruction in block.instructions.iter() {
                if instruction.assigned_ssa_var() == v {
                    return true;
                }
            }
        }
        false
    }
    fn find_invalid_terminators(&self) -> Vec<(BlockId, InvalidTerminatorError)> {
        let mut errors = Vec::new();
        for block in self.blocks.iter() {
            match block.terminator {
                Terminator::Goto(target) => {
                    if !self.has_block(target) {
                        errors.push((block.id, InvalidTerminatorError::MissingBlockId));
                    }
                }
                Terminator::CondBranch {
                    on,
                    true_target,
                    false_target,
                } => {
                    if !self.has_block(true_target) {
                        errors.push((block.id, InvalidTerminatorError::MissingBlockId));
                    }
                    if !self.has_block(false_target) {
                        errors.push((block.id, InvalidTerminatorError::MissingBlockId));
                    }
                    if !self.has_assigned_var(on) {
                        errors.push((block.id, InvalidTerminatorError::MissingSsaVar));
                    }
                }
                Terminator::Return(var) => {
                    if !self.has_assigned_var(var) {
                        errors.push((block.id, InvalidTerminatorError::MissingSsaVar));
                    }
                }
            }
        }
        errors
    }

    fn check_errors(&self) -> Result<(), InvalidFnCodeErrors> {
        let errors = InvalidFnCodeErrors {
            duplicate_assignments: self.find_duplicate_assignments(),
            duplicate_block_ids: self.find_duplicate_block_ids(),
            invalid_terminators: self.find_invalid_terminators(),
        };
        if errors.duplicate_assignments.is_empty()
            && errors.duplicate_block_ids.is_empty()
            && errors.invalid_terminators.is_empty()
        {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SsaError {
    RedefinedVar(SsaVar),
    UnsetVar(SsaVar),
    UnknownFn(FnId),
    FnTypeMismatch(SsaVar),
    FnArgLenMismatch(SsaVar),
    NonBooleanBranch(BlockId),
    UnknownBlock(BlockId),
    ReturnTypeError(BlockId),
    DuplicateBlockId(BlockId),
}

// Type checks the instruction, updating `var_types`, the types of SSA variables.
// Adds any errors to `errors`.
fn typecheck_instruction(
    instruction: &Instruction,
    context: &Context,
    var_types: &mut HashMap<SsaVar, Type>,
    errors: &mut HashSet<SsaError>,
) {
    match instruction {
        Instruction::AssignBool(assigned, _) => {
            if var_types.insert(*assigned, Type::Bool).is_some() {
                errors.insert(SsaError::RedefinedVar(*assigned));
            }
        }
        Instruction::AssignF64(assigned, _) => {
            if var_types.insert(*assigned, Type::Float).is_some() {
                errors.insert(SsaError::RedefinedVar(*assigned));
            }
        }
        Instruction::AssignI64(assigned, _) => {
            if var_types.insert(*assigned, Type::Int).is_some() {
                errors.insert(SsaError::RedefinedVar(*assigned));
            }
        }
        Instruction::AssignFn(assigned, f_id) => {
            let f = context.fns.get(f_id);
            if f.is_none() {
                errors.insert(SsaError::UnknownFn(*f_id));
                return;
            }
            if var_types.insert(*assigned, f.unwrap().as_type()).is_some() {
                errors.insert(SsaError::RedefinedVar(*assigned));
            }
        }
        Instruction::AssignVar(assigned, other) => {
            let other_ty = var_types.get(other).cloned();
            if other_ty.is_none() {
                errors.insert(SsaError::UnsetVar(*other));
                return;
            }
            if var_types.insert(*assigned, other_ty.unwrap()).is_some() {
                errors.insert(SsaError::RedefinedVar(*assigned));
            }
        }
        Instruction::Call(assigned, func, args) => {
            let fn_ty = var_types.get(func).cloned();
            if fn_ty.is_none() {
                errors.insert(SsaError::UnsetVar(*func));
                return;
            }
            let fn_ty = fn_ty.unwrap();
            if !fn_ty.is_fn() {
                errors.insert(SsaError::FnTypeMismatch(*func));
                return;
            }
            // Insert the return type assuming the args typecheck_ssa
            // so we don't emit errors for downstream uses of the assigned.
            let return_type = fn_ty.unwrap_fn_return_type().clone();
            if var_types.insert(*assigned, return_type).is_some() {
                errors.insert(SsaError::RedefinedVar(*assigned));
            }
            let arg_types = fn_ty.unwrap_fn_arg_types();
            if arg_types.len() != args.len() {
                errors.insert(SsaError::FnArgLenMismatch(*func));
                return;
            }
            for (expected_ty, var) in arg_types.iter().zip(args.iter()) {
                let arg_ty = var_types.get(var).cloned();
                if arg_ty.is_none() {
                    errors.insert(SsaError::UnsetVar(*var));
                    return;
                }
                if arg_ty.unwrap() != *expected_ty {
                    errors.insert(SsaError::FnTypeMismatch(*func));
                    return;
                }
            }
        }
    }
}

fn typecheck_terminator(
    block: &BasicBlock,
    var_types: &HashMap<SsaVar, Type>,
    function: &FnCode,
    errors: &mut HashSet<SsaError>,
) {
    match block.terminator {
        Terminator::CondBranch {
            on,
            true_target,
            false_target,
        } => {
            let on_ty = var_types.get(&on);
            if on_ty.is_none() {
                errors.insert(SsaError::UnsetVar(on));
                return;
            }
            if *on_ty.unwrap() != Type::Bool {
                errors.insert(SsaError::NonBooleanBranch(block.id));
            }
            if !function.has_block(true_target) {
                errors.insert(SsaError::UnknownBlock(true_target));
            }
            if !function.has_block(false_target) {
                errors.insert(SsaError::UnknownBlock(false_target));
            }
        }
        Terminator::Goto(block) => {
            if !function.has_block(block) {
                errors.insert(SsaError::UnknownBlock(block));
            }
        }
        Terminator::Return(var) => {
            if let Some(ty) = var_types.get(&var) {
                if Some(ty) != function.return_type.as_ref() {
                    errors.insert(SsaError::ReturnTypeError(block.id));
                }
            } else {
                errors.insert(SsaError::UnsetVar(var));
            }
        }
    }
}

// Runs each instruction in order to ensure all SsaVars are indeed assigned
// once, before use, and are used with the correct types.
fn typecheck_ssa(function: &FnCode, context: &Context) -> HashSet<SsaError> {
    let mut var_types = HashMap::new();
    let mut errors = HashSet::new();
    let mut seen_block_ids = HashSet::new();

    for (var, ty) in function.args.iter() {
        if var_types.insert(*var, ty.clone()).is_some() {
            errors.insert(SsaError::RedefinedVar(*var));
        }
    }

    for block in function.blocks.iter() {
        if !seen_block_ids.insert(block.id) {
            errors.insert(SsaError::DuplicateBlockId(block.id));
        }
        for instruction in block.instructions.iter() {
            typecheck_instruction(instruction, context, &mut var_types, &mut errors);
        }
        typecheck_terminator(block, &var_types, function, &mut errors);
    }
    errors
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_duplicate_block_ids() {
        let f = FnCode {
            args: vec![],
            return_type: Some(Type::Bool),
            blocks: vec![
                BasicBlock {
                    id: BlockId(1),
                    instructions: vec![Instruction::AssignBool(SsaVar(0), true)],
                    terminator: Terminator::Return(SsaVar(0)),
                },
                BasicBlock {
                    id: BlockId(1),
                    instructions: vec![Instruction::AssignBool(SsaVar(1), true)],
                    terminator: Terminator::Return(SsaVar(0)),
                },
            ],
        };
        assert!(!f.check_errors().unwrap_err().duplicate_block_ids.is_empty());
    }
    #[test]
    fn error_duplicate_ssa_assignments() {
        let f = FnCode {
            args: vec![],
            return_type: Some(Type::Int),
            blocks: vec![BasicBlock {
                id: BlockId(1),
                instructions: vec![
                    Instruction::AssignBool(SsaVar(0), true),
                    Instruction::AssignBool(SsaVar(0), false),
                ],
                terminator: Terminator::Return(SsaVar(0)),
            }],
        };
        assert!(!f
            .check_errors()
            .unwrap_err()
            .duplicate_assignments
            .is_empty());
    }
    #[test]
    fn error_invalid_terminator() {
        let f = FnCode {
            args: vec![],
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    instructions: vec![Instruction::AssignBool(SsaVar(0), true)],
                    terminator: Terminator::CondBranch {
                        on: SsaVar(0),
                        true_target: BlockId(500), // Error!
                        false_target: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    instructions: vec![],
                    terminator: Terminator::Goto(BlockId(20)), // Error!
                },
                BasicBlock {
                    id: BlockId(2),
                    instructions: vec![],
                    terminator: Terminator::Return(SsaVar(30)), // Error!
                },
            ],
            return_type: Some(Type::Int),
        };
        assert_eq!(f.check_errors().unwrap_err().invalid_terminators.len(), 3);
    }
    #[test]
    fn test_2_plus_2() {
        // TODO: Add arithmetic to the fn def.
        let mut context = Context::new();
        let add_i32 = context.add_fn(FnDef {
            name: "add_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Int,
        });
        let code = FnCode {
            args: vec![],
            return_type: Some(Type::Int),
            blocks: vec![BasicBlock {
                id: BlockId(0),
                instructions: vec![
                    Instruction::AssignI64(SsaVar(0), 2),
                    Instruction::AssignI64(SsaVar(1), 2),
                    Instruction::AssignFn(SsaVar(2), add_i32),
                    Instruction::Call(SsaVar(3), SsaVar(2), vec![SsaVar(0), SsaVar(1)]),
                ],
                terminator: Terminator::Return(SsaVar(3)),
            }],
        };
        assert_eq!(typecheck_ssa(&code, &context).len(), 0);
    }
    #[test]
    fn test_arg0_plus_arg1() {
        // TODO: Add arithmetic to the fn def.
        let mut context = Context::new();
        let add_i32 = context.add_fn(FnDef {
            name: "add_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Int,
        });
        let code = FnCode {
            args: vec![(SsaVar(0), Type::Int), (SsaVar(1), Type::Int)],
            blocks: vec![BasicBlock {
                id: BlockId(0),
                instructions: vec![
                    Instruction::AssignFn(SsaVar(2), add_i32),
                    Instruction::Call(SsaVar(3), SsaVar(2), vec![SsaVar(0), SsaVar(1)]),
                ],
                terminator: Terminator::Return(SsaVar(3)),
            }],
            return_type: Some(Type::Int),
        };
        assert_eq!(typecheck_ssa(&code, &context), HashSet::new());
    }

    #[test]
    fn test_add_relu_branch() {
        // TODO: Add arithmetic to the fn def.
        let mut context = Context::new();
        let add_i32 = context.add_fn(FnDef {
            name: "add_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Int,
        });
        let geq_i32 = context.add_fn(FnDef {
            name: "geq_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Bool,
        });
        let code = FnCode {
            args: vec![(SsaVar(0), Type::Int), (SsaVar(1), Type::Int)],
            return_type: Some(Type::Int),
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    instructions: vec![
                        // %3 = %0 + %1
                        Instruction::AssignFn(SsaVar(2), add_i32),
                        Instruction::Call(SsaVar(3), SsaVar(2), vec![SsaVar(0), SsaVar(1)]),
                        // %6 = %3 >= 0
                        Instruction::AssignFn(SsaVar(4), geq_i32),
                        Instruction::AssignI64(SsaVar(5), 0),
                        Instruction::Call(SsaVar(6), SsaVar(4), vec![SsaVar(3), SsaVar(5)]),
                    ],
                    terminator: Terminator::CondBranch {
                        on: SsaVar(6),
                        true_target: BlockId(1),
                        false_target: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1), // return %3
                    instructions: vec![],
                    terminator: Terminator::Return(SsaVar(3)),
                },
                BasicBlock {
                    id: BlockId(2),
                    instructions: vec![], // return 0
                    terminator: Terminator::Return(SsaVar(5)),
                },
            ],
        };
        assert_eq!(typecheck_ssa(&code, &context), HashSet::new());
    }

    #[test]
    fn error_arg_and_instruction_share_ssa_var() {
        // TODO: Add arithmetic to the fn def.
        let mut context = Context::new();
        let add_i32 = context.add_fn(FnDef {
            name: "add_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Int,
        });
        let code = FnCode {
            args: vec![(SsaVar(0), Type::Int), (SsaVar(1), Type::Int)],
            return_type: Some(Type::Int),
            blocks: vec![BasicBlock {
                id: BlockId(0),
                instructions: vec![
                    Instruction::AssignFn(SsaVar(0), add_i32), // Redefined 0!
                    Instruction::Call(SsaVar(3), SsaVar(0), vec![SsaVar(1), SsaVar(1)]),
                ],
                terminator: Terminator::Return(SsaVar(3)),
            }],
        };
        // Note that `Call` still "works" even with %0 redefined. We want
        // succinct errors so we don't make everything else fail after the first error.
        assert_eq!(
            typecheck_ssa(&code, &context),
            HashSet::from([SsaError::RedefinedVar(SsaVar(0)),])
        );
    }
}
