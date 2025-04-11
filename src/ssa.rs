// Silica SSA IR.
//
// After typechecking an AST, we lower to this IR.
// It is an explicitly-typed, but polymorphic, and in SSA form.
// The simplified control-flow enables linear type checking
// and lifetime analysis.
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SsaVar(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BlockId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    // BTreemap for consistent iteration order.
    Phi(SsaVar, BTreeMap<BlockId, SsaVar>),
}
impl Instruction {
    fn assigned_ssa_var(&self) -> SsaVar {
        match self {
            Instruction::AssignBool(s, _)
            | Instruction::AssignI64(s, _)
            | Instruction::AssignF64(s, _)
            | Instruction::AssignFn(s, _)
            | Instruction::AssignVar(s, _)
            | Instruction::Call(s, _, _)
            | Instruction::Phi(s, _) => *s,
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

impl FnCode {
    fn has_block(&self, id: BlockId) -> bool {
        self.blocks.iter().any(|block| block.id == id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SsaError {
    // Basic assignment and usage errors.
    RedefinedVar(SsaVar),
    UnsetVar(SsaVar),

    // Function related errors.
    UnknownFn(FnId),
    FnTypeMismatch(SsaVar),
    // Assigned var, index of mismatched arg.
    FnArgTypeMismatch(SsaVar, usize),
    FnArgLenMismatch(SsaVar),

    // Terminator errors.
    NonBooleanBranch(BlockId),
    GoToUnknownBlock(BlockId, BlockId), // terminator id, invalid id
    ReturnTypeError(BlockId),
    DuplicateBlockId(BlockId),

    // Phi related errors.
    EmptyPhi(SsaVar),
    InconsistentPhiType(SsaVar),          // Assigned var.
    PhiFromUnknownBlock(SsaVar, BlockId), // Assigned var, invalid block
}

// Type checks the instruction, updating `var_types`, the types of SSA variables.
// Adds any errors to `errors`.
fn typecheck_instruction(
    instruction: &Instruction,
    context: &Context,
    function: &FnCode,
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
            for (i, (expected_ty, var)) in arg_types.iter().zip(args.iter()).enumerate() {
                let arg_ty = var_types.get(var).cloned();
                if arg_ty.is_none() {
                    errors.insert(SsaError::UnsetVar(*var));
                    return;
                }
                if arg_ty.unwrap() != *expected_ty {
                    errors.insert(SsaError::FnArgTypeMismatch(*assigned, i));
                    return;
                }
            }
        }
        Instruction::Phi(assigned, upstreams) => {
            // Look at each of the upstream ssa vars and
            // use them to determine the type of assigned.
            let mut ty = None;
            for (source_block, var) in upstreams.iter() {
                let arg_ty = var_types.get(var);
                if arg_ty.is_none() {
                    errors.insert(SsaError::UnsetVar(*var));
                    continue;
                }
                if ty.is_none() {
                    ty = arg_ty.cloned();
                }
                if ty.as_ref() != arg_ty {
                    errors.insert(SsaError::InconsistentPhiType(*assigned));
                }
                if !function.has_block(*source_block) {
                    errors.insert(SsaError::PhiFromUnknownBlock(*assigned, *source_block));
                }
            }
            if ty.is_none() {
                errors.insert(SsaError::EmptyPhi(*assigned));
                return;
            }
            if var_types.insert(*assigned, ty.unwrap()).is_some() {
                errors.insert(SsaError::RedefinedVar(*assigned));
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
                errors.insert(SsaError::GoToUnknownBlock(block.id, true_target));
            }
            if !function.has_block(false_target) {
                errors.insert(SsaError::GoToUnknownBlock(block.id, false_target));
            }
        }
        Terminator::Goto(next_block) => {
            if !function.has_block(next_block) {
                errors.insert(SsaError::GoToUnknownBlock(block.id, next_block));
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
            typecheck_instruction(instruction, context, function, &mut var_types, &mut errors);
        }
        typecheck_terminator(block, &var_types, function, &mut errors);
    }
    errors
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
                    terminator: Terminator::Return(SsaVar(1)),
                },
            ],
        };
        assert_eq!(
            typecheck_ssa(&f, &Context::new()),
            HashSet::from([SsaError::DuplicateBlockId(BlockId(1)),])
        );
    }
    #[test]
    fn error_duplicate_ssa_assignments() {
        let f = FnCode {
            args: vec![],
            return_type: Some(Type::Bool),
            blocks: vec![BasicBlock {
                id: BlockId(1),
                instructions: vec![
                    Instruction::AssignBool(SsaVar(0), true),
                    Instruction::AssignBool(SsaVar(0), false),
                ],
                terminator: Terminator::Return(SsaVar(0)),
            }],
        };
        assert_eq!(
            typecheck_ssa(&f, &Context::new()),
            HashSet::from([SsaError::RedefinedVar(SsaVar(0)),])
        );
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
        assert_eq!(
            typecheck_ssa(&f, &Context::new()),
            HashSet::from([
                SsaError::GoToUnknownBlock(BlockId(0), BlockId(500)),
                SsaError::GoToUnknownBlock(BlockId(1), BlockId(20)),
                SsaError::UnsetVar(SsaVar(30)),
            ])
        );
    }
    #[test]
    fn two_plus_two() {
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
    fn arg0_plus_arg1() {
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
    fn sign_fn_with_phi() {
        let mut context = Context::new();
        let geq_i32 = context.add_fn(FnDef {
            name: "geq_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Bool,
        });
        let code = FnCode {
            // Sign function.
            args: vec![(SsaVar(0), Type::Int)],
            return_type: Some(Type::Int),
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    instructions: vec![
                        // %3 = %0 >= 0
                        Instruction::AssignFn(SsaVar(1), geq_i32),
                        Instruction::AssignI64(SsaVar(2), 0),
                        Instruction::Call(SsaVar(3), SsaVar(1), vec![SsaVar(0), SsaVar(2)]),
                    ],
                    terminator: Terminator::CondBranch {
                        on: SsaVar(3),
                        true_target: BlockId(1),
                        false_target: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    // True case: Set 1.
                    instructions: vec![Instruction::AssignI64(SsaVar(4), 1)],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(2),
                    // False case: Set -1.
                    instructions: vec![Instruction::AssignI64(SsaVar(5), -1)],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(3),
                    instructions: vec![Instruction::Phi(
                        SsaVar(6),
                        BTreeMap::from([(BlockId(1), SsaVar(4)), (BlockId(2), SsaVar(5))]),
                    )],
                    terminator: Terminator::Return(SsaVar(6)),
                },
            ],
        };
        assert_eq!(typecheck_ssa(&code, &context), HashSet::new());
    }

    #[test]
    fn error_phi_sign_fn_invalid_block() {
        let mut context = Context::new();
        let geq_i32 = context.add_fn(FnDef {
            name: "geq_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Bool,
        });
        let code = FnCode {
            // Sign function.
            args: vec![(SsaVar(0), Type::Int)],
            return_type: Some(Type::Int),
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    instructions: vec![
                        // %3 = %0 >= 0
                        Instruction::AssignFn(SsaVar(1), geq_i32),
                        Instruction::AssignI64(SsaVar(2), 0),
                        Instruction::Call(SsaVar(3), SsaVar(1), vec![SsaVar(0), SsaVar(2)]),
                    ],
                    terminator: Terminator::CondBranch {
                        on: SsaVar(3),
                        true_target: BlockId(1),
                        false_target: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    // True case: Set 1.
                    instructions: vec![Instruction::AssignI64(SsaVar(4), 1)],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(2),
                    // False case: Set -1.
                    instructions: vec![Instruction::AssignI64(SsaVar(5), -1)],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(3),
                    instructions: vec![Instruction::Phi(
                        SsaVar(6),
                        BTreeMap::from([(BlockId(1000), SsaVar(4)), (BlockId(2), SsaVar(5))]),
                    )],
                    terminator: Terminator::Return(SsaVar(6)),
                },
            ],
        };
        assert_eq!(
            typecheck_ssa(&code, &context),
            HashSet::from([SsaError::PhiFromUnknownBlock(SsaVar(6), BlockId(1000)),])
        );
    }
    #[test]
    fn error_phi_sign_fn_empty_phi() {
        let mut context = Context::new();
        let geq_i32 = context.add_fn(FnDef {
            name: "geq_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Bool,
        });
        let code = FnCode {
            // Sign function.
            args: vec![(SsaVar(0), Type::Int)],
            return_type: Some(Type::Int),
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    instructions: vec![
                        // %3 = %0 >= 0
                        Instruction::AssignFn(SsaVar(1), geq_i32),
                        Instruction::AssignI64(SsaVar(2), 0),
                        Instruction::Call(SsaVar(3), SsaVar(1), vec![SsaVar(0), SsaVar(2)]),
                    ],
                    terminator: Terminator::CondBranch {
                        on: SsaVar(3),
                        true_target: BlockId(1),
                        false_target: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    // True case: Set 1.
                    instructions: vec![Instruction::AssignI64(SsaVar(4), 1)],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(2),
                    // False case: Set -1.
                    instructions: vec![Instruction::AssignI64(SsaVar(5), -1)],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(3),
                    instructions: vec![Instruction::Phi(SsaVar(6), BTreeMap::new())],
                    terminator: Terminator::Return(SsaVar(6)),
                },
            ],
        };
        assert_eq!(
            typecheck_ssa(&code, &context),
            HashSet::from([SsaError::EmptyPhi(SsaVar(6)), SsaError::UnsetVar(SsaVar(6)),])
        );
    }
    #[test]
    fn error_phi_sign_fn_inconsistent_types() {
        let mut context = Context::new();
        let geq_i32 = context.add_fn(FnDef {
            name: "geq_i32".to_string(),
            arg_types: vec![Type::Int, Type::Int],
            return_type: Type::Bool,
        });
        let code = FnCode {
            // Sign function.
            args: vec![(SsaVar(0), Type::Int)],
            return_type: Some(Type::Int),
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    instructions: vec![
                        // %3 = %0 >= 0
                        Instruction::AssignFn(SsaVar(1), geq_i32),
                        Instruction::AssignI64(SsaVar(2), 0),
                        Instruction::Call(SsaVar(3), SsaVar(1), vec![SsaVar(0), SsaVar(2)]),
                    ],
                    terminator: Terminator::CondBranch {
                        on: SsaVar(3),
                        true_target: BlockId(1),
                        false_target: BlockId(2),
                    },
                },
                BasicBlock {
                    id: BlockId(1),
                    // True case: Set 1.
                    instructions: vec![Instruction::AssignI64(SsaVar(4), 1)],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(2),
                    // False case: Set -1.
                    instructions: vec![
                        Instruction::AssignF64(SsaVar(5), -1.0), // Oops float.
                    ],
                    terminator: Terminator::Goto(BlockId(3)),
                },
                BasicBlock {
                    id: BlockId(3),
                    instructions: vec![Instruction::Phi(
                        SsaVar(6),
                        BTreeMap::from([(BlockId(1), SsaVar(4)), (BlockId(2), SsaVar(5))]),
                    )],
                    terminator: Terminator::Return(SsaVar(6)),
                },
            ],
        };
        assert_eq!(
            typecheck_ssa(&code, &context),
            HashSet::from([SsaError::InconsistentPhiType(SsaVar(6)),])
        );
    }

    #[test]
    fn error_wrong_arg_type() {
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
                    Instruction::AssignF64(SsaVar(1), 2.0), // Oops, float.
                    Instruction::AssignFn(SsaVar(2), add_i32),
                    Instruction::Call(SsaVar(3), SsaVar(2), vec![SsaVar(0), SsaVar(1)]),
                ],
                terminator: Terminator::Return(SsaVar(3)),
            }],
        };
        assert_eq!(
            typecheck_ssa(&code, &context),
            HashSet::from([SsaError::FnArgTypeMismatch(SsaVar(3), 1),])
        );
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
