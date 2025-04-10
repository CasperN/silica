// Abstract syntax tree and type checking.

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct TypeVar(u32);

#[derive(Debug, Clone, PartialEq)]
enum LValue {
    Variable(String),
    // Deref, Field access, etc
}

#[derive(Debug, Clone, PartialEq)]
enum Expression {
    L(LValue),
    LiteralInt(i64),
    LiteralBool(bool),
    LiteralFloat(f64),
    LiteralUnit,
    If(Box<Expression>, Box<Expression>, Box<Expression>),
    Call(Box<Expression>, Vec<Expression>),
    Block(Vec<Statement>),
    Lambda(Vec<String>, Box<Expression>),
}
#[derive(Debug, Clone, PartialEq)]
enum Statement {
    Assign(LValue, Box<Expression>),
    Let {
        variable: String,
        mutable: bool,
        ty: Option<Type>,
        value: Expression,
    },
    Expression(Expression),
    Return(Expression),
    // Perform / handle
    // Ref/Deref
}

#[derive(Debug, Clone, PartialEq)]
struct FnDecl {
    name: String,
    args: Vec<(String, Type)>,
    ret: Type,
    // If no body is provided, its assumed to be external.
    body: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
enum Declaration {
    Fn(FnDecl),
    // Structs, unions, effects, traits, etc
}

#[derive(Default, Debug, Clone, PartialEq)]
struct Program(Vec<Declaration>);

#[derive(Debug, Clone, PartialEq)]
enum Type {
    Var(TypeVar),
    Int,
    Bool,
    Float,
    Unit,
    Fn(Vec<Type>, Box<Type>),
}
impl Type {
    fn contains_type_var(&self, type_var: TypeVar) -> bool {
        match self {
            Self::Int | Self::Bool | Self::Float | Self::Unit => false,
            Self::Var(self_type_var) => *self_type_var == type_var,
            Self::Fn(arg_types, ret_ty) => {
                arg_types.iter().any(|ty| ty.contains_type_var(type_var))
                    || ret_ty.contains_type_var(type_var)
            }
        }
    }
    fn substitute(&mut self, subs: &Substitutions) {
        match self {
            Self::Int | Self::Bool | Self::Float | Self::Unit => {}
            Self::Var(tv) => {
                if let Some(sub) = subs.0.get(tv) {
                    *self = sub.clone();
                }
            }
            Self::Fn(arg_types, ret_ty) => {
                for ty in arg_types.iter_mut() {
                    ty.substitute(subs);
                }
                ret_ty.substitute(subs);
            }
        }
    }
}

#[derive(Debug, Default, PartialEq)]
struct TypeContext {
    variable_types: HashMap<String, Type>,
    next_type_var: u32,
}
impl TypeContext {
    fn new() -> Self {
        Self::default()
    }
    fn get_variable_type(&self, variable: &str) -> Option<&Type> {
        self.variable_types.get(variable)
    }
    fn substitute(&mut self, subs: &Substitutions) {
        for ty in self.variable_types.values_mut() {
            ty.substitute(subs);
        }
    }
    // Variables should be inserted via the returned shadow, which
    // is a context-guard that when finished, undoes any insertions.
    // The shadow allows for the same underlying HashMap to be reused,
    // while preserving the state of the TypeContext when the shadow falls
    // out of scope.
    fn shadow(&mut self) -> ShadowTypeContext {
        ShadowTypeContext {
            type_context: self,
            shadowed: HashMap::new(),
            finished: false,
        }
    }
    // Returns a new unknown type.
    fn new_type_var(&mut self) -> Type {
        let v = self.next_type_var;
        self.next_type_var += 1;
        Type::Var(TypeVar(v))
    }
}

#[derive(Debug)]
struct ShadowTypeContext<'a> {
    type_context: &'a mut TypeContext,
    shadowed: HashMap<String, Option<Type>>,
    finished: bool,
}
impl<'a> ShadowTypeContext<'a> {
    fn insert(&mut self, name: String, ty: Type) {
        if self.shadowed.contains_key(&name) {
            // Original already saved.
            self.type_context.variable_types.insert(name, ty);
        } else {
            let original = self.type_context.variable_types.insert(name.clone(), ty);
            self.shadowed.insert(name, original);
        }
    }
    fn context_contains(&mut self, name: &str) -> bool {
        self.type_context.variable_types.contains_key(name)
    }

    fn context(&mut self) -> &mut TypeContext {
        self.type_context
    }
    fn finish(self) {}
}
impl<'a> Drop for ShadowTypeContext<'a> {
    fn drop(&mut self) {
        for (var_name, original_type) in self.shadowed.drain() {
            match original_type {
                Some(ty) => {
                    self.type_context.variable_types.insert(var_name, ty);
                }
                None => {
                    self.type_context.variable_types.remove(&var_name);
                }
            }
        }
    }
}

#[derive(Debug, Default, PartialEq)]
struct Substitutions(HashMap<TypeVar, Type>);
impl Substitutions {
    fn new() -> Self {
        Self::default()
    }
    // Applies `other` after `self`.
    fn and_then(&mut self, other: &Self) -> &mut Self {
        for ty in self.0.values_mut() {
            ty.substitute(other);
        }
        for (tv, ty) in other.0.iter() {
            self.0.insert(*tv, ty.clone());
        }
        self
    }
    fn insert(&mut self, tv: TypeVar, ty: Type) {
        self.0.insert(tv, ty);
    }
}

#[derive(PartialEq, Debug)]
enum Error {
    UnknownName(String),
    NotUnifiable(Type, Type),
    InfiniteTypeError(TypeVar, Type),
    DuplicateArgNames(Expression),
    DuplicateTopLevelName(String),
}

fn unify(left: &Type, right: &Type) -> Result<Substitutions, Error> {
    match (left, right) {
        (Type::Int, Type::Int) | (Type::Float, Type::Float) | (Type::Bool, Type::Bool) => {
            Ok(Substitutions::new())
        }
        (Type::Var(tv), ty) | (ty, Type::Var(tv)) => {
            if ty.contains_type_var(*tv) {
                return Err(Error::InfiniteTypeError(*tv, ty.clone()));
            }
            let mut subs = Substitutions::new();
            subs.insert(*tv, ty.clone());
            Ok(subs)
        }
        (Type::Fn(left_arg_types, left_ret_ty), Type::Fn(right_arg_types, right_ret_ty)) => {
            if left_arg_types.len() != right_arg_types.len() {
                return Err(Error::NotUnifiable(left.clone(), right.clone()));
            }
            let mut subs = Substitutions::new();
            for (left_arg_ty, right_arg_ty) in left_arg_types.iter().zip(right_arg_types.iter()) {
                subs.and_then(&unify(&left_arg_ty, &right_arg_ty)?);
            }
            subs.and_then(&unify(&left_ret_ty, &right_ret_ty)?);
            Ok(subs)
        }
        _ => Err(Error::NotUnifiable(left.clone(), right.clone())),
    }
}

fn infer(
    context: &mut TypeContext,
    expression: &Expression,
) -> Result<(Type, Substitutions), Error> {
    match expression {
        Expression::L(LValue::Variable(name)) => {
            if let Some(ty) = context.get_variable_type(name) {
                Ok((ty.clone(), Substitutions::new()))
            } else {
                Err(Error::UnknownName(name.to_string()))
            }
        }
        Expression::LiteralUnit => Ok((Type::Unit, Substitutions::new())),
        Expression::LiteralInt(_) => Ok((Type::Int, Substitutions::new())),
        Expression::LiteralBool(_) => Ok((Type::Bool, Substitutions::new())),
        Expression::LiteralFloat(_) => Ok((Type::Float, Substitutions::new())),
        Expression::If(cond, t_expr, f_expr) => {
            let (mut cond_ty, mut subs) = infer(context, cond)?;
            cond_ty.substitute(&subs);
            subs.and_then(&unify(&cond_ty, &Type::Bool)?);

            let (mut t_ty, t_subs) = infer(context, &t_expr)?;
            subs.and_then(&t_subs);
            t_ty.substitute(&subs);

            let (mut f_ty, f_subs) = infer(context, &f_expr)?;
            subs.and_then(&f_subs);
            f_ty.substitute(&subs);

            subs.and_then(&unify(&t_ty, &f_ty)?);

            Ok((t_ty, subs))
        }
        Expression::Call(fn_expr, arg_exprs) => {
            let mut arg_types = vec![];
            let mut subs = Substitutions::new();
            for arg_expr in arg_exprs {
                let (mut arg_type, arg_subs) = infer(context, arg_expr)?;
                subs.and_then(&arg_subs);
                arg_type.substitute(&subs);
                context.substitute(&subs);
                arg_types.push(arg_type);
            }
            let (mut fn_ty, fn_subs) = infer(context, fn_expr)?;
            subs.and_then(&fn_subs);
            fn_ty.substitute(&subs);
            context.substitute(&subs);

            let mut return_type = context.new_type_var();
            subs.and_then(&unify(
                &fn_ty,
                &Type::Fn(arg_types, Box::new(return_type.clone())),
            )?);
            fn_ty.substitute(&subs);
            return_type.substitute(&subs);
            context.substitute(&subs);

            Ok((return_type, subs))
        }
        Expression::Lambda(arg_names, body) => {
            let arg_name_set: HashSet<_> = arg_names.iter().collect();
            if arg_name_set.len() != arg_names.len() {
                return Err(Error::DuplicateArgNames(expression.clone()));
            }
            let mut shadow = context.shadow();
            for name in arg_names.iter() {
                let var_type = shadow.context().new_type_var();
                shadow.insert(name.clone(), var_type);
            }
            let res = infer(shadow.context(), body);
            shadow.finish();
            res
        }
        Expression::Block(statements) => {
            let mut subs = Substitutions::new();
            let mut shadow = context.shadow();
            let mut last_statement_type = Type::Unit;
            for statement in statements {
                match statement {
                    Statement::Expression(expr) => {
                        let (e_ty, e_subs) = infer(shadow.context(), expr)?;
                        subs.and_then(&e_subs);
                        last_statement_type = e_ty;
                    }
                    Statement::Let {
                        variable,
                        mutable,
                        ty: declared_type,
                        value,
                    } => {
                        assert!(!mutable); // TODO.
                        let context = shadow.context();
                        let (mut value_type, value_subs) = infer(context, value)?;
                        subs.and_then(&value_subs);
                        value_type.substitute(&subs);
                        if let Some(ty) = declared_type {
                            subs.and_then(&unify(ty, &value_type)?);
                        }
                        context.substitute(&subs);
                        shadow.insert(variable.clone(), value_type);
                        last_statement_type = Type::Unit;
                    }
                    Statement::Assign(_name, _expr) => todo!(),
                    Statement::Return(_expr) => todo!(),
                }
            }
            shadow.finish();
            Ok((last_statement_type, subs))
        }
    }
}

fn typecheck_program(program: &Program) -> Result<(), Error> {
    // First, load declarations into typing context.
    let mut context = TypeContext::new();
    let mut shadow = context.shadow();
    for declaration in program.0.iter() {
        match declaration {
            Declaration::Fn(FnDecl {
                name,
                args,
                ret,
                body: _,
            }) => {
                if shadow.context_contains(name.as_str()) {
                    return Err(Error::DuplicateTopLevelName(name.clone()));
                }
                let arg_types = args.iter().map(|(_, ty)| ty.clone()).collect();
                let fn_type = Type::Fn(arg_types, Box::new(ret.clone()));
                shadow.insert(name.clone(), fn_type);
            }
        }
    }
    // Second, typecheck function bodies.
    for declaration in program.0.iter() {
        match declaration {
            Declaration::Fn(FnDecl {
                name: _,
                args,
                ret,
                body,
            }) => {
                if body.is_none() {
                    continue; // External fn, take it for granted.
                }
                let body = body.as_ref().unwrap();

                // Declare a new shadow to insert fn variables.
                let mut shadow = shadow.context().shadow();
                for (arg_name, arg_ty) in args.iter() {
                    shadow.insert(arg_name.to_string(), arg_ty.clone());
                }
                let (mut ty, subs) = infer(shadow.context(), body)?;
                ty.substitute(&subs);
                unify(&ty, ret)?;
                shadow.finish();
            }
        }
    }
    shadow.finish();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_two_plus_two() {
        let program = &Program(vec![
            Declaration::Fn(FnDecl {
                name: "plus".to_string(),
                args: vec![("a".to_string(), Type::Int), ("b".to_string(), Type::Int)],
                ret: Type::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                name: "main".to_string(),
                args: vec![],
                ret: Type::Int,
                body: Some(Expression::Block(vec![
                    Statement::Let { variable: "a".to_string(), mutable: false, ty: None, value: Expression::LiteralInt(2) },
                    Statement::Let { variable: "b".to_string(), mutable: false, ty: None, value: Expression::LiteralInt(2) },
                    Statement::Let {
                        variable: "c".to_string(),
                        mutable: false,
                        ty: None, 
                        value: Expression::Call(
                            Box::new(Expression::L(LValue::Variable("plus".to_string()))),
                            vec![
                                Expression::L(LValue::Variable("a".to_string())),
                                Expression::L(LValue::Variable("b".to_string())),
                            ]
                        )
                    },
                    Statement::Expression(Expression::L(LValue::Variable("c".to_string())))
                ])),
            })
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn test_shadow_earlier_variables() {
        let program = &Program(vec![
            Declaration::Fn(FnDecl {
                name: "main".to_string(),
                args: vec![],
                ret: Type::Bool,
                body: Some(Expression::Block(vec![
                    Statement::Let { variable: "a".to_string(), mutable: false, ty: None, value: Expression::LiteralInt(2) },
                    Statement::Let { variable: "a".to_string(), mutable: false, ty: None, value: Expression::LiteralFloat(2.0) },
                    Statement::Let { variable: "a".to_string(), mutable: false, ty: None, value: Expression::LiteralBool(true) },
                    Statement::Expression(Expression::L(LValue::Variable("a".to_string())))
                ])),
            })
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn test_if_expression_unifies() {
        let program = &Program(vec![
            Declaration::Fn(FnDecl {
                name: "round".to_string(),
                args: vec![("a".to_string(), Type::Float)],
                ret: Type::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                name: "main".to_string(),
                args: vec![],
                ret: Type::Int,
                body: Some(Expression::Block(vec![
                    Statement::Let { variable: "a".to_string(), mutable: false, ty: None, value: Expression::LiteralBool(true) },
                    Statement::Let { variable: "b".to_string(), mutable: false, ty: None, value: Expression::LiteralInt(2) },
                    Statement::Let { variable: "c".to_string(), mutable: false, ty: None, value: Expression::LiteralFloat(2.0) },
                    
                    Statement::Expression(
                        Expression::If(
                            Box::new(Expression::L(LValue::Variable("a".to_string()))),
                            Box::new(Expression::L(LValue::Variable("b".to_string()))),
                            Box::new(Expression::Call(
                                Box::new(Expression::L(LValue::Variable("round".to_string()))),
                                vec![Expression::L(LValue::Variable("c".to_string()))],
                            )),
                        )
                    )
                ])),
            })
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
}
