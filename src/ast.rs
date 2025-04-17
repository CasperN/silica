// Abstract syntax tree and type checking.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct TypeVar(u32);

#[derive(Debug, Clone, PartialEq)]
pub enum LValue {
    Variable(String),
    // Deref, Field access, etc
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    L(LValue, Option<Type>),
    LiteralInt(i64),
    LiteralBool(bool),
    LiteralFloat(f64),
    LiteralUnit,
    If {
        condition: Box<Expression>,
        true_expr: Box<Expression>,
        false_expr: Box<Expression>,
        ty: Option<Type>,
    },
    Call {
        fn_expr: Box<Expression>,
        arg_exprs: Vec<Expression>,
        ty: Option<Type>,
    },
    Block {
        statements: Vec<Statement>,
        ty: Option<Type>,
    },
    Lambda {
        bindings: Vec<SoftBinding>,
        body: Box<Expression>,
        ty: Option<Type>,
    },
}
impl Expression {
    fn init_type_var(&mut self, context: &mut TypeContext) -> &mut Self {
        match self {
            Self::LiteralInt(_)
            | Self::LiteralBool(_)
            | Self::LiteralFloat(_)
            | Self::LiteralUnit => {}
            Self::L(_, ty)
            | Self::If { ty, .. }
            | Self::Block { ty, .. }
            | Self::Call { ty, .. }
            | Self::Lambda { ty, .. } => {
                if ty.is_none() {
                    *ty = Some(context.new_type_var())
                }
            }
        }
        self
    }
    // Gets the type of the expression, assigning a new variable from the
    // context if one hasn't been created yet.
    fn unwrap_type(&mut self) -> &Type {
        match self {
            Self::LiteralInt(_) => &Type::Int,
            Self::LiteralBool(_) => &Type::Bool,
            Self::LiteralFloat(_) => &Type::Float,
            Self::LiteralUnit => &Type::Unit,
            Self::L(_, ty)
            | Self::If { ty, .. }
            | Self::Block { ty, .. }
            | Self::Call { ty, .. }
            | Self::Lambda { ty, .. } => ty
                .as_ref()
                .expect("unwrap_type called before init_type_var"),
        }
    }
    fn substitute(&mut self, subs: &Substitutions) -> &mut Self {
        match self {
            Self::L(_, _)
            | Self::LiteralInt(_)
            | Self::LiteralBool(_)
            | Self::LiteralFloat(_)
            | Self::LiteralUnit => {}
            Self::If { ty, .. }
            | Self::Block { ty, .. }
            | Self::Call { ty, .. }
            | Self::Lambda { ty, .. } => {
                ty.as_mut()
                    .expect(
                        "Substituting type vars in an expression before init_type_var was called.",
                    )
                    .substitute(subs);
            }
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Assign(LValue, Expression),
    Let {
        binding: SoftBinding,
        value: Expression,
    },
    Expression(Expression),
    Return(Expression),
    // Perform / handle
    // Ref / Deref
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedBinding {
    pub name: String,
    pub ty: Type,
    pub mutable: bool,
}
impl TypedBinding {
    pub fn new(name: impl Into<String>, ty: impl Into<Type>, mutable: bool) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
            mutable,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SoftBinding {
    pub name: String,
    pub ty: Option<Type>,
    pub mutable: bool,
}
impl SoftBinding {
    pub fn new(name: impl Into<String>, ty: impl Into<Option<Type>>, mutable: bool) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
            mutable,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub name: String,
    pub args: Vec<TypedBinding>,
    pub return_type: Type,
    // If no body is provided, its assumed to be external.
    pub body: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Fn(FnDecl),
    // Structs, unions, effects, traits, etc
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Program(pub Vec<Declaration>);

// Var(TypeVar) cannot be constructed outside of this file.
#[allow(private_interfaces)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Type {
    Var(TypeVar),
    Int,
    Bool,
    Float,
    Unit,
    Fn(Vec<Type>, Box<Type>),
}
impl Default for Type {
    fn default() -> Self {
        Type::Unit
    }
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

#[derive(PartialEq, Debug)]
struct VariableInfo {
    ty: Type,
    mutable: bool,
}

// TODO: Move TypeContext, ShadowTypeContext, and friends into a sub-module to protect field access.
#[derive(Debug, Default, PartialEq)]
struct TypeContext {
    variables: HashMap<String, VariableInfo>,
    next_type_var: u32,
    return_type: Type,
}
impl TypeContext {
    fn new() -> Self {
        Self::default()
    }
    fn variable_info(&self, variable: &str) -> Option<&VariableInfo> {
        self.variables.get(variable)
    }
    fn substitute(&mut self, subs: &Substitutions) {
        for var in self.variables.values_mut() {
            var.ty.substitute(subs);
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
            shadowed_variables: HashMap::new(),
            finished: false,
            shadowed_return_type: None,
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
    shadowed_variables: HashMap<String, Option<VariableInfo>>,
    shadowed_return_type: Option<Type>,
    finished: bool,
}
impl<'a> ShadowTypeContext<'a> {
    fn insert(&mut self, name: String, ty: Type, mutable: bool) {
        let info = VariableInfo { ty, mutable };
        if let Entry::Vacant(entry) = self.shadowed_variables.entry(name.clone()) {
            let original = self.type_context.variables.insert(name, info);
            entry.insert(original);
        } else {
            // The original was already saved in `shadowed_variables`,
            // no need to touch that.
            self.type_context.variables.insert(name, info);
        }
    }
    fn insert_binding(&mut self, binding: TypedBinding) {
        let TypedBinding { name, ty, mutable } = binding;
        self.insert(name, ty, mutable);
    }
    fn insert_soft_binding(&mut self, binding: SoftBinding) {
        let SoftBinding { name, ty, mutable } = binding;
        let ty = ty.unwrap_or_else(|| self.context().new_type_var());
        self.insert(name, ty, mutable);
    }
    fn set_return_type(&mut self, ty: Type) {
        if self.shadowed_return_type.is_some() {
            panic!("Return type shadowed multiple times.");
        }
        self.shadowed_return_type = Some(ty);
        core::mem::swap(
            self.shadowed_return_type.as_mut().unwrap(),
            &mut self.type_context.return_type,
        );
    }

    fn context_contains(&mut self, name: &str) -> bool {
        self.type_context.variables.contains_key(name)
    }

    fn context(&mut self) -> &mut TypeContext {
        self.type_context
    }
    fn finish(self) {}
}
impl<'a> Drop for ShadowTypeContext<'a> {
    fn drop(&mut self) {
        for (var_name, original_type) in self.shadowed_variables.drain() {
            match original_type {
                Some(ty) => {
                    self.type_context.variables.insert(var_name, ty);
                }
                None => {
                    self.type_context.variables.remove(&var_name);
                }
            }
        }
        if let Some(ty) = self.shadowed_return_type.take() {
            self.type_context.return_type = ty;
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
    InfiniteType(TypeVar, Type),
    DuplicateArgNames(Expression),
    DuplicateTopLevelName(String),
    AssignToImmutableBinding(String),
    TopLevelFnArgsMustBeTyped(String),
}

fn unify(left: &Type, right: &Type) -> Result<Substitutions, Error> {
    match (left, right) {
        (Type::Int, Type::Int)
        | (Type::Float, Type::Float)
        | (Type::Bool, Type::Bool)
        | (Type::Unit, Type::Unit) => Ok(Substitutions::new()),
        (Type::Var(tv), ty) | (ty, Type::Var(tv)) => {
            if ty.contains_type_var(*tv) {
                return Err(Error::InfiniteType(*tv, ty.clone()));
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
                subs.and_then(&unify(left_arg_ty, right_arg_ty)?);
            }
            subs.and_then(&unify(left_ret_ty, right_ret_ty)?);
            Ok(subs)
        }
        _ => Err(Error::NotUnifiable(left.clone(), right.clone())),
    }
}

fn infer(context: &mut TypeContext, expression: &mut Expression) -> Result<Substitutions, Error> {
    expression.init_type_var(context);
    match expression {
        Expression::L(LValue::Variable(name), ty) => {
            let ty = ty.get_or_insert_with(|| context.new_type_var());
            if let Some(info) = context.variable_info(name) {
                unify(ty, &info.ty)
            } else {
                Err(Error::UnknownName(name.to_string()))
            }
        }
        Expression::LiteralUnit => Ok(Substitutions::new()),
        Expression::LiteralInt(_) => Ok(Substitutions::new()),
        Expression::LiteralBool(_) => Ok(Substitutions::new()),
        Expression::LiteralFloat(_) => Ok(Substitutions::new()),
        Expression::If {
            condition,
            true_expr,
            false_expr,
            ty: _,
        } => {
            let mut subs = infer(context, condition)?;
            let cond_ty = condition.unwrap_type();
            subs.and_then(&unify(cond_ty, &Type::Bool)?);
            condition.substitute(&subs);

            let t_subs = infer(context, true_expr)?;
            subs.and_then(&t_subs);
            true_expr.substitute(&subs);

            let f_subs = infer(context, false_expr)?;
            subs.and_then(&f_subs);
            false_expr.substitute(&subs);

            let t_ty = true_expr.unwrap_type();
            let f_ty = false_expr.unwrap_type();
            subs.and_then(&unify(t_ty, f_ty)?);

            Ok(subs)
        }
        Expression::Call {
            fn_expr,
            arg_exprs,
            ty: return_type,
        } => {
            let mut arg_types = vec![];
            let mut subs = Substitutions::new();
            for arg_expr in arg_exprs {
                let arg_subs = infer(context, arg_expr)?;
                subs.and_then(&arg_subs);
                arg_expr.substitute(&subs);
                context.substitute(&subs);
                arg_types.push(arg_expr.unwrap_type().clone());
            }
            let fn_subs = infer(context, fn_expr)?;
            subs.and_then(&fn_subs);

            fn_expr.substitute(&subs);
            context.substitute(&subs);

            let return_type = return_type.as_mut().unwrap();
            subs.and_then(&unify(
                fn_expr.unwrap_type(),
                &Type::Fn(arg_types, Box::new(return_type.clone())),
            )?);
            fn_expr.substitute(&subs);
            return_type.substitute(&subs);
            context.substitute(&subs);

            Ok(subs)
        }
        Expression::Lambda {
            bindings,
            body,
            ty: lambda_type,
        } => {
            let arg_name_set: HashSet<_> = bindings.iter().map(|b| b.name.as_str()).collect();
            if arg_name_set.len() != bindings.len() {
                return Err(Error::DuplicateArgNames(expression.clone()));
            }
            let mut lambda_return_type = context.new_type_var();
            let mut shadow = context.shadow();
            shadow.set_return_type(lambda_return_type.clone());

            for binding in bindings.iter() {
                shadow.insert_soft_binding(binding.clone());
            }
            let mut subs = infer(shadow.context(), body)?;
            subs.and_then(&unify(body.unwrap_type(), &lambda_return_type)?);

            body.substitute(&subs);
            shadow.context().substitute(&subs);
            lambda_return_type.substitute(&subs);

            let mut arg_types = Vec::new();
            for binding in bindings.iter() {
                let arg_type = shadow
                    .context()
                    .variable_info(binding.name.as_str())
                    .unwrap()
                    .ty
                    .clone();
                arg_types.push(arg_type);
            }
            if lambda_type.is_none() {
                *lambda_type = Some(shadow.context().new_type_var());
            }
            let lambda_type = lambda_type.as_mut().unwrap();
            subs.and_then(&unify(
                lambda_type,
                &Type::Fn(arg_types, Box::new(lambda_return_type)),
            )?);
            lambda_type.substitute(&subs);
            shadow.finish();
            Ok(subs)
        }
        Expression::Block {
            statements,
            ty: block_ty,
        } => {
            let mut subs = Substitutions::new();
            let mut shadow = context.shadow();
            let mut last_statement_type = Type::Unit;
            for statement in statements {
                match statement {
                    Statement::Expression(expr) => {
                        let e_subs = infer(shadow.context(), expr)?;
                        subs.and_then(&e_subs);
                        last_statement_type =
                            expr.unwrap_type().clone();
                    }
                    Statement::Let { binding, value } => {
                        let context = shadow.context();
                        let value_subs = infer(context, value)?;
                        subs.and_then(&value_subs);
                        value.substitute(&subs);
                        let value_type = value.unwrap_type().clone();
                        if let Some(binding_ty) = &binding.ty {
                            subs.and_then(&(unify(binding_ty, &value_type))?);
                        }
                        context.substitute(&subs);
                        value.substitute(&subs);
                        shadow.insert(binding.name.clone(), value_type.clone(), binding.mutable);
                        last_statement_type = Type::Unit;
                    }
                    Statement::Assign(LValue::Variable(name), expr) => {
                        let context = shadow.context();
                        let e_subs = infer(context, expr)?;
                        subs.and_then(&e_subs);
                        let expr_ty = expr
                            .substitute(&subs)
                            .unwrap_type()
                            .clone();

                        // Check whether the expession can be assigned to the variable.
                        let variable_info = context.variable_info(name);
                        if variable_info.is_none() {
                            return Err(Error::UnknownName(name.to_string()));
                        }
                        let variable_info = variable_info.unwrap();
                        if !variable_info.mutable {
                            return Err(Error::AssignToImmutableBinding(name.to_string()));
                        }
                        subs.and_then(&unify(&expr_ty, &variable_info.ty)?);
                        context.substitute(&subs);
                        expr.substitute(&subs);
                        last_statement_type = Type::Unit;
                    }
                    Statement::Return(expr) => {
                        let e_subs = infer(shadow.context(), expr)?;
                        subs.and_then(&e_subs);
                        subs.and_then(&dbg!(unify(
                            expr.unwrap_type(),
                            &shadow.context().return_type
                        ))?);
                        last_statement_type = expr.unwrap_type().clone();
                        // TODO: Probably should issue a warning for unreachable statements.
                    }
                }
            }
            subs.and_then(&unify(block_ty.as_ref().unwrap(), &last_statement_type)?);
            shadow.finish();
            Ok(subs)
        }
    }
}

fn typecheck_program(program: &mut Program) -> Result<(), Error> {
    // First, load declarations into typing context.
    let mut context = TypeContext::new();
    let mut shadow = context.shadow();
    for declaration in program.0.iter() {
        match declaration {
            Declaration::Fn(FnDecl {
                name,
                args,
                return_type,
                body: _,
            }) => {
                if shadow.context_contains(name.as_str()) {
                    return Err(Error::DuplicateTopLevelName(name.clone()));
                }
                let mut arg_types = vec![];
                for binding in args.iter() {
                    arg_types.push(binding.ty.clone());
                }
                let fn_type = Type::Fn(arg_types, Box::new(return_type.clone()));
                shadow.insert(name.clone(), fn_type, false); // TODO: args need to declare mutability.
            }
        }
    }
    // Second, typecheck function bodies.
    for declaration in program.0.iter_mut() {
        match declaration {
            Declaration::Fn(FnDecl {
                name: _,
                args,
                return_type,
                body,
            }) => {
                if body.is_none() {
                    continue; // External fn, take it for granted.
                }
                let body = body.as_mut().unwrap();

                // Declare a new shadow to insert fn variables.
                let mut shadow = shadow.context().shadow();
                for binding in args.iter() {
                    shadow.insert_binding(binding.clone()); // TODO: variables need to declare mutaiblity.
                }
                shadow.set_return_type(return_type.clone());
                infer(shadow.context(), body)?;
                unify(return_type, body.unwrap_type())?;
                shadow.finish();
            }
        }
    }
    shadow.finish();
    Ok(())
}

impl From<i64> for Expression {
    fn from(value: i64) -> Self {
        Expression::LiteralInt(value)
    }
}
impl From<f64> for Expression {
    fn from(value: f64) -> Self {
        Expression::LiteralFloat(value)
    }
}
impl From<bool> for Expression {
    fn from(value: bool) -> Self {
        Expression::LiteralBool(value)
    }
}
impl From<()> for Expression {
    fn from(_: ()) -> Self {
        Expression::LiteralUnit
    }
}
impl From<&str> for Statement {
    fn from(value: &str) -> Self {
        Statement::Expression(Expression::L(LValue::Variable(value.to_string()), None))
    }
}
impl From<i64> for Statement {
    fn from(value: i64) -> Self {
        Statement::Expression(Expression::LiteralInt(value))
    }
}
impl From<&str> for Expression {
    fn from(value: &str) -> Self {
        Expression::L(LValue::Variable(value.to_string()), None)
    }
}
impl From<&str> for LValue {
    fn from(value: &str) -> Self {
        LValue::Variable(value.to_string())
    }
}
impl From<Expression> for Statement {
    fn from(expr: Expression) -> Self {
        Statement::Expression(expr)
    }
}

#[cfg(test)]
pub mod test_helpers {
    use super::*;

    pub fn let_stmt(
        name: &str,
        ty: impl Into<Option<Type>>,
        mutable: bool,
        value: impl Into<Expression>,
    ) -> Statement {
        Statement::Let {
            binding: SoftBinding {
                name: name.to_string(),
                ty: ty.into(),
                mutable,
            },
            value: value.into(),
        }
    }
    pub fn return_stmt(expr: impl Into<Expression>) -> Statement {
        Statement::Return(expr.into())
    }

    pub fn call_expr(f: impl Into<Expression>, arg_exprs: Vec<Expression>) -> Expression {
        Expression::Call {
            fn_expr: Box::new(f.into()),
            arg_exprs,
            ty: None,
        }
    }
    pub fn assign_stmt(left: impl Into<LValue>, right: impl Into<Expression>) -> Statement {
        Statement::Assign(left.into(), right.into())
    }
    pub fn if_expr(
        cond: impl Into<Expression>,
        true_expr: impl Into<Expression>,
        false_expr: impl Into<Expression>,
    ) -> Expression {
        Expression::If {
            condition: Box::new(cond.into()),
            true_expr: Box::new(true_expr.into()),
            false_expr: Box::new(false_expr.into()),
            ty: None,
        }
    }
    pub fn block_expr(statements: Vec<Statement>) -> Expression {
        Expression::Block {
            statements,
            ty: None,
        }
    }
    pub fn lambda_expr(bindings: Vec<SoftBinding>, body: impl Into<Expression>) -> Expression {
        Expression::Lambda {
            bindings,
            body: Box::new(body.into()),
            ty: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::*;
    use super::*;
    #[test]
    fn two_plus_two() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::Int, false),
                    TypedBinding::new("b", Type::Int, false),
                ],
                return_type: Type::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                name: "main".to_string(),
                args: vec![],
                return_type: Type::Int,
                body: Some(block_expr(vec![
                    let_stmt("a", None, false, 2),
                    let_stmt("b", None, false, 2),
                    let_stmt(
                        "c",
                        None,
                        false,
                        call_expr("plus", vec!["a".into(), "b".into()]),
                    ),
                    "c".into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn shadow_earlier_variables() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            args: vec![],
            return_type: Type::Bool,
            body: Some(block_expr(vec![
                let_stmt("a", None, false, 2),
                let_stmt("a", None, false, 2.0),
                let_stmt("a", None, false, true),
                "a".into(),
            ])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn assign_to_mutable_binding() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            args: vec![TypedBinding::new("a", Type::Int, true)], // mutable.
            return_type: Type::Unit,
            body: Some(block_expr(vec![assign_stmt("a", 3)])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn error_assign_wrong_type_to_mutable_binding() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            args: vec![],
            return_type: Type::Unit,
            body: Some(block_expr(vec![
                let_stmt("a", None, true, 2),
                assign_stmt("a", ()),
            ])),
        })]);
        assert_eq!(
            typecheck_program(program),
            Err(Error::NotUnifiable(Type::Unit, Type::Int))
        );
    }
    #[test]
    fn error_assign_to_immutable_binding() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            args: vec![],
            return_type: Type::Unit,
            body: Some(block_expr(vec![
                let_stmt("a", None, false, 2),
                assign_stmt("a", 3),
            ])),
        })]);
        assert_eq!(
            typecheck_program(program),
            Err(Error::AssignToImmutableBinding("a".to_string()))
        );
    }
    #[test]
    fn if_expression_unifies() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                name: "round".to_string(),
                args: vec![TypedBinding::new("a", Type::Float, false)],
                return_type: Type::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                name: "main".to_string(),
                args: vec![],
                return_type: Type::Int,
                body: Some(block_expr(vec![
                    let_stmt("a", None, false, true),
                    let_stmt("b", None, false, 2),
                    let_stmt("c", None, false, 2.0),
                    Statement::Expression(if_expr("a", "b", call_expr("round", vec!["c".into()]))),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn use_lambda() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::Int, false),
                    TypedBinding::new("b", Type::Int, false),
                ],
                return_type: Type::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                name: "main".to_string(),
                args: vec![],
                return_type: Type::Int,
                body: Some(block_expr(vec![
                    let_stmt(
                        "plus_two",
                        None,
                        false,
                        lambda_expr(
                            vec![SoftBinding::new("x", None, false)],
                            call_expr("plus", vec![2.into(), "x".into()]),
                        ),
                    ),
                    let_stmt("b", None, true, 2),
                    assign_stmt("b", call_expr("plus_two", vec!["b".into()])),
                    "b".into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn return_lambda() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::Int, false),
                    TypedBinding::new("b", Type::Int, false),
                ],
                return_type: Type::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                name: "main".to_string(),
                args: vec![],
                return_type: Type::Fn(vec![Type::Int], Box::new(Type::Int)),
                body: Some(block_expr(vec![
                    let_stmt(
                        "plus_two",
                        None,
                        false,
                        lambda_expr(
                            vec![SoftBinding::new("x", None, false)],
                            call_expr("plus", vec![2.into(), "x".into()]),
                        ),
                    ),
                    "plus_two".into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn return_in_lambda() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::Int, false),
                    TypedBinding::new("b", Type::Int, false),
                ],
                return_type: Type::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                name: "main".to_string(),
                args: vec![],
                return_type: Type::Fn(vec![Type::Int], Box::new(Type::Int)),
                body: Some(block_expr(vec![
                    let_stmt(
                        "plus_two",
                        None,
                        false,
                        lambda_expr(
                            vec![SoftBinding::new("x", None, false)],
                            block_expr(vec![return_stmt(call_expr(
                                "plus",
                                vec![2.into(), "x".into()],
                            ))]),
                        ),
                    ),
                    return_stmt("plus_two"),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn return_in_one_branch() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            args: vec![],
            return_type: Type::Int,
            body: Some(block_expr(vec![
                // Explicit return in true branch, implicit return in false.
                if_expr(true, block_expr(vec![return_stmt(2)]), 3).into(),
            ])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn error_return_wrong_type_in_one_branch() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            args: vec![],
            return_type: Type::Int,
            body: Some(block_expr(vec![
                // Explicit return in true branch, implicit return in false.
                if_expr(true, block_expr(vec![return_stmt(2.5)]), 3).into(),
            ])),
        })]);
        assert!(matches!(typecheck_program(program), Err(Error::NotUnifiable(_, _))));
    }
}
