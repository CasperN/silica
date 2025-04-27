// Abstract syntax tree and type checking.
#![allow(clippy::result_large_err)]
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum LValue {
    Variable(String),
    Field(Box<Expression>, String), // Deref, Field access, etc
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    L(LValue, Type),
    LiteralInt(i64),
    LiteralBool(bool),
    LiteralFloat(f64),
    LiteralUnit,
    LiteralStruct {
        name: String,
        fields: HashMap<String, Expression>,
        ty: Type,
    },
    If {
        condition: Box<Expression>,
        true_expr: Box<Expression>,
        false_expr: Box<Expression>,
        ty: Type,
    },
    Call {
        fn_expr: Box<Expression>,
        arg_exprs: Vec<Expression>,
        return_type: Type,
    },
    Block {
        statements: Vec<Statement>,
        ty: Type,
    },
    Lambda {
        bindings: Vec<SoftBinding>,
        body: Box<Expression>,
        lambda_type: Type,
    },
    Perform {
        name: Option<String>,
        op: String,
        arg: Box<Expression>,
        resume_type: Type,
    },
}
impl Expression {
    // Clones the type of the expression.
    fn get_type(&self) -> Type {
        match self {
            Self::LiteralInt(_) => Type::int(),
            Self::LiteralBool(_) => Type::bool_(),
            Self::LiteralFloat(_) => Type::float(),
            Self::LiteralUnit => Type::unit(),
            Self::L(_, ty)
            | Self::If { ty, .. }
            | Self::Block { ty, .. }
            | Self::Call {
                return_type: ty, ..
            }
            | Self::Lambda {
                lambda_type: ty, ..
            }
            | Self::LiteralStruct {
                name: _,
                fields: _,
                ty,
            }
            | Self::Perform {
                name: _,
                op: _,
                arg: _,
                resume_type: ty,
            } => {
                let mut ty = ty.clone();
                ty.compress();
                ty
            }
        }
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

#[derive(Default, Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub forall: Vec<u32>,
    pub name: String,
    pub args: Vec<TypedBinding>,
    pub return_type: Type,
    // If no body is provided, its assumed to be external.
    pub body: Option<Expression>,
}
impl FnDecl {
    fn as_type(&self) -> Type {
        let arg_types = self.args.iter().map(|b| b.ty.clone()).collect();
        Type::forall(
            self.forall.clone(),
            Type::func(arg_types, self.return_type.clone()),
        )
    }
}

// A coroutine function.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct CoDecl {
    pub forall: Vec<u32>,
    pub name: String,
    pub args: Vec<TypedBinding>,
    pub return_type: Type,
    pub ops: OpSet,
    // If no body is provided, its assumed to be external.
    pub body: Option<Expression>,
}

impl CoDecl {
    fn as_type(&self) -> Type {
        let arg_types = self.args.iter().map(|b| b.ty.clone()).collect();
        let co_type = Type::new(TypeI::Co(self.return_type.clone(), self.ops.clone()));
        Type::forall(self.forall.clone(), Type::func(arg_types, co_type))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    name: String,
    params: HashSet<u32>,
    fields: HashMap<String, Type>,
}
impl StructDecl {
    fn new(name: &str, fields: &[(&str, Type)]) -> Self {
        let fields: HashMap<String, Type> = fields
            .iter()
            .map(|(name, ty)| (name.to_string(), ty.clone()))
            .collect();
        let mut params = HashSet::new();
        for field_type in fields.values() {
            field_type.insert_type_params(&mut params);
        }
        StructDecl {
            name: name.to_string(),
            params,
            fields,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectDecl {
    name: String,
    params: HashSet<u32>, // TODO: wrap effect params in a newtype
    ops: HashMap<String, (Type, Type)>,
}
impl EffectDecl {
    fn unwrap_op_type(&self, op: &str) -> (Type, Type) {
        self.ops
            .get(op)
            .unwrap_or_else(|| panic!("Op {op} not in effect decl: {self:?}"))
            .clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Fn(FnDecl),
    Co(CoDecl),
    Struct(StructDecl),
    Effect(EffectDecl), // enums, traits, etc
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Program(pub Vec<Declaration>);

#[derive(Debug, Clone, PartialEq)]
pub struct StructInstance {
    params: HashMap<u32, Type>,
    decl: Rc<StructDecl>,
}
impl StructInstance {
    fn field_type(&self, field: &str) -> Result<Type, Error> {
        if let Some(field_type) = self.decl.fields.get(field) {
            let mut field_type = field_type.clone();
            if !self.params.is_empty() {
                field_type.instantiate_with(&self.params);
            }
            dbg!(field, &field_type, self);
            Ok(field_type)
        } else {
            Err(Error::UnrecognizedField(field.to_string()))
        }
    }
}

#[derive(Default, Clone, PartialEq)]
pub struct Type(Rc<RefCell<TypeI>>);

// Make it transparent over the contents.
impl std::fmt::Debug for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner();
        write!(f, "{inner:?}")
    }
}

#[derive(Debug, Clone, PartialEq)]
struct EffectInstance {
    params: HashMap<u32, Type>,
    decl: Rc<EffectDecl>,
}
impl EffectInstance {
    fn unwrap_op_type(&self, op_name: &str) -> (Type, Type) {
        let (mut p, mut r) = self.decl.unwrap_op_type(op_name);
        p.instantiate_with(&self.params);
        r.instantiate_with(&self.params);
        (p, r)
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct OpSet {
    anonymous_ops: HashMap<String, (Type, Type)>,
    effects: Vec<(EffectInstance, HashSet<String>)>,
    named_effects: HashMap<String, (EffectInstance, HashSet<String>)>,
}

impl OpSet {
    pub fn empty() -> Self {
        Self::default()
    }
    fn get(&self, name_or_effect: Option<&str>, op_name: &str) -> Option<(Type, Type)> {
        if name_or_effect.is_none() {
            // It must be a decl free / anonymous op.
            return self.anonymous_ops.get(op_name).cloned();
        }
        let name_or_effect = name_or_effect.unwrap();
        if let Some((instance, ops)) = self.named_effects.get(name_or_effect) {
            if ops.contains(op_name) {
                return Some(instance.unwrap_op_type(op_name));
            } else {
                // TODO: Error, op not in opset.
                return None;
            }
        }
        for (instance, ops) in self.effects.iter() {
            if instance.decl.name == name_or_effect {
                if ops.contains(op_name) {
                    return Some(instance.unwrap_op_type(op_name));
                } else {
                    // TODO: Error, op not in opset.
                    return None;
                }
            }
        }
        None
    }
    fn iter_types(&self) -> impl Iterator<Item = Type> + '_ {
        let mut types = Vec::new();
        for (p, r) in self.anonymous_ops.values() {
            types.push(p.clone());
            types.push(r.clone());
        }
        for (instance, _) in self.named_effects.values() {
            types.extend(instance.params.values().cloned());
        }
        for (instance, _) in self.effects.iter() {
            types.extend(instance.params.values().cloned());
        }
        types.into_iter()
    }
    fn iter_types_mut(&mut self) -> impl Iterator<Item = &'_ mut Type>  {
        let mut types = Vec::new();
        for (p, r) in self.anonymous_ops.values_mut() {
            types.push(p);
            types.push(r);
        }
        for (instance, _) in self.named_effects.values_mut() {
            types.extend(instance.params.values_mut());
        }
        for (instance, _) in self.effects.iter_mut() {
            types.extend(instance.params.values_mut());
        }
        types.into_iter()
    }
    fn insert_anonymous_effect(
        &mut self,
        name: &str,
        perform_type: Type,
        resume_type: Type,
    ) -> Result<(), Error> {
        match self.anonymous_ops.entry(name.to_string()) {
            Entry::Occupied(_) => Err(Error::DuplicateAnonymousOpName(name.to_string())),
            Entry::Vacant(entry) => {
                entry.insert((perform_type, resume_type));
                Ok(())
            }
        }
    }
    fn insert_declared_op(
        &mut self,
        name: Option<&str>,
        instance: &EffectInstance,
        op_name: &str,
    ) -> Result<(), Error> {
        if !instance.decl.ops.contains_key(op_name) {
            return Err(Error::NoMatchingOpInEffectDecl(op_name.to_string(), instance.decl.as_ref().clone()));
        }
        if name.is_none() {
            // First, try to add it to the unnamed effects.
            for (existing_instance, ops) in self.effects.iter_mut() {
                if existing_instance == instance {
                    ops.insert(op_name.to_string());
                    return Ok(());
                }
                if existing_instance.decl.name == instance.decl.name && existing_instance != instance {
                    return Err(Error::OpSetNameConflict(instance.decl.name.to_string()));
                }
            }
            // Before we append the new effect instance to the unnamed effects, we ensure there is no
            // ambiguity introduced with named effects:
            for (named_instance, _) in self.named_effects.values() {
                if named_instance.decl.name == instance.decl.name {
                    return Err(Error::OpSetNameConflict(instance.decl.name.to_string()));
                }
            }
            self.effects.push((instance.clone(), HashSet::from([op_name.to_string()])));
            return Ok(());
        }
        let name = name.unwrap().to_string();
        match self.named_effects.entry(name) {
            Entry::Occupied(mut entry) => {
                let existing_instance = &entry.get().0;
                if existing_instance != instance {
                    return Err(Error::NamedEffectInstanceMismatch(existing_instance.clone(), instance.clone()));
                }
                entry.get_mut().1.insert(op_name.to_string());
            },
            Entry::Vacant(entry) => {
                // Need to check that there's no ambiguity introduced with the unnamed effects before inserting.
                for (existing_instance, _) in self.effects.iter() {
                    if existing_instance.decl.name == instance.decl.name {
                        return Err(Error::NamedEffectInstanceMismatch(existing_instance.clone(), instance.clone()));
                    }
                }
                entry.insert((instance.clone(), HashSet::from_iter([op_name.to_string()])));
            }
        }
        Ok(())
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
enum TypeI {
    Int,
    Bool,
    Float,
    #[default]
    Unit,
    Fn(Vec<Type>, Type),
    Forall(Vec<u32>, Type),
    StructInstance(StructInstance),
    Co(Type, OpSet),

    // Unknown type variable. Unifies with any other type.
    // Should not appear in generic types.
    Unknown,

    // A type parameter. It should not appear during unification as
    // types need to be instantiated before then.
    Param(u32),

    // Points to another type that this is an alias for, as per the union-find algorithm.
    Follow(Type),
}

impl Type {
    fn new(i: TypeI) -> Self {
        Self(Rc::new(RefCell::new(i)))
    }
    pub fn bool_() -> Self {
        Self::new(TypeI::Bool)
    }
    pub fn int() -> Self {
        Self::new(TypeI::Int)
    }
    pub fn float() -> Self {
        Self::new(TypeI::Float)
    }
    pub fn unit() -> Self {
        Self::new(TypeI::Unit)
    }
    pub fn func(args: Vec<Type>, ret: Type) -> Self {
        Self::new(TypeI::Fn(args, ret))
    }
    pub fn param(id: u32) -> Self {
        Self::new(TypeI::Param(id))
    }
    pub fn unknown() -> Self {
        Self::new(TypeI::Unknown)
    }
    pub fn forall(params: Vec<u32>, ty: Type) -> Self {
        Self::new(TypeI::Forall(params, ty))
    }
    pub fn struct_(s: StructInstance) -> Self {
        Self::new(TypeI::StructInstance(s))
    }
    fn inner(&self) -> std::cell::Ref<TypeI> {
        self.0.borrow()
    }
    // Recursively compresses away any instances of `Follow` in the type.
    fn compress(&mut self) -> &mut Self {
        self.compress_direct_path();
        // TODO: Do you really need to compress direct path recursively?
        match &mut *self.0.borrow_mut() {
            TypeI::Bool
            | TypeI::Float
            | TypeI::Int
            | TypeI::Unit
            | TypeI::Unknown
            | TypeI::Param(_) => {}
            TypeI::Follow(_) => unreachable!(), // compress direct path.
            TypeI::Fn(args, ret) => {
                for arg in args.iter_mut() {
                    arg.compress();
                }
                ret.compress();
            }
            TypeI::Forall(_, ty) => {
                ty.compress();
            }
            TypeI::StructInstance(StructInstance { params, decl: _ }) => {
                for param_type in params.values_mut() {
                    param_type.compress();
                }
            }
            TypeI::Co(ty, ops) => {
                ty.compress();
                for ty in ops.iter_types_mut() {
                    ty.compress();
                }
            }
        }
        self
    }
    // Compresses Follow paths (e.g. Follow(Follow(T)) -> T ).
    fn compress_direct_path(&mut self) -> &mut Self {
        // Fast path, we're not following anyone.
        if !matches!(&*self.inner(), TypeI::Follow(_)) {
            return self;
        }
        // Follow the type references to the end.
        let mut end = self.clone();
        loop {
            let next = if let TypeI::Follow(other) = &*end.0.borrow() {
                other.clone()
            } else {
                break;
            };
            end = next;
        }
        // Traverse the path again and make everyone point to end.
        let mut follower = self.clone();
        loop {
            let next = if let TypeI::Follow(other) = &*follower.0.borrow() {
                other.clone()
            } else {
                break;
            };
            *follower.0.borrow_mut() = TypeI::Follow(end.clone());
            follower = next;
        }
        // This Type can point to the end-type, omitting any Follow nodes.
        *self = end.clone();
        self
    }

    fn contains_ptr(&self, other: &Type) -> bool {
        if Rc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        match &*self.0.borrow() {
            TypeI::Int
            | TypeI::Bool
            | TypeI::Float
            | TypeI::Unit
            | TypeI::Param(_)
            | TypeI::Unknown => false,
            TypeI::Follow(ty) => ty.contains_ptr(other),
            TypeI::Forall(_, ty) => ty.contains_ptr(other),
            TypeI::Fn(arg_types, ret_type) => {
                arg_types.iter().any(|a| a.contains_ptr(other)) || ret_type.contains_ptr(other)
            }
            TypeI::StructInstance(StructInstance { params, decl: _ }) => {
                // We assume that declarations cannot contain unification cycles,
                // so just check the params.
                params.values().any(|p| p.contains_ptr(other))
            }
            TypeI::Co(ty, ops) => {
                ty.contains_ptr(other) || ops.iter_types().any(|ty| ty.contains_ptr(other))
            }
        }
    }
    fn point_to(&mut self, other: &Self) -> Result<(), Error> {
        if other.contains_ptr(self) {
            return Err(Error::InfiniteType(self.clone(), other.clone()));
        }
        *self.0.borrow_mut() = TypeI::Follow(other.clone());
        Ok(())
    }

    fn instantiate_with(&mut self, subs: &HashMap<u32, Type>) {
        let mut i = self.inner().clone();
        match &mut i {
            TypeI::Int | TypeI::Bool | TypeI::Float | TypeI::Unit => {}
            TypeI::Param(b) => {
                *self = subs
                    .get(b)
                    .cloned()
                    .unwrap_or_else(|| panic!("Cannot instantiate type var {b}"));
                return;
            }
            TypeI::Unknown => {
                panic!("Instantiating a type with Unknown type variables within.");
            }
            TypeI::Fn(arg_types, ret_ty) => {
                for ty in arg_types.iter_mut() {
                    ty.instantiate_with(subs);
                }
                ret_ty.instantiate_with(subs);
            }
            TypeI::Forall(forall_vars, ty) => {
                forall_vars.retain(|forall_var| !subs.contains_key(forall_var));
                ty.instantiate_with(subs);
            }
            TypeI::StructInstance(StructInstance { params, decl: _ }) => {
                for param in params.values_mut() {
                    param.instantiate_with(subs);
                }
            }
            TypeI::Follow(_) => panic!("Instantiating a TypeI::Follow."),
            TypeI::Co(ty, ops) => {
                ty.instantiate_with(subs);
                for ty in ops.iter_types_mut() {
                    ty.instantiate_with(subs);
                }
            }
        }
        *self = Self::new(i)
    }

    // Instantiates Forall types with new type variables.
    fn instantiate(self) -> Self {
        let i = self.inner().clone();
        if let TypeI::Forall(params, mut ty) = i {
            let mut subs = HashMap::new();
            for b in params {
                subs.insert(b, Type::unknown());
            }
            ty.instantiate_with(&subs);
            ty
        } else {
            self
        }
    }
    fn is_polymorphic(&self) -> bool {
        matches!(&*self.inner(), TypeI::Forall(_, _))
    }
    fn insert_type_params(&self, output: &mut HashSet<u32>) {
        match &*self.inner() {
            TypeI::Int | TypeI::Bool | TypeI::Float | TypeI::Unit | TypeI::Unknown => {}
            TypeI::Param(b) => {
                output.insert(*b);
            }
            TypeI::Fn(arg_types, ret_ty) => {
                for arg_type in arg_types.iter() {
                    arg_type.insert_type_params(output);
                }
                ret_ty.insert_type_params(output);
            }
            // Free type variables are only asked for by struct generalization.
            // but we don't expect generic members.
            TypeI::Forall(_, _) => panic!(
                "Free type variables are only asked for by struct generalization... \
                but we don't expect generic members."
            ),
            TypeI::StructInstance(StructInstance { params, decl: _ }) => {
                for param in params.values() {
                    param.insert_type_params(output)
                }
            }
            TypeI::Follow(_) => panic!("TypeI::Follow used."),
            TypeI::Co(ty, ops) => {
                ty.insert_type_params(output);
                for ty in ops.iter_types() {
                    ty.insert_type_params(output);
                }
            }
        }
    }
}

#[derive(PartialEq, Debug)]
enum NamedItem {
    Variable(VariableInfo),
    Struct(Rc<StructDecl>),
    Effect(Rc<EffectDecl>),
}

#[derive(PartialEq, Debug)]
struct VariableInfo {
    ty: Type,
    mutable: bool,
}

// TODO: Move TypeContext, ShadowTypeContext, and friends into a sub-module to protect field access.
#[derive(Debug, Default, PartialEq)]
struct TypeContext {
    names: HashMap<String, NamedItem>,
    next_type_var: u32,
    // TODO: the return type and op-set need indicators to say whether they are being inferred or checked.
    // FnDecls are checked, lambdas and such are inferred. OpSet perhas should be optional too, to distinguish
    // between "this perform is not allowed" and "this is not a coroutine."
    return_type: Type,
    ops: Option<OpSet>,
}
impl TypeContext {
    fn new() -> Self {
        Self::default()
    }
    fn variable_info(&self, variable: &str) -> Option<&VariableInfo> {
        if let Some(NamedItem::Variable(info)) = self.names.get(variable) {
            Some(info)
        } else {
            None
        }
    }
    fn instantiate_struct(&mut self, name: &str) -> Result<StructInstance, Error> {
        let decl = if let Some(NamedItem::Struct(decl)) = self.names.get(name) {
            decl.clone()
        } else {
            return Err(Error::NoSuchStruct(name.to_string()));
        };
        let mut params = HashMap::new();
        for param_id in decl.params.iter() {
            params.insert(*param_id, Type::unknown());
        }
        Ok(StructInstance { params, decl })
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
            shadowed_ops: None,
        }
    }
    // When entering the body of a lambda, or `co {..}`, block, the semantics of
    // `return` and allowable performed ops changes.
    fn enter_activation_frame(
        &mut self,
        mut return_type: Type,
        mut ops: Option<OpSet>,
    ) -> ShadowTypeContext {
        std::mem::swap(&mut return_type, &mut self.return_type);
        std::mem::swap(&mut ops, &mut self.ops);
        ShadowTypeContext {
            type_context: self,
            shadowed_variables: HashMap::new(),
            shadowed_return_type: Some(return_type),
            shadowed_ops: Some(ops),
            finished: false,
        }
    }
}

#[derive(Debug)]
struct ShadowTypeContext<'a> {
    type_context: &'a mut TypeContext,
    shadowed_variables: HashMap<String, Option<NamedItem>>,
    shadowed_return_type: Option<Type>,
    shadowed_ops: Option<Option<OpSet>>,
    finished: bool,
}
impl<'a> ShadowTypeContext<'a> {
    fn insert_variable(&mut self, name: String, ty: Type, mutable: bool) {
        let info = NamedItem::Variable(VariableInfo { ty, mutable });
        if let Entry::Vacant(entry) = self.shadowed_variables.entry(name.clone()) {
            let original = self.type_context.names.insert(name, info);
            entry.insert(original);
        } else {
            // The original was already saved in `shadowed_variables`,
            // no need to touch that.
            self.type_context.names.insert(name, info);
        }
    }
    fn insert_binding(&mut self, binding: TypedBinding) {
        let TypedBinding { name, ty, mutable } = binding;
        self.insert_variable(name, ty, mutable);
    }
    fn insert_soft_binding(&mut self, binding: SoftBinding) {
        let SoftBinding { name, ty, mutable } = binding;
        let ty = ty.unwrap_or_else(Type::unknown);
        self.insert_variable(name, ty, mutable);
    }
    // Defines a struct. Returns if there was a previous definition.
    fn define_struct(&mut self, decl: StructDecl) -> bool {
        let name = decl.name.clone();
        let decl = NamedItem::Struct(Rc::new(decl));
        if let Entry::Vacant(entry) = self.shadowed_variables.entry(name.clone()) {
            let original = self.type_context.names.insert(name, decl);
            entry.insert(original);
            false
        } else {
            // The original was already saved in `shadowed_variables`,
            // no need to touch that.
            self.type_context.names.insert(name, decl);
            true
        }
    }
    fn define_effect(&mut self, decl: EffectDecl) -> bool {
        let name = decl.name.clone();
        let decl = NamedItem::Effect(Rc::new(decl));
        if let Entry::Vacant(entry) = self.shadowed_variables.entry(name.clone()) {
            let original = self.type_context.names.insert(name, decl);
            entry.insert(original);
            false
        } else {
            // The original was already saved in `shadowed_variables`,
            // no need to touch that.
            self.type_context.names.insert(name, decl);
            true
        }
    }
    fn context_contains(&mut self, name: &str) -> bool {
        self.type_context.names.contains_key(name)
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
                    self.type_context.names.insert(var_name, ty);
                }
                None => {
                    self.type_context.names.remove(&var_name);
                }
            }
        }
        if let Some(ty) = self.shadowed_return_type.take() {
            self.type_context.return_type = ty;
        }
    }
}

#[derive(PartialEq, Debug)]
enum Error {
    UnknownName(String),
    NoSuchStruct(String),
    NotUnifiable(Type, Type),
    InfiniteType(Type, Type),
    DuplicateArgNames(Expression),
    DuplicateTopLevelName(String),
    AssignToImmutableBinding(String),
    TopLevelFnArgsMustBeTyped(String),
    UnrecognizedField(String),
    FieldAccessToNonStruct(Expression, String),
    NoMatchingOpInContext {
        effect_or_instance: Option<String>,
        op_name: String,
    },
    PerformedEffectsNotallowedInContext {
        effect_or_instance: Option<String>,
        op_name: String,
    },
    DuplicateAnonymousOpName(String),
    OpSetNameConflict(String),
    NoMatchingOpInEffectDecl(String, EffectDecl),
    NamedEffectInstanceMismatch(EffectInstance, EffectInstance),
}

// Unifies two types via interior mutability.
// Errors if the types ae not unifiable.
fn unify(left: &Type, right: &Type) -> Result<(), Error> {
    let mut left = left.clone();
    let mut right = right.clone();
    left.compress();
    right.compress();

    // `.inner` borrows from the Refcell. That borrow must end before
    // `point_to` runs, which borrows mutably. Hence, we use a
    // matches! statement instead of handling this case in the match.
    if matches!(&*left.inner(), TypeI::Unknown) {
        left.point_to(&right)?;
        return Ok(());
    }
    if matches!(&*right.inner(), TypeI::Unknown) {
        right.point_to(&left)?;
        return Ok(());
    }
    // Assign the std::Rc::Ref RAII objects.
    // They can't be temporaries in the match statement.
    let result = match (&*left.inner(), &*right.inner()) {
        (TypeI::Int, TypeI::Int)
        | (TypeI::Float, TypeI::Float)
        | (TypeI::Bool, TypeI::Bool)
        | (TypeI::Unit, TypeI::Unit) => Ok(()),
        (TypeI::Fn(left_arg_types, left_ret_ty), TypeI::Fn(right_arg_types, right_ret_ty)) => {
            if left_arg_types.len() != right_arg_types.len() {
                return Err(Error::NotUnifiable(left.clone(), right.clone()));
            }
            unify(left_ret_ty, right_ret_ty)?;
            for (l, r) in left_arg_types.iter().zip(right_arg_types.iter()) {
                unify(l, r)?;
            }
            Ok(())
        }
        (
            TypeI::StructInstance(StructInstance {
                params: left_params,
                decl: left_decl,
            }),
            TypeI::StructInstance(StructInstance {
                params: right_params,
                decl: right_decl,
            }),
        ) if left_decl == right_decl => {
            assert_eq!(left_params.len(), right_params.len());
            for (param_id, left_ty) in left_params.iter() {
                let right_ty = right_params.get(param_id).unwrap_or_else(|| {
                    panic!(
                        "Two instance of the same struct should \
                        have the same type param ids. \
                        Left:{left:?}, Right:{right:?}"
                    )
                });
                unify(left_ty, right_ty)?
            }
            Ok(())
        }
        _ => Err(Error::NotUnifiable(left.clone(), right.clone())),
    };
    result
}

// Infers the type of the given expression in the given context.
// Mutates the expression to set the type.
fn infer(context: &mut TypeContext, expression: &mut Expression) -> Result<(), Error> {
    match expression {
        Expression::L(LValue::Variable(name), ty) => {
            if let Some(info) = context.variable_info(name) {
                if info.ty.is_polymorphic() {
                    // This only applies if name is a top level polymorphic function.
                    unify(ty, &info.ty.clone().instantiate())?;
                    Ok(())
                } else {
                    unify(ty, &info.ty)
                }
            } else {
                Err(Error::UnknownName(name.to_string()))
            }
        }
        Expression::L(LValue::Field(expr, field), ty) => {
            infer(context, expr)?;
            if let TypeI::StructInstance(struct_instance) = &*expr.get_type().inner() {
                let field_type = struct_instance.field_type(field)?;
                unify(&field_type, ty)?;
            } else {
                return Err(Error::FieldAccessToNonStruct(*expr.clone(), field.clone()));
            }
            Ok(())
        }
        Expression::LiteralUnit => Ok(()),
        Expression::LiteralInt(_) => Ok(()),
        Expression::LiteralBool(_) => Ok(()),
        Expression::LiteralFloat(_) => Ok(()),
        Expression::LiteralStruct { name, fields, ty } => {
            let struct_instance = context.instantiate_struct(name)?;
            for (field_name, field_expr) in fields.iter_mut() {
                let decl_ty = struct_instance.field_type(field_name)?;
                infer(context, field_expr)?;
                unify(&field_expr.get_type(), &decl_ty)?;
            }
            unify(ty, &Type::struct_(struct_instance))?;
            Ok(())
        }
        Expression::If {
            condition,
            true_expr,
            false_expr,
            ty: _,
        } => {
            infer(context, condition)?;
            let cond_ty = condition.get_type();
            unify(&cond_ty, &Type::bool_())?;

            infer(context, true_expr)?;
            infer(context, false_expr)?;
            let t_ty = true_expr.get_type();
            let f_ty = false_expr.get_type();
            unify(&t_ty, &f_ty)?;

            Ok(())
        }
        Expression::Call {
            fn_expr,
            arg_exprs,
            return_type,
        } => {
            infer(context, fn_expr)?;

            let mut arg_types = vec![];
            for arg_expr in arg_exprs.iter_mut() {
                infer(context, arg_expr)?;
                arg_types.push(arg_expr.get_type().clone());
            }

            unify(
                &fn_expr.get_type(),
                &Type::func(arg_types, return_type.clone()),
            )?;
            Ok(())
        }
        Expression::Lambda {
            bindings,
            body,
            lambda_type,
        } => {
            let arg_name_set: HashSet<_> = bindings.iter().map(|b| b.name.as_str()).collect();
            if arg_name_set.len() != bindings.len() {
                return Err(Error::DuplicateArgNames(expression.clone()));
            }
            let lambda_return_type = Type::unknown();
            let mut shadow = context.enter_activation_frame(
                lambda_return_type.clone(),
                None, // TODO: Lambdas can't perform effects... yet?
            );
            for binding in bindings.iter() {
                shadow.insert_soft_binding(binding.clone());
            }
            infer(shadow.context(), body)?;
            unify(&body.get_type(), &lambda_return_type)?;

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
            unify(lambda_type, &Type::func(arg_types, lambda_return_type))?;
            shadow.finish();
            Ok(())
        }
        Expression::Perform {
            name,
            op,
            arg,
            resume_type: expr_ty,
        } => {
            if context.ops.is_none() {
                return Err(Error::PerformedEffectsNotallowedInContext {
                    effect_or_instance: name.clone(),
                    op_name: op.clone(),
                });
            }
            dbg!(&name, &op, &arg);
            if let Some((perform_ty, resume_ty)) =
                context.ops.as_ref().unwrap().get(name.as_deref(), op)
            {
                unify(&arg.get_type(), &perform_ty)?;
                unify(expr_ty, &resume_ty)?;
                Ok(())
            } else {
                Err(Error::NoMatchingOpInContext {
                    effect_or_instance: name.clone(),
                    op_name: op.clone(),
                })
            }
        }
        Expression::Block {
            statements,
            ty: block_ty,
        } => {
            let mut shadow = context.shadow();
            let mut last_statement_type = Type::unit();
            for statement in statements {
                match statement {
                    Statement::Expression(expr) => {
                        infer(shadow.context(), expr)?;
                        last_statement_type = expr.get_type().clone();
                    }
                    Statement::Let { binding, value } => {
                        let context = shadow.context();
                        infer(context, value)?;
                        let value_type = value.get_type().clone();
                        if let Some(binding_ty) = &binding.ty {
                            unify(binding_ty, &value_type)?;
                        }
                        shadow.insert_variable(
                            binding.name.clone(),
                            value_type.clone(),
                            binding.mutable,
                        );
                        last_statement_type = Type::unit();
                    }
                    Statement::Assign(LValue::Variable(name), expr) => {
                        let context = shadow.context();
                        infer(context, expr)?;

                        // Check whether the expession can be assigned to the variable.
                        let variable_info = context.variable_info(name);
                        if variable_info.is_none() {
                            return Err(Error::UnknownName(name.to_string()));
                        }
                        let variable_info = variable_info.unwrap();
                        if !variable_info.mutable {
                            return Err(Error::AssignToImmutableBinding(name.to_string()));
                        }
                        unify(&expr.get_type(), &variable_info.ty)?;
                        last_statement_type = Type::unit();
                    }
                    Statement::Assign(LValue::Field(_, _), _) => {
                        todo!()
                    }
                    Statement::Return(expr) => {
                        infer(shadow.context(), expr)?;
                        unify(&expr.get_type(), &shadow.context().return_type)?;
                        last_statement_type = expr.get_type().clone();
                        // TODO: Probably should issue a warning for unreachable statements.
                    }
                }
            }
            unify(block_ty, &last_statement_type)?;
            shadow.finish();
            Ok(())
        }
    }
}

fn typecheck_program(program: &mut Program) -> Result<(), Error> {
    // First, load declarations into typing context.
    let mut context = TypeContext::new();
    let mut shadow = context.shadow();
    for declaration in program.0.iter() {
        match declaration {
            Declaration::Fn(fn_decl) => {
                if shadow.context_contains(&fn_decl.name) {
                    return Err(Error::DuplicateTopLevelName(fn_decl.name.clone()));
                }
                let mut arg_types = vec![];
                for binding in fn_decl.args.iter() {
                    arg_types.push(binding.ty.clone());
                }
                // TODO: Verify that top level declarations have no free type variables.
                // TODO: args need to declare mutability.
                shadow.insert_variable(fn_decl.name.clone(), fn_decl.as_type(), false);
            }
            Declaration::Co(co_decl) => {
                if shadow.context_contains(&co_decl.name) {
                    return Err(Error::DuplicateTopLevelName(co_decl.name.clone()));
                }
                let mut arg_types = vec![];
                for binding in co_decl.args.iter() {
                    arg_types.push(binding.ty.clone());
                }
                // TODO: Verify that top level declarations have no free type variables.
                // TODO: args need to declare mutability.
                shadow.insert_variable(co_decl.name.clone(), co_decl.as_type(), false);
            }
            Declaration::Struct(struct_declaration) => {
                let redefined = shadow.define_struct(struct_declaration.clone());
                if redefined {
                    return Err(Error::DuplicateTopLevelName(
                        struct_declaration.name.clone(),
                    ));
                }
            }
            Declaration::Effect(effect_declaration) => {
                let redefined = shadow.define_effect(effect_declaration.clone());
                if redefined {
                    return Err(Error::DuplicateTopLevelName(
                        effect_declaration.name.clone(),
                    ));
                }
            }
        }
    }
    // Second, typecheck function bodies.
    for declaration in program.0.iter_mut() {
        match declaration {
            Declaration::Fn(FnDecl {
                forall: _,
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
                let mut shadow = shadow
                    .context()
                    .enter_activation_frame(return_type.clone(), None);
                for binding in args.iter() {
                    shadow.insert_binding(binding.clone()); // TODO: variables need to declare mutaiblity.
                }
                infer(shadow.context(), body)?;
                unify(return_type, &body.get_type())?;
                shadow.finish();
            }
            Declaration::Co(CoDecl {
                forall: _,
                name: _,
                args,
                return_type,
                ops,
                body,
            }) => {
                if body.is_none() {
                    continue; // External fn, take it for granted.
                }
                let body = body.as_mut().unwrap();

                // Declare a new shadow to insert fn variables.
                let mut shadow = shadow
                    .context()
                    .enter_activation_frame(return_type.clone(), Some(ops.clone()));
                for binding in args.iter() {
                    shadow.insert_binding(binding.clone()); // TODO: variables need to declare mutaiblity.
                }
                infer(shadow.context(), body)?;
                unify(return_type, &body.get_type())?;
                shadow.finish();
            }
            Declaration::Struct(_) | Declaration::Effect(_) => {}
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
impl From<LValue> for Expression {
    fn from(l: LValue) -> Self {
        Expression::L(l, Type::unknown())
    }
}
impl From<&str> for Statement {
    fn from(value: &str) -> Self {
        Statement::Expression(Expression::L(
            LValue::Variable(value.to_string()),
            Type::unknown(),
        ))
    }
}
impl From<i64> for Statement {
    fn from(value: i64) -> Self {
        Statement::Expression(Expression::LiteralInt(value))
    }
}
impl From<&str> for Expression {
    fn from(value: &str) -> Self {
        Expression::L(LValue::Variable(value.to_string()), Type::unknown())
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
impl From<LValue> for Statement {
    fn from(l: LValue) -> Self {
        Statement::Expression(l.into())
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
            return_type: Type::unknown(),
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
            ty: Type::unknown(),
        }
    }
    pub fn block_expr(statements: Vec<Statement>) -> Expression {
        Expression::Block {
            statements,
            ty: Type::unknown(),
        }
    }
    pub fn lambda_expr(bindings: Vec<SoftBinding>, body: impl Into<Expression>) -> Expression {
        Expression::Lambda {
            bindings,
            body: Box::new(body.into()),
            lambda_type: Type::unknown(),
        }
    }
    pub fn field(expr: impl Into<Expression>, name: &str) -> LValue {
        LValue::Field(Box::new(expr.into()), name.into())
    }
    pub fn perform_anon(op: &str, arg: impl Into<Expression>) -> Expression {
        Expression::Perform {
            name: None,
            op: op.to_string(),
            arg: Box::new(arg.into()),
            resume_type: Type::unknown(),
        }
    }
    pub fn perform(name: &str, op: &str, arg: impl Into<Expression>) -> Expression {
        Expression::Perform {
            name: Some(name.to_string()),
            op: op.to_string(),
            arg: Box::new(arg.into()),
            resume_type: Type::unknown(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::*;
    use super::*;

    #[test]
    fn unify_type_vars() {
        let mut v0 = Type::unknown();
        let mut v1 = Type::unknown();
        unify(&v0, &v1).unwrap();
        assert_eq!(v0.compress(), v1.compress());
    }
    #[test]
    fn left_unify_many_type_vars() {
        let mut types = Vec::new();
        for _ in 0..10 {
            types.push(Type::unknown());
        }
        for i in 0..9 {
            unify(&types[i], &types[i + 1]).unwrap();
        }
        unify(&types[9], &Type::unit()).unwrap();
        assert_eq!(*types[0].compress(), Type::unit());
    }
    #[test]
    fn right_unify_many_type_vars() {
        let mut types = Vec::new();
        for _ in 0..10 {
            types.push(Type::unknown());
        }
        for i in 0..9 {
            unify(&types[i + 1], &types[i]).unwrap();
        }
        unify(&types[9], &Type::unit()).unwrap();
        assert_eq!(*types[0].compress(), Type::unit());
    }
    #[test]
    fn unify_fn_return_type() {
        let type_var_0 = Type::unknown();
        let mut f1 = Type::func(vec![], type_var_0);
        let mut f2 = Type::func(vec![], Type::int());
        unify(&f1, &f2).unwrap();

        assert_eq!(f1.compress(), f2.compress());
    }
    #[test]
    fn unify_fns() {
        let type_var_0 = Type::unknown();
        let type_var_1 = Type::unknown();
        let mut f1 = Type::func(vec![type_var_0.clone()], type_var_0.clone());
        let mut f2 = Type::func(vec![type_var_1], Type::int());
        unify(&f1, &f2).unwrap();

        assert_eq!(f1.compress(), f2.compress());
    }

    #[test]
    fn two_plus_two() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
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
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::bool_(),
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
            forall: vec![],
            name: "main".to_string(),
            args: vec![TypedBinding::new("a", Type::int(), true)], // mutable.
            return_type: Type::unit(),
            body: Some(block_expr(vec![assign_stmt("a", 3)])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn error_assign_wrong_type_to_mutable_binding() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::unit(),
            body: Some(block_expr(vec![
                let_stmt("a", None, true, 2),
                assign_stmt("a", ()),
            ])),
        })]);
        assert_eq!(
            typecheck_program(program),
            Err(Error::NotUnifiable(Type::unit(), Type::int()))
        );
    }
    #[test]
    fn error_assign_to_immutable_binding() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::unit(),
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
                forall: vec![],
                name: "round".to_string(),
                args: vec![TypedBinding::new("a", Type::float(), false)],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
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
                forall: vec![],
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
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
                forall: vec![],
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::func(vec![Type::int()], Type::int()),
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
                forall: vec![],
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::func(vec![Type::int()], Type::int()),
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
    fn use_polymorphic_top_level_fn() {
        // id(2) + id(2)
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: vec![0],
                name: "id".to_string(),
                args: vec![TypedBinding::new("a", Type::param(0), false)],
                return_type: Type::param(0),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![call_expr(
                    "plus",
                    vec![
                        call_expr("id", vec![2.into()]),
                        call_expr("id", vec![2.into()]),
                    ],
                )
                .into()])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn use_polymorphic_top_level_fn_polymorphically() {
        // id(2) + id(2)
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: vec![0],
                name: "id".to_string(),
                args: vec![TypedBinding::new("a", Type::param(0), false)],
                return_type: Type::param(0),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::float(),
                body: Some(block_expr(vec![
                    let_stmt("x", None, false, call_expr("id", vec![12.34.into()])),
                    let_stmt("b", None, false, call_expr("id", vec![false.into()])),
                    if_expr("b", "x", 23.45).into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn error_polymorphic_top_level_fn() {
        // id(2) + id(2)
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: vec![0],
                name: "id".to_string(),
                args: vec![TypedBinding::new("a", Type::param(0), false)],
                return_type: Type::param(0),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::func(vec![Type::int()], Type::int()),
                body: Some(block_expr(vec![call_expr(
                    "plus",
                    vec![
                        call_expr("id", vec![2.into()]),
                        call_expr("id", vec![2.0.into()]),
                    ],
                )
                .into()])),
            }),
        ]);
        let t = typecheck_program(program);
        assert!(t.is_err());
    }

    #[test]
    fn return_in_one_branch() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::int(),
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
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::int(),
            body: Some(block_expr(vec![
                // Explicit return in true branch, implicit return in false.
                if_expr(true, block_expr(vec![return_stmt(2.5)]), 3).into(),
            ])),
        })]);
        assert!(matches!(
            typecheck_program(program),
            Err(Error::NotUnifiable(_, _))
        ));
    }
    #[test]
    fn error_infinite_type() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::int(),
            body: Some(block_expr(vec![
                // Explicit return in true branch, implicit return in false.
                let_stmt(
                    "x",
                    None,
                    false,
                    lambda_expr(
                        vec![SoftBinding {
                            name: "x".to_string(),
                            ty: None,
                            mutable: false,
                        }],
                        call_expr("x", vec!["x".into()]),
                    ),
                ),
                2.into(),
            ])),
        })]);
        assert!(matches!(
            typecheck_program(program),
            Err(Error::InfiniteType(_, _))
        ));
    }

    #[test]
    fn empty_block_has_type_unit() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::unit(),
            body: Some(block_expr(vec![])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn block_with_just_lets_has_type_unit() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::unit(),
            body: Some(block_expr(vec![
                let_stmt("x", None, false, 1),
                let_stmt("y", None, false, false),
                let_stmt("z", None, false, 19.95),
            ])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn factorial_fn() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "mul".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "sub".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "leq".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::bool_(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "factorial".to_string(),
                args: vec![TypedBinding::new("x", Type::int(), false)],
                return_type: Type::int(),
                body: Some(if_expr(
                    call_expr("leq", vec!["x".into(), 1.into()]),
                    1,
                    call_expr(
                        "mul",
                        vec![
                            "x".into(),
                            call_expr(
                                "factorial",
                                vec![call_expr("sub", vec!["x".into(), 1.into()])],
                            ),
                        ],
                    ),
                )),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn higher_order_functions() {
        let fn_int_to_int = Type::func(vec![Type::int()], Type::int());

        let program = &mut Program(vec![
            // Declare external fn apply(f: fn(Int)->Int, x: Int) -> Int;
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "apply".to_string(),
                args: vec![
                    TypedBinding::new("f", fn_int_to_int.clone(), false),
                    TypedBinding::new("x", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None, // External
            }),
            // Declare external fn make_adder(y: Int) -> fn(Int)->Int;
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "make_adder".to_string(),
                args: vec![TypedBinding::new("y", Type::int(), false)],
                return_type: fn_int_to_int.clone(),
                body: None, // External
            }),
            // Declare fn main() -> Int {
            //   let add5 = make_adder(5);
            //   apply(add5, 10)
            // }
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    // let add_5 = make_adder(5)
                    let_stmt("add5", None, false, call_expr("make_adder", vec![5.into()])),
                    // apply(add5, 10)
                    call_expr("apply", vec!["add5".into(), 10.into()]).into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn struct_field_access() {
        let pair = StructDecl::new(
            "Pair",
            &[("left", Type::param(0)), ("right", Type::param(1))],
        );

        let program = &mut Program(vec![
            Declaration::Struct(pair),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "p",
                        None,
                        false,
                        Expression::LiteralStruct {
                            name: "Pair".to_string(),
                            fields: HashMap::from_iter([
                                ("left".into(), true.into()),
                                ("right".into(), 123.into()),
                            ]),
                            ty: Type::unknown(),
                        },
                    ),
                    field("p", "left").into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn perform_anonymous_effect() {
        let mut ops = OpSet::empty();
        ops.insert_anonymous_effect("foo", Type::int(), Type::bool_()).unwrap();
        let program = &mut Program(vec![Declaration::Co(CoDecl {
            forall: vec![],
            name: "main".to_string(),
            args: vec![],
            return_type: Type::bool_(),
            ops,
            body: Some(block_expr(vec![
                let_stmt("p", None, false, perform_anon("foo", 1)),
                "p".into(),
            ])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn perform_named_effects() {
        let state_effect = EffectDecl {
            name: "State".to_string(),
            params: HashSet::from_iter([0]),
            ops: HashMap::from_iter([
                ("get".into(), (Type::unit(), Type::param(0))),
                ("set".into(), (Type::param(0), Type::unit())),
            ])
        };
        let int_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: HashMap::from_iter([(0, Type::int())])
        };
        let bool_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: HashMap::from_iter([(0, Type::bool_())])
        };

        let mut ops = OpSet::empty();
        ops.insert_declared_op(Some("int_state"), &int_state_instance, "get").unwrap();
        ops.insert_declared_op(Some("int_state"), &int_state_instance, "set").unwrap();
        ops.insert_declared_op(Some("bool_state"), &bool_state_instance, "get").unwrap();
        ops.insert_declared_op(Some("bool_state"), &bool_state_instance, "set").unwrap();

        let program = &mut Program(vec![
            Declaration::Effect(state_effect),
            Declaration::Co(CoDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops,
                body: Some(block_expr(vec![
                    let_stmt("x", Type::int(), false, perform("int_state", "get", ())),
                    perform("int_state", "set", "x").into(),
                    let_stmt("y", Type::bool_(), false, perform("bool_state", "get", ())),
                    perform("bool_state", "set", "y").into(),
                    "y".into()
                ])),
            })
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn conflicting_parametric_effects_one_named() {
        let state_effect = EffectDecl {
            name: "State".to_string(),
            params: HashSet::from_iter([0]),
            ops: HashMap::from_iter([
                ("get".into(), (Type::unit(), Type::param(0))),
                ("set".into(), (Type::param(0), Type::unit())),
            ])
        };
        let int_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: HashMap::from_iter([(0, Type::int())])
        };
        let bool_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: HashMap::from_iter([(0, Type::bool_())])
        };
        let mut ops = OpSet::empty();
        ops.insert_declared_op(Some("int_state"), &int_state_instance, "get").unwrap();
        assert_eq!(
            ops.insert_declared_op(None, &bool_state_instance, "set"),
            Err(Error::OpSetNameConflict("State".to_string())),
        );
    }
    #[test]
    fn conflicting_parametric_effects_both_unnamed() {
        let state_effect = EffectDecl {
            name: "State".to_string(),
            params: HashSet::from_iter([0]),
            ops: HashMap::from_iter([
                ("get".into(), (Type::unit(), Type::param(0))),
                ("set".into(), (Type::param(0), Type::unit())),
            ])
        };
        let int_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: HashMap::from_iter([(0, Type::int())])
        };
        let bool_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: HashMap::from_iter([(0, Type::bool_())])
        };
        let mut ops = OpSet::empty();
        ops.insert_declared_op(None, &int_state_instance, "get").unwrap();
        assert_eq!(
            ops.insert_declared_op(None, &bool_state_instance, "set"),
            Err(Error::OpSetNameConflict("State".to_string())),
        );
    }
}
