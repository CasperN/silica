// Abstract syntax tree and type checking.
#![allow(clippy::result_large_err)]
use crate::union_find::UnionFindRef;
use std::cell::{Ref, RefMut};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

// *************************************************************************************************
//  Types
// *************************************************************************************************

#[derive(Default, Clone, PartialEq)]
pub struct Type(UnionFindRef<TypeI>);

// Make it transparent over the contents.
impl std::fmt::Debug for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = self.clone();
        let inner = s.inner();
        write!(f, "{inner:?}")
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
enum TypeI {
    #[default]
    Unit,
    Int,
    Bool,
    Float,
    Never, // Never aka Bottom aka the empty type.
    Fn(Vec<Type>, Type),
    StructInstance(StructInstance),
    Co(Type, OpSet),

    // TODO: Consider deleting -- only top level decls should be polymorphic.
    Forall(BTreeSet<String>, Type),

    // Unknown type variable. Unifies with any other type.
    // Should not appear in generic types.
    Unknown(Vec<Constraint>),

    // A type parameter. It should not appear during unification as
    // types need to be instantiated before then.
    Param(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Constraint {
    // Field name and field type constraint.
    HasField(String, Type),
}

impl Type {
    fn new(i: TypeI) -> Self {
        Self(UnionFindRef::new(i))
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
    pub fn never() -> Self {
        Self::new(TypeI::Never)
    }
    pub fn func(args: Vec<Type>, ret: Type) -> Self {
        Self::new(TypeI::Fn(args, ret))
    }
    pub fn param(id: &str) -> Self {
        Self::new(TypeI::Param(id.to_string()))
    }
    pub fn unknown() -> Self {
        Self::new(TypeI::Unknown(vec![]))
    }
    pub fn forall(params: BTreeSet<String>, ty: Type) -> Self {
        Self::new(TypeI::Forall(params, ty))
    }
    pub fn struct_(s: StructInstance) -> Self {
        Self::new(TypeI::StructInstance(s))
    }
    pub fn co(ty: Type, ops: impl Into<OpSet>) -> Type {
        Self::new(TypeI::Co(ty, ops.into()))
    }
    fn inner(&mut self) -> Ref<TypeI> {
        self.0.inner()
    }
    fn inner_mut(&mut self) -> RefMut<TypeI> {
        self.0.inner_mut()
    }

    fn contains_type(&self, other: &Type) -> bool {
        if self.0.ptr_eq(&other.0) {
            return true;
        }
        match &*self.clone().inner() {
            TypeI::Int
            | TypeI::Bool
            | TypeI::Float
            | TypeI::Unit
            | TypeI::Never
            | TypeI::Param(_)
            | TypeI::Unknown(_) => false,
            TypeI::Forall(_, ty) => ty.contains_type(other),
            TypeI::Fn(arg_types, ret_type) => {
                arg_types.iter().any(|a| a.contains_type(other)) || ret_type.contains_type(other)
            }
            TypeI::StructInstance(StructInstance { params, decl: _ }) => {
                // We assume that declarations cannot contain unification cycles,
                // so just check the params.
                params.values().any(|p| p.contains_type(other))
            }
            TypeI::Co(ty, ops) => {
                ty.contains_type(other)
                    || ops
                        .clone()
                        .inner()
                        .iter_types()
                        .any(|ty| ty.contains_type(other))
            }
        }
    }
    fn contains_opset(&self, other: &OpSet) -> bool {
        match &*self.clone().inner() {
            TypeI::Int
            | TypeI::Bool
            | TypeI::Float
            | TypeI::Unit
            | TypeI::Param(_)
            | TypeI::Never
            | TypeI::Unknown(_) => false,
            TypeI::Forall(_, ty) => ty.contains_opset(other),
            TypeI::Fn(arg_types, ret_type) => {
                arg_types.iter().any(|a| a.contains_opset(other)) || ret_type.contains_opset(other)
            }
            TypeI::StructInstance(StructInstance { params, decl: _ }) => {
                // We assume that declarations cannot contain unification cycles,
                // so just check the params.
                params.values().any(|p| p.contains_opset(other))
            }
            TypeI::Co(ty, ops) => ty.contains_opset(other) || ops.contains_opset(other),
        }
    }

    fn point_to(&mut self, other: &Self) -> Result<(), Error> {
        if other.contains_type(self) {
            return Err(Error::InfiniteType(self.clone(), other.clone()));
        }
        self.0.follow(&other.0);
        Ok(())
    }

    fn instantiate_with(&mut self, subs: &BTreeMap<String, Type>) {
        let mut i = self.inner().clone();
        match &mut i {
            TypeI::Int | TypeI::Bool | TypeI::Float | TypeI::Unit | TypeI::Never => {}
            TypeI::Param(b) => {
                *self = subs
                    .get(b)
                    .cloned()
                    .unwrap_or_else(|| panic!("Cannot instantiate type var {b}"));
                return;
            }
            TypeI::Unknown(_) => {
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
            TypeI::Co(ty, ops) => {
                ty.instantiate_with(subs);
                for ty in ops.mut_inner().iter_types_mut() {
                    ty.instantiate_with(subs);
                }
            }
        }
        *self = Self::new(i)
    }

    // Instantiates Forall types with new type variables.
    fn instantiate(mut self) -> Self {
        let i = self.inner().clone();
        if let TypeI::Forall(params, mut ty) = i {
            let mut subs = BTreeMap::new();
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
        matches!(&*self.clone().inner(), TypeI::Forall(_, _))
    }
    fn insert_type_params(&self, output: &mut BTreeSet<String>) {
        match &*self.clone().inner() {
            TypeI::Int
            | TypeI::Bool
            | TypeI::Float
            | TypeI::Unit
            | TypeI::Never
            | TypeI::Unknown(_) => {}
            TypeI::Param(b) => {
                output.insert(b.clone());
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
            TypeI::Co(ty, ops) => {
                ty.insert_type_params(output);
                for ty in ops.clone().inner().iter_types() {
                    ty.insert_type_params(output);
                }
            }
        }
    }
    // A concrete type will be unchanged under unification.
    fn is_concrete(&self) -> bool {
        match &*self.clone().inner() {
            TypeI::Bool
            | TypeI::Int
            | TypeI::Float
            | TypeI::Unit
            | TypeI::Param(_)
            | TypeI::Never => true,
            TypeI::Unknown(_) => false,
            TypeI::Forall(_, ty) => ty.is_concrete(),
            TypeI::Fn(args, ret) => ret.is_concrete() && args.iter().all(|a| a.is_concrete()),
            TypeI::StructInstance(StructInstance { params, decl: _ }) => {
                params.values().all(|p| p.is_concrete())
            }
            TypeI::Co(ty, ops) => ty.is_concrete() && ops.clone().inner().is_concrete(),
        }
    }

    fn unwrap_ops(&self) -> OpSet {
        if let TypeI::Co(_, ops) = &*self.clone().inner() {
            return ops.clone();
        }
        panic!("unwrap_ops called on a non-coroutine.");
    }
}

// *************************************************************************************************
// OpSets
// *************************************************************************************************

// Union find wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct OpSet(UnionFindRef<OpSetI>);

// A set of ops that are being performed.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct OpSetI {
    anonymous_ops: HashMap<String, (Type, Type)>,
    named_effects: HashMap<String, (EffectInstance, HashSet<String>)>,
    super_sets: Vec<OpSet>,
    done_extending: bool,
}

impl From<OpSetI> for OpSet {
    fn from(value: OpSetI) -> Self {
        Self::new(value)
    }
}
impl Default for OpSet {
    fn default() -> Self {
        Self::new(OpSetI::empty())
    }
}

impl OpSet {
    fn new(opset: OpSetI) -> Self {
        Self(UnionFindRef::new(opset))
    }
    fn empty_non_extensible() -> Self {
        Self::new(OpSetI::empty_non_extensible())
    }
    pub fn empty_extensible() -> Self {
        Self::new(OpSetI::empty())
    }
    fn is_empty(&self) -> bool {
        self.0.clone().inner().is_empty()
    }
    fn is_extendable(&self) -> bool {
        !self.0.clone().inner().done_extending
    }
    fn inner(&mut self) -> Ref<OpSetI> {
        self.0.inner()
    }
    fn mut_inner(&mut self) -> RefMut<OpSetI> {
        self.0.inner_mut()
    }
    fn clone_inner(&self) -> OpSetI {
        self.0.clone_inner()
    }
    fn follow(&self, other: &OpSet) {
        if other.contains_opset(self) {
            panic!("Looped introduced when following. Left {self:?}. Right: {other:?}.");
        }
        self.0.clone().follow(&other.0);
    }
    /// When self is extended with new ops, add them to `other` too.
    fn when_extended_update(&mut self, other: &Self) {
        if other.contains_opset(self) {
            return;
        }
        if self.is_extendable() {
            self.mut_inner().super_sets.push(other.clone());
        }
    }

    /// Adds all currently known effects of `other` into `self`.
    fn subsume(&mut self, other: &OpSetI) -> Result<(), Error> {
        let mut self_inner = self.mut_inner();
        for (op_name, (perform_type, resume_type)) in other.anonymous_ops.iter() {
            self_inner.unify_add_anonymous_effect(op_name, perform_type, resume_type)?;
        }
        for (effect_name, (instance, ops)) in other.named_effects.iter() {
            for op_name in ops {
                self_inner.unify_add_declared_op(Some(effect_name), instance, op_name)?;
            }
        }
        Ok(())
    }
    fn contains_opset(&self, other: &OpSet) -> bool {
        if self.0.ptr_eq(&other.0) {
            return true;
        }
        self.clone().inner().contains_opset(other)
    }
}
impl OpSetI {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn empty_non_extensible() -> Self {
        let mut x = Self::empty();
        x.mark_done_extending();
        x
    }
    pub fn mark_done_extending(&mut self) -> &mut Self {
        self.done_extending = true;
        self
    }
    fn is_empty(&self) -> bool {
        self.anonymous_ops.is_empty() && self.named_effects.is_empty()
    }

    // An Opset is concrete if all the types therein are concrete.
    // Note that we consider extendable OpSets to be concrete.
    fn is_concrete(&self) -> bool {
        self.iter_types().all(|ty| ty.is_concrete())
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
                return None;
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
        types.into_iter()
    }
    fn iter_types_mut(&mut self) -> impl Iterator<Item = &'_ mut Type> {
        let mut types = Vec::new();
        for (p, r) in self.anonymous_ops.values_mut() {
            types.push(p);
            types.push(r);
        }
        for (instance, _) in self.named_effects.values_mut() {
            types.extend(instance.params.values_mut());
        }
        types.into_iter()
    }
    fn remove_anonymous_effect_or_die(&mut self, name: &str) {
        self.anonymous_ops
            .remove(name)
            .unwrap_or_else(|| panic!("Expected {name} in {self:?}"));
    }
    fn remove_named_effect_or_die(&mut self, effect_name: &str, op_name: &str) {
        if let Some((_, ops)) = self.named_effects.get_mut(effect_name) {
            if !ops.remove(op_name) {
                panic!("Expected {effect_name}.{op_name} in {self:?}");
            }
        } else {
            panic!("Expected {effect_name} effect in {self:?}");
        }
    }

    // TODO: Should not be public. Made public as a hack to construct OpSets in the parser.
    pub fn unify_add_anonymous_effect_or_die(
        &mut self,
        name: &str,
        perform_type: &Type,
        resume_type: &Type,
    ) -> &mut Self {
        self.unify_add_anonymous_effect(name, perform_type, resume_type)
            .unwrap()
    }

    fn unify_add_anonymous_effect(
        &mut self,
        name: &str,
        perform_type: &Type,
        resume_type: &Type,
    ) -> Result<&mut Self, Error> {
        match self.anonymous_ops.entry(name.to_string()) {
            Entry::Occupied(entry) => {
                let (existing_perform_type, existing_resume_type) = entry.get();
                unify(perform_type, existing_perform_type)?;
                unify(resume_type, existing_resume_type)?;
            }
            Entry::Vacant(entry) => {
                if self.done_extending {
                    return Err(Error::OpSetNotExtendable(self.clone()));
                }
                entry.insert((perform_type.clone(), resume_type.clone()));
                for super_set in self.super_sets.iter() {
                    super_set.clone().mut_inner().unify_add_anonymous_effect(
                        name,
                        perform_type,
                        resume_type,
                    )?;
                }
            }
        }
        Ok(self)
    }
    fn unify_add_declared_op(
        &mut self,
        name: Option<&str>,
        instance: &EffectInstance,
        op_name: &str,
    ) -> Result<&mut Self, Error> {
        if !instance.decl.ops.contains_key(op_name) {
            return Err(Error::NoMatchingOpInEffectDecl(
                op_name.to_string(),
                instance.decl.as_ref().clone(),
            ));
        }
        // Unnamed declared effects take their effect declaration name by default.
        let name_str = name.unwrap_or(&instance.decl.name).to_string();
        match self.named_effects.entry(name_str) {
            Entry::Occupied(mut entry) => {
                let (existing_instance, existing_ops) = entry.get_mut();
                if existing_instance.decl != instance.decl {
                    return Err(Error::EffectDeclMismatch(
                        instance.decl.as_ref().clone(),
                        existing_instance.decl.as_ref().clone(),
                    ));
                }
                // Unify the params map.
                unify_params(&existing_instance.params, &instance.params, || {
                    panic!(
                        "Two instances of the same effect should \
                        have the same type param ids. \
                        Left:{existing_instance:?}, 15:{instance:?}"
                    )
                })?;
                existing_ops.insert(op_name.to_string());
            }
            Entry::Vacant(entry) => {
                if self.done_extending {
                    return Err(Error::OpSetNotExtendable(self.clone()));
                }
                entry.insert((instance.clone(), HashSet::from_iter([op_name.to_string()])));
                for super_set in self.super_sets.iter() {
                    super_set
                        .clone()
                        .mut_inner()
                        .unify_add_declared_op(name, instance, op_name)?;
                }
            }
        }
        Ok(self)
    }
    fn get_anonymous_op(&self, name: &str) -> Option<(Type, Type)> {
        self.anonymous_ops.get(name).cloned()
    }
    fn contains_opset(&self, other: &OpSet) -> bool {
        self.super_sets.iter().any(|s| s.contains_opset(other))
            && self.iter_types().any(|t| t.contains_opset(other))
    }
}

// *************************************************************************************************
//  Parsed Types
// *************************************************************************************************

// TODO: Get rid of strings here.
#[derive(Default, Debug, Clone, PartialEq)]
pub enum ParsedType {
    #[default]
    Unspecified,
    Unit,
    Int,
    Bool,
    Float,
    Never,
    Fn(Vec<ParsedType>, Box<ParsedType>),
    Co(Box<ParsedType>, Vec<ParsedOp>),
    Named(String, Vec<ParsedType>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParsedOp {
    Anonymous(String, ParsedType, ParsedType),
    NamedEffect {
        name: Option<String>,
        effect: String,
        op_name: Option<String>,
    },
}

fn resolve_type(context: &TypeContext, ty: &ParsedType) -> Result<Type, Error> {
    match ty {
        ParsedType::Unspecified => Ok(Type::unknown()),
        ParsedType::Unit => Ok(Type::unit()),
        ParsedType::Int => Ok(Type::int()),
        ParsedType::Bool => Ok(Type::bool_()),
        ParsedType::Float => Ok(Type::float()),
        ParsedType::Never => Ok(Type::never()),
        ParsedType::Fn(args, ret) => {
            let mut resolved_args = Vec::new();
            for arg in args {
                resolved_args.push(resolve_type(context, arg)?);
            }
            let ret = resolve_type(context, ret)?;
            Ok(Type::func(resolved_args, ret))
        }
        ParsedType::Co(ret, ops) => {
            let ret = resolve_type(context, ret)?;
            let mut anonymous_ops = HashMap::new();
            let mut named_effects = HashMap::new();

            for op in ops {
                match op {
                    ParsedOp::Anonymous(op_name, p_ty, r_ty) => {
                        let perform_type = resolve_type(context, p_ty)?;
                        let resume_type = resolve_type(context, r_ty)?;
                        if anonymous_ops
                            .insert(op_name.clone(), (perform_type, resume_type))
                            .is_some()
                        {
                            return Err(Error::DuplicateOpName(op_name.clone()));
                        }
                    }
                    ParsedOp::NamedEffect {
                        name,
                        effect,
                        op_name,
                    } => {
                        if let Some(NamedItem::Effect(decl)) = context.names.get(effect) {
                            let name = name.clone().unwrap_or_else(|| decl.name.clone());
                            match named_effects.entry(name) {
                                Entry::Vacant(entry) => {
                                    let ops: HashSet<String> = if let Some(op_name) = op_name {
                                        [op_name.clone()].into_iter().collect()
                                    } else {
                                        // If no op_name is specified, add all of them.
                                        decl.ops.keys().cloned().collect()
                                    };
                                    assert!(decl.params.is_empty());
                                    let instance = EffectInstance {
                                        params: BTreeMap::new(),
                                        decl: decl.clone(),
                                    };
                                    entry.insert((instance, ops));
                                }
                                Entry::Occupied(mut entry) => {
                                    let (_decl, ops) = entry.get_mut();
                                    if let Some(op_name) = op_name {
                                        if ops.insert(op_name.clone()) {
                                            return Err(Error::DuplicateOpName(op_name.clone()));
                                        }
                                    } else {
                                        return Err(Error::DuplicateNamedEffect(effect.clone()));
                                    }
                                }
                            }
                        } else {
                            return Err(Error::ExpectedNamedEffect(effect.clone()));
                        }
                    }
                }
            }
            let opset = OpSet::new(OpSetI {
                anonymous_ops,
                named_effects,
                super_sets: Vec::new(),
                done_extending: true,
            });
            Ok(Type::co(ret, opset))
        }
        ParsedType::Named(name, parsed_params) => {
            match context.names.get(name) {
                // TODO: Type parameters, like foo<T> need to be supported by type context and
                // resolved here.
                Some(NamedItem::Struct(decl)) => {
                    if decl.params.len() != parsed_params.len() {
                        return Err(Error::ParameterCountMismatch(
                            name.clone(),
                            parsed_params.clone(),
                        ));
                    }
                    // TODO: Params in struct decl should be in order, not sorted :/
                    // This is pretty wrong, unless params so happen to be sorted.
                    let mut params = BTreeMap::new();
                    for (parsed_param, name) in parsed_params.iter().zip(decl.params.iter()) {
                        let ty = resolve_type(context, parsed_param)?;
                        if params.insert(name.clone(), ty).is_some() {
                            panic!("fixme.")
                        }
                    }
                    Ok(Type::struct_(StructInstance {
                        params,
                        decl: decl.clone(),
                    }))
                }
                _ => {
                    Err(Error::ExpectedNamedType(name.clone()))
                }
            }
        }
    }
}

// *************************************************************************************************
//  Unification
// *************************************************************************************************

fn constrain(ty: &mut TypeI, constraint: Constraint) -> Result<(), Error> {
    let cloned_type = Type::new(ty.clone());
    match (ty, constraint) {
        (TypeI::Unknown(constraints), constraint) => {
            constraints.push(constraint);
            Ok(())
        }
        (TypeI::StructInstance(instance), Constraint::HasField(name, field_ty)) => {
            unify(&instance.field_type(&name)?, &field_ty)
        }
        (_, constraint) => Err(Error::InapplicableConstraint(cloned_type, constraint)),
    }
}

fn unify_params(
    left_params: &BTreeMap<String, Type>,
    right_params: &BTreeMap<String, Type>,
    panic_msg: impl Fn() -> String,
) -> Result<(), Error> {
    assert_eq!(left_params.len(), right_params.len());
    for (param_id, left_ty) in left_params.iter() {
        let right_ty = right_params
            .get(param_id)
            .unwrap_or_else(|| panic!("{}", panic_msg()));
        unify(left_ty, right_ty)?
    }
    Ok(())
}

fn unify_opsets(mut left_ops: OpSet, mut right_ops: OpSet) -> Result<(), Error> {
    assert!(!left_ops.0.ptr_eq(&right_ops.0));

    // Ensure all ops in left and right are on both sides.
    left_ops.subsume(&right_ops.inner())?;
    right_ops.subsume(&left_ops.inner())?;

    // We wil make right follow left. If right is not extendable, left shouldn't be either.
    if !right_ops.is_extendable() {
        left_ops.mut_inner().done_extending = true;
    }
    right_ops.follow(&left_ops);

    Ok(())
}

// Unifies two types via interior mutability.
// Errors if the types ae not unifiable.
fn unify(left: &Type, right: &Type) -> Result<(), Error> {
    // Allow for unification with self while avoiding double borrowing.
    if left.0.ptr_eq(&right.0) {
        return Ok(());
    }
    let left_clone = left.clone();
    let right_clone = right.clone();
    let mut left = left.clone();
    let mut right = right.clone();

    // `.inner` borrows from the Refcell. That borrow must end before
    // `point_to` runs, which borrows mutably. Hence, we record whether
    // we will point left to right or right to left in a variable and apply
    // it after the match statement.
    let mut point_left_to_right: Option<bool> = None;

    let result = match (&mut *left.inner_mut(), &mut *right.inner_mut()) {
        (TypeI::Int, TypeI::Int)
        | (TypeI::Float, TypeI::Float)
        | (TypeI::Bool, TypeI::Bool)
        | (TypeI::Unit, TypeI::Unit) => Ok(()),
        // The empty type unifies with everything.
        (_, TypeI::Never) => {
            point_left_to_right = Some(false);
            Ok(())
        }
        (TypeI::Never, _) => {
            point_left_to_right = Some(true);
            Ok(())
        }
        // Unify unknown types, respecting constraints.
        (TypeI::Unknown(constraints), right_inner) => {
            for constraint in constraints.drain(..) {
                constrain(right_inner, constraint)?;
            }
            point_left_to_right = Some(true);
            Ok(())
        }
        (left_inner, TypeI::Unknown(constraints)) => {
            for constraint in constraints.drain(..) {
                constrain(left_inner, constraint)?;
            }
            point_left_to_right = Some(false);
            Ok(())
        }
        (TypeI::Fn(left_arg_types, left_ret_ty), TypeI::Fn(right_arg_types, right_ret_ty)) => {
            if left_arg_types.len() != right_arg_types.len() {
                return Err(Error::NotUnifiable(left_clone, right_clone));
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
        ) if left_decl == right_decl => unify_params(left_params, right_params, || {
            panic!(
                "Two instance of the same struct should \
                    have the same type param ids. \
                    Left:{left_clone:?}, Right:{right_clone:?}"
            )
        }),
        (TypeI::Co(left_ty, left_ops), TypeI::Co(right_ty, right_ops)) => {
            unify(left_ty, right_ty)?;
            unify_opsets(left_ops.clone(), right_ops.clone())?;
            Ok(())
        }
        (TypeI::Param(l), TypeI::Param(r)) if l == r => Ok(()),
        _ => Err(Error::NotUnifiable(left_clone, right_clone)),
    };
    // Finish unifying unknowns (after the dynamic borrow)
    // by making one an alias of the other.
    if let Some(left_to_right) = point_left_to_right {
        if left_to_right {
            left.point_to(&right)?;
        } else {
            right.point_to(&left)?;
        }
    }
    result
}

// *************************************************************************************************
//  Expressions
// *************************************************************************************************

#[derive(Debug, Clone, PartialEq)]
pub enum LValue {
    Variable(String),
    Field(Box<Expression>, String), // Deref, Field access, etc
}

// TODO: Consider struct TypedExpression(Expression, Type) and remove the type fields.

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
    Co(Box<Expression>, Type, OpSet),
    Perform {
        name: Option<String>,
        op: String,
        arg: Box<Expression>,
        resume_type: Type,
    },
    Propagate(Box<Expression>, Type),
    Handle {
        co: Box<Expression>,
        // Initially arm, finally arm.
        return_arm: Option<(SoftBinding, Box<Expression>)>,
        op_arms: Vec<HandleOpArm>,
        ty: Type,
    }, // TODO: UFCS/method-call, ref/deref.
}

#[derive(Debug, Clone, PartialEq)]
pub struct HandleOpArm {
    pub op_name: String,
    // TODO: This only handles anonymous effects. What about named effects?
    pub performed_variable: SoftBinding,
    pub body: Expression,
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
            }
            | Self::Handle { ty, .. }
            | Self::Propagate(_, ty) => ty.clone(),
            Self::Co(_, return_ty, ops) => Type::co(return_ty.clone(), ops.clone()),
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
    Resume(Expression),
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
    pub parsed_type: ParsedType,
    pub ty: Type,
    pub mutable: bool,
}
impl SoftBinding {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parsed_type: ParsedType::Unspecified,
            ty: Type::unknown(),
            mutable: false,
        }
    }
    pub fn new_mut(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parsed_type: ParsedType::Unspecified,
            ty: Type::unknown(),
            mutable: true,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub forall: BTreeSet<String>,
    pub name: String,
    pub args: Vec<TypedBinding>,
    pub return_type: Type,  // TODO: This should be the parsed type...
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
    pub forall: BTreeSet<String>,
    pub name: String,
    pub args: Vec<TypedBinding>,
    pub return_type: Type,
    pub ops: OpSetI,
    // If no body is provided, its assumed to be external.
    pub body: Option<Expression>,
}

impl CoDecl {
    fn as_type(&self) -> Type {
        let arg_types = self.args.iter().map(|b| b.ty.clone()).collect();
        let co_type = Type::co(self.return_type.clone(), self.ops.clone());
        Type::forall(self.forall.clone(), Type::func(arg_types, co_type))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: String,
    pub params: BTreeSet<String>,
    pub fields: HashMap<String, Type>,
}
impl StructDecl {
    fn new(name: &str, fields: &[(&str, Type)]) -> Self {
        let fields: HashMap<String, Type> = fields
            .iter()
            .map(|(name, ty)| (name.to_string(), ty.clone()))
            .collect();
        let mut params = BTreeSet::new();
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
    pub name: String,
    pub params: BTreeSet<String>,
    pub ops: HashMap<String, (Type, Type)>,
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
    params: BTreeMap<String, Type>,
    decl: Rc<StructDecl>,
}
impl StructInstance {
    fn field_type(&self, field: &str) -> Result<Type, Error> {
        if let Some(field_type) = self.decl.fields.get(field) {
            let mut field_type = field_type.clone();
            if !self.params.is_empty() {
                field_type.instantiate_with(&self.params);
            }
            Ok(field_type)
        } else {
            Err(Error::UnrecognizedField(field.to_string()))
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
struct EffectInstance {
    params: BTreeMap<String, Type>,
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

// *************************************************************************************************
//  Type Context
// *************************************************************************************************

// TODO: Move TypeContext, ShadowTypeContext, and friends into a sub-module to protect field access.
#[derive(Debug, Default, PartialEq)]
struct TypeContext {
    names: HashMap<String, NamedItem>,
    next_type_var: u32,
    // TODO: the return type and op-set need indicators to say whether they are being inferred or
    // checked. FnDecls are checked, lambdas and such are inferred. OpSetI perhaps should be optional
    // too, to distinguish between "this perform is not allowed" and "this is not a coroutine."
    return_type: Type,
    resume_type: Option<Type>,
    ops: OpSet,
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
        let mut params = BTreeMap::new();
        for param_id in decl.params.iter() {
            params.insert(param_id.clone(), Type::unknown());
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
            shadowed_resume_type: None,
        }
    }
    // When entering the body of a lambda, or `co {..}`, block, the semantics of
    // `return` and allowable performed ops changes.
    fn enter_activation_frame(
        &mut self,
        mut return_type: Type,
        mut ops: OpSet,
    ) -> ShadowTypeContext {
        std::mem::swap(&mut return_type, &mut self.return_type);
        std::mem::swap(&mut ops, &mut self.ops);
        let resume_type = self.resume_type.take();
        ShadowTypeContext {
            type_context: self,
            shadowed_variables: HashMap::new(),
            shadowed_return_type: Some(return_type),
            shadowed_ops: Some(ops),
            shadowed_resume_type: Some(resume_type),
            finished: false,
        }
    }
}

#[derive(Debug)]
struct ShadowTypeContext<'a> {
    type_context: &'a mut TypeContext,
    shadowed_variables: HashMap<String, Option<NamedItem>>,
    shadowed_return_type: Option<Type>,
    shadowed_resume_type: Option<Option<Type>>,
    shadowed_ops: Option<OpSet>,
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
    fn insert_soft_binding(&mut self, binding: SoftBinding) -> Type {
        let SoftBinding { name, ty, mutable, parsed_type: _ } = binding;
        self.insert_variable(name, ty.clone(), mutable);
        ty
    }
    fn set_resume_type(&mut self, ty: &Type) {
        if self.shadowed_return_type.is_some() {
            panic!("Resume type set multiple times.");
        }
        self.shadowed_resume_type = Some(self.context().resume_type.take());
        self.context().resume_type = Some(ty.clone());
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
        if let Some(ops) = self.shadowed_ops.take() {
            self.type_context.ops = ops;
        }
        if let Some(resume_type) = self.shadowed_resume_type.take() {
            self.type_context.resume_type = resume_type;
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
    DuplicateOpName(String),
    NoMatchingOpInEffectDecl(String, EffectDecl),
    NamedEffectInstanceMismatch(EffectInstance, EffectInstance),
    FnDeclMustHaveConcreteTypes(FnDecl),
    CoDeclMustHaveConcreteTypes(CoDecl),
    StructDeclMustHaveConcreteTypes(StructDecl),
    EffectDeclMismatch(EffectDecl, EffectDecl),
    OpSetNotUnifiable(OpSetI, OpSetI),
    OpSetNotExtendable(OpSetI),
    InapplicableConstraint(Type, Constraint),
    ExpressisonTypeNotInferred(Expression),
    UnexpectedResume(Expression),
    ParameterCountMismatch(String, Vec<ParsedType>),
    ExpectedNamedType(String),
    ExpectedNamedEffect(String),
    DuplicateNamedEffect(String),
}

// *************************************************************************************************
//  Inference
// *************************************************************************************************

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
            constrain(
                &mut expr.get_type().inner_mut(),
                Constraint::HasField(field.clone(), ty.clone()),
            )
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
            let mut shadow = context
                .enter_activation_frame(lambda_return_type.clone(), OpSet::empty_non_extensible());
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
        Expression::Co(expr, return_ty, ops) => {
            let mut shadow = context.enter_activation_frame(return_ty.clone(), ops.clone());
            infer(shadow.context(), expr)?;
            unify(return_ty, &expr.get_type())?;
            shadow.finish();
            Ok(())
        }
        Expression::Perform {
            name,
            op,
            arg,
            resume_type: expr_ty,
        } => {
            infer(context, arg)?;

            if let Some((perform_ty, resume_ty)) =
                context.ops.clone_inner().get(name.as_deref(), op)
            {
                unify(&arg.get_type(), &perform_ty)?;
                unify(expr_ty, &resume_ty)?;
                Ok(())
            } else if !context.ops.is_extendable() {
                Err(Error::PerformedEffectsNotallowedInContext {
                    effect_or_instance: name.clone(),
                    op_name: op.clone(),
                })
            } else if name.is_none() {
                // Perform an anonymous effect.
                context
                    .ops
                    .mut_inner()
                    .unify_add_anonymous_effect(op, &arg.get_type(), expr_ty)?;
                Ok(())
            } else {
                // TODO: Perform named effect. Need to constrain the unknown effect.
                // Similar to constraining unknown types with fields.
                Err(Error::NoMatchingOpInContext {
                    effect_or_instance: name.clone(),
                    op_name: op.clone(),
                })
            }
        }
        Expression::Propagate(expr, ty) => {
            infer(context, expr)?;
            let mut co_ops = OpSet::empty_extensible();
            unify(&expr.get_type(), &Type::co(ty.clone(), co_ops.clone()))?;
            context.ops.subsume(&co_ops.inner())?;
            co_ops.when_extended_update(&context.ops);
            Ok(())
        }
        Expression::Handle {
            co,
            return_arm,
            op_arms,
            ty: handle_expr_ty,
        } => {
            infer(context, co)?;
            // Get a name for `co`'s type and ops, adding the handled ops to co_ops.
            let co_return_ty = Type::unknown();
            let mut co_ops = OpSet::empty_extensible();
            for arm in op_arms.iter() {
                co_ops.mut_inner().unify_add_anonymous_effect(
                    &arm.op_name,
                    &Type::unknown(),
                    &Type::unknown(),
                )?;
            }
            unify(
                &co.get_type(),
                &Type::co(co_return_ty.clone(), co_ops.clone()),
            )?;

            // Infer the return arm.
            if let Some((binding, expr)) = return_arm {
                let mut shadow = context.shadow();
                let original_return_ty = shadow.insert_soft_binding(binding.clone());
                unify(&original_return_ty, &co_return_ty)?;

                infer(shadow.context(), expr)?;
                unify(&expr.get_type(), handle_expr_ty)?;
            } else {
                // An unspecified return arm is the identity function, so `co`'s returned value
                // becomes the handle's evaluated value.
                unify(&co_return_ty, handle_expr_ty)?;
            }

            // Infer the op arms.
            for arm in op_arms.iter_mut() {
                let mut shadow = context.shadow();
                let p_var_ty = shadow.insert_soft_binding(arm.performed_variable.clone());
                let (p_ty, r_ty) = co_ops.inner().get_anonymous_op(&arm.op_name).unwrap();
                unify(&p_ty, &p_var_ty)?;
                shadow.set_resume_type(&r_ty);
                infer(shadow.context(), &mut arm.body)?;
                // TODO: If the arm ends in a resume, then it does not need to unify here.
                unify(&arm.body.get_type(), handle_expr_ty)?;
            }
            // Clone the handled ops and removed the handled ones to get the remaining ops.
            // Guaranteed not to die because they were just added.
            let mut remaining_ops = co_ops.mut_inner().clone();
            for arm in op_arms.iter() {
                remaining_ops.remove_anonymous_effect_or_die(&arm.op_name);
            }
            context.ops.subsume(&remaining_ops)?;
            co_ops.when_extended_update(&context.ops);
            Ok(())
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
                        unify(&binding.ty, &resolve_type(context, &binding.parsed_type)?)?;
                        infer(context, value)?;
                        unify(&binding.ty, &value.get_type())?;
                        shadow.insert_variable(
                            binding.name.clone(),
                            value.get_type(),
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
                        last_statement_type = Type::never();
                        // TODO: Probably should issue a warning for unreachable statements.
                    }
                    Statement::Resume(expr) => {
                        infer(shadow.context(), expr)?;
                        if let Some(resume_type) = shadow.context().resume_type.as_ref() {
                            unify(&expr.get_type(), resume_type)?;
                        } else {
                            return Err(Error::UnexpectedResume(expression.clone()));
                        }
                        last_statement_type = Type::never();
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

// TODO: Accumulate errors in a list rather than returning the first one.
fn check_expression_concrete(expression: &Expression) -> Result<(), Error> {
    // TODO: Its not currently an error that some expressions do not have concrete types
    // at the end of inference. This needs to be implemented and added to `typecheck_program`.
    if !expression.get_type().is_concrete() {
        return Err(Error::ExpressisonTypeNotInferred(expression.clone()));
    }
    match expression {
        Expression::LiteralUnit
        | Expression::LiteralBool(_)
        | Expression::LiteralInt(_)
        | Expression::LiteralFloat(_)
        | Expression::L(_, _) => {}
        Expression::If {
            condition,
            true_expr,
            false_expr,
            ty: _,
        } => {
            check_expression_concrete(condition)?;
            check_expression_concrete(true_expr)?;
            check_expression_concrete(false_expr)?;
        }
        Expression::Call {
            fn_expr,
            arg_exprs,
            return_type: _,
        } => {
            check_expression_concrete(fn_expr)?;
            for arg_expr in arg_exprs {
                check_expression_concrete(arg_expr)?
            }
        }
        Expression::Lambda { body, .. } => {
            check_expression_concrete(body)?;
        }
        Expression::LiteralStruct { fields, .. } => {
            for field_expr in fields.values() {
                check_expression_concrete(field_expr)?;
            }
        }
        Expression::Perform { arg, .. } => {
            check_expression_concrete(arg)?;
        }
        Expression::Propagate(expr, _) => {
            check_expression_concrete(expr)?;
        }
        Expression::Handle {
            co,
            return_arm,
            op_arms,
            ty: _,
        } => {
            check_expression_concrete(co)?;
            if let Some((_, expr)) = return_arm {
                check_expression_concrete(expr)?;
            }
            for arm in op_arms {
                check_expression_concrete(&arm.body)?;
            }
        }
        Expression::Co(expr, _, _) => {
            check_expression_concrete(expr)?;
        }
        Expression::Block { statements, .. } => {
            for statement in statements {
                check_statement_concrete(statement)?;
            }
        }
    };
    Ok(())
}
fn check_statement_concrete(statement: &Statement) -> Result<(), Error> {
    match statement {
        Statement::Assign(_, expr) => {
            check_expression_concrete(expr)?;
        }
        Statement::Expression(expr) => {
            check_expression_concrete(expr)?;
        }
        Statement::Let { binding: _, value } => {
            check_expression_concrete(value)?;
        }
        Statement::Return(expr) | Statement::Resume(expr) => {
            check_expression_concrete(expr)?;
        }
    }
    Ok(())
}

fn typecheck_program(program: &mut Program) -> Result<(), Error> {
    // First, load declarations into typing context.
    let mut context = TypeContext::new();
    let mut shadow = context.shadow();
    for declaration in program.0.iter() {
        match declaration {
            Declaration::Fn(fn_decl) => {
                if !(fn_decl.args.iter().all(|binding| binding.ty.is_concrete())
                    && fn_decl.return_type.is_concrete())
                {
                    return Err(Error::FnDeclMustHaveConcreteTypes(fn_decl.clone()));
                }
                if shadow.context_contains(&fn_decl.name) {
                    return Err(Error::DuplicateTopLevelName(fn_decl.name.clone()));
                }
                let mut arg_types = vec![];
                for binding in fn_decl.args.iter() {
                    arg_types.push(binding.ty.clone());
                }
                shadow.insert_variable(fn_decl.name.clone(), fn_decl.as_type(), false);
            }
            Declaration::Co(co_decl) => {
                if !(co_decl.args.iter().all(|binding| binding.ty.is_concrete())
                    && co_decl.return_type.is_concrete()
                    && co_decl.ops.is_concrete())
                {
                    return Err(Error::CoDeclMustHaveConcreteTypes(co_decl.clone()));
                }
                if shadow.context_contains(&co_decl.name) {
                    return Err(Error::DuplicateTopLevelName(co_decl.name.clone()));
                }
                let mut arg_types = vec![];
                for binding in co_decl.args.iter() {
                    arg_types.push(binding.ty.clone());
                }
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
                    .enter_activation_frame(return_type.clone(), OpSet::empty_non_extensible());
                for binding in args.iter() {
                    shadow.insert_binding(binding.clone());
                }
                infer(shadow.context(), body)?;
                unify(return_type, &body.get_type())?;
                shadow.finish();
                check_expression_concrete(body)?;
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
                    .enter_activation_frame(return_type.clone(), ops.clone().into());
                for binding in args.iter() {
                    shadow.insert_binding(binding.clone());
                }
                infer(shadow.context(), body)?;
                unify(return_type, &body.get_type())?;
                shadow.finish();
                check_expression_concrete(body)?;
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
impl From<f64> for Statement {
    fn from(value: f64) -> Self {
        Statement::Expression(Expression::LiteralFloat(value))
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
        value: impl Into<Expression>,
    ) -> Statement {
        Statement::Let {
            binding: SoftBinding::new(name),
            value: value.into(),
        }
    }
    pub fn full_let_stmt(
        name: &str,
        type_: Type,
        mutable: bool,
        value: impl Into<Expression>,
    ) -> Statement {
        Statement::Let {
            binding: SoftBinding { name: name.to_string(),
                parsed_type: ParsedType::Unspecified, ty: type_, mutable },
            value: value.into(),
        }
    }
    pub fn let_mut_stmt(
        name: &str,
        value: impl Into<Expression>,
    ) -> Statement {
        Statement::Let {
            binding: SoftBinding::new_mut(name),
            value: value.into(),
        }
    }
    pub fn return_stmt(expr: impl Into<Expression>) -> Statement {
        Statement::Return(expr.into())
    }
    pub fn resume_stmt(expr: impl Into<Expression>) -> Statement {
        Statement::Resume(expr.into())
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
    pub fn propagate(expr: impl Into<Expression>) -> Expression {
        Expression::Propagate(Box::new(expr.into()), Type::unknown())
    }
    pub fn co_expr(expr: impl Into<Expression>) -> Expression {
        Expression::Co(Box::new(expr.into()), Type::unknown(), OpSet::default())
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
        assert_eq!(*v0.inner(), *v1.inner());
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
        assert_eq!(*types[0].inner(), TypeI::Unit);
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
        assert_eq!(*types[0].inner(), TypeI::Unit);
    }
    #[test]
    fn unify_fn_return_type() {
        let type_var_0 = Type::unknown();
        let mut f1 = Type::func(vec![], type_var_0);
        let mut f2 = Type::func(vec![], Type::int());
        unify(&f1, &f2).unwrap();

        assert_eq!(*f1.inner(), *f2.inner());
    }
    #[test]
    fn unify_fns() {
        let type_var_0 = Type::unknown();
        let type_var_1 = Type::unknown();
        let mut f1 = Type::func(vec![type_var_0.clone()], type_var_0.clone());
        let mut f2 = Type::func(vec![type_var_1], Type::int());
        unify(&f1, &f2).unwrap();

        assert_eq!(*f1.inner(), *f2.inner());
    }

    #[test]
    fn two_plus_two() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt("a", 2),
                    let_stmt("b", 2),
                    let_stmt(
                        "c",
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
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::bool_(),
            body: Some(block_expr(vec![
                let_stmt("a", 2),
                let_stmt("a", 2.0),
                let_stmt("a", true),
                "a".into(),
            ])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn assign_to_mutable_binding() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: BTreeSet::new(),
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
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::unit(),
            body: Some(block_expr(vec![
                let_mut_stmt("a", 2),
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
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::unit(),
            body: Some(block_expr(vec![
                let_stmt("a", 2),
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
                forall: BTreeSet::new(),
                name: "round".to_string(),
                args: vec![TypedBinding::new("a", Type::float(), false)],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt("a", true),
                    let_stmt("b", 2),
                    let_stmt("c", 2.0),
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
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "plus_two",
                        lambda_expr(
                            vec![SoftBinding::new("x")],
                            call_expr("plus", vec![2.into(), "x".into()]),
                        ),
                    ),
                    let_mut_stmt("b", 2),
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
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::func(vec![Type::int()], Type::int()),
                body: Some(block_expr(vec![
                    let_stmt(
                        "plus_two",
                        lambda_expr(
                            vec![SoftBinding::new("x")],
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
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::func(vec![Type::int()], Type::int()),
                body: Some(block_expr(vec![
                    let_stmt(
                        "plus_two",
                        lambda_expr(
                            vec![SoftBinding::new("x")],
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
                forall: BTreeSet::from_iter(["T".to_string()].into_iter()),
                name: "id".to_string(),
                args: vec![TypedBinding::new("a", Type::param("T"), false)],
                return_type: Type::param("T"),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
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
                forall: BTreeSet::from_iter(["T".to_string()].into_iter()),
                name: "id".to_string(),
                args: vec![TypedBinding::new("a", Type::param("T"), false)],
                return_type: Type::param("T"),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::float(),
                body: Some(block_expr(vec![
                    let_stmt("x", call_expr("id", vec![12.34.into()])),
                    let_stmt("b", call_expr("id", vec![false.into()])),
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
                forall: BTreeSet::from_iter(["T".to_string()].into_iter()),
                name: "id".to_string(),
                args: vec![TypedBinding::new("a", Type::param("T"), false)],
                return_type: Type::param("T"),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
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
            forall: BTreeSet::new(),
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
            forall: BTreeSet::new(),
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
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::int(),
            body: Some(block_expr(vec![
                // Explicit return in true branch, implicit return in false.
                let_stmt(
                    "x",
                    lambda_expr(
                        vec![SoftBinding::new("x")],
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
            forall: BTreeSet::new(),
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
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::unit(),
            body: Some(block_expr(vec![
                let_stmt("x", 1),
                let_stmt("y", false),
                let_stmt("z", 19.95),
            ])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn factorial_fn() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "mul".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "sub".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "leq".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::bool_(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
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
                forall: BTreeSet::new(),
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
                forall: BTreeSet::new(),
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
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    // let add_5 = make_adder(5)
                    let_stmt("add5", call_expr("make_adder", vec![5.into()])),
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
            &[("left", Type::param("T")), ("right", Type::param("U"))],
        );

        let program = &mut Program(vec![
            Declaration::Struct(pair),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "p",
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
    fn struct_field_access_wrong_name() {
        let pair = StructDecl::new(
            "Pair",
            &[("left", Type::param("T")), ("right", Type::param("U"))],
        );

        let program = &mut Program(vec![
            Declaration::Struct(pair),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "p",
                        Expression::LiteralStruct {
                            name: "Pair".to_string(),
                            fields: HashMap::from_iter([
                                ("left".into(), true.into()),
                                ("right".into(), 123.into()),
                            ]),
                            ty: Type::unknown(),
                        },
                    ),
                    field("p", "whoopsie").into(),
                ])),
            }),
        ]);
        assert_eq!(
            typecheck_program(program),
            Err(Error::UnrecognizedField("whoopsie".into()))
        );
    }
    #[test]
    fn field_access_not_a_struct() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::bool_(),
            body: Some(block_expr(vec![
                let_stmt("p", ()),
                field("p", "whoopsie").into(),
            ])),
        })]);
        assert_eq!(
            typecheck_program(program),
            Err(Error::InapplicableConstraint(
                Type::unit(),
                Constraint::HasField("whoopsie".to_string(), Type::unknown())
            ))
        );
    }
    #[test]
    fn delayed_field_access_correct_field_names() {
        let pair = StructDecl::new(
            "Pair",
            &[("left", Type::param("T")), ("right", Type::param("U"))],
        );

        let program = &mut Program(vec![
            Declaration::Struct(pair),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::from_iter(["T".to_string()].into_iter()),
                name: "default".to_string(),
                args: vec![],
                return_type: Type::param("T"),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    // Initialize `p` with some polymorphic default fn.
                    let_mut_stmt("p", call_expr("default", vec![])),
                    // Use and constrain "p" before its type is known.
                    let_stmt(
                        "r",
                        if_expr(field("p", "left"), field("p", "right"), 42),
                    ),
                    // Assign to "p" to give it a type.
                    assign_stmt(
                        "p",
                        Expression::LiteralStruct {
                            name: "Pair".to_string(),
                            fields: HashMap::from_iter([
                                ("left".into(), true.into()),
                                ("right".into(), 123.into()),
                            ]),
                            ty: Type::unknown(),
                        },
                    ),
                    "r".into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn delayed_field_access_incorrect_field_names() {
        let pair = StructDecl::new(
            "Pair",
            &[("left", Type::param("T")), ("right", Type::param("U"))],
        );

        let program = &mut Program(vec![
            Declaration::Struct(pair),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::from_iter(["T".to_string()].into_iter()),
                name: "default".to_string(),
                args: vec![],
                return_type: Type::param("T"),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    // Initialize `p` with some polymorphic default fn.
                    let_mut_stmt("p", call_expr("default", vec![])),
                    // Use and constrain "p" before its type is known.
                    let_stmt(
                        "r",
                        if_expr(field("p", "left"), field("p", "whoopsie"), 42),
                    ),
                    // Assign to "p" to give it a type.
                    assign_stmt(
                        "p",
                        Expression::LiteralStruct {
                            name: "Pair".to_string(),
                            fields: HashMap::from_iter([
                                ("left".into(), true.into()),
                                ("right".into(), 123.into()),
                            ]),
                            ty: Type::unknown(),
                        },
                    ),
                    "r".into(),
                ])),
            }),
        ]);
        assert_eq!(
            typecheck_program(program),
            Err(Error::UnrecognizedField("whoopsie".to_string()))
        );
    }
    #[test]
    fn curried_pair_polymorphic_fn_test() {
        let pair = StructDecl::new(
            "Pair",
            &[("left", Type::param("T")), ("right", Type::param("U"))],
        );
        let pair_type = |left_type, right_type| {
            Type::struct_(StructInstance {
                decl: Rc::new(pair.clone()),
                params: BTreeMap::from_iter([
                    ("T".to_string(), left_type),
                    ("U".to_string(), right_type),
                ]),
            })
        };
        let program = &mut Program(vec![
            Declaration::Struct(pair.clone()),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::from_iter(["T".to_string(), "U".to_string()]),
                name: "make_pair".to_string(),
                args: vec![TypedBinding::new("a", Type::param("T"), false)],
                return_type: Type::func(
                    vec![Type::param("U")],
                    pair_type(Type::param("T"), Type::param("U")),
                ),
                body: Some(block_expr(vec![lambda_expr(
                    vec![SoftBinding::new("b")],
                    Expression::LiteralStruct {
                        name: "Pair".to_string(),
                        fields: HashMap::from_iter([
                            ("left".into(), "a".into()),
                            ("right".into(), "b".into()),
                        ]),
                        ty: Type::unknown(),
                    },
                )
                .into()])),
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: pair_type(
                    pair_type(Type::bool_(), Type::int()),
                    pair_type(Type::int(), Type::float()),
                ),
                body: Some(block_expr(vec![
                    let_stmt(
                        "bool_int",
                        call_expr(call_expr("make_pair", vec![true.into()]), vec![123.into()]),
                    ),
                    let_stmt(
                        "int_float",
                        call_expr(call_expr("make_pair", vec![123.into()]), vec![12.3.into()]),
                    ),
                    call_expr(
                        call_expr("make_pair", vec!["bool_int".into()]),
                        vec!["int_float".into()],
                    )
                    .into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn perform_anonymous_effect() {
        let mut ops = OpSetI::empty();
        ops.unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap();
        ops.mark_done_extending();
        let program = &mut Program(vec![Declaration::Co(CoDecl {
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::bool_(),
            ops,
            body: Some(block_expr(vec![
                let_stmt("p", perform_anon("foo", 1)),
                "p".into(),
            ])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn perform_named_effects() {
        let state_effect = EffectDecl {
            name: "State".to_string(),
            params: BTreeSet::from_iter(["T".to_string()]),
            ops: HashMap::from_iter([
                ("get".into(), (Type::unit(), Type::param("T"))),
                ("set".into(), (Type::param("T"), Type::unit())),
            ]),
        };
        let int_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: BTreeMap::from_iter([("T".to_string(), Type::int())]),
        };
        let bool_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: BTreeMap::from_iter([("T".to_string(), Type::bool_())]),
        };

        let mut ops = OpSetI::empty();
        ops.unify_add_declared_op(Some("int_state"), &int_state_instance, "get")
            .unwrap();
        ops.unify_add_declared_op(Some("int_state"), &int_state_instance, "set")
            .unwrap();
        ops.unify_add_declared_op(Some("bool_state"), &bool_state_instance, "get")
            .unwrap();
        ops.unify_add_declared_op(Some("bool_state"), &bool_state_instance, "set")
            .unwrap();
        ops.mark_done_extending();

        let program = &mut Program(vec![
            Declaration::Effect(state_effect),
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops,
                body: Some(block_expr(vec![
                    full_let_stmt("x", Type::int(), false, perform("int_state", "get", ())),
                    perform("int_state", "set", "x").into(),
                    full_let_stmt("y", Type::bool_(), false, perform("bool_state", "get", ())),
                    perform("bool_state", "set", "y").into(),
                    "y".into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn two_state_effects_ok_because_one_is_named() {
        let state_effect = EffectDecl {
            name: "State".to_string(),
            params: BTreeSet::from_iter(["T".to_string()]),
            ops: HashMap::from_iter([
                ("get".into(), (Type::unit(), Type::param("T"))),
                ("set".into(), (Type::param("T"), Type::unit())),
            ]),
        };
        let int_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: BTreeMap::from_iter([("T".to_string(), Type::int())]),
        };
        let bool_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: BTreeMap::from_iter([("T".to_string(), Type::bool_())]),
        };
        let mut ops = OpSetI::empty();
        ops.unify_add_declared_op(Some("int_state"), &int_state_instance, "get")
            .unwrap();
        assert!(ops
            .unify_add_declared_op(None, &bool_state_instance, "set")
            .is_ok());
    }
    #[test]
    fn conflicting_parametric_effects_both_unnamed() {
        let state_effect = EffectDecl {
            name: "State".to_string(),
            params: BTreeSet::from_iter(["T".to_string()]),
            ops: HashMap::from_iter([
                ("get".into(), (Type::unit(), Type::param("T"))),
                ("set".into(), (Type::param("T"), Type::unit())),
            ]),
        };
        let int_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: BTreeMap::from_iter([("T".to_string(), Type::int())]),
        };
        let bool_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: BTreeMap::from_iter([("T".to_string(), Type::bool_())]),
        };
        let mut ops = OpSetI::empty();
        ops.unify_add_declared_op(None, &int_state_instance, "get")
            .unwrap();
        // TODO: This probably could use a more informative error.
        assert_eq!(
            dbg!(ops.unify_add_declared_op(None, &bool_state_instance, "set")),
            Err(Error::NotUnifiable(Type::int(), Type::bool_()))
        );
    }
    #[test]
    fn empty_opset_is_identity_under_unification() {
        let state_effect = EffectDecl {
            name: "State".to_string(),
            params: BTreeSet::from_iter(["T".to_string()]),
            ops: HashMap::from_iter([
                ("get".into(), (Type::unit(), Type::param("T"))),
                ("set".into(), (Type::param("T"), Type::unit())),
            ]),
        };
        let int_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: BTreeMap::from_iter([("T".to_string(), Type::int())]),
        };
        let bool_state_instance = EffectInstance {
            decl: Rc::new(state_effect.clone()),
            params: BTreeMap::from_iter([("T".to_string(), Type::bool_())]),
        };
        let mut ops = OpSetI::empty();
        ops.unify_add_declared_op(Some("int_state"), &int_state_instance, "get")
            .unwrap();
        ops.unify_add_declared_op(Some("bool_state"), &bool_state_instance, "set")
            .unwrap();
        ops.unify_add_anonymous_effect("foo", &Type::unit(), &Type::float())
            .unwrap();
        ops.mark_done_extending();
        let original_ops = ops.clone();
        let ops = OpSet::new(ops);
        let was_empty = OpSet::new(OpSetI::empty());
        unify_opsets(ops.clone(), was_empty.clone()).unwrap();
        assert_eq!(ops.clone_inner(), original_ops);
        assert_eq!(was_empty.clone_inner(), original_ops);
    }
    #[test]
    fn test_propagate_subset_of_effects() {
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let mut foo_bar_ops = OpSetI::empty();
        foo_bar_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .mark_done_extending();

        let program = &mut Program(vec![
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops: foo_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("foo", 1)),
                    "p".into(),
                ])),
            }),
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "propagates_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops: foo_bar_ops,
                body: Some(block_expr(vec![propagate(call_expr(
                    "performs_foo",
                    vec![],
                ))
                .into()])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn test_propagate_subset_of_effects_fails_in_fn() {
        let mut ops = OpSetI::empty();
        ops.unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap();
        ops.mark_done_extending();
        let program = &mut Program(vec![
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops: ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("foo", 1)),
                    "p".into(),
                ])),
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "cannot_propagate_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                body: Some(block_expr(vec![propagate(call_expr(
                    "performs_foo",
                    vec![],
                ))
                .into()])),
            }),
        ]);
        // TODO: What would the real error be?
        assert_eq!(
            typecheck_program(program),
            Err(Error::OpSetNotExtendable(OpSetI::empty_non_extensible()))
        );
    }
    #[test]
    fn test_lambda_coroutine() {
        let mut ops = OpSetI::empty();
        ops.unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap();
        ops.mark_done_extending();
        let program = &mut Program(vec![
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops: ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("foo", 1)),
                    "p".into(),
                ])),
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "propagates_foo".to_string(),
                args: vec![],
                return_type: Type::func(vec![], Type::co(Type::bool_(), ops)),
                body: Some(block_expr(vec![
                    let_stmt(
                        "x",
                        lambda_expr(
                            vec![],
                            co_expr(block_expr(vec![return_stmt(propagate(call_expr(
                                "performs_foo",
                                vec![],
                            )))])),
                        ),
                    ),
                    "x".into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn test_propagate_two_sets_of_effects() {
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let mut bar_ops = OpSetI::empty();
        bar_ops
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .mark_done_extending();

        let mut foo_bar_ops = OpSetI::empty();
        foo_bar_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .mark_done_extending();

        let program = &mut Program(vec![
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops: foo_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("foo", 1)),
                    "p".into(),
                ])),
            }),
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_bar".to_string(),
                args: vec![],
                return_type: Type::float(),
                ops: bar_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("bar", ())),
                    "p".into(),
                ])),
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "propagates_foo".to_string(),
                args: vec![],
                return_type: Type::co(Type::unit(), foo_bar_ops),
                body: Some(block_expr(vec![co_expr(block_expr(vec![
                    let_stmt(
                        "f",
                        propagate(call_expr("performs_foo", vec![])),
                    ),
                    let_stmt(
                        "b",
                        propagate(call_expr("performs_bar", vec![])),
                    ),
                ]))
                .into()])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn delayed_opset_inference_okay() {
        // test the case of |c1, c2| co { c1?; c2?; } which requires proper propagation of
        // unknown opsets. Success case.
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let mut bar_ops = OpSetI::empty();
        bar_ops
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .mark_done_extending();

        let mut foo_bar_ops = OpSetI::empty();
        foo_bar_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .mark_done_extending();

        let program = &mut Program(vec![
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops: foo_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("foo", 1)),
                    "p".into(),
                ])),
            }),
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_bar".to_string(),
                args: vec![],
                return_type: Type::float(),
                ops: bar_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("bar", ())),
                    "p".into(),
                ])),
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::co(Type::unit(), foo_bar_ops.clone()),
                body: Some(block_expr(vec![
                    let_stmt(
                        "lambda",
                        lambda_expr(
                            vec![
                                SoftBinding::new("f"),
                                SoftBinding::new("b"),
                            ],
                            co_expr(block_expr(vec![
                                propagate("f").into(),
                                propagate("b").into(),
                                return_stmt(()),
                            ])),
                        ),
                    ),
                    call_expr(
                        "lambda",
                        vec![
                            call_expr("performs_foo", vec![]),
                            call_expr("performs_bar", vec![]),
                        ],
                    )
                    .into(),
                ])),
            }),
        ]);
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn delayed_opset_inference_unify_too_large() {
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let mut bar_ops = OpSetI::empty();
        bar_ops
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .mark_done_extending();
        let mut foo_bar_baz_ops = OpSetI::empty();
        foo_bar_baz_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .unify_add_anonymous_effect("baz", &Type::float(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let program = &mut Program(vec![
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops: foo_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("foo", 1)),
                    "p".into(),
                ])),
            }),
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_bar".to_string(),
                args: vec![],
                return_type: Type::float(),
                ops: bar_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("bar", ())),
                    "p".into(),
                ])),
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                // The returned co should not be empty. ERROR!
                return_type: Type::co(Type::unit(), foo_bar_baz_ops),
                body: Some(block_expr(vec![
                    let_stmt(
                        "lambda",
                        lambda_expr(
                            vec![
                                SoftBinding::new("f"),
                                SoftBinding::new("b"),
                            ],
                            co_expr(block_expr(vec![
                                propagate("f").into(),
                                propagate("b").into(),
                                return_stmt(()),
                            ])),
                        ),
                    ),
                    call_expr(
                        "lambda",
                        vec![
                            call_expr("performs_foo", vec![]),
                            call_expr("performs_bar", vec![]),
                        ],
                    )
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        // TODO: This test is incorrect, the program typechecks even though lambda returns a
        // coroutine that will never perform baz. This isn't unsound, but we want to catch those
        // errors.
        assert_eq!(result, Ok(()));
        // assert!(matches!(result, Err(Error::OpSetNotExtendable(_))), "result:`{:?}`", result);
    }
    #[test]
    fn delayed_opset_inference_unify_too_small() {
        // test the case of |c1, c2| co { c1?; c2?; } which requires proper propagation of
        // unknown opsets. Unify with a too-small opset.
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let mut bar_ops = OpSetI::empty();
        bar_ops
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .mark_done_extending();
        let program = &mut Program(vec![
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_foo".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                ops: foo_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("foo", 1)),
                    "p".into(),
                ])),
            }),
            Declaration::Co(CoDecl {
                forall: BTreeSet::new(),
                name: "performs_bar".to_string(),
                args: vec![],
                return_type: Type::float(),
                ops: bar_ops.clone(),
                body: Some(block_expr(vec![
                    let_stmt("p", perform_anon("bar", ())),
                    "p".into(),
                ])),
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                // The returned co should have foo and bar. ERROR.
                return_type: Type::co(Type::unit(), foo_ops),
                body: Some(block_expr(vec![
                    let_stmt(
                        "lambda",
                        lambda_expr(
                            vec![
                                SoftBinding::new("f"),
                                SoftBinding::new("b"),
                            ],
                            co_expr(block_expr(vec![
                                propagate("f").into(),
                                propagate("b").into(),
                                return_stmt(()),
                            ])),
                        ),
                    ),
                    call_expr(
                        "lambda",
                        vec![
                            call_expr("performs_foo", vec![]),
                            call_expr("performs_bar", vec![]),
                        ],
                    )
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        assert!(
            matches!(result, Err(Error::OpSetNotExtendable(_))),
            "result:`{:?}`",
            result
        );
    }
    #[test]
    fn perform_in_co_error() {
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            // The returned co should not be empty. ERROR!
            return_type: Type::co(Type::unit(), OpSetI::empty_non_extensible()),
            body: Some(block_expr(vec![co_expr(perform_anon("foo", 42)).into()])),
        })]);
        let result = typecheck_program(program);
        // Tried to extend the empty opset.
        assert!(
            matches!(result, Err(Error::OpSetNotExtendable(_))),
            "result:`{:?}`",
            result
        );
    }

    #[test]
    fn perform_in_co_good() {
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::co(Type::bool_(), foo_ops),
            body: Some(block_expr(vec![co_expr(perform_anon("foo", 42)).into()])),
        })]);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn coroutine_unification_error_different_ops() {
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let mut bar_ops = OpSetI::empty();
        bar_ops
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .mark_done_extending();

        let int_foo = Type::co(Type::int(), foo_ops);
        let int_bar = Type::co(Type::int(), bar_ops);
        let unification = unify(&int_foo, &int_bar);
        assert!(unification.is_err());
    }

    #[test]
    fn coroutine_unification_error_more_ops() {
        let mut foo_ops = OpSetI::empty();
        foo_ops
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let mut bar_ops = OpSetI::empty();
        bar_ops
            .unify_add_anonymous_effect("bar", &Type::unit(), &Type::float())
            .unwrap()
            .unify_add_anonymous_effect("foo", &Type::int(), &Type::bool_())
            .unwrap()
            .mark_done_extending();

        let int_foo = Type::co(Type::int(), foo_ops);
        let int_bar = Type::co(Type::int(), bar_ops);
        let unification = unify(&int_foo, &int_bar);
        assert!(unification.is_err());
    }
    #[test]
    fn error_expression_type_not_inferred() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::from_iter(["T".to_string()].into_iter()),
                name: "default".to_string(),
                args: vec![],
                return_type: Type::param("T"),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::unit(),
                body: Some(block_expr(vec![
                    let_stmt("foo", call_expr("default", vec![])),
                    return_stmt(()),
                ])),
            }),
        ]);
        assert!(matches!(
            typecheck_program(program),
            Err(Error::ExpressisonTypeNotInferred(Expression::Call { .. }))
        ));
    }
    #[test]
    fn propagate_co() {
        let program = &mut Program(vec![Declaration::Fn(FnDecl {
            forall: BTreeSet::new(),
            name: "main".to_string(),
            args: vec![],
            return_type: Type::int(),
            body: Some(block_expr(vec![
                let_stmt(
                    "performs_bar",
                    co_expr(perform_anon("bar", 53)),
                ),
                propagate("performs_bar").into(),
            ])),
        })]);
        let result = typecheck_program(program);
        assert!(matches!(result, Err(Error::OpSetNotExtendable(_))));
    }
    #[test]
    fn handle_two_anonymous_effects() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "performs_foo",
                        co_expr(perform_anon("foo", 53)),
                    ),
                    let_stmt(
                        "performs_bar",
                        co_expr(perform_anon("bar", 53)),
                    ),
                    let_stmt(
                        "both",
                        co_expr(call_expr(
                            "plus",
                            vec![propagate("performs_bar"), propagate("performs_foo")],
                        )),
                    ),
                    Expression::Handle {
                        co: Box::new("both".into()),
                        ty: Type::unknown(),
                        return_arm: None,
                        op_arms: vec![
                            HandleOpArm {
                                op_name: "foo".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: 42.into(), // Ignore the effect.
                            },
                            HandleOpArm {
                                op_name: "bar".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: 42.into(), // Ignore the effect.
                            },
                        ],
                    }
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn handle_two_anonymous_effects_resuming() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "something".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::float(), false),
                ],
                return_type: Type::bool_(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::bool_(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "performs_foo",
                        co_expr(perform_anon("foo", 53.42)),
                    ),
                    let_stmt(
                        "performs_bar",
                        co_expr(perform_anon("bar", 53)),
                    ),
                    let_stmt(
                        "calls_something",
                        co_expr(call_expr(
                            "something",
                            vec![propagate("performs_bar"), propagate("performs_foo")],
                        )),
                    ),
                    Expression::Handle {
                        co: Box::new("calls_something".into()),
                        ty: Type::unknown(),
                        return_arm: None,
                        op_arms: vec![
                            HandleOpArm {
                                op_name: "foo".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: block_expr(vec![resume_stmt("x")]),
                            },
                            HandleOpArm {
                                op_name: "bar".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: block_expr(vec![resume_stmt("x")]),
                            },
                        ],
                    }
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn handles_foo_but_not_bar() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "performs_foo_and_bar",
                        co_expr(block_expr(vec![
                            full_let_stmt("bar", Type::bool_(), false, perform_anon("bar", 4.2)),
                            perform_anon("foo", 53).into(),
                        ])),
                    ),
                    Expression::Handle {
                        co: Box::new("performs_foo_and_bar".into()),
                        ty: Type::unknown(),
                        return_arm: None,
                        op_arms: vec![HandleOpArm {
                            op_name: "foo".to_string(),
                            performed_variable: SoftBinding::new("x"),
                            body: "x".into(),
                        }],
                    }
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        assert!(matches!(result, Err(Error::OpSetNotExtendable(_))));
    }
    #[test]
    fn handles_bar_incorrectly() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "performs_foo_and_bar",
                        co_expr(block_expr(vec![
                            full_let_stmt("bar", Type::bool_(), false, perform_anon("bar", 4.2)),
                            perform_anon("foo", 53).into(),
                        ])),
                    ),
                    Expression::Handle {
                        co: Box::new("performs_foo_and_bar".into()),
                        ty: Type::unknown(),
                        return_arm: None,
                        op_arms: vec![
                            HandleOpArm {
                                op_name: "foo".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: "x".into(),
                            },
                            HandleOpArm {
                                op_name: "bar".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: "x".into(),
                            },
                        ],
                    }
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        assert_eq!(result, Err(Error::NotUnifiable(Type::float(), Type::int())));
    }
    #[test]
    fn handles_irrelevant() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "plus".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::int(), false),
                ],
                return_type: Type::int(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "performs_foo",
                        co_expr(block_expr(vec![perform_anon("foo", 53).into()])),
                    ),
                    Expression::Handle {
                        co: Box::new("performs_foo".into()),
                        ty: Type::unknown(),
                        return_arm: None,
                        op_arms: vec![
                            HandleOpArm {
                                op_name: "foo".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: "x".into(),
                            },
                            HandleOpArm {
                                op_name: "irrelevant".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: block_expr(vec![
                                    resume_stmt(()), // Give the resume a unit type.
                                    "x".into(),      // Returned x makes perform type an int.
                                ]),
                            },
                        ],
                    }
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        // TODO: This test is incorrect. While the program typechecks, it should be rejected because
        // the "irrelevant" arm is handling an effect that will never be performed. This should be
        // at least a warning, if not an error.
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn handle_return_arm_unifies() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "something".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::float(), false),
                ],
                return_type: Type::bool_(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "performs_foo",
                        co_expr(perform_anon("foo", 53.42)),
                    ),
                    let_stmt(
                        "performs_bar",
                        co_expr(perform_anon("bar", 53)),
                    ),
                    let_stmt(
                        "calls_something",
                        co_expr(call_expr(
                            "something",
                            vec![propagate("performs_bar"), propagate("performs_foo")],
                        )),
                    ),
                    Expression::Handle {
                        co: Box::new("calls_something".into()),
                        ty: Type::unknown(),
                        return_arm: Some((
                            SoftBinding::new("r"),
                            Box::new(12.into()),
                        )),
                        op_arms: vec![
                            HandleOpArm {
                                op_name: "foo".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: block_expr(vec![resume_stmt("x")]),
                            },
                            HandleOpArm {
                                op_name: "bar".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: block_expr(vec![42.into()]),
                            },
                        ],
                    }
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn handle_return_arm_does_not_unify() {
        let program = &mut Program(vec![
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "something".to_string(),
                args: vec![
                    TypedBinding::new("a", Type::int(), false),
                    TypedBinding::new("b", Type::float(), false),
                ],
                return_type: Type::bool_(),
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![
                    let_stmt(
                        "performs_foo",
                        co_expr(perform_anon("foo", 53.42)),
                    ),
                    let_stmt(
                        "performs_bar",
                        co_expr(perform_anon("bar", 53)),
                    ),
                    let_stmt(
                        "calls_something",
                        co_expr(call_expr(
                            "something",
                            vec![propagate("performs_bar"), propagate("performs_foo")],
                        )),
                    ),
                    Expression::Handle {
                        co: Box::new("calls_something".into()),
                        ty: Type::unknown(),
                        return_arm: Some((
                            SoftBinding::new("r"),
                            Box::new(12.into()),
                        )),
                        op_arms: vec![
                            HandleOpArm {
                                op_name: "foo".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                // Never type.
                                body: block_expr(vec![resume_stmt("x")]),
                            },
                            HandleOpArm {
                                op_name: "bar".to_string(),
                                performed_variable: SoftBinding::new("x"),
                                body: block_expr(vec![
                                    42.32.into(), // Not an int!
                                ]),
                            },
                        ],
                    }
                    .into(),
                ])),
            }),
        ]);
        let result = typecheck_program(program);
        assert_eq!(result, Err(Error::NotUnifiable(Type::float(), Type::int())));
    }
}
