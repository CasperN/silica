// Abstract syntax tree and type checking.
#![allow(clippy::result_large_err)]
use crate::parse::{
    CoDecl, Declaration, EffectDecl, FnDecl, ParsedOp, ParsedType, Program, StructDecl,
};
use crate::union_find::UnionFindRef;
use std::cell::{Ref, RefMut};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
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
    Forall(Vec<String>, Type),

    // Unknown type variable. Unifies with any other type.
    // Should not appear in generic types.
    Unknown(Vec<Constraint>),

    // A type parameter. It should not appear during unification as
    // types need to be instantiated before then.
    Param(String),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Constraint {
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
    pub fn forall(params: Vec<String>, ty: Type) -> Self {
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
                let mut opset_inner = ops.mut_inner();
                for (p, r) in opset_inner.anonymous_ops.values_mut() {
                    p.instantiate_with(subs);
                    r.instantiate_with(subs);
                }
                for effect in opset_inner.named_effects.values_mut() {
                    match effect {
                        DeclaredOps::Known(instance, _ops) => {
                            for ty in instance.params.values_mut() {
                                ty.instantiate_with(subs);
                            }
                        }
                        DeclaredOps::Unknown(ops) => {
                            for (p, r) in ops.values_mut() {
                                p.instantiate_with(subs);
                                r.instantiate_with(subs);
                            }
                        }
                    }
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
}

// *************************************************************************************************
// OpSets
// *************************************************************************************************

// Union find wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct OpSet(UnionFindRef<OpSetI>);

#[derive(Debug, Clone, PartialEq)]
enum DeclaredOps {
    Known(EffectInstance, HashSet<String>),
    Unknown(HashMap<String, (Type, Type)>),
}

// A set of ops that are being performed.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct OpSetI {
    anonymous_ops: HashMap<String, (Type, Type)>,
    named_effects: HashMap<String, DeclaredOps>,
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
        for (effect_name, effect) in other.named_effects.iter() {
            match effect {
                DeclaredOps::Known(instance, ops) => {
                    for op_name in ops {
                        self_inner.unify_add_declared_op(Some(effect_name), instance, op_name)?;
                    }
                }
                DeclaredOps::Unknown(ops) => {
                    for (op, (p, r)) in ops.iter() {
                        self_inner.unify_add_unknown_declared_op(effect_name, op, p, r)?;
                    }
                }
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
    // An Opset is concrete if all the types therein are concrete.
    // Note that we consider extendable OpSets to be concrete.
    fn is_concrete(&self) -> bool {
        if !self.iter_types().all(|ty| ty.is_concrete()) {
            return false;
        }
        for e in self.named_effects.values() {
            if let DeclaredOps::Unknown(_) = e {
                return false;
            }
        }
        true
    }
    fn get(&self, name_or_effect: Option<&str>, op_name: &str) -> Option<(Type, Type)> {
        if name_or_effect.is_none() {
            // It must be a decl free / anonymous op.
            return self.anonymous_ops.get(op_name).cloned();
        }
        let name_or_effect = name_or_effect.unwrap();
        match self.named_effects.get(name_or_effect) {
            Some(DeclaredOps::Known(instance, ops)) => {
                if ops.contains(op_name) {
                    Some(instance.unwrap_op_type(op_name))
                } else {
                    None
                }
            }
            Some(DeclaredOps::Unknown(ops)) => ops.get(op_name).cloned(),
            None => None,
        }
    }
    fn iter_types(&self) -> impl Iterator<Item = Type> + '_ {
        let mut types = Vec::new();
        for (p, r) in self.anonymous_ops.values() {
            types.push(p.clone());
            types.push(r.clone());
        }
        for effect in self.named_effects.values() {
            match effect {
                DeclaredOps::Known(instance, _) => {
                    types.extend(instance.params.values().cloned());
                }
                DeclaredOps::Unknown(ops) => {
                    for (p, r) in ops.values() {
                        types.push(p.clone());
                        types.push(r.clone());
                    }
                }
            }
        }
        types.into_iter()
    }
    fn remove_anonymous_effect_or_die(&mut self, name: &str) {
        if self.anonymous_ops.remove(name).is_none() {
            panic!("Expected {name} in {self:?}");
        }
    }
    fn remove_named_effect_or_die(&mut self, effect_name: &str, op_name: &str) {
        match self.named_effects.get_mut(effect_name) {
            None => panic!("Expected {effect_name} effect in {self:?}"),
            Some(DeclaredOps::Known(_, ops)) => {
                if !ops.remove(op_name) {
                    panic!("Expected {effect_name}.{op_name} in {self:?}");
                }
            }
            Some(DeclaredOps::Unknown(ops)) => {
                if ops.remove(op_name).is_none() {
                    panic!("Expected {effect_name}.{op_name} in {self:?}");
                }
            }
        }
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

    fn unify_add_unknown_declared_op(
        &mut self,
        effect_name: &str,
        op_name: &str,
        perform_type: &Type,
        resume_type: &Type,
    ) -> Result<&mut Self, Error> {
        match self.named_effects.entry(effect_name.to_string()) {
            Entry::Vacant(entry) => {
                if self.done_extending {
                    return Err(Error::OpSetNotExtendable(self.clone()));
                }
                let mut ops = HashMap::new();
                ops.insert(
                    op_name.to_string(),
                    (perform_type.clone(), resume_type.clone()),
                );
                entry.insert(DeclaredOps::Unknown(ops));
                Ok(self)
            }
            Entry::Occupied(mut entry) => match entry.get_mut() {
                DeclaredOps::Unknown(ops) => match ops.entry(op_name.to_string()) {
                    Entry::Vacant(entry) => {
                        if self.done_extending {
                            return Err(Error::OpSetNotExtendable(self.clone()));
                        }
                        entry.insert((perform_type.clone(), resume_type.clone()));
                        Ok(self)
                    }
                    Entry::Occupied(entry) => {
                        let (p, r) = entry.get();
                        unify(perform_type, p)?;
                        unify(resume_type, r)?;
                        Ok(self)
                    }
                },
                DeclaredOps::Known(instance, ops) => {
                    if !instance.decl.ops.contains_key(op_name) {
                        return Err(Error::NoMatchingOpInEffectDecl(
                            op_name.to_string(),
                            (*instance.decl).clone(),
                        ));
                    }
                    if !ops.contains(op_name) && self.done_extending {
                        return Err(Error::OpSetNotExtendable(self.clone()));
                    }
                    let (p, r) = instance.unwrap_op_type(op_name);
                    unify(perform_type, &p)?;
                    unify(resume_type, &r)?;
                    Ok(self)
                }
            },
        }
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
                let effect_entry = entry.get_mut();
                let mut new_effect_entry = None;
                match effect_entry {
                    DeclaredOps::Known(existing_instance, existing_ops) => {
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
                    DeclaredOps::Unknown(existing_ops) => {
                        if !existing_ops.contains_key(op_name) && self.done_extending {
                            return Err(Error::OpSetNotExtendable(self.clone()));
                        }
                        let mut new_ops = HashSet::new();
                        new_ops.insert(op_name.to_string());
                        for (op, (p, r)) in existing_ops.iter() {
                            if !instance.decl.ops.contains_key(op) {
                                return Err(Error::NoMatchingOpInEffectDecl(
                                    op_name.to_string(),
                                    (*instance.decl).clone(),
                                ));
                            }
                            let (p2, r2) = instance.unwrap_op_type(op);
                            unify(p, &p2)?;
                            unify(r, &r2)?;
                            new_ops.insert(op.to_string());
                        }
                        new_effect_entry = Some(DeclaredOps::Known(instance.clone(), new_ops));
                    }
                }
                if let Some(e) = new_effect_entry {
                    *effect_entry = e;
                }
            }
            Entry::Vacant(entry) => {
                if self.done_extending {
                    return Err(Error::OpSetNotExtendable(self.clone()));
                }
                entry.insert(DeclaredOps::Known(
                    instance.clone(),
                    HashSet::from_iter([op_name.to_string()]),
                ));
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
    fn get_declared_op(&self, effect: &str, op: &str) -> Option<(Type, Type)> {
        self.named_effects.get(effect).and_then(|e| match e {
            DeclaredOps::Known(decl, ops) => {
                if ops.contains(op) {
                    Some(decl.unwrap_op_type(op))
                } else {
                    None
                }
            }
            DeclaredOps::Unknown(ops) => ops.get(op).cloned(),
        })
    }
    fn contains_opset(&self, other: &OpSet) -> bool {
        self.super_sets.iter().any(|s| s.contains_opset(other))
            && self.iter_types().any(|t| t.contains_opset(other))
    }
}

// *************************************************************************************************
//  Resolve Types
// *************************************************************************************************

// TODO: Resolved structs and effects shouldn't have UNSPECIFIED in there.

fn resolve_struct(mut shadow: ShadowTypeContext, parsed: &StructDecl) -> Result<StructType, Error> {
    let mut fields = HashMap::new();
    for param in parsed.params.iter() {
        shadow.insert_type_param(param);
    }
    for (field_name, field_ty) in parsed.fields.iter() {
        let ty = resolve_type(shadow.context(), field_ty)?;
        if fields.insert(field_name.clone(), ty).is_some() {
            panic!("Duplicate field name {field_name} that should have been caught earlier.");
        }
    }

    shadow.finish();
    Ok(StructType {
        params: parsed.params.clone(),
        name: parsed.name.clone(),
        fields,
    })
}

fn resolve_effect(mut shadow: ShadowTypeContext, parsed: &EffectDecl) -> Result<EffectType, Error> {
    for param in parsed.params.iter() {
        shadow.insert_type_param(param);
    }
    let mut ops = HashMap::new();
    for op in parsed.ops.iter() {
        match op {
            ParsedOp::Anonymous(name, p_ty, r_ty) => {
                let p_ty = resolve_type(shadow.context(), p_ty)?;
                let r_ty = resolve_type(shadow.context(), r_ty)?;
                if ops.insert(name.clone(), (p_ty, r_ty)).is_some() {
                    panic!("Duplicate anonymous op {name} that should have been caught earlier.");
                }
            }
            ParsedOp::NamedEffect { .. } => todo!(),
        }
    }
    Ok(EffectType {
        name: parsed.name.clone(),
        params: parsed.params.clone(),
        ops,
    })
}

fn resolve_fn_type(mut shadow: ShadowTypeContext, parsed: &FnDecl) -> Result<Type, Error> {
    let mut params = Vec::new();
    for param in parsed.forall.iter() {
        shadow.insert_type_param(param);
        params.push(param.clone());
        // TODO: what if dup?
    }
    let mut arg_types = Vec::new();
    for arg_binding in parsed.args.iter() {
        arg_types.push(resolve_type(shadow.context(), &arg_binding.parsed_type)?);
    }
    let ret = resolve_type(shadow.context(), &parsed.return_type)?;
    shadow.finish();
    Ok(Type::forall(params, Type::func(arg_types, ret)))
}

fn resolve_co_fn_type(mut shadow: ShadowTypeContext, parsed: &CoDecl) -> Result<Type, Error> {
    let mut params = Vec::new();
    for param in parsed.forall.iter() {
        shadow.insert_type_param(param);
        params.push(param.clone());
        // TODO: what if dup?
    }
    let mut arg_types = Vec::new();
    for arg_binding in parsed.args.iter() {
        arg_types.push(resolve_type(shadow.context(), &arg_binding.parsed_type)?);
    }
    let ret = resolve_type(shadow.context(), &parsed.return_type)?;
    let ops = resolve_ops(shadow.context(), &parsed.ops)?;
    shadow.finish();
    Ok(Type::forall(
        params,
        Type::func(arg_types, Type::co(ret, ops)),
    ))
}

fn resolve_type(context: &TypeContext, ty: &ParsedType) -> Result<Type, Error> {
    match ty {
        ParsedType::Unspecified => Ok(Type::unknown()),
        ParsedType::Unit => Ok(Type::unit()),
        ParsedType::Int => Ok(Type::int()),
        ParsedType::Bool => Ok(Type::bool_()),
        ParsedType::Float => Ok(Type::float()),
        ParsedType::Never => Ok(Type::never()),
        ParsedType::Ops(_) => panic!(),
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
            let opset = resolve_ops(context, ops)?;
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
                Some(NamedItem::TypeParam(param)) => {
                    if !parsed_params.is_empty() {
                        return Err(Error::HigherKindedTypesNotSupported(
                            name.to_string(),
                            parsed_params.to_vec(),
                        ));
                    }
                    Ok(Type::param(param))
                }
                found => Err(Error::ExpectedNamedType(name.clone(), found.cloned())),
            }
        }
    }
}

fn resolve_ops(context: &TypeContext, ops: &[ParsedOp]) -> Result<OpSet, Error> {
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
                params,
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
                            // FIXME: decl.params is sorted in alphabetical order, but this "zip"
                            // assumes it to be in declaration order.
                            let mut resolved_params = BTreeMap::new();
                            for (name, param) in decl.params.iter().zip(params.iter()) {
                                resolved_params.insert(name.clone(), resolve_type(context, param)?);
                            }
                            let instance = EffectInstance {
                                params: resolved_params,
                                decl: decl.clone(),
                            };
                            entry.insert(DeclaredOps::Known(instance, ops));
                        }
                        Entry::Occupied(mut entry) => match entry.get_mut() {
                            DeclaredOps::Known(_decl, ops) => {
                                if let Some(op_name) = op_name {
                                    if ops.insert(op_name.clone()) {
                                        return Err(Error::DuplicateOpName(op_name.clone()));
                                    }
                                } else {
                                    return Err(Error::DuplicateNamedEffect(effect.clone()));
                                }
                            }
                            DeclaredOps::Unknown(_) => {
                                panic!("Resolving opset with unknown ops?! {op:?}")
                            }
                        },
                    }
                } else {
                    return Err(Error::ExpectedNamedEffect(effect.clone()));
                }
            }
        }
    }
    Ok(OpSet::new(OpSetI {
        anonymous_ops,
        named_effects,
        super_sets: Vec::new(),
        done_extending: true,
    }))
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
        bindings: Vec<Binding>,
        body: Box<Expression>,
        lambda_type: Type,
    },
    Co(Box<Expression>, Type, OpSet),
    Perform {
        effect_name: Option<String>,
        op_name: String,
        arg: Box<Expression>,
        resume_type: Type,
    },
    Propagate(Box<Expression>, Type),
    Handle {
        co: Box<Expression>,
        // Initially arm, finally arm.
        return_arm: Option<(Binding, Box<Expression>)>,
        op_arms: Vec<HandleOpArm>,
        ty: Type,
    }, // TODO: UFCS/method-call, ref/deref.
}

#[derive(Debug, Clone, PartialEq)]
pub struct HandleOpArm {
    pub effect_name: Option<String>,
    pub op_name: String,
    // TODO: This only handles anonymous effects. What about named effects?
    pub performed_variable: Binding,
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
                effect_name: _,
                op_name: _,
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
    Assign(Expression, Expression),
    Let { binding: Binding, value: Expression },
    Expression(Expression),
    Return(Expression),
    Resume(Expression),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: String,
    pub parsed_type: ParsedType,
    pub ty: Type,
    pub mutable: bool,
}
#[allow(dead_code)]
impl Binding {
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
    pub fn new_typed(name: impl Into<String>, parsed_type: ParsedType) -> Self {
        Self {
            name: name.into(),
            parsed_type,
            ty: Type::unknown(),
            mutable: false,
        }
    }
    pub fn new_typed_mut(name: impl Into<String>, parsed_type: ParsedType) -> Self {
        Self {
            name: name.into(),
            parsed_type,
            ty: Type::unknown(),
            mutable: true,
        }
    }
    pub fn has_specified_type(&self) -> bool {
        self.parsed_type.is_specified()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructType {
    pub name: String,
    pub params: Vec<String>,
    pub fields: HashMap<String, Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectType {
    pub name: String,
    pub params: Vec<String>,
    pub ops: HashMap<String, (Type, Type)>,
}
impl EffectType {
    fn unwrap_op_type(&self, op: &str) -> (Type, Type) {
        self.ops
            .get(op)
            .unwrap_or_else(|| panic!("Op {op} not in effect decl: {self:?}"))
            .clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructInstance {
    params: BTreeMap<String, Type>,
    decl: Rc<StructType>,
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
    decl: Rc<EffectType>,
}
impl EffectInstance {
    fn unwrap_op_type(&self, op_name: &str) -> (Type, Type) {
        let (mut p, mut r) = self.decl.unwrap_op_type(op_name);
        p.instantiate_with(&self.params);
        r.instantiate_with(&self.params);
        (p, r)
    }
}

#[derive(PartialEq, Debug, Clone)]
pub(crate) enum NamedItem {
    Variable(VariableInfo),
    Struct(Rc<StructType>),
    Effect(Rc<EffectType>),
    TypeParam(String),
}

#[derive(PartialEq, Debug, Clone)]
pub(crate) struct VariableInfo {
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
}
impl<'a> ShadowTypeContext<'a> {
    fn insert_named_item(&mut self, name: String, item: NamedItem) {
        if let Entry::Vacant(entry) = self.shadowed_variables.entry(name.clone()) {
            let original = self.type_context.names.insert(name, item);
            entry.insert(original);
        } else {
            // The original was already saved in `shadowed_variables`,
            // no need to touch that.
            self.type_context.names.insert(name, item);
        }
    }
    fn insert_variable(&mut self, name: String, ty: Type, mutable: bool) {
        let info = NamedItem::Variable(VariableInfo { ty, mutable });
        self.insert_named_item(name, info);
    }
    fn insert_binding(&mut self, binding: Binding) -> Type {
        let Binding {
            name,
            ty,
            mutable,
            parsed_type: _,
        } = binding;
        self.insert_variable(name, ty.clone(), mutable);
        ty
    }
    fn insert_type_param(&mut self, name: &str) {
        let item = NamedItem::TypeParam(name.to_string());
        self.insert_named_item(name.to_string(), item)
    }
    fn set_resume_type(&mut self, ty: &Type) {
        if self.shadowed_return_type.is_some() {
            panic!("Resume type set multiple times.");
        }
        self.shadowed_resume_type = Some(self.context().resume_type.take());
        self.context().resume_type = Some(ty.clone());
    }

    // Defines a struct. Returns if there was a previous definition.
    fn define_struct(&mut self, decl: StructType) -> bool {
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
    fn define_effect(&mut self, decl: EffectType) -> bool {
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
pub(crate) enum Error {
    UnknownName(String),
    NoSuchStruct(String),
    NotUnifiable(Type, Type),
    InfiniteType(Type, Type),
    DuplicateArgNames(Expression),
    DuplicateTopLevelName(String),
    AssignToImmutableBinding(String),
    UnrecognizedField(String),
    PerformedEffectsNotallowedInContext {
        effect_or_instance: Option<String>,
        op_name: String,
    },
    DuplicateOpName(String),
    NoMatchingOpInEffectDecl(String, EffectType),
    FnDeclMustHaveConcreteTypes(FnDecl),
    CoDeclMustHaveConcreteTypes(CoDecl),
    StructDeclMustHaveConcreteTypes(StructType),
    EffectDeclMustHaveConcreteTypes(EffectType),
    EffectDeclMismatch(EffectType, EffectType),
    OpSetNotExtendable(OpSetI),
    InapplicableConstraint(Type, Constraint),
    ExpressisonTypeNotInferred(Expression),
    UnexpectedResume(Expression),
    ParameterCountMismatch(String, Vec<ParsedType>),
    HigherKindedTypesNotSupported(String, Vec<ParsedType>),
    ExpectedNamedType(String, Option<NamedItem>),
    ExpectedNamedEffect(String),
    DuplicateNamedEffect(String),
    InvalidLValue(Expression),
}

// *************************************************************************************************
//  Inference
// *************************************************************************************************
// TODO: Move resolution over here.
// TODO: Break up type checking and type inference, i.e. bidirectional type checking?

/// Checks if an expression is a valid L-value (a "place").
/// If it is, and if it's mutable, returns the type of the place.
/// Otherwise, returns an appropriate error.
fn check_lvalue(context: &TypeContext, expr: &Expression) -> Result<Type, Error> {
    match expr {
        Expression::L(LValue::Variable(name), ty) => {
            let var_info = context
                .variable_info(name)
                .ok_or_else(|| Error::UnknownName(name.clone()))?;

            if !var_info.mutable {
                return Err(Error::AssignToImmutableBinding(name.clone()));
            }
            Ok(ty.clone())
        }
        Expression::L(LValue::Field(obj_expr, _field_name), ty) => {
            check_lvalue(context, obj_expr)?;
            Ok(ty.clone())
        }
        _ => {
            // Any other kind of expression (a literal, a function call, an arithmetic operation, etc.)
            // is an R-value and cannot be assigned to.
            Err(Error::InvalidLValue(expr.clone()))
        }
    }
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
                shadow.insert_binding(binding.clone());
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
            effect_name,
            op_name,
            arg,
            resume_type: expr_ty,
        } => {
            infer(context, arg)?;

            if let Some((perform_ty, resume_ty)) = context
                .ops
                .clone_inner()
                .get(effect_name.as_deref(), op_name)
            {
                unify(&arg.get_type(), &perform_ty)?;
                unify(expr_ty, &resume_ty)?;
                Ok(())
            } else if !context.ops.is_extendable() {
                Err(Error::PerformedEffectsNotallowedInContext {
                    effect_or_instance: effect_name.clone(),
                    op_name: op_name.clone(),
                })
            } else if effect_name.is_none() {
                // Perform an anonymous effect.
                context.ops.mut_inner().unify_add_anonymous_effect(
                    op_name,
                    &arg.get_type(),
                    expr_ty,
                )?;
                Ok(())
            } else {
                // Structurally constrain a declared effect.
                context.ops.mut_inner().unify_add_unknown_declared_op(
                    effect_name.as_ref().unwrap(),
                    op_name,
                    &arg.get_type(),
                    expr_ty,
                )?;
                Ok(())
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
                if let Some(effect_name) = &arm.effect_name {
                    co_ops.mut_inner().unify_add_unknown_declared_op(
                        effect_name,
                        &arm.op_name,
                        &Type::unknown(),
                        &Type::unknown(),
                    )?;
                } else {
                    co_ops.mut_inner().unify_add_anonymous_effect(
                        &arm.op_name,
                        &Type::unknown(),
                        &Type::unknown(),
                    )?;
                }
            }
            unify(
                &co.get_type(),
                &Type::co(co_return_ty.clone(), co_ops.clone()),
            )?;
            // Infer the return arm.
            if let Some((binding, expr)) = return_arm {
                let mut shadow = context.shadow();
                let original_return_ty = shadow.insert_binding(binding.clone());
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
                let p_var_ty = shadow.insert_binding(arm.performed_variable.clone());
                let (p_ty, r_ty) = if let Some(effect_name) = &arm.effect_name {
                    co_ops
                        .inner()
                        .get_declared_op(effect_name, &arm.op_name)
                        .unwrap()
                } else {
                    co_ops.inner().get_anonymous_op(&arm.op_name).unwrap()
                };
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
                if let Some(effect_name) = &arm.effect_name {
                    remaining_ops.remove_named_effect_or_die(effect_name, &arm.op_name);
                } else {
                    remaining_ops.remove_anonymous_effect_or_die(&arm.op_name);
                }
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
                    Statement::Assign(left, right) => {
                        infer(shadow.context(), left)?;
                        infer(shadow.context(), right)?;
                        let lvalue_type = check_lvalue(shadow.context(), left)?;
                        unify(&lvalue_type, &right.get_type())?;
                        last_statement_type = Type::unit()
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
        Statement::Assign(left, right) => {
            check_expression_concrete(left)?;
            check_expression_concrete(right)?;
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

pub fn typecheck_program(program: &mut Program) -> Result<(), Error> {
    // First, load declarations into typing context.
    // TODO: Allow for circular dependencies between types at the top level.
    let mut context = TypeContext::new();
    let mut shadow = context.shadow();
    for declaration in program.0.iter() {
        match declaration {
            Declaration::Fn(fn_decl) => {
                let local_context = shadow.context();
                let mut local_shadow = local_context.shadow();
                for param in fn_decl.forall.iter() {
                    local_shadow.insert_type_param(param);
                }

                for binding in fn_decl.args.iter() {
                    unify(
                        &binding.ty,
                        &resolve_type(local_shadow.context(), &binding.parsed_type)?,
                    )?;
                    if !binding.ty.is_concrete() {
                        return Err(Error::FnDeclMustHaveConcreteTypes(fn_decl.clone()));
                    }
                }
                if !resolve_type(local_shadow.context(), &fn_decl.return_type)?.is_concrete() {
                    return Err(Error::FnDeclMustHaveConcreteTypes(fn_decl.clone()));
                }
                local_shadow.finish();
                if shadow.context_contains(&fn_decl.name) {
                    return Err(Error::DuplicateTopLevelName(fn_decl.name.clone()));
                }
                let fn_type = resolve_fn_type(shadow.context().shadow(), fn_decl)?;
                shadow.insert_variable(fn_decl.name.clone(), fn_type, false);
            }
            Declaration::Co(co_decl) => {
                let local_context = shadow.context();
                let mut local_shadow = local_context.shadow();
                for param in co_decl.forall.iter() {
                    local_shadow.insert_type_param(param);
                }
                for binding in co_decl.args.iter() {
                    unify(
                        &binding.ty,
                        &resolve_type(local_shadow.context(), &binding.parsed_type)?,
                    )?;
                    if !binding.ty.is_concrete() {
                        return Err(Error::CoDeclMustHaveConcreteTypes(co_decl.clone()));
                    }
                }
                for op in co_decl.ops.iter() {
                    if !op.is_specified() {
                        return Err(Error::CoDeclMustHaveConcreteTypes(co_decl.clone()));
                    }
                }
                if !resolve_type(local_shadow.context(), &co_decl.return_type)?.is_concrete() {
                    return Err(Error::CoDeclMustHaveConcreteTypes(co_decl.clone()));
                }
                local_shadow.finish();
                if shadow.context_contains(&co_decl.name) {
                    return Err(Error::DuplicateTopLevelName(co_decl.name.clone()));
                }
                let mut arg_types = vec![];
                for binding in co_decl.args.iter() {
                    arg_types.push(binding.ty.clone());
                }
                let co_fn_ty = resolve_co_fn_type(shadow.context().shadow(), co_decl)?;
                shadow.insert_variable(co_decl.name.clone(), co_fn_ty, false);
            }
            Declaration::Struct(struct_declaration) => {
                let struct_type = resolve_struct(shadow.context().shadow(), struct_declaration)?;

                if !struct_type.fields.values().all(|ty| ty.is_concrete()) {
                    return Err(Error::StructDeclMustHaveConcreteTypes(struct_type));
                }

                let redefined = shadow.define_struct(struct_type.clone());
                if redefined {
                    return Err(Error::DuplicateTopLevelName(struct_type.name.clone()));
                }
            }
            Declaration::Effect(effect_declaration) => {
                let effect_type = resolve_effect(shadow.context().shadow(), effect_declaration)?;
                if !effect_type
                    .ops
                    .values()
                    .all(|(p, r)| p.is_concrete() && r.is_concrete())
                {
                    return Err(Error::EffectDeclMustHaveConcreteTypes(effect_type));
                }
                let redefined = shadow.define_effect(effect_type.clone());
                if redefined {
                    return Err(Error::DuplicateTopLevelName(effect_type.name.clone()));
                }
            }
        }
    }
    // Second, typecheck function bodies.
    for declaration in program.0.iter_mut() {
        match declaration {
            Declaration::Fn(FnDecl {
                forall,
                name: _,
                args,
                return_type,
                body,
            }) => {
                if body.is_none() {
                    continue; // External fn, take it for granted.
                }
                // Local shadow.
                let mut shadow = shadow.context().shadow();
                for param in forall {
                    shadow.insert_type_param(param);
                }
                let return_type = resolve_type(shadow.context(), return_type)?;
                let body = body.as_mut().unwrap();
                // Declare a new shadow to insert fn variables.
                let mut shadow = shadow
                    .context()
                    .enter_activation_frame(return_type.clone(), OpSet::empty_non_extensible());
                for binding in args.iter() {
                    shadow.insert_binding(binding.clone());
                }
                infer(shadow.context(), body)?;
                unify(&return_type, &body.get_type())?;
                shadow.finish();
                check_expression_concrete(body)?;
            }
            Declaration::Co(CoDecl {
                forall,
                name: _,
                args,
                return_type,
                ops: parsed_ops,
                body,
            }) => {
                if body.is_none() {
                    continue; // External fn, take it for granted.
                }
                // Local shadow.
                let mut shadow = shadow.context().shadow();
                for param in forall {
                    shadow.insert_type_param(param);
                }
                let return_type = resolve_type(shadow.context(), return_type)?;
                let ops = resolve_ops(shadow.context(), parsed_ops)?;

                let body = body.as_mut().unwrap();
                // Declare a new shadow to insert fn variables.
                let mut shadow = shadow
                    .context()
                    .enter_activation_frame(return_type.clone(), ops);
                for binding in args.iter() {
                    shadow.insert_binding(binding.clone());
                }
                infer(shadow.context(), body)?;
                unify(&return_type, &body.get_type())?;
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
#[allow(dead_code)]
pub mod test_helpers {
    use super::*;

    pub fn let_stmt(name: &str, value: impl Into<Expression>) -> Statement {
        Statement::Let {
            binding: Binding::new(name),
            value: value.into(),
        }
    }
    pub fn let_mut_stmt(name: &str, value: impl Into<Expression>) -> Statement {
        Statement::Let {
            binding: Binding::new_mut(name),
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
    pub fn assign_stmt(left: impl Into<Expression>, right: impl Into<Expression>) -> Statement {
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
    pub fn lambda_expr(bindings: Vec<Binding>, body: impl Into<Expression>) -> Expression {
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
            effect_name: None,
            op_name: op.to_string(),
            arg: Box::new(arg.into()),
            resume_type: Type::unknown(),
        }
    }
    pub fn perform(name: &str, op: &str, arg: impl Into<Expression>) -> Expression {
        Expression::Perform {
            effect_name: Some(name.to_string()),
            op_name: op.to_string(),
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
    use super::*;
    use crate::parse::parse_program_or_die;

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
        let program = &mut parse_program_or_die(
            r#"
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> i64 {
                let a = 2;
                let b = 2;
                let c = plus(a, b);
                c
            }
        "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn shadow_earlier_variables() {
        let program = &mut parse_program_or_die(
            r#"
            fn main() -> bool {
                let a = 2;
                let a = 2.0;
                let a = true;
                a
            }
        "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn assign_to_mutable_binding() {
        let program = &mut parse_program_or_die(
            r#"
            fn main(mut a: i64) -> unit {
                a = 3;
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn error_assign_wrong_type_to_mutable_binding() {
        let program = &mut parse_program_or_die(
            r#"
            fn main() -> unit {
                let mut a = 2;
                a = ();
            }
            "#,
        );
        assert_eq!(
            typecheck_program(program),
            Err(Error::NotUnifiable(Type::int(), Type::unit()))
        );
    }
    #[test]
    fn error_assign_to_immutable_binding() {
        let program = &mut parse_program_or_die(
            r#"
            fn main() -> unit {
                let a = 2;
                a = 3;
            }
            "#,
        );
        assert_eq!(
            typecheck_program(program),
            Err(Error::AssignToImmutableBinding("a".to_string()))
        );
    }
    #[test]
    fn if_expression_unifies() {
        let program = &mut parse_program_or_die(
            r#"
            fn round(a: f64) -> i64;
            fn main() -> i64 {
                let a = true;
                let b = 2;
                let c = 2.0;
                if a then b else round(c)
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn use_lambda() {
        let program = &mut parse_program_or_die(
            r#"
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> i64 {
                let plus_two = |x| plus(2, x);
                let mut b = 2;
                b = plus_two(b);
                b
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn return_lambda() {
        let program = &mut parse_program_or_die(
            r#"
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> fn(i64) -> i64 {
                let plus_two = |x| plus(2, x);
                plus_two
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn return_in_lambda() {
        let program = &mut parse_program_or_die(
            r#"
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> fn(i64) -> i64 {
                let plus_two = |x| { return plus(2, x); };
                return plus_two;
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn use_polymorphic_top_level_fn() {
        let program = &mut parse_program_or_die(
            r#"
            fn id<T>(a: T) -> T;
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> i64 {
                plus(id(2), id(2))
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn use_polymorphic_top_level_fn_polymorphically() {
        let program = &mut parse_program_or_die(
            r#"
            fn id<T>(a: T) -> T;
            fn main() -> f64 {
                let x = id(12.34);
                let b = id(false);
                if b then x else 23.45
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn error_polymorphic_top_level_fn() {
        let program = &mut parse_program_or_die(
            r#"
            fn id<T>(a: T) -> T;
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> i64 {
                plus(id(2), id(2.0))
            }
            "#,
        );
        let t = typecheck_program(program);
        assert!(t.is_err());
    }

    #[test]
    fn return_in_one_branch() {
        let program = &mut parse_program_or_die(
            r#"
            fn main() -> i64 {
                if true then {
                    return 2;
                } else {
                    3
                }
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn error_return_wrong_type_in_one_branch() {
        let program = &mut parse_program_or_die(
            r#"
            fn main() -> i64 {
                if true then {
                    return 2.5;
                } else {
                    3
                }
            }
            "#,
        );
        assert!(matches!(
            typecheck_program(program),
            Err(Error::NotUnifiable(_, _))
        ));
    }
    #[test]
    fn error_infinite_type() {
        let program = &mut parse_program_or_die(
            r#"
            fn main() -> i64 {
                let x = |x| x(x);
                2
            }
            "#,
        );
        assert!(matches!(
            typecheck_program(program),
            Err(Error::InfiniteType(_, _))
        ));
    }

    #[test]
    fn empty_block_has_type_unit() {
        let program = &mut parse_program_or_die(
            r#"
        fn main() -> unit {
            // implicit return unit
        }
        "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn block_with_just_lets_has_type_unit() {
        let program = &mut parse_program_or_die(
            r#"
            fn main() -> unit {
                let x = 1;
                let y = false;
                let z = 19.95;
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn factorial_fn() {
        let program = &mut parse_program_or_die(
            r#"
            fn mul(a: i64, b: i64) -> i64;
            fn sub(a: i64, b: i64) -> i64;
            fn leq(a: i64, b: i64) -> bool;

            fn factorial(x: i64) -> i64 {
                if leq(x, 1) then 1 else mul(x, factorial(sub(x, 1)))
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn higher_order_functions() {
        let program = &mut parse_program_or_die(
            r#"
            fn apply(f: fn(i64) -> i64, x: i64) -> i64;
            fn make_adder(y: i64) -> fn(i64) -> i64;

            fn main() -> i64 {
                let add5 = make_adder(5);
                apply(add5, 10)
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn struct_field_access() {
        let program = &mut parse_program_or_die(
            r#"
            struct Pair<T, U> {
                left: T,
                right: U,
            }

            fn main() -> bool {
                let p = Pair { left: true, right: 123 };
                p.left
            }
            "#,
        );
        assert_eq!(typecheck_program(program), Ok(()));
    }
    #[test]
    fn struct_field_access_wrong_name() {
        let program = &mut parse_program_or_die(
            r#"
            struct Pair<T, U> {
                left: T,
                right: U,
            }

            fn main() -> bool {
                let p = Pair { left: true, right: 123 };
                p.whoopsie
            }
            "#,
        );
        assert_eq!(
            typecheck_program(program),
            Err(Error::UnrecognizedField("whoopsie".into()))
        );
    }
    #[test]
    fn field_access_not_a_struct() {
        let source = r#"
        fn main() -> bool {
            let p = ();
            p.whoopsie
        }
        "#;
        let program = &mut parse_program_or_die(source);
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
        let source = r#"
        struct Pair<T, U> {
            left: T,
            right: U,
        }
        fn default<T>() -> T;
        fn main() -> i64 {
            let mut p = default();
            let r = if p.left then p.right else 42;
            p = Pair { left: true, right: 123 };
            r
        }
        "#;
        let program = &mut parse_program_or_die(source);
        assert_eq!(typecheck_program(program), Ok(()));
    }

    #[test]
    fn delayed_field_access_incorrect_field_names() {
        let result = typecheck_program(&mut parse_program_or_die(
            r"
            struct Pair<T, U> {
                left: T,
                right: U,
            }
            fn default<T>() -> T;
            fn main() -> i64 {
                // Initialize `p` with some unknown polymorphic type.
                let mut p = default();
                // Use `p`
                let r = if p.left then p.whoopsie else p.right;
                // Only later constrain `p` and find out that it has no `whoopsie` field.
                p = Pair { left: true, right: 123 };
                r
            }
            ",
        ));
        assert_eq!(
            result,
            Err(Error::UnrecognizedField("whoopsie".to_string()))
        );
    }
    #[test]
    fn curried_pair_polymorphic_fn_test() {
        let result = typecheck_program(&mut parse_program_or_die(
            r"
            struct Pair<T, U> {
                left: T,
                right: U,
            }
            fn make_pair<T, U>(a: T) -> fn(U) -> Pair<T, U> {
                |b| Pair { left: a, right: b }
            }
            fn main() -> Pair<Pair<bool, i64>, Pair<i64, f64>> {
                let bool_int = make_pair(true)(123);
                let int_float = make_pair(123)(12.3);
                make_pair(bool_int)(int_float)
            }
            ",
        ));
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn perform_anonymous_effect() {
        let result = typecheck_program(&mut parse_program_or_die(
            r"
            co main() -> bool ! foo(i64 -> bool) {
                perform foo(1)
            }
            ",
        ));
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn perform_anonymous_effect_parameterized() {
        let result = typecheck_program(&mut parse_program_or_die(
            r"
            co run_foo<T, U>(x: T) -> U ! foo(T -> U) {
                perform foo(x)
            }
            fn main() -> i64 {
                run_foo(1.2) handle { foo(x: f64) => { resume 42 } }
            }
            ",
        ));
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn perform_named_effects() {
        let result = typecheck_program(&mut parse_program_or_die(
            r"
            effect State<T> {
                get: unit -> T,
                set: T -> unit,
            }
            co main() ! int_state: State<i64>, bool_state: State<f64> {
                let x = perform int_state.get(());
                perform int_state.set(42);
                let y = perform bool_state.get(());
                perform bool_state.set(y);
            }
            ",
        ));
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn two_state_effects_ok_because_one_is_named() {
        let source = r#"
        effect State<T> {
            get: unit -> T,
            set: T -> unit,
        }

        co main() ! int_state: State<i64>, State<bool> {
            perform int_state.get(());
            perform State.set(true); // Unnamed, defaults to 'State'
        }
        "#;
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
    }
    #[test]
    fn conflicting_parametric_effects_both_unnamed() {
        let state_effect = EffectType {
            name: "State".to_string(),
            params: vec!["T".to_string()],
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
        let state_effect = EffectType {
            name: "State".to_string(),
            params: vec!["T".to_string()],
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
    // TODO Replace above test with this one. Requires support for performing named effects.
    // #[test]
    // fn conflicting_parametric_effects_both_unnamed2() {
    //     let source = r#"
    //     effect State<T> {
    //         get: unit -> T,
    //         set: T -> unit,
    //     }
    //     fn main() {
    //         let impossible = co {
    //             perform State.get(()); // First usage infers T=i64 for the 'State' effect
    //             perform State.set(true); // Second usage infers T=bool
    //         };
    //     }
    //     "#;
    //     // TODO: This probably could use a more informative error.
    //     assert_eq!(typecheck_program(&mut parse_program_or_die(source)),
    //                 Err(Error::NotUnifiable(Type::int(), Type::bool_())));
    // }
    #[test]
    fn test_propagate_subset_of_effects() {
        let result = typecheck_program(&mut parse_program_or_die(
            r"
            co performs_foo() -> bool ! foo(i64 -> bool) {
                perform foo(1)
            }
            co propagates_foo_and_bar() -> bool ! foo(i64 -> bool), bar(unit -> f64) {
                perform bar(());
                performs_foo()?
            }
            ",
        ));
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn test_propagate_subset_of_effects_fails_in_fn() {
        let result = typecheck_program(&mut parse_program_or_die(
            r"
            co performs_foo() -> bool ! foo(i64 -> bool) {
                perform foo(1)
            }
            fn cannot_propagate_foo() -> bool {
                performs_foo()?
            }
            ",
        ));
        // TODO: More informative error message?
        assert_eq!(
            result,
            Err(Error::OpSetNotExtendable(OpSetI::empty_non_extensible()))
        );
    }
    #[test]
    fn test_lambda_coroutine() {
        let result = typecheck_program(&mut parse_program_or_die(
            r"
            co performs_foo() -> bool ! foo(i64 -> bool) {
                perform foo(1)
            }
            fn propagates_foo() -> fn() -> Co<bool ! foo(i64 -> bool)> {
                || co { return performs_foo()? }
            }
            ",
        ));

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn test_propagate_two_sets_of_effects() {
        let mut program = parse_program_or_die(
            r"
            co performs_foo() -> bool ! foo(i64->bool), baz(i64 -> unit) {
                let f = perform foo(1);
                perform baz(42);
                f
            }
            co performs_bar() -> f64 ! bar(unit -> f64), baz(i64->unit) {
                let b = perform bar(());
                perform baz(42);
                b
            }
            co main() ! foo(i64 -> bool), bar(unit -> f64), baz(i64 -> unit) {
                let f = performs_foo()?;
                let b = performs_bar()?;
            }
            ",
        );
        let result = typecheck_program(&mut program);
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn delayed_opset_inference_okay() {
        // test the case of |c1, c2| co { c1?; c2?; } which requires proper propagation of
        // unknown opsets. Success case.
        let mut program = parse_program_or_die(
            r"
            co performs_foo() -> bool ! foo(i64->bool) {
                perform foo(1)
            }
            co performs_bar() -> f64 ! bar(unit -> f64) {
                perform bar(())
            }
            co main() ! foo(i64 -> bool), bar(unit -> f64) {
                let lambda = |f, b| co { f?; b?; return (); };
                lambda(performs_foo(), performs_bar())?
            }
            ",
        );
        let result = typecheck_program(&mut program);
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn delayed_opset_inference_unify_too_large() {
        let mut program = parse_program_or_die(
            r"
            co performs_foo() -> bool ! foo(i64->bool) {
                perform foo(1)
            }
            co performs_bar() -> f64 ! bar(unit -> f64) {
                perform bar(())
            }
            co main() ! foo(i64 -> bool), bar(unit -> f64), baz(f64 -> bool) {
                let lambda = |f, b| co { f?; b?; return (); };
                lambda(performs_foo(), performs_bar())?
            }
            ",
        );
        let result = typecheck_program(&mut program);
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
        let mut program = parse_program_or_die(
            r"
            co performs_foo() -> bool ! foo(i64->bool) {
                perform foo(1)
            }
            co performs_bar() -> f64 ! bar(unit -> f64) {
                perform bar(())
            }
            co main() ! foo(i64 -> bool) {
                let lambda = |f, b| { f?; b?; return (); };
                // Main cannot perform bar.
                lambda(performs_foo(), performs_bar())
            }
            ",
        );
        let result = typecheck_program(&mut program);
        assert!(
            matches!(result, Err(Error::OpSetNotExtendable(_))),
            "result:`{:?}`",
            result
        );
    }
    #[test]
    fn perform_in_co_error() {
        let source = r#"
        fn main() -> Co<unit> {
            co { perform foo(42) }
        }
        "#;
        let result = typecheck_program(&mut parse_program_or_die(source));
        // Tried to extend the empty opset.
        assert!(
            matches!(result, Err(Error::OpSetNotExtendable(_))),
            "result:`{:?}`",
            result
        );
    }

    #[test]
    fn perform_in_co_good() {
        let source = r#"
        fn main() -> Co<bool ! foo(i64 -> bool)> {
            co { perform foo(42) }
        }
        "#;
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
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
        let mut program = parse_program_or_die(
            r"
            fn default<T>() -> T;
            fn main() {
                let foo = default();
            }
            ",
        );
        assert!(matches!(
            typecheck_program(&mut program),
            Err(Error::ExpressisonTypeNotInferred(Expression::Call { .. }))
        ));
    }
    #[test]
    fn propagate_co() {
        let mut program = parse_program_or_die(
            r"
            fn main() -> i64 {
                let performs_bar = co perform bar(53);
                performs_bar?
            }
            ",
        );
        let result = typecheck_program(&mut program);
        assert!(matches!(result, Err(Error::OpSetNotExtendable(_))));
    }
    #[test]
    fn handle_two_anonymous_effects() {
        let mut program = parse_program_or_die(
            r"
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> i64 {
                let performs_foo = co perform foo(53);
                let performs_bar = co perform bar(53);
                let calls_something = co plus(performs_bar?, performs_foo?);
                calls_something handle {
                    foo(x) => { resume 42 },
                    bar(x) => { resume 42 },
                }
            }
            ",
        );
        let result = typecheck_program(&mut program);
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn handle_two_anonymous_effects_resuming() {
        let mut program = parse_program_or_die(
            r"
            fn something(a: i64, b: f64) -> bool;
            fn main() -> bool {
                let performs_foo = co perform foo(53.42);
                let performs_bar = co perform bar(53);
                let calls_something = co something(performs_bar?, performs_foo?);
                calls_something handle {
                    foo(x) => { resume x },
                    bar(x) => { resume x },
                }
            }
            ",
        );
        let result = typecheck_program(&mut program);
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn handles_foo_but_not_bar() {
        let mut program = parse_program_or_die(
            r"
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> i64 {
                let performs_foo_and_bar = co {
                    let bar: bool = perform bar(4.2);
                    perform foo(53)
                };
                performs_foo_and_bar handle {
                    foo(x) => x,
                    // `bar` not handled
                }
            }
            ",
        );
        let result = typecheck_program(&mut program);
        // TODO: This is kind of an opaque error.
        assert!(matches!(result, Err(Error::OpSetNotExtendable(_))));
    }
    #[test]
    fn handles_bar_incorrectly() {
        let mut program = parse_program_or_die(
            r"
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> i64 {
                let performs_foo_and_bar = co {
                    let bar: bool = perform bar(4.2);
                    perform foo(53)
                };
                performs_foo_and_bar handle {
                    foo(x) => x,
                    bar(x) => x,
                }
            }
            ",
        );
        let result = typecheck_program(&mut program);
        assert_eq!(result, Err(Error::NotUnifiable(Type::float(), Type::int())));
    }
    #[test]
    fn handles_irrelevant() {
        let mut program = parse_program_or_die(
            r"
            fn plus(a: i64, b: i64) -> i64;
            fn main() -> i64 {
                let performs_foo = co perform foo(53);
                performs_foo handle {
                    foo(x) => x,
                    // It can be inferred that irrelevant is unit -> i64 so typechecking passes.
                    irrelevant(x) => { resume(()); x },
                }
            }
            ",
        );
        let result = typecheck_program(&mut program);
        // TODO: This test is incorrect. While the program typechecks, it should be rejected because
        // the "irrelevant" arm is handling an effect that will never be performed. This should be
        // at least a warning, if not an error.
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn handle_return_arm_unifies() {
        let mut program = parse_program_or_die(
            r"
            fn something(a: i64, b: f64) -> bool;
            fn main() -> i64 {
                let performs_foo = co perform foo(53.42);
                let performs_bar = co perform bar(53);
                let calls_something = co something(performs_bar?, performs_foo?);
                calls_something handle {
                    return r => { 12 },
                    foo(x) => { resume x },
                    bar(x) => { resume 42 },
                }
            }
            ",
        );

        let result = typecheck_program(&mut program);
        assert_eq!(result, Ok(()));
    }
    #[test]
    fn handle_return_arm_does_not_unify() {
        let mut program = parse_program_or_die(
            r"
            fn something(a: i64, b: f64) -> bool;

            fn main() -> bool {
                let performs_foo = co perform foo(53.42);
                let performs_bar = co perform bar(53);
                let calls_something = co something(performs_bar?, performs_foo?);
                calls_something handle {
                    return r => (),
                    foo(x) => { resume(x) },
                    bar(x) => { resume(42) },
                }
            }
            ",
        );
        let result = typecheck_program(&mut program);
        assert_eq!(
            result,
            Err(Error::NotUnifiable(Type::bool_(), Type::unit()))
        );
    }
    #[test]
    fn unspecified_type_in_struct() {
        let result = typecheck_program(&mut parse_program_or_die(r" struct Foo { foo: _ } "));
        assert!(matches!(
            result,
            Err(Error::StructDeclMustHaveConcreteTypes(_))
        ));
    }

    #[test]
    fn unspecified_type_in_effect() {
        let result = typecheck_program(&mut parse_program_or_die(
            r" effect Foo { foo: unit -> _ } ",
        ));
        assert!(matches!(
            result,
            Err(Error::EffectDeclMustHaveConcreteTypes(_))
        ));
    }
    #[test]
    fn unspecified_type_in_fn_arg() {
        let result = typecheck_program(&mut parse_program_or_die(r" fn foo(x: _); "));
        assert!(matches!(result, Err(Error::FnDeclMustHaveConcreteTypes(_))));
    }
    #[test]
    fn unspecified_type_in_fn_return() {
        let result = typecheck_program(&mut parse_program_or_die(r" fn foo(x: f64) -> _; "));
        assert!(
            matches!(result, Err(Error::FnDeclMustHaveConcreteTypes(_))),
            "{:?}",
            result
        );
    }
    #[test]
    fn unspecified_type_in_fn_return_parameter() {
        let result = typecheck_program(&mut parse_program_or_die(
            r" 
            struct Foo<T> { foo: T }
            fn foo(x: f64) -> Foo<_>;
            ",
        ));
        assert!(
            matches!(result, Err(Error::FnDeclMustHaveConcreteTypes(_))),
            "{:?}",
            result
        );
    }
    #[test]
    fn unspecified_type_in_co_arg() {
        let result = typecheck_program(&mut parse_program_or_die(r" co foo(x: _); "));
        assert!(matches!(result, Err(Error::CoDeclMustHaveConcreteTypes(_))));
    }
    #[test]
    fn unspecified_type_in_co_anon_op() {
        let result = typecheck_program(&mut parse_program_or_die(r" co foo() ! foo(unit -> _); "));
        assert!(
            matches!(result, Err(Error::CoDeclMustHaveConcreteTypes(_))),
            "{:?}",
            result
        );
    }
    #[test]
    fn unspecified_type_in_co_return() {
        let result = typecheck_program(&mut parse_program_or_die(r" co foo(x: f64) -> _; "));
        assert!(
            matches!(result, Err(Error::CoDeclMustHaveConcreteTypes(_))),
            "{:?}",
            result
        );
    }
    #[test]
    fn field_assignment() {
        let source = r#"
        struct Point { x: i64, y: i64 }

        fn main() -> i64 {
            let mut p = Point { x: 1, y: 2 };
            p.x = 10;
            p.x
        }
        "#;
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
    }
    #[test]
    fn field_assignment_type_error() {
        let source = r#"
        struct Point { x: i64, y: i64 }

        fn main() -> i64 {
            let mut p = Point { x: 1, y: 2 };
            p.x = 10.1;
            0
        }
        "#;
        assert_eq!(
            typecheck_program(&mut parse_program_or_die(source)),
            Err(Error::NotUnifiable(Type::int(), Type::float()))
        );
    }
    #[test]
    fn assign_to_invalid_lvalue() {
        let source = r#"
        fn main() -> i64 {
            10 = 20;
            1
        }
        "#;
        assert_eq!(
            typecheck_program(&mut parse_program_or_die(source)),
            Err(Error::InvalidLValue(10.into()))
        );
    }
    #[test]
    fn assign_to_field_of_field() {
        let source = r#"
        struct Pair<L, R> { left: L, right: R }
        fn main() -> i64 {
            let mut p = Pair { left: Pair { left: 3, right: 2.0 }, right: 3.0 };
            p.left.left = 4;
            p.left.right = 3.2;
            p.left.left
        }
        "#;
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
    }

    #[test]
    fn perform_declared_effect() {
        let source = r#"
        effect State<T> { get: unit -> T, set: T -> unit }
        co foo() -> i64 ! s: State<i64> {
            perform s.get(())
        }
        "#;
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
    }
    #[test]
    fn handle_declared_effect() {
        let source = r#"
        effect State<T> { get: unit -> T, set: T -> unit }
        co run_state(x: i64) -> i64 ! State<i64>;
        fn main() -> i64 {
            run_state(42) handle {
                State.get(x) => { resume(44) },
                State.set(s) => { resume(()) },
            }
        }
        "#;
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
    }
    #[test]
    fn error_handle_structural_named_effect_not_bound_to_decl() {
        let source = r#"
        effect State<T> { get: unit -> T, set: T -> unit }
        fn main() -> i64 {
            let c = co {
                let s = perform State.get(());
                perform State.set(s);
                42
            };
            c handle {
                State.get(x) => { resume(44) },
                State.set(s) => { resume(()) },
            }
        }
        "#;
        let result = typecheck_program(&mut parse_program_or_die(source));
        // The co expression type is not concrete because we don't know that it binds to the State
        // effect: It could be some _other_ effect, locally named "State", with get and set.
        assert!(
            matches!(result, Err(Error::ExpressisonTypeNotInferred(_))),
            "{result:?}"
        );
    }
    #[test]
    fn bind_structural_effect_to_declaration_via_variable_binding() {
        let source = r#"
        effect State<T> { get: unit -> T, set: T -> unit }
        fn main() -> i64 {
            let c: Co< _ ! State<_>> = co {
                let s = perform State.get(());
                perform State.set(s);
                42
            };
            c handle {
                State.get(x) => { resume(44) },
                State.set(s) => { resume(()) },
            }
        }
        "#;
        // The type annotation on `c` is required for this to pass, as it binds the named effect
        // `State` to the actual declared `State` effect.
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
    }
    #[test]
    fn error_perform_effect_with_inconsistent_parameterized_type() {
        let source = r#"
        effect State<T> { get: unit -> T, set: T -> unit }
        fn main() {
            let c: Co< _ ! State<_>> = co {
                let s = perform State.get(());
                perform State.set(100);
            };
            c handle {
                State.get(x) => { resume(3.14) },
                State.set(s) => { resume(()) },
            }
        }
        "#;
        let result = typecheck_program(&mut parse_program_or_die(source));
        assert_eq!(result, Err(Error::NotUnifiable(Type::float(), Type::int())));
    }
    #[test]
    fn perform_multiple_named_effects() {
        let source = r#"
        effect State<T> { get: unit -> T, set: T -> unit }
        co do_stuff() ! b: State<bool>, i: State<i64> {
            if perform b.get(()) then {
                perform i.set(42);
            } else {
                perform i.set(300);
            }
        }
        fn main() {
            let mut i_state = 0;
            do_stuff() handle {
                b.get(x) => { resume(true) },
                b.set(s) => { resume(()) },
                i.get(x) => { resume(i_state) },
                i.set(x) => { i_state = x; resume(()) },
            }
        }
        "#;
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
    }
    #[test]
    fn handle_named_effects_in_parts() {
        let source = r#"
        effect State<T> { get: unit -> T, set: T -> unit }
        co do_stuff() ! b: State<bool>, i: State<i64> {
            if perform b.get(()) then {
                perform i.set(42);
            } else {
                perform i.set(300);
            }
        }
        fn main() {
            let mut i_state = 0;
            // `d` is a coroutine that only handled some of do_stuff();
            let d = co do_stuff() handle {
                b.get(x) => { resume(true) },
                i.set(x) => { i_state = x; resume(()) },
            };
            d handle {
                b.set(s) => { resume(()) },
                i.get(x) => { resume(i_state) },
            }
        }
        "#;
        assert_eq!(typecheck_program(&mut parse_program_or_die(source)), Ok(()));
    }
    #[test]
    fn error_unconstrained_inferred_effect_signature() {
        let mut program = parse_program_or_die(
            r#"
            fn default<T>() -> T;
            fn main() -> unit {
                let c = co {
                    perform foo(default()); // The perform type is inferred as Unknown
                };
                // 'c' is never used, so the effect { foo(_ -> unit) } is not
                // constrained further. This should be an error.
            }
            "#,
        );
        assert!(matches!(
            typecheck_program(&mut program),
            Err(Error::ExpressisonTypeNotInferred(_))
        ));
    }
    #[test]
    fn error_unconstrained_type_in_handled_effect() {
        let mut program = parse_program_or_die(
            r#"
            fn default<T>() -> T;
            fn main() -> i64 { // The function signature itself is concrete.
                let c = co {
                    perform get_state(());
                };

                c handle {
                    // The 'get_state' effect is handled and does not escape.
                    get_state(p) => { resume(default()) }, // The resume type is Unknown!

                    // The handle expression has a concrete return type.
                    return v => 42,
                }
            }
            "#,
        );
        // The resume type for 'get_state' is inferred to be an unconstrained
        // polymorphic type `T`, which should be an error.
        assert!(matches!(
            typecheck_program(&mut program),
            Err(Error::ExpressisonTypeNotInferred(_))
        ));
    }

    // TODO: Recursive types? Mutually recursive types? Forward declarations?
    // TODO: Top level comments in parse?
}
