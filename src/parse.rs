use std::collections::{hash_map::Entry, HashMap, HashSet};

use crate::ast::{Binding, Expression, HandleOpArm, LValue, OpSet, Statement, Type};
use tree_sitter::{Language, Node, Parser};

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

struct SourceTree<'source> {
    source: &'source str,
    tree: tree_sitter::Tree,
}
impl<'source> SourceTree<'source> {
    fn parse(source: &'source str, errors: &mut Vec<ParseError<'source>>) -> Option<Self> {
        if let Some(tree) = parse(source) {
            Some(Self { source, tree })
        } else {
            errors.push(ParseError::NoTree);
            None
        }
    }
    fn root<'tree>(
        &'tree self,
        errors: &mut Vec<ParseError<'source>>,
    ) -> Option<SourceNode<'source, 'tree>> {
        // SourceNode::new assumes we've already verified root_node and only verifies that the
        // immediate children are valid.
        let root_node = self.tree.root_node();
        if root_node.kind() != "source_file" {
            errors.push(ParseError::UnexpectedNodeType {
                expected: "source_file",
                found: root_node.kind(),
                node_text: self.source,
            });
        }
        SourceNode::new(self.tree.root_node(), self.source, errors)
    }
}

#[derive(Clone, Copy)]
struct SourceNode<'source, 'tree> {
    source: &'source str,
    node: Node<'tree>,
}
impl<'s, 't> std::fmt::Debug for SourceNode<'s, 't> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:?}", self.kind(), self.text())
    }
}

impl<'t, 's> SourceNode<'s, 't> {
    fn new(node: Node<'t>, source: &'s str, errors: &mut Vec<ParseError<'s>>) -> Option<Self> {
        if node.is_error() || node.is_missing() {
            // Assume the error has already been added by the caller.
            return None;
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.is_error() {
                errors.push(ParseError::Unparsable {
                    context_kind: node.kind(),
                    text: child.utf8_text(source.as_bytes()).unwrap(),
                });
            }
        }
        Some(SourceNode { node, source })
    }

    fn kind(&self) -> &'static str {
        self.node.kind()
    }

    fn text(self) -> &'s str {
        // self.source is utf8 and the node's span shouldn't slice it in an invalid way, so
        // utf8 decoding should not fail unless there is an error in tree-sitter or the grammar.
        self.node
            .utf8_text(self.source.as_bytes())
            .expect("Node text decode failed")
    }

    fn optional_child(self, field: &str, errors: &mut Vec<ParseError<'s>>) -> Option<Self> {
        self.node
            .child_by_field_name(field)
            .and_then(|n| Self::new(n, self.source, errors))
    }

    // Returns the child if its there. If not, appends an error to `errors``.
    fn required_child<'e>(
        self,
        field_name: &'static str,
        errors: &'e mut Vec<ParseError<'s>>,
    ) -> Option<Self> {
        if let Some(node) = self.optional_child(field_name, errors) {
            if !node.node.is_missing() && !node.node.is_error() && !node.text().is_empty() {
                return Some(node);
            }
        }
        errors.push(ParseError::MissingField {
            node_kind: self.kind(),
            field_name,
        });
        None
    }

    fn optional_child_by_id(
        self,
        field_id: usize,
        errors: &mut Vec<ParseError<'s>>,
    ) -> Option<Self> {
        self.node
            .child(field_id)
            .and_then(|node| Self::new(node, self.source, errors))
    }

    // Returns the child if its there. If not, appends an error to `errors``.
    fn required_child_by_id<'e>(
        self,
        field_id: usize,
        errors: &'e mut Vec<ParseError<'s>>,
    ) -> Option<Self> {
        if let Some(node) = self.optional_child_by_id(field_id, errors) {
            if !node.node.is_missing() && !node.node.is_error() && !node.text().is_empty() {
                return Some(node);
            }
        }
        errors.push(ParseError::MissingIndex {
            node_kind: self.node.kind(),
            child_index: field_id,
        });
        None
    }

    fn children(self, errors: &mut Vec<ParseError<'s>>) -> Vec<Self> {
        let mut cursor = self.node.walk();
        let mut children = Vec::new();
        for node in self.node.children(&mut cursor) {
            if let Some(c) = Self::new(node, self.source, errors) {
                children.push(c);
            }
        }
        children
    }
    fn children_by_field_name(
        self,
        field_name: &str,
        errors: &mut Vec<ParseError<'s>>,
    ) -> Vec<Self> {
        let mut cursor = self.node.walk();
        let mut children = Vec::new();
        for node in self.node.children_by_field_name(field_name, &mut cursor) {
            if let Some(c) = Self::new(node, self.source, errors) {
                children.push(c);
            }
        }
        children
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError<'source> {
    UnexpectedNodeType {
        expected: &'static str,
        found: &'source str,
        node_text: &'source str,
    },
    MissingField {
        node_kind: &'static str,
        field_name: &'source str,
    },
    MissingIndex {
        node_kind: &'static str,
        child_index: usize,
    },
    InvalidLiteral {
        kind: &'static str,
        text: &'source str,
        error: String,
    },
    DuplicateItem {
        duplicated_item: &'source str,
        item_type: &'static str,
        context_name: &'source str,
        context_type: &'static str,
    },
    Unparsable {
        context_kind: &'static str,
        text: &'source str,
    },
    NoTree,
    UnknownPrimitiveType(&'source str),
}

// Helper type alias for results
type ParseResult<'source, T> = Result<T, ParseError<'source>>;

// *************************************************************************************************
//  Parsed Types
// *************************************************************************************************

// TODO: Get rid of strings here.
// TODO Rename this to just parse::Type?
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
        params: Vec<ParsedType>,
        op_name: Option<String>,
    },
}
impl ParsedOp {
    pub fn anon(name: &str, p: ParsedType, r: ParsedType) -> Self {
        Self::Anonymous(name.to_string(), p, r)
    }
    // Does not contain ParsedType::Unspecified.
    fn is_specified(&self) -> bool {
        match self {
            Self::Anonymous(_name, p_ty, r_ty) => p_ty.is_specified() && r_ty.is_specified(),
            Self::NamedEffect { .. } => true,
        }
    }
}

impl ParsedType {
    pub fn func(args: impl AsRef<[Self]>, ret: Self) -> Self {
        Self::Fn(args.as_ref().to_vec(), Box::new(ret))
    }
    pub fn co(ret: ParsedType, ops: impl AsRef<[ParsedOp]>) -> Self {
        Self::Co(Box::new(ret), ops.as_ref().to_vec())
    }
    pub fn named(name: &str) -> Self {
        Self::Named(name.to_string(), vec![])
    }
    pub fn parameterized(name: &str, params: &[ParsedType]) -> Self {
        Self::Named(name.to_string(), params.to_vec())
    }

    pub fn is_specified(&self) -> bool {
        match self {
            Self::Unspecified => false,
            Self::Bool | Self::Float | Self::Int | Self::Never | Self::Unit => true,
            Self::Fn(args, ret) => ret.is_specified() && args.iter().all(|a| a.is_specified()),
            Self::Named(_name, params) => params.iter().all(|p| p.is_specified()),
            Self::Co(ret, ops) => ret.is_specified() && ops.iter().all(|op| op.is_specified()),
        }
    }
}

// TODO: str instead of string?
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StructDecl {
    pub(crate) name: String,
    pub(crate) params: Vec<String>,
    pub(crate) fields: Vec<(String, ParsedType)>,
}
impl StructDecl {
    pub fn new(name: impl Into<String>, fields: &[(&str, ParsedType)]) -> Self {
        Self::parameterized(name, &[], fields)
    }
    pub fn parameterized(
        name: impl Into<String>,
        params: &[&str],
        fields: &[(&str, ParsedType)],
    ) -> Self {
        Self {
            name: name.into(),
            params: params.iter().map(|p| p.to_string()).collect(),
            fields: fields
                .iter()
                .map(|(n, t)| (n.to_string(), t.clone()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectDecl {
    pub name: String,
    pub params: Vec<String>,
    pub ops: Vec<ParsedOp>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub forall: Vec<String>,
    pub name: String,
    pub args: Vec<Binding>,
    pub return_type: ParsedType,
    // If no body is provided, its assumed to be external.
    pub body: Option<Expression>,
}

// A coroutine function.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct CoDecl {
    pub forall: Vec<String>,
    pub name: String,
    pub args: Vec<Binding>,
    pub return_type: ParsedType,
    pub ops: Vec<ParsedOp>,
    // If no body is provided, its assumed to be external.
    pub body: Option<Expression>,
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

// *************************************************************************************************
//  Parse functions
// *************************************************************************************************

/// Converts a tree-sitter Tree into an ast::Program
pub fn parse_ast_program<'s>(source: &'s str, errors: &mut Vec<ParseError<'s>>) -> Option<Program> {
    let source_tree = SourceTree::parse(source, errors)?;
    let root_node = source_tree.root(errors)?;

    let mut declarations = Vec::new();

    for child in root_node.children(errors) {
        let decl = match child.kind() {
            "function_declaration" => parse_fn_decl(child, errors),
            "struct_declaration" => parse_struct_decl(child, errors),
            "effect_declaration" => parse_effect_decl(child, errors),
            "coroutine_declaration" => parse_co_decl(child, errors),
            other_kind => {
                errors.push(ParseError::UnexpectedNodeType {
                    expected: "function/struct/effect declaration",
                    found: other_kind,
                    node_text: child.text(),
                });
                None
            }
        };
        if let Some(decl) = decl {
            declarations.push(decl);
        }
    }

    Some(Program(declarations))
}

// --- Node Conversion Functions ---
fn parse_effect_decl<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Declaration> {
    let name = node.required_child("name", errors).map(|n| n.text());
    let params = node
        .optional_child("parameters", errors)
        .map(|n| parse_generic_params(n, errors))
        .unwrap_or_default();

    let mut seen_anonymous_ops = HashSet::new();
    let mut ops = Vec::new();
    for child_node in node.children(errors) {
        if child_node.kind() == "operation_signature" {
            let op_name = child_node
                .required_child("op_name", errors)
                .map(|n| n.text());
            let perform_type = child_node
                .required_child("perform_type", errors)
                .and_then(|n| parse_type(n, errors));
            let resume_type = child_node
                .required_child("resume_type", errors)
                .and_then(|n| parse_type(n, errors));

            if let (Some(op_name), Some(perform_type), Some(resume_type)) =
                (op_name, perform_type, resume_type)
            {
                if !seen_anonymous_ops.insert(op_name) {
                    errors.push(ParseError::DuplicateItem {
                        duplicated_item: op_name,
                        item_type: "operation name",
                        context_name: name.unwrap_or_default(),
                        context_type: "effect",
                    });
                } else {
                    ops.push(ParsedOp::anon(op_name, perform_type, resume_type));
                }
            }
        }
    }
    name.map(|name| {
        Declaration::Effect(EffectDecl {
            name: name.to_string(),
            params,
            ops,
        })
    })
}

fn parse_fn_decl<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Declaration> {
    let name = node
        .required_child("name", errors)
        .map(|n| n.text().to_string());
    let params = node
        .optional_child("parameters", errors)
        .map(|n| parse_generic_params(n, errors))
        .unwrap_or_default();

    let mut args = Vec::new();
    for arg in node.children_by_field_name("args", errors) {
        if let Some(binding) = parse_binding(arg, errors) {
            args.push(binding);
        }
    }

    let return_type = node
        .optional_child("return_type", errors)
        .and_then(|n| parse_type(n, errors))
        .unwrap_or(ParsedType::Unit);

    let body = node
        .required_child("body", errors)
        .and_then(|body_node| match body_node.kind() {
            "block_expression" => parse_block_expr(body_node, errors),
            ";" => None, // External function
            found => {
                errors.push(ParseError::UnexpectedNodeType {
                    expected: "block_expression or ;",
                    found,
                    node_text: body_node.text(),
                });
                None
            }
        });

    name.map(|name| {
        Declaration::Fn(FnDecl {
            forall: params,
            name,
            args,
            return_type,
            body,
        })
    })
}
fn parse_co_decl<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Declaration> {
    let name = node.required_child("name", errors)?.text().to_string();
    let params = node
        .optional_child("parameters", errors)
        .map(|n| parse_generic_params(n, errors))
        .unwrap_or_default();

    let mut args = Vec::new();
    for arg in node.children_by_field_name("args", errors) {
        if let Some(binding) = parse_binding(arg, errors) {
            args.push(binding);
        }
    }

    let return_type = node
        .optional_child("return_type", errors)
        .and_then(|n| parse_type(n, errors))
        .unwrap_or(ParsedType::Unit);
    let ops = node
        .optional_child("effects", errors)
        .and_then(|n| parse_effects(n, errors))
        .unwrap_or_default();

    let body = node
        .required_child("body", errors)
        .and_then(|body_node| match body_node.kind() {
            "block_expression" => parse_block_expr(body_node, errors),
            ";" => None, // External function
            found => {
                errors.push(ParseError::UnexpectedNodeType {
                    expected: "block_expression or ;",
                    found,
                    node_text: body_node.text(),
                });
                None
            }
        });

    Some(Declaration::Co(CoDecl {
        forall: params,
        name,
        args,
        return_type,
        ops,
        body,
    }))
}

fn parse_effects<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Vec<ParsedOp>> {
    let mut parsed_ops = Vec::new();
    for e in node.children_by_field_name("effects", errors) {
        match e.kind() {
            "anonymous_op_type" => {
                let op_name = e
                    .required_child("name", errors)
                    .map(|n| n.text().to_string());
                let perform_type = e
                    .required_child("perform_type", errors)
                    .and_then(|n| parse_type(n, errors));
                let resume_type = e
                    .required_child("resume_type", errors)
                    .and_then(|n| parse_type(n, errors));
                if let (Some(o), Some(p), Some(r)) = (op_name, perform_type, resume_type) {
                    parsed_ops.push(ParsedOp::anon(&o, p, r));
                }
            }
            "declared_effect" => {
                let name = e
                    .optional_child("effect_name", errors)
                    .map(|n| n.text().to_string());
                let effect = e
                    .required_child("effect_type", errors)
                    .map(|n| n.text().to_string());
                let params = e
                    .optional_child("type_params", errors)
                    .map(|node| parse_type_list(node, errors))
                    .unwrap_or_default();
                let op_name = e
                    .optional_child("op_name", errors)
                    .map(|n| n.text().to_string());
                if let Some(effect) = effect {
                    parsed_ops.push(ParsedOp::NamedEffect {
                        name,
                        effect,
                        params,
                        op_name,
                    });
                }
            }
            found => {
                errors.push(ParseError::UnexpectedNodeType {
                    expected: "effect",
                    found,
                    node_text: e.text(),
                });
            }
        }
    }
    Some(parsed_ops)
}

fn parse_type_list<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Vec<ParsedType> {
    let mut types = Vec::new();
    for n in node.children_by_field_name("types", errors) {
        if let Some(ty) = parse_type(n, errors) {
            types.push(ty);
        }
    }
    types
}

fn parse_struct_decl<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Declaration> {
    let struct_name = node.required_child("name", errors).map(|n| n.text())?;
    let params = node
        .optional_child("parameters", errors)
        .map(|node| parse_generic_params(node, errors))
        .unwrap_or_default();

    let mut seen_field_names = HashSet::new();
    let mut fields = Vec::new();
    for field_node in node.children_by_field_name("fields", errors) {
        let field_name = field_node.required_child("name", errors).map(|n| n.text());
        let field_ty = field_node
            .required_child("type", errors)
            .and_then(|n| parse_type(n, errors))
            .unwrap_or_default();
        if let Some(field_name) = field_name {
            if !seen_field_names.insert(field_name) {
                errors.push(ParseError::DuplicateItem {
                    duplicated_item: field_name,
                    item_type: "field",
                    context_name: struct_name,
                    context_type: "struct",
                });
            } else {
                fields.push((field_name.to_string(), field_ty));
            }
        }
    }
    Some(Declaration::Struct(StructDecl {
        name: struct_name.to_string(),
        params,
        fields,
    }))
}

fn parse_generic_params<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut params = Vec::new();
    for child in node.children(errors) {
        match child.kind() {
            "<" | "," | ">" => continue,
            "identifier" => {
                let parameter = child.text();
                if !seen.insert(parameter) {
                    errors.push(ParseError::DuplicateItem {
                        duplicated_item: parameter,
                        item_type: "parameter",
                        context_name: "",
                        context_type: "generic parameter list",
                    });
                }
                params.push(parameter.to_string());
            }
            other => errors.push(ParseError::UnexpectedNodeType {
                expected: "identifier",
                found: other,
                node_text: child.text(),
            }),
        }
    }
    params
}

fn parse_type<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<ParsedType> {
    match node.kind() {
        "primitive_type" => {
            match node.text() {
                // TODO: Update `Type`
                // "i8" => Some(Type::I8), // Assuming these variants exist in ast::Type
                // "u8" => Some(Type::U8),
                // "i16" => Some(Type::I16),
                // "u16" => Some(Type::U16),
                // "i32" => Some(Type::I32),
                // "u32" => Some(Type::U32),
                // "i64" => Some(Type::I64), // Was Type::int()
                // "u64" => Some(Type::U64),
                // "i128" => Some(Type::I128),
                // "u128" => Some(Type::U128),
                // "isize" => Some(Type::Isize),
                // "usize" => Some(Type::Usize),
                // "f64" => Some(Type::F64), // Was Type::Float
                "f64" => Some(ParsedType::Float),
                "i64" => Some(ParsedType::Int),
                "bool" => Some(ParsedType::Bool),
                "unit" => Some(ParsedType::Unit),
                "!" => Some(ParsedType::Never),
                "_" => Some(ParsedType::Unspecified),
                other => {
                    errors.push(ParseError::UnknownPrimitiveType(other));
                    None
                }
            }
        }
        "function_type" => {
            let mut arg_types = Vec::new();
            for a in node.children_by_field_name("arg_types", errors) {
                if let Some(t) = parse_type(a, errors) {
                    arg_types.push(t);
                }
            }
            let return_type = node
                .required_child("return_type", errors)
                .and_then(|n| parse_type(n, errors))?;
            Some(ParsedType::Fn(arg_types, Box::new(return_type)))
        }
        "coroutine_type" => {
            let return_type = node
                .required_child("return_type", errors)
                .and_then(|node| parse_type(node, errors))?;
            let ops = node
                .optional_child("effects", errors)
                .and_then(|n| parse_effects(n, errors))
                .unwrap_or_default();
            Some(ParsedType::Co(Box::new(return_type), ops))
        }
        "named_type" => {
            let name = node
                .required_child("name", errors)
                .map(|n| n.text().to_string());
            let params = node
                .optional_child("type_params", errors)
                .map(|node| parse_type_list(node, errors))
                .unwrap_or_default();
            name.map(|name| ParsedType::Named(name, params))
        }
        _ => {
            errors.push(ParseError::UnexpectedNodeType {
                expected: "type",
                found: node.kind(),
                node_text: node.text(),
            });
            None
        }
    }
}

// --- Statement Parsing ---
fn parse_statement<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Statement> {
    match node.kind() {
        "let_statement" => parse_let_statement(node, errors),
        "assignment_statement" => parse_assignment_statement(node, errors),
        "return_statement" => parse_return_statement(node, errors),
        "resume_statement" => parse_resume_statement(node, errors),
        "expression_statement" => parse_expression_statement(node, errors),
        found => {
            errors.push(ParseError::UnexpectedNodeType {
                expected: "statement",
                found,
                node_text: node.text(),
            });
            None
        }
    }
}

fn parse_let_statement<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Statement> {
    let binding = node
        .required_child("binding", errors)
        .and_then(|n| parse_binding(n, errors));
    let value = node
        .optional_child("value", errors)
        .and_then(|n| parse_expr(n, errors));

    Some(Statement::Let {
        binding: binding?,
        value: value?,
    })
}

// --- Expression Parsing ---
fn parse_expr<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    // Use node.kind() to dispatch to specific build functions
    match node.kind() {
        "if_expression" => parse_if_expr(node, errors),
        "lambda_expression" => parse_lambda_expr(node, errors),
        "call_expression" => parse_call_expr(node, errors),
        "block_expression" => parse_block_expr(node, errors),
        "literal" | "integer_literal" | "float_literal" | "boolean_literal" | "unit_literal" => {
            parse_literal_expr(node, errors)
        }
        "variable" => parse_variable_expr(node, errors),
        "parenthesized_expression" => {
            // TODO: This probably doesn't do well with tuples.
            parse_expr(node.required_child_by_id(1, errors)?, errors)
        }
        "perform_expression" => parse_perform_expression(node, errors),
        "propagate_expression" => node
            .required_child_by_id(0, errors)
            .and_then(|node| parse_expr(node, errors))
            .map(|e| Expression::Propagate(Box::new(e), Type::unknown())),
        "co_expression" => {
            // child 0 is "co", child 1 is the actual expression being wrapped.
            node.required_child_by_id(1, errors)
                .and_then(|node| parse_expr(node, errors))
                .map(|e| Expression::Co(Box::new(e), Type::unknown(), OpSet::empty_extensible()))
        }
        "struct_literal_expression" => parse_struct_literal(node, errors),
        "handle_expression" => parse_handle_expression(node, errors),
        "field_access_expression" => parse_field_access(node, errors),
        // Potentially handle _primary_expression or _l_value if needed by grammar structure
        _ => {
            errors.push(ParseError::UnexpectedNodeType {
                expected: "expression",
                found: node.kind(),
                node_text: node.text(),
            });
            None
        }
    }
}

fn parse_field_access<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let obj = node
        .required_child("object", errors)
        .and_then(|node| parse_expr(node, errors))?;
    let field = node
        .required_child("field", errors)
        .map(|node| node.text().to_string())?;
    Some(Expression::L(
        LValue::Field(Box::new(obj), field),
        Type::unknown(),
    ))
}

fn parse_handle_expression<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let coroutine = node
        .required_child("coroutine", errors)
        .and_then(|node| parse_expr(node, errors))?;
    let mut return_arm = None;
    let mut op_arms = Vec::new();
    for arm in node.children_by_field_name("arms", errors) {
        match arm.kind() {
            "return_arm" => {
                if return_arm.is_some() {
                    errors.push(ParseError::DuplicateItem {
                        duplicated_item: "return_arm",
                        item_type: "handle_arm",
                        context_name: "",
                        context_type: "handle expression",
                    });
                    continue;
                }
                let binding = arm
                    .required_child("binding", errors)
                    .and_then(|node| parse_binding(node, errors));
                let expr = arm
                    .required_child("expr", errors)
                    .and_then(|node| parse_expr(node, errors));
                if let (Some(binding), Some(expr)) = (binding, expr) {
                    return_arm = Some((binding, Box::new(expr)));
                }
            }
            "op_arm" => {
                // TODO: Use effect name once handled in AST.
                // let effect_name = node
                //     .optional_child("effect_name", errors)
                //     .map(|node| node.text());
                let op_name = arm
                    .required_child("op_name", errors)
                    .map(|node| node.text().to_string());
                let binding = arm
                    .required_child("binding", errors)
                    .and_then(|node| parse_binding(node, errors));
                let expr = arm
                    .required_child("expr", errors)
                    .and_then(|node| parse_expr(node, errors));
                if let (Some(op_name), Some(binding), Some(expr)) = (op_name, binding, expr) {
                    op_arms.push(HandleOpArm {
                        op_name: op_name.to_string(),
                        performed_variable: binding,
                        body: expr,
                    })
                }
            }
            found => {
                errors.push(ParseError::UnexpectedNodeType {
                    expected: "handle arm",
                    found,
                    node_text: arm.text(),
                });
            }
        }
    }
    Some(Expression::Handle {
        co: Box::new(coroutine),
        return_arm,
        op_arms,
        ty: Type::unknown(),
    })
}

fn parse_struct_literal<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let struct_name = node.required_child("name", errors).map(|n| n.text())?;
    let mut fields = HashMap::new();
    for field in node.children_by_field_name("fields", errors) {
        let field_name = field.required_child("name", errors).map(|f| f.text());
        let value = field
            .required_child("value", errors)
            .and_then(|n| parse_expr(n, errors));
        if let (Some(field_name), Some(value)) = (field_name, value) {
            match fields.entry(field_name.to_string()) {
                Entry::Occupied(_) => {
                    errors.push(ParseError::DuplicateItem {
                        duplicated_item: field_name,
                        item_type: "field",
                        context_name: struct_name,
                        context_type: "struct",
                    });
                }
                Entry::Vacant(e) => {
                    e.insert(value);
                }
            }
        }
    }
    Some(Expression::LiteralStruct {
        name: struct_name.to_string(),
        fields,
        ty: Type::unknown(),
    })
}

fn parse_perform_expression<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let qualifier = node.optional_child("effect_name", errors);
    let op_name = node.required_child("op_name", errors)?;
    // If the arg expression is an error, default to unit.
    let arg = node
        .required_child("argument", errors)
        .and_then(|node| parse_expr(node, errors))
        .unwrap_or(Expression::LiteralUnit);

    Some(Expression::Perform {
        name: qualifier.map(|n| n.text().to_string()),
        op: op_name.text().to_string(),
        arg: Box::new(arg),
        resume_type: Type::unknown(),
    })
}

fn parse_literal_expr<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let text = node.text();
    let child = node.required_child_by_id(0, errors)?;
    match child.kind() {
        "integer_literal" => text
            .parse::<i64>()
            .map(Expression::LiteralInt)
            .map_err(|e| {
                errors.push(ParseError::InvalidLiteral {
                    kind: "integer",
                    text,
                    error: e.to_string(),
                })
            })
            .ok(),
        "float_literal" => text
            .parse::<f64>()
            .map(Expression::LiteralFloat)
            .map_err(|e| {
                errors.push(ParseError::InvalidLiteral {
                    kind: "float",
                    text,
                    error: e.to_string(),
                })
            })
            .ok(),
        "boolean_literal" => match text {
            "true" => Some(Expression::LiteralBool(true)),
            "false" => Some(Expression::LiteralBool(false)),
            other => {
                errors.push(ParseError::InvalidLiteral {
                    kind: "boolean",
                    text: other,
                    error: "Not a boolean.".to_string(),
                });
                None
            }
        },
        "unit_literal" => Some(Expression::LiteralUnit),
        other_kind => {
            errors.push(ParseError::UnexpectedNodeType {
                expected: "literal", // TODO: What did we expect?
                found: other_kind,
                node_text: text,
            });
            None
        }
    }
}

fn parse_variable_expr<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let var = node.required_child_by_id(0, errors)?.text().to_string();
    Some(Expression::L(LValue::Variable(var), Type::unknown()))
}

fn parse_block_expr<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let mut statements = Vec::new();
    for child_node in node.children_by_field_name("statements", errors) {
        if let Some(statement) = parse_statement(child_node, errors) {
            statements.push(statement);
        }
    }
    if let Some(node) = node.optional_child("final_expression", errors) {
        if let Some(expr) = parse_expr(node, errors) {
            statements.push(Statement::Expression(expr));
        }
    }
    Some(Expression::Block {
        statements,
        ty: Type::unknown(),
    })
}

// Parses both soft and typed bindings from the Grammar.
fn parse_binding<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Binding> {
    let mutable = node
        .optional_child_by_id(0, errors)
        .map_or(false, |n| n.kind() == "mut");
    let name = node
        .required_child("name", errors)
        .map(|n| n.text().to_string());
    let ty = node
        .optional_child("type", errors)
        .and_then(|n| parse_type(n, errors))
        .unwrap_or(ParsedType::Unspecified);

    name.map(|name| Binding {
        name,
        parsed_type: ty,
        ty: Type::unknown(),
        mutable,
    })
}

fn parse_l_value<'s>(node: SourceNode<'s, '_>, errors: &mut Vec<ParseError<'s>>) -> Option<LValue> {
    match node.kind() {
        "identifier" => Some(LValue::Variable(node.text().to_string())),
        found => {
            errors.push(ParseError::UnexpectedNodeType {
                expected: "lvalue",
                found,
                node_text: node.text(),
            });
            None
        }
    }
}

fn parse_assignment_statement<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Statement> {
    let left = node
        .required_child("left", errors)
        .and_then(|n| parse_l_value(n, errors));
    let right = node
        .required_child("right", errors)
        .and_then(|n| parse_expr(n, errors));

    Some(Statement::Assign(left?, right?))
}

fn parse_return_statement<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Statement> {
    node.required_child("value", errors)
        .and_then(|n| parse_expr(n, errors))
        .map(Statement::Return)
}

fn parse_resume_statement<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Statement> {
    node.required_child("value", errors)
        .and_then(|n| parse_expr(n, errors))
        .map(Statement::Resume)
}

fn parse_expression_statement<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Statement> {
    node.required_child_by_id(0, errors)
        .and_then(|n| parse_expr(n, errors))
        .map(Statement::Expression)
}

fn parse_if_expr<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let condition = node
        .required_child("condition", errors)
        .and_then(|n| parse_expr(n, errors))
        .map(Box::new);
    let true_expr = node
        .required_child("consequence", errors)
        .and_then(|n| parse_expr(n, errors))
        .map(Box::new);
    let false_expr = node
        .optional_child("alternative", errors)
        .and_then(|n| parse_expr(n, errors))
        .unwrap_or(Expression::Block {
            statements: vec![],
            ty: Type::unknown(),
        }); // Default else
    if let (Some(condition), Some(true_expr)) = (condition, true_expr) {
        Some(Expression::If {
            condition,
            true_expr,
            false_expr: Box::new(false_expr),
            ty: Type::unknown(),
        })
    } else {
        None
    }
}

fn parse_lambda_expr<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let mut bindings = Vec::new();
    for child in node.children_by_field_name("lambda_args", errors) {
        dbg!(child.kind());
        if let Some(b) = parse_binding(child, errors) {
            bindings.push(b);
        }
    }
    let body = node
        .required_child("body", errors)
        .and_then(|n| parse_expr(n, errors))
        .map(Box::new);
    body.map(|body| Expression::Lambda {
        bindings,
        body,
        lambda_type: Type::unknown(),
    })
}

fn parse_call_expr<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Expression> {
    let fn_expr = node
        .required_child("function", errors)
        .and_then(|n| parse_expr(n, errors))
        .map(Box::new);
    let mut arg_exprs = Vec::new();
    for child in node.children_by_field_name("call_args", errors) {
        if let Some(expr) = parse_expr(child, errors) {
            arg_exprs.push(expr);
        }
    }
    fn_expr.map(|fn_expr| Expression::Call {
        fn_expr,
        arg_exprs,
        return_type: Type::unknown(),
    })
}

// Parses a program and panics if there were any errors.
// Intended for use in testing.
pub fn parse_program_or_die(source: &str) -> Program {
    let mut errors = Vec::new();
    let p = parse_ast_program(source, &mut errors);
    if !errors.is_empty() {
        panic!("Expected no errors, found: {errors:?})");
    }
    p.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::test_helpers::*;

    #[test]
    fn parses_simple_main() {
        let source_code = "fn main() -> i64 { 2 }";
        let tree = parse(source_code).unwrap();

        let root_node = tree.root_node();
        assert_eq!(root_node.kind(), "source_file");

        let mut errors = vec![];
        assert_eq!(
            parse_ast_program(source_code, &mut errors),
            Some(Program(vec![Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: ParsedType::Int,
                body: Some(block_expr(vec![2.into()]))
            })]))
        );
        assert_eq!(&errors, &[]);
    }
    #[test]
    fn use_lambda() {
        let source = r"
            fn plus(a: i64, b: i64) -> i64;

            fn main() -> i64 {
                let plus_two = |x| plus(2, x);
                let mut b = 2;
                b = plus_two(b);
                b
            }
        ";
        let program = Program(vec![
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "plus".to_string(),
                args: vec![
                    Binding::new_typed("a", ParsedType::Int),
                    Binding::new_typed("b", ParsedType::Int),
                ],
                return_type: ParsedType::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: ParsedType::Int,
                body: Some(block_expr(vec![
                    let_stmt(
                        "plus_two",
                        lambda_expr(
                            vec![Binding::new("x")],
                            call_expr("plus", vec![2.into(), "x".into()]),
                        ),
                    ),
                    let_mut_stmt("b", 2),
                    assign_stmt("b", call_expr("plus_two", vec!["b".into()])),
                    "b".into(),
                ])),
            }),
        ]);
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(&errors, &[]);
        assert_eq!(parsed, Some(program));
    }
    #[test]
    fn return_lambda() {
        let source = r"
            fn plus(a: i64, b: i64) -> i64;

            fn main() -> fn(i64)->i64 {
                let plus_two = |x| plus(2, x);
                plus_two
            }
        ";
        let program = Program(vec![
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "plus".to_string(),
                args: vec![
                    Binding::new_typed("a", ParsedType::Int),
                    Binding::new_typed("b", ParsedType::Int),
                ],
                return_type: ParsedType::Int,
                body: None,
            }),
            Declaration::Fn(FnDecl {
                forall: vec![],
                name: "main".to_string(),
                args: vec![],
                return_type: ParsedType::func([ParsedType::Int], ParsedType::Int),
                body: Some(block_expr(vec![
                    let_stmt(
                        "plus_two",
                        lambda_expr(
                            vec![Binding::new("x")],
                            call_expr("plus", vec![2.into(), "x".into()]),
                        ),
                    ),
                    "plus_two".into(),
                ])),
            }),
        ]);
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(&errors, &[]);
        assert_eq!(parsed, Some(program));
    }
    #[test]
    fn declare_struct() {
        let source = r"struct FooBar { foo: i64, bar: f64 }";
        let program = Program(vec![Declaration::Struct(StructDecl {
            params: vec![],
            name: "FooBar".to_string(),
            fields: vec![
                ("foo".to_string(), ParsedType::Int),
                ("bar".to_string(), ParsedType::Float),
            ],
        })]);
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(&errors, &[]);
        assert_eq!(parsed, Some(program));
    }
    #[test]
    fn declare_generic_fn() {
        let source = r"fn id<T>(x: T) -> T { x }";
        let program = Program(vec![Declaration::Fn(FnDecl {
            forall: ["T".to_string()].into_iter().collect(),
            name: "id".to_string(),
            args: vec![Binding {
                name: "x".to_string(),
                parsed_type: ParsedType::Named("T".to_string(), vec![]),
                ty: Type::unknown(),
                mutable: false,
            }],
            return_type: ParsedType::named("T"),
            body: Some(block_expr(vec!["x".into()])),
        })]);
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(&errors, &[]);
        assert_eq!(parsed, Some(program));
    }
    #[test]
    fn declare_generic_struct() {
        let source = r"struct FooBar<F, B> { foo: F, bar: B }";
        let program = Program(vec![Declaration::Struct(StructDecl::parameterized(
            "FooBar",
            &["F", "B"],
            &[
                ("foo", ParsedType::named("F")),
                ("bar", ParsedType::named("B")),
            ],
        ))]);
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(&errors, &[]);
        assert_eq!(parsed, Some(program));
    }
    #[test]
    fn declare_effect() {
        let source = r"effect IntState { get: unit -> i64, set: i64 -> unit }";
        let program = Program(vec![Declaration::Effect(EffectDecl {
            name: "IntState".to_string(),
            params: vec![],
            ops: vec![
                ParsedOp::anon("get", ParsedType::Unit, ParsedType::Int),
                ParsedOp::anon("set", ParsedType::Int, ParsedType::Unit),
            ],
        })]);
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(&errors, &[]);
        assert_eq!(parsed, Some(program));
    }

    #[test]
    fn declare_generic_effect() {
        let source = r"effect State<T> { get: unit -> T, set: T -> unit }";
        let program = Program(vec![Declaration::Effect(EffectDecl {
            name: "State".to_string(),
            params: vec!["T".to_string()],
            ops: vec![
                ParsedOp::anon("get", ParsedType::Unit, ParsedType::named("T")),
                ParsedOp::anon("set", ParsedType::named("T"), ParsedType::Unit),
            ],
        })]);
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(&errors, &[]);
        assert_eq!(parsed, Some(program));
    }
    #[test]
    fn struct_with_repeated_fields() {
        let source = r"
        struct Foo {
            foo: i64,
            foo: bool,
            bar: unit,
            bar: f64,
        }";
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(
            &errors,
            &[
                ParseError::DuplicateItem {
                    duplicated_item: "foo",
                    item_type: "field",
                    context_name: "Foo",
                    context_type: "struct"
                },
                ParseError::DuplicateItem {
                    duplicated_item: "bar",
                    item_type: "field",
                    context_name: "Foo",
                    context_type: "struct"
                }
            ]
        );
        let program = Program(vec![Declaration::Struct(StructDecl::new(
            "Foo",
            &[("foo", ParsedType::Int), ("bar", ParsedType::Unit)],
        ))]);
        assert_eq!(parsed, Some(program));
    }
    #[test]
    fn missing_struct_field_type() {
        let source_code = r#"
        struct Point {
            x: i64,
            y:,
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![ParseError::MissingField {
            node_kind: "struct_field_declaration",
            field_name: "type",
        }];
        assert_eq!(errors, expected_errors);

        let expected_ast = Some(Program(vec![Declaration::Struct(StructDecl {
            name: "Point".to_string(),
            params: vec![],
            fields: vec![
                ("x".to_string(), ParsedType::Int),
                ("y".to_string(), ParsedType::Unspecified),
            ],
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn unexpected_identifier_in_struct() {
        let source_code = r#"
        struct Point {
            x: i64,
            garbage borked 123 tokens
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![ParseError::Unparsable {
            context_kind: "struct_declaration",
            text: "garbage borked 123 tokens",
        }];
        assert_eq!(errors, expected_errors);

        let expected_ast = Some(Program(vec![Declaration::Struct(StructDecl::new(
            "Point",
            &[("x", ParsedType::Int)],
        ))]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn unexpected_identifier_in_effect() {
        let source_code = r#"
        effect State<T> {
            get: unit -> T,
            set (the inner garbage tokens): T -> unit,
            the outer garbage tokens
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![
            ParseError::Unparsable {
                context_kind: "effect_declaration",
                text: "the outer garbage tokens",
            },
            ParseError::Unparsable {
                context_kind: "operation_signature",
                text: "(the inner garbage tokens)",
            },
        ];
        assert_eq!(errors, expected_errors);

        let expected_ast = Some(Program(vec![Declaration::Effect(EffectDecl {
            name: "State".to_string(),
            params: vec!["T".to_string()],
            ops: vec![
                ParsedOp::anon("get", ParsedType::Unit, ParsedType::named("T")),
                ParsedOp::anon("set", ParsedType::named("T"), ParsedType::Unit),
            ],
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn missing_fn_name() {
        let source_code = r#"
        fn <T>(x: T) -> T { x }
        "#;
        let mut errors = Vec::new();
        parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![ParseError::Unparsable {
            context_kind: "source_file",
            text: "fn <T>(x: T) -> T { x }",
        }];
        assert_eq!(errors, expected_errors);
    }
    #[test]
    fn extra_fn_return_type() {
        let source_code = r#"
        fn id<T>(x: T) -> T -> T { x }
        "#;
        let mut errors = Vec::new();
        parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![ParseError::Unparsable {
            context_kind: "function_declaration",
            text: "-> T",
        }];
        assert_eq!(errors, expected_errors);
    }
    #[test]
    fn perform_op() {
        let source_code = r#"
        fn main() {
            perform foo(1)
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            return_type: ParsedType::Unit,
            body: Some(block_expr(vec![perform_anon("foo", 1).into()])),
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn missing_op_arg() {
        let source_code = r#"
        fn main() {
            perform foo()
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![ParseError::MissingField {
            node_kind: "perform_expression",
            field_name: "argument",
        }];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            return_type: ParsedType::Unit,
            body: Some(block_expr(vec![perform_anon("foo", ()).into()])),
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn propagate_expression() {
        let source_code = r#"
        fn main() {
            foo()?
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            return_type: ParsedType::Unit,
            body: Some(block_expr(vec![propagate(call_expr("foo", vec![])).into()])),
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn co_expression() {
        let source_code = r#"
        fn main() {
            co foo()
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            return_type: ParsedType::Unit,
            body: Some(block_expr(vec![co_expr(call_expr("foo", vec![])).into()])),
        })]));
        assert_eq!(parsed, expected_ast);
    }

    #[test]
    fn struct_literal_expression() {
        let source_code = r#"
        fn main() {
            Foo { foo: 1, bar: true }
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            return_type: ParsedType::Unit,
            body: Some(block_expr(vec![Expression::LiteralStruct {
                name: "Foo".to_string(),
                ty: Type::unknown(),
                fields: [
                    ("foo".to_string(), 1.into()),
                    ("bar".to_string(), true.into()),
                ]
                .into_iter()
                .collect(),
            }
            .into()])),
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn duplicate_field_in_struct_literal() {
        let source_code = r#"
        fn main() {
            Foo { foo: 1, bar: true, bar: true }
        }
        "#;
        let mut errors = vec![];
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![ParseError::DuplicateItem {
            duplicated_item: "bar",
            item_type: "field",
            context_name: "Foo",
            context_type: "struct",
        }];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            return_type: ParsedType::Unit,
            body: Some(block_expr(vec![Expression::LiteralStruct {
                name: "Foo".to_string(),
                ty: Type::unknown(),
                fields: [
                    ("foo".to_string(), 1.into()),
                    ("bar".to_string(), true.into()),
                ]
                .into_iter()
                .collect(),
            }
            .into()])),
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn handle_expr() {
        let source_code = r#"
        fn main() {
            foo() handle {
                return x => bar(x, x),
                foo(mut y: unit) => baz(y),
                bar(z) => {
                    resume z(1);
                },
            }
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            return_type: ParsedType::Unit,
            body: Some(block_expr(vec![Expression::Handle {
                co: Box::new(call_expr("foo", vec![])),
                return_arm: Some((
                    Binding::new("x"),
                    Box::new(call_expr("bar", vec!["x".into(), "x".into()])),
                )),
                op_arms: vec![
                    HandleOpArm {
                        op_name: "foo".to_string(),
                        performed_variable: Binding {
                            name: "y".into(),
                            parsed_type: ParsedType::Unit,
                            ty: Type::unknown(),
                            mutable: true,
                        },
                        body: call_expr("baz", vec!["y".into()]),
                    },
                    HandleOpArm {
                        op_name: "bar".to_string(),
                        performed_variable: Binding::new("z"),
                        body: block_expr(vec![resume_stmt(call_expr("z", vec![1.into()]))]),
                    },
                ],
                ty: Type::unknown(),
            }
            .into()])),
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn field_access() {
        let source_code = r#"
        fn main() {
            foo().bar
        }
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Fn(FnDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            return_type: ParsedType::Unit,
            body: Some(block_expr(vec![Expression::L(
                LValue::Field(Box::new(call_expr("foo", vec![])), "bar".to_string()),
                Type::unknown(),
            )
            .into()])),
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn co_function() {
        let source_code = r#"
        co main() -> unit ! foo(i64 -> f64);
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Co(CoDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            ops: vec![ParsedOp::anon("foo", ParsedType::Int, ParsedType::Float)],
            return_type: ParsedType::Unit,
            body: None,
        })]));
        assert_eq!(parsed, expected_ast);
    }
    #[test]
    fn co_function_implicit_return_type() {
        let source_code = r#"
        co main() ! foo(i64 -> f64);
        "#;
        let mut errors = Vec::new();
        let parsed = parse_ast_program(source_code, &mut errors);

        let expected_errors = vec![];
        assert_eq!(errors, expected_errors);
        let expected_ast = Some(Program(vec![Declaration::Co(CoDecl {
            name: "main".to_string(),
            forall: Default::default(),
            args: vec![],
            ops: vec![ParsedOp::anon("foo", ParsedType::Int, ParsedType::Float)],
            return_type: ParsedType::Unit,
            body: None,
        })]));
        assert_eq!(parsed, expected_ast);
    }
}
