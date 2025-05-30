use std::collections::{hash_map::Entry, BTreeSet, HashMap};

use crate::ast::{
    Declaration, EffectDecl, Expression, FnDecl, LValue, Program, SoftBinding, Statement,
    StructDecl, Type, TypedBinding,
};
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

// TODO: Some of these strings can be borrowed.
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
    UnknownPrimitiveType(String),
}

// Helper type alias for results
type ParseResult<'source, T> = Result<T, ParseError<'source>>;

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

    let mut ops = HashMap::new();
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
                match ops.entry(op_name.to_string()) {
                    Entry::Occupied(_) => {
                        errors.push(ParseError::DuplicateItem {
                            duplicated_item: op_name,
                            item_type: "operation name",
                            context_name: name.unwrap_or_default(),
                            context_type: "effect",
                        });
                    }
                    Entry::Vacant(e) => {
                        e.insert((perform_type, resume_type));
                    }
                }
            }
        } else {
            dbg!(child_node.kind(), child_node.node);
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

    let args = node
        .required_child("args", errors)
        .map(|n| parse_fn_arg_list(n, errors))
        .unwrap_or_default();

    let return_type = node
        .optional_child("return_type", errors)
        .and_then(|n| parse_type(n, errors))
        .unwrap_or(Type::unit());

    // Check if body is ';' (external) or a block expression
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

fn parse_struct_decl<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<Declaration> {
    let struct_name = node.required_child("name", errors).map(|n| n.text());
    let params = node
        .optional_child("generic_parameters", errors)
        .map(|node| parse_generic_params(node, errors))
        .unwrap_or_default();

    let mut fields = HashMap::new();
    if let Some(fields_node) = node.required_child("fields", errors) {
        for field_node in fields_node.children(errors) {
            if field_node.kind() == "struct_field_declaration" {
                let (field_name, field_type) = match parse_struct_field(field_node, errors) {
                    Some((f, t)) => (f, t),
                    None => continue,
                };

                match fields.entry(field_name.to_string()) {
                    Entry::Occupied(_) => {
                        errors.push(ParseError::DuplicateItem {
                            duplicated_item: field_name,
                            item_type: "field",
                            context_name: struct_name.unwrap_or_default(),
                            context_type: "struct",
                        });
                    }
                    Entry::Vacant(e) => {
                        e.insert(field_type);
                    }
                }
            }
        }
    }
    struct_name.map(|s| {
        Declaration::Struct(StructDecl {
            params,
            name: s.to_string(),
            fields,
        })
    })
}

fn parse_struct_field<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<(&'s str, Type)> {
    let name = node.required_child("name", errors).map(|n| n.text());
    let ty = node
        .required_child("type", errors)
        .and_then(|n| parse_type(n, errors));
    Some((name?, ty?))
}

fn parse_generic_params<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> BTreeSet<String> {
    let mut params = BTreeSet::new();
    for child in node.children(errors) {
        match child.kind() {
            "<" | "," | ">" => continue,
            "identifier" => {
                let parameter = child.text();
                if !params.insert(parameter.to_string()) {
                    errors.push(ParseError::DuplicateItem {
                        duplicated_item: parameter,
                        item_type: "parameter",
                        context_name: "",
                        context_type: "generic parameter list",
                    });
                }
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

fn parse_fn_arg_list<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Vec<TypedBinding> {
    let mut args = Vec::new();
    for child in node.children(errors) {
        if matches!(child.kind(), "(" | "," | ")") {
            continue;
        }
        if child.kind() != "fn_arg" {
            errors.push(ParseError::UnexpectedNodeType {
                expected: "fn_arg",
                found: child.kind(),
                node_text: child.text(),
            });
            continue;
        }
        let mutable = child
            .optional_child_by_id(0, errors)
            .map_or(false, |n| n.kind() == "mut");
        let name = child
            .required_child("name", errors)
            .map(|n| n.text().to_string());
        let ty = child
            .required_child("type", errors)
            .and_then(|n| parse_type(n, errors));

        if let (Some(name), Some(ty)) = (name, ty) {
            args.push(TypedBinding { name, ty, mutable });
        }
    }
    args
}

// --- Type Parsing ---
fn parse_type<'s>(node: SourceNode<'s, '_>, errors: &mut Vec<ParseError<'s>>) -> Option<Type> {
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
                "f64" => Some(Type::float()),
                "i64" => Some(Type::int()),
                "bool" => Some(Type::bool_()),
                "unit" => Some(Type::unit()),
                other => {
                    errors.push(ParseError::UnknownPrimitiveType(other.to_string()));
                    None
                }
            }
        }
        "function_type" => {
            let mut arg_types = Vec::new();
            let mut return_type = Type::unit();
            let mut found_arrow = false;
            for child in node.children(errors) {
                match child.kind() {
                    "fn" | "(" | "," | ")" => {}
                    "->" => {
                        found_arrow = true;
                    }
                    // TODO: Is thie comprehensive or correct?
                    "primitive_type" | "function_type" | "named_type" | "_type" => {
                        if found_arrow {
                            if let Some(ty) = parse_type(child, errors) {
                                return_type = ty;
                            }
                        } else if let Some(ty) = parse_type(child, errors) {
                            arg_types.push(ty);
                            // TODO: Some kind of unknown type placeholder?
                        }
                    }
                    found => {
                        errors.push(ParseError::UnexpectedNodeType {
                            expected: "type",
                            found,
                            node_text: child.text(),
                        });
                    }
                }
            }
            Some(Type::func(arg_types, return_type))
        }
        // TODO: Named type support
        "named_type" | "identifier" => Some(Type::param(node.text())),
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
        .and_then(|n| parse_soft_binding(n, errors));
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
    if let Some(node) = node.required_child("final_expression", errors) {
        if let Some(expr) = parse_expr(node, errors) {
            statements.push(Statement::Expression(expr));
        }
    }
    Some(Expression::Block {
        statements,
        ty: Type::unknown(),
    })
}

fn parse_soft_binding<'s>(
    node: SourceNode<'s, '_>,
    errors: &mut Vec<ParseError<'s>>,
) -> Option<SoftBinding> {
    let mutable = node
        .optional_child_by_id(0, errors)
        .map_or(false, |n| n.kind() == "mut");
    let name = node
        .required_child("name", errors)
        .map(|n| n.text().to_string());
    let ty = node
        .optional_child("type", errors)
        .and_then(|n| parse_type(n, errors));

    name.map(|name| SoftBinding { name, ty, mutable })
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
    for child in node.required_child("lambda_args", errors)?.children(errors) {
        if child.kind() == "soft_binding" {
            if let Some(b) = parse_soft_binding(child, errors) {
                bindings.push(b);
            }
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
    for child in node.required_child("arguments", errors)?.children(errors) {
        if !matches!(child.kind(), "(" | ")" | ",") {
            if let Some(expr) = parse_expr(child, errors) {
                arg_exprs.push(expr);
            }
        }
    }
    fn_expr.map(|fn_expr| Expression::Call {
        fn_expr,
        arg_exprs,
        return_type: Type::unknown(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use super::*;
    use crate::ast::{test_helpers::*, StructDecl};

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
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
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
        let mut errors = vec![];
        let parsed = parse_ast_program(source, &mut errors);
        assert_eq!(&errors, &[]);
        assert_eq!(parsed, Some(program));
    }
    #[test]
    fn declare_struct() {
        let source = r"struct FooBar { foo: i64, bar: f64 }";
        let program = Program(vec![Declaration::Struct(StructDecl {
            params: BTreeSet::new(),
            name: "FooBar".to_string(),
            fields: HashMap::from_iter(
                [
                    ("foo".to_string(), Type::int()),
                    ("bar".to_string(), Type::float()),
                ]
                .into_iter(),
            ),
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
            args: vec![TypedBinding {
                name: "x".to_string(),
                ty: Type::param("T"),
                mutable: false,
            }],
            return_type: Type::param("T"),
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
        let program = Program(vec![Declaration::Struct(StructDecl {
            params: BTreeSet::new(),
            name: "FooBar".to_string(),
            fields: HashMap::from_iter(
                [
                    ("foo".to_string(), Type::param("F")),
                    ("bar".to_string(), Type::param("B")),
                ]
                .into_iter(),
            ),
        })]);
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
            params: BTreeSet::default(),
            ops: [
                ("get".to_string(), (Type::unit(), Type::int())),
                ("set".to_string(), (Type::int(), Type::unit())),
            ]
            .into_iter()
            .collect(),
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
            params: ["T".to_string()].into_iter().collect(),
            ops: [
                ("get".to_string(), (Type::unit(), Type::param("T"))),
                ("set".to_string(), (Type::param("T"), Type::unit())),
            ]
            .into_iter()
            .collect(),
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
        let program = Program(vec![Declaration::Struct(StructDecl {
            name: "Foo".to_string(),
            params: Default::default(),
            fields: [
                ("foo".to_string(), Type::int()),
                ("bar".to_string(), Type::unit()),
            ]
            .into_iter()
            .collect(),
        })]);
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
            params: BTreeSet::new(),
            fields: [("x".to_string(), Type::int())].into_iter().collect(),
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
            context_kind: "struct_field_list",
            text: "garbage borked 123 tokens",
        }];
        assert_eq!(errors, expected_errors);

        let expected_ast = Some(Program(vec![Declaration::Struct(StructDecl {
            name: "Point".to_string(),
            params: BTreeSet::new(),
            fields: [("x".to_string(), Type::int())].into_iter().collect(),
        })]));
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
            params: ["T".to_string()].into_iter().collect(),
            ops: [
                ("get".to_string(), (Type::unit(), Type::param("T"))),
                ("set".to_string(), (Type::param("T"), Type::unit())),
            ]
            .into_iter()
            .collect(),
        })]));
        assert_eq!(parsed, expected_ast);
    }
}
