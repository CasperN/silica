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
    fn parse(source: &'source str) -> BuildResult<Self> {
        let tree = parse(source).ok_or(ParseError::NoTree)?;
        Ok(Self { source, tree })
    }
    fn root<'tree>(&'tree self) -> BuildResult<SourceNode<'tree, 'source>> {
        let root_node = SourceNode {
            node: self.tree.root_node(),
            source: self.source,
        };
        if root_node.kind() != "source_file" {
            return Err(ParseError::UnexpectedNodeType {
                expected: "source_file",
                found: root_node.kind().to_string(),
                node_text: root_node.source.to_string(),
            });
        }
        Ok(root_node)
    }
}

#[derive(Clone, Copy)]
struct SourceNode<'tree, 'source> {
    node: Node<'tree>,
    source: &'source str,
}
impl<'t, 's> SourceNode<'t, 's> {
    fn kind(&self) -> &'static str {
        self.node.kind()
    }

    fn is_type(&self) -> bool {
        // TODO: Is this correct/comprehensive? WARNING LLM GENERATED.
        matches!(
            self.kind(),
            "primitive_type" | "function_type" | "named_type" | "_type"
        )
    }

    fn text(&self) -> &'s str {
        // self.source is utf8 and the node's span shouldn't slice it in an invalid way, so
        // utf8 decoding should not fail unless there is an error in tree-sitter or the grammar.
        self.node
            .utf8_text(self.source.as_bytes())
            .expect("Node text decode failed")
    }

    fn child_by_field_name_opt(self, field: &str) -> Option<Self> {
        self.node.child_by_field_name(field).map(|node| Self {
            node,
            source: self.source,
        })
    }

    fn child_by_field_name(self, field: &str) -> BuildResult<Self> {
        self.child_by_field_name_opt(field)
            .ok_or_else(|| ParseError::MissingField {
                node_kind: self.node.kind(),
                field_name: field.to_string(),
            })
    }
    fn child_by_id_opt(self, field_id: usize) -> Option<Self> {
        self.node.child(field_id).map(|node| Self {
            node,
            source: self.source,
        })
    }

    fn child_by_id(self, field_id: usize) -> BuildResult<Self> {
        self.child_by_id_opt(field_id)
            .ok_or_else(|| ParseError::MissingIndex {
                node_kind: self.node.kind(),
                child_index: field_id,
            })
    }

    fn children(self) -> Vec<Self> {
        let mut cursor = self.node.walk();
        let mut children = Vec::new();
        for node in self.node.children(&mut cursor) {
            children.push(Self {
                node,
                source: self.source,
            });
        }
        children
    }
    fn children_by_field_name(self, field_name: &str) -> Vec<Self> {
        let mut cursor = self.node.walk();
        let mut children = Vec::new();
        for node in self.node.children_by_field_name(field_name, &mut cursor) {
            children.push(Self {
                node,
                source: self.source,
            });
        }
        children
    }
}

// TODO: Some of these strings can be borrowed.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedNodeType {
        expected: &'static str,
        found: String,
        node_text: String,
    },
    MissingField {
        node_kind: &'static str,
        field_name: String,
    },
    MissingIndex {
        node_kind: &'static str,
        child_index: usize,
    },
    InvalidLiteral {
        kind: &'static str,
        text: String,
        error: String,
    },
    DuplicateItem {
        duplicated_item: String,
        item_type: &'static str,
        context_name: String,
        context_type: &'static str,
    },
    NoTree,
    UnknownPrimitiveType(String),
}

// Helper type alias for results
type BuildResult<T> = Result<T, ParseError>;

fn get_required_child<'a>(
    node: &Node<'a>,
    source: &str,
    child: &'static str,
) -> BuildResult<Node<'a>> {
    let _ = source;
    node.child_by_field_name(child)
        .ok_or_else(|| ParseError::MissingField {
            node_kind: node.kind(),
            field_name: child.to_string(),
        })
}
fn get_required_child_by_id<'a>(
    node: &Node<'a>,
    source: &str,
    field_id: usize,
) -> BuildResult<Node<'a>> {
    let _ = source;
    node.child(field_id)
        .ok_or_else(|| ParseError::MissingIndex {
            node_kind: node.kind(),
            child_index: field_id,
        })
}

// --- Main Conversion Function ---

/// Converts a tree-sitter Tree into an ast::Program
pub fn build_ast_program(source: &str) -> BuildResult<Program> {
    let source_tree = SourceTree::parse(source)?;
    let root_node = source_tree.root()?;

    let mut declarations = Vec::new();

    for child in root_node.children() {
        match child.kind() {
            "function_declaration" => {
                declarations.push(build_fn_decl(child)?);
            }
            "struct_declaration" => {
                declarations.push(build_struct_decl(child)?);
            }
            "effect_declaration" => {
                declarations.push(build_effect_decl(child)?);
            }
            other_kind => {
                return Err(ParseError::UnexpectedNodeType {
                    expected: "function/struct/effect declaration",
                    found: other_kind.to_string(),
                    node_text: child.text().to_string(),
                })
            }
        }
    }

    Ok(Program(declarations))
}

// --- Node Conversion Functions ---
fn build_effect_decl(node: SourceNode) -> BuildResult<Declaration> {
    let name_node = node.child_by_field_name("name")?; // [cite: 792]
    let name = name_node.text().to_string(); // [cite: 879]

    let generic_params_node = node.child_by_field_name("generic_parameters").ok();
    let params = if let Some(gp_node) = generic_params_node {
        build_generic_params(gp_node)?
    } else {
        BTreeSet::new()
    };

    let mut ops = HashMap::new();

    // Iterate through children of the effect_declaration node to find operation signatures
    for child_node in node.children() {
        if child_node.kind() == "operation_signature" {
            let op_name = child_node.child_by_field_name("op_name")?.text();
            let perform_type = child_node
                .child_by_field_name("perform_type")
                .and_then(build_type)?;
            let resume_type = child_node
                .child_by_field_name("resume_type")
                .and_then(build_type)?;

            match ops.entry(op_name.to_string()) {
                Entry::Occupied(_) => {
                    return Err(ParseError::DuplicateItem {
                        duplicated_item: op_name.to_string(),
                        item_type: "operation name",
                        context_name: name.to_string(),
                        context_type: "effect",
                    });
                }
                Entry::Vacant(e) => {
                    e.insert((perform_type, resume_type));
                }
            }
        }
    }

    Ok(Declaration::Effect(EffectDecl { name, params, ops }))
}

fn build_fn_decl(node: SourceNode) -> BuildResult<Declaration> {
    let name = node.child_by_field_name("name")?.text().to_string();
    let params = if let Some(node) = node.child_by_field_name_opt("parameters") {
        build_generic_params(node)?
    } else {
        BTreeSet::new()
    };

    let args = node
        .child_by_field_name("args")
        .and_then(build_fn_arg_list)?;

    let return_type = if let Some(n) = node.child_by_field_name_opt("return_type") {
        build_type(n)?
    } else {
        Type::unit()
    };

    // Check if body is ';' (external) or a block expression
    let body_node = node.child_by_field_name("body")?;
    let body = match body_node.kind() {
        "block_expression" => Some(build_block_expr(body_node)?),
        ";" => None, // External function
        _ => {
            return Err(ParseError::UnexpectedNodeType {
                expected: "block_expression or ;",
                found: body_node.kind().to_string(),
                node_text: body_node.text().to_string(),
            })
        }
    };

    Ok(Declaration::Fn(FnDecl {
        forall: params,
        name,
        args,
        return_type,
        body,
    }))
}
fn build_struct_decl(node: SourceNode) -> BuildResult<Declaration> {
    let name = node.child_by_field_name("name")?.text().to_string();
    let params = node
        .child_by_field_name("generic_parameters")
        .map(|node| build_generic_params(node))
        .unwrap_or(Ok(BTreeSet::new()))?;
    let fields = node
        .child_by_field_name("fields")
        .and_then(build_fields_map)?;

    Ok(Declaration::Struct(StructDecl {
        params,
        name,
        fields,
    }))
}

fn build_generic_params(node: SourceNode) -> BuildResult<BTreeSet<String>> {
    let mut params = BTreeSet::new();
    for child in node.children() {
        if child.kind() == "identifier" {
            let child_text = child.text();
            if !params.insert(child_text.to_string()) {
                return Err(ParseError::DuplicateItem {
                    duplicated_item: child_text.to_string(),
                    item_type: "parameter",
                    context_name: "".to_string(),
                    context_type: "generic parameter list",
                });
            }
        }
        // Skip '<' ')' ',' tokens if they appear as unnamed children
    }
    Ok(params)
}

fn build_fields_map(node: SourceNode) -> BuildResult<HashMap<String, Type>> {
    // { field_name: field_type, }
    let mut fields = HashMap::new();
    for child in node.children() {
        if child.kind() == "struct_field_declaration" {
            let (field_name, field_type) = build_struct_field(child)?;
            match fields.entry(field_name.to_string()) {
                Entry::Occupied(_) => {
                    return Err(ParseError::DuplicateItem {
                        duplicated_item: field_name,
                        item_type: "field",
                        context_name: "".to_string(),
                        context_type: "struct",
                    });
                }
                Entry::Vacant(e) => {
                    e.insert(field_type);
                }
            }
        }
    }
    Ok(fields)
}

fn build_struct_field(node: SourceNode) -> BuildResult<(String, Type)> {
    let name = node.child_by_field_name("name")?.text().to_string();
    let ty = node.child_by_field_name("type").and_then(build_type)?;
    Ok((name, ty))
}

fn build_fn_arg_list(node: SourceNode) -> BuildResult<Vec<TypedBinding>> {
    // parameter_list: $ => seq('(', optional(sepBy(',', $.parameter)), ')')
    let mut params = Vec::new();
    for child in node.children() {
        if child.kind() == "fn_arg" {
            params.push(build_fn_arg(child)?);
        }
        // Skip '(' ')' ',' tokens if they appear as unnamed children
    }
    Ok(params)
}

fn build_fn_arg(node: SourceNode) -> BuildResult<TypedBinding> {
    // parameter: $ => seq(optional('mut'), name: $.identifier, ':', type: $._type)
    let mutable = node.child_by_id(0).map_or(false, |n| n.kind() == "mut"); // Check first child
    let name = node.child_by_field_name("name")?.text().to_string();
    let ty = node.child_by_field_name("type").and_then(build_type)?;

    Ok(TypedBinding { name, ty, mutable })
}

// --- Type Parsing ---
fn build_type(node: SourceNode) -> BuildResult<Type> {
    match node.kind() {
        "primitive_type" => {
            match node.text() {
                // TODO: Update `Type`
                // "i8" => Ok(Type::I8), // Assuming these variants exist in ast::Type
                // "u8" => Ok(Type::U8),
                // "i16" => Ok(Type::I16),
                // "u16" => Ok(Type::U16),
                // "i32" => Ok(Type::I32),
                // "u32" => Ok(Type::U32),
                // "i64" => Ok(Type::I64), // Was Type::int()
                // "u64" => Ok(Type::U64),
                // "i128" => Ok(Type::I128),
                // "u128" => Ok(Type::U128),
                // "isize" => Ok(Type::Isize),
                // "usize" => Ok(Type::Usize),
                // "f64" => Ok(Type::F64), // Was Type::Float
                "f64" => Ok(Type::float()),
                "i64" => Ok(Type::int()),
                "bool" => Ok(Type::bool_()),
                "unit" => Ok(Type::unit()),
                other => Err(ParseError::UnknownPrimitiveType(other.to_string())),
            }
        }
        "function_type" => {
            // fn_type: $ => seq('fn', '(', optional(sepBy(',', $._type)), ')', '->', $._type)
            let mut arg_types = Vec::new();
            let mut return_type = None;
            let mut found_arrow = false;
            // TODO: What is going on here? WARNING LLM GENERATED.
            for child in node.children() {
                if child.kind() == "->" {
                    found_arrow = true;
                    continue; // Skip arrow token
                }
                if child.is_type() {
                    // Helper function needed
                    if found_arrow {
                        return_type = Some(build_type(child)?);
                    } else {
                        // Assume types before '->' are args
                        arg_types.push(build_type(child)?);
                    }
                }
                // Skip 'fn', '(', ')', ',' tokens
            }
            Ok(Type::func(arg_types, return_type.unwrap_or(Type::unit())))
        }
        // TODO: Named type support
        "named_type" | "identifier" => Ok(Type::param(node.text())),
        _ => Err(ParseError::UnexpectedNodeType {
            expected: "type",
            found: node.kind().to_string(),
            node_text: node.text().to_string(),
        }),
    }
}

// --- Statement Parsing ---
fn build_statement(node: SourceNode) -> BuildResult<Statement> {
    match node.kind() {
        "let_statement" => build_let_statement(node),
        "assignment_statement" => build_assignment_statement(node),
        "return_statement" => build_return_statement(node),
        "expression_statement" => build_expression_statement(node),
        _ => Err(ParseError::UnexpectedNodeType {
            expected: "statement",
            found: node.kind().to_string(),
            node_text: node.text().to_string(),
        }),
    }
}

fn build_let_statement(node: SourceNode) -> BuildResult<Statement> {
    let binding = node
        .child_by_field_name("binding")
        .and_then(build_soft_binding)?;
    let value = node
        .child_by_field_name("value")
        .and_then(build_expression)?;

    Ok(Statement::Let { binding, value })
}

// --- Expression Parsing ---
fn build_expression(node: SourceNode) -> BuildResult<Expression> {
    // Use node.kind() to dispatch to specific build functions
    match node.kind() {
        "if_expression" => build_if_expr(node),
        "lambda_expression" => build_lambda_expr(node),
        "call_expression" => build_call_expr(node),
        "block_expression" => build_block_expr(node),
        "literal" | "integer_literal" | "float_literal" | "boolean_literal" | "unit_literal" => {
            build_literal_expr(node)
        }
        "variable" => build_variable_expr(node),
        "parenthesized_expression" => {
            // TODO: This probably doesn't do well with tuples.
            build_expression(node.child_by_id(1)?)
        }
        // Potentially handle _primary_expression or _l_value if needed by grammar structure
        _ => Err(ParseError::UnexpectedNodeType {
            expected: "expression",
            found: node.kind().to_string(),
            node_text: node.text().to_string(),
        }),
    }
}

fn build_literal_expr(node: SourceNode) -> BuildResult<Expression> {
    let text = node.text();
    let child = node.child_by_id(0)?;
    match child.kind() {
        "integer_literal" => text
            .parse::<i64>()
            .map(Expression::LiteralInt)
            .map_err(|e| ParseError::InvalidLiteral {
                kind: "integer",
                text: text.to_string(),
                error: e.to_string(),
            }),
        "float_literal" => text
            .parse::<f64>()
            .map(Expression::LiteralFloat)
            .map_err(|e| ParseError::InvalidLiteral {
                kind: "float",
                text: text.to_string(),
                error: e.to_string(),
            }),
        "boolean_literal" => match text {
            "true" => Ok(Expression::LiteralBool(true)),
            "false" => Ok(Expression::LiteralBool(false)),
            _ => unreachable!(), // Grammar should prevent this
        },
        "unit_literal" => Ok(Expression::LiteralUnit),
        other_kind => Err(ParseError::UnexpectedNodeType {
            expected: "literal", // TODO: What did we expect?
            found: other_kind.to_string(),
            node_text: text.to_string(),
        }),
    }
}

fn build_variable_expr(node: SourceNode) -> BuildResult<Expression> {
    // Assuming 'variable' node wraps an 'identifier' node
    let var = node.child_by_id(0)?.text().to_string();
    Ok(Expression::L(LValue::Variable(var), Type::unknown()))
}

fn build_block_expr(node: SourceNode) -> BuildResult<Expression> {
    let mut statements = Vec::new();
    for s in node.children_by_field_name("statements") {
        statements.push(build_statement(s)?);
    }
    if let Ok(node) = node.child_by_field_name("final_expression") {
        statements.push(Statement::Expression(build_expression(node)?));
    }
    Ok(Expression::Block {
        statements,
        ty: Type::unknown(),
    })
}

// --- TODO: Implement remaining build functions ---

fn build_soft_binding(node: SourceNode) -> BuildResult<SoftBinding> {
    let mutable = node.child_by_id(0).map_or(false, |n| n.kind() == "mut");
    let name = node.child_by_field_name("name")?.text().to_string();
    let ty = node
        .child_by_field_name_opt("type")
        .map(build_type)
        .transpose()?;

    Ok(SoftBinding { name, ty, mutable })
}

fn build_l_value(node: SourceNode) -> BuildResult<LValue> {
    match node.kind() {
        "identifier" => Ok(LValue::Variable(node.text().to_string())),
        _ => Err(ParseError::UnexpectedNodeType {
            expected: "lvalue",
            found: node.kind().into(),
            node_text: node.text().to_string(),
        }),
    }
}

fn build_assignment_statement(node: SourceNode) -> BuildResult<Statement> {
    let left = node.child_by_field_name("left").and_then(build_l_value)?;
    let right = node
        .child_by_field_name("right")
        .and_then(build_expression)?;
    Ok(Statement::Assign(left, right))
}

fn build_return_statement(node: SourceNode) -> BuildResult<Statement> {
    let expr = node
        .child_by_field_name("value")
        .and_then(build_expression)?;
    Ok(Statement::Return(expr))
}

fn build_expression_statement(node: SourceNode) -> BuildResult<Statement> {
    // expression_statement: $ => seq($._expression, ';')
    let expr = node.child_by_id(0).and_then(build_expression)?;
    Ok(Statement::Expression(expr))
}

fn build_if_expr(node: SourceNode) -> BuildResult<Expression> {
    let condition = node
        .child_by_field_name("condition")
        .and_then(build_expression)
        .map(Box::new)?;
    let true_expr = node
        .child_by_field_name("consequence")
        .and_then(build_expression)
        .map(Box::new)?;

    let false_expr = match node.child_by_field_name_opt("alternative") {
        Some(node) => build_block_expr(node)?,
        None => Expression::Block {
            statements: vec![],
            ty: Type::unknown(),
        }, // Default else
    };

    Ok(Expression::If {
        condition,
        true_expr,
        false_expr: Box::new(false_expr),
        ty: Type::unknown(),
    })
}

fn build_lambda_expr(node: SourceNode) -> BuildResult<Expression> {
    let bindings = node
        .child_by_field_name("lambda_args")
        .and_then(build_lambda_parameter_list)?;
    let body = node
        .child_by_field_name("body")
        .and_then(build_expression)
        .map(Box::new)?;

    Ok(Expression::Lambda {
        bindings,
        body,
        lambda_type: Type::unknown(),
    })
}

fn build_lambda_parameter_list(node: SourceNode) -> BuildResult<Vec<SoftBinding>> {
    // lambda_parameter_list: $ => seq('(', optional(sepBy(',', $.soft_binding)), ')')
    let mut params = Vec::new();
    for child in node.children() {
        if child.kind() == "soft_binding" {
            params.push(build_soft_binding(child)?);
        }
    }
    Ok(params)
}

fn build_call_expr(node: SourceNode) -> BuildResult<Expression> {
    // call_expression: $ => prec.left(1, seq(function: $._expression, arguments: $.argument_list))
    let fn_expr = node
        .child_by_field_name("function")
        .and_then(build_expression)
        .map(Box::new)?;
    let arg_exprs = node
        .child_by_field_name("arguments")
        .and_then(build_call_arg_list)?;

    Ok(Expression::Call {
        fn_expr,
        arg_exprs,
        return_type: Type::unknown(),
    })
}

fn build_call_arg_list(node: SourceNode) -> BuildResult<Vec<Expression>> {
    // argument_list: $ => seq('(', optional(sepBy(',', $._expression)), ')')
    let mut args = Vec::new();
    for child in node.children() {
        // Need to handle potential intermediate nodes depending on `sepBy` implementation in
        // grammar. Assuming direct children are expressions for now
        if !matches!(child.kind(), "(" | ")" | ",") {
            args.push(build_expression(child)?);
        }
    }
    Ok(args)
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

        assert_eq!(
            build_ast_program(source_code),
            Ok(Program(vec![Declaration::Fn(FnDecl {
                forall: BTreeSet::new(),
                name: "main".to_string(),
                args: vec![],
                return_type: Type::int(),
                body: Some(block_expr(vec![2.into()]))
            })]))
        );
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
        let expected_program = Program(vec![
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
        assert_eq!(build_ast_program(source), Ok(expected_program));
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
        assert_eq!(build_ast_program(source), Ok(program));
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
        assert_eq!(build_ast_program(source), Ok(program));
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
        assert_eq!(build_ast_program(source), Ok(program));
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
        assert_eq!(build_ast_program(source), Ok(program));
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
        assert_eq!(build_ast_program(source), Ok(program));
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
        assert_eq!(build_ast_program(source), Ok(program));
    }
}
