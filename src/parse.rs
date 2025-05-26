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

// TODO: Refactor these errors
// - `kind` probably can be a static str
// - They need helper functions.
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
    let tree = parse(source).ok_or(ParseError::NoTree)?;
    let root_node = tree.root_node();
    if root_node.kind() != "source_file" {
        return Err(ParseError::UnexpectedNodeType {
            expected: "source_file",
            found: root_node.kind().to_string(),
            node_text: get_node_text(&root_node, source).to_string(),
        });
    }

    let mut declarations = Vec::new();
    let mut cursor = root_node.walk(); // Use cursor for iterating children

    for child_node in root_node.children(&mut cursor) {
        // Depending on grammar, might need to check child_node.kind() == "_declaration" first
        match child_node.kind() {
            "function_declaration" => {
                declarations.push(build_fn_decl(&child_node, source)?);
            }
            "struct_declaration" => {
                declarations.push(build_struct_decl(&child_node, source)?);
            }
            "effect_declaration" => {
                declarations.push(build_effect_decl(&child_node, source)?);
            }
            _ => {
                unimplemented!("{}", child_node.kind());
            }
        }
    }

    Ok(Program(declarations))
}

// --- Node Conversion Functions ---
fn build_effect_decl(node: &Node, source: &str) -> BuildResult<Declaration> {
    let name_node = get_required_child(node, source, "name")?; // [cite: 792]
    let name = get_node_text(&name_node, source).to_string(); // [cite: 879]

    let generic_params_node = node.child_by_field_name("generic_parameters");
    let params = if let Some(gp_node) = generic_params_node {
        build_generic_params(&gp_node, source)? //
    } else {
        BTreeSet::new()
    };

    let mut ops = HashMap::new();
    let mut cursor = node.walk(); // [cite: 796]

    // Iterate through children of the effect_declaration node to find operation signatures
    for child_node in node.children(&mut cursor) {
        dbg!(child_node.kind());
        if child_node.kind() == "operation_signature" {
            let op_name_node = get_required_child(&child_node, source, "op_name")?;
            let op_name = get_node_text(&op_name_node, source).to_string();

            let perform_type_node = get_required_child(&child_node, source, "perform_type")?;
            let perform_type = build_type(&perform_type_node, source)?;

            let resume_type_node = get_required_child(&child_node, source, "resume_type")?;
            let resume_type = build_type(&resume_type_node, source)?;

            match ops.entry(op_name.clone()) {
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

fn build_fn_decl(node: &Node, source: &str) -> BuildResult<Declaration> {
    let name_node = get_required_child(node, source, "name")?;
    let params_node = node.child_by_field_name("parameters");
    let args_node = get_required_child(node, source, "args")?;
    let return_type_node_opt = node.child_by_field_name("return_type"); // Optional field
    let body_node = get_required_child(node, source, "body")?;

    let name = get_node_text(&name_node, source);
    let params = if let Some(node) = params_node {
        dbg!(&node);
        build_generic_params(&node, source)?
    } else {
        BTreeSet::new()
    };
    let args = build_fn_arg_list(&args_node, source)?;
    let return_type = match return_type_node_opt {
        Some(n) => build_type(&n, source)?,
        None => Type::unit(), // Default return type if not specified
    };

    // Check if body is ';' (external) or a block expression
    let body = match body_node.kind() {
        "block_expression" => Some(build_block_expr(&body_node, source)?),
        ";" => None, // External function
        _ => {
            return Err(ParseError::UnexpectedNodeType {
                expected: "block_expression or ;",
                found: body_node.kind().to_string(),
                node_text: get_node_text(&body_node, source).to_string(),
            })
        }
    };

    Ok(Declaration::Fn(FnDecl {
        forall: params,
        name: name.to_string(),
        args,
        return_type,
        body,
    }))
}
fn build_struct_decl(node: &Node, source: &str) -> BuildResult<Declaration> {
    let name_node = get_required_child(node, source, "name")?;
    let generic_params_node = node.child_by_field_name("generic_parameters");
    let field_nodes = get_required_child(node, source, "fields")?;

    let name = get_node_text(&name_node, source).to_string();

    let params = generic_params_node
        .map(|node| build_generic_params(&node, source))
        .unwrap_or(Ok(BTreeSet::new()))?;

    let fields = build_fields_map(&field_nodes, source)?;

    Ok(Declaration::Struct(StructDecl {
        params,
        name,
        fields,
    }))
}

fn build_generic_params(node: &Node, source: &str) -> BuildResult<BTreeSet<String>> {
    let mut params = BTreeSet::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            let child_text = get_node_text(&child, source);
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

fn build_fields_map(node: &Node, source: &str) -> BuildResult<HashMap<String, Type>> {
    // { field_name: field_type, }
    let mut fields = HashMap::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "struct_field_declaration" {
            let (field_name, field_type) = build_struct_field(&child, source)?;
            match fields.entry(field_name.to_string()) {
                Entry::Occupied(_) => {
                    // TODO: Better error.
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

fn build_struct_field(node: &Node, source: &str) -> BuildResult<(String, Type)> {
    let name_node = get_required_child(node, source, "name")?;
    let type_node = get_required_child(node, source, "type")?;
    let name = get_node_text(&name_node, source).to_string();
    let ty = build_type(&type_node, source)?;
    Ok((name, ty))
}

fn build_fn_arg_list(node: &Node, source: &str) -> BuildResult<Vec<TypedBinding>> {
    // parameter_list: $ => seq('(', optional(sepBy(',', $.parameter)), ')')
    let mut params = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "fn_arg" {
            params.push(build_fn_arg(&child, source)?);
        }
        // Skip '(' ')' ',' tokens if they appear as unnamed children
    }
    Ok(params)
}

fn build_fn_arg(node: &Node, source: &str) -> BuildResult<TypedBinding> {
    // parameter: $ => seq(optional('mut'), name: $.identifier, ':', type: $._type)
    let is_mutable = node.child(0).map_or(false, |n| n.kind() == "mut"); // Check first child
    let name_node = get_required_child(node, source, "name")?;
    let type_node = get_required_child(node, source, "type")?;

    Ok(TypedBinding {
        name: get_node_text(&name_node, source).to_string(),
        ty: build_type(&type_node, source)?,
        mutable: is_mutable,
    })
}

// --- Type Parsing ---
fn build_type(node: &Node, source: &str) -> BuildResult<Type> {
    match node.kind() {
        "primitive_type" => {
            let type_name = get_node_text(node, source);
            match type_name {
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
            let mut cursor = node.walk();
            let mut found_arrow = false;
            for child in node.children(&mut cursor) {
                if child.kind() == "->" {
                    found_arrow = true;
                    continue; // Skip arrow token
                }
                if is_type_node(&child) {
                    // Helper function needed
                    if found_arrow {
                        return_type = Some(build_type(&child, source)?);
                    } else {
                        // Assume types before '->' are args
                        arg_types.push(build_type(&child, source)?);
                    }
                }
                // Skip 'fn', '(', ')', ',' tokens
            }
            Ok(Type::func(arg_types, return_type.unwrap_or(Type::unit())))
        }
        // TODO: Named type support
        "named_type" | "identifier" => Ok(Type::param(get_node_text(node, source))),
        _ => Err(ParseError::UnexpectedNodeType {
            expected: "type",
            found: node.kind().to_string(),
            node_text: get_node_text(node, source).to_string(),
        }),
    }
}

// --- Statement Parsing ---
fn build_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    match node.kind() {
        "let_statement" => build_let_statement(node, source),
        "assignment_statement" => build_assignment_statement(node, source),
        "return_statement" => build_return_statement(node, source),
        "expression_statement" => build_expression_statement(node, source),
        _ => Err(ParseError::UnexpectedNodeType {
            expected: "statement",
            found: node.kind().to_string(),
            node_text: get_node_text(node, source).to_string(),
        }),
    }
}

fn build_let_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    let binding_node = get_required_child(node, source, "binding")?;
    let value_node = get_required_child(node, source, "value")?;

    Ok(Statement::Let {
        binding: build_soft_binding(&binding_node, source)?,
        value: build_expression(&value_node, source)?,
    })
}

// --- Expression Parsing ---
fn build_expression(node: &Node, source: &str) -> BuildResult<Expression> {
    // Use node.kind() to dispatch to specific build functions
    match node.kind() {
        "if_expression" => build_if_expr(node, source),
        "lambda_expression" => build_lambda_expr(node, source),
        "call_expression" => build_call_expr(node, source),
        "block_expression" => build_block_expr(node, source),
        "literal" | "integer_literal" | "float_literal" | "boolean_literal" | "unit_literal" => {
            build_literal_expr(node, source)
        }
        "variable" => build_variable_expr(node, source),
        "parenthesized_expression" => {
            let inner_node = get_required_child_by_id(node, source, 1)?;
            build_expression(&inner_node, source)
        }
        // Potentially handle _primary_expression or _l_value if needed by grammar structure
        _ => Err(ParseError::UnexpectedNodeType {
            expected: "expression",
            found: node.kind().to_string(),
            node_text: get_node_text(node, source).to_string(),
        }),
    }
}

fn build_literal_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    let text = get_node_text(node, source);
    // Go into the literal.
    let child = get_required_child_by_id(node, source, 0)?;

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

fn build_variable_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    // Assuming 'variable' node wraps an 'identifier' node
    let ident_node = get_required_child_by_id(node, source, 0)?;
    Ok(Expression::L(
        LValue::Variable(get_node_text(&ident_node, source).to_string()),
        Type::unknown(),
    ))
}

fn build_block_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    let mut statements = Vec::new();
    let mut cursor = node.walk();
    for statement_node in node.children_by_field_name("statements", &mut cursor) {
        statements.push(build_statement(&statement_node, source)?);
    }
    if let Some(node) = node.child_by_field_name("final_expression") {
        statements.push(Statement::Expression(build_expression(&node, source)?));
    }
    Ok(Expression::Block {
        statements,
        ty: Type::unknown(),
    })
}

// --- TODO: Implement remaining build functions ---

fn build_soft_binding(node: &Node, source: &str) -> BuildResult<SoftBinding> {
    // soft_binding: $ => seq(optional('mut'), name: $.identifier, optional(seq(':', type: $._type)))
    let is_mutable = node.child(0).map_or(false, |n| n.kind() == "mut");
    let name_node = get_required_child(node, source, "name")?;
    let type_node_opt = node.child_by_field_name("type"); // Optional

    Ok(SoftBinding {
        name: get_node_text(&name_node, source).to_string(),
        ty: type_node_opt.map(|n| build_type(&n, source)).transpose()?,
        mutable: is_mutable,
    })
}

fn build_l_value(node: &Node, source: &str) -> BuildResult<LValue> {
    match node.kind() {
        "identifier" => Ok(LValue::Variable(get_node_text(node, source).to_string())),
        _ => Err(ParseError::UnexpectedNodeType {
            expected: "lvalue",
            found: node.kind().into(),
            node_text: get_node_text(node, source).to_string(),
        }),
    }
}

fn build_assignment_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    let left_node = get_required_child(node, source, "left")?;
    let right_node = get_required_child(node, source, "right")?;
    Ok(Statement::Assign(
        build_l_value(&left_node, source)?,
        build_expression(&right_node, source)?,
    ))
}

fn build_return_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    let value_node = get_required_child(node, source, "value")?;
    Ok(Statement::Return(build_expression(&value_node, source)?))
}

fn build_expression_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    // expression_statement: $ => seq($._expression, ';')
    let expr_node = get_required_child_by_id(node, source, 0)?;
    Ok(Statement::Expression(build_expression(&expr_node, source)?))
}

fn build_if_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    let cond_node = get_required_child(node, source, "condition")?;
    // TODO: Change these weird names.
    let cons_node = get_required_child(node, source, "consequence")?;
    let alt_node_opt = node.child_by_field_name("alternative");

    let false_expr = match alt_node_opt {
        Some(alt_node) => build_block_expr(&alt_node, source)?, // Assume block expr
        None => Expression::Block {
            statements: vec![Statement::Expression(Expression::LiteralUnit)],
            ty: Type::unknown(),
        }, // Default else
    };

    Ok(Expression::If {
        condition: Box::new(build_expression(&cond_node, source)?),
        true_expr: Box::new(build_block_expr(&cons_node, source)?), // Assume block expr
        false_expr: Box::new(false_expr),
        ty: Type::unknown(),
    })
}

fn build_lambda_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    let params_node = get_required_child(node, source, "lambda_args")?;
    // TODO: Currently, body expects a block expression. Can it just be an expression?
    let body_node = get_required_child(node, source, "body")?;

    Ok(Expression::Lambda {
        bindings: build_lambda_parameter_list(&params_node, source)?,
        body: Box::new(build_expression(&body_node, source)?),
        lambda_type: Type::unknown(),
    })
}

fn build_lambda_parameter_list(node: &Node, source: &str) -> BuildResult<Vec<SoftBinding>> {
    // lambda_parameter_list: $ => seq('(', optional(sepBy(',', $.soft_binding)), ')')
    let mut params = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "soft_binding" {
            params.push(build_soft_binding(&child, source)?);
        }
    }
    Ok(params)
}

fn build_call_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    // call_expression: $ => prec.left(1, seq(function: $._expression, arguments: $.argument_list))
    let func_node = get_required_child(node, source, "function")?;
    let args_node = get_required_child(node, source, "arguments")?;

    Ok(Expression::Call {
        fn_expr: Box::new(build_expression(&func_node, source)?),
        arg_exprs: build_call_arg_list(&args_node, source)?,
        return_type: Type::unknown(),
    })
}

fn build_call_arg_list(node: &Node, source: &str) -> BuildResult<Vec<Expression>> {
    // argument_list: $ => seq('(', optional(sepBy(',', $._expression)), ')')
    let mut args = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        // Need to handle potential intermediate nodes depending on `sepBy` implementation in grammar
        // Assuming direct children are expressions for now
        if !matches!(child.kind(), "(" | ")" | ",") {
            args.push(build_expression(&child, source)?);
        }
    }
    Ok(args)
}

// --- Helper Functions ---

/// Helper to get the source text for a node. Panics on error for simplicity here.
fn get_node_text<'a>(node: &Node<'a>, source: &'a str) -> &'a str {
    node.utf8_text(source.as_bytes())
        .expect("Node text decode failed")
}

/// Helper to check if a node represents a type rule in the grammar
fn is_type_node(node: &Node) -> bool {
    matches!(
        node.kind(),
        "primitive_type" | "function_type" | "named_type" | "_type"
    )
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
