use std::collections::{hash_map::Entry, HashMap, HashSet};

use crate::ast::{
    Declaration, Expression, FnDecl, LValue, Program, SoftBinding, Statement, StructDecl, Type,
    TypedBinding,
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
pub enum AstBuildError {
    UnexpectedNodeType {
        expected: String,
        found: String,
        node_text: String,
    },
    MissingChild {
        node_kind: String,
        field_name: Option<String>, // Use field name if available
        child_index: Option<usize>, // Or index
    },
    InvalidLiteral {
        kind: String,
        text: String,
        error: String,
    },
    // Add more specific errors as needed
    Other(String),
}

// Implement std::error::Error etc. if desired

// Helper type alias for results
type BuildResult<T> = Result<T, AstBuildError>;

// --- Main Conversion Function ---

/// Converts a tree-sitter Tree into an ast::Program
pub fn build_ast_program(source: &str) -> BuildResult<Program> {
    let tree = parse(source).ok_or(AstBuildError::Other("TODO".to_string()))?;
    let root_node = tree.root_node();
    let expected_kind = "source_file";
    if root_node.kind() != expected_kind {
        return Err(AstBuildError::UnexpectedNodeType {
            expected: expected_kind.to_string(),
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
            "ERROR" => {
                panic!()
            }
            _ => {
                unimplemented!("{}", child_node.kind());
            }
        }
    }

    Ok(Program(declarations))
}

// --- Node Conversion Functions ---

fn build_fn_decl(node: &Node, source: &str) -> BuildResult<Declaration> {
    let name_node =
        node.child_by_field_name("name")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "function_declaration".to_string(),
                field_name: Some("name".to_string()),
                child_index: None,
            })?;
    let params_node =
        node.child_by_field_name("parameters")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "function_declaration".to_string(),
                field_name: Some("parameters".to_string()),
                child_index: None,
            })?;
    let return_type_node_opt = node.child_by_field_name("return_type"); // Optional field
    let body_node =
        node.child_by_field_name("body")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "function_declaration".to_string(),
                field_name: Some("body".to_string()),
                child_index: None,
            })?;

    let name = get_node_text(&name_node, source);
    let params = build_parameter_list(&params_node, source)?;
    let return_type = match return_type_node_opt {
        Some(n) => build_type(&n, source)?,
        None => Type::unit(), // Default return type if not specified
    };

    // Check if body is ';' (external) or a block expression
    let body = match body_node.kind() {
        "block_expression" => Some(build_block_expr(&body_node, source)?),
        ";" => None, // External function
        _ => {
            return Err(AstBuildError::UnexpectedNodeType {
                expected: "block_expression or ;".to_string(),
                found: body_node.kind().to_string(),
                node_text: get_node_text(&body_node, source).to_string(),
            })
        }
    };

    Ok(Declaration::Fn(FnDecl {
        forall: vec![], // TODO: Parse polymorphic function declarations.
        name: name.to_string(),
        args: params,
        return_type,
        body,
    }))
}
fn build_struct_decl(node: &Node, source: &str) -> BuildResult<Declaration> {
    let name_node =
        node.child_by_field_name("name")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "struct_declaration".to_string(),
                field_name: Some("name".to_string()),
                child_index: None,
            })?;
    let generic_params_node = node.child_by_field_name("parameters");
    let field_nodes =
        node.child_by_field_name("fields")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "struct_declaration".to_string(),
                field_name: Some("fields".to_string()),
                child_index: None,
            })?;

    let name = get_node_text(&name_node, source).to_string();

    let params = generic_params_node
        .map(|node| build_generic_params(&node, source))
        .unwrap_or(Ok(HashSet::new()))?;

    let fields = build_fields_map(&field_nodes, source)?;

    Ok(Declaration::Struct(StructDecl {
        params,
        name,
        fields,
    }))
}

fn build_generic_params(node: &Node, source: &str) -> BuildResult<HashSet<u32>> {
    let _ = (node, source);
    todo!()
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
                    return Err(AstBuildError::Other(format!("Name conflict {field_name}")));
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
    let name_node =
        node.child_by_field_name("name")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "struct_field".to_string(),
                field_name: Some("name".to_string()),
                child_index: None,
            })?;
    let type_node =
        node.child_by_field_name("type")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "struct_field".to_string(),
                field_name: Some("type".to_string()),
                child_index: None,
            })?;
    let name = get_node_text(&name_node, source).to_string();
    let ty = build_type(&type_node, source)?;
    Ok((name, ty))
}

fn build_parameter_list(node: &Node, source: &str) -> BuildResult<Vec<TypedBinding>> {
    // parameter_list: $ => seq('(', optional(sepBy(',', $.parameter)), ')')
    let mut params = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "parameter" {
            params.push(build_parameter(&child, source)?);
        }
        // Skip '(' ')' ',' tokens if they appear as unnamed children
    }
    Ok(params)
}

fn build_parameter(node: &Node, source: &str) -> BuildResult<TypedBinding> {
    // parameter: $ => seq(optional('mut'), name: $.identifier, ':', type: $._type)
    let is_mutable = node.child(0).map_or(false, |n| n.kind() == "mut"); // Check first child
    let name_node =
        node.child_by_field_name("name")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "parameter".to_string(),
                field_name: Some("name".to_string()),
                child_index: None,
            })?;
    let type_node =
        node.child_by_field_name("type")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "parameter".to_string(),
                field_name: Some("type".to_string()),
                child_index: None,
            })?;

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
                _ => Err(AstBuildError::Other(format!(
                    "Unknown primitive type: {}",
                    type_name
                ))),
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
        // "named_type" | "identifier" => Ok(Type::Named(get_node_text(node, source))), // Assuming Type::Named
        _ => Err(AstBuildError::UnexpectedNodeType {
            expected: "type".to_string(),
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
        _ => Err(AstBuildError::UnexpectedNodeType {
            expected: "statement".to_string(),
            found: node.kind().to_string(),
            node_text: get_node_text(node, source).to_string(),
        }),
    }
}

fn build_let_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    let binding_node =
        node.child_by_field_name("binding")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "let_statement".into(),
                field_name: Some("binding".into()),
                child_index: None,
            })?;
    let value_node =
        node.child_by_field_name("value")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "let_statement".into(),
                field_name: Some("value".into()),
                child_index: None,
            })?;

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
            let inner_node = node.child(1).ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "parenthesized_expression".into(),
                field_name: None,
                child_index: Some(1),
            })?; // Skip '('
            build_expression(&inner_node, source)
        }
        // Potentially handle _primary_expression or _l_value if needed by grammar structure
        _ => Err(AstBuildError::UnexpectedNodeType {
            expected: "expression".to_string(),
            found: node.kind().to_string(),
            node_text: get_node_text(node, source).to_string(),
        }),
    }
}

fn build_literal_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    let text = get_node_text(node, source);
    // Go into the literal.
    match node.named_child(0).map(|n| n.kind()) {
        Some("integer_literal") => text
            .parse::<i64>()
            .map(Expression::LiteralInt)
            .map_err(|e| AstBuildError::InvalidLiteral {
                kind: "integer".into(),
                text: text.to_string(),
                error: e.to_string(),
            }),
        Some("float_literal") => text
            .parse::<f64>()
            .map(Expression::LiteralFloat)
            .map_err(|e| AstBuildError::InvalidLiteral {
                kind: "float".into(),
                text: text.to_string(),
                error: e.to_string(),
            }),
        Some("boolean_literal") => match text {
            "true" => Ok(Expression::LiteralBool(true)),
            "false" => Ok(Expression::LiteralBool(false)),
            _ => unreachable!(), // Grammar should prevent this
        },
        Some("unit_literal") => Ok(Expression::LiteralUnit),
        Some(_) => Err(AstBuildError::UnexpectedNodeType {
            expected: "literal".to_string(), // TODO: What did we expect?
            found: node.kind().to_string(),
            node_text: text.to_string(),
        }),
        None => Err(AstBuildError::MissingChild {
            node_kind: "literal".to_string(),
            field_name: None,
            child_index: None,
        }),
    }
}

fn build_variable_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    // Assuming 'variable' node wraps an 'identifier' node
    let ident_node = node.child(0).ok_or_else(|| AstBuildError::MissingChild {
        node_kind: "variable".into(),
        field_name: None,
        child_index: Some(0),
    })?;
    Ok(Expression::L(
        LValue::Variable(get_node_text(&ident_node, source).to_string()),
        Type::unknown(),
    ))
}

fn build_block_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    // block_expression: $ => seq('{', repeat($._statement), optional($._expression), '}')
    let mut statements = Vec::new();
    let mut final_expr = None;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "{" | "}" => {} // Skip braces
            "comment" => {} // Skip comments
            // Check if it's a statement kind before checking for expression kind
            "let_statement"
            | "assignment_statement"
            | "return_statement"
            | "expression_statement" => {
                // If we already found a final expression, it's an error or needs specific handling
                if final_expr.is_some() {
                    return Err(AstBuildError::Other(
                        "Statement after final expression in block".into(),
                    ));
                }
                statements.push(build_statement(&child, source)?);
            }
            // Assume anything else is the final expression (needs refinement)
            _ => {
                if final_expr.is_some() {
                    let current = child.to_string();
                    return Err(AstBuildError::Other(
                        format!("Multiple final expressions in block? existing: {final_expr:?} current: {current}"),
                    ));
                }
                final_expr = Some(build_expression(&child, source)?);
            }
        }
    }
    // AST requires block body to be Vec<Statement>, handle final expression
    if let Some(expr) = final_expr {
        statements.push(Statement::Expression(expr)); // Convert final expr to statement
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
    let name_node =
        node.child_by_field_name("name")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "name".to_string(),
                field_name: None,
                child_index: None,
            })?;
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
        _ => Err(AstBuildError::UnexpectedNodeType {
            expected: "lvalue".into(),
            found: node.kind().into(),
            node_text: get_node_text(node, source).to_string(),
        }),
    }
}

fn build_assignment_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    let left_node =
        node.child_by_field_name("left")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "assignee".to_string(),
                field_name: None,
                child_index: None,
            })?;
    let right_node =
        node.child_by_field_name("right")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "assignment_value".to_string(),
                field_name: None,
                child_index: None,
            })?;
    Ok(Statement::Assign(
        build_l_value(&left_node, source)?,
        build_expression(&right_node, source)?,
    ))
}

fn build_return_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    let value_node =
        node.child_by_field_name("value")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "return value".to_string(),
                field_name: None,
                child_index: None,
            })?;
    Ok(Statement::Return(build_expression(&value_node, source)?))
}

fn build_expression_statement(node: &Node, source: &str) -> BuildResult<Statement> {
    // expression_statement: $ => seq($._expression, ';')
    let expr_node = node.child(0).ok_or_else(|| AstBuildError::MissingChild {
        node_kind: "expression".to_string(),
        field_name: None,
        child_index: None,
    })?;
    Ok(Statement::Expression(build_expression(&expr_node, source)?))
}

fn build_if_expr(node: &Node, source: &str) -> BuildResult<Expression> {
    let cond_node =
        node.child_by_field_name("condition")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "condition".to_string(),
                field_name: None,
                child_index: None,
            })?;
    let cons_node =
        node.child_by_field_name("consequence")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "consequence".to_string(),
                field_name: None,
                child_index: None,
            })?;
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
    let params_node =
        node.child_by_field_name("parameters")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "parameters".to_string(),
                field_name: None,
                child_index: None,
            })?;
    // TODO: Currently, body expects a block expression. Can it just be an expression?
    let body_node =
        node.child_by_field_name("body")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "body".to_string(),
                field_name: None,
                child_index: None,
            })?;

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
    let func_node =
        node.child_by_field_name("function")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "function".to_string(),
                field_name: None,
                child_index: None,
            })?;
    let args_node =
        node.child_by_field_name("arguments")
            .ok_or_else(|| AstBuildError::MissingChild {
                node_kind: "arguments".to_string(),
                field_name: None,
                child_index: None,
            })?;

    Ok(Expression::Call {
        fn_expr: Box::new(build_expression(&func_node, source)?),
        arg_exprs: build_argument_list(&args_node, source)?,
        return_type: Type::unknown(),
    })
}

fn build_argument_list(node: &Node, source: &str) -> BuildResult<Vec<Expression>> {
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
    use std::collections::{HashMap, HashSet};

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
                forall: vec![],
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
        assert_eq!(build_ast_program(source), Ok(program));
    }
    #[test]
    fn declare_struct() {
        let source = r"struct FooBar { foo: i64, bar: f64 }";
        let program = Program(vec![Declaration::Struct(StructDecl {
            params: HashSet::new(),
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
    // TODO Declare generic struct and generic function.
}
