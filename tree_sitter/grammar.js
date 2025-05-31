/**
 * @file Parser for the Silica programming language
 * @author Casper Neo <casperneo1712@gmail.com>
 * @license MIT
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

function sepBy1(sep, rule) {
  return seq(rule, repeat(seq(sep, rule)), optional(sep));
}

function sepBy(sep, rule) {
  return optional(sepBy1(sep, rule));
}

// A delimited list where the start and end are required and there may be a trailing seperator.
function delimited(start, rule, sep, end) {
  return seq(start, sepBy(sep, rule), end);
}

module.exports = grammar({
  name: 'silica',

  // Extras includes whitespace and comments, handled automatically between rules
  extras: $ => [
    /\s+/,      // Whitespace
    $.comment,
  ],

  rules: {
    source_file: $ => repeat($._declaration),

    _declaration: $ => choice(
      $.function_declaration,
      $.struct_declaration,
      $.effect_declaration
      // TODO: coroutine_declaration.
    ),

    // --- Effect Declaration ---
    effect_declaration: $ => seq(
      'effect',
      field('name', $.identifier),
      field('parameters', optional($.generic_parameter_list)),
      delimited("{", $.operation_signature, ",", "}")
    ),

    operation_signature: $ => seq(
      field('op_name', $.identifier),
      ':',
      field('perform_type', $._type),
      '->',
      field('resume_type', $._type)
    ),

    // --- Function Declaration ---
    function_declaration: $ => seq(
      'fn',
      field('name', $.identifier),
      field('parameters', optional($.generic_parameter_list)),
      delimited('(', field("args", $.fn_arg), ',', ')'),
      optional(seq('->', field('return_type', $._type))),
      field('body', choice($.block_expression, ';'))
    ),

    fn_arg: $ => seq( // Corresponds to TypedBinding
      optional('mut'),
      field('name', $.identifier),
      ':',
      field('type', $._type)
    ),

    // --- Struct Declaration ---
    struct_declaration: $ => seq(
      'struct',
      field('name', $.identifier),
      field('parameters', optional($.generic_parameter_list)),
      delimited("{", field('fields', $.struct_field_declaration), ",", "}"),
      // TODO: What about unit structs?
    ),

    struct_field_declaration: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type)
    ),

    // --- Types (for annotations) ---
    _type: $ => choice(
      $.primitive_type,
      $.function_type,
      $.named_type
    ),

    primitive_type: $ => choice(
      'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'i128', 'u128',
      'isize', 'usize', 'f64', 'bool', 'unit'  // TODO unit should be "()"
    ),

    function_type: $ => seq(
      'fn',
      delimited("(", field("arg_types", $._type), ",", ")"),
      '->',
      field("return_type", $._type),
    ),

    named_type: $ => $.identifier, // Simple identifier for now

    // --- Statements ---
    _statement: $ => choice(
      $.let_statement,
      $.assignment_statement,
      $.return_statement,
      $.expression_statement
      // Add perform, handle statements later
    ),

    let_statement: $ => seq(
      'let',
      field('binding', $.soft_binding),
      '=',
      field('value', $._expression),
      ';'
    ),

    assignment_statement: $ => seq(
      field('left', $._l_value),
      '=',
      field('right', $._expression),
      ';'
    ),

    return_statement: $ => seq(
      'return',
      field('value', $._expression),
      ';'
    ),

    expression_statement: $ => seq(
      $._expression,
      ';'
    ),

    // --- Expressions ---
    _expression: $ => choice(
      $.perform_expression,
      $.call_expression,
      $.if_expression,
      $.lambda_expression,
      $.block_expression,
      $._primary_expression
      // Add binary/unary operators with precedence later
    ),

    _primary_expression: $ => choice(
      $.literal,
      $.variable,
      $.parenthesized_expression
    ),

    parenthesized_expression: $ => seq('(', $._expression, ')'),

    variable: $ => $._l_value,

    _l_value: $ => choice(
        // TODO: Deref, Field access, etc.
        $.identifier
    ),

    call_expression: $ => prec.left(1, seq(
      field('function', $._expression),
      delimited("(", field("call_args", $._expression), ",", ")")
    )),

    if_expression: $ => seq(
      'if',
      field('condition', $._expression),
      field('consequence', $.block_expression),
      optional(seq('else', field('alternative', $.block_expression)))
    ),

    lambda_expression: $ => seq(
      delimited("|", field("lambda_args", $.soft_binding), ",", "|"),
      field('body', $._expression),
    ),

    block_expression: $ => seq(
      '{',
        field("statements", repeat($._statement)),
        field("final_expression", optional($._expression)),
      '}'
    ),

    perform_expression: $ => seq(
        'perform',
        optional(seq(
          field('effect_name', $.identifier),
          "."
        )),
        field("op_name", $.identifier),
        '(',
        field('argument', $._expression),
        ')',
    ),

    // --- Literals ---
    literal: $ => choice(
      $.integer_literal,
      $.float_literal,
      $.boolean_literal,
      $.unit_literal
    ),

    integer_literal: $ => token(/\d+/),
    float_literal: $ => token(/\d+\.\d*|\.\d+/), // Simplified
    boolean_literal: $ => choice('true', 'false'),
    unit_literal: $ => '()',

    // --- Identifier ---
    // Tree-sitter automatically handles keywords vs identifiers if keywords are defined as strings
    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    // --- Comment ---
    comment: $ => token(seq('//', /.*/)),

    // --- Other helpers ---

    // Soft binding used in `let` and lambda parameters
    soft_binding: $ => seq(
        optional('mut'),
        field('name', $.identifier),
        optional(seq(':', field('type', $._type)))
    ),

    // TODO: Consider constraining params, e.g. T: MyTrait.
    generic_parameter_list: $ =>  delimited("<", field("parameter_name", $.identifier), ",", ">"),

  }
});