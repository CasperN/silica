/**
 * @file Parser for the Silica programming language
 * @author Casper Neo <casperneo1712@gmail.com>
 * @license MIT
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

function sepBy1(sep, rule) {
  return seq(rule, repeat(seq(sep, rule)));
}

function sepBy(sep, rule) {
  return optional(sepBy1(sep, rule));
}

module.exports = grammar({
  name: 'silica',

  // Extras includes whitespace and comments, handled automatically between rules
  extras: $ => [
    /\s+/,      // Whitespace
    $.comment,
  ],

  // Rules definition
  rules: {
    // Top-level rule: a source file contains zero or more declarations
    source_file: $ => repeat($._declaration),

    _declaration: $ => choice(
      $.function_declaration,
      $.struct_declaration,
      $.effect_declaration
      // Add rules for enum, trait declarations later
    ),

    // --- Effect Declaration ---
    effect_declaration: $ => seq(
      'effect',
      field('name', $.identifier),
      field('generic_parameters', optional($.generic_parameter_list)),
      '{',
      optional(sepBy(',', $.operation_signature)),
      optional(','), // Optional trailing comma for the operations list itself
      '}'
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
      field('args', $.fn_args),
      optional(seq('->', field('return_type', $._type))),
      field('body', choice($.block_expression, ';'))
    ),

    fn_args: $ => seq(
      '(',
      optional(sepBy(',', $.fn_arg)),
      ')'
    ),

    fn_arg: $ => seq( // Corresponds to TypedBinding
      optional('mut'),
      field('name', $.identifier),
      ':',
      field('type', $._type)
    ),

    // --- Structs ---
    struct_declaration: $ => seq(
      'struct',
      field('name', $.identifier),
      field('parameters', optional($.generic_parameter_list)),
      field('fields', $.struct_field_list),
      // TODO: What about unit structs?
    ),


    struct_field_list: $ => seq(
      '{',
      optional(sepBy(',', $.struct_field_declaration)),
      optional(','), // Optional trailing comma for the field list itself
      '}'
    ),

    struct_field_declaration: $ => seq(
      // TODO: Add visibility modifiers (pub, etc.) later if needed
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
      '(',
      optional(sepBy(',', $._type)),
      ')',
      '->',
      $._type
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

    // Soft binding used in `let` and lambda parameters
    soft_binding: $ => seq(
        optional('mut'),
        field('name', $.identifier),
        optional(seq(':', field('type', $._type)))
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
        $.identifier
    ),

    call_expression: $ => prec.left(1, seq(
      field('function', $._expression),
      field('arguments', $.argument_list)
    )),

    argument_list: $ => seq(
        '(',
        optional(sepBy(',', $._expression)),
        ')'
    ),

    if_expression: $ => seq(
      'if',
      field('condition', $._expression),
      field('consequence', $.block_expression),
      optional(seq('else', field('alternative', $.block_expression)))
    ),

    lambda_expression: $ => seq(
      '|',
      field('lambda_args', optional($.soft_bindings)),
      '|',
      field('body', $._expression)
    ),

    soft_bindings: $ => sepBy1(',', $.soft_binding),

    block_expression: $ => seq(
      '{',
        field("statements", repeat($._statement)),
        field("final_expression", optional($._expression)),
      '}'
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

    // Other helpers. 
    generic_parameter_list: $ => seq(
      '<',
      sepBy1(',', field('parameter_name', $.identifier)), // Simple identifiers for generic params for now
                                                          // Could be $._type for bounds later: T: MyTrait
      optional(','), // Optional trailing comma
      '>'
    ),

  }
});