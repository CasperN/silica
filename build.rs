use std::path::PathBuf;
use std::process::Command;

fn main() {
    eprintln!("Running tree-sitter generate..."); // Log for debugging build
    Command::new("tree-sitter")
        .arg("generate")
        // Tree-sitter CLI generates files in ./src relative to its CWD.
        // Ensure the command runs where grammar.js is located if it's not at the crate root.
        .current_dir("tree_sitter")
        .output()
        .expect("Failed to call tree-sitter CLI");

    let parser_src_dir = PathBuf::from("tree_sitter/src");
    let parser_include_dir = PathBuf::from("tree_siter/src/tree_sitter");

    let mut c_config = cc::Build::new();
    c_config.include(&parser_include_dir); // Include directory for header files
    c_config
        .flag_if_supported("-Wno-unused-parameter") // Suppress common warnings in generated code
        .flag_if_supported("-Wno-unused-but-set-variable")
        // .flag_if_supported("-Wno-trigraphs"); // Just in case
    ;
    // Compile parser.c
    let parser_path = parser_src_dir.join("parser.c");
    c_config.file(&parser_path);
    println!("cargo:rerun-if-changed={}", parser_path.to_str().unwrap());

    // Compile the static library (adjust name as desired)
    c_config.compile("tree-sitter-silica-parser");

    // Optional: Tell cargo to rerun build script if grammar changes,
    // *if* you want to regenerate parser.c automatically (requires tree-sitter-cli dependency)
    // Simpler approach is to manually run `tree-sitter generate` and commit parser.c
    println!("cargo:rerun-if-changed=tree_sitter/grammar.js");
}
