# Silica Language Design Document

#### Document Purpose and Audience:
This document defines the Silica programming language: its core features, design rationale, implementation state, and trade-offs. It serves as a reference for ongoing design and implementation. The intended audience includes the primary designer, future AI collaborators, and programmers familiar with Rust/C++ but not necessarily PL theory.

## 1. Overview & Philosophy
Silica is envisioned as an experimental systems programming language combining fine-grained low-level control with powerful abstractions via a generalized algebraic effect system . It targets LLVM for efficient native code generation, leveraging effects for compositional design while retaining direct resource control. Foundational goals include strong, compile-time memory and resource safety (inspired by Rust/linearity), detailed in subsequent sections. Silica draws inspiration from Ante, C, C++, Haskell, Hylo, Koka, and Rust.

## 2. Core Language Features

### 2.1 Type System Foundations
* Foundation (HM Hybrid): Uses a Hindley-Milner inspired system. Top-level functions require full type annotations (generics, lifetimes, effects). Local type inference (Algorithm W-like) deduces types for let bindings within functions.
* Polymorphism:
  * Universal (forall) types: (forall T: TraitBound...). This allows for standard generic code reusable across different types satisfying specified constraints (traits). Traits define shared interfaces or properties that types can implement. Generic functions are preserved in Silica’s Mid-level IR (MIR) for analysis before potential monomorphization in the LLVM backend.
  * Existential (exists) types: (exists S: TraitBound...). This lets generics provide a hidden type S, that only has some defined interface (trait bounds). Primarily used for abstracting types, especially the opaque state/implementation of coroutines (see 2.3). Requires explicit annotations. Practical Explanation: Think of it like returning an abstract "iterator" or "future" where you know what it does (its interface or trait bounds) but not its exact internal type.
* Function Signatures & Effects: Effectful functions always return an opaque coroutine object (Co<T!E>) via existential types (fn foo() -> S where exists S: Co<T!E> + Bounds...). Effects enable library implementations of features like exceptions, generators, and async/await, promoting composability (e.g., Co<T, Fail<E>, Async> combines failure and async effects). Pure functions return values directly (fn add(x: i32, y: i32) -> i32).
* Algebraic Effects (Motivation): Effects provide a single, type-safe framework to implement exceptions, callbacks, generators, async/await as library features. These would normally be language features. Effects are also more simply composable . In Rust, Future<Result<Iterator<Item=T>, E>> is very different from Iterator<Item=Future<Result<T, E>>>. With effects, they’d both be Co<T, Fail<E>, Async>. Function signatures explicitly declare the effects that their returned coroutine might perform.

### 2.2 Linearity, Ownership, and Movability
Silica incorporate linear types (values consumed exactly once) for compile-time resource safety, preventing leaks and double-use (e.g. in memory or file handles). This strict default is made ergonomic via the following relaxations:
* **References & Borrowing**: Allows temporary, non-consuming access (`&'a T`, `&'a mut T`, `&'a out T`, `&'a deinit T`). Borrowing doesn't consume linear values, but lifetime rules prevent dangling references (Rust NLL-inspired, analyzed on MIR). Definite assignment analysis tracks initialization status for &out/&deinit safety.
  * &'a T: Immutable shared borrow.
  * &'a mut T: Mutable exclusive borrow (allows deinit/reinit within borrow).
  * &'a out T: Output exclusive borrow (must be initialized).
  * &'a deinit T: Deinitialization exclusive borrow (must be deinitialized).

* **Opt-in implicits**: The following traits further relax linearity rules by controlling implicit destruction and copying:
  * ImplicitDrop (Affine): Allows values to be used at most once . If unused by scope end, its drop method is called automatically. Enables affinity.

        trait ImplicitDrop {
          // Non-effectful cleanup logic.
          fn drop(&deinit self);
        }

      *Note:* `drop` cannot perform effects. Effectful cleanup requires an explicit function like `destroy(&deinit self) -> Co<() ! E>`.
  * ImplicitCopy (Relevant/Unrestricted): Allows values to be used one or more times (relevant) or any number of times (unrestricted, if also ImplicitDrop). Implicit copies are made via the copy method when needed.

        trait ImplicitCopy { // Note: Does NOT require ImplicitDrop
          // Performs a copy. Allows user-defined logic (not just memcpy).
          fn copy(&self, &out other: Self);
        }

* **Movability**: Types are immovable by default.
  * Rationale: Stable addresses; enables safe in-place init/deinit via `&out`/`&deinit`; avoids Pin complexity.
  * Requires opt-in via the Move trait for types that can be safely moved (potentially via user-defined logic).

        trait Move {
          // Performs a move, consuming `self`. Allows user-defined logic (not just memcpy).
          // The `&out other` parameter receives the moved-to value.
          fn move(&deinit self, &out other: Self);
        }

  * **Pass-by-value requires the `Move` trait.** Attempting to pass a type that does *not* implement `Move` by value results in a **compile-time error**. This enforces the default immovability of types.
  * For types that *do* implement `Move`, pass-by-value transfers ownership to the callee. The `Move::move` operation consumes the source (`&deinit self`) and initializes the destination (`&out other`).

  * **Aspirational Goal (Move/Copy Elision):** For types implementing `Move` (or `ImplicitCopy`), the compiler *may* optimize away the actual call to the `move` (or `copy`) method, directly constructing the value in the destination or reusing the source memory location, similar to Guaranteed RVO/Copy Elision in C++.

* To transfer ownership *responsibility* (specifically, the obligation to deinitialize) for an **immovable type** without moving its memory, pass it via an exclusive **`&deinit T`** reference. The caller retains the storage allocation, but the callee is responsible for deinitialization.

* It is generally best practice to mark types as `Move`, `ImplicitDrop` and `ImplicitCopy` where possible and semantically correct.
Aggregate types (and closures/coroutines) can only have those traits if all their members (and captured state) have these properties.

### 2.3 Effects, Coroutines, and Syntax
Algebraic Effects are a generalization of exceptions and coroutines that can model what would normally be language-level features at the library level. Most notably, these include generators, async/await, and exceptions. They can be tought of as "resumable exceptions".

The following example demonstrates two algebraic effects, `UserPrompt` and `Fail`.
  ```
  // 1. Define effects
  effect UserPrompt { ask(String) -> String } // Resumable
  effect Fail<E> { fail(E) -> ! }             // Non-resumable

  // 2. Define a function using effects
  fn calculator() -> S
    where exists S: Co<i32!UserPrompt, Fail<String>>
  {
    // `perform` transfers control to an enclosing handler, which may
    // transfer control back to this function with a value.
    let a = perform UserPrompt::ask("give me a");
    let b = perform UserPrompt::ask("give me b");

    // Assume `parse_int()` is an effectful function that may return an i32
    // but also perform Fail<()>.

    let int_a = parse_int() handle {
      // handle Fail<()> by performing our own Fail<String>. 
      Fail::fail(()) => { perform Fail::fail("Failed to parse `a`") },
      // handle a returned value with the identify function.
      return i => i,
    };

    let int_b parse_int() handle {
      Fail::fail(()) => { perform Fail::fail("Failed to parse `b`") },
      // The identity return block was the default and need not be specified.
    }

    // The last expression in the block is returned.
    int_a + int_b  
  }
  ```

* Core Mechanism: Effects separate what operation (defined by effect `E { op(X)->Y }`) from how it's handled. `perform E.op(x)` suspends the current coroutine; control transfers outwards to the nearest enclosing handle block that matches E.op.
* Postfix Handlers (handle): Primary way to manage coroutine execution and handle effects. Applied postfix: `expr handle { Effect.op(arg, resume) => { ... }, return result => { ... } }`. Control transfers to the matching arm; the handler decides when/how to resume. Resumption may be delayed (e.g., for async).
* `?` Operator (Sugar): Manages coroutine execution: returns success value T or propagates unhandled effects E. Operates on Co<T!E> values. Equivalent to a default forwarding handle block:
Semantics: `expr?` is roughly equivalent to installing a default handler that forwards any effect using perform:
    ```
    // Conceptual desugaring of `expr?` where expr yields a Co<T!E>
    expr handle {
        // For any effect E with operation op that the coroutine might perform:
        E.op(arg, resume) => {
            // Re-perform the effect outwards using `perform`.
            // This searches for the next enclosing handler.
            let result = perform E.op(arg);
            // Transfer control back to the original coroutine with the result
            // obtained from handling the effect higher up the stack.
            resume(result)
        }
        return v => v,  // the handle returns v, and thus `expr?` evaluates to v.
    }
    ```

* Usage: Simplifies chaining effectful operations. Can only be used within another effectful function context that includes the propagated effects E.
* Handling Scoping: Effects performed within a handler arm (i.e., inside the `{...}` following `=>`) are not caught by the handler itself; perform always propagates outwards from the arm to the next enclosing handler scope. (Subject to the tunneling caveat below)
* Tunneling: Handle blocks in the body of an effect-parameterized function will always propagate the parameterized effect, even if the implementation handles effects of the same type. The parameterized effect "tunnels through" the body of a polymorphic function. This prevents accidental handling as generic code cannot know the type of an effect its generic over. [link to algebraic effect tunneling paper here]
* Representation: Effectful functions return opaque coroutines (Co<T!E>) via existential types (-> S where exists S: Co<...> + Bounds...). Bounds reflect captured state properties. Co trait interface TBD.
Syntax Notes: Imperative core; expression-oriented; Rust-like keywords. Postfix match; `->` pointer access; No implicit deref; Loops via stdlib functions using Yield/Break effects.

## 3. Design Trade-offs
This section summarizes key design choices where Silica prioritizes certain goals over others, accepting the associated trade-offs.

* **Linearity vs. Flexibility**: Silica prioritizes strict linearity by default for maximum resource safety (exactly-once usage). Flexibility and ergonomics are then recovered through opt-in relaxations (discussed above). Trade-off: Requires explicit opt-in for common affine/unrestricted patterns vs. Silica’s linear-by-default.
* **Immovability vs. Ergonomics**: Silica chooses immovability by default, requiring types to opt into the `Move` trait if they can be safely moved. This simplifies handling of immovable types, including coroutines that emit pointers, enables safe in-place initialization/destruction patterns (via `&out`/`&deinit`), avoids complexities like Rust's `Pin`, and may simplify self-referential types and FFI (TBD). Trade-off: More control for specifying that most things are `Move`.
* **Effects vs. Built-ins**: Silica uses its algebraic effect system as the primary, unified mechanism for handling computational contexts like errors, concurrency, generators, etc., instead of providing separate built-in language features for each. The goal is greater composability and uniformity. Trade-off: Introduces the complexity of the effect system, and libraries may be less ergonomic than dedicated language features.
* **Syntax vs. Convention**: Silica adopts some less conventional syntax, notably postfix `handle` and `match` blocks. This choice aims for potentially greater fluency when chaining operations on a value. Trade-off: Reduced familiarity for programmers accustomed to more standard prefix syntax.

## 4. Implementation State & Plans
* **Parser:** An initial `tree-sitter` grammar has been implemented in `grammar.js`. `parse.rs` accesses the generated parser via the Rust tree-sitter bindings and uses them to build the basic AST structures defined in `ast.rs`. This needs additional testing and integration.
* **AST & Type Checker:** `ast.rs` implements basic AST nodes (literals, if, call, block, lambda, let, assign). It contains a Hindley-Milner based type inference system (`infer`, `unify`) that operates on this AST. During type checking, it **mutates the AST nodes in-place** to store the inferred type information within an `Option<Type>` field associated with each expression node. This approach annotates the AST with types without generating a separate Typed AST structure. It successfully handles basic type checking, variable scoping, and mutability checks for assignments on the current language subset. Lacks AST representation and type checking logic for core Silica features: references, effects/coroutines, aggregates (structs/enums), traits, and explicit generics. `Return` statement handling within blocks is currently unimplemented.
* **Mid Level SSA IR** `ssa.rs` defines basic data structures for an SSA IR (`SsaVar`, `Instruction`, `BasicBlock`, etc.) and includes an SSA validation pass (`typecheck_ssa`) for this basic structure. Lifetime and definite initialization analysis is planned to be implemented at this layer.
* **Implementation Plan**:
  1.  Enhance Parser (`grammar.js`, `parse.rs`) and AST/TypeChecker (`ast.rs`) to support more language features (e.g., structs, basic references).
  2.  Implement Lowering from the type-annotated AST to the SSA/MIR form (`sst.rs`).
  3.  Enhance SSA/MIR representation and implement required analysis passes (lifetimes, definite initialization, effect handling).
  4.  Develop LLVM backend code generation from MIR.
  

## 5. AI Assistant Instructions
Context & Workflow: These instructions are in a copy of the canonical design document. It is maintained externally by the user, in a markdown file, under source control. It was previously in a Google doc. This collaborative session uses separate immersive artifacts (identified by specific IDs listed below) to focus discussion and edits on individual sections. If this session does not contain the immersive artifacts, they may be copied from this file. Prioritize the content within the *most recent versions* of these relevant section artifacts over potentially incomplete or contradictory information from earlier conversation history. Workflow: Edits should be requested for specific sections using their designated IDs. Only the requested section's artifact will be regenerated. The user will integrate changes into their canonical document. Multiple sections may need to be queried to establish context.

Section IDs:
  * `silica_overview` (Section 1)
  * `silica_types_foundations` (Section 2.1)
  * `silica_linearity_ownership` (Section 2.2)
  * `silica_effects_syntax` (Section 2.3)
  * `silica_tradeoffs` (Section 3)
  * `silica_implementation` (Section 4)
  * `silica_ai_assistant_instructions` (This Section)

The primary designer (the user) is the authority on design decisions. The design is iterative.

The target audience for explanations is programmers familiar with Rust/C++ but not necessarily PL theory. Define jargon clearly.

The primary motivation for Silica's extensive use of algebraic effects is *compositionality*.

Key Design Concepts & Decisions Summary:
  * Linearity (exactly-once default), Affinity (`ImplicitDrop`), `ImplicitCopy` (independent of Drop), `Move` trait.
  * `Move` / `Copy` allow user-defined logic (NOT necessarily `memcpy`).
  * Default Immovability. Pass-by-value requires `Move` trait.
  * Reference Types: `&out T`, `&deinit T`, `&T`, `&mut T`. `&deinit T` used for ownership transfer of immovable types.
  * Algebraic Effects: Compositionality focus, Tunneling behavior, all effectful functions return `Co<T!E>` via existential `S`.
  * Lifetimes: Rust NLL-inspired, analyzed on MIR.
  * Type System: HM-hybrid foundation. Inference mutates AST in-place.
  * Implementation Target: MIR for analysis, LLVM backend.
  * Syntax: Postfix `handle`/`match`, `->` pointer access, no implicit deref, `?` operator.
  * `?` Operator: Manages `Co` execution + propagates effects via forwarding `handle`.
  * `handle`: Transfers control; `resume` can be delayed (async).
  * `perform`: Inside handler arm always forwards outwards.

**Core Challenge:** The main complexity lies in the interaction between linearity, lifetimes, effects (including their execution model), immovability, initialization tracking (`&out`/`&deinit`), and generics, particularly within the MIR analysis passes.

**Key Decisions Recap**: Simulate loops/break with effects; postfix handle/match; explicit `resume`; default immovability; linearity relaxed by traits/borrows; polymorphic MIR; definite initialization and lifetime analysis in MIR; existential types for coroutines; `ImplicitDrop` sig: `fn drop(&deinit self);`; Only `Co` return for effectful fns; `?` manages/propagates; `perform` forwards control to the next enclosing handler block; pass-by-value requires `Move`.

**Document Check Best Practices (Checklist for AI)**:
  * Check Document: Ensure the correct immersive document is generated. Incremental edits should be applied to the relevant subsection’s document.
  * Check Target Audience Readability (Rust/C++ Dev): Review explanations for novel concepts (effects, linearity, `&deinit`, immovability) to ensure clarity without requiring deep PL theory. Use analogies. Confirm differences from Rust/C++ are explicit (e.g., non-memcpy Move/Copy, effect handling, stricter move rules).
  * Check Future Gemini Readability: Verify this document provides sufficient context, definitions, key decisions, and TBD points for an AI assistant without prior conversation history to effectively assist.
  * Check for DRYness: Ensure concepts are explained clearly once and not unnecessarily repeated. Ensure concepts are introduced before they are used. Double check that concepts are not repeated between sections.
  * Check Internal Consistency: Verify that design decisions (e.g., `perform` scoping, `?` semantics, trait definitions, move rules) are reflected consistently across all relevant sections, including examples and tips.
  * Check Revision Annotations: When updating, mark changed sections with (Revised Date) and summarize edits. Remove these markers when producing a final export version if requested.
  * Check Heading Levels: Ensure major sections (1, 2, 3...) use `##` and subsections (2.1, 2.2...) use `###` consistently.
  * Check Export Cleanup: Ensure internal status notes or revision markers are removed before generating a final "export" version when requested.

Useful Search Terms: "algebraic effects existential types", "linear types lifetimes", "affine types resource management", "scoped effect handlers", "effect tunneling", "SSA for coroutines", "definite assignment analysis", "immovable types language design", "continuation passing style compilation", "LLVM coroutine intrinsics", "Rust MIR borrow checking", "output reference types", "guaranteed copy elision semantics", "effect handlers driving computation", "asynchronous algebraic effects".