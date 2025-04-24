# Silica Language Design Document

#### Document Purpose and Audience:
This document defines the Silica programming language: its core features, design rationale, implementation state, and trade-offs. It serves as a reference for ongoing design and implementation. The intended audience includes the primary designer, future AI collaborators, and programmers familiar with Rust/C++ but not necessarily PL theory.

## 1. Overview & Philosophy
Silica is envisioned as an experimental systems programming language combining fine-grained low-level control and algebraic effects, which manifest as explicit coroutines. It targets LLVM for efficient native code generation, leveraging effects for compositional design while retaining direct resource control. Foundational goals include strong, compile-time memory and resource safety (with linear types and lifetimes), detailed in subsequent sections. Silica draws inspiration from Ante, C, C++, Haskell, Hylo, Koka, Rust, and Zig.

### 1.1 Core Pillars
Silica's design rests on four core pillars, aiming to provide a unique blend of safety, control, and expressiveness suitable for systems programming:

1. Algebraic Effects: A unified, type-safe framework for managing computational effects (coroutines, async/await, exceptions, state, etc.) as composable library features rather than disparate language constructs. (See 1.2)

2. Linearity, Lifetimes, and Resource Safety: Compile-time guarantees for memory and resource management (files, sockets, locks) without requiring a garbage collector, achieved through linear types, ownership, and borrowing with lifetimes. (See 1.3)

3. Immovability and Fine-Grained Control: Default immovability of types combined with explicit Move semantics and specialized reference types (&out, &deinit) provides precise control over data layout, initialization, and resource handling, crucial for low-level programming and FFI. (See 1.4)

4. Uniformity: Where possible, we want to minimize special compiler and standard library "magic" for standard library types, compared to user-defined types. The effects system aims to achieve this for control flow. The trait based dereferencing and assignment (2.4) aims to achieve this for smart pointers, in contrast with the special status of `Box` in Rust.

### 1.2 Algebraic Effects: Motivation and Overview
A central goal of Silica is to provide a more composable and unified way to handle computational effects compared to traditional approaches. Imperative languages often rely on distinct, built-in mechanisms for error handling (try/catch), asynchronous operations (async/await), and generators (yield). While useful, such constructs cannot be implemented within the language. Functional languages sometimes use abstractions like Monads, which, while powerful, are complex to compose (consider, `Future<Result<Iterator<Item=T>, E>>` versus `Iterator<Item=Future<Result<T, E>>>` and ask whether either, both, or neither properly model a fallible async stream of `T`).

Silica adopts **algebraic effects** (Section 2.3) as its unifying mechanism. For programmers familiar with exceptions, algebraic effects can be initially understood as resumable checked exceptions:
  * **Checked:** Like checked exceptions (or Rust's Result), effects performed by a function must be declared in its signature and handled by callers (or propagated). This prevents "surprise" side effects or unhandled errors.
  * **Resumable:** Unlike standard exceptions which unwind the stack, effect handlers can resume the computation exactly where the effect was performed, potentially passing a value back. This capability is fundamental and require effectful functions to be implemented as coroutines.

This resumption capability unlocks significant expressive power, allowing effects to model a wide spectrum of computational patterns within a single framework:
* **Exceptions:** Naturally, exceptions from other languages can be framed as an effect that resumes exactly zero times.
* **Generators:** Generators are coroutines that resume 0 or 1 time after each yielded item. Generators can potentially be infinite and the option to resume 0 times lets callers consume a finite number of yielded items.
* **Async/Await:** Async tasks yield control whenever they are awaiting something. The executor will resume them exactly one time per suspension, running them to completion.
* **Transactions / Checkpoints:** Resuming from a suspension point multiple times lets you retry computations, possibly with different arguments. This can be used to facilitate retry-able transactions with transient errors. It can also be used for backtracking, e.g. when parsing an ambiguous grammar or searching a tree.

By unifying these patterns, algebraic effects aim to enhance composability (effects form a "flat" set, easier to combine) and modularity (code performing effects can be reused under different handlers implementing different semantics, e.g., synchronous vs. asynchronous execution).

### 1.3 Linearity, Lifetimes, and Resource Safety
Drawing inspiration from Rust and linear logic, Silica aims to provide strong compile-time safety guarantees without relying on a garbage collector, which is crucial for predictable performance in systems programming.

* **Memory Safety:** The combination of linear types (use-exactly-once by default), ownership, borrowing (&'a T, &'a mut T), and lifetime analysis (Rust NLL-inspired) prevents common errors like use-after-free, double-frees, and data races (in concurrent contexts, TBD).

* **Resource Management:** Silica goes beyond Rust’s affine types (use at most once) with linear types (use exactly once). It ensures any resource represented as a linear type (file handles, sockets, mutex guards, etc.) is properly acquired, used according to its rules (e.g., exclusive access), and are also released **exactly** (not at most) once, preventing leaks or logic errors.

### 1.4 Immovability and Fine-Grained Control
Predictable memory layout and explicit control over data movement are often required in systems programming, especially for FFI, performance optimization, or managing self-referential data structures (like certain coroutines or intrusive collections).

* **Default Immovability:** Types in Silica are immovable by default. Their memory location is stable unless they explicitly opt into being movable via the Move trait. This prevents accidental moves and simplifies reasoning about pointers.

* **Custom Move and Copy:** Like Cpp and unlike Rust, Silica allows for custom move and copy logic that are not restricted memcpy. These functions are also not guaranteed to be called if the compiler can optimize away the move or copy.

* **Initialization/Deinitialization references:** Specialized reference types (`&'a out T` for initialization, `&'a deinit T` for deinitialization) allow safe, in-place manipulation of memory, even for immovable types.

This combination provides developers with fine-grained control over data lifecycle and representation when needed, while still benefiting from the high-level safety guarantees of the type and effect systems.

## 2. Core Language Features

### 2.1 Type System Foundations
* Foundation (HM Hybrid): Uses a Hindley-Milner inspired system.
  * Top-level functions require full type annotations (generics, lifetimes, effects).
  * Local type inference (Algorithm W-like) deduces types for let bindings within functions. Inference stops at the function boundary.
* Polymorphism:
  * Universal (forall) types: (forall T: TraitBound...).
    * This allows for standard generic code reusable across different types satisfying specified constraints (traits).
    * Traits define shared interfaces or properties that types can implement (similar to Rust traits or Haskell type classes).
  * Existential (exists) types: (exists S: TraitBound...).
    * This lets generics provide a hidden type S, that only has some defined interface (trait bounds).
    * Primarily used for abstracting types, especially the opaque state/implementation of coroutines (see 2.3).
    * Requires explicit annotations.
    * Practical Explanation: Think of it like returning an abstract "iterator" or "future" where you know what it does (its interface or trait bounds) but not its exact internal type.
* Algebraic Effects: 
  * Algebraic effects is PL Jargon for **resumable checked exceptions**.
    * Effects are like **exceptions**, in that a `perform FooEffect.foo(x)` statement transfers control flow (and a value) to some outer `handle { FooEffect.foo(x) => ... }` block, analogous to how `raise` transfers control to a `catch` block.
    * They are **checked**, in the sense that all effects that a function may perform will be in its type signature and code that does not handle or explicitly propagate a performed effect is considered an error.
    * Unlike most exception systems, effects are **resumable**, in that the `handle` block may transfer control flow (and a value) _back to where `perform` was invoked_.
  * Effects provide a single, type-safe framework to implement exceptions, callbacks, generators, async/await as library features. These would normally be language features.
    * Effects are more simpler to compose than Monads: In Rust, `Future<Result<Iterator<Item=T>, E>>` is very different from `Iterator<Item=Future<Result<T, E>>>.` With effects, they’d both be `Co<() ! Yield<T>, Fail<E>, Async>`. In essence, effects form a "flat" set, while monads have some encapsulation order.
  * In Silica, effectful functions return coroutine objects.
    * The set of effect operations that the coroutine may perform are in the type signature of the coroutine.
    * Functions may be polymorphic over the effects they perform.

### 2.2 Linearity, Ownership, and Movability
Silica incorporate linear types (values consumed exactly once) for compile-time resource safety, preventing leaks and double-use (e.g. in memory or file handles). This strict default is made ergonomic via the following relaxations:
* **References & Borrowing**: Allows temporary, non-consuming access (`&'a T`, `&'a mut T`, `&'a out T`, `&'a deinit T`). Borrowing doesn't consume linear values, but lifetime rules prevent dangling references (Rust NLL-inspired, analyzed on MIR). Definite assignment analysis tracks initialization status for &out/&deinit safety.
  * `&'a T`: Immutable shared borrow.
  * `&'a mut T`: Mutable exclusive borrow (allows deinit/reinit within borrow).
  * `&'a out T`: Output exclusive borrow (must be initialized).
  * `&'a deinit T`: Deinitialization exclusive borrow (must be deinitialized).

* **Opt-in implicits**: The following traits further relax linearity rules by controlling implicit destruction and copying:
  * **ImplicitDrop (Affine):** Allows values to be used at most once . If unused by scope end, its drop method is called automatically. Enables affinity.
      ```
      trait ImplicitDrop {
        // Non-effectful cleanup logic.
        fn drop(&deinit self);
      }
      ```
      *Note:* `drop` cannot perform effects. Effectful cleanup requires an explicit function like `destroy(&deinit self) -> Co<() ! E>`.
  * **ImplicitCopy (Relevant):** Allows values to be used one or more times (relevant) or any number of times (unrestricted, if also ImplicitDrop). Implicit copies are made via the copy method when needed.
      ```
      trait Copy {
        // Performs a copy. Allows user-defined logic (not just memcpy).
        fn copy(&self, &out other: Self);
      }
      // Allows the compiler to call `copy` implicitly.
      trait ImplicitCopy : Copy {}
      ```
  * **ImplicitCopy + ImplicitDrop (Unrestricted):** Types that are both implicit copy and implicit drop may be used any number of times. 

* **Movability**: Types are immovable by default.
  * Stable addresses are required for values that are pointed-to from outside of the language, e.g. Mutex. First class support avoids complexity like Rust's `Pin`.
  * Safe in-place init/deinit via `&out`/`&deinit`.
  * Requires opt-in via the Move trait for types that can be safely moved (potentially via user-defined logic).
      ```
      trait Move {
        // Performs a move, consuming `self`. Allows user-defined logic (not just memcpy).
        // The `&out other` parameter receives the moved-to value.
        fn move(&deinit self, &out other: Self);
      }
      ```
  * **Pass-by-value requires the `Move` trait.** Attempting to pass a type that does *not* implement `Move` by value results in a **compile-time error**. This enforces the default immovability of types.
  * For types that *do* implement `Move`, pass-by-value transfers ownership to the callee. The `Move::move` operation consumes the source (`&deinit self`) and initializes the destination (`&out other`).

  * **Move/Copy Elision:** For types implementing `Move` (or `ImplicitCopy`), the compiler *may* optimize away the actual call to the `move` (or `copy`) method, directly constructing the value in the destination or reusing the source memory location, similar to Guaranteed RVO/Copy Elision in C++. Code defined in `move` and `copy` methods are not guaranteed to execute.

  * **Coroutines and Movability:**  Coroutines that allow references to their own state (even self-referentially) will generally not implement `Move` or `Copy` for safety reasons. To store or transfer such immovable coroutines (e.g., in async executors), they can be placed on the heap using a movable Box type (e.g., `Box<Co<T!E>>`, as `Box` implements `Move`).

* To transfer ownership *responsibility* (specifically, the obligation to deinitialize) for an **immovable type** without moving its memory, pass it via an exclusive **`&deinit T`** reference. The caller retains the storage allocation, but the callee is responsible for deinitialization.

* It is generally best practice to mark types as `Move`, `ImplicitDrop` and `Copy` where possible and semantically correct.

*  Aggregate types (structs/enums/closures/coroutines) can only implement `Move`, `ImplicitDrop` and `Copy` if all their members (and captured state) have these properties.

* **Deriving Defaults:** Mechanisms to automatically derive default implementations for `Move`, `ImplicitCopy`, `ImplicitDrop`, and `Copy` for aggregate types will be necessary for ergonomics.

### 2.3 Effects and Coroutines
* **Core Idea:** The compiler tracks the set of operations that a function may perform. If the set is non-empty, the function will return a coroutine object. When run, any performed operations yield control and may subsequently be resumed.
* **Effect Definitions:** Operations are defined in an `effect` block. Each operation has an identifier, a _perform type_ and a _resumption type_.
    ```silica
    effect FooService {
      get_config: () -> &ServiceConfig;
      get_user: UserId -> User;
      update_user: User -> ();
      internal_error: String -> !;  // `!` or "Never" is the empty type. This op cannot be resumed.
    }
    ```
  * Effect operation signatures can involve types containing lifetimes tied to the coroutine's execution context. A key safety rule is that references derived from the coroutine's internal state have limited lifetimes bounded by suspension points [^lifetime_note].

  [^lifetime_note]: Note on Coroutine Reference Lifetimes: References derived from a coroutine's internal state (e.g., returned via `perform`) are generally invalidated when the corresponding `resume` is invoked or the next suspension occurs. Conceptually, this can be modeled using Stacked Borrows: the resume capability holds a temporary borrow of the coroutine state, and the performed references are derived from it, and calling resume consumes the capability's borrow, invalidating derived references, particularly the performed reference. Enforcing this requires compiler support.

* **Performing Effects:** Operations are invoked via `perform Effect.operation(...)` syntax. Type parameters for generic effects (`S` in `State<S>`) are typically inferred from usage context.
    ```silica
    // Conceptual perform syntax
    let count = perform State.get();
    perform Log.info("Got count");
    ```
  * **Named effects:** If a function performs operations from multiple instance of the same polymorphic effect, the effect must be named.
    * Since we don't specify types at `perform` or `handle` sites, the effects must be named. 
    * E.g. `perform my_int_state.get(())` or `perform string_state.set(x)`.
  * **Declararation-free operations:** Operations may also be defined inline, without the `effect` declaration. 
    * E.g., the expression `y = perform my_new_operation(x)` defines an operation called `my_new_operation` whose perform type `typeof(x)` and resume type `typeof(y)` are inferred.
    * Such operations unify by name and perform/resume type.
    * Declaration free operations are useful for oneoff operations that aren't used in enough places to warrant an `effect declaration. The tradeoff is analogous to using tuples vs predefined structs.
* **Propagating Effects (`?` Operator):**
    * `expr?` operates on a Coroutine value yielded by `expr`.
    * It runs the coroutine. If the coroutine performs an effect `E_op`, `?` propagates `E_op` outwards.
      * This requires `E_op` to be in the current function scope's operation set.
      * If the coroutine eventually completes with value `T`, then `expr?` evaluates to `T`.
* **Coroutine Values:** Functions whose execution involves `perform` (or `?`) yield control, so they return a **Coroutine** value. This value encapsulates the suspended computation, its eventual result type (`T`), and the set of effects (`E`) it might perform when resumed.
* **Handling Effects (`handle` Block):**
    * A unified handler syntax intercepts and provides semantics for specific operations.
    * **Conceptual Syntax:** `coroutine_expr handle { InitialClause?, ReturnClause?, OpClause*?, FinallyClause? }`
    * `OpClause` form: `Effect.Operation(args), resume => { /* handler body */ }`. It matches a specific performed operation.
    * `resume(value)`: Resumes the original computation, passing `value` back as the result of the `perform`. `value` must match the operation's resumption type.
    * The `InitialClause`, e.g. `initially => ...` is executed before the coroutine is run. The other clauses may reference state from this clause, but not from each other.
    * The `ReturnClause`, e.g. `return x => ... ` is executed if the coroutine finishes and returns its final value. Guaranteed move elision is desirable for simple returns like `Return(v) => v`, which is the default behavior if unspecified.
    * The `FinallyClause`, e.g. `finally => ...`, this code is executed if control leaves this `handle` block, never to return, due to a propaged effect operation that is not resumed. The coroutine this `handle` block evalutes to will have a `destroy` method that will run this code.
    * **resume:** Within the handler arm, `resume` represents the unique, linear continuation capability (FnOnce) for resuming from this specific suspension point. Calling `resume(value)` consumes the `resume` and passes `value` back to the perform site, and transfers control away from the handler arm. Code immediately following a `resume()` call within the same block is unreachable.
    * **resume.into():** For scenarios like async executors needing to store and move the continuation, `resume.into()` can be used. This consumes the resume capability and yields a new, _movable_ object (conceptually `impl FnOnce(ResumeType) -> Co<T!E>`) that owns the continuation state. This requires the underlying coroutine state to implement `Move`. Calling the resulting object later will run the moved coroutine from the current suspension point.
    * **Evaluation** A `expr handle {...}`  block, that does not handle all effects from the `expr` coroutine, evaluates to a coroutine. If all effects are handled, then it evaluates to `expr`'s return type.
* **Effect Tunneling:** Functions polymorphic over an effect parameter `E` cannot handle operations in `E`, even if those operations happen to collide with operations used in the polymoprphic function body (otherwise, the implementation of the generic function is revealed to the caller). Such operations "tunnel through" and are propagated outwards as part of the function's `E` effect parameter, bypassing local handlers for the operation. Concrete effects not matching `E` are handled normally.

### 2.4 Access, Calls, and Assignment (UFCS)
Silica aims for a uniform and explicit system for accessing members, calling methods, and handling assignment, especially concerning pointer-like types and custom smart pointers.

* **Uniform Function Call Syntax (UFCS):**
  * Method calls use the dot operator: `receiver.method(args...)`.
  * This syntax desugars to a standard function call: `method(receiver, args...)`.
* Method resolution first checks for fields named method on the receiver's type. If no field exists, it searches for applicable functions (standalone or trait methods) named method that accept receiver as the first argument.
* **Postfix Dereference Operators:** To provide explicit control over accessing the value or place underlying a pointer-like type `P<T>`, Silica uses postfix dereference operators defined by traits:
  ```
  trait Deref<'a> { type Target: 'a; fn deref(&'a self) -> Self::Target; }
  trait DerefMut<'a>: Deref<'a> { type TargetMut: 'a; fn deref_mut(&'a mut self) -> Self::TargetMut; }
  trait DerefOut<'a> { type TargetOut: 'a; fn deref_out(&'a mut self) -> Self::TargetOut; }
  trait DerefDeinit<'a> { type TargetDeinit: 'a; fn deref_deinit(&'a deinit self) -> Self::TargetDeinit; }
  ```
  * These traits are invoked by `ptr.*`, `ptr.*mut`, `ptr.*out`, and `ptr.*deinit` respectively.
  * These traits allow custom smart pointers and proxy types to define how they provide access to the underlying data or place. The associated types (`Target`, `TargetMut`, `TargetOut`, `TargetDeinit`) are not restricted to primitive references.
    * **Effectful Dereference:** These dereference operations themselves can be effectful (return `Co<TargetType ! E>`). The `?` operator chains naturally: `ptr.*?` or `ptr.*mut?`.
  * **Note on moving the referent:** `ptr.*deinit`, gives you a `&'deinit` reference or an equivalent proxy and ensures `ptr` points to uninitialized memory after the borrow, but this doesn't exactly give you the moved value. A further call to `Move::move`, or some other method, is required to move the referent to a new location.
  
* **Assignment (`=`):** The assignment operator `=` has distinct semantics based on the initialization state of the left-hand side (LHS), determined by definite assignment analysis. It is not overloadable via traits; custom update logic requires explicit methods.

  * **Initialization:** `lhs = rhs` where `lhs` is uninitialized performs in-place initialization.
    * ` lhs` represents the uninitialized place (e.g., an uninitialized variable or the result of `ptr.*out`).
    * If `rhs` is a constructor call or literal (e.g., `Foo { ... }`), the compiler generates code to initialize the fields of lhs directly using the rhs initializers, without creating a temporary rhs object.
    * If `rhs` is an existing value, standard move/copy semantics (potentially using `Move::move` or `ImplicitCopy::copy`) are used to initialize lhs from rhs.
    * The compiler marks `lhs` as initialized.
    
  * **Re-assignment:** `lhs = rhs` where `lhs` is already initialized:
    1. If `lhs`'s type implements `ImplicitDrop`, the existing value in lhs is dropped with `ImplicitDrop::drop`. If lhs's type is strictly linear (no `ImplicitDrop`), re-assignment is a compile-time error as the existing value cannot be implicitly discarded.
    2. The value `rhs` is moved or copied into the lhs location.

This system allows custom smart pointers to integrate cleanly with method calls (via UFCS and dereference traits) and initialization (via DerefOut and compiler-driven initialization), while keeping the semantics of re-assignment (=) simple and predictable.

### 2.5 `defer` blocks
* Functions and coroutines may place `defer { ... }` statements which execute when control permanently leaves this activation frame. This includes when the function or coroutine returns, or when a suspended coroutine is explicitly dropped.
  * The intended use is for potentially effectful cleanup of linear resources.
* `defer` blocks _may not_ be placed in the `OpClause` of `handle` blocks. At least until "code after resume" (5.2) is supported. 

## 3. Design Rationale and Open Questions
This section summarizes key design choices where Silica prioritizes certain goals over others, accepting the associated trade-offs.

## 3.1 Accepted Trade offs
This subsection explains the _why_ behind major finalized design choices, clarifying the guiding principles and accepted consequences.

* **Linearity vs. Flexibility**: Silica prioritizes strict linearity by default for maximum resource safety (exactly-once usage). Flexibility and ergonomics are then recovered through opt-in relaxations (discussed above). Trade-off: Requires explicit opt-in for common affine/unrestricted patterns vs. Silica’s linear-by-default.
* **Immovability vs. Ergonomics**: Silica chooses immovability by default, requiring types to opt into the `Move` trait if they can be safely moved. This simplifies handling of immovable types, including coroutines that emit pointers, enables safe in-place initialization/destruction patterns (via `&out`/`&deinit`), avoids complexities like Rust's `Pin`, and may simplify self-referential types and FFI (TBD). Trade-off: More control for specifying that most things are `Move`.
* **Effects vs. Built-ins**: Silica uses its algebraic effect system as the primary, unified mechanism for handling computational contexts like errors, concurrency, generators, etc., instead of providing separate built-in language features for each. The goal is greater composability and uniformity. Trade-off: Introduces the complexity of the effect system, and libraries may be less ergonomic than dedicated language features.
* **Syntax vs. Convention**: Silica adopts some less conventional syntax, notably postfix `handle` and `match` blocks. This choice aims for potentially greater fluency when chaining operations on a value. Trade-off: Reduced familiarity for programmers accustomed to more standard prefix syntax.
* **Explicitness vs. Ergonomics**: Silica's decision to use explicit postfix dereference operators (`.*`, `.*mut`, etc.) and desugar assignment to specific methods (`assign`/`reassign`) prioritizes explicitness and control over the potential ergonomic convenience of more implicit systems (like Rust's overloaded dot operator or C++'s operator=). Trade-off: More verbose for some common operations but clearer semantics and potentially easier reasoning.
* **Generality vs. Compiler Complexity**: Making fundamental operations like dereferencing and assignment extensible via traits (to support custom proxies and smart pointers uniformly) adds generality but increases compiler complexity. The compiler needs robust trait resolution and deep integration with analyses like definite assignment. Trade-off: More powerful user-defined types vs. increased compiler implementation effort.

### 3.2 Open Questions
This subsection outlines areas where the design is still evolving, involves known difficult interactions, or requires further investigation and decision-making.

* **ImplicitDrop vs. Effectful Deallocation:**
  * **Tension:** The current design defines `ImplicitDrop::drop` as non-effectful (to potentially enable ASAP destruction and TCO), but many resources (like `malloc` and `free` from C) require effectful cleanup.
  * **Current Status:** Requires types needing effectful cleanup (like Box) to rely on linearity and an explicit, effectful destroy method, rather than automatic ImplicitDrop.
  * **Open Question:** Is this separation optimal? Does it place too much burden on the programmer compared to call destructors that can have side effects? Should `drop` be allowed to perform effects, potentially allowing for  benefits?

* **Type-State Analysis / Flow-Sensitive Typing for Initialization:**
  * **Tension:** Achieving efficient and ergonomic initialization patterns (like `empty_box.*out = value;` implicitly transitioning `empty_box`'s type from `EmptyBox` to `Box`) without compiler magic specific to `Box` is desirable. One approach involves explicit conversion (`unsafe assume initialized`). An alternative involves tracking the state with flow-sensitive typing and allowing the variable's type to change after initialization.
  * **Current Status:** True flow-sensitive type mutation significantly complicates the planned HM-hybrid type system foundation. However, the ergonomics of the type-state annotation approach are appealing.
  * **Open Question:** Can a limited form of flow-sensitive type-state tracking for initialization be integrated cleanly into the HM-hybrid system without undue complexity? Or is the explicit conversion function the most pragmatic path despite slightly increased verbosity?

* **Allocation/Deallocation as Effects:**
  * **Tension:** Should fundamental operations like heap memory allocation and deallocation be modeled as algebraic effects? Modeling them as effects makes their usage explicit and integrates them into the handler system (allowing custom allocators via handlers) but means types like `Box` cannot use `ImplicitDrop` for deallocation. Treating them as non-effectful built-ins simplifies `ImplicitDrop` for `Box` but makes allocation less explicit and harder to abstract over. `malloc` and `free` are trivially resumptive (always resume exactly once) which does not harness the power of algebraic effects (resuming multiple times, after a long suspension, or not at all). `perform/resume` may also be expensive for such common operations.
  * **Current Status:** Assumed to be effects in some examples, but not definitively decided. Leaning towards "not an effect".
  * **Open Question:** What is the best way to model allocation/deallocation to balance explicitness, composability, performance, and integration with resource management traits?

* **ASAP Destruction vs. RAII:**
  * **Tension:** Should automatic cleanup for affine types (`ImplicitDrop`) occur immediately after the last use (ASAP) or at the end of the lexical scope (RAII)?
  * **Current Status:** Undecided. ASAP enables TCO (given non-effectful drop) and earlier resource release but requires complex compiler analysis and changes the programmer's mental model slightly. RAII is simpler conceptually and lexically predictable but hinders TCO and can hold resources longer. `defer` seems like `RAII`.
  * **Open Question:** Which model best fits Silica's goals and implementation constraints?

* **Object Safety & Effects:**
  * **Tension:** How should traits with effectful methods interact with dynamic dispatch (dyn Trait)?
  * **Current Status:** A basic strategy is outlined (effects must be declared in the trait, trait objects carry the union of effects), but the precise semantics, especially regarding effect polymorphism and potential dynamic effect dispatch, need detailed specification.
  **Open Question:** What are the exact rules and limitations for object safety concerning various effect patterns (polymorphism, labels, etc.)?

* **Lifetimes Across perform/resume:**
  * **Tension:** Ensuring the safety of references held across suspension points is complex, even with explicit ? markers.
  * **Current Status:** The Stacked Borrows model provides a conceptual basis, and the compiler's MIR borrow checker is expected to enforce safety.
  * **Open Question:** What are the precise, formal rules the borrow checker will use? How will complex handler interactions or effect polymorphism affect this analysis?

## 4. Implementation State & Plans

## 4.1 Immediate state and Immediate next steps
* **Parser:** An initial `tree-sitter` grammar has been implemented in `grammar.js`. `parse.rs` accesses the generated parser via the Rust tree-sitter bindings and uses them to build the basic AST structures defined in `ast.rs`. This needs additional testing and integration.
* **AST & Type Checker:** `ast.rs` implements basic AST nodes (literals, if, call, block, lambda, let, assign, structs). It contains a Hindley-Milner based type inference system (`infer`, `unify`).
  * During type checking, it **mutates the AST nodes in-place** to store the inferred type information within an `Type` field associated with each expression node.
  * Efficient type unification is achieved using interior mutability and shared pointers.
  * It successfully handles basic type checking, variable scoping, and mutability checks for assignments on the current language subset. Lacks AST representation and type checking logic for core Silica features: references, effects/coroutines, enums, traits, and explicit generics.
  * Next step: Supporting effects.
    * `TypeContext` currently supports tracking the current function's return type. We'll rename this to the `final_return_type` for disambiguation. A similar approach can be used to track which operations may be introduced or propagated from the function. If a function has a non-empty op-set the return type must be a coroutine like `Co<final_return_type ! op-set >`. If the opset is empty, the function then `return_type = final_return_type`.
    * The other situation where a coroutine may appear are after `handle` blocks. This seems complex, so the implementation should start with effect declarations and `?` before working on `handle` blocks.
* **Mid Level SSA IR**
  * `ssa.rs` defines basic data structures for an SSA IR (`SsaVar`, `Instruction`, `BasicBlock`, etc.) and includes an SSA validation pass (`typecheck_ssa`) for this basic structure.
  * Development on this file is paused until the type checker supports effects and references.
  * Lifetime and definite initialization analysis is planned to be implemented at this layer.
  * Generic functions are preserved in the MIR for analysis before potential monomorphization in the LLVM backend.
* **Implementation Plan**:
  1.  Enhance Parser (`grammar.js`, `parse.rs`) and AST/TypeChecker (`ast.rs`) to support more language features (e.g., structs, basic references).
      1.  The parser does not yet support structs, through the AST/Typechecker does.
      2.  References and Effects are the likely next step for the type checker
  2.  Implement Lowering from the type-annotated AST to the SSA/MIR form (`sst.rs`).
      1.  Implement last use inference
      2.  Implement lifetime infernece and checking
      3.  Develop LLVM backend code generation from MIR.


## 4.2 LLVM Coroutine Backend Strategy
* Silica targets LLVM for code generation. To implement algebraic effects (which manifest as coroutines), Silica will leverage LLVM's built-in coroutine support.

* **Chosen Backend:** The implementation will use the standard, mature LLVM state machine lowering strategy (switched-resume model). This involves transforming effectful Silica functions into state machines using LLVM intrinsics (`@llvm.coro.begin`, `@llvm.coro.suspend`, `@llvm.coro.resume`, `@llvm.coro.destroy`, etc.) and associated optimization passes (`coro-split`, `coro-elide`, `coro-cleanup`).

* **Rationale:** This backend is well-supported, actively developed, benefits from dedicated optimization passes (especially Heap Allocation Elision via `coro-elide`), and is the same foundation used by C++ and Rust's coroutine/async implementations. Alternative LLVM strategies (like returned-continuations) are considered less mature, less optimized, and potentially unstable.

* **Frame Allocation:** Coroutine state machine frames will aim for stack allocation by default (leveraging `coro-elide`). Heap allocation via `Box<Co<T!E>>` will be required for recursive coroutines or when storing coroutines polymorphically (e.g., in executor task queues).

**Performance Considerations:**
* The overhead of suspension/resumption (state save/restore, indirect calls) is acknowledged.
* Heap Allocation Elision (HAO) is critical for performance but known to be fragile; generating HAO-friendly LLVM IR will be important.
* Link-Time Optimization (LTO) will be important for optimizing across coroutine/handler boundaries.

**Trivial Resumption Optimization (TRO):**
* To mitigate overhead for simple, synchronous-like effects, "trivial resumption" optimizations are planned. An operation is considered _trivially resumed_ if each `perform` is resumed exactly once, without copying or moving the coroutine. These do not require suspension of the coroutine, the handle logic may be implemented as a function called inside of the coroutine.
  * **Front End Annotation** Operations marked appropriately (e.g., #[TriviallyResumed], TBD) may cause non-trivial handling to be a front end compile error.
  * Whole program analysis may also determine that an operation is always trivially resumed, enabling suspension point elision during LTO. If the handler can be statically determined, we could elide passing of the function pointer and perform a direct call at the `perform` site. 

## 4.2 Concurrency Model: Structured Concurrency
Silica will adopt structured concurrency as its primary model for managing concurrent tasks, leveraging the algebraic effect system for its implementation.

* **Core Principle:** The lifetime of concurrent tasks ("children") is strictly bound by a lexical scope. A scope cannot exit until all tasks spawned within it have terminated. This prevents "orphan" tasks and ensures predictable resource management. This model aligns exceptionally well with Silica's features:
  * **Lifetimes:** Parent tasks waiting for children guarantees that data borrowed from the parent scope by child tasks remains valid for the children's entire lifetime, eliminating a major class of concurrency bugs.
  * **Linearity:** Ensures resources passed to or used by concurrent tasks are consumed correctly according to their type rules.
  * **Effects/Handlers:** The concurrency scope is defined naturally using an effect handler. Operations like spawning tasks (`perform Concurrent.spawn(...)`) are handled within this scope.
* **Cancellation:**
  * Cancellation is cooperative and initiated by the parent scope/handler.
  * Checks for cancellation occur implicitly at concurrency-related suspension points within child tasks (e.g., when performing scheduler yields, channel operations, or potentially specific effect operations designated as cancellation points).
    * Upon detecting cancellation, the parent calls the child coroutine's explicit `destroy` method (see Section 2.3) is called by the concurrency handler to ensure graceful consumption of linear resources (effects performed during destroy/defer unwinding must be resumable).  

## 5. Alternatives Considered (Deferred or Rejected)

This section documents significant features or design choices that were considered but ultimately deferred or rejected for the current iteration of Silica, along with the rationale.

### 5.1 Polymorphic Closures and `let` Generalization

* **Alternative Considered:** Implementing Hindley-Milner style `let` polymorphism (where `let bound_fn = |x| ...` can be automatically generalized to a polymorphic type like `forall T...`) and/or full higher-rank polymorphism allowing polymorphic closures (`f: forall T. Fn(T) -> T`) to be passed as arguments.

* **Potential Benefits:**
    * Increased local abstraction and code reuse (defining polymorphic helpers locally via `let`).
    * Greater expressiveness, enabling certain functional programming patterns more concisely.

* **Reasons for Deferral / Why Not (Currently):**
    * **Implementation Complexity:** Implementing generalization and/or higher-rank types complicates the type inference engine (`ast.rs::infer`, `unify`) and the `Type` representation (requiring type schemes, potentially complex instantiation logic).
    * **Performance/Monomorphization:** While local `let` polymorphism is compatible with monomorphization, implementing higher-rank polymorphism (especially for function arguments) efficiently via pure monomorphization is challenging. It often requires complex compiler optimizations or risks needing runtime mechanisms like dictionary passing or boxing, which conflicts with Silica's performance goals and explicit control philosophy.
    * **Focus on Core Goals:** The primary initial focus for Silica is exploring the interaction of algebraic effects, linearity/lifetimes, and low-level control. The added expressiveness of these advanced polymorphism features is considered secondary for now.
    * **Alignment with Rust/C++:** Standard Rust/C++ do not support these forms of polymorphism for local functions/closures. Sticking to explicit top-level generics keeps the core type system simpler and potentially more familiar to the target audience.

* **Decision:** Defer implementation of `let` generalization and higher-rank polymorphism for closures. Polymorphism will initially be supported only via explicitly annotated generic parameters on top-level function declarations (`fn foo<T>(...)`). This decision prioritizes focusing on effects and lifetimes and keeps the initial type system complexity manageable. This may be revisited later.

### 5.2 Code After resume in Handlers
* **Alternative Considered:** Allowing handler code to execute after a `resume()` call returns control to the handler arm.

* **Potential Benefits:** Allows certain handler patterns (e.g., wrapping `resume` with logging or resource management) more directly.

* **Reasons for Deferral:** Significant implementation complexity:
  * Requires preserving handler execution context across resumptions
  * Introduces potential for stack overflow issues in common effect loops as `handle` becomes recursive and reentrant.

* **Decision:** Defer. `resume(...)` transfers control away from the handler arm, making code after it unreachable. If the coroutine suspends again, control flow enters from the start of a handler arm.

### 5.3 Arrow Syntax for UFCS Sugar (`->`, `->mut`, etc.)
* **Alternative Considered:** Using C++/Rust-like arrow syntax (`ptr->method()`, `ptr->mut method()`) as sugar for UFCS calls involving dereferencing.

* **Reasons for Rejection:** While familiar, this syntax chains poorly with multiple indirections `((*p)->method())` and does not compose cleanly with the `?` operator for effectful/fallible dereferencing (`method((*ptr)?` is less ergonomic than `ptr.*.?.method()`). Decision: Adopt postfix dereference operators (`.*`, `.*mut`, etc.) combined with the standard dot (`.`) for UFCS calls for better consistency and composability with effects.

### Custom `=` Assign, Reassign, Setters
* **Alternative Considered:** Desugaring the assignment operator (`=`) to methods defined by traits (`Assign` for initialization, `Reassign` for updates), allowing user-defined types/proxies to customize assignment behavior.
* **Reasons for Rejection:** While enabling custom logic for proxies, this approach adds potential "magic" or implicit conversion feel to the fundamental assignment operator.
* **Decision:** Keep assignment (`=`) as a built-in operation with fixed semantics (compiler-driven in-place initialization for uninitialized LHS using literals/constructors, or standard drop-then-move/copy for initialized LHS). Custom update logic requires explicit method calls.


### 5.4 Interconnected Value Systems (Limited Self-Reference):
* **Alternative Considered** Silica may support a pattern where functions return or accept arguments representing a temporary "system" of values with interconnected lifetimes. This simulates limited, safe self-reference, primarily for initialization or temporary access, without requiring a general 'self lifetime feature.
* **Mechanism:** This involves passing or returning an aggregate (like a struct with named fields, or a tuple) containing an immovable container type along with one or more references borrowing from it. The type signature must express the components and their interconnected lifetimes (potentially via annotations like `(borrowed_mut('a) Box<T>, &'a out T)` - syntax TBD). Guaranteed RVO ensures correct memory placement when returning such systems from functions without invalidating internal references established during creation.
  * **Properties & Usage:** Such an interconnected system is inherently Immovable (as a unit) and Linear (must be consumed). The caller typically destructures it immediately (`let (container, borrow) = ...;`) or passes it via `&deinit`. The borrow checker, understanding the type signature's constraints, enforces that the container part cannot be used improperly while the borrow part is live. Once the borrow ends, the restriction on the container is lifted.
  * **Limitations:** Due to the internal borrowing relationships established at creation, obtaining a general mutable reference (`&mut`) to the entire system is disallowed, as it could invalidate internal references. Similarly, `&out` or `&deinit` references to the system cannot decay to `&mut`. Only explicitly destroying/destructuring the system (via `&deinit` or consuming moves) is permitted after its constituent borrows expire. This pattern provides safety for specific initialization/temporary access scenarios but does not enable general-purpose, persistent, mutable self-referential struct types without unsafe.
* **Reasons for Deferral:** Seems to be of limited use. I can only think of the `fn EmptyBox::fill(self) -> (borrowed_mut('a) Box<T>, &'a out T)` use case. Maybe first class type-states can model this use case better, while also being useful in other ways.

### 5.5 Overloadable Assignment (Assign/Reassign Traits)
* **Alternative Considered:** Desugaring the assignment operator (=) to methods defined by traits (Assign for initialization, Reassign for updates), allowing user-defined types/proxies to customize assignment behavior.
* **Reasons for Rejection:** While enabling custom logic for proxies, this approach complicates type inference (trait resolution depends on both LHS and RHS types) and adds potential "magic" or implicit conversion feel to the fundamental assignment operator. Decision: Keep assignment (=) as a built-in operation with fixed semantics (compiler-driven in-place initialization for uninitialized LHS using literals/constructors, or standard drop-then-move/copy for initialized LHS). Custom update logic requires explicit method calls.

## 6. AI Assistant Instructions
* **Goal:** Assist in designing and implementing Silica. Be curious, ask clarifying questions, offer constructive criticism, identify inconsistencies, and propose solutions/alternatives based on PL theory and practice (especially Rust, Haskell, ML, Koka influences). The primary designer (the user) is the authority on design decisions. The design is iterative. Explain complex concepts succinctly. Prioritize consistent forwards momentum.
* **Style:** Collaborative, inquisitive, detailed, encouraging. Reference specific sections/decisions when relevant.
* **Assume:** The author is an expert in Rust/C++ and is familiar with PL concepts (types, inference, scope, linear types, algebraic effects), but not necessarily deep theory (e.g., category theory, advanced dependent types).

Key Design Concepts & Decisions Summary:
  * Linearity (exactly-once default), Affinity (`ImplicitDrop`), `ImplicitCopy` (independent of Drop), `Move` trait.
  * `Move` / `Copy` allow user-defined logic (NOT necessarily `memcpy`).
  * Default Immovability. Pass-by-value requires `Move` trait.
  * Reference Types: `&out T`, `&deinit T`, `&T`, `&mut T`. `&deinit T` used for ownership transfer of immovable types.
  * Algebraic Effects: Compositionality focus, Tunneling behavior, all effectful functions return `Co<T!E>` via existential `S`.
  * resume capability (`FnOnce`), `resume.into()` creates movable FnOnce (requires Move on state).
  * Lifetimes: Rust NLL-inspired, analyzed on MIR.
  * Type System: HM-hybrid foundation. Inference mutates AST in-place.
  * Implementation Target: MIR for analysis, LLVM backend.
  * Syntax: Postfix `handle`/`match`, `->` pointer access, no implicit deref, `?` operator.
  * `?` Operator: Manages `Co` execution + propagates effects via forwarding `handle`.
  * `handle`: Transfers control; `resume` can be delayed (async).
  * `perform`: Inside handler arm always forwards outwards.
  * pass-by-value requires `Move`
  * efficient unification via interior mutability/Union-Find.
  * Uniformity for smart pointers (avoid Rust's special Box/Rc/Arc behaviors)
  * Postfix dereference operators (`.*`, `.*mut`, `.*out`, `.*deinit`) via traits
  * Assignment (`=`): Built-in operation. Performs in-place initialization if LHS uninitialized (compiler handles literals/constructors). Performs drop+move/copy if LHS initialized (error if LHS is linear). Not overloadable via traits; custom updates require explicit methods.
  * Guaranteed Cleanup: Function-scoped defer (LIFO) planned (implementation deferred). Runs on function termination or coroutine destruction. Incompatible with TCO. Effects in `defer` must be resumable to ensure linearity.

* **Checklist Integration:** When asked to review or regenerate, mentally use the checklist below.
* **Checklist:** (For AI use when reviewing/generating)
  * Check for Clarity: Ensure explanations are understandable to the target audience. Define terms. Provide rationale. Use examples.
  * Check Target Audience Readability (Rust/C++ Dev): Review explanations for novel concepts (effects, linearity, `&deinit`, immovability) to ensure clarity without requiring deep PL theory. Use analogies. Confirm differences from Rust/C++ are explicit.
    * Crucially, when motivating algebraic effects, avoid examples like State or IO which are often seen as trivial non-issues in imperative languages. Focus on demonstrating value through patterns like generators, async/await, exceptions, transactions/retries, etc.
    * Be mindful when explaining the postfix dereference and assignment desugaring, as these differ significantly from C++/Rust conventions and require clear justification.
  * Check each section meets its goals: (Core Language: final state; Trade-offs: priorities; Implementation: current state + plan; Alternatives: rejected or deferred ideas; AI Instructions: guidance).
  * Check for Completeness: Ensure major features covered. Capture assumptions, definitions, decisions, TBD points for future AI context.
  * Check for DRYness: Ensure concepts explained once, introduced before use, no major repetition between sections.
  * Check Internal Consistency: Verify that design decisions (e.g., `perform` scoping, `?` semantics, trait definitions, move rules) are reflected consistently across all relevant sections, including examples and tips.
  * Check Revision Annotations: When updating, mark changed sections with (Revised Date) and summarize edits. Remove these markers when producing a final export version if requested.
  * Check Heading Levels: Ensure `##` for major, `###` for subsections.
  * Check Export Cleanup: Ensure internal status notes or revision markers are removed before generating a final "export" version when requested.

Useful Search Terms: "algebraic effects existential types", "linear types lifetimes", "affine types resource management", "scoped effect handlers", "effect tunneling", "SSA for coroutines", "definite assignment analysis", "immovable types language design", "continuation passing style compilation", "LLVM coroutine intrinsics", "Rust MIR borrow checking", "output reference types", "guaranteed copy elision semantics", "effect handlers driving computation", "asynchronous algebraic effects".