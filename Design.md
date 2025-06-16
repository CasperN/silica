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

4. Uniformity: Where possible, we want to minimize special compiler and standard library "magic" for standard library types, compared to user-defined types. The effects system aims to achieve this for control flow.

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
* **Polymorphism:**
  * Universal (forall) types: (forall T: TraitBound...).
    * This allows for standard generic code reusable across different types satisfying specified constraints (traits).
    * Traits define shared interfaces or properties that types can implement (similar to Rust traits or Haskell type classes).
  * Existential (exists) types: (exists S: TraitBound...).
    * This lets generics provide a hidden type S, that only has some defined interface (trait bounds).
    * Primarily used for abstracting types, especially the opaque state/implementation of coroutines (see 2.3).
    * Requires explicit annotations.
    * Practical Explanation: Think of it like returning an abstract "iterator" or "future" where you know what it does (its interface or trait bounds) but not its exact internal type.
* **Algebraic Effects:** 
  * Algebraic effects is PL Jargon for **resumable checked exceptions**.
    * Effects are like **exceptions**, in that a `perform FooEffect.foo(x)` statement transfers control flow (and a value) to some outer `handle { FooEffect.foo(x) => ... }` block, analogous to how `raise` transfers control to a `catch` block.
    * They are **checked**, in the sense that all effects that a function may perform will be in its type signature and code that does not handle or explicitly propagate a performed effect is considered an error.
    * Unlike most exception systems, effects are **resumable**, in that the `handle` block may transfer control flow (and a value) _back to where `perform` was invoked_.
  * Effects provide a single, type-safe framework to implement exceptions, callbacks, generators, async/await as library features. These would normally be language features.
    * Effects are more simpler to compose than Monads: In Rust, `Future<Result<Iterator<Item=T>, E>>` is very different from `Iterator<Item=Future<Result<T, E>>>.` With effects, they’d both be `Co<() ! Yield<T>, Fail<E>, Async>`. In essence, effects form a "flat" set, while monads have some encapsulation order.
  * In Silica, effectful functions return coroutine objects.
    * The set of effect operations that the coroutine may perform are in the type signature of the coroutine.
    * Functions may be polymorphic over the effects they perform.
* **Syntax:** Silica's syntax is largely Rust inspired.
  ```
  effect Fail<E> { fail: E -> ! }
  struct DivByZero;
  co div(a: i32, b: i32) -> i32 ! Fail<DivByZero> {
    if b == 0 {
      perform Fail.fail(DivByZero);
    }
    a / b
  }
  ```
  * The `effect` keyword declares an effect and associated operations (2.4)
  * The `struct` keyword declares an aggregate type, in this case with no members.
  * The `co` keyword, in this case, a coroutine constructor function, `div` which eventually returns a 32 bit integer, `i32`, and may perform a ` Fail<DivByZero>` effect.
  * The `perform` keyword suspends the coroutine with the `Fail.fail` effect operation. This operation cannot be resumed because the operation resumes with the empty type, `!`.

### 2.2 Linearity and Movability
Silica incorporate linear types (values consumed exactly once) for compile-time resource safety, preventing leaks and double-use (e.g. in memory or file handles). This strict default is made ergonomic via the following relaxations:
* **Borrowing:** Temporary, non-consuming access to values is permitted via references. Borrowing does not consume a linear value, but its usage is governed by lifetime rules (Rust NLL-inspired, analyzed on MIR) to prevent dangling references. Specific kinds of state-tracking references offer more fine-grained control during borrows (See Section 2.3).

* **Opt-in implicits**: The following traits further relax linearity rules by controlling implicit destruction and copying:
  * **ImplicitDrop (Affine):** Allows values to be used at most once . If unused by scope end, its drop method is called automatically. Enables affinity.
      ```
      trait ImplicitDrop {
        // Non-effectful cleanup logic.
        fn drop(&deinit self);
      }
      ```
      * *Note:* `drop` cannot perform effects. Effectful cleanup requires an explicit function like `co destroy(&deinit self) -> () ! E`. See `2.6` on `defer` blocks for effectful clean up.
      * *Note:* Allocation and deallocation are not considered effects.
      * The API for explicit, effectful, cleanup is `trait Destroy { effects E; co destroy(&deinit self) -> () ! E }`
  * **ImplicitCopy (Relevant):** Allows values to be used one or more times (relevant) or any number of times (unrestricted, if also ImplicitDrop). Implicit copies are made via the copy method when needed.
      ```
      trait Copy {
        effects E;
        // Performs a copy. Allows user-defined logic (not just memcpy).
        co copy<'a>(&'a self, dest: &'a out Self) -> () ! E + 'a;
      }
      // Allows the compiler to call `copy` implicitly.
      trait ImplicitCopy {
        fn copy(&self, &out Self);
      }
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
  * **Pass-by-value requires `Move` or `ImplicitCopy`.** Attempting to pass by value a type that does *not* implement either of these traits results in a **compile-time error**. This enforces the default immovability of types.
    * The compiler will prioritize try moving values into functions with `Move`.
    * If the value is used later and the trait is `ImplicitCopy`, then the prior pass-by-value will be a copy.
    * In the rare scenario where `Move` is not implemented for the type, but `ImplicitCopy` is, the value will be copied.
    * *Note: the `Move::move` operation consumes the source (`&deinit self`) and initializes the destination (`&out other`), transfering ownership of any internal resources.*

  * **Move/Copy Elision:** For types implementing `Move` (or `ImplicitCopy`), the compiler *may* optimize away the actual call to the `move` (or `copy`) method, directly constructing the value in the destination or reusing the source memory location, similar to Guaranteed RVO/Copy Elision in C++. Code defined in `move` and `copy` methods are not guaranteed to execute.

  * **Coroutines and Movability:**  Coroutines that allow references to their own state (even self-referentially) will generally not implement `Move` or `Copy` for safety reasons. To store or transfer such immovable coroutines (e.g., in async executors), they can be placed on the heap using a movable Box type (e.g., `Box<Co<T!E>>`, as `Box` implements `Move`).

* To transfer ownership *responsibility* (specifically, the obligation to deinitialize) for an **immovable type** without moving its memory, pass it via an exclusive **`&deinit T`** reference. The caller retains the storage allocation, but the callee is responsible for deinitialization.

* It is generally best practice to mark types as `Move`, `ImplicitDrop` and `Copy` where possible and semantically correct.

*  Aggregate types (structs/enums/closures/coroutines) can only implement `Move`, `ImplicitDrop` and `Copy` if all their members (and captured state) have these properties.

* **Deriving Defaults:** Mechanisms to automatically derive default implementations for `Move`, `ImplicitCopy`, `ImplicitDrop`, and `Copy` for aggregate types will be necessary for ergonomics.

### 2.3: Flow-Sensitive Typing for Initialization State

Silica provides fine-grained control over memory initialization state via a system of non-owning references coupled with flow-sensitive typing. This enables safe in-place operations, with strong compile-time guarantees and better ergonomics than patterns like `MaybeUninit` or requiring intialization on the stack before moving into a heap allocation (in Rust).

* **Initialization State Tracking:** The type system tracks the initialization state of memory locations accessed through exclusive references along two dimensions: the **current state** (is it initialized now?) and the **required final state** (must it be initialized or uninitialized when the reference's lifetime ends?). This yields **four** fundamental exclusive states.

* **Transmutation Principle:** Operations that change the current initialization state (from initialized to uninitialized, or vice-versa) cause the reference type itself to **transmute**. This transmutation reflects the new current state while preserving the original required final state associated with the reference's lifetime. This "type transmutation" is inherently **flow-sensitive**, meaning the compiler tracks the changes in the type of the variable based on the operations performed along each control flow path.

* **Triggering State Changes:**
  * **Initialization:** Occurs via direct assignment (`x = value;`) or by lending a `&out T` sub-reference, which will initialize the location by the end of its lifetime.
  * **Deinitialization:** Occurs via moving the value out, explicitly calling a destructor (e.g., `drop(value)` conceptually), destructuring an aggregate reference and recursively deinitializing its parts, or by lending a `&deinit T` sub-reference which will deinitialize the location by the end of its lifetime.

* **Exclusive Reference Types & Contracts:** The four primary exclusive reference types embody the different state combinations:

  * `&'a mut T`: **Mutable**: Current State: Initialized, Final State: Initialized.
  * `&'a out T`: **Output**: Current State: Uninitialized, Final State: Initialized.
  * `&'a deinit T`: **Deinitialization**: Current State: Initialized, Final State: Uninitialized.
  * `&'a uninit T`: **Uninitialized**: Current State: Uninitialized, Final State: Uninitialized.

* **Linearity**: The current initialization state of `&mut T` and `&uninit T` is their final intialization state, so these references are `ImplicitDrop`, or affine. However, `&out T` and `&deinit T` have unfulfilled obligations to initialize or deinitialize their referent and therere cannot be implicitly dropped. They are linear types.

* **Transmutation Rules Enumerated** 
  * **Initialization Occurs:**
    * `&out T` becomes `&mut T` (Initialzed end state is preserved).
    * `&uninit T` becomes `&deinit T` (Uninitialized end state is preserved).
  * **Deinitialization Occurs:**
    * `&mut T` becomes `&out T`
    * `&deinit T` becomes `&uninit T`
  * **Subreference rules:** A reference may be borrowed from another reference so long as the current state matches, e.g. `&mut T` may be borrowed from `&mut T` or `&deinit T`. Note that borrowing a `&out T` or `&deinit T` subreference will change the referent's initialization state, causing the outer reference to transmute. 

* **Destructuring References:** A reference to an aggregate type may be destructured into references to the aggregate's components. These component references have the same current initialization state and end initialization obligation as the original reference.

* **Unification Rule:** When control flow paths merge, the stateful reference type of a variable must be identical on all incoming paths. If different types (representing different states or end-obligations) merge, it is a compile-time error.

* **Transmuting Library Types:** Silica allows library-defined owned types to opt into compile-time state tracking, enabling safer state machines and resource management protocols. This is primarily intended for types managing a resource with binary operational states (e.g., Initialized/Uninitialized, Open/Closed, Locked/Unlocked).
  - **Declaration (`~?`):** A type definition prefixed with `~?` signals it has compiler-tracked binary state. Conventionally, `TypeName<T>` refers to the primary state (e.g., Initialized) and `~TypeName<T>` refers to the alternative state (e.g., Uninitialized).
  - **State-Specific Method Implementations:** Methods can be defined to operate only on a specific state or on either state:
    - `impl TypeName<T> { ... }` Defines methods for the primary state.
    - `impl ~TypeName<T> { ... }` Defines methods for the alternative state.
    - `impl ~?TypeName<T> { ... }` Defines methods valid in either state.
  - The compiler uses flow analysis to determine the current state of a variable and resolves methods accordingly.
  - **State Transmutation (`~&mut self`):** Methods that transition the object's state use the special receiver type, `~&'a mut self` which indicates transmutation to the other state after the end of the lifetime `'a`[^instant_of_transmutation].
    - Calling such a method consumes the object in its current state and causes the compiler's flow analysis to track the variable as being in the other state subsequently. This is the primary mechanism for implementing state transitions (e.g., `Box::drop_inner(~&mut self)`, `~Box::init(~&mut self, T)`).
  - Only variables may be transmuted. Methods that transmute the type cannot be called on references.

[^instant_of_transmutation]: Note, since the variable is considered borrowed is unusable for the lifetime `'a` it is equally valid to think of transmutation as happening at the start of the lifetime.


### 2.4 Effects and Coroutines
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
    * If an effect is unnamed, we give a default name from the effect declaration, e.g. `State.set`.
  * **Declararation-free operations:** Operations may also be defined inline, without the `effect` declaration. 
    * E.g., the expression `y = perform my_new_operation(x)` defines an operation called `my_new_operation` whose perform type `typeof(x)` and resume type `typeof(y)` are inferred.
    * Such operations unify by name and perform/resume type.
    * Declaration free operations are useful for oneoff operations that aren't used in enough places to warrant an `effect declaration. The tradeoff is analogous to using tuples vs predefined structs.
* **Propagating Effects (`?` Operator):**
    * `expr?` operates on a Coroutine value yielded by `expr`.
    * It runs the coroutine. If the coroutine performs an effect `E_op`, `?` propagates `E_op` outwards.
      * This requires `E_op` to be in the current function scope's operation set.
      * If the coroutine eventually completes with value `T`, then `expr?` evaluates to `T`.
    * **Relationship to `handle`:** `expr?` may be considered syntatic sugar for `expr handle {}`.
* **Coroutine Values:** Functions whose execution involves `perform` (or `?`) yield control, so they return a **Coroutine** value. This value encapsulates the suspended computation, its eventual result type (`T`), and the set of effects (`E`) it might perform when resumed.
* **Handling Effects (`handle` Block):**
    * **Conceptual Syntax:** `coroutine_expr handle { InitialClause?, ReturnClause?, OpClause*?, FinallyClause? }`
    * `OpClause` form: `Effect.Operation(args), resume => { /* handler body */ }`. It matches a specific performed operation.
    * `resume(value)`: Resumes the original computation, passing `value` back as the result of the `perform`. `value` must match the operation's resumption type.
    * The `InitialClause`, e.g. `initially => ...` is executed before the coroutine is run. The other clauses may reference state from this clause, but not from each other.
    * The `ReturnClause`, e.g. `return x => ... ` is executed if the coroutine finishes and returns its final value. Guaranteed move elision is desirable for simple returns like `Return(v) => v`, which is the default behavior if unspecified.
    * The `FinallyClause`, e.g. `finally => ...`, this code is executed if control leaves this `handle` block, never to return, due to a propaged effect operation that is not resumed. The coroutine this `handle` block evalutes to will have a `destroy` method that will run this code.
    * **resume:** Within the handler arm, `resume` represents the unique, linear continuation capability (FnOnce) for resuming from this specific suspension point. Calling `resume(value)` consumes the `resume` and passes `value` back to the perform site, and transfers control away from the handler arm. Code immediately following a `resume()` call within the same block is unreachable.
    * **resume.into():** For scenarios like async executors needing to store and move the continuation, `resume.into()` can be used. This consumes the resume capability and yields a new, _movable_ object (conceptually `impl FnOnce(ResumeType) -> Co<T!E>`) that owns the continuation state. This requires the underlying coroutine state to implement `Move`. Calling the resulting object later will run the moved coroutine from the current suspension point.
    * **Evaluation** A `expr handle {...}`  block evaluates to the type of the return arm, or if that return arm is not specified, the return type of the coroutine that `expr` evalautes to. Any unhandled operations are propagated into the local context. If any operations are propagated, the `handle` expression must be in a coroutine.
* **Effect Tunneling:** Functions polymorphic over an effect parameter `E` cannot handle operations in `E`, even if those operations happen to collide with operations used in the polymoprphic function body (otherwise, the implementation of the generic function is revealed to the caller). Such operations "tunnel through" and are propagated outwards as part of the function's `E` effect parameter, bypassing local handlers for the operation. Concrete effects not matching `E` are handled normally.
* **Coroutine Literals:** `co $expression`, e.g. `co foo()? + bar()?`, is an expression that evaluates to a coroutine. In the example, the coroutine returns the result of the addition and proapagtes effects from `foo()` and `bar()`.
* **Effectless coroutines:** It is technically possible for a coroutine to have no effects, `co 1 + 1`. Such coroutines behave like lazily evalauted "thunks" in the Haskell sense.
* **Coroutines are Existential Types:** That is, each coroutine in the source code has a unique (unnamable) type with a possibly uniquely sized state machine.
  * `Co<T!E>` is a trait (aka typeclass) that abstracts all coroutines that eventually return `T` and perform `E`.
* **Coroutine traits:**
  * **Coroutine State:** Coroutines that emit references to their activation frame whose lifetime crosses a suspension point cannot be safely moved or copied. Coroutines that store immovable or non-copyable data in their activation frame across a suspension point, are similarly immovable or non-copyable.
  * **Opt-in `Move`:** Coroutines may opt into movability declaring they implement the `Move` trait, e.g. `co foo() -> T!E + Move`, however the compiler will enforce the aforementioned properties at compile time.
    * **Move via Box or Reference:** Immovable coroutines can be made movable by placing them behind a reference or pointer. If `S: Co<T!E>` then `&mut S: Co<T!E> + Move` and `Box<S>: Co<T!E> + Move`.
  * **Opt-in `Copy`:** Analogously, coroutines can `+ Copy` or `+ ImplicitCopy` to opt into those traits and associated restrictions too.
  * **Opt-in `ImplicitDrop`:** Coroutines that store a linear resource across a suspension point cannot be `ImplicitDrop`.
    * Coroutines can `+ ImplicitDrop` to get that trait however the compiler will enforce that no linear resources are stored across suspension points in such coroutines.
  * **Opt-in `Destroy`:** Coroutines can be explicitly destroyed if they clean up their linear resources using `defer` blocks (Section 2.6).

### 2.5 Access, Calls, and Dereferencing
Silica distinguishes between direct member access and access through indirection (pointers/references), providing separate operators with distinct capabilities. It introduces an auto-dereferencing mechanism specifically designed to handle potential effects during chained access.

* **Explicit Dereference (`^` Operator):**
  * A postfix derference operator is used to convert references into lvalues, or "place expressions" which may be assigned to (or read from if initialized). 

* **Field Access and Unified Function Call Syntax (UFCS):**
  * The dot operator (`.`) is used for direct field access (`value.field`) on structs and for Uniform Function Call Syntax (UFCS) on methods (`value.method(args)`) desugaring to `method(value, args)`.
  * Unlike in Rust, the `.` operator performs **no automatic dereferencing**. It operates solely on the immediate type of the receiver expression (`value`). To access members through any pointer or reference type using `.`, an explicit dereference (with `^`) must be performed first, e.g. `ptr^.method()`.
  * To call a callable field, to avoid ambiguity with UFCS resolution, you have to parenthesize the field access, e.g., `(foo.bar)(args)`, because `foo.bar(args)` is interpreted as `bar(foo, args)`.
  * Note that you can use arbitrary expressions in UFCS position, e.g., `foo.(bar(arg1))(arg2)` desugars to `bar(arg1)(foo, arg2)`.

* **Chained Derferencing Access (`->` Operator):**
  * The arrow operator (`->`) is the mechanism for accessing data through potentially multiple levels of indirection provided by pointer, reference types, or coroutine types.
  It performs automatic, potentially chained dereferencing through smart pointers that implement the `AutoDeref` and `AutoDerefMut` traits. 
  * Conceptual `AutoDeref` trait (`AutoDerefMut` is analogous):
    ```
    trait AutoDeref {
      type T;
      effects E;
      co deref(&self) -> T ! E;
    }
    ```
  * `foo->` desugars to some expresion matching the regex, `foo(\.deref()\?\^)+`, the compiler infers how many times to call `.deref()` and whether to invoke `?`. 
    * When resolving `expr->`
      1. Initialize `current = expr.deref()`
      2. propagate effects and dereference: `current = current?^`
      3. if the type of `current` has a `.deref` method, call it, and go to 2. 
  * **Chained Dereferencing with method calls:** A `.` UFCS operator is still required. `foo->.bar(x)` would be how you write `bar(foo->, x)`. Unlike in Cpp, `foo->bar()` would be a syntax error.

### 2.6 `defer` blocks
* Functions and coroutines may place `defer { ... }` statements which execute when control permanently leaves this activation frame. This includes when the function or coroutine returns, or when a suspended coroutine is explicitly dropped.
  * The intended use is for potentially effectful cleanup of linear resources.
  * *Note:* Control does not permenantly leaves a coroutine if it is suspended and resumed. Code in `defer` may run some time after suspension if the coroutine is implicitly dropped or explicitly destroyed.
* `defer` blocks _may not_ be placed in the `OpClause` of `handle` blocks. At least until "code after resume" (5.2) is supported.
* **Performing or Propagating Effects** in `defer` blocks is allowed, however it is a compile-time error to perform a nonresumable effect, e.g. `Fail<E>: E -> !`, as the intent of `defer` blocks is to enable the effectful cleanup of linear resources.
* `defer` blocks are conceptually "copy-pasted" and executed after statement where the return value is written, which makes `defer` blocks incompatible with tail call optimization (TCO). After defer blocks are "pasted" ASAP destruction is used, `ImplicitDrop::drop` is called as soon as possible, respecting lifetime constraints.

### 2.7 Raw pointers
* **Primitive Types:** Silica provides primitive raw pointer types, requiring `unsafe` blocks for dereferencing (`^`) or casting back to safe references.
    * `Ptr<T>`: A non-nullable, potentially aliased raw pointer to `T`. Offers no compile-time aliasing guarantees.
    * `RestrictPtr<T>`: A non-nullable, non-aliased raw pointer to `T`. Guarantees exclusive access within its scope (like C's `restrict`), enabling optimizations (`noalias` attribute).
* **Alignment:** Both `Ptr<T>` and `RestrictPtr<T>` are assumed by the compiler, for optimization purposes to be aligned to `T`'s alignment.
* **Nullability:** `Ptr<T>` and `RestrictPtr<T>` are assumed to be non-null, for optimization purposes. Nullable pointers use `Option<Ptr<T>>` or `Option<RestrictPtr<T>>`. Compiler NPO ensures these match C's nullable pointer ABI. 
* **`unsafe` Responsibilities:** When converting raw pointers back to safe references, the programmer must guarantee the pointer points within a live allocation, respects aliasing rules (especially for `RestrictPtr` or when creating `&mut`), and points to memory in the correct initialization state.

### 2.8 Guaranteed Return/Perform/Resume Value Optimizations (RVO/PVO/other RVO)
Silica guarantees that immovable types may be constructed in place across the return, perform, and resume boundaries. This is a generalization of RVO in Cpp.
* **Functions:** Functions that return a non-scalar type will be rewritten to take an `&out ReturnType` reference, so the final return value may be constructed in-place.
* **Coroutines:** The coroutine ABI will have a `set_output_ptrs` function which will be called at the start of every `handle` block so, even if the coroutine is moved, it has the return place to construct the final return value. This function will set two pointers in the coroutine state:
  1. A *return place pointer*, for guaranteed RVO
  2. A *communication place pointer*, which points to a place in the callee's stack that is large enough for:
     1. a discriminant, which tells the callee whether the coroutine is returning or performing an effect, and which effect
     2. a pointer to the coroutine's activation frame where the resumption value should be written
     3. any perform values 

### 2.9 Expression Precedence
1.  `( ... )`, `{...}` (Grouping)
2.  Literals, Variables, paths, `|args| expr` (lambdas)
3.  `expr(...)`, `expr.field`, `expr.method(...)`, `expr[...]` (Calls, Fields, UFCS Calls, Indexing) [Left-associative]
4.  `expr^`, `expr->`, `expr->mut`, `expr&`, `expr &mut`, `expr &out`, `expr &deinit` (Postfix Deref/Ref) [Left-associative]
5.  `expr?` (Postfix Propagation) [Left-associative]
6.  `expr handle { ... }`, `expr match { ... }` (Postfix Blocks) [Left-associative]
7.  `-expr`, `!expr` (Prefix Unary) [Right-associative]
8.  `*`, `/`, `%` (Multiplicative) [Left-associative]
9.  `+`, `-` (Additive) [Left-associative]
10. `<<`, `>>` (Bitwise Shifts) [Left-associative]
11. `==`, `!=`, `<`, `>`, `<=`, `>=` (Comparison) [Non-associative]
12. `&&` (Logical AND) [Left-associative]
13. `||` (Logical OR) [Left-associative]
14. `if cond {then} else {else_}`, `block { ... }` (Control Flow / Block Expressions)
15. `co expr` (Coroutine Literal Expression) [Prefix]
16. `perform ...` (Effect Operation)


### 2.10 `VTable<Trait>`
- `VTable<Trait>` is a recognized type in the system that's thought of as a struct of run time type information, 
including function pointers, a type id, and the type's size.
- There will be some API for using this with a type erased pointer. 
- TODO:


## 3. Design Rationale and Open Questions
This section summarizes key design choices where Silica prioritizes certain goals over others, accepting the associated trade-offs.

### 3.1 Accepted Trade offs
This subsection explains the _why_ behind major finalized design choices, clarifying the guiding principles and accepted consequences.

* **Linearity vs. Flexibility**: Silica prioritizes strict linearity by default for maximum resource safety (exactly-once usage). Flexibility and ergonomics are then recovered through opt-in relaxations (discussed above). Trade-off: Requires explicit opt-in for common affine/unrestricted patterns vs. Silica’s linear-by-default.
* **Immovability vs. Ergonomics**: Silica chooses immovability by default, requiring types to opt into the `Move` trait if they can be safely moved. This simplifies handling of immovable types, including coroutines that emit pointers, enables safe in-place initialization/destruction patterns (via `&out`/`&deinit`), avoids complexities like Rust's `Pin`, and may simplify self-referential types and FFI (TBD). Trade-off: More control for specifying that most things are `Move`.
* **Effects vs. Built-ins**: Silica uses its algebraic effect system as the primary, unified mechanism for handling computational contexts like errors, concurrency, generators, etc., instead of providing separate built-in language features for each. The goal is greater composability and uniformity. Trade-off: Introduces the complexity of the effect system, and libraries may be less ergonomic than dedicated language features.
* **Syntax vs. Convention**: Silica adopts some less conventional syntax, notably postfix `handle` and `match` blocks. This choice aims for potentially greater fluency when chaining operations on a value. Trade-off: Reduced familiarity for programmers accustomed to more standard prefix syntax.

### 3.2 Open Questions
This subsection outlines areas where the design is still evolving, involves known difficult interactions, or requires further investigation and decision-making.
* **Object Safety & Effects:**
  * **Tension:** How should traits with effectful methods interact with dynamic dispatch (dyn Trait)?
  * **Current Status:** A basic strategy is outlined (effects must be declared in the trait, trait objects carry the union of effects), but the precise semantics, especially regarding effect polymorphism and potential dynamic effect dispatch, need detailed specification.
  **Open Question:** What are the exact rules and limitations for object safety concerning various effect patterns (polymorphism, labels, etc.)?

* **Lifetimes Across perform/resume (Formal Rules):**
    * **Tension:** Ensuring the safety of references held across suspension points (`perform`/`resume`) is complex.
    * **Current Status:** Basic rule established: references to transient local state are invalidated by suspension/resumption (enforced by MIR analysis). Stacked Borrows provides a conceptual model.
    * **Open Question:** What are the precise, formal rules the borrow checker will use to track lifetimes interacting with coroutine state, captures, effect arguments, and resumption values, especially considering potential coroutine moves (`resume.into()`)? How does this interact robustly with effect polymorphism (`!E`)?

* **Object Safety & Effects (Refined Direction):**
    * **Tension:** How to enable dynamic dispatch (`dyn Trait`) for traits whose methods return existential coroutine types (`-> T ! E + Props`), especially without relying on intrinsic `Box` heap allocation?
    * **Current Direction:** Leaning towards a "Placement Strategy" pattern. This involves the caller passing an `Allocator` and a strategy object/closure (e.g., via `MyPtr::placer(&alloc)`) to the `dyn Trait` method. The implementation uses this to allocate space for the concrete coroutine state `S` and initialize the caller's desired smart pointer/handle type (`MyPtr`, `Arc`, etc.) via an `unsafe` placement method.
    * **Open Questions:** What is the precise API for the `PlacementStrategy` trait or the `unsafe` placement method required on smart pointers? What information (layout, vtables) must be passed? How is the type-erased handle returned by the placement method defined, including its necessary `Co` vtable? How are allocator lifetimes (`'alloc`) managed safely with the handle?

* **Flow-Sensitive Owned Types (`~?Type`) - Formal Semantics:**
    * **Tension:** Integrating flow-sensitive state tracking for owned variables (`~?Type`) affects many parts of the type system.
    * **Current Status:** Syntax (`~?Name`, `impl Name`, `impl ~Name`, `~&mut self`) and core concepts (binary state, flow tracking, state frozen during borrow, merge unification) are defined. Transmutation timing via `~&'a mut self` decided (end of `'a`). Type state must match when control flow merges.
    * **Open Question:** How exactly does it interact with generics (bounds needed?), complex control flow (loops, match), coroutine state saving/restoration, and potentially effectful transmuting methods? What are the precise semantics and constraints on the `~&mut self` receiver?

* **First-Class VTables (`VTable<Trait>`) - API Definition:**
    * **Tension:** Using `VTable<Trait>` seems key for type erasure (especially object safety) without compiler magic for specific types.
    * **Current Status:** Concept agreed upon - `VTable<Trait>` holds layout, function pointers (including combined destroy+dealloc). Marked as TODO in Sec 2.10.
    * **Open Question:** What is the exact structure of `VTable<Trait>`? What is the safe/unsafe API for creating `VTable` values (compiler generated?) and for invoking methods through a `(data_ptr, vtable_ptr)` pair?

* **`handle` Consumption Mechanism Trait:**
    * **Tension:** `handle` needs a standard way to consume library-defined smart pointers (`MyPtr<dyn Co<T!E>>`) to manage the underlying coroutine, especially immovable ones, without special knowledge of `Box`.
    * **Current Status:** Considering a `DerefDeinit` trait to allow the `handle` block to consume the coroutine.
    * **Open Question:** How exactly will this work?

## 4. Implementation State & Plans

### 4.1 Immediate state and Immediate next steps
TODO: Rewrite.


### 4.2 LLVM Coroutine Backend Strategy
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


### 5.3 Custom `=` Assign, Reassign, Setters
* **Alternative Considered:** Desugaring the assignment operator (`=`) to methods defined by traits (`Assign` for initialization, `Reassign` for updates), allowing user-defined types/proxies to customize assignment behavior.
* **Reasons for Rejection:** While enabling custom logic for proxies, this approach adds potential "magic" or implicit conversion feel to the fundamental assignment operator.
* **Decision:** Keep assignment (`=`) as a built-in operation with fixed semantics (compiler-driven in-place initialization for uninitialized LHS using literals/constructors, or standard drop-then-move/copy for initialized LHS). Custom update logic requires explicit method calls.


### 5.4 **Allocation/Deallocation as Effects:**
* **Alternative Considered:** Modelling heap memory allocation and deallocation as algebraic effects. Modeling them as effects makes their usage explicit but means types like `Box` cannot use `ImplicitDrop` for deallocation.
* **Reasons for Rejection:** In addition to the ergonomic costs, `malloc` and `free` are trivially resumptive (always resume exactly once) which does not harness the power of algebraic effects (resuming multiple times, after a long suspension, or not at all). `perform/resume` may be too expensive for such common operations, though trivial resumption optimization (TRO) may help. 

## 6. Library Design Patterns and Idioms

### 6.1 Concurrency Model: Structured Concurrency
Silica will adopt structured concurrency as its primary model for managing concurrent tasks, leveraging the algebraic effect system for its implementation.

* **Core Principle:** The lifetime of concurrent tasks ("children") is strictly bound by a lexical scope. A scope cannot exit until all tasks spawned within it have terminated. This prevents "orphan" tasks and ensures predictable resource management. This model aligns exceptionally well with Silica's features:
  * **Lifetimes:** Parent tasks waiting for children guarantees that data borrowed from the parent scope by child tasks remains valid for the children's entire lifetime, eliminating a major class of concurrency bugs.
  * **Linearity:** Ensures resources passed to or used by concurrent tasks are consumed correctly according to their type rules.
  * **Effects/Handlers:** The concurrency scope is defined naturally using an effect handler. Operations like spawning tasks (`perform Concurrent.spawn(...)`) are handled within this scope.
* **Cancellation:**
  * Cancellation is cooperative and initiated by the parent scope/handler.
  * Checks for cancellation occur implicitly at concurrency-related suspension points within child tasks (e.g., when performing scheduler yields, channel operations, or potentially specific effect operations designated as cancellation points).
    * Upon detecting cancellation, the parent calls the child coroutine's explicit `destroy` method.
      * As per Section 2.6, this runs the coroutine's `defer` blocks to ensure graceful consumption of linear resources.

## 7. AI Assistant Instructions
* **Goal:** Assist in designing and implementing Silica. Be curious, ask clarifying questions, offer constructive criticism, identify inconsistencies, and propose solutions/alternatives based on PL theory and practice (especially Rust, Haskell, ML, Koka influences). The primary designer (the user) is the authority on design decisions. The design is iterative. Explain complex concepts succinctly. Prioritize consistent forwards momentum.
* **Style:** Collaborative, inquisitive, detailed, encouraging. Reference specific sections/decisions when relevant.
* **Assume:** The author is an expert in Rust/C++ and is familiar with PL concepts (types, inference, scope, linear types, algebraic effects), but not necessarily deep theory (e.g., category theory, advanced dependent types).

**Key Design Concepts & Decisions Summary:**
  * **Core Pillars:** Algebraic Effects (resumable checked exceptions), Linearity/Lifetimes, Default Immovability/Fine-Grained Control, Uniformity (minimize compiler magic for specific types like `Box`).
  * **Linearity:** Default (exactly-once). Relaxed by:
      * Borrowing (`&`, `&mut`, `&out`, `&deinit`, `&uninit`).
      * `ImplicitDrop` trait (Affine - at most once). Drop logic must be non-effectful (dealloc OK).
      * `ImplicitCopy` trait (Relevant - one or more uses, requires pure `Copy` base). Copy logic must be non-effectful (alloc OK).
      * `ImplicitDrop` + `ImplicitCopy` = Unrestricted (any number of uses).
  * **Movability:** Default Immovable. Opt-in via `Move` trait. `Move::move(&deinit self, &out other: Self)` consumes source, inits dest. Pass-by-value requires `Move` or `ImplicitCopy` (preferring `Move` unless value used after call). Coroutines often immovable if self-referential; use `Box<Co>` or `&mut Co` for movability.
  * **Copyability:** `Copy` trait (`fn copy(&self, &out Self) -> Co<()!E>`) allows potentially effectful, explicit copy. `ImplicitCopy` marker trait (`ImplicitCopy: Copy`) requires `Copy::copy` impl to be effect-free, allows compiler copies.
  * **Destruction:** `ImplicitDrop` trait for affine types (effect-free `drop(&deinit self)`). `Destroy` trait (`co destroy(&deinit self) -> ()!E`) for explicit, potentially effectful cleanup. Hybrid drop timing: `ImplicitDrop` uses ASAP (after last use, considering `defer`), `defer` blocks use RAII (scope exit).
  * **References & Flow Sensitivity:** Flow-sensitive typing tracks init state for `&out`, `&deinit`, `&mut`, `&uninit`. Transmutation occurs based on use (e.g., init via `&out` yields `&mut`). Control flow merges require identical states (unification error otherwise). State is frozen during standard borrows (`&`, `&mut`). References to transient local state invalidated across suspension.
  * **Owned Type State (`~?Type`):** Library types can opt-in (`struct ~?Name`) to flow-sensitive binary state tracking (e.g., `Name` vs `~Name`). `impl Name`/`impl ~Name` for state-specific methods. `impl ~?Name` for methods that work on both states. `~&mut self` receiver signals state transmutation after method call. Requires sophisticated flow analysis for owned variables. State frozen during standard borrows.
  * **Effects & Coroutines:** Algebraic effects model (resumable checked exceptions). `effect Name {op: P->R}` defines interfaces. `perform EffectName.op(p)` or `perform instance.op(p)` invokes. Unnamed instances use default name (`EffectName`). Multiple instances require type level name. Declaration-free ops allowed. Effect signatures mandatory on `co` functions. Effects tunnel through polymorphic functions.
  * **`fn` vs `co`:** `fn` defines regular functions (return `T`, no effects allowed). `co` defines coroutines (return `Co<T!E>[Props]`, effects allowed). Use `co coro_name(...) -> T ! E + Prop1 + Prop2;` signature.
  * **Coroutine Literals/Lambdas:** `co { body }` creates a coroutine value. `|args...| co { body }` creates a lambda returning a coroutine value (inferred type `Fn*(...) -> Co<...>`).
  * **Handling (`handle`, `?`):** `handle` consumes coroutine (likely via a standard trait like `ConsumableViaGuard` / `IntoCoHandle` needing definition for smart ptrs), manages execution, runs cleanup via RAII guard drop or extracted functions. `?` runs coroutine, propagates allowed effects, yields `T` on completion (using RVO).
  * **RVO/PVO/ResumeVO:** Guaranteed via ABI. Handler passes two conceptual pointers to the coroutine before resume: `FinalDestPtr<T>` (for `return T`) and `CommAreaPtr` (for `perform P` + discriminant). Resume value `R` placed directly into coroutine frame slot (pointer provided by coroutine on suspend). `T`, `P`, `R` do not need `Move`/`Copy`.
  * **Operators:**
    * `.`: Pure UFCS (`expr.m(a)` -> `m(expr, a)`) or field access (`expr.f`). No auto-deref. Field access only if no `()`. Prioritize fields and methods before UFCS. Callable fields need `(expr.f)(a)`.
    * `->`: Chained, potentially effectful *borrowing*. Defaults to immutable (`->ref`, using `AutoDeref::deref`). `->mut` uses `AutoDerefMut::deref_mut`. Result `&T` or `&mut T`. Followed by `.` for UFCS call/field access on result. Does not handle `&out`/`&deinit`.
    * `^`: Postfix dereference to "place".
    * `^&`/`^&mut`/`^&out`/`^&deinit`: Postfix explicit borrow from place. `^&out`/`^&deinit` trigger state transmutation on the variable providing the place if it's a `~?Type`. Restricted to variables, not complex expressions ending in `->`.
  * **Type System:** HM-hybrid, local inference, top-level annotations. ZST function items implementing `Fn*` traits. Overload resolution based on first argument (requires extended inference engine). `VTable<Trait>` concept for type erasure. Numeric literals require constraints (`α: Numeric`), error on ambiguity.
  * **Object Safety:** Open question. Direction is towards a "Placement Strategy" pattern using `Allocator` trait + `unsafe` placement method on smart pointer types, avoiding intrinsic `Box`.
  * **Allocators:** `Allocator` trait defined (using `RestrictPtr`, returning values not `Co`). Allocation/deallocation are *not* algebraic effects.
  * **Temporary Lifetime Extension:** Desired. Planned implementation via implicit MIR variables + ASAP destruction.

* **Design Review Checklist:** (For AI use when reviewing/generating this document)
  * Check for Clarity: Ensure explanations are understandable to the target audience. Define terms. Provide rationale. Use examples.
  * Check Target Audience Readability (Rust/C++ Dev): Review explanations for novel concepts (effects, linearity, `&deinit`, immovability, `~?Type`) to ensure clarity without requiring deep PL theory. Confirm differences from Rust/C++ are explicit.
    * Crucially, when motivating algebraic effects, avoid examples like State or IO which are often seen as trivial non-issues in imperative languages. Focus on demonstrating value through patterns like generators, async/await, exceptions, transactions/retries, etc.
  * Check each section meets its goals: (Core Language: final state; Trade-offs: priorities; Implementation: current state + plan; Alternatives: rejected or deferred ideas; AI Instructions: guidance).
  * Check for Completeness: Ensure major features covered. Capture assumptions, definitions, decisions, TBD points for future AI context.
  * Check for DRYness: Ensure concepts explained once, introduced before use, no major repetition between sections.
  * Check Internal Consistency: Verify that design decisions (e.g., `perform` scoping, `?` semantics, trait definitions, move rules, `~?Type` rules, operator precedence) are reflected consistently across all relevant sections, including examples and tips.
  * Check Revision Annotations: When updating, mark changed sections with (Revised Date) and summarize edits. Remove these markers when producing a final export version if requested.
  * Check Heading Levels: Ensure `##` for major, `###` for subsections.
  * Check Export Cleanup: Ensure internal status notes or revision markers are removed before generating a final "export" version when requested.
  * Ensure internal consistency on the distinct roles of `.`, `^`, `?`, and `->`.
  * Require explicit instructions for generating the full document. Prefer to only generate subsections, for easier review, and to avoid risking crashes in the AI environment due to large outputs.
  *  Remember `~TypeName` denotes an *uninitialized state* concept, not Rust destructor syntax.
  *  Remember `handle` *consumes* its argument via a trait mechanism (specific trait TBD)
  *  Avoid re-introducing rejected ideas unless explicitly asked to reconsider.

* **Onboarding Instructions:** Follow these instructions step by step to familiarize yourself with the Silica project.
  1. Review this design document.
     1. List the important features and consider the interactions between pairs of features.
     2. Compare and contrast Silica against other languages, such as C++, Rust, Python, Koka, Ocaml, and Haskell.
     3. List the points where the design is light on details
     4. For each of those points, how would an experienced language designer flesh out the details? What are the potential options and how do they compare when considering Silica's values and goals.
     5. Enumerate the source of potential complexity, then think through potential resolution strategies. Note that Silica is still in development and while issues need to be surfaced, they need not be immediately solved.
     6. Enumerate the sources of potential ergonomic friction or missing features in Silica. What can it learn from other languages. For each potential feature consider whether it is compatible with Silica's design and values.
  2. Review the langauge's implementation files
     1. See `grammar.js` for the tree-sitter grammar. 
        1. Think through what features are missing. The grammar will only be expanded as and when the type system is capable of modeling the associated complexity.
     2. See `parse.rs` for Rust code that uses the generated tree-sitter parser and converts it to datastructures in Rust
     3. See `ast.rs` to see the type system, inference, and a long list of tests.
        1. Describe the current implementation state of the type system. Compare it to the implemented features in the parser/grammar. Are there gaps?
        2. Compare the implementation state to the overall design.
           1. How should the type system be augmented to bring it closer to the design.
           2. For each potential direction, what is the smallest incremental steps be
     4. Review the tests
        1. Are there tests that are redundant with each other?
        2. Are there tests that are not testing what their name suggests?
        3. Are there test programs that are too complex and unfocused for the test?
        4. Are there errors or features that are untested?
        5. Is there both a success and failure case for each tested feature?
     5. Review the TODOs in the three files.
        1. For each TODO, describe what changes need to resolve the TODO.
        2. For each TODO, decide whether it is stale, blocked by other features, blocked by refactoring, or if its readily fixable.
     6. Review the major logical pieces and enumerate the places where the implementation diverges from best practices, or established theory, or just differ from mature compilers. 
        1. Is this divergence justified by Silica's design or goals?
        2. If not, what changes should be made to improve the Silica design or compiler implementation?
  3. Having thought through the implementation detials, step back and review the design again.
     1. Based on your above analysis of the design and review of the implementation state, consider how this design document should be updated.
     2. Enumerate the ways where the implementation has diverged from the design.
     3. For each divergence, propose whether the design or implementation should be updated, based on the values of the language.
     4. Apply the design review checklist, step by step, and note what changes need to be made.
  4. Propose a roadmap to bring the implementation closer to the intended design
     1. Based on the gaps and TODOs identified, enumerate the major work items that should be in a roadmap
     2. Identify dependencies and interactions between items in the roadmap, if two items are interdependent, they should be broken into smaller work items.
     3. Ensure work items form a directed acyclic graph (a DAG) and then identify which items are unblocked and may be worked on first.
     4. Of the unblocked work items, decide which should come first, consider ease of implementation and important to the long term goals. Prioritize progress and momentum.
  5. Having followed these steps, reflect on these onboarding instructions and your thinking
     1. Check your previous thinking. Enumerate corrections.
     2. Critique these onboarding instructions. What steps should be expanded or added to better explore the language?
     3. List out your changes to the onboarding instructions
     4. Follow your new instructions.

**Useful Search Terms:** "algebraic effects existential types", "linear types lifetimes", "affine types resource management", "scoped effect handlers", "effect tunneling", "SSA for coroutines", "definite assignment analysis", "immovable types language design", "continuation passing style compilation", "LLVM coroutine intrinsics", "Rust MIR borrow checking", "output reference types", "guaranteed copy elision semantics", "effect handlers driving computation", "asynchronous algebraic effects", "flow sensitive type systems owned types", "typestate analysis implementation", "Hindley-Milner overloading".