# cons_expr TODOs

A prioritized list of features for making cons_expr a practical embedded Scheme-like language for C++ integration.

## Critical (Safety & Correctness)

- [ ] **Fix Division by Zero**
  - Uncomment and implement error handling for division by zero (line 1049)
  - Prevent crashes in embedded contexts

- [ ] **Improved Lexical Scoping**
  - Fix variable capture in closures
  - Fix scoping issues in lambdas
  - Essential for predictable behavior

- [ ] **Memory Usage Optimizer**
  - Implement "defragment" function mentioned in TODOs
  - Critical for long-running embedded scripts with memory constraints
  - Memory leaks in embedded contexts can affect the host application

- [ ] **Better Error Propagation**
  - Ensure errors bubble up properly to C++ caller
  - Add context about what went wrong
  - Allow C++ code to catch and handle script errors gracefully

## High Priority (Core Functionality)

- [ ] **C++ ↔ Script Data Exchange**
  - Streamlined passing of data between C++ and script
  - Simple conversion between C++ types and Scheme types
  - Example: `auto result = evaluator.call<int>("my-function", 10, "string", std::vector{1,2,3})`

- [ ] **Basic Type Predicates**
  - Core set: `number?`, `string?`, `list?`, `procedure?`
  - Essential for type checking within scripts
  - Allows scripts to handle mixed-type data from C++

- [ ] **List Utilities**
  - `length` - Count elements in a list
  - `map` - Transform lists (basic functional building block)
  - `filter` - Filter lists based on predicate
  - These operations are fundamental and tedious to implement in scripts

- [ ] **Transparent C++ Function Registration**
  - Automatic type conversion for registered C++ functions
  - Example: `evaluator.register_function("add", [](int a, int b) { return a + b; })`
  - Simpler than current approach while maintaining type safety

## Medium Priority (Usability & Performance)

- [ ] **Constant Folding**
  - Optimize expressions that can be evaluated at compile time
  - Performance boost for embedded use
  - Makes constexpr evaluation more efficient

- [ ] **Basic Math Functions**
  - Minimal set: `abs`, `min`, `max`
  - Common operations that C++ code might expect

- [ ] **Vector Support**
  - Random access data structure
  - More natural for interfacing with C++ std::vector
  - Useful for passing arrays of data between C++ and script

- [ ] **Script Function Memoization**
  - Cache results of pure functions
  - Performance optimization for embedded use
  - Example: `(define-memoized fibonacci (lambda (n) ...))`

- [ ] **Script Interrupt/Timeout**
  - Allow C++ to interrupt long-running scripts
  - Set execution time limits
  - Essential for embedded use where scripts shouldn't block main application

## Optional Enhancements

- [ ] **Debugging Support**
  - Script debugging facilities
  - Integration with C++ debugging tools
  - Breakpoints, variable inspection
  - Makes embedded scripts easier to maintain

- [ ] **Profiling Tools**
  - Measure script performance
  - Identify hotspots for optimization
  - Useful for optimizing embedded scripts

- [ ] **Sandbox Mode**
  - Restrict which functions a script can access
  - Limit resource usage
  - Important for security in embedded contexts

- [ ] **Script Hot Reloading**
  - Update scripts without restarting application
  - Useful for development and game scripting

- [ ] **Incremental GC**
  - Non-blocking memory management
  - Important for real-time applications

## Implementation Notes

1. **Comparison with Other Embedded Schemes**:
   - Unlike Guile/Chicken: Focus on C++23 integration over standalone usage
   - Unlike TinyScheme: Prioritize constexpr/compile-time evaluation
   - Like ChaiScript: Emphasize tight C++ integration, but with Scheme syntax

2. **Key Differentiation**:
   - Compile-time script evaluation via constexpr
   - No dynamic allocation requirement
   - C++23 features for cleaner integration
   - Fixed buffer sizes for embedded environments

3. **Design Philosophy**:
   - Favor predictable performance over language completeness
   - Favor C++ compatibility over Scheme compatibility
   - Treat scripts as extensions of C++, not standalone programs

4. **Use Cases to Consider**:
   - Game scripting (behaviors, AI)
   - Configuration (loading settings)
   - Rule engines (business logic)
   - UI event handling
   - Embedded device scripting

5. **C++ Integration Best Practices**:
   - Use strong typing when passing data between C++ and script
   - Keep scripts focused on high-level logic
   - Implement performance-critical code in C++
   - Use scripts for parts that need runtime modification