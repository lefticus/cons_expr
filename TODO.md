# cons_expr TODOs

A prioritized list of features for making cons_expr a practical embedded Scheme-like language for C++ integration.

## Critical (Safety & Correctness)

- [ ] **Optional Safe Numeric Types**
  - Create a new header file `numerics.hpp` with optional safe numeric types
  - Implement `Rational` type for exact fraction arithmetic as a replacement for `int_type`
  - Implement `Safe` template for both integral and floating point types with checked operations
  - Allow users to choose these types to enhance safety and exactness:
    - `Rational<BaseType>` - Exact fraction arithmetic for integer operations
    - `Safe<T>` - Error checking wrapper for any numeric type (int or float)
  - Keep default numeric operations as-is (division by zero will signal/crash)
  - Follow "pay for what you use" principle - users who need safety/exactness should explicitly opt in

- [X] **Improved Lexical Scoping**
  - Fix variable capture in closures
  - Fix scoping issues in lambdas
  - Essential for predictable behavior

- [ ] **Memory Usage Optimizer (Compaction)**
  - Implement non-member "compact" function in utility.hpp as an opt-in feature
  - Use two-phase mark-and-compact approach:
    1. Mark Phase: Identify all reachable elements from global_scope
    2. Compact Phase: Create new containers and remap indices
  - Critical for long-running embedded scripts with memory constraints
  - Allows reclaiming space from unreachable values in fixed-size containers
  - Avoid in-place compaction which is more complex and error-prone

- [ ] **Better Error Propagation**
  - Ensure errors bubble up properly to C++ caller
  - Add context about what went wrong
  - Allow C++ code to catch and handle script errors gracefully
  - Implement container capacity error detection and reporting:
    1. Add detection functions to identify when SmallVector containers enter error state
    2. Propagate container errors during evaluation and parsing
    3. Create specific error types for container overflow errors
    4. Ensure container errors are reported with container-specific context
    5. Add tests to verify correct error reporting for container capacity issues

## High Priority (Core Functionality)

- [ ] **C++ ↔ Script Data Exchange**
  - Expand the existing function call mechanism with container support
  - Add automatic conversion between Scheme lists and C++ containers:
    - std::vector ↔ Scheme lists
    - std::map/std::unordered_map ↔ Scheme association lists
    - std::tuple ↔ Scheme lists of fixed size
  - Add constexpr tests for C++ ↔ Scheme function calls
  - Example goal: `auto result = evaluator.call<int>("my-function", 10, "string", std::vector{1,2,3})`

- [X] **Basic Type Predicates**
  - Core set: `number?`, `string?`, `list?`, `procedure?`, etc.
  - Implemented with a flexible variadic template approach 
  - Essential for type checking within scripts
  - Allows scripts to handle mixed-type data from C++

- [ ] **List Utilities**
  - `length` - Count elements in a list
  - `map` - Transform lists (basic functional building block)
  - `filter` - Filter lists based on predicate
  - `foldl`/`foldr` - Reduce a list to a single value (sum, product, etc.)
  - `reverse` - Reverse a list
  - `member` - Check if an element is in a list
  - `assoc` - Look up key-value pairs in an association list
  - These operations are fundamental and tedious to implement in scripts
  - Implementation should follow functional programming patterns with immutability

- [ ] **Transparent C++ Function Registration**
  - Build on existing template<auto Func> function registration
  - Add support for lambdas and function objects with deduced types
  - Example: `evaluator.register_function("add", [](int a, int b) { return a + b; })`
  - Implement converters for more complex C++ types:
    - Support for std::optional return values
    - Support for std::expected return values for error handling
    - Support for user-defined types with conversion traits
  - Create a cleaner API that maintains type safety but reduces template verbosity

## Medium Priority (Usability & Performance)

- [ ] **Add `letrec` Support**
  - Support recursive bindings in `let` expressions
  - Support mutual recursion without forward declarations
  - Follow standard Scheme semantics for `letrec`
  - Implementation approach:
    - Build on existing self-referential closure mechanism
    - Create a new scope where all variables are pre-defined (but uninitialized)
    - Evaluate right-hand sides in that scope
    - Bind results to the pre-defined variables
    - Syntax: `(letrec ((name1 value1) (name2 value2) ...) body ...)`
  - This complements the current `let` which uses sequential binding

- [ ] **Constant Folding**
  - Optimize expressions that can be evaluated at compile time
  - Performance boost for embedded use
  - Makes constexpr evaluation more efficient
  - Implementation strategy:
    - Add a "pure" flag to function pointers that guarantees no side effects
    - During parsing phase, identify expressions with only pure operations
    - Pre-evaluate these expressions and replace with their result
    - Add caching for common constant expressions
    - Implementation should preserve semantics exactly
  - Potential optimizations:
    - Arithmetic expressions with constant operands: `(+ 1 2 3)` → `6`
    - Constant string operations: `(string-append "hello" " " "world")` → `"hello world"`
    - Pure function calls with constant arguments
    - Condition expressions with constant predicates: `(if true x y)` → `x`

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

6. **Safe Numerics Implementation Plan**:
   - **Design Goals**:
     - Provide optional numeric types with guaranteed safety
     - Make them drop-in replacements for standard numeric types
     - Support both C++ and Scheme semantics
     - Maintain constexpr compatibility
   
   - **Components**:
     1. **Rational<BaseType>**:
        - Exact representation of fractions (e.g., 1/3) without rounding errors
        - Replace int_type for exact arithmetic
        - Store as numerator/denominator pair of BaseType
        - Support all basic operations while preserving exactness
        - Detect division by zero and handle gracefully
        - Optional normalization (dividing by GCD)
        - Example:
          ```cpp
          template<std::integral BaseType>
          struct Rational {
              BaseType numerator;
              BaseType denominator;  // never zero
              
              // Various arithmetic operations...
              constexpr Rational operator+(const Rational& other) const;
              constexpr Rational operator/(const Rational& other) const {
                  if (other.numerator == 0) {
                      // Handle division by zero - could set error flag or return NaN equivalent
                  }
                  return Rational{numerator * other.denominator, denominator * other.numerator};
              }
          };
          ```
     
     2. **Safe<T>**:
        - Wrapper around any numeric type with error checking
        - Can be used for both int_type and real_type
        - Detect overflow, underflow, division by zero
        - Hold error state internally
        - Example:
          ```cpp
          template<typename T>
          struct Safe {
              T value;
              bool error_state = false;
              
              constexpr Safe operator/(const Safe& other) const {
                  if (other.value == 0) {
                      return Safe{0, true}; // Error state true
                  }
                  return Safe{value / other.value};
              }
          };
          ```
   
   - **Integration Strategy**:
     ```cpp
     // Example usage in cons_expr instances:
     
     // Use Rational for exact arithmetic with fractions
     using ExactEval = lefticus::cons_expr<
         std::uint16_t,
         char,
         lefticus::Rational<int>,  // Replace int_type with Rational
         double                    // Keep regular floating point
     >;
     
     // Use Safe wrappers for error detection
     using SafeEval = lefticus::cons_expr<
         std::uint16_t,
         char,
         lefticus::Safe<int>,     // Safe integer operations
         lefticus::Safe<double>   // Safe floating point operations
     >;
     
     // Combine both approaches
     using SafeExactEval = lefticus::cons_expr<
         std::uint16_t,
         char,
         lefticus::Safe<lefticus::Rational<int>>,  // Safe exact arithmetic
         lefticus::Safe<double>                    // Safe floating point
     >;
     ```

7. **List Utilities Implementation Plan**:
   - **Design Goals**:
     - Provide standard functional list operations
     - Maintain immutability of data
     - Support both literal_list_type and list_type where appropriate
     - Follow Scheme/LISP conventions
     - Maximize constexpr compatibility
   
   - **Core Functions**:
     1. **length**:
        ```cpp
        // Basic list length calculation
        [[nodiscard]] static constexpr SExpr length(cons_expr &engine, LexicalScope &scope, list_type params)
        {
          if (params.size != 1) { return engine.make_error(str("(length list)"), params); }
          
          const auto list_result = engine.eval_to<literal_list_type>(scope, engine.values[params[0]]);
          if (!list_result) { return engine.make_error(str("expected list"), list_result.error()); }
          
          return SExpr{ Atom(static_cast<int_type>(list_result->items.size)) };
        }
        ```
     
     2. **map**:
        ```cpp
        // Transform a list by applying a function to each element
        [[nodiscard]] static constexpr SExpr map(cons_expr &engine, LexicalScope &scope, list_type params)
        {
          if (params.size != 2) { return engine.make_error(str("(map function list)"), params); }
          
          const auto func = engine.eval(scope, engine.values[params[0]]);
          const auto list_result = engine.eval_to<literal_list_type>(scope, engine.values[params[1]]);
          
          if (!list_result) { return engine.make_error(str("expected list"), list_result.error()); }
          
          // Create a new list with the results of applying the function to each element
          Scratch result{ engine.object_scratch };
          
          for (const auto &item : engine.values[list_result->items]) {
            // Apply function to each item
            std::array<SExpr, 1> args{ item };
            const auto mapped_item = engine.invoke_function(scope, func, engine.values.insert_or_find(args));
            
            // Check for container errors after each operation
            if (engine.has_container_error()) {
              return engine.make_container_error();
            }
            
            result.push_back(mapped_item);
          }
          
          return SExpr{ LiteralList{ engine.values.insert_or_find(result) } };
        }
        ```
     
     3. **filter**:
        ```cpp
        // Filter a list based on a predicate function
        [[nodiscard]] static constexpr SExpr filter(cons_expr &engine, LexicalScope &scope, list_type params)
        {
          if (params.size != 2) { return engine.make_error(str("(filter predicate list)"), params); }
          
          const auto pred = engine.eval(scope, engine.values[params[0]]);
          const auto list_result = engine.eval_to<literal_list_type>(scope, engine.values[params[1]]);
          
          if (!list_result) { return engine.make_error(str("expected list"), list_result.error()); }
          
          // Create a new list with only elements that satisfy the predicate
          Scratch result{ engine.object_scratch };
          
          for (const auto &item : engine.values[list_result->items]) {
            // Apply predicate to each item
            std::array<SExpr, 1> args{ item };
            const auto pred_result = engine.invoke_function(scope, pred, engine.values.insert_or_find(args));
            
            // Check if predicate returned true
            const auto bool_result = engine.eval_to<bool>(scope, pred_result);
            if (!bool_result) {
              return engine.make_error(str("predicate must return boolean"), pred_result);
            }
            
            // Add item to result if predicate is true
            if (*bool_result) {
              result.push_back(item);
            }
          }
          
          return SExpr{ LiteralList{ engine.values.insert_or_find(result) } };
        }
        ```
     
   - **Additional Functions**:
     - `foldl`/`foldr` for reduction operations
     - `reverse` for creating a reversed copy of a list
     - `member` for checking list membership
     - `assoc` for working with association lists (key-value pairs)
   
   - **Registration**:
     ```cpp
     // Add to consteval cons_expr() constructor
     add(str("length"), SExpr{ FunctionPtr{ length, FunctionPtr::Type::other } });
     add(str("map"), SExpr{ FunctionPtr{ map, FunctionPtr::Type::other } });
     add(str("filter"), SExpr{ FunctionPtr{ filter, FunctionPtr::Type::other } });
     // Add other list utility functions...
     ```

8. **Memory Compaction Implementation Plan**:
   - **Design Goals**:
     - Create a non-member utility function for memory compaction
     - Safely reduce memory usage by removing unreachable items
     - Preserve all reachable values with correct indexing
     - Support constexpr operation
     - Zero dynamic allocation

   - **Implementation Strategy**:
     ```cpp
     // Non-member compact function in utility.hpp
     template<ConsExpr Eval>
     constexpr void compact(Eval& evaluator) {
       using size_type = typename Eval::size_type;
       
       // Phase 1: Mark all reachable elements
       std::array<bool, Eval::BuiltInStringsSize> string_reachable{};
       std::array<bool, Eval::BuiltInValuesSize> value_reachable{};
       
       // Start from global scope and recursively mark everything reachable
       for (const auto& [name, value] : evaluator.global_scope) {
         mark_reachable_string(name, string_reachable, evaluator);
         mark_reachable_value(value, string_reachable, value_reachable, evaluator);
       }
       
       // Phase 2: Build index mapping tables
       std::array<size_type, Eval::BuiltInStringsSize> string_index_map{};
       std::array<size_type, Eval::BuiltInValuesSize> value_index_map{};
       
       size_type new_string_idx = 0;
       for (size_type i = 0; i < evaluator.strings.small_size_used; ++i) {
         if (string_reachable[i]) {
           string_index_map[i] = new_string_idx++;
         }
       }
       
       size_type new_value_idx = 0;
       for (size_type i = 0; i < evaluator.values.small_size_used; ++i) {
         if (value_reachable[i]) {
           value_index_map[i] = new_value_idx++;
         }
       }
       
       // Phase 3: Create new containers with only reachable elements
       auto new_strings = evaluator.strings;
       auto new_values = evaluator.values;
       auto new_global_scope = evaluator.global_scope;
       
       // Reset counters
       new_strings.small_size_used = 0;
       new_values.small_size_used = 0;
       new_global_scope.small_size_used = 0;
       
       // Copy and remap strings
       for (size_type i = 0; i < evaluator.strings.small_size_used; ++i) {
         if (string_reachable[i]) {
           new_strings.small[string_index_map[i]] = evaluator.strings.small[i];
           new_strings.small_size_used++;
         }
       }
       
       // Copy and remap values (recursively update all indices)
       for (size_type i = 0; i < evaluator.values.small_size_used; ++i) {
         if (value_reachable[i]) {
           new_values.small[value_index_map[i]] = rewrite_indices(
             evaluator.values.small[i], string_index_map, value_index_map);
           new_values.small_size_used++;
         }
       }
       
       // Rebuild global scope with remapped indices
       for (const auto& [name, value] : evaluator.global_scope) {
         using string_type = typename Eval::string_type;
         
         string_type new_name{string_index_map[name.start], name.size};
         auto new_value = rewrite_indices(value, string_index_map, value_index_map);
         
         new_global_scope.push_back({new_name, new_value});
       }
       
       // Replace the old containers with the new ones
       evaluator.strings = std::move(new_strings);
       evaluator.values = std::move(new_values);
       evaluator.global_scope = std::move(new_global_scope);
       
       // Reset error states that may have been set
       evaluator.strings.error_state = false;
       evaluator.values.error_state = false;
       evaluator.global_scope.error_state = false;
     }
     
     // Helper function to mark reachable strings
     template<ConsExpr Eval>
     constexpr void mark_reachable_string(
       const typename Eval::string_type& str, 
       std::array<bool, Eval::BuiltInStringsSize>& string_reachable,
       const Eval& evaluator) {
       // Mark the string itself
       string_reachable[str.start] = true;
     }
     
     // Helper function to mark reachable values recursively
     template<ConsExpr Eval>
     constexpr void mark_reachable_value(
       const typename Eval::SExpr& expr,
       std::array<bool, Eval::BuiltInStringsSize>& string_reachable,
       std::array<bool, Eval::BuiltInValuesSize>& value_reachable,
       const Eval& evaluator) {
       
       // Handle different variant types in SExpr
       std::visit([&](const auto& value) {
         using T = std::decay_t<decltype(value)>;
         
         if constexpr (std::is_same_v<T, typename Eval::Atom>) {
           // Handle atomic types
           std::visit([&](const auto& atom) {
             using AtomT = std::decay_t<decltype(atom)>;
             
             // Mark strings in atoms
             if constexpr (std::is_same_v<AtomT, typename Eval::string_type> ||
                           std::is_same_v<AtomT, typename Eval::identifier_type> ||
                           std::is_same_v<AtomT, typename Eval::symbol_type>) {
               mark_reachable_string(atom, string_reachable, evaluator);
             }
             // Other atom types don't contain references
           }, value);
         } 
         else if constexpr (std::is_same_v<T, typename Eval::list_type>) {
           // Mark all elements in the list
           value_reachable[value.start] = true;
           for (size_type i = 0; i < value.size; ++i) {
             const auto& list_item = evaluator.values.small[value.start + i];
             mark_reachable_value(list_item, string_reachable, value_reachable, evaluator);
           }
         }
         else if constexpr (std::is_same_v<T, typename Eval::literal_list_type>) {
           // Mark all elements in the literal list
           mark_reachable_value(
             typename Eval::SExpr{value.items}, 
             string_reachable, value_reachable, evaluator);
         }
         else if constexpr (std::is_same_v<T, typename Eval::Closure>) {
           // Mark parameter names and statements
           value_reachable[value.parameter_names.start] = true;
           value_reachable[value.statements.start] = true;
           
           // Mark all parameter names
           for (size_type i = 0; i < value.parameter_names.size; ++i) {
             mark_reachable_value(
               evaluator.values.small[value.parameter_names.start + i],
               string_reachable, value_reachable, evaluator);
           }
           
           // Mark all statements
           for (size_type i = 0; i < value.statements.size; ++i) {
             mark_reachable_value(
               evaluator.values.small[value.statements.start + i],
               string_reachable, value_reachable, evaluator);
           }
           
           // Mark self identifier if present
           if (value.has_self_reference()) {
             mark_reachable_string(value.self_identifier, string_reachable, evaluator);
           }
         }
         // Other types like FunctionPtr don't contain references to track
       }, expr.value);
     }
     
     // Helper function to recursively rewrite indices in all data structures
     template<ConsExpr Eval>
     constexpr typename Eval::SExpr rewrite_indices(
       const typename Eval::SExpr& expr,
       const std::array<typename Eval::size_type, Eval::BuiltInStringsSize>& string_map,
       const std::array<typename Eval::size_type, Eval::BuiltInValuesSize>& value_map) {
       
       using SExpr = typename Eval::SExpr;
       
       return std::visit([&](const auto& value) -> SExpr {
         using T = std::decay_t<decltype(value)>;
         
         if constexpr (std::is_same_v<T, typename Eval::Atom>) {
           // Rewrite indices in atom types if needed
           return SExpr{std::visit([&](const auto& atom) {
             using AtomT = std::decay_t<decltype(atom)>;
             
             if constexpr (std::is_same_v<AtomT, typename Eval::string_type>) {
               return typename Eval::Atom{typename Eval::string_type{
                 string_map[atom.start], atom.size}};
             }
             else if constexpr (std::is_same_v<AtomT, typename Eval::identifier_type>) {
               return typename Eval::Atom{typename Eval::identifier_type{
                 string_map[atom.start], atom.size}};
             }
             else if constexpr (std::is_same_v<AtomT, typename Eval::symbol_type>) {
               return typename Eval::Atom{typename Eval::symbol_type{
                 string_map[atom.start], atom.size}};
             }
             else {
               // Other atoms don't need remapping
               return typename Eval::Atom{atom};
             }
           }, value)};
         }
         else if constexpr (std::is_same_v<T, typename Eval::list_type>) {
           // Remap list indices
           return SExpr{typename Eval::list_type{
             value_map[value.start], value.size}};
         }
         else if constexpr (std::is_same_v<T, typename Eval::literal_list_type>) {
           // Remap literal list indices
           return SExpr{typename Eval::literal_list_type{
             typename Eval::list_type{value_map[value.items.start], value.items.size}}};
         }
         else if constexpr (std::is_same_v<T, typename Eval::Closure>) {
           // Remap closure indices
           typename Eval::Closure new_closure;
           new_closure.parameter_names = {
             value_map[value.parameter_names.start], value.parameter_names.size};
           new_closure.statements = {
             value_map[value.statements.start], value.statements.size};
           
           // Remap self identifier if present
           if (value.has_self_reference()) {
             new_closure.self_identifier = {
               string_map[value.self_identifier.start], value.self_identifier.size};
           }
           
           return SExpr{new_closure};
         }
         else {
           // Other types like FunctionPtr don't contain indices
           return SExpr{value};
         }
       }, expr.value);
     }
     ```

9. **Container Error Detection Plan**:
   - **Problems**: 
     1. SmallVector sets error_state flags when capacity limits are exceeded, but these errors are not currently propagated or reported
     2. **Critical Issue**: SmallVector's higher-level insert methods don't check for failures:
        - The base insert() sets error_state when capacity is exceeded but returns a potentially invalid index
        - insert_or_find() and insert(SpanType values) call the base insert() but don't check if it succeeded
        - These methods continue to use potentially invalid indices from the base insert()
        - This propagates bad values into the KeyType results and makes overflow errors extremely difficult to debug
        - Need to ensure these methods check error_state and handle failures appropriately
   - **Root cause**: Running out of capacity in one of the fixed-size containers:
     - global_scope: Fixed number of symbols/variables
     - strings: Fixed space for string data
     - values: Fixed number of SExpr values
     - Various scratch spaces used during evaluation
   - **Implementation Strategy**:
     - Phase 1 - Error Detection:
       - Add helper method to detect error states in all containers
       - Check both global and local scope objects
       - Check all containers at key points during evaluation
     - Phase 2 - Error Propagation:
       - Modify evaluation functions to check for errors before/after operations
       - Propagate container errors to the caller via error SExpr
       - Ensure error states from containers bubble up through the call stack
     - Phase 3 - Error Reporting:
       - Create specific error messages for different container types
       - Include container size/capacity information in error messages
       - Add helper to identify which specific container is in error state
       - **Critical**: Handle the circular dependency where creating error strings might itself fail:
         - Pre-allocate/reserve all error message strings during initialization
         - Or use numeric error codes that don't require string allocation
         - Or implement a fallback mechanism that avoids string allocation for error reports
         - Ensure error reporting path doesn't allocate additional strings when strings container is full
     - Phase 4 - Testing Plan:
       1. **Test global_scope overflow**:
          - Create a test that defines variables until global_scope capacity is exceeded
          - Verify correct error code/message is returned
          - Check that subsequent evaluation operations fail appropriately
       
       2. **Test strings table overflow**:
          - Create a test that adds unique strings until strings capacity is exceeded
          - Verify overflow is detected and reported correctly
          - Test both direct string creation and indirect string creation (via identifiers)
       
       3. **Test values table overflow**:
          - Create a test with deeply nested expressions that exceed values capacity
          - Create a test with many list elements that exceed values capacity
          - Verify appropriate errors are generated
       
       4. **Test scratch space overflows**:
          - Create tests that overflow each scratch space (object_scratch, string_scratch, etc.)
          - Verify errors are propagated correctly to the caller
       
       5. **Test local scope overflow**:
          - Create a test with deeply nested lexical scopes or many local variables
          - Verify scope overflow errors are detected
       
       6. **Test error propagation paths**:
          - Test that errors propagate correctly through eval, parse, and other functions
          - Verify that container errors take precedence over other errors
       
       7. **Test error reporting mechanism**:
          - Verify that container errors can be reported even when strings container is full
          - Test fallback mechanisms for error reporting
       
       8. **Integration tests**:
          - Test interaction between various overflow scenarios
          - Verify that the system remains in a stable state after overflow
       
       9. **Test Implementation Considerations**:
          - **Initialization vs. Runtime Overflow**: 
            - Container sizes must be large enough to accommodate built-ins
            - Test both initialization failure and runtime overflow separately
          
          - **Testing Approaches**:
            1. **Staged Overflow Testing**: 
               - Start with containers just large enough for initialization
               - Then incrementally add more items until each container overflows
               - Use custom subclass or wrapper that exposes current capacity usage
            
            2. **Container-Specific Testing**:
               - For global_scope: Test with many variable definitions
               - For strings: Test with many unique string literals
               - For values: Test with deeply nested expressions or long lists
               - For scratch spaces: Test operations that heavily use each scratch space
            
            3. **Custom Construction Testing**:
               - Create a test helper that allows partial initialization
               - Skip adding built-ins that aren't needed for specific tests
               - Use smaller containers for specific overflow scenarios
            
            4. **Two-Phase Testing**:
               - Phase 1: Test error detection during initialization
               - Phase 2: Test error detection during evaluation
            
            5. **SmallVector Insert Methods Testing**:
               - Create unit tests specifically for the SmallVector class
               - Test insert() with exact capacity limits to verify error_state is set correctly
               - Test insert(SpanType) with values that exceed capacity
               - Test insert_or_find() with values that exceed capacity
               - Verify returned KeyType values are safe and valid even in error cases
               - Check that partially inserted values are handled correctly
   - **Expected Result**:
     - Clearer error messages when capacity limits are reached
     - Better debugging experience when working with constrained container sizes
     - More robust error handling in embedded environments
   - **Core Implementation Strategy**:
     1. **Fix SmallVector Higher-Level Insert Methods**:
        ```cpp
        // Current problematic implementation of insert(SpanType)
        constexpr KeyType insert(SpanType values) noexcept
        {
          size_type last = 0;
          for (const auto &value : values) { last = insert(value); }
          return KeyType{ static_cast<size_type>(last - values.size() + 1), static_cast<size_type>(values.size()) };
        }
        
        // Fix: Check error_state after each insert and return a safe KeyType on error
        constexpr KeyType insert(SpanType values) noexcept
        {
          if (values.empty()) { return KeyType{0, 0}; } // Safe empty KeyType
          
          const auto start_idx = small_size_used;
          size_type inserted = 0;
          
          for (const auto &value : values) {
            const auto idx = insert(value);
            if (error_state) {
              // We hit capacity - return a KeyType with the correct elements we did manage to insert
              return KeyType{start_idx, inserted};
            }
            inserted++;
          }
          
          return KeyType{start_idx, inserted};
        }
        
        // Current problematic implementation of insert_or_find
        constexpr KeyType insert_or_find(SpanType values) noexcept
        {
          if (const auto small_found = std::search(begin(), end(), values.begin(), values.end()); small_found != end()) {
            return KeyType{ static_cast<size_type>(std::distance(begin(), small_found)),
              static_cast<size_type>(values.size()) };
          } else {
            return insert(values); // Doesn't check if insert succeeded
          }
        }
        
        // Fix: Check error_state after insert and handle appropriately
        constexpr KeyType insert_or_find(SpanType values) noexcept
        {
          if (const auto small_found = std::search(begin(), end(), values.begin(), values.end()); small_found != end()) {
            return KeyType{ static_cast<size_type>(std::distance(begin(), small_found)),
              static_cast<size_type>(values.size()) };
          } else {
            const auto before_error = error_state;
            const auto result = insert(values);
            
            // If we had no error before but have one now, the insert failed
            if (!before_error && error_state) {
              // Could return a special error KeyType or just the best approximation we have
              // For safety, might want to return KeyType{0, 0} to avoid propagating bad indices
            }
            
            return result;
          }
        }
        ```
     2. **Container Error Detection**:
        ```cpp
        // Add method to check container error states
        [[nodiscard]] constexpr bool has_container_error() const noexcept {
          return global_scope.error_state || 
                 strings.error_state || 
                 values.error_state || 
                 object_scratch.error_state || 
                 variables_scratch.error_state || 
                 string_scratch.error_state;
        }
        
        // Add method to check scope error state
        [[nodiscard]] constexpr bool has_scope_error(const LexicalScope &scope) const noexcept {
          return scope.error_state;
        }
        
        // Add method to check all error states including passed scope
        [[nodiscard]] constexpr bool has_any_error(const LexicalScope &scope) const noexcept {
          return has_container_error() || has_scope_error(scope);
        }
        ```
     
     2. **Error Checking in Evaluation**:
        ```cpp
        [[nodiscard]] constexpr SExpr eval(LexicalScope &scope, const SExpr expr) {
          // Check for container errors first
          if (has_any_error(scope)) {
            return create_container_error(scope);
          }
          
          // Existing evaluation logic...
          
          // Check again after evaluation
          if (has_any_error(scope)) {
            return create_container_error(scope);
          }
          
          return result;
        }
        ```
        
   - **Possible Error Reporting Approaches**:
     1. **Pre-allocation Strategy**:
        - Reserve a set of predefined error strings during initialization
        - Use indices instead of direct references for error messages
        - This ensures error reporting never needs to allocate new strings
     2. **Error Code Strategy**:
        - Define an enum of error codes (e.g., STRING_CAPACITY_EXCEEDED)
        - Return error codes directly inside the Error type
        - Let the hosting application map codes to messages
     3. **Two-Phase Error Reporting**:
        - Add a "container_error_type" field to Error type
        - When container errors occur, set numeric type without creating strings
        - Only generate detailed error messages if string container has capacity
        - Fall back to generic error codes when strings are full
     4. **Extend Error Type**:
        - Modify Error type to hold either string reference or direct error code
        - Avoid string allocation when reporting container capacity errors
        - Use the direct error code path when strings container is full
   - **Example Implementation Sketch**:
     ```cpp
     // Add error codes enum
     enum struct ContainerErrorCode : std::uint8_t {
       NONE,
       GLOBAL_SCOPE_FULL,
       STRINGS_FULL,
       VALUES_FULL,
       SCRATCH_SPACE_FULL
     };
     
     // Modify Error struct to include container error code
     template<std::unsigned_integral SizeType> struct Error {
       using size_type = SizeType;
       IndexedString<size_type> expected; // Existing field
       IndexedList<size_type> got;        // Existing field
       ContainerErrorCode container_error{ContainerErrorCode::NONE}; // New field
       
       // Constructor for regular errors (unchanged)
       constexpr Error(IndexedString<size_type> exp, IndexedList<size_type> g) 
         : expected(exp), got(g), container_error(ContainerErrorCode::NONE) {}
       
       // New constructor for container errors (no string allocation)
       constexpr Error(ContainerErrorCode code) 
         : expected{0, 0}, got{0, 0}, container_error(code) {}
         
       [[nodiscard]] constexpr bool is_container_error() const { 
         return container_error != ContainerErrorCode::NONE; 
       }
     };
     
     // Then usage would be like:
     if (strings.error_state) {
       return SExpr{Error{ContainerErrorCode::STRINGS_FULL}};
     }
     ```

## Coverage Analysis

### How to Run Branch Coverage Report

The project has a pre-configured `build-coverage` directory for generating coverage reports. To run a branch coverage analysis:

```bash
# 1. Build the coverage-configured project (don't reconfigure!)
cmake --build ./build-coverage

# 2. Run all tests to generate coverage data
cd ./build-coverage && ctest

# 3. Generate branch coverage report for cons_expr.hpp
cd /home/jason/cons_expr/build-coverage
gcovr --txt-metric branch --filter ../include/cons_expr/cons_expr.hpp --gcov-ignore-errors=no_working_dir_found .
```

**Note**: The `--gcov-ignore-errors=no_working_dir_found` flag is needed to ignore errors from dependency coverage data (Catch2, etc.) that we don't need for our analysis.

## Branch Coverage Tests to Add

Based on coverage analysis showing 36% branch coverage for `include/cons_expr/cons_expr.hpp`, these specific test cases should be added to improve coverage to ~55-65%.

**IMPORTANT**: All tests must use `STATIC_CHECK` and be constexpr-capable for compatibility with the `constexpr_tests` target. Follow existing test patterns in `constexpr_tests.cpp`.

### 1. **SmallVector Overflow Tests** (Lines 187, 192) - **HIGH PRIORITY**
**File**: `constexpr_tests.cpp`
```cpp
TEST_CASE("SmallVector overflow scenarios", "[utility]") {
    constexpr auto test = []() constexpr {
        // Create engine with smaller capacity for testing
        cons_expr<32, char, int, double> engine;  // Reduced capacity
        
        // Test error state after exceeding capacity
        for (int i = 0; i < 35; ++i) {  // Exceed capacity
            engine.values.insert(engine.True);
        }
        return engine.values.error_state;
    };
    
    STATIC_CHECK(test());
    
    constexpr auto test2 = []() constexpr {
        cons_expr<32, char, int, double> engine;
        
        // Test string capacity overflow
        for (int i = 0; i < 100; ++i) {
            std::string_view test_str = "test_string_content";
            engine.strings.insert(std::span{test_str.data(), test_str.size()});
        }
        return engine.strings.error_state;
    };
    
    STATIC_CHECK(test2());
}
```

### 2. **Number Parsing Edge Cases** (Lines 263, 283, 288, 296, 310, 319, 334, 343, 351) - **HIGH PRIORITY**
**File**: `constexpr_tests.cpp`
```cpp
TEST_CASE("Number parsing edge cases", "[parser]") {
    constexpr auto test_lone_minus = []() constexpr {
        // Test lone minus sign
        auto result = parse_number<int>("-");
        return !result.first;  // Should fail parsing
    };
    STATIC_CHECK(test_lone_minus());
    
    constexpr auto test_scientific_notation = []() constexpr {
        // Test 'e'/'E' notation variations
        auto float_result = parse_number<double>("123e5");
        return float_result.first && (float_result.second == 12300000.0);
    };
    STATIC_CHECK(test_scientific_notation());
    
    constexpr auto test_invalid_exponent = []() constexpr {
        // Test invalid exponent characters
        auto bad_exp = parse_number<double>("1.5eZ");
        return !bad_exp.first;  // Should fail
    };
    STATIC_CHECK(test_invalid_exponent());
    
    constexpr auto test_incomplete_exponent = []() constexpr {
        // Test incomplete exponent (starts but no digits)
        auto incomplete_exp = parse_number<double>("1.5e");
        return !incomplete_exp.first;  // Should fail
    };
    STATIC_CHECK(test_incomplete_exponent());
    
    constexpr auto test_negative_exponent = []() constexpr {
        // Test negative exponent
        auto neg_exp = parse_number<double>("1.5e-2");
        return neg_exp.first && (neg_exp.second == 0.015);
    };
    STATIC_CHECK(test_negative_exponent());
}
```

### 3. **Parser Null Pointer Handling** (Lines 601, 639, 651) - **HIGH PRIORITY**
**File**: `constexpr_tests.cpp`
```cpp
TEST_CASE("Parser safety edge cases", "[parser]") {
    constexpr auto test_null_pointer = []() constexpr {
        cons_expr<> engine;
        
        // Test null sexpr in get_if
        const decltype(engine)::SExpr* null_ptr = nullptr;
        auto result = engine.get_if<int>(null_ptr);
        return result == nullptr;
    };
    STATIC_CHECK(test_null_pointer());
    
    constexpr auto test_unterminated_string = []() constexpr {
        cons_expr<> engine;
        
        // Test unterminated string in parser
        auto [parsed, remaining] = engine.parse("\"unterminated");
        if (parsed.size == 0) return false;
        
        auto& first_expr = engine.values[parsed[0]];
        return std::holds_alternative<decltype(engine)::error_type>(first_expr.value);
    };
    STATIC_CHECK(test_unterminated_string());
}
```

### 4. **Token Parsing Edge Cases** (Lines 367, 372, 389, 392, 410, 415, 417) - **MEDIUM PRIORITY**
**File**: `constexpr_tests.cpp`
```cpp
TEST_CASE("Token parsing edge cases", "[parser]") {
    constexpr auto test_line_endings = []() constexpr {
        // Test end-of-line characters
        auto token1 = next_token("\r\n   token");
        return token1.parsed == "token";
    };
    STATIC_CHECK(test_line_endings());
    
    constexpr auto test_quote_character = []() constexpr {
        // Test quote character
        auto token2 = next_token("'symbol");
        return token2.parsed == "'";
    };
    STATIC_CHECK(test_quote_character());
    
    constexpr auto test_parentheses = []() constexpr {
        // Test parentheses
        auto token3 = next_token(")rest");
        return token3.parsed == ")";
    };
    STATIC_CHECK(test_parentheses());
    
    constexpr auto test_unterminated_string_token = []() constexpr {
        // Test unterminated string
        auto token4 = next_token("\"unterminated string");
        return token4.parsed == "\"unterminated string";
    };
    STATIC_CHECK(test_unterminated_string_token());
    
    constexpr auto test_empty_token = []() constexpr {
        // Test empty token at end
        auto token5 = next_token("");
        return token5.parsed.empty();
    };
    STATIC_CHECK(test_empty_token());
}
```

### 5. **String Escape Processing** (Lines 494, 538, 548) - **MEDIUM PRIORITY**
**File**: `constexpr_tests.cpp`
```cpp
TEST_CASE("String escape edge cases", "[strings]") {
    constexpr auto test_error_equality = []() constexpr {
        cons_expr<> engine;
        
        // Test error type equality comparison
        auto error1 = engine.make_error("test error", engine.empty_indexed_list);
        auto error2 = engine.make_error("test error", engine.empty_indexed_list);
        auto err1 = std::get<decltype(engine)::error_type>(error1.value);
        auto err2 = std::get<decltype(engine)::error_type>(error2.value);
        return err1 == err2;
    };
    STATIC_CHECK(test_error_equality());
    
    constexpr auto test_unknown_escape = []() constexpr {
        cons_expr<> engine;
        
        // Test unknown escape character
        auto bad_escape = engine.process_string_escapes("test\\q");
        return std::holds_alternative<decltype(engine)::error_type>(bad_escape.value);
    };
    STATIC_CHECK(test_unknown_escape());
    
    constexpr auto test_unterminated_escape = []() constexpr {
        cons_expr<> engine;
        
        // Test unterminated escape (string ends with backslash)
        auto unterminated = engine.process_string_escapes("test\\");
        return std::holds_alternative<decltype(engine)::error_type>(unterminated.value);
    };
    STATIC_CHECK(test_unterminated_escape());
}
```

### 6. **Quote Depth Handling** (Lines 745, 754, 762-773) - **MEDIUM PRIORITY**
**File**: `constexpr_tests.cpp`
```cpp
TEST_CASE("Quote depth handling", "[parser]") {
    constexpr auto test_multiple_quotes = []() constexpr {
        cons_expr<> engine;
        
        // Test multiple quote levels
        auto [parsed, _] = engine.parse("'''symbol");
        return parsed.size == 1;
    };
    STATIC_CHECK(test_multiple_quotes());
    
    constexpr auto test_quote_booleans = []() constexpr {
        cons_expr<> engine;
        
        // Test quote with different token types
        auto [parsed2, _2] = engine.parse("'true");
        auto [parsed3, _3] = engine.parse("'false");
        return parsed2.size == 1 && parsed3.size == 1;
    };
    STATIC_CHECK(test_quote_booleans());
    
    constexpr auto test_quote_literals = []() constexpr {
        cons_expr<> engine;
        
        // Test quote with strings, numbers
        auto [parsed4, _4] = engine.parse("'\"hello\"");
        auto [parsed5, _5] = engine.parse("'123");
        auto [parsed6, _6] = engine.parse("'123.45");
        return parsed4.size == 1 && parsed5.size == 1 && parsed6.size == 1;
    };
    STATIC_CHECK(test_quote_literals());
}
```

### 7. **Error Propagation** (Lines 779, 780, 784-796) - **LOWER PRIORITY**
**File**: `constexpr_tests.cpp`
```cpp
TEST_CASE("Float vs int parsing priority", "[parser]") {
    constexpr auto test_float_parsing = []() constexpr {
        cons_expr<> engine;
        
        // Test case where int parsing fails but float parsing succeeds
        auto [parsed, _] = engine.parse("123.456");
        if (parsed.size == 0) return false;
        
        auto& expr = engine.values[parsed[0]];
        auto* atom = std::get_if<decltype(engine)::Atom>(&expr.value);
        if (atom == nullptr) return false;
        
        return std::holds_alternative<double>(*atom);
    };
    STATIC_CHECK(test_float_parsing());
    
    constexpr auto test_identifier_fallback = []() constexpr {
        cons_expr<> engine;
        
        // Test case where both int and float parsing fail
        auto [parsed2, _2] = engine.parse("not_a_number");
        if (parsed2.size == 0) return false;
        
        auto& expr2 = engine.values[parsed2[0]];
        auto* atom2 = std::get_if<decltype(engine)::Atom>(&expr2.value);
        if (atom2 == nullptr) return false;
        
        return std::holds_alternative<decltype(engine)::identifier_type>(*atom2);
    };
    STATIC_CHECK(test_identifier_fallback());
}
```

### **Implementation Priority & Expected Impact**:
1. **Phase 1**: SmallVector overflow + Number parsing + Null pointer handling (should get coverage to ~48-52%)
2. **Phase 2**: Token parsing + String escape processing (should get coverage to ~52-58%)  
3. **Phase 3**: Quote depth + Error propagation (should get coverage to ~55-65%)

### **Test Organization**:
- **ALL tests must be added to the `constexpr_tests` target** and use `STATIC_CHECK` patterns
- Tests can be added to existing test files or new test files as appropriate
- Tests must be evaluable at compile-time to work with the `constexpr_tests` target
- Follow the existing patterns in the constexpr test files for consistency
- Use reduced template parameters (e.g., `cons_expr<32, char, int, double>`) for overflow testing
