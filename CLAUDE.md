# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Configure: `cmake -S . -B ./build`
- Build: `cmake --build ./build`
- Run tests: `cd ./build && ctest -C Debug`
- Run specific test: `cd ./build && ctest -C Debug -R "unittests.*"` or `ctest -C Debug -R "constexpr.*"`
- Build constexpr tests: `cmake --build ./build --target constexpr_tests`
- Build runtime assertion tests: `cmake --build ./build --target relaxed_constexpr_tests`
- Run a specific test category: `./test/relaxed_constexpr_tests "[category]"`

## Testing
- The library is designed to run both at compile-time and runtime
- `constexpr_tests` target compiles tests with static assertions
  - Will fail to compile if tests fail since they use static assertions
  - Makes debugging difficult as you won't see which specific test failed
- `relaxed_constexpr_tests` target compiles with runtime assertions
  - Preferred for debugging since it shows which specific tests fail
  - Use this target when developing/debugging: 
    ```bash
    cmake --build ./build --target relaxed_constexpr_tests && ./build/test/relaxed_constexpr_tests
    ```

### Catch2 Command Line Arguments
- Run specific test tag: `./build/test/relaxed_constexpr_tests "[tag]"`
- Run tests with specific name: `./build/test/relaxed_constexpr_tests "quote function"`
- Increase verbosity: `./build/test/relaxed_constexpr_tests --verbosity high`
- List all tests: `./build/test/relaxed_constexpr_tests --list-tests`
- List all tags: `./build/test/relaxed_constexpr_tests --list-tags`
- Show help: `./build/test/relaxed_constexpr_tests --help`

### Command-line Debugging
- Prefer using the `cons_expr` command-line tool for quick debugging and iteration
- Build it with: `cmake --build ./build --target cons_expr_cli`
- Test expressions directly with: `./build/src/cons_expr_cli/cons_expr_cli --exec "(expression-to-test)"`
- This is faster than rebuilding and running test suites for quick iteration

### Writing Tests
- All tests should pass in both modes (constexpr and runtime)
- Catch2 is used for testing framework
- Use the TEST_CASE macro with meaningful name and tags
- Split complex tests into smaller, focused tests
- All constexpr tests should use STATIC_CHECK to ensure they can be evaluated at compile time
- When testing parsing functions:
  - Use `std::string_view` when passing string literals
  - Remember `parse()` always returns a list containing the parsed expressions
  - Navigate the result carefully by checking the types at each level
- Separate parse tests (parser_tests.cpp) from evaluation tests (constexpr_tests.cpp)

## Code Style
- C++23 standard
- No C++ extensions (CMAKE_CXX_EXTENSIONS OFF)
- Treat warnings as errors
- Code is header-only library (include/cons_expr)
- Header files follow #ifndef/#define guard pattern
- Entire system is `constexpr` capable unless it uses IO
- Use modern C++ style casts over C-style casts
- Avoid macros completely except for header guards
- Prefer templates, constexpr functions or concepts over macros

## Naming and Structure
- Namespace: lefticus
- Use snake_case for variables and functions
- Classes/structs use PascalCase
- Template parameters use PascalCase
- All objects are immutable once captured

## Error Handling
- Avoid exceptions and dynamic allocations
- Use std::expected for error handling
- Check bounds and sizes before access

## Parser and Expression Structure
- `parse()` function always returns a list containing the parsed expressions
- Even a single expression is wrapped in a list
- Expression types are stored as variants
- Elements can be:
  - Atoms (identifiers, symbols, strings, numbers, booleans)
  - Lists (collections of elements)
  - Literal lists (quoted lists that aren't evaluated)
  - Closures (lambda functions with environment)
  - Errors (with expected and got information)
- The parser handles nested structures, quotes, and comments
- Use string_view for all string literals in parser tests

## Known Issues
- String handling: Special attention needed for escaped quotes in strings
