#include <catch2/catch_test_macros.hpp>

#include <cons_expr/cons_expr.hpp>
#include <cstdint>
#include <string_view>
#include <variant>

using IntType = int;
using FloatType = double;

namespace {
template<typename Result> constexpr Result evaluate_to(std::string_view input)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  return evaluator.evaluate_to<Result>(input).value();
}

template<typename Result> constexpr bool evaluate_expected(std::string_view input, auto result)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  return evaluator.evaluate_to<Result>(input).value() == result;
}

// Helper to check if an expression results in an error
constexpr bool is_error(std::string_view input)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  auto result = evaluator.evaluate(input);
  return std::holds_alternative<lefticus::Error<std::uint16_t>>(result.value);
}
}// namespace

TEST_CASE("Error handling in diverse contexts", "[error]")
{
  // Test the error? predicate
  STATIC_CHECK(evaluate_to<bool>("(error? (car '()))") == true);
  STATIC_CHECK(evaluate_to<bool>("(error? 42)") == false);
  STATIC_CHECK(evaluate_to<bool>("(error? \"hello\")") == false);
  STATIC_CHECK(evaluate_to<bool>("(error? (lambda (x) x))") == false);

  // Test various error cases
  STATIC_CHECK(is_error("(+ 1 \"string\")"));// Type mismatch
  STATIC_CHECK(is_error("undefined-var"));// Undefined identifier
  STATIC_CHECK(is_error("(+ 1)"));// Wrong number of arguments
  STATIC_CHECK(is_error("(42 1 2 3)"));// Invalid function call
}

TEST_CASE("List bounds checking and error conditions", "[error][list]")
{
  // Test car on empty list
  STATIC_CHECK(is_error("(car '())"));
  STATIC_CHECK(evaluate_to<bool>("(error? (car '()))") == true);

  // Test cdr on empty list (now also returns error)
  STATIC_CHECK(is_error("(cdr '())"));
  STATIC_CHECK(evaluate_to<bool>("(error? (cdr '()))") == true);

  // Test car on non-list types
  STATIC_CHECK(is_error("(car 42)"));
  STATIC_CHECK(is_error("(car \"string\")"));
  STATIC_CHECK(is_error("(car true)"));
  STATIC_CHECK(is_error("(car 'symbol)"));// symbols are not lists

  // Test cdr on non-list types
  STATIC_CHECK(is_error("(cdr 42)"));
  STATIC_CHECK(is_error("(cdr \"string\")"));
  STATIC_CHECK(is_error("(cdr true)"));
  STATIC_CHECK(is_error("(cdr 'symbol)"));// symbols are not lists
}

TEST_CASE("Type mismatch error handling", "[error][type]")
{
  // Test different type mismatches
  STATIC_CHECK(is_error("(+ 5 \"hello\")"));// Number expected but got string
  STATIC_CHECK(is_error("(and true 42)"));// Boolean expected but got number
  STATIC_CHECK(is_error("(car 42)"));// List expected but got atom
  STATIC_CHECK(is_error("(apply 42 '(1 2 3))"));// Function expected but got number
}

TEST_CASE("Error propagation in nested expressions", "[error][propagation]")
{
  // Error in argument evaluation should propagate
  STATIC_CHECK(is_error("(+ (undefined-var) 5)"));
}

TEST_CASE("Error handling in get_list and get_list_range", "[error][helper]")
{
  // Test errors in function calls requiring specific list structures
  STATIC_CHECK(is_error("(cond 42)"));// cond requires list clauses
  STATIC_CHECK(is_error("(let 42 body)"));// let requires binding pairs
  STATIC_CHECK(is_error("(define)"));// define requires identifier and value
  STATIC_CHECK(is_error("(let ((x)) x)"));// Malformed let bindings
}

TEST_CASE("Lambda parameter validation", "[error][lambda]")
{
  // Lambda with no body
  STATIC_CHECK(is_error("(lambda (x))"));

  // Invalid parameter list
  STATIC_CHECK(is_error("(lambda 42 body)"));

  // Calling lambda with wrong number of args
  STATIC_CHECK(is_error("((lambda (x y) (+ x y)) 1)"));
}

TEST_CASE("Container overflow: values overflow returns error", "[error][overflow]")
{
  constexpr auto test = []() constexpr {
    // BuiltInValuesSize=10: constructor uses 0 values, so 10 slots available.
    // Parsing (+ 1 2 3 4 5 6 7 8 9 10 11 12) creates 13 SExprs in a list, overflowing values.
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType, 64, 1540, 10> engine;
    auto result = engine.evaluate("(+ 1 2 3 4 5 6 7 8 9 10 11 12)");
    return std::holds_alternative<lefticus::Error<std::uint16_t>>(result.value);
  };
  STATIC_CHECK(test());
}

TEST_CASE("Container overflow: strings overflow returns error", "[error][overflow]")
{
  constexpr auto test = []() constexpr {
    // Constructor uses ~173 chars of strings. Capacity 256 leaves ~83 chars headroom.
    // 4 unique 25-char identifiers = 100 chars, overflows remaining capacity.
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType, 64, 256, 279> engine;
    auto result = engine.evaluate(
      "(+ abcdefghijklmnopqrstuvwxy zyxwvutsrqponmlkjihgfedcba mnopqrstuvwxyzabcdefghijk qponmlkjihgfedcbazyxwvuts)");
    return std::holds_alternative<lefticus::Error<std::uint16_t>>(result.value);
  };
  STATIC_CHECK(test());
}

TEST_CASE("Container overflow: object_scratch overflow returns error", "[error][overflow]")
{
  constexpr auto test = []() constexpr {
    lefticus::cons_expr<> engine;
    // object_scratch capacity is 32. Each nested parse() call adds entries.
    // 33+ nesting levels will overflow the scratch.
    auto result = engine.evaluate(
      "(+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ "
      "(+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ (+ 1 1"
      ") 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1"
      ") 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1) 1)");
    return std::holds_alternative<lefticus::Error<std::uint16_t>>(result.value);
  };
  STATIC_CHECK(test());
}

TEST_CASE("Container overflow: normal operations still succeed", "[error][overflow]")
{
  // Regression guard: default-capacity engine works fine
  STATIC_CHECK(evaluate_to<int>("(+ 1 2 3)") == 6);
  STATIC_CHECK(evaluate_to<bool>("(error? (+ 1 2))") == false);
}
