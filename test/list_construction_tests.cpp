#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>
#include <internal_use_only/config.hpp>

using IntType = int;
using FloatType = double;

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

TEST_CASE("Cons function with various types", "[list][cons]")
{
  // Basic cons with a number and a list
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '(2 3)) '(1 2 3))") == true);

  // Cons with a string
  STATIC_CHECK(evaluate_to<bool>("(== (cons \"hello\" '(\"world\")) '(\"hello\" \"world\"))") == true);

  // Cons with a boolean
  STATIC_CHECK(evaluate_to<bool>("(== (cons true '(false)) '(true false))") == true);

  // Cons with a symbol
  STATIC_CHECK(evaluate_to<bool>("(== (cons 'a '(b c)) '(a b c))") == true);

  // Cons with an empty list
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '()) '(1))") == true);

  // Cons with a nested list
  STATIC_CHECK(evaluate_to<bool>("(== (cons '(1 2) '(3 4)) '((1 2) 3 4))") == true);
}

TEST_CASE("Append function with various lists", "[list][append]")
{
  // Basic append with two simple lists
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 2) '(3 4)) '(1 2 3 4))") == true);

  // Append with an empty first list
  STATIC_CHECK(evaluate_to<bool>("(== (append '() '(1 2)) '(1 2))") == true);

  // Append with an empty second list
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 2) '()) '(1 2))") == true);

  // Append with two empty lists
  STATIC_CHECK(evaluate_to<bool>("(== (append '() '()) '())") == true);

  // Append with nested lists
  STATIC_CHECK(evaluate_to<bool>("(== (append '((1) 2) '(3 (4))) '((1) 2 3 (4)))") == true);

  // Append with mixed content
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 \"two\") '(true 3.0)) '(1 \"two\" true 3.0))") == true);
}

TEST_CASE("Car function with various lists", "[list][car]")
{
  // Basic car of a simple list
  STATIC_CHECK(evaluate_to<int>("(car '(1 2 3))") == 1);

  // Car of a list with mixed types
  STATIC_CHECK(evaluate_expected<std::string_view>("(car '(\"hello\" 2 3))", "hello"));

  // Car of a list with a nested list
  STATIC_CHECK(evaluate_to<bool>("(== (car '((1 2) 3 4)) '(1 2))") == true);

  // Car of a single-element list
  STATIC_CHECK(evaluate_to<int>("(car '(42))") == 42);

  // Car of a quoted symbol list
  STATIC_CHECK(evaluate_to<bool>("(== (car '(a b c)) 'a)") == true);
}

TEST_CASE("Cdr function with various lists", "[list][cdr]")
{
  // Basic cdr of a simple list
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(1 2 3)) '(2 3))") == true);

  // Cdr of a list with mixed types
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(\"hello\" 2 3)) '(2 3))") == true);

  // Cdr of a list with a nested list
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '((1 2) 3 4)) '(3 4))") == true);

  // Cdr of a single-element list returns empty list
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(42)) '())") == true);

  // Cdr of a two-element list
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(1 2)) '(2))") == true);
}

TEST_CASE("Complex list construction", "[list][complex]")
{
  // Combining cons, car, and cdr
  STATIC_CHECK(evaluate_to<bool>(R"(
    (== (cons (car '(1 2)) 
              (cdr '(3 4 5)))
        '(1 4 5))
  )") == true);

  // Nested cons calls
  STATIC_CHECK(evaluate_to<bool>(R"(
    (== (cons 1 (cons 2 (cons 3 '())))
        '(1 2 3))
  )") == true);

  // Combining append with cons
  STATIC_CHECK(evaluate_to<bool>(R"(
    (== (append (cons 1 '(2)) 
                (cons 3 '(4)))
        '(1 2 3 4))
  )") == true);

  // Building complex nested structures
  STATIC_CHECK(evaluate_to<bool>(R"(
    (== (cons (cons 1 '(2)) 
              (cons (cons 3 '(4)) 
                    '()))
        '((1 2) (3 4)))
  )") == true);
}

TEST_CASE("List construction edge cases", "[list][edge]")
{
  // Cons with both arguments being lists
  STATIC_CHECK(evaluate_to<bool>("(== (cons '(1) '(2)) '((1) 2))") == true);

  // Nested empty lists
  STATIC_CHECK(evaluate_to<bool>("(== (cons '() '()) '(()))") == true);

  // Triple-nested cons
  STATIC_CHECK(evaluate_to<bool>(R"(
    (== (cons 1 (cons 2 (cons 3 '())))
        '(1 2 3))
  )") == true);
}
