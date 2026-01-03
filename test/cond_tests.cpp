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

TEST_CASE("Cond expression basic usage", "[cond]")
{
  // Basic cond with one matching clause
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((< 5 10) 1)
      (else 2))
  )") == 1);

  // Basic cond with else clause
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((> 5 10) 1)
      (else 2))
  )") == 2);

  // Cond with multiple conditions
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((> 5 10) 1)
      ((< 5 10) 2)
      (else 3))
  )") == 2);

  // Cond with multiple conditions, evaluating last one
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((> 5 10) 1)
      ((> 5 20) 2)
      (else 3))
  )") == 3);
}

TEST_CASE("Cond with complex expressions", "[cond]")
{
  // Cond with expressions in conditions
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((< (+ 2 3) (* 2 3)) 1)
      ((> (+ 2 3) (* 2 3)) 2)
      (else 3))
  )") == 1);

  // Cond with expressions in results
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((< 5 10) (+ 1 2))
      (else (- 10 5)))
  )") == 3);

  // Nested cond expressions
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((< 5 10) (cond 
                  ((> 3 1) 1)
                  (else 2)))
      (else 3))
  )") == 1);
}

TEST_CASE("Cond without else clause", "[cond]")
{
  // Cond with multiple conditions but no else, with a match
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((> 5 10) 1)
      ((< 5 10) 2))
  )") == 2);

  // Cond with no else and no matching condition should error
  STATIC_CHECK(is_error(R"(
    (cond 
      ((> 5 10) 1)
      ((> 5 20) 2))
  )"));
}

TEST_CASE("Cond with side effects", "[cond]")
{
  // Only the matching condition's result should be evaluated
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 5)
    (define y 10)
    (cond 
      ((< x y) x)
      (else (/ x 0))) ; This would error if evaluated
  )") == 5);

  // Similarly, condition expressions should be evaluated in sequence
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((< 5 10) 1)
      ((/ 1 0) 2)) ; This division by zero should not occur
  )") == 1);
}

TEST_CASE("Cond with boolean conditions", "[cond]")
{
  // Directly using boolean values
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      (true 1)
      (else 2))
  )") == 1);

  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      (false 1)
      (else 2))
  )") == 2);

  // Using boolean expressions
  STATIC_CHECK(evaluate_to<int>(R"(
    (cond 
      ((and (< 5 10) (> 5 1)) 1)
      (else 2))
  )") == 1);
}

TEST_CASE("Cond error handling", "[cond][error]")
{
  // Malformed cond syntax
  STATIC_CHECK(is_error("(cond)"));
  STATIC_CHECK(is_error("(cond 1 2 3)"));

  // Condition clause not a list
  STATIC_CHECK(is_error("(cond 42 else)"));

  // Condition clause without result
  STATIC_CHECK(is_error("(cond ((< 5 10)))"));

  // Non-boolean condition (should be okay actually)
  // STATIC_CHECK(evaluate_to<int>("(cond (1 42) (else 0))") == 42);
}
