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

TEST_CASE("Basic type predicates", "[types][predicates]")
{
  // integer?
  STATIC_CHECK(evaluate_to<bool>("(integer? 42)") == true);
  STATIC_CHECK(evaluate_to<bool>("(integer? 3.14)") == false);
  STATIC_CHECK(evaluate_to<bool>("(integer? \"hello\")") == false);
  STATIC_CHECK(evaluate_to<bool>("(integer? '(1 2 3))") == false);
  
  // real?
  STATIC_CHECK(evaluate_to<bool>("(real? 3.14)") == true);
  STATIC_CHECK(evaluate_to<bool>("(real? 42)") == false);
  STATIC_CHECK(evaluate_to<bool>("(real? \"hello\")") == false);
  
  // string?
  STATIC_CHECK(evaluate_to<bool>("(string? \"hello\")") == true);
  STATIC_CHECK(evaluate_to<bool>("(string? 42)") == false);
  STATIC_CHECK(evaluate_to<bool>("(string? 3.14)") == false);
  
  // boolean?
  STATIC_CHECK(evaluate_to<bool>("(boolean? true)") == true);
  STATIC_CHECK(evaluate_to<bool>("(boolean? false)") == true);
  STATIC_CHECK(evaluate_to<bool>("(boolean? 42)") == false);
  STATIC_CHECK(evaluate_to<bool>("(boolean? \"true\")") == false);
  
  // symbol?
  STATIC_CHECK(evaluate_to<bool>("(symbol? 'abc)") == true);
  STATIC_CHECK(evaluate_to<bool>("(symbol? \"abc\")") == false);
  STATIC_CHECK(evaluate_to<bool>("(symbol? 42)") == false);
}

TEST_CASE("Composite type predicates", "[types][predicates]")
{
  // number?
  STATIC_CHECK(evaluate_to<bool>("(number? 42)") == true);
  STATIC_CHECK(evaluate_to<bool>("(number? 3.14)") == true);
  STATIC_CHECK(evaluate_to<bool>("(number? \"42\")") == false);
  STATIC_CHECK(evaluate_to<bool>("(number? '(1 2 3))") == false);
  
  // list?
  STATIC_CHECK(evaluate_to<bool>("(list? '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(list? '(1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(list? (list 1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(list? 42)") == false);
  STATIC_CHECK(evaluate_to<bool>("(list? \"hello\")") == false);
  
  // procedure?
  STATIC_CHECK(evaluate_to<bool>("(procedure? (lambda (x) x))") == true);
  STATIC_CHECK(evaluate_to<bool>("(procedure? +)") == true);
  STATIC_CHECK(evaluate_to<bool>("(procedure? 42)") == false);
  STATIC_CHECK(evaluate_to<bool>("(procedure? '(1 2 3))") == false);
  
  // atom?
  STATIC_CHECK(evaluate_to<bool>("(atom? 42)") == true);
  STATIC_CHECK(evaluate_to<bool>("(atom? \"hello\")") == true);
  STATIC_CHECK(evaluate_to<bool>("(atom? true)") == true);
  STATIC_CHECK(evaluate_to<bool>("(atom? 'abc)") == true);
  STATIC_CHECK(evaluate_to<bool>("(atom? '(1 2 3))") == false);
  STATIC_CHECK(evaluate_to<bool>("(atom? (lambda (x) x))") == false);
}

TEST_CASE("Type predicates in expressions", "[types][predicates]")
{
  // Using predicates in if expressions
  STATIC_CHECK(evaluate_to<int>(R"(
    (if (number? 42) 
        1 
        0)
  )") == 1);
  
  STATIC_CHECK(evaluate_to<int>(R"(
    (if (string? 42) 
        1 
        0)
  )") == 0);
  
  // Using predicates in lambda functions
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define type-checker
      (lambda (x)
        (cond
          ((number? x) true)
          ((string? x) true)
          (else false))))
    
    (type-checker 42)
  )") == true);
  
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define type-checker
      (lambda (x)
        (cond
          ((number? x) true)
          ((string? x) true)
          (else false))))
    
    (type-checker '(1 2 3))
  )") == false);
}
