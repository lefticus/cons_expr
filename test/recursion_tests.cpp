#include <catch2/catch_test_macros.hpp>

#include <cons_expr/cons_expr.hpp>
#include <cstdint>
#include <string_view>

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
}// namespace

TEST_CASE("Y-Combinator", "[recursion]")
{
  STATIC_CHECK(evaluate_to<int>(
                 R"(
;; Y combinator definition
(define Y
  (lambda (f)
    ((lambda (x) (f (lambda (y) ((x x) y))))
     (lambda (x) (f (lambda (y) ((x x) y)))))))

;; Factorial using Y combinator
(define factorial
  (Y (lambda (fact)
       (lambda (n)
         (if (== n 0)
             1
             (* n (fact (- n 1))))))))

(factorial 5)
)") == 120);
}


TEST_CASE("expressive 'define' 1 level", "[recursion]")
{
  STATIC_CHECK(evaluate_to<int>(
                 R"(
(define factorial
  (lambda (n)
    (if (== n 0)
        1
        (* n (factorial (- n 1))))))

(factorial 1)
)") == 1);
}

TEST_CASE("expressive 'define' 5 levels", "[recursion]")
{
  STATIC_CHECK(evaluate_to<int>(
                 R"(
(define factorial
  (lambda (n)
    (if (== n 0)
        1
        (* n (factorial (- n 1))))))

(factorial 5)
)") == 120);
}
