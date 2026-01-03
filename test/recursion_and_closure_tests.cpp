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
}// namespace

TEST_CASE("Recursive lambda passed to another lambda", "[recursion][closure]")
{
  STATIC_CHECK(evaluate_to<int>(R"(
    ; Higher-order function that applies a function n times
    (define apply-n-times
      (lambda (f n x)
        (if (== n 0)
            x
            (f (apply-n-times f (- n 1) x)))))
    
    ; Use it to calculate 2^10
    (define double (lambda (x) (* 2 x)))
    (apply-n-times double 10 1)
  )") == 1024);
}


TEST_CASE("Deep recursive function with closure", "[recursion][closure]")
{
  STATIC_CHECK(evaluate_to<int>(R"(
    ; Recursive Fibonacci function
    (define fibonacci
      (lambda (n)
        (cond
          ((== n 0) 0)
          ((== n 1) 1)
          (else (+ (fibonacci (- n 1))
                   (fibonacci (- n 2)))))))
    
    (fibonacci 10)
  )") == 55);
}

TEST_CASE("Closure with self-reference error handling", "[recursion][closure][error]")
{
  // Create an evaluator for checking error cases
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

  // Test incorrect number of parameters
  auto result = evaluator.evaluate(R"(
    (define factorial
      (lambda (n)
        (if (== n 0)
            1
            (* n (factorial (- n 1))))))
    
    (factorial 5 10) ; Too many arguments
  )");

  REQUIRE(std::holds_alternative<lefticus::Error<std::uint16_t>>(result.value));
}

TEST_CASE("Complex nested scoping scenarios", "[recursion][closure][scoping]")
{
  STATIC_CHECK(evaluate_to<int>(R"(
    (define make-adder
      (lambda (x)
        (lambda (y)
          (+ x y))))
    
    (define add5 (make-adder 5))
    (define add10 (make-adder 10))
    
    (+ (add5 3) (add10 7))
  )") == 25);// (5+3) + (10+7)

  // More complex nesting with let and lambda
  STATIC_CHECK(evaluate_to<int>(R"(
    (let ((x 10))
      (let ((f (lambda (y) (+ x y))))
        (let ((x 20)) ; This x should not affect the closure
          (f 5))))
  )") == 15);// 10 + 5, not 20 + 5
}
