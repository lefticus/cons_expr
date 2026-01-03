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


// Function to check if an evaluation fails with an error
constexpr bool expect_error(std::string_view input)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  auto result = evaluator.evaluate(input);
  // Check if the result is an error type
  return std::holds_alternative<lefticus::cons_expr<>::error_type>(result.value);
}
}

// ----- Basic Scoping Tests -----

TEST_CASE("Basic identifier scoping", "[scoping][basic]")
{
  // Simple undefined identifier - should fail
  STATIC_CHECK(expect_error("undefined_variable"));

  // Basic define at global scope
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    x
  )") == 10);

  // Shadowing a global definition with a local one
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    (let ((x 20))
      x)
  )") == 20);

  // Outer scope still has original value
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    (let ((x 20))
      x)
    x
  )") == 10);

  // Multiple nestings - innermost wins
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    (let ((x 20))
      (let ((x 30))
        x))
  )") == 30);
}

TEST_CASE("Lambda scoping", "[scoping][lambda]")
{
  // Basic lambda parameter scoping
  STATIC_CHECK(evaluate_to<int>(R"(
    ((lambda (x) x) 42)
  )") == 42);

  // Lambda parameters shadow global scope
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    ((lambda (x) x) 42)
  )") == 42);

  // Lambda body can access global scope for non-shadowed variables
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    ((lambda (y) x) 42)
  )") == 10);

  // Lambda parameter shadows global with same name,
  // but can still access other globals
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    (define y 20)
    ((lambda (x) (+ x y)) 30)
  )") == 50);
}

TEST_CASE("Let scoping", "[scoping][let]")
{
  // Basic let binding
  STATIC_CHECK(evaluate_to<int>(R"(
    (let ((x 10)) x)
  )") == 10);

  // Multiple bindings in same let
  STATIC_CHECK(evaluate_to<int>(R"(
    (let ((x 10) (y 20)) (+ x y))
  )") == 30);

  // Let bindings are not visible outside their scope
  STATIC_CHECK(expect_error(R"(
    (let ((x 10)) x)
    x
  )"));

  // Later bindings in the same let can't see earlier ones
  STATIC_CHECK(expect_error(R"(
    (let ((x 10) (y (+ x 1))) y)
  )"));
}

TEST_CASE("Define scoping", "[scoping][define]")
{
  // Redefining a global is allowed
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    (define x 20)
    x
  )") == 20);

  // Define in nested scopes
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    (let ((y 20))
      (define z 30)
      (+ x (+ y z)))
  )") == 60);

  // Define in a lambda body creates a new binding in that scope
  STATIC_CHECK(evaluate_to<int>(R"(
    (define counter 0)
    (define inc-counter
      (lambda ()
        (define counter (+ counter 1))
        counter))
    (inc-counter) ; this introduces a new counter in the lambda's scope
    counter       ; global counter remains unchanged
  )") == 0);
}

TEST_CASE("Recursive functions", "[scoping][recursion]")
{
  // Basic recursive function
  STATIC_CHECK(evaluate_to<int>(R"(
    (define fact
      (lambda (n)
        (if (== n 0)
            1
            (* n (fact (- n 1))))))
    (fact 5)
  )") == 120);

  // Shadowing a recursive function parameter with local binding
  STATIC_CHECK(evaluate_to<int>(R"(
    (define fact
      (lambda (n)
        (let ((n 10)) ; This shadows the parameter
          n)))
    (fact 5)
  )") == 10);

  // Ensure recursion still works with shadowed globals
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 10)
    (define fact
      (lambda (n)
        (if (== n 0)
            1
            (let ((x (* n (fact (- n 1)))))
              x))))
    (fact 5)
  )") == 120);
}

TEST_CASE("Lexical closure capture", "[scoping][closure]")
{
  // Basic closure capturing
  STATIC_CHECK(evaluate_to<int>(R"(
    (define make-adder
      (lambda (x)
        (lambda (y) (+ x y))))
    (define add5 (make-adder 5))
    (add5 10)
  )") == 15);

  // Nested closures capturing different variables
  STATIC_CHECK(evaluate_to<int>(R"(
    (define make-adder
      (lambda (x)
        (lambda (y)
          (lambda (z)
            (+ x (+ y z))))))
    (define add5 (make-adder 5))
    (define add5and10 (add5 10))
    (add5and10 15)
  )") == 30);

  // Captured variables are immutable in the closure (except for self-recursion)
  // This system captures values at definition time, not references
  STATIC_CHECK(evaluate_to<int>(R"(
    (define x 5)
    (define get-x (lambda () x))
    (define x 10)
    (get-x)
  )") == 5);
}

TEST_CASE("Complex scoping scenarios", "[scoping][complex]")
{
  // Simplified version using just the regular Y-combinator pattern
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define Y
      (lambda (f)
        ((lambda (x) (f (lambda (y) ((x x) y))))
         (lambda (x) (f (lambda (y) ((x x) y)))))))
    
    ; A simpler even function using just regular recursion
    (define is-even?
      (Y (lambda (self)
           (lambda (n)
             (if (== n 0)
                 true
                 (if (== n 1)
                     false
                     (self (- n 2))))))))
                     
    (is-even? 10)
  )") == true);

  // Higher-order functions with scoping
  STATIC_CHECK(evaluate_to<int>(R"(
    (define apply-twice
      (lambda (f x)
        (f (f x))))
    
    (define add5
      (lambda (n)
        (+ n 5)))
    
    (apply-twice add5 10)
  )") == 20);

  // IIFE (Immediately Invoked Function Expression) pattern
  STATIC_CHECK(evaluate_to<int>(R"(
    ((lambda (x)
       (define square (lambda (y) (* y y)))
       (square x))
     7)
  )") == 49);

  // Demonstrating that attempts to create stateful closures don't work
  // because we can't mutate captured variables
  STATIC_CHECK(evaluate_to<int>(R"(
    (define make-adder
      (lambda (x)
        (lambda (y) (+ x y))))
    
    (define add10 (make-adder 10))
    (add10 5)  ; Always returns x+y (10+5)
  )") == 15);
}

TEST_CASE("Edge cases in scoping", "[scoping][edge]")
{
  // Empty body in lambda
  STATIC_CHECK(expect_error(R"(
    ((lambda (x)) 42)
  )"));

  // Empty body in let returns the last expression evaluated (which is nothing)
  STATIC_CHECK(evaluate_to<std::monostate>(R"(
    (let ((x 10)))
  )") == std::monostate{});

  // Self-shadowing in nested let
  STATIC_CHECK(evaluate_to<int>(R"(
    (let ((x 10))
      (let ((x (+ x 5)))
        x))
  )") == 15);

  // Recursive let (not supported in most schemes)
  STATIC_CHECK(expect_error(R"(
    (let ((fact (lambda (n)
                  (if (== n 0)
                      1
                      (* n (fact (- n 1)))))))
      (fact 5))
  )"));

  // Named let for recursion (not implemented)
  STATIC_CHECK(expect_error(R"(
    (let loop ((n 5) (acc 1))
      (if (== n 0)
          acc
          (loop (- n 1) (* acc n))))
  )"));
}

TEST_CASE("Y Combinator for anonymous recursion", "[scoping][y-combinator]")
{
  // Using Y-combinator to make an anonymous recursive function
  STATIC_CHECK(evaluate_to<int>(R"(
    (define Y
      (lambda (f)
        ((lambda (x) (f (lambda (y) ((x x) y))))
         (lambda (x) (f (lambda (y) ((x x) y)))))))
    
    ((Y (lambda (fact)
          (lambda (n)
            (if (== n 0)
                1
                (* n (fact (- n 1)))))))
     5)
  )") == 120);

  // Y-combinator with captured variable from outer scope
  STATIC_CHECK(evaluate_to<int>(R"(
    (define Y
      (lambda (f)
        ((lambda (x) (f (lambda (y) ((x x) y))))
         (lambda (x) (f (lambda (y) ((x x) y)))))))
    
    (define multiplier 2)
    
    ((Y (lambda (fact)
          (lambda (n)
            (if (== n 0)
                1
                (* multiplier (fact (- n 1)))))))
     5)
  )") == 32);// 2^5
}

TEST_CASE("Recursive lambda passed to another lambda", "[scoping][recursion][lambda-passing]")
{
  // Define a recursive function, pass it to another function, and verify it still works
  STATIC_CHECK(evaluate_to<int>(R"(
    ; Define a recursive factorial function
    (define factorial
      (lambda (n)
        (if (== n 0)
            1
            (* n (factorial (- n 1))))))
    
    ; Define a function that applies its argument to 5
    (define apply-to-5
      (lambda (f)
        (f 5)))
    
    ; Pass the recursive function to apply-to-5
    (apply-to-5 factorial)
  )") == 120);

  // More complex case with a higher-order function that uses the passed function multiple times
  STATIC_CHECK(evaluate_to<int>(R"(
    ; Define a recursive Fibonacci function
    (define fib
      (lambda (n)
        (if (< n 2)
            n
            (+ (fib (- n 1)) (fib (- n 2))))))
    
    ; Define a function that adds the results of applying a function to two arguments
    (define apply-and-add
      (lambda (f x y)
        (+ (f x) (f y))))
    
    ; Pass the recursive function to apply-and-add
    (apply-and-add fib 5 6)
  )") == 13);// fib(5) + fib(6) = 5 + 8 = 13

  // A recursive function that returns another recursive function
  STATIC_CHECK(evaluate_to<int>(R"(
    ; Define a recursive function that returns a specialized power function
    (define make-power-fn
      (lambda (exponent)
        (lambda (base)
          (if (== exponent 0)
              1
              (* base ((make-power-fn (- exponent 1)) base))))))
    
    ; Get the cube function and apply it to 2
    (define cube (make-power-fn 3))
    (cube 2)
  )") == 8);// 2³ = 8
}
