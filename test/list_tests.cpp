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

// Basic List Creation Tests
TEST_CASE("Basic list creation", "[lists]")
{
  // Creating an empty list
  STATIC_CHECK(evaluate_to<bool>("(== '() '())") == true);

  // Creating a list with one element
  STATIC_CHECK(evaluate_to<bool>("(== '(1) '(1))") == true);

  // Creating a list with multiple elements
  STATIC_CHECK(evaluate_to<bool>("(== '(1 2 3) '(1 2 3))") == true);

  // Using the list function
  STATIC_CHECK(evaluate_to<bool>("(== (list) '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (list 1) '(1))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (list 1 2 3) '(1 2 3))") == true);

  // List with expressions that need to be evaluated
  STATIC_CHECK(evaluate_to<bool>("(== (list (+ 1 2) (* 3 4)) '(3 12))") == true);
}

// List Equality and Comparison
TEST_CASE("List equality and comparison", "[lists]")
{
  // Basic equality checks
  STATIC_CHECK(evaluate_to<bool>("(== '(1 2 3) '(1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== '(1 2 3) '(1 2 4))") == false);
  STATIC_CHECK(evaluate_to<bool>("(!= '(1 2 3) '(1 2 4))") == true);
  STATIC_CHECK(evaluate_to<bool>("(!= '(1 2 3) '(1 2 3))") == false);

  // Different length lists
  STATIC_CHECK(evaluate_to<bool>("(== '(1 2) '(1 2 3))") == false);
  STATIC_CHECK(evaluate_to<bool>("(== '(1 2 3) '(1 2))") == false);

  // Empty list comparisons
  STATIC_CHECK(evaluate_to<bool>("(== '() '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(!= '() '(1))") == true);

  // Nested list equality
  STATIC_CHECK(evaluate_to<bool>("(== '((1 2) 3) '((1 2) 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== '((1 2) 3) '((1 3) 3))") == false);
}

// List Access with car/cdr
TEST_CASE("List access with car/cdr", "[lists]")
{
  // Basic car (first element) tests
  STATIC_CHECK(evaluate_to<IntType>("(car '(1 2 3))") == 1);
  STATIC_CHECK(evaluate_to<IntType>("(car '(42))") == 42);

  // Basic cdr (rest of list) tests
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(1 2 3)) '(2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(1)) '())") == true);

  // Combined car/cdr tests
  STATIC_CHECK(evaluate_to<IntType>("(car (cdr '(1 2 3)))") == 2);
  STATIC_CHECK(evaluate_to<IntType>("(car (cdr (cdr '(1 2 3))))") == 3);

  // Nested list access
  STATIC_CHECK(evaluate_to<bool>("(== (car '((1 2) 3 4)) '(1 2))") == true);
  STATIC_CHECK(evaluate_to<IntType>("(car (car '((1 2) 3 4)))") == 1);
  STATIC_CHECK(evaluate_to<bool>("(== (cdr (car '((1 2) 3 4))) '(2))") == true);
}

// List Construction with cons
TEST_CASE("List construction with cons", "[lists]")
{
  // Basic cons (add to front of list)
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '()) '(1))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '(2)) '(1 2))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '(2 3)) '(1 2 3))") == true);

  // Building a list with multiple cons calls
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 (cons 2 (cons 3 '()))) '(1 2 3))") == true);

  // Cons with symbols
  STATIC_CHECK(evaluate_to<bool>("(== (cons 'a '(b c)) '(a b c))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (cons 'a (cons 'b '(c))) '(a b c))") == true);

  // Cons with evaluated expressions
  STATIC_CHECK(evaluate_to<bool>("(== (cons (+ 1 2) '(4 5)) '(3 4 5))") == true);

  // Cons with nested lists
  STATIC_CHECK(evaluate_to<bool>("(== (cons '(1 2) '(3 4)) '((1 2) 3 4))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 (cons '(2 3) '(4 5))) '(1 (2 3) 4 5))") == true);
}

// List Combination with append
TEST_CASE("List combination with append", "[lists]")
{
  // Basic append (combine lists)
  STATIC_CHECK(evaluate_to<bool>("(== (append '() '()) '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1) '()) '(1))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (append '() '(1)) '(1))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 2) '(3 4)) '(1 2 3 4))") == true);

  // Multiple append operations
  STATIC_CHECK(evaluate_to<bool>("(== (append (append '(1) '(2)) '(3)) '(1 2 3))") == true);

  // Append with nested lists
  STATIC_CHECK(evaluate_to<bool>("(== (append '((1 2)) '((3 4))) '((1 2) (3 4)))") == true);

  // Append with evaluated expressions
  STATIC_CHECK(evaluate_to<bool>("(== (append (list (+ 1 2)) (list (* 2 2))) '(3 4))") == true);
}

// List Evaluation and Quoted Lists
TEST_CASE("List evaluation and quoted lists", "[lists][quote]")
{
  // Quote vs list literals
  STATIC_CHECK(evaluate_to<bool>("(== (quote (1 2 3)) '(1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (quote ()) '())") == true);

  // Quoted expressions aren't evaluated
  STATIC_CHECK(evaluate_to<bool>("(== (quote (+ 1 2)) '(+ 1 2))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== '(+ 1 2) '(+ 1 2))") == true);

  // Eval on quoted expressions
  STATIC_CHECK(evaluate_to<IntType>("(eval (quote (+ 1 2)))") == 3);
  STATIC_CHECK(evaluate_to<IntType>("(eval '(+ 1 2))") == 3);

  // Nested quotes
  STATIC_CHECK(evaluate_to<bool>("(== (quote (quote (1 2 3))) '(quote (1 2 3)))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (eval (quote (quote (1 2 3)))) '(1 2 3))") == true);
}

// List Functions and Higher-Order Functions
TEST_CASE("List functions and higher-order functions", "[lists][functions]")
{
  // Apply function to list
  STATIC_CHECK(evaluate_to<IntType>("(apply + '(1 2 3))") == 6);
  STATIC_CHECK(evaluate_to<IntType>("(apply * '(2 3))") == 6);

  // Define functions that operate on lists
  STATIC_CHECK(evaluate_to<IntType>(R"(
    (define first-element (lambda (lst) (car lst)))
    (first-element '(10 20 30))
  )") == 10);

  STATIC_CHECK(evaluate_to<bool>(R"(
    (define rest-of-list (lambda (lst) (cdr lst)))
    (== (rest-of-list '(10 20 30)) '(20 30))
  )") == true);

  // Function that builds a list
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define build-list (lambda (a b c) (list a b c)))
    (== (build-list 1 2 3) '(1 2 3))
  )") == true);
}

// Nested Lists and Complex Structures
TEST_CASE("Nested lists and complex structures", "[lists][complex]")
{
  // Deeply nested lists equality test
  STATIC_CHECK(evaluate_to<bool>("(== '(1 (2 (3 (4)))) '(1 (2 (3 (4)))))") == true);

  // Nested list access
  STATIC_CHECK(evaluate_to<bool>("(== '(1 (2 3) 4) '(1 (2 3) 4))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (car (cdr '(1 (2 3) 4))) '(2 3))") == true);
  STATIC_CHECK(evaluate_to<IntType>("(car (car (cdr '(1 (2 3) 4))))") == 2);
  STATIC_CHECK(evaluate_to<bool>("(== (cdr (car (cdr '(1 (2 3) 4)))) '(3))") == true);

  // Building complex nested structures
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define nested 
      (cons 1 
        (cons (cons 2 
                (cons 3 '())) 
          (cons 4 '()))))
    (== nested '(1 (2 3) 4))
  )") == true);
}

// Empty List Edge Cases
TEST_CASE("Empty list edge cases", "[lists][edge]")
{
  // Various ways to represent empty lists
  STATIC_CHECK(evaluate_to<bool>("(== '() '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (quote ()) '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (list) '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(1)) '())") == true);

  // Combining with empty lists
  STATIC_CHECK(evaluate_to<bool>("(== (append '() '()) '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 2) '()) '(1 2))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (append '() '(1 2)) '(1 2))") == true);

  // Cons with empty list
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '()) '(1))") == true);
}

// List Manipulation Algorithms
TEST_CASE("List manipulation algorithms", "[lists][algorithms]")
{
  // Simple list operation tests
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define simple-fn
      (lambda (lst)
        (if (== lst '())
            true
            false)))
    
    (simple-fn '())
  )") == true);

}
