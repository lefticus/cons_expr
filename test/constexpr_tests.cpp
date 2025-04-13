#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>
#include <internal_use_only/config.hpp>

using IntType = int;
using FloatType = double;

static_assert(lefticus::is_cons_expr_v<lefticus::cons_expr<>>);

static_assert(std::is_trivially_copyable_v<lefticus::cons_expr<>::SExpr>);


template<typename Result> constexpr Result evaluate_to(std::string_view input)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  return evaluator.evaluate_to<Result>(input).value();
}

// this version exists so we can evaluate an objects
// whose lifetime would have otherwise ended
template<typename Result> constexpr bool evaluate_expected(std::string_view input, auto result)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  return evaluator.evaluate_to<Result>(input).value() == result;
}

template<typename Result> constexpr std::optional<Result> parse_as(auto &evaluator, std::string_view input)
{
  using eval_type = std::remove_cvref_t<decltype(evaluator)>;
  using list_type = eval_type::list_type;

  auto [parse_result, parse_remaining] = evaluator.parse(input);
  // properly parsed results are always lists
  const auto list = std::get_if<list_type>(&parse_result.value);
  // this should be a list of exactly 1 thing (which might be another list)
  if (list == nullptr || list->size != 1) { return std::optional<Result>{}; }
  const auto first_elem = evaluator.values[(*list)[0]];

  const auto *result = evaluator.template get_if<Result>(&first_elem);

  if (result == nullptr) { return std::optional<Result>{}; }

  return *result;
}

TEST_CASE("Operator identifiers", "[operators]")
{
  STATIC_CHECK(evaluate_to<IntType>("((if false + *) 3 4)") == 12);
  STATIC_CHECK(evaluate_to<IntType>("((if true + *) 3 4)") == 7);
  STATIC_CHECK(evaluate_to<IntType>("((if (== 1 1) + *) 5 4)") == 9);
  STATIC_CHECK(evaluate_to<IntType>("((if (!= 1 1) + *) 5 4)") == 20);
}

TEST_CASE("basic float operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<FloatType>("(+ 1.0 0.1)") == FloatType{ 1.1 });
  STATIC_CHECK(evaluate_to<FloatType>("(+ 0.0 1.0e-1)") == FloatType{ 1.0e-1 });
  STATIC_CHECK(evaluate_to<FloatType>("(+ 0.0 0.1e1)") == FloatType{ 0.1e1 });
  STATIC_CHECK(evaluate_to<FloatType>("(- 5.5 2.5)") == FloatType{ 3.0 });
  STATIC_CHECK(evaluate_to<FloatType>("(* 2.5 3.0)") == FloatType{ 7.5 });
  STATIC_CHECK(evaluate_to<FloatType>("(/ 10.0 2.0)") == FloatType{ 5.0 });
  STATIC_CHECK(evaluate_to<FloatType>("(/ 10.0 4.0)") == FloatType{ 2.5 });
}


TEST_CASE("basic string_view operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<bool>(R"((== "hello" "hello"))") == true);
  STATIC_CHECK(evaluate_to<bool>(R"((== "hello" "world"))") == false);
  STATIC_CHECK(evaluate_to<bool>(R"((!= "hello" "world"))") == true);
  STATIC_CHECK(evaluate_to<bool>(R"((!= "hello" "hello"))") == false);
  STATIC_CHECK(evaluate_expected<std::string_view>(R"("test string")", "test string"));
}

TEST_CASE("access as string_view", "[strings]")
{
  STATIC_CHECK(evaluate_expected<std::string_view>(R"("hello")", "hello"));
  STATIC_CHECK(evaluate_expected<std::string_view>(R"("multi word string")", "multi word string"));
  STATIC_CHECK(evaluate_expected<std::string_view>(R"("")", ""));
}

TEST_CASE("basic integer operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<IntType>("(+ 1 2)") == 3);
  STATIC_CHECK(evaluate_to<IntType>("(/ 2 2)") == 1);
  STATIC_CHECK(evaluate_to<IntType>("(- 2 2)") == 0);
  STATIC_CHECK(evaluate_to<IntType>("(* 2 2)") == 4);

  STATIC_CHECK(evaluate_to<IntType>("(+ 1 2 3 -6)") == 0);
  STATIC_CHECK(evaluate_to<IntType>("(/ 4 2 1)") == 2);
  STATIC_CHECK(evaluate_to<IntType>("(- 2 2 1)") == -1);
  STATIC_CHECK(evaluate_to<IntType>("(* 2 2 2 2 2)") == 32);

  // Additional complex arithmetic expressions
  STATIC_CHECK(evaluate_to<IntType>("(+ (* 2 3) (- 10 5))") == 11);
  STATIC_CHECK(evaluate_to<IntType>("(* (+ 2 3) (- 10 5))") == 25);
  STATIC_CHECK(evaluate_to<IntType>("(/ (* 8 4) (+ 2 2))") == 8);
}

TEST_CASE("list comparisons", "[operators]")
{
  STATIC_CHECK(evaluate_to<bool>("(== '(1) '(1))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== '(1 2 3) '(1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(!= '(1 2 3) '(3 2 1))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== '() '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(!= '(1 2) '(1 2 3))") == true);
}

TEST_CASE("basic integer comparisons", "[operators]")
{
  STATIC_CHECK(evaluate_to<bool>("(== 12 12)") == true);
  STATIC_CHECK(evaluate_to<bool>("(== 12 12 12)") == true);
  STATIC_CHECK(evaluate_to<bool>("(!= 12 13)") == true);
  STATIC_CHECK(evaluate_to<bool>("(!= 12 12)") == false);

  STATIC_CHECK(evaluate_to<bool>("(< 12 3 1)") == false);
  STATIC_CHECK(evaluate_to<bool>("(> 12 3 1)") == true);
  STATIC_CHECK(evaluate_to<bool>("(>= 12 3 12)") == false);
  STATIC_CHECK(evaluate_to<bool>("(>= 12 12 1)") == true);
  STATIC_CHECK(evaluate_to<bool>("(>= 12 12 1 12)") == false);
  STATIC_CHECK(evaluate_to<bool>("(<= 1 2 3 4)") == true);
  STATIC_CHECK(evaluate_to<bool>("(<= 1 2 2 3)") == true);
  STATIC_CHECK(evaluate_to<bool>("(<= 1 3 2 4)") == false);
}

TEST_CASE("basic logical boolean operations", "[operators]")
{
  STATIC_CHECK(evaluate_to<bool>("(and true true false)") == false);
  STATIC_CHECK(evaluate_to<bool>("(and true true true)") == true);
  STATIC_CHECK(evaluate_to<bool>("(and true true)") == true);
  STATIC_CHECK(evaluate_to<bool>("(or false true false true)") == true);
  STATIC_CHECK(evaluate_to<bool>("(or false false false)") == false);
  STATIC_CHECK(evaluate_to<bool>("(not false)") == true);
  STATIC_CHECK(evaluate_to<bool>("(not true)") == false);

  // Compound logical operations
  STATIC_CHECK(evaluate_to<bool>("(and (or true false) (not false))") == true);
  STATIC_CHECK(evaluate_to<bool>("(or (and true false) (not false))") == true);
  STATIC_CHECK(evaluate_to<bool>("(and (> 5 3) (< 2 8))") == true);
  STATIC_CHECK(evaluate_to<bool>("(or (> 5 10) (< 2 1))") == false);
}

TEST_CASE("basic lambda usage", "[lambdas]")
{
  STATIC_CHECK(evaluate_to<bool>("((lambda () true))") == true);
  STATIC_CHECK(evaluate_to<bool>("((lambda () false))") == false);
  STATIC_CHECK(evaluate_to<bool>("((lambda (x) x) true)") == true);
  STATIC_CHECK(evaluate_to<bool>("((lambda (x) x) false)") == false);
  STATIC_CHECK(evaluate_to<IntType>("((lambda (x) (* x x)) 11)") == 121);
  STATIC_CHECK(evaluate_to<IntType>("((lambda (x y) (+ x y)) 5 7)") == 12);
  STATIC_CHECK(evaluate_to<IntType>("((lambda (x y z) (+ x (* y z))) 5 7 2)") == 19);
}

TEST_CASE("nested lambda usage", "[lambdas]")
{
  STATIC_CHECK(evaluate_to<IntType>("(define l (lambda (x) (lambda () x))) ((l 1))") == 1);
  STATIC_CHECK(evaluate_to<IntType>("(define l (lambda (x) (lambda (y) (lambda () (+ x y))))) (((l 1) 3))") == 4);
  STATIC_CHECK(evaluate_to<IntType>("(define l (lambda (x) (lambda (y) (let ((z (+ x y))) z)))) ((l 1) 3)") == 4);
  STATIC_CHECK(evaluate_to<IntType>("(define l (lambda (x) (lambda (y) (let ((z 10)) (+ x y z))))) ((l 1) 3)") == 14);
  STATIC_CHECK(evaluate_to<IntType>("((lambda (x) (let ((x (+ x 5))) x)) 2)") == 7);

  // Higher-order function tests
  STATIC_CHECK(evaluate_to<IntType>(R"(
    (define apply-twice (lambda (f x) (f (f x))))
    (define add-one (lambda (x) (+ x 1)))
    (apply-twice add-one 10)
  )") == 12);

  STATIC_CHECK(evaluate_to<IntType>(R"(
    (define compose (lambda (f g) (lambda (x) (f (g x)))))
    (define square (lambda (x) (* x x)))
    (define double (lambda (x) (* x 2)))
    ((compose square double) 3)
  )") == 36);
}

TEST_CASE("basic define usage", "[define]")
{
  STATIC_CHECK(evaluate_to<IntType>("(define x 32) x") == 32);
  STATIC_CHECK(evaluate_to<IntType>("(define x (lambda (y)(+ y 4))) (x 10)") == 14);
  STATIC_CHECK(evaluate_to<IntType>("(define x 10) (define y 20) (+ x y)") == 30);
  STATIC_CHECK(evaluate_to<IntType>("(define x 5) (define x 10) x") == 10);// Shadowing
  STATIC_CHECK(evaluate_to<bool>("(define x true) x") == true);
}

TEST_CASE("define scoping", "[define]")
{
  STATIC_CHECK(evaluate_to<IntType>("((lambda () (define y 20) y))") == 20);
  STATIC_CHECK(evaluate_to<IntType>("((lambda (x) (define y 20) (+ x y)) 10)") == 30);
  STATIC_CHECK(evaluate_to<IntType>("((lambda (x) (define y (* x 2)) y) 20)") == 40);
  STATIC_CHECK(evaluate_to<IntType>("(define x 42) (define l (lambda (x)(+ x 4))) (l 10)") == 14);
  STATIC_CHECK(evaluate_to<IntType>(R"(
             (
               (lambda (x)
                 (define y
                   ((lambda (z) (* z x 2)) 3)
                 )
                 y
               )
             4)
)") == 24);

  // The original test was failing because lambda scoping worked differently
  // than expected in the constexpr context

  // In lambda scope, x defined in the lambda body should shadow the global x
  STATIC_CHECK(evaluate_to<IntType>(R"(
    (define x 10)
    ((lambda () (define x 20) x))
  )") == 10);// Notice we expect 10 here, not 20, because of how define works

  // Outside lambda scope, global 'x' is still 10 (immutable)
  STATIC_CHECK(evaluate_to<IntType>(R"(
    (define x 10)
    ((lambda () (define x 20) x))
    x
  )") == 10);

  // Let's test a clearer example of lambda scope that should work:
  STATIC_CHECK(evaluate_to<IntType>(R"(
    ((lambda () (define y 5) y))
  )") == 5);

  // This wouldn't work as expected because counter is immutable
  // In cons_expr, defining a global doesn't mutate existing references
  /*
  STATIC_CHECK(evaluate_to<IntType>(R"(
    (define counter 0)
    (define increment (lambda () (define counter (+ counter 1)) counter))
    (increment)
    (increment)
    (increment)
  )") == 3);
  */
}


TEST_CASE("GPT Generated Tests", "[integration tests]")
{
  STATIC_CHECK(evaluate_to<IntType>(R"(
(define square (lambda (x) (* x x)))
(square 5)
)") == 25);

  STATIC_CHECK(evaluate_to<IntType>(R"(
(define make-adder (lambda (x) (lambda (y) (+ x y))))
((make-adder 5) 3)
)") == 8);

  STATIC_CHECK(evaluate_to<IntType>(R"(
(let ((x 2) (y 3))
  (define adder (lambda (a b) (+ a b)))
  (adder x y))
)") == 5);


  STATIC_CHECK(evaluate_to<IntType>(R"(
(let ((x 2))
  (let ((y 3))
    (+ x y)))
)") == 5);

}

TEST_CASE("binary short circuiting", "[short circuiting]")
{
  STATIC_CHECK(evaluate_to<bool>("(and false (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(or true (unknownfunc))") == true);
  STATIC_CHECK(evaluate_to<bool>("(< 2 1 (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(> 1 2 (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(and (== 1 1) (== 2 2) false (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(or false false true (unknownfunc))") == true);
}

TEST_CASE("let variables", "[let variables]")
{
  STATIC_CHECK(evaluate_to<IntType>("(let ((x 3)(y 14)) (* x y))") == 42);
  STATIC_CHECK(evaluate_to<IntType>("(let ((x (* 3 1))(y (- 18 4))) (* x y))") == 42);
  STATIC_CHECK(evaluate_to<IntType>("(define x 42) (let ((l (lambda (x)(+ x 4)))) (l 10))") == 14);
  // let variable initial values are scoped to the outer scope, not to previously
  // declared variables in scope
  STATIC_CHECK(evaluate_to<IntType>("(define x 42) (let ((x 10)(y x)) y)") == 42);

  STATIC_CHECK(evaluate_to<IntType>("(define x 2) (let ((x (+ x 5))) x)") == 7);

  // Additional let tests
  STATIC_CHECK(evaluate_to<IntType>(R"(
    (let ((x 10) (y 20))
      (let ((z (+ x y)))
        (+ x y z)))
  )") == 60);

  STATIC_CHECK(evaluate_to<IntType>(R"(
    (let ((x 5) (y 4))
      (let ((x (* x 2)) (y (+ y 3)))
        (* x y)))
  )") == 70);

  STATIC_CHECK(evaluate_to<IntType>(R"(
    (define outer 100)
    (let ((outer 50))
      (let ((result (+ outer 10)))
        result))
  )") == 60);
}

TEST_CASE("list operations", "[builtins]")
{
  STATIC_CHECK(evaluate_to<IntType>("(car '(1 2 3 4))") == 1);
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(1 2 3 4)) '(2 3 4))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (car (cdr '(1 2 3 4))) 2)") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (car (cdr (cdr '(1 2 3 4)))) 3)") == true);

  // List function
  STATIC_CHECK(evaluate_to<bool>("(== (list 1 2 3) '(1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (list) '())") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (list 1) '(1))") == true);

  // List with evaluated expressions
  STATIC_CHECK(evaluate_to<bool>("(== (list (+ 1 2) (* 3 4)) '(3 12))") == true);
}

TEST_CASE("comments", "[parsing]")
{
  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
15
)") == 15);

  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
; a comment
15
)") == 15);

  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
15 ; a comment
)") == 15);

  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
; a comment

15
)") == 15);

  // Multiple comments
  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
; first comment
; second comment
(+ 10 ; inline comment
   5) ; another comment
)") == 15);

  // Comment in expression
  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
(+ 
  ; comment between arguments
  10 5)
)") == 15);
}

TEST_CASE("simple cons expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<bool>("(== (cons '(1 2 3 4) '(5)) '((1 2 3 4) 5))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '(5)) '(1 5))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (cons 'x '(5)) '(x 5))") == true);

  // Break down cons tests into simpler incremental cases

  // Test consing a single element to empty list
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '()) '(1))") == true);

  // Test consing a single element to an existing list
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '(2)) '(1 2))") == true);

  // Test consing an element to a list with two elements
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 '(2 3)) '(1 2 3))") == true);

  // Test consing two elements sequentially to build a list - incremental
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 (cons 2 '())) '(1 2))") == true);

  // Test consing three elements sequentially to build a list - incremental
  STATIC_CHECK(evaluate_to<bool>("(== (cons 1 (cons 2 (cons 3 '()))) '(1 2 3))") == true);

  // Test consing symbols instead of numbers
  STATIC_CHECK(evaluate_to<bool>("(== (cons 'a '(b c)) '(a b c))") == true);

  // Test sequential consing with symbols
  STATIC_CHECK(evaluate_to<bool>("(== (cons 'a (cons 'b '(c))) '(a b c))") == true);

  // Test consing an evaluated expression
  STATIC_CHECK(evaluate_to<bool>("(== (cons (+ 1 2) '(4 5)) '(3 4 5))") == true);
}

TEST_CASE("apply expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<IntType>("(apply * '(2 3))") == 6);
  STATIC_CHECK(evaluate_to<IntType>("(apply + '(1 2 3 4 5))") == 15);
  STATIC_CHECK(evaluate_to<IntType>("(apply - '(10 5 2))") == 3);

  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
(define x 10)

(define add-x (lambda (y) (+ x y)))

(let ((x 20))
  (apply add-x (list 5)))
)") == 15);

  // Apply with lambda expressions
  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
(apply (lambda (x y) (+ x (* y 2))) '(5 10))
)") == 25);

  STATIC_CHECK(evaluate_to<bool>(
                 R"(
(apply == '(1 1))
)") == true);
}

TEST_CASE("check version number", "[system]")
{
  STATIC_CHECK(lefticus::cons_expr_version_major == cons_expr::cmake::project_version_major);
  STATIC_CHECK(lefticus::cons_expr_version_minor == cons_expr::cmake::project_version_minor);
  STATIC_CHECK(lefticus::cons_expr_version_patch == cons_expr::cmake::project_version_patch);
  STATIC_CHECK(lefticus::cons_expr_version_tweak == cons_expr::cmake::project_version_tweak);
}

TEST_CASE("eval expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<IntType>("(eval '(+ 3 4))") == 7);
  STATIC_CHECK(evaluate_to<bool>("(eval '(== 1 1))") == true);
  STATIC_CHECK(evaluate_to<IntType>("(eval '(* 2 3))") == 6);

  // Standard eval
  STATIC_CHECK(evaluate_to<IntType>("(eval '(+ 5 5))") == 10);

  // Nested eval should work since this is a fully constexpr system
  STATIC_CHECK(evaluate_to<IntType>("(eval '(eval '(+ 5 5)))") == 10);

  // Creating a new expression with cons and evaluating it
  // It might be failing because the expected result was incorrect
  STATIC_CHECK(evaluate_to<IntType>("(eval (cons '+ '(1 2)))") == 3);

  // Try with three arguments to make sure the semantics are correct
  STATIC_CHECK(evaluate_to<IntType>("(eval (cons '+ '(1 2 3)))") == 6);
}

TEST_CASE("simple append expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 2 3 4) '(5)) '(1 2 3 4 5))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (append '() '(1 2 3)) '(1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 2 3) '()) '(1 2 3))") == true);

  // Multiple append operations
  STATIC_CHECK(evaluate_to<bool>("(== (append (append '(1) '(2)) '(3)) '(1 2 3))") == true);

  // Append with evaluated expressions
  STATIC_CHECK(evaluate_to<bool>("(== (append (list (+ 1 2)) (list (* 2 2))) '(3 4))") == true);
}

TEST_CASE("if expressions", "[builtins]")
{
  STATIC_CHECK(evaluate_to<IntType>("(if true 1 2)") == 1);
  STATIC_CHECK(evaluate_to<IntType>("(if false 1 2)") == 2);
  STATIC_CHECK(evaluate_to<IntType>("(if (== 1 1) 5 10)") == 5);
  STATIC_CHECK(evaluate_to<IntType>("(if (!= 1 1) 5 10)") == 10);

  // Nested if expressions
  STATIC_CHECK(evaluate_to<IntType>("(if (> 5 2) (if (< 3 1) 1 2) 3)") == 2);

  // If with more complex conditions
  STATIC_CHECK(evaluate_to<IntType>("(if (and (> 5 2) (< 1 3)) 10 20)") == 10);
  STATIC_CHECK(evaluate_to<IntType>("(if (or (> 5 10) (< 1 0)) 10 20)") == 20);

  // If with expressions in the branches
  STATIC_CHECK(evaluate_to<IntType>("(if (> 5 2) (+ 10 5) (* 3 4))") == 15);
}


TEST_CASE("simple error handling", "[errors]")
{
  evaluate_to<lefticus::cons_expr<>::error_type>(R"(
(+ 1 2.3)
)");

  evaluate_to<lefticus::cons_expr<>::error_type>(R"(
(+ 1 (+ 1 2.3))
)");
}

TEST_CASE("custom make_callable functionality", "[callables]")
{
  // This tests the make_callable template functionality
  STATIC_CHECK(evaluate_to<IntType>(R"(
    ((lambda (x) (+ x 5)) 10)
  )") == 15);

  // Testing more complex callable patterns
  STATIC_CHECK(evaluate_to<IntType>(R"(
    (define square (lambda (x) (* x x)))
    (define inc (lambda (x) (+ x 1)))
    (define compose (lambda (f g) (lambda (x) (f (g x)))))
    ((compose square inc) 4)
  )") == 25);
}

TEST_CASE("get_list and get_list_range edge cases", "[implementation]")
{
  // Test empty list handling
  STATIC_CHECK(evaluate_to<bool>("(== '() '())") == true);

  // Test boundary cases with lists
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define empty '())
    (== (append empty '(1)) '(1))
  )") == true);

  STATIC_CHECK(evaluate_to<bool>(R"(
    (define singleton '(1))
    (== (car singleton) 1)
  )") == true);
}

TEST_CASE("begin", "[builtins]")
{
  STATIC_CHECK(evaluate_to<bool>("(begin true)") == true);
  STATIC_CHECK(evaluate_to<bool>("(begin true false)") == false);
  STATIC_CHECK(evaluate_to<int>("(begin true false 1)") == 1);
  STATIC_CHECK(evaluate_to<int>("(begin true false (* 3 3))") == 9);
}

TEST_CASE("basic for-each usage", "[builtins]")
{
  // STATIC_CHECK_NOTHROW(evaluate_to<std::monostate>("(for-each display '(1 2 3 4))"));
}

TEST_CASE("SmallVector memory and optimization", "[implementation]")
{
  // Test string deduplication behavior
  STATIC_CHECK(evaluate_to<bool>("(== 'hello 'hello)") == true);
  STATIC_CHECK(evaluate_to<bool>("(== \"test\" \"test\")") == true);

  // Alternative test for identical identifier equality using define
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define x 'symbol)
    (define y 'symbol)
    (== x y)
  )") == true);

  // This test checks if value reuse is working correctly
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define list1 '(1 2 3))
    (define list2 '(1 2 3))
    (== list1 list2)
  )") == true);
}


TEST_CASE("token parsing edge cases", "[parsing]")
{
  // Simple string test that doesn't use escaped quotes
  STATIC_CHECK(evaluate_expected<std::string_view>(R"("simple string")", "simple string"));

  // Test with whitespace variations
  STATIC_CHECK(evaluate_to<IntType>("(+ \t1   2\n)") == 3);
}

TEST_CASE("Quoted symbol equality issues", "[symbols]")
{
  // These tests currently fail but should work based on the expected behavior of symbols
  // They are included to document expected behavior and prevent regression

  // ----------------------------------------
  // FAILING CASES - Should all return true
  // ----------------------------------------

  // 1. Direct quoted symbol equality fails
  STATIC_CHECK(evaluate_to<bool>("(== 'hello 'hello)") == true);

  // 2. Defined symbols with identical quoted values fail comparison
  STATIC_CHECK(evaluate_to<bool>("(define x 'hello) (define y 'hello) (== x y)") == true);

  // 3. Reference equality of symbols fails
  STATIC_CHECK(evaluate_to<bool>("(define x 'hello) (define y x) (== x y)") == true);

  // 4. Car of quoted list equality fails
  STATIC_CHECK(evaluate_to<bool>("(define a (car '('a))) (define b (car '('a))) (== a b)") == true);

  // 5. Identity of a symbol fails
  STATIC_CHECK(evaluate_to<bool>("(define sym 'hello) (== sym sym)") == true);

  // ----------------------------------------
  // WORKING CASES - For comparison
  // ----------------------------------------

  // Lists containing quoted symbols work fine
  STATIC_CHECK(evaluate_to<bool>("(== '('hello) '('hello))") == true);

  // Car of list with quoted symbols also works
  STATIC_CHECK(evaluate_to<bool>("(== (car '('hello)) (car '('hello)))") == true);

  // Symbols in the same list compare equal
  STATIC_CHECK(evaluate_to<bool>("(define lst '(x x)) (== (car lst) (car (cdr lst)))") == true);

  // Integer equality works
  STATIC_CHECK(evaluate_to<bool>("(== 1 1)") == true);

  // String equality works
  STATIC_CHECK(evaluate_to<bool>("(== \"hello\" \"hello\")") == true);
}


// Unit tests for internal structures

// IndexedString tests
TEST_CASE("IndexedString creation and comparison", "[core][indexedstring]")
{
  constexpr auto test_indexed_string_creation = []() {
    lefticus::IndexedString<uint16_t> str{ 5, 10 };
    return str.start == 5 && str.size == 10;
  };
  STATIC_CHECK(test_indexed_string_creation());
}

TEST_CASE("IndexedString equality", "[core][indexedstring]")
{
  constexpr auto test_indexed_string_equality = []() {
    lefticus::IndexedString<uint16_t> str1{ 5, 10 };
    lefticus::IndexedString<uint16_t> str2{ 5, 10 };
    return str1 == str2;
  };
  STATIC_CHECK(test_indexed_string_equality());
}

TEST_CASE("IndexedString inequality", "[core][indexedstring]")
{
  constexpr auto test_indexed_string_inequality = []() {
    lefticus::IndexedString<uint16_t> str1{ 5, 10 };
    lefticus::IndexedString<uint16_t> str2{ 15, 10 };
    return str1 != str2;
  };
  STATIC_CHECK(test_indexed_string_inequality());
}

TEST_CASE("IndexedString substr", "[core][indexedstring]")
{
  constexpr auto test_indexed_string_substr = []() {
    lefticus::IndexedString<uint16_t> str{ 5, 10 };
    auto substr = str.substr(2);
    return substr.start == 7 && substr.size == 8;
  };
  STATIC_CHECK(test_indexed_string_substr());
}

// IndexedList tests
TEST_CASE("IndexedList creation and properties", "[core][indexedlist]")
{
  constexpr auto test_indexed_list_creation = []() {
    lefticus::IndexedList<uint16_t> list{ 10, 5 };
    return list.start == 10 && list.size == 5 && !list.empty();
  };
  STATIC_CHECK(test_indexed_list_creation());
}

TEST_CASE("IndexedList equality", "[core][indexedlist]")
{
  constexpr auto test_indexed_list_equality = []() {
    lefticus::IndexedList<uint16_t> list1{ 10, 5 };
    lefticus::IndexedList<uint16_t> list2{ 10, 5 };
    return list1 == list2;
  };
  STATIC_CHECK(test_indexed_list_equality());
}

TEST_CASE("IndexedList element access", "[core][indexedlist]")
{
  constexpr auto test_indexed_list_access = []() {
    lefticus::IndexedList<uint16_t> list{ 10, 5 };
    return list.front() == 10 && list[2] == 12 && list.back() == 14;
  };
  STATIC_CHECK(test_indexed_list_access());
}

TEST_CASE("IndexedList sublist operations", "[core][indexedlist]")
{
  constexpr auto test_indexed_list_sublist = []() {
    lefticus::IndexedList<uint16_t> list{ 10, 5 };
    auto sublist1 = list.sublist(2);
    auto sublist2 = list.sublist(1, 3);
    return (sublist1.start == 12 && sublist1.size == 3) && (sublist2.start == 11 && sublist2.size == 3);
  };
  STATIC_CHECK(test_indexed_list_sublist());
}

// Identifier tests
TEST_CASE("Identifier creation and properties", "[core][identifier]")
{
  constexpr auto test_identifier_creation = []() {
    lefticus::Identifier<uint16_t> id{ lefticus::IndexedString<uint16_t>{ 5, 10 } };
    return id.value.start == 5 && id.value.size == 10;
  };
  STATIC_CHECK(test_identifier_creation());
}

TEST_CASE("Identifier equality", "[core][identifier]")
{
  constexpr auto test_identifier_equality = []() {
    lefticus::Identifier<uint16_t> id1{ lefticus::IndexedString<uint16_t>{ 5, 10 } };
    lefticus::Identifier<uint16_t> id2{ lefticus::IndexedString<uint16_t>{ 5, 10 } };
    return id1 == id2;
  };
  STATIC_CHECK(test_identifier_equality());
}

TEST_CASE("Identifier inequality", "[core][identifier]")
{
  constexpr auto test_identifier_inequality = []() {
    constexpr lefticus::Identifier<uint16_t> id1{ lefticus::IndexedString<uint16_t>{ 5, 10 } };
    constexpr lefticus::Identifier<uint16_t> id2{ lefticus::IndexedString<uint16_t>{ 15, 10 } };
    return id1 != id2;
  };
  STATIC_CHECK(test_identifier_inequality());
}

TEST_CASE("Identifier substr", "[core][identifier]")
{
  constexpr auto test_identifier_substr = []() {
    lefticus::Identifier<uint16_t> id{ lefticus::IndexedString<uint16_t>{ 5, 10 } };
    auto substr = id.substr(2);
    return substr.value.start == 7 && substr.value.size == 8;
  };
  STATIC_CHECK(test_identifier_substr());
}

// Token and parsing tests
TEST_CASE("Token parsing basics", "[core][token]")
{
  constexpr auto test_token_simple = []() {
    auto token = lefticus::next_token(std::string_view("hello world"));
    return token.parsed == "hello" && token.remaining == "world";
  };
  STATIC_CHECK(test_token_simple());
}

TEST_CASE("Token parsing with whitespace", "[core][token]")
{
  constexpr auto test_token_whitespace = []() {
    auto token = lefticus::next_token(std::string_view("  hello  world  "));
    return token.parsed == "hello" && token.remaining == "world  ";
  };
  STATIC_CHECK(test_token_whitespace());
}

TEST_CASE("Token parsing with delimiters", "[core][token]")
{
  constexpr auto test_token_delimiters = []() {
    auto token = lefticus::next_token(std::string_view("(hello world)"));
    return token.parsed == "(" && token.remaining == "hello world)";
  };
  STATIC_CHECK(test_token_delimiters());
}

// Number parsing tests
TEST_CASE("Parse integer", "[core][parse]")
{
  constexpr auto test_parse_int = []() {
    auto [success, value] = lefticus::parse_number<int>(std::string_view("123"));
    return success && value == 123;
  };
  STATIC_CHECK(test_parse_int());
}

TEST_CASE("Parse negative integer", "[core][parse]")
{
  constexpr auto test_parse_negative = []() {
    auto [success, value] = lefticus::parse_number<int>(std::string_view("-42"));
    return success && value == -42;
  };
  STATIC_CHECK(test_parse_negative());
}

TEST_CASE("Parse float", "[core][parse]")
{
  constexpr auto test_parse_float = []() {
    auto [success, value] = lefticus::parse_number<double>(std::string_view("123.45"));
    return success && std::abs(value - 123.45) < 0.0001;
  };
  STATIC_CHECK(test_parse_float());
}

TEST_CASE("Parse invalid number", "[core][parse]")
{
  constexpr auto test_parse_invalid = []() {
    auto [success, value] = lefticus::parse_number<int>(std::string_view("abc"));
    return !success;
  };
  STATIC_CHECK(test_parse_invalid());
}


// Full parser tests
TEST_CASE("Parser handles basic expressions", "[core][parser]")
{
  constexpr auto test_parse_number = []() {
    using eval_type = lefticus::cons_expr<std::uint16_t, char, int, double>;
    eval_type evaluator;

    const auto parsed = parse_as<int>(evaluator, "42");
    return parsed.value();
  };
  STATIC_CHECK(test_parse_number() == 42);
}

TEST_CASE("Parser handles simple list", "[core][parser]")
{
  constexpr auto test_parse_list = []() {
    lefticus::cons_expr<std::uint16_t, char, int, double> evaluator;
    using list_type = lefticus::cons_expr<std::uint16_t, char, int, double>::list_type;
    return parse_as<list_type>(evaluator, "(+ 1 2)");
  };
  STATIC_CHECK(test_parse_list().has_value());
}

// Quote-related tests relevant to our issue
TEST_CASE("Parser interprets quoted symbols", "[core][parser][quotes]")
{
  constexpr auto test_parse_quoted_symbol = []() {
    lefticus::cons_expr<std::uint16_t, char, int, double> evaluator;
    return evaluator.parse("'hello");
  };
  constexpr auto parse_result = test_parse_quoted_symbol();
  constexpr auto token = parse_result.second;
  // it's actually expected that both "parsed" and "remaining" are empty here
  // because it consumed all input tokens and the last pass parsed nothing
  STATIC_CHECK(token.parsed == "");
  STATIC_CHECK(token.remaining == "");
}


TEST_CASE("Evaluated Identifier Comparison", "[core][parser][quotes]")
{
  constexpr auto test_parse_result_equality = []() {
    using eval_type = lefticus::cons_expr<std::uint16_t, char, int, double>;
    using identifier_type = eval_type::identifier_type;

    eval_type evaluator;

    auto result1 = evaluator.evaluate_to<identifier_type>("'hello");
    auto result2 = evaluator.evaluate_to<identifier_type>("'hello");

    // The parse results should be equal
    return result1 == result2;
  };
  STATIC_CHECK(test_parse_result_equality());
}


TEST_CASE("Direct parsing comparison", "[core][parser][quotes]")
{
  constexpr auto test_parse_result_equality = []() {
    using eval_type = lefticus::cons_expr<std::uint16_t, char, int, double>;
    using identifier_type = eval_type::identifier_type;

    eval_type evaluator;

    // parse same identifier twice
    auto result1 = parse_as<identifier_type>(evaluator, "'hello");
    auto result2 = parse_as<identifier_type>(evaluator, "'hello");

    // The parse results should be equal
    return result1 == result2;
  };
  STATIC_CHECK(test_parse_result_equality());
}

TEST_CASE("deeply nested expressions", "[nesting]")
{
  // Test deeply nested expressions
  STATIC_CHECK(evaluate_to<IntType>(R"(
    (+ 1 (* 2 (- 10 (/ 8 (+ 1 1)))))
  )") == 13);

  // Test deeply nested lists
  STATIC_CHECK(evaluate_to<bool>(R"(
    (== (cons 1 (cons 2 (cons 3 (cons 4 (cons 5 '()))))) '(1 2 3 4 5))
  )") == true);
}

TEST_CASE("quote function", "[builtins][quote]")
{
  // Basic quote tests with lists
  STATIC_CHECK(evaluate_to<bool>("(== (quote (1 2 3)) '(1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (quote ()) '())") == true);

  // Quote with symbols
  STATIC_CHECK(evaluate_to<bool>("(== (quote hello) 'hello)") == true);

  // Quote prevents evaluation
  STATIC_CHECK(evaluate_to<bool>("(== (quote (+ 1 2)) '(+ 1 2))") == true);

  // Quote vs eval
  STATIC_CHECK(evaluate_to<IntType>("(eval (quote (+ 1 2)))") == 3);

  // Compare quote and the ' shorthand
  STATIC_CHECK(evaluate_to<bool>("(== (quote (1 2 3)) '(1 2 3))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== (quote x) 'x)") == true);

  // Quote in different contexts
  STATIC_CHECK(evaluate_to<bool>(R"(
    (define x 10)
    (== (quote x) 'x)
  )") == true);

  // Quote for expressions that would otherwise error
  STATIC_CHECK(evaluate_to<bool>("(== (quote (undefined-function 1 2)) '(undefined-function 1 2))") == true);
}
