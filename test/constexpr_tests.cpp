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

TEST_CASE("Operator identifiers", "[operators]")
{
  STATIC_CHECK(evaluate_to<IntType>("((if false + *) 3 4)") == 12);
  STATIC_CHECK(evaluate_to<IntType>("((if true + *) 3 4)") == 7);
}

TEST_CASE("basic float operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<FloatType>("(+ 1.0 0.1)") == FloatType{ 1.1 });
  STATIC_CHECK(evaluate_to<FloatType>("(+ 0.0 1.0e-1)") == FloatType{ 1.0e-1 });
  STATIC_CHECK(evaluate_to<FloatType>("(+ 0.0 0.1e1)") == FloatType{ 0.1e1 });
}


TEST_CASE("basic string_view operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<bool>(R"((== "hello" "hello"))") == true);
}

TEST_CASE("access as string_view", "[strings]")
{
  STATIC_CHECK(evaluate_expected<std::string_view>(R"("hello")", "hello"));
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
}

TEST_CASE("list comparisons", "[operators]") { STATIC_CHECK(evaluate_to<bool>("(== '(1) '(1))") == true); }

TEST_CASE("basic integer comparisons", "[operators]")
{
  STATIC_CHECK(evaluate_to<bool>("(== 12 12)") == true);
  STATIC_CHECK(evaluate_to<bool>("(== 12 12 12)") == true);

  STATIC_CHECK(evaluate_to<bool>("(< 12 3 1)") == false);
  STATIC_CHECK(evaluate_to<bool>("(> 12 3 1)") == true);
  STATIC_CHECK(evaluate_to<bool>("(>= 12 3 12)") == false);
  STATIC_CHECK(evaluate_to<bool>("(>= 12 12 1)") == true);
  STATIC_CHECK(evaluate_to<bool>("(>= 12 12 1 12)") == false);
}

TEST_CASE("basic logical boolean operations", "[operators]")
{
  STATIC_CHECK(evaluate_to<bool>("(and true true false)") == false);
  STATIC_CHECK(evaluate_to<bool>("(or false true false true)") == true);
  STATIC_CHECK(evaluate_to<bool>("(not false)") == true);
  STATIC_CHECK(evaluate_to<bool>("(not true)") == false);
  STATIC_CHECK(evaluate_to<bool>("(not false)") == true);
}

TEST_CASE("basic lambda usage", "[lambdas]")
{
  STATIC_CHECK(evaluate_to<bool>("((lambda () true))") == true);
  STATIC_CHECK(evaluate_to<bool>("((lambda () false))") == false);
  STATIC_CHECK(evaluate_to<bool>("((lambda (x) x) true)") == true);
  STATIC_CHECK(evaluate_to<IntType>("((lambda (x) (* x x)) 11)") == 121);
}

TEST_CASE("nested lambda usage", "[lambdas]")
{
  STATIC_CHECK(evaluate_to<IntType>("(define l (lambda (x) (lambda () x))) ((l 1))") == 1);
  STATIC_CHECK(evaluate_to<IntType>("(define l (lambda (x) (lambda (y) (lambda () (+ x y))))) (((l 1) 3))") == 4);
  STATIC_CHECK(evaluate_to<IntType>("(define l (lambda (x) (lambda (y) (let ((z (+ x y))) z)))) ((l 1) 3)") == 4);
  STATIC_CHECK(evaluate_to<IntType>("(define l (lambda (x) (lambda (y) (let ((z 10)) (+ x y z))))) ((l 1) 3)") == 14);
  STATIC_CHECK(evaluate_to<IntType>("((lambda (x) (let ((x (+ x 5))) x)) 2)") == 7);
}

TEST_CASE("basic define usage", "[define]")
{
  STATIC_CHECK(evaluate_to<IntType>("(define x 32) x") == 32);
  STATIC_CHECK(evaluate_to<IntType>("(define x (lambda (y)(+ y 4))) (x 10)") == 14);
}

TEST_CASE("define scoping", "[define]")
{
  STATIC_CHECK(evaluate_to<IntType>("((lambda () (define y 20)  y))") == 20);
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


  //  STATIC_CHECK(evaluate_to<IntType>(R"(
  //(if (>= 5 3) 'true 'false)
  //
  //)") == 5);
}

TEST_CASE("binary short circuiting", "[short circuiting]")
{
  STATIC_CHECK(evaluate_to<bool>("(and false (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(or true (unknownfunc))") == true);
  STATIC_CHECK(evaluate_to<bool>("(< 2 1 (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(> 1 2 (unknownfunc))") == false);
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
}

TEST_CASE("simple car expression", "[builtins]") { STATIC_CHECK(evaluate_to<IntType>("(car '(1 2 3 4))") == 1); }

TEST_CASE("simple cdr expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(1 2 3 4)) '(2 3 4))") == true);
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
}


TEST_CASE("simple cons expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<bool>("(== ( cons '(1 2 3 4) '(5) ) '((1 2 3 4) 5))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== ( cons 1 '(5) ) '(1 5))") == true);
  STATIC_CHECK(evaluate_to<bool>("(== ( cons 'x '(5) ) '(x 5))") == true);
}

TEST_CASE("apply expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<IntType>("(apply * '(2 3))") == 6);

  STATIC_CHECK(evaluate_to<IntType>(
                 R"(
(define x 10)

(define add-x (lambda (y) (+ x y)))

(let ((x 20))
  (apply add-x (list 5)))
)") == 15);
}

TEST_CASE("check version number", "[system]")
{
  STATIC_CHECK(lefticus::cons_expr_version_major == cons_expr::cmake::project_version_major);
  STATIC_CHECK(lefticus::cons_expr_version_minor == cons_expr::cmake::project_version_minor);
  STATIC_CHECK(lefticus::cons_expr_version_patch == cons_expr::cmake::project_version_patch);
  STATIC_CHECK(lefticus::cons_expr_version_tweak == cons_expr::cmake::project_version_tweak);
}

TEST_CASE("eval expression", "[builtins]") { STATIC_CHECK(evaluate_to<IntType>("(eval '(+ 3 4))") == 7); }

TEST_CASE("simple append expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 2 3 4) '(5)) '(1 2 3 4 5))") == true);
}


TEST_CASE("simple do expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<IntType>("(do () (true 0))") == 0);

  STATIC_CHECK(evaluate_to<IntType>(R"(
(do ((i 1 (+ i 1))
     (sum 0 (+ sum i)))
    ((> i 10) sum)
)
)") == 55);
}

TEST_CASE("simple error handling", "[errors]")
{
  evaluate_to<lefticus::cons_expr<>::error_type>(R"(
(+ 1 2.3)
)");

  evaluate_to<lefticus::cons_expr<>::error_type>(R"(
(define x (do (b) (true 0)))
(eval x)
)");

  evaluate_to<lefticus::cons_expr<>::error_type>(R"(
(+ 1 (+ 1 2.3))
)");
}


TEST_CASE("scoped do expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<IntType>(R"(

((lambda (count)
   (do ((i 1 (+ i 1))
         (sum 0 (+ sum i)))
        ((> i count) sum)
   )
) 10)

)") == 55);
}

TEST_CASE("basic for-each usage", "[builtins]")
{
  // STATIC_CHECK_NOTHROW(evaluate_to<std::monostate>("(for-each display '(1 2 3 4))"));
}
