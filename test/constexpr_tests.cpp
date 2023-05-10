#include <catch2/catch_test_macros.hpp>


#include <cons_expr/cons_expr.hpp>

static_assert(std::is_trivially_copyable_v<lefticus::cons_expr<>::SExpr>);

// we'll be exactly 16k, because we can
static_assert(sizeof(lefticus::cons_expr<>) == 16384);

consteval auto build_cons_expr()
{
  lefticus::cons_expr<> result;
  result.add("x", 42);
  return result;
}


constexpr auto evaluate(std::string_view input)
{
  lefticus::cons_expr<> evaluator;

  return evaluator.sequence(evaluator.global_scope, evaluator.parse(input).first.to_list(evaluator));
}

template<typename Result> constexpr Result evaluate_to(std::string_view input)
{
  return std::get<Result>(std::get<lefticus::cons_expr<>::Atom>(evaluate(input).value));
}

TEST_CASE("test constexpr construction")
{
  auto eval = build_cons_expr();
  // CHECK(eval.eval_to<int>("x") == 42);
}

TEST_CASE("Operator identifiers", "[operators]")
{
  STATIC_CHECK(evaluate_to<int>("((if false + *) 3 4)") == 12);
  STATIC_CHECK(evaluate_to<int>("((if true + *) 3 4)") == 7);
}

TEST_CASE("basic float operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<double>("(+ 1.0 0.1)") == 1.1);
  STATIC_CHECK(evaluate_to<double>("(+ 0.0 1.0e-1)") == 1.0e-1);
  STATIC_CHECK(evaluate_to<double>("(+ 0.0 0.1e1)") == 0.1e1);
}


TEST_CASE("basic string_view operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<bool>(R"((== "hello" "hello"))") == true);
}

TEST_CASE("basic integer operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<int>("(+ 1 2)") == 3);
  STATIC_CHECK(evaluate_to<int>("(/ 2 2)") == 1);
  STATIC_CHECK(evaluate_to<int>("(- 2 2)") == 0);
  STATIC_CHECK(evaluate_to<int>("(* 2 2)") == 4);

  STATIC_CHECK(evaluate_to<int>("(+ 1 2 3 -6)") == 0);
  STATIC_CHECK(evaluate_to<int>("(/ 4 2 1)") == 2);
  STATIC_CHECK(evaluate_to<int>("(- 2 2 1)") == -1);
  STATIC_CHECK(evaluate_to<int>("(* 2 2 2 2 2)") == 32);
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
}

TEST_CASE("basic lambda usage", "[lambdas]")
{
  STATIC_CHECK(evaluate_to<bool>("((lambda () true))") == true);
  STATIC_CHECK(evaluate_to<bool>("((lambda () false))") == false);
  STATIC_CHECK(evaluate_to<bool>("((lambda (x) x) true)") == true);
  STATIC_CHECK(evaluate_to<int>("((lambda (x) (* x x)) 11)") == 121);
}

TEST_CASE("nested lambda usage", "[lambdas]")
{
  STATIC_CHECK(evaluate_to<int>("(define l (lambda (x) (lambda () x))) ((l 1))") == 1);
  STATIC_CHECK(evaluate_to<int>("(define l (lambda (x) (lambda (y) (lambda () (+ x y))))) (((l 1) 3))") == 4);
  STATIC_CHECK(evaluate_to<int>("(define l (lambda (x) (lambda (y) (let ((z (+ x y))) z)))) ((l 1) 3)") == 4);
  STATIC_CHECK(evaluate_to<int>("(define l (lambda (x) (lambda (y) (let ((z 10)) (+ x y z))))) ((l 1) 3)") == 14);
  STATIC_CHECK(evaluate_to<int>("((lambda(x) (let ((x (+ x 5))) x)) 2)") == 7);
}

TEST_CASE("basic define usage", "[define]")
{
  STATIC_CHECK(evaluate_to<int>("(define x 32) x") == 32);
  STATIC_CHECK(evaluate_to<int>("(define x (lambda (y)(+ y 4))) (x 10)") == 14);
}

TEST_CASE("define scoping", "[define]")
{
  STATIC_CHECK(evaluate_to<int>("((lambda () (define y 20)  y))") == 20);
  STATIC_CHECK(evaluate_to<int>("((lambda (x) (define y 20) (+ x y)) 10)") == 30);
  STATIC_CHECK(evaluate_to<int>("((lambda (x) (define y (* x 2)) y) 20)") == 40);
  STATIC_CHECK(evaluate_to<int>("(define x 42) (define l (lambda (x)(+ x 4))) (l 10)") == 14);
  STATIC_CHECK(evaluate_to<int>(R"(
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


TEST_CASE("binary short circuiting", "[short circuiting]")
{
  STATIC_CHECK(evaluate_to<bool>("(and false (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(or true (unknownfunc))") == true);
  STATIC_CHECK(evaluate_to<bool>("(< 2 1 (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(> 1 2 (unknownfunc))") == false);
}

TEST_CASE("let variables", "[let variables]")
{
  STATIC_CHECK(evaluate_to<int>("(let ((x 3)(y 14)) (* x y))") == 42);
  STATIC_CHECK(evaluate_to<int>("(let ((x (* 3 1))(y (- 18 4))) (* x y))") == 42);
  STATIC_CHECK(evaluate_to<int>("(define x 42) (let ((l (lambda (x)(+ x 4)))) (l 10))") == 14);
  // let variable initial values are scoped to the outer scope, not to previously
  // declared variables in scope
  STATIC_CHECK(evaluate_to<int>("(define x 42) (let ((x 10)(y x)) y)") == 42);
}

TEST_CASE("simple car expression", "[builtins]") { STATIC_CHECK(evaluate_to<int>("(car '(1 2 3 4))") == 1); }

TEST_CASE("simple cdr expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<bool>("(== (cdr '(1 2 3 4)) '(2 3 4))") == true);
}

TEST_CASE("comments", "[parsing]")
{
  STATIC_CHECK(evaluate_to<int>(
                 R"(
15
)") == 15);

  STATIC_CHECK(evaluate_to<int>(
                 R"(
; a comment
15
)") == 15);

  STATIC_CHECK(evaluate_to<int>(
                 R"(
15 ; a comment
)") == 15);

  STATIC_CHECK(evaluate_to<int>(
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
  STATIC_CHECK(evaluate_to<int>("(apply * '(2 3))") == 6);

  STATIC_CHECK(evaluate_to<int>(
                 R"(
(define x 10)

(define add-x (lambda (y) (+ x y)))

(let ((x 20))
  (apply add-x (list 5)))
)") == 15);

}


TEST_CASE("eval expression", "[builtins]") { STATIC_CHECK(evaluate_to<int>("(eval '(+ 3 4))") == 7); }

TEST_CASE("simple append expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<bool>("(== (append '(1 2 3 4) '(5)) '(1 2 3 4 5))") == true);
}


TEST_CASE("simple do expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<int>(R"(
(do ((i 1 (+ i 1))
     (sum 0 (+ sum i)))
    ((> i 10) sum)
)
)") == 55);
}


TEST_CASE("scoped do expression", "[builtins]")
{
  STATIC_CHECK(evaluate_to<int>(R"(

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
