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
  lefticus::cons_expr<>::Context context;

  return evaluator.sequence(context, evaluator.parse(input).first.to_list(evaluator));
}

template<typename Result> constexpr Result evaluate_to(std::string_view input)
{
  return std::get<Result>(std::get<lefticus::cons_expr<>::Atom>(evaluate(input).value));
}

TEST_CASE("test constexpr construction") {
  auto eval = build_cons_expr();
  //CHECK(eval.eval_to<int>("x") == 42);
}

TEST_CASE("Operator identifiers", "[operators]") {
  STATIC_CHECK(evaluate_to<int>("((if false + *) 3 4)") == 12);
  STATIC_CHECK(evaluate_to<int>("((if true + *) 3 4)") == 7);
}

TEST_CASE("basic float operators", "[operators]")
{
  STATIC_CHECK(evaluate_to<double>("(+ 1.0 0.1)") == 1.1);
  STATIC_CHECK(evaluate_to<double>("(+ 0.0 1.0e-1)") == 1.0e-1);
  STATIC_CHECK(evaluate_to<double>("(+ 0.0 0.1e1)") == 0.1e1);
}


TEST_CASE("basic string_view operators", "[operators]") {
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
}

TEST_CASE("basic define usage", "[define]")
{
  STATIC_CHECK(evaluate_to<int>("(define x 32) x") == 32);
  STATIC_CHECK(evaluate_to<int>("(define x (lambda (y)(+ y 4))) (x 10)") == 14);
  STATIC_CHECK(evaluate_to<int>("(define x 42) (define l (lambda (x)(+ x 4))) (l 10)") == 14);
}

TEST_CASE("binary short circuiting", "[short circuiting]")
{
  STATIC_CHECK(evaluate_to<bool>("(and false (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(or true (unknownfunc))") == true);
  STATIC_CHECK(evaluate_to<bool>("(< 2 1 (unknownfunc))") == false);
  STATIC_CHECK(evaluate_to<bool>("(> 1 2 (unknownfunc))") == false);
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


TEST_CASE("basic for-each usage", "[builtins]")
{
  // STATIC_CHECK_NOTHROW(evaluate_to<std::monostate>("(for-each display '(1 2 3 4))"));
}
