#include <catch2/catch_test_macros.hpp>


#include <cons_expr/cons_expr.hpp>

static_assert(std::is_trivially_copyable_v<lefticus::cons_expr<>::SExpr>);

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
