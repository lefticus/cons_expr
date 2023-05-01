#include <catch2/catch_test_macros.hpp>


#include <cons_expr/cons_expr.hpp>

auto evaluate(std::string_view input)
{
  lefticus::cons_expr<> evaluator;
  lefticus::cons_expr<>::Context context;

  return evaluator.sequence(context, std::get<lefticus::cons_expr<>::List>(evaluator.parse(input).first.value));
}

template<typename Result> Result evaluate_to(std::string_view input)
{
  return std::get<Result>(std::get<lefticus::cons_expr<>::Atom>(evaluate(input).value));
}

TEST_CASE("basic float operators", "[operators]")
{
  CHECK(evaluate_to<double>("(+ 1.0 0.1)") == 1.1);
  CHECK(evaluate_to<double>("(+ 0.0 1.0e-1)") == 1.0e-1);
  CHECK(evaluate_to<double>("(+ 0.0 0.1e1)") == 0.1e1);
}

TEST_CASE("basic integer operators", "[operators]")
{
  CHECK(evaluate_to<int>("(+ 1 2)") == 3);
  CHECK(evaluate_to<int>("(/ 2 2)") == 1);
  CHECK(evaluate_to<int>("(- 2 2)") == 0);
  CHECK(evaluate_to<int>("(* 2 2)") == 4);

  CHECK(evaluate_to<int>("(+ 1 2 3 -6)") == 0);
  CHECK(evaluate_to<int>("(/ 4 2 1)") == 2);
  CHECK(evaluate_to<int>("(- 2 2 1)") == -1);
  CHECK(evaluate_to<int>("(* 2 2 2 2 2)") == 32);
}

TEST_CASE("basic integer comparisons", "[operators]")
{
  CHECK(evaluate_to<bool>("(< 12 3 1)") == false);
  CHECK(evaluate_to<bool>("(> 12 3 1)") == true);
  CHECK(evaluate_to<bool>("(>= 12 3 12)") == false);
  CHECK(evaluate_to<bool>("(>= 12 12 1)") == true);
  CHECK(evaluate_to<bool>("(>= 12 12 1 12)") == false);
}

TEST_CASE("basic logical boolean operations", "[operators]")
{
  CHECK(evaluate_to<bool>("(and true true false)") == false);
  CHECK(evaluate_to<bool>("(or false true false true)") == true);
}

TEST_CASE("basic lambda usage", "[lambdas]")
{
  CHECK(evaluate_to<bool>("((lambda () true))") == true);
  CHECK(evaluate_to<bool>("((lambda () false))") == false);
  CHECK(evaluate_to<bool>("((lambda (x) x) true)") == true);
}

TEST_CASE("basic callable usage", "[c++ api]")
{
  lefticus::cons_expr<> evaluator;
  auto func = evaluator.make_callable<int(int, int, int)>("+");
  CHECK(func(1, 2, 3) == 6);

  auto func2 = evaluator.make_callable<int(int)>("(lambda (x) (* x x))");
  CHECK(func2(10) == 100);
}


TEST_CASE("basic for-each usage", "[builtins]")
{
  // CHECK_NOTHROW(evaluate_to<std::monostate>("(for-each display '(1 2 3 4))"));
}

struct UDT
{
};

template<typename Result> Result evaluate_to_with_UDT(std::string_view input)
{
  lefticus::cons_expr<UDT> evaluator;
  lefticus::cons_expr<UDT>::Context context;

  auto parsed = evaluator.parse(input);
  const auto &items = std::get<lefticus::cons_expr<UDT>::List>(parsed.first.value);

  if (!items.empty()) {
    for (std::size_t idx = 0; idx < items.size() - 1; ++idx) { evaluator.eval(context, items[idx]); }
  }
  return evaluator.eval(context, std::get<lefticus::cons_expr<UDT>::List>(parsed.first.value).front());
  return std::get<Result>(std::get<lefticus::cons_expr<>::Atom>(evaluate(input).value));
}
