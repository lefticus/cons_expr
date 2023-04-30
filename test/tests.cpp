#include <catch2/catch_test_macros.hpp>


#include <cons_expr/cons_expr.hpp>

auto evaluate(std::string_view input)
{
  schmxpr::Schmxpr<> evaluator;
  schmxpr::Schmxpr<>::Context context;

  auto parsed = evaluator.parse(input);
  const auto &items = std::get<schmxpr::Schmxpr<>::List>(parsed.first.value);

  if (!items.empty()) {
    for (std::size_t idx = 0; idx < items.size() - 1; ++idx) { evaluator.eval(context, items[idx]); }
  }
  return evaluator.eval(context, std::get<schmxpr::Schmxpr<>::List>(parsed.first.value).back());
}

template<typename Result> Result evaluate_to(std::string_view input)
{
  return std::get<Result>(std::get<schmxpr::Schmxpr<>::Atom>(evaluate(input).value));
}


TEST_CASE("basic integer operators", "[operators]")
{
  REQUIRE(evaluate_to<int>("(+ 1 2)") == 3);
  REQUIRE(evaluate_to<int>("(/ 2 2)") == 1);
  REQUIRE(evaluate_to<int>("(- 2 2)") == 0);
  REQUIRE(evaluate_to<int>("(* 2 2)") == 4);

  REQUIRE(evaluate_to<int>("(+ 1 2 3 -6)") == 0);
  REQUIRE(evaluate_to<int>("(/ 4 2 1)") == 2);
  REQUIRE(evaluate_to<int>("(- 2 2 1)") == -1);
  REQUIRE(evaluate_to<int>("(* 2 2 2 2 2)") == 32);
}

TEST_CASE("basic integer comparisons", "[operators]")
{
  REQUIRE(evaluate_to<bool>("(< 12 3 1)") == false);
  REQUIRE(evaluate_to<bool>("(> 12 3 1)") == true);
  REQUIRE(evaluate_to<bool>("(>= 12 3 12)") == false);
  REQUIRE(evaluate_to<bool>("(>= 12 12 1)") == true);
}

TEST_CASE("basic logical boolean operations", "[operators]")
{
  REQUIRE(evaluate_to<bool>("(and true true false)") == false);
  REQUIRE(evaluate_to<bool>("(xor false true false true)") == false);
  REQUIRE(evaluate_to<bool>("(xor false true false false)") == true);
  REQUIRE(evaluate_to<bool>("(or false true false true)") == true);
}

TEST_CASE("basic lambda usage", "[lambdas]")
{
  REQUIRE(evaluate_to<bool>("((lambda () true))") == true);
  REQUIRE(evaluate_to<bool>("((lambda () false))") == false);
  REQUIRE(evaluate_to<bool>("((lambda (x) x) true)") == true);
}

TEST_CASE("basic for-each usage", "[builtins]")
{
  //REQUIRE_NOTHROW(evaluate_to<std::monostate>("(for-each display '(1 2 3 4))"));
}

struct UDT
{

};

template<typename Result> Result evaluate_to_with_UDT(std::string_view input)
{
  schmxpr::Schmxpr<UDT> evaluator;
  schmxpr::Schmxpr<UDT>::Context context;

  auto parsed = evaluator.parse(input);
  const auto &items = std::get<schmxpr::Schmxpr<UDT>::List>(parsed.first.value);

  if (!items.empty()) {
    for (std::size_t idx = 0; idx < items.size() - 1; ++idx) { evaluator.eval(context, items[idx]); }
  }
  return evaluator.eval(context, std::get<schmxpr::Schmxpr<UDT>::List>(parsed.first.value).front());
  return std::get<Result>(std::get<schmxpr::Schmxpr<>::Atom>(evaluate(input).value));
}
