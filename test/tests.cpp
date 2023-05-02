#include <catch2/catch_test_macros.hpp>


#include <cons_expr/cons_expr.hpp>

auto evaluate(std::string_view input)
{
  lefticus::cons_expr<> evaluator;
  lefticus::cons_expr<>::Context context;

  auto parse_result = evaluator.parse(input);
  auto list = parse_result.first.to_list(evaluator);

  return evaluator.sequence(context, list);
}

template<typename Result> Result evaluate_to(std::string_view input)
{
  return std::get<Result>(std::get<lefticus::cons_expr<>::Atom>(evaluate(input).value));
}


TEST_CASE("basic callable usage", "[c++ api]")
{
  lefticus::cons_expr<> evaluator;
  auto func = evaluator.make_callable<int(int, int, int)>("+");
  CHECK(func(1, 2, 3) == 6);

  auto func2 = evaluator.make_callable<int(int)>("(lambda (x) (* x x))");
  CHECK(func2(10) == 100);
}

//TEST_CASE("basic for-each usage", "[builtins]")
//{
//  CHECK_NOTHROW(evaluate_to<std::monostate>("(for-each display '(1 2 3 4))"));
//}

/*
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
  */