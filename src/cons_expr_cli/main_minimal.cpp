#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>
#include <iostream>

constexpr auto evaluate(std::string_view input)
{
  lefticus::cons_expr evaluator;

  const auto [parse_result, remaining] = evaluator.parse(input);
  const auto result = evaluator.sequence(evaluator.global_scope, parse_result);

  std::cout << "Result: " << lefticus::to_string(evaluator, false, result) << '\n';
}


int main(int argc, const char **argv) { evaluate(argv[1]); }
