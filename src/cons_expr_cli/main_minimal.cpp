#include <cons_expr/cons_expr.hpp>


#include <cons_expr/cons_expr.hpp>


constexpr auto evaluate(std::string_view input)
{
  lefticus::cons_expr evaluator;

  const auto result = evaluator.parse(input).first.value;

  return evaluator.sequence(evaluator.global_scope, *std::get_if<typename lefticus::cons_expr<>::list_type>(&result));
}


int main(int argc, const char **argv) { evaluate(argv[0]); }
