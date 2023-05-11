#include <cons_expr/cons_expr.hpp>
#include <iostream>

constexpr int add(int x, int y) {
  return x + y;
}

void display(int i) {
  std::cout << i << '\n';
}

auto evaluate(std::string_view input)
{
  lefticus::cons_expr<> evaluator;

  evaluator.add<&add>("add");
  evaluator.add<&display>("display");

  return evaluator.sequence(evaluator.global_scope, evaluator.parse(input).first.to_list(evaluator));
}

template<typename Result> Result evaluate_to(std::string_view input)
{
  return std::get<Result>(std::get<lefticus::cons_expr<>::Atom>(evaluate(input).value));
}

int main()
{
  evaluate(R"(
(display (do ((i 1 (+ i 1))
     (count 0 (add count 1)))
    ((> i 10000000) count)
))
)");

}
