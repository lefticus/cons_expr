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

  return evaluator.sequence(evaluator.global_scope, evaluator.parse(input).first.to_list());
}

template<typename Result> Result evaluate_to(std::string_view input)
{
  return std::get<Result>(std::get<lefticus::cons_expr<>::Atom>(evaluate(input).value));
}

int main()
{
  evaluate(R"(
(define count 
  (lambda (min max)
    (display
      (do ((i min (+ i 1))
           (value 0 (add value 1)))
          ((> i max) value)
      )
    )
  )
)

(count 1 1000000)
(count 10 1000000)
(count -10 10000)
)");

}
