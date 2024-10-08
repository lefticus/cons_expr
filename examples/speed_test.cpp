#include <cons_expr/cons_expr.hpp>
#include <iostream>

constexpr long long add(long long x, long long y) { return x + y; }

void display(long long i) { std::cout << i << '\n'; }

using cons_expr_type = lefticus::cons_expr<std::uint16_t, char, long long, double>;

auto evaluate(std::string_view input)
{
  cons_expr_type evaluator;

  evaluator.add<&add>("add");
  evaluator.add<&display>("display");

  return evaluator.sequence(
    evaluator.global_scope, std::get<typename cons_expr_type::list_type>(evaluator.parse(input).first.value));
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
