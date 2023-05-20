#include <cons_expr/cons_expr.hpp>
#include <format>

constexpr int add(int x, int y) { return x + y; }

consteval auto make_scripted_function()
{
  lefticus::cons_expr<> evaluator;

  evaluator.add<&add>("add");

  constexpr static std::string_view input =
    R"(
(define sum
  (lambda (min max) 
    (do ((i min (+ i 1))
         (sum 0 (+ sum i)))
        ((> i max) sum)
    )
  )
)
    )";


  [[maybe_unused]] const auto result =
      evaluator.sequence(evaluator.global_scope, std::get<lefticus::IndexedList>(evaluator.parse(input).first.value));

  return evaluator.make_standalone_callable<int(int, int)>("sum");
}


int main()
{
  // the kicker here is that this lambda is a full self contained script environment
  // that was all parsed and optimized at compile-time
  auto func = make_scripted_function();

  auto print_sum = [&](int from, int to) {
    std::puts(std::format("sum({} to {}) = {}", from, to, func(from, to).value()).c_str());
  };

  print_sum(101, 132414);
  print_sum(1, 1222222);
  print_sum(-10, 10);
}
