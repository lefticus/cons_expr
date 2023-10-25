#include <cons_expr/cons_expr.hpp>
#include <format>

using cons_expr_type = lefticus::cons_expr<std::uint16_t, char, long long, long double>;

constexpr long long add(long long x, long long y) { return x + y; }

consteval auto make_scripted_function()
{
  cons_expr_type evaluator;

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


  [[maybe_unused]] const auto result = evaluator.sequence(
    evaluator.global_scope, std::get<typename cons_expr_type::list_type>(evaluator.parse(input).first.value));

  return std::bind_front(evaluator.make_callable<long long(long long, long long)>("sum"), evaluator);
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
