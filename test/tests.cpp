#include <catch2/catch_test_macros.hpp>
#include <iostream>


#include <cons_expr/cons_expr.hpp>

void display(long long i) { std::cout << i << '\n'; }

auto evaluate(std::string_view input)
{
  lefticus::cons_expr<> evaluator;

  evaluator.add<display>("display");

  auto parse_result = evaluator.parse(input);
  auto list = std::get<typename lefticus::cons_expr<>::list_type>(parse_result.first.value);

  return evaluator.sequence(evaluator.global_scope, list);
}

template<typename Result> Result evaluate_to(std::string_view input)
{
  return std::get<Result>(std::get<lefticus::cons_expr<>::Atom>(evaluate(input).value));
}


TEST_CASE("basic callable usage", "[c++ api]")
{
  lefticus::cons_expr<> evaluator;
  auto func = evaluator.make_callable<long long (long long , long long, long long)>("+");
  CHECK(func(1, 2, 3) == 6);

  auto func2 = evaluator.make_callable<long long(long long)>("(lambda (x) (* x x))");
  CHECK(func2(10) == 100);
}

TEST_CASE("GPT Generated Tests", "[integration tests]")
{
  CHECK(evaluate_to<long long>(R"(
(define make-adder-multiplier
  (lambda (a)
    (lambda (b)
      (do ((i 0 (+ i 1))
           (sum 0 (+ sum (let ((x (+ a i)))
                            (if (>= x b)
                                (define y (* x 2))
                                (define y (* x 3)))
                            (do ((j 0 (+ j 1))
                                 (inner-sum 0 (+ inner-sum y)))
                                ((>= j i) inner-sum))))))
          ((>= i 5) sum)))))

((make-adder-multiplier 2) 3)
)") == 100);
}

TEST_CASE("member functions", "[function]")
{
  struct Test
  {
    void set(int i) { m_i = i; }

    int get() const { return m_i; }

    int m_i{ 0 };
  };

  lefticus::cons_expr<std::uint16_t, char, int, float, 100, 100, 100, Test *> evaluator;
  evaluator.add<&Test::set>("set");
  evaluator.add<&Test::get>("get");


  auto eval = [&](const std::string_view input) {
    return evaluator.sequence(evaluator.global_scope, std::get<decltype(evaluator)::list_type>(evaluator.parse(input).first.value));
  };

  Test myobj;

  myobj.m_i = 42;

  evaluator.add("myobj", &myobj);

  CHECK(myobj.m_i == 42);
  eval("(set myobj 10)");
  CHECK(myobj.m_i == 10);
  eval("(set myobj (+ (get myobj) 12)");
  CHECK(myobj.m_i == 22);

  CHECK(evaluator.template eval_to<int>(evaluator.global_scope, eval("(get myobj)")) == 22);
}

TEST_CASE("basic for-each usage", "[builtins]")
{
  CHECK_NOTHROW(evaluate_to<std::monostate>("(for-each display '(1 2 3 4))"));
}

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