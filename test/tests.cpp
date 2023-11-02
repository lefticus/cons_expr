#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <cons_expr/cons_expr.hpp>

template<typename char_type> using cons_expr_type = lefticus::cons_expr<std::uint16_t, char_type>;

void display(cons_expr_type<char>::int_type i) { std::cout << i << '\n'; }

auto evaluate(std::basic_string_view<char> input)
{
  cons_expr_type<char> evaluator;

  evaluator.template add<display>("display");

  auto parse_result = evaluator.parse(input);
  auto list = std::get<cons_expr_type<char>::list_type>(parse_result.first.value);

  return evaluator.sequence(evaluator.global_scope, list);
}

template<typename Result, typename char_type = char> Result evaluate_to(std::basic_string_view<char_type> input)
{
  return std::get<Result>(std::get<typename cons_expr_type<char_type>::Atom>(evaluate(input).value));
}

template<typename char_type = char> auto evaluate_non_char(std::basic_string_view<char_type> input)
{
  cons_expr_type<char_type> evaluator;

  auto parse_result = evaluator.parse(input);
  auto list = std::get<typename cons_expr_type<char_type>::list_type>(parse_result.first.value);

  return evaluator.sequence(evaluator.global_scope, list);
}

template<typename Result, typename char_type = char>
Result evaluate_non_char_to(std::basic_string_view<char_type> input)
{
  return std::get<Result>(std::get<typename cons_expr_type<char_type>::Atom>(evaluate_non_char(input).value));
}

TEST_CASE("non-char characters", "[c++ api]") { CHECK(evaluate_non_char_to<int, wchar_t>(L"(+ 1 2 3 4)") == 10); }

TEST_CASE("basic callable usage", "[c++ api]")
{
  cons_expr_type<char> evaluator;
  using int_type = cons_expr_type<char>::int_type;

  auto func = evaluator.make_callable<int_type(int_type, int_type, int_type)>("+");
  CHECK(func(evaluator, 1, 2, 3) == 6);

  auto func2 = evaluator.make_callable<int_type(int_type)>("(lambda (x) (* x x))");
  CHECK(func2(evaluator, 10) == 100);
}

TEST_CASE("GPT Generated Tests", "[integration tests]")
{
  CHECK(evaluate_to<typename cons_expr_type<char>::int_type, char>(R"(
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
    return evaluator.sequence(
      evaluator.global_scope, std::get<decltype(evaluator)::list_type>(evaluator.parse(input).first.value));
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
  CHECK_NOTHROW(evaluate_to<std::monostate, char>("(for-each display '(1 2 3 4))"));
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