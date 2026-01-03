#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>

#include <cons_expr/cons_expr.hpp>
#include <string_view>
#include <variant>

template<typename char_type> using cons_expr_type = lefticus::cons_expr<std::uint16_t, char_type>;

namespace {
void display(cons_expr_type<char>::int_type value) { std::cout << value << '\n'; }

auto evaluate(std::basic_string_view<char> input)
{
  cons_expr_type<char> evaluator;

  evaluator.template add<display>("display");

  auto parse_result = evaluator.parse(input);
  return evaluator.sequence(evaluator.global_scope, parse_result.first);
}

template<typename Result, typename char_type = char> Result evaluate_to(std::basic_string_view<char_type> input)
{
  return std::get<Result>(std::get<typename cons_expr_type<char_type>::Atom>(evaluate(input).value));
}

template<typename char_type = char> auto evaluate_non_char(std::basic_string_view<char_type> input)
{
  cons_expr_type<char_type> evaluator;

  auto parse_result = evaluator.parse(input);
  return evaluator.sequence(evaluator.global_scope, parse_result.first);
}

template<typename Result, typename char_type = char>
Result evaluate_non_char_to(std::basic_string_view<char_type> input)
{
  return std::get<Result>(std::get<typename cons_expr_type<char_type>::Atom>(evaluate_non_char(input).value));
}
}// namespace

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


TEST_CASE("member functions", "[function]")
{
  struct Test
  {
    void set(int new_i) { m_i = new_i; }

    [[nodiscard]] int get() const { return m_i; }

    int m_i{ 0 };
  };

  lefticus::cons_expr<std::uint16_t, char, int, float, 500, 500, 500, Test *> evaluator;
  evaluator.add<&Test::set>("set");
  evaluator.add<&Test::get>("get");


  auto eval = [&](const std::string_view input) {
    return evaluator.sequence(evaluator.global_scope, evaluator.parse(input).first);
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

TEST_CASE("SmallVector error handling", "[core][smallvector]")
{
  constexpr auto test_smallvector_error = []() {
    // Create a SmallVector with small capacity
    lefticus::SmallVector<uint16_t, char, 2, char, std::string_view> vec{};

    // Add elements until we reach capacity
    vec.push_back('a');
    vec.push_back('b');

    // This should set error_state to true
    vec.push_back('c');

    // Check that error_state is set
    return vec.error_state == true && vec.size() == static_cast<uint16_t>(2);
  };

  STATIC_CHECK(test_smallvector_error());
}

TEST_CASE("SmallVector const operator[]", "[core][smallvector]")
{
  constexpr auto test_const_access = []() {
    lefticus::SmallVector<uint16_t, char, 5, char, std::string_view> vec{};
    vec.push_back('a');
    vec.push_back('b');
    vec.push_back('c');

    // Create a const reference and access elements
    const auto &const_vec = vec;
    return const_vec[static_cast<uint16_t>(0)] == 'a' && const_vec[static_cast<uint16_t>(1)] == 'b'
           && const_vec[static_cast<uint16_t>(2)] == 'c';
  };

  STATIC_CHECK(test_const_access());
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
