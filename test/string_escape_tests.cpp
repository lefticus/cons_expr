#include <catch2/catch_test_macros.hpp>

#include <cons_expr/cons_expr.hpp>
#include <cstdint>
#include <string_view>
#include <variant>

using IntType = int;
using FloatType = double;

namespace {
template<typename Result> constexpr Result evaluate_to(std::string_view input)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  return evaluator.evaluate_to<Result>(input).value();
}

template<typename Result> constexpr bool evaluate_expected(std::string_view input, auto result)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  return evaluator.evaluate_to<Result>(input).value() == result;
}
}

TEST_CASE("String escape processing", "[string][escape]")
{
  // Test basic string with no escapes
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello world\"", "hello world"));

  // Test each escape sequence
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\nworld\"", "hello\nworld"));
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\tworld\"", "hello\tworld"));
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\rworld\"", "hello\rworld"));
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\fworld\"", "hello\fworld"));
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\bworld\"", "hello\bworld"));

  // Test escaped quotes and backslashes
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\\"world\"", "hello\"world"));
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\\\world\"", "hello\\world"));

  // Test multiple escapes in a single string
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\n\\tworld\\r\"", "hello\n\tworld\r"));

  // Test escapes at start and end
  STATIC_CHECK(evaluate_expected<std::string_view>("\"\\nhello\"", "\nhello"));
  STATIC_CHECK(evaluate_expected<std::string_view>("\"hello\\n\"", "hello\n"));

  // Test empty string with escapes
  STATIC_CHECK(evaluate_expected<std::string_view>("\"\\n\"", "\n"));
  STATIC_CHECK(evaluate_expected<std::string_view>("\"\\t\\r\\n\"", "\t\r\n"));
}

TEST_CASE("String escape error cases", "[string][escape][error]")
{
  // Create an evaluator for checking error cases
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

  // Test invalid escape sequence
  auto invalid_escape = evaluator.evaluate(R"("hello\xworld")");
  REQUIRE(std::holds_alternative<lefticus::Error<std::uint16_t>>(invalid_escape.value));

  // Test unterminated escape at end of string
  auto unterminated_escape = evaluator.evaluate(R"("hello\")");
  REQUIRE(std::holds_alternative<lefticus::Error<std::uint16_t>>(unterminated_escape.value));
}

TEST_CASE("String operations on escaped strings", "[string][escape][operations]")
{
  // Test comparing strings with escapes
  STATIC_CHECK(evaluate_to<bool>("(== \"hello\\nworld\" \"hello\\nworld\")") == true);
  STATIC_CHECK(evaluate_to<bool>("(== \"hello\\nworld\" \"hello\\tworld\")") == false);

  // Test using escaped strings in expressions
  STATIC_CHECK(evaluate_expected<std::string_view>(R"(
    (let ((greeting "Hello\nWorld!"))
      greeting)
  )",
    "Hello\nWorld!"));

  // Test string predicates with escaped strings
  STATIC_CHECK(evaluate_to<bool>("(string? \"hello\\nworld\")") == true);
}

TEST_CASE("String escape edge cases", "[string][escape][edge]")
{
  // Test consecutive escapes
  STATIC_CHECK(evaluate_expected<std::string_view>("\"\\n\\r\\t\"", "\n\r\t"));

  // Test empty string
  STATIC_CHECK(evaluate_expected<std::string_view>("\"\"", ""));

  // Test string with just an escaped character
  STATIC_CHECK(evaluate_expected<std::string_view>("\"\\n\"", "\n"));
}

// Branch Coverage Enhancement Tests - Missing String Cases

TEST_CASE("String escape error conditions for coverage", "[string][escape][coverage]")
{
  constexpr auto test_unknown_escape = []() constexpr {
    lefticus::cons_expr<> engine;

    // Test unknown escape character
    auto bad_escape = engine.process_string_escapes("test\\q");
    return std::holds_alternative<decltype(engine)::error_type>(bad_escape.value);
  };
  STATIC_CHECK(test_unknown_escape());

  constexpr auto test_unterminated_escape = []() constexpr {
    lefticus::cons_expr<> engine;

    // Test unterminated escape (string ends with backslash)
    auto unterminated = engine.process_string_escapes("test\\");
    return std::holds_alternative<decltype(engine)::error_type>(unterminated.value);
  };
  STATIC_CHECK(test_unterminated_escape());
}
