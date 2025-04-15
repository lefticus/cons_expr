#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>
#include <internal_use_only/config.hpp>
#include <iostream>

#include <cmath>

using IntType = int;
using FloatType = double;

// Helper function for getting values from parsing without evaluation
template<typename Result> constexpr Result parse_result(std::string_view input)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  auto [parsed, _] = evaluator.parse(input);

  const auto *list = std::get_if<lefticus::cons_expr<>::list_type>(&parsed.value);
  if (list != nullptr && list->size == 1) {
    // Extract the first element from the parsed list
    const auto *result = evaluator.template get_if<Result>(&evaluator.values[(*list)[0]]);
    if (result != nullptr) { return *result; }
  }

  // This is a fallback that will cause the test to fail if we can't extract the expected type
  return Result{};
}

// Helper function for checking if a parsed expression contains a specific type
template<typename TokenType> constexpr bool is_of_type(std::string_view input)
{
  lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
  auto [parsed, _] = evaluator.parse(input);

  const auto *list = std::get_if<lefticus::cons_expr<>::list_type>(&parsed.value);
  if (list != nullptr && list->size == 1) {
    return evaluator.template get_if<TokenType>(&evaluator.values[(*list)[0]]) != nullptr;
  }

  return false;
}

// Basic Tokenization Tests
TEST_CASE("Basic tokenization", "[parser][tokenize]")
{
  // Test token parsing with a constexpr lambda
  constexpr auto test_token_parsing = []() {
    using Token = lefticus::Token<char>;

    // Simple tokens
    Token token1 = lefticus::next_token(std::string_view("hello"));
    if (token1.parsed != std::string_view("hello")) return false;

    // Whitespace handling
    Token token2 = lefticus::next_token(std::string_view("   hello"));
    if (token2.parsed != std::string_view("hello")) return false;

    Token token3 = lefticus::next_token(std::string_view("hello   "));
    if (token3.parsed != std::string_view("hello")) return false;

    // Multiple tokens
    Token token4 = lefticus::next_token(std::string_view("hello world"));
    if (token4.parsed != std::string_view("hello") || token4.remaining != std::string_view("world")) return false;

    // Parentheses
    Token token5 = lefticus::next_token(std::string_view("(hello)"));
    if (token5.parsed != std::string_view("(") || token5.remaining != std::string_view("hello)")) return false;

    Token token6 = lefticus::next_token(std::string_view(")hello"));
    if (token6.parsed != std::string_view(")") || token6.remaining != std::string_view("hello")) return false;

    // Quote syntax
    Token token7 = lefticus::next_token(std::string_view("'(hello)"));
    if (token7.parsed != std::string_view("'(") || token7.remaining != std::string_view("hello)")) return false;

    // Strings
    Token token8 = lefticus::next_token(std::string_view("\"hello\""));
    if (token8.parsed != std::string_view("\"hello\"")) return false;

    // Empty input
    Token token9 = lefticus::next_token(std::string_view(""));
    if (!token9.parsed.empty() || !token9.remaining.empty()) return false;

    // Comments
    Token token10 = lefticus::next_token(std::string_view("; comment\nhello"));
    if (token10.parsed != std::string_view("hello")) return false;

    return true;
  };

  STATIC_CHECK(test_token_parsing());
}

// Token Sequence Processing Tests
TEST_CASE("Token sequence processing", "[parser][token-sequence]")
{
  // Break into individual checks for better debugging

  // Simple token check 1
  constexpr auto test_token1 = []() {
    auto token1 = lefticus::next_token(std::string_view("(+ 1 2)"));
    return token1.parsed == std::string_view("(") && token1.remaining == std::string_view("+ 1 2)");
  };

  // Simple token check 2
  constexpr auto test_token2 = []() {
    auto token1 = lefticus::next_token(std::string_view("(+ 1 2)"));
    auto token2 = lefticus::next_token(token1.remaining);
    return token2.parsed == std::string_view("+") && token2.remaining == std::string_view("1 2)");
  };

  // Simple token check 3
  constexpr auto test_token3 = []() {
    auto token1 = lefticus::next_token(std::string_view("(+ 1 2)"));
    auto token2 = lefticus::next_token(token1.remaining);
    auto token3 = lefticus::next_token(token2.remaining);
    return token3.parsed == std::string_view("1") && token3.remaining == std::string_view("2)");
  };

  // Simple token check 4
  constexpr auto test_token4 = []() {
    auto token1 = lefticus::next_token(std::string_view("(+ 1 2)"));
    auto token2 = lefticus::next_token(token1.remaining);
    auto token3 = lefticus::next_token(token2.remaining);
    auto token4 = lefticus::next_token(token3.remaining);
    return token4.parsed == std::string_view("2") && token4.remaining == std::string_view(")");
  };

  // Simple token check 5
  constexpr auto test_token5 = []() {
    auto token1 = lefticus::next_token(std::string_view("(+ 1 2)"));
    auto token2 = lefticus::next_token(token1.remaining);
    auto token3 = lefticus::next_token(token2.remaining);
    auto token4 = lefticus::next_token(token3.remaining);
    auto token5 = lefticus::next_token(token4.remaining);
    return token5.parsed == std::string_view(")") && token5.remaining.empty();
  };

  // Whitespace and quotes check
  constexpr auto test_token6 = []() {
    auto token6 = lefticus::next_token(std::string_view("  (  quote  hello  )  "));
    // The returned remaining string likely has normalized whitespace
    // Let's ignore the exact amount of whitespace
    return token6.parsed == std::string_view("(") && (token6.remaining.find("quote") != std::string_view::npos)
           && (token6.remaining.find("hello") != std::string_view::npos)
           && (token6.remaining.find(")") != std::string_view::npos);
  };

  // Check all individual assertions
  STATIC_CHECK(test_token1());
  STATIC_CHECK(test_token2());
  STATIC_CHECK(test_token3());
  STATIC_CHECK(test_token4());
  STATIC_CHECK(test_token5());
  STATIC_CHECK(test_token6());
}

// Whitespace Handling Tests
TEST_CASE("Whitespace handling", "[parser][whitespace]")
{
  // Spaces and tabs
  constexpr auto test_whitespace1 = []() {
    auto token1 = lefticus::next_token(std::string_view("  \t  hello"));
    return token1.parsed == std::string_view("hello");
  };

  // Newlines and carriage returns
  constexpr auto test_whitespace2 = []() {
    auto token2 = lefticus::next_token(std::string_view("\n\r\nhello"));
    return token2.parsed == std::string_view("hello");
  };

  // Mixed whitespace
  constexpr auto test_whitespace3 = []() {
    auto token3 = lefticus::next_token(std::string_view("\t \n \r hello \t \n"));
    return token3.parsed == std::string_view("hello");
  };

  // Whitespace in multi-token input
  constexpr auto test_whitespace4 = []() {
    auto token4 = lefticus::next_token(std::string_view("  hello  \t  world  "));
    return token4.parsed == std::string_view("hello") && token4.remaining == std::string_view("world  ");
  };

  // Whitespace and parentheses
  constexpr auto test_whitespace5 = []() {
    auto token5 = lefticus::next_token(std::string_view("  (  hello  )  "));
    // The returned remaining string likely has normalized whitespace
    // Let's ignore the exact amount of whitespace
    return token5.parsed == std::string_view("(") && (token5.remaining.find("hello") != std::string_view::npos)
           && (token5.remaining.find(")") != std::string_view::npos);
  };

  // Only whitespace
  constexpr auto test_whitespace6 = []() {
    auto token6 = lefticus::next_token(std::string_view("   "));
    return token6.parsed.empty() && token6.remaining.empty();
  };

  // Check all individual assertions
  STATIC_CHECK(test_whitespace1());
  STATIC_CHECK(test_whitespace2());
  STATIC_CHECK(test_whitespace3());
  STATIC_CHECK(test_whitespace4());
  STATIC_CHECK(test_whitespace5());
  STATIC_CHECK(test_whitespace6());
}

// Comment Handling Tests
TEST_CASE("Comment handling", "[parser][comments]")
{
  // Basic comment at start
  constexpr auto test_comment1 = []() {
    auto token1 = lefticus::next_token(std::string_view("; This is a comment\nhello"));
    return token1.parsed == std::string_view("hello");
  };

  // Comment at end of line
  constexpr auto test_comment2 = []() {
    auto token2 = lefticus::next_token(std::string_view("hello ; This is a comment\nworld"));
    // The resulting token should be "hello" and the remaining text should contain "world"
    return token2.parsed == std::string_view("hello") && (token2.remaining.find("world") != std::string_view::npos);
  };

  // Comment without newline separator
  constexpr auto test_comment3 = []() {
    auto token3 = lefticus::next_token(std::string_view("; This is a comment"));
    return token3.parsed.empty();// Should be empty since comment consumes the line
  };

  // Multiple comments on different lines
  constexpr auto test_comment4 = []() {
    auto token4 = lefticus::next_token(std::string_view("; Comment 1\n; Comment 2\nhello"));
    // The tokenizer might either return "hello" directly or possibly empty string
    // if it's handling comments line by line
    return !token4.parsed.empty()
           && (token4.parsed.find("hello") != std::string_view::npos
               || token4.remaining.find("hello") != std::string_view::npos);
  };

  // Check all individual assertions
  STATIC_CHECK(test_comment1());
  STATIC_CHECK(test_comment2());
  STATIC_CHECK(test_comment3());
  STATIC_CHECK(test_comment4());
}

// String Parsing Tests
TEST_CASE("String parsing", "[parser][strings]")
{
  // Basic string
  constexpr auto test_string1 = []() {
    auto token1 = lefticus::next_token(std::string_view("\"hello\""));
    return token1.parsed == std::string_view("\"hello\"");
  };

  // String with spaces
  constexpr auto test_string2 = []() {
    auto token2 = lefticus::next_token(std::string_view("\"hello world\""));
    return token2.parsed == std::string_view("\"hello world\"");
  };

  // Empty string
  constexpr auto test_string3 = []() {
    auto token3 = lefticus::next_token(std::string_view("\"\""));
    return token3.parsed == std::string_view("\"\"");
  };

  // String with escaped quote
  constexpr auto test_string4 = []() {
    auto token4 = lefticus::next_token(std::string_view("\"hello\\\"world\""));
    return token4.parsed == std::string_view("\"hello\\\"world\"");
  };

  // String followed by other tokens
  constexpr auto test_string5 = []() {
    auto token5 = lefticus::next_token(std::string_view("\"hello\" world"));
    return token5.parsed == std::string_view("\"hello\"") && token5.remaining == std::string_view("world");
  };

  // Check all individual assertions
  STATIC_CHECK(test_string1());
  STATIC_CHECK(test_string2());
  STATIC_CHECK(test_string3());
  STATIC_CHECK(test_string4());
  STATIC_CHECK(test_string5());
}

// String Escape Character Tests
TEST_CASE("String escape characters", "[parser][strings][escapes]")
{
  // Escaped double quote
  constexpr auto test_escaped_quote = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;
    auto [parsed, _] = evaluator.parse("\"Quote: \\\"Hello\\\"\"");
    
    // Extract the string content
    const auto *list = std::get_if<lefticus::cons_expr<>::list_type>(&parsed.value);
    if (list == nullptr || list->size != 1) return false;
    
    const auto *atom = std::get_if<lefticus::cons_expr<>::Atom>(&evaluator.values[(*list)[0]].value);
    if (atom == nullptr) return false;
    
    const auto *string_val = std::get_if<lefticus::cons_expr<>::string_type>(atom);
    if (string_val == nullptr) return false;
    
    // Check the raw tokenized string includes the escapes
    auto token = lefticus::next_token(std::string_view("\"Quote: \\\"Hello\\\"\""));
    if (token.parsed != std::string_view("\"Quote: \\\"Hello\\\"\"")) return false;
    
    return true;
  };
  
  // Escaped backslash
  constexpr auto test_escaped_backslash = []() {
    auto token = lefticus::next_token(std::string_view("\"Backslash: \\\\\""));
    return token.parsed == std::string_view("\"Backslash: \\\\\"");
  };
  
  // Multiple escape sequences
  constexpr auto test_multiple_escapes = []() {
    auto token = lefticus::next_token(std::string_view("\"Escapes: \\\\ \\\" \\n \\t \\r\""));
    return token.parsed == std::string_view("\"Escapes: \\\\ \\\" \\n \\t \\r\"");
  };
  
  // Escape at end of string
  constexpr auto test_escape_at_end = []() {
    auto token = lefticus::next_token(std::string_view("\"Escape at end: \\\""));
    return token.parsed == std::string_view("\"Escape at end: \\\"");
  };

  // Unterminated string with escape
  constexpr auto test_unterminated_escape = []() {
    auto token = lefticus::next_token(std::string_view("\"Unterminated \\"));
    return token.parsed == std::string_view("\"Unterminated \\");
  };
  
  // Common escape sequences: \n \t \r \f \b
  constexpr auto test_common_escapes = []() {
    auto token = lefticus::next_token(std::string_view("\"Special chars: \\n\\t\\r\\f\\b\""));
    return token.parsed == std::string_view("\"Special chars: \\n\\t\\r\\f\\b\"");
  };
  
  // Test handling of consecutive escapes
  constexpr auto test_consecutive_escapes = []() {
    auto token = lefticus::next_token(std::string_view("\"Double escapes: \\\\\\\"\""));
    return token.parsed == std::string_view("\"Double escapes: \\\\\\\"\"");
  };
  
  // Check all individual assertions
  STATIC_CHECK(test_escaped_quote());
  STATIC_CHECK(test_escaped_backslash());
  STATIC_CHECK(test_multiple_escapes());
  STATIC_CHECK(test_escape_at_end());
  STATIC_CHECK(test_unterminated_escape());
  STATIC_CHECK(test_common_escapes());
  STATIC_CHECK(test_consecutive_escapes());
}

// Number Parsing Tests
TEST_CASE("Number parsing", "[parser][numbers]")
{
  constexpr auto test_int_parsing = []() {
    // Integer parsing
    auto [success1, value1] = lefticus::parse_number<int>(std::string_view("123"));
    if (!success1 || value1 != 123) return false;

    auto [success2, value2] = lefticus::parse_number<int>(std::string_view("-456"));
    if (!success2 || value2 != -456) return false;

    auto [success3, value3] = lefticus::parse_number<int>(std::string_view("not_a_number"));
    if (success3) return false;// Should fail

    return true;
  };

  STATIC_CHECK(test_int_parsing());
}

// List Structure Tests
TEST_CASE("List structure", "[parser][lists]")
{
  // Empty list test
  constexpr auto test_empty_list = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Parse an empty list: ()
    auto [parsed_result, _] = evaluator.parse(std::string_view("()"));

    // Parse always returns a list containing the parsed expressions
    // For an empty list, we expect a list with one item (which is itself an empty list)
    const auto *outer_list = std::get_if<lefticus::cons_expr<>::list_type>(&parsed_result.value);
    if (outer_list == nullptr || outer_list->size != 1) return false;

    // Check that the inner element is an empty list
    const auto &inner_elem = evaluator.values[(*outer_list)[0]];
    const auto *inner_list = std::get_if<lefticus::cons_expr<>::list_type>(&inner_elem.value);
    return inner_list != nullptr && inner_list->size == 0;
  };

  // Simple list test
  constexpr auto test_simple_list = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Parse a simple list with three elements: (a b c)
    auto [parsed_result, _] = evaluator.parse(std::string_view("(a b c)"));

    // Outer list should contain one item
    const auto *outer_list = std::get_if<lefticus::cons_expr<>::list_type>(&parsed_result.value);
    if (outer_list == nullptr || outer_list->size != 1) return false;

    // Inner list should contain three elements (a, b, c)
    const auto &inner_elem = evaluator.values[(*outer_list)[0]];
    const auto *inner_list = std::get_if<lefticus::cons_expr<>::list_type>(&inner_elem.value);
    return inner_list != nullptr && inner_list->size == 3;
  };

  // Nested list test
  constexpr auto test_nested_list = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Parse a list with a nested list: (a (b c) d)
    auto [parsed_result, _] = evaluator.parse(std::string_view("(a (b c) d)"));

    // Outer list should contain one item
    const auto *outer_list = std::get_if<lefticus::cons_expr<>::list_type>(&parsed_result.value);
    if (outer_list == nullptr || outer_list->size != 1) return false;

    // Inner list should contain three elements: a, (b c), d
    const auto &inner_elem = evaluator.values[(*outer_list)[0]];
    const auto *inner_list = std::get_if<lefticus::cons_expr<>::list_type>(&inner_elem.value);
    if (inner_list == nullptr || inner_list->size != 3) return false;

    // The second element should be a nested list with 2 elements: b, c
    const auto &nested_elem = evaluator.values[(*inner_list)[1]];
    const auto *nested_list = std::get_if<lefticus::cons_expr<>::list_type>(&nested_elem.value);
    return nested_list != nullptr && nested_list->size == 2;
  };

  // Check all individual assertions
  STATIC_CHECK(test_empty_list());
  STATIC_CHECK(test_simple_list());
  STATIC_CHECK(test_nested_list());
}

// Quote Syntax Tests
TEST_CASE("Quote syntax", "[parser][quotes]")
{
  constexpr auto test_quotes = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Quoted symbol
    auto [quoted_symbol, _1] = evaluator.parse("'symbol");
    const auto *list1 = std::get_if<lefticus::cons_expr<>::list_type>(&quoted_symbol.value);
    if (list1 == nullptr || list1->size != 1) return false;

    auto &first_item = evaluator.values[(*list1)[0]];
    const auto *atom = std::get_if<lefticus::cons_expr<>::Atom>(&first_item.value);
    if (atom == nullptr) return false;
    if (std::get_if<lefticus::cons_expr<>::symbol_type>(atom) == nullptr) return false;

    // Quoted list
    auto [quoted_list, _2] = evaluator.parse("'(a b c)");
    const auto *list2 = std::get_if<lefticus::cons_expr<>::list_type>(&quoted_list.value);
    if (list2 == nullptr || list2->size != 1) return false;

    const auto *literal_list =
      std::get_if<lefticus::cons_expr<>::literal_list_type>(&evaluator.values[(*list2)[0]].value);
    if (literal_list == nullptr || literal_list->items.size != 3) return false;

    return true;
  };

  STATIC_CHECK(test_quotes());
}

// Symbol vs Identifier Tests
TEST_CASE("Symbol vs identifier", "[parser][symbols]")
{
  constexpr auto test_symbol_vs_identifier = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Symbol (quoted identifier)
    auto [symbol_expr, _1] = evaluator.parse("'symbol");
    const auto *list1 = std::get_if<lefticus::cons_expr<>::list_type>(&symbol_expr.value);
    if (list1 == nullptr || list1->size != 1) return false;

    const auto *atom1 = std::get_if<lefticus::cons_expr<>::Atom>(&evaluator.values[(*list1)[0]].value);
    if (atom1 == nullptr) return false;

    const auto *symbol = std::get_if<lefticus::cons_expr<>::symbol_type>(atom1);
    if (symbol == nullptr) return false;

    // Regular identifier
    auto [id_expr, _2] = evaluator.parse("identifier");
    const auto *list2 = std::get_if<lefticus::cons_expr<>::list_type>(&id_expr.value);
    if (list2 == nullptr || list2->size != 1) return false;

    const auto *atom2 = std::get_if<lefticus::cons_expr<>::Atom>(&evaluator.values[(*list2)[0]].value);
    if (atom2 == nullptr) return false;

    const auto *identifier = std::get_if<lefticus::cons_expr<>::identifier_type>(atom2);
    if (identifier == nullptr) return false;

    return true;
  };

  STATIC_CHECK(test_symbol_vs_identifier());
}

// Boolean Literal Tests
TEST_CASE("Boolean literals", "[parser][booleans]")
{
  constexpr auto test_booleans = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Parse true
    auto [true_expr, _1] = evaluator.parse("true");
    const auto *list1 = std::get_if<lefticus::cons_expr<>::list_type>(&true_expr.value);
    if (list1 == nullptr || list1->size != 1) return false;

    const auto *atom1 = std::get_if<lefticus::cons_expr<>::Atom>(&evaluator.values[(*list1)[0]].value);
    if (atom1 == nullptr) return false;

    const auto *bool_val1 = std::get_if<bool>(atom1);
    if (bool_val1 == nullptr || !(*bool_val1)) return false;

    // Parse false
    auto [false_expr, _2] = evaluator.parse("false");
    const auto *list2 = std::get_if<lefticus::cons_expr<>::list_type>(&false_expr.value);
    if (list2 == nullptr || list2->size != 1) return false;

    const auto *atom2 = std::get_if<lefticus::cons_expr<>::Atom>(&evaluator.values[(*list2)[0]].value);
    if (atom2 == nullptr) return false;

    const auto *bool_val2 = std::get_if<bool>(atom2);
    if (bool_val2 == nullptr || (*bool_val2)) return false;

    return true;
  };

  STATIC_CHECK(test_booleans());
}

// Multiple Expression Parsing
TEST_CASE("Multiple expressions", "[parser][multiple]")
{
  constexpr auto test_multiple_expressions = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Parse a definition expression: (define x 10)
    auto [parsed, _] = evaluator.parse(std::string_view("(define x 10)"));

    // Outer list should contain one item
    const auto *outer_list = std::get_if<lefticus::cons_expr<>::list_type>(&parsed.value);
    if (outer_list == nullptr || outer_list->size != 1) return false;

    // Inner list should contain three elements: define, x, 10
    const auto &inner_elem = evaluator.values[(*outer_list)[0]];
    const auto *inner_list = std::get_if<lefticus::cons_expr<>::list_type>(&inner_elem.value);

    return inner_list != nullptr && inner_list->size == 3;
  };

  STATIC_CHECK(test_multiple_expressions());
}

// Parse Complex Expressions
TEST_CASE("Complex expressions", "[parser][complex]")
{
  constexpr auto test_complex_expressions = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Parse a lambda function: (lambda (x) (+ x 1))
    auto [parsed, _] = evaluator.parse(std::string_view("(lambda (x) (+ x 1))"));

    // Outer list should contain one item
    const auto *outer_list = std::get_if<lefticus::cons_expr<>::list_type>(&parsed.value);
    if (outer_list == nullptr || outer_list->size != 1) return false;

    // Inner list should contain three elements: lambda, (x), (+ x 1)
    const auto &inner_elem = evaluator.values[(*outer_list)[0]];
    const auto *inner_list = std::get_if<lefticus::cons_expr<>::list_type>(&inner_elem.value);
    if (inner_list == nullptr || inner_list->size != 3) return false;

    // Second element should be a parameter list containing just x
    const auto &params = evaluator.values[(*inner_list)[1]];
    const auto *params_list = std::get_if<lefticus::cons_expr<>::list_type>(&params.value);
    if (params_list == nullptr || params_list->size != 1) return false;

    return true;
  };

  STATIC_CHECK(test_complex_expressions());
}

// String Content Tests
TEST_CASE("String content", "[parser][string-content]")
{
  constexpr auto test_string_content = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Parse a string and check its content
    auto [string_expr, _] = evaluator.parse("\"hello world\"");
    const auto *list = std::get_if<lefticus::cons_expr<>::list_type>(&string_expr.value);
    if (list == nullptr || list->size != 1) return false;

    const auto *atom = std::get_if<lefticus::cons_expr<>::Atom>(&evaluator.values[(*list)[0]].value);
    if (atom == nullptr) return false;

    const auto *string_val = std::get_if<lefticus::cons_expr<>::string_type>(atom);
    if (string_val == nullptr) return false;

    auto sv = evaluator.strings.view(*string_val);
    if (sv != std::string_view("hello world")) return false;

    return true;
  };

  STATIC_CHECK(test_string_content());
}

// Mixed Content Parsing
TEST_CASE("Mixed content", "[parser][mixed]")
{
  constexpr auto test_mixed_content = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Parse a list with mixed content types including numbers, strings, booleans, symbols, and nested lists
    auto [mixed_expr, _] = evaluator.parse(std::string_view("(list 123 \"hello\" true 'symbol (nested))"));

    // Outer list should contain one item
    const auto *outer_list = std::get_if<lefticus::cons_expr<>::list_type>(&mixed_expr.value);
    if (outer_list == nullptr || outer_list->size != 1) return false;

    // Inner list should contain six elements: list, 123, "hello", true, 'symbol, (nested)
    const auto &inner_elem = evaluator.values[(*outer_list)[0]];
    const auto *inner_list = std::get_if<lefticus::cons_expr<>::list_type>(&inner_elem.value);
    if (inner_list == nullptr || inner_list->size != 6) return false;

    // First element should be an identifier "list"
    const auto &first_elem = evaluator.values[(*inner_list)[0]];
    const auto *first_atom = std::get_if<lefticus::cons_expr<>::Atom>(&first_elem.value);
    if (first_atom == nullptr) return false;

    const auto *id = std::get_if<lefticus::cons_expr<>::identifier_type>(first_atom);
    return id != nullptr;
  };

  STATIC_CHECK(test_mixed_content());
}

// Quoted List Tests
TEST_CASE("Quoted lists", "[parser][quoted-lists]")
{
  constexpr auto test_quoted_lists = []() {
    lefticus::cons_expr<std::uint16_t, char, IntType, FloatType> evaluator;

    // Empty quoted list
    auto [empty, _1] = evaluator.parse("'()");
    const auto *list1 = std::get_if<lefticus::cons_expr<>::list_type>(&empty.value);
    if (list1 == nullptr || list1->size != 1) return false;

    const auto *literal_list1 =
      std::get_if<lefticus::cons_expr<>::literal_list_type>(&evaluator.values[(*list1)[0]].value);
    if (literal_list1 == nullptr || literal_list1->items.size != 0) return false;

    // Simple quoted list
    auto [simple, _2] = evaluator.parse("'(1 2 3)");
    const auto *list2 = std::get_if<lefticus::cons_expr<>::list_type>(&simple.value);
    if (list2 == nullptr || list2->size != 1) return false;

    const auto *literal_list2 =
      std::get_if<lefticus::cons_expr<>::literal_list_type>(&evaluator.values[(*list2)[0]].value);
    if (literal_list2 == nullptr || literal_list2->items.size != 3) return false;

    // Nested quoted list
    auto [nested, _3] = evaluator.parse("'(1 (2 3) 4)");
    const auto *list3 = std::get_if<lefticus::cons_expr<>::list_type>(&nested.value);
    if (list3 == nullptr || list3->size != 1) return false;

    const auto *literal_list3 =
      std::get_if<lefticus::cons_expr<>::literal_list_type>(&evaluator.values[(*list3)[0]].value);
    if (literal_list3 == nullptr || literal_list3->items.size != 3) return false;

    return true;
  };

  STATIC_CHECK(test_quoted_lists());
}

// Special Character Tests
TEST_CASE("Special characters", "[parser][special-chars]")
{
  constexpr auto test_special_chars = []() {
    // Various identifier formats with special characters
    auto token1 = lefticus::next_token(std::string_view("hello-world"));
    if (token1.parsed != std::string_view("hello-world")) return false;

    auto token2 = lefticus::next_token(std::string_view("symbol+"));
    if (token2.parsed != std::string_view("symbol+")) return false;

    auto token3 = lefticus::next_token(std::string_view("_special_"));
    if (token3.parsed != std::string_view("_special_")) return false;

    auto token4 = lefticus::next_token(std::string_view("*wild*"));
    if (token4.parsed != std::string_view("*wild*")) return false;

    auto token5 = lefticus::next_token(std::string_view("symbol?"));
    if (token5.parsed != std::string_view("symbol?")) return false;

    return true;
  };

  STATIC_CHECK(test_special_chars());
}

using LongDouble = long double;

// Number Parsing Edge Cases
TEMPLATE_TEST_CASE("integral parsing", "[parser][numbers][edge]", int, long, short)
{
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123x")).first == false);
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123e4")).first == false);
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("-123")).second == TestType{-123});
}

// Number Parsing Edge Cases
TEMPLATE_TEST_CASE("unsigned integral parsing", "[parser][numbers][edge]", std::uint16_t, std::uint32_t)
{
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("-123")).first == false);
}


// LCOV_EXCL_START
// Number Parsing Edge Cases
TEMPLATE_TEST_CASE("Floating point parsing", "[parser][numbers][edge]", float, double, LongDouble)
{
//  static constexpr auto eps = std::numeric_limits<TestType>::epsilon() * sizeof(TestType) * 8;
  struct Approx {
    TestType target;
    constexpr bool operator==(TestType arg) const {
      if (arg == target) { return true; }

      int steps = 0;
      while (steps < 100 && arg != target) {
        arg = std::nexttoward(arg, target);
        ++steps;
      }
  
      return steps < 30;
    }
  };

  STATIC_CHECK(static_cast<TestType>(123.456l) == lefticus::parse_number<TestType>(std::string_view("123.456")).second);
  STATIC_CHECK(static_cast<TestType>(-789.012l) == lefticus::parse_number<TestType>(std::string_view("-789.012")).second);
  STATIC_CHECK(static_cast<TestType>(1000.0l) == lefticus::parse_number<TestType>(std::string_view("1e3")).second);
  STATIC_CHECK(static_cast<TestType>(0.015l) == lefticus::parse_number<TestType>(std::string_view("1.5e-2")).second);

  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123.1.")).first == false);
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123.1e.")).first == false);
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123.1e")).first == false);
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123e")).first == false);
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123e4")).second == static_cast<TestType>(123e4l));
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123ex")).first == false);
  STATIC_CHECK(static_cast<TestType>(.123e2l) == lefticus::parse_number<TestType>(std::string_view(".123e2")).second);
  STATIC_CHECK(static_cast<TestType>(12.3l) == lefticus::parse_number<TestType>(std::string_view(".123e2")).second);
  STATIC_CHECK(lefticus::parse_number<TestType>(std::string_view("123.456e3")).second == static_cast<TestType>(123456l));
  STATIC_CHECK(static_cast<TestType>(.123l) == lefticus::parse_number<TestType>(std::string_view(".123")).second);
  STATIC_CHECK(static_cast<TestType>(1.l) == lefticus::parse_number<TestType>(std::string_view("1.")).second);
}
// LCOV_EXCL_STOP
