#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>
#include <fmt/format.h>
#include <string>
#include <string_view>

// Fuzzer that tests the cons_expr parser and evaluator
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
  // Create a string view from the fuzz data
  std::string_view script(reinterpret_cast<const char *>(data), size);

  // Initialize the cons_expr evaluator
  lefticus::cons_expr<> evaluator;

  // Try to parse the script
  auto [parse_result, remaining] = evaluator.parse(script);

  // Evaluate the parsed expression
  // Don't care about the result, just want to make sure nothing crashes
  [[maybe_unused]] auto result = evaluator.sequence(evaluator.global_scope, parse_result);

  return 0;// Non-zero return values are reserved for future use
}