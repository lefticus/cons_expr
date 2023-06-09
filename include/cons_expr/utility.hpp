#ifndef CONS_EXPR_UTILITY_HPP
#define CONS_EXPR_UTILITY_HPP

#include "cons_expr.hpp"

#include <format>

namespace lefticus {
template<typename> inline constexpr bool is_cons_expr_v = false;

template<typename SizeType,
  typename CharType,
  typename IntType,
  typename FloatType,
  SizeType BuiltInSymbolsSize,
  SizeType BuiltInStringsSize,
  SizeType BuiltInValuesSize,
  typename... UserTypes>
inline constexpr bool is_cons_expr_v<lefticus::cons_expr<SizeType,
  CharType,
  IntType,
  FloatType,
  BuiltInSymbolsSize,
  BuiltInStringsSize,
  BuiltInValuesSize,
  UserTypes...>> = true;

template<typename T>
concept ConsExpr = is_cons_expr_v<T>;


template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::SExpr &input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const bool input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::float_type input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::int_type input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::Closure &);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const std::monostate &);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::Atom &input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::function_ptr &);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::list_type &list);
template<ConsExpr Eval>
std::string to_string(const Eval &, bool annotate, const typename Eval::literal_list_type &list);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::string_type &string);
template<ConsExpr Eval> std::string to_string(const Eval &, bool, const typename Eval::error_type &);


template<ConsExpr Eval>
std::string to_string([[maybe_unused]] const Eval &eval, bool, const typename Eval::error_type &error)
{
  return std::format(
    "[Expected: {} got: {}]", to_string(eval, false, error.expected), to_string(eval, false, error.got));
}

template<ConsExpr Eval>
std::string to_string([[maybe_unused]] const Eval &, bool, const typename Eval::Closure &closure)
{
  return std::format("[closure parameters {{{}, {}}} statements {{{}, {}}}]",
    closure.parameter_names.start,
    closure.parameter_names.size,
    closure.statements.start,
    closure.statements.size);
}

template<ConsExpr Eval> std::string to_string([[maybe_unused]] const Eval &, bool, const std::monostate &)
{
  return "[nil]";
}


template<ConsExpr Eval>
std::string to_string(const Eval &engine, bool annotate, const typename Eval::identifier_type &id)
{
  if (annotate) {
    return std::format("[identifier] {{{}, {}}} {}", id.value.start, id.value.size, engine.strings.view(id.value));
  } else {
    return std::string{ engine.strings.view(id.value) };
  }
}


template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const bool input)
{
  std::string result;
  if (annotate) { result = "[bool] "; }
  if (input) {
    return result + "true";
  } else {
    return result + "false";
  }
}

template<ConsExpr Eval> std::string to_string(const Eval &engine, bool annotate, const typename Eval::Atom &input)
{
  return std::visit([&](const auto &value) { return to_string(engine, annotate, value); }, input);
}

template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::float_type input)
{
  std::string result;
  if (annotate) { result = "[floating_point] "; }

  return result + std::format("{}", input);
}

template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::int_type input)
{
  std::string result;
  if (annotate) { result = "[int] "; }
  return result + std::format("{}", input);
}

template<ConsExpr Eval> std::string to_string(const Eval &, bool, const typename Eval::FunctionPtr &func)
{
  return std::format("[function_ptr {}]", reinterpret_cast<const void *>(func.ptr));
}


template<ConsExpr Eval> std::string to_string(const Eval &engine, bool annotate, const typename Eval::list_type &list)
{
  std::string result;

  if (annotate) { result += std::format("[list] {{{}, {}}} ", list.start, list.size); }
  result += "(";

  if (!list.empty()) {
    for (const auto &item : engine.values[list.sublist(0, list.size - 1)]) {
      result += to_string(engine, false, item) + ' ';
    }
    result += to_string(engine, false, engine.values[list.back()]);
  }
  result += ")";
  return result;
}

template<ConsExpr Eval>
std::string to_string(const Eval &engine, bool annotate, const typename Eval::literal_list_type &list)
{
  std::string result;
  if (annotate) { result += std::format("[literal list] {{{}, {}}} ", list.items.start, list.items.size); }
  return result + "'" + to_string(engine, false, list.items);
}

template<ConsExpr Eval>
std::string to_string(const Eval &engine, bool annotate, const typename Eval::string_type &string)
{
  if (annotate) {
    return std::format("[identifier] {{{}, {}}} \"{}\"", string.start, string.size, engine.strings.view(string));
  } else {
    return std::format("\"{}\"", engine.strings.view(string));
  }
}

template<ConsExpr Eval> std::string to_string(const Eval &engine, bool annotate, const typename Eval::SExpr &input)
{
  return std::visit([&](const auto &value) { return to_string(engine, annotate, value); }, input.value);
}
}// namespace lefticus

#endif