#ifndef CONS_EXPR_UTILITY_HPP
#define CONS_EXPR_UTILITY_HPP

#include "cons_expr.hpp"

#include <format>

namespace lefticus {
template<typename> inline constexpr bool is_cons_expr_v = false;

template<std::size_t BuiltInSymbolsSize,
  std::size_t BuiltInStringsSize,
  std::size_t BuiltInValuesSize,
  typename... UserTypes>
inline constexpr bool
  is_cons_expr_v<lefticus::cons_expr<BuiltInSymbolsSize, BuiltInStringsSize, BuiltInValuesSize, UserTypes...>> = true;

template<typename T>
concept ConsExpr = is_cons_expr_v<T>;


template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::SExpr &input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const bool input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const double input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const int input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::Closure &);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const std::monostate &);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::Atom &input);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const typename Eval::function_ptr &);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const lefticus::IndexedList &list);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const lefticus::LiteralList &list);
template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const lefticus::IndexedString &string);


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

template<ConsExpr Eval> std::string to_string(const Eval &engine, bool annotate, const lefticus::Identifier &id)
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

template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const double input)
{
  std::string result;
  if (annotate) { result = "[double] "; }

  return result + std::format("{}", input);
}

template<ConsExpr Eval> std::string to_string(const Eval &, bool annotate, const int input)
{
  std::string result;
  if (annotate) { result = "[int] "; }
  return result + std::format("{}", input);
}

template<ConsExpr Eval> std::string to_string(const Eval &, bool, const typename Eval::FunctionPtr &func)
{
  return std::format("[function_ptr {}]", reinterpret_cast<const void *>(func.ptr));
}

template<ConsExpr Eval> std::string to_string(const Eval &engine, bool annotate, const lefticus::IndexedList &list)
{
  std::string result;

  if (annotate) { result += std::format("[list] {{{}, {}}} ", list.start, list.size); }
  result += "(";

  if (!list.empty()) {
    for (const auto &item : engine.values[list.sublist(0, list.size - 1)]) { result += to_string(engine, false, item) + ' '; }
    result += to_string(engine, false, engine.values[list.back()]);
  }
  result += ")";
  return result;
}

template<ConsExpr Eval> std::string to_string(const Eval &engine, bool annotate, const lefticus::LiteralList &list)
{
  std::string result;
  if (annotate) { result += std::format("[literal list] {{{}, {}}} ", list.items.start, list.items.size); }
  return result + "'" + to_string(engine, false, list.items);
}

template<ConsExpr Eval> std::string to_string(const Eval &engine, bool annotate, const lefticus::IndexedString &string)
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
}

#endif