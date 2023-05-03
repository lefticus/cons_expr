#ifndef CONS_EXPR_HPP
#define CONS_EXPR_HPP

#include <fmt/format.h>

#include <charconv>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <stacktrace>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

// https://en.wikipedia.org/wiki/Greenspun%27s_tenth_rule
//
// Any sufficiently complicated C or Fortran program contains an ad hoc,
// informally-specified, bug-ridden, slow implementation of half of Common Lisp.

/*
┌─────────────────────────┐┌─────────────────────────┐┌──────────────────────────┐
│ LISP is over half a     ││ I wonder if the cycles  ││   /  These are your    \ │
│ century old and it      ││ will continue forever.  ││   |father's parentheses| │
│ still has this perfect  ││\________________  _____/││  / ____________________/ │
│ timeless air about it.  ││              /-\|/      ││  |/                      │
│\_______________  ______/││              | |        ││ /-\ ((()            /-\  │
│                \|       ││              \-/        ││ | | ((()            | |  │
│  ╔══            /-\     ││          \   /|         ││ \-/ ((()            \-/  │
│  ║ |            | |     ││          |\ / |         ││  |---- )            /|\  │
│  ║-/       \    \-/     ││       ------  |         ││  |    (            / | \ │
│  /|\       |\   /|      ││ ________________|\____  ││  |         _______   |   │
│  || \   ------ - |*     ││/                       \││ / \       /elegant\ / \  │
│   |      |   |/--|*     ││ A few coders from each  ││/   \      |weapons|/   \ │
│  / \     |   ||****     ││ new generation will re- ││ _____|\___\       /______│
│ /   \    |   ||*  *     ││ discover the LISP arts. ││for a more...civilized age│
└─────────────────────────┘└─────────────────────────┘└──────────────────────────┘

 originally from: https://xkcd.com/297/  https://xkcd.com/license.html
 */


/// Goals
//
// * always stay small and hackable. At most 1,000 lines, total, ever.
// * s-expression based, Scheme-inspired embedded language for C++
// * constexpr evaluation of script possible
// * constexpr creation of interpreter possible
// * small memory footprint, fast
// * extreme type safety, no implicit conversions
// * all objects are immutable, copies are preferred over sharing
// * "impossible" to have memory errors because references don't naturally exist
// * allow the user to add pointer types if they choose to, for sharing of data
// between script and C++
// * C++23 as a minimum
// * never thread safe

namespace lefticus {
template<typename T>
concept not_bool_or_ptr = !std::same_as<std::remove_cvref_t<T>, bool> && !std::is_pointer_v<std::remove_cvref_t<T>>;

static constexpr auto plus_equal = [](auto &lhs, const auto &rhs) -> auto &
  requires not_bool_or_ptr<decltype(lhs)> && requires { lhs += rhs; }
{
  return lhs += rhs;
};

static constexpr auto multiply_equal = [](auto &lhs, const auto &rhs) -> auto &
  requires not_bool_or_ptr<decltype(lhs)> && requires { lhs *= rhs; }
{
  return lhs *= rhs;
};

static constexpr auto division_equal = [](auto &lhs, const auto &rhs) -> auto &
  requires not_bool_or_ptr<decltype(lhs)> && requires { lhs /= rhs; }
{
  return lhs /= rhs;
};

static constexpr auto minus_equal = [](auto &lhs, const auto &rhs) -> auto &
  requires not_bool_or_ptr<decltype(lhs)> && requires { lhs -= rhs; }
{
  return lhs -= rhs;
};

static constexpr auto less_than = [](const auto &lhs, const auto &rhs) -> bool
  requires not_bool_or_ptr<decltype(lhs)> && requires { lhs < rhs; }
{
  return lhs < rhs;
};
static constexpr auto greater_than = [](const auto &lhs, const auto &rhs) -> bool
  requires not_bool_or_ptr<decltype(lhs)> && requires { lhs > rhs; }
{
  return lhs > rhs;
};

static constexpr auto less_than_equal = [](const auto &lhs, const auto &rhs) -> bool
  requires not_bool_or_ptr<decltype(lhs)> && requires { lhs <= rhs; }
{
  return lhs <= rhs;
};

static constexpr auto greater_than_equal = [](const auto &lhs, const auto &rhs) -> bool
  requires not_bool_or_ptr<decltype(lhs)> && requires { lhs >= rhs; }
{
  return lhs >= rhs;
};

static constexpr auto equal = [](const auto &lhs, const auto &rhs) -> bool
  requires requires { lhs == rhs; }
{
  return lhs == rhs;
};

static constexpr auto not_equal = [](const auto &lhs, const auto &rhs) -> bool
  requires requires { lhs != rhs; }
{
  return lhs != rhs;
};

static constexpr auto logical_and = []<std::same_as<bool> Param>(Param lhs, Param rhs) -> bool { return lhs && rhs; };
static constexpr auto logical_or = []<std::same_as<bool> Param>(Param lhs, Param rhs) -> bool { return lhs || rhs; };

static constexpr bool logical_not(bool lhs) { return !lhs; }

struct Token
{
  std::string_view parsed;
  std::string_view remaining;
};

[[nodiscard]] constexpr std::pair<bool, int> parse_int(std::string_view input)
{
  int parsed = 0;
  auto result = std::from_chars(input.data(), input.data() + input.size(), parsed);
  if (result.ec == std::errc() && result.ptr == input.data() + input.size()) {
    return { true, parsed };
  } else {
    return { false, parsed };
  }
}

template<typename T> [[nodiscard]] constexpr std::pair<bool, T> parse_float(std::string_view input)
{
  if (input == "-") { return { false, 0 }; }

  enum class State { Start, IntegerPart, FractionPart, ExponentPart, ExponentStart, Fail };
  struct ParseState
  {
    State state = State::Start;
    T value_sign = 1;
    long long value = 0LL;
    long long frac = 0LL;
    long long frac_exp = 0LL;
    long long exp_sign = 1LL;
    long long exp = 0LL;

    [[nodiscard]] static constexpr auto pow(const T base, long long power) noexcept
    {
      auto result = decltype(base)(1);
      if (power > 0) {
        for (int iteration = 0; iteration < power; ++iteration) { result *= base; }
      } else if (power < 0) {
        for (int iteration = 0; iteration > power; --iteration) { result /= base; }
      }
      return result;
    };

    [[nodiscard]] constexpr auto next_state(State new_state) const noexcept
    {
      auto retval = *this;
      retval.state = new_state;
      return retval;
    }

    [[nodiscard]] constexpr auto float_value() const noexcept -> std::pair<bool, T>
    {
      if (state == State::Fail || state == State::Start || state == State::ExponentStart) { return { false, 0 }; }

      return { true,
        (static_cast<T>(value_sign) * (static_cast<T>(value) + static_cast<T>(frac) * pow(static_cast<T>(10), frac_exp))
          * pow(static_cast<T>(10), exp_sign * exp)) };
    }
  };

  ParseState current_state;

  auto parse_digit = [](auto &value, char ch) {
    if (ch >= '0' && ch <= '9') {
      value = value * 10 + ch - '0';
      return true;
    } else {
      return false;
    }
  };

  auto parse_char = [parse_digit](ParseState state, char c) -> ParseState {
    switch (state.state) {
    case State::Start:
      if (c == '-') {
        state.value_sign = -1;
      } else if (!parse_digit(state.value, c)) {
        return state.next_state(State::Fail);
      }
      return state.next_state(State::IntegerPart);
    case State::IntegerPart:
      if (parse_digit(state.value, c)) {
        return state;
      } else if (c == '.') {
        return state.next_state(State::FractionPart);
      } else if (c == 'e' || c == 'E') {
        return state.next_state(State::ExponentPart);
      }
      return state.next_state(State::Fail);
    case State::FractionPart:
      if (parse_digit(state.frac, c)) {
        state.frac_exp--;
        return state;
      } else if (c == 'e' || c == 'E') {
        return state.next_state(State::ExponentStart);
      }
      return state.next_state(State::Fail);
    case State::ExponentStart:
      if (c == '-') {
        state.exp_sign = -1;
      } else if (!parse_digit(state.exp, c)) {
        return state.next_state(State::Fail);
      }
      return state.next_state(State::ExponentPart);
    case State::ExponentPart:
      if (parse_digit(state.exp, c)) { return state; }
      return state.next_state(State::Fail);
    case State::Fail:
      return state;
    }

    return state.next_state(State::Fail);
  };

  for (const char current_c : input) { current_state = parse_char(current_state, current_c); }

  return current_state.float_value();
}


[[nodiscard]] constexpr Token next_token(std::string_view input)
{
  constexpr auto is_whitespace = [](auto character) {
    return character == ' ' || character == '\t' || character == '\n';
  };

  constexpr auto consume = [=](auto ws_input, auto predicate) {
    auto begin = ws_input.begin();
    while (begin != ws_input.end() && predicate(*begin)) { ++begin; }
    return std::string_view{ begin, ws_input.end() };
  };

  constexpr auto make_token = [=](auto token_input, std::size_t size) {
    return Token{ token_input.substr(0, size), consume(token_input.substr(size), is_whitespace) };
  };

  input = consume(input, is_whitespace);

  if (input.starts_with('(') || input.starts_with(')')) { return make_token(input, 1); }

  if (input.starts_with("'(")) { return make_token(input, 2); }

  if (input.starts_with('"')) {
    bool in_escape = false;
    auto location = std::next(input.begin());
    while (location != input.end()) {
      if (*location == '\\') {
        in_escape = true;
      } else if (*location == '"' && !in_escape) {
        ++location;
        break;
      } else {
        in_escape = false;
      }
      ++location;
    }

    return make_token(input, static_cast<std::size_t>(std::distance(input.begin(), location)));
  }

  const auto value =
    consume(input, [=](char character) { return !is_whitespace(character) && character != ')' && character != '('; });
  return make_token(input, static_cast<std::size_t>(std::distance(input.begin(), value.begin())));
}

struct IndexedString
{
  std::size_t start;
  std::size_t length;
  [[nodiscard]] constexpr auto size() const { return length; }
  [[nodiscard]] constexpr bool operator==(const IndexedString &) const noexcept = default;
};

struct IndexedList
{
  std::size_t start;
  std::size_t length;
  [[nodiscard]] constexpr auto size() const { return length; }
};
struct LiteralList
{
  IndexedList items;
};

struct Identifier
{
  enum struct Map { builtin, global, local };
  struct Location
  {
    Map map;
    std::size_t index;
  };
  IndexedString value;
  mutable std::optional<Location> found;
};

template<std::size_t BuiltInSymbolsSize = 32,
  std::size_t BuiltInStringsSize = 255,
  std::size_t BuiltInValuesSize = 0,
  typename... UserTypes>
struct cons_expr
{
  struct SExpr;


  struct Context
  {
    std::vector<std::pair<IndexedString, SExpr>> objects;
  };

  using function_ptr = SExpr (*)(cons_expr &, Context &, std::span<const SExpr>);

  [[nodiscard]] static constexpr SExpr for_each(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    if (params.size() != 2) { throw std::runtime_error("Wrong number of parameters to for-each expression"); }

    const auto &list = engine.to_list(engine.eval_to<LiteralList>(context, params[1]).items);

    for (auto itr = list.begin(); itr != list.end(); ++itr) {
      [[maybe_unused]] const auto result =
        engine.invoke_function(context, params[0], std::span<const SExpr>{ itr, std::next(itr) });
    }

    return SExpr{ Atom{ std::monostate{} } };
  }

  using Atom = std::variant<std::monostate, bool, int, double, IndexedString, Identifier, UserTypes...>;

  struct Lambda;

  struct SExpr
  {
    std::variant<Atom, IndexedList, LiteralList, Lambda, function_ptr> value;
    constexpr std::span<const SExpr> to_list(const cons_expr &engine) const
    {
      if (const auto *list = std::get_if<IndexedList>(&value); list != nullptr) { return engine.to_list(*list); }

      throw std::runtime_error("SExpr is not a list");
    }
  };

  std::array<std::pair<IndexedString, SExpr>, BuiltInSymbolsSize> builtin_symbols{};
  std::vector<std::pair<IndexedString, SExpr>> symbols;
  std::size_t builtin_symbols_size = 0;
  std::array<char, BuiltInStringsSize> builtin_strings{};
  std::size_t builtin_strings_size = 0;
  std::vector<char> strings;
  std::array<SExpr, BuiltInValuesSize> builtin_values{};
  std::size_t builtin_values_size = 0;
  std::vector<SExpr> values;


  struct Lambda
  {
    IndexedList parameter_names;
    IndexedList captured_names;
    IndexedList captured_values;
    IndexedList statements;

    [[nodiscard]] constexpr SExpr invoke(cons_expr &engine, Context &context, std::span<const SExpr> parameters)
    {
      if (parameters.size() != parameter_names.size()) {
        throw std::runtime_error("Incorrect number of parameters for lambda");
      }

      const auto captured_names_list = engine.to_list(captured_names);
      const auto captured_values_list = engine.to_list(captured_values);

      // restore context
      Context new_context;
      for (std::size_t index = 0; index < captured_names_list.size(); ++index) {
        new_context.objects.emplace_back(
          std::get<Identifier>(std::get<Atom>(captured_names_list[index].value)).value, captured_values_list[index]);
      }

      const auto parameter_names_list = engine.to_list(parameter_names);

      // set up params
      // technically I'm evaluating the params lazily while invoking the lambda, not before
      // does it matter?
      for (std::size_t index = 0; index < parameter_names_list.size(); ++index) {
        new_context.objects.emplace_back(std::get<Identifier>(std::get<Atom>(parameter_names_list[index].value)).value,
          engine.eval(context, parameters[index]));
      }


      return engine.sequence(new_context, engine.to_list(statements));
    }
  };

  [[nodiscard]] constexpr std::span<const SExpr> to_list(IndexedList indexedlist) const
  {
    if (indexedlist.start >= BuiltInValuesSize) {
      return std::span<const SExpr>(values).subspan(indexedlist.start, indexedlist.length);
    } else {
      return std::span<const SExpr>(builtin_values).subspan(indexedlist.start, indexedlist.length);
    }
  }

  [[nodiscard]] constexpr std::string_view to_string_view(IndexedString indexedstring) const
  {
    if (indexedstring.start >= BuiltInValuesSize) {
      return std::string_view(strings).substr(indexedstring.start, indexedstring.length);
    } else {
      return std::string_view(builtin_strings).substr(indexedstring.start, indexedstring.length);
    }
  }



  [[nodiscard]] constexpr IndexedString add_string(std::string_view value)
  {
    if (const auto builtin_location =
          std::search(builtin_strings.begin(), builtin_strings.end(), value.begin(), value.end());
        builtin_location != builtin_strings.end()) {
      return IndexedString{ static_cast<std::size_t>(std::distance(builtin_strings.begin(), builtin_location)),
        value.size() };
    } else if (const auto location = std::search(strings.begin(), strings.end(), value.begin(), value.end());
               location != strings.end()) {
      return IndexedString{ static_cast<std::size_t>(std::distance(strings.begin(), location)) + BuiltInStringsSize,
        value.size() };
    } else {
      strings.insert(strings.end(), value.begin(), value.end());
      return IndexedString{ strings.size() - value.size() + BuiltInStringsSize, value.size() };
    }
  };

  [[nodiscard]] constexpr IndexedList add_list(std::vector<SExpr> items)
  {
    const auto start = values.size();
    values.insert(values.end(), std::make_move_iterator(items.begin()), std::make_move_iterator(items.end()));

    return IndexedList{ start + BuiltInValuesSize, values.size() - start };
  };

  [[nodiscard]] constexpr std::pair<SExpr, Token> parse(std::string_view input)
  {
    std::vector<SExpr> retval;

    auto token = next_token(input);

    while (!token.parsed.empty()) {
      if (token.parsed == "(") {
        auto [parsed, remaining] = parse(token.remaining);
        retval.push_back(parsed);
        token = remaining;
      } else if (token.parsed == "'(") {
        auto [parsed, remaining] = parse(token.remaining);
        retval.push_back(SExpr{ LiteralList{ std::get<IndexedList>(parsed.value) } });
        token = remaining;
      } else if (token.parsed == ")") {
        break;
      } else if (token.parsed == "true") {
        retval.push_back(SExpr{ Atom{ true } });
      } else if (token.parsed == "false") {
        retval.push_back(SExpr{ Atom{ false } });
      } else {
        if (token.parsed.starts_with('"')) {
          // quoted string
          if (!token.parsed.ends_with('"')) { throw std::runtime_error("Unterminated string"); }
          // note that this doesn't remove escaped characters like it should yet
          retval.push_back(SExpr{ Atom(add_string(token.parsed.substr(1, token.parsed.size() - 2))) });
        } else if (auto [int_did_parse, int_value] = parse_int(token.parsed); int_did_parse) {
          retval.push_back(SExpr{ Atom(int_value) });
        } else if (auto [float_did_parse, float_value] = parse_float<double>(token.parsed); float_did_parse) {
          retval.push_back(SExpr{ Atom(float_value) });
        } else {
          retval.push_back(SExpr{ Atom(Identifier{ add_string(token.parsed), {} }) });
        }
      }
      token = next_token(token.remaining);
    }
    return std::pair<SExpr, Token>(SExpr{ add_list(retval) }, token);
  }

  consteval IndexedString add_builtin(std::string_view value)
  {
    const auto end = std::next(builtin_strings.begin(), static_cast<std::ptrdiff_t>(builtin_strings_size));
    const auto location = std::search(builtin_strings.begin(),
      end,
      value.begin(),
      value.end());
    if (location != end) {
      return IndexedString{ static_cast<std::size_t>(std::distance(builtin_strings.begin(), location)), value.size() };
    } else {
      std::copy(
        value.begin(), value.end(), std::next(builtin_strings.begin(), static_cast<std::ptrdiff_t>(builtin_strings_size)));
      const auto start = builtin_strings_size;
      builtin_strings_size += value.size();
      return IndexedString{ start, value.size() };
    }
  }

  consteval void add_builtin(std::string_view name, SExpr value)
  {
    builtin_symbols[builtin_symbols_size] = std::pair{ add_builtin(name), value };
    ++builtin_symbols_size;
  }

  consteval cons_expr()
  {
    add_builtin("+", SExpr{ binary_left_fold<plus_equal> });
    add_builtin("*", SExpr{ binary_left_fold<multiply_equal> });
    add_builtin("-", SExpr{ binary_left_fold<minus_equal> });
    add_builtin("/", SExpr{ binary_left_fold<division_equal> });
    add_builtin("<", SExpr{ binary_boolean_apply_pairwise<less_than> });
    add_builtin(">", SExpr{ binary_boolean_apply_pairwise<greater_than> });
    add_builtin("<=", SExpr{ binary_boolean_apply_pairwise<less_than_equal> });
    add_builtin(">=", SExpr{ binary_boolean_apply_pairwise<greater_than_equal> });
    add_builtin("and", SExpr{ binary_boolean_apply_pairwise<logical_and> });
    add_builtin("or", SExpr{ binary_boolean_apply_pairwise<logical_or> });
    add_builtin("if", SExpr{ ifer });
    add_builtin("not", SExpr{ make_evaluator<logical_not>() });
    add_builtin("==", SExpr{ binary_boolean_apply_pairwise<equal> });
    add_builtin("!=", SExpr{ binary_boolean_apply_pairwise<not_equal> });
    add_builtin("for-each", SExpr{ for_each });
    add_builtin("list", SExpr{ list });
    add_builtin("lambda", SExpr{ lambda });
    add_builtin("do", SExpr{ doer });
  }

  [[nodiscard]] constexpr SExpr sequence(Context &context, std::span<const SExpr> statements)
  {
    if (!statements.empty()) {
      for (const auto &statement : statements.subspan(0, statements.size() - 1)) {
        [[maybe_unused]] const auto result = eval(context, statement);
      }
      return eval(context, statements.back());
    } else {
      return SExpr{ Atom{ std::monostate{} } };
    }
  }

  [[nodiscard]] constexpr SExpr
    invoke_function(Context &context, const SExpr &function, std::span<const SExpr> parameters)
  {
    SExpr resolved_function = eval(context, function);

    if (auto *lambda = std::get_if<Lambda>(&resolved_function.value); lambda != nullptr) {
      return lambda->invoke(*this, context, parameters);
    } else {
      return get_function(resolved_function)(*this, context, parameters);
    }
  }

  [[nodiscard]] constexpr function_ptr get_function(const SExpr &expr)
  {
    if (const auto *func = std::get_if<function_ptr>(&expr.value); func != nullptr) { return *func; }
    throw std::runtime_error("Does not evaluate to a function");
  }

  template<auto Func, typename Ret, typename... Param>
  [[nodiscard]] constexpr static function_ptr make_evaluator(Ret (*)(Param...))
  {
    return function_ptr{ [](cons_expr &engine, Context &context, std::span<const SExpr> params) -> SExpr {
      if (params.size() != sizeof...(Param)) { throw std::runtime_error("wrong param count"); }

      auto impl = [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        if constexpr (std::is_same_v<void, Ret>) {
          Func(engine.eval_to<std::remove_cvref_t<Param>>(context, params[Idx])...);
          return SExpr{ Atom{ std::monostate{} } };
        } else {
          return SExpr{ Func(engine.eval_to<Param>(context, params[Idx])...) };
        }
      };

      return impl(std::make_index_sequence<sizeof...(Param)>{});
    } };
  }

  template<auto Func> [[nodiscard]] constexpr static function_ptr make_evaluator()
  {
    return make_evaluator<Func>(Func);
  }

  template<auto Func> constexpr void add(std::string_view name)
  {
    symbols.emplace_back(add_string(name), SExpr{ make_evaluator<Func>(Func) });
  }

  template<typename Value> constexpr void add(std::string_view name, Value &&value)
  {
    symbols.emplace_back(add_string(name), SExpr{ Atom{ std::forward<Value>(value) } });
  }


  [[nodiscard]] constexpr SExpr eval(Context &context, const SExpr &expr)
  {
    if (const auto *indexedlist = std::get_if<IndexedList>(&expr.value); indexedlist != nullptr) {
      // if it's a non-empty list, then I need to evaluate it as a function
      auto list = to_list(*indexedlist);
      if (!list.empty()) { return invoke_function(context, list[0], { std::next(list.begin()), list.end() }); }
    } else if (const auto *atom = std::get_if<Atom>(&expr.value); atom != nullptr) {
      // if it's an identifier, we need to be able to find it
      if (const auto *id = std::get_if<Identifier>(atom); id != nullptr) {
        if (id->found.has_value()) {
          switch (const auto &[map, index] = *id->found; map) {
          case Identifier::Map::global:
            return symbols[index].second;
          case Identifier::Map::local:
            return context.objects[index].second;
          case Identifier::Map::builtin:
            return builtin_symbols[index].second;
          }
        }

        for (std::size_t index = 0; const auto &object : context.objects) {
          if (object.first == id->value) {
            id->found = typename Identifier::Location{ Identifier::Map::local, index };
            return object.second;
          }
          ++index;
        }

        for (std::size_t index = 0; const auto &object : symbols) {
          if (object.first == id->value) {
            id->found = typename Identifier::Location{ Identifier::Map::global, index };
            return object.second;
          }
          ++index;
        }

        for (std::size_t index = 0; const auto &object : builtin_symbols) {
          if (object.first == id->value) {
            id->found = typename Identifier::Location{ Identifier::Map::builtin, index };
            return object.second;
          }
          ++index;
        }

        throw std::runtime_error("id not found");
      }
    }
    return expr;
  }

  template<typename Type> [[nodiscard]] constexpr Type eval_to(Context &context, const SExpr &expr)
  {
    if constexpr (std::is_same_v<Type, LiteralList> || std::is_same_v<Type, IndexedList> || std::is_same_v<Type, Lambda>
                  || std::is_same_v<Type, function_ptr>) {
      if (const auto *obj = std::get_if<Type>(&expr.value); obj != nullptr) { return *obj; }
    } else {
      if (const auto *atom = std::get_if<Atom>(&expr.value); atom != nullptr) {
        if (const auto *value = std::get_if<Type>(atom); value != nullptr) {
          return *value;
        } else if (!std::holds_alternative<Identifier>(*atom)) {
          throw std::runtime_error("wrong type");
        }
      }
    }
    return eval_to<Type>(context, eval(context, expr));
  }

  [[nodiscard]] static constexpr SExpr list(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    std::vector<SExpr> result;

    for (const auto &param : params) { result.push_back(engine.eval(context, param)); }

    return SExpr{ LiteralList{ engine.add_list(result) } };
  }

  [[nodiscard]] static constexpr SExpr lambda(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    if (params.size() < 2) { throw std::runtime_error("Wrong number of parameters to lambda expression"); }

    std::vector<SExpr> context_ids;
    std::vector<SExpr> context_values;
    for (const auto &object : context.objects) {
      context_ids.push_back(SExpr{ Identifier{ object.first, {} } });
      context_values.push_back(object.second);
    }

    return SExpr{ Lambda{ std::get<IndexedList>(params[0].value),
      engine.add_list(context_ids),
      engine.add_list(context_values),
      { engine.add_list({ std::next(params.begin()), params.end() }) } } };
  }

  [[nodiscard]] static constexpr SExpr doer(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    if (params.size() < 2) { throw std::runtime_error("Wrong number of parameters to do expression"); }

    std::vector<std::pair<std::size_t, SExpr>> variables;

    const auto setup_variable = [&](const auto &expr) {
      auto elements = expr.to_list(engine);
      if (elements.size() != 3) { throw std::runtime_error(""); }

      const auto index = context.objects.size();
      context.objects.emplace_back(
        engine.eval_to<Identifier>(context, elements[0]).value, engine.eval(context, elements[1]));
      variables.emplace_back(index, elements[2]);
    };

    const auto setup_variables = [&](const auto &expr) {
      auto elements = expr.to_list(engine);
      for (const auto &variable : elements) { setup_variable(variable); }
    };

    setup_variables(params[0]);

    const auto terminators = params[1].to_list(engine);

    std::vector<std::pair<std::size_t, SExpr>> new_values;

    // continue while terminator test is false
    while (!engine.eval_to<bool>(context, terminators[0])) {
      // evaluate body
      [[maybe_unused]] const auto result = engine.sequence(context, params.subspan(2));

      // iterate loop variables
      for (const auto &[index, expr] : variables) { new_values.emplace_back(index, engine.eval(context, expr)); }

      // update values
      for (auto &&[index, value] : new_values) { context.objects[index].second = std::move(value); }

      new_values.clear();
    }

    // evaluate sequence of termination expressions
    return engine.sequence(context, terminators.subspan(1));
  }

  [[nodiscard]] static constexpr SExpr ifer(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    if (params.size() != 3) { throw std::runtime_error("Wrong number of parameters to if expression"); }

    if (engine.eval_to<bool>(context, params[0])) {
      return engine.eval(context, params[1]);
    } else {
      return engine.eval(context, params[2]);
    }
  }

  template<typename Signature>
  [[nodiscard]] auto make_callable(auto function)
    requires std::is_function_v<Signature>
  {
    auto impl = [this, function]<typename Ret, typename... Params>(Ret (*)(Params...)) {
      // this is fragile, we need to check parsing better
      Context temp_ctx;

      return [callable = eval(temp_ctx, to_list(std::get<IndexedList>(parse(function).first.value))[0]), this](
               Params... params) {
        Context ctx;
        std::array<SExpr, sizeof...(Params)> args{ SExpr{ Atom{ params } }... };
        return eval_to<Ret>(ctx, invoke_function(ctx, callable, args));
      };
    };

    return impl(std::add_pointer_t<Signature>{ nullptr });
  }

  template<auto Op>
  [[nodiscard]] static constexpr SExpr
    binary_left_fold(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    auto sum = [&engine, &context, params]<typename Param>(Param first) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        for (const auto &next : params.subspan(1)) { Op(first, engine.eval_to<Param>(context, next)); }

        return SExpr{ Atom{ first } };
      } else {
        throw std::runtime_error("Operator not supported for types");
      }
    };

    if (params.size() > 1) { return std::visit(sum, std::get<Atom>(engine.eval(context, params[0]).value)); }

    throw std::runtime_error("Not enough params");
  }

  template<auto Op>
  [[nodiscard]] static constexpr SExpr
    binary_boolean_apply_pairwise(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    auto sum = [&engine, &context, params]<typename Param>(Param first) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        auto second = engine.eval_to<Param>(context, params[1]);
        bool result = Op(first, second);
        bool odd = true;
        for (const auto &next : params.subspan(2)) {
          if (odd) {
            first = engine.eval_to<Param>(context, next);
            result = result && Op(second, first);
          } else {
            second = engine.eval_to<Param>(context, next);
            result = result && Op(first, second);
          }
          odd = !odd;
        }

        return SExpr{ Atom{ result } };
      } else {
        throw std::runtime_error("Operator not supported for types");
      }
    };

    if (params.size() > 1) { return std::visit(sum, std::get<Atom>(engine.eval(context, params[0]).value)); }

    throw std::runtime_error("Not enough params");
  }
};


}// namespace lefticus


/// TODO
// * add the ability to define / let things
// * replace function identifiers with function pointers while parsing
// * add cons car cdr eval apply
// * sort out copying on return of objects when possible
// * convert constexpr `defined` objects into static string views and static spans
// * remove exceptions I guess?
// * fix short circuiting

#endif
