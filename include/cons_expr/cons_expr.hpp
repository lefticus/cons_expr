#ifndef CONS_EXPR_HPP
#define CONS_EXPR_HPP

#include <fmt/format.h>

#include <charconv>
#include <iostream>
#include <span>
#include <stacktrace>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>


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
 */


namespace lefticus {
template<typename T>
concept not_bool_or_ptr = !std::same_as<std::remove_cvref_t<T>, bool> && !std::is_pointer_v<std::remove_cvref_t<T>>;

static constexpr auto plus_equal = [](auto &lhs, const auto &rhs) -> auto &
  requires requires {
    requires not_bool_or_ptr<decltype(lhs)>;
    lhs += rhs;
  }
{
  return lhs += rhs;
};

static constexpr auto multiply_equal = [](auto &lhs, const auto &rhs) -> auto &
  requires requires {
    requires not_bool_or_ptr<decltype(lhs)>;
    lhs *= rhs;
  }
{
  return lhs *= rhs;
};

static constexpr auto division_equal = [](auto &lhs, const auto &rhs) -> auto &
  requires requires {
    requires not_bool_or_ptr<decltype(lhs)>;
    lhs /= rhs;
  }
{
  return lhs /= rhs;
};

static constexpr auto minus_equal = [](auto &lhs, const auto &rhs) -> auto &
  requires requires {
    requires not_bool_or_ptr<decltype(lhs)>;
    lhs -= rhs;
  }
{
  return lhs -= rhs;
};

static constexpr auto less_than = [](const auto &lhs, const auto &rhs) -> auto
  requires requires {
    requires not_bool_or_ptr<decltype(lhs)>;
    lhs < rhs;
  }
{
  return lhs < rhs;
};
static constexpr auto greater_than = [](const auto &lhs, const auto &rhs) -> auto
  requires requires {
    requires not_bool_or_ptr<decltype(lhs)>;
    lhs > rhs;
  }
{
  return lhs > rhs;
};

static constexpr auto less_than_equal = [](const auto &lhs, const auto &rhs) -> auto
  requires requires {
    requires not_bool_or_ptr<decltype(lhs)>;
    lhs <= rhs;
  }
{
  return lhs <= rhs;
};
static constexpr auto greater_than_equal = [](const auto &lhs, const auto &rhs) -> auto
  requires requires {
    requires not_bool_or_ptr<decltype(lhs)>;
    lhs >= rhs;
  }
{
  return lhs >= rhs;
};

static constexpr auto equal = [](const auto &lhs, const auto &rhs) -> auto
  requires requires { lhs == rhs; }
{
  return lhs == rhs;
};

static constexpr auto not_equal = [](const auto &lhs, const auto &rhs) -> auto
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

inline std::pair<bool, int> parse_int(std::string_view input)
{
  int parsed = 0;
  auto result = std::from_chars(input.data(), input.data() + input.size(), parsed);
  if (result.ec == std::errc() && result.ptr == input.data() + input.size()) {
    return { true, parsed };
  } else {
    return { false, parsed };
  }
}

template<typename T> constexpr std::pair<bool, T> parse_float(std::string_view input)
{
  auto integer = 0LL;
  T sign = 1;
  auto fraction = 0LL;
  auto exponent = 0LL;
  auto fraction_exponent = 0LL;
  auto exp_sign = 1LL;
  enum class State { Start, IntegerPart, FractionPart, ExponentPart } state = State::Start;

  for (const char c : input) {
    switch (state) {
    case State::Start:
      if (c == '-') {
        sign = -1;
      } else if (c >= '0' && c <= '9') {
        integer = c - '0';
      } else {
        return { false, 0 };
      }
      state = State::IntegerPart;
      break;
    case State::IntegerPart:
      if (c >= '0' && c <= '9') {
        integer = integer * 10 + (c - '0');
      } else if (c == '.') {
        state = State::FractionPart;
      } else if (c == 'e' || c == 'E') {
        state = State::ExponentPart;
      } else {
        return { false, 0 };
      }
      break;
    case State::FractionPart:
      if (c >= '0' && c <= '9') {
        fraction = fraction * 10 + (c - '0');
        fraction_exponent--;
      } else if (c == 'e' || c == 'E') {
        state = State::ExponentPart;
      } else {
        return { false, 0 };
      }
      break;
    case State::ExponentPart:
      if (c == '-') {
        exp_sign = -1;
      } else if (c >= '0' && c <= '9') {
        exponent = exponent * 10 + (c - '0') * exp_sign;
      } else {
        return { false, 0 };
      }
      break;
    }
  }

  auto pow = [](const auto base, auto power) {
    auto result  = decltype(base)(1);
      if (power > 0) {
        for (int iteration = 0; iteration < power; ++iteration) { result *= base; }
      } else if (power < 1) {
        for (int iteration = 0; iteration > power; --iteration) { result /= base; }
      }
    return result;
};

  return { true,
    (static_cast<T>(sign) * (static_cast<T>(integer)
                          + static_cast<T>(fraction) * pow(static_cast<T>(10), fraction_exponent))
     * pow(static_cast<T>(10), exponent)) };
}


constexpr Token next_token(std::string_view input)
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

template<typename... UserTypes> struct cons_expr
{
  struct SExpr;

  struct Identifier
  {
    std::string_view value;
  };

  struct Context
  {
    std::vector<std::pair<std::string_view, SExpr>> objects;
  };

  using function_ptr = SExpr (*)(cons_expr &, Context &, std::span<const SExpr>);

  static constexpr SExpr for_each(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    if (params.size() != 2) { throw std::runtime_error("Wrong number of parameters to for-each expression"); }

    const auto &list = engine.eval_to<LiteralList>(context, params[1]);

    for (auto itr = list.items.begin(); itr != list.items.end(); ++itr) {
      engine.invoke_function(context, params[0], std::span<const SExpr>{ itr, std::next(itr) });
    }

    return SExpr{ Atom{ std::monostate{} } };
  }

  using Atom = std::variant<std::monostate, bool, int, double, Identifier, std::string, UserTypes...>;
  using List = std::vector<SExpr>;

  std::array<std::pair<std::string_view, SExpr>, 20> built_ins;
  std::vector<std::pair<std::string, SExpr>> symbols;

  struct LiteralList
  {
    std::vector<SExpr> items;
  };
  struct Lambda
  {
    Context captured_context;
    std::vector<std::string_view> parameter_names;
    std::vector<SExpr> statements;
  };

  struct SExpr
  {
    std::variant<Atom, List, LiteralList, Lambda, function_ptr> value;
  };

  static constexpr std::pair<SExpr, Token> parse(std::string_view input)
  {
    List retval;

    auto token = next_token(input);

    while (!token.parsed.empty()) {
      if (token.parsed == "(") {
        auto [parsed, remaining] = parse(token.remaining);
        retval.push_back(parsed);
        token = remaining;
      } else if (token.parsed == "'(") {
        auto [parsed, remaining] = parse(token.remaining);
        retval.push_back(SExpr{ LiteralList{ std::get<List>(parsed.value) } });
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
          retval.push_back(SExpr{ Atom(std::string(token.parsed.substr(1, token.parsed.size() - 2))) });
        } else if (auto [int_did_parse, int_value] = parse_int(token.parsed); int_did_parse) {
          retval.push_back(SExpr{ Atom(int_value) });
        } else if (auto [float_did_parse, float_value] = parse_float<double>(token.parsed); float_did_parse) {
          retval.push_back(SExpr{ Atom(float_value) });
        } else {
          // to-do, parse float
          // for now just assume identifier
          retval.push_back(SExpr{ Atom(Identifier{ token.parsed }) });
        }
      }
      token = next_token(token.remaining);
    }
    return std::pair<SExpr, Token>(SExpr{ retval }, token);
  }

  static constexpr auto make_built_ins()
  {
    decltype(built_ins) retval;
    retval[0] = { "+", SExpr{ binary_left_fold<plus_equal> } };
    retval[1] = { "*", SExpr{ binary_left_fold<multiply_equal> } };
    retval[2] = { "-", SExpr{ binary_left_fold<minus_equal> } };
    retval[3] = { "/", SExpr{ binary_left_fold<division_equal> } };
    retval[4] = { "<", SExpr{ binary_boolean_apply_pairwise<less_than> } };
    retval[5] = { ">", SExpr{ binary_boolean_apply_pairwise<greater_than> } };
    retval[6] = { "<=", SExpr{ binary_boolean_apply_pairwise<less_than_equal> } };
    retval[7] = { ">=", SExpr{ binary_boolean_apply_pairwise<greater_than_equal> } };
    retval[8] = { "and", SExpr{ binary_boolean_apply_pairwise<logical_and> } };
    retval[9] = { "or", SExpr{ binary_boolean_apply_pairwise<logical_or> } };
    retval[11] = { "if", SExpr{ ifer } };
    retval[12] = { "not", SExpr{ make_evaluator<logical_not>() } };
    retval[13] = { "==", SExpr{ binary_boolean_apply_pairwise<equal> } };
    retval[14] = { "!=", SExpr{ binary_boolean_apply_pairwise<not_equal> } };
    retval[15] = { "for-each", SExpr{ for_each } };
    retval[16] = { "list", SExpr{ list } };
    retval[17] = { "lambda", SExpr{ lambda } };
    return retval;
  }

  // should be `consteval` capable, but not in GCC 12.2 yet
  cons_expr() : built_ins(make_built_ins()) {}

  constexpr SExpr sequence(Context &context, std::span<const SExpr> statements)
  {
    if (!statements.empty()) {
      for (const auto &statement : statements.subspan(0, statements.size() - 1)) { eval(context, statement); }
      return eval(context, statements.back());
    } else {
      return SExpr{ Atom{ std::monostate{} } };
    }
  }

  constexpr SExpr invoke_function(Context &context, const SExpr &function, std::span<const SExpr> parameters)
  {
    SExpr resolved_function = eval(context, function);

    if (auto *lambda = std::get_if<Lambda>(&resolved_function.value); lambda != nullptr) {
      if (parameters.size() != lambda->parameter_names.size()) {
        throw std::runtime_error("Incorrect number of parameters for lambda");
      }
      Context new_context = lambda->captured_context;
      for (std::size_t index = 0; index < parameters.size(); ++index) {
        new_context.objects.emplace_back(lambda->parameter_names[index], eval(context, parameters[index]));
      }
      return sequence(new_context, lambda->statements);
    } else {
      return get_function(resolved_function)(*this, context, parameters);
    }
  }

  constexpr function_ptr get_function(const SExpr &expr)
  {
    if (const auto *func = std::get_if<function_ptr>(&expr.value); func != nullptr) { return *func; }
    throw std::runtime_error("Does not evaluate to a function");
  }

  template<auto Func, typename Ret, typename... Param> constexpr static function_ptr make_evaluator(Ret (*)(Param...))
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

  template<auto Func> constexpr static function_ptr make_evaluator() { return make_evaluator<Func>(Func); }

  template<auto Func> constexpr void add(std::string_view name)
  {
    symbols.emplace_back(std::string(name), Atom{ make_evaluator<Func>(Func) });
  }

  template<typename Value> constexpr void add(std::string_view name, Value &&value)
  {
    symbols.emplace_back(std::string{ name }, Atom{ std::forward<Value>(value) });
  }

  constexpr SExpr eval(Context &context, const SExpr &expr)
  {
    if (const auto *list = std::get_if<List>(&expr.value); list != nullptr) {
      // if it's a non-empty list, then I need to evaluate it as a function
      if (!list->empty()) { return invoke_function(context, (*list)[0], { std::next(list->begin()), list->end() }); }
    } else if (const auto *atom = std::get_if<Atom>(&expr.value); atom != nullptr) {
      // if it's an identifier, we need to be able to find it
      if (const auto *id = std::get_if<Identifier>(atom); id != nullptr) {
        for (const auto &object : context.objects) {
          if (object.first == id->value) { return object.second; }
        }

        for (const auto &object : symbols) {
          if (object.first == id->value) { return object.second; }
        }

        for (const auto &object : built_ins) {
          if (object.first == id->value) { return object.second; }
        }

        throw std::runtime_error("id not found");
      }
    }
    return expr;
  }

  template<typename Type> constexpr Type eval_to(Context &context, const SExpr &expr)
  {
    if constexpr (std::is_same_v<Type, LiteralList> || std::is_same_v<Type, List> || std::is_same_v<Type, Lambda>
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

  static constexpr SExpr list(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    LiteralList result;

    for (const auto &param : params) { result.items.push_back(engine.eval(context, param)); }

    return SExpr{ result };
  }

  static constexpr SExpr lambda(cons_expr &, Context &context, std::span<const SExpr> params)
  {
    if (params.size() < 2) { throw std::runtime_error("Wrong number of parameters to lambda expression"); }

    std::vector<std::string_view> parameter_names;

    for (const auto &name : std::get<List>(params[0].value)) {
      parameter_names.push_back(std::get<Identifier>(std::get<Atom>(name.value)).value);
    }

    return SExpr{ Lambda{ context, parameter_names, { std::next(params.begin()), params.end() } } };
  }

  static constexpr SExpr ifer(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    if (params.size() != 3) { throw std::runtime_error("Wrong number of parameters to if expression"); }

    if (engine.eval_to<bool>(context, params[0])) {
      return engine.eval(context, params[1]);
    } else {
      return engine.eval(context, params[2]);
    }
  }

  template<typename Signature>
  auto make_callable(auto function)
    requires std::is_function_v<Signature>
  {
    auto impl = [this, function]<typename Ret, typename... Params>(Ret (*)(Params...)) {
      // this is fragile, we need to check parsing better
      Context temp_ctx;

      return [callable = eval(temp_ctx, std::get<List>(parse(function).first.value)[0]), this](Params... params) {
        Context ctx;
        std::array<SExpr, sizeof...(Params)> args{ SExpr{ Atom{ params } }... };
        return eval_to<Ret>(ctx, invoke_function(ctx, callable, args));
      };
    };

    return impl(std::add_pointer_t<Signature>{ nullptr });
  }

  template<auto Op>
  static constexpr SExpr binary_left_fold(cons_expr &engine, Context &context, std::span<const SExpr> params)
  {
    auto sum = [&engine, &context, params]<typename Param>(Param first) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        for (const auto &next : params.subspan(1)) { Op(first, engine.eval_to<Param>(context, next)); }

        return SExpr{ Atom{ first } };
      } else {
        throw std::runtime_error("Operator not supported for types");
      }
    };

    if (!params.empty()) { return std::visit(sum, std::get<Atom>(engine.eval(context, params[0]).value)); }

    throw std::runtime_error("Not enough params");
  }

  template<auto Op>
  static constexpr SExpr
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

        return SExpr{ result };
      } else {
        throw std::runtime_error("Operator not supported for types");
      }
    };

    if (!params.empty()) { return std::visit(sum, std::get<Atom>(engine.eval(context, params[0]).value)); }

    throw std::runtime_error("Not enough params");
  }
};
}// namespace lefticus

/// Goals
// https://en.wikipedia.org/wiki/Greenspun%27s_tenth_rule
//
// Any sufficiently complicated C or Fortran program contains an ad hoc,
// informally-specified, bug-ridden, slow implementation of half of Common Lisp.
//
// * always stay small and hackable. At most 1,000 lines, total, ever.
// Preferably
//   more like 500 lines
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

/// TODO
// * add the ability to define / let things
// * replace function identifiers with function pointers while parsing
// * "compile" identifiers to be exact indexes into appropriate maps
// * add cons car cdr
// * parse / test doubles
// * reenable constexpr stuff
// * sort out copying on return of objects when possible
// * convert constexpr `defined` objects into static string views and static spans
// * remove exceptions I guess
// * `fold_left` things are really not correctly named

#endif
