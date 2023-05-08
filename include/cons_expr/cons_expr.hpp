#ifndef CONS_EXPR_HPP
#define CONS_EXPR_HPP

#include <charconv>
#include <functional>
#include <ranges>
#include <span>
#include <stacktrace>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

// https://en.wikipedia.org/wiki/Greenspun%27s_tenth_rule
//
// Any sufficiently complicated C or Fortran program contains an ad hoc,
// informally-specified, bug-ridden, slow implementation of half of Common Lisp.

// ┌─────────────────────────┐┌─────────────────────────┐┌──────────────────────────┐
// │ LISP is over half a     ││ I wonder if the cycles  ││   /  These are your    \ │
// │ century old and it      ││ will continue forever.  ││   |father's parentheses| │
// │ still has this perfect  ││\________________  _____/││  / ____________________/ │
// │ timeless air about it.  ││              /-\|/      ││  |/                      │
// │\_______________  ______/││              | |        ││ /-\ ((()            /-\  │
// │                \|       ││              \-/        ││ | | ((()            | |  │
// │  ╔══            /-\     ││          \   /|         ││ \-/ ((()            \-/  │
// │  ║ |            | |     ││          |\ / |         ││  |---- )            /|\  │
// │  ║-/       \    \-/     ││       ------  |         ││  |    (            / | \ │
// │  /|\       |\   /|      ││ ________________|\____  ││  |         _______   |   │
// │  || \   ------ - |*     ││/                       \││ / \       /elegant\ / \  │
// │   |      |   |/--|*     ││ A few coders from each  ││/   \      |weapons|/   \ │
// │  / \     |   ||****     ││ new generation will re- ││ _____|\___\       /______│
// │ /   \    |   ||*  *     ││ discover the LISP arts. ││for a more...civilized age│
// └─────────────────────────┘└─────────────────────────┘└──────────────────────────┘
//  originally from: https://xkcd.com/297/  https://xkcd.com/license.html


/// Goals
//
// * always stay small and hackable. At most 1,000 lines, total, ever, with comments
// * s-expression based, Scheme-inspired embedded language for C++
// * constexpr evaluation of script possible
// * constexpr creation of interpreter possible
// * small memory footprint, fast
// * extreme type safety, no implicit conversions
// * all objects are immutable, copies are preferred over sharing
// * allow the user to add pointer types if they choose to, for sharing of data
//   between script and C++
// * C++23 as a minimum
// * never thread safe


namespace lefticus {

template<typename Contained, std::size_t SmallSize, typename KeyType, typename SpanType = std::span<const Contained>>
struct SmallOptimizedVector
{
  std::array<Contained, SmallSize> small;
  std::size_t small_size_used = 0;
  std::vector<Contained> rest;
  static constexpr auto small_capacity = SmallSize;

  [[nodiscard]] constexpr auto max_index() const noexcept
  {
    if (!rest.empty()) {
      return rest.size() + SmallSize;
    } else {
      return small_size_used;
    }
  }
  [[nodiscard]] constexpr Contained &operator[](std::size_t index)
  {
    if (index < SmallSize) {
      return small[index];
    } else {
      return rest[index - SmallSize];
    }
  }

  [[nodiscard]] constexpr const Contained &operator[](std::size_t index) const
  {
    if (index < SmallSize) {
      return small[index];
    } else {
      return rest[index - SmallSize];
    }
  }

  [[nodiscard]] constexpr SpanType operator[](KeyType range) const
  {
    if (range.start >= SmallSize) {
      return SpanType{ std::span<const Contained>(rest).subspan(range.start - SmallSize, range.size) };
    } else {
      return SpanType{ std::span<const Contained>(small).subspan(range.start, range.size) };
    }
  }

  template<typename... Param> constexpr auto emplace_back(Param &&...param)
  {
    return insert(Contained{ std::forward<Param>(param)... });
  }

  constexpr std::size_t insert(Contained obj, bool force_rest = false)
  {
    if (!rest.empty() || force_rest || small_size_used == SmallSize) {
      rest.push_back(std::move(obj));
      return (rest.size() - 1) + SmallSize;
    } else {
      small[small_size_used] = std::move(obj);
      return small_size_used++;
    }
  }

  constexpr auto small_end() { return std::next(small.begin(), static_cast<std::ptrdiff_t>(small_size_used)); }

  struct Iterator
  {
    using difference_type = std::ptrdiff_t;
    using value_type = Contained;
    using pointer = const Contained *;
    using reference = const Contained &;
    using iterator_category = std::bidirectional_iterator_tag;

    const SmallOptimizedVector *container;
    std::size_t index;

    constexpr bool operator==(const Iterator &) const noexcept = default;
    constexpr bool operator!=(const Iterator &) const noexcept = default;

    constexpr const auto &operator*() const { return (*container)[index]; }
    constexpr auto &operator++()
    {
      ++index;
      if (index == container->small_size_used && !container->rest.empty()) { index = SmallSize; }
      return *this;
    }
    [[nodiscard]] constexpr auto operator++(int)
    {
      auto result = *this;
      ++(*this);
      return result;
    }
    [[nodiscard]] constexpr auto operator--(int)
    {
      auto result = *this;
      --(*this);
      return result;
    }

    constexpr auto &operator--()
    {
      if (index == SmallSize) {
        index = container->small_size_used - 1;
      } else {
        --index;
      }
      return *this;
    }
  };

  [[nodiscard]] constexpr auto view(KeyType span) const noexcept
  {
    struct View
    {
      KeyType span;
      const SmallOptimizedVector *container;
      constexpr auto begin() const { return Iterator{ container, span.start }; }
      constexpr auto end() const { return Iterator{ container, span.start + span.size }; }
    };

    return View{ span, this };
  }

  constexpr auto view() const
  {
    if (rest.empty()) {
      return view(KeyType{ 0, small_size_used });
    } else {
      return view(KeyType{ 0, rest.size() + SmallSize });
    }
  }

  constexpr KeyType insert_or_find(SpanType values)
  {
    if (const auto small_found = std::search(small.begin(), small_end(), values.begin(), values.end());
        small_found != small_end()) {
      return KeyType{ static_cast<std::size_t>(std::distance(small.begin(), small_found)), values.size() };
    } else if (const auto rest_found = std::search(rest.begin(), rest.end(), values.begin(), values.end());
               rest_found != rest.end()) {
      return KeyType{ static_cast<std::size_t>(std::distance(rest.begin(), rest_found)) + SmallSize, values.size() };
    } else {
      return insert(values);
    }
  }

  constexpr KeyType insert(SpanType values)
  {
    const bool force_rest = (values.size() + small_size_used) > SmallSize;
    std::size_t last = 0;
    for (const auto &value : values) { last = insert(value, force_rest); }
    return KeyType{ last - values.size() + 1, values.size() };
  }
};

template<typename T>
concept not_bool_or_ptr = !std::same_as<std::remove_cvref_t<T>, bool> && !std::is_pointer_v<std::remove_cvref_t<T>>;

static constexpr auto adds = []<not_bool_or_ptr T>(const T &lhs, const T &rhs)
  requires requires { lhs + rhs; }
{
  return lhs + rhs;
};

static constexpr auto multiplies = []<not_bool_or_ptr T>(const T &lhs, const T &rhs)
  requires requires { lhs *rhs; }
{
  return lhs * rhs;
};

static constexpr auto divides = []<not_bool_or_ptr T>(const T &lhs, const T &rhs)
  requires requires { lhs / rhs; }
{
  return lhs / rhs;
};

static constexpr auto subtracts = []<not_bool_or_ptr T>(const T &lhs, const T &rhs)
  requires requires { lhs - rhs; }
{
  return lhs - rhs;
};

static constexpr auto less_than = []<not_bool_or_ptr T>(const T &lhs, const T &rhs)
  requires requires { lhs < rhs; }
{
  return lhs < rhs;
};
static constexpr auto greater_than = []<not_bool_or_ptr T>(const T &lhs, const T &rhs)
  requires requires { lhs > rhs; }
{
  return lhs > rhs;
};

static constexpr auto less_than_equal = []<not_bool_or_ptr T>(const T &lhs, const T &rhs)
  requires requires { lhs <= rhs; }
{
  return lhs <= rhs;
};

static constexpr auto greater_than_equal = []<not_bool_or_ptr T>(const T &lhs, const T &rhs)
  requires requires { lhs >= rhs; }
{
  return lhs >= rhs;
};


static constexpr auto equal = []<typename T>(const T &lhs, const T &rhs) -> bool
  requires requires { lhs == rhs; }
{
  return lhs == rhs;
};

static constexpr auto not_equal = []<typename T>(const T &lhs, const T &rhs) -> bool
  requires requires { lhs != rhs; }
{
  return lhs != rhs;
};

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
  static constexpr std::pair<bool, T> failure{ false, 0 };
  if (input == "-") { return failure; }

  enum struct State {
    Start,
    IntegerPart,
    FractionPart,
    ExponentPart,
    ExponentStart,
  };

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

    [[nodiscard]] constexpr auto float_value() const noexcept -> std::pair<bool, T>
    {
      if (state == State::Start || state == State::ExponentStart) { return { false, 0 }; }

      return { true,
        (static_cast<T>(value_sign) * (static_cast<T>(value) + static_cast<T>(frac) * pow(static_cast<T>(10), frac_exp))
          * pow(static_cast<T>(10), exp_sign * exp)) };
    }
  };

  ParseState state;

  auto parse_digit = [](auto &value, char ch) {
    if (ch >= '0' && ch <= '9') {
      value = value * 10 + ch - '0';
      return true;
    } else {
      return false;
    }
  };

  for (const char c : input) {
    switch (state.state) {
    case State::Start:
      if (c == '-') {
        state.value_sign = -1;
      } else if (!parse_digit(state.value, c)) {
        return failure;
      }
      state.state = State::IntegerPart;
      break;
    case State::IntegerPart:
      if (c == '.') {
        state.state = State::FractionPart;
      } else if (c == 'e' || c == 'E') {
        state.state = State::ExponentPart;
      } else if (!parse_digit(state.value, c)) {
        return failure;
      }
      break;
    case State::FractionPart:
      if (parse_digit(state.frac, c)) {
        state.frac_exp--;
      } else if (c == 'e' || c == 'E') {
        state.state = State::ExponentStart;
      } else {
        return failure;
      }
      break;
    case State::ExponentStart:
      if (c == '-') {
        state.exp_sign = -1;
      } else if (!parse_digit(state.exp, c)) {
        return failure;
      }
      state.state = State::ExponentPart;
      break;
    case State::ExponentPart:
      if (!parse_digit(state.exp, c)) { return failure; }
    }
  }

  return state.float_value();
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
  std::size_t size;
  [[nodiscard]] constexpr bool operator==(const IndexedString &) const noexcept = default;
};

struct IndexedList
{
  std::size_t start;
  std::size_t size;
  [[nodiscard]] constexpr bool operator==(const IndexedList &) const noexcept = default;
};

struct LiteralList
{
  IndexedList items;
  [[nodiscard]] constexpr bool operator==(const LiteralList &) const noexcept = default;
};

struct Identifier
{
  IndexedString value;
  [[nodiscard]] constexpr bool operator==(const Identifier &) const noexcept = default;
};

template<std::size_t BuiltInSymbolsSize = 64,
  std::size_t BuiltInStringsSize = 1540,
  std::size_t BuiltInValuesSize = 279,
  typename... UserTypes>
struct cons_expr
{
  struct SExpr;

  using LexicalScope = SmallOptimizedVector<std::pair<IndexedString, SExpr>, BuiltInSymbolsSize, IndexedList>;


  using function_ptr = SExpr (*)(cons_expr &, LexicalScope &, std::span<const SExpr>);

  struct FunctionPtr
  {
    function_ptr ptr;
    [[nodiscard]] constexpr bool operator==(const FunctionPtr &) const noexcept { return false; }
  };

  [[nodiscard]] static constexpr SExpr for_each(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    if (params.size() != 2) { throw std::runtime_error("Wrong number of parameters to for-each expression"); }

    const auto &list = engine.values[engine.eval_to<LiteralList>(scope, params[1]).items];

    for (auto itr = list.begin(); itr != list.end(); ++itr) {
      [[maybe_unused]] const auto result =
        engine.invoke_function(scope, params[0], std::span<const SExpr>{ itr, std::next(itr) });
    }

    return SExpr{ Atom{ std::monostate{} } };
  }

  using Atom = std::variant<std::monostate, bool, int, double, IndexedString, Identifier, UserTypes...>;

  struct Closure;

  struct SExpr
  {
    std::variant<Atom, IndexedList, LiteralList, Closure, FunctionPtr> value;
    [[nodiscard]] constexpr bool operator==(const SExpr &) const noexcept = default;

    constexpr std::span<const SExpr> to_list(const cons_expr &engine) const
    {
      if (const auto *list = std::get_if<IndexedList>(&value); list != nullptr) { return engine.values[*list]; }

      throw std::runtime_error("SExpr is not a list");
    }
  };

  static_assert(std::is_trivially_copyable_v<SExpr> && std::is_trivially_destructible_v<SExpr>,
    "cons_expr does not work well with non-trivial types");

  template<typename Result> [[nodiscard]] constexpr const Result *get_if(const SExpr *sexpr) const
  {
    if (sexpr == nullptr) { return nullptr; }

    if constexpr (std::is_same_v<Result, Atom> || std::is_same_v<Result, IndexedList>
                  || std::is_same_v<Result, LiteralList> || std::is_same_v<Result, Closure>
                  || std::is_same_v<Result, FunctionPtr>) {
      return std::get_if<Result>(&sexpr->value);
    } else {
      if (const auto *atom = std::get_if<Atom>(&sexpr->value)) {
        return std::get_if<Result>(atom);
      } else {
        return nullptr;
      }
    }
  }

  LexicalScope global_scope{};
  SmallOptimizedVector<char, BuiltInStringsSize, IndexedString, std::string_view> strings{};
  SmallOptimizedVector<SExpr, BuiltInValuesSize, IndexedList> values{};

  struct Closure
  {
    IndexedList parameter_names;
    IndexedList statements;

    [[nodiscard]] constexpr bool operator==(const Closure &) const noexcept = default;

    [[nodiscard]] constexpr SExpr
      invoke(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> parameters) const
    {
      if (parameters.size() != parameter_names.size) {
        throw std::runtime_error("Incorrect number of parameters for lambda");
      }

      LexicalScope new_scope;

      const auto parameter_names_list = engine.values[parameter_names];

      // set up params
      // technically I'm evaluating the params lazily while invoking the lambda, not before
      // does it matter?
      for (std::size_t index = 0; index < parameter_names_list.size(); ++index) {
        new_scope.emplace_back(std::get<Identifier>(std::get<Atom>(parameter_names_list[index].value)).value,
          engine.eval(scope, parameters[index]));
      }

      return engine.sequence(new_scope, engine.values[statements]);
    }
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
          retval.push_back(SExpr{ Atom(strings.insert_or_find(token.parsed.substr(1, token.parsed.size() - 2))) });
        } else if (auto [int_did_parse, int_value] = parse_int(token.parsed); int_did_parse) {
          retval.push_back(SExpr{ Atom(int_value) });
        } else if (auto [float_did_parse, float_value] = parse_float<double>(token.parsed); float_did_parse) {
          retval.push_back(SExpr{ Atom(float_value) });
        } else {
          retval.push_back(SExpr{ Atom(Identifier{ strings.insert_or_find(token.parsed) }) });
        }
      }
      token = next_token(token.remaining);
    }
    return std::pair<SExpr, Token>(SExpr{ values.insert_or_find(retval) }, token);
  }


  consteval cons_expr()
  {
    add("+", SExpr{ FunctionPtr{ binary_left_fold<adds> } });
    add("*", SExpr{ FunctionPtr{ binary_left_fold<multiplies> } });
    add("-", SExpr{ FunctionPtr{ binary_left_fold<subtracts> } });
    add("/", SExpr{ FunctionPtr{ binary_left_fold<divides> } });
    add("<", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<less_than> } });
    add(">", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<greater_than> } });
    add("<=", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<less_than_equal> } });
    add(">=", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<greater_than_equal> } });
    add("and", SExpr{ FunctionPtr{ logical_and } });
    add("or", SExpr{ FunctionPtr{ logical_or } });
    add("if", SExpr{ FunctionPtr{ ifer } });
    add("not", SExpr{ FunctionPtr{ make_evaluator<logical_not>() } });
    add("==", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<equal> } });
    add("!=", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<not_equal> } });
    add("for-each", SExpr{ FunctionPtr{ for_each } });
    add("list", SExpr{ FunctionPtr{ list } });
    add("lambda", SExpr{ FunctionPtr{ lambda } });
    add("do", SExpr{ FunctionPtr{ doer } });
    add("define", SExpr{ FunctionPtr{ definer } });
    add("let", SExpr{ FunctionPtr{ letter } });
  }

  [[nodiscard]] constexpr SExpr sequence(LexicalScope &scope, std::span<const SExpr> statements)
  {
    if (!statements.empty()) {
      for (const auto &statement : statements.subspan(0, statements.size() - 1)) {
        [[maybe_unused]] const auto result = eval(scope, statement);
      }
      return eval(scope, statements.back());
    } else {
      return SExpr{ Atom{ std::monostate{} } };
    }
  }

  [[nodiscard]] constexpr SExpr
    invoke_function(LexicalScope &scope, const SExpr &function, std::span<const SExpr> parameters)
  {
    const SExpr resolved_function = eval(scope, function);

    if (auto *lambda = get_if<Closure>(&resolved_function); lambda != nullptr) {
      return lambda->invoke(*this, scope, parameters);
    } else {
      return get_function(resolved_function)(*this, scope, parameters);
    }
  }

  [[nodiscard]] constexpr function_ptr get_function(const SExpr &expr)
  {
    if (const auto *func = get_if<FunctionPtr>(&expr); func != nullptr) { return func->ptr; }
    throw std::runtime_error("Does not evaluate to a function");
  }

  template<auto Func, typename Ret, typename... Param>
  [[nodiscard]] constexpr static function_ptr make_evaluator(Ret (*)(Param...))
  {
    return function_ptr{ [](cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params) -> SExpr {
      if (params.size() != sizeof...(Param)) { throw std::runtime_error("wrong param count"); }

      auto impl = [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        if constexpr (std::is_same_v<void, Ret>) {
          Func(engine.eval_to<std::remove_cvref_t<Param>>(scope, params[Idx])...);
          return SExpr{ Atom{ std::monostate{} } };
        } else {
          return SExpr{ Func(engine.eval_to<Param>(scope, params[Idx])...) };
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
    global_scope.emplace_back(strings.insert_or_find(name), SExpr{ make_evaluator<Func>() });
  }

  constexpr void add(std::string_view name, SExpr value)
  {
    global_scope.insert(std::pair<IndexedString, SExpr>(strings.insert_or_find(name), std::move(value)));
  }

  template<typename Value> constexpr void add(std::string_view name, Value &&value)
  {
    global_scope.insert(
      std::pair<IndexedString, SExpr>(strings.insert_or_find(name), SExpr{ Atom{ std::forward<Value>(value) } }));
  }


  [[nodiscard]] constexpr SExpr eval(LexicalScope &scope, const SExpr &expr)
  {
    if (const auto *indexedlist = get_if<IndexedList>(&expr); indexedlist != nullptr) {
      // if it's a non-empty list, then I need to evaluate it as a function
      auto list = values[*indexedlist];
      if (!list.empty()) { return invoke_function(scope, list[0], { std::next(list.begin()), list.end() }); }
    } else if (const auto *id = get_if<Identifier>(&expr); id != nullptr) {
      for (const auto &[key, value] : scope.view() | std::ranges::views::reverse) {
        if (key == id->value) { return value; }
      }

      throw std::runtime_error("id not found");
    }
    return expr;
  }

  template<typename Type> [[nodiscard]] constexpr Type eval_to(LexicalScope &scope, const SExpr &expr)
  {
    if constexpr (std::is_same_v<Type, SExpr>) {
      return expr;
    } else if constexpr (std::is_same_v<Type, LiteralList> || std::is_same_v<Type, IndexedList>
                         || std::is_same_v<Type, Closure> || std::is_same_v<Type, FunctionPtr>) {
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
    return eval_to<Type>(scope, eval(scope, expr));
  }

  [[nodiscard]] static constexpr SExpr list(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    std::vector<SExpr> result;

    for (const auto &param : params) { result.push_back(engine.eval(scope, param)); }

    return SExpr{ LiteralList{ engine.values.insert(result) } };
  }

  constexpr std::vector<IndexedString> get_lambda_parameter_names(const SExpr &sexpr)
  {
    std::vector<IndexedString> retval;
    if (auto *parameter_list = get_if<IndexedList>(&sexpr); parameter_list != nullptr) {
      for (const auto &expr : values[*parameter_list]) {
        if (auto *local_id = get_if<Identifier>(&expr); local_id != nullptr) { retval.push_back(local_id->value); }
      }
    }
    return retval;
  }

  [[nodiscard]] static constexpr SExpr lambda(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    if (params.size() < 2) { throw std::runtime_error("Wrong number of parameters to lambda expression"); }

    // replace all references to captured values with constant copies
    std::vector<SExpr> fixed_statements;

    auto locals = engine.get_lambda_parameter_names(params[0]);

    for (const auto &statement : params.subspan(1)) {
      // all of current scope is const and capturable
      fixed_statements.push_back(engine.fix_identifiers(statement, locals, scope));
    }

    return SExpr{ Closure{
      std::get<IndexedList>(params[0].value), { engine.values.insert_or_find(fixed_statements) } } };
  }

  [[nodiscard]] static constexpr SExpr definer(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    if (params.size() != 2) { throw std::runtime_error("Wrong number of parameters to define expression"); }
    scope.emplace_back(engine.eval_to<Identifier>(scope, params[0]).value,
      engine.fix_identifiers(engine.eval(scope, params[1]), {}, scope));
    return SExpr{ Atom{ std::monostate{} } };
  }

  [[nodiscard]] constexpr SExpr fix_identifiers(const SExpr &input,
    std::span<const IndexedString> local_identifiers,
    const LexicalScope &local_constants)
  {
    if (auto *list = get_if<IndexedList>(&input); list != nullptr) {
      if (list->size != 0) {
        auto first_index = list->start;
        const auto &elem = values[first_index];
        if (auto *id = get_if<Identifier>(&elem); id != nullptr) {
          auto string = strings[id->value];
          if (string == "lambda") {
            std::vector<IndexedString> new_locals{ local_identifiers.begin(), local_identifiers.end() };
            auto lambda_locals = get_lambda_parameter_names(values[first_index + 1]);
            new_locals.insert(new_locals.end(), lambda_locals.begin(), lambda_locals.end());

            std::vector<SExpr> new_lambda;
            // fixup pointer to "lambda" function
            new_lambda.push_back(fix_identifiers(values[first_index], new_locals, local_constants));
            new_lambda.push_back(values[first_index + 1]);

            for (auto index = first_index + 2; index < list->size + list->start; ++index) {
              new_lambda.push_back(fix_identifiers(values[index], new_locals, local_constants));
            }

            return SExpr{ values.insert_or_find(new_lambda) };
          } else if (string == "let") {
            std::vector<IndexedString> new_locals{ local_identifiers.begin(), local_identifiers.end() };

            std::vector<SExpr> new_parameters;

            // collect all locals
            for (const auto &param : values[first_index + 1].to_list(*this)) {
              auto param_list = param.to_list(*this);
              new_locals.push_back(get_if<Identifier>(&param_list[0])->value);
            }

            for (const auto &param : values[first_index + 1].to_list(*this)) {
              auto param_list = param.to_list(*this);
              std::vector<SExpr> new_param;
              new_param.push_back(param_list[0]);
              new_param.push_back(fix_identifiers(param_list[1], local_identifiers, local_constants));
              new_parameters.push_back(SExpr{ values.insert_or_find(new_param) });
            }

            std::vector<SExpr> new_let;
            // fixup pointer to "let" function
            new_let.push_back(fix_identifiers(values[first_index], new_locals, local_constants));
            new_let.push_back(SExpr{ values.insert_or_find(new_parameters) });

            for (auto index = first_index + 2; index < list->size + list->start; ++index) {
              new_let.push_back(fix_identifiers(values[index], new_locals, local_constants));
            }

            return SExpr{ values.insert_or_find(new_let) };
          } else if (string == "define") {
            std::vector<IndexedString> new_locals{ local_identifiers.begin(), local_identifiers.end() };
            new_locals.push_back(get_if<Identifier>(&values[first_index + 1])->value);

            std::vector<SExpr> new_define;
            new_define.push_back(fix_identifiers(values[first_index], local_identifiers, local_constants));
            new_define.push_back(values[first_index + 1]);
            new_define.push_back(fix_identifiers(values[first_index + 2], new_locals, local_constants));
            return SExpr{ values.insert_or_find(new_define) };
          } else if (string == "do") {
            // we don't want to fix up things that set their own scope (yet)
            return input;
          }
        }
      }
      std::vector<SExpr> result;
      result.reserve(list->size);
      for (const auto &value : values[*list]) {
        result.push_back(fix_identifiers(value, local_identifiers, local_constants));
      }
      return SExpr{ this->values.insert_or_find(result) };
    } else if (auto *id = get_if<Identifier>(&input); id != nullptr) {
      for (const auto &local : local_identifiers) {
        if (local == id->value) {
          // do something smarter later, but abort for now because it's in the variable scope
          return input;
        }
      }

      // we're hoping it's a constant
      for (const auto &object : local_constants.view() | std::ranges::views::reverse) {
        if (object.first == id->value) { return object.second; }
      }

      return input;
    }

    return input;
  }

  [[nodiscard]] static constexpr SExpr letter(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    if (params.empty()) { throw std::runtime_error("Wrong number of parameters to let expression"); }

    std::vector<std::pair<std::size_t, SExpr>> variables;

    auto new_scope = scope;

    const auto setup_variable = [&](const auto &expr) {
      auto elements = expr.to_list(engine);
      if (elements.size() != 2) { throw std::runtime_error(""); }

      new_scope.emplace_back(engine.eval_to<Identifier>(scope, elements[0]).value, engine.eval(scope, elements[1]));
    };

    const auto setup_variables = [&](const auto &expr) {
      auto elements = expr.to_list(engine);
      for (const auto &variable : elements) { setup_variable(variable); }
    };

    setup_variables(params[0]);

    // evaluate body
    return engine.sequence(new_scope, params.subspan(1));
  }


  [[nodiscard]] static constexpr SExpr doer(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    if (params.size() < 2) { throw std::runtime_error("Wrong number of parameters to do expression"); }

    std::vector<std::pair<std::size_t, SExpr>> variables;

    std::vector<IndexedString> variable_names;

    auto new_scope = scope;

    const auto setup_variable = [&](const auto &expr) {
      auto elements = expr.to_list(engine);
      if (elements.size() != 3) { throw std::runtime_error(""); }

      const auto index = new_scope.max_index();
      new_scope.emplace_back(engine.eval_to<Identifier>(scope, elements[0]).value, engine.eval(scope, elements[1]));
      variables.emplace_back(index, elements[2]);
    };

    const auto setup_variables = [&](const auto &expr) {
      auto elements = expr.to_list(engine);
      for (const auto &variable : elements) { setup_variable(variable); }
    };

    setup_variables(params[0]);

    for (auto &variable : variables) { variable.second = engine.fix_identifiers(variable.second, variable_names, {}); }

    // make copy of scope first, then build this from entire scope
    for (const auto &local : new_scope.view()) { variable_names.push_back(local.first); }

    const auto terminators = params[1].to_list(engine);

    std::vector<std::pair<std::size_t, SExpr>> new_values;

    auto fixed_up_terminator = engine.fix_identifiers(terminators[0], variable_names, {});

    // continue while terminator test is false
    while (!engine.eval_to<bool>(new_scope, fixed_up_terminator)) {
      // evaluate body
      [[maybe_unused]] const auto result = engine.sequence(new_scope, params.subspan(2));

      // iterate loop variables
      for (const auto &[index, expr] : variables) { new_values.emplace_back(index, engine.eval(new_scope, expr)); }

      // update values
      for (auto &&[index, value] : new_values) { new_scope[index].second = std::move(value); }

      new_values.clear();
    }

    // evaluate sequence of termination expressions
    return engine.sequence(new_scope, terminators.subspan(1));
  }

  [[nodiscard]] static constexpr SExpr ifer(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    if (params.size() != 3) { throw std::runtime_error("Wrong number of parameters to if expression"); }

    if (engine.eval_to<bool>(scope, params[0])) {
      return engine.eval(scope, params[1]);
    } else {
      return engine.eval(scope, params[2]);
    }
  }

  template<typename Signature>
  [[nodiscard]] auto make_callable(auto function)
    requires std::is_function_v<Signature>
  {
    auto impl = [this, function]<typename Ret, typename... Params>(Ret (*)(Params...)) {
      // this is fragile, we need to check parsing better

      return [callable = eval(global_scope, values[std::get<IndexedList>(parse(function).first.value)][0]), this](
               Params... params) {
        std::array<SExpr, sizeof...(Params)> args{ SExpr{ Atom{ params } }... };
        return eval_to<Ret>(global_scope, invoke_function(global_scope, callable, args));
      };
    };

    return impl(std::add_pointer_t<Signature>{ nullptr });
  }

  template<auto Op>
  [[nodiscard]] static constexpr SExpr
    binary_left_fold(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    auto sum = [&engine, &scope, params]<typename Param>(Param first) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        for (const auto &next : params.subspan(1)) { first = Op(first, engine.eval_to<Param>(scope, next)); }

        return SExpr{ Atom{ first } };
      } else {
        throw std::runtime_error("Operator not supported for types");
      }
    };

    if (params.size() > 1) { return std::visit(sum, std::get<Atom>(engine.eval(scope, params[0]).value)); }

    throw std::runtime_error("Not enough params");
  }

  [[nodiscard]] static constexpr SExpr
    logical_and(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    for (const auto &next : params) {
      if (!engine.eval_to<bool>(scope, next)) { return SExpr{ Atom{ false } }; }
    }

    return SExpr{ Atom{ true } };
  }

  [[nodiscard]] static constexpr SExpr logical_or(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    for (const auto &next : params) {
      if (engine.eval_to<bool>(scope, next)) { return SExpr{ Atom{ true } }; }
    }

    return SExpr{ Atom{ false } };
  }

  template<auto Op>
  [[nodiscard]] static constexpr SExpr
    binary_boolean_apply_pairwise(cons_expr &engine, LexicalScope &scope, std::span<const SExpr> params)
  {
    auto sum = [&engine, &scope, params]<typename Param>(Param first) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        auto second = engine.eval_to<Param>(scope, params[1]);
        bool result = Op(first, second);
        bool odd = true;
        for (const auto &next : params.subspan(2)) {
          if (!result) { return SExpr{ Atom{ false } }; }

          if (odd) {
            first = engine.eval_to<Param>(scope, next);
            result = result && Op(second, first);
          } else {
            second = engine.eval_to<Param>(scope, next);
            result = result && Op(first, second);
          }

          odd = !odd;
        }

        return SExpr{ Atom{ result } };
      } else {
        throw std::runtime_error("Operator not supported for types");
      }
    };

    if (params.size() > 1) { return std::visit(sum, std::get<Atom>(engine.eval(scope, params[0]).value)); }

    throw std::runtime_error("Not enough params");
  }
};


}// namespace lefticus


/// TODO
// * add cons car cdr eval apply
// * remove exceptions I guess?
// * check propogation of lambda constants down into do/lambda/whatever below it
// * make allocator aware
#endif
