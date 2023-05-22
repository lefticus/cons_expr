#ifndef CONS_EXPR_HPP
#define CONS_EXPR_HPP

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <stacktrace>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
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

/// Why?
// I (Jason Turner / lefticus) failed to complete my LISP project in Comparative Languages in CS at Virginia Tech
// in approximately Spring 1999. This is my revenge.

/// Goals
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

/// Notes
// it's a scheme-like language with a few caveats:
// * Once an object is captured or used, it's immutable
// * `==` `true` and `false` stray from `=` `#t` and `#f` of scheme
// * Pair types don't exist, only lists
// * only indices and values are passed, for safety during resize of `values` object
// Triviality of types is critical to design and structure of this system

/// To do
// * We probably want some sort of "defragment" at some point
// * Consider removing exceptions and return first-class error objects instead
// * Add constant folding capability
// * Allow functions to be registered as "pure" so they can be folded!
// * make allocator aware
// * make char type templatable

namespace lefticus {

inline constexpr int cons_expr_version_major{ 0 };
inline constexpr int cons_expr_version_minor{ 0 };
inline constexpr int cons_expr_version_patch{ 1 };
inline constexpr int cons_expr_version_tweak{};


template<std::unsigned_integral SizeType,
  typename Contained,
  SizeType SmallSize,
  typename KeyType,
  typename SpanType = std::span<const Contained>>
struct SmallOptimizedVector
{
  using size_type = SizeType;

  std::array<Contained, SmallSize> small;
  size_type small_size_used = 0;
  std::vector<Contained> rest;


  static constexpr auto small_capacity = SmallSize;

  [[nodiscard]] constexpr auto size() const noexcept
  {
    if (!rest.empty()) {
      return static_cast<size_type>(rest.size() + SmallSize);
    } else {
      return small_size_used;
    }
  }
  [[nodiscard]] constexpr Contained &operator[](size_type index)
  {
    if (index < SmallSize) {
      return small[index];
    } else {
      return rest[index - SmallSize];
    }
  }

  [[nodiscard]] constexpr const Contained &operator[](size_type index) const
  {
    if (index < SmallSize) {
      return small[index];
    } else {
      return rest[index - SmallSize];
    }
  }

  [[nodiscard]] constexpr SpanType view(KeyType range) const
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

  constexpr size_type insert(Contained obj, bool force_rest = false)
  {
    if (!rest.empty() || force_rest || small_size_used == SmallSize) {
      rest.push_back(std::move(obj));
      return static_cast<size_type>(rest.size() - 1) + SmallSize;
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
    size_type index;

    [[nodiscard]] constexpr bool operator==(const Iterator &other) const noexcept { return index == other.index; }
    [[nodiscard]] constexpr bool operator!=(const Iterator &) const noexcept = default;

    constexpr const auto &operator*() const noexcept { return (*container)[index]; }
    constexpr auto &operator++() noexcept
    {
      ++index;
      return *this;
    }
    [[nodiscard]] constexpr auto operator++(int) noexcept
    {
      auto result = *this;
      ++(*this);
      return result;
    }

    constexpr auto &operator--() noexcept
    {
      --index;
      return *this;
    }

    [[nodiscard]] constexpr auto operator--(int) noexcept
    {
      auto result = *this;
      --(*this);
      return result;
    }
  };

  struct View
  {
    KeyType span;
    const SmallOptimizedVector *container;
    [[nodiscard]] constexpr const auto &operator[](size_type offset) const noexcept
    {
      return (*container)[span.start + offset];
    }
    [[nodiscard]] constexpr auto empty() const noexcept { return span.empty(); }
    [[nodiscard]] constexpr auto size() const noexcept { return span.size; }
    [[nodiscard]] constexpr auto begin() const noexcept { return Iterator{ container, span.start }; }
    [[nodiscard]] constexpr auto end() const noexcept
    {
      return Iterator{ container, static_cast<size_type>(span.start + span.size) };
    }
  };

  [[nodiscard]] constexpr auto operator[](KeyType span) const noexcept { return View{ span, this }; }
  [[nodiscard]] constexpr auto begin() const noexcept { return Iterator{ this, 0 }; }
  [[nodiscard]] constexpr auto end() const noexcept { return Iterator{ this, size() }; }

  constexpr KeyType insert_or_find(SpanType values)
  {
    if (const auto small_found = std::search(small.begin(), small_end(), values.begin(), values.end());
        small_found != small_end()) {
      return KeyType{ static_cast<size_type>(std::distance(small.begin(), small_found)),
        static_cast<size_type>(values.size()) };
    } else if (const auto rest_found = std::search(rest.begin(), rest.end(), values.begin(), values.end());
               rest_found != rest.end()) {
      return KeyType{ static_cast<size_type>(
                        static_cast<size_type>(std::distance(rest.begin(), rest_found)) + SmallSize),
        static_cast<size_type>(values.size()) };
    } else {
      return insert(values);
    }
  }

  constexpr KeyType insert(SpanType values)
  {
    const bool force_rest = !rest.empty() || ((values.size() + small_size_used) > SmallSize);
    size_type last = 0;
    for (const auto &value : values) { last = insert(value, force_rest); }
    return KeyType{ static_cast<size_type>(last - values.size() + 1), static_cast<size_type>(values.size()) };
  }
};

template<typename T>
concept not_bool_or_ptr = !std::same_as<std::remove_cvref_t<T>, bool> && !std::is_pointer_v<std::remove_cvref_t<T>>;
template<typename T>
concept addable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs + rhs; };
template<typename T>
concept multipliable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs *rhs; };
template<typename T>
concept dividable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs / rhs; };
template<typename T>
concept subtractable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs - rhs; };
template<typename T>
concept lt_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs < rhs; };
template<typename T>
concept gt_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs > rhs; };
template<typename T>
concept lt_eq_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs <= rhs; };
template<typename T>
concept gt_eq_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs >= rhs; };
template<typename T>
concept eq_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs == rhs; };
template<typename T>
concept not_eq_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs != rhs; };

static constexpr auto adds = []<addable T>(const T &lhs, const T &rhs) { return lhs + rhs; };
static constexpr auto multiplies = []<multipliable T>(const T &lhs, const T &rhs) { return lhs * rhs; };
static constexpr auto divides = []<dividable T>(const T &lhs, const T &rhs) { return lhs / rhs; };
static constexpr auto subtracts = []<subtractable T>(const T &lhs, const T &rhs) { return lhs - rhs; };
static constexpr auto less_than = []<lt_comparable T>(const T &lhs, const T &rhs) { return lhs < rhs; };
static constexpr auto greater_than = []<gt_comparable T>(const T &lhs, const T &rhs) { return lhs > rhs; };
static constexpr auto lt_equal = []<lt_eq_comparable T>(const T &lhs, const T &rhs) { return lhs <= rhs; };
static constexpr auto gt_equal = []<gt_eq_comparable T>(const T &lhs, const T &rhs) { return lhs >= rhs; };
static constexpr auto equal = []<eq_comparable T>(const T &lhs, const T &rhs) -> bool { return lhs == rhs; };
static constexpr auto not_equal = []<not_eq_comparable T>(const T &lhs, const T &rhs) -> bool { return lhs != rhs; };
static constexpr bool logical_not(bool lhs) { return !lhs; }

template<typename CharType> struct Token
{
  using char_type = CharType;
  std::basic_string_view<char_type> parsed;
  std::basic_string_view<char_type> remaining;
};

template<std::integral IntType, typename CharType>
[[nodiscard]] constexpr std::pair<bool, IntType> parse_int(std::basic_string_view<CharType> input)
{
  IntType parsed = 0;
  auto result = std::from_chars(input.data(), input.data() + input.size(), parsed);
  if (result.ec == std::errc() && result.ptr == input.data() + input.size()) {
    return { true, parsed };
  } else {
    return { false, parsed };
  }
}

template<std::floating_point T, typename CharType>
[[nodiscard]] constexpr std::pair<bool, T> parse_float(std::basic_string_view<CharType> input)
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


template<typename CharType> [[nodiscard]] constexpr Token<CharType> next_token(std::basic_string_view<CharType> input)
{
  constexpr auto is_eol = [](auto ch) { return ch == '\n' || ch == '\r'; };
  constexpr auto is_whitespace = [=](auto ch) { return ch == ' ' || ch == '\t' || is_eol(ch); };

  constexpr auto consume = [=](auto ws_input, auto predicate) {
    auto begin = ws_input.begin();
    while (begin != ws_input.end() && predicate(*begin)) { ++begin; }
    return std::basic_string_view<CharType>{ begin, ws_input.end() };
  };

  constexpr auto make_token = [=](auto token_input, std::size_t size) {
    return Token{ token_input.substr(0, size), consume(token_input.substr(size), is_whitespace) };
  };

  input = consume(input, is_whitespace);

  // comment
  if (input.starts_with(';')) {
    input = consume(input, [=](char ch) { return not is_eol(ch); });
    input = consume(input, is_whitespace);
  }

  // list
  if (input.starts_with('(') || input.starts_with(')')) { return make_token(input, 1); }

  // literal list
  if (input.starts_with("'(")) { return make_token(input, 2); }

  // quoted string
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

  // everything else
  const auto value = consume(input, [=](char ch) { return !is_whitespace(ch) && ch != ')' && ch != '('; });

  return make_token(input, static_cast<std::size_t>(std::distance(input.begin(), value.begin())));
}

template<std::unsigned_integral SizeType> struct IndexedString
{
  using size_type = SizeType;
  size_type start{ 0 };
  size_type size{ 0 };
  [[nodiscard]] constexpr bool operator==(const IndexedString &) const noexcept = default;
  [[nodiscard]] constexpr auto front() const noexcept { return start; }
  [[nodiscard]] constexpr auto substr(const size_type from) const noexcept
  {
    return IndexedString{ static_cast<size_type>(start + from), static_cast<size_type>(size - from) };
  }
};

template<std::unsigned_integral SizeType> struct IndexedList
{
  using size_type = SizeType;
  size_type start{ 0 };
  size_type size{ 0 };
  [[nodiscard]] constexpr bool operator==(const IndexedList &) const noexcept = default;
  [[nodiscard]] constexpr bool empty() const noexcept { return size == 0; }
  [[nodiscard]] constexpr auto front() const noexcept { return start; }
  [[nodiscard]] constexpr size_type operator[](size_type index) const noexcept { return start + index; }
  [[nodiscard]] constexpr size_type back() const noexcept { return static_cast<size_type>(start + size - 1); }
  [[nodiscard]] constexpr auto sublist(const size_type from,
    const size_type distance = std::numeric_limits<size_type>::max()) const noexcept
  {
    if (distance == std::numeric_limits<size_type>::max()) {
      return IndexedList{ static_cast<size_type>(start + from), static_cast<size_type>(size - from) };
    } else {
      return IndexedList{ static_cast<size_type>(start + from), distance };
    };
  }
};

template<std::unsigned_integral SizeType> struct LiteralList
{
  using size_type = SizeType;
  IndexedList<size_type> items;
  [[nodiscard]] constexpr auto front() const noexcept { return items.front(); }
  [[nodiscard]] constexpr auto sublist(const size_type from) const noexcept
  {
    return LiteralList{ items.sublist(from) };
  }
  [[nodiscard]] constexpr bool operator==(const LiteralList &) const noexcept = default;
};

template<std::unsigned_integral SizeType> struct Identifier
{
  using size_type = SizeType;
  IndexedString<size_type> value;
  [[nodiscard]] constexpr auto front() const noexcept { return value.front(); }
  [[nodiscard]] constexpr auto substr(const size_type from) const noexcept { return Identifier{ value.substr(from) }; }
  [[nodiscard]] constexpr bool operator==(const Identifier &) const noexcept = default;
};

template<std::unsigned_integral SizeType> struct Error
{
  using size_type = SizeType;
  IndexedString<size_type> expected;
  IndexedList<size_type> got;
  [[nodiscard]] constexpr bool operator==(const Error &) const noexcept = default;
};


template<std::unsigned_integral SizeType = std::uint16_t,
  typename CharType = char,
  std::signed_integral IntegralType = long long,
  std::floating_point FloatType = long double,
  SizeType BuiltInSymbolsSize = 64,
  SizeType BuiltInStringsSize = 1540,
  SizeType BuiltInValuesSize = 279,
  typename... UserTypes>
struct cons_expr
{
  using char_type = CharType;
  using size_type = SizeType;
  using int_type = IntegralType;
  using float_type = FloatType;
  using string_type = IndexedString<size_type>;
  using identifier_type = Identifier<size_type>;
  using list_type = IndexedList<size_type>;
  using literal_list_type = LiteralList<size_type>;
  using error_type = Error<size_type>;

  struct SExpr;
  struct Closure;

  using LexicalScope = SmallOptimizedVector<size_type, std::pair<string_type, SExpr>, BuiltInSymbolsSize, list_type>;
  using function_ptr = SExpr (*)(cons_expr &, LexicalScope &, list_type);
  using Atom = std::variant<std::monostate, bool, int_type, float_type, string_type, identifier_type, UserTypes...>;

  struct FunctionPtr
  {
    enum struct Type { other, do_expr, let_expr, lambda_expr, define_expr };
    function_ptr ptr{ nullptr };
    Type type{ Type::other };

    [[nodiscard]] constexpr bool operator==(const FunctionPtr &other) const noexcept
    {
      // this pointer comparison is giving me a problem in constexpr context
      // it feels like a bug in GCC, but not sure
      if consteval {
        return type != Type::other && type == other.type;
      } else {
        return ptr == other.ptr;
      }
    }
  };

  struct SExpr
  {
    std::variant<Atom, list_type, literal_list_type, Closure, FunctionPtr, error_type> value;

    [[nodiscard]] constexpr bool operator==(const SExpr &) const noexcept = default;
  };

  static_assert(std::is_trivially_copyable_v<SExpr> && std::is_trivially_destructible_v<SExpr>,
    "cons_expr does not work well with non-trivial types");

  template<typename Result> [[nodiscard]] constexpr const Result *get_if(const SExpr *sexpr) const
  {
    if (sexpr == nullptr) { return nullptr; }

    if constexpr (std::is_same_v<Result, Atom> || std::is_same_v<Result, list_type>
                  || std::is_same_v<Result, literal_list_type> || std::is_same_v<Result, Closure>
                  || std::is_same_v<Result, FunctionPtr> || std::is_same_v<Result, error_type>) {
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
  SmallOptimizedVector<size_type, char_type, BuiltInStringsSize, string_type, std::basic_string_view<char_type>>
    strings{};
  SmallOptimizedVector<size_type, SExpr, BuiltInValuesSize, list_type> values{};


  struct Closure
  {
    list_type parameter_names;
    list_type statements;

    [[nodiscard]] constexpr bool operator==(const Closure &) const noexcept = default;

    [[nodiscard]] constexpr SExpr invoke(cons_expr &engine, LexicalScope &scope, list_type params) const
    {
      if (params.size != parameter_names.size) {
        return engine.make_error("Incorrect number of params for lambda", params);
      }

      // Closures contain all of their own scope
      LexicalScope new_scope;

      // set up params
      // technically I'm evaluating the params lazily while invoking the lambda, not before. Does it matter?
      for (const auto [name, parameter] : std::views::zip(engine.values[parameter_names], engine.values[params])) {
        new_scope.emplace_back(engine.get_if<identifier_type>(&name)->value, engine.eval(scope, parameter));
      }

      std::vector<SExpr> fixed_statements;
      for (const auto &statement : engine.values[statements]) {
        fixed_statements.push_back(engine.fix_identifiers(statement, {}, new_scope));
      }

      return engine.sequence(new_scope, engine.values.insert_or_find(fixed_statements));
    }
  };

  [[nodiscard]] constexpr std::pair<SExpr, Token<CharType>> parse(std::basic_string_view<char_type> input) noexcept
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
        if (const auto *list = std::get_if<list_type>(&parsed.value); list != nullptr) {
          retval.push_back(SExpr{ LiteralList{ *list } });
        } else {
          retval.push_back(make_error("parsed list", parsed));
        }
        token = remaining;
      } else if (token.parsed == ")") {
        break;
      } else if (token.parsed == "true") {
        retval.push_back(SExpr{ Atom{ true } });
      } else if (token.parsed == "false") {
        retval.push_back(SExpr{ Atom{ false } });
      } else {
        if (token.parsed.starts_with('"')) {
          // note that this doesn't remove escaped characters like it should yet
          const auto string = strings.insert_or_find(token.parsed.substr(1, token.parsed.size() - 2));
          // quoted string
          if (token.parsed.ends_with('"')) {
            retval.push_back(SExpr{ Atom(string) });
          } else {
            retval.push_back(make_error("terminated string", SExpr{ Atom(string) }));
          }
        } else if (auto [int_did_parse, int_value] = parse_int<int_type>(token.parsed); int_did_parse) {
          retval.push_back(SExpr{ Atom(int_value) });
        } else if (auto [float_did_parse, float_value] = parse_float<float_type>(token.parsed); float_did_parse) {
          retval.push_back(SExpr{ Atom(float_value) });
        } else {
          retval.push_back(SExpr{ Atom(Identifier{ strings.insert_or_find(token.parsed) }) });
        }
      }
      token = next_token(token.remaining);
    }
    return std::pair<SExpr, Token<CharType>>(SExpr{ values.insert_or_find(retval) }, token);
  }

  // Guaranteed to be initialized at compile time
  consteval cons_expr()
  {
    add("+", SExpr{ FunctionPtr{ binary_left_fold<adds>, FunctionPtr::Type::other } });
    add("*", SExpr{ FunctionPtr{ binary_left_fold<multiplies>, FunctionPtr::Type::other } });
    add("-", SExpr{ FunctionPtr{ binary_left_fold<subtracts>, FunctionPtr::Type::other } });
    add("/", SExpr{ FunctionPtr{ binary_left_fold<divides>, FunctionPtr::Type::other } });
    add("<=", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<lt_equal>, FunctionPtr::Type::other } });
    add(">=", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<gt_equal>, FunctionPtr::Type::other } });
    add("<", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<less_than>, FunctionPtr::Type::other } });
    add(">", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<greater_than>, FunctionPtr::Type::other } });
    add("and", SExpr{ FunctionPtr{ logical_and, FunctionPtr::Type::other } });
    add("or", SExpr{ FunctionPtr{ logical_or, FunctionPtr::Type::other } });
    add("if", SExpr{ FunctionPtr{ ifer, FunctionPtr::Type::other } });
    add("not", SExpr{ FunctionPtr{ make_evaluator<logical_not>(), FunctionPtr::Type::other } });
    add("==", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<equal>, FunctionPtr::Type::other } });
    add("!=", SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<not_equal>, FunctionPtr::Type::other } });
    add("for-each", SExpr{ FunctionPtr{ for_each, FunctionPtr::Type::other } });
    add("list", SExpr{ FunctionPtr{ list, FunctionPtr::Type::other } });
    add("lambda", SExpr{ FunctionPtr{ lambda, FunctionPtr::Type::lambda_expr } });
    add("do", SExpr{ FunctionPtr{ doer, FunctionPtr::Type::do_expr } });
    add("define", SExpr{ FunctionPtr{ definer, FunctionPtr::Type::define_expr } });
    add("let", SExpr{ FunctionPtr{ letter, FunctionPtr::Type::let_expr } });
    add("car", SExpr{ FunctionPtr{ car, FunctionPtr::Type::other } });
    add("cdr", SExpr{ FunctionPtr{ cdr, FunctionPtr::Type::other } });
    add("cons", SExpr{ FunctionPtr{ cons, FunctionPtr::Type::other } });
    add("append", SExpr{ FunctionPtr{ append, FunctionPtr::Type::other } });
    add("eval", SExpr{ FunctionPtr{ evaler, FunctionPtr::Type::other } });
    add("apply", SExpr{ FunctionPtr{ applier, FunctionPtr::Type::other } });
  }

  [[nodiscard]] constexpr SExpr sequence(LexicalScope &scope, list_type expressions)
  {
    auto result = SExpr{ Atom{ std::monostate{} } };
    std::ranges::for_each(values[expressions], [&, engine = this](auto expr) { result = engine->eval(scope, expr); });
    return result;
  }

  [[nodiscard]] constexpr SExpr invoke_function(LexicalScope &scope, const SExpr function, list_type params)
  {
    const SExpr resolved_function = eval(scope, function);

    if (auto *closure = get_if<Closure>(&resolved_function); closure != nullptr) [[unlikely]] {
      return closure->invoke(*this, scope, params);
    } else if (auto *func = get_if<FunctionPtr>(&resolved_function); func != nullptr) {
      return (func->ptr)(*this, scope, params);
    }

    return make_error("Function", function);
  }

  template<auto Func, typename Ret, typename... Param>
  [[nodiscard]] constexpr static function_ptr make_evaluator() noexcept
  {
    return function_ptr{ [](cons_expr &engine, LexicalScope &scope, list_type params) -> SExpr {
      if (params.size != sizeof...(Param)) { return engine.make_error("wrong param count for function", params); }

      auto impl = [&]<size_type... Idx>(std::integer_sequence<size_type, Idx...>) {
        if constexpr (std::is_same_v<void, Ret>) {
          std::invoke(Func, engine.eval_to<std::remove_cvref_t<Param>>(scope, engine.values[params[Idx]]).value()...);
          return SExpr{ Atom{ std::monostate{} } };
        } else {
          return SExpr{ std::invoke(
            Func, engine.eval_to<std::remove_cvref_t<Param>>(scope, engine.values[params[Idx]]).value()...) };
        }
      };

      return impl(std::make_integer_sequence<size_type, static_cast<size_type>(sizeof...(Param))>{});
    } };
  }

  template<auto Func, typename Ret, typename... Param>
  [[nodiscard]] constexpr static function_ptr make_evaluator(Ret (*)(Param...)) noexcept
  {
    return make_evaluator<Func, Ret, Param...>();
  }

  template<auto Func, typename Ret, typename Type, typename... Param>
  [[nodiscard]] constexpr static function_ptr make_evaluator(Ret (Type::*)(Param...) const) noexcept
  {
    return make_evaluator<Func, Ret, Type *, Param...>();
  }

  template<auto Func, typename Ret, typename Type, typename... Param>
  [[nodiscard]] constexpr static function_ptr make_evaluator(Ret (Type::*)(Param...)) noexcept
  {
    return make_evaluator<Func, Ret, Type *, Param...>();
  }

  template<auto Func> [[nodiscard]] constexpr static function_ptr make_evaluator() noexcept
  {
    return make_evaluator<Func>(Func);
  }

  template<auto Func> constexpr void add(std::basic_string_view<char_type> name)
  {
    global_scope.emplace_back(strings.insert_or_find(name), SExpr{ FunctionPtr{ make_evaluator<Func>() } });
  }

  constexpr void add(std::basic_string_view<char_type> name, SExpr value)
  {
    global_scope.emplace_back(strings.insert_or_find(name), std::move(value));
  }

  template<typename Value> constexpr void add(std::basic_string_view<char_type> name, Value &&value)
  {
    global_scope.emplace_back(strings.insert_or_find(name), SExpr{ Atom{ std::forward<Value>(value) } });
  }

  [[nodiscard]] constexpr SExpr eval(LexicalScope &scope, const SExpr expr)
  {
    if (const auto *indexed_list = get_if<list_type>(&expr); indexed_list != nullptr) {
      // if it's a non-empty list, then we need to evaluate it as a function
      if (!indexed_list->empty()) {
        return invoke_function(scope, values[(*indexed_list)[0]], indexed_list->sublist(1));
      }
    } else if (const auto *id = get_if<identifier_type>(&expr); id != nullptr) {
      for (const auto &[key, value] : scope | std::ranges::views::reverse) {
        if (key == id->value) { return value; }
      }

      const auto string = strings.view(id->value);

      // is quoted identifier, handle appropriately
      if (string.starts_with('\'')) { return SExpr{ Atom{ id->substr(1) } }; }

      return make_error("id not found", expr);
    }
    return expr;
  }

  template<typename Type>
  [[nodiscard]] constexpr std::expected<Type, SExpr> eval_to(LexicalScope &scope, const SExpr expr) noexcept
  {
    if constexpr (std::is_same_v<Type, SExpr>) {
      return expr;
    } else if constexpr (std::is_same_v<Type, literal_list_type> || std::is_same_v<Type, list_type>
                         || std::is_same_v<Type, Closure> || std::is_same_v<Type, FunctionPtr>
                         || std::is_same_v<Type, error_type>) {
      if (const auto *obj = std::get_if<Type>(&expr.value); obj != nullptr) { return *obj; }
    } else {
      if (const auto *atom = std::get_if<Atom>(&expr.value); atom != nullptr) {
        if (const auto *value = std::get_if<Type>(atom); value != nullptr) {
          return *value;
        } else if (!std::holds_alternative<identifier_type>(*atom)) {
          return std::unexpected(expr);
        }
      }
    }
    if (const auto *err = std::get_if<error_type>(&expr.value); err != nullptr) { return std::unexpected(expr); }
    return eval_to<Type>(scope, eval(scope, expr));
  }

  [[nodiscard]] static constexpr SExpr list(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    std::vector<SExpr> result;
    result.reserve(params.size);

    for (const auto &param : engine.values[params]) { result.push_back(engine.eval(scope, param)); }

    return SExpr{ LiteralList{ engine.values.insert_or_find(result) } };
  }

  constexpr std::vector<string_type> get_lambda_parameter_names(const SExpr &sexpr)
  {
    std::vector<string_type> retval;
    if (auto *parameter_list = get_if<list_type>(&sexpr); parameter_list != nullptr) {
      retval.reserve(parameter_list->size);
      for (const auto &expr : values[*parameter_list]) {
        if (auto *local_id = get_if<identifier_type>(&expr); local_id != nullptr) { retval.push_back(local_id->value); }
      }
    }
    return retval;
  }

  [[nodiscard]] static constexpr SExpr lambda(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    if (params.size < 2) { return engine.make_error("Incorrect number of params to lambda", params); }

    auto locals = engine.get_lambda_parameter_names(engine.values[params[0]]);

    // replace all references to captured values with constant copies
    std::vector<SExpr> fixed_statements;
    fixed_statements.reserve(params.size);
    for (const auto &statement : engine.values[params.sublist(1)]) {
      // all of current scope is const and capturable
      fixed_statements.push_back(engine.fix_identifiers(statement, locals, scope));
    }

    return SExpr{ Closure{
      std::get<list_type>(engine.values[params[0]].value), { engine.values.insert_or_find(fixed_statements) } } };
  }

  [[nodiscard]] constexpr SExpr fix_do_identifiers(list_type list,
    size_type first_index,
    std::span<const string_type> local_identifiers,
    const LexicalScope &local_constants)
  {
    std::vector<string_type> new_locals{ local_identifiers.begin(), local_identifiers.end() };

    std::vector<SExpr> new_params;

    // collect all locals
    const auto *params = get_if<list_type>(&values[first_index + 1]);
    if (params == nullptr) { return make_error("malformed do expression", list); }

    for (const auto &param : values[*params]) {
      auto param_list = get_if<list_type>(&param);
      if (param_list == nullptr || param_list->size < 2) { return make_error("malformed do expression", list); }

      auto id = get_if<identifier_type>(&values[(*param_list)[0]]);
      if (id == nullptr) { return make_error("malformed do expression", list); }
      new_locals.push_back(id->value);
    }

    for (const auto &param : values[*params]) {
      auto param_list = get_if<list_type>(&param);
      if (param_list == nullptr || param_list->size < 2) { return make_error("malformed do expression", list); }

      std::vector<SExpr> new_param;
      new_param.push_back(values[(*param_list)[0]]);
      new_param.push_back(fix_identifiers(values[(*param_list)[1]], local_identifiers, local_constants));
      // increment thingy (optional)
      if (param_list->size == 3) {
        new_param.push_back(fix_identifiers(values[(*param_list)[2]], new_locals, local_constants));
      }
      new_params.push_back(SExpr{ values.insert_or_find(new_param) });
    }

    std::vector<SExpr> new_do;
    // fixup pointer to "let" function
    new_do.push_back(fix_identifiers(values[first_index], new_locals, local_constants));

    // add parameter setup
    new_do.push_back(SExpr{ values.insert_or_find(new_params) });

    for (auto value : values[list.sublist(2)]) {
      new_do.push_back(fix_identifiers(value, new_locals, local_constants));
    }

    return SExpr{ values.insert_or_find(new_do) };
  }

  [[nodiscard]] constexpr SExpr fix_let_identifiers(list_type list,
    size_type first_index,
    std::span<const string_type> local_identifiers,
    const LexicalScope &local_constants)
  {
    std::vector<string_type> new_locals{ local_identifiers.begin(), local_identifiers.end() };

    std::vector<SExpr> new_params;

    const auto params = get_if<list_type>(&values[static_cast<size_type>(first_index + 1)]);
    if (params == nullptr) { return make_error("malformed let expression", list); }

    // collect all locals
    for (const auto &param : values[*params]) {
      auto param_list = get_if<list_type>(&param);
      if (param_list == nullptr || param_list->size < 2) { return make_error("malformed let expression", list); }

      auto id = get_if<identifier_type>(&values[(*param_list)[0]]);
      if (id == nullptr) { return make_error("malformed let expression", list); }
      new_locals.push_back(id->value);
    }

    for (const auto &param : values[*params]) {
      auto param_list = get_if<list_type>(&param);
      if (param_list == nullptr || param_list->size < 2) { return make_error("malformed do expression", list); }

      std::vector<SExpr> new_param;
      new_param.push_back(values[(*param_list)[0]]);
      new_param.push_back(fix_identifiers(values[(*param_list)[1]], local_identifiers, local_constants));
      // increment thingy (optional)
      if (param_list->size == 3) {
        new_param.push_back(fix_identifiers(values[(*param_list)[2]], new_locals, local_constants));
      }
      new_params.push_back(SExpr{ values.insert_or_find(new_param) });
    }

    std::vector<SExpr> new_let;
    // fixup pointer to "let" function
    new_let.push_back(fix_identifiers(values[first_index], new_locals, local_constants));
    new_let.push_back(SExpr{ values.insert_or_find(new_params) });

    for (size_type index = first_index + 2; index < list.size + list.start; ++index) {
      new_let.push_back(fix_identifiers(values[index], new_locals, local_constants));
    }

    return SExpr{ values.insert_or_find(new_let) };
  }

  [[nodiscard]] constexpr SExpr fix_define_identifiers(size_type first_index,
    std::span<const string_type> local_identifiers,
    const LexicalScope &local_constants)
  {
    std::vector<string_type> new_locals{ local_identifiers.begin(), local_identifiers.end() };

    const auto *id = get_if<identifier_type>(&values[static_cast<size_type>(first_index + 1)]);

    if (id == nullptr) { return make_error("malformed define expression", values[first_index + 1]); }
    new_locals.push_back(id->value);

    std::array<SExpr, 3> new_define{ fix_identifiers(values[first_index], local_identifiers, local_constants),
      values[first_index + 1],
      fix_identifiers(values[first_index + 2], new_locals, local_constants) };
    return SExpr{ values.insert_or_find(new_define) };
  }


  [[nodiscard]] constexpr SExpr fix_lambda_identifiers(list_type list,
    size_type first_index,
    std::span<const string_type> local_identifiers,
    const LexicalScope &local_constants)
  {
    std::vector<string_type> new_locals{ local_identifiers.begin(), local_identifiers.end() };
    auto lambda_locals = get_lambda_parameter_names(values[first_index + 1]);
    new_locals.insert(new_locals.end(), lambda_locals.begin(), lambda_locals.end());

    std::vector<SExpr> new_lambda;
    // fixup pointer to "lambda" function
    new_lambda.push_back(fix_identifiers(values[first_index], new_locals, local_constants));
    new_lambda.push_back(values[first_index + 1]);

    for (size_type index = first_index + 2; index < list.size + list.start; ++index) {
      new_lambda.push_back(fix_identifiers(values[index], new_locals, local_constants));
    }

    return SExpr{ values.insert_or_find(new_lambda) };
  }

  [[nodiscard]] constexpr SExpr
    fix_identifiers(SExpr input, std::span<const string_type> local_identifiers, const LexicalScope &local_constants)
  {
    if (auto *list = get_if<list_type>(&input); list != nullptr) {
      if (list->size != 0) {
        auto first_index = list->start;
        const auto &elem = values[first_index];
        std::basic_string_view<char_type> id = "";
        auto fp_type = FunctionPtr::Type::other;
        if (auto *id_atom = get_if<identifier_type>(&elem); id_atom != nullptr) { id = strings.view(id_atom->value); }
        if (auto *fp = get_if<FunctionPtr>(&elem); fp != nullptr) { fp_type = fp->type; }

        if (fp_type == FunctionPtr::Type::lambda_expr || id == "lambda") {
          return fix_lambda_identifiers(*list, first_index, local_identifiers, local_constants);
        } else if (fp_type == FunctionPtr::Type::let_expr || id == "let") {
          return fix_let_identifiers(*list, first_index, local_identifiers, local_constants);
        } else if (fp_type == FunctionPtr::Type::define_expr || id == "define") {
          return fix_define_identifiers(first_index, local_identifiers, local_constants);
        } else if (fp_type == FunctionPtr::Type::do_expr || id == "do") {
          return fix_do_identifiers(*list, first_index, local_identifiers, local_constants);
        }
      }
      std::vector<SExpr> result;
      result.reserve(list->size);
      for (const auto &value : values[*list]) {
        result.push_back(fix_identifiers(value, local_identifiers, local_constants));
      }
      return SExpr{ this->values.insert_or_find(result) };
    } else if (auto *id = get_if<identifier_type>(&input); id != nullptr) {
      for (const auto &local : local_identifiers) {
        if (local == id->value) {
          // do something smarter later, but abort for now because it's in the variable scope
          return input;
        }
      }

      // we're hoping it's a constant
      for (const auto &object : local_constants | std::ranges::views::reverse) {
        if (object.first == id->value) { return object.second; }
      }

      return input;
    }

    return input;
  }

  [[nodiscard]] constexpr SExpr make_error(std::basic_string_view<char_type> description, list_type context) noexcept
  {
    return SExpr{ Error{ strings.insert_or_find(description), context } };
  }

  template<std::same_as<SExpr>... Param>
  [[nodiscard]] constexpr SExpr make_error(std::basic_string_view<char_type> description, Param... context) noexcept
  {
    std::array<SExpr, sizeof...(Param)> params{ context... };
    return make_error(description, values.insert_or_find(params));
  }

  //
  // built-ins
  //
  [[nodiscard]] static constexpr SExpr letter(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.empty()) { return engine.make_error("(let ((var1 val1) ...))", params); }

    std::vector<std::pair<std::size_t, SExpr>> variables;

    auto variable_elems = engine.values[params[0]];
    auto *variable_list = engine.get_if<list_type>(&variable_elems);
    if (variable_list == nullptr) { return engine.make_error("((var1 val1) ...)", variable_elems); }

    auto new_scope = scope;

    for (const auto variable : engine.values[*variable_list]) {
      auto *variable_elements = engine.get_if<list_type>(&variable);
      if (variable_elements == nullptr || variable_elements->size != 2) {
        return engine.make_error("(var1 val1)", variable);
      }
      auto variable_id = engine.eval_to<identifier_type>(scope, engine.values[(*variable_elements)[0]]);
      if (!variable_id) { return engine.make_error("expected identifier", variable_id.error()); }
      new_scope.emplace_back(variable_id->value, engine.eval(scope, engine.values[(*variable_elements)[1]]));
    }

    // evaluate body
    return engine.sequence(new_scope, params.sublist(1));
  }


  [[nodiscard]] static constexpr SExpr doer(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size < 2) {
      return engine.make_error(
        "(do ((var1 val1 [iter_expr1]) ...) (terminate_condition [result...]) [body...])", params);
    }

    std::vector<std::pair<std::size_t, SExpr>> variables;
    std::vector<string_type> variable_names;

    auto *variable_list = engine.get_if<list_type>(&engine.values[params[0]]);

    if (variable_list == nullptr) {
      return engine.make_error("((var1 val1 [iter_expr1]) ...)", engine.values[params[0]]);
    }

    auto new_scope = scope;

    for (const auto &variable : engine.values[*variable_list]) {
      auto *variable_parts = engine.get_if<list_type>(&variable);
      if (variable_parts == nullptr || variable_parts->size < 2 || variable_parts->size > 3) {
        return engine.make_error("(var1 val1 [iter_expr1])", variable);
      }

      auto variable_parts_list = engine.values[*variable_parts];

      const auto index = new_scope.size();
      const auto id = engine.eval_to<identifier_type>(scope, variable_parts_list[0]);

      if (!id) { return engine.make_error("identifier", id.error()); }

      // initial value
      new_scope.emplace_back(id->value, engine.eval(scope, variable_parts_list[1]));

      // increment expression
      if (variable_parts->size == 3) { variables.emplace_back(index, variable_parts_list[2]); }
    }

    for (auto &variable : variables) {
      variable.second = engine.fix_identifiers(variable.second, variable_names, scope);
    }

    for (const auto &local : new_scope) { variable_names.push_back(local.first); }

    const auto terminator_param = engine.values[params[1]];
    const auto *terminator_list = engine.get_if<list_type>(&terminator_param);
    if (terminator_list == nullptr || terminator_list->size == 0) {
      return engine.make_error("(terminator_condition [result...])", terminator_param);
    }
    const auto terminators = engine.values[*terminator_list];

    // reuse the storage created for the new values on each iteration
    std::vector<std::pair<size_type, SExpr>> new_values;

    auto fixed_up_terminator = engine.fix_identifiers(terminators[0], variable_names, scope);

    // continue while terminator test is false

    bool end = false;
    while (!end) {
      const auto condition = engine.eval_to<bool>(new_scope, fixed_up_terminator);
      if (!condition) { return engine.make_error("boolean condition", condition.error()); }
      end = *condition;
      if (!end) {
        // evaluate body
        [[maybe_unused]] const auto result = engine.sequence(new_scope, params.sublist(2));

        // iterate loop variables
        for (const auto &[index, expr] : variables) { new_values.emplace_back(index, engine.eval(new_scope, expr)); }

        // update values
        for (auto &&[index, value] : new_values) { new_scope[index].second = std::move(value); }

        new_values.clear();
      }
    }

    // evaluate sequence of termination expressions
    return engine.sequence(new_scope, terminators.span.sublist(1));
  }

  [[nodiscard]] static constexpr SExpr append(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size != 2) { return engine.make_error("(append LiteralList LiteralList)", params); }

    auto first = engine.eval_to<literal_list_type>(scope, engine.values[params[0]]);
    auto second = engine.eval_to<literal_list_type>(scope, engine.values[params[1]]);

    if (!first || !second) { return engine.make_error("(append LiteralList LiteralList)", params); }

    std::vector<SExpr> result;

    for (const auto &value : engine.values[first->items]) { result.push_back(value); }
    for (const auto &value : engine.values[second->items]) { result.push_back(value); }

    return SExpr{ LiteralList{ engine.values.insert_or_find(result) } };
  }

  [[nodiscard]] static constexpr SExpr cons(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size != 2) { return engine.make_error("(cons Expr LiteralList)", params); }

    auto front = engine.eval(scope, engine.values[params[0]]);
    auto list = engine.eval_to<literal_list_type>(scope, engine.values[params[1]]);

    if (!list) { return engine.make_error("(cons Expr LiteralList)", params); }

    std::vector<SExpr> result;

    if (const auto *list_front = std::get_if<literal_list_type>(&front.value); list_front != nullptr) {
      result.push_back(SExpr{ list_front->items });
    } else {
      result.push_back(front);
    }

    for (const auto &value : engine.values[list->items]) { result.push_back(value); }

    return SExpr{ LiteralList{ engine.values.insert_or_find(result) } };
  }

  [[nodiscard]] static constexpr SExpr cdr(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size != 1) { return engine.make_error("(cdr Non-Empty-LiteralList)", params); }

    auto list = engine.eval_to<literal_list_type>(scope, engine.values[params[0]]);
    if (!list || list->items.size == 0) { return engine.make_error("(cdr Non-Empty-LiteralList)", params); }

    return SExpr{ list->sublist(1) };
  }

  [[nodiscard]] static constexpr SExpr car(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size != 1) { return engine.make_error("(car Non-Empty-LiteralList)", params); }

    auto list = engine.eval_to<literal_list_type>(scope, engine.values[params[0]]);
    if (!list || list->items.size == 0) { return engine.make_error("(car Non-Empty-LiteralList)", params); }

    return engine.values[list->front()];
  }

  [[nodiscard]] static constexpr SExpr applier(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size != 2) { return engine.make_error("(apply Function LiteralList)", params); }

    auto applied_params = engine.eval_to<literal_list_type>(scope, engine.values[params[1]]);
    if (!applied_params) { return engine.make_error("(apply Function LiteralList)", params); }

    return engine.invoke_function(scope, engine.values[params[0]], applied_params->items);
  }

  [[nodiscard]] static constexpr SExpr evaler(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size != 1) { return engine.make_error("(eval LiteralList)", params); }
    auto evaled_params = engine.eval_to<literal_list_type>(scope, engine.values[params[0]]);
    if (!evaled_params) { return engine.make_error("(eval LiteralList)", params); }

    return engine.eval(engine.global_scope, SExpr{ evaled_params->items });
  }

  [[nodiscard]] static constexpr SExpr ifer(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size != 3) { return engine.make_error("(if bool-cond then else)", params); }

    const auto condition = engine.eval_to<bool>(scope, engine.values[params[0]]);

    if (!condition) { return engine.make_error("boolean condition", condition.error()); }

    if (*condition) {
      return engine.eval(scope, engine.values[params[1]]);
    } else {
      return engine.eval(scope, engine.values[params[2]]);
    }
  }

  [[nodiscard]] static constexpr SExpr for_each(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    if (params.size != 2) { return engine.make_error("(for_each Function (param...))", params); }

    const auto list = engine.eval_to<literal_list_type>(scope, engine.values[params[1]]);
    if (!list) { return engine.make_error("(for_each Function (param...))", params); }

    const auto func = engine.values[params[0]];
    for (size_type index = 0; index < list->items.size; ++index) {
      [[maybe_unused]] const auto result = engine.invoke_function(scope, func, list->items.sublist(index, 1));
    }

    return SExpr{ Atom{ std::monostate{} } };
  }

  [[nodiscard]] static constexpr SExpr definer(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    if (params.size != 2) { return engine.make_error("(define Identifier Expression)", params); }
    const auto id = engine.eval_to<identifier_type>(scope, engine.values[params[0]]);
    if (!id) { return engine.make_error("(define Identifier Expression)", params); }
    scope.emplace_back(id->value, engine.fix_identifiers(engine.eval(scope, engine.values[params[1]]), {}, scope));
    return SExpr{ Atom{ std::monostate{} } };
  }

  // make a callable that captures the current engine by value
  template<typename Signature>
  [[nodiscard]] constexpr auto make_standalone_callable(std::basic_string_view<char_type> function) noexcept
    requires std::is_function_v<Signature>
  {
    auto impl = [this, function]<typename Ret, typename... Params>(Ret (*)(Params...)) {
      // this is fragile, we need to check parsing better

      return
        [engine = *this, callable = eval(global_scope, values[std::get<list_type>(parse(function).first.value)][0])](
          Params... params) mutable {
          std::array<SExpr, sizeof...(Params)> args{ SExpr{ Atom{ params } }... };
          return engine.template eval_to<Ret>(engine.global_scope,
            engine.invoke_function(engine.global_scope, callable, engine.values.insert_or_find(args)));
        };
    };

    return impl(std::add_pointer_t<Signature>{ nullptr });
  }

  // take a string_view and return a C++ function object
  // of unspecified type.
  template<typename Signature>
  [[nodiscard]] constexpr auto make_callable(std::basic_string_view<char_type> function) noexcept
    requires std::is_function_v<Signature>
  {
    auto impl = [this, function]<typename Ret, typename... Params>(Ret (*)(Params...)) {
      // this is fragile, we need to check parsing better

      return [callable = eval(global_scope, values[std::get<list_type>(parse(function).first.value)][0]), this](
               Params... params) {
        std::array<SExpr, sizeof...(Params)> args{ SExpr{ Atom{ params } }... };
        return eval_to<Ret>(global_scope, invoke_function(global_scope, callable, values.insert_or_find(args)));
      };
    };

    return impl(std::add_pointer_t<Signature>{ nullptr });
  }

  template<auto Op>
  [[nodiscard]] static constexpr SExpr
    binary_left_fold(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    auto fold = [&engine, &scope, params]<typename Param>(Param first) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        for (const auto &next : engine.values[params.sublist(1)]) {
          const auto result = engine.eval_to<Param>(scope, next);
          if (!result) { return engine.make_error("same types for operator", SExpr{ first }, result.error()); }
          first = Op(first, *result);
        }

        return SExpr{ Atom{ first } };
      } else {
        return engine.make_error("operator not supported for types", params);
      }
    };

    if (params.size > 1) {
      return std::visit(fold, std::get<Atom>(engine.eval(scope, engine.values[params[0]]).value));
    }

    return engine.make_error("operator requires at east two parameters", params);
  }

  [[nodiscard]] static constexpr SExpr logical_and(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    for (const auto &next : engine.values[params]) {
      const auto result = engine.eval_to<bool>(scope, next);
      if (!result) { return engine.make_error("parameter not boolean", result.error()); }
      if (!result.value()) { return SExpr{ Atom{ false } }; }
    }

    return SExpr{ Atom{ true } };
  }

  [[nodiscard]] static constexpr SExpr logical_or(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    for (const auto &next : engine.values[params]) {
      const auto result = engine.eval_to<bool>(scope, next);
      if (!result) { return engine.make_error("parameter not boolean", result.error()); }
      if (result.value()) { return SExpr{ Atom{ true } }; }
    }

    return SExpr{ Atom{ false } };
  }

  template<auto Op>
  [[nodiscard]] static constexpr SExpr
    binary_boolean_apply_pairwise(cons_expr &engine, LexicalScope &scope, list_type params) noexcept
  {
    auto sum = [&engine, &scope, params]<typename Param>(Param next) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        for (const auto &next_sexpr : engine.values[params.sublist(1)]) {
          const auto result = engine.eval_to<Param>(scope, next_sexpr);
          if (!result) { return engine.make_error("same types for operator", SExpr{ next }, result.error()); }
          const auto prev = std::exchange(next, *result);
          if (!Op(prev, next)) { return SExpr{ Atom{ false } }; }
        }

        return SExpr{ Atom{ true } };
      } else {
        return engine.make_error("supported types", params);
      }
    };

    if (params.size < 2) { return engine.make_error("at least 2 parameters", params); }
    auto first_param = engine.eval(scope, engine.values[params[0]]).value;

    // For working directly on "LiteralList" objects
    if (const auto *list = std::get_if<literal_list_type>(&first_param); list != nullptr) { return sum(*list); }

    if (const auto *atom = std::get_if<Atom>(&first_param); atom != nullptr) { return std::visit(sum, *atom); }

    return engine.make_error("supported types", params);
  }
};


}// namespace lefticus

#endif