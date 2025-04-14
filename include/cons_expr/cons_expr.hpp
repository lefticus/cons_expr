/*
MIT License

Copyright (c) 2023-2024 Jason Turner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef CONS_EXPR_HPP
#define CONS_EXPR_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

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
// * no exceptions or dynamic allocations

/// Notes
// This is a scheme-like language with a few caveats:
// * Once an object is captured or used, it's immutable
// * `==` `true` and `false` stray from `=` `#t` and `#f` of scheme
// * Pair types don't exist, only lists
// * Only indices and values are passed, for safety during resize of `values` object
// Triviality of types is critical to design and structure of this system
// Triviality lets us greatly simplify the copy/move/forward discussion
//
// Supported Scheme Features:
// * Core Data Types: numbers (int/float), strings, booleans, lists, symbols
// * List Operations: car, cdr, cons, append, list, quote
// * Control Structures: if, cond, begin
// * Variable Binding: let, define
// * Functions: lambda, apply
// * Higher-order Functions: for-each
// * Evaluation Control: eval
// * Basic Arithmetic: +, -, *, /
// * Comparisons: <, >, ==, !=, <=, >=
// * Boolean Logic: and, or, not

/// To do
// * We probably want some sort of "defragment" at some point
// * Add constant folding capability
// * Allow functions to be registered as "pure" so they can be folded!


namespace lefticus {

inline constexpr int cons_expr_version_major{ 0 };
inline constexpr int cons_expr_version_minor{ 0 };
inline constexpr int cons_expr_version_patch{ 1 };
inline constexpr int cons_expr_version_tweak{};


template<typename char_type> struct chars
{
  [[nodiscard]] static consteval std::string_view str(const char *input) noexcept
    requires std::is_same_v<char_type, char>
  {
    return input;
  }

  template<std::size_t Size>
  [[nodiscard]] static consteval auto str(const char (&input)[Size]) noexcept// NOLINT c-arrays
    requires(!std::is_same_v<char_type, char>)
  {
    struct Result
    {
      char_type data[Size];// NOLINT c-arrays
      constexpr operator std::basic_string_view<char_type>() { return { data, Size - 1 }; }// NOLINT implicit
    };

    Result result;
    std::copy(std::begin(input), std::end(input), std::begin(result.data));
    return result;
  }

  [[nodiscard]] static consteval char_type ch(const char input) noexcept { return input; }
};


template<std::unsigned_integral SizeType,
  typename Contained,
  SizeType SmallSize,
  typename KeyType,
  typename SpanType = std::span<const Contained>>
struct SmallVector
{
  using size_type = SizeType;
  using span_type = SpanType;

  std::array<Contained, SmallSize> small;
  size_type small_size_used = 0;
  bool error_state = false;

  static constexpr auto small_capacity = SmallSize;

  [[nodiscard]] constexpr Contained &operator[](size_type index) noexcept { return small[index]; }
  [[nodiscard]] constexpr const Contained &operator[](size_type index) const noexcept { return small[index]; }
  [[nodiscard]] constexpr auto size() const noexcept { return small_size_used; }
  [[nodiscard]] constexpr auto begin() const noexcept { return small.begin(); }
  [[nodiscard]] constexpr auto begin() noexcept { return small.begin(); }

  [[nodiscard]] constexpr auto end() const noexcept
  {
    return std::next(small.begin(), static_cast<std::ptrdiff_t>(small_size_used));
  }

  [[nodiscard]] constexpr auto end() noexcept
  {
    return std::next(small.begin(), static_cast<std::ptrdiff_t>(small_size_used));
  }

  [[nodiscard]] constexpr SpanType view(KeyType range) const noexcept
  {
    return SpanType{ std::span<const Contained>(small).subspan(range.start, range.size) };
  }
  [[nodiscard]] constexpr auto operator[](KeyType span) const noexcept { return view(span); }

  constexpr void push_back(auto &&param) noexcept { insert(param); }
  constexpr void emplace_back(auto &&...param) noexcept { insert(Contained{ param... }); }

  constexpr void resize(SizeType new_size) noexcept
  {
    small_size_used = std::min(new_size, SmallSize);
    if (new_size > SmallSize) { error_state = true; }
  }

  constexpr size_type insert(Contained obj) noexcept
  {
    if (small_size_used < small_capacity) {
      small[small_size_used] = std::move(obj);
      return small_size_used++;
    } else {
      error_state = true;
      return small_size_used;
    }
  }

  constexpr KeyType insert_or_find(SpanType values) noexcept
  {
    if (const auto small_found = std::search(begin(), end(), values.begin(), values.end()); small_found != end()) {
      return KeyType{ static_cast<size_type>(std::distance(begin(), small_found)),
        static_cast<size_type>(values.size()) };
    } else {
      return insert(values);
    }
  }

  constexpr KeyType insert(SpanType values) noexcept
  {
    size_type last = 0;
    for (const auto &value : values) { last = insert(value); }
    return KeyType{ static_cast<size_type>(last - values.size() + 1), static_cast<size_type>(values.size()) };
  }
};

template<typename T>
concept not_bool_or_ptr = !std::same_as<std::remove_cvref_t<T>, bool> && !std::is_pointer_v<std::remove_cvref_t<T>>;

// clang-format off
template<typename T> concept addable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs + rhs; };
template<typename T> concept multipliable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs *rhs; };
template<typename T> concept dividable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs / rhs; };
template<typename T> concept subtractable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs - rhs; };
template<typename T> concept lt_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs < rhs; };
template<typename T> concept gt_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs > rhs; };
template<typename T> concept lt_eq_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs <= rhs; };
template<typename T> concept gt_eq_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs >= rhs; };
template<typename T> concept eq_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs == rhs; };
template<typename T> concept not_eq_comparable = not_bool_or_ptr<T> && requires(T lhs, T rhs) { lhs != rhs; };
// clang-format on

inline constexpr auto adds = []<addable T>(const T &lhs, const T &rhs) { return lhs + rhs; };
inline constexpr auto multiplies = []<multipliable T>(const T &lhs, const T &rhs) { return lhs * rhs; };
inline constexpr auto divides = []<dividable T>(const T &lhs, const T &rhs) { return lhs / rhs; };
inline constexpr auto subtracts = []<subtractable T>(const T &lhs, const T &rhs) { return lhs - rhs; };
inline constexpr auto less_than = []<lt_comparable T>(const T &lhs, const T &rhs) { return lhs < rhs; };
inline constexpr auto greater_than = []<gt_comparable T>(const T &lhs, const T &rhs) { return lhs > rhs; };
inline constexpr auto lt_equal = []<lt_eq_comparable T>(const T &lhs, const T &rhs) { return lhs <= rhs; };
inline constexpr auto gt_equal = []<gt_eq_comparable T>(const T &lhs, const T &rhs) { return lhs >= rhs; };
inline constexpr auto equal = []<eq_comparable T>(const T &lhs, const T &rhs) -> bool { return lhs == rhs; };
inline constexpr auto not_equal = []<not_eq_comparable T>(const T &lhs, const T &rhs) -> bool { return lhs != rhs; };
inline constexpr bool logical_not(bool lhs) { return !lhs; }

template<typename CharType> struct Token
{
  using char_type = CharType;
  using string_view_type = std::basic_string_view<char_type>;
  string_view_type parsed;
  string_view_type remaining;
};

template<typename CharType>
Token(std::basic_string_view<CharType>, std::basic_string_view<CharType>) -> Token<CharType>;

template<typename T, typename CharType>
[[nodiscard]] constexpr std::pair<bool, T> parse_number(std::basic_string_view<CharType> input) noexcept
{
  static constexpr std::pair<bool, T> failure{ false, 0 };
  if (input == chars<CharType>::str("-")) { return failure; }

  enum struct State : std::uint8_t {
    Start,
    IntegerPart,
    FractionPart,
    ExponentPart,
    ExponentStart,
  };

  State state = State::Start;
  T value_sign = 1;
  long long value = 0LL;
  long long frac = 0LL;
  long long frac_exp = 0LL;
  long long exp_sign = 1LL;
  long long exp = 0LL;

  constexpr auto pow_10 = [](long long power) noexcept {
    auto result = T{ 1 };
    if (power > 0) {
      for (int iteration = 0; iteration < power; ++iteration) { result *= T{ 10 }; }
    } else if (power < 0) {
      for (int iteration = 0; iteration > power; --iteration) { result /= T{ 10 }; }
    }
    return result;
  };

  const auto parse_digit = [](auto &cur_value, auto ch) {
    if (ch >= chars<CharType>::ch('0') && ch <= chars<CharType>::ch('9')) {
      cur_value = cur_value * 10 + ch - chars<CharType>::ch('0');
      return true;
    } else {
      return false;
    }
  };

  for (const auto ch : input) {
    switch (state) {
    case State::Start:
      if (ch == chars<CharType>::ch('-')) {
        value_sign = -1;
      } else if (!parse_digit(value, ch)) {
        return failure;
      }
      state = State::IntegerPart;
      break;
    case State::IntegerPart:
      if (ch == chars<CharType>::ch('.')) {
        state = State::FractionPart;
      } else if (ch == chars<CharType>::ch('e') || ch == chars<CharType>::ch('E')) {
        state = State::ExponentStart;
      } else if (!parse_digit(value, ch)) {
        return failure;
      }
      break;
    case State::FractionPart:
      if (parse_digit(frac, ch)) {
        frac_exp--;
      } else if (ch == chars<CharType>::ch('e') || ch == chars<CharType>::ch('E')) {
        state = State::ExponentStart;
      } else {
        return failure;
      }
      break;
    case State::ExponentStart:
      if (ch == chars<CharType>::ch('-')) {
        exp_sign = -1;
      } else if (!parse_digit(exp, ch)) {
        return failure;
      }
      state = State::ExponentPart;
      break;
    case State::ExponentPart:
      if (!parse_digit(exp, ch)) { return failure; }
    }
  }

  if constexpr (std::is_integral_v<T>) {
    if (state != State::IntegerPart) { return failure; }
    return { true, value_sign * static_cast<T>(value) };
  } else {
    if (state == State::Start || state == State::ExponentStart) { return { false, 0 }; }

    return { true,
      (static_cast<T>(value_sign) * (static_cast<T>(value) + static_cast<T>(frac) * pow_10(frac_exp))
        * pow_10(exp_sign * exp)) };
  }
}


template<typename CharType> [[nodiscard]] constexpr Token<CharType> next_token(std::basic_string_view<CharType> input)
{
  using chars = lefticus::chars<CharType>;

  constexpr auto is_eol = [](auto ch) { return ch == chars::ch('\n') || ch == chars::ch('\r'); };
  constexpr auto is_whitespace = [=](auto ch) { return ch == chars::ch(' ') || ch == chars::ch('\t') || is_eol(ch); };

  constexpr auto consume = [=](auto ws_input, auto predicate) {
    auto begin = ws_input.begin();
    while (begin != ws_input.end() && predicate(*begin)) { ++begin; }
    return std::basic_string_view<CharType>{ begin, ws_input.end() };
  };

  constexpr auto make_token = [=](std::basic_string_view<CharType> token_input, std::size_t size) {
    return Token{ token_input.substr(0, size), consume(token_input.substr(size), is_whitespace) };
  };

  input = consume(input, is_whitespace);

  // comment
  if (input.starts_with(chars::ch(';'))) {
    input = consume(input, [=](auto ch) { return not is_eol(ch); });
    input = consume(input, is_whitespace);
  }

  // list
  if (input.starts_with(chars::ch('(')) || input.starts_with(chars::ch(')'))) { return make_token(input, 1); }

  // literal list
  if (input.starts_with(chars::str("'("))) { return make_token(input, 2); }

  // quoted string
  if (input.starts_with(chars::ch('"'))) {
    bool in_escape = false;
    auto location = std::next(input.begin());
    while (location != input.end()) {
      if (*location == chars::ch('\\')) {
        in_escape = true;
      } else if (*location == chars::ch('"') && !in_escape) {
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
  const auto value =
    consume(input, [=](auto ch) { return !is_whitespace(ch) && ch != chars::ch(')') && ch != chars::ch('('); });

  return make_token(input, static_cast<std::size_t>(std::distance(input.begin(), value.begin())));
}

// Tagged string base template
template<std::unsigned_integral SizeType, typename Tag> struct TaggedIndexedString
{
  using size_type = SizeType;
  size_type start{ 0 };
  size_type size{ 0 };
  [[nodiscard]] constexpr bool operator==(const TaggedIndexedString &) const noexcept = default;
  [[nodiscard]] constexpr auto front() const noexcept { return start; }
  [[nodiscard]] constexpr auto substr(const size_type from) const noexcept
  {
    return TaggedIndexedString{ static_cast<size_type>(start + from), static_cast<size_type>(size - from) };
  }
};

// Type aliases for the concrete string types
template<std::unsigned_integral SizeType> using IndexedString = TaggedIndexedString<SizeType, struct StringTag>;
template<std::unsigned_integral SizeType> using Identifier = TaggedIndexedString<SizeType, struct IdentifierTag>;
template<std::unsigned_integral SizeType> using Symbol = TaggedIndexedString<SizeType, struct SymbolTag>;

template<std::unsigned_integral SizeType, typename Tag>
[[nodiscard]] constexpr auto to_string(const TaggedIndexedString<SizeType, Tag> input)
{
  return IndexedString<SizeType>{ input.start, input.size };
}
template<std::unsigned_integral SizeType, typename Tag>
[[nodiscard]] constexpr auto to_identifier(const TaggedIndexedString<SizeType, Tag> input)
{
  return Identifier<SizeType>{ input.start, input.size };
}
template<std::unsigned_integral SizeType, typename Tag>
[[nodiscard]] constexpr auto to_symbol(const TaggedIndexedString<SizeType, Tag> input)
{
  return Symbol<SizeType>{ input.start, input.size };
}

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
  [[nodiscard]] constexpr auto sublist(const size_type from) const noexcept
  {
    return IndexedList{ static_cast<size_type>(start + from), static_cast<size_type>(size - from) };
  }

  [[nodiscard]] constexpr auto sublist(const size_type from, const size_type distance) const noexcept
  {
    return IndexedList{ static_cast<size_type>(start + from), distance };
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


template<std::unsigned_integral SizeType> struct Error
{
  using size_type = SizeType;
  IndexedString<size_type> expected;
  IndexedList<size_type> got;
  [[nodiscard]] constexpr bool operator==(const Error &) const noexcept = default;
};

template<std::unsigned_integral SizeType> Error(IndexedString<SizeType>, IndexedList<SizeType>) -> Error<SizeType>;


template<std::unsigned_integral SizeType = std::uint16_t,
  typename CharType = char,
  std::signed_integral IntegralType = int,
  std::floating_point FloatType = double,
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
  using string_view_type = std::basic_string_view<char_type>;
  using identifier_type = Identifier<size_type>;
  using symbol_type = Symbol<size_type>;
  using list_type = IndexedList<size_type>;
  using literal_list_type = LiteralList<size_type>;
  using error_type = Error<size_type>;

  template<typename Contained> using stack_vector = SmallVector<size_type, Contained, 32, IndexedList<size_type>>;

  struct SExpr;
  struct Closure;

  template<std::size_t Size>
  [[nodiscard]] static consteval auto str(char const (&input)[Size]) noexcept// NOLINT(modernize-avoid-c-arrays)
  {
    return chars<char_type>::str(input);
  }

  template<typename Type>
  [[nodiscard]] static constexpr bool visit_helper(SExpr &result, auto Func, auto &variant) noexcept
  {
    if (auto *value = std::get_if<Type>(&variant); value != nullptr) {
      result = Func(*value);
      return true;
    }
    return false;
  }

  template<typename... Type> static constexpr SExpr visit(auto visitor, const std::variant<Type...> &value) noexcept
  {
    SExpr result{};
    // || will make this short circuit and stop on first matching function
    ((visit_helper<Type>(result, visitor, value) || ...));
    return result;
  }

  using LexicalScope = SmallVector<size_type, std::pair<string_type, SExpr>, BuiltInSymbolsSize, list_type>;
  using function_ptr = SExpr (*)(cons_expr &, LexicalScope &, list_type);
  using Atom =
    std::variant<std::monostate, bool, int_type, float_type, string_type, identifier_type, symbol_type, UserTypes...>;

  struct FunctionPtr
  {
    enum struct Type : std::uint8_t { other, do_expr, let_expr, lambda_expr, define_expr };
    function_ptr ptr{ nullptr };
    Type type{ Type::other };

    [[nodiscard]] constexpr bool operator==(const FunctionPtr &other) const
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

  template<typename T>
  inline static constexpr bool is_sexpr_type_v =
    std::is_same_v<T, literal_list_type> || std::is_same_v<T, list_type> || std::is_same_v<T, Closure>
    || std::is_same_v<T, FunctionPtr> || std::is_same_v<T, error_type> || std::is_same_v<T, Atom>;


  struct SExpr
  {
    std::variant<Atom, list_type, literal_list_type, Closure, FunctionPtr, error_type> value;

    [[nodiscard]] constexpr bool operator==(const SExpr &) const noexcept = default;
  };

  static_assert(std::is_trivially_copyable_v<SExpr> && std::is_trivially_destructible_v<SExpr>,
    "cons_expr does not work with non-trivial types");

  template<typename Result> [[nodiscard]] static constexpr const Result *get_if(const SExpr *sexpr) noexcept
  {
    if (sexpr == nullptr) { return nullptr; }

    if constexpr (is_sexpr_type_v<Result>) {
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
  SmallVector<size_type, char_type, BuiltInStringsSize, string_type, string_view_type> strings{};
  SmallVector<size_type, SExpr, BuiltInValuesSize, list_type> values{};

  SmallVector<size_type, SExpr, 32, IndexedList<size_type>> object_scratch{};
  SmallVector<size_type, std::pair<size_type, SExpr>, 32, IndexedList<size_type>> variables_scratch{};
  SmallVector<size_type, string_type, 32, IndexedList<size_type>> string_scratch{};

  template<typename ScratchTo> struct Scratch
  {
    constexpr explicit Scratch(ScratchTo &t_data) noexcept : data(&t_data) {}
    constexpr explicit Scratch(ScratchTo &t_data, auto initial_values) noexcept : Scratch(t_data)
    {
      for (const auto &obj : initial_values) { push_back(obj); }
    }
    Scratch(const Scratch &) = delete;
    constexpr Scratch(Scratch &&other) noexcept
      : data{ std::exchange(other.data, nullptr) }, initial_size{ other.initial_size },
        current_size{ other.current_size }
    {}
    auto &operator=(Scratch &&) = delete;
    auto &operator=(const Scratch &&) = delete;
    constexpr ~Scratch() noexcept
    {
      if (data != nullptr) { data->resize(initial_size); }
    }
    [[nodiscard]] constexpr auto begin() const noexcept { return std::next(data->begin(), initial_size); }
    [[nodiscard]] constexpr auto end() const noexcept { return std::next(data->begin(), current_size); }
    constexpr void emplace_back(auto &&...param) noexcept
    {
      assert(data->size() == current_size);
      data->emplace_back(param...);
      current_size = data->size();
    }
    constexpr void push_back(auto obj) noexcept
    {
      assert(data->size() == current_size);
      data->push_back(obj);
      current_size = data->size();
    }

  private:
    ScratchTo *data;
    size_type initial_size = data->size();
    size_type current_size = data->size();
  };


  struct Closure
  {
    list_type parameter_names;
    list_type statements;

    [[nodiscard]] constexpr bool operator==(const Closure &) const = default;

    [[nodiscard]] constexpr SExpr invoke(cons_expr &engine, LexicalScope &scope, list_type params) const
    {
      if (params.size != parameter_names.size) {
        return engine.make_error(str("Incorrect number of params for lambda"), params);
      }

      // Closures contain all of their own scope
      LexicalScope param_scope = scope;

      // overwrite scope with the things we know we need params to be named

      // set up params
      // technically I'm evaluating the params lazily while invoking the lambda, not before. Does it matter?
      for (const auto [name, parameter] : std::views::zip(engine.values[parameter_names], engine.values[params])) {
        param_scope.emplace_back(to_string(*engine.get_if<identifier_type>(&name)), engine.eval(scope, parameter));
      }

      // TODO set up tail call elimination for last element of the sequence being evaluated?
      return engine.sequence(param_scope, statements);
    }
  };

  // Process escape sequences in a string literal
  [[nodiscard]] constexpr string_type process_string_escapes(string_view_type str) 
  {
    // Create a temporary buffer for the processed string
    // Using 64 as a reasonable initial size for most string literals
    SmallVector<size_type, CharType, 64, CharType, string_view_type> temp_buffer{};
    
    bool in_escape = false;
    for (const auto& ch : str) {
      if (in_escape) {
        // Handle the escape sequence
        switch (ch) {
          case '"':  temp_buffer.push_back('"'); break;  // Escaped quote
          case '\\': temp_buffer.push_back('\\'); break; // Escaped backslash
          case 'n':  temp_buffer.push_back('\n'); break; // Newline
          case 't':  temp_buffer.push_back('\t'); break; // Tab
          case 'r':  temp_buffer.push_back('\r'); break; // Carriage return
          case 'f':  temp_buffer.push_back('\f'); break; // Form feed
          case 'b':  temp_buffer.push_back('\b'); break; // Backspace
          default:   temp_buffer.push_back(ch); break;   // Other characters as-is
        }
        in_escape = false;
      } else if (ch == '\\') {
        in_escape = true;
      } else {
        temp_buffer.push_back(ch);
      }
    }
    
    // Now use insert_or_find to deduplicate the processed string
    string_view_type processed_view(temp_buffer.small.data(), temp_buffer.size());
    return strings.insert_or_find(processed_view);
  }

  [[nodiscard]] constexpr std::pair<SExpr, Token<CharType>> parse(string_view_type input)
  {
    Scratch retval{ object_scratch };

    auto token = next_token(input);

    while (!token.parsed.empty()) {
      if (token.parsed == str("(")) {
        auto [parsed, remaining] = parse(token.remaining);
        retval.push_back(parsed);
        token = remaining;
      } else if (token.parsed == str("'(")) {
        auto [parsed, remaining] = parse(token.remaining);
        if (const auto *list = std::get_if<list_type>(&parsed.value); list != nullptr) {
          retval.push_back(SExpr{ LiteralList{ *list } });
        } else {
          retval.push_back(make_error(str("parsed list"), parsed));
        }
        token = remaining;
      } else if (token.parsed == str(")")) {
        break;
      } else if (token.parsed == str("true")) {
        retval.push_back(SExpr{ Atom{ true } });
      } else if (token.parsed == str("false")) {
        retval.push_back(SExpr{ Atom{ false } });
      } else {
        if (token.parsed.starts_with('"')) {
          // Process quoted string with proper escape character handling
          if (token.parsed.ends_with('"')) {
            // Extract the string content (remove surrounding quotes)
            string_view_type raw_content = token.parsed.substr(1, token.parsed.size() - 2);
            
            // Process escape sequences and get the deduplicated string
            const auto string = process_string_escapes(raw_content);
            retval.push_back(SExpr{ Atom(string) });
          } else {
            retval.push_back(make_error(str("terminated string"), SExpr{ Atom(strings.insert_or_find(token.parsed)) }));
          }
        } else if (auto [int_did_parse, int_value] = parse_number<int_type>(token.parsed); int_did_parse) {
          retval.push_back(SExpr{ Atom(int_value) });
        } else if (auto [float_did_parse, float_value] = parse_number<float_type>(token.parsed); float_did_parse) {
          retval.push_back(SExpr{ Atom(float_value) });
        } else if (token.parsed.starts_with('\'')) {
          retval.push_back(SExpr{ Atom(to_symbol(strings.insert_or_find(token.parsed.substr(1)))) });
        } else {
          retval.push_back(SExpr{ Atom(to_identifier(strings.insert_or_find(token.parsed))) });
        }
      }
      token = next_token(token.remaining);
    }
    return std::pair<SExpr, Token<CharType>>(SExpr{ values.insert_or_find(retval) }, token);
  }

  // Guaranteed to be initialized at compile time
  consteval cons_expr() noexcept
  {
    add(str("+"), SExpr{ FunctionPtr{ binary_left_fold<adds>, FunctionPtr::Type::other } });
    add(str("*"), SExpr{ FunctionPtr{ binary_left_fold<multiplies>, FunctionPtr::Type::other } });
    add(str("-"), SExpr{ FunctionPtr{ binary_left_fold<subtracts>, FunctionPtr::Type::other } });
    add(str("/"), SExpr{ FunctionPtr{ binary_left_fold<divides>, FunctionPtr::Type::other } });
    add(str("<="), SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<lt_equal>, FunctionPtr::Type::other } });
    add(str(">="), SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<gt_equal>, FunctionPtr::Type::other } });
    add(str("<"), SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<less_than>, FunctionPtr::Type::other } });
    add(str(">"), SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<greater_than>, FunctionPtr::Type::other } });
    add(str("and"), SExpr{ FunctionPtr{ logical_and, FunctionPtr::Type::other } });
    add(str("or"), SExpr{ FunctionPtr{ logical_or, FunctionPtr::Type::other } });
    add(str("if"), SExpr{ FunctionPtr{ ifer, FunctionPtr::Type::other } });
    add(str("not"), SExpr{ FunctionPtr{ make_evaluator<logical_not>(), FunctionPtr::Type::other } });
    add(str("=="), SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<equal>, FunctionPtr::Type::other } });
    add(str("!="), SExpr{ FunctionPtr{ binary_boolean_apply_pairwise<not_equal>, FunctionPtr::Type::other } });
    add(str("for-each"), SExpr{ FunctionPtr{ for_each, FunctionPtr::Type::other } });
    add(str("list"), SExpr{ FunctionPtr{ list, FunctionPtr::Type::other } });
    add(str("lambda"), SExpr{ FunctionPtr{ lambda, FunctionPtr::Type::lambda_expr } });
    add(str("define"), SExpr{ FunctionPtr{ definer, FunctionPtr::Type::define_expr } });
    add(str("let"), SExpr{ FunctionPtr{ letter, FunctionPtr::Type::let_expr } });
    add(str("car"), SExpr{ FunctionPtr{ car, FunctionPtr::Type::other } });
    add(str("cdr"), SExpr{ FunctionPtr{ cdr, FunctionPtr::Type::other } });
    add(str("cons"), SExpr{ FunctionPtr{ cons, FunctionPtr::Type::other } });
    add(str("append"), SExpr{ FunctionPtr{ append, FunctionPtr::Type::other } });
    add(str("eval"), SExpr{ FunctionPtr{ evaler, FunctionPtr::Type::other } });
    add(str("apply"), SExpr{ FunctionPtr{ applier, FunctionPtr::Type::other } });
    add(str("quote"), SExpr{ FunctionPtr{ quoter, FunctionPtr::Type::other } });
    add(str("begin"), SExpr{ FunctionPtr{ begin, FunctionPtr::Type::other } });
    add(str("cond"), SExpr{ FunctionPtr{ cond, FunctionPtr::Type::other } });
    add(str("error?"), SExpr{ FunctionPtr{ error_p, FunctionPtr::Type::other } });
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

    if (auto *closure = get_if<Closure>(&resolved_function); closure != nullptr) {
      return closure->invoke(*this, scope, params);
    } else if (auto *func = get_if<FunctionPtr>(&resolved_function); func != nullptr) {
      return (func->ptr)(*this, scope, params);
    }

    return make_error(str("Function"), function);
  }


  template<auto Func, typename Ret, typename... Param> [[nodiscard]] constexpr static function_ptr make_evaluator()
  {
    return function_ptr{ [](cons_expr &engine, LexicalScope &scope, list_type params) -> SExpr {
      if (params.size != sizeof...(Param)) { return engine.make_error(str("wrong param count for function"), params); }

      auto impl = [&]<size_type... Idx>(std::integer_sequence<size_type, Idx...>) {
        std::tuple evaled_params{ engine.eval_to<std::remove_cvref_t<Param>>(scope, engine.values[params[Idx]])... };

        SExpr error;

        // See if any parameter evaluations errored
        const bool errored = !([&] {
          if (std::get<Idx>(evaled_params).has_value()) {
            return true;
          } else {
            error = std::get<Idx>(evaled_params).error();
            return false;
          }
        }() && ...);

        if (errored) { return engine.make_error(str("parameter type mismatch"), error); }

        // types have already been verified, so I can just `*` the expected safely to avoid exception checks
        if constexpr (std::is_same_v<void, Ret>) {
          std::invoke(Func, *std::get<Idx>(evaled_params)...);
          return SExpr{ Atom{ std::monostate{} } };
        } else {
          return SExpr{ std::invoke(Func, *std::get<Idx>(evaled_params)...) };
        }
      };

      return impl(std::make_integer_sequence<size_type, static_cast<size_type>(sizeof...(Param))>{});
    } };
  }

  template<auto Func, typename Ret, typename... Param>
  [[maybe_unused]] [[nodiscard]] constexpr static function_ptr make_evaluator(Ret (*)(Param...))
  {
    return make_evaluator<Func, Ret, Param...>();
  }

  template<auto Func, typename Ret, typename Type, typename... Param>
  [[nodiscard]] constexpr static function_ptr make_evaluator(Ret (Type::*)(Param...) const)
  {
    return make_evaluator<Func, Ret, Type *, Param...>();
  }

  template<auto Func, typename Ret, typename Type, typename... Param>
  [[maybe_unused]] [[nodiscard]] constexpr static function_ptr make_evaluator(Ret (Type::*)(Param...))
  {
    return make_evaluator<Func, Ret, Type *, Param...>();
  }

  template<auto Func> [[nodiscard]] constexpr static function_ptr make_evaluator()
  {
    return make_evaluator<Func>(Func);
  }

  constexpr auto add(string_view_type name, SExpr value)
  {
    return global_scope.emplace_back(strings.insert_or_find(name), value);
  }

  template<auto Func> constexpr auto add(string_view_type name)
  {
    return add(name, SExpr{ FunctionPtr{ make_evaluator<Func>() } });
  }

  template<typename Value> constexpr auto add(string_view_type name, Value value)
  {
    return add(name, SExpr{ Atom{ value } });
  }

  [[nodiscard]] constexpr SExpr eval(LexicalScope &scope, const SExpr expr)
  {
    if (const auto *indexed_list = get_if<list_type>(&expr); indexed_list != nullptr) {
      // if it's a non-empty list, then we need to evaluate it as a function
      if (!indexed_list->empty()) {
        return invoke_function(scope, values[(*indexed_list)[0]], indexed_list->sublist(1));
      }
    } else if (const auto *id = get_if<identifier_type>(&expr); id != nullptr) {
      for (const auto &[key, value] : scope | std::views::reverse) {
        if (key == to_string(*id)) { return value; }
      }

      return make_error(str("id not found"), expr);
    }
    return expr;
  }

  template<typename Type>
  [[nodiscard]] constexpr std::expected<Type, SExpr> eval_to(LexicalScope &scope, const SExpr expr)
  {
    if constexpr (std::is_same_v<Type, SExpr>) {
      return eval(scope, expr);
    } else if constexpr (is_sexpr_type_v<Type>) {
      if (const auto *obj = std::get_if<Type>(&expr.value); obj != nullptr) { return *obj; }
    } else {
      if (const auto *atom = std::get_if<Atom>(&expr.value); atom != nullptr) {
        if constexpr (std::is_same_v<Type, string_view_type>) {
          if (const auto *value = std::get_if<string_type>(atom); value != nullptr) { return strings.view(*value); }
        } else {
          if (const auto *value = std::get_if<Type>(atom); value != nullptr) {
            return *value;
          } else if (!std::holds_alternative<identifier_type>(*atom)) {
            return std::unexpected(expr);
          }
        }
      }
    }

    // if things aren't changing, then we abort, because it's not going to happen
    // this should be cleaned up somehow to avoid move
    if (auto next = eval(scope, expr); next == expr) {
      return std::unexpected(expr);
    } else {
      return eval_to<Type>(scope, std::move(next));
    }
  }

  // (list 1 2 3) -> '(1 2 3)
  // (list (+ 1 2) (+ 3 4)) -> '(3 7)
  [[nodiscard]] static constexpr SExpr list(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    // Evaluate each parameter and add it to a new list
    Scratch result{ engine.object_scratch };
    for (const auto &param : engine.values[params]) { result.push_back(engine.eval(scope, param)); }
    return SExpr{ LiteralList{ engine.values.insert_or_find(result) } };
  }

  constexpr auto get_lambda_parameter_names(const SExpr &sexpr)
  {
    Scratch retval{ string_scratch };
    if (auto *parameter_list = get_if<list_type>(&sexpr); parameter_list != nullptr) {
      for (const auto &expr : values[*parameter_list]) {
        if (auto *local_id = get_if<identifier_type>(&expr); local_id != nullptr) {
          retval.push_back(to_string(*local_id));
        }
      }
    }
    return retval;
  }

  // (lambda (x y) (+ x y)) -> #<closure>
  // ((lambda (x) (* x x)) 5) -> 25
  [[nodiscard]] static constexpr SExpr lambda(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    if (params.size < 2) { return engine.make_error(str("(lambda ([params...]) [statement...])"), params); }

    // Extract parameter names from first argument
    auto locals = engine.get_lambda_parameter_names(engine.values[params[0]]);

    // Replace all references to captured values with constant copies
    // This is how we create the closure object - by fixing all identifiers
    Scratch fixed_statements{ engine.object_scratch };

    for (const auto &statement : engine.values[params.sublist(1)]) {
      // All of current scope is const and capturable
      fixed_statements.push_back(engine.fix_identifiers(statement, locals, scope));
    }

    // Create the closure with parameter list and fixed statements
    const auto list = engine.get_if<list_type>(&engine.values[params[0]]);
    if (list) { return SExpr{ Closure{ *list, { engine.values.insert_or_find(fixed_statements) } } }; }

    return engine.make_error(str("(lambda ([params...]) [statement...])"), params);
  }

  [[nodiscard]] constexpr std::expected<list_type, SExpr> get_list(SExpr expr,
    string_view_type message,
    size_type min = 0,
    size_type max = std::numeric_limits<size_type>::max())
  {
    const auto *items = std::get_if<list_type>(&expr.value);
    if (items == nullptr || items->size < min || items->size > max) {
      return std::unexpected(make_error(message, expr));
    }

    return *items;
  }

  [[nodiscard]] constexpr std::expected<typename decltype(values)::span_type, SExpr> get_list_range(SExpr expr,
    string_view_type message,
    size_type min = 0,
    size_type max = std::numeric_limits<size_type>::max())
  {
    auto list = get_list(expr, message, min, max);
    if (!list) { return std::unexpected(list.error()); }
    return values[*list];
  }


  [[nodiscard]] constexpr SExpr fix_let_identifiers(list_type list,
    size_type first_index,
    std::span<const string_type> local_identifiers,
    const LexicalScope &local_constants)
  {
    Scratch new_locals{ string_scratch, local_identifiers };

    Scratch new_params{ object_scratch };

    const auto params =
      get_list_range(values[static_cast<size_type>(first_index + 1)], str("malformed let expression"));
    if (!params) { return params.error(); }

    for (const auto &param : *params) {
      const auto param_list = get_list(param, str("malformed let expression"), 2, 2);
      if (!param_list) { return param_list.error(); }

      auto *id = get_if<identifier_type>(&values[(*param_list)[0]]);
      if (id == nullptr) { return make_error(str("malformed let expression"), list); }
      new_locals.push_back(to_string(*id));

      std::array new_param{ values[(*param_list)[0]],
        fix_identifiers(values[(*param_list)[1]], local_identifiers, local_constants) };

      new_params.push_back(SExpr{ values.insert_or_find(new_param) });
    }

    Scratch new_let{ object_scratch };

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
    Scratch new_locals{ string_scratch, local_identifiers };

    const auto *id = get_if<identifier_type>(&values[static_cast<size_type>(first_index + 1)]);

    if (id == nullptr) { return make_error(str("malformed define expression"), values[first_index + 1]); }
    new_locals.push_back(to_string(*id));

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
    auto lambda_locals = get_lambda_parameter_names(values[first_index + 1]);
    Scratch new_locals{ string_scratch, local_identifiers };
    for (const auto &value : lambda_locals) { new_locals.push_back(value); }

    Scratch new_lambda{ object_scratch,
      std::array{ fix_identifiers(values[first_index], new_locals, local_constants), values[first_index + 1] } };

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
        string_view_type id;
        auto fp_type = FunctionPtr::Type::other;
        if (auto *id_atom = get_if<identifier_type>(&elem); id_atom != nullptr) {
          id = strings.view(to_string(*id_atom));
        }
        if (auto *fp = get_if<FunctionPtr>(&elem); fp != nullptr) { fp_type = fp->type; }

        if (fp_type == FunctionPtr::Type::lambda_expr || id == str("lambda")) {
          return fix_lambda_identifiers(*list, first_index, local_identifiers, local_constants);
        } else if (fp_type == FunctionPtr::Type::let_expr || id == str("let")) {
          return fix_let_identifiers(*list, first_index, local_identifiers, local_constants);
        } else if (fp_type == FunctionPtr::Type::define_expr || id == str("define")) {
          return fix_define_identifiers(first_index, local_identifiers, local_constants);
        }
      }


      Scratch result{ object_scratch };
      for (const auto &value : values[*list]) {
        result.push_back(fix_identifiers(value, local_identifiers, local_constants));
      }
      return SExpr{ this->values.insert_or_find(result) };
    } else if (auto *id = get_if<identifier_type>(&input); id != nullptr) {
      for (const auto &local : local_identifiers | std::views::reverse) {
        // do something smarter later, but abort for now because it's in the variable scope
        if (local == to_string(*id)) { return input; }
      }

      for (const auto &object : local_constants | std::views::reverse) {
        if (object.first == to_string(*id)) { return object.second; }
      }

      return input;
    }

    return input;
  }

  [[nodiscard]] constexpr SExpr make_error(string_view_type description, list_type context)
  {
    return SExpr{ Error{ strings.insert_or_find(description), context } };
  }

  [[nodiscard]] constexpr SExpr make_error(string_view_type description, auto... value)
  {
    return make_error(description, values.insert_or_find(std::array{ SExpr{ value }... }));
  }


  //
  // built-ins
  //
  [[nodiscard]] static constexpr SExpr letter(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    if (params.empty()) { return engine.make_error(str("(let ((var1 val1) ...) [expr...])"), params); }

    auto new_scope = scope;

    const auto variable_list = engine.get_list_range(engine.values[params[0]], str("((var1 val1) ...)"));
    if (!variable_list) { return variable_list.error(); }

    for (const auto variable : *variable_list) {
      const auto variable_elements = engine.get_list_range(variable, str("(var1 val1)"), 2, 2);
      if (!variable_elements) { return variable_elements.error(); }

      auto variable_id = engine.eval_to<identifier_type>(scope, (*variable_elements)[0]);
      if (!variable_id) { return engine.make_error(str("expected identifier"), variable_id.error()); }
      new_scope.emplace_back(to_string(*variable_id), engine.eval(scope, (*variable_elements)[1]));
    }

    // evaluate body
    return engine.sequence(new_scope, params.sublist(1));
  }

  template<typename Type>
  [[nodiscard]] constexpr std::expected<Type, SExpr>
    eval_to(LexicalScope &scope, list_type params, string_view_type expected)
  {
    if (params.size != 1) { return std::unexpected(make_error(expected, params)); }
    auto first = eval_to<Type>(scope, values[params[0]]);
    if (!first) { return std::unexpected(make_error(expected, params)); }

    return *first;
  }

  template<typename Type1, typename Type2>
  [[nodiscard]] constexpr std::expected<std::tuple<Type1, Type2>, SExpr>
    eval_to(LexicalScope &scope, list_type params, string_view_type expected)
  {
    if (params.size != 2) { return std::unexpected(make_error(expected, params)); }

    auto first = eval_to<Type1>(scope, values[params[0]]);
    auto second = eval_to<Type2>(scope, values[params[1]]);

    if (!first || !second) { return std::unexpected(make_error(expected, params)); }

    return std::tuple<Type1, Type2>{ *first, *second };
  }

  [[nodiscard]] static constexpr SExpr append(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    auto evaled_params =
      engine.eval_to<literal_list_type, literal_list_type>(scope, params, str("(append LiteralList LiteralList)"));
    if (!evaled_params) { return evaled_params.error(); }
    const auto &[first, second] = *evaled_params;

    Scratch result{ engine.object_scratch };

    for (const auto &value : engine.values[first.items]) { result.push_back(value); }
    for (const auto &value : engine.values[second.items]) { result.push_back(value); }

    return SExpr{ LiteralList{ engine.values.insert_or_find(result) } };
  }

  // (cons 1 '(2 3)) -> '(1 2 3)
  // (cons '(a) '(b c)) -> '((a) b c)
  [[nodiscard]] static constexpr SExpr cons(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    auto evaled_params = engine.eval_to<SExpr, literal_list_type>(scope, params, str("(cons Expr LiteralList)"));
    if (!evaled_params) { return evaled_params.error(); }
    const auto &[front, list] = *evaled_params;

    Scratch result{ engine.object_scratch };

    if (const auto *list_front = std::get_if<literal_list_type>(&front.value); list_front != nullptr) {
      // First element is a list, add it as a nested list
      result.push_back(SExpr{ list_front->items });
    } else if (const auto *atom = std::get_if<Atom>(&front.value); atom != nullptr) {
      if (const auto *identifier_front = std::get_if<symbol_type>(atom); identifier_front != nullptr) {
        // Convert symbol to identifier when adding to result list
        // Note: should maybe fix this so quoted lists are always lists of symbols?
        result.push_back(SExpr{ Atom{ to_identifier(*identifier_front) } });
      } else {
        // Regular atom, keep as-is
        result.push_back(front);
      }
    } else {
      // Any other expression type
      result.push_back(front);
    }

    // Add the remaining elements from the second list
    for (const auto &value : engine.values[list.items]) { result.push_back(value); }

    return SExpr{ LiteralList{ engine.values.insert_or_find(result) } };
  }

  // Helper for monadic-style error handling
  // If operation succeeded, calls callable with the result
  // If operation failed, propagates the error
  template<typename ValueType>
  [[nodiscard]] static constexpr SExpr error_or_else(const std::expected<ValueType, SExpr> &obj, auto callable)
  {
    if (obj) {
      return callable(*obj);
    } else {
      return obj.error();
    }
  }

  // Empty indexed list for reuse
  static constexpr IndexedList<size_type> empty_indexed_list{ 0, 0 };
  
  // (cdr '(1 2 3)) -> '(2 3)
  // (cdr '(1)) -> '()
  [[nodiscard]] static constexpr SExpr cdr(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    return error_or_else(
      engine.eval_to<literal_list_type>(scope, params, str("(cdr LiteralList)")), [&](const auto &list) {
        // If the list has one or zero elements, return empty list
        if (list.items.size <= 1) {
          return SExpr{ literal_list_type{ empty_indexed_list } };
        }
        return SExpr{ list.sublist(1) };
      });
  }

  // (car '(1 2 3)) -> 1
  // (car '((a b) c)) -> '(a b)
  [[nodiscard]] static constexpr SExpr car(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    return error_or_else(
      engine.eval_to<literal_list_type>(scope, params, str("(car Non-Empty-LiteralList)")), [&](const auto &list) {
        // Check if list is empty
        if (list.items.size == 0) { return engine.make_error(str("car: cannot take car of empty list"), params); }

        // Get the first element of the list
        const auto &elem = engine.values[list.items.front()];

        // If the element is a list_type, return it as a literal_list_type
        if (const auto *nested_list = std::get_if<list_type>(&elem.value); nested_list != nullptr) {
          return SExpr{ literal_list_type{ *nested_list } };
        }

        return elem;
      });
  }

  [[nodiscard]] static constexpr SExpr applier(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    return error_or_else(engine.eval_to<SExpr, literal_list_type>(scope, params, str("(apply Function LiteralList)")),
      [&](const auto &evaled_params) {
        return engine.invoke_function(scope, std::get<0>(evaled_params), std::get<1>(evaled_params).items);
      });
  }

  [[nodiscard]] static constexpr SExpr begin(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    return engine.sequence(scope, params);
  }


  [[nodiscard]] static constexpr SExpr evaler(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    return error_or_else(engine.eval_to<literal_list_type>(scope, params, str("(eval LiteralList)")),
      [&](const auto &list) { return engine.eval(engine.global_scope, SExpr{ list.items }); });
  }

  // (cond ((< 5 10) "less") ((> 5 10) "greater") (else "equal")) -> "less"
  // (cond ((= 5 10) "equal") ((> 5 10) "greater") (else "less")) -> "less"
  [[nodiscard]] static constexpr SExpr cond(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    // Evaluate each condition pair in sequence
    for (const auto &entry : engine.values[params]) {
      const auto cond = engine.eval_to<list_type>(scope, entry);
      if (!cond || cond->size != 2) { return engine.make_error(str("(condition statement)"), cond.error()); }

      // Check for the special 'else' case - always matches and returns its expression
      if (const auto *cond_str = get_if<identifier_type>(&engine.values[(*cond)[0]]);
        cond_str != nullptr && engine.strings.view(to_string(*cond_str)) == str("else")) {
        // we've reached the "else" condition
        return engine.eval(scope, engine.values[(*cond)[1]]);
      } else {
        // Evaluate the condition to check if it's true
        const auto condition = engine.eval_to<bool>(scope, engine.values[(*cond)[0]]);
        if (!condition) { return engine.make_error(str("boolean condition"), condition.error()); }

        // If this condition matches, evaluate and return its expression
        if (*condition) { return engine.eval(scope, engine.values[(*cond)[1]]); }
      }
    }

    // No matching condition, including no else clause
    return engine.make_error(str("No matching condition found"), params);
  }


  // (if true 1 2) -> 1
  // (if false 1 2) -> 2
  // (if (< 5 10) (+ 1 2) (- 10 5)) -> 3
  [[nodiscard]] static constexpr SExpr ifer(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    // need to be careful to not execute unexecuted branches
    if (params.size != 3) { return engine.make_error(str("(if bool-cond then else)"), params); }

    // Evaluate the condition to a boolean
    const auto condition = engine.eval_to<bool>(scope, engine.values[params[0]]);
    if (!condition) { return engine.make_error(str("boolean condition"), condition.error()); }

    // Only evaluate the branch that needs to be taken
    if (*condition) {
      return engine.eval(scope, engine.values[params[1]]);  // true branch
    } else {
      return engine.eval(scope, engine.values[params[2]]);  // false branch
    }
  }

  [[nodiscard]] static constexpr SExpr for_each(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    auto evaled_params = engine.eval_to<SExpr, literal_list_type>(scope, params, str("(for_each Function (param...))"));
    if (!evaled_params) { return evaled_params.error(); }
    const auto &[func, applied_params] = *evaled_params;

    for (size_type index = 0; index < applied_params.items.size; ++index) {
      [[maybe_unused]] const auto result = engine.invoke_function(scope, func, applied_params.items.sublist(index, 1));
    }

    return SExpr{ Atom{ std::monostate{} } };
  }
  
  // error?: Check if the expression is an error
  [[nodiscard]] static constexpr SExpr error_p(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    if (params.size != 1) { return engine.make_error(str("(error? expr)"), params); }
    
    // Evaluate the expression
    auto expr = engine.eval(scope, engine.values[params[0]]);
    
    // Check if it's an error type
    const bool is_error = std::holds_alternative<error_type>(expr.value);
    
    return SExpr{ Atom(is_error) };
  }

  [[nodiscard]] static constexpr SExpr quoter(cons_expr &engine, LexicalScope &, list_type params)
  {
    if (params.size != 1) { return engine.make_error(str("(quote expr)"), params); }

    const auto &expr = engine.values[params[0]];

    // If it's a list, convert it to a literal list
    if (const auto *list = std::get_if<list_type>(&expr.value); list != nullptr) {
      // Special case for empty lists - use a canonical empty list with start index 0
      if (list->size == 0) {
        return SExpr{ literal_list_type{ empty_indexed_list } };
      }
      return SExpr{ literal_list_type{ *list } };
    }
    // If it's an identifier, convert it to a symbol
    else if (const auto *atom = std::get_if<Atom>(&expr.value); atom != nullptr) {
      if (const auto *id = std::get_if<identifier_type>(atom); id != nullptr) {
        return SExpr{ Atom{ symbol_type{ to_symbol(*id) } } };
      }
    }

    // Otherwise return as is
    return expr;
  }

  [[nodiscard]] static constexpr SExpr definer(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    return error_or_else(engine.eval_to<identifier_type, SExpr>(scope, params, str("(define Identifier Expression)")),
      [&](const auto &evaled) {
        scope.emplace_back(to_string(std::get<0>(evaled)), engine.fix_identifiers(std::get<1>(evaled), {}, scope));
        return SExpr{ Atom{ std::monostate{} } };
      });
  }

  // take a string_view and return a C++ function object
  // of unspecified type.
  template<typename Signature>
  [[nodiscard]] constexpr auto make_callable(SExpr callable) noexcept
    requires std::is_function_v<Signature>
  {
    auto impl = [this, callable]<typename Ret, typename... Params>(Ret (*)(Params...)) {
      return [callable](cons_expr &engine, Params... params) {
        std::array<SExpr, sizeof...(Params)> args{ SExpr{ Atom{ params } }... };
        if constexpr (std::is_same_v<void, Ret>) {
          engine.eval(engine.global_scope,
            engine.invoke_function(engine.global_scope, callable, engine.values.insert_or_find(args)));
        } else {
          return engine.eval_to<Ret>(engine.global_scope,
            engine.invoke_function(engine.global_scope, callable, engine.values.insert_or_find(args)));
        }
      };
    };

    return impl(std::add_pointer_t<Signature>{ nullptr });
  }

  template<typename Signature>
  [[nodiscard]] constexpr auto make_callable(std::basic_string_view<char_type> function) noexcept
    requires std::is_function_v<Signature>
  {
    // this is fragile, we need to check parsing better
    return make_callable<Signature>(eval(global_scope, values[std::get<list_type>(parse(function).first.value)][0]));
  }


  template<typename T> [[nodiscard]] constexpr auto eval_transform(LexicalScope &scope)
  {
    return std::views::transform([&scope, this](const SExpr param) { return eval_to<T>(scope, param); });
  }

  template<auto Op>
  [[nodiscard]] static constexpr SExpr binary_left_fold(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    auto fold = [&engine, &scope, params]<typename Param>(Param first) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        for (const auto &elem : engine.values[params.sublist(1)]) {
          const auto &next = engine.eval_to<Param>(scope, elem);
          if (!next) { return engine.make_error(str("same types for operator"), SExpr{ first }, next.error()); }
          first = Op(first, *next);
        }

        return SExpr{ Atom{ first } };
      } else {
        return engine.make_error(str("operator not supported for types"), params);
      }
    };

    if (params.size > 1) {
      const auto param1 = engine.eval(scope, engine.values[params[0]]);
      if (const auto *atom = std::get_if<Atom>(&param1.value); atom != nullptr) {
        return visit(fold, *atom);
      } else {
        return engine.make_error(str("operator not supported for types"), params);
      }
    }

    return engine.make_error(str("operator requires at east two parameters"), params);
  }

  [[nodiscard]] static constexpr SExpr logical_and(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    for (const auto &next : engine.values[params] | engine.eval_transform<bool>(scope)) {
      if (!next) { return engine.make_error(str("parameter not boolean"), next.error()); }
      if (!(*next)) { return SExpr{ Atom{ false } }; }
    }

    return SExpr{ Atom{ true } };
  }

  [[nodiscard]] static constexpr SExpr logical_or(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    for (const auto &next : engine.values[params] | engine.eval_transform<bool>(scope)) {
      if (!next) { return engine.make_error(str("parameter not boolean"), next.error()); }
      if (*next) { return SExpr{ Atom{ true } }; }
    }

    return SExpr{ Atom{ false } };
  }

  template<auto Op>
  [[nodiscard]] static constexpr SExpr
    binary_boolean_apply_pairwise(cons_expr &engine, LexicalScope &scope, list_type params)
  {
    auto sum = [&engine, &scope, params]<typename Param>(Param next) -> SExpr {
      if constexpr (requires(Param p1, Param p2) { Op(p1, p2); }) {
        for (const auto &elem : engine.values[params.sublist(1)]) {
          const auto &result = engine.eval_to<Param>(scope, elem);
          if (!result) { return engine.make_error(str("same types for operator"), SExpr{ next }, result.error()); }
          const auto prev = std::exchange(next, *result);
          if (!Op(prev, next)) { return SExpr{ Atom{ false } }; }
        }

        return SExpr{ Atom{ true } };
      } else {
        return engine.make_error(str("supported types"), params);
      }
    };

    if (params.size < 2) { return engine.make_error(str("at least 2 parameters"), params); }
    auto first_param = engine.eval(scope, engine.values[params[0]]).value;

    // For working directly on "LiteralList" objects
    if (const auto *list = std::get_if<literal_list_type>(&first_param); list != nullptr) { return sum(*list); }

    if (const auto *atom = std::get_if<Atom>(&first_param); atom != nullptr) { return visit(sum, *atom); }

    return engine.make_error(str("supported types"), params);
  }

  [[nodiscard]] constexpr SExpr evaluate(string_view_type input)
  {
    const auto result = parse(input).first;
    const auto *list = std::get_if<list_type>(&result.value);

    if (list != nullptr) { return sequence(global_scope, *list); }
    return result;
  }

  template<typename Result> [[nodiscard]] constexpr std::expected<Result, SExpr> evaluate_to(string_view_type input)
  {
    return eval_to<Result>(global_scope, evaluate(input));
  }
};


}// namespace lefticus

#endif
