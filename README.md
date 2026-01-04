# cons_expr

[![ci](https://github.com/lefticus/cons_expr/actions/workflows/ci.yml/badge.svg)](https://github.com/lefticus/cons_expr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lefticus/cons_expr/branch/main/graph/badge.svg)](https://codecov.io/gh/lefticus/cons_expr)
[![CodeQL](https://github.com/lefticus/cons_expr/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/lefticus/cons_expr/actions/workflows/codeql-analysis.yml)


## About cons_expr

**cons_expr** is:

* Scheme-inspired
   * scheme syntax and features
   * does *not* support pairs or macros
   * any value once captured is const
   * any redefinition of a value shadows the previous value
* Constexpr-capable
   * `cons_expr::cons_expr` is `consteval` meaning that it is *always* 0-cost to construct a cons_expr interpreter
   * Any script can be executed at compile time.
   * Why execute script at compile time?
     * I don't know your other use cases
     * But you can use this feature to set up your entire scripting environment at compile time, then execute scripts at runtime
* Simple and small
   * I'm attempting to keep the code as compact as possible. As of now it is "feature complete." Any additional features will go as add-ons in the "utility" header
   * I'm not happy with the `fix_identifiers` family of functions. They need to be simplified and made better.
   * Compilation of entire engine is just a few seconds, even with optimizations.
* Easy to use
   * Include one header and go! See [examples](examples).
* Embeddable scripting language 
   * Bi-directional support for functions between script and C++.
   * All supported types must be known at [compile time](https://github.com/lefticus/cons_expr/blob/main/test/tests.cpp#L57C1-L91).
* For C++23
   * Currently only known to work with GCC 13.1.

## Examples

### Basic Example

It can be as simple as:

```cpp
#include <cons_expr/cons_expr.hpp>
#include <print>

void greet(std::string_view name) { std::println("Hello {}!", name); }

int main() {
  lefticus::cons_expr<> evaluator;

  // add C++ function
  evaluator.add<greet>("greet");

  // call C++ function from script
  const auto _ = evaluator.evaluate(R"(
    (greet "Jason")
  )");
}
```

Play with this in [Compiler Explorer](https://compiler-explorer.com/z/MvvqTdvMK).

### Calling Script Functions from C++

```cpp
#include <cons_expr/cons_expr.hpp>
#include <iostream>

int main() {
  lefticus::cons_expr<> evaluator;

  // you should check this for errors
  [[maybe_unused]] const auto result = evaluator.evaluate(R"(
    (define math
      (lambda (x y) 
        (- (* x y) (+ x y))
      )
    )
  )");  

  // we can treat script functions as C++ functions
  auto func = evaluator.make_callable<int (int, int)>("math");

  std::cout << func(evaluator, 5, 4).value();

  // or bind them and make them more natural
  auto math = std::bind_front(func, std::ref(evaluator));

  std::cout << math(5, 4).value();

  // or bind them *by value* and make them self contained
  auto math2 = std::bind_front(func, evaluator);

  std::cout << math2(5, 4).value();
}
```

Play with this example in [Compiler Explorer](https://compiler-explorer.com/z/WGa54G9Ee).

### More Complete Bi-Directional Example

```cpp
#include <cmath>
#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>
#include <numbers>

namespace lefticus {
constexpr double cos(double input) { return std::cos(input); }
}  // namespace lefticus

int main() {
  lefticus::cons_expr<> evaluator;
  // adding a function
  evaluator.add<lefticus::cos>("cos");

  // adding a lambda, not the + to force it into a 
  // function pointer
  evaluator.add<+[](double input) { return std::sin(input); }>("sin");

  // adding a global
  evaluator.add("pi", std::numbers::pi_v<double>);

  // calling the functions I just added
  // (floating point isn't exact :D)
  const auto result = evaluator.evaluate("(+ (cos pi) (sin pi))");

  // if the above had an error then we'd get a pretty-print
  // of it with this helper function
  
  // using the to_string helper from the utility.hpp
  std::puts(lefticus::to_string(evaluator, true, result).c_str());
}
```

Play with this example in [Compiler Explorer](https://compiler-explorer.com/z/dGYbG88YE).

### And Remember: Everything is `constexpr` Capable

```cpp
#include <https://raw.githubusercontent.com/lefticus/cons_expr/refs/heads/develop/include/cons_expr/cons_expr.hpp>

// the entire system is constexpr capable
consteval int do_math(int x, int y) {
  lefticus::cons_expr<> evaluator;

  // you should check this for errors
  [[maybe_unused]] const auto result = evaluator.evaluate(R"(
    (define math
      (lambda (x y) 
        (- (* x y) (+ x y))
      )
    )
  )");  

  auto func = evaluator.make_callable<int (int, int)>("math");
  return func(evaluator, x, y).value();
}

int main() {
  // doing this in a consteval function above guarantees this is done 
  // at compile time.
  return do_math(4, 2);
}
```

Play with this example in [Compiler Explorer](https://compiler-explorer.com/z/jr51cYYv7).

 
## Command Line Inspection Tool

`ccons_expr` can be used to execute scripts and inspect the state of the runtime system live

[![asciicast](https://asciinema.org/a/ZJWpwSjkFqt7Fl750HpeiT3Eg.svg)](https://asciinema.org/a/ZJWpwSjkFqt7Fl750HpeiT3Eg)

## Online Builds To Play With The Syntax

- Main: [https://lefticus.github.io/cons_expr/](https://lefticus.github.io/cons_expr/)
- Develop: [https://lefticus.github.io/cons_expr/develop/](https://lefticus.github.io/cons_expr/develop/)


## Important Notes

* Objects are never destroyed, but because they are treated as immutable, they are reused as much as possible.
* All types contained in the script are required to be `trivial`
* If you expand beyond the statically allocated storage, dynamic storage is utilized. If you maintain dynamic storage from compile-time to run-time, you will get a compile time error.


