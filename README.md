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
   * `cons_expr::cons_expr` is `constinit` meaning that it is *always* 0-cost to construct a cons_expr interpreter
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


## Command Line Inspection Tool

`ccons_expr` can be used to execute scripts and inspect the state of the runtime system live

[![asciicast](https://asciinema.org/a/ZJWpwSjkFqt7Fl750HpeiT3Eg.svg)](https://asciinema.org/a/ZJWpwSjkFqt7Fl750HpeiT3Eg)


## Important Notes

* Objects are never destroyed, but because they are treated as immutable, they are reused as much as possible.
* All types contained in the script are required to be `trivial`
* If you expand beyond the statically allocated storage, dynamic storage is utilized. If you maintain dynamic storage from compile-time to run-time, you will get a compile time error.


