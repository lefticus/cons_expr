cmake_minimum_required(VERSION 3.15...3.23)

project(CmakeConfigPackageTests LANGUAGES CXX)

# ---- Test as standalone project the exported config package ----

if(PROJECT_IS_TOP_LEVEL OR TEST_INSTALLED_VERSION)
  enable_testing()

  find_package(cons_expr CONFIG REQUIRED) # for intro, project_options, ...

  if(NOT TARGET cons_expr_options)
    message(FATAL_ERROR "Required config package not found!")
    return() # be strictly paranoid for Template Janitor github action! CK
  endif()
endif()

# ---- Dependencies ----

include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

# Provide a simple smoke test to make sure that the CLI works and can display a --help message
add_test(NAME cli.has_help COMMAND cons_expr_cli --help)

# Provide a test to verify that the version being reported from the application
# matches the version given to CMake. This will be important once you package
# your program. Real world shows that this is the kind of simple mistake that is easy
# to make, but also easy to test for.
add_test(NAME cli.version_matches COMMAND cons_expr_cli --version)
set_tests_properties(cli.version_matches PROPERTIES PASS_REGULAR_EXPRESSION "${PROJECT_VERSION}")

# Test direct script execution
add_test(NAME cli.direct_script COMMAND cons_expr_cli --exec "(+ 1 2)")
set_tests_properties(cli.direct_script PROPERTIES PASS_REGULAR_EXPRESSION "3")

# Test file input handling with a simple script
add_test(NAME cli.file_input COMMAND cons_expr_cli --file "${CMAKE_CURRENT_SOURCE_DIR}/test_script.scm")
set_tests_properties(cli.file_input PROPERTIES PASS_REGULAR_EXPRESSION "30")

# Test file input with a non-existent file
add_test(NAME cli.missing_file COMMAND cons_expr_cli --file "non_existent_file.scm")
set_tests_properties(cli.missing_file PROPERTIES 
                     WILL_FAIL TRUE
                     FAIL_REGULAR_EXPRESSION "File not found")

add_executable(tests tests.cpp)
target_link_libraries(
  tests
  PRIVATE cons_expr::cons_expr
          cons_expr::cons_expr_warnings
          cons_expr::cons_expr_options)


target_link_system_libraries(
  tests
  PRIVATE Catch2::Catch2WithMain)

if(WIN32 AND BUILD_SHARED_LIBS)
  add_custom_command(
    TARGET tests
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
    COMMAND_EXPAND_LISTS)
endif()

# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
# to whatever you want, or use different for different binaries
catch_discover_tests(
  tests
  TEST_PREFIX
  "unittests."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "unittests."
  OUTPUT_SUFFIX
  .xml)

# Add a file containing a set of constexpr tests
add_executable(constexpr_tests constexpr_tests.cpp 
                     list_tests.cpp 
                     parser_tests.cpp 
                     recursion_tests.cpp 
                     scoping_tests.cpp 
                     type_predicate_tests.cpp
                     string_escape_tests.cpp
                     error_handling_tests.cpp
                     recursion_and_closure_tests.cpp
                     cond_tests.cpp
                     list_construction_tests.cpp)
target_link_libraries(
  constexpr_tests
  PRIVATE cons_expr::cons_expr
          cons_expr::cons_expr_warnings
          cons_expr::cons_expr_options)

target_link_system_libraries(
  constexpr_tests
  PRIVATE Catch2::Catch2WithMain)

target_include_directories(constexpr_tests PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")

catch_discover_tests(
  constexpr_tests
  TEST_PREFIX
  "constexpr."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "constexpr."
  OUTPUT_SUFFIX
  .xml)

# Disable the constexpr portion of the test, and build again this allows us to have an executable that we can debug when
# things go wrong with the constexpr testing
add_executable(relaxed_constexpr_tests constexpr_tests.cpp 
                             list_tests.cpp 
                             parser_tests.cpp 
                             recursion_tests.cpp 
                             scoping_tests.cpp 
                             type_predicate_tests.cpp
                             string_escape_tests.cpp
                             error_handling_tests.cpp
                             recursion_and_closure_tests.cpp
                             cond_tests.cpp
                             list_construction_tests.cpp)
target_link_libraries(
  relaxed_constexpr_tests
  PRIVATE cons_expr::cons_expr
          cons_expr::cons_expr_warnings
          cons_expr::cons_expr_options
          Catch2::Catch2WithMain)
target_compile_definitions(relaxed_constexpr_tests PRIVATE -DCATCH_CONFIG_RUNTIME_STATIC_REQUIRE)

catch_discover_tests(
  relaxed_constexpr_tests
  TEST_PREFIX
  "relaxed_constexpr."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "relaxed_constexpr."
  OUTPUT_SUFFIX
  .xml)

target_include_directories(relaxed_constexpr_tests PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")
