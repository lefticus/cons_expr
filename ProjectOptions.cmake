include(cmake/SystemLink.cmake)
include(cmake/LibFuzzer.cmake)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)


macro(cons_expr_supports_sanitizers)
  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)
    set(SUPPORTS_UBSAN ON)
  else()
    set(SUPPORTS_UBSAN OFF)
  endif()

  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
    set(SUPPORTS_ASAN OFF)
  else()
    set(SUPPORTS_ASAN ON)
  endif()
endmacro()

macro(cons_expr_setup_options)
  option(cons_expr_ENABLE_HARDENING "Enable hardening" ON)
  option(cons_expr_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  cmake_dependent_option(
    cons_expr_ENABLE_GLOBAL_HARDENING
    "Attempt to push hardening options to built dependencies"
    ON
    cons_expr_ENABLE_HARDENING
    OFF)

  cons_expr_supports_sanitizers()

  if(NOT PROJECT_IS_TOP_LEVEL OR cons_expr_PACKAGING_MAINTAINER_MODE)
    option(cons_expr_ENABLE_IPO "Enable IPO/LTO" OFF)
    option(cons_expr_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(cons_expr_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(cons_expr_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    option(cons_expr_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(cons_expr_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
    option(cons_expr_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(cons_expr_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(cons_expr_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(cons_expr_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(cons_expr_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
    option(cons_expr_ENABLE_PCH "Enable precompiled headers" OFF)
    option(cons_expr_ENABLE_CACHE "Enable ccache" OFF)
  else()
    option(cons_expr_ENABLE_IPO "Enable IPO/LTO" ON)
    option(cons_expr_WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
    option(cons_expr_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(cons_expr_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ${SUPPORTS_ASAN})
    option(cons_expr_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(cons_expr_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ${SUPPORTS_UBSAN})
    option(cons_expr_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(cons_expr_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(cons_expr_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(cons_expr_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(cons_expr_ENABLE_CPPCHECK "Enable cpp-check analysis" ON)
    option(cons_expr_ENABLE_PCH "Enable precompiled headers" OFF)
    option(cons_expr_ENABLE_CACHE "Enable ccache" ON)
  endif()

  if(NOT PROJECT_IS_TOP_LEVEL)
    mark_as_advanced(
      cons_expr_ENABLE_IPO
      cons_expr_WARNINGS_AS_ERRORS
      cons_expr_ENABLE_USER_LINKER
      cons_expr_ENABLE_SANITIZER_ADDRESS
      cons_expr_ENABLE_SANITIZER_LEAK
      cons_expr_ENABLE_SANITIZER_UNDEFINED
      cons_expr_ENABLE_SANITIZER_THREAD
      cons_expr_ENABLE_SANITIZER_MEMORY
      cons_expr_ENABLE_UNITY_BUILD
      cons_expr_ENABLE_CLANG_TIDY
      cons_expr_ENABLE_CPPCHECK
      cons_expr_ENABLE_COVERAGE
      cons_expr_ENABLE_PCH
      cons_expr_ENABLE_CACHE)
  endif()

  cons_expr_check_libfuzzer_support(LIBFUZZER_SUPPORTED)
  if(LIBFUZZER_SUPPORTED AND (cons_expr_ENABLE_SANITIZER_ADDRESS OR cons_expr_ENABLE_SANITIZER_THREAD OR cons_expr_ENABLE_SANITIZER_UNDEFINED))
    set(DEFAULT_FUZZER ON)
  else()
    set(DEFAULT_FUZZER OFF)
  endif()

  option(cons_expr_BUILD_FUZZ_TESTS "Enable fuzz testing executable" ${DEFAULT_FUZZER})

endmacro()

macro(cons_expr_global_options)
  if(cons_expr_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    cons_expr_enable_ipo()
  endif()

  cons_expr_supports_sanitizers()

  if(cons_expr_ENABLE_HARDENING AND cons_expr_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR cons_expr_ENABLE_SANITIZER_UNDEFINED
       OR cons_expr_ENABLE_SANITIZER_ADDRESS
       OR cons_expr_ENABLE_SANITIZER_THREAD
       OR cons_expr_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    message("${cons_expr_ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${cons_expr_ENABLE_SANITIZER_UNDEFINED}")
    cons_expr_enable_hardening(cons_expr_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()
endmacro()

macro(cons_expr_local_options)
  if(PROJECT_IS_TOP_LEVEL)
    include(cmake/StandardProjectSettings.cmake)
  endif()

  add_library(cons_expr_warnings INTERFACE)
  add_library(cons_expr_options INTERFACE)

  include(cmake/CompilerWarnings.cmake)
  cons_expr_set_project_warnings(
    cons_expr_warnings
    ${cons_expr_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
    "")

  if(cons_expr_ENABLE_USER_LINKER)
    include(cmake/Linker.cmake)
    configure_linker(cons_expr_options)
  endif()

  include(cmake/Sanitizers.cmake)
  cons_expr_enable_sanitizers(
    cons_expr_options
    ${cons_expr_ENABLE_SANITIZER_ADDRESS}
    ${cons_expr_ENABLE_SANITIZER_LEAK}
    ${cons_expr_ENABLE_SANITIZER_UNDEFINED}
    ${cons_expr_ENABLE_SANITIZER_THREAD}
    ${cons_expr_ENABLE_SANITIZER_MEMORY})

  set_target_properties(cons_expr_options PROPERTIES UNITY_BUILD ${cons_expr_ENABLE_UNITY_BUILD})

  if(cons_expr_ENABLE_PCH)
    target_precompile_headers(
      cons_expr_options
      INTERFACE
      <vector>
      <string>
      <utility>)
  endif()

  if(cons_expr_ENABLE_CACHE)
    include(cmake/Cache.cmake)
    cons_expr_enable_cache()
  endif()

  include(cmake/StaticAnalyzers.cmake)
  if(cons_expr_ENABLE_CLANG_TIDY)
    cons_expr_enable_clang_tidy(cons_expr_options ${cons_expr_WARNINGS_AS_ERRORS})
  endif()

  if(cons_expr_ENABLE_CPPCHECK)
    cons_expr_enable_cppcheck(${cons_expr_WARNINGS_AS_ERRORS} "" # override cppcheck options
    )
  endif()

  if(cons_expr_ENABLE_COVERAGE)
    include(cmake/Tests.cmake)
    cons_expr_enable_coverage(cons_expr_options)
  endif()

  if(cons_expr_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(cons_expr_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()

  if(cons_expr_ENABLE_HARDENING AND NOT cons_expr_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR cons_expr_ENABLE_SANITIZER_UNDEFINED
       OR cons_expr_ENABLE_SANITIZER_ADDRESS
       OR cons_expr_ENABLE_SANITIZER_THREAD
       OR cons_expr_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    cons_expr_enable_hardening(cons_expr_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()

endmacro()
