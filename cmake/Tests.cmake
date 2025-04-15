function(cons_expr_enable_coverage project_name)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    target_compile_options(${project_name} INTERFACE --coverage -O0 -g)
    target_link_libraries(${project_name} INTERFACE --coverage)
    
    # Create a custom target for generating coverage reports
    if(cons_expr_ENABLE_COVERAGE)
      add_custom_target(
        coverage_report
        # First reset coverage data
        COMMAND find . -name "*.gcda" -delete
        COMMAND find . -name "coverage.info" -delete

        # Run the tests
        COMMAND ctest -C Debug
        # Use a separate script to run the coverage commands
        COMMAND lcov --capture --directory . --output-file coverage.info --exclude \"/home/jason/cons_expr/test/*\" --exclude \"/usr/*\" --exclude \"/home/jason/cons_expr/build-coverage/_deps/*\" --output-file coverage.info
        COMMAND genhtml coverage.info --output-directory coverage_report
        COMMAND lcov --list coverage.info
        COMMENT "Resetting coverage counters, running tests, and generating coverage report"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
      add_custom_command(
        TARGET coverage_report
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated in ${CMAKE_BINARY_DIR}/coverage_report/index.html"
      )
    endif()
  endif()
endfunction()
