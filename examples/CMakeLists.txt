add_executable(speed_test speed_test.cpp)

target_link_libraries(
  speed_test
  PRIVATE cons_expr::cons_expr
          cons_expr::cons_expr_options
          cons_expr::cons_expr_warnings)


add_executable(compile_test compile_test.cpp)

target_link_libraries(
  compile_test
  PRIVATE cons_expr::cons_expr
          cons_expr::cons_expr_options
          cons_expr::cons_expr_warnings)

