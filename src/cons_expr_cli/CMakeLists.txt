add_executable(cons_expr_cli main.cpp)

target_link_libraries(
  cons_expr_cli
  PRIVATE cons_expr::cons_expr
          cons_expr::cons_expr_options
          cons_expr::cons_expr_warnings)

target_link_system_libraries(
  cons_expr_cli
  PRIVATE spdlog::spdlog
          CLI11::CLI11)

target_include_directories(cons_expr_cli PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")
set_target_properties(cons_expr_cli PROPERTIES OUTPUT_NAME "cons_expr")
