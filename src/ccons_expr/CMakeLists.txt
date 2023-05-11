add_executable(ccons_expr main.cpp)

target_link_libraries(
  ccons_expr
  PRIVATE cons_expr::cons_expr
          cons_expr::cons_expr_options
          cons_expr::cons_expr_warnings)

target_link_system_libraries(
  ccons_expr
  PRIVATE
          CLI11::CLI11
          spdlog::spdlog
          ftxui::screen
          ftxui::dom
          ftxui::component)

target_include_directories(ccons_expr PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")