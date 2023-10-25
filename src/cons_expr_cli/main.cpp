
#include <CLI/CLI.hpp>
#include <format>
#include <spdlog/spdlog.h>

#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>

#include <internal_use_only/config.hpp>

using cons_expr_type = lefticus::cons_expr<>;

void display(cons_expr_type::int_type i) { std::cout << i << '\n'; }


int main(int argc, const char **argv)
{
  try {
    CLI::App app{ std::format("{} version {}", cons_expr::cmake::project_name, cons_expr::cmake::project_version) };

    std::optional<std::string> script;
    bool show_version = false;
    app.add_flag("--version", show_version, "Show version information");
    app.add_option("--exec", script, "Script to execute");

    CLI11_PARSE(app, argc, argv);

    if (show_version) {
      std::puts(std::format("{}", cons_expr::cmake::project_version).c_str());
      return EXIT_SUCCESS;
    }

    lefticus::cons_expr<> evaluator;

    evaluator.add<display>("display");

    if (script) {
      std::cout << lefticus::to_string(evaluator,
        false,
        evaluator.sequence(evaluator.global_scope,
          std::get<cons_expr_type::list_type>(evaluator.parse(*script).first.value)));
      std::cout << '\n';
    }
  } catch (const std::exception &e) {
    spdlog::error("Unhandled exception in main: {}", e.what());
  }
}
