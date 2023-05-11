
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <format>

#include <cons_expr/cons_expr.hpp>

#include <internal_use_only/config.hpp>

void display(int i) { std::cout << i << '\n'; }


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
      [[maybe_unused]] const auto result = evaluator.sequence(evaluator.global_scope, evaluator.parse(*script).first.to_list(evaluator));
    }
  } catch (const std::exception &e) {
    spdlog::error("Unhandled exception in main: {}", e.what());
  }
}
