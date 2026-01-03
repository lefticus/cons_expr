#include <CLI/CLI.hpp>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <optional>
#include <ostream>
#include <spdlog/spdlog.h>
#include <sstream>

#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>

#include <internal_use_only/config.hpp>
#include <stdexcept>
#include <string>

using cons_expr_type = lefticus::cons_expr<>;
namespace fs = std::filesystem;

namespace {
void display(cons_expr_type::int_type value) { std::cout << value << '\n'; }

// Read a file into a string
std::string read_file(const fs::path &path)
{
  if (!fs::exists(path)) { throw std::runtime_error(std::format("File not found: {}", path.string())); }

  std::ifstream const file(path, std::ios::in | std::ios::binary);
  if (!file) { throw std::runtime_error(std::format("Failed to open file: {}", path.string())); }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}
}// namespace

int main(int argc, const char **argv)
{
  try {
    CLI::App app{ std::format("{} version {}", cons_expr::cmake::project_name, cons_expr::cmake::project_version) };

    std::optional<std::string> script;
    std::optional<std::string> file_path;
    bool show_version = false;
    app.add_flag("--version", show_version, "Show version information");
    app.add_option("--exec", script, "Script to execute directly");
    app.add_option("--file", file_path, "File containing script to execute");

    CLI11_PARSE(app, argc, argv);

    if (show_version) {
      std::puts(std::format("{}", cons_expr::cmake::project_version).c_str());
      return EXIT_SUCCESS;
    }

    lefticus::cons_expr<> evaluator;

    evaluator.add<display>("display");

    // Process script from command line
    if (script) {
      std::cout << "Executing script from command line...\n";
      std::cout << lefticus::to_string(
        evaluator, false, evaluator.sequence(evaluator.global_scope, evaluator.parse(*script).first));
      std::cout << '\n';
    }

    // Process script from file
    if (file_path) {
      try {
        std::cout << "Executing script from file: " << *file_path << '\n';
        std::string const file_content = read_file(fs::path(*file_path));

        auto [parse_result, remaining] = evaluator.parse(file_content);
        auto result = evaluator.sequence(evaluator.global_scope, parse_result);

        std::cout << "Result: " << lefticus::to_string(evaluator, false, result) << '\n';
      } catch (const std::exception &e) {
        spdlog::error("Error processing file '{}': {}", *file_path, e.what());
        return EXIT_FAILURE;
      }
    }

    // If no script or file provided, display usage
    if (!script && !file_path) {

      while (true) {
        std::cout << "cons_expr> " << std::flush;

        std::string line;
        std::getline(std::cin, line);

        if (!std::cin.good()) { break; }


        auto [parse_result, remaining] = evaluator.parse(line);
        auto result = evaluator.sequence(evaluator.global_scope, parse_result);

        std::cout << lefticus::to_string(evaluator, false, result) << '\n';
      }
    }

  } catch (const std::exception &e) {
    spdlog::error("Unhandled exception in main: {}", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
