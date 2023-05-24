#include <format>
#include <string>
#include <vector>

#include "ftxui/component/captured_mouse.hpp"// for ftxui
#include "ftxui/component/component.hpp"// for Input, Renderer, ResizableSplitLeft
#include "ftxui/component/component_base.hpp"// for ComponentBase, Component
#include "ftxui/component/screen_interactive.hpp"// for ScreenInteractive
#include "ftxui/dom/elements.hpp"// for operator|, separator, text, Element, flex, vbox, border
#include "ftxui/dom/table.hpp"// for operator|, separator, text, Element, flex, vbox, border

#include <cons_expr/cons_expr.hpp>
#include <cons_expr/utility.hpp>

#include <internal_use_only/config.hpp>


int main([[maybe_unused]] int argc, [[maybe_unused]] const char *argv[])
{

  lefticus::cons_expr evaluator;


  std::string content_1;
  std::string content_2;


  // evaluator.add<display>("display");

  std::vector<std::string> entries;
  std::vector<std::string> characters;

  int selected = 0;
  int character_selected = 0;

  auto update_objects = [&]() {
    entries.clear();
    std::size_t index = 0;
    for (auto itr = evaluator.values.small.begin(); itr != evaluator.values.small_end(); ++itr) {
      entries.push_back(std::format("{}: {}", index, to_string(evaluator, true, *itr)));
      ++index;
    }

    index = evaluator.values.small_capacity;
    for (const auto &value : evaluator.values.rest) {
      entries.push_back(std::format("{}: {}", index, to_string(evaluator, true, value)));
      ++index;
    }

    characters.clear();
    index = 0;
    for (auto itr = evaluator.strings.small.begin(); itr != evaluator.strings.small_end(); ++itr) {
      characters.push_back(std::format("{}: '{}'", index, *itr));
      ++index;
    }
  };

  update_objects();

  auto do_evaluate = [&]() {
    content_2 += "\n> " + content_1 + "\n";

    try {
      content_2 += to_string(
        evaluator, false, evaluator.sequence(evaluator.global_scope, evaluator.parse(content_1).first.to_list()));
    } catch (const std::exception &e) {
      content_2 += std::string("Error: ") + e.what();
    }
    // content_1.clear();
    update_objects();
  };


  auto textarea_1 = ftxui::Input(&content_1);
  auto output_1 = ftxui::Input(&content_2);
  auto button = ftxui::Button("Evaluate", do_evaluate);
  int size = 50;
  auto resizeable_bits = ftxui::ResizableSplitLeft(textarea_1, output_1, &size);

  auto radiobox = ftxui::Menu(&entries, &selected);
  auto characterbox = ftxui::Menu(&characters, &character_selected);
  bool forget_history = false;
  auto forget_history_check = ftxui::Checkbox("Forget History", &forget_history);

  auto layout = ftxui::Container::Horizontal({ characterbox, radiobox, resizeable_bits, button, forget_history_check });

  auto get_stats = [&]() {
    return ftxui::vbox({ ftxui::text(std::format("Data Sizes: cons_expr<> {} SExpr {} Atom {}",
                           sizeof(lefticus::cons_expr<>),
                           sizeof(lefticus::cons_expr<>::SExpr),
                           sizeof(lefticus::cons_expr<>::Atom))),
      ftxui::text(std::format("string used: {}+{}  symbols used: {}+{}  values used: {}+{}",
        evaluator.strings.small_size_used,
        evaluator.strings.rest.size(),
        evaluator.global_scope.small_size_used,
        evaluator.global_scope.rest.size(),
        evaluator.values.small_size_used,
        evaluator.values.rest.size())),
      ftxui::text(std::format(
        "GIT SHA: {}  version string: {}", cons_expr::cmake::git_sha, cons_expr::cmake::project_version)) });
  };

  auto component = ftxui::Renderer(layout, [&] {
    return ftxui::hbox({ characterbox->Render() | ftxui::vscroll_indicator | ftxui::frame,
             ftxui::separator(),
             ftxui::vbox({ radiobox->Render() | ftxui::vscroll_indicator | ftxui::frame
                             | ftxui::size(ftxui::HEIGHT, ftxui::EQUAL, 10),
               ftxui::separator(),
               resizeable_bits->Render() | ftxui::flex,
               ftxui::separator(),
               ftxui::hbox({ button->Render(), get_stats(), forget_history_check->Render() }) })
               | ftxui::flex })
           | ftxui::border;
  });

  auto screen = ftxui::ScreenInteractive::Fullscreen();
  screen.Loop(component);
}
