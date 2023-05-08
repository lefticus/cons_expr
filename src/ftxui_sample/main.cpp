#include <memory>// for allocator, __shared_ptr_access, shared_ptr
#include <string>// for string

#include <format>

#include "ftxui/component/captured_mouse.hpp"// for ftxui
#include "ftxui/component/component.hpp"// for Input, Renderer, ResizableSplitLeft
#include "ftxui/component/component_base.hpp"// for ComponentBase, Component
#include "ftxui/component/screen_interactive.hpp"// for ScreenInteractive
#include "ftxui/dom/elements.hpp"// for operator|, separator, text, Element, flex, vbox, border
#include "ftxui/dom/table.hpp"// for operator|, separator, text, Element, flex, vbox, border

#include <cons_expr/cons_expr.hpp>


std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::cons_expr<>::SExpr &input);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const bool input);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const double input);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const int input);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::cons_expr<>::Closure &);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const std::monostate &);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::cons_expr<>::Atom &input);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::cons_expr<>::function_ptr &);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::IndexedList &list);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::LiteralList &list);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::IndexedString &string);


std::string to_string([[maybe_unused]] const lefticus::cons_expr<> &, bool, const lefticus::cons_expr<>::Closure &closure)
{
  return std::format("[closure parameters {{{}, {}}} statements {{{}, {}}}]",
    closure.parameter_names.start,
    closure.parameter_names.size,
    closure.statements.start,
    closure.statements.size);
}

std::string to_string([[maybe_unused]] const lefticus::cons_expr<> &, bool, const std::monostate &) { return "[nil]"; }

std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::Identifier &id)
{
  if (annotate) {
    return std::format("[identifier] {{{}, {}}} {}", id.value.start, id.value.size, engine.strings[id.value]);
  } else {
    return std::string{ engine.strings[id.value] };
  }
}


std::string to_string(const lefticus::cons_expr<> &, bool annotate, const bool input)
{
  std::string result;
  if (annotate) { result = "[bool] "; }
  if (input) {
    return result + "true";
  } else {
    return result + "false";
  }
}

std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::cons_expr<>::Atom &input)
{
  return std::visit([&](const auto &value) { return to_string(engine, annotate, value); }, input);
}

std::string to_string(const lefticus::cons_expr<> &, bool annotate, const double input)
{
  std::string result;
  if (annotate) { result = "[double] "; }

  return result + std::format("{}", input);
}
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const int input)
{
  std::string result;
  if (annotate) { result = "[int] "; }
  return result + std::format("{}", input);
}

std::string to_string(const lefticus::cons_expr<> &, bool, const lefticus::cons_expr<>::function_ptr &func)
{
  return std::format("[function_ptr {}]", reinterpret_cast<const void *>(func));
}
std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::IndexedList &list)
{
  std::string result;

  if (annotate) { result += std::format("[list] {{{}, {}}} ", list.start, list.size); }
  result += "(";
  const auto span = engine.values[list];

  if (!span.empty()) {
    for (const auto &item : span.subspan(0, span.size() - 1)) { result += to_string(engine, false, item) + ' '; }
    result += to_string(engine, false, span.back());
  }
  result += ")";
  return result;
}

std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::LiteralList &list)
{
  std::string result;
  if (annotate) { result += std::format("[literal list] {{{}, {}}} ", list.items.start, list.items.size); }
  return result + "'" + to_string(engine, false, list.items);
}

std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::IndexedString &string)
{
  if (annotate) {
    return std::format("[identifier] {{{}, {}}} \"{}\"", string.start, string.size, engine.strings[string]);
  } else {
    return std::format("\"{}\"", engine.strings[string]);
  }
}

std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::cons_expr<>::SExpr &input)
{
  return std::visit([&](const auto &value) { return to_string(engine, annotate, value); }, input.value);
}


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
      content_2 +=
        to_string(evaluator, false, evaluator.sequence(evaluator.global_scope, evaluator.parse(content_1).first.to_list(evaluator)));
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

  auto layout = ftxui::Container::Horizontal({ characterbox, radiobox, resizeable_bits, button });

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
        evaluator.values.rest.size())) });
  };

  auto component = ftxui::Renderer(layout, [&] {
    return ftxui::hbox({ characterbox->Render() | ftxui::vscroll_indicator | ftxui::frame,
             ftxui::separator(),
             ftxui::vbox({ radiobox->Render() | ftxui::vscroll_indicator | ftxui::frame
                             | ftxui::size(ftxui::HEIGHT, ftxui::EQUAL, 10),
               ftxui::separator(),
               resizeable_bits->Render() | ftxui::flex,
               ftxui::separator(),
               ftxui::hbox({ button->Render(), get_stats() }) })
               | ftxui::flex })
           | ftxui::border;
  });

  auto screen = ftxui::ScreenInteractive::Fullscreen();
  screen.Loop(component);
}
