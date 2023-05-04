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
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::cons_expr<>::Lambda &);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const std::monostate &);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::cons_expr<>::Atom &input);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::cons_expr<>::function_ptr &);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::IndexedList &list);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::LiteralList &list);
std::string to_string(const lefticus::cons_expr<> &, bool annotate, const lefticus::IndexedString &string);


std::string to_string([[maybe_unused]] const lefticus::cons_expr<> &, bool , const lefticus::cons_expr<>::Lambda &)
{
  return "[lambda]";
}

std::string to_string([[maybe_unused]] const lefticus::cons_expr<> &, bool , const std::monostate &) { return "nil"; }

std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::Identifier &id)
{
  if (annotate) {
    return std::format("[identifier] {{{}, {}}} {}", id.value.start, id.value.size, engine.strings[id.value]);
  } else {
    return std::string{engine.strings[id.value]};
  }
}


std::string to_string(const lefticus::cons_expr<> &, bool , const bool input)
{
  if (input) {
    return "true";
  } else {
    return "false";
  }
}

std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::cons_expr<>::Atom &input)
{
  return std::visit([&](const auto &value) { return to_string(engine, annotate, value); }, input);
}

std::string to_string(const lefticus::cons_expr<> &, bool , const double input) { return std::format("{}f", input); }
std::string to_string(const lefticus::cons_expr<> &, bool , const int input) { return std::format("{}", input); }

std::string to_string(const lefticus::cons_expr<> &, bool , const lefticus::cons_expr<>::function_ptr &)
{
  return "function_ptr";
}
std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::IndexedList &list)
{
  std::string result;

  if (annotate) {
    result += std::format("[list] {{{}, {}}} ", list.start, list.size);
  }
  result += "(";
  const auto span = engine.values[list];

  if (!span.empty()) {
    for (const auto &item : span.subspan(0, span.size() - 1)) { result += to_string(engine, annotate, item) + ' '; }
    result += to_string(engine, annotate, span.back());
  }
  result += ")";
  return result;
}

std::string to_string(const lefticus::cons_expr<> &engine, bool annotate, const lefticus::LiteralList &list)
{
  std::string result;
  if (annotate) { result += std::format("[literal list] {{{}, {}}} ", list.items.start, list.items.size); }
  return result + "'" + to_string(engine, annotate, list.items);
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
  int selected = 0;

  auto update_objects = [&]() {
    entries.clear();
    std::size_t index = 0;
    for (auto itr = evaluator.values.small.begin(); itr != evaluator.values.small_end(); ++itr) {
      entries.push_back(std::format("{}: {}", index, to_string(evaluator, true, *itr)));
      ++index;
    }
  };

  update_objects();

  auto do_evaluate = [&]() {
    lefticus::cons_expr<>::Context context;

    content_2 += "\n> " + content_1 + "\n";

    try {
      content_2 += to_string(evaluator, false, evaluator.sequence(context, evaluator.parse(content_1).first.to_list(evaluator)));
    } catch (const std::exception &e) {
      content_2 += std::string("Error: ") + e.what();
    }
    //content_1.clear();
    update_objects();
  };


  auto textarea_1 = ftxui::Input(&content_1);
  auto output_1 = ftxui::Input(&content_2);
  auto button = ftxui::Button("Evaluate", do_evaluate);
  int size = 50;
  auto resizeable_bits = ftxui::ResizableSplitLeft(textarea_1, output_1, &size);

  auto radiobox = ftxui::Menu(&entries, &selected);

  auto layout = ftxui::Container::Vertical({ radiobox, resizeable_bits, button });

  auto component = ftxui::Renderer(layout, [&] {
    return ftxui::vbox({ radiobox->Render() | ftxui::vscroll_indicator | ftxui::frame
                           | ftxui::size(ftxui::HEIGHT, ftxui::EQUAL, 10),
             ftxui::separator(),
             resizeable_bits->Render() | ftxui::flex,
             button->Render() })
           | ftxui::border;
  });

  auto screen = ftxui::ScreenInteractive::Fullscreen();
  screen.Loop(component);
}

// Copyright 2020 Arthur Sonzogni. All rights reserved.
// Use of this source code is governed by the MIT license that can be found in
// the LICENSE file.
