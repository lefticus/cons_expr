#include <cons_expr/cons_expr.hpp>

template<typename Contained, typename Allocator = std::allocator<Contained>> struct null_container
{
  constexpr const Contained *begin() const { return &dummyobj; }

  constexpr const Contained *end() const { return &dummyobj; }

  constexpr const Contained &operator[](const std::size_t) const { return dummyobj; }
  constexpr Contained &operator[](const std::size_t) { return dummyobj; }


  constexpr void push_back(const Contained &) {}

  Contained dummyobj;

  constexpr bool empty() const { return true; }
  constexpr std::size_t size() const { return 0; }
};

int main(int argc, const char **argv)
{
  //  lefticus::cons_expr<std::uint16_t, wchar_t> evaluator1;

  //  lefticus::cons_expr<lefticus::cons_expr_settings<std::uint16_t, char, int, float, 64, 1024, 256, null_container>>
  //  evaluator;
  //   lefticus::cons_expr<lefticus::cons_expr_settings<std::uint16_t, char, int, float, 64, 128, 64, std::vector>>
  //   evaluator;

  lefticus::cons_expr evaluator;

  evaluator.sequence(
    evaluator.global_scope, std::get<typename lefticus::cons_expr<>::list_type>(evaluator.parse(argv[1]).first.value));
}
