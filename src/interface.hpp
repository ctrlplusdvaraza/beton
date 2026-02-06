#pragma once

#include <vector>

enum class Direction : int
{
    Descending = -1,
    Ascending = 0
};

inline Direction operator!(Direction dir)
{
    if (dir == Direction::Ascending) { return Direction::Descending; }

    return Direction::Ascending;
}

template <typename T>
class Bitonic
{
  private:
    using iter = typename std::vector<T>::iterator;

  public:
    static void cpu_sort_recursive(iter begin, iter end, Direction direction);
    static void cpu_sort_iterative(iter begin, iter end, Direction direction);
    static void gpu_sort(iter begin, iter end, Direction direction);

  private:
    static void cpu_merge(iter begin, iter end, Direction direction);
    static void gpu_merge(iter begin, iter end, Direction direction);

    static void cpu_comp_and_swap(iter first, iter second, Direction direction);
    static void cpu_comp_and_swap(iter first, iter second, bool direction);
    static void gpu_comp_and_swap(iter first, iter second, Direction direction);
};
