#pragma once

#include <vector>

enum class Direction
{
    Descending = 0,
    Ascending = 1
};

template <typename T>
class Bitonic
{
  private:
    using iter = typename std::vector<T>::iterator;

  public:
    static void cpu_sort(iter begin, iter end, Direction direction);
    static void gpu_sort(iter begin, iter end, Direction direction);

  private:
    static void cpu_merge(iter begin, iter end, Direction direction);
    static void gpu_merge(iter begin, iter end, Direction direction);

    static void cpu_comp_and_swap(iter first, iter second, Direction direction);
    static void gpu_comp_and_swap(iter first, iter second, Direction direction);
};
