#pragma once

#include <vector>

namespace Bitonic
{

enum class Direction : int
{
    Descending = 0,
    Ascending = 1
};

inline Direction operator!(Direction dir)
{
    if (dir == Direction::Ascending) { return Direction::Descending; }

    return Direction::Ascending;
}

/* --------------------------------------------------------------------------------------------- */

void cpu_sort_recursive(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                        Direction direction);

/* --------------------------------------------------------------------------------------------- */

void cpu_sort_iterative_0(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction);

void cpu_sort_iterative_1(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction);

void cpu_sort_iterative_2(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction);

void cpu_sort_iterative_3(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction);

/* --------------------------------------------------------------------------------------------- */

void gpu_naive_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                    Direction direction);

void gpu_naive_sort_better(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                           Direction direction);

void gpu_naive_sort_best(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                         Direction direction);

/* --------------------------------------------------------------------------------------------- */

void gpu_local_sort_naive(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction);

void gpu_local_sort_better(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                           Direction direction);

void gpu_local_sort_best(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                         Direction direction);

/* --------------------------------------------------------------------------------------------- */

void gpu_advanced_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                       Direction direction);

void gpu_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end,
              Direction direction);


}; // namespace Bitonic
