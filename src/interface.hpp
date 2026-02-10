#pragma once

#include <vector>

namespace Bitonic
{

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

void cpu_sort_recursive(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                        Direction direction);

void cpu_sort_iterative(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                        Direction direction);

void cpu_sort_iterative_2(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                        Direction direction);

void gpu_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end,
              Direction direction);

void gpu_stupid_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                     Direction direction);

}; // namespace Bitonic
