#include <algorithm>
#include <cstddef>
#include <vector>

#include "interface.hpp"

void Bitonic::cpu_comp_and_swap(iter first, iter second, Direction direction)
{
    if ((direction == Direction::Ascending && *first > *second) ||
        (direction == Direction::Descending && *first < *second))
    {
        std::iter_swap(first, second);
    }
}

void Bitonic::cpu_merge(iter begin, iter end, Direction direction)
{
    std::ptrdiff_t size = end - begin;

    if (size <= 1)
    {
        return;
    }

    std::ptrdiff_t half = size / 2;

    for (std::ptrdiff_t i = 0; i < half; ++i)
    {
        Bitonic::cpu_comp_and_swap(begin + i, begin + half + i, direction);
    }

    Bitonic::cpu_merge(begin, begin + half, direction);
    Bitonic::cpu_merge(begin + half, end, direction);
}

void Bitonic::cpu_sort(iter begin, iter end, Direction direction)
{
    std::ptrdiff_t size = end - begin;

    if (size <= 1)
    {
        return;
    }

    std::ptrdiff_t half = size / 2;

    Bitonic::cpu_sort(begin, begin + half, Direction::Ascending);
    Bitonic::cpu_sort(begin + half, end, Direction::Descending);

    Bitonic::cpu_merge(begin, end, direction);
}
