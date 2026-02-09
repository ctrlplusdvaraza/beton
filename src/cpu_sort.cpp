#include <algorithm>
#include <cstddef>

#include "interface.hpp"

namespace Bitonic
{

namespace details
{

void cpu_comp_and_swap(std::vector<int>::iterator first, std::vector<int>::iterator second,
                       Direction direction)
{
    if ((direction == Direction::Ascending && *first > *second) ||
        (direction == Direction::Descending && *first < *second))
    {
        std::iter_swap(first, second);
    }
}

void cpu_merge(std::vector<int>::iterator begin, std::vector<int>::iterator end,
               Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    std::ptrdiff_t half = size / 2;

    for (std::ptrdiff_t i = 0; i < half; ++i)
    {
        cpu_comp_and_swap(begin + i, begin + half + i, direction);
    }

    cpu_merge(begin, begin + half, direction);
    cpu_merge(begin + half, end, direction);
}

} // namespace details

void cpu_sort_recursive(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                        Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    std::ptrdiff_t half = size / 2;

    cpu_sort_recursive(begin, begin + half, Direction::Ascending);
    cpu_sort_recursive(begin + half, end, Direction::Descending);

    details::cpu_merge(begin, end, direction);
}

void cpu_sort_iterative(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                        Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    for (int block_size = 2; block_size <= size; block_size *= 2)
    {
        for (int dist = block_size / 2; dist > 0; dist /= 2)
        {
            for (int block_idx = 0; block_idx < size / dist; block_idx += 2)
            {
                for (int pos = block_idx * dist; pos < block_idx * dist + dist; ++pos)
                {
                    int partner = pos ^ dist;
                    bool use_original_direction = (pos & block_size) == 0;
                    Direction local_direction = use_original_direction ? direction : !direction;
                    details::cpu_comp_and_swap(begin + pos, begin + partner, local_direction);
                }
            }
        }
    }
}

} // namespace Bitonic
