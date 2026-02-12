#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>

#include "interface.hpp"

namespace Bitonic
{

namespace details
{

void cpu_comp_and_swap(int& first, int& second, Direction direction)
{
    if ((first > second) == (direction == Direction::Ascending)) { std::swap(first, second); }
}

void cpu_comp_and_swap(int& first, int& second, int direction)
{
    if ((first > second) == (direction == 1)) { std::swap(first, second); }
}

void cpu_merge(std::vector<int>::iterator begin, std::vector<int>::iterator end,
               Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    std::ptrdiff_t half = size / 2;

    for (std::ptrdiff_t i = 0; i < half; ++i)
    {
        cpu_comp_and_swap(begin[i], begin[half + i], direction);
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

void cpu_sort_iterative_0(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    for (std::size_t block_size = 2; block_size <= size; block_size *= 2)
    {
        for (std::size_t dist = block_size / 2; dist > 0; dist /= 2)
        {
            for (std::size_t pos = 0; pos < size; ++pos)
            {
                std::size_t partner = pos ^ dist;
                if (partner > pos)
                {
                    bool use_original_direction = (pos & block_size) == 0;
                    Direction local_direction = use_original_direction ? direction : !direction;

                    details::cpu_comp_and_swap(begin[pos], begin[partner], local_direction);
                }
            }
        }
    }
}

void cpu_sort_iterative_1(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                        Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    for (std::size_t block_size = 2; block_size <= size; block_size *= 2)
    {
        for (std::size_t dist = block_size / 2; dist > 0; dist /= 2)
        {
            for (std::size_t block_idx = 0; block_idx < size / dist; block_idx += 2)
            {
                for (std::size_t pos = block_idx * dist; pos < block_idx * dist + dist; ++pos)
                {
                    std::size_t partner = pos ^ dist;

                    bool use_original_direction = (pos & block_size) == 0;
                    Direction local_direction = use_original_direction ? direction : !direction;

                    details::cpu_comp_and_swap(begin[pos], begin[partner], local_direction);
                }
            }
        }
    }
}

void cpu_sort_iterative_2(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    int dir = static_cast<int>(direction);

    for (std::size_t block_size = 2; block_size <= size; block_size *= 2)
    {
        for (std::size_t dist = block_size / 2; dist > 0; dist /= 2)
        {
            for (std::size_t pos = 0; pos < size / 2; ++pos)
            {
                std::size_t block_index = pos / dist;
                std::size_t correct_pos = pos + block_index * dist;

                std::size_t partner = correct_pos ^ dist;

                bool use_reversed_direction = (correct_pos & block_size) != 0;
                int local_direction = dir ^ use_reversed_direction;

                details::cpu_comp_and_swap(begin[correct_pos], begin[partner], local_direction);
            }
        }
    }
}

void cpu_sort_iterative_3(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    std::size_t size_log = log2(size);

    int dir = static_cast<int>(direction);

    for (std::size_t block_size_log = 1; block_size_log <= size_log; block_size_log += 1)
    {
        std::size_t block_size = 1ul << block_size_log;

        for (int dist_log = block_size_log - 1; dist_log >= 0; dist_log -= 1)
        {
            std::size_t dist = 1ul << dist_log;
            
            for (std::size_t pos = 0; pos < size / 2; ++pos)
            {
                std::size_t block_index = pos >> dist_log;
                std::size_t correct_pos = pos + (block_index << dist_log);

                std::size_t partner = correct_pos ^ dist;

                bool use_reversed_direction = (correct_pos & block_size) != 0;
                int local_direction = dir ^ use_reversed_direction;

                details::cpu_comp_and_swap(begin[correct_pos], begin[partner], local_direction);
            }
        }
    }
}

} // namespace Bitonic
