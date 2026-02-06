#pragma once

#include <algorithm>
#include <cstddef>

#include "interface.hpp"

template <typename T>
void Bitonic<T>::cpu_comp_and_swap(iter first, iter second, Direction direction)
{
    if ((direction == Direction::Ascending && *first > *second) ||
        (direction == Direction::Descending && *first < *second))
    {
        std::iter_swap(first, second);
    }
}

template <typename T>
void Bitonic<T>::cpu_comp_and_swap(iter first, iter second, bool direction)
{
    if ((direction == 1 && *first > *second) ||
        (direction == 0 && *first < *second))
    {
        std::iter_swap(first, second);
    }
}

template <typename T>
void Bitonic<T>::cpu_merge(iter begin, iter end, Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    std::ptrdiff_t half = size / 2;

    for (std::ptrdiff_t i = 0; i < half; ++i)
    {
        Bitonic::cpu_comp_and_swap(begin + i, begin + half + i, direction);
    }

    Bitonic::cpu_merge(begin, begin + half, direction);
    Bitonic::cpu_merge(begin + half, end, direction);
}

template <typename T>
void Bitonic<T>::cpu_sort_recursive(iter begin, iter end, Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }

    std::ptrdiff_t half = size / 2;

    Bitonic::cpu_sort_recursive(begin, begin + half, Direction::Ascending);
    Bitonic::cpu_sort_recursive(begin + half, end, Direction::Descending);

    Bitonic::cpu_merge(begin, end, direction);
}

// template <typename T>
// void Bitonic<T>::cpu_sort_iterative(iter begin, iter end, Direction direction)
// {
//     std::ptrdiff_t size = end - begin;
//     if (size <= 1) { return; }

//     for (int block_size = 2; block_size <= size; block_size *= 2)
//     {
//         for (int dist = block_size / 2; dist > 0; dist /= 2)
//         {
//             for (int pos = 0; pos < size; ++pos)
//             {
//                 int partner = pos ^ dist;
//                 if (partner > pos)
//                 {
//                     bool use_original_direction = (pos & block_size) == 0;
//                     Direction local_direction = use_original_direction ? direction : !direction;

//                     Bitonic::cpu_comp_and_swap(begin + pos, begin + partner, local_direction);
//                 }
//             }
//         }
//     }
// }

template <typename T>
void Bitonic<T>::cpu_sort_iterative(iter begin, iter end, Direction direction)
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
                    Bitonic::cpu_comp_and_swap(begin + pos, begin + partner, local_direction);
                }
            }
        }
    }
}
