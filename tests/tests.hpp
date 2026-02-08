#pragma once

#include <chrono>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include "bitonic_sort.hpp"

using SortFunction =
    std::function<void(std::vector<int>::iterator, std::vector<int>::iterator, Direction)>;

void TestBitonicSortsCorrectness(
    std::size_t start_size, std::size_t end_size,
    const std::vector<std::pair<std::string, SortFunction>>& sort_functions);

void CompareSortsPerformance(
    std::size_t start_size, std::size_t end_size,
    const std::vector<std::pair<std::string, SortFunction>>& sort_functions);


inline bool IsPowerOfTwo(std::size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& ostream, const std::vector<T>& vector)
{
    ostream << '[';
    for (std::size_t i = 0; i < vector.size(); ++i)
    {
        if (i > 0) { ostream << ", "; }
        ostream << vector[i];
    }
    ostream << ']';
    return ostream;
}

inline void RandFill(std::vector<int>& vector, int modulo = 1000000)
{
    static std::random_device rd;
    static std::mt19937_64 gen(rd());

    std::uniform_int_distribution<int> dist(-modulo, modulo - 1);

    for (auto& elem : vector)
    {
        elem = dist(gen);
    }
}
