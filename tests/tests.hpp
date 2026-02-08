#pragma once

#include <iostream>
#include <vector>

#include "bitonic_sort.hpp"

using SortFunction =
    std::function<void(std::vector<int>::iterator, std::vector<int>::iterator, Direction)>;

void TestBitonicSortsCorrectness(std::size_t start_size, std::size_t end_size);

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
    srand(time(nullptr));

    for (auto& elem : vector)
    {
        elem = (rand() % (2 * modulo)) - modulo;
    }
}
