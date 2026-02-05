#pragma once

#include <iostream>
#include <vector>

#include "bitonic_sort.hpp"

using SortFunc = void (*)(std::vector<int>::iterator, std::vector<int>::iterator, Direction);

template <typename T>
std::ostream& operator<<(std::ostream& ostream, const std::vector<T>& vector);

void RandFill(std::vector<int>& vector);

void TestBitonicSortsCorrectness(std::size_t start_size, std::size_t end_size);
// void CompareBitonicSortsPerformance(std::size_t start_size, std::size_t end_size);
