#include "tests.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

double TestSortPerformance(SortFunction sort_func, const std::vector<int>& base,
                           std::size_t repeats)
{
    double total_time = 0.0;

    for (std::size_t i = 0; i < repeats; ++i)
    {
        std::vector<int> arr = base;
        auto start = std::chrono::high_resolution_clock::now();
        sort_func(arr.begin(), arr.end(), Direction::Ascending);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return total_time / repeats;
}

void CompareSortsPerformance(
    std::size_t start_size, std::size_t end_size,
    const std::vector<std::pair<std::string, SortFunction>>& sort_functions)
{
    std::cout << "Running performance tests..." << std::endl;

    if (!IsPowerOfTwo(start_size) || !IsPowerOfTwo(end_size))
    {
        std::cout << "Sizes should be powers of 2" << std::endl;
        return;
    }

    std::cout << std::setw(10) << "Size";
    for (const auto& [name, _] : sort_functions)
    {
        std::cout << std::setw(30) << name + ", ms";
    }
    std::cout << std::endl;

    for (std::size_t size = start_size; size <= end_size; size *= 2)
    {
        std::vector<int> base(size);
        RandFill(base);

        std::cout << std::setw(10) << size;

        for (const auto& [_, sort_func] : sort_functions)
        {
            double avg_time = TestSortPerformance(sort_func, base, 5);
            std::cout << std::setw(30) << std::fixed << std::setprecision(3) << avg_time;
        }
        std::cout << std::endl;
    }
}
