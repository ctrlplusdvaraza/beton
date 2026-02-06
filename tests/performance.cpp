#include "tests.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void TestSortPerformance(SortFunc sort_func, std::size_t start_size, std::size_t end_size,
                         const std::string& name, std::size_t repeats)
{
    if (start_size % 2 != 0 || end_size % 2 != 0)
    {
        std::cout << "Sizes should be a multiple of 2" << std::endl;
        return;
    }

    std::cout << "=== Performance test for: " << name << " ===" << std::endl;
    std::cout << std::setw(10) << "Size" << std::setw(20) << "Ascending (ms)" << std::setw(20)
              << "Descending (ms)" << std::endl;

    for (std::size_t size = start_size; size <= end_size; size *= 2)
    {
        std::vector<int> base(size);
        RandFill(base);

        double total_time_asc = 0.0;
        double total_time_desc = 0.0;

        for (std::size_t i = 0; i < repeats; ++i)
        {
            {
                std::vector<int> arr = base;
                auto start = std::chrono::high_resolution_clock::now();
                sort_func(arr.begin(), arr.end(), Direction::Ascending);
                auto end = std::chrono::high_resolution_clock::now();
                total_time_asc += std::chrono::duration<double, std::milli>(end - start).count();
            }

            {
                std::vector<int> arr = base;
                auto start = std::chrono::high_resolution_clock::now();
                sort_func(arr.begin(), arr.end(), Direction::Descending);
                auto end = std::chrono::high_resolution_clock::now();
                total_time_desc += std::chrono::duration<double, std::milli>(end - start).count();
            }
        }

        // Среднее время
        double avg_time_asc = total_time_asc / repeats;
        double avg_time_desc = total_time_desc / repeats;

        std::cout << std::setw(10) << size << std::setw(20) << std::fixed << std::setprecision(3)
                  << avg_time_asc << std::setw(20) << std::fixed << std::setprecision(3)
                  << avg_time_desc << std::endl;
    }

    std::cout << "==============================================" << std::endl;
}

