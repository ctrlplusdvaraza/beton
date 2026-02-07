#include "tests.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

double TestSortPerformance(SortFunc sort_func, std::vector<int> base, std::size_t repeats)
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

    double avg_time = total_time / repeats;
    return avg_time;
}

double TestStdSortPerformance(StdSortFunc sort_func, std::vector<int> base, std::size_t repeats)
{
    double total_time = 0.0;

    for (std::size_t i = 0; i < repeats; ++i)
    {
        std::vector<int> arr = base;
        auto start = std::chrono::high_resolution_clock::now();
        sort_func(arr.begin(), arr.end());
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_time = total_time / repeats;
    return avg_time;
}

void CompareBitonicSortsPerformance(std::size_t start_size, std::size_t end_size)
{
    std::cout << "Running performance tests..." << std::endl;

    if (start_size % 2 != 0 || end_size % 2 != 0)
    {
        std::cout << "Sizes should be a multiple of 2" << std::endl;
        return;
    }

    std::cout << std::setw(10) << "Size" << std::setw(20) << "std::sort (ms)" << std::setw(20)
              << "cpu_sort_recursive (ms)" << std::setw(20) << "cpu_sort_iterative (ms)"
              << std::setw(20) << "gpu_sort (ms)" << std::endl;

    for (std::size_t size = start_size; size <= end_size; size *= 2)
    {
        std::vector<int> base(size);
        RandFill(base);

        double avg_std_sort = TestStdSortPerformance(std::sort, base, 5);
        double avg_cpu_rec_sort = TestSortPerformance(Bitonic<int>::cpu_sort_recursive, base, 5);
        double avg_cpu_iter_sort = TestSortPerformance(Bitonic<int>::cpu_sort_iterative, base, 5);
        double avg_gpu_sort = TestSortPerformance(Bitonic<int>::gpu_sort, base, 5);

        std::cout << std::setw(10) << size << std::setw(20) << std::fixed << std::setprecision(3)
                  << avg_std_sort << std::setw(20) << std::fixed << std::setprecision(3)
                  << avg_cpu_rec_sort << std::setw(20) << std::fixed << std::setprecision(3)
                  << avg_cpu_iter_sort << std::setw(20) << std::fixed << std::setprecision(3)
                  << avg_gpu_sort << std::endl;
    }
}
