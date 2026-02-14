#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

#include "tests.hpp"

void StdSort(std::vector<int>::iterator begin, std::vector<int>::iterator end,
             Bitonic::Direction direction)
{
    std::sort(begin, end);
}

int main()
try
{
    std::size_t correctness_start_size = 1ull << 10;
    std::size_t correctness_end_size = 1ull << 20;

    std::vector<std::pair<std::string, SortFunction>> functions_for_correctness_test = {
        {"recursive (cpu)", Bitonic::cpu_sort_recursive},

        {"iterative 0 (cpu)", Bitonic::cpu_sort_iterative_0},
        {"iterative 1 (cpu)", Bitonic::cpu_sort_iterative_1},
        {"iterative 2 (cpu)", Bitonic::cpu_sort_iterative_2},
        {"iterative 3 (cpu)", Bitonic::cpu_sort_iterative_3},

        {"naive (gpu)", Bitonic::gpu_naive_sort},
        {"naive better (gpu)", Bitonic::gpu_naive_sort_better},
        {"naive best (gpu)", Bitonic::gpu_naive_sort_best},

        {"local naive (gpu)", Bitonic::gpu_local_sort_naive},
        {"local better (gpu)", Bitonic::gpu_local_sort_better},
        {"local best (gpu)", Bitonic::gpu_local_sort_best},
    };

    TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size,
                                functions_for_correctness_test);

    std::size_t performance_start_size = 1ull << 20;
    std::size_t performance_end_size = 1ull << 25;

    std::vector<std::pair<std::string, SortFunction>> functions_for_performance_test_1 = {
        {"std::sort", StdSort},
        {"local best (gpu)", Bitonic::gpu_local_sort_best},
    };

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_performance_test_1);

    std::vector<std::pair<std::string, SortFunction>> functions_for_performance_test_2 = {
        {"local naive (gpu)", Bitonic::gpu_local_sort_naive},
        {"local better (gpu)", Bitonic::gpu_local_sort_better},
        {"local best (gpu)", Bitonic::gpu_local_sort_best},
    };

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_performance_test_2);

    std::vector<std::pair<std::string, SortFunction>> functions_for_performance_test_3 = {
        {"naive (gpu)", Bitonic::gpu_naive_sort},
        {"naive better (gpu)", Bitonic::gpu_naive_sort_better},
        {"naive best (gpu)", Bitonic::gpu_naive_sort_best},
    };

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_performance_test_3);

    std::vector<std::pair<std::string, SortFunction>> functions_for_performance_test_4 = {
        {"recursive (cpu)", Bitonic::cpu_sort_recursive},
        {"iterative 0 (cpu)", Bitonic::cpu_sort_iterative_0},
        {"iterative 1 (cpu)", Bitonic::cpu_sort_iterative_1},
        {"iterative 2 (cpu)", Bitonic::cpu_sort_iterative_2},
        {"iterative 3 (cpu)", Bitonic::cpu_sort_iterative_3}
    };

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_performance_test_4);
}
catch (const cl::Error& err)
{
    std::cerr << "OpenCL exception occured!\n" << err.err() << err.what() << std::endl;
    return -1;
}

