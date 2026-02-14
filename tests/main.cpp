#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

#include "tests.hpp"

void StdSort(std::vector<int>::iterator begin, std::vector<int>::iterator end,
             Bitonic::Direction direction)
{
    std::sort(begin, end);
}

// int main()
// try
// {
//     std::size_t correctness_start_size = 1ull << 10;
//     std::size_t correctness_end_size = 1ull << 20;

//     std::vector<std::pair<std::string, SortFunction>> functions_for_correctness_test = {
//         {"bitonic (gpu)", Bitonic::gpu_sort},
//         {"bitonic naive (gpu)", Bitonic::gpu_naive_sort},
//         {"bitonic recursive (cpu)", Bitonic::cpu_sort_recursive},
//         {"bitonic iterative (cpu)", Bitonic::cpu_sort_iterative}};

//     TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size,
//                                 functions_for_correctness_test);

//     std::size_t performance_start_size = 1ull << 6;
//     std::size_t performance_end_size = 1ull << 27;

//     std::vector<std::pair<std::string, SortFunction>> functions_for_test_1 = {
//         {"std:sort", StdSort}, {"bitonic (gpu)", Bitonic::gpu_sort}, {"naive (gpu)",
//         Bitonic::gpu_naive_sort}};

//     CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_test_1);

//     std::vector<std::pair<std::string, SortFunction>> functions_for_test_2 = {
//         {"bitonic recursive (cpu)", Bitonic::cpu_sort_recursive},
//         {"bitonic iterative (cpu)", Bitonic::cpu_sort_iterative}};

//     CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_test_2);
// }
// catch (const cl::Error& err)
// {
//     std::cerr << "OpenCL exception occured!\n" << err.err() << err.what() << std::endl;
//     return -1;
// }

int main()
try
{
    std::size_t correctness_start_size = 1ull << 10;
    std::size_t correctness_end_size = 1ull << 20;

    std::vector<std::pair<std::string, SortFunction>> functions_for_correctness_test = 
    {
        // {"bitonic naive (gpu)", Bitonic::gpu_naive_sort},
        // {"bitonic better (gpu)", Bitonic::gpu_naive_sort_better},
        {"naive best (gpu)", Bitonic::gpu_naive_sort_best},
        {"local sort naive (gpu)", Bitonic::gpu_local_sort_naive},
        {"local sort better (gpu)", Bitonic::gpu_local_sort_better},
        {"local sort best (gpu)", Bitonic::gpu_local_sort_best},
        {"local sort advanced (gpu)", Bitonic::gpu_advanced_sort},
        //{"gpu_sort (gpu)", Bitonic::gpu_sort}
    };

    TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size,
                               functions_for_correctness_test);

    std::size_t performance_start_size = 1ull << 20;
    std::size_t performance_end_size = 1ull << 25;

    std::vector<std::pair<std::string, SortFunction>> functions_for_test_1 = {
        // {"std::sort", StdSort},
        // {"bitonic iterative (cpu)", Bitonic::cpu_sort_iterative_1},
        // {"naive (gpu)", Bitonic::gpu_naive_sort},
        // {"naive better (gpu)", Bitonic::gpu_naive_sort_better},
        {"naive best (gpu)", Bitonic::gpu_naive_sort_best},
        {"local sort naive (gpu)", Bitonic::gpu_local_sort_naive},
        {"local sort better (gpu)", Bitonic::gpu_local_sort_better},
        {"local sort best (gpu)", Bitonic::gpu_local_sort_best},
        {"local sort advanced (gpu)", Bitonic::gpu_advanced_sort},
        // {"gpu_sort (gpu)", Bitonic::gpu_sort}
    };

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_test_1);
}
catch (const cl::Error& err)
{
    std::cerr << "OpenCL exception occured!\n" << err.err() << err.what() << std::endl;
    return -1;
}
