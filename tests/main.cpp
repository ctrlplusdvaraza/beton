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
        {"bitonic (gpu)", Bitonic::gpu_sort},
        {"bitonic stupid (gpu)", Bitonic::gpu_stupid_sort},
        {"bitonic recursive (cpu)", Bitonic::cpu_sort_recursive},
        {"bitonic iterative (cpu)", Bitonic::cpu_sort_iterative}};

    TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size,
                                functions_for_correctness_test);

    std::size_t performance_start_size = 1ull << 6;
    std::size_t performance_end_size = 1ull << 27;

    std::vector<std::pair<std::string, SortFunction>> functions_for_test_1 = {
        {"std:sort", StdSort}, {"bitonic (gpu)", Bitonic::gpu_sort}, {"stupid (gpu)",
        Bitonic::gpu_stupid_sort}};

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_test_1);

    std::vector<std::pair<std::string, SortFunction>> functions_for_test_2 = {
        {"bitonic recursive (cpu)", Bitonic::cpu_sort_recursive},
        {"bitonic iterative (cpu)", Bitonic::cpu_sort_iterative}};

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_test_2);
}
catch (const cl::Error& err)
{
    std::cerr << "OpenCL exception occured!\n" << err.err() << err.what() << std::endl;
    return -1;
}

// int main()
// try
// {
//     // std::size_t correctness_start_size = 1ull << 10;
//     // std::size_t correctness_end_size = 1ull << 20;

//     // std::vector<std::pair<std::string, SortFunction>> functions_for_correctness_test = {
//     //     {"bitonic (gpu)", Bitonic::gpu_sort},
//     //     {"bitonic stupid (gpu)", Bitonic::gpu_stupid_sort},
//     //     {"bitonic recursive (cpu)", Bitonic::cpu_sort_recursive},
//     //     {"bitonic iterative (cpu)", Bitonic::cpu_sort_iterative}};

//     // TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size,
//     //                             functions_for_correctness_test);

//     std::size_t performance_start_size = 1ull << 25;
//     std::size_t performance_end_size = 1ull << 25;

//     std::vector<std::pair<std::string, SortFunction>> functions_for_test_1 = {
//         {"stupid (gpu)", Bitonic::gpu_stupid_sort}};

//     CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_test_1);
// }
// catch (const cl::Error& err)
// {
//     std::cerr << "OpenCL exception occured!\n" << err.err() << err.what() << std::endl;
//     return -1;
// }
