#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

#include "tests.hpp"

bool init_platform(std::size_t preferred_platform_idx = 0)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty())
    {
        std::cerr << "No available OpenCL platforms found!" << std::endl;
        return false;
    }

    if (preferred_platform_idx >= platforms.size())
    {
        std::cerr << "Platform with such idx does not exist" << std::endl;
        return false;
    }

    std::cout << "Available OpenCL platforms:" << std::endl;
    for (std::size_t i = 0; i < platforms.size(); i++)
    {
        const auto& platform = platforms[i];

        std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
        std::cout << "[" << i << "] " << platform_name << std::endl;
    }

    auto platform = cl::Platform::setDefault(platforms[preferred_platform_idx]);
    if (platforms[preferred_platform_idx] != platform)
    {
        std::cout << "Error setting default platform.\n";
        return false;
    }

    std::cout << "Selected platform [" << preferred_platform_idx << "]" << std::endl;

    return true;
}

void StdSort(std::vector<int>::iterator begin, std::vector<int>::iterator end, Bitonic::Direction direction)
{
    if (direction == Bitonic::Direction::Ascending) { std::sort(begin, end); }
    else
    {
        std::sort(begin, end, std::greater<int>());
    }
}

int main()
try
{
    if (!init_platform()) { return -1; }

    std::size_t correctness_start_size = 1ull << 10;
    std::size_t correctness_end_size = 1ull << 20;

    std::vector<std::pair<std::string, SortFunction>> functions_for_correctness_test = {
        {"std:sort", StdSort},
        {"bitonic (gpu)", Bitonic::gpu_sort},
        {"bitonic recursive (cpu)", Bitonic::cpu_sort_recursive},
        {"bitonic iterative (cpu)", Bitonic::cpu_sort_iterative}};

    TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size,
                                functions_for_correctness_test);

    std::size_t performance_start_size = 1ull << 6;
    std::size_t performance_end_size = 1ull << 27;

    std::vector<std::pair<std::string, SortFunction>> functions_for_test_1 = {
        {"std:sort", StdSort}, {"bitonic (gpu)", Bitonic::gpu_sort}};

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
