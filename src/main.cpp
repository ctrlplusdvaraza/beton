#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "bitonic_cl.hpp"
#include "bitonic_sort.hpp"

const std::string kOpenClBuildArgs = "-cl-std=CL3.0";

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

int main(int argc, char** argv)
try
{
    if (!init_platform()) { return -1; }

    // std::cout << "Using the following kernel code:\n" << kBitonicClSrc << std::endl;

    cl::Program bitonic_sort_program(kBitonicClSrc);

    try
    {
        bitonic_sort_program.build(kOpenClBuildArgs);
    }
    catch (...)
    {
        cl_int err = CL_SUCCESS;
        auto build_info = bitonic_sort_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&err);

        for (auto& kv : build_info) { std::cerr << kv.second << std::endl; }
        return -1;
    }

    std::cout << "Built kernel successfully!" << std::endl;

    auto bitonic_kernel = cl::KernelFunctor<cl::Buffer>(bitonic_sort_program, "bitonic_sort");

    // bitonicKernel(cl::EnqueueArgs);

    //
    // std::vector<int> arr1 = {2, 1, 4, 3, 6, 5, 8, 7, 20, 18, 19, 21, 23, 22, 34, 25};
    // std::cout << "Int array: ";
    // for (auto& elem : arr1) { std::cout << elem << " "; }
    // std::cout << std::endl;
    //
    // Bitonic<int>::cpu_sort_iterative(arr1.begin(), arr1.end(), Direction::Ascending);
    // for (auto& elem : arr1) { std::cout << elem << " "; }
    // std::cout << std::endl;
    //
    // std::cout << "Double array: ";
    // std::vector<double> arr2 =
    // {2.5, 1.1, 4.2, 3.4, 6.3, 5.7, 8.9, 7.3, 20.2, 18.5, 19.7, 21.1, 23.2, 22.7, 34.3, 25.2}; for
    // (auto& elem : arr2) { std::cout << elem << " "; } std::cout << std::endl;
    //
    // Bitonic<double>::cpu_sort_iterative(arr2.begin(), arr2.end(), Direction::Descending);
    // for (auto& elem : arr2) { std::cout << elem << " "; }
    // std::cout << std::endl;
    //
    return 0;
}
catch (const cl::Error& err)
{
    std::cerr << "OpenCL exception occured!\n" << err.err() << err.what() << std::endl;
    return -1;
}
