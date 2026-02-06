#include "interface.hpp"
#define CL_HPP_ENABLE_EXCEPTIONS

#include "bitonic_sort.hpp"
#include "gpu_sort.hpp"
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

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

    return 0;
}
catch (const cl::Error& err)
{
    std::cerr << "OpenCL exception occured!\n" << err.err() << err.what() << std::endl;
    return -1;
}
