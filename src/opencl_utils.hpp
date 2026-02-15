#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#include <CL/opencl.hpp>

namespace Bitonic::details
{

inline bool init_platform(std::size_t preferred_platform_idx = 0)
{
    static bool is_platform_initialized = false;

    if (is_platform_initialized) { return true; }

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

    is_platform_initialized = true;

    return true;
}

inline void build_kernels(bool& are_kernels_compiled, cl::Program& program)
{
    static const std::string kOpenClBuildArgs = "-cl-std=CL3.0";

    if (are_kernels_compiled) { return; }

    try
    {
        program.build(kOpenClBuildArgs.c_str());
        are_kernels_compiled = true;
    }
    catch (...)
    {
        cl_int err = CL_SUCCESS;
        auto build_info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&err);

        for (auto& kv : build_info)
        {
            std::cerr << kv.second << std::endl;
        }
        throw;
    }
}

} // namespace Bitonic::details
