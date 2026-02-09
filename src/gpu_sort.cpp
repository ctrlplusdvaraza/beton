#include <CL/opencl.hpp>
#include <iostream>
#include <type_traits>

#include "bitonic_cl.hpp"
#include "interface.hpp"

namespace Bitonic
{

const std::string kOpenClBuildArgs = "-cl-std=CL3.0";

void gpu_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end, Direction direction)
{
    static bool are_kernels_compiled = false;
    static cl::Program bitonic_sort_program(kBitonicClSrc);

    try
    {
        bitonic_sort_program.build(kOpenClBuildArgs);
        are_kernels_compiled = true;
    }
    catch (...)
    {
        cl_int err = CL_SUCCESS;
        auto build_info = bitonic_sort_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&err);

        for (auto& kv : build_info)
        {
            std::cerr << kv.second << std::endl;
        }
        throw;
    }

    auto bitonic_sort_kernel =
        cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg>(bitonic_sort_program, "bsort_init");

    auto bitonic_stage_0_kernel = cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, cl_uint>(
        bitonic_sort_program, "bsort_stage_0");

    auto bitonic_stage_n_kernel =
        cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, cl_uint, cl_uint>(bitonic_sort_program,
                                                                           "bsort_stage_n");

    auto bitonic_merge_kernel = cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, cl_uint, cl_uint>(
        bitonic_sort_program, "bsort_merge");

    auto bitonic_merge_last_kernel = cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, cl_uint>(
        bitonic_sort_program, "bsort_merge_last");

    auto ctx = cl::Context::getDefault();
    auto command_queue = cl::CommandQueue(ctx);
    auto device = cl::Device::getDefault();
    auto max_workgroup_sz = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    cl::Buffer buf(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                   std::distance(begin, end) * sizeof(float), &(*begin));

    std::size_t local_sz = max_workgroup_sz;
    std::size_t global_sz = (std::distance(begin, end)) / 8;
    std::size_t ldata_sz = 2 * sizeof(cl_float4) * local_sz;

    auto ev = bitonic_sort_kernel(
        cl::EnqueueArgs(command_queue, cl::NDRange(global_sz), cl::NDRange(local_sz)), buf,
        cl::Local(ldata_sz));

    ev.wait();

    std::size_t num_stages = global_sz / local_sz;

    for (std::size_t high_stage = 2; high_stage < num_stages; high_stage *= 2)
    {
        for (std::size_t stage = high_stage; stage > 1; stage /= 2)
        {
            bitonic_stage_n_kernel(
                cl::EnqueueArgs(command_queue, cl::NDRange(global_sz), cl::NDRange(local_sz)), buf,
                cl::Local(ldata_sz), stage, high_stage);
        }

        bitonic_stage_0_kernel(
            cl::EnqueueArgs(command_queue, cl::NDRange(global_sz), cl::NDRange(local_sz)), buf,
            cl::Local(ldata_sz), high_stage);
    }

    for (std::size_t stage = num_stages; stage > 1; stage /= 2)
    {
        bitonic_merge_kernel(
            cl::EnqueueArgs(command_queue, cl::NDRange(global_sz), cl::NDRange(local_sz)), buf,
            cl::Local(ldata_sz), stage, static_cast<int>(direction));
    }

    bitonic_merge_last_kernel(
        cl::EnqueueArgs(command_queue, cl::NDRange(global_sz), cl::NDRange(local_sz)), buf,
        cl::Local(ldata_sz), static_cast<int>(direction));


    command_queue.enqueueReadBuffer(buf, CL_TRUE, 0, std::distance(begin, end) * sizeof(float),
                                    &(*begin));
}

} // namespace Bitonic
