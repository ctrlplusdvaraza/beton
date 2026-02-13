#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>
#include <type_traits>

#include "bitonic_cl.hpp"
#include "interface.hpp"
#include "opencl_utils.hpp"

namespace Bitonic
{

void gpu_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end, Direction inp_direction)
{
    int direction = (inp_direction == Direction::Ascending) ? 0 : -1;

    static bool is_platform_initialized = false;
    if (!is_platform_initialized) { details::init_platform(); }

    static bool are_kernels_compiled = false;
    static cl::Program bitonic_sort_program(kBitonicClSrc);
    if (!are_kernels_compiled)
    {
        details::build_kernels(are_kernels_compiled, bitonic_sort_program);
    }


    cl::Context context = cl::Context::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::CommandQueue command_queue(context);
    cl::size_type max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();


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


    std::size_t array_size = std::distance(begin, end);
    if (array_size <= 1) { return; }

    cl::Buffer array(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, array_size * sizeof(int),
                     &(*begin));

    std::size_t local_size = max_workgroup_size; //workgroup size
    std::size_t global_size = array_size / 8; // work items count
    std::size_t ldata_size = 2 * sizeof(cl_int4) * local_size;

    // INIT bsort
    auto ev = bitonic_sort_kernel(
        cl::EnqueueArgs(command_queue, cl::NDRange(global_size), cl::NDRange(local_size)), array,
        cl::Local(ldata_size));

    ev.wait();

    std::size_t num_stages = global_size / local_size; // num stages = workgroup count 

    for (std::size_t high_stage = 2; high_stage < num_stages; high_stage *= 2)
    {
        for (std::size_t stage = high_stage; stage > 1; stage /= 2)
        {
            // if high stage fit in local memory -> use bsort_stage_n_local, else -> bsort_stage_n_global
            bitonic_stage_n_kernel(
                cl::EnqueueArgs(command_queue, cl::NDRange(global_size), cl::NDRange(local_size)),
                array, cl::Local(ldata_size), stage, high_stage);
        }

        bitonic_stage_0_kernel(
            cl::EnqueueArgs(command_queue, cl::NDRange(global_size), cl::NDRange(local_size)),
            array, cl::Local(ldata_size), high_stage);
    }

    for (std::size_t stage = num_stages; stage > 1; stage /= 2)
    {
        bitonic_merge_kernel(
            cl::EnqueueArgs(command_queue, cl::NDRange(global_size), cl::NDRange(local_size)),
            array, cl::Local(ldata_size), stage, direction);
    }

    bitonic_merge_last_kernel(
        cl::EnqueueArgs(command_queue, cl::NDRange(global_size), cl::NDRange(local_size)), array,
        cl::Local(ldata_size), direction);


    command_queue.enqueueReadBuffer(array, CL_TRUE, 0, array_size * sizeof(int), &(*begin));
}

} // namespace Bitonic
