#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <cmath>
#include <iostream>

#include "interface.hpp"
#include "local_cl.hpp"
#include "opencl_utils.hpp"

namespace Bitonic
{

void gpu_local_sort_naive(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                          Direction direction)
{
    static bool is_platform_initialized = false;
    if (!is_platform_initialized) { details::init_platform(); }

    static bool are_kernels_compiled = false;
    static cl::Program bitonic_sort_program(kLocalClSrc);
    if (!are_kernels_compiled)
    {
        details::build_kernels(are_kernels_compiled, bitonic_sort_program);
    }

    cl::Kernel kernel_global(bitonic_sort_program, "bitonic_step_global");
    cl::Kernel kernel_local(bitonic_sort_program, "bitonic_step_local");

    cl::Context context = cl::Context::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::CommandQueue command_queue(context);

    cl::size_type max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();


    std::size_t array_size = std::distance(begin, end);
    if (array_size <= 1) { return; }

    cl::Buffer array(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, array_size * sizeof(int),
                     &(*begin));

    const cl_int dir = (direction == Direction::Ascending) ? 1 : 0;

    cl::NDRange global_range(array_size / 2);
    cl::NDRange local_range(max_workgroup_size);

    for (cl_uint block_size = 2; block_size <= array_size; block_size *= 2)
    {
        for (cl_uint dist = block_size / 2; dist > 0; dist /= 2)
        {
            if (dist <= max_workgroup_size)
            {
                kernel_local.setArg(0, array);
                kernel_local.setArg(1, cl::Local(max_workgroup_size * 2 * sizeof(int)));
                kernel_local.setArg(2, block_size);
                kernel_local.setArg(3, dist);
                kernel_local.setArg(4, dir);

                command_queue.enqueueNDRangeKernel(kernel_local, cl::NullRange, global_range,
                                                   local_range);
            }

            else
            {
                kernel_global.setArg(0, array);
                kernel_global.setArg(1, block_size);
                kernel_global.setArg(2, dist);
                kernel_global.setArg(3, dir);

                command_queue.enqueueNDRangeKernel(kernel_global, cl::NullRange, global_range,
                                                   local_range);
            }
        }
    }

    command_queue.enqueueReadBuffer(array, CL_TRUE, 0, array_size * sizeof(int), &(*begin));
}

/* --------------------------------------------------------------------------------------------- */

void gpu_local_sort_better(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                           Direction direction)
{
    static bool is_platform_initialized = false;
    if (!is_platform_initialized) { details::init_platform(); }

    static bool are_kernels_compiled = false;
    static cl::Program bitonic_sort_program(kLocalClSrc);
    if (!are_kernels_compiled)
    {
        details::build_kernels(are_kernels_compiled, bitonic_sort_program);
    }

    cl::Kernel kernel_global(bitonic_sort_program, "bitonic_step_global");
    cl::Kernel kernel_local(bitonic_sort_program, "bitonic_local");
    cl::Kernel kernel_local_step(bitonic_sort_program, "bitonic_step_local");

    cl::Context context = cl::Context::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::CommandQueue command_queue(context);

    cl::size_type max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();


    std::size_t array_size = std::distance(begin, end);
    if (array_size <= 1) { return; }

    cl::Buffer array(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, array_size * sizeof(int),
                     &(*begin));

    const cl_int dir = (direction == Direction::Ascending) ? 1 : 0;

    cl::NDRange global_range(array_size / 2);
    cl::NDRange local_range(max_workgroup_size);

    cl_uint elems_per_workgroup = max_workgroup_size * 2;


    kernel_local.setArg(0, array);
    kernel_local.setArg(1, cl::Local(elems_per_workgroup * sizeof(int)));
    kernel_local.setArg(2, dir);

    command_queue.enqueueNDRangeKernel(kernel_local, cl::NullRange, global_range, local_range);


    for (cl_uint block_size = elems_per_workgroup * 2; block_size <= array_size; block_size *= 2)
    {
        for (cl_uint dist = block_size / 2; dist > 0; dist /= 2)
        {
            if (dist <= max_workgroup_size)
            {
                kernel_local_step.setArg(0, array);
                kernel_local_step.setArg(1, cl::Local(elems_per_workgroup * sizeof(int)));
                kernel_local_step.setArg(2, block_size);
                kernel_local_step.setArg(3, dist);
                kernel_local_step.setArg(4, dir);

                command_queue.enqueueNDRangeKernel(kernel_local_step, cl::NullRange, global_range,
                                                   local_range);
            }

            else
            {
                kernel_global.setArg(0, array);
                kernel_global.setArg(1, block_size);
                kernel_global.setArg(2, dist);
                kernel_global.setArg(3, dir);

                command_queue.enqueueNDRangeKernel(kernel_global, cl::NullRange, global_range,
                                                   local_range);
            }
        }
    }

    command_queue.enqueueReadBuffer(array, CL_TRUE, 0, array_size * sizeof(int), &(*begin));
}

/* --------------------------------------------------------------------------------------------- */

void gpu_local_sort_best(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                         Direction direction)
{
    static bool is_platform_initialized = false;
    if (!is_platform_initialized) { details::init_platform(); }

    static bool are_kernels_compiled = false;
    static cl::Program bitonic_sort_program(kLocalClSrc);
    if (!are_kernels_compiled)
    {
        details::build_kernels(are_kernels_compiled, bitonic_sort_program);
    }

    cl::Kernel kernel_global(bitonic_sort_program, "bitonic_step_global");
    cl::Kernel kernel_local(bitonic_sort_program, "bitonic_local");
    cl::Kernel kernel_local_step(bitonic_sort_program, "bitonic_big_step_local");

    cl::Context context = cl::Context::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::CommandQueue command_queue(context);

    cl::size_type max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();


    std::size_t array_size = std::distance(begin, end);
    if (array_size <= 1) { return; }

    cl::Buffer array(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, array_size * sizeof(int),
                     &(*begin));

    const cl_int dir = (direction == Direction::Ascending) ? 1 : 0;

    cl::NDRange global_range(array_size / 2);
    cl::NDRange local_range(max_workgroup_size);

    cl_uint elems_per_workgroup = max_workgroup_size * 2;


    kernel_local.setArg(0, array);
    kernel_local.setArg(1, cl::Local(elems_per_workgroup * sizeof(int)));
    kernel_local.setArg(2, dir);

    command_queue.enqueueNDRangeKernel(kernel_local, cl::NullRange, global_range, local_range);


    for (cl_uint block_size = elems_per_workgroup * 2; block_size <= array_size; block_size *= 2)
    {
        for (cl_uint dist = block_size / 2; dist > max_workgroup_size; dist /= 2)
        {
            kernel_global.setArg(0, array);
            kernel_global.setArg(1, block_size);
            kernel_global.setArg(2, dist);
            kernel_global.setArg(3, dir);

            command_queue.enqueueNDRangeKernel(kernel_global, cl::NullRange, global_range,
                                               local_range);
        }

        kernel_local_step.setArg(0, array);
        kernel_local_step.setArg(1, cl::Local(elems_per_workgroup * sizeof(int)));
        kernel_local_step.setArg(2, block_size);
        kernel_local_step.setArg(3, static_cast<uint>(max_workgroup_size));
        kernel_local_step.setArg(4, dir);

        command_queue.enqueueNDRangeKernel(kernel_local_step, cl::NullRange, global_range,
                                           local_range);
    }

    command_queue.enqueueReadBuffer(array, CL_TRUE, 0, array_size * sizeof(int), &(*begin));
}

} // namespace Bitonic
