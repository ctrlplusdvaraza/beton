#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>

#include "interface.hpp"
#include "opencl_utils.hpp"
#include "stupid_cl.hpp"

namespace Bitonic
{

void gpu_stupid_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                     Direction direction)
{
    static bool is_platform_initialized = false;
    if (!is_platform_initialized) { details::init_platform(); }

    static bool are_kernels_compiled = false;
    static cl::Program bitonic_sort_program(kStupidClSrc);
    if (!are_kernels_compiled)
    {
        details::build_kernels(are_kernels_compiled, bitonic_sort_program);
    }


    cl::Context context = cl::Context::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::CommandQueue command_queue(context);
    cl::size_type max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();


    cl::Kernel kernel(bitonic_sort_program, "bitonic_step");

    std::size_t array_size = std::distance(begin, end);
    if (array_size <= 1) { return; }

    cl::Buffer array(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, array_size * sizeof(int),
                     &(*begin));

    const cl_int dir = (direction == Direction::Ascending) ? 0 : -1;

    for (cl_uint block_size = 2; block_size <= array_size; block_size *= 2)
    {
        for (cl_uint dist = block_size / 2; dist > 0; dist /= 2)
        {
            kernel.setArg(0, array);
            kernel.setArg(1, block_size);
            kernel.setArg(2, dist);
            kernel.setArg(3, dir);

            command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(array_size / 2),
                                               cl::NullRange);
            command_queue.finish();

        }
    }

    command_queue.enqueueReadBuffer(array, CL_TRUE, 0, array_size * sizeof(int), &(*begin));
}

} // namespace Bitonic
