#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "advanced_cl.hpp"
#include "interface.hpp"
#include "opencl_utils.hpp"

#define ELEMS_PER_THREAD 4

namespace Bitonic
{

void pad_to_8(std::string& s, char pad = ' ') {
    if (s.size() % 16 != 0) {
        s.append(16 - (s.size() % 16), pad);
    }
}

void print_array(std::vector<int>::iterator begin, std::vector<int>::iterator end, std::string name="") {
    pad_to_8(name);
    std::cout << name << ": "; 
    for (auto to = begin; to != end; to++) {
        std::cout << *(to) << " ";
    }
    std::cout << "\n";
}

static void gpu_advanced_sort__(std::vector<int>::iterator begin, std::vector<int>::iterator end)
{
    static bool is_platform_initialized = false;
    if (!is_platform_initialized) { details::init_platform(); }

    static bool are_kernels_compiled = false;
    static cl::Program bitonic_sort_program(kAdvancedClSrc);
    if (!are_kernels_compiled)
    {
        details::build_kernels(are_kernels_compiled, bitonic_sort_program);
    }

    print_array(begin, end, "initial");

    cl::Kernel kernel_local_max_slm(bitonic_sort_program, "bitonic_local_max_slm");
    cl::Kernel kernel_global(bitonic_sort_program, "bitonic_step_global");
    cl::Kernel kernel_local_step(bitonic_sort_program, "bitonic_big_step_local");

    cl::Context context = cl::Context::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::CommandQueue command_queue(context);

    cl::size_type max_workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    max_workgroup_size = 2;


    cl_ulong local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    cl_uint max_slm_elems = local_mem_size / sizeof(int);
    cl_uint elems_per_thread = max_slm_elems / max_workgroup_size;


    std::size_t array_size = std::distance(begin, end);
    if (array_size <= 1) { return; }

    cl::Buffer array(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, array_size * sizeof(int),
                     &(*begin));

    cl::NDRange local_range(max_workgroup_size);
    cl::NDRange global_range_1(array_size / (ELEMS_PER_THREAD * 2));

    kernel_local_max_slm.setArg(0, array);
    kernel_local_max_slm.setArg(1, cl::Local(max_workgroup_size * ELEMS_PER_THREAD * 2 * sizeof(int)));
    command_queue.enqueueNDRangeKernel(kernel_local_max_slm, cl::NullRange, global_range_1, local_range);

    cl::NDRange global_range_2(array_size / 2);
    cl_uint elems_per_workgroup = max_workgroup_size * 2;


    for (cl_uint block_size = max_workgroup_size * (ELEMS_PER_THREAD * 2) * 2; block_size <= array_size; block_size *= 2)
    {   
        // for (cl_uint dist = block_size / 2; dist >= 1; dist /= 2) 
        for (cl_uint dist = block_size / 2; dist >= max_workgroup_size * ELEMS_PER_THREAD; dist /= 2) 
        {
            kernel_global.setArg(0, array);
            kernel_global.setArg(1, block_size);
            kernel_global.setArg(2, dist);
            kernel_global.setArg(3, /*ascending*/1);

            command_queue.enqueueNDRangeKernel(kernel_global, cl::NullRange, cl::NDRange(array_size / 2),
                                               local_range);
        }
        // continue;
    
        uint local_start_dist = max_workgroup_size / 2; // in int4
    
        std::cout << "block size : " << block_size << ", local_start_dist : " << local_start_dist << "\n";
        command_queue.enqueueReadBuffer(array, CL_TRUE, 0, array_size * sizeof(int), &(*begin));
        print_array(begin, end, "afterglobal");

        kernel_local_step.setArg(0, array);
        kernel_local_step.setArg(1, cl::Local(max_workgroup_size * ELEMS_PER_THREAD * sizeof(int)));
        kernel_local_step.setArg(2, block_size);
        kernel_local_step.setArg(3, local_start_dist);
        kernel_local_step.setArg(4, /*ascending*/1);
        command_queue.enqueueNDRangeKernel(kernel_local_step, cl::NullRange, cl::NDRange(array_size / ELEMS_PER_THREAD), local_range);

        command_queue.enqueueReadBuffer(array, CL_TRUE, 0, array_size * sizeof(int), &(*begin));
        print_array(begin, end, "afterlocal");
    }

    command_queue.enqueueReadBuffer(array, CL_TRUE, 0, array_size * sizeof(int), &(*begin));
}

void gpu_advanced_sort(std::vector<int>::iterator begin, std::vector<int>::iterator end, Direction direction) {
    gpu_advanced_sort__(begin, end);
    if (direction == Direction::Descending) 
    {
        std::reverse(begin, end);
    } 
}

} // namespace Bitonic
