#include <CL/cl.h>
#include <iostream>

#include <iostream>
#include "interface.hpp"

bool IsOpenClAvailable()
{
    cl_uint n = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &n);
    return (err == CL_SUCCESS) && (n > 0);
}

int main()
{
    if (!IsOpenClAvailable())
    {
        std::cerr << "OpenCL is not available!!!" << std::endl;
        return 1;
    }
    std::cout << "OpenCL is working, lets go boiss" << std::endl;

    std::vector<int> arr = {1, 4, 8, 2, 6, 9, 2, 24, 324312, 0, -6, 345, 345, 22, 123, 6578};
    Bitonic::cpu_sort(arr.begin(), arr.end(), Bitonic::Direction::Ascending);
    for (auto& elem : arr)
    {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    return 0;
}
