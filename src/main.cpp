#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

#include "bitonic_sort.hpp"

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

    std::cout << "Int array:" << std::endl;
    std::vector<int> arr1 = {1, 4, 8, 2, 6, 9, 2, 24, 324312, 0, -6, 345, 345, 22, 123, 6578};
    Bitonic<int>::cpu_sort_recursive(arr1.begin(), arr1.end(), Direction::Ascending);
    for (auto& elem : arr1) { std::cout << elem << " "; }
    std::cout << std::endl;

    std::cout << "Double array:" << std::endl;
    std::vector<double> arr2 = {1.1,     4.7, 8,  2,   6,   9,  2.35, 24.4,
                                32431.2, 0,   -6, 345, 345, 22, 123,  6578};
    Bitonic<double>::cpu_sort_iterative(arr2.begin(), arr2.end(), Direction::Ascending);
    for (auto& elem : arr2) { std::cout << elem << " "; }
    std::cout << std::endl;

    return 0;
}

