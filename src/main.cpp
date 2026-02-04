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

    std::vector<int> arr1 = {2, 1, 4, 3, 6, 5, 8, 7, 20, 18, 19, 21, 23, 22, 34, 25};
    std::cout << "Int array: ";
    for (auto& elem : arr1) { std::cout << elem << " "; }
    std::cout << std::endl;

    Bitonic<int>::cpu_sort_iterative(arr1.begin(), arr1.end(), Direction::Ascending);
    std::cout << std::endl;

    std::cout << "Double array: ";
    std::vector<double> arr2 = {2.5, 1.1, 4.2, 3.4, 6.3, 5.7, 8.9, 7.3, 20.2, 18.5, 19.7, 21.1, 23.2, 22.7, 34.3, 25.2};
    for (auto& elem : arr2) { std::cout << elem << " "; }
    std::cout << std::endl;

    Bitonic<double>::cpu_sort_iterative(arr2.begin(), arr2.end(), Direction::Descending);
    std::cout << std::endl;

    return 0;
}

