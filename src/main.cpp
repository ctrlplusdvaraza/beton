#include <CL/cl.h>
#include <iostream>

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

    return 0;
}
