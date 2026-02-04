#pragma once

#include <CL/opencl.hpp>
#include <type_traits>

#include "interface.hpp"

template <typename T>
void Bitonic<T>::gpu_sort(iter begin, iter end, Direction direction)
{
    std::ptrdiff_t size = end - begin;
    if (size <= 1) { return; }


}

