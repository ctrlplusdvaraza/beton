#include <iostream>
#include <vector>

enum class Direction
{
    Descending = 0,
    Ascending = 1
};

using iter = std::vector<int>::iterator;

void CompAndSwap(iter first, iter second, Direction direction)
{
    if ((direction == Direction::Ascending && *first > *second) ||
        (direction == Direction::Descending && *first < *second))
    {
        std::iter_swap(first, second);
    }
}

void BitonicMerge(iter begin, iter end, Direction direction)
{
    ptrdiff_t size = end - begin;

    if (size <= 1)
    {
        return;
    }

    ptrdiff_t half = size / 2;

    for (ptrdiff_t i = 0; i < half; ++i)
    {
        CompAndSwap(begin + i, begin + half + i, direction);
    }

    BitonicMerge(begin, begin + half, direction);
    BitonicMerge(begin + half, end, direction);
}

void BitonicSort(iter begin, iter end, Direction direction)
{
    ptrdiff_t size = end - begin;

    if (size <= 1)
    {
        return;
    }

    ptrdiff_t half = size / 2;

    BitonicSort(begin, begin + half, Direction::Ascending);
    BitonicSort(begin + half, end, Direction::Descending);

    BitonicMerge(begin, end, direction);
}


int main()
{
    std::vector<int> arr = {1, 2, 65, 4534, 134, 23, 234, 8};

    BitonicSort(arr.begin(), arr.end(), Direction::Ascending);

    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}
