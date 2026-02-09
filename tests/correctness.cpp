#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <numeric>

#include "tests.hpp"

bool IsSorted(const std::vector<int>& vector, Bitonic::Direction direction)
{
    if (direction == Bitonic::Direction::Descending)
    {
        return std::is_sorted(vector.begin(), vector.end(), std::greater<int>());
    }
    else
    {
        return std::is_sorted(vector.begin(), vector.end());
    }
}

void TestSortCorrectness(SortFunction sort_func, std::size_t start_size, std::size_t end_size,
                         const std::string& name)
{
    std::size_t total_tests = 0;
    std::size_t passed_tests = 0;

    for (std::size_t size = start_size; size <= end_size; size *= 2)
    {
        std::vector<int> base(size);
        RandFill(base);

        // Ascending test
        {
            std::vector<int> arr_ascending = base;
            sort_func(arr_ascending.begin(), arr_ascending.end(), Bitonic::Direction::Ascending);
            bool ok_asc = IsSorted(arr_ascending, Bitonic::Direction::Ascending);

            ++total_tests;
            if (!ok_asc)
            {
                std::cout << "FAILED TEST: " << name << " (size = " << size
                          << ", ASCENDING) NOT SORTED" << std::endl;
                std::cout << "RESULT: " << arr_ascending << std::endl;
            }
            else
            {
                ++passed_tests;
            }
        }

        // Descending test
        {
            std::vector<int> arr_descending = base;
            sort_func(arr_descending.begin(), arr_descending.end(), Bitonic::Direction::Descending);
            bool ok_desc = IsSorted(arr_descending, Bitonic::Direction::Descending);

            ++total_tests;
            if (!ok_desc)
            {
                std::cout << "FAILED TEST: " << name << " (size = " << size
                          << ", DESCENDING) NOT SORTED" << std::endl;
                std::cout << "RESULT: " << arr_descending << std::endl;
            }
            else
            {
                ++passed_tests;
            }
        }
    }

    if (passed_tests == total_tests)
    {
        std::cout << name << ": ALL TESTS PASSED (" << passed_tests << "/" << total_tests << ")"
                  << std::endl;
    }
    else
    {
        std::cout << name << ": FAILED (" << passed_tests << "/" << total_tests << " passed)"
                  << std::endl;
    }
}

void TestBitonicSortsCorrectness(
    std::size_t start_size, std::size_t end_size,
    const std::vector<std::pair<std::string, SortFunction>>& sort_functions)
{
    std::cout << "Running tests to verify sorting correctness..." << std::endl;

    if (!IsPowerOfTwo(start_size) || !IsPowerOfTwo(end_size))
    {
        std::cout << "Sizes should be powers of 2" << std::endl;
        return;
    }

    for (const auto& [name, sort_func] : sort_functions)
    {
        TestSortCorrectness(sort_func, start_size, end_size, name);
    }
}
