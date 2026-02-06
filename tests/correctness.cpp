#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <numeric>

#include "tests.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& ostream, const std::vector<T>& vector)
{
    ostream << '[';
    for (std::size_t i = 0; i < vector.size(); ++i)
    {
        if (i > 0) { ostream << ", "; }
        ostream << vector[i];
    }
    ostream << ']';
    return ostream;
}

void RandFill(std::vector<int>& vector, int modulo)
{
    srand(time(nullptr));

    for (auto& elem : vector)
    {
        elem = rand() % modulo;
    }
}

bool IsSorted(const std::vector<int>& vector, Direction direction)
{
    if (direction == Direction::Descending)
    {
        return std::is_sorted(vector.begin(), vector.end(), std::greater<int>());
    }
    else
    {
        return std::is_sorted(vector.begin(), vector.end());
    }
}

void TestSortCorrectness(SortFunc sort_func, std::size_t start_size, std::size_t end_size,
                         const std::string& name)
{
    if (start_size % 2 != 0 || end_size % 2 != 0)
    {
        std::cout << "Sizes should be a multiple of 2" << std::endl;
        return;
    }

    std::size_t total_tests = 0;
    std::size_t passed_tests = 0;

    for (std::size_t size = start_size; size <= end_size; size *= 2)
    {
        std::vector<int> base(size);
        RandFill(base);

        // Ascending test
        {
            std::vector<int> arr_ascending = base;
            sort_func(arr_ascending.begin(), arr_ascending.end(), Direction::Ascending);
            bool ok_asc = IsSorted(arr_ascending, Direction::Ascending);

            ++total_tests;
            if (!ok_asc)
            {
                std::cout << "FAILED TEST: " << name << " (size = " << size
                          << ", ASCENDING) NOT SORTED" << std::endl;
            }
            else
            {
                ++passed_tests;
            }
        }

        // Descending test
        {
            std::vector<int> arr_descending = base;
            sort_func(arr_descending.begin(), arr_descending.end(), Direction::Descending);
            bool ok_desc = IsSorted(arr_descending, Direction::Descending);

            ++total_tests;
            if (!ok_desc)
            {
                std::cout << "FAILED TEST: " << name << " (size = " << size
                          << ", DESCENDING) NOT SORTED" << std::endl;
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

void TestBitonicSortsCorrectness(std::size_t start_size, std::size_t end_size)
{
    std::cout << "Running tests to verify sorting correctness..." << std::endl;

    TestSortCorrectness(Bitonic<int>::cpu_sort_iterative, start_size, end_size, "CPU iterative");
    TestSortCorrectness(Bitonic<int>::cpu_sort_recursive, start_size, end_size, "CPU recursive");
    TestSortCorrectness(Bitonic<int>::gpu_sort, start_size, end_size, "GPU");
}
