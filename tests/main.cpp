#include "tests.hpp"

int main()
{
    std::size_t correctness_start_size = 1ull << 3;
    std::size_t correctness_end_size = 1ull << 20;
    TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size);

    std::size_t performance_start_size = 1ull << 3;
    std::size_t performance_end_size = 1ull << 20;
    // CompareBitonicSortsPerformance(START_SIZE, END_SIZE);

    TestSortPerformance(Bitonic<int>::cpu_sort_recursive, performance_start_size, performance_end_size, "cpu_sort_recursive");
    TestSortPerformance(Bitonic<int>::cpu_sort_iterative, performance_start_size, performance_end_size, "cpu_sort_iterative");
}
