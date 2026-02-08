#include "tests.hpp"

void StdSort(std::vector<int>::iterator begin, std::vector<int>::iterator end, Direction direction)
{
    std::sort(begin, end);
}

int main()
{
    std::size_t correctness_start_size = 1ull << 10;
    std::size_t correctness_end_size = 1ull << 20;
    TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size);

    std::size_t performance_start_size = 1ull << 6;
    std::size_t performance_end_size = 1ull << 27;

    std::vector<std::pair<std::string, SortFunction>> functions_for_test_1 = {
        {"std:sort", StdSort}, {"bitonic (gpu)", Bitonic<int>::gpu_sort}};

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_test_1);

    std::vector<std::pair<std::string, SortFunction>> functions_for_test_2 = {
        {"bitonic recursive (cpu)", Bitonic<int>::cpu_sort_recursive},
        {"bitonic iterative (cpu)", Bitonic<int>::cpu_sort_iterative}};

    CompareSortsPerformance(performance_start_size, performance_end_size, functions_for_test_2);
}
