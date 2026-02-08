#include "tests.hpp"

int main()
{
    std::size_t correctness_start_size = 1ull << 10;
    std::size_t correctness_end_size = 1ull << 20;
    TestBitonicSortsCorrectness(correctness_start_size, correctness_end_size);

    std::size_t performance_start_size = 1ull << 6;
    std::size_t performance_end_size = 1ull << 27;
    CompareBitonicSortsPerformance(performance_start_size, performance_end_size);
}
