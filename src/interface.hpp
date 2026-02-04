#include <vector>

class Bitonic
{
  public:
    enum class Direction
    {
        Descending = 0,
        Ascending = 1
    };

  private:
    using iter = std::vector<int>::iterator;

  public:
    static void cpu_sort(iter begin, iter end, Direction direction);
    static void gpu_sort(iter begin, iter end, Direction direction);

  private:
    static void cpu_merge(iter begin, iter end, Direction direction);
    static void gpu_merge(iter begin, iter end, Direction direction);

    static void cpu_comp_and_swap(iter first, iter second, Direction direction);
    static void gpu_comp_and_swap(iter first, iter second, Direction direction);
};
