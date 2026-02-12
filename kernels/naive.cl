/*
The most naive implementation of bitonic sort.

In bitonic_step_naive work items most of the time *idle* (do nothing).
In bitonic_step_better work items most of the time *stalled* (waiting for data from memory).

Conclusion: To improve performance, we need to reduce memory latency (for example, by using local
memory)
*/

#define SWAP(a, b)                                                                                 \
    {                                                                                              \
        int t = (a);                                                                               \
        (a) = (b);                                                                                 \
        (b) = t;                                                                                   \
    }

#define COMP_AND_SWAP(A, B, DIR)                                                                   \
    {                                                                                              \
        if (((A) > (B)) == ((DIR) == 1)) { SWAP(A, B); }                                           \
    }


/* 1 = ASCENDING, 0 = DESCENDING */
__kernel void bitonic_step_naive(__global int* array, const uint block_size, const uint dist,
                                 int direction)
{
    uint pos = get_global_id(0);

    uint partner = pos ^ dist;
    if (partner < pos) { return; }

    int use_reversed_direction = (pos & block_size) != 0;
    int local_direction = direction ^ use_reversed_direction;

    COMP_AND_SWAP(array[pos], array[partner], local_direction);
}

__kernel void bitonic_step_better(__global int* array, const uint block_size, const uint dist,
                                  int direction)
{
    uint pos = get_global_id(0);

    uint block_index = pos / dist;
    pos += block_index * dist;

    uint partner = pos ^ dist;

    int use_reversed_direction = (pos & block_size) != 0;
    int local_direction = direction ^ use_reversed_direction;

    COMP_AND_SWAP(array[pos], array[partner], local_direction);
}

__kernel void bitonic_step_best(__global int* array, const uint block_size,
                                const uint block_size_log, const uint dist_log, int direction)
{
    uint pos = get_global_id(0);

    uint dist = 1ul << dist_log;

    uint block_index = pos >> dist_log;
    pos += block_index << dist_log;

    uint partner = pos ^ dist;

    int use_reversed_direction = (pos & block_size) != 0;
    int local_direction = direction ^ use_reversed_direction;

    COMP_AND_SWAP(array[pos], array[partner], local_direction);
}
