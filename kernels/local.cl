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

__kernel void bitonic_step_global(__global int* array, const uint block_size, const uint dist,
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

__kernel void bitonic_step_local(__global int* g_data, __local int* l_data, const uint block_size,
                                 const uint dist, int direction)
{
    uint group_size = get_local_size(0);
    uint global_offset = get_group_id(0) * (group_size * 2);

    uint lid = get_local_id(0);

    l_data[lid] = g_data[global_offset + lid];
    l_data[lid + group_size] = g_data[global_offset + lid + group_size];

    barrier(CLK_LOCAL_MEM_FENCE);


    uint pos = get_local_id(0);

    uint block_index = pos / dist;
    pos += block_index * dist;

    uint partner = pos ^ dist;

    uint global_pos = pos + global_offset;

    int use_reversed_direction = (global_pos & block_size) != 0;
    int local_direction = direction ^ use_reversed_direction;

    COMP_AND_SWAP(l_data[pos], l_data[partner], local_direction);


    barrier(CLK_LOCAL_MEM_FENCE);

    g_data[global_offset + lid] = l_data[lid];
    g_data[global_offset + lid + group_size] = l_data[lid + group_size];
}

__kernel void bitonic_local(__global int* g_data, __local int* l_data, int direction)
{
    int group_size = get_local_size(0);
    int global_offset = get_group_id(0) * (group_size * 2);

    int lid = get_local_id(0);

    l_data[lid] = g_data[global_offset + lid];
    l_data[lid + group_size] = g_data[global_offset + lid + group_size];

    barrier(CLK_LOCAL_MEM_FENCE);

    int limit = group_size * 2;

    for (int block_size = 2; block_size <= limit; block_size *= 2)
    {
        for (int dist = block_size / 2; dist > 0; dist /= 2)
        {
            uint pos = lid;

            uint block_index = pos / dist;
            pos += block_index * dist;

            uint partner = pos ^ dist;

            uint global_pos = pos + global_offset;

            uint use_reversed_direction = (global_pos & block_size) != 0;
            int local_direction = direction ^ use_reversed_direction;

            COMP_AND_SWAP(l_data[pos], l_data[partner], local_direction);

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    g_data[global_offset + lid] = l_data[lid];
    g_data[global_offset + lid + group_size] = l_data[lid + group_size];
}

__kernel void bitonic_big_step_local(__global int* g_data, __local int* l_data,
                                     const uint block_size, int direction)
{
    int group_size = get_local_size(0);
    int global_offset = get_group_id(0) * (group_size * 2);

    int lid = get_local_id(0);

    l_data[lid] = g_data[global_offset + lid];
    l_data[lid + group_size] = g_data[global_offset + lid + group_size];

    barrier(CLK_LOCAL_MEM_FENCE);

    int limit = group_size * 2;

    for (int dist = block_size / 2; dist > 0; dist /= 2)
    {
        uint pos = lid;

        uint block_index = pos / dist;
        pos += block_index * dist;

        uint partner = pos ^ dist;

        uint global_pos = pos + global_offset;

        uint use_reversed_direction = (global_pos & block_size) != 0;
        int local_direction = direction ^ use_reversed_direction;

        COMP_AND_SWAP(l_data[pos], l_data[partner], local_direction);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    g_data[global_offset + lid] = l_data[lid];
    g_data[global_offset + lid + group_size] = l_data[lid + group_size];
}
