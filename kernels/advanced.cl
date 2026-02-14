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

#define ELEMS_PER_THREAD 4

static inline void int4_compare_swap(int4* a, int4* b, int dir/*asc=0, des=-1*/)
{
    dir *= -1;
    int4 add3 = (int4)(4, 5, 6, 7);
    int4 ta = *a;
    int4 comp = ((*a < *b) ^ dir) * 4 + add3;
    *a = shuffle2(*a, *b, as_uint4(comp));
    *b = shuffle2(*b, ta, as_uint4(comp));
}

static inline void int4_sort(int4* input, int dir/*asc=0, des=-1*/) 
{
    dir *= -1;
    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 mask3 = (uint4)(3, 2, 1, 0);

    int4 add1 = (int4)(1, 1, 3, 3);
    int4 add2 = (int4)(2, 3, 2, 3);
    int4 add3 = (int4)(1, 2, 2, 3);

    int4 comp;

    comp = ((*input < shuffle(*input, mask1)) ^ dir);
    *input = shuffle(*input, as_uint4(comp + add1));

    comp = ((*input < shuffle(*input, mask2)) ^ dir);
    *input = shuffle(*input, as_uint4(comp * 2 + add2));

    comp = ((*input < shuffle(*input, mask3)) ^ dir);
    *input = shuffle(*input, as_uint4(comp + add3));
}

__kernel void bitonic_local_max_slm(__global int4* g_data, __local int4* l_data, int direction)
{
    
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint group_size = get_local_size(0);
    const uint global_offset = get_group_id(0) * get_local_size(0);
    uint gid_offset_int4 = gid * group_size;

    int4 private_block = g_data[global_offset + lid];

    int4_sort(&private_block, lid % 2);
    l_data[lid] = private_block;
    barrier(CLK_LOCAL_MEM_FENCE);


    for (uint block_size = 2; block_size <= group_size; block_size *= 2) // block = block_size * int4 
    {
        for (uint dist = block_size / 2; dist > 0; dist /= 2) 
        {
            uint pos = lid;
            uint partner = pos ^ dist;
            
            if (partner > pos) {
                uint global_pos_int = (gid_offset_int4 + pos) * ELEMS_PER_THREAD;
                uint mask = block_size * ELEMS_PER_THREAD;
                int local_direction = direction ^ ((global_pos_int & mask) != 0);
                int4_compare_swap(&l_data[pos], &l_data[partner], local_direction);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        uint base_global_int = (gid_offset_int4 + lid) * ELEMS_PER_THREAD;
        uint mask = block_size * ELEMS_PER_THREAD;
        int local_dir = direction ^ ((base_global_int & mask) != 0);
        int4_sort(&l_data[lid], (local_dir + 1) % 2);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    g_data[global_offset + lid] = l_data[lid];
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

__kernel void bitonic_big_step_local(__global int* g_data, __local int* l_data,
                                     const uint block_size, const uint current_dist, int direction)
{
    int group_size = get_local_size(0);
    int global_offset = get_group_id(0) * (group_size * 2);

    int lid = get_local_id(0);

    l_data[lid] = g_data[global_offset + lid];
    l_data[lid + group_size] = g_data[global_offset + lid + group_size];

    barrier(CLK_LOCAL_MEM_FENCE);


    for (int dist = current_dist; dist > 0; dist /= 2)
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
