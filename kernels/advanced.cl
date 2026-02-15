#define ELEMS_PER_THREAD 4
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

static inline void int4_compare_swap(int4* a, int4* b, int dir)
{
    int4 add3 = (int4)(4, 5, 6, 7);
    int4 ta = *a;
    int4 comp = ((*a < *b) ^ dir) * 4 + add3;
    *a = shuffle2(*a, *b, as_uint4(comp));
    *b = shuffle2(*b, ta, as_uint4(comp));
}

static inline void int4_sort(int4* input, int dir)
{
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

static inline void sort_bitonic_int4(int4* input, int dir)
{ 
    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 mask3 = (uint4)(3, 2, 1, 0);

    int4 add1 = (int4)(1, 1, 3, 3);
    int4 add2 = (int4)(2, 3, 2, 3);
    int4 add3 = (int4)(1, 2, 2, 3);
     int4 comp;


    comp = ((*input < shuffle(*input, mask2)) ^ dir);
    *input = shuffle(*input, as_uint4(comp * 2 + add2));

    comp = ((*input < shuffle(*input, mask1)) ^ dir);
    *input = shuffle(*input, as_uint4(comp + add1));
}

static inline void bitonic_local_finalize(__local int4* l_data, int id, int dir)
{
    int4 input1 = l_data[id];
    int4 input2 = l_data[id + 1];
    int4_compare_swap(&input1, &input2, dir);
    sort_bitonic_int4(&input1, dir);
    sort_bitonic_int4(&input2, dir);
    l_data[id] = input1;
    l_data[id + 1] = input2;
}

static inline void bitonic_local_merge(__local int4* l_data, int start_stride, int dir) {
    for (int stride = start_stride; stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        int id = get_local_id(0) + (get_local_id(0) / stride) * stride;
        int4_compare_swap(&l_data[id], &l_data[id + stride], dir);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void bitonic_local_max_slm(__global int4* g_data, __local int4* l_data)
{   
    int id = get_local_id(0) * 2;
    int global_start = get_group_id(0) * get_local_size(0) * 2 + id;

    int4 input1 = g_data[global_start];
    int4 input2 = g_data[global_start + 1];

    int dir = get_local_id(0) % 2 * -1;

    int4_sort(&input1, 0);  // ascending
    int4_sort(&input2, -1); // descending
    int4_compare_swap(&input1, &input2, dir);
    sort_bitonic_int4(&input1, dir);
    sort_bitonic_int4(&input2, dir);

    l_data[id] = input1;
    l_data[id + 1] = input2;

    /* Create bitonic set */
    for (int size = 2; size < get_local_size(0); size <<= 1)
    {
        dir = (get_local_id(0) / size & 1) * -1;
        bitonic_local_merge(l_data, size, dir);
        bitonic_local_finalize(l_data, /*id=*/get_local_id(0) * 2, dir);
    }

    /* Perform bitonic merge */
    dir = (get_group_id(0) % 2) * -1;
    bitonic_local_merge(l_data, get_local_size(0), dir);
    
    /* Perform final sort */
    id = get_local_id(0) * 2;
    input1 = l_data[id];
    input2 = l_data[id + 1];
    int4_compare_swap(&input1, &input2, dir);
    sort_bitonic_int4(&input1, dir);
    sort_bitonic_int4(&input2, dir);
    
    g_data[global_start] = input1;
    g_data[global_start + 1] = input2;
}

__kernel void bitonic_step_global(__global int* array, const uint block_size, const uint dist, int direction)
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
    const int group_size = get_local_size(0);
    const int global_offset = get_group_id(0) * group_size;
    const int lid = get_local_id(0);

    l_data[lid] = g_data[global_offset + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int dist = current_dist; dist > 0; dist /= 2)
    {
        uint pos = lid;
        uint partner = pos ^ dist;
        if (partner > pos) {
            uint global_pos = pos + global_offset;

            uint use_reversed_direction = (global_pos & block_size) != 0;
            int local_direction = direction ^ use_reversed_direction;

            COMP_AND_SWAP(l_data[pos], l_data[partner], local_direction);

        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    g_data[global_offset + lid] = l_data[lid];
}


// __kernel void bitonic_big_step_local_int4(__global int4* g_data,
//                                           __local int4* l_data,
//                                           const uint block_size,       // in ints
//                                           const uint current_dist,     // in ints
//                                           int direction)
// {
//     const uint lid = get_local_id(0);
//     const uint group_size = get_local_size(0);          // number of int4 slots per half-chunk
//     const uint global_offset = get_group_id(0) * (group_size * 2u); // offset in int4 units

//     // load two int4s per thread into local memory (cover 2*group_size int4 slots)
//     int4 v0 = g_data[global_offset + lid];
//     int4 v1 = g_data[global_offset + lid + group_size];
//     l_data[lid] = v0;
//     l_data[lid + group_size] = v1;

//     barrier(CLK_LOCAL_MEM_FENCE);

//     // convert sizes from ints -> int4 units
//     const uint block_size4 = block_size / ELEMS_PER_THREAD;     // must be integer
//     uint dist4 = current_dist / ELEMS_PER_THREAD;               // loop variable in int4 units

//     for (; dist4 > 0; dist4 >>= 1)
//     {
//         // pos computation same pattern as original but in int4 indices
//         uint pos = lid;
//         uint block_index = pos / dist4;
//         pos += block_index * dist4;

//         uint partner = pos ^ dist4;

//         // do compare-swap only once per pair
//         if (partner > pos)
//         {
//             // compute element-granular index (in ints) for direction decision
//             uint global_pos_int = (global_offset + pos) * ELEMS_PER_THREAD;
//             // block_size is in ints => use it directly as mask
//             uint use_reversed_direction = (global_pos_int & block_size) != 0u;
//             int local_direction = direction ^ (int)use_reversed_direction;

//             // use int4 compare-swap (replace COMP_AND_SWAP for int4 lanes)
//             int4_compare_swap(&l_data[pos], &l_data[partner], local_direction);
//         }

//         barrier(CLK_LOCAL_MEM_FENCE);
//     }

//     // write back both int4 slots
//     g_data[global_offset + lid] = l_data[lid];
//     g_data[global_offset + lid + group_size] = l_data[lid + group_size];
// }

