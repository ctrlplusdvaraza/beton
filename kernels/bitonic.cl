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


// Input: a bitonic sequence in local memory
// Output: the same block is fully sorted in dir
static inline void bitonic_local_merge(__local int4* l_data, int start_stride, int dir) {
    for (int stride = start_stride; stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        int id = get_local_id(0) + (get_local_id(0) / stride) * stride;
        int4_compare_swap(&l_data[id], &l_data[id + stride], dir);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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

// TODO try bubble sort for 8, 16 elements
// sorting array with size = wgrp_size * 8 (only one work group is sorting)
__kernel void bsort_init(__global int4* g_data, __local int4* l_data)
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

__kernel void bsort_stage_0(__global int4* g_data, __local int4* l_data, uint high_stage)
{
    const int local_size = get_local_size(0);
    const int local_id   = get_local_id(0);
    const int group_id   = get_group_id(0);

    int dir = (group_id / high_stage & 1) ? -1 : 0;
    int group_base = group_id * (local_size * 2);

    l_data[local_id]                 = g_data[group_base + local_id];           
    l_data[local_id + local_size]    = g_data[group_base + local_id + local_size]; 
    barrier(CLK_LOCAL_MEM_FENCE);

    bitonic_local_merge(l_data, local_size, dir);
    int id2 = local_id * 2;
    bitonic_local_finalize(l_data, id2, dir);
    g_data[group_base + id2]     = l_data[id2];
    g_data[group_base + id2 + 1] = l_data[id2 + 1];
}

/* Perform successive stages of the bitonic sort */
__kernel void bsort_stage_n(__global int4* g_data, __local int4* l_data, uint stage, uint high_stage)
{
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);

    int dir             = (get_group_id(0) / high_stage & 1) * -1;
    int global_start    = (get_group_id(0) + (get_group_id(0) / stage) * stage) * get_local_size(0) + get_local_id(0);
    int global_offset   = stage * get_local_size(0);

    l_data[local_id]                 = g_data[global_start];
    l_data[local_id + local_size]    = g_data[global_start + global_offset];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int4_compare_swap(&l_data[local_id], &l_data[local_id + local_size], dir);
    g_data[global_start] = l_data[local_id];
    g_data[global_start + global_offset] = l_data[local_id + local_size];
}

/* Sort the bitonic set */
__kernel void bsort_merge(__global int4* g_data, __local int4* l_data, uint stage, int dir)
{
    int global_start    = (get_group_id(0) + (get_group_id(0) / stage) * stage) * get_local_size(0) + get_local_id(0);
    int global_offset   = stage * get_local_size(0);

    int4 input1 = g_data[global_start];
    int4 input2 = g_data[global_start + global_offset];
    int4_compare_swap(&input1, &input2, dir);
    g_data[global_start] = input1;
    g_data[global_start + global_offset] = input2;
}

/* Perform final step of the bitonic merge */
__kernel void bsort_merge_last(__global int4* g_data, __local int4* l_data, int dir)
{
    /* Determine location of data in global memory */
    int id;
    id = get_local_id(0);
    int global_start = get_group_id(0) * get_local_size(0) * 2 + id;

    /* Perform initial swap */
    int4 input1 = g_data[global_start];
    int4 input2 = g_data[global_start + get_local_size(0)];

    int4 temp1 = input1;
    int4 temp2 = input2;
    int4_compare_swap(&temp1, &temp2, dir);
    l_data[id] = temp1;
    l_data[id + get_local_size(0)] = temp2;


    bitonic_local_merge(l_data, get_local_size(0) / 2, dir);

    /* Perform final sort */
    id = get_local_id(0) * 2;
    input1 = l_data[id];
    input2 = l_data[id + 1];
    int4_compare_swap(&input1, &input2, dir);
    sort_bitonic_int4(&input1, dir);
    sort_bitonic_int4(&input2, dir);
    g_data[global_start + get_local_id(0)] = input1;
    g_data[global_start + get_local_id(0) + 1] = input2;
}
