static inline void int4_compare_swap(int4* a, int4* b, int dir, int4 add3)
{
    int4 ta = *a;
    int4 comp = ((*a < *b) ^ dir) * 4 + add3;
    *a = shuffle2(*a, *b, as_uint4(comp));
    *b = shuffle2(*b, ta, as_uint4(comp));
}

static inline void int4_sort(int4* input, int dir, uint4 mask1, uint4 mask2, uint4 mask3, int4 add1,
                             int4 add2, int4 add3)
{
    int4 comp;

    comp = ((*input < shuffle(*input, mask1)) ^ dir);
    *input = shuffle(*input, as_uint4(comp + add1));

    comp = ((*input < shuffle(*input, mask2)) ^ dir);
    *input = shuffle(*input, as_uint4(comp * 2 + add2));

    comp = ((*input < shuffle(*input, mask3)) ^ dir);
    *input = shuffle(*input, as_uint4(comp + add3));
}

static inline void sort_bitonic_int4(int4* input, int dir, uint4 mask1, uint4 mask2, int4 add1,
                                     int4 add2)
{
    int4 comp;

    comp = ((*input < shuffle(*input, mask2)) ^ dir);
    *input = shuffle(*input, as_uint4(comp * 2 + add2));

    comp = ((*input < shuffle(*input, mask1)) ^ dir);
    *input = shuffle(*input, as_uint4(comp + add1));
}

// TODO try bubble sort for 8, 16 elements
// sorting array with size = wgrp_size * 8 (only one work group is sorting)
__kernel void bsort_init(__global int4* g_data, __local int4* l_data)
{
    int dir;
    uint id, global_start, size, stride;
    int4 input1, input2, temp;
    int4 comp;

    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 mask3 = (uint4)(3, 2, 1, 0);

    int4 add1 = (int4)(1, 1, 3, 3);
    int4 add2 = (int4)(2, 3, 2, 3);
    int4 add3 = (int4)(1, 2, 2, 3);

    id = get_local_id(0) * 2;
    global_start = get_group_id(0) * get_local_size(0) * 2 + id;

    input1 = g_data[global_start];
    input2 = g_data[global_start + 1];

    int4_sort(&input1, 0, mask1, mask2, mask3, add1, add2, add3);  // ascending
    int4_sort(&input2, -1, mask1, mask2, mask3, add1, add2, add3); // descending

    /* Swap corresponding elements of input 1 and 2 */
    add3 = (int4)(4, 5, 6, 7);

    dir = get_local_id(0) % 2 * -1;
    int4_compare_swap(&input1, &input2, dir, add3);
    sort_bitonic_int4(&input1, dir, mask1, mask2, add1, add2);
    sort_bitonic_int4(&input2, dir, mask1, mask2, add1, add2);

    l_data[id] = input1;
    l_data[id + 1] = input2;

    /* Create bitonic set */
    for (size = 2; size < get_local_size(0); size <<= 1)
    {
        dir = (get_local_id(0) / size & 1) * -1;

        for (stride = size; stride > 1; stride >>= 1) // distance
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            id = get_local_id(0) + (get_local_id(0) / stride) * stride;
            int4_compare_swap(&l_data[id], &l_data[id + stride], dir, add3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        id = get_local_id(0) * 2;

        input1 = l_data[id];
        input2 = l_data[id + 1];

        int4_compare_swap(&input1, &input2, dir, add3);
        sort_bitonic_int4(&input1, dir, mask1, mask2, add1, add2);
        sort_bitonic_int4(&input2, dir, mask1, mask2, add1, add2);

        l_data[id] = input1;
        l_data[id + 1] = input2;
    }

    /* Perform bitonic merge */
    dir = (get_group_id(0) % 2) * -1;
    for (stride = get_local_size(0); stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        id = get_local_id(0) + (get_local_id(0) / stride) * stride;
        int4_compare_swap(&l_data[id], &l_data[id + stride], dir, add3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform final sort */
    id = get_local_id(0) * 2;
    input1 = l_data[id];
    input2 = l_data[id + 1];

    int4_compare_swap(&input1, &input2, dir, add3);
    sort_bitonic_int4(&input1, dir, mask1, mask2, add1, add2);
    sort_bitonic_int4(&input2, dir, mask1, mask2, add1, add2);

    g_data[global_start] = input1;
    g_data[global_start + 1] = input2;
}

__kernel void bsort_stage_0(__global int4* g_data, __local int4* l_data, uint high_stage)
{
    int dir;
    uint id, global_start, stride;
    int4 input1, input2, temp;
    int4 comp;

    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 mask3 = (uint4)(3, 2, 1, 0);

    int4 add1 = (int4)(1, 1, 3, 3);
    int4 add2 = (int4)(2, 3, 2, 3);
    int4 add3 = (int4)(4, 5, 6, 7);

    id = get_local_id(0);
    dir = (get_group_id(0) / high_stage & 1) * -1;
    global_start = get_group_id(0) * get_local_size(0) * 2 + id;

    /* Perform initial swap */
    input1 = g_data[global_start];
    input2 = g_data[global_start + get_local_size(0)];
    comp = (input1 < input2 ^ dir) * 4 + add3;
    l_data[id] = shuffle2(input1, input2, as_uint4(comp));
    l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

    /* Perform bitonic merge */
    for (stride = get_local_size(0) / 2; stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        id = get_local_id(0) + (get_local_id(0) / stride) * stride;
        int4_compare_swap(&l_data[id], &l_data[id + stride], dir, add3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform final sort */
    id = get_local_id(0) * 2;
    input1 = l_data[id];
    input2 = l_data[id + 1];

    int4_compare_swap(&input1, &input2, dir, add3);
    sort_bitonic_int4(&input1, dir, mask1, mask2, add1, add2);
    sort_bitonic_int4(&input2, dir, mask1, mask2, add1, add2);

    /* Store output in global memory */
    g_data[global_start + get_local_id(0)] = input1;
    g_data[global_start + get_local_id(0) + 1] = input2;
}

/* Perform successive stages of the bitonic sort */
__kernel void bsort_stage_n(__global int4* g_data, __local int4* l_data, uint stage,
                            uint high_stage)
{
    int dir;
    int4 input1, input2;
    int4 comp, add;
    uint global_start, global_offset;

    add = (int4)(4, 5, 6, 7);

    /* Determine location of data in global memory */
    dir = (get_group_id(0) / high_stage & 1) * -1;
    global_start =
        (get_group_id(0) + (get_group_id(0) / stage) * stage) * get_local_size(0) + get_local_id(0);
    global_offset = stage * get_local_size(0);

    /* Perform swap */
    input1 = g_data[global_start];
    input2 = g_data[global_start + global_offset];
    comp = (input1 < input2 ^ dir) * 4 + add;
    g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
    g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

/* Sort the bitonic set */
__kernel void bsort_merge(__global int4* g_data, __local int4* l_data, uint stage, int dir)
{
    int4 input1, input2;
    int4 comp, add;
    uint global_start, global_offset;

    add = (int4)(4, 5, 6, 7);

    /* Determine location of data in global memory */
    global_start =
        (get_group_id(0) + (get_group_id(0) / stage) * stage) * get_local_size(0) + get_local_id(0);
    global_offset = stage * get_local_size(0);

    input1 = g_data[global_start];
    input2 = g_data[global_start + global_offset];
    int4_compare_swap(&input1, &input2, dir, add);
    g_data[global_start] = input1;
    g_data[global_start + global_offset] = input2;
}

/* Perform final step of the bitonic merge */
__kernel void bsort_merge_last(__global int4* g_data, __local int4* l_data, int dir)
{
    uint id, global_start, stride;
    int4 input1, input2, temp;
    int4 comp;

    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 mask3 = (uint4)(3, 2, 1, 0);

    int4 add1 = (int4)(1, 1, 3, 3);
    int4 add2 = (int4)(2, 3, 2, 3);
    int4 add3 = (int4)(4, 5, 6, 7);

    /* Determine location of data in global memory */
    id = get_local_id(0);
    global_start = get_group_id(0) * get_local_size(0) * 2 + id;

    /* Perform initial swap */
    input1 = g_data[global_start];
    input2 = g_data[global_start + get_local_size(0)];

    int4 temp1 = input1;
    int4 temp2 = input2;
    int4_compare_swap(&temp1, &temp2, dir, add3);
    l_data[id] = temp1;
    l_data[id + get_local_size(0)] = temp2;

    /* Perform bitonic merge */
    for (stride = get_local_size(0) / 2; stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        id = get_local_id(0) + (get_local_id(0) / stride) * stride;
        int4_compare_swap(&l_data[id], &l_data[id + stride], dir, add3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform final sort */
    id = get_local_id(0) * 2;
    input1 = l_data[id];
    input2 = l_data[id + 1];

    int4_compare_swap(&input1, &input2, dir, add3);
    sort_bitonic_int4(&input1, dir, mask1, mask2, add1, add2);
    sort_bitonic_int4(&input2, dir, mask1, mask2, add1, add2);

    g_data[global_start + get_local_id(0)] = input1;
    g_data[global_start + get_local_id(0) + 1] = input2;
}
