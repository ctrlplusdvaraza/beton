void bubble_sort4(__local float4* l_data, uint start_idx, uint direction)
// direction 0 - ascending, 1 - descending
{
    float arr[4];
    ((float4*)arr)[0] = l_data[start_idx];

    for (uint i = 0; i < 4; i++)
    {
        for (uint j = 0; j < 4; j++)
        {
            if (((arr[i] < arr[j]) && direction == 0) || ((arr[i] > arr[j]) && direction == 1))
            {
                float temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
    l_data[start_idx] = ((float4*)arr)[0];
}


void bubble_sort8(__local float4* l_data, uint start_idx, uint direction)
// direction 0 - ascending, 1 - descending
{
    float arr[8];
    ((float4*)arr)[0] = l_data[start_idx];
    ((float4*)arr)[1] = l_data[start_idx + 1];

    for (uint i = 0; i < 8; i++)
    {
        for (uint j = 0; j < 8; j++)
        {
            if (((arr[i] < arr[j]) && direction == 0) || ((arr[i] > arr[j]) && direction == 1))
            {
                float temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
    l_data[start_idx] = ((float4*)arr)[0];
    l_data[start_idx + 1] = ((float4*)arr)[1];
}

#define COMPARE_AND_SWAP1(A, B, DIRECTION)                                                         \
    if (((A > B) && (DIRECTION == 0)) || ((A < B) && (DIRECTION == 1)))                            \
    {                                                                                              \
        float tmp = A;                                                                             \
        A = B;                                                                                     \
        B = tmp;                                                                                   \
    }


void cas_float4(__local float4* first, __local float4* second, uint direction)
{
    COMPARE_AND_SWAP1(first->x, second->x, direction);
    COMPARE_AND_SWAP1(first->y, second->y, direction);
    COMPARE_AND_SWAP1(first->z, second->z, direction);
    COMPARE_AND_SWAP1(first->w, second->w, direction);
}

#define VECTOR_SORT(input, dir)                                                                    \
    comp = abs(input > shuffle(input, mask2)) ^ dir;                                               \
    input = shuffle(input, comp * 2 + add2);                                                       \
    comp = abs(input > shuffle(input, mask1)) ^ dir;                                               \
    input = shuffle(input, comp + add1);


#define VECTOR_SWAP(in1, in2, dir)                                                                 \
    input1 = in1;                                                                                  \
    input2 = in2;                                                                                  \
    comp = (abs(input1 > input2) ^ dir) * 4 + add3;                                                \
    in1 = shuffle2(input1, input2, comp);                                                          \
    in2 = shuffle2(input2, input1, comp);


__kernel void bitonic_sort_init(__global float4* g_data, __local float4* l_data,
                                ulong len) // len is for float4 indexation (i.e. absolute sz / 4)
{
    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 swap = (uint4)(0, 0, 1, 1);
    uint4 add1 = (uint4)(0, 0, 2, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 add2 = (uint4)(0, 1, 0, 1);
    uint4 add3 = (uint4)(0, 1, 2, 3);

    ulong m = get_local_size(0);

    // up to cur_idx + 8m
    ulong cur_idx = get_group_id(0) * 2 * m;
    // per workitem copy 4 + 4 float4 to l_data, then we will get 8M copies and fill l_data

    ulong local_idx = get_local_id(0) * 2;

    l_data[local_idx] = g_data[cur_idx + local_idx];
    l_data[local_idx + 1] = g_data[cur_idx + local_idx + 1];

    bubble_sort8(l_data, local_idx, get_local_id(0) % 2);

    for (uint size = 2; size < get_local_size(0); size *= 2)
    {
        uint dir = get_local_id(0) / size % 2;
        for (uint stride = size; stride > 1; stride /= 2)
        {
            barrier(CLK_LOCAL_MEM_FENCE);

            uint id = get_local_id(0) + (get_local_id(0) / stride) * stride;
            cas_float4(&l_data[id], &l_data[id + stride], dir);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        uint4 add3 = (uint4)(0, 1, 2, 3);
        int id = get_local_id(0) * 2;

        float4 input1 = l_data[id];
        float4 input2 = l_data[id + 1];

        float4 temp = input1;
        uint4 comp = (abs(input1 > input2) ^ dir) * 4 + add3;
        input1 = shuffle2(input1, input2, comp);
        input2 = shuffle2(input2, temp, comp);

        VECTOR_SORT(input1, dir);
        VECTOR_SORT(input2, dir);

        l_data[id] = input1;
        l_data[id + 1] = input2;
    }

    uint dir = get_group_id(0) % 2;
    for (uint stride = get_local_size(0); stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint id = get_local_id(0) + (get_local_id(0) / stride) * stride;
        cas_float4(&l_data[id], &l_data[id + stride], dir);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint id = get_local_id(0) * 2;
    float4 input1 = l_data[id];
    float4 input2 = l_data[id + 1];
    float4 temp = input1;
    uint4 comp = (abs(input1 > input2) ^ dir) * 4 + add3;
    input1 = shuffle2(input1, input2, comp);
    input2 = shuffle2(input2, temp, comp);
    VECTOR_SORT(input1, dir);
    VECTOR_SORT(input2, dir);

    uint global_start = get_group_id(0) * get_local_size(0) * 2 + id;
    g_data[global_start] = input1;
    g_data[global_start + 1] = input2;
}


__kernel void bsort_stage_0(__global float4 *g_data, __local float4 *l_data, 
                            uint high_stage) {

   int dir;
   uint id, global_start, stride;
   float4 input1, input2, temp;
   int4 comp;

    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 swap = (uint4)(0, 0, 1, 1);
    uint4 add1 = (uint4)(0, 0, 2, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 add2 = (uint4)(0, 1, 0, 1);
    uint4 add3 = (uint4)(0, 1, 2, 3);

   uint4 mask3 = (uint4)(3, 2, 1, 0);


   /* Determine data location in global memory */
   id = get_local_id(0);
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];
   comp = (input1 < input2 ^ dir) * 4 + add3;
   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      cas_float4(&l_data[id], &l_data[id + stride], dir);
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));

   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);

   /* Store output in global memory */
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
}


__kernel void bsort_stage_n(__global float4 *g_data, __local float4 *l_data, 
                            uint stage, uint high_stage) {

   int dir;
   float4 input1, input2;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   /* Determine location of data in global memory */
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   /* Perform swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + global_offset];
   comp = (input1 < input2 ^ dir) * 4 + add;
   g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

__kernel void bsort_merge(__global float4* g_data, __local float4* l_data, uint stage, uint dir)
{
    uint tid = get_global_id(0);              // 0 .. (N4/2 - 1)
    uint lsz = get_local_size(0);
    uint offset = stage * lsz;                // in float4 elements

    // Map tid into 2*offset blocks:
    uint i = (tid % offset) + (tid / offset) * (2 * offset);
    uint j = i + offset;

    float4 input1 = g_data[i];
    float4 input2 = g_data[j];

    uint4 add  = (uint4)(4, 5, 6, 7);
    uint4 comp = (uint4)abs((input1 < input2) ^ (int)dir);
    comp = comp * 4 + add;

    g_data[i] = shuffle2(input1, input2, comp);
    g_data[j] = shuffle2(input2, input1, comp);
}


__kernel void bsort_merge_last(__global float4* g_data, __local float4* l_data, uint dir)
{
    uint id, global_start, stride;
    float4 input1, input2, temp;
    uint4 comp;
    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 add1 = (uint4)(1, 1, 3, 3);
    uint4 add2 = (uint4)(2, 3, 2, 3);
    uint4 add3 = (uint4)(4, 5, 6, 7);

    id = get_local_id(0);
    global_start = get_group_id(0) * get_local_size(0) * 2 + id;

    input1 = g_data[global_start];
    input2 = g_data[global_start + get_local_size(0)];

    comp = (uint4)abs((input1 < input2) ^ (int)dir) * 4 + add3;
    l_data[id] = shuffle2(input1, input2, comp);
    l_data[id + get_local_size(0)] = shuffle2(input2, input1, comp);

    for (stride = get_local_size(0) / 2; stride > 1; stride >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        id = get_local_id(0) + (get_local_id(0) / stride) * stride;
        cas_float4(&l_data[id], &l_data[id + stride], dir);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform final sort */
    id = get_local_id(0) * 2;
    input1 = l_data[id];
    input2 = l_data[id + 1];
    temp = input1;

    add3 = (uint4)(4, 5, 6, 7);
    int4 cmp0 = (input1 < input2);
    int4 cmp1 = cmp0 ^ (int4)(dir);
    comp = (uint4)abs(cmp1) * 4u + add3;


    input1 = shuffle2(input1, input2, as_uint4(comp));
    input2 = shuffle2(input2, temp, as_uint4(comp));

    VECTOR_SORT(input1, dir);
    VECTOR_SORT(input2, dir);

uint base = get_group_id(0) * get_local_size(0) * 2;  // float4 index base
uint out  = get_local_id(0) * 2;                      // 2 float4 per WI

g_data[base + out]     = input1;
g_data[base + out + 1] = input2;


}
