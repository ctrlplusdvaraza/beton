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

static inline void int4_compare_swap(__local int* la, __local int* lb, int dir)
{
    for (int i = 0; i < 4; i++) {
        COMP_AND_SWAP(la[i], lb[i], dir);
    }

    // int4 a = {la[0], la[1], la[2], la[3]};
    // int4 b = {lb[0], lb[1], lb[2], lb[3]};

    // // int4 add3 = (int4)(4, 5, 6, 7);
    // int4 ta = a;
    // // int4 comp = ((a < b) ^ dir) * 4 + add3;
    // // a = shuffle2(a, b, as_uint4(comp));
    // // b = shuffle2(b, ta, as_uint4(comp));


    // int4 comp = ((a < b) ^ dir);
    // int4 minMask = comp * 4;          // 0,0,0,0 or 4,4,4,4
    // int4 maxMask = (1 - comp) * 4;     // 4,4,4,4 or 0,0,0,0
    // int4 add3 = (int4)(0, 1, 2, 3);

    // a = shuffle2(a, b, as_uint4(minMask + add3));  // Select smaller values
    // b = shuffle2(ta, b, as_uint4(maxMask + add3)); // Select larger values



    // la[0] = a.x;
    // la[1] = a.y;
    // la[2] = a.z;
    // la[3] = a.w;

    // lb[0] = b.x;
    // lb[1] = b.y;
    // lb[2] = b.z;
    // lb[3] = b.w;
}

__kernel void bitonic_local_max_slm(__global int* g_data, __local int* l_data, int direction)
{
    uint group_size = get_local_size(0);
    uint total_local_size = group_size * ELEMS_PER_THREAD;
    uint global_offset = get_group_id(0) * total_local_size;
    uint lid = get_local_id(0);

    int private_block[ELEMS_PER_THREAD];

    uint base_idx = lid * ELEMS_PER_THREAD;
    for (uint i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        uint global_idx = global_offset + base_idx + i;
        private_block[i] = g_data[global_idx];
    }

    // printf("%d : %d %d %d %d\n", lid, private_block[0], private_block[1] , private_block[2], private_block[3]);

    for (uint block_size = 2; block_size <= ELEMS_PER_THREAD; block_size *= 2)
    {
        for (uint dist = block_size / 2; dist > 0; dist /= 2)
        {
            for (uint pos = 0; pos < ELEMS_PER_THREAD / 2; ++pos)
            {
                uint block_index = pos / dist;
                uint correct_pos = pos + block_index * dist;

                uint partner = correct_pos ^ dist;


                // uint global_pos = pos + global_offset;
                uint global_pos = (lid * ELEMS_PER_THREAD) + correct_pos;

                // Use the global index for the mask check
                int use_reversed_direction = (global_pos & block_size) != 0;
                int local_direction = direction ^ use_reversed_direction;

                // int use_reversed_direction = (correct_pos & block_size) != 0;
                // int local_direction = direction ^ use_reversed_direction;

                COMP_AND_SWAP(private_block[correct_pos], private_block[partner], local_direction);
            }
        }
    }

    for (uint i = 0; i < ELEMS_PER_THREAD; i++)
    {
        l_data[base_idx + i] = private_block[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    for (uint block_size = 2; block_size <= group_size; block_size *= 2) 
    {
        // 1. Inter-Thread Phase: Swap whole int4 vectors between threads
        // Handles strides: block_size/2 ... down to 1 thread
        for (uint dist = block_size / 2; dist > 0; dist /= 2) 
        {
            uint pos = lid;
            uint partner = pos ^ dist;
            
            if (partner > pos)
            {
                uint global_pos = pos * ELEMS_PER_THREAD + global_offset;
                
                // FIX 1: Scale block_size to elements for the mask
                uint mask = block_size * ELEMS_PER_THREAD; 
                uint use_reversed_direction = (global_pos & mask) != 0;
                int local_direction = direction ^ use_reversed_direction;

                int4_compare_swap(&l_data[pos * ELEMS_PER_THREAD], &l_data[partner * ELEMS_PER_THREAD], local_direction);
            }
            // FIX 2: Mandatory Barrier
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        uint global_pos = lid * ELEMS_PER_THREAD + global_offset;
        uint mask = block_size * ELEMS_PER_THREAD; 
        
        // Calculate direction for THIS thread (applies to all 4 elements)
        int local_dir = direction ^ ((global_pos & mask) != 0);

        // Load all 4 elements into registers
        uint base = lid * ELEMS_PER_THREAD;
        int v0 = l_data[base + 0];
        int v1 = l_data[base + 1];
        int v2 = l_data[base + 2];
        int v3 = l_data[base + 3];

        // Stride 2: Compare (0,2) and (1,3)
        COMP_AND_SWAP(v0, v2, local_dir);
        COMP_AND_SWAP(v1, v3, local_dir);

        // Stride 1: Compare (0,1) and (2,3)
        COMP_AND_SWAP(v0, v1, local_dir);
        COMP_AND_SWAP(v2, v3, local_dir);

        // Store fully sorted quad back to memory
        l_data[base + 0] = v0;
        l_data[base + 1] = v1;
        l_data[base + 2] = v2;
        l_data[base + 3] = v3;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        g_data[global_offset + base_idx + i] = l_data[base_idx + i];
    }
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
