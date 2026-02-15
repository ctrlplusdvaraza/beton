__kernel void bitonic_local_max_slm(__global int4* g_data, __local int4* l_data, const int direction)
{
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint group_size = get_local_size(0);
    const uint global_offset = get_group_id(0) * get_local_size(0);

    int4 private_block = g_data[global_offset + lid];

    int4_sort(&private_block, lid & 1);
    l_data[lid] = private_block;
    barrier(CLK_LOCAL_MEM_FENCE);


    for (uint block_size = 2; block_size <= group_size; block_size *= 2) // block = block_size * int4 
    {
        for (uint dist = block_size / 2; dist > 0; dist /= 2) 
        {
            uint pos = lid;
            uint partner = pos ^ dist;
            
            if (partner > pos) {
                uint global_pos_int = (global_offset + pos);
                // if the element sits in the upper half of the block the direction is inverted
                int local_direction = direction ^ ((global_pos_int & block_size) != 0); 
                int4_compare_swap(&l_data[pos], &l_data[partner], local_direction);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        uint base_global_int = (global_offset + lid);
        int local_dir = direction ^ ((base_global_int & block_size) == 0);
        int4_sort(&l_data[lid], local_dir);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    g_data[global_offset + lid] = l_data[lid];
}