__kernel void bitonic_sort_init(__global float4* g_data, __local float4* l_data, ulong len, __global float4* output)  // len is for float4 indexation (i.e. absolute sz / 4)
{
    ulong m = get_local_size(0);

    // up to cur_idx + 8m
    ulong cur_idx = get_group_id(0) * 2 * m; 
    // per workitem copy 4 + 4 float4 to l_data, then we will get 8M copies and fill l_data

    ulong local_idx = get_local_id(0) * 2;

    l_data[local_idx] = g_data[cur_idx + local_idx];
    l_data[local_idx + 1] = g_data[cur_idx + local_idx + 1];
    
    for(int i = 0; i < 2 * m; i++)
    {
        output[i] = l_data[i];
    }
}
