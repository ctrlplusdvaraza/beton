__kernel void bitonic_step(__global int* array, const uint block_size, const uint dist,
                           int direction)
{
    uint pos = get_global_id(0);
    uint partner = pos ^ dist;
    if (partner < pos) { return; }

    int use_original_direction = (pos & block_size) == 0;
    int local_direction = use_original_direction ? direction : (-1 - direction);

    int a = array[pos];
    int b = array[partner];

    int swap = ((a > b) == (local_direction == 0));
    if (swap)
    {
        array[pos] = b;
        array[partner] = a;
    }
}
