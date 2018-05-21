__kernel void scan_blelloch(__global float * a, __global float * r,
                            __global float * partial_sums,
                            __local float * b, int n) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_id = get_group_id(0);
    uint block_size = get_local_size(0);
    uint dp = 1;

    if (gid < n) {
        b[lid] = a[gid];
    } else {
        b[lid] = 0;
    }

    for (uint s = block_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) {
            uint i = dp * (2 * lid + 1) - 1;
            uint j = dp * (2 * lid + 2) - 1;
            b[j] += b[i];
        }

        dp <<= 1;
    }

    if (lid == 0) {
        partial_sums[group_id] = b[block_size - 1];

        if (gid < n) {
            uint last_index = group_id * block_size + block_size - 1;
            r[last_index] = b[block_size - 1];
        }

        b[block_size - 1] = 0;
    }


    for(uint s = 1; s < block_size; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < s) {
            uint i = dp * (2 * lid + 1) - 1;
            uint j = dp * (2 * lid + 2) - 1;

            float t = b[j];
            b[j] += b[i];
            b[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < block_size - 1 && gid < n) {
        r[gid] = b[lid + 1];
    }
}

__kernel void add_rest_elements(__global float * partial_sums, __global float * c, int n) {
    uint gid = get_global_id(0);
    uint group_id = get_group_id(0);

    if (gid > n - 1 || group_id == 0) {
        return;
    }

    c[gid] += partial_sums[group_id - 1];
}