__kernel void convolution(__global float * a, __global float * b, __global float * c, int n, int m)
{
    // get coords
    int i = get_global_id(0);
    int j = get_global_id(1);
    int hm = (m - 1) / 2;

    if (i < n && j < n) {
        float res = 0;
        for (int k = -hm; k <= hm; k++) {
            for (int l = -hm; l <= hm; l++)  {
                int ai = i + k;
                int aj = j + l;
                int bi = k + hm;
                int bj = l + hm;

                float first = 0;
                if (ai < 0 || aj < 0 || ai >= n || aj >= n) {
                    first = 0;
                } else {
                    first = a[ai * n + aj];
                }

                res += first * b[bi * m + bj];
            }
        }

        c[i * n + j] = res;
    }
}