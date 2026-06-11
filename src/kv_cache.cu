#include "kv_cache.h"
#include <cuda_runtime.h>

// 把当前 token 的 k,v 写进缓存第 pos 个槽位。布局 [H, S, D] 连续显存。
__global__ void append_kv(const float* k_new, const float* v_new,
                          float* k_cache, float* v_cache,
                          int pos, int D, int S) {
    int h = blockIdx.x;       // 第几个 head
    int d = threadIdx.x;      // head_dim 第几维
    int dst = h * S * D + pos * D + d;   // 偏移 = 这一行的核心
    k_cache[dst] = k_new[h * D + d];  //h是当前计算的head索引，pos是当前的token索引，d是当前计算的head内的D维度索引
    v_cache[dst] = v_new[h * D + d];
}

void launch_append(const float* k_new, const float* v_new,
                   float* k_cache, float* v_cache,
                   int pos, int H, int D, int S) {
    append_kv<<<H, D>>>(k_new, v_new, k_cache, v_cache, pos, D, S);
}