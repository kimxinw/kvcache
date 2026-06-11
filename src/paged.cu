#include "kv_cache.h"
#include <cuda_runtime.h>

// 物理缓存池布局: [num_blocks, H, BLOCK, D] —— 每个物理块装 BLOCK 个 token、所有 head。
// block_table: [num_logical_blocks], 逻辑块号 -> 物理块号。
// 取 token pos 的某 head 的 K/V, 要先查表再算块内偏移。
__device__ inline int kv_offset(const int* block_table, int pos,
                                int h, int H, int BLOCK, int D) {
    int phys = block_table[pos / BLOCK];          // 1) 查表得物理块号
    int in   = pos % BLOCK;                        // 2) 块内第几个 token
    // 3) 池布局 [num_blocks, H, BLOCK, D] 下, 该 token 该 head 的起始偏移:
    return ((phys * H + h) * BLOCK + in) * D;
}

__global__ void paged_decode(const float* q,
                             const float* k_pool, const float* v_pool,
                             const int* block_table,
                             float* out, int cur_len,
                             int H, int D, int BLOCK, float scale) {
    int h = blockIdx.x, d = threadIdx.x;
    extern __shared__ float smem[];
    float* q_s   = smem;                  // [D]
    float* score = smem + D;              // [cur_len] (上限由调用方保证 shmem 够)

    q_s[d] = q[h * D + d];
    __syncthreads();

    for (int pos = d; pos < cur_len; pos += D) {   // score = q·K[pos]*scale
        // ===== TODO: 用 kv_offset 间接取 K[pos] 这一行, 算点积 =====
        // 对照 L1 的写法, 唯一区别是 K 的基址不再是 kh+pos*D, 而是:
          const float* krow = k_pool + kv_offset(block_table, pos, h, H, BLOCK, D);
          float acc = 0.f;
          for (int i = 0; i < D; ++i) acc += q_s[i] * krow[i];
          score[pos] = acc * scale;
        // ================================================================
    }
    __syncthreads();

    if (d == 0) {                          // softmax(和 L1 一样)
        float m = score[0];
        for (int i = 1; i < cur_len; ++i) m = fmaxf(m, score[i]);
        float s = 0.f;
        for (int i = 0; i < cur_len; ++i) { score[i] = expf(score[i]-m); s += score[i]; }
        for (int i = 0; i < cur_len; ++i) score[i] /= s;
    }
    __syncthreads();

    float acc = 0.f;                       // 加权和: V 也要间接取
    for (int pos = 0; pos < cur_len; ++pos) {
        const float* vrow = v_pool + kv_offset(block_table, pos, h, H, BLOCK, D);
        acc += score[pos] * vrow[d];
    }
    out[h * D + d] = acc;
}

void launch_paged(const float* q, const float* k_pool, const float* v_pool,
                  const int* block_table, float* out, int cur_len,
                  int H, int D, int BLOCK, float scale) {
    size_t shmem = (D + cur_len) * sizeof(float);
    paged_decode<<<H, D, shmem>>>(q, k_pool, v_pool, block_table,
                                  out, cur_len, H, D, BLOCK, scale);
}