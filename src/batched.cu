// batched.cu —— 多序列批量 decode。grid = (N, H): blockIdx.x 选序列, blockIdx.y 选 head。
// 三段式 (算分 -> softmax -> 加权和), 基址和长度按序列区分。
// 两个版本算法相同, 唯一区别是 K/V 怎么取:
//   - batched_decode : 连续 cache [N,H,MAXS,D], 直接 base + pos*D。
//   - batched_paged  : 共享池 [NB,H,BLOCK,D] + 每序列 block_table, 查表间接取。
//
// Tier 2: softmax 不再 if(d==0) 串行, 改成 D 个线程协作的并行 reduce (max/sum)。
// 共享内存布局: [ q_s(D) | score(cap) | red(D) ]。
#include "kv_cache.h"
#include <cuda_runtime.h>
#include <cfloat>

// D 个线程对 red[0..D) 做树形归约 (op=0 取 max, op=1 求和), 结果广播在 red[0]。
__device__ inline float block_reduce(float* red, int d, int D, int op) {
    __syncthreads();
    for (int stride = D/2; stride > 0; stride >>= 1) {
        if (d < stride) {
            if (op == 0) red[d] = fmaxf(red[d], red[d+stride]);
            else         red[d] = red[d] + red[d+stride];
        }
        __syncthreads();
    }
    float r = red[0];   // 先存进寄存器
    __syncthreads();    // 关键: 确保所有线程读完 red[0] 后, 下阶段才会改写 red
    return r;
}

// 对 score[0..L) 做 in-place softmax, D 个线程并行。red 是大小 D 的暂存区。
__device__ inline void softmax_inplace(float* score, int L, float* red, int d, int D) {
    float local = -FLT_MAX;                          // 1) 并行求 max
    for (int i = d; i < L; i += D) local = fmaxf(local, score[i]);
    red[d] = local;
    float m = block_reduce(red, d, D, 0);

    float lsum = 0.f;                                // 2) 并行 exp + 求和
    for (int i = d; i < L; i += D) { float e = expf(score[i]-m); score[i] = e; lsum += e; }
    red[d] = lsum;
    float sum = block_reduce(red, d, D, 1);

    float inv = 1.0f / sum;                          // 3) 并行归一化
    for (int i = d; i < L; i += D) score[i] *= inv;
    __syncthreads();
}

// ---------- 连续布局批量 decode ----------
__global__ void batched_decode(const float* q, const float* k_cache, const float* v_cache,
                               const int* cur_len, float* out,
                               int H, int D, int MAXS, int cap, float scale) {
    int s = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int L = cur_len[s];
    extern __shared__ float smem[];
    float* q_s   = smem;            // [D]
    float* score = smem + D;        // [cap]
    float* red   = smem + D + cap;  // [D]

    const float* qh = q       + ((size_t)s*H + h)*D;
    const float* kh = k_cache + ((size_t)s*H + h)*MAXS*D;
    const float* vh = v_cache + ((size_t)s*H + h)*MAXS*D;
    float* outh     = out     + ((size_t)s*H + h)*D;

    q_s[d] = qh[d];
    __syncthreads();

    for (int pos = d; pos < L; pos += D) {           // score = q·K[pos]*scale
        float acc = 0.f;
        for (int i = 0; i < D; ++i) acc += q_s[i] * kh[pos*D + i];
        score[pos] = acc * scale;
    }
    __syncthreads();

    softmax_inplace(score, L, red, d, D);            // 并行 softmax

    float acc = 0.f;                                 // out[d] = Σ prob[pos]*V[pos][d]
    for (int pos = 0; pos < L; ++pos) acc += score[pos] * vh[pos*D + d];
    outh[d] = acc;
}

// ---------- 分页布局批量 decode ----------
__device__ inline int bp_offset(const int* table, int pos, int h, int H, int BLOCK, int D) {
    int phys = table[pos / BLOCK];
    int in   = pos % BLOCK;
    return ((phys * H + h) * BLOCK + in) * D;
}

__global__ void batched_paged(const float* q, const float* k_pool, const float* v_pool,
                              const int* table_all, const int* tab_off, const int* cur_len,
                              float* out, int H, int D, int BLOCK, int cap, float scale) {
    int s = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int L = cur_len[s];
    const int* table = table_all + tab_off[s];
    extern __shared__ float smem[];
    float* q_s   = smem;
    float* score = smem + D;
    float* red   = smem + D + cap;

    const float* qh = q   + ((size_t)s*H + h)*D;
    float* outh     = out + ((size_t)s*H + h)*D;

    q_s[d] = qh[d];
    __syncthreads();

    for (int pos = d; pos < L; pos += D) {
        const float* krow = k_pool + bp_offset(table, pos, h, H, BLOCK, D);
        float acc = 0.f;
        for (int i = 0; i < D; ++i) acc += q_s[i] * krow[i];
        score[pos] = acc * scale;
    }
    __syncthreads();

    softmax_inplace(score, L, red, d, D);

    float acc = 0.f;
    for (int pos = 0; pos < L; ++pos) {
        const float* vrow = v_pool + bp_offset(table, pos, h, H, BLOCK, D);
        acc += score[pos] * vrow[d];
    }
    outh[d] = acc;
}

void launch_batched_decode(const float* q, const float* k_cache, const float* v_cache,
                           const int* cur_len, float* out,
                           int N, int H, int D, int MAXS, int cap, float scale) {
    dim3 grid(N, H);
    size_t shmem = ((size_t)2*D + cap) * sizeof(float);
    batched_decode<<<grid, D, shmem>>>(q, k_cache, v_cache, cur_len, out, H, D, MAXS, cap, scale);
}

void launch_batched_paged(const float* q, const float* k_pool, const float* v_pool,
                          const int* table_all, const int* tab_off, const int* cur_len,
                          float* out, int N, int H, int D, int BLOCK, int cap, float scale) {
    dim3 grid(N, H);
    size_t shmem = ((size_t)2*D + cap) * sizeof(float);
    batched_paged<<<grid, D, shmem>>>(q, k_pool, v_pool, table_all, tab_off, cur_len,
                                      out, H, D, BLOCK, cap, scale);
}
