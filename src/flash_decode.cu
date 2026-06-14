#include "kv_cache.h"
#include <cuda_runtime.h>
#include <cfloat>

// Flash-attention 风格的 decode：在线 softmax (running max/sum) + split-K。
//
// 与 decode_attn.cu 的区别：
//   decode_attn 把整张 score[S] 落 shared memory，再两遍串行 softmax（找 max -> exp+sum）。
//   这里不 materialize score 表：每来一个 position 就用 running m/l 边算边 rescale 累加器，
//   单遍扫完即得结果（数值上等价，显存只占 O(D) 而非 O(S)）。
//   再把 KV 序列切成 num_splits 段交给多个 block 并行，最后一个 reduce kernel 合并。
//
// 布局：q[H,D]  k_cache/v_cache[H,S,D]  out[H,D]，与 decode_attn 一致。
// 约定：blockDim.x = D，且 D 为 2 的幂（树形 reduce）。RTX 3060 demo 里 D=64。

// ---- 阶段①：每个 block 处理 (head h, split) 的一段 KV，输出局部 (m, l, acc[D]) ----
// grid = (H, num_splits)，blockDim = D。
__global__ void flash_split(const float* q,
                            const float* k_cache, const float* v_cache,
                            float* part_acc,   // [H, num_splits, D]  m-移位后的未归一累加器
                            float* part_m,     // [H, num_splits]     该段 running max
                            float* part_l,     // [H, num_splits]     该段 running sum
                            int cur_len, int D, int S, float scale, int num_splits) {
    // 索引布局是：先按 head 分组，每个 head 下有 num_splits 个段，每个段有 D 个浮点数。
    int h     = blockIdx.x;
    int split = blockIdx.y;
    int d     = threadIdx.x;

    // 这一段负责的 KV 区间 [j0, j1)
    int chunk = (cur_len + num_splits - 1) / num_splits;
    int j0 = split * chunk;
    int j1 = min(j0 + chunk, cur_len);

    extern __shared__ float smem[];//shared memory，同一block内所有线程共享
    float* q_s = smem;          // [D]  query 搬进 shared 复用
    float* red = smem + D;      // [D]  点积的树形 reduce 缓冲

    const float* qh = q + h * D;
    const float* kh = k_cache + h * S * D;
    const float* vh = v_cache + h * S * D;

    q_s[d] = qh[d];
    __syncthreads();

    // 在线 softmax 的 running 状态：m/l 是标量，每个线程各存一份（都从同一个 score 推出，恒等）。
    // acc是定义在核函数（__global__）内的自动变量（位于寄存器），每个线程各持有一个独立副本，
    //每个线程都能访问自己的 acc，但不同线程的 acc 是不同内存位置，互不干扰。acc 按维度 d 分到每个线程的寄存器里。
    //acc是线程私有的累加器
    float m = -FLT_MAX, l = 0.f, acc = 0.f;

    for (int j = j0; j < j1; ++j) {
        // 1) score_j = (q · K_j) * scale —— 树形 reduce over D 
        red[d] = q_s[d] * kh[j * D + d];
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        float s = red[0] * scale;   // 所有线程读同一个标量
        __syncthreads();            // 读完再进下一轮，避免下次写 red 时还有线程在读

        // 2) 在线更新 running max / sum / 累加器
        float m_new = fmaxf(m, s);
        float corr  = __expf(m - m_new);   // 旧累加量的缩放因子（首个元素时 m=-FLT_MAX -> corr=0）
        float p     = __expf(s - m_new);   // 当前 token 的权重
        l   = l   * corr + p;
        acc = acc * corr + p * vh[j * D + d];
        m   = m_new;
    }

    // 3) 写出该段局部结果
    long base = ((long)h * num_splits + split) * D + d;
    part_acc[base] = acc;
    if (d == 0) {
        part_m[h * num_splits + split] = m;
        part_l[h * num_splits + split] = l;
    }
}

// ---- 阶段②：跨 split 合并 —— 标准 flash 合并公式 ----
// grid = H，blockDim = D。
__global__ void flash_reduce(const float* part_acc, const float* part_m, const float* part_l,
                             float* out, int D, int num_splits) {
    int h = blockIdx.x;
    int d = threadIdx.x;

    extern __shared__ float pm[];   // [num_splits]  各段的 m，先求全局最大
    for (int s = d; s < num_splits; s += D) pm[s] = part_m[h * num_splits + s];
    __syncthreads();

    float M = -FLT_MAX;
    for (int s = 0; s < num_splits; ++s) M = fmaxf(M, pm[s]);

    float acc = 0.f, l = 0.f;
    for (int s = 0; s < num_splits; ++s) {
        float w = __expf(pm[s] - M);                       // 把各段对齐到全局 max
        acc += w * part_acc[((long)h * num_splits + s) * D + d];
        l   += w * part_l[h * num_splits + s];
    }
    out[h * D + d] = acc / l;
}

void launch_flash_decode(const float* q, const float* k_cache, const float* v_cache,
                         float* out, int cur_len, int H, int D, int S, float scale,
                         int num_splits) {
    if (num_splits < 1) num_splits = 1;
    if (num_splits > cur_len) num_splits = cur_len;   // 空段没意义

    float *part_acc, *part_m, *part_l;
    cudaMalloc(&part_acc, (size_t)H * num_splits * D * sizeof(float));
    cudaMalloc(&part_m,   (size_t)H * num_splits     * sizeof(float));
    cudaMalloc(&part_l,   (size_t)H * num_splits     * sizeof(float));

    dim3 grid_s(H, num_splits);
    size_t shmem_s = (size_t)(2 * D) * sizeof(float);   // q_s[D] + red[D]
    flash_split<<<grid_s, D, shmem_s>>>(q, k_cache, v_cache,
                                        part_acc, part_m, part_l,
                                        cur_len, D, S, scale, num_splits);

    size_t shmem_r = (size_t)num_splits * sizeof(float);
    flash_reduce<<<H, D, shmem_r>>>(part_acc, part_m, part_l, out, D, num_splits);

    cudaFree(part_acc);
    cudaFree(part_m);
    cudaFree(part_l);
}
