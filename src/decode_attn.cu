#include "kv_cache.h"
#include <cuda_runtime.h>

// 一个 block 处理一个 head; blockDim.x = D。三段:算分 -> softmax -> 加权和。
__global__ void decode_attn(const float* q,
                            const float* k_cache, const float* v_cache,
                            float* out, int cur_len, int D, int S, float scale) {
    int h = blockIdx.x; //按照grid分头
    int d = threadIdx.x;//按照线程分维度 0..D-1
    extern __shared__ float smem[];       // 大小 = D + S
    float* q_s   = smem;                  // [D]
    float* score = smem + D;              // [S]

    const float* qh = q + h * D;
    const float* kh = k_cache + h * S * D;
    const float* vh = v_cache + h * S * D;

    q_s[d] = qh[d];                       // 1) q 搬进 shared
    __syncthreads();

    for (int pos = d; pos < cur_len; pos += D) {   // 2) score = q·K[pos]*scale
        float acc = 0.f;
        for (int i = 0; i < D; ++i) acc += q_s[i] * kh[pos * D + i];
        score[pos] = acc * scale;
    }
    __syncthreads();
    // 屏障：Block 内所有线程都必须到达这个位置才继续
    // 保证共享内存可见性：一个线程写入 smem 的内容，其他线程在 __syncthreads() 之后可以安全读取

    // ====== 3) TODO: 把 score[0..cur_len) 做成 softmax 概率 ======
    // 暂时先让 0 号线程串行做
      if (d == 0) {
          float m = score[0];
          for (int i = 1; i < cur_len; ++i) m = fmaxf(m, score[i]); // 数值稳定: 减最大值
          float s = 0.f;
          for (int i = 0; i < cur_len; ++i) { score[i] = expf(score[i]-m); s += score[i]; }
          for (int i = 0; i < cur_len; ++i) score[i] /= s;
      }
    // 跑通后再换成 block 内并行 reduce 求 max / sum(L3 会用到, 现在先不急)。
    __syncthreads();
    // ====================================================================

    float acc = 0.f;                      // 4) out[d] = Σ_pos prob[pos]*V[pos][d]
    for (int pos = 0; pos < cur_len; ++pos) acc += score[pos] * vh[pos * D + d];
    out[h * D + d] = acc;
}

void launch_decode(const float* q, const float* k_cache, const float* v_cache,
                   float* out, int cur_len, int H, int D, int S, float scale) {
    size_t shmem = (D + S) * sizeof(float);
    decode_attn<<<H, D, shmem>>>(q, k_cache, v_cache, out, cur_len, D, S, scale);
}