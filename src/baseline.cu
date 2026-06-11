#include "kv_cache.h"
#include <cuda_runtime.h>

// 模拟"没有 KV cache":每步都假装重新投影出全部 K/V(一次 [cur_len,D]x[D,D] 的乘法),
// 再走和 decode_attn 相同的 attention。重算的那部分就是 KV cache 省掉的计算。
__global__ void recompute_attn(const float* q,
                               const float* k_cache, const float* v_cache,
                               const float* Wk, const float* Wv,
                               float* out, int cur_len, int D, int S, float scale) {
    int h = blockIdx.x, d = threadIdx.x;
    extern __shared__ float smem[];
    float* q_s   = smem;
    float* score = smem + D;
    float* k_re  = smem + D + S;   // 重算出来的 K[pos] 暂存一行

    const float* qh = q + h * D;
    const float* kh = k_cache + h * S * D;
    const float* vh = v_cache + h * S * D;

    q_s[d] = qh[d];
    __syncthreads();

    // 关键差异:每个位置都重新算一遍 K[pos] = K_in[pos] @ Wk(O(t·D²) 的重复计算)
    for (int pos = 0; pos < cur_len; ++pos) {
        float acc = 0.f;
        for (int i = 0; i < D; ++i) acc += kh[pos * D + i] * Wk[i * D + d];
        k_re[d] = acc;
        __syncthreads();
        if (d == 0) {
            float s = 0.f;
            for (int i = 0; i < D; ++i) s += q_s[i] * k_re[i];
            score[pos] = s * scale;
        }
        __syncthreads();
    }

    if (d == 0) {
        float m = score[0];
        for (int i = 1; i < cur_len; ++i) m = fmaxf(m, score[i]);
        float s = 0.f;
        for (int i = 0; i < cur_len; ++i) { score[i] = expf(score[i]-m); s += score[i]; }
        for (int i = 0; i < cur_len; ++i) score[i] /= s;
    }
    __syncthreads();

    float acc = 0.f;
    for (int pos = 0; pos < cur_len; ++pos) acc += score[pos] * vh[pos * D + d];
    out[h * D + d] = acc;
}

void launch_recompute(const float* q, const float* k_cache, const float* v_cache,
                      const float* Wk, const float* Wv,
                      float* out, int cur_len, int H, int D, int S, float scale) {
    size_t shmem = (D + S + D) * sizeof(float);
    recompute_attn<<<H, D, shmem>>>(q, k_cache, v_cache, Wk, Wv, out, cur_len, D, S, scale);
} 