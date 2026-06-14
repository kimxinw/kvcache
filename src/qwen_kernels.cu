// qwen_kernels.cu —— 见 qwen_kernels.h。全程 fp32。
#include "qwen_kernels.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

// ---------- embedding gather: out[L,H] = embed[ids[r]] ----------
__global__ void k_embed(float* out, const float* embed, const int* ids, int L, int H) {
    int r = blockIdx.x;
    int tok = ids[r];
    const float* src = embed + (size_t)tok * H;
    float* dst = out + (size_t)r * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = src[i];
}
void qk_embed_gather(float* out, const float* embed, const int* ids, int L, int H) {
    k_embed<<<L, 256>>>(out, embed, ids, L, H);
}

// ---------- RMSNorm: 每行 out = x * rsqrt(mean(x^2)+eps) * w ----------
__global__ void k_rmsnorm(float* out, const float* in, const float* w, int H, float eps) {
    int r = blockIdx.x;
    const float* x = in + (size_t)r * H;
    float* y = out + (size_t)r * H;
    __shared__ float red[256];
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) local += x[i] * x[i];
    red[threadIdx.x] = local; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) y[i] = x[i] * inv * w[i];
}
void qk_rmsnorm(float* out, const float* in, const float* w, int L, int H, float eps) {
    k_rmsnorm<<<L, 256>>>(out, in, w, H, eps);
}

// ---------- 行广播加 bias: x[L,N] += bias[N] ----------
__global__ void k_add_bias(float* x, const float* bias, int L, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < L * N) x[idx] += bias[idx % N];
}
void qk_add_bias(float* x, const float* bias, int L, int N) {
    int n = L * N; k_add_bias<<<(n + 255) / 256, 256>>>(x, bias, L, N);
}

// ---------- 残差: x += y ----------
__global__ void k_add(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
void qk_add_residual(float* x, const float* y, int n) {
    k_add<<<(n + 255) / 256, 256>>>(x, y, n);
}

// ---------- RoPE (HF rotate_half 风格)。x 视作 [L*n_heads, D]，每个 (row,head) 一个 D 向量 ----------
// 位置 pos = pos_base + row；inv_freq[t] = theta^(-2t/D)，t in [0,D/2)。
__global__ void k_rope(float* x, int n_heads, int D, int pos_base, float theta) {
    int blk = blockIdx.x;            // = row*n_heads + head
    int row = blk / n_heads;
    int half = D >> 1;
    int t = threadIdx.x;             // [0, half)
    float* base = x + (size_t)blk * D;
    float pos = (float)(pos_base + row);
    float inv = __expf(-logf(theta) * (2.0f * t / D));
    float ang = pos * inv;
    float c = cosf(ang), s = sinf(ang);
    float x1 = base[t], x2 = base[t + half];
    base[t]        = x1 * c - x2 * s;
    base[t + half] = x2 * c + x1 * s;
}
void qk_rope(float* x, int L, int n_heads, int D, int pos_base, float theta) {
    k_rope<<<L * n_heads, D / 2>>>(x, n_heads, D, pos_base, theta);
}

// ---------- SwiGLU: out = silu(gate) * up ----------
__global__ void k_silu_mul(float* out, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; out[i] = (g / (1.f + __expf(-g))) * up[i]; }
}
void qk_silu_mul(float* out, const float* gate, const float* up, int n) {
    k_silu_mul<<<(n + 255) / 256, 256>>>(out, gate, up, n);
}

// ---------- 写 K/V 入分页池 [num_blocks, Hkv, BLOCK, D] ----------
__global__ void k_write_kv(const float* k, const float* v, float* k_pool, float* v_pool,
                           const int* bt, int start_pos, int Hkv, int D, int BLOCK) {
    int blk = blockIdx.x;            // = r*Hkv + hk
    int r = blk / Hkv, hk = blk % Hkv;
    int pos = start_pos + r;
    int phys = bt[pos / BLOCK], in = pos % BLOCK;
    size_t dst = (((size_t)phys * Hkv + hk) * BLOCK + in) * D;
    size_t src = (size_t)blk * D;
    int d = threadIdx.x;
    k_pool[dst + d] = k[src + d];
    v_pool[dst + d] = v[src + d];
}
void qk_write_kv(const float* k, const float* v, float* k_pool, float* v_pool,
                 const int* block_table_dev, int start_pos, int L, int Hkv, int D, int BLOCK) {
    k_write_kv<<<L * Hkv, D>>>(k, v, k_pool, v_pool, block_table_dev, start_pos, Hkv, D, BLOCK);
}

// ---------- GQA 因果分页 attention，flash 在线 softmax ----------
__global__ void k_paged_attn(const float* q, float* out, const float* k_pool, const float* v_pool,
                             const int* bt, int q_pos_base, int Hq, int Hkv, int D, int BLOCK, float scale) {
    int qi = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int kv_head = h / (Hq / Hkv);
    int Lc = q_pos_base + qi + 1;            // 因果长度
    extern __shared__ float smem[];
    float* q_s = smem;                       // [D]
    float* red = smem + D;                   // [D]
    const float* qh = q + ((size_t)qi * Hq + h) * D;
    float* outh    = out + ((size_t)qi * Hq + h) * D;
    q_s[d] = qh[d];
    __syncthreads();

    float m = -FLT_MAX, l = 0.f, acc = 0.f;
    for (int pos = 0; pos < Lc; ++pos) {
        int phys = bt[pos / BLOCK], in = pos % BLOCK;
        size_t off = (((size_t)phys * Hkv + kv_head) * BLOCK + in) * D;
        red[d] = q_s[d] * k_pool[off + d];
        __syncthreads();
        for (int s = D >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; __syncthreads(); }
        float sc = red[0] * scale;
        __syncthreads();
        float vd = v_pool[off + d];
        float m_new = fmaxf(m, sc);
        float corr = __expf(m - m_new);
        float p = __expf(sc - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vd;
        m = m_new;
    }
    outh[d] = (l > 0.f) ? acc / l : 0.f;
}
void qk_paged_causal_attn(const float* q, float* out, const float* k_pool, const float* v_pool,
                          const int* block_table_dev, int q_pos_base,
                          int Nq, int Hq, int Hkv, int D, int BLOCK, float scale) {
    dim3 grid(Nq, Hq);
    size_t shmem = (size_t)2 * D * sizeof(float);
    k_paged_attn<<<grid, D, shmem>>>(q, out, k_pool, v_pool, block_table_dev, q_pos_base,
                                     Hq, Hkv, D, BLOCK, scale);
}

// ---------- argmax ----------
__global__ void k_argmax(const float* x, int n, int* out_idx) {
    __shared__ float vbest[256];
    __shared__ int   ibest[256];
    float vb = -FLT_MAX; int ib = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) if (x[i] > vb) { vb = x[i]; ib = i; }
    vbest[threadIdx.x] = vb; ibest[threadIdx.x] = ib; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (vbest[threadIdx.x + s] > vbest[threadIdx.x]) {
                vbest[threadIdx.x] = vbest[threadIdx.x + s];
                ibest[threadIdx.x] = ibest[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out_idx = ibest[0];
}
void qk_argmax(const float* logits, int n, int* d_idx) {
    k_argmax<<<1, 256>>>(logits, n, d_idx);
}

// ===================== 里程碑2 批量 kernel =====================

// 逐行位置 RoPE（与 k_rope 同算法，位置改成 pos[row]）
__global__ void k_rope_pos(float* x, int n_heads, int D, const int* pos, float theta) {
    int blk = blockIdx.x;            // = row*n_heads + head
    int row = blk / n_heads;
    int half = D >> 1, t = threadIdx.x;
    float* base = x + (size_t)blk * D;
    float p = (float)pos[row];
    float inv = __expf(-logf(theta) * (2.0f * t / D));
    float ang = p * inv, c = cosf(ang), s = sinf(ang);
    float x1 = base[t], x2 = base[t + half];
    base[t]        = x1 * c - x2 * s;
    base[t + half] = x2 * c + x1 * s;
}
void qk_rope_pos(float* x, int n_rows, int n_heads, int D, const int* pos, float theta) {
    k_rope_pos<<<n_rows * n_heads, D / 2>>>(x, n_heads, D, pos, theta);
}

// 批量写 KV：第 r 行(序列) 的块表 = table_all+tab_off[r]，写位置 wpos[r]
__global__ void k_batched_write_kv(const float* k, const float* v, float* k_pool, float* v_pool,
                                   const int* table_all, const int* tab_off, const int* wpos,
                                   int Hkv, int D, int BLOCK) {
    int blk = blockIdx.x;            // = r*Hkv + hk
    int r = blk / Hkv, hk = blk % Hkv;
    const int* table = table_all + tab_off[r];
    int pos = wpos[r];
    int phys = table[pos / BLOCK], in = pos % BLOCK;
    size_t dst = (((size_t)phys * Hkv + hk) * BLOCK + in) * D;
    size_t src = (size_t)blk * D;
    int d = threadIdx.x;
    k_pool[dst + d] = k[src + d];
    v_pool[dst + d] = v[src + d];
}
void qk_batched_write_kv(const float* k, const float* v, float* k_pool, float* v_pool,
                         const int* table_all, const int* tab_off, const int* wpos,
                         int N, int Hkv, int D, int BLOCK) {
    k_batched_write_kv<<<N * Hkv, D>>>(k, v, k_pool, v_pool, table_all, tab_off, wpos, Hkv, D, BLOCK);
}

// 批量 GQA 分页 decode attention：grid=(N,Hq)，每序列查询注意 [0,cur_len[s])
__global__ void k_batched_paged_decode_attn(const float* q, float* out,
                                            const float* k_pool, const float* v_pool,
                                            const int* table_all, const int* tab_off, const int* cur_len,
                                            int Hq, int Hkv, int D, int BLOCK, float scale) {
    int s = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int kv_head = h / (Hq / Hkv);
    int L = cur_len[s];
    const int* table = table_all + tab_off[s];
    extern __shared__ float smem[];
    float* q_s = smem;             // [D]
    float* red = smem + D;         // [D]
    const float* qh = q + ((size_t)s * Hq + h) * D;
    float* outh    = out + ((size_t)s * Hq + h) * D;
    q_s[d] = qh[d];
    __syncthreads();

    float m = -FLT_MAX, l = 0.f, acc = 0.f;
    for (int pos = 0; pos < L; ++pos) {
        int phys = table[pos / BLOCK], in = pos % BLOCK;
        size_t off = (((size_t)phys * Hkv + kv_head) * BLOCK + in) * D;
        red[d] = q_s[d] * k_pool[off + d];
        __syncthreads();
        for (int st = D >> 1; st > 0; st >>= 1) { if (d < st) red[d] += red[d + st]; __syncthreads(); }
        float sc = red[0] * scale;
        __syncthreads();
        float vd = v_pool[off + d];
        float m_new = fmaxf(m, sc);
        float corr = __expf(m - m_new), p = __expf(sc - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vd;
        m = m_new;
    }
    outh[d] = (l > 0.f) ? acc / l : 0.f;
}
void qk_batched_paged_decode_attn(const float* q, float* out, const float* k_pool, const float* v_pool,
                                  const int* table_all, const int* tab_off, const int* cur_len,
                                  int N, int Hq, int Hkv, int D, int BLOCK, float scale) {
    dim3 grid(N, Hq);
    size_t shmem = (size_t)2 * D * sizeof(float);
    k_batched_paged_decode_attn<<<grid, D, shmem>>>(q, out, k_pool, v_pool, table_all, tab_off,
                                                    cur_len, Hq, Hkv, D, BLOCK, scale);
}
