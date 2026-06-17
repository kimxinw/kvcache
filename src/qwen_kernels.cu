// qwen_kernels.cu —— 见 qwen_kernels.h。
//   激活/权重/KV 池全程 fp16 (half) 存储，kernel 内部一律用 fp32 累加/三角函数，
//   只在读入(h2f)和写回(f2h)处转换。argmax 读的是 lm_head 的 fp32 logits，保持 float。
#include "qwen_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <math.h>

__device__ __forceinline__ float h2f(half x) { return __half2float(x); }
__device__ __forceinline__ half  f2h(float x) { return __float2half(x); }

// ---------- embedding gather: out[L,H] = embed[ids[r]] ----------
__global__ void k_embed(half* out, const half* embed, const int* ids, int L, int H,cudaStream_t stream) {
    int r = blockIdx.x;
    int tok = ids[r];
    const half* src = embed + (size_t)tok * H;
    half* dst = out + (size_t)r * H;
    for(int i =threadIdx.x;i<H;i+=blockDim.x)dst[i] = src[i];   // half 直拷，无需转换
}
void qk_embed_gather(half* out, const half* embed, const int* ids, int L, int H,cudaStream_t stream) {
    k_embed<<<L, 256,0,stream>>>(out, embed, ids, L, H,stream);
}

// ---------- RMSNorm: 每行 out = x * rsqrt(mean(x^2)+eps) * w ----------
__global__ void k_rmsnorm(half* out, const half* in, const half* w, int H, float eps,cudaStream_t stream) {
    int r = blockIdx.x;
    const half* x = in + (size_t)r * H;
    half* y = out + (size_t)r * H;
    __shared__ float red[256];
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) { float v = h2f(x[i]); local += v * v; }
    red[threadIdx.x] = local; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) y[i] = f2h(h2f(x[i]) * inv * h2f(w[i]));
}
void qk_rmsnorm(half* out, const half* in, const half* w, int L, int H, float eps,cudaStream_t stream) {
    k_rmsnorm<<<L, 256,0,stream>>>(out, in, w, H, eps,stream);
}

// ---------- 行广播加 bias: x[L,N] += bias[N] ----------
__global__ void k_add_bias(half* x, const half* bias, int L, int N,cudaStream_t stream) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < L * N) x[idx] = f2h(h2f(x[idx]) + h2f(bias[idx % N]));
}
void qk_add_bias(half* x, const half* bias, int L, int N,cudaStream_t stream ) {
    int n = L * N; k_add_bias<<<(n + 255) / 256, 256,0,stream>>>(x, bias, L, N,stream);
}

// ---------- 残差: x += y ----------
__global__ void k_add(half* x, const half* y, int n,cudaStream_t stream) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = f2h(h2f(x[i]) + h2f(y[i]));
}
void qk_add_residual(half* x, const half* y, int n,cudaStream_t stream) {
    k_add<<<(n + 255) / 256, 256,0,stream>>>(x, y, n,stream);
}

// ---------- RoPE (HF rotate_half 风格)。x 视作 [L*n_heads, D]，每个 (row,head) 一个 D 向量 ----------
// 位置 pos = pos_base + row；inv_freq[t] = theta^(-2t/D)，t in [0,D/2)。
__global__ void k_rope(half* x, int n_heads, int D, int pos_base, float theta,cudaStream_t stream) {
    int blk = blockIdx.x;            // = row*n_heads + head
    int row = blk / n_heads;
    int half_d = D >> 1;
    int t = threadIdx.x;             // [0, half)
    half* base = x + (size_t)blk * D;
    float pos = (float)(pos_base + row);
    float inv = __expf(-logf(theta) * (2.0f * t / D));
    float ang = pos * inv;
    float c = cosf(ang), s = sinf(ang);
    float x1 = h2f(base[t]), x2 = h2f(base[t + half_d]);
    base[t]          = f2h(x1 * c - x2 * s);
    base[t + half_d] = f2h(x2 * c + x1 * s);
}
void qk_rope(half* x, int L, int n_heads, int D, int pos_base, float theta,cudaStream_t stream) {
    k_rope<<<L * n_heads, D / 2,0,stream>>>(x, n_heads, D, pos_base, theta,stream);
}

__global__ void kcuda_rope(half* x, int n_heads, int D, int* pos_base, float theta,cudaStream_t stream) {
    int blk = blockIdx.x;            // = row*n_heads + head
    int row = blk / n_heads;
    int half_d = D >> 1;
    int t = threadIdx.x;             // [0, half)
    half* base = x + (size_t)blk * D;
    float pos = (float)(*pos_base + row);
    float inv = __expf(-logf(theta) * (2.0f * t / D));
    float ang = pos * inv;
    float c = cosf(ang), s = sinf(ang);
    float x1 = h2f(base[t]), x2 = h2f(base[t + half_d]);
    base[t]          = f2h(x1 * c - x2 * s);
    base[t + half_d] = f2h(x2 * c + x1 * s);
}
void qkcuda_rope(half* x, int L, int n_heads, int D, int* pos_base, float theta,cudaStream_t stream) {
    kcuda_rope<<<L * n_heads, D / 2,0,stream>>>(x, n_heads, D, pos_base, theta,stream);
}

// ---------- SwiGLU: out = silu(gate) * up ----------
__global__ void k_silu_mul(half* out, const half* gate, const half* up, int n,cudaStream_t stream) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = h2f(gate[i]); out[i] = f2h((g / (1.f + __expf(-g))) * h2f(up[i])); }
}
void qk_silu_mul(half* out, const half* gate, const half* up, int n,cudaStream_t stream) {
    k_silu_mul<<<(n + 255) / 256, 256,0,stream>>>(out, gate, up, n,stream);
}

// ---------- 写 K/V 入分页池 [num_blocks, Hkv, BLOCK, D] ----------
__global__ void k_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                           const int* bt, int start_pos, int Hkv, int D, int BLOCK,cudaStream_t stream) {
    int blk = blockIdx.x;            // = r*Hkv + hk
    int r = blk / Hkv, hk = blk % Hkv;//此处的线性关系看CPU调用核函数的签名L*Hkv
    int pos = start_pos + r;
    int phys = bt[pos / BLOCK], in = pos % BLOCK;
    size_t dst = (((size_t)phys * Hkv + hk) * BLOCK + in) * D;
    size_t src = (size_t)blk * D;
    int d = threadIdx.x;
    k_pool[dst + d] = k[src + d];
    v_pool[dst + d] = v[src + d];
}
void qk_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                 const int* block_table_dev, int start_pos, int L, int Hkv, int D, int BLOCK,cudaStream_t stream) {
    k_write_kv<<<L * Hkv, D,0,stream>>>(k, v, k_pool, v_pool, block_table_dev, start_pos, Hkv, D, BLOCK,stream);
}

__global__ void kcuda_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                           const int* bt, int* start_pos, int Hkv, int D, int BLOCK,cudaStream_t stream) {
    int blk = blockIdx.x;            // = r*Hkv + hk
    int r = blk / Hkv, hk = blk % Hkv;//此处的线性关系看CPU调用核函数的签名L*Hkv
    int pos = *start_pos + r;
    int phys = bt[pos / BLOCK], in = pos % BLOCK;
    size_t dst = (((size_t)phys * Hkv + hk) * BLOCK + in) * D;
    size_t src = (size_t)blk * D;
    int d = threadIdx.x;
    k_pool[dst + d] = k[src + d];
    v_pool[dst + d] = v[src + d];
}
void qkcuda_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                 const int* block_table_dev, int* start_pos, int L, int Hkv, int D, int BLOCK,cudaStream_t stream) {
    kcuda_write_kv<<<L * Hkv, D,0,stream>>>(k, v, k_pool, v_pool, block_table_dev, start_pos, Hkv, D, BLOCK,stream);
}

// ---------- GQA 因果分页 attention，flash 在线 softmax ----------
__global__ void k_paged_attn(const half* q, half* out, const half* k_pool, const half* v_pool,
                             const int* bt, int q_pos_base, int Hq, int Hkv, int D, int BLOCK, float scale,cudaStream_t stream) {
    int qi = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int kv_head = h / (Hq / Hkv);            //GQA/MQA核心：Hq/Hkv个query共享一个kv head
    int Lc = q_pos_base + qi + 1;            // 因果长度，当前query能看到多少个token(include itself)
    extern __shared__ float smem[];
    float* q_s = smem;                       // [D]
    float* red = smem + D;                   // [D]
    const half* qh = q + ((size_t)qi * Hq + h) * D;
    half* outh     = out + ((size_t)qi * Hq + h) * D;
    q_s[d] = h2f(qh[d]);
    __syncthreads();

    float m = -FLT_MAX, l = 0.f, acc = 0.f;
    for (int pos = 0; pos < Lc; ++pos) {
        int phys = bt[pos / BLOCK], in = pos % BLOCK;//虚拟快、物理块一个块都有BLOCK个token
        size_t off = (((size_t)phys * Hkv + kv_head) * BLOCK + in) * D;
        red[d] = q_s[d] * h2f(k_pool[off + d]);//k+pool[num_blocks, Hkv, BLOCK, D]
        __syncthreads();
        for (int s = D >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; __syncthreads(); }
        float sc = red[0] * scale;
        __syncthreads();
        float vd = h2f(v_pool[off + d]);//k_pool、v_pool本质上都是一维数组
        float m_new = fmaxf(m, sc);
        float corr = __expf(m - m_new);
        float p = __expf(sc - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vd;
        m = m_new;
    }
    outh[d] = f2h((l > 0.f) ? acc / l : 0.f);
}
void qk_paged_causal_attn(const half* q, half* out, const half* k_pool, const half* v_pool,
                          const int* block_table_dev, int q_pos_base,
                          int Nq, int Hq, int Hkv, int D, int BLOCK, float scale,cudaStream_t stream) {
    dim3 grid(Nq, Hq);
    size_t shmem = (size_t)2 * D * sizeof(float);
    k_paged_attn<<<grid, D, shmem,stream>>>(q, out, k_pool, v_pool, block_table_dev, q_pos_base,
                                     Hq, Hkv, D, BLOCK, scale,stream);
}

__global__ void kcuda_paged_attn(const half* q, half* out, const half* k_pool, const half* v_pool,
                             const int* bt, int* q_pos_base, int Hq, int Hkv, int D, int BLOCK, float scale,cudaStream_t stream) {
    int qi = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int kv_head = h / (Hq / Hkv);            //GQA/MQA核心：Hq/Hkv个query共享一个kv head
    int Lc = *q_pos_base + qi + 1;            // 因果长度，当前query能看到多少个token(include itself)
    extern __shared__ float smem[];
    float* q_s = smem;                       // [D]
    float* red = smem + D;                   // [D]
    const half* qh = q + ((size_t)qi * Hq + h) * D;
    half* outh     = out + ((size_t)qi * Hq + h) * D;
    q_s[d] = h2f(qh[d]);
    __syncthreads();

    float m = -FLT_MAX, l = 0.f, acc = 0.f;
    for (int pos = 0; pos < Lc; ++pos) {
        int phys = bt[pos / BLOCK], in = pos % BLOCK;//虚拟快、物理块一个块都有BLOCK个token
        size_t off = (((size_t)phys * Hkv + kv_head) * BLOCK + in) * D;
        red[d] = q_s[d] * h2f(k_pool[off + d]);//k+pool[num_blocks, Hkv, BLOCK, D]
        __syncthreads();
        for (int s = D >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; __syncthreads(); }
        float sc = red[0] * scale;
        __syncthreads();
        float vd = h2f(v_pool[off + d]);//k_pool、v_pool本质上都是一维数组
        float m_new = fmaxf(m, sc);
        float corr = __expf(m - m_new);
        float p = __expf(sc - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vd;
        m = m_new;
    }
    outh[d] = f2h((l > 0.f) ? acc / l : 0.f);
}
void qkcuda_paged_causal_attn(const half* q, half* out, const half* k_pool, const half* v_pool,
                          const int* block_table_dev, int* q_pos_base,
                          int Nq, int Hq, int Hkv, int D, int BLOCK, float scale,cudaStream_t stream) {
    dim3 grid(Nq, Hq);
    size_t shmem = (size_t)2 * D * sizeof(float);
    kcuda_paged_attn<<<grid, D, shmem,stream>>>(q, out, k_pool, v_pool, block_table_dev, q_pos_base,
                                     Hq, Hkv, D, BLOCK, scale,stream);
}

// ---------- argmax (logits 仍是 fp32：lm_head GEMM 输出 CUDA_R_32F) ----------
__global__ void k_argmax(const float* x, int n, int* out_idx,cudaStream_t stream) {
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
void qk_argmax(const float* logits, int n, int* d_idx,cudaStream_t stream) {
    k_argmax<<<1, 256,0,stream>>>(logits, n, d_idx,stream);
}

// ===================== 里程碑2 批量 kernel =====================

// 逐行位置 RoPE（与 k_rope 同算法，位置改成 pos[row]）
__global__ void k_rope_pos(half* x, int n_heads, int D, const int* pos, float theta) {
    int blk = blockIdx.x;            // = row*n_heads + head
    int row = blk / n_heads;
    int half_d = D >> 1, t = threadIdx.x;
    half* base = x + (size_t)blk * D;
    float p = (float)pos[row];
    float inv = __expf(-logf(theta) * (2.0f * t / D));
    float ang = p * inv, c = cosf(ang), s = sinf(ang);
    float x1 = h2f(base[t]), x2 = h2f(base[t + half_d]);
    base[t]          = f2h(x1 * c - x2 * s);
    base[t + half_d] = f2h(x2 * c + x1 * s);
}
void qk_rope_pos(half* x, int n_rows, int n_heads, int D, const int* pos, float theta) {
    k_rope_pos<<<n_rows * n_heads, D / 2>>>(x, n_heads, D, pos, theta);
}

// 批量写 KV：第 r 行(序列) 的块表 = table_all+tab_off[r]，写位置 wpos[r]
__global__ void k_batched_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                                   const int* table_all, const int* tab_off, const int* wpos,
                                   int Hkv, int D, int BLOCK) {
    int blk = blockIdx.x;            // = r*Hkv + hk
    //这里blockIdx.x = r*Hkv + hk 纯个人喜好，也可以blockIdx.x = r,blockIdx.y = hk
    //不过下面CPU主机端调用函数时，需要dim grid(L,Hkv),k_batched_write_kv<<<grid,D>>>
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
void qk_batched_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                         const int* table_all, const int* tab_off, const int* wpos,
                         int N, int Hkv, int D, int BLOCK) {
    k_batched_write_kv<<<N * Hkv, D>>>(k, v, k_pool, v_pool, table_all, tab_off, wpos, Hkv, D, BLOCK);
}

// 批量 GQA 分页 decode attention：grid=(N,Hq)，每序列查询注意 [0,cur_len[s])
__global__ void k_batched_paged_decode_attn(const half* q, half* out,
                                            const half* k_pool, const half* v_pool,
                                            const int* table_all, const int* tab_off, const int* cur_len,
                                            int Hq, int Hkv, int D, int BLOCK, float scale) {
    //该kernels函数分配了N*Hq个block，每个block有D个线程进行并行计算
    int s = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int kv_head = h / (Hq / Hkv);
    int L = cur_len[s];
    const int* table = table_all + tab_off[s];
    extern __shared__ float smem[];
    float* q_s = smem;             // [D]
    float* red = smem + D;         // [D]
    const half* qh = q + ((size_t)s * Hq + h) * D;
    half* outh     = out + ((size_t)s * Hq + h) * D;
    q_s[d] = h2f(qh[d]);
    __syncthreads();

    float m = -FLT_MAX, l = 0.f, acc = 0.f;
    for (int pos = 0; pos < L; ++pos) {
        int phys = table[pos / BLOCK], in = pos % BLOCK;
        size_t off = (((size_t)phys * Hkv + kv_head) * BLOCK + in) * D;
        red[d] = q_s[d] * h2f(k_pool[off + d]);
        __syncthreads();
        for (int st = D >> 1; st > 0; st >>= 1) { if (d < st) red[d] += red[d + st]; __syncthreads(); }
        float sc = red[0] * scale;
        __syncthreads();
        float vd = h2f(v_pool[off + d]);
        float m_new = fmaxf(m, sc);
        float corr = __expf(m - m_new), p = __expf(sc - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vd;
        m = m_new;
    }
    outh[d] = f2h((l > 0.f) ? acc / l : 0.f);
}
void qk_batched_paged_decode_attn(const half* q, half* out, const half* k_pool, const half* v_pool,
                                  const int* table_all, const int* tab_off, const int* cur_len,
                                  int N, int Hq, int Hkv, int D, int BLOCK, float scale) {
    dim3 grid(N, Hq);
    size_t shmem = (size_t)2 * D * sizeof(float);
    k_batched_paged_decode_attn<<<grid, D, shmem>>>(q, out, k_pool, v_pool, table_all, tab_off,
                                                    cur_len, Hq, Hkv, D, BLOCK, scale);
}
