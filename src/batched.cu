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
//文中的所有d，都是线程索引
//block_reduce中，多个线程用来求一个token的maxD和sumD
//softmax
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
    int s = blockIdx.x, h = blockIdx.y, d = threadIdx.x;//s是batch序列索引,h是该batch下的第几个head，d是维度索引[0,D-1];
    int L = cur_len[s];//第s批prom的序列长度（token数）
    extern __shared__ float smem[];
    //smem 本身就是指向共享内存起始地址的指针。
    float* q_s   = smem;            // [D]
    //q_s 指向这块内存的开头，被当作一个长度为 D 的 float 数组使用（即占用 D * sizeof(float) 字节）。
    float* score = smem + D;        // [cap]
    //smem + D 是指针算术：跳过前 D 个 float 元素（即 q_s 占用的空间），指向接下来的位置
    //score 被当作长度为 cap 的数组使用（占用 cap * sizeof(float) 字节）。
    float* red   = smem + D + cap;  // [D]
    //smem + D + cap 再跳过 score 占用的 cap 个元素，指向更后面的位置。
    //red被当做长度为 D 的数组使用（占用D * sizeof(float)字节）

    const float* qh = q       + ((size_t)s*H + h)*D;
    const float* kh = k_cache + ((size_t)s*H + h)*MAXS*D;//MAXS是所有批次里最长的序列token个数
    const float* vh = v_cache + ((size_t)s*H + h)*MAXS*D;
    float* outh     = out     + ((size_t)s*H + h)*D;

    q_s[d] = qh[d];
    __syncthreads();
    //这里的pos=d,对应于不同的线程，D是总的线程数，由CPU主机端调用时通过<<<H,D>>>传入
    //pos是token的索引，就是该batch下的第几个token
    for (int pos = d; pos < L; pos += D) {           // score = q·K[pos]*scale
        float acc = 0.f;
        for (int i = 0; i < D; ++i) acc += q_s[i] * kh[pos*D + i];//kh[pos*D+i]代表第pos个token的第i维;
        score[pos] = acc * scale;
        //此时该批次下的第pos个token的q*kT记录在acc中了，乘以scal==1/sqrt(D)后记录在score[pos]中
    }
    __syncthreads();
    // 屏障：Block 内所有线程都必须到达这个位置才继续
    // 保证共享内存可见性：一个线程写入 smem 的内容，其他线程在 __syncthreads() 之后可以安全读取
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
    //D是线程数
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

// ---------- 分页布局批量 decode · flash 风格在线 softmax (引擎权威 kernel) ----------
// 与上面的 batched_paged 算法等价、输出一致, 区别在 softmax 的做法:
//   batched_paged : 先把整张 score[cap] 落 shared, 再两遍并行 reduce (max -> exp+sum)。
//   这里(flash)   : 不 materialize score —— 单遍扫 KV, 用 running (m,l,acc) 边算边 rescale,
//                   shared 只占 2*D(q_s + red), 与序列长度 cap 无关。长序列省下 O(cap) shared,
//                   也省掉一遍 KV 扫描。这是 ModelRunner/Engine 实际调用的那条路径。
// GQA 友好点: 取 K/V 用的 head 是 'h'(= q 的 head); 接 GQA 时把这里换成 h / (n_q/n_kv) 的
//   kv_head 映射、并让 grid.y = n_q_heads 即可, 池布局(n_kv_heads)与本 kernel 结构都不动。
__global__ void batched_paged_flash(const float* q, const float* k_pool, const float* v_pool,
                                    const int* table_all, const int* tab_off, const int* cur_len,
                                    float* out, int H, int D, int BLOCK, float scale) {
    int s = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int L = cur_len[s];
    const int* table = table_all + tab_off[s];
    extern __shared__ float smem[];
    float* q_s = smem;        // [D]
    float* red = smem + D;    // [D] 点积树形 reduce 缓冲

    const float* qh = q   + ((size_t)s*H + h)*D;
    float* outh     = out + ((size_t)s*H + h)*D;

    q_s[d] = qh[d];
    __syncthreads();

    // running 状态: m/l 为标量(每线程各存一份, 恒等); acc 按维度 d 分到各线程寄存器。
    float m = -FLT_MAX, l = 0.f, acc = 0.f;
    for (int pos = 0; pos < L; ++pos) {
        // 1) score = (q · K_pos) * scale —— 对 D 维树形 reduce
        const float* krow = k_pool + bp_offset(table, pos, h, H, BLOCK, D);
        red[d] = q_s[d] * krow[d];
        __syncthreads();
        for (int stride = D >> 1; stride > 0; stride >>= 1) {
            if (d < stride) red[d] += red[d + stride];
            __syncthreads();
        }
        float sc = red[0] * scale;
        __syncthreads();                      // 读完 red[0] 再进下一轮(下次要改写 red)
        // 2) 在线更新 running max / sum / 累加器
        const float* vrow = v_pool + bp_offset(table, pos, h, H, BLOCK, D);
        float m_new = fmaxf(m, sc);
        float corr  = __expf(m - m_new);      // 旧累加量缩放(首个元素 m=-FLT_MAX -> corr=0)
        float p     = __expf(sc - m_new);     // 当前 token 权重
        l   = l   * corr + p;
        acc = acc * corr + p * vrow[d];
        m   = m_new;
    }
    outh[d] = (l > 0.f) ? acc / l : 0.f;
}

// 与 launch_batched_paged 同签名(cap 不再需要, 仅为 drop-in 替换保留), 故 ModelRunner 换一行即可。
void launch_batched_paged_flash(const float* q, const float* k_pool, const float* v_pool,
                                const int* table_all, const int* tab_off, const int* cur_len,
                                float* out, int N, int H, int D, int BLOCK, int cap, float scale) {
    (void)cap;
    dim3 grid(N, H);
    size_t shmem = (size_t)2 * D * sizeof(float);   // q_s[D] + red[D], 与序列长度无关
    batched_paged_flash<<<grid, D, shmem>>>(q, k_pool, v_pool, table_all, tab_off , cur_len,
                                            out, H, D, BLOCK, scale);
}
