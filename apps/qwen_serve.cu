// qwen_serve.cu —— 里程碑2：把 Qwen2.5-0.5B 真前向接进 continuous batching。
//   多条 prompt 并发：共享分页 KV 池 [NL, num_blocks, Hkv, BLOCK, D] + 每序列块表；
//   admit 时单序列 prefill，之后所有 running 序列每轮跑一步【批量 GQA 分页 decode】；
//   各自贪心 argmax、各自遇 eos / max_new 退场并归还块，腾出的块供 waiting 序列 admit。
//
//   线性层 cuBLAS；embed/RMSNorm/RoPE/SwiGLU/写KV/attention 手写 kernel（含批量版）。
//   IO：stdin 读 "NSEQ\n" 然后每行 "max_new id0 id1 ..."；stdout 每行打印一条序列生成的 id。
#include "qwen_loader.h"
#include "qwen_kernels.h"
#include "block_alloc.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <iostream>

static const int H = 896, Hq = 14, Hkv = 2, D = 64, IM = 4864, NL = 24;
static const int VOCAB = 151936, BLOCK = 16, MAXB = 8, NUM_BLOCKS = 512;
static const float EPS = 1e-6f, THETA = 1000000.0f;
static const int EOS_A = 151645, EOS_B = 151643;

static cublasHandle_t g_cublas;
// fp16 输入/输出，fp32 累加。转置约定不变(row-major 当 col-major)。
static void linear(half* y, const half* x, const half* W, int M, int N, int K) {
    const float one = 1.f, zero = 0.f;
    cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                 &one, W, CUDA_R_16F, K, x, CUDA_R_16F, K,
                 &zero, y, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
// lm_head 专用：fp16 输入，fp32 输出 logits 给 argmax。
static void linear_logits(float* y, const half* x, const half* W, int M, int N, int K) {
    const float one = 1.f, zero = 0.f;
    cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                 &one, W, CUDA_R_16F, K, x, CUDA_R_16F, K,
                 &zero, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

struct Seq {
    int id, prompt_len, max_new, cur_len = 0;
    std::vector<int> tokens;        // prompt + generated
    std::vector<int> block_table;   // 逻辑块 -> 物理块
    bool finished = false;
    int gen() const { return (int)tokens.size() - prompt_len; }
};

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "data/qwen05b";
    QwenWeights w; w.load(dir);
    cublasCreate(&g_cublas);
    float scale = 1.f / sqrtf((float)D);

    // ---- 读入多条请求 ----
    std::vector<Seq> store;
    { int nseq; std::cin >> nseq;
      for (int i = 0; i < nseq; ++i) {
        Seq s; s.id = i; std::cin >> s.max_new;
        std::string rest; std::getline(std::cin, rest);
        std::istringstream ss(rest); int t;
        while (ss >> t) s.tokens.push_back(t);
        s.prompt_len = (int)s.tokens.size();
        store.push_back(s);
      } }
    int NSEQ = (int)store.size();
    fprintf(stderr, "[serve] %d requests, max_batch=%d, pool=%d blocks\n", NSEQ, MAXB, NUM_BLOCKS);

    // ---- 设备缓冲（按最大行数 = max(prompt_len, MAXB)）----
    int maxP = MAXB; for (auto& s : store) maxP = std::max(maxP, s.prompt_len);
    int MR = maxP;
    auto dm = [](size_t n){ half* p; cudaMalloc(&p, n*sizeof(half)); return p; };
    half *x=dm((size_t)MR*H), *nrm=dm((size_t)MR*H);
    half *Q=dm((size_t)MR*Hq*D), *K=dm((size_t)MR*Hkv*D), *V=dm((size_t)MR*Hkv*D);
    half *att=dm((size_t)MR*Hq*D), *oo=dm((size_t)MR*H);
    half *gate=dm((size_t)MR*IM), *up=dm((size_t)MR*IM), *down=dm((size_t)MR*H);
    float *logits; cudaMalloc(&logits, (size_t)MAXB*VOCAB*sizeof(float));   // lm_head 输出 fp32
    int *d_ids; cudaMalloc(&d_ids, (size_t)MR*sizeof(int));
    int *d_idx; cudaMalloc(&d_idx, MAXB*sizeof(int));
    int *d_tab; cudaMalloc(&d_tab, NUM_BLOCKS*sizeof(int));
    int *d_off; cudaMalloc(&d_off, (MAXB+1)*sizeof(int));
    int *d_pos; cudaMalloc(&d_pos, MAXB*sizeof(int));   // 写位置/attn 长度复用
    int *d_len; cudaMalloc(&d_len, MAXB*sizeof(int));

    size_t layer_stride = (size_t)NUM_BLOCKS * Hkv * BLOCK * D;
    half *k_pool=dm(NL*layer_stride), *v_pool=dm(NL*layer_stride);
    BlockAllocator alloc(NUM_BLOCKS);

    // ---- 单序列 prefill：写 KV 入池，返回首个生成 token ----
    auto prefill = [&](Seq& s) -> int {
        int P = s.prompt_len;
        int nb = blocks_needed(P, BLOCK);
        for (int i = 0; i < nb; ++i) s.block_table.push_back(alloc.alloc());
        cudaMemcpy(d_tab, s.block_table.data(), nb*sizeof(int), cudaMemcpyHostToDevice);
        int off0[2] = {0, nb}; cudaMemcpy(d_off, off0, 2*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ids, s.tokens.data(), P*sizeof(int), cudaMemcpyHostToDevice);
        qk_embed_gather(x, w.ptr("model.embed_tokens.weight"), d_ids, P, H);
        for (int l = 0; l < NL; ++l) {
            half* kp = k_pool + (size_t)l*layer_stride;
            half* vp = v_pool + (size_t)l*layer_stride;
            qk_rmsnorm(nrm, x, w.lp(l,"input_layernorm.weight"), P, H, EPS);
            linear(Q, nrm, w.lp(l,"self_attn.q_proj.weight"), P, Hq*D, H); qk_add_bias(Q, w.lp(l,"self_attn.q_proj.bias"), P, Hq*D);
            linear(K, nrm, w.lp(l,"self_attn.k_proj.weight"), P, Hkv*D, H); qk_add_bias(K, w.lp(l,"self_attn.k_proj.bias"), P, Hkv*D);
            linear(V, nrm, w.lp(l,"self_attn.v_proj.weight"), P, Hkv*D, H); qk_add_bias(V, w.lp(l,"self_attn.v_proj.bias"), P, Hkv*D);
            qk_rope(Q, P, Hq, D, 0, THETA);
            qk_rope(K, P, Hkv, D, 0, THETA);
            qk_write_kv(K, V, kp, vp, d_tab, 0, P, Hkv, D, BLOCK);
            qk_paged_causal_attn(Q, att, kp, vp, d_tab, 0, P, Hq, Hkv, D, BLOCK, scale);
            linear(oo, att, w.lp(l,"self_attn.o_proj.weight"), P, H, Hq*D);
            qk_add_residual(x, oo, P*H);
            qk_rmsnorm(nrm, x, w.lp(l,"post_attention_layernorm.weight"), P, H, EPS);
            linear(gate, nrm, w.lp(l,"mlp.gate_proj.weight"), P, IM, H);
            linear(up,   nrm, w.lp(l,"mlp.up_proj.weight"),   P, IM, H);
            qk_silu_mul(gate, gate, up, P*IM);
            linear(down, gate, w.lp(l,"mlp.down_proj.weight"), P, H, IM);
            qk_add_residual(x, down, P*H);
        }
        qk_rmsnorm(nrm, x + (size_t)(P-1)*H, w.ptr("model.norm.weight"), 1, H, EPS);
        linear_logits(logits, nrm, w.ptr("lm_head.weight"), 1, VOCAB, H);
        qk_argmax(logits, VOCAB, d_idx);
        int t; cudaMemcpy(&t, d_idx, sizeof(int), cudaMemcpyDeviceToHost);
        s.cur_len = P; s.tokens.push_back(t);
        return t;
    };

    // ---- 批量 decode 一步：对所有 running 序列各推进 1 token ----
    auto decode_step = [&](std::vector<Seq*>& run) {
        int N = (int)run.size();//batch size
        std::vector<int> h_ids(N), h_tab, h_off(N+1), h_pos(N), h_len(N);
        for (int r = 0; r < N; ++r) {
            Seq* s = run[r];
            h_ids[r] = s->tokens.back();        // 待喂入的最新 token
            int pos = s->cur_len;               // 它将占据的位置
            if (blocks_needed(pos+1, BLOCK) > (int)s->block_table.size())
                s->block_table.push_back(alloc.alloc());
            h_off[r] = (int)h_tab.size();
            for (int b : s->block_table) h_tab.push_back(b);
            h_pos[r] = pos;
            s->cur_len = pos + 1;
            h_len[r] = s->cur_len;              // attn 因果长度 = 写入后总长
        }
        h_off[N] = (int)h_tab.size();
        cudaMemcpy(d_ids, h_ids.data(), N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tab, h_tab.data(), h_tab.size()*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_off, h_off.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos, h_pos.data(), N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_len, h_len.data(), N*sizeof(int), cudaMemcpyHostToDevice);

        qk_embed_gather(x, w.ptr("model.embed_tokens.weight"), d_ids, N, H);
        for (int l = 0; l < NL; ++l) {
            half* kp = k_pool + (size_t)l*layer_stride;
            half* vp = v_pool + (size_t)l*layer_stride;
            qk_rmsnorm(nrm, x, w.lp(l,"input_layernorm.weight"), N, H, EPS);
            linear(Q, nrm, w.lp(l,"self_attn.q_proj.weight"), N, Hq*D, H); qk_add_bias(Q, w.lp(l,"self_attn.q_proj.bias"), N, Hq*D);
            linear(K, nrm, w.lp(l,"self_attn.k_proj.weight"), N, Hkv*D, H); qk_add_bias(K, w.lp(l,"self_attn.k_proj.bias"), N, Hkv*D);
            linear(V, nrm, w.lp(l,"self_attn.v_proj.weight"), N, Hkv*D, H); qk_add_bias(V, w.lp(l,"self_attn.v_proj.bias"), N, Hkv*D);
            qk_rope_pos(Q, N, Hq, D, d_pos, THETA);
            qk_rope_pos(K, N, Hkv, D, d_pos, THETA);
            qk_batched_write_kv(K, V, kp, vp, d_tab, d_off, d_pos, N, Hkv, D, BLOCK);
            qk_batched_paged_decode_attn(Q, att, kp, vp, d_tab, d_off, d_len, N, Hq, Hkv, D, BLOCK, scale);
            linear(oo, att, w.lp(l,"self_attn.o_proj.weight"), N, H, Hq*D);
            qk_add_residual(x, oo, N*H);
            qk_rmsnorm(nrm, x, w.lp(l,"post_attention_layernorm.weight"), N, H, EPS);
            linear(gate, nrm, w.lp(l,"mlp.gate_proj.weight"), N, IM, H);
            linear(up,   nrm, w.lp(l,"mlp.up_proj.weight"),   N, IM, H);
            qk_silu_mul(gate, gate, up, N*IM);
            linear(down, gate, w.lp(l,"mlp.down_proj.weight"), N, H, IM);
            qk_add_residual(x, down, N*H);
        }
        qk_rmsnorm(nrm, x, w.ptr("model.norm.weight"), N, H, EPS);    // 全 N 行 final norm
        linear_logits(logits, nrm, w.ptr("lm_head.weight"), N, VOCAB, H);
        for (int r = 0; r < N; ++r) qk_argmax(logits + (size_t)r*VOCAB, VOCAB, d_idx + r);
        std::vector<int> nxt(N); cudaMemcpy(nxt.data(), d_idx, N*sizeof(int), cudaMemcpyDeviceToHost);
        for (int r = 0; r < N; ++r) run[r]->tokens.push_back(nxt[r]);
    };

    // ---- continuous batching 主循环 ----
    int next_admit = 0, total_steps = 0, peak = 0, reserved = 0;
    std::vector<Seq*> running;
    while (true) {
        // 退场
        for (size_t i = 0; i < running.size();) {
            Seq* s = running[i];
            int last = s->tokens.back();
            if (s->gen() >= s->max_new || last == EOS_A || last == EOS_B) {
                s->finished = true; alloc.free_blocks(s->block_table); s->block_table.clear();
                reserved -= blocks_needed(s->prompt_len + s->max_new, BLOCK);
                running[i] = running.back(); running.pop_back();
            } else ++i;
        }
        // 入场（块够就 admit + prefill）
        while ((int)running.size() < MAXB && next_admit < NSEQ) {
            Seq& s = store[next_admit];
            int need = blocks_needed(s.prompt_len + s.max_new, BLOCK);
            if (reserved + need > NUM_BLOCKS) break;   // 累计预留防 OOM（含已 admit 序列的未来增长）
            prefill(s);
            reserved += need;
            running.push_back(&s);
            next_admit++;
        }
        if (running.empty() && next_admit >= NSEQ) break;
        peak = std::max(peak, (int)running.size());
        if (!running.empty()) { decode_step(running); total_steps++; }
    }
    cudaDeviceSynchronize();
    fprintf(stderr, "[serve] done. decode_steps=%d peak_batch=%d\n", total_steps, peak);

    // 输出每条序列生成的 token id（去掉 prompt）
    for (auto& s : store) {
        for (int i = s.prompt_len; i < (int)s.tokens.size(); ++i)
            printf("%d%c", s.tokens[i], i+1<(int)s.tokens.size()?' ':'\n');
    }
    return 0;
}
