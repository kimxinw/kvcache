// qwen_infer.cu —— 里程碑1：单序列 Qwen2.5-0.5B-Instruct 真前向 + 贪心 decode。
//   线性层走 cuBLAS；embed/RMSNorm/RoPE/SwiGLU/写KV/GQA因果分页attention 走手写 kernel。
//   IO 约定：stdin 读 prompt token id（空白分隔），stdout 打印生成的 token id。
//   tokenizer 在 Python 侧（tools/run_qwen.py）。
//
//   KV 池本里程碑自管：[NL, blocks, Hkv, BLOCK, D]，单序列用恒等 block_table（逻辑块i->物理块i），
//   仍走分页 kernel 的查表路径。里程碑2 换 KVCacheManager + 多序列真块表。
//
// 编译见 CMakeLists：链接 cublas。
#include "qwen_loader.h"
#include "qwen_kernels.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>

// ---- Qwen2.5-0.5B-Instruct 结构常量 ----
static const int H = 896, Hq = 14, Hkv = 2, D = 64, IM = 4864, NL = 24;
static const int VOCAB = 151936, BLOCK = 16;
static const float EPS = 1e-6f, THETA = 1000000.0f;

static cublasHandle_t g_cublas;

// decode 稳态计时汇报：丢弃 warmup 步后，对 per-token 墙钟取中位数。
// 输出走 stderr（stdout 只留 token id 行，diff_cudagraph.py 仍能解析）。
static void report_decode_timing(std::vector<double>& us, const char* tag) {
    if (us.empty()) { fprintf(stderr, "[bench] %s: no timed steps\n", tag); return; }
    std::sort(us.begin(), us.end());
    double med = us[us.size() / 2];
    double sum = 0; for (double v : us) sum += v;
    double mean = sum / us.size();
    fprintf(stderr, "[bench] %s steps=%zu median=%.1f us/tok mean=%.1f us/tok %.1f tok/s\n",
            tag, us.size(), med, mean, 1e6 / med);
}

// y[M,N] = x[M,K] @ W[N,K]^T   (W 是 PyTorch [out=N, in=K] row-major)
//   fp16 输入/输出，fp32 累加 (compute=CUBLAS_COMPUTE_32F)。alpha/beta 按 compute type 用 float。
//   转置约定不变：把 row-major 当 col-major，等价 gemm(T,N, N,M,K)。
static void linear(half* y, const half* x, const half* W, int M, int N, int K) {
    const float one = 1.f, zero = 0.f;
    cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                 &one, W, CUDA_R_16F, K, x, CUDA_R_16F, K,
                 &zero, y, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
// lm_head 专用：fp16 输入，fp32 输出 (logits 直接给 argmax，避免 fp16 在 151936 类上丢精度)。
static void linear_logits(float* y, const half* x, const half* W, int M, int N, int K) {
    const float one = 1.f, zero = 0.f;
    cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                 &one, W, CUDA_R_16F, K, x, CUDA_R_16F, K,
                 &zero, y, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "data/qwen05b";
    int max_new = (argc > 2) ? atoi(argv[2]) : 32;

    QwenWeights w; w.load(dir);
    cublasCreate(&g_cublas);

    // 读 prompt ids
    std::vector<int> prompt;
    { int t; while (scanf("%d", &t) == 1) prompt.push_back(t); }
    int P = (int)prompt.size();
    if (P == 0) { fprintf(stderr, "no prompt ids on stdin\n"); return 1; }
    fprintf(stderr, "[infer] prompt_len=%d max_new=%d\n", P, max_new);

    int Lmax = P;                                  // prefill 是最大 M
    int total_max = P + max_new + 4;
    int blocks_pl = (total_max + BLOCK - 1) / BLOCK + 1;   // 每层块数
    float scale = 1.f / sqrtf((float)D);

    // ---- 设备缓冲 ----
    auto dmalloc = [](size_t n){ half* p; cudaMalloc(&p, n*sizeof(half)); return p; };
    half *x = dmalloc((size_t)Lmax*H), *nrm = dmalloc((size_t)Lmax*H);
    half *Q = dmalloc((size_t)Lmax*Hq*D), *K = dmalloc((size_t)Lmax*Hkv*D), *V = dmalloc((size_t)Lmax*Hkv*D);
    half *att = dmalloc((size_t)Lmax*Hq*D), *oo = dmalloc((size_t)Lmax*H);
    half *gate = dmalloc((size_t)Lmax*IM), *up = dmalloc((size_t)Lmax*IM), *down = dmalloc((size_t)Lmax*H);
    float *logits; cudaMalloc(&logits, VOCAB*sizeof(float));   // lm_head 输出 fp32
    int *d_ids; cudaMalloc(&d_ids, (size_t)Lmax*sizeof(int));
    int *d_idx; cudaMalloc(&d_idx, sizeof(int));
    // 每层 KV 池基址 = l * (blocks_pl*Hkv*BLOCK*D)
    size_t layer_stride = (size_t)blocks_pl * Hkv * BLOCK * D;
    half *k_pool = dmalloc(NL * layer_stride), *v_pool = dmalloc(NL * layer_stride);
    // 恒等 block_table
    std::vector<int> bt(blocks_pl); for (int i = 0; i < blocks_pl; ++i) bt[i] = i;
    int *d_bt; cudaMalloc(&d_bt, blocks_pl*sizeof(int));//指针d_bt分配在主机CPU上，通过cudaMalloc分配GPU显存，指向GPU的地址
    cudaMemcpy(d_bt, bt.data(), blocks_pl*sizeof(int), cudaMemcpyHostToDevice);

    // 一段前向：x[M,H] 在位置 [pos_base, pos_base+M)，写 KV、跑 attention，返回最后一行的 argmax。
    auto forward = [&](half* xbuf, int M, int pos_base) -> int {
        //TODO CUDA graph
        for (int l = 0; l < NL; ++l) {
            half* kp = k_pool + (size_t)l * layer_stride;
            half* vp = v_pool + (size_t)l * layer_stride;
            qk_rmsnorm(nrm, xbuf, w.lp(l, "input_layernorm.weight"), M, H, EPS);
            linear(Q, nrm, w.lp(l, "self_attn.q_proj.weight"), M, Hq*D, H);
            qk_add_bias(Q, w.lp(l, "self_attn.q_proj.bias"), M, Hq*D);
            linear(K, nrm, w.lp(l, "self_attn.k_proj.weight"), M, Hkv*D, H);
            qk_add_bias(K, w.lp(l, "self_attn.k_proj.bias"), M, Hkv*D);
            linear(V, nrm, w.lp(l, "self_attn.v_proj.weight"), M, Hkv*D, H);
            qk_add_bias(V, w.lp(l, "self_attn.v_proj.bias"), M, Hkv*D);
            qk_rope(Q, M, Hq, D, pos_base, THETA);
            qk_rope(K, M, Hkv, D, pos_base, THETA);
            qk_write_kv(K, V, kp, vp, d_bt, pos_base, M, Hkv, D, BLOCK);
            qk_paged_causal_attn(Q, att, kp, vp, d_bt, pos_base, M, Hq, Hkv, D, BLOCK, scale);
            linear(oo, att, w.lp(l, "self_attn.o_proj.weight"), M, H, Hq*D);  // 无 bias
            qk_add_residual(xbuf, oo, M*H);
            qk_rmsnorm(nrm, xbuf, w.lp(l, "post_attention_layernorm.weight"), M, H, EPS);
            linear(gate, nrm, w.lp(l, "mlp.gate_proj.weight"), M, IM, H);
            linear(up,   nrm, w.lp(l, "mlp.up_proj.weight"),   M, IM, H);
            qk_silu_mul(gate, gate, up, M*IM);
            linear(down, gate, w.lp(l, "mlp.down_proj.weight"), M, H, IM);
            qk_add_residual(xbuf, down, M*H);
        }
        // 末位 token -> final norm -> logits -> argmax
        half* xlast = xbuf + (size_t)(M-1)*H;
        qk_rmsnorm(nrm, xlast, w.ptr("model.norm.weight"), 1, H, EPS);
        linear_logits(logits, nrm, w.ptr("lm_head.weight"), 1, VOCAB, H);
        qk_argmax(logits, VOCAB, d_idx);
        int next; cudaMemcpy(&next, d_idx, sizeof(int), cudaMemcpyDeviceToHost);
        return next;
    };
    // ---- prefill ----
    cudaMemcpy(d_ids, prompt.data(), P*sizeof(int), cudaMemcpyHostToDevice);
    qk_embed_gather(x, w.ptr("model.embed_tokens.weight"), d_ids, P, H);
    std::vector<int> gen;
    int next = forward(x, P, 0);
    gen.push_back(next);
#ifdef CUDAG
    // ---- decode：逐 token，单行前向，pos = P + (k-1) ---- 
    //CUDA graph begin
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int* d_posbase,*d_tokenid;
    cudaMalloc(&d_posbase,sizeof(int));
    cudaMalloc(&d_tokenid,sizeof(int));
    half* x1 = x;   // 复用首行
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cublasSetStream(g_cublas, stream);
    cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
    qk_embed_gather(x1, w.ptr("model.embed_tokens.weight"), d_tokenid, 1, H,stream);
    for (int l = 0; l < NL; ++l) {
        half* kp = k_pool + (size_t)l * layer_stride;
        half* vp = v_pool + (size_t)l * layer_stride;
        qk_rmsnorm(nrm, x1, w.lp(l, "input_layernorm.weight"), 1, H, EPS,stream);
        linear(Q, nrm, w.lp(l, "self_attn.q_proj.weight"), 1, Hq*D, H);
        qk_add_bias(Q, w.lp(l, "self_attn.q_proj.bias"), 1, Hq*D,stream);
        linear(K, nrm, w.lp(l, "self_attn.k_proj.weight"), 1, Hkv*D, H);
        qk_add_bias(K, w.lp(l, "self_attn.k_proj.bias"), 1, Hkv*D,stream);
        linear(V, nrm, w.lp(l, "self_attn.v_proj.weight"), 1, Hkv*D, H);
        qk_add_bias(V, w.lp(l, "self_attn.v_proj.bias"), 1, Hkv*D,stream);
        qkcuda_rope(Q, 1, Hq, D, d_posbase, THETA,stream);
        qkcuda_rope(K, 1, Hkv, D, d_posbase, THETA,stream);
        qkcuda_write_kv(K, V, kp, vp, d_bt, d_posbase, 1, Hkv, D, BLOCK,stream);
        qkcuda_paged_causal_attn(Q, att, kp, vp, d_bt, d_posbase, 1, Hq, Hkv, D, BLOCK, scale,stream);
        linear(oo, att, w.lp(l, "self_attn.o_proj.weight"), 1, H, Hq*D);  // 无 bias
        qk_add_residual(x1, oo, 1*H,stream);
        qk_rmsnorm(nrm, x1, w.lp(l, "post_attention_layernorm.weight"), 1, H, EPS,stream);
        linear(gate, nrm, w.lp(l, "mlp.gate_proj.weight"), 1, IM, H);
        linear(up,   nrm, w.lp(l, "mlp.up_proj.weight"),   1, IM, H);
        qk_silu_mul(gate, gate, up, 1*IM,stream);
        linear(down, gate, w.lp(l, "mlp.down_proj.weight"), 1, H, IM);
        qk_add_residual(x1, down, 1*H,stream);
    }
    // 末位 token -> final norm -> logits -> argmax
    half* xlast = x1 + (size_t)(1-1)*H;
    qk_rmsnorm(nrm, xlast, w.ptr("model.norm.weight"), 1, H, EPS,stream);
    linear_logits(logits, nrm, w.ptr("lm_head.weight"), 1, VOCAB, H);
    qk_argmax(logits, VOCAB, d_idx,stream);
    cudaError_t err;
    err = cudaStreamEndCapture(stream, &graph);
    if(err!=cudaSuccess){
        fprintf(stderr,"Capture err: %s\n",cudaGetErrorString(err));
        exit(0);
    }
    if(graph!=nullptr){
        err = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        if(err!=cudaSuccess){
            fprintf(stderr,"Instantiate err: %s\n",cudaGetErrorString(err));
            exit(0);
        }
    }else{
        fprintf(stderr,"graph is nullptr after capture!\n");
    }
    
    //CUDA graph end
    const int WARMUP = std::min(16, max_new / 4);   // 丢掉 graph 首次 replay / 升频 / workspace 首分配
    std::vector<double> step_us;
    for (int k = 1; k < max_new; ++k) {
        int pos = P + k - 1;
        auto t0 = std::chrono::steady_clock::now();
        cudaMemcpy(d_tokenid, &gen.back(), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_posbase,&pos,sizeof(int),cudaMemcpyHostToDevice);
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        int next;
        cudaMemcpy(&next,d_idx,sizeof(int),cudaMemcpyDeviceToHost);
        auto t1 = std::chrono::steady_clock::now();
        gen.push_back(next);
        if (k > WARMUP) step_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report_decode_timing(step_us, "graph");
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_posbase);
    cudaFree(d_tokenid);
#else 
    // ---- decode：逐 token，单行前向，pos = P + (k-1) ----
    half* x1 = x;   // 复用首行
    const int WARMUP = std::min(16, max_new / 4);   // 与 graph 路径对齐
    std::vector<double> step_us;
    for (int k = 1; k < max_new; ++k) {
        int pos = P + k - 1;
        auto t0 = std::chrono::steady_clock::now();
        cudaMemcpy(d_ids, &gen.back(), sizeof(int), cudaMemcpyHostToDevice);
        qk_embed_gather(x1, w.ptr("model.embed_tokens.weight"), d_ids, 1, H);
        next = forward(x1, 1, pos);   // forward 末尾 D2H 隐式 sync，墙钟即真实 per-token 延迟
        auto t1 = std::chrono::steady_clock::now();
        gen.push_back(next);
        if (k > WARMUP) step_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report_decode_timing(step_us, "eager");

#endif
    // 输出生成的 token id
    for (size_t i = 0; i < gen.size(); ++i) printf("%d%c", gen[i], i+1<gen.size()?' ':'\n');
    cudaDeviceSynchronize();
    return 0;
}
