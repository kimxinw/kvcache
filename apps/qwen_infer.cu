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
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>

// ---- Qwen2.5-0.5B-Instruct 结构常量 ----
static const int H = 896, Hq = 14, Hkv = 2, D = 64, IM = 4864, NL = 24;
static const int VOCAB = 151936, BLOCK = 16;
static const float EPS = 1e-6f, THETA = 1000000.0f;

static cublasHandle_t g_cublas;

// y[M,N] = x[M,K] @ W[N,K]^T   (W 是 PyTorch [out=N, in=K] row-major)
static void linear(float* y, const float* x, const float* W, int M, int N, int K) {
    const float one = 1.f, zero = 0.f;
    // 把 row-major 数据当 col-major：见推导，等价 sgemm(T,N, N,M,K)。
    cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &one, W, K, x, K, &zero, y, N);
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
    auto dmalloc = [](size_t n){ float* p; cudaMalloc(&p, n*sizeof(float)); return p; };
    float *x = dmalloc((size_t)Lmax*H), *nrm = dmalloc((size_t)Lmax*H);
    float *Q = dmalloc((size_t)Lmax*Hq*D), *K = dmalloc((size_t)Lmax*Hkv*D), *V = dmalloc((size_t)Lmax*Hkv*D);
    float *att = dmalloc((size_t)Lmax*Hq*D), *oo = dmalloc((size_t)Lmax*H);
    float *gate = dmalloc((size_t)Lmax*IM), *up = dmalloc((size_t)Lmax*IM), *down = dmalloc((size_t)Lmax*H);
    float *logits = dmalloc(VOCAB);
    int *d_ids; cudaMalloc(&d_ids, (size_t)Lmax*sizeof(int));
    int *d_idx; cudaMalloc(&d_idx, sizeof(int));
    // 每层 KV 池基址 = l * (blocks_pl*Hkv*BLOCK*D)
    size_t layer_stride = (size_t)blocks_pl * Hkv * BLOCK * D;
    float *k_pool = dmalloc(NL * layer_stride), *v_pool = dmalloc(NL * layer_stride);
    // 恒等 block_table
    std::vector<int> bt(blocks_pl); for (int i = 0; i < blocks_pl; ++i) bt[i] = i;
    int *d_bt; cudaMalloc(&d_bt, blocks_pl*sizeof(int));//指针d_bt分配在主机CPU上，通过cudaMalloc分配GPU显存，指向GPU的地址
    cudaMemcpy(d_bt, bt.data(), blocks_pl*sizeof(int), cudaMemcpyHostToDevice);

    // 一段前向：x[M,H] 在位置 [pos_base, pos_base+M)，写 KV、跑 attention，返回最后一行的 argmax。
    auto forward = [&](float* xbuf, int M, int pos_base) -> int {
        //TODO CUDA graph
        for (int l = 0; l < NL; ++l) {
            float* kp = k_pool + (size_t)l * layer_stride;
            float* vp = v_pool + (size_t)l * layer_stride;
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
        float* xlast = xbuf + (size_t)(M-1)*H;
        qk_rmsnorm(nrm, xlast, w.ptr("model.norm.weight"), 1, H, EPS);
        linear(logits, nrm, w.ptr("lm_head.weight"), 1, VOCAB, H);
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

    // ---- decode：逐 token，单行前向，pos = P + (k-1) ---- 
    //CUDA graph begin
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int* d_posbase,*d_tokenid;
    cudaMalloc(&d_posbase,sizeof(int));
    cudaMalloc(&d_tokenid,sizeof(int));
    float* x1 = x;   // 复用首行
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cublasSetStream(g_cublas, stream);
    cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
    qk_embed_gather(x1, w.ptr("model.embed_tokens.weight"), d_tokenid, 1, H,stream);
    for (int l = 0; l < NL; ++l) {
        float* kp = k_pool + (size_t)l * layer_stride;
        float* vp = v_pool + (size_t)l * layer_stride;
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
    float* xlast = x1 + (size_t)(1-1)*H;
    qk_rmsnorm(nrm, xlast, w.ptr("model.norm.weight"), 1, H, EPS,stream);
    linear(logits, nrm, w.ptr("lm_head.weight"), 1, VOCAB, H);
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
    for (int k = 1; k < max_new; ++k) {
        int pos = P + k - 1;
        cudaMemcpy(d_tokenid, &gen.back(), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_posbase,&pos,sizeof(int),cudaMemcpyHostToDevice);
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        int next;
        cudaMemcpy(&next,d_idx,sizeof(int),cudaMemcpyDeviceToHost);
        gen.push_back(next);
    }
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_posbase);
    cudaFree(d_tokenid);

    // 输出生成的 token id
    for (size_t i = 0; i < gen.size(); ++i) printf("%d%c", gen[i], i+1<gen.size()?' ':'\n');
    cudaDeviceSynchronize();
    return 0;
}
