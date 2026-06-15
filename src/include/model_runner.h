#pragma once
// model_runner.h —— GPU 执行器: 对一个 running batch 真跑【一步 paged decode】。
//
// 取代旧 scheduler.h 里那个也叫 Engine、却把「KV 物理池 + 执行 + 计时」混在一起的结构体。
// 现在职责:
//   - KV 物理池      -> KVCacheManager 拥有 (传引用进来用)
//   - 每步暂存(q/out + gather 三件套 table/off/len) + cudaEvent 计时 -> ModelRunner 自持
//   - 块分配 / block_table 维护 -> KVCacheManager
//   - 何时跑、跑哪些序列 -> Engine/Scheduler
//
// 当前 MHA: 要求 n_heads == kvm.n_kv_heads (池的 head 维)。GQA 时 q 用 n_q_heads、
//   池用 n_kv_heads, 需 attention kernel 做 q->kv 头映射, 那是后续 kernel 改造的事。
#include "kv_cache_manager.h"
#include "sequence.h"
#include "kv_cache.h"          // launch_batched_paged 声明 (定义在 batched.cu)
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef MR_CK
#define MR_CK(x) do{ cudaError_t e=(x); if(e){ \
    printf("CUDA err %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)
#endif

struct ModelRunner {
    int n_heads, head_dim, block_size, max_batch, max_pool_blocks;
    float scale;
    float *dq = nullptr, *dout = nullptr;          // [max_batch, n_heads, head_dim]
    int   *dtab = nullptr, *dtoff = nullptr, *dlen = nullptr;
    cudaEvent_t ev0{}, ev1{};
    std::vector<int> h_tab, h_off, h_len;          // 每步重建的 gather 暂存

    ModelRunner(int n_heads_, int head_dim_, int block_size_, int max_batch_, int max_pool_blocks_)
      : n_heads(n_heads_), head_dim(head_dim_), block_size(block_size_),
        max_batch(max_batch_), max_pool_blocks(max_pool_blocks_),
        scale(1.f / sqrtf((float)head_dim_)) {
        size_t qn = (size_t)max_batch * n_heads * head_dim;
        MR_CK(cudaMalloc(&dq,   qn * sizeof(float)));  MR_CK(cudaMemset(dq, 0, qn * sizeof(float)));
        MR_CK(cudaMalloc(&dout, qn * sizeof(float)));
        MR_CK(cudaMalloc(&dtab,  (size_t)max_pool_blocks * sizeof(int)));
        MR_CK(cudaMalloc(&dtoff, (size_t)(max_batch + 1) * sizeof(int)));
        MR_CK(cudaMalloc(&dlen,  (size_t)max_batch * sizeof(int)));
        MR_CK(cudaEventCreate(&ev0)); MR_CK(cudaEventCreate(&ev1));
        h_tab.reserve(max_pool_blocks); h_off.reserve(max_batch + 1); h_len.reserve(max_batch);
    }
    ~ModelRunner() {
        cudaFree(dq); cudaFree(dout); cudaFree(dtab); cudaFree(dtoff); cudaFree(dlen);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // 对 batch 真跑一步 paged decode (读 kvm 的池), 返回实测 µs。
    // 把每条序列的 block_table 拼进 table_all、记录偏移与长度, 再调权威 kernel。
    double decode_step(const std::vector<Sequence*>& batch, KVCacheManager& kvm) {
        int n = (int)batch.size();
        if (n == 0) return 0.0;
        h_tab.clear(); h_off.clear(); h_len.clear();
        int cap = 1;
        for (Sequence* s : batch) {
            h_off.push_back((int)h_tab.size());            // 第 s 条 block_table 起点
            for (int b : s->block_table) h_tab.push_back(b);
            int L = s->cur_len > 0 ? s->cur_len : 1;       // 至少 1, 防 0 长度
            h_len.push_back(L); cap = std::max(cap, L);
        }
        h_off.push_back((int)h_tab.size());
        MR_CK(cudaMemcpy(dtab,  h_tab.data(), h_tab.size() * sizeof(int), cudaMemcpyHostToDevice));
        MR_CK(cudaMemcpy(dtoff, h_off.data(), h_off.size() * sizeof(int), cudaMemcpyHostToDevice));
        MR_CK(cudaMemcpy(dlen,  h_len.data(), (size_t)n   * sizeof(int), cudaMemcpyHostToDevice));
        MR_CK(cudaEventRecord(ev0));
        // 权威 kernel: 两遍 softmax 的批量分页 decode。
        //   实测它比 flash 版(launch_batched_paged_flash)快 ~3x: decode 下两遍版 score 阶段按
        //   位置并行、零 reduction; flash 在线 softmax 强制每位置一次 D 维树形 reduce(大量 sync)。
        //   flash/在线 softmax 的省 shared+单遍优势属于 prefill/长上下文/低批量, 不是这里的场景。
        launch_batched_paged(dq, kvm.k_pool, kvm.v_pool, dtab, dtoff, dlen, dout,
                             n, n_heads, head_dim, block_size, cap, scale);
        MR_CK(cudaEventRecord(ev1)); MR_CK(cudaEventSynchronize(ev1));
        float ms; MR_CK(cudaEventElapsedTime(&ms, ev0, ev1));
        return ms * 1e3;   // -> µs
    }
};
