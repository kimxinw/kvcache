#pragma once
// kv_cache_manager.h —— 分页 KV 内存的【唯一真相源】。
//
// 把原先散落三处的逻辑收拢到一个对象:
//   (1) 物理块 free-list      —— 复用 BlockAllocator (block_alloc.h)
//   (2) 设备端 KV 物理池       —— 这里 cudaMalloc/拥有, 布局与 batched_paged kernel 完全一致:
//                                 [num_blocks, n_kv_heads, block_size, head_dim]
//   (3) 每条序列的 block_table —— 由 allocate_prompt / append_token 维护到 Sequence 上
//
// 设计上只暴露【块粒度原语】, 不内置 "worst-case 预留 / 抢占" 这类【调度策略】——
//   那些是 Scheduler 的事, 它可以基于 free_blocks() + blocks_needed() 自行实现预留。
//   这样 cache 管理与调度策略彻底解耦, 换调度算法不必改内存层。
//
// GQA 友好: 池的 head 维是 n_kv_heads(KV 头数), 当前 MHA 下 = n_q_heads; 接真模型时
//   只需令 n_kv_heads < n_q_heads, 池与本类无需改动, attention kernel 做 q->kv 头映射即可。
#include "block_alloc.h"
#include "sequence.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef KVM_CK
#define KVM_CK(x) do{ cudaError_t e=(x); if(e){ \
    printf("CUDA err %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)
#endif

struct KVCacheManager {
    int num_blocks, block_size, n_kv_heads, head_dim;
    BlockAllocator alloc;
    float* k_pool = nullptr;       // [num_blocks, n_kv_heads, block_size, head_dim]
    float* v_pool = nullptr;

    KVCacheManager(int num_blocks_, int block_size_, int n_kv_heads_, int head_dim_)
      : num_blocks(num_blocks_), block_size(block_size_),
        n_kv_heads(n_kv_heads_), head_dim(head_dim_), alloc(num_blocks_) {
        size_t n = pool_elems();
        KVM_CK(cudaMalloc(&k_pool, n * sizeof(float)));
        KVM_CK(cudaMalloc(&v_pool, n * sizeof(float)));
        KVM_CK(cudaMemset(k_pool, 0, n * sizeof(float)));
        KVM_CK(cudaMemset(v_pool, 0, n * sizeof(float)));
    }
    ~KVCacheManager() { cudaFree(k_pool); cudaFree(v_pool); }
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    size_t pool_elems() const {
        return (size_t)num_blocks * n_kv_heads * block_size * head_dim;
    }
    int free_blocks() const { return alloc.free_count(); }
    int used_blocks() const { return alloc.used_count(); }

    // 复位块账目 (free list 全满)。设备池内容无需动 (反正会被覆写)。
    void reset() { alloc.reset(); }

    // 该序列最后一块还剩多少 token 槽位 (= 容量 - 已用)。
    int slack(const Sequence& s) const {
        return (int)s.block_table.size() * block_size - s.cur_len;
    }

    // 为 prompt 分配物理块并写入 block_table。块不足返回 false (不改动任何状态)。
    bool allocate_prompt(Sequence& s) {
        int need = blocks_needed(s.prompt_len, block_size);
        if (alloc.free_count() < need) return false;
        for (int i = 0; i < need; ++i) s.block_table.push_back(alloc.alloc());
        s.cur_len = s.prompt_len;
        return true;
    }

    // decode 一步: KV 长度 +1, 跨块边界时借一个新物理块。块不足返回 false (cur_len 不前进)。
    bool append_token(Sequence& s) {
        int need = blocks_needed(s.cur_len + 1, block_size);
        if (need > (int)s.block_table.size()) {
            int b = alloc.alloc();
            if (b < 0) return false;
            s.block_table.push_back(b);
        }
        s.cur_len++;
        return true;
    }

    // 归还该序列所有物理块, 清空其 block_table。
    void free(Sequence& s) {
        alloc.free_blocks(s.block_table);
        s.block_table.clear();
    }

    // 物理块 phys 的某个 kv_head 在池中的【元素】起始偏移 (与 batched_paged 的 bp_offset 同口径)。
    // 用于把新 token 的 K/V 写进池 (append kernel) 或调试核对。
    size_t slot_offset(int phys, int kv_head, int in_block) const {
        return (((size_t)phys * n_kv_heads + kv_head) * block_size + in_block) * head_dim;
    }
};
