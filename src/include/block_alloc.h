#pragma once
// 极简物理块分配器 (主机端)。模拟 vLLM 的 KV block pool:
//   池里有 total_blocks 个等大物理块, 用 free list 管理。
//   序列按需 alloc() 物理块, 结束时 free() 整条归还 -> 无外部碎片 (block 粒度对齐)。
// 这是 paged attention "显存几乎零浪费" 的来源: 不必为每条序列预留 max_seq_len。
#include <vector>
#include <cstdio>
#include <cstdlib>

struct BlockAllocator {
    int total_blocks;
    std::vector<int> free_list;          // 可用物理块号 (栈式复用)

    explicit BlockAllocator(int n) : total_blocks(n) {
        free_list.reserve(n);
        for (int i = n - 1; i >= 0; --i) free_list.push_back(i);  // 0..n-1 入栈
    }

    int free_count() const { return (int)free_list.size(); }
    int used_count() const { return total_blocks - (int)free_list.size(); }

    // 取一个物理块; 池空则返回 -1 (调用方判 OOM)
    int alloc() {
        if (free_list.empty()) return -1;
        int b = free_list.back(); free_list.pop_back();
        return b;
    }
    // 归还一条序列占用的所有物理块
    void free_blocks(const std::vector<int>& blocks) {
        for (int b : blocks) free_list.push_back(b);
    }
};

// 给定一条序列长度, 计算需要多少个物理块
static inline int blocks_needed(int seq_len, int block) {
    return (seq_len + block - 1) / block;
}
