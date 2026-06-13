#pragma once
// engine.h —— 顶层引擎: 串起三层 —— 内存层 KVCacheManager · 执行层 ModelRunner · 策略层(本文件)。
//
// Engine 是【唯一的 continuous batching 实现】(iteration-level scheduling):
//   每个 decode step 之前【退场已完成 -> 立刻归还块】、【从 waiting 按到达时刻补进新请求
//   直到 batch 满或池放不下】, 然后对 running 跑一步。demo 与基准的 continuous 行都走它。
//
// 支持两种到达模型 (由请求的 arrival_us 决定, 无需切代码):
//   - 全 0 (离线饱和): 队列从头就满, 隔离纯调度效率的吞吐基准; wall == gpu。
//   - 泊松/任意到达    : 队列空时墙钟 clock 空转推进到下一个到达时刻(不计 GPU 忙时), 体现排队延迟。
//
// worst-case 预留(防生成途中 OOM, 故免抢占)在【本层】做 —— 这是调度策略;
//   KVCacheManager 只给块原语。换调度算法只动这里, 不碰内存层与 kernel。
//
// static (request-level) batching 是【故意次优的对照基线】, 见 scheduler.h 的 run_static ——
//   它复用同一套 KVCacheManager + ModelRunner, 只是策略循环不同, 用来量化 continuous 的收益。
#include "kv_cache_manager.h"
#include "model_runner.h"
#include "sequence.h"
#include <deque>
#include <vector>
#include <algorithm>

struct EngineConfig {
    int n_heads;     // = n_kv_heads (当前 MHA)
    int head_dim;
    int block_size;  // 每个物理块容纳的 token 数
    int num_blocks;  // KV 池物理块总数 (显存预算)
    int max_batch;   // 最大并发 running 序列数
};

// 一次运行的汇总指标 (continuous 与 static 基线共用)。
struct EngineStats {
    const char* name        = "";
    int       iters         = 0;   // 总 decode step 数
    long long tokens        = 0;   // 总生成 token 数
    double    gpu_us        = 0;   // GPU 忙时之和
    double    wall_us       = 0;   // 端到端墙钟 (含等待到达的空转) = makespan
    long long slot_steps    = 0;   // Σ 每步 batch 宽度 (含 static 完成序列的空转 slot)
    long long useful_steps  = 0;   // Σ 每步真正在生成的序列数 (continuous 恒 == slot_steps)
    int       peak_blocks   = 0;   // 峰值物理块占用
    int       finished      = 0;   // 完成的请求数
    double    avg_latency_us= 0;   // 请求平均端到端时间 (到达 -> 完成)
    double    p99_latency_us= 0;
};

// 从一组延迟样本填 avg / p99 (两种策略复用)。
inline void fill_latency_stats(EngineStats& R, std::vector<double>& lat) {
    if (lat.empty()) { R.avg_latency_us = R.p99_latency_us = 0; return; }
    std::sort(lat.begin(), lat.end());
    double s = 0; for (double x : lat) s += x;
    R.avg_latency_us = s / lat.size();
    R.p99_latency_us = lat[std::min((size_t)(lat.size() * 0.99), lat.size() - 1)];
}

struct Engine {
    EngineConfig cfg;
    KVCacheManager kvm;
    ModelRunner    runner;
    std::deque<Sequence>   store;       // 拥有所有序列对象 (deque: push_back 不失效指针)
    std::deque<Sequence*>  waiting;     // 按到达时刻非降序 (要求 add_request 按序调用)
    std::vector<Sequence*> running;
    std::vector<double>    latencies;   // 完成序列的端到端延迟样本
    int    reserved_blocks = 0;
    double clock = 0;                   // 墙钟 (µs)
    int    next_id = 0;
    EngineStats stats;

    explicit Engine(const EngineConfig& c)
      : cfg(c),
        kvm(c.num_blocks, c.block_size, c.n_heads, c.head_dim),
        runner(c.n_heads, c.head_dim, c.block_size, c.max_batch, c.num_blocks) {
        stats.name = "continuous";
    }

    // 清空所有运行期状态 (同一个 Engine 实例再跑一条 trace 前调用)。
    void reset() {
        kvm.reset(); store.clear(); waiting.clear(); running.clear(); latencies.clear();
        reserved_blocks = 0; clock = 0; next_id = 0;
        stats = EngineStats{}; stats.name = "continuous";
    }

    int worstcase_blocks(const Sequence& s) const {
        return blocks_needed(s.prompt_len + s.max_new_tokens, cfg.block_size);
    }

    // 新请求入列。注意: 必须按 arrival_us 非降序调用 (基准的 trace 天然如此; 离线场景全 0)。
    void add_request(int prompt_len, int max_new_tokens, double arrival_us = 0.0) {
        store.push_back(Sequence{});
        Sequence& s = store.back();
        s.id = next_id++;
        s.prompt_len = prompt_len;
        s.max_new_tokens = max_new_tokens;
        s.arrival_us = arrival_us;
        s.status = SeqStatus::Waiting;
        waiting.push_back(&s);
    }

    // 跑一个迭代。返回 false 表示全部完成。
    bool step() {
        // 1) 退场: 完成的序列归还块 + 预留, 记录延迟
        for (size_t i = 0; i < running.size();) {
            Sequence* s = running[i];
            if (s->done()) {
                kvm.free(*s);
                reserved_blocks -= worstcase_blocks(*s);
                latencies.push_back(clock - s->arrival_us);
                s->status = SeqStatus::Finished;
                stats.finished++;
                running[i] = running.back(); running.pop_back();   // O(1) 删除
            } else ++i;
        }
        // 2) 入场: 凡已到达(arrival_us <= clock)的, 在 batch 宽度与池预留双约束下尽量补满
        while ((int)running.size() < cfg.max_batch && !waiting.empty()
               && waiting.front()->arrival_us <= clock) {
            Sequence* s = waiting.front();
            int fp = worstcase_blocks(*s);
            if (reserved_blocks + fp > cfg.num_blocks) break;   // 池放不下, 等退场腾预留
            if (!kvm.allocate_prompt(*s)) break;                // 预留已保证, 防御性兜底
            reserved_blocks += fp;
            s->status = SeqStatus::Running;
            running.push_back(s);
            waiting.pop_front();
        }
        // 3) 队列空: 全完成则结束; 否则墙钟空转到下一个到达时刻
        if (running.empty()) {
            if (waiting.empty()) return false;
            clock = std::max(clock, waiting.front()->arrival_us);
            return true;
        }
        // 4) 真跑一步 paged decode -> 推进墙钟
        double us = runner.decode_step(running, kvm);
        stats.gpu_us += us; clock += us; stats.iters++;
        stats.slot_steps  += (int)running.size();
        stats.useful_steps += (int)running.size();          // continuous: 每个 slot 都在干活
        // 5) 每条生成一个 token (KV+1, 跨块借新块; worst-case 已预留, 不会 OOM)
        for (Sequence* s : running) { kvm.append_token(*s); stats.tokens++; }
        stats.peak_blocks = std::max(stats.peak_blocks, kvm.used_blocks());
        return true;
    }

    // 跑到所有请求完成, 填好 stats。
    EngineStats run() {
        while (step()) {}
        stats.wall_us = clock;
        fill_latency_stats(stats, latencies);
        return stats;
    }
};
