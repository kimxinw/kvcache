#pragma once
// scheduler.h —— 基准用的【workload trace 类型】+【static (request-level) 批处理基线】。
//
// 历史: 这里曾自带一个执行器(也叫 Engine)+ 块管理 + continuous/static 两套循环, 与别处重复。
//   重构后职责收敛:
//     - 执行(跑一步 paged decode)     -> ModelRunner
//     - 块/池管理                      -> KVCacheManager
//     - continuous batching(主力实现)  -> Engine (engine.h)
//   本文件只剩两样东西:
//     (1) Request: 基准喂数据的 workload trace(到达时刻 + prompt 长 + 要生成多少)。
//     (2) run_static: 故意次优的对照基线 —— 一组请求一起跑, 必须等组内【最长】序列结束才换下一组,
//         早完成的 slot 空转(kernel 照样空算 + 块不还)。它复用同一套 KVCacheManager + ModelRunner,
//         唯一目的是量化 continuous(Engine) 的收益。
//
// 用法 (见 bench_continuous.cu):
//   Engine eng(cfg);
//   EngineStats st = run_static(reqs, eng.kvm, eng.runner, cfg.max_batch);  // 基线
//   eng.reset(); for(r:reqs) eng.add_request(r.prompt_len, r.gen_len, r.arrival_us);
//   EngineStats co = eng.run();                                              // 主力
#include "engine.h"
#include <vector>
#include <algorithm>

// 基准输入: 一条请求的 workload。运行期状态不在这里(在 Sequence)。
struct Request {
    int    id          = 0;
    double arrival_us  = 0;   // 到达墙钟时刻 (µs); 离线饱和压测时全 = 0
    int    prompt_len  = 0;
    int    gen_len     = 0;   // 要生成的 token 数 (= Sequence.max_new_tokens)
};

// ============================ static (request-level) 基线 ============================
inline EngineStats run_static(std::vector<Request> reqs, KVCacheManager& kvm,
                              ModelRunner& runner, int maxB) {
    EngineStats R; R.name = "static";
    std::sort(reqs.begin(), reqs.end(),
              [](const Request& a, const Request& b){ return a.arrival_us < b.arrival_us; });
    kvm.reset();

    // 由 trace 建运行期 Sequence (arrival 顺序)。seqs 定长, 不再扩容 -> 指针稳定。
    std::vector<Sequence> seqs(reqs.size());
    for (size_t i = 0; i < reqs.size(); ++i) {
        seqs[i].id = reqs[i].id;
        seqs[i].prompt_len     = reqs[i].prompt_len;
        seqs[i].max_new_tokens = reqs[i].gen_len;
        seqs[i].arrival_us     = reqs[i].arrival_us;
    }
    auto wc = [&](const Sequence& s){
        return blocks_needed(s.prompt_len + s.max_new_tokens, kvm.block_size);
    };

    std::vector<double> lat;//FP64
    size_t idx = 0; double clock = 0;

    while (idx < seqs.size()) {
        if (seqs[idx].arrival_us > clock) clock = seqs[idx].arrival_us;   // 空转等第一条到达
        // ---- 组建一组: 此刻【已到达】的最多 maxB 条 (受池预留约束); 晚到的留给下一组 ----
        std::vector<Sequence*> group; int reserved = 0;
        while ((int)group.size() < maxB && idx < seqs.size() && seqs[idx].arrival_us <= clock) {
            int fp = wc(seqs[idx]);
            if (reserved + fp > kvm.num_blocks) break;
            kvm.allocate_prompt(seqs[idx]); reserved += fp;
            group.push_back(&seqs[idx]); ++idx;
        }
        if (group.empty()) break;                                  // 单条都放不下 -> 池配置过小
        // ---- 跑到组内全部完成: batch 宽度 = group.size() 不变(完成的 slot 空转) ----
        int active = (int)group.size();
        while (active > 0) {
            double us = runner.decode_step(group, kvm);
            R.gpu_us += us; clock += us; R.iters++;
            R.slot_steps += (int)group.size();                     // 宽度含已完成的空转 slot
            int still = 0;
            for (Sequence* s : group) {
                if (!s->done()) {                                  // 本步开始时仍需生成
                    kvm.append_token(*s); R.tokens++; ++still;
                    if (s->done()) { lat.push_back(clock - s->arrival_us); R.finished++; }
                }
            }
            R.useful_steps += still;                               // 本步真正干活的序列数
            R.peak_blocks = std::max(R.peak_blocks, kvm.used_blocks());
            active = 0; for (Sequence* s : group) if (!s->done()) ++active;
        }
        // ---- 整组归还 (短序列占的块到这里才释放) ----
        for (Sequence* s : group) kvm.free(*s);
    }
    R.wall_us = clock;
    fill_latency_stats(R, lat);
    return R;
}
