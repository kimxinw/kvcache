#pragma once
// sequence.h —— 引擎运行期「一条序列」的状态。
//
// 这是把 KV 内存与调度解耦的第一块：Sequence 只描述【逻辑/运行期状态】——
//   token 进度 + 它占用的 block_table + 状态机 + 停止条件 + 到达时刻(用于延迟统计)；
//   它不知道物理块怎么分配(归 KVCacheManager)、何时被调度(归 Engine/Scheduler)。
//
// 注意: 旧 demo 的 Request 把 workload(arrival/gen) 与运行期字段混在一起;
//   现在运行期状态在 Sequence, workload trace(给基准喂数据)是独立的 Request(见 scheduler.h)。
#include <vector>

enum class SeqStatus { Waiting, Running, Finished };

struct Sequence {
    int    id            = -1;
    int    prompt_len    = 0;      // prefill 的 KV 长度
    int    cur_len       = 0;      // 当前 KV 长度 = prompt + 已生成 token 数
    int    max_new_tokens= 0;      // 停止条件: 生成满这么多 token 即结束 (sampling param)
    double arrival_us    = 0;      // 到达墙钟时刻 (µs); 离线饱和场景全 = 0。延迟 = 完成时刻 - 此值
    std::vector<int> tokens;       // token id（接真模型后填；纯调度基准可留空）
    std::vector<int> block_table;  // 逻辑块号 -> 物理块号（分页 KV 的间接表）
    SeqStatus status = SeqStatus::Waiting;

    int  num_generated() const { return cur_len - prompt_len; }
    bool done()          const { return num_generated() >= max_new_tokens; }
    bool is_finished()   const { return status == SeqStatus::Finished; }
};
