#pragma once
// scheduler.h —— continuous batching 调度器 (主机端) + 真跑 paged decode kernel。
//
// 复现 vLLM 的 "iteration-level scheduling": 每个 decode step 之前重组 batch ——
//   退场已完成的序列(立刻归还其物理块)、从 waiting 队列补进新请求 —— 让 batch 始终接近满载。
// 对照组 static (request-level) batching: 一组请求一起跑, 必须等组内【最长】序列结束才换下一组;
//   早完成的序列仍占着 slot 和物理块(kernel 照样空算 + 显存不还) —— 这正是 continuous batching 消除的浪费。
//
// 两种策略每个 step 都真实调用 launch_batched_paged(batched.cu) 对【当前 running 集合】跑一步,
// per-step 延迟用 cudaEvent 实测 —— 所以 makespan/吞吐是真数字, 差异来自 (a) 总迭代数 (b) 每步 batch 占用率。
//
// 计时模型: clock 是【墙钟】(µs), 每跑一步推进 step 实测时延; 等待新请求到达时空转推进(不计 GPU 忙时)。
//   - wall_us : 端到端墙钟 (含空闲) -> 吞吐 = NREQ / wall。
//   - gpu_us  : GPU 忙时之和 -> gpu_util = gpu_us / wall。
//   - 请求延迟 = 完成时刻 - 到达时刻 (arrival_us)。离线饱和场景 arrival_us 全 = 0, wall == gpu。
//
// 显存模型 (本 demo 不做抢占): 入场时按 worst-case 长度(prompt+gen)预留块数, 但物理块按 token【惰性】分配。
//   预留保证生成途中绝不 OOM(无需抢占); 惰性分配让 peak_blocks 反映真实占用。
//   continuous 一旦序列完成立刻释放其预留 -> 同一池子下能更快补进新请求 -> batch 更满。
#include "block_alloc.h"
#include "kv_cache.h"
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef SCHED_CK
#define SCHED_CK(x) do { cudaError_t e=(x); if(e){ \
    printf("CUDA err %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)
#endif

// 一条请求 (prompt + 期望生成长度)。运行期字段记录占用的物理块与起止迭代, 用于算延迟。
struct Request {
    int id;
    double arrival_us; // 到达墙钟时刻 (µs)。离线饱和压测时全 = 0
    int prompt_len;    // prefill 的 KV 长度
    int gen_len;       // 需要生成的 token 数 (decode 步数)
    // ---- 运行期 ----
    int cur_len   = 0; // 当前 KV 长度 = prompt + 已生成
    int generated = 0; // 已生成 token 数
    int start_iter  = -1;
    int finish_iter = -1;
    std::vector<int> blocks; // 占用的物理块号 (paged)
};

// 一次调度跑完的汇总指标
struct SchedResult {
    const char* name = "";
    int       iters        = 0;  // 总迭代数 (= 总 step 数)
    double    gpu_us       = 0;  // GPU 忙时之和
    double    wall_us      = 0;  // 端到端墙钟 (含等待空闲) = makespan
    long long slot_steps   = 0;  // Σ 每步 batch 宽度 (含 static 的空转 slot)
    long long useful_steps = 0;  // Σ 每步真正在生成的序列数
    int       peak_blocks  = 0;  // 峰值物理块占用
    double    avg_latency_us = 0; // 请求平均端到端时间 (到达 -> 完成)
    double    p99_latency_us = 0;
};

// ---- GPU 执行器: 持有设备端 KV 池 / q / out / table 暂存, 对当前 batch 真跑一步 paged decode ----
// 注: 本基准只测调度效率, KV 内容无关紧要(数值正确性已由 bench_throughput 证明), 池填 0 即可。
struct Engine {
    int H, D, BLOCK, B; float scale; int pool_blocks;
    float *dq=nullptr,*dkp=nullptr,*dvp=nullptr,*dout=nullptr;
    int   *dtab=nullptr,*dtoff=nullptr,*dlen=nullptr;
    cudaEvent_t ev0{}, ev1{};
    std::vector<int> h_tab, h_off, h_len;          // 每步重建的 block_table / 偏移 / 长度

    Engine(int H_,int D_,int BLOCK_,int maxB,int pool_blocks_)
      : H(H_),D(D_),BLOCK(BLOCK_),B(maxB),scale(1.f/sqrtf((float)D_)),pool_blocks(pool_blocks_) {
        size_t pool_n=(size_t)pool_blocks*H*BLOCK*D;
        SCHED_CK(cudaMalloc(&dkp,pool_n*4)); SCHED_CK(cudaMalloc(&dvp,pool_n*4));
        SCHED_CK(cudaMemset(dkp,0,pool_n*4)); SCHED_CK(cudaMemset(dvp,0,pool_n*4));
        SCHED_CK(cudaMalloc(&dq,(size_t)B*H*D*4)); SCHED_CK(cudaMemset(dq,0,(size_t)B*H*D*4));
        SCHED_CK(cudaMalloc(&dout,(size_t)B*H*D*4));
        SCHED_CK(cudaMalloc(&dtab,(size_t)pool_blocks*4));
        SCHED_CK(cudaMalloc(&dtoff,(size_t)(B+1)*4));
        SCHED_CK(cudaMalloc(&dlen,(size_t)B*4));
        SCHED_CK(cudaEventCreate(&ev0)); SCHED_CK(cudaEventCreate(&ev1));
        h_tab.reserve(pool_blocks); h_off.reserve(B+1); h_len.reserve(B);
    }
    ~Engine(){ cudaFree(dkp);cudaFree(dvp);cudaFree(dq);cudaFree(dout);
               cudaFree(dtab);cudaFree(dtoff);cudaFree(dlen);
               cudaEventDestroy(ev0);cudaEventDestroy(ev1); }
    Engine(const Engine&)=delete; Engine& operator=(const Engine&)=delete;

    // 对当前 running batch 真跑一步 paged decode, 返回实测 µs。
    double decode_step(const std::vector<Request*>& batch){
        int n=(int)batch.size(); if(n==0) return 0.0;
        h_tab.clear(); h_off.clear(); h_len.clear();
        int cap=1;
        for(int s=0;s<n;++s){
            h_off.push_back((int)h_tab.size());          // 第 s 条 block_table 在 table_all 里的起点
            Request* r=batch[s];
            for(int b : r->blocks) h_tab.push_back(b);
            int L = r->cur_len>0 ? r->cur_len : 1;       // 至少 1, 防 0 长度
            h_len.push_back(L); cap = std::max(cap,L);
        }
        h_off.push_back((int)h_tab.size());
        SCHED_CK(cudaMemcpy(dtab, h_tab.data(), h_tab.size()*4, cudaMemcpyHostToDevice));
        SCHED_CK(cudaMemcpy(dtoff,h_off.data(), h_off.size()*4, cudaMemcpyHostToDevice));
        SCHED_CK(cudaMemcpy(dlen, h_len.data(), (size_t)n*4,    cudaMemcpyHostToDevice));
        SCHED_CK(cudaEventRecord(ev0));
        launch_batched_paged(dq,dkp,dvp,dtab,dtoff,dlen,dout,n,H,D,BLOCK,cap,scale);
        SCHED_CK(cudaEventRecord(ev1)); SCHED_CK(cudaEventSynchronize(ev1));
        float ms; SCHED_CK(cudaEventElapsedTime(&ms,ev0,ev1));
        return ms*1e3;   // -> µs
    }
};

// 序列生成一个 token: KV 长度 +1, 跨 block 边界时借新块。worst-case 已预留, 故 alloc 不会失败。
static inline void grow_one(Request& r, BlockAllocator& alloc, int BLOCK){
    int need = blocks_needed(r.cur_len+1, BLOCK);
    if(need > (int)r.blocks.size()) r.blocks.push_back(alloc.alloc());  // 已预留, 必有空块
    r.cur_len++; r.generated++;
}

// 入场: 预留 worst-case 块数(防 OOM), 但只惰性分配 prompt 所需的块。返回是否成功。
static inline bool admit(Request& r, BlockAllocator& alloc, int& reserved, int pool_blocks,
                         int BLOCK, int iter){
    int fp = blocks_needed(r.prompt_len + r.gen_len, BLOCK);   // worst-case 足迹
    if(reserved + fp > pool_blocks) return false;              // 池放不下, 等有序列退场
    int np = blocks_needed(r.prompt_len, BLOCK);
    for(int b=0;b<np;++b) r.blocks.push_back(alloc.alloc());
    r.cur_len    = r.prompt_len;
    r.start_iter = iter;
    reserved    += fp;
    return true;
}
static inline void release(Request& r, BlockAllocator& alloc, int& reserved, int BLOCK){
    alloc.free_blocks(r.blocks); r.blocks.clear();
    reserved -= blocks_needed(r.prompt_len + r.gen_len, BLOCK);
}

static inline void finalize_latency(SchedResult& R, std::vector<double>& comp){
    std::vector<double> v = comp; std::sort(v.begin(), v.end());
    double s=0; for(double x:v) s+=x;
    R.avg_latency_us = v.empty()?0:s/v.size();
    R.p99_latency_us = v.empty()?0:v[std::min((size_t)(v.size()*0.99), v.size()-1)];
}

// ============================ continuous (iteration-level) ============================
inline SchedResult run_continuous(std::vector<Request> reqs, Engine& eng, int pool_blocks, int maxB){
    SchedResult R; R.name="continuous";
    std::sort(reqs.begin(),reqs.end(),[](const Request&a,const Request&b){return a.arrival_us<b.arrival_us;});
    BlockAllocator alloc(pool_blocks);
    int reserved=0;
    std::vector<Request*> running;
    std::vector<double> comp(reqs.size(), 0.0);
    size_t next=0; double clock=0;

    while(true){
        // 1) 退场: 完成的序列归还块 + 预留, 记录端到端延迟 = 完成时刻 - 到达时刻
        for(size_t i=0;i<running.size();){
            Request* r=running[i];
            if(r->generated>=r->gen_len){
                release(*r, alloc, reserved, eng.BLOCK);
                comp[r->id]=clock - r->arrival_us; r->finish_iter=R.iters;
                running[i]=running.back(); running.pop_back();   // O(1) 删除
            } else ++i;
        }
        // 2) 入场: 凡已到达(arrival_us<=clock)的请求, 在 batch 宽度与池预留双约束下尽量补满
        while((int)running.size()<maxB && next<reqs.size() && reqs[next].arrival_us<=clock){
            if(admit(reqs[next], alloc, reserved, pool_blocks, eng.BLOCK, R.iters)){
                running.push_back(&reqs[next]); ++next;
            } else break;   // 块不够, 等下一轮(会有序列完成腾出预留)
        }
        if(running.empty()){
            if(next>=reqs.size()) break;                    // 全部完成
            clock = std::max(clock, reqs[next].arrival_us);  // 空转等下一个请求到达 (不计 GPU 忙时)
            continue;
        }
        // 3) 真跑一步 decode -> 推进墙钟
        double us = eng.decode_step(running);
        R.gpu_us += us; clock += us; R.iters++;
        R.slot_steps += (int)running.size(); R.useful_steps += (int)running.size();
        // 4) 每条生成一个 token
        for(Request* r : running) grow_one(*r, alloc, eng.BLOCK);
        R.peak_blocks = std::max(R.peak_blocks, alloc.used_count());
    }
    R.wall_us = clock;
    finalize_latency(R, comp);
    return R;
}

// ============================ static (request-level) ============================
inline SchedResult run_static(std::vector<Request> reqs, Engine& eng, int pool_blocks, int maxB){
    SchedResult R; R.name="static";
    std::sort(reqs.begin(),reqs.end(),[](const Request&a,const Request&b){return a.arrival_us<b.arrival_us;});
    BlockAllocator alloc(pool_blocks);
    std::vector<double> comp(reqs.size(), 0.0);
    size_t idx=0; double clock=0;

    while(idx<reqs.size()){
        if(reqs[idx].arrival_us > clock) clock = reqs[idx].arrival_us;  // 空转等第一条到达
        // ---- 组建一组: 取此刻【已到达】的、最多 maxB 条 (受池预留约束); 晚到的留给下一组 ----
        std::vector<Request*> group; int reserved=0;
        while((int)group.size()<maxB && idx<reqs.size() && reqs[idx].arrival_us<=clock){
            if(!admit(reqs[idx], alloc, reserved, pool_blocks, eng.BLOCK, R.iters)) break;
            group.push_back(&reqs[idx]); ++idx;
        }
        if(group.empty()) break;                                  // 单条都放不下 -> 池配置过小
        // ---- 跑到组内全部完成: batch 宽度 = group.size() 始终不变(完成的 slot 空转) ----
        int active=(int)group.size();
        while(active>0){
            double us = eng.decode_step(group); R.gpu_us += us; clock += us; R.iters++;
            R.slot_steps += (int)group.size();                    // 宽度含已完成的空转 slot
            int still=0;
            for(Request* r : group){
                if(r->generated < r->gen_len){
                    grow_one(*r, alloc, eng.BLOCK); ++still;
                    if(r->generated >= r->gen_len){ comp[r->id]=clock - r->arrival_us; r->finish_iter=R.iters; }
                }
            }
            R.useful_steps += still;                              // 本步真正干活的序列数
            R.peak_blocks = std::max(R.peak_blocks, alloc.used_count());
            active=0; for(Request* r : group) if(r->generated < r->gen_len) ++active;
        }
        // ---- 整组归还 (只有到这里短序列占的块才被释放) ----
        for(Request* r : group){ alloc.free_blocks(r->blocks); r->blocks.clear(); }
    }
    R.wall_us = clock;
    finalize_latency(R, comp);
    return R;
}
