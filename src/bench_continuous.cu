// bench_continuous.cu —— continuous (iteration-level) vs static (request-level) batching 对比。
//
// 同一条变长 trace, 同一套 paged decode kernel, 每个 step 都真跑 GPU(实测 per-step 延迟)。
// 关键变量是【输出长度的方差】: 方差越大, static 整批等组内最长序列结束的浪费越严重, continuous 优势越大。
//
// 两种到达模型:
//   ARRIVAL_RATE=0 (默认): 离线饱和 —— 所有请求 t=0 就位, 队列从头就满, 隔离纯"调度效率"的吞吐基准。
//   ARRIVAL_RATE>0       : 在线泊松 —— 请求按 λ req/s 的泊松过程到达(指数间隔), 体现排队延迟。
//                          此时 continuous 能即时把空 slot 让给刚到的请求, static 必须等整组结束才接新请求。
//
// 报告: 迭代数 / makespan(墙钟) / 吞吐(req·s, tok/s) / 平均 batch 宽度 / slot 利用率 / GPU 利用率 /
//       峰值块 / 平均&P99 延迟(到达->完成)。
//
// 可调 (环境变量): NREQ(256) BATCH(32) PROMPT_MAX(256) GEN_MAX(512) BUDGET_MB(2048) ARRIVAL_RATE(0)。
#include "scheduler.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

static int    env_int(const char* k,int d)   { const char* v=getenv(k); return v?atoi(v):d; }
static double env_dbl(const char* k,double d) { const char* v=getenv(k); return v?atof(v):d; }

int main(){
    const int H=8, D=64, BLOCK=16;
    const int NREQ       = env_int("NREQ", 256);
    const int BATCH      = env_int("BATCH", 32);
    const int PROMPT_MAX = env_int("PROMPT_MAX", 256);
    const int GEN_MAX    = env_int("GEN_MAX", 512);
    const int BUDGET_MB  = env_int("BUDGET_MB", 2048);
    const double RATE    = env_dbl("ARRIVAL_RATE", 0.0);             // req/s, 0 = 离线饱和
    const int per_block_bytes = H*BLOCK*D*2*4;                       // K+V, 1 block = H*BLOCK*D*2 floats
    const int pool_blocks = (int)((double)BUDGET_MB*1048576.0/per_block_bytes);

    // ---- 生成 trace: prompt/gen 都偏向短(r*r), 但 gen 跨度大(真实服务里输出长度极不均) ----
    // 到达: RATE>0 时按泊松过程(指数间隔)累积 arrival_us; 否则全 0。
    std::vector<Request> reqs(NREQ);
    unsigned seed=2024; long long gen_total=0, prompt_total=0; int gmin=1<<30, gmax=0;
    double t_arr=0;
    auto rnd=[&](){ seed=seed*1103515245u+12345u; return (seed>>9)/8388608.0f; };  // [0,1)
    for(int i=0;i<NREQ;++i){
        float rp=rnd(), rg=rnd();
        int p = 16 + (int)((PROMPT_MAX-16)*rp*rp);
        int g = 8  + (int)((GEN_MAX-8)  *rg*rg);
        if(RATE>0){ double u=rnd(); if(u<1e-9)u=1e-9; t_arr += -log(u)/RATE*1e6; }  // 指数间隔(µs)
        reqs[i].id=i; reqs[i].arrival_us=(RATE>0?t_arr:0.0); reqs[i].prompt_len=p; reqs[i].gen_len=g;
        gen_total+=g; prompt_total+=p;
        gmin=std::min(gmin,g); gmax=std::max(gmax,g);
    }

    Engine eng(H,D,BLOCK,BATCH,pool_blocks);

    SchedResult cs = run_static(reqs, eng, pool_blocks, BATCH);     // reqs 按值传入, 两次互不干扰
    SchedResult cc = run_continuous(reqs, eng, pool_blocks, BATCH);

    // ---- 报告 ----
    printf("# config: NREQ=%d BATCH=%d H=%d D=%d BLOCK=%d  pool=%d blocks (%d MB)  arrival_rate=%.0f req/s%s\n",
           NREQ,BATCH,H,D,BLOCK,pool_blocks,BUDGET_MB,RATE, RATE>0?"":" (offline saturated)");
    printf("# trace: prompt mean=%.0f  gen[min=%d mean=%.0f max=%d]  total_gen_tokens=%lld\n",
           (double)prompt_total/NREQ, gmin,(double)gen_total/NREQ,gmax, gen_total);

    auto row=[&](const SchedResult& R){
        double wall=R.wall_us;
        printf("%-11s %7d %11.1f %9.0f %9.0f %9.1f %8.1f%% %8.1f%% %8d %9.1f %9.1f\n",
            R.name, R.iters, wall/1000.0,
            NREQ/(wall*1e-6), gen_total/(wall*1e-6),
            (double)R.slot_steps/R.iters, 100.0*R.useful_steps/R.slot_steps,
            100.0*R.gpu_us/wall, R.peak_blocks,
            R.avg_latency_us/1000.0, R.p99_latency_us/1000.0);
    };
    printf("\n%-11s %7s %11s %9s %9s %9s %9s %9s %8s %9s %9s\n",
        "scheme","iters","wall_ms","req/s","tok/s","avg_batch","slot_eff","gpu_util","peak_blk",
        "avgLat_ms","p99Lat_ms");
    row(cs); row(cc);

    printf("\n# continuous vs static: 吞吐 %.2fx | makespan %.2fx | 平均延迟 %.2fx | P99 延迟 %.2fx\n",
        cs.wall_us/cc.wall_us, cs.wall_us/cc.wall_us,
        cs.avg_latency_us/cc.avg_latency_us, cs.p99_latency_us/cc.p99_latency_us);
    printf("# slot 利用率: static %.1f%% (整批等最长序列, 完成的 slot 空转) vs continuous %.1f%% (随完随补)\n",
        100.0*cs.useful_steps/cs.slot_steps, 100.0*cc.useful_steps/cc.slot_steps);

    // ---- 机器可读 CSV ----
    FILE* f=fopen("data/continuous_summary.csv","w");
    if(f){
        fprintf(f,"metric,static,continuous\n");
        fprintf(f,"iters,%d,%d\n",cs.iters,cc.iters);
        fprintf(f,"makespan_ms,%.1f,%.1f\n",cs.wall_us/1000.0,cc.wall_us/1000.0);
        fprintf(f,"req_per_s,%.1f,%.1f\n",NREQ/(cs.wall_us*1e-6),NREQ/(cc.wall_us*1e-6));
        fprintf(f,"tok_per_s,%.0f,%.0f\n",gen_total/(cs.wall_us*1e-6),gen_total/(cc.wall_us*1e-6));
        fprintf(f,"avg_batch,%.1f,%.1f\n",(double)cs.slot_steps/cs.iters,(double)cc.slot_steps/cc.iters);
        fprintf(f,"slot_eff_pct,%.1f,%.1f\n",
            100.0*cs.useful_steps/cs.slot_steps,100.0*cc.useful_steps/cc.slot_steps);
        fprintf(f,"gpu_util_pct,%.1f,%.1f\n",100.0*cs.gpu_us/cs.wall_us,100.0*cc.gpu_us/cc.wall_us);
        fprintf(f,"peak_blocks,%d,%d\n",cs.peak_blocks,cc.peak_blocks);
        fprintf(f,"avg_lat_ms,%.1f,%.1f\n",cs.avg_latency_us/1000.0,cc.avg_latency_us/1000.0);
        fprintf(f,"p99_lat_ms,%.1f,%.1f\n",cs.p99_latency_us/1000.0,cc.p99_latency_us/1000.0);
        fclose(f);
        printf("# wrote data/continuous_summary.csv\n");
    }
    return 0;
}
