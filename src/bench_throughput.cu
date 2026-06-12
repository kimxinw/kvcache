// bench_throughput.cu —— 多序列变长压测: 连续 KV cache vs 分页 (paged) 的显存与吞吐对比。
//
// 这是体现 PagedAttention 真正价值的实验。一批 N 条【长度各异】的序列各 decode 一步:
//   连续方案: 每条必须按 max_seq_len 预留 [H, MAXS, D] (不能动态增长) -> 短序列大量浪费。
//   分页方案: 每条只按实际长度向共享池借 ceil(len/BLOCK) 个物理块 -> 显存几乎不浪费。
// 报告: 预留显存 / 实际占用 / 利用率 / 单步延迟 / 吞吐, 外加"固定显存预算下能并发多少条"。
//
// 可调 (环境变量): NSEQ (默认 128), LMAX (默认 2048), BUDGET_MB (默认 4096)。
#include "kv_cache.h"
#include "block_alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e){ \
    printf("CUDA err %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

// 确定性伪随机, 让连续/分页两套布局填入完全相同的逻辑数据 -> 输出应一致
static inline float hval(unsigned s,unsigned h,unsigned pos,unsigned d,unsigned salt){
    unsigned x = (((s*131u+h)*131u+pos)*131u+d) + salt*2654435761u;
    x ^= x>>13; x *= 0x9E3779B1u; x ^= x>>16;
    return (x>>9)/8388608.0f*2.0f - 1.0f;   // [-1,1)
}

template <class F> static float time_us(F launch, int warmup, int iters) {
    for (int i=0;i<warmup;++i) launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i=0;i<iters;++i) launch();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms/iters*1e3f;
}

static int env_int(const char* k, int dft){ const char* v=getenv(k); return v?atoi(v):dft; }

int main() {
    const int H = 8, D = 64, BLOCK = 16;
    const float scale = 1.0f/sqrtf((float)D);
    const int N    = env_int("NSEQ", 128);
    const int LMAX = env_int("LMAX", 2048);
    const double BUDGET_MB = env_int("BUDGET_MB", 4096);
    const int warmup = 20, iters = 100;

    // ---- 变长序列: 用 r*r 偏向短序列 (贴近真实服务: 多数请求短, 少数很长) ----
    std::vector<int> len(N);
    unsigned seed = 12345; long long sum_len = 0; int max_len = 0;
    for (int s=0;s<N;++s){
        seed = seed*1103515245u + 12345u;
        float r = (seed>>9)/8388608.0f;          // [0,1)
        int L = 16 + (int)((LMAX-16) * r*r);
        len[s]=L; sum_len+=L; if(L>max_len)max_len=L;
    }
    double mean_len = (double)sum_len/N;
    const int cap = max_len;                      // shared mem 上界

    // ---- 分页: 给每条序列分配物理块, 拼 block_table ----
    std::vector<int> nb(N), tab_off(N+1,0), table_all;
    int total_blocks = 0;
    for (int s=0;s<N;++s){ nb[s]=blocks_needed(len[s],BLOCK); total_blocks+=nb[s]; }
    BlockAllocator alloc(total_blocks);           // 恰好够用的池
    table_all.reserve(total_blocks);
    for (int s=0;s<N;++s){
        tab_off[s] = (int)table_all.size();
        for (int b=0;b<nb[s];++b){ int p=alloc.alloc(); table_all.push_back(p); }
    }
    tab_off[N] = (int)table_all.size();

    // ---- 显存账本 ----
    double per_elem = sizeof(float);
    double contig_reserved = (double)N*H*LMAX*D*per_elem*2;          // K+V, 每条预留 MAXS
    double contig_used     = (double)sum_len*H*D*per_elem*2;         // 真实用到的行
    double paged_used      = (double)total_blocks*H*BLOCK*D*per_elem*2
                           + table_all.size()*sizeof(int);           // 池 + block_table
    auto MB=[](double b){return b/1048576.0;};

    // ---- 填充主机缓冲并上设备 ----
    std::vector<int> dlen=len;
    size_t contig_n = (size_t)N*H*LMAX*D;
    size_t pool_n   = (size_t)total_blocks*H*BLOCK*D;
    std::vector<float> kc(contig_n,0.f), vc(contig_n,0.f);
    std::vector<float> kp(pool_n,0.f),   vp(pool_n,0.f);
    std::vector<float> q((size_t)N*H*D);

    for (int s=0;s<N;++s)
      for (int h=0;h<H;++h){
        for (int d=0;d<D;++d) q[((size_t)s*H+h)*D+d] = hval(s,h,0,d,1);
        for (int pos=0;pos<len[s];++pos)
          for (int d=0;d<D;++d){
            float kv_k = hval(s,h,pos,d,2), kv_v = hval(s,h,pos,d,3);
            kc[(((size_t)s*H+h)*LMAX+pos)*D+d] = kv_k;        // 连续
            vc[(((size_t)s*H+h)*LMAX+pos)*D+d] = kv_v;
            int phys = table_all[tab_off[s] + pos/BLOCK];     // 分页
            int in   = pos % BLOCK;
            kp[(((size_t)phys*H+h)*BLOCK+in)*D+d] = kv_k;
            vp[(((size_t)phys*H+h)*BLOCK+in)*D+d] = kv_v;
          }
      }

    float *dq,*dkc,*dvc,*dkp,*dvp,*doc,*dop; int *dlen_d,*dtab,*dtoff;
    CK(cudaMalloc(&dq,  q.size()*4));
    CK(cudaMalloc(&dkc, contig_n*4)); CK(cudaMalloc(&dvc, contig_n*4));
    CK(cudaMalloc(&dkp, pool_n*4));   CK(cudaMalloc(&dvp, pool_n*4));
    CK(cudaMalloc(&doc, q.size()*4)); CK(cudaMalloc(&dop, q.size()*4));
    CK(cudaMalloc(&dlen_d, N*4));
    CK(cudaMalloc(&dtab, table_all.size()*4)); CK(cudaMalloc(&dtoff,(N+1)*4));
    CK(cudaMemcpy(dq, q.data(), q.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkc,kc.data(),contig_n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvc,vc.data(),contig_n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkp,kp.data(),pool_n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvp,vp.data(),pool_n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlen_d,dlen.data(),N*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtab,table_all.data(),table_all.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtoff,tab_off.data(),(N+1)*4, cudaMemcpyHostToDevice));

    auto run_c=[&]{ launch_batched_decode(dq,dkc,dvc,dlen_d,doc,N,H,D,LMAX,cap,scale); };
    auto run_p=[&]{ launch_batched_paged (dq,dkp,dvp,dtab,dtoff,dlen_d,dop,N,H,D,BLOCK,cap,scale); };

    // ---- 正确性: 两套布局应给出相同结果 ----
    run_c(); run_p(); CK(cudaDeviceSynchronize());
    std::vector<float> oc(q.size()), op(q.size());
    CK(cudaMemcpy(oc.data(),doc,q.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(op.data(),dop,q.size()*4,cudaMemcpyDeviceToHost));
    float md=0.f; for(size_t i=0;i<q.size();++i) md=fmaxf(md,fabsf(oc[i]-op[i]));

    // ---- 计时 ----
    float t_c = time_us(run_c, warmup, iters);
    float t_p = time_us(run_p, warmup, iters);

    // ---- 固定显存预算下的并发上限 (KV 只算 K+V) ----
    double per_seq_contig = (double)H*LMAX*D*per_elem*2;                 // 必须按 MAXS 预留
    double per_seq_paged  = (double)blocks_needed((int)mean_len,BLOCK)*H*BLOCK*D*per_elem*2;
    double budget = BUDGET_MB*1048576.0;
    long max_c = (long)(budget/per_seq_contig);
    long max_p = (long)(budget/per_seq_paged);

    // ---- 输出 ----
    printf("# config: N=%d H=%d D=%d BLOCK=%d LMAX=%d\n", N,H,D,BLOCK,LMAX);
    printf("# lengths: min=%d max=%d mean=%.0f  total_blocks=%d (avg %.1f blk/seq)\n",
           16, max_len, mean_len, total_blocks, (double)total_blocks/N);
    printf("# correctness contiguous-vs-paged: max_abs_diff=%.6f -> %s\n",
           md, md<1e-3f?"PASS":"FAIL");
    printf("\nscheme,      reserved_MB, used_MB, util_%%, step_us, tokens_per_s\n");
    printf("contiguous,  %10.1f, %7.1f, %5.1f, %7.2f, %10.0f\n",
           MB(contig_reserved), MB(contig_used), 100.0*contig_used/contig_reserved,
           t_c, N/(t_c*1e-6f));
    printf("paged,       %10.1f, %7.1f, %5.1f, %7.2f, %10.0f\n",
           MB(paged_used), MB(paged_used), 100.0, t_p, N/(t_p*1e-6f));
    printf("\n# memory: paged 用 %.1f MB vs 连续预留 %.1f MB  -> 省 %.1fx\n",
           MB(paged_used), MB(contig_reserved), contig_reserved/paged_used);
    printf("# fixed budget %.0f MB 可并发: contiguous=%ld 条, paged=%ld 条  -> %.1fx 更多\n",
           BUDGET_MB, max_c, max_p, (double)max_p/max_c);
    printf("# 单步延迟 paged/contig = %.2fx (间接寻址代价)\n", t_p/t_c);

    // ---- 机器可读 CSV (给 ref/plot.py 画图) ----
    FILE* f = fopen("data/throughput_summary.csv", "w");
    if (f) {
        fprintf(f, "metric,contiguous,paged\n");
        fprintf(f, "reserved_MB,%.1f,%.1f\n", MB(contig_reserved), MB(paged_used));
        fprintf(f, "used_MB,%.1f,%.1f\n",     MB(contig_used),     MB(paged_used));
        fprintf(f, "util_pct,%.1f,%.1f\n",    100.0*contig_used/contig_reserved, 100.0);
        fprintf(f, "step_us,%.2f,%.2f\n",     t_c, t_p);
        fprintf(f, "tokens_per_s,%.0f,%.0f\n", N/(t_c*1e-6f), N/(t_p*1e-6f));
        fprintf(f, "max_seqs_budget,%ld,%ld\n", max_c, max_p);
        fclose(f);
        printf("# wrote data/throughput_summary.csv\n");
    }
    cudaFree(dq);cudaFree(dkc);cudaFree(dvc);cudaFree(dkp);cudaFree(dvp);
    cudaFree(doc);cudaFree(dop);cudaFree(dlen_d);cudaFree(dtab);cudaFree(dtoff);
    return 0;
}
