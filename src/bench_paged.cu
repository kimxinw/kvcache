// bench_paged.cu —— 连续 KV cache decode vs 分页 (paged) decode 的单序列对比。
//
// 目的: 在【完全相同的输入】上跑 launch_decode 和 launch_paged,
//   (1) 验证两者数值一致 (paged 只是改了取数方式, 结果应相同);
//   (2) 测单步 decode 延迟, 量化 paged 因 block_table 间接寻址带来的开销。
//
// 注意结论方向: 单序列下 paged 预期【略慢】于连续 —— 这正是 paged 的代价,
//   它换来的显存收益要在多序列变长压测 (bench_throughput) 里才体现。
#include "kv_cache.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e){ \ 
    printf("CUDA err %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

// 简单可复现 RNG, 填 [-1,1] 区间
static void fill_rand(std::vector<float>& v, unsigned seed) {
    for (auto& x : v) { 
        seed = seed*1103515245u + 12345u; 
        x = (seed>>9)/8388608.0f*2.0f - 1.0f; 
    }
}

// 计时单个 kernel 启动 (warmup 后取平均, 返回微秒)
template <class F>
static float time_us(F launch, int warmup, int iters) {
    for (int i=0;i<warmup;++i) launch();
    CK(cudaDeviceSynchronize());
    cudaEvent_t ev0,ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    CK(cudaEventRecord(ev0));
    for (int i=0;i<iters;++i) launch();
    CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
    float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1));
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return ms/iters*1e3f;   // -> us
}

int main() {
    const int H = 8, D = 64, BLOCK = 16;
    const float scale = 1.0f/sqrtf((float)D);
    const int warmup = 30, iters = 300;

    printf("# H=%d D=%d BLOCK=%d  (identity block table)\n", H, D, BLOCK);
    printf("seq_len, decode_us, paged_us, overhead_%%, max_abs_diff, ok\n");

    for (int S = 128; S <= 4096; S *= 2) {
        const int NB = (S + BLOCK - 1) / BLOCK;          // 逻辑块数 = 物理块数 (identity)

        // ---- 主机端造数据 ----
        std::vector<float> q(H*D), K(H*S*D), V(H*S*D);
        fill_rand(q, 1); fill_rand(K, 2); fill_rand(V, 3);

        // 连续布局 [H,S,D] 直接就是 K/V
        // 分页池 [NB,H,BLOCK,D], identity table: 逻辑块 i -> 物理块 i
        std::vector<float> kpool((size_t)NB*H*BLOCK*D, 0), vpool((size_t)NB*H*BLOCK*D, 0);
        std::vector<int>   table(NB);
        for (int b=0;b<NB;++b) table[b]=b;
        for (int lb=0; lb<NB; ++lb)
            for (int h=0; h<H; ++h)
                for (int t=0; t<BLOCK; ++t) {
                    int pos = lb*BLOCK + t; if (pos>=S) break;
                    for (int d=0; d<D; ++d) {
                        size_t src = ((size_t)h*S + pos)*D + d;
                        size_t dst = (((size_t)lb*H + h)*BLOCK + t)*D + d;
                        kpool[dst]=K[src]; vpool[dst]=V[src];
                    }
                }

        // ---- 设备端 ----
        float *dq,*dKc,*dVc,*dkp,*dvp,*dout_c,*dout_p; int *dtab;
        CK(cudaMalloc(&dq,    H*D*sizeof(float)));
        CK(cudaMalloc(&dKc,   (size_t)H*S*D*sizeof(float)));
        CK(cudaMalloc(&dVc,   (size_t)H*S*D*sizeof(float)));
        CK(cudaMalloc(&dkp,   kpool.size()*sizeof(float)));
        CK(cudaMalloc(&dvp,   vpool.size()*sizeof(float)));
        CK(cudaMalloc(&dout_c,H*D*sizeof(float)));
        CK(cudaMalloc(&dout_p,H*D*sizeof(float)));
        CK(cudaMalloc(&dtab,  NB*sizeof(int)));
        CK(cudaMemcpy(dq, q.data(),  H*D*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dKc,K.data(),  (size_t)H*S*D*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dVc,V.data(),  (size_t)H*S*D*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dkp,kpool.data(), kpool.size()*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dvp,vpool.data(), vpool.size()*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dtab,table.data(), NB*sizeof(int), cudaMemcpyHostToDevice));

        auto run_decode = [&]{ launch_decode(dq, dKc, dVc, dout_c, S, H, D, S, scale); };
        auto run_paged  = [&]{ launch_paged (dq, dkp, dvp, dtab, dout_p, S, H, D, BLOCK, scale); };

        // ---- 正确性: 两条路径输出应一致 ----
        run_decode(); run_paged(); CK(cudaDeviceSynchronize());
        std::vector<float> oc(H*D), op(H*D);
        CK(cudaMemcpy(oc.data(), dout_c, H*D*sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(op.data(), dout_p, H*D*sizeof(float), cudaMemcpyDeviceToHost));
        float md=0.f; for (int i=0;i<H*D;++i) md=fmaxf(md, fabsf(oc[i]-op[i]));

        // ---- 计时 ----
        float t_c = time_us(run_decode, warmup, iters);
        float t_p = time_us(run_paged,  warmup, iters);
        printf("%d, %.2f, %.2f, %+.1f, %.6f, %s\n",
               S, t_c, t_p, (t_p/t_c-1.0f)*100.0f, md, md<1e-3f?"PASS":"FAIL");

        cudaFree(dq);cudaFree(dKc);cudaFree(dVc);cudaFree(dkp);cudaFree(dvp);
        cudaFree(dout_c);cudaFree(dout_p);cudaFree(dtab);
    }
    return 0;
}
