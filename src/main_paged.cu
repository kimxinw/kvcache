#include "kv_cache.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

static std::vector<float> load_bin(const char* p, int n) {
    std::vector<float> v(n); FILE* f = fopen(p, "rb");
    if (!f) { printf("!! 缺 %s, 先 python ref/ref_attn.py\n", p); exit(1); }
    fread(v.data(), sizeof(float), n, f); 
    fclose(f); 
    return v;
}

int main() {
    const int H = 8, S = 128, D = 64, BLOCK = 16;
    const int NB = (S + BLOCK - 1) / BLOCK;          // 逻辑块数 = 8
    const float scale = 1.0f / sqrtf((float)D);

    auto q   = load_bin("data/q.bin",       H * D);
    auto K   = load_bin("data/K.bin",       H * S * D);   // 源布局 [H,S,D]
    auto V   = load_bin("data/V.bin",       H * S * D);
    auto ref = load_bin("data/out_ref.bin", H * D);

    // 决定 block table: 逻辑块 i -> 物理块 perm[i]。可以恒等, 可打乱。
    std::vector<int> perm(NB); 
    std::iota(perm.begin(), perm.end(), 0);
    bool SHUFFLE = (getenv("SHUFFLE") != nullptr);
    if (SHUFFLE) { 
        unsigned seed=42; 
        for(int i=NB-1;i>0;--i){
            seed=seed*1103515245+12345; 
            std::swap(perm[i], perm[seed%(i+1)]);
        } 
    }

    // 按 block table 把 [H,S,D] 的源数据搬进物理池 [NB,H,BLOCK,D]
    std::vector<float> kpool(NB*H*BLOCK*D, 0), vpool(NB*H*BLOCK*D, 0);
    for (int lb = 0; lb < NB; ++lb) {
        int phys = perm[lb];
        for (int h = 0; h < H; ++h)
            for (int t = 0; t < BLOCK; ++t) {
                int pos = lb*BLOCK + t; 
                if (pos >= S) break;
                for (int dd = 0; dd < D; ++dd) {
                    size_t src = ((size_t)h*S + pos)*D + dd;
                    size_t dst = (((size_t)phys*H + h)*BLOCK + t)*D + dd;
                    kpool[dst] = K[src]; 
                    vpool[dst] = V[src];
                }
            }
    }

    float *dq,*dkp,*dvp,*dout; int *dbt;
    cudaMalloc(&dq, H*D*4); 
    cudaMalloc(&dkp, kpool.size()*4); 
    cudaMalloc(&dvp, vpool.size()*4);
    cudaMalloc(&dout, H*D*4); 
    cudaMalloc(&dbt, NB*4);
    cudaMemcpy(dq, q.data(), H*D*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkp, kpool.data(), kpool.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dvp, vpool.data(), vpool.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbt, perm.data(), NB*4, cudaMemcpyHostToDevice);

    launch_paged(dq, dkp, dvp, dbt, dout, S, H, D, BLOCK, scale);

    std::vector<float> out(H*D);
    cudaMemcpy(out.data(), dout, H*D*4, cudaMemcpyDeviceToHost);
    float md = 0.f; for (int i=0;i<H*D;++i) md = fmaxf(md, fabsf(out[i]-ref[i]));
    printf("%s | max abs diff = %.6f -> %s\n",
           SHUFFLE ? "[shuffled table]" : "[identity table]",
           md, md < 1e-3f ? "PASS ✅" : "FAIL ❌");
    return 0;
}