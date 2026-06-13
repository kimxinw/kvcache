// test_paged_flash.cu —— 验证权威 kernel batched_paged_flash(在线 softmax) 与已验证的
//   batched_paged(两遍 softmax) 在随机输入下输出逐元素一致。两者读同一池 + 同一组 block_table,
//   故一致即证明 flash 版数值正确。
// 编译: nvcc -std=c++17 -arch=sm_86 -I src/include tests/test_paged_flash.cu src/batched.cu -o build/test_paged_flash
#include "kv_cache.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

int main() {
    const int H = 8, D = 64, BLOCK = 4;          // D 须为 2 的幂(树形 reduce)
    const int num_blocks = 16;
    const float scale = 1.0f / sqrtf((float)D);

    // 3 条变长序列, 顺序分配互不重叠的物理块。
    int cur_len[] = {5, 8, 13};
    const int N = 3;
    std::vector<int> tab_off = {0};
    std::vector<int> table_all;
    int next = 0;
    for (int s = 0; s < N; ++s) {
        int nb = (cur_len[s] + BLOCK - 1) / BLOCK;
        for (int b = 0; b < nb; ++b) table_all.push_back(next++);
        tab_off.push_back((int)table_all.size());
    }
    int cap = 0; for (int s = 0; s < N; ++s) cap = cur_len[s] > cap ? cur_len[s] : cap;

    // 随机填池 + q。
    srand(7);
    auto rnd = []{ return (float)rand()/RAND_MAX*2.f - 1.f; };
    size_t pool_n = (size_t)num_blocks*H*BLOCK*D;
    std::vector<float> kp(pool_n), vp(pool_n), q((size_t)N*H*D);
    for (auto& x : kp) x = rnd();
    for (auto& x : vp) x = rnd();
    for (auto& x : q)  x = rnd();

    float *dq,*dkp,*dvp,*dout1,*dout2; int *dtab,*dtoff,*dlen;
    cudaMalloc(&dq, q.size()*4); cudaMalloc(&dkp, pool_n*4); cudaMalloc(&dvp, pool_n*4);
    cudaMalloc(&dout1, (size_t)N*H*D*4); cudaMalloc(&dout2, (size_t)N*H*D*4);
    cudaMalloc(&dtab, table_all.size()*4); cudaMalloc(&dtoff, tab_off.size()*4); cudaMalloc(&dlen, N*4);
    cudaMemcpy(dq, q.data(), q.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkp, kp.data(), pool_n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dvp, vp.data(), pool_n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtab, table_all.data(), table_all.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtoff, tab_off.data(), tab_off.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dlen, cur_len, N*4, cudaMemcpyHostToDevice);

    launch_batched_paged      (dq,dkp,dvp,dtab,dtoff,dlen,dout1,N,H,D,BLOCK,cap,scale);
    launch_batched_paged_flash(dq,dkp,dvp,dtab,dtoff,dlen,dout2,N,H,D,BLOCK,cap,scale);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    std::vector<float> o1((size_t)N*H*D), o2((size_t)N*H*D);
    cudaMemcpy(o1.data(), dout1, o1.size()*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o2.data(), dout2, o2.size()*4, cudaMemcpyDeviceToHost);

    float maxdiff = 0.f;
    for (size_t i = 0; i < o1.size(); ++i) maxdiff = fmaxf(maxdiff, fabsf(o1[i]-o2[i]));
    int pass = (err == cudaSuccess) && (maxdiff < 1e-4f);
    printf("%s | max abs diff (flash vs two-pass) = %.3e -> %s\n",
           cudaGetErrorString(err), maxdiff, pass ? "PASS ✅" : "FAIL ❌");
    return pass ? 0 : 1;
}
