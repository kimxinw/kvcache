#include "kv_cache.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

static float time_step(void (*fn)(), int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    cudaEvent_t s, e; 
    cudaEventCreate(&s); 
    cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(e); 
    cudaEventSynchronize(e);
    float ms; 
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;   // 单步平均毫秒
}

// 用全局变量把参数喂给无参函数指针
static float *g_q, *g_Kc, *g_Vc, *g_Wk, *g_Wv, *g_out;
static int g_len, g_H, g_D, g_S; 
static float g_scale;
static void run_cache()     { launch_decode(g_q, g_Kc, g_Vc, g_out, g_len, g_H, g_D, g_S, g_scale); }
static void run_recompute() { launch_recompute(g_q, g_Kc, g_Vc, g_Wk, g_Wv, g_out, g_len, g_H, g_D, g_S, g_scale); }

int main() {
    const int H = 8, S = 1024, D = 64;
    const float scale = 1.0f / sqrtf((float)D);
    g_H = H; g_D = D; g_S = S; g_scale = scale;

    size_t cache_sz = (size_t)H * S * D * sizeof(float);
    cudaMalloc(&g_q,  H * D * sizeof(float));
    cudaMalloc(&g_Kc, cache_sz);   
    cudaMalloc(&g_Vc, cache_sz);
    cudaMalloc(&g_Wk, D * D * sizeof(float));  
    cudaMalloc(&g_Wv, D * D * sizeof(float));
    cudaMalloc(&g_out, H * D * sizeof(float));
    cudaMemset(g_q, 1, H * D * sizeof(float));        // 随便填,测速不看正确性
    cudaMemset(g_Kc, 1, cache_sz); 
    cudaMemset(g_Vc, 1, cache_sz);
    cudaMemset(g_Wk, 0, D * D * sizeof(float));

    printf("seq_len, cache_us, recompute_us, speedup\n");
    for (int len = 64; len <= S; len *= 2) {
        g_len = len;
        float c = time_step(run_cache,     20, 200) * 1e3f;   // 转微秒
        float r = time_step(run_recompute, 20, 200) * 1e3f;
        printf("%d, %.2f, %.2f, %.2fx\n", len, c, r, r / c);
    }
    return 0;
}