#include "kv_cache.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

static std::vector<float> load_bin(const char* path, int n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("!! 打不开 %s,先跑 python ref/ref_attn.py\n", path); exit(1); }
    fread(v.data(), sizeof(float), n, f); fclose(f);
    return v;
}

int main() {
    const int H = 8, S = 128, D = 64;     // 必须和 ref_attn.py 一致
    const float scale = 1.0f / sqrtf((float)D);

    auto q   = load_bin("data/q.bin",       H * D);
    auto K   = load_bin("data/K.bin",       H * S * D);
    auto V   = load_bin("data/V.bin",       H * S * D);
    auto ref = load_bin("data/out_ref.bin", H * D);

    float *dq, *dK, *dV, *dKc, *dVc, *dout, *dkt, *dvt;
    cudaMalloc(&dq,  H * D * sizeof(float));
    cudaMalloc(&dK,  H * S * D * sizeof(float));
    cudaMalloc(&dV,  H * S * D * sizeof(float));
    cudaMalloc(&dKc, H * S * D * sizeof(float));   // 缓存
    cudaMalloc(&dVc, H * S * D * sizeof(float));
    cudaMalloc(&dout, H * D * sizeof(float));
    cudaMalloc(&dkt, H * D * sizeof(float));          // 单 token 的 [H,D] 缓冲
    cudaMalloc(&dvt, H * D * sizeof(float));
    cudaMemcpy(dq, q.data(), H * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), H * S * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), H * S * D * sizeof(float), cudaMemcpyHostToDevice);

    // 用 append 把 S 个 token 逐个填进缓存(模拟 decode 一步步追加的过程)
    // for (int pos = 0; pos < S; ++pos)
    //     launch_append(dK + pos * D, dV + pos * D, dKc, dVc, pos, H, D, S);
    // 注意: dK 布局是 [H,S,D], 这里偏移取的是 head0 的第 pos 行。


    // 逐个 token 填缓存:先按 stride 抽出第 pos 个 token 的连续 [H,D],再 append。
    for (int pos = 0; pos < S; ++pos) {
        // 从 [H,S,D] 里取 token pos 的 [H,D];跨 head 间隔是 S*D 个 float(cudaMemcpy2D函数的拷贝间距参数)
        cudaMemcpy2D(dkt, D * sizeof(float),
                     dK + pos * D, S * D * sizeof(float),
                     D * sizeof(float), H, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(dvt, D * sizeof(float),
                     dV + pos * D, S * D * sizeof(float),
                     D * sizeof(float), H, cudaMemcpyDeviceToDevice);
        launch_append(dkt, dvt, dKc, dVc, pos, H, D, S);
    }

    launch_decode(dq, dKc, dVc, dout, S, H, D, S, scale);

    std::vector<float> out(H * D);
    cudaMemcpy(out.data(), dout, H * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();

    float maxdiff = 0.f;
    for (int i = 0; i < H * D; ++i) maxdiff = fmaxf(maxdiff, fabsf(out[i] - ref[i]));
    printf("cuda: %s | max abs diff = %.6f -> %s\n",
           cudaGetErrorString(err), maxdiff, maxdiff < 1e-3f ? "PASS ✅" : "FAIL ❌");
    return 0;
}
