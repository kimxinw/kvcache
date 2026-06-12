#include "kv_cache.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// flash_decode 正确性测试：在线 softmax + split-K，对拍 ref/ref_attn.py 的 out_ref.bin。
// 用多个 num_splits（整除 / 不整除 / 极端）验证 split-K 的 reduce 在各种切法下都对。

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

    float *dq, *dK, *dV, *dout;
    cudaMalloc(&dq,  H * D * sizeof(float));
    cudaMalloc(&dK,  H * S * D * sizeof(float));
    cudaMalloc(&dV,  H * S * D * sizeof(float));
    cudaMalloc(&dout, H * D * sizeof(float));
    cudaMemcpy(dq, q.data(), H * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), H * S * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), H * S * D * sizeof(float), cudaMemcpyHostToDevice);

    int splits[] = {1, 4, 7, 16, 128};   // 1=纯在线softmax; 7不整除128; 128=每段1个token
    int all_pass = 1;
    for (int ns : splits) {
        launch_flash_decode(dq, dK, dV, dout, S, H, D, S, scale, ns);
        cudaDeviceSynchronize();

        std::vector<float> out(H * D);
        cudaMemcpy(out.data(), dout, H * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();

        float maxdiff = 0.f;
        for (int i = 0; i < H * D; ++i) maxdiff = fmaxf(maxdiff, fabsf(out[i] - ref[i]));
        int pass = (err == cudaSuccess) && (maxdiff < 1e-3f);
        all_pass &= pass;
        printf("num_splits=%3d | %s | max abs diff = %.6f -> %s\n",
               ns, cudaGetErrorString(err), maxdiff, pass ? "PASS ✅" : "FAIL ❌");
    }
    printf("---- flash_decode %s ----\n", all_pass ? "ALL PASS ✅" : "SOME FAIL ❌");
    return 0;
}
