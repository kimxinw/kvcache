#pragma once
// qwen_loader.h —— 读 manifest.tsv + weights.bin，把整块 fp16 权重一次性传上 GPU。
//   每个 tensor 的设备指针 = (char*)d_base + offset_bytes。按 HF 原名查询。
//   (embed 与 lm_head 因 tie 重复存了一份，fp16 整块 ~1.26GB，3060 12G 放得下。)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

#ifndef QW_CK
#define QW_CK(x) do{ cudaError_t e=(x); if(e){ \
    printf("CUDA err %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)
#endif

struct Tensor { size_t off_bytes; std::vector<int> shape; };

struct QwenWeights {
    half* d_base = nullptr;                 // 整块权重 (设备)
    std::unordered_map<std::string, Tensor> tmap;

    void load(const std::string& dir) {
        std::string tsv = dir + "/manifest.tsv";
        std::string bin = dir + "/weights.bin";
        // 1) 解析 manifest.tsv
        std::ifstream mf(tsv);
        if (!mf) { printf("cannot open %s\n", tsv.c_str()); exit(1); }
        std::string line;
        size_t max_end = 0;
        while (std::getline(mf, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string name, shape_csv; size_t off, nb;
            std::getline(ss, name, '\t');
            ss >> off; ss.ignore(1); ss >> nb; ss.ignore(1);
            std::getline(ss, shape_csv);
            Tensor t; t.off_bytes = off;
            std::stringstream sc(shape_csv); std::string d;
            while (std::getline(sc, d, ',')) if (!d.empty()) t.shape.push_back(std::stoi(d));
            tmap[name] = t;
            max_end = std::max(max_end, off + nb);
        }
        // 2) 读 weights.bin 到 host，再整体上 GPU
        std::ifstream bf(bin, std::ios::binary | std::ios::ate);
        if (!bf) { printf("cannot open %s\n", bin.c_str()); exit(1); }
        size_t bytes = (size_t)bf.tellg();
        if (bytes < max_end) { printf("weights.bin too small: %zu < %zu\n", bytes, max_end); exit(1); }
        bf.seekg(0);
        std::vector<char> host(bytes);
        bf.read(host.data(), bytes);
        QW_CK(cudaMalloc((void**)&d_base, bytes));
        QW_CK(cudaMemcpy(d_base, host.data(), bytes, cudaMemcpyHostToDevice));
        printf("[loader] uploaded %.1f MB, %zu tensors\n", bytes/1e6, tmap.size());
    }

    const Tensor& info(const std::string& name) const {
        auto it = tmap.find(name);
        if (it == tmap.end()) { printf("missing tensor %s\n", name.c_str()); exit(1); }
        return it->second;
    }
    // off_bytes 是字节偏移；用 char* 推进再转回 half*，不依赖元素大小。
    half* ptr(const std::string& name) const {
        return reinterpret_cast<half*>(reinterpret_cast<char*>(d_base) + info(name).off_bytes);
    }
    // 便捷：层 ℓ 的某权重
    half* lp(int l, const char* suffix) const {
        char buf[128]; snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, suffix);
        return ptr(buf);
    }
    ~QwenWeights() { if (d_base) cudaFree(d_base); }
};
