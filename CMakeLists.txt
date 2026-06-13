cmake_minimum_required(VERSION 3.18)
project(kvcache LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)          # vscode 补全信息

# 默认 GPU 架构 (RTX 3060 = sm_86)；可用 -DCMAKE_CUDA_ARCHITECTURES=xx 覆盖
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ---- paged: 分页 attention 正确性测试 (identity / SHUFFLE 两种 block table) ----
add_executable(paged src/main_paged.cu src/paged.cu)
target_include_directories(paged PRIVATE src/include)

# ---- kvcache: 连续 KV cache decode 正确性测试 ----
add_executable(kvcache src/main.cu src/kv_cache.cu src/decode_attn.cu)
target_include_directories(kvcache PRIVATE src/include)

# ---- bench: 连续 cache vs 重算 (为什么需要 KV cache) ----
add_executable(bench src/bench.cu src/decode_attn.cu src/baseline.cu)
target_include_directories(bench PRIVATE src/include)

# ---- bench_paged: 连续 decode vs 分页 decode (单序列延迟 + 正确性) ----
add_executable(bench_paged src/bench_paged.cu src/decode_attn.cu src/paged.cu)
target_include_directories(bench_paged PRIVATE src/include)

# ---- bench_throughput: 多序列变长压测 (显存 + 吞吐, paged 的真正价值) ----
add_executable(bench_throughput src/bench_throughput.cu src/batched.cu)
target_include_directories(bench_throughput PRIVATE src/include)

# ---- flash: 在线 softmax + split-K decode 正确性测试 ----
add_executable(flash src/main_flash.cu src/flash_decode.cu)
target_include_directories(flash PRIVATE src/include)

# ---- continuous: continuous vs static batching 调度器对比 (吞吐 / 延迟 / slot 利用率) ----
add_executable(continuous src/bench_continuous.cu src/batched.cu)
target_include_directories(continuous PRIVATE src/include)
