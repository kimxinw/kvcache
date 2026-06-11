cmake_minimum_required(VERSION 3.18)
project(kvcache LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)#vscode补全信息
set(CMAKE_CUDA_ARCHITECTURE 86)#GPU架构号

add_executable(paged src/main_paged.cu src/paged.cu)
target_include_directories(paged PRIVATE src/include)
target_compile_options(paged PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>)

#add_executable(bench src/bench.cu src/decode_attn.cu src/baseline.cu)
#target_include_directories(bench PRIVATE src/include)
#target_compile_options(bench PRIVATE
    #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>)

#add_executable(kvcache src/main.cu src src/kv_cache.cu src/decode_attn.cu)
#target_include_directories(kvcache PRIVATE src/include)