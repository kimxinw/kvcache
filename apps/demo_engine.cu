// demo_engine.cu —— 第④步骨架验证: 用新核心 (Engine -> ModelRunner -> KVCacheManager
//   -> launch_batched_paged) 跑一批合成请求的端到端离线 continuous batching。
//
// 这里只验证【调度+内存+执行主干能跑通且账目对得上】:
//   生成的总 token 数 == Σ max_new_tokens, 完成请求数 == 请求总数, 峰值块不超池容量。
// attention 数值正确性已由 ref 对拍证明, 故 q 填 0 即可。
//
// 编译: nvcc -std=c++17 -I src/include apps/demo_engine.cu src/batched.cu -o build/demo_engine
#include "engine.h"
#include <cstdio>
#include <cstdlib>

int main() {
    EngineConfig cfg{ /*n_heads*/8, /*head_dim*/64, /*block_size*/16,
                      /*num_blocks*/512, /*max_batch*/32 };
    Engine eng(cfg);

    srand(123);
    const int NREQ = 256;
    long long expected = 0;
    int max_prompt = 0;
    for (int i = 0; i < NREQ; ++i) {
        int prompt = 8  + rand() % 120;    // 8..127
        int gen    = 16 + rand() % 200;    // 16..215
        eng.add_request(prompt, gen);
        expected += gen;
        if (prompt + gen > max_prompt) max_prompt = prompt + gen;
    }

    eng.run();
    const EngineStats& st = eng.stats;
    double util = st.iters ? (double)st.tokens / st.slot_steps : 0.0;
    printf("NREQ=%d  iters=%d  finished=%d\n", NREQ, st.iters, st.finished);
    printf("tokens=%lld (expected %lld)\n", st.tokens, expected);
    printf("gpu_us=%.0f  avg_step=%.1f us  slot_util=%.1f%%\n",
           st.gpu_us, st.iters ? st.gpu_us / st.iters : 0.0, util * 100);
    printf("peak_blocks=%d / %d  (max worstcase footprint per seq ~ %d blocks)\n",
           st.peak_blocks, cfg.num_blocks, (max_prompt + cfg.block_size - 1) / cfg.block_size);

    bool ok = (st.tokens == expected) && (st.finished == NREQ)
              && (st.peak_blocks <= cfg.num_blocks);
    printf(ok ? "\n=== ENGINE SKELETON OK ===\n" : "\n=== ENGINE MISMATCH ===\n");
    return ok ? 0 : 1;
}
