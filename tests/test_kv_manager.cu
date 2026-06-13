// test_kv_manager.cu —— KVCacheManager / Sequence 单元测试 (host 端逻辑 + 设备池分配)。
// 验证: prompt 分配 / decode 跨块借新块 / OOM 返回 false / free 归还 / 偏移口径。
// 编译: nvcc -std=c++17 -I src/include tests/test_kv_manager.cu -o build/test_kv_manager
#include "kv_cache_manager.h"
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, msg) do{ if(!(cond)){ printf("  [FAIL] %s\n", msg); g_fail++; } \
                             else        { printf("  [ ok ] %s\n", msg); } }while(0)

int main() {
    // 池: 8 个物理块, 每块 4 token, 2 个 kv head, head_dim=8。
    KVCacheManager kvm(/*num_blocks*/8, /*block_size*/4, /*n_kv_heads*/2, /*head_dim*/8);
    printf("pool_elems=%zu free_blocks=%d\n", kvm.pool_elems(), kvm.free_blocks());
    CHECK(kvm.free_blocks() == 8, "init free=8");

    // --- 序列 A: prompt_len=5 -> ceil(5/4)=2 块 ---
    Sequence a; a.id = 0; a.prompt_len = 5;
    CHECK(kvm.allocate_prompt(a), "A allocate_prompt ok");
    CHECK((int)a.block_table.size() == 2, "A has 2 blocks");
    CHECK(a.cur_len == 5, "A cur_len=5");
    CHECK(kvm.free_blocks() == 6, "free=6 after A");
    CHECK(kvm.slack(a) == 3, "A slack=3 (2*4-5)");

    // --- decode: 5->6,7,8 不该借块(容量8); 8->9 跨界借第3块 ---
    kvm.append_token(a); // 6
    kvm.append_token(a); // 7
    kvm.append_token(a); // 8
    CHECK((int)a.block_table.size() == 2, "A still 2 blocks at len=8");
    CHECK(kvm.slack(a) == 0, "A slack=0 at len=8");
    kvm.append_token(a); // 9 -> 借第3块
    CHECK((int)a.block_table.size() == 3, "A grew to 3 blocks at len=9");
    CHECK(a.cur_len == 9 && a.num_generated() == 4, "A len=9 generated=4");
    CHECK(kvm.free_blocks() == 5, "free=5 after A grew");

    // --- OOM: 再分配直到耗尽 ---
    Sequence b; b.id = 1; b.prompt_len = 20; // ceil(20/4)=5 块, 恰好用完剩余 5 块
    CHECK(kvm.allocate_prompt(b), "B allocate_prompt ok (uses last 5)");
    CHECK(kvm.free_blocks() == 0, "free=0, pool exhausted");
    Sequence c; c.id = 2; c.prompt_len = 1;
    CHECK(!kvm.allocate_prompt(c), "C allocate_prompt fails (OOM)");
    CHECK(c.block_table.empty(), "C left untouched on OOM");
    CHECK(!kvm.append_token(b), "B append fails (no free block to grow)");

    // --- free A -> 归还 3 块, free 应回到 3 ---
    kvm.free(a);
    CHECK(a.block_table.empty(), "A block_table cleared after free");
    CHECK(kvm.free_blocks() == 3, "free=3 after releasing A");
    CHECK(kvm.append_token(b), "B append now ok (blocks freed)");

    // --- 偏移口径自检: 与 batched_paged 的 bp_offset 一致 ---
    //   bp_offset(pos) = ((phys*H + h)*BLOCK + pos%BLOCK)*D, 这里 in_block=pos%BLOCK
    size_t off = kvm.slot_offset(/*phys*/3, /*kv_head*/1, /*in_block*/2);
    size_t expect = (((size_t)3*2 + 1)*4 + 2)*8; // = ((3*2+1)*4+2)*8
    CHECK(off == expect, "slot_offset matches pool layout");

    printf(g_fail ? "\n=== %d CHECK(S) FAILED ===\n" : "\n=== ALL CHECKS PASSED ===\n", g_fail);
    return g_fail ? 1 : 0;
}
