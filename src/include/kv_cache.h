#pragma once
void launch_append(const float* k_new, const float* v_new,
                   float* k_cache, float* v_cache,
                   int pos, int H, int D, int S);
void launch_decode(const float* q, const float* k_cache, const float* v_cache,
                   float* out, int cur_len, int H, int D, int S, float scale);
void launch_recompute(const float* q,const float* k_cache,const float* v_cache,const float* Wk, const float* Wv,
                      float* out, int cur_len, int H, int D, int S, float scale);
void launch_paged(const float* q, const float* k_pool, const float* v_pool,
                  const int* block_table, float* out, int cur_len,
                  int H, int D, int BLOCK, float scale);

// ---- 批量 (多序列) decode：grid = (N 个序列, H 个 head) ----
// 连续布局: 每条序列预留 [H, MAXS, D]; 整块 cache = [N, H, MAXS, D]; cur_len[N] 给每条真实长度。
void launch_batched_decode(const float* q, const float* k_cache, const float* v_cache,
                           const int* cur_len, float* out,
                           int N, int H, int D, int MAXS, int cap, float scale);
// 分页布局: 共享物理池 [num_blocks, H, BLOCK, D]; 所有序列的 block_table 拼接在 table_all 里,
//   第 s 条从 tab_off[s] 开始; cur_len[s] 给真实长度; cap = 最长序列(定 shared mem 上界)。
void launch_batched_paged(const float* q, const float* k_pool, const float* v_pool,
                          const int* table_all, const int* tab_off, const int* cur_len,
                          float* out, int N, int H, int D, int BLOCK, int cap, float scale);

// ---- flash decode: 在线 softmax (running max/sum) + split-K ----
// 单序列连续布局 q[H,D] / k_cache,v_cache[H,S,D] / out[H,D]，与 launch_decode 同口径。
// num_splits: 把长度 cur_len 的 KV 切成几段并行；段间用第二个 reduce kernel 合并。
void launch_flash_decode(const float* q, const float* k_cache, const float* v_cache,
                         float* out, int cur_len, int H, int D, int S, float scale,
                         int num_splits);
