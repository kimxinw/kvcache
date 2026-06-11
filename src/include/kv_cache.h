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