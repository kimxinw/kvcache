#pragma once
// qwen_kernels.h —— Qwen2 前向里手写 CUDA 的那部分(项目亮点)：
//   embed / RMSNorm / RoPE / SwiGLU / 残差 / 写 KV 入分页池 / GQA 因果分页 attention / argmax。
//   线性层(QKV/O/MLP/lm_head) 走 cuBLAS，不在这里。
void qk_embed_gather(float* out, const float* embed, const int* ids, int L, int H);
void qk_rmsnorm(float* out, const float* in, const float* w, int L, int H, float eps);
void qk_add_bias(float* x, const float* bias, int L, int N);
void qk_add_residual(float* x, const float* y, int n);
void qk_rope(float* x, int L, int n_heads, int D, int pos_base, float theta);
void qk_silu_mul(float* out, const float* gate, const float* up, int n);
// 把 prefill/decode 算出的 K/V (已过 RoPE 的 K) 散写进分页池 [num_blocks, Hkv, BLOCK, D]。
void qk_write_kv(const float* k, const float* v, float* k_pool, float* v_pool,
                 const int* block_table_dev, int start_pos, int L, int Hkv, int D, int BLOCK);
// GQA 因果分页 attention：q[Nq,Hq,D] 第 qi 个查询位于绝对位置 (q_pos_base+qi)，
//   因果地注意池中 [0, q_pos_base+qi]；kv_head = h / (Hq/Hkv)。flash 在线 softmax。
void qk_paged_causal_attn(const float* q, float* out, const float* k_pool, const float* v_pool,
                          const int* block_table_dev, int q_pos_base,
                          int Nq, int Hq, int Hkv, int D, int BLOCK, float scale);
// argmax over logits[n] -> *d_idx (设备端单 int)。
void qk_argmax(const float* logits, int n, int* d_idx);

// ===== 里程碑2 批量(多序列)版本：continuous batching 的 decode 步 =====
// 逐行位置的 RoPE：x[n_rows*n_heads, D]，第 row 行的位置由 pos[row] 给(各序列位置不同)。
void qk_rope_pos(float* x, int n_rows, int n_heads, int D, const int* pos, float theta);
// 批量写 KV：k,v 是 [N,Hkv,D]，每序列一个新 token。第 r 行的块表 = table_all+tab_off[r]，
//   写入位置 wpos[r]。池为单层切片 [num_blocks,Hkv,BLOCK,D]。
void qk_batched_write_kv(const float* k, const float* v, float* k_pool, float* v_pool,
                         const int* table_all, const int* tab_off, const int* wpos,
                         int N, int Hkv, int D, int BLOCK);
// 批量 GQA 分页 decode attention：q[N,Hq,D] 每序列一个查询，注意各自 [0,cur_len[s]) 的页。
//   kv_head=h/(Hq/Hkv)。flash 在线 softmax。池为单层切片。
void qk_batched_paged_decode_attn(const float* q, float* out, const float* k_pool, const float* v_pool,
                                  const int* table_all, const int* tab_off, const int* cur_len,
                                  int N, int Hq, int Hkv, int D, int BLOCK, float scale);
