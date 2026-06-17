#pragma once
#include<cuda_runtime.h>
#include<cuda_fp16.h>
// qwen_kernels.h —— Qwen2 前向里手写 CUDA 的那部分(项目亮点)：
//   embed / RMSNorm / RoPE / SwiGLU / 残差 / 写 KV 入分页池 / GQA 因果分页 attention / argmax。
//   线性层(QKV/O/MLP/lm_head) 走 cuBLAS，不在这里。
//   激活/权重/KV 池全程 fp16 (half)，kernel 内部 fp32 累加；argmax 读 lm_head 的 fp32 logits。
void qk_embed_gather(half* out, const half* embed, const int* ids, int L, int H,cudaStream_t stream = 0);
void qk_rmsnorm(half* out, const half* in, const half* w, int L, int H, float eps,cudaStream_t stream = 0);
void qk_add_bias(half* x, const half* bias, int L, int N,cudaStream_t stream = 0);
void qk_add_residual(half* x, const half* y, int n,cudaStream_t stream = 0);
void qk_rope(half* x, int L, int n_heads, int D, int pos_base, float theta,cudaStream_t stream = 0);
void qk_silu_mul(half* out, const half* gate, const half* up, int n,cudaStream_t stream = 0);
// 把 prefill/decode 算出的 K/V (已过 RoPE 的 K) 散写进分页池 [num_blocks, Hkv, BLOCK, D]。
void qk_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                 const int* block_table_dev, int start_pos, int L, int Hkv, int D, int BLOCK,cudaStream_t stream = 0);
// GQA 因果分页 attention：q[Nq,Hq,D] 第 qi 个查询位于绝对位置 (q_pos_base+qi)，
//   因果地注意池中 [0, q_pos_base+qi]；kv_head = h / (Hq/Hkv)。flash 在线 softmax。
void qk_paged_causal_attn(const half* q, half* out, const half* k_pool, const half* v_pool,
                          const int* block_table_dev, int q_pos_base,
                          int Nq, int Hq, int Hkv, int D, int BLOCK, float scale,cudaStream_t stream = 0);
// argmax over logits[n] -> *d_idx (设备端单 int)。logits 是 fp32。
void qk_argmax(const float* logits, int n, int* d_idx,cudaStream_t stream = 0);

// ===== 里程碑2 批量(多序列)版本：continuous batching 的 decode 步 =====
// 逐行位置的 RoPE：x[n_rows*n_heads, D]，第 row 行的位置由 pos[row] 给(各序列位置不同)。
void qk_rope_pos(half* x, int n_rows, int n_heads, int D, const int* pos, float theta);
// 批量写 KV：k,v 是 [N,Hkv,D]，每序列一个新 token。第 r 行的块表 = table_all+tab_off[r]，
//   写入位置 wpos[r]。池为单层切片 [num_blocks,Hkv,BLOCK,D]。
void qk_batched_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                         const int* table_all, const int* tab_off, const int* wpos,
                         int N, int Hkv, int D, int BLOCK);
// 批量 GQA 分页 decode attention：q[N,Hq,D] 每序列一个查询，注意各自 [0,cur_len[s]) 的页。
//   kv_head=h/(Hq/Hkv)。flash 在线 softmax。池为单层切片。
void qk_batched_paged_decode_attn(const half* q, half* out, const half* k_pool, const half* v_pool,
                                  const int* table_all, const int* tab_off, const int* cur_len,
                                  int N, int Hq, int Hkv, int D, int BLOCK, float scale);

//CUDA graph专属kernels函数
//greph是录一次，一直用，传入指针地址，每次循环修改指针指向地址的内容
void qkcuda_rope(half* x, int L, int n_heads, int D, int* pos_base, float theta,cudaStream_t stream = 0);
void qkcuda_write_kv(const half* k, const half* v, half* k_pool, half* v_pool,
                 const int* block_table_dev, int* start_pos, int L, int Hkv, int D, int BLOCK,cudaStream_t stream = 0);
void qkcuda_paged_causal_attn(const half* q, half* out, const half* k_pool, const half* v_pool,
                          const int* block_table_dev, int* q_pos_base,
                          int Nq, int Hq, int Hkv, int D, int BLOCK, float scale,cudaStream_t stream = 0);
