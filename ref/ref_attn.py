import numpy as np, os

# heads, 已缓存长度, head_dim
H, S, D = 8, 128, 64     
# 随机种子，保证结果可复现              
rng = np.random.default_rng(0)
# Q/K/V
q = rng.standard_normal((H, D)).astype(np.float32)
K = rng.standard_normal((H, S, D)).astype(np.float32)
V = rng.standard_normal((H, S, D)).astype(np.float32)
# Q*KT
score = np.einsum('hd,hsd->hs', q, K) / np.sqrt(D)
score -= score.max(-1, keepdims=True)
p = np.exp(score); p /= p.sum(-1, keepdims=True)
out = np.einsum('hs,hsd->hd', p, V).astype(np.float32)
os.makedirs('data', exist_ok=True)
for name, arr in [('q', q), ('K', K), ('V', V), ('out_ref', out)]:
    arr.tofile(f'data/{name}.bin')        # 裸 float32, C++ 端直接 fread
print(f'dumped test case  H={H} S={S} D={D}')