#!/usr/bin/env python
# dump_weights.py —— 把 Qwen2.5-0.5B-Instruct 的权重以 fp32、row-major 顺序铺进
#   一个扁平 weights.bin，并生成 manifest.json(name/shape/offset_bytes/nbytes)。
# C++ 引擎按 name 查 manifest，读 blob，上传 GPU。
#
#   - 全程 fp32：cuBLAS sgemm 最简单，且与 fp32 HF oracle 对拍数值最干净。
#   - PyTorch Linear.weight 形状是 [out, in]，y = x @ W^T (+b)。这里原样存 [out,in]
#     row-major，转置约定留给 C++ 的 linear 帮手处理。
import os, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qwen05b")
os.makedirs(OUT_DIR, exist_ok=True)

print("loading model (fp32, cpu)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
sd = model.state_dict()

# tie_word_embeddings=true：lm_head.weight 可能不在 state_dict，复用 embed。
if "lm_head.weight" not in sd:
    sd = dict(sd)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]

bin_path = os.path.join(OUT_DIR, "weights.bin")
manifest = {"model": MODEL, "dtype": "float32", "tensors": []}
offset = 0
with open(bin_path, "wb") as f:
    for name, t in sd.items():
        arr = t.detach().contiguous().to(torch.float32).numpy()
        assert arr.dtype == np.float32
        b = arr.tobytes()
        f.write(b)
        manifest["tensors"].append({
            "name": name,
            "shape": list(arr.shape),
            "offset": offset,
            "nbytes": len(b),
        })
        offset += len(b)

manifest["total_bytes"] = offset
cfg = model.config
manifest["config"] = {
    "hidden_size": cfg.hidden_size,
    "num_hidden_layers": cfg.num_hidden_layers,
    "num_attention_heads": cfg.num_attention_heads,
    "num_key_value_heads": cfg.num_key_value_heads,
    "head_dim": cfg.hidden_size // cfg.num_attention_heads,
    "intermediate_size": cfg.intermediate_size,
    "vocab_size": cfg.vocab_size,
    "rms_norm_eps": cfg.rms_norm_eps,
    "rope_theta": cfg.rope_theta,
    "tie_word_embeddings": cfg.tie_word_embeddings,
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"wrote {bin_path} ({offset/1e6:.1f} MB), {len(manifest['tensors'])} tensors", flush=True)
print("config:", json.dumps(manifest["config"]), flush=True)
