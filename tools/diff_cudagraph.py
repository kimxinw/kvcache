#!/usr/bin/env python
# diff_cudagraph.py —— CUDA graph 对拍回归。
#   同一 prompt 下 qwen_infer(定义 CUDAG，graph capture/replay)与
#   qwen_infer_eager(无 CUDAG，eager 逐 token forward)生成的 token id 必须逐位相同。
#   decode 是纯 argmax 贪心、确定性 → 任何分叉 = graph 路径有 bug。
#   用法：python tools/diff_cudagraph.py        (退出码 0=全过, 1=有分叉)
import os, sys, subprocess
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 必须在 import transformers 之前：
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  #   huggingface_hub 在 import 时即读取并缓存该值
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HERE  = os.path.dirname(os.path.abspath(__file__))
DIR   = os.path.join(HERE, "..", "data", "qwen05b")
BIN_G = os.path.join(HERE, "..", "build", "qwen_infer")        # graph 版
BIN_E = os.path.join(HERE, "..", "build", "qwen_infer_eager")  # eager 基线
tok = AutoTokenizer.from_pretrained(MODEL)

# (prompt, max_new)；max_new 取大些，多覆盖 decode/重放步，更易暴露 pos 累积错位
CASES = [
    ("Give me a short introduction to large language models.", 64),
    ("What is the capital of France?",                         48),
    ("Explain recursion in one sentence.",                     48),
    ("Write a haiku about the ocean.",                         64),
]

def encode(p):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors=None)["input_ids"]

def is_ids(l):                       # 纯整数行 = 生成 id 行，跳过 [loader] 等日志
    p = l.split()
    return bool(p) and all(t.lstrip("-").isdigit() for t in p)

def run(bin_path, ids, max_new):
    stdin = " ".join(map(str, ids)) + "\n"
    out = subprocess.run([bin_path, DIR, str(max_new)], input=stdin,
                         capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if is_ids(l)]
    if not lines:
        sys.stderr.write(out.stderr)
        raise RuntimeError(f"{bin_path}: 未解析到 token id 行 (rc={out.returncode})")
    return list(map(int, lines[-1].split()))

all_ok = True
for p, mn in CASES:
    ids = encode(p)
    g = run(BIN_G, ids, mn)
    e = run(BIN_E, ids, mn)
    ok = (g == e)
    all_ok &= ok
    print("=" * 70)
    print(f"[PROMPT]  {p}  (max_new={mn})")
    print(f"[对拍]    {'MATCH ✓' if ok else 'MISMATCH ✗'}   graph={len(g)} eager={len(e)} tok")
    if not ok:
        n = min(len(g), len(e))
        i = next((k for k in range(n) if g[k] != e[k]), n)   # 首个分叉位置
        gi = g[i] if i < len(g) else "<eof>"
        ei = e[i] if i < len(e) else "<eof>"
        print(f"  首个分叉 @ idx {i}:  graph={gi}  eager={ei}")
        print(f"  graph: {g}")
        print(f"  eager: {e}")

print("=" * 70)
print("CUDA graph 对拍总判:", "ALL PASS ✅" if all_ok else "FAIL ❌")
sys.exit(0 if all_ok else 1)
