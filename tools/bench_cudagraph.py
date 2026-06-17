#!/usr/bin/env python
# bench_cudagraph.py —— CUDA graph 的 decode launch-overhead 性能对比。
#   diff_cudagraph.py 只验证生成 token 的正确性；这里量 graph 真正的卖点：
#   batch=1 decode 下，graph 把每步 ~360 次 kernel launch 收成 1 次 graphLaunch，
#   消掉 CPU 关键路径上的 launch 开销 → per-token 延迟下降。
#   两个 binary 各自在稳态(丢 warmup)打印 [bench] 计时行，本脚本解析并对比。
#   用法：python tools/bench_cudagraph.py
import os, re, sys, subprocess
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 必须在 import transformers 之前：
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  #   huggingface_hub 在 import 时即读取并缓存该值
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HERE  = os.path.dirname(os.path.abspath(__file__))
DIR   = os.path.join(HERE, "..", "data", "qwen05b")
BIN_G = os.path.join(HERE, "..", "build", "qwen_infer")        # graph 版
BIN_E = os.path.join(HERE, "..", "build", "qwen_infer_eager")  # eager 基线
tok = AutoTokenizer.from_pretrained(MODEL)

PROMPT  = "Give me a short introduction to large language models."
MAX_NEW = 256       # 取大，让稳态步数足够、中位数稳定

# 解析 binary 打到 stderr 的：[bench] graph steps=.. median=XX us/tok mean=.. YY tok/s
BENCH_RE = re.compile(r"median=([\d.]+) us/tok.*?([\d.]+) tok/s")

def encode(p):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors=None)["input_ids"]

def run(bin_path, ids, max_new):
    stdin = " ".join(map(str, ids)) + "\n"
    out = subprocess.run([bin_path, DIR, str(max_new)], input=stdin,
                         capture_output=True, text=True)
    m = BENCH_RE.search(out.stderr)
    if not m:
        sys.stderr.write(out.stderr)
        raise RuntimeError(f"{bin_path}: 未解析到 [bench] 计时行 (rc={out.returncode})")
    return float(m.group(1)), float(m.group(2))   # (us/tok, tok/s)

ids = encode(PROMPT)
print(f"[prompt] {PROMPT!r}  prompt_len={len(ids)}  max_new={MAX_NEW}")
e_us, e_tps = run(BIN_E, ids, MAX_NEW)
g_us, g_tps = run(BIN_G, ids, MAX_NEW)

print("=" * 60)
print(f"{'mode':<8}{'us/tok':>14}{'tok/s':>12}")
print(f"{'eager':<8}{e_us:>14.1f}{e_tps:>12.1f}")
print(f"{'graph':<8}{g_us:>14.1f}{g_tps:>12.1f}")
print("-" * 60)
print(f"speedup  {e_us / g_us:>13.2f}x   省 {e_us - g_us:.1f} us/tok "
      f"(消掉每步 ~360 次 kernel launch 的 CPU 开销)")
print("=" * 60)
