#!/usr/bin/env python
# run_serve.py —— 里程碑2 驱动：把多条 chat prompt 编码成 token id 喂给 qwen_serve，
#   读回每条序列生成的 id 并解码成文本。并做一条正确性校验：
#   含与 ref_tokens.json 相同 prompt 的那条，batch 下贪心结果必须 == 单序列 ref(贪心序列独立)。
import os, sys, json, subprocess
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qwen05b")
BIN = os.path.join(os.path.dirname(__file__), "..", "build", "qwen_serve")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
tok = AutoTokenizer.from_pretrained(MODEL)

PROMPTS = [
    "Give me a short introduction to large language models.",   # 与 ref 同，用于校验
    "Write a haiku about the ocean.",
    "What is the capital of France?",
    "Explain recursion in one sentence.",
]
MAX_NEW = [32, 24, 16, 28]

def encode(p):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors=None)["input_ids"]

reqs = [encode(p) for p in PROMPTS]
stdin = f"{len(reqs)}\n" + "\n".join(
    f"{MAX_NEW[i]} " + " ".join(map(str, ids)) for i, ids in enumerate(reqs)) + "\n"

out = subprocess.run([BIN, DIR], input=stdin, capture_output=True, text=True)
sys.stderr.write(out.stderr)
def is_ids(l):
    p = l.split()
    return p and all(t.lstrip("-").isdigit() for t in p)
lines = [l for l in out.stdout.splitlines() if is_ids(l)]
gen = [list(map(int, l.split())) for l in lines]

print("=" * 70)
for p, g in zip(PROMPTS, gen):
    print(f"[PROMPT] {p}")
    print(f"[GEN]    {tok.decode(g)}\n")

# 正确性校验：第 0 条 == ref gen_ids
ref = json.load(open(os.path.join(DIR, "ref_tokens.json")))["gen_ids"]
ok = gen[0] == ref
print("=" * 70)
print("正确性校验 (batch 第0条 vs 单序列 HF ref):", "PASS ✅" if ok else "FAIL ❌")
if not ok:
    print(" ref:", ref)
    print(" got:", gen[0])
