#!/usr/bin/env python
# ref_qwen.py —— HF 参考 oracle。两个用途：
#   (1) token 级：固定 prompt 经 chat template，greedy decode，dump prompt/gen token ids。
#       —— 里程碑1 验收：C++ 引擎逐 token argmax 必须与这里完全一致。
#   (2) tensor 级：对 prompt 跑一次 prefill，dump 逐层 hidden_states + 末位 logits。
#       —— C++ 分层对拍：argmax 偏了就二分查是哪一层先崩。
import os, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qwen05b")
os.makedirs(OUT_DIR, exist_ok=True)
PROMPT = "Give me a short introduction to large language models."
MAX_NEW = 32

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": PROMPT},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt")
prompt_ids = enc.input_ids[0].tolist()
print("prompt text:\n", text)
print("prompt_ids (%d):" % len(prompt_ids), prompt_ids)

# (1) greedy decode —— 这就是 C++ 要复现的金标准 token 序列
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                         num_beams=1, repetition_penalty=1.0)
gen_ids = out[0].tolist()[len(prompt_ids):]
print("gen_ids (%d):" % len(gen_ids), gen_ids)
print("gen text:", tok.decode(gen_ids))

with open(os.path.join(OUT_DIR, "ref_tokens.json"), "w") as f:
    json.dump({"prompt": PROMPT, "prompt_ids": prompt_ids, "gen_ids": gen_ids,
               "eos_token_id": tok.eos_token_id}, f, indent=2)

# (2) prefill 逐层 hidden + 末位 logits
with torch.no_grad():
    o = model(**enc, output_hidden_states=True)
hs = torch.stack(o.hidden_states, dim=0)[:, 0]   # [L+1, seq, hidden]
logits_last = o.logits[0, -1]                    # [vocab]
np.savez(os.path.join(OUT_DIR, "ref_hidden.npz"),
         hidden=hs.float().numpy(),
         logits_last=logits_last.float().numpy(),
         prompt_ids=np.array(prompt_ids, dtype=np.int64),
         next_id=int(logits_last.argmax()))
print("prefill next argmax:", int(logits_last.argmax()),
      "==", gen_ids[0], "?", int(logits_last.argmax()) == gen_ids[0])
print("wrote ref_tokens.json, ref_hidden.npz to", OUT_DIR)
