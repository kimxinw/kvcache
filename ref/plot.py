#!/usr/bin/env python3
# 画三组图:
#   1) data/bench.png           连续 KV cache vs 重算 (为什么要 cache)   [若 data/bench.csv 存在]
#   2) data/bench_paged.png     连续 decode vs 分页 decode 单序列延迟    [data/bench_paged.csv]
#   3) data/paged_vs_contig.png 多序列变长: 显存 / 并发 / 吞吐对比       [data/throughput_summary.csv]
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    """读 CSV, 跳过 # 注释行, 表头/字段去空格。"""
    rows, header = [], None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if header is None:
            header = parts
        else:
            rows.append(dict(zip(header, parts)))
    return rows


# ---------- 1) 连续 cache vs 重算 ----------
if os.path.exists("data/bench.csv"):
    rows = read_rows("data/bench.csv")
    x = [int(r["seq_len"]) for r in rows]
    c = [float(r["cache_us"]) for r in rows]
    r_ = [float(r["recompute_us"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(x, c, "o-", label="KV cache  O(t)")
    a1.plot(x, r_, "s-", label="recompute  O(t²)")
    a1.set(xlabel="sequence length", ylabel="per-step latency (µs)", title="decode cost")
    a1.legend()
    a2.plot(x, [ri / ci for ri, ci in zip(r_, c)], "d-", color="C2")
    a2.set(xlabel="sequence length", ylabel="speedup ×", title="KV cache speedup")
    fig.tight_layout(); fig.savefig("data/bench.png", dpi=130)
    print("saved data/bench.png")


# ---------- 2) 连续 decode vs 分页 decode 单序列延迟 ----------
if os.path.exists("data/bench_paged.csv"):
    rows = read_rows("data/bench_paged.csv")
    x  = [int(r["seq_len"]) for r in rows]
    dc = [float(r["decode_us"]) for r in rows]
    pg = [float(r["paged_us"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(x, dc, "o-", label="contiguous decode")
    a1.plot(x, pg, "s-", label="paged decode")
    a1.set(xlabel="sequence length", ylabel="per-step latency (µs)",
           title="single-seq latency", xscale="log", yscale="log")
    a1.legend()
    a2.plot(x, [(p / d - 1) * 100 for p, d in zip(pg, dc)], "d-", color="C3")
    a2.set(xlabel="sequence length", ylabel="paged overhead (%)",
           title="cost of block-table indirection", xscale="log")
    fig.tight_layout(); fig.savefig("data/bench_paged.png", dpi=130)
    print("saved data/bench_paged.png")


# ---------- 3) 多序列变长: 显存 / 并发 / 吞吐 ----------
if os.path.exists("data/throughput_summary.csv"):
    m = {r["metric"]: r for r in read_rows("data/throughput_summary.csv")}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.2))

    # 显存: 连续(预留 vs 真用) 对比 paged(只占真用)
    cont_res = float(m["reserved_MB"]["contiguous"]); cont_use = float(m["used_MB"]["contiguous"])
    pg_use   = float(m["reserved_MB"]["paged"])
    a1.bar(["contiguous\nreserved", "contiguous\nactually used", "paged\nused"],
           [cont_res, cont_use, pg_use], color=["#c0392b", "#e59866", "#27ae60"])
    a1.set(ylabel="KV memory (MB)", title=f"memory: paged saves {cont_res/pg_use:.1f}×")
    for i, v in enumerate([cont_res, cont_use, pg_use]):
        a1.text(i, v, f"{v:.0f}", ha="center", va="bottom")

    # 固定预算下可并发序列数
    mc = int(m["max_seqs_budget"]["contiguous"]); mp = int(m["max_seqs_budget"]["paged"])
    a2.bar(["contiguous", "paged"], [mc, mp], color=["#c0392b", "#27ae60"])
    a2.set(ylabel="max concurrent seqs", title=f"fixed budget: paged fits {mp/mc:.1f}× more")
    for i, v in enumerate([mc, mp]):
        a2.text(i, v, str(v), ha="center", va="bottom")

    # 同 batch 下吞吐 (paged 略低 = 间接寻址代价)
    tc = float(m["tokens_per_s"]["contiguous"]); tp = float(m["tokens_per_s"]["paged"])
    a3.bar(["contiguous", "paged"], [tc, tp], color=["#c0392b", "#27ae60"])
    a3.set(ylabel="tokens / s (same batch)", title=f"throughput: paged = {tp/tc:.2f}×")
    for i, v in enumerate([tc, tp]):
        a3.text(i, v, f"{v:.0f}", ha="center", va="bottom")

    fig.tight_layout(); fig.savefig("data/paged_vs_contig.png", dpi=130)
    print("saved data/paged_vs_contig.png")
