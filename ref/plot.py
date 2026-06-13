#!/usr/bin/env python3
# 画四组图:
#   1) data/bench.png            连续 KV cache vs 重算 (为什么要 cache)  [若 data/bench.csv 存在]
#   2) data/bench_paged.png      连续 decode vs 分页 decode 单序列延迟   [data/bench_paged.csv]
#   3) data/paged_vs_contig.png  多序列变长: 显存 / 并发 / 吞吐对比      [data/throughput_summary.csv]
#   4) data/continuous_vs_static.png  continuous vs static batching     [data/continuous_summary.csv]
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


# ---------- 4) continuous vs static batching ----------
if os.path.exists("data/continuous_summary.csv"):
    m = {r["metric"]: r for r in read_rows("data/continuous_summary.csv")}
    S, C = "#c0392b", "#27ae60"          # static 红 / continuous 绿
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.2))

    # 吞吐 (tokens/s) —— continuous 随完随补, 迭代数更少
    ts = float(m["tok_per_s"]["static"]); tc = float(m["tok_per_s"]["continuous"])
    a1.bar(["static", "continuous"], [ts, tc], color=[S, C])
    a1.set(ylabel="tokens / s", title=f"throughput: continuous = {tc/ts:.2f}×")
    for i, v in enumerate([ts, tc]):
        a1.text(i, v, f"{v:.0f}", ha="center", va="bottom")

    # slot 利用率 —— static 整批等最长序列, 大量 slot 空转
    us = float(m["slot_eff_pct"]["static"]); uc = float(m["slot_eff_pct"]["continuous"])
    a2.bar(["static", "continuous"], [us, uc], color=[S, C])
    a2.set(ylabel="useful slot-steps (%)", title="batch slot utilization", ylim=(0, 105))
    for i, v in enumerate([us, uc]):
        a2.text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    # 延迟 (avg / p99) —— 分组柱状
    avg = [float(m["avg_lat_ms"]["static"]), float(m["avg_lat_ms"]["continuous"])]
    p99 = [float(m["p99_lat_ms"]["static"]), float(m["p99_lat_ms"]["continuous"])]
    xpos = [0, 1]; w = 0.35
    a3.bar([x - w/2 for x in xpos], avg, w, label="avg", color=[S, C])
    a3.bar([x + w/2 for x in xpos], p99, w, label="p99", color=[S, C], alpha=0.55)
    a3.set(ylabel="latency (ms)", title="request latency (lower better)")
    a3.set_xticks(xpos); a3.set_xticklabels(["static", "continuous"]); a3.legend()
    for x, va, vp in zip(xpos, avg, p99):
        a3.text(x - w/2, va, f"{va:.0f}", ha="center", va="bottom", fontsize=8)
        a3.text(x + w/2, vp, f"{vp:.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(); fig.savefig("data/continuous_vs_static.png", dpi=130)
    print("saved data/continuous_vs_static.png")
