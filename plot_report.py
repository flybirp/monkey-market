#!/usr/bin/env python3
"""出图：随机 K 线长什么样 + 形态预测力证据。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["PingFang HK", "PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "#111417"
plt.rcParams["axes.facecolor"] = "#111417"
plt.rcParams["text.color"] = "#e6e6e6"
plt.rcParams["axes.labelcolor"] = "#e6e6e6"
plt.rcParams["xtick.color"] = "#9aa0a6"
plt.rcParams["ytick.color"] = "#9aa0a6"
plt.rcParams["axes.edgecolor"] = "#3a4046"

HERE = Path(__file__).parent
UP, DN = "#ef4b4b", "#22c55e"   # 中国习惯：涨红跌绿


def candles(ax, o, c, h, l, lw=0.8):
    idx = np.arange(len(o))
    up = c >= o
    for i in idx:
        col = UP if up[i] else DN
        ax.plot([i, i], [l[i], h[i]], color=col, lw=lw, zorder=1)
        lo, hi = min(o[i], c[i]), max(o[i], c[i])
        ax.add_patch(Rectangle((i - 0.32, lo), 0.64, max(hi - lo, 1e-3),
                               facecolor=col, edgecolor=col, zorder=2))
    ax.set_xlim(-1, len(o))


def main():
    df = pd.read_csv(HERE / "data" / "fake7.csv", parse_dates=["date"])
    res = pd.read_csv(HERE / "pattern_test.csv")

    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    fig.suptitle("Monkey Market — 100% 抛硬币生成的 K 线，照样长成教科书的样子",
                 fontsize=16, color="#f5f5f5", y=0.975)

    # ① 1000 天走势
    a = ax[0, 0]
    a.plot(df.close.values, color="#7aa2f7", lw=0.9)
    a.set_title("① fake7：1000 个交易日的“股价”（纯随机游走）", fontsize=11, pad=8)
    a.set_ylabel("价格")
    a.set_xlabel("交易日（2014-01-02 ~ 2018-02-01）")
    a.grid(alpha=0.15)

    # ② 局部 K 线 + 形态标注
    b = ax[0, 1]
    s, n = 300, 70
    sub = df.iloc[s:s + n]
    o, c, h, l = sub.open.values, sub.close.values, sub.high.values, sub.low.values
    candles(b, o, c, h, l)
    body = np.abs(c - o)
    rng = np.where(h - l > 0, h - l, np.nan)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    hammer = (lower >= 2 * body) & (upper <= body) & (body > 0)
    po, pc = np.r_[np.nan, o[:-1]], np.r_[np.nan, c[:-1]]
    engulf = (pc < po) & (c > o) & (c >= po) & (o <= pc)
    for i in np.flatnonzero(hammer):
        b.annotate("锤子线", (i, l[i]), textcoords="offset points", xytext=(0, -16),
                   ha="center", fontsize=8, color="#fbbf24",
                   arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=0.8))
    for i in np.flatnonzero(engulf):
        b.annotate("看涨吞没", (i, h[i]), textcoords="offset points", xytext=(0, 12),
                   ha="center", fontsize=8, color="#a78bfa",
                   arrowprops=dict(arrowstyle="->", color="#a78bfa", lw=0.8))
    b.set_title(f"② 放大 {n} 天：锤子线 / 吞没形态应有尽有（共命中 {int(hammer.sum() + engulf.sum())} 次）",
                fontsize=11, pad=8)
    b.set_ylabel("价格")
    b.grid(alpha=0.12, axis="y")

    # ③ 形态预测力：超额收益 + bootstrap 95% 置信区间
    d = ax[1, 0]
    x = np.arange(len(res))
    ex = res["5日超额%"].values
    se5 = np.where(res["t_boot(5日)"].abs() > 1e-9, np.abs(ex / res["t_boot(5日)"].values), np.nan)
    d.bar(x - 0.2, ex, 0.4, color="#7aa2f7", label="未来 5 日超额")
    d.errorbar(x - 0.2, ex, yerr=1.96 * se5, fmt="none", ecolor="#c8d3f5", lw=1, capsize=3)
    ex20 = res["20日超额%"].values
    se20 = np.where(res["t_boot(20日)"].abs() > 1e-9, np.abs(ex20 / res["t_boot(20日)"].values), np.nan)
    d.bar(x + 0.2, ex20, 0.4, color="#f7768e", label="未来 20 日超额")
    d.errorbar(x + 0.2, ex20, yerr=1.96 * se20, fmt="none", ecolor="#f7c8d0", lw=1, capsize=3)
    d.axhline(0, color="#8b949e", lw=1)
    d.set_xticks(x)
    d.set_xticklabels([s.split("(")[0] for s in res["形态"]], rotation=35, ha="right", fontsize=8.5)
    d.set_ylabel("超额收益 %")
    d.set_title("③ 形态出现后的超额收益：全部包含 0（无一显著）", fontsize=11, pad=8)
    d.legend(fontsize=9, frameon=False)
    d.grid(alpha=0.12, axis="y")

    # ④ 日收益分布 vs 正态
    e = ax[1, 1]
    files = sorted((HERE / "data").glob("fake*.csv"))[:300]
    r = np.concatenate([np.diff(np.log(pd.read_csv(f, usecols=["close"]).close.values)) for f in files])
    e.hist(r * 100, bins=120, density=True, color="#7aa2f7", alpha=0.85)
    xs = np.linspace(r.min(), r.max(), 300) * 100
    e.plot(xs, np.exp(-0.5 * ((xs - r.mean() * 100) / (r.std() * 100)) ** 2) / (r.std() * 100 * np.sqrt(2 * np.pi)),
           color="#fbbf24", lw=1.6, label=f"正态拟合 σ={r.std() * 100:.2f}%")
    e.set_title(f"④ 日收益分布（{r.size:,} 个样本）：钟形、无厚尾", fontsize=11, pad=8)
    e.set_xlabel("日收益 %")
    e.legend(fontsize=9, frameon=False)
    e.grid(alpha=0.12)

    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    out = HERE / "monkey_report.png"
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
