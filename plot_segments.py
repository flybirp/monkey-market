#!/usr/bin/env python3
"""分段稳健性热力图：随机数据 vs 真实 A 股。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Songti SC", "PingFang SC", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "#111417"
plt.rcParams["axes.facecolor"] = "#111417"
plt.rcParams["text.color"] = "#e6e6e6"
plt.rcParams["xtick.color"] = "#9aa0a6"
plt.rcParams["ytick.color"] = "#9aa0a6"

HERE = Path(__file__).parent
SEG_LABEL = ["S1 (250d)", "S2 (250d)", "S3 (250d)", "S4 (250d)"]


def load(tag):
    df = pd.read_csv(HERE / f"segment_{tag}.csv", index_col=0)
    cols = [c for c in df.columns if c.endswith("超额%")]
    m = df[cols].values
    labels = [i.split("(")[0].strip() for i in df.index]
    return labels, m, df


def panel(ax, labels, m, title):
    v = float(np.percentile(np.abs(m), 92))
    v = max(v, 0.08)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(SEG_LABEL, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="#111417" if abs(m[i, j]) > 0.45 * v else "#e6e6e6")
    ax.set_title(title, fontsize=11.5, pad=10)
    return im


def main():
    lab_m, m_m, _ = load("monkey")
    lab_r, m_r, _ = load("real")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    fig.suptitle("形态出现后 5 日超额收益（%）按 250 天分段 — 方向是否同向？",
                 fontsize=15, color="#f5f5f5", y=0.97)
    panel(axes[0], lab_m, m_m,
          f"① 纯随机（抛硬币）：各段在 0 附近随机抖动\n四段同向者 {int((np.abs(np.sign(m_m).sum(1)) == 4).sum())}/11")
    im = panel(axes[1], lab_r, m_r,
               f"② 真实 A 股（1600 只）：全样本看着很'显著'，分段后大幅翻号\n"
               f"四段同向者 {int((np.abs(np.sign(m_r).sum(1)) == 4).sum())}/11  ← 只是各段行情周期的影子")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("5 日超额收益 %", color="#e6e6e6")
    cb.ax.tick_params(colors="#9aa0a6")
    fig.text(0.5, 0.02, "红=正超额 / 蓝=负超额；若形态真携带信息，同一形态应在四段同色",
             ha="center", fontsize=9.5, color="#9aa0a6")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = HERE / "segment_heatmap.png"
    fig.savefig(out, dpi=130)
    print("saved ->", out)


if __name__ == "__main__":
    main()
