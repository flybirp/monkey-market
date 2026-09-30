#!/usr/bin/env python3
"""块自助采样实验总图：信号随块长的"复活曲线" + 跳空/微观结构对照。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import block_bootstrap as bb
from compare_all import load_dir, aligned_pool

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Songti SC", "PingFang SC", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "#111417"
plt.rcParams["axes.facecolor"] = "#111417"
plt.rcParams["text.color"] = "#e6e6e6"
plt.rcParams["axes.labelcolor"] = "#e6e6e6"
plt.rcParams["xtick.color"] = "#9aa0a6"
plt.rcParams["ytick.color"] = "#9aa0a6"
plt.rcParams["axes.edgecolor"] = "#3a4046"

HERE = Path(__file__).parent
ORDER = ["fake(纯随机)", "bb L=1", "bb L=5", "bb L=20", "bb L=60", "REAL(真股)"]
XLAB = ["fake\n纯随机", "L=1\n单日洗牌", "L=5", "L=20", "L=60", "REAL\n真股"]
COL = {"锤子线 Hammer": "#f7768e", "突破20日新高": "#7aa2f7",
       "跌破20日新低": "#9ece6a", "MA5下穿MA20(死叉)": "#e0af68"}


def panel_a(ax):
    d = pd.read_csv(HERE / "pattern_test_all.csv")
    for k, col in COL.items():
        s = d[d["形态"] == k].set_index("数据集")["5日超额%"]
        ax.plot(range(len(ORDER)), [s.get(o, np.nan) for o in ORDER], "o-",
                color=col, lw=2, ms=6, label=k)
    ax.axhline(0, color="#5a6066", lw=1, ls="--")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(XLAB, fontsize=9)
    ax.set_ylabel("形态后 5 日超额收益 (%)")
    ax.set_title("① 信号随块长的复活曲线：L=1 归零，L≥5 立刻回来", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.15)


def panel_b(ax):
    d = pd.read_csv(HERE / "pattern_test_all.csv")
    n5 = [int(((d["数据集"] == o) & (d["p(5日)"] < 0.05)).sum()) for o in ORDER]
    n20 = [int(((d["数据集"] == o) & (d["p(20日)"] < 0.05)).sum()) for o in ORDER]
    x = np.arange(len(ORDER)); w = 0.38
    ax.bar(x - w / 2, n5, w, color="#7aa2f7", label="5 日")
    ax.bar(x + w / 2, n20, w, color="#f7768e", label="20 日")
    ax.axhline(11 * 0.05, color="#e0af68", ls="--", lw=1.2, label="期望假阳性 0.55")
    for i, (a_, b_) in enumerate(zip(n5, n20)):
        ax.text(i - w / 2, a_ + 0.2, str(a_), ha="center", fontsize=8, color="#7aa2f7")
        ax.text(i + w / 2, b_ + 0.2, str(b_), ha="center", fontsize=8, color="#f7768e")
    ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=9)
    ax.set_ylim(0, 12); ax.set_ylabel("显著形态个数 / 共 11 个")
    ax.set_title("② 显著个数：只有把序列自相关彻底打碎，形态才失效", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.12, axis="y")


def panel_c(ax):
    cal = bb.load_calendar()
    pool = bb.pick_stocks(cal, 1001)
    real = np.stack([a for _, a in pool])
    boot = load_dir(HERE / "data_boot")
    fake = load_dir(HERE / "data", limit=1001)
    for m, name, col in ((fake, "fake 纯随机", "#9aa0a6"),
                         (boot, "bb L=20 合成", "#7aa2f7"),
                         (real, "REAL 真股", "#f7768e")):
        g = np.abs(m[:, 1:, 0] / m[:, :-1, 1] - 1.0).ravel() * 100
        g = g[g < 6]
        ax.hist(g, bins=90, range=(0, 3), histtype="step", lw=1.6,
                color=col, label=f"{name}  跳空率 {np.mean(np.abs(m[:,1:,0]/m[:,:-1,1]-1.0) > 1e-9)*100:.1f}%")
    ax.set_yscale("log")
    ax.set_xlabel("|今日开盘 / 昨收 − 1|  (%)")
    ax.set_ylabel("天数（对数）")
    ax.set_title("③ 跳空幅度分布：bb 完美继承真股，fake 一根没有", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.12)


def panel_d(ax, ret_acf=False):
    """|r| 的自相关函数（波动率聚集）—— 块长的作用一目了然。"""
    cal = bb.load_calendar()
    pool = bb.pick_stocks(cal, 1001)
    real = np.stack([a for _, a in pool])
    sets = [(np.abs(_r(load_dir(HERE / "data", limit=1001))), "fake 纯随机", "#9aa0a6", "--"),
            (np.abs(_r(load_dir(HERE / "data_boot_L1"))), "bb L=1", "#c0caf5", "--"),
            (np.abs(_r(load_dir(HERE / "data_boot"))), "bb L=20", "#7aa2f7", "-"),
            (np.abs(_r(load_dir(HERE / "data_boot_L60"))), "bb L=60", "#9ece6a", "-"),
            (np.abs(_r(real)), "REAL 真股", "#f7768e", "-")]
    lags = np.arange(1, 61)
    for x_, name, col, ls in sets:
        acf = _acf(x_.ravel(), lags)
        ax.plot(lags, acf, ls, color=col, lw=1.8, label=name)
    ax.axhline(0, color="#5a6066", lw=1)
    ax.set_xlabel("滞后天数")
    ax.set_ylabel("|r| 自相关（波动率聚集）")
    ax.set_title("④ 什么被保留、什么被摧毁：波动聚集只在块内活着", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.12)


def _r(m):
    c = m[:, :, 1]
    return (c[:, 1:] / c[:, :-1] - 1.0).ravel()


def _acf(x, lags):
    x = x - x.mean()
    d = np.dot(x, x)
    return np.array([np.dot(x[:-l], x[l:]) / d for l in lags])


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panel_a(axes[0][0]); panel_b(axes[0][1])
    panel_c(axes[1][0]); panel_d(axes[1][1])
    fig.suptitle("猴子市场 · 块自助采样（Block Bootstrap）：技术形态的预测力到底来自哪里",
                 fontsize=15, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = HERE / "boot_report.png"
    fig.savefig(out, dpi=125)
    print("已写出", out)


if __name__ == "__main__":
    main()
