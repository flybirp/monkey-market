#!/usr/bin/env python3
"""把猴子交易员 / 买入持有的收益率分布画到正态曲线上。"""
from pathlib import Path
from statistics import NormalDist

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
plt.rcParams["axes.labelcolor"] = "#e6e6e6"
plt.rcParams["xtick.color"] = "#9aa0a6"
plt.rcParams["ytick.color"] = "#9aa0a6"
plt.rcParams["axes.edgecolor"] = "#3a4046"

HERE = Path(__file__).parent
MONKEY, HOLD = "#7aa2f7", "#f7768e"


def _norm_pdf(xs, mu, sd):
    return np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _hist(ax, x, color, title, bins=60):
    lo, hi = np.percentile(x, [0.3, 99.7])
    rng = max(hi - lo, 1e-3)
    lo, hi = lo - 0.08 * rng, hi + 0.08 * rng
    xs = np.linspace(lo, hi, 400)
    ax.hist(x * 100, bins=bins, range=(lo * 100, hi * 100), density=True,
            color=color, alpha=0.55, edgecolor="none")
    mu, sd = x.mean(), x.std(ddof=1)
    # 注意单位：直方图按"个百分点"计密度，正态 pdf 也要换成同一尺度（否则差 100 倍）
    ax.plot(xs * 100, _norm_pdf(xs * 100, mu * 100, sd * 100), color="#fbbf24", lw=2,
            label=f"正态拟合 μ={mu * 100:+.1f}%  σ={sd * 100:.1f}%")
    ax.axvline(mu * 100, color="#fbbf24", lw=1, ls="--", alpha=0.8)
    ax.axvline(0, color="#8b949e", lw=1)
    ax.set_xlim(lo * 100, hi * 100)
    ax.set_xlabel("最终收益率 %")
    ax.set_ylabel("概率密度")
    ax.set_title(title, fontsize=11.5, pad=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(alpha=0.12)
    return mu, sd


def _qq(ax, series: dict):
    nd = NormalDist()
    for name, (x, color) in series.items():
        n = len(x)
        p = (np.arange(1, n + 1) - 0.5) / n
        theo = np.array([nd.inv_cdf(pi) for pi in p])
        samp = np.sort(x)
        z = (samp - samp.mean()) / samp.std(ddof=1)
        ax.scatter(theo, z, s=7, color=color, alpha=0.6, label=name)
    lim = 3.6
    ax.plot([-lim, lim], [-lim, lim], color="#8b949e", lw=1, ls="--")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("正态理论分位数")
    ax.set_ylabel("实际分位数（标准化）")
    ax.set_title("③ Q-Q 图：贴线 = 正态；往上翘 = 右偏长尾（暴富尾巴）", fontsize=11.5, pad=9)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(alpha=0.12)


def main(df: pd.DataFrame, fee: float = 0.0, fee_mean: float | None = None,
         tag: str = "", src: str = "纯随机 fake 个股"):
    m = df.monkey_ret.to_numpy(float)
    b = df.buyhold_ret.to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    fig.suptitle(f"猴子交易员（每天随机 buy/hold/sell）× {len(df)} 次独立实验 — 收益率分布 vs 正态分布"
                 f"（费率 {fee:.2%}）", fontsize=15, color="#f5f5f5", y=0.99)

    mu_m, sd_m = _hist(axes[0], m, MONKEY,
                       f"① 随机买卖：钟形但右偏（偏度 {pd.Series(m).skew():+.2f}，峰度 {pd.Series(m).kurt():+.2f}）\n"
                       f"胜率 {(m > 0).mean() * 100:.1f}%｜翻倍 {(m > 1).mean() * 100:.1f}%｜腰斩 {(m < -0.5).mean() * 100:.1f}%")
    mu_b, sd_b = _hist(axes[1], b, HOLD,
                       f"② 买入持有：暴富尾巴更长（偏度 {pd.Series(b).skew():+.2f}，峰度 {pd.Series(b).kurt():+.2f}）\n"
                       f"胜率 {(b > 0).mean() * 100:.1f}%｜翻倍 {(b > 1).mean() * 100:.1f}%｜腰斩 {(b < -0.5).mean() * 100:.1f}%")
    _qq(axes[2], {"随机买卖": (m, MONKEY), "买入持有": (b, HOLD)})
    if fee_mean is not None:
        axes[0].axvline(fee_mean * 100, color="#ef4b4b", lw=1.6, ls=":")
        axes[0].annotate(f"千二手续费后均值 {fee_mean * 100:+.1f}%", xy=(fee_mean * 100, axes[0].get_ylim()[1] * 0.55),
                         xytext=(6, 0), textcoords="offset points", fontsize=8.5, color="#ef4b4b")

    fig.text(0.5, 0.015,
             f"随机买卖：均值 {mu_m * 100:+.2f}%  波动 {sd_m * 100:.1f}%   "
             f"买入持有：均值 {mu_b * 100:+.2f}%  波动 {sd_b * 100:.1f}%   "
             f"→ 频繁进出把波动从 {sd_b * 100:.0f}% 压到 {sd_m * 100:.0f}%，均值也从 {mu_b * 100:+.1f}% 掉到 {mu_m * 100:+.1f}%；\n"
             f"   收益/波动比 {mu_m / sd_m:.3f} vs {mu_b / sd_b:.3f} —— 只是把两条尾巴一起削掉，并没有变好",
             ha="center", fontsize=9.5, color="#9aa0a6")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    out = HERE / f"monkey_trader{tag}.png"
    fig.savefig(out, dpi=130)
    print("saved ->", out)


if __name__ == "__main__":
    main(pd.read_csv(HERE / "monkey_trader_runs.csv"))
