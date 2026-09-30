#!/usr/bin/env python3
"""
猴子市场生成器 —— 纯随机 K 线实验（证伪技术分析的第一步）

规则（完全不含任何"市场记忆"，只有抛硬币）：
  1. 每个交易日，猴子抛 32 次公平硬币决定 32 个 tick 的涨跌；
  2. 每个 tick 的对数收益 = +sigma / -sigma（等幅，方向由硬币决定）；
  3. 当日 open  = 昨收（价格连贯，无跳空缺口）；
     当日 close = 第 32 次抛完之后的价格；
     当日 high/low = 当日 33 个价格点（open + 32 个 tick）的极值；
  4. 连续 1000 个交易日 = 一只个股的完整历史；
  5. 重复 fake0 ... fake1000 共 1001 只个股（每只独立种子，可复现）。

日期取自 ~/Documents/mainland_data_2014 的真实 A 股交易日历（2014-01-02 起）。
输出格式：date,open,close,high,low

用法：
  python generate_monkeys.py                    # 默认 1001 只 × 1000 天
  python generate_monkeys.py --stocks 100 --days 1000 --out data_small
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

DEFAULTS = dict(
    days=1000,          # 每只个股的交易日数
    stocks=1001,        # fake0 ~ fake1000
    steps=32,           # 每日抛硬币次数（日内 tick 数）
    sigma=0.003,        # 单个 tick 的对数收益幅度 -> 日波动率 ≈ sigma*sqrt(32) ≈ 1.7~1.8%
    jitter=0.6,         # tick 幅度抖动: amp ~ U(1-jitter, 1+jitter)，0=严格等幅
    s0=10.00,           # 起始价（第 1 天开盘价）
    seed0=20140102,     # 基础随机种子，fake_i 使用 seed0 + i
    decimals=2,         # 价格保留小数位
)


def build_calendar(days: int) -> list[str]:
    cal = (HERE / "trading_calendar.csv").read_text().split()
    assert cal[0] == "date"
    dates = cal[1:]
    if len(dates) < days:
        raise SystemExit(f"交易日历只有 {len(dates)} 天，不足 {days}")
    return dates[:days]


def gen_one(rng: np.random.Generator, days: int, steps: int, sigma: float,
            s0: float, decimals: int, jitter: float = 0.0) -> np.ndarray:
    """返回 shape=(days,4) 的 [open, close, high, low]（已取整）。"""
    n = days * steps
    # 抛硬币：+1 / -1 只决定方向
    tick = rng.integers(0, 2, size=n).astype(np.float64) * 2.0 - 1.0
    if jitter > 0:  # 每步幅度随机（避免等幅二项导致 14% 的"平盘日"）
        tick *= 1.0 + jitter * rng.uniform(-1.0, 1.0, size=n)
    logret = tick * sigma
    # 全局对数价格：L[0]=log(s0)，之后每个 tick 累加
    L = np.empty(n + 1)
    L[0] = np.log(s0)
    np.cumsum(logret, out=L[1:])
    L[1:] += L[0]
    P = np.exp(L)

    o = P[0:n:steps]                  # 每天起点（第 0 天开盘 = s0，之后 = 昨收）
    c = P[steps::steps]               # 每天第 32 tick 后的价格
    T = P[1:].reshape(days, steps)    # 每天 32 个 tick 后的价格
    h = np.maximum(o, T.max(axis=1))  # 极值含 open 本身
    lo = np.minimum(o, T.min(axis=1))

    # 取整到分，并修复取整可能造成的 OHLC 矛盾
    f = 10.0 ** decimals
    o = np.round(o * f) / f
    c = np.round(c * f) / f
    h = np.round(h * f) / f
    lo = np.round(lo * f) / f
    h = np.maximum.reduce([o, c, h])
    lo = np.minimum.reduce([o, c, lo])
    return np.column_stack([o, c, h, lo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DEFAULTS["days"])
    ap.add_argument("--stocks", type=int, default=DEFAULTS["stocks"])
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--sigma", type=float, default=DEFAULTS["sigma"])
    ap.add_argument("--jitter", type=float, default=DEFAULTS["jitter"])
    ap.add_argument("--s0", type=float, default=DEFAULTS["s0"])
    ap.add_argument("--seed0", type=int, default=DEFAULTS["seed0"])
    ap.add_argument("--out", type=str, default="data")
    a = ap.parse_args()

    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    dates = build_calendar(a.days)

    t0 = time.time()
    fmt = "%." + str(DEFAULTS["decimals"]) + "f"
    for i in range(a.stocks):
        rng = np.random.default_rng(a.seed0 + i)
        ohlc = gen_one(rng, a.days, a.steps, a.sigma, a.s0, DEFAULTS["decimals"], a.jitter)
        lines = [
            f"{d},{fmt % r[0]},{fmt % r[1]},{fmt % r[2]},{fmt % r[3]}"
            for d, r in zip(dates, ohlc)
        ]
        (out / f"fake{i}.csv").write_text("date,open,close,high,low\n" + "\n".join(lines) + "\n")
        if (i + 1) % 100 == 0:
            print(f"  fake{i} done  ({time.time() - t0:.1f}s)")

    meta = dict(DEFAULTS, **vars(a))
    amp2 = 1.0 + a.jitter ** 2 / 3.0
    meta["daily_sigma_approx"] = round(a.sigma * (a.steps * amp2) ** 0.5 * 100, 3)
    meta["date_range"] = [dates[0], dates[-1]]
    meta["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["coin_flips_total"] = a.stocks * a.days * a.steps
    (HERE / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\n完成：{a.stocks} 只 × {a.days} 天 -> {out}")
    print(f"日期区间 {dates[0]} ~ {dates[-1]}｜每 tick ±{a.sigma:.2%}｜日波动率≈{meta['daily_sigma_approx']}%")
    print(f"总抛硬币次数 {meta['coin_flips_total']:,}｜耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
