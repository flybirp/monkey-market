#!/usr/bin/env python3
"""
monkey-buy-hold-sell：让猴子当交易员。

每天开盘前，猴子随机三选一 —— 买(buy) / 持有(hold) / 卖(sell)：
  * buy  : 空仓才买得进（已持仓则视为 hold）
  * sell : 持仓才卖得出（已空仓则视为 hold）
  * hold : 不动
成交价用当日收盘价结算；现金与持仓都按复利滚动。

对照组 = 同一只股票的"第一天买入、持有 1000 天"（Buy & Hold）。
在 1000 只纯随机个股上各跑一次 = 1000 次独立实验，把收益率分布画到正态曲线上。

要看的核心问题：
  1. 随机买卖的收益分布长什么样？均值是不是 0？
  2. 1000 只猴子交易员里，能冒出多少"翻倍股神"？（纯运气的上限）
  3. 频繁进出能不能打败 buy & hold？
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"


def load_close(data_dir: Path = DATA, limit: int | None = None, days: int | None = None,
               start: str = "2014-01-02") -> tuple[list[str], np.ndarray]:
    """返回 (names, close 矩阵 (S, D))。默认读随机数据，也可指向真实 A 股日线目录。"""
    if data_dir == DATA:
        files = sorted(DATA.glob("fake*.csv"), key=lambda p: int(p.stem[4:]))
        if limit:
            files = files[:limit]
        cols = [pd.read_csv(f, usecols=["close"]).to_numpy().ravel() for f in files]
        return [f.stem for f in files], np.array(cols)

    names, cols = [], []
    for f in sorted(data_dir.glob("*.csv")):           # 真实数据：按文件名字母序
        if limit and len(cols) >= limit:
            break
        a = pd.read_csv(f, usecols=["date", "close"], nrows=days or 10 ** 6)
        if days and (len(a) < days or str(a["date"].iloc[0])[:10] != start):
            continue                                    # 剔除起点不对 / 天数不足的
        cols.append(a["close"].to_numpy(float))
        names.append(f.stem)
    return names, np.array(cols)


def simulate(c: np.ndarray, fee: float = 0.0, seed: int = 20260930) -> pd.DataFrame:
    """随机 buy/hold/sell 策略。返回每只股票的最终收益、交易次数、暴露度等。"""
    S, D = c.shape
    r = c[:, 1:] / c[:, :-1] - 1.0          # 每日收益（第 t 天相对第 t-1 天）
    rng = np.random.default_rng(seed)
    pos = np.zeros(S, dtype=bool)           # 持仓状态
    eq = np.ones(S)                         # 权益（复利）
    trades = np.zeros(S, dtype=np.int64)
    held = np.zeros(S, dtype=np.int64)

    for t in range(D - 1):
        act = rng.integers(0, 3, size=S)    # 0=buy 1=sell 2=hold
        prev = pos
        do_buy = (act == 0) & (~prev)
        do_sell = (act == 1) & prev
        pos = prev.copy()
        pos[do_buy] = True
        pos[do_sell] = False
        traded = do_buy | do_sell
        trades += traded
        eq *= np.where(traded, 1.0 - fee, 1.0) * np.where(pos, 1.0 + r[:, t], 1.0)
        held += pos

    buyhold = c[:, -1] / c[:, 0] - 1.0      # 对照组：买入并持有全程
    return pd.DataFrame({
        "monkey_ret": eq - 1.0,
        "buyhold_ret": buyhold,
        "alpha": (eq - 1.0) - buyhold,
        "trades": trades,
        "exposure": held / (D - 1),
        "final_equity": eq,
    })


def describe(x: np.ndarray, name: str) -> dict:
    x = np.asarray(x, dtype=float)
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return {
        "策略": name,
        "样本": len(x),
        "均值%": round(x.mean() * 100, 2),
        "中位数%": round(np.median(x) * 100, 2),
        "标准差%": round(x.std(ddof=1) * 100, 2),
        "偏度": round(pd.Series(x).skew(), 3),
        "超额峰度": round(pd.Series(x).kurt(), 3),
        "胜率%": round((x > 0).mean() * 100, 1),
        "P1%": round(q[0] * 100, 1), "P5%": round(q[1] * 100, 1),
        "P25%": round(q[2] * 100, 1), "P50%": round(q[3] * 100, 1),
        "P75%": round(q[4] * 100, 1), "P95%": round(q[5] * 100, 1),
        "P99%": round(q[6] * 100, 1),
        "翻倍占比%": round((x > 1.0).mean() * 100, 2),
        "腰斩占比%": round((x < -0.5).mean() * 100, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000, help="跑多少只股票（= 多少次独立实验）")
    ap.add_argument("--data", type=str, default=str(DATA), help="数据目录，默认随机数据；也可指向真实 A 股日线目录")
    ap.add_argument("--days", type=int, default=None, help="每只取多少天（真实数据用 1000）")
    ap.add_argument("--tag", type=str, default="", help="输出文件后缀")
    ap.add_argument("--fee", type=float, default=0.0, help="单边交易费率，如 0.002 = 千二")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    names, c = load_close(Path(a.data).expanduser(), a.runs, a.days)
    real = Path(a.data).expanduser() != DATA
    df = simulate(c, fee=a.fee, seed=a.seed)
    df.insert(0, "stock", names)
    df.to_csv(HERE / f"monkey_trader_runs{a.tag}.csv", index=False)

    print(f"=== 猴子交易员 vs 买入持有｜{len(df)} 次独立实验｜费率 {a.fee:.3%}｜数据源: "
          f"{'真实 A 股 ' + a.data if real else '纯随机 fake 个股'} ===")
    stat = pd.DataFrame([describe(df.monkey_ret, "🐒 随机买卖"), describe(df.buyhold_ret, "🛒 买入持有")])
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(stat.to_string(index=False))

    win = float((df.alpha > 0).mean())
    print(f"\n随机交易跑赢买入持有的比例: {win * 100:.1f}%   alpha 均值 {df.alpha.mean() * 100:+.2f}%")
    print(f"平均交易次数 {df.trades.mean():.0f} 次 / {c.shape[1] - 1} 天，平均持仓暴露 {df.exposure.mean() * 100:.1f}%")
    top = df.nlargest(5, "monkey_ret")
    print("\n1000 只猴子里的 Top5（纯抛硬币，无任何技巧）:")
    for _, row in top.iterrows():
        print(f"  {row['stock']}: 随机买卖 {row['monkey_ret'] * 100:+.0f}%  "
              f"(同期买入持有 {row['buyhold_ret'] * 100:+.0f}%)  交易 {int(row['trades'])} 次")
    god = float((df.monkey_ret > 1.0).mean())
    print(f"\n收益翻倍以上的'股神'占比: {god * 100:.2f}%（{int(god * len(df))} 只）—— 全部由运气产生")

    fee_mean = None
    if a.fee == 0:
        df_fee = simulate(c, fee=0.002, seed=a.seed)
        fee_mean = float(df_fee.monkey_ret.mean())
        print(f"\n[附加] 加上千二单边手续费后: 随机买卖均值 {df_fee.monkey_ret.mean() * 100:+.2f}%"
              f"（无费时 {df.monkey_ret.mean() * 100:+.2f}%），胜率 {(df_fee.monkey_ret > 0).mean() * 100:.1f}%")

    (HERE / "monkey_trader_meta.json").write_text(json.dumps(
        {"runs": len(df), "fee": a.fee, "seed": a.seed, "days": int(c.shape[1]),
         "mean_monkey": float(df.monkey_ret.mean()), "mean_buyhold": float(df.buyhold_ret.mean()),
         "sharpe_like_monkey": float(df.monkey_ret.mean() / df.monkey_ret.std(ddof=1)),
         "sharpe_like_buyhold": float(df.buyhold_ret.mean() / df.buyhold_ret.std(ddof=1)),
         "god_ratio": god, "exposure": float(df.exposure.mean()), "trades": float(df.trades.mean()),
         "data_source": "real" if real else "fake"},
        ensure_ascii=False, indent=2))

    if not a.no_plot:
        import plot_trader
        plot_trader.main(df, fee=a.fee, fee_mean=fee_mean, tag=a.tag,
                         src="真实 A 股" if real else "纯随机 fake 个股")
    print(f"\n明细: {HERE / f'monkey_trader_runs{a.tag}.csv'}")
    print(f"图: {HERE / f'monkey_trader{a.tag}.png'}")


if __name__ == "__main__":
    main()
