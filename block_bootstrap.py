#!/usr/bin/env python3
"""
块自助采样（Block Bootstrap）合成器 —— 解决 fake0~fake1000 "太干净、没有跳空" 的问题

思路（来自 flybirp 的伪代码，本文件为实现版）：
    真股 A 的 1000 根日 K 里，把每一天压缩成 4 个「尺度无关」的比例特征
        r = close[t] / close[t-1]    收盘收益（含跳空贡献）
        g = open[t]  / close[t-1]    开盘跳空幅度 ★保留跳空的关键
        h = high[t]  / close[t]      上影 / 收盘
        l = low[t]   / close[t]      下影 / 收盘
    切成连续块（长度 L），块内顺序保留、块间随机重排，再从首根 K 累乘重建。

这样得到的数据：
    保留：单日波动率分布、跳空分布、上下影结构、涨跌停、波动率聚集（块内短期自相关）
    摧毁：牛熊周期、事件时序、任何跨块的长记忆 —— 正是我们想证伪的"图形"

用法：
    python block_bootstrap.py --L 20            # -> data_boot/     (boot0..boot1000)
    python block_bootstrap.py --L 60            # -> data_boot_L60/ (boot0..boot1000)
    python block_bootstrap.py --L 20 --L2 60    # 一次跑两套
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REAL_DIR = Path.home() / "Documents" / "mainland_data_2014"
CALENDAR = HERE / "trading_calendar.csv"

DAYS = 1000               # 与 fake 数据一致：1000 个交易日
N_STOCK = 1001            # fake0 ~ fake1000
EXTREME = 0.15            # 单日 |r-1| 超过此值的股票整只剔除（ST 复牌等一次性跳变）
BASE_SEED = 20140102


# ---------------------------------------------------------------- 数据准备
def load_calendar(n: int = DAYS) -> list[str]:
    df = pd.read_csv(CALENDAR, usecols=["date"])
    return [str(d) for d in df["date"].tolist()[:n]]


def pick_stocks(cal: list[str], n_need: int, verbose: bool = True):
    """按代码顺序挑出前 n_need 只「干净」的真股，返回 [(code, ohlc(D,4)), ...]。

    干净 = 2014-01-02 已上市（首行即该日）+ 自有交易日 >= DAYS 根 + OHLC 全正自洽
           + 窗口内无 |r-1| > EXTREME 的一次性跳变（ST 复牌等）。

    注：不要求与联合日历逐日对齐。停牌股会往后多取几十天，但块自助法本来
    就摧毁了时间轴，且输出的日期标签统一用联合日历（与 fake 数据格式一致），
    保证 1001 只合成股逐行对齐、可直接做横截面中性化。
    """
    picked, n_bad = [], {"short": 0, "late": 0, "dirty": 0, "extreme": 0, "err": 0}
    files = sorted(REAL_DIR.glob("*.csv"))
    for f in files:
        if len(picked) >= n_need:
            break
        code = f.stem
        try:
            df = pd.read_csv(f, usecols=["date", "open", "close", "high", "low"],
                             nrows=DAYS + 400)
            if str(df["date"].iloc[0]) != cal[0]:
                n_bad["late"] += 1           # 2014-01-02 之后才上市
                continue
            if len(df) < DAYS:
                n_bad["short"] += 1
                continue
            a = df[["open", "close", "high", "low"]].to_numpy(dtype=np.float64)[:DAYS]
        except Exception:
            n_bad["err"] += 1
            continue
        if not np.isfinite(a).all() or (a <= 0).any():
            n_bad["dirty"] += 1
            continue
        # OHLC 自洽性（脏数据直接丢）
        if (a[:, 2] < np.maximum(a[:, 0], a[:, 1]) - 1e-9).any() or \
           (a[:, 3] > np.minimum(a[:, 0], a[:, 1]) + 1e-9).any():
            n_bad["dirty"] += 1
            continue
        r = a[1:, 1] / a[:-1, 1]
        if np.max(np.abs(r - 1.0)) > EXTREME:
            n_bad["extreme"] += 1
            continue
        picked.append((code, a))
    if verbose:
        print(f"  扫描 {len(files)} 只真股 -> 取到 {len(picked)} 只  剔除: {n_bad}")
    return picked


# ---------------------------------------------------------------- 核心算法
def make_blocks(feat: np.ndarray, L: int, rng: np.random.Generator):
    """把 (nfeat, 4) 的特征切成整块 + 1 个余块，随机起点，块间无放回洗牌。

    返回洗牌后的特征序列，长度 == nfeat（因此重建后总长度不变）。
    """
    nfeat = len(feat)
    nb = nfeat // L
    rem = nfeat - nb * L
    s = int(rng.integers(0, rem + 1)) if rem > 0 else 0

    body = feat[s: s + nb * L]
    blocks = [body[i * L:(i + 1) * L] for i in range(nb)]
    if rem > 0:
        # 两端残段拼成 1 个余块一起洗（长度恒为 rem，总长守恒）
        leftover = np.concatenate([feat[:s], feat[s + nb * L:]], axis=0)
        blocks.append(leftover)

    rng.shuffle(blocks)                       # 无放回重排
    return np.concatenate(blocks, axis=0)


def synth_one(a: np.ndarray, L: int, seed: int, decimals: int = 2):
    """输入真股 OHLC (D,4) -> 输出合成 OHLC (D,4)，逐块洗牌后累乘重建。"""
    rng = np.random.default_rng(seed)
    o, c, hi, lo = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    prev = c[:-1]
    feat = np.column_stack([
        c[1:] / prev,          # r  收盘收益
        o[1:] / prev,          # g  开盘跳空
        hi[1:] / c[1:],        # h  上影 / 收盘
        lo[1:] / c[1:],        # l  下影 / 收盘
    ])
    seq = make_blocks(feat, L, rng)

    q = 10.0 ** decimals
    new = np.empty_like(a)
    new[0] = np.round(a[0] * q) / q          # 首根照搬（没有前收）
    pc = new[0, 1]
    for k in range(len(seq)):
        r, g, h, l = seq[k]
        cc = np.round(pc * r * q) / q
        oo = np.round(pc * g * q) / q
        hh = np.round(cc * h * q) / q
        ll = np.round(cc * l * q) / q
        hh = max(hh, oo, cc)                 # 修正：比例还原后 open 换了位置
        ll = min(ll, oo, cc)
        if ll <= 0:                          # 价格不能为 0
            ll = 0.01
        new[k + 1] = (oo, cc, hh, ll)
        pc = cc
    return new


# ---------------------------------------------------------------- 体检
def quick_stats(mats: list[np.ndarray]) -> dict:
    """对一组合成/真实 OHLC 算关键统计量，用于跟真股对照。"""
    o = np.stack([m[:, 0] for m in mats])
    c = np.stack([m[:, 1] for m in mats])
    hi = np.stack([m[:, 2] for m in mats])
    lo = np.stack([m[:, 3] for m in mats])

    r = c[:, 1:] / c[:, :-1] - 1.0
    g = o[:, 1:] / c[:, :-1] - 1.0
    rr = r.ravel()
    gg = np.abs(g).ravel()

    # |r| 的一阶自相关（波动率聚集）
    ar = np.abs(r)
    x, y = ar[:, :-1].ravel(), ar[:, 1:].ravel()
    vol_clust = float(np.corrcoef(x, y)[0, 1])
    x2, y2 = r[:, :-1].ravel(), r[:, 1:].ravel()
    mom = float(np.corrcoef(x2, y2)[0, 1])

    sd = float(np.std(rr, ddof=1))
    m = rr.mean()
    kurt = float(np.mean((rr - m) ** 4) / sd ** 4 - 3.0)
    skew = float(np.mean((rr - m) ** 3) / sd ** 3)

    viol = int(((hi < np.maximum(o, c) - 1e-9).sum() + (lo > np.minimum(o, c) + 1e-9).sum()))
    limit = float(np.mean(np.abs(rr) > 0.095))

    return {
        "sigma": round(sd * 100, 4),
        "skew": round(skew, 3),
        "excess_kurtosis": round(kurt, 3),
        "gap_ratio_pct": round(float(np.mean(gg > 1e-9)) * 100, 2),   # 有跳空的天数占比
        "gap_abs_mean_pct": round(float(np.mean(gg)) * 100, 4),
        "gap_abs_median_pct": round(float(np.median(gg)) * 100, 4),
        "gap_abs_p95_pct": round(float(np.percentile(gg, 95)) * 100, 4),
        "vol_cluster_absr_lag1": round(vol_clust, 4),
        "return_lag1": round(mom, 4),
        "near_limit_pct": round(limit * 100, 3),
        "ohlc_violations": viol,
        "n_days": int(r.size),
    }


# ---------------------------------------------------------------- 主流程
def run(L: int, out_dir: Path, cal: list[str], pool, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    mats = []
    for i, (code, a) in enumerate(pool):
        new = synth_one(a, L, BASE_SEED + i * 7919)
        df = pd.DataFrame({
            "date": cal,
            "open": np.round(new[:, 0], 2),
            "close": np.round(new[:, 1], 2),
            "high": np.round(new[:, 2], 2),
            "low": np.round(new[:, 3], 2),
        })
        df.to_csv(out_dir / f"boot{i}.csv", index=False, float_format="%.2f")
        mats.append(new)
        if (i + 1) % 200 == 0:
            print(f"    L={L}: {i + 1}/{len(pool)}", flush=True)

    pd.DataFrame({"idx": range(len(pool)),
                  "source_code": [c for c, _ in pool]}).to_csv(
        out_dir / "sources.csv", index=False)

    st = quick_stats(mats)
    st["L"] = L
    st["n_stocks"] = len(pool)
    st["extreme_filter"] = EXTREME
    st["seed"] = BASE_SEED
    (out_dir / "meta.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
    print(f"  [{tag}] L={L} -> {out_dir}  {st}")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=20, help="主块长")
    ap.add_argument("--L2", type=int, default=None, help="补充块长（同时再跑一套）")
    ap.add_argument("--stocks", type=int, default=N_STOCK)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cal = load_calendar()
    print(f"日历 {len(cal)} 天: {cal[0]} ~ {cal[-1]}")
    print("挑选真股（块池 = 每只股票洗自己的牌）...")
    pool = pick_stocks(cal, args.stocks)
    if len(pool) < args.stocks:
        print(f"  ⚠️ 只取到 {len(pool)} 只（要求 {args.stocks}）", file=sys.stderr)

    out1 = Path(args.outdir) if args.outdir else HERE / "data_boot"
    res = {}
    res[args.L] = run(args.L, out1, cal, pool, "主")
    if args.L2:
        out2 = HERE / f"data_boot_L{args.L2}"
        res[args.L2] = run(args.L2, out2, cal, pool, "补充")

    # 真股基准（同一批股票、同一窗口）
    real = quick_stats([a for _, a in pool])
    real["L"] = "REAL"
    print(f"  [真股基准] {real}")
    (HERE / "boot_stats.json").write_text(
        json.dumps({"real": real, "boot": res}, ensure_ascii=False, indent=2))
    print("\n已写出 boot_stats.json")


if __name__ == "__main__":
    main()
