#!/usr/bin/env python3
"""
三方对照的诊断脚本：真股 / 纯随机 fake / 各块长 block-bootstrap

除了微观结构统计，额外算两个"方法学体检"指标：

1. 市场共同因子强度（日收益矩阵第一主成分解释的方差占比）
   —— 真股 ~30~40%（个股同涨同跌）；逐股独立洗牌会把它打没，
      这是 block bootstrap 的已知代价，必须在结论里说明。

2. corr(过去20日收益, 未来20日收益)
   —— 无放回洗牌（整段序列是同一批收益的重排）会人为制造
      "总和守恒"式的长期负相关。这个指标用来量化该伪影。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import block_bootstrap as bb

HERE = Path(__file__).parent


def load_dir(d: Path, limit=None, days=1000):
    files = sorted(p for p in d.glob("*.csv") if p.name not in ("sources.csv",))
    rows = []
    for f in files:
        if limit and len(rows) >= limit:
            break
        a = pd.read_csv(f, usecols=["date", "open", "close", "high", "low"], nrows=days)
        if len(a) < days or str(a["date"].iloc[0])[:10] != "2014-01-02":
            continue
        rows.append(a[["open", "close", "high", "low"]].to_numpy(float))
    return np.stack(rows)


def diagnose(m: np.ndarray) -> dict:
    c = m[:, :, 1]
    r = c[:, 1:] / c[:, :-1] - 1.0
    r = r - r.mean(axis=1, keepdims=True)          # 去个股均值
    # 1. 市场共同因子：第一主成分解释方差占比
    #    注意不能再减日横截面均值 —— 那正好把市场因子减掉了
    _, s, _ = np.linalg.svd(r, full_matrices=False)
    pc1 = float(s[0] ** 2 / np.sum(s ** 2))
    #    以及等权市场收益（日横截面均值）的波动占比，直观版市场因子
    mkt = r.mean(axis=0)
    mkt_share = float(mkt.var() / r.mean(axis=1).size / 0) if False else \
        float(mkt.var() / (r.var() / r.shape[0] + mkt.var()))
    # 2. 过去20日 vs 未来20日（重叠窗口，无放回洗牌伪影探测器）
    c20p = c[:, 20:] / c[:, :-20] - 1.0            # t-20 -> t
    f20 = c[:, 20:] / c[:, :-20] - 1.0
    p20 = (c[:, 20:-20] / c[:, :-40]) - 1.0        # 过去20日（截至 t）
    q20 = (c[:, 40:] / c[:, 20:-20]) - 1.0         # 未来20日（从 t 起）
    corr20 = float(np.corrcoef(p20.ravel(), q20.ravel())[0, 1])
    del c20p, f20
    return {"pc1_market_factor_%": round(pc1 * 100, 2),
            "mkt_var_share_%": round(mkt_share * 100, 2),
            "corr_past20_future20": round(corr20, 4)}


def aligned_pool(cal: list[str], limit: int = 1001):
    """严格日期对齐（1000 个联合日历日一天不缺）的真股子集。

    主池 1001 只为凑够样本放宽了对齐 —— 停牌股往后错行，会把横截面共同因子
    稀释掉（实测 PC1 从 50.7% 掉到 7.1%）。市场因子必须在严格对齐子集上量。
    """
    cs = set(cal)
    arrs, codes = [], []
    for f in sorted(bb.REAL_DIR.glob("*.csv")):
        d = pd.read_csv(f, usecols=["date", "open", "close", "high", "low"])
        if str(d["date"].iloc[0]) != cal[0]:
            continue
        d["date"] = d["date"].astype(str)
        d = d[d["date"].isin(cs)].drop_duplicates("date").set_index("date").reindex(cal)
        if d.isna().any().any():
            continue
        a = d[["open", "close", "high", "low"]].to_numpy(float)
        if (a <= 0).any() or not np.isfinite(a).all():
            continue
        if np.max(np.abs(a[1:, 1] / a[:-1, 1] - 1.0)) > bb.EXTREME:
            continue
        arrs.append(a); codes.append(f.stem)
        if len(arrs) >= limit:
            break
    return codes, np.stack(arrs)


def main():
    cal = bb.load_calendar()
    pool = bb.pick_stocks(cal, 1001)
    real = np.stack([a for _, a in pool])

    sets = {
        "REAL(真股)": real,
        "fake(纯随机)": load_dir(HERE / "data", limit=1001),
        "bb L=1": load_dir(HERE / "data_boot_L1"),
        "bb L=5": load_dir(HERE / "data_boot_L5"),
        "bb L=20": load_dir(HERE / "data_boot"),
        "bb L=60": load_dir(HERE / "data_boot_L60"),
    }

    rows = []
    for name, m in sets.items():
        st = bb.quick_stats([x for x in m])
        st.update(diagnose(m))
        st["数据集"] = name
        rows.append(st)

    df = pd.DataFrame(rows).set_index("数据集")
    cols = ["sigma", "skew", "excess_kurtosis", "gap_ratio_pct", "gap_abs_median_pct",
            "gap_abs_p95_pct", "vol_cluster_absr_lag1", "return_lag1", "near_limit_pct",
            "pc1_market_factor_%", "mkt_var_share_%", "corr_past20_future20",
            "ohlc_violations"]
    df = df[cols]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(df.to_string())

    # 市场因子必须在严格对齐子集上量（主池 1001 只因停牌错行被稀释）
    acodes, areal = aligned_pool(cal, 1001)
    extra = []
    for name, d in (("bb L=20", HERE / "data_boot"), ("bb L=60", HERE / "data_boot_L60")):
        m = load_dir(d)
        extra.append((name, float(diagnose(m)["pc1_market_factor_%"])))
    print(f"\n=== 市场共同因子（严格日期对齐子集，{len(acodes)} 只）===")
    print(f"  REAL(真股) PC1 = {diagnose(areal)['pc1_market_factor_%']:.2f}%")
    for name, v in extra:
        print(f"  {name}      PC1 = {v:.2f}%   <- 逐股独立洗牌把个股同涨同跌打没了")
    df.loc["REAL(对齐子集)"] = [np.nan] * len(cols)
    df.loc["REAL(对齐子集)", "pc1_market_factor_%"] = diagnose(areal)["pc1_market_factor_%"]
    df.to_csv(HERE / "three_way_stats.csv")

    # 合并各口径的形态检验结果
    tags = {"fake(纯随机)": "", "bb L=1": "_boot1", "bb L=5": "_boot5",
            "bb L=20": "_boot20", "bb L=60": "_boot60", "REAL(真股)": "_real1001"}
    frames = []
    for name, t in tags.items():
        f = HERE / f"pattern_test{t}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["数据集"] = name
            frames.append(d)
    if frames:
        allp = pd.concat(frames)
        allp.to_csv(HERE / "pattern_test_all.csv", index=False)
        piv = allp.pivot_table(index="形态", columns="数据集",
                               values="5日超额%", aggfunc="first")
        piv2 = allp.pivot_table(index="形态", columns="数据集",
                                values="t_boot(5日)", aggfunc="first")
        print("\n=== 5日超额收益(%) ===")
        print(piv.round(3).to_string())
        print("\n=== t_boot(5日) ===")
        print(piv2.round(2).to_string())
        print("\n显著个数 (p<0.05, 5日):")
        print(allp[allp["p(5日)"] < 0.05].groupby("数据集").size().to_string())
        print("\n已写出 three_way_stats.csv / pattern_test_all.csv")


if __name__ == "__main__":
    main()
