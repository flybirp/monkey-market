#!/usr/bin/env python3
"""
猴子市场体检 + 教科书形态检验

A. 体检：OHLC 自洽性、价格连贯性、收益率统计特征（涨跌比、波动率、峰度、连阳连阴）
B. 证伪核心：在 100% 纯随机的数据上跑一遍"教科书形态"识别，检验形态出现后的
   未来 5/20 日超额收益是否显著 ≠ 0。

统计口径要点（否则会自己骗自己）：
  * 形态事件在个股内高度重叠（连续多天满足条件 + 未来窗口重叠），
    朴素 t 检验会把标准误低估好几倍 -> 假阳性。
  * 因此采用【个股聚类 t 检验】：每只股票只贡献 1 个观测（该股形态后超额均值），
    1001 只股票互相独立，标准误用个股间方差估计。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).parent
DATA = HERE / "data"


def load_matrix(data_dir: Path = DATA, limit: int | None = None, days: int | None = None):
    """返回 (names, o, c, h, l)，均为 shape=(S, D) 的 float 矩阵（逐文件读入，低内存）。

    data_dir 可指向 monkey_market/data（随机数据）或 ~/Documents/mainland_data_2014（真实数据）。
    """
    files = sorted(p for p in data_dir.glob("*.csv") if p.name != "sources.csv")
    names, arrs = [], []
    for f in files:
        if limit is not None and len(arrs) >= limit:
            break
        a = pd.read_csv(f, usecols=["date", "open", "close", "high", "low"], nrows=days or 100000)
        if days and len(a) < days:
            continue
        if days and str(a["date"].iloc[0])[:10] != "2014-01-02":   # 只保留从起点就有数据的
            continue
        arrs.append(a[["open", "close", "high", "low"]].to_numpy())
        names.append(f.stem)
    S, D = len(arrs), len(arrs[0])
    mats = {k: np.empty((S, D)) for k in range(4)}
    for j, a in enumerate(arrs):
        for k in range(4):
            mats[k][j] = a[:, k]
    return names, mats[0], mats[1], mats[2], mats[3]


def load_matrix_old():
    files = sorted(DATA.glob("fake*.csv"), key=lambda p: int(p.stem[4:]))
    names = [f.stem for f in files]
    first = pd.read_csv(files[0], usecols=["open", "close", "high", "low"]).to_numpy()
    S, D = len(files), len(first)
    mats = {k: np.empty((S, D)) for k in range(4)}
    for j, f in enumerate(files):
        a = first if j == 0 else pd.read_csv(f, usecols=["open", "close", "high", "low"]).to_numpy()
        for k in range(4):
            mats[k][j] = a[:, k]
    return names, mats[0], mats[1], mats[2], mats[3]


def market_mean(data_dir: Path, days: int, limit: int | None = None):
    """全市场每日未来 5/20 日收益均值（横截面），用于市场中性化。"""
    files = sorted(p for p in data_dir.glob("*.csv") if p.name != "sources.csv")
    s5 = np.zeros(days); n5 = np.zeros(days)
    s20 = np.zeros(days); n20 = np.zeros(days)
    used = 0
    for f in files:
        if limit is not None and used >= limit:
            break
        a = pd.read_csv(f, usecols=["date", "close"], nrows=days)
        if len(a) < days or str(a["date"].iloc[0])[:10] != "2014-01-02":
            continue
        c = a["close"].to_numpy(float)
        f5 = np.full(days, np.nan); f5[:-5] = c[5:] / c[:-5] - 1
        f20 = np.full(days, np.nan); f20[:-20] = c[20:] / c[:-20] - 1
        for src, ss, nn in ((f5, s5, n5), (f20, s20, n20)):
            ok = ~np.isnan(src)
            ss[ok] += src[ok]; nn[ok] += 1
        used += 1
    return np.where(n5 > 0, s5 / np.maximum(n5, 1), 0.0), np.where(n20 > 0, s20 / np.maximum(n20, 1), 0.0)


# ---------------- A. 体检 ----------------
def sanity(o, c, h, l) -> None:
    bad_ohlc = int((h < np.maximum(o, c) - 1e-9).sum() + (l > np.minimum(o, c) + 1e-9).sum())
    bad_link = int((np.abs(o[:, 1:] - c[:, :-1]) > 0.011).sum())
    r = np.diff(np.log(c), axis=1).ravel()
    print("=== A. 数据体检 ===")
    print(f"样本: {c.shape[0]} 只 × {c.shape[1]} 天 = {r.size:,} 个日收益")
    print(f"OHLC 矛盾 (high<max(o,c) 或 low>min(o,c)): {bad_ohlc}")
    print(f"价格断裂 (今日open≠昨收): {bad_link}")
    print(f"上涨天数占比: {(r > 0).mean():.4f}   下跌: {(r < 0).mean():.4f}   平盘: {(r == 0).mean():.4f}")
    print(f"日收益标准差: {r.std(ddof=1) * 100:.3f}%   均值: {r.mean() * 100:+.4f}%")
    print(f"偏度: {pd.Series(r).skew():+.3f}   超额峰度: {pd.Series(r).kurt():+.3f}  (正态=0)")
    # 游程长度（向量化，避免生成 50 万个 Python 数组对象）
    s = np.sign(r)
    chg = np.flatnonzero(np.diff(s) != 0) + 1
    start = np.concatenate([[0], chg])
    lens = np.diff(np.concatenate([start, [s.size]]))
    sgn = s[start]
    up, dn = lens[sgn > 0], lens[sgn < 0]
    print(f"连阳段 {up.size:,} 段, 平均 {up.mean():.2f} 天, 最长 {up.max()} 天")
    print(f"连阴段 {dn.size:,} 段, 平均 {dn.mean():.2f} 天, 最长 {dn.max()} 天")
    print(f"≥5连阳的段数: {(up >= 5).sum():,}   ≥5连阴: {(dn >= 5).sum():,}")
    print("  ↑ '五连阳/七连阴'这种教科书式强趋势，纯抛硬币里照样成批出现\n")


# ---------------- B. 形态识别 ----------------
def _shift1(x):
    out = np.full_like(x, np.nan)
    out[:, 1:] = x[:, :-1]
    return out


def _rmean(x, w):
    cs = np.cumsum(np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1), axis=1)
    out = np.full_like(x, np.nan)
    out[:, w - 1:] = (cs[:, w:] - cs[:, :-w]) / w
    return out


def _rmax(x, w):
    out = np.full_like(x, np.nan)
    out[:, w - 1:] = sliding_window_view(x, w, axis=1).max(axis=-1)
    return out


def _rmin(x, w):
    out = np.full_like(x, np.nan)
    out[:, w - 1:] = sliding_window_view(x, w, axis=1).min(axis=-1)
    return out


def patterns(o, c, h, l) -> dict[str, np.ndarray]:
    body = np.abs(c - o)
    rng = np.where(h - l <= 0, np.nan, h - l)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    po, pc = _shift1(o), _shift1(c)
    ma5, ma20 = _rmean(c, 5), _rmean(c, 20)
    up, dn = (c > o), (c < o)
    u1 = np.pad(up[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    u2 = np.pad(up[:, :-2], ((0, 0), (2, 0)), constant_values=False)
    d1 = np.pad(dn[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    d2 = np.pad(dn[:, :-2], ((0, 0), (2, 0)), constant_values=False)
    p = {
        "十字星 Doji": body <= 0.1 * rng,
        "锤子线 Hammer": (lower >= 2 * body) & (upper <= body) & (body > 0),
        "射击之星": (upper >= 2 * body) & (lower <= body) & (body > 0),
        "看涨吞没": (pc < po) & up & (c >= po) & (o <= pc),
        "看跌吞没": (pc > po) & dn & (c <= po) & (o >= pc),
        "三连阳(第3根)": up & u1 & u2,
        "三连阴(第3根)": dn & d1 & d2,
        "MA5上穿MA20(金叉)": (ma5 > ma20) & (_shift1(ma5) <= _shift1(ma20)),
        "MA5下穿MA20(死叉)": (ma5 < ma20) & (_shift1(ma5) >= _shift1(ma20)),
        "突破20日新高": c >= _shift1(_rmax(c, 20)),
        "跌破20日新低": c <= _shift1(_rmin(c, 20)),
    }
    return {k: np.nan_to_num(v.astype(float), nan=0.0).astype(bool) for k, v in p.items()}


def _cluster_t(d: np.ndarray):
    """个股聚类 t 检验：1001 只股票 = 1001 个独立观测。"""
    d = d[~np.isnan(d)]
    if d.size < 30:
        return float("nan"), float("nan")
    se = d.std(ddof=1) / math.sqrt(d.size)
    if se == 0:
        return float("nan"), float("nan")
    t = d.mean() / se
    return t, math.erfc(abs(t) / math.sqrt(2))


def chunk_stats(o, c, h, l, m5=None, m20=None) -> dict:
    """一批股票（~200 只）的统计中间量，避免全量矩阵把内存打爆。

    对每只股票记录 4 个量（按事件加权，无偏）：
      sA/nA = 形态命中样本的未来收益和 / 个数； sAll/nAll = 该股票全部样本的和 / 个数
    """
    S, D = c.shape
    f5 = np.full((S, D), np.nan); f5[:, :-5] = c[:, 5:] / c[:, :-5] - 1
    f20 = np.full((S, D), np.nan); f20[:, :-20] = c[:, 20:] / c[:, :-20] - 1
    if m5 is not None:      # 横截面市场中性化：剔除"整个市场当天在涨/跌"的时序效应
        f5 = f5 - m5[None, :]
        f20 = f20 - m20[None, :]
    ok5, ok20 = ~np.isnan(f5), ~np.isnan(f20)
    g5, g20 = np.where(ok5, f5, 0.0), np.where(ok20, f20, 0.0)

    out = {}
    for k, M in patterns(o, c, h, l).items():
        m5, m20 = M & ok5, M & ok20
        a5, a20 = np.where(m5, f5, 0.0), np.where(m20, f20, 0.0)
        out[k] = dict(
            cnt=int(M.sum()), total=int(M.size),
            sA5=a5.sum(1), nA5=m5.sum(1).astype(float),
            sA20=a20.sum(1), nA20=m20.sum(1).astype(float),
            sAll5=g5.sum(1), nAll5=ok5.sum(1).astype(float),
            sAll20=g20.sum(1), nAll20=ok20.sum(1).astype(float),
            # 朴素口径用的：命中样本的收益数组（用于方差）
            vA5=(a5.sum(1), (np.where(m5, f5 ** 2, 0.0)).sum(1)),
            vA20=(a20.sum(1), (np.where(m20, f20 ** 2, 0.0)).sum(1)),
        )
        del m5, m20, a5, a20
    return out


def _boot(sA, nA, sAll, nAll, B=2000, seed=42):
    """按个股 bootstrap（个股间独立，个股内重叠相关被整只吸收）。"""
    S = len(sA)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(B, S))
    sa = sA[idx].sum(1); na = nA[idx].sum(1)
    sb = sAll[idx].sum(1); nb = nAll[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        th = sa / na - sb / nb
    th = th[np.isfinite(th)]
    return th.mean(), th.std(ddof=1)


def merge(chunks: list[dict]) -> pd.DataFrame:
    rows = []
    for k in chunks[0]:
        cat = lambda f: np.concatenate([x[k][f] for x in chunks])  # noqa: E731
        sA5, nA5 = cat("sA5"), cat("nA5")
        sA20, nA20 = cat("sA20"), cat("nA20")
        sAll5, nAll5 = cat("sAll5"), cat("nAll5")
        sAll20, nAll20 = cat("sAll20"), cat("nAll20")
        cnt = int(sum(x[k]["cnt"] for x in chunks))
        total = int(sum(x[k]["total"] for x in chunks))
        v5 = np.concatenate([x[k]["vA5"][1] for x in chunks])
        v20 = np.concatenate([x[k]["vA20"][1] for x in chunks])

        def report(sA, nA, sAll, nAll, v):
            theta = sA.sum() / nA.sum() - sAll.sum() / nAll.sum()      # 事件加权超额
            mu, se = _boot(sA, nA, sAll, nAll)
            t = theta / se if se > 0 else float("nan")
            p = math.erfc(abs(t) / math.sqrt(2))
            n = nA.sum()
            m = sA.sum() / n
            sd = math.sqrt(max(v.sum() / n - m ** 2, 1e-18))
            t_naive = (m - sAll.sum() / nAll.sum()) / (sd / math.sqrt(n))  # 把每个事件当独立 -> 虚高
            return theta, t, p, t_naive

        e5, t5, p5, tn5 = report(sA5, nA5, sAll5, nAll5, v5)
        e20, t20, p20, tn20 = report(sA20, nA20, sAll20, nAll20, v20)
        rows.append({
            "形态": k,
            "出现次数": cnt,
            "占比%": round(cnt / total * 100, 2),
            "5日超额%": round(e5 * 100, 4),
            "t_boot(5日)": round(t5, 2),
            "p(5日)": round(p5, 3),
            "t朴素(5日)": round(tn5, 2),
            "20日超额%": round(e20 * 100, 4),
            "t_boot(20日)": round(t20, 2),
            "p(20日)": round(p20, 3),
            "t朴素(20日)": round(tn20, 2),
        })
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DATA), help="数据目录（默认随机数据）")
    ap.add_argument("--limit", type=int, default=None, help="最多用多少只")
    ap.add_argument("--days", type=int, default=None, help="每只取多少天（真实数据用 1000）")
    ap.add_argument("--tag", type=str, default="", help="输出文件后缀")
    ap.add_argument("--demean", action="store_true", help="横截面市场中性化")
    a = ap.parse_args()
    names, o, c, h, l = load_matrix(Path(a.data).expanduser(), a.limit, a.days)
    sanity(o, c, h, l)
    print("=== B. 教科书形态预测力检验（数据 100% 由抛硬币生成）===")
    print("口径：事件加权超额收益，标准误按【个股 bootstrap】估计（个股内重叠相关被整只吸收）\n")
    m5 = m20 = None
    if a.demean:
        m5, m20 = market_mean(Path(a.data).expanduser(), c.shape[1], a.limit)
        print(f"已做横截面市场中性化（每日全市场 {int(a.limit or 0)} 只去均值）")
    chunks = []
    B = 200
    for i in range(0, len(names), B):
        sl = slice(i, i + B)
        chunks.append(chunk_stats(o[sl], c[sl], h[sl], l[sl], m5, m20))
        print(f"  chunk {i}-{min(i + B, len(names))} done", flush=True)
    res = merge(chunks)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print()
    print(res.to_string(index=False))
    out_csv = HERE / f"pattern_test{a.tag}.csv"
    res.to_csv(out_csv, index=False)
    sig5 = int((res["p(5日)"] < 0.05).sum())
    sig20 = int((res["p(20日)"] < 0.05).sum())
    naive5 = int((res["t朴素(5日)"].abs() > 1.96).sum())
    naive20 = int((res["t朴素(20日)"].abs() > 1.96).sum())
    print(f"\n正确口径(bootstrap)显著: 5日 {sig5}/{len(res)}   20日 {sig20}/{len(res)}   期望假阳性 {len(res) * 0.05:.1f}")
    print(f"朴素口径(把重叠事件当独立)显著: 5日 {naive5}/{len(res)}  20日 {naive20}/{len(res)}  <- 这就是绝大多数'技术信号'的来源")
    print(f"明细: {out_csv}")


if __name__ == "__main__":
    main()
