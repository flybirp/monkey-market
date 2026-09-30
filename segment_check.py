#!/usr/bin/env python3
"""
分段稳健性检验：把 1000 天切成若干段，分别统计形态后的 5 日超额收益。

判据：
  * 若某个"技术信号"只在某一段显著、其他段翻号/归零 —— 它捕捉的是那一段的行情周期，
    不是形态本身的信息。
  * 真正的稳定信号应当在所有段同号同向。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import analyze as A

HERE = Path(__file__).parent


def run(tag: str, data_dir: str, days: int, limit: int, segs: int, demean: bool):
    names, o, c, h, l = A.load_matrix(Path(data_dir).expanduser(), limit, days)
    m5 = m20 = None
    if demean:
        m5, m20 = A.market_mean(Path(data_dir).expanduser(), days, limit)
    D = c.shape[1]
    edges = np.linspace(0, D, segs + 1).astype(int)
    print(f"\n===== {tag}｜{len(names)} 只 × {len(names[0]) and D} 天｜{segs} 段 =====")
    print(f"段边界(交易日索引): {list(edges)}")

    rows = {}
    for si in range(segs):
        a, b = edges[si], edges[si + 1]
        chunks = []
        for i in range(0, len(names), 200):
            sl = slice(i, i + 200)
            chunks.append(A.chunk_stats(o[sl, a:b], c[sl, a:b], h[sl, a:b], l[sl, a:b],
                                        None if m5 is None else m5[a:b],
                                        None if m20 is None else m20[a:b]))
        res = A.merge(chunks).set_index("形态")
        rows[f"S{si+1} 超额%"] = res["5日超额%"]
        rows[f"S{si+1} t"] = res["t_boot(5日)"]
    out = A.pd.DataFrame(rows)
    A.pd.set_option("display.width", 220)
    A.pd.set_option("display.max_columns", 60)
    print(out.to_string())
    # 一致性：各段同号的比例
    ex_cols = [c for c in out.columns if c.endswith("超额%")]
    sign = np.sign(out[ex_cols].values)
    consistent = (np.abs(sign.sum(axis=1)) == segs).sum()
    print(f"\n各段同号（方向一致）的形态: {consistent}/{len(out)}")
    out.to_csv(HERE / f"segment_{tag}.csv")


if __name__ == "__main__":
    segs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run("monkey", str(HERE / "data"), 1000, None, segs, False)
    run("real", "~/Documents/mainland_data_2014", 1000, 1600, segs, True)
