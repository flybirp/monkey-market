#!/usr/bin/env python3
"""从真实 A 股数据提取交易日历：2014-01-02 起的 N 个交易日。

取多只股票日期的并集，避免单只股票停牌导致日历缺日。
输出：trading_calendar.csv (date,)
"""
import sys
from pathlib import Path

SRC = Path.home() / "Documents" / "mainland_data_2014"
START = "2014-01-02"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = Path(__file__).parent / "trading_calendar.csv"


def main():
    files = sorted(SRC.glob("*.csv"))
    if not files:
        raise SystemExit(f"找不到数据: {SRC}")
    # 取前 400 只股票（覆盖度足够，且都是 2014 年前上市的老股）
    sample = files[:400]
    dates = set()
    for f in sample:
        with f.open() as fh:
            next(fh, None)
            for line in fh:
                d = line.split(",", 1)[0].strip()
                if d >= START:
                    dates.add(d)
    cal = sorted(dates)[:N]
    if len(cal) < N:
        raise SystemExit(f"日历不足: 只有 {len(cal)} 个交易日 (需要 {N})")
    OUT.write_text("date\n" + "\n".join(cal) + "\n")
    print(f"交易日 {len(cal)} 个: {cal[0]} ~ {cal[-1]}  -> {OUT}")


if __name__ == "__main__":
    main()
