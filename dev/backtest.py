import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from model import forecast_market, baseline_forecast_market, QUANTILES

KIT = os.path.join(os.path.dirname(__file__), "..")


def pinball(y, q):
    tot = 0.0
    for tau, qi in zip(QUANTILES, q):
        d = y - qi
        tot += max(tau * d, (tau - 1) * d)
    return 100.0 * tot / len(QUANTILES)


def eval_market(prices, horizon, split_offset):
    """split_offset: how many minutes before the end to cut training."""
    n = len(prices)
    if split_offset >= n:
        return None
    test_len = min(horizon, split_offset)
    if test_len < 1:
        return None
    train = prices[: n - split_offset]
    actual = prices[n - split_offset: n - split_offset + test_len]
    if len(train) < 5:
        return None
    mine_rows = forecast_market(train, test_len)
    base_rows = baseline_forecast_market(train, test_len)
    mine_loss = 0.0
    base_loss = 0.0
    mine_width = 0.0
    base_width = 0.0
    mine_cov = 0
    base_cov = 0
    cnt = 0
    for h in range(test_len):
        y = actual[h]
        mine_loss += pinball(y, mine_rows[h])
        base_loss += pinball(y, base_rows[h])
        mine_width += mine_rows[h][4] - mine_rows[h][0]
        base_width += base_rows[h][4] - base_rows[h][0]
        if mine_rows[h][0] <= y <= mine_rows[h][4]:
            mine_cov += 1
        if base_rows[h][0] <= y <= base_rows[h][4]:
            base_cov += 1
        cnt += 1
    return mine_loss / cnt, base_loss / cnt, cnt, mine_width / cnt, base_width / cnt, mine_cov / cnt, base_cov / cnt


def main():
    cases = ["p00", "p01", "p02", "p03", "p04", "p05"]
    splits = [240, 480, 720]  # multiple walk-forward cut points, minutes before end
    total_mine = 0.0
    total_base = 0.0
    total_n = 0
    total_mw = total_bw = total_mc = total_bc = 0.0
    per_case = {}
    for c in cases:
        with open(os.path.join(KIT, "input", f"{c}.in")) as f:
            d = json.load(f)
        horizon = d["horizon_minutes"]
        case_mine = 0.0
        case_base = 0.0
        case_n = 0
        case_mw = case_bw = case_mc = case_bc = 0.0
        for m in d["markets"]:
            prices = m["prices"]
            for off in splits:
                r = eval_market(prices, horizon, off)
                if r is None:
                    continue
                ml, bl, cnt, mw, bw, mc, bc = r
                case_mine += ml * cnt
                case_base += bl * cnt
                case_mw += mw * cnt
                case_bw += bw * cnt
                case_mc += mc * cnt
                case_bc += bc * cnt
                case_n += cnt
        per_case[c] = (case_mine / case_n if case_n else None, case_base / case_n if case_n else None, case_n,
                        case_mw / case_n, case_bw / case_n, case_mc / case_n, case_bc / case_n)
        total_mine += case_mine
        total_base += case_base
        total_mw += case_mw
        total_bw += case_bw
        total_mc += case_mc
        total_bc += case_bc
        total_n += case_n

    print(f"{'case':6} {'mine':>10} {'base':>10} {'n':>8} {'mw':>8} {'bw':>8} {'mcov':>6} {'bcov':>6}")
    for c in cases:
        m, b, n, mw, bw, mc, bc = per_case[c]
        print(f"{c:6} {m:10.5f} {b:10.5f} {n:8d} {mw:8.4f} {bw:8.4f} {mc:6.3f} {bc:6.3f}")
    print("---")
    print(f"TOTAL mine={total_mine/total_n:.6f} base={total_base/total_n:.6f} n={total_n}")
    print(f"width mine={total_mw/total_n:.4f} base={total_bw/total_n:.4f} cov_mine={total_mc/total_n:.3f} cov_base={total_bc/total_n:.3f}")
    print(f"improvement: {(1 - (total_mine/total_n)/(total_base/total_n))*100:.2f}%")


if __name__ == "__main__":
    main()
