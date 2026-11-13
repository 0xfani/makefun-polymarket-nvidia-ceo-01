import json
import math
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(__file__))
KIT = os.path.join(os.path.dirname(__file__), "..")
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def pinball(y, q):
    tot = 0.0
    for tau, qi in zip(QUANTILES, q):
        d = y - qi
        tot += max(tau * d, (tau - 1) * d)
    return 100.0 * tot / len(QUANTILES)


def eval_market(model_mod, prices, horizon, split_offset):
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
    mine_rows = model_mod.forecast_market(train, test_len)
    mine_loss = 0.0
    cnt = 0
    for h in range(test_len):
        y = actual[h]
        mine_loss += pinball(y, mine_rows[h])
        cnt += 1
    return mine_loss, cnt


def run(model_mod, cases, splits):
    total = 0.0
    total_n = 0
    for c in cases:
        with open(os.path.join(KIT, "input", f"{c}.in")) as f:
            d = json.load(f)
        horizon = d["horizon_minutes"]
        for m in d["markets"]:
            prices = m["prices"]
            for off in splits:
                r = eval_market(model_mod, prices, horizon, off)
                if r is None:
                    continue
                ml, cnt = r
                total += ml
                total_n += cnt
    return total / total_n


if __name__ == "__main__":
    import model as model_mod
    cases = ["p00", "p01", "p02", "p03", "p04", "p05"]
    splits = [240, 480, 720]

    for floor_frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for shrink in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            model_mod.FLOOR_FRAC = floor_frac
            model_mod.GLOBAL_SHRINK = shrink
            loss = run(model_mod, cases, splits)
            print(f"floor={floor_frac} shrink={shrink} -> loss={loss:.6f}")
