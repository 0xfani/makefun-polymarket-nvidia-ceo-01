#!/usr/bin/env python3
"""
polymarket-nvidia-ceo-forecast-01 entry.

Forecasts each market's Yes-price quantiles for the 240 minutes after the
case's checkpoint from its own 72h per-minute price history, offline,
Python 3.13 stdlib only.

Model (own-market empirical quantile bands, in price space):
  - For each market, compute the empirical distribution of h-step price
    changes (price[t+h] - price[t]) over several anchor horizons h, using
    every overlapping window in the market's own history.
  - Fit a power law (|change quantile| ~ a * h^b) per quantile level across
    those anchors, so the spread can be evaluated at every minute 1..240.
  - Scale the spread by how close the current price sits to 0/1 (a
    Bernoulli-variance ratio against the market's own historical average
    price), since prices near the boundary are stickier than mid-range ones.
  - Blend a fraction of this empirical spread with the house baseline's
    fixed-sigma random-walk spread as a safety floor, so a market that has
    been quiet for 72h is not forecast with near-zero spread (a single real
    jump the training window never saw would then cost heavily).
  - The point forecast (median) is simply the last observed price: no drift,
    consistent with an efficient-market / random-walk prior, which
    back-testing (see scratch/backtest.py in this repo's git history) did
    not show reliably improved on.

This file is deliberately self-contained (no imports from this repo) so the
grader's read-only /bot mount with only main.py present still runs cleanly.
"""
import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
BASE_SIGMA = 0.0025  # house baseline's per-sqrt-minute spread; used as a safety floor

HORIZON_ANCHORS = [1, 2, 4, 8, 16, 32, 64, 120, 180, 240]
MIN_SAMPLES_PER_ANCHOR = 20
MIN_HISTORY_FOR_EMPIRICAL = 60  # minutes; below this, just use the baseline shape
FLOOR_FRAC = 0.4   # blend-in fraction of the baseline's spread (safety floor)
GLOBAL_SHRINK = 0.4  # overall multiplier on the empirical (non-floor) spread


def price6(x):
    x = min(1.0, max(0.0, x))
    return round(x * 1e6) / 1e6 + 0.0


def _empirical_quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _fit_power(xs, ys):
    """Fit |y| = a * x^b in log-log space; returns (a, b, sign) or None."""
    pts = [(x, y) for x, y in zip(xs, ys) if abs(y) > 1e-9 and x > 0]
    if len(pts) < 2:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(abs(y)) for _, y in pts]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    den = sum((lx[i] - mx) ** 2 for i in range(n)) or 1e-9
    b = num / den
    a = math.exp(my - b * mx)
    sign = 1.0 if sum(y for _, y in pts) >= 0 else -1.0
    return a, b, sign


def _bern_var(p):
    return max(p * (1 - p), 1e-6)


def build_spread_fn(prices):
    """Returns spread_fn(h) -> 5 deltas-from-last-price for minute h (1-indexed)."""
    n = len(prices)
    x0 = prices[-1]

    if n < MIN_HISTORY_FOR_EMPIRICAL:
        def fallback(h):
            return [z * BASE_SIGMA * math.sqrt(h) for z in Z]
        return fallback

    avg_p = sum(prices) / n
    hist_var = _bern_var(avg_p)
    cur_var = _bern_var(x0)
    vol_ratio = math.sqrt(cur_var / hist_var) if hist_var > 0 else 1.0
    vol_ratio = min(max(vol_ratio, 0.15), 4.0)

    anchors = []
    spread_by_h = {}
    for h in HORIZON_ANCHORS:
        if h >= n:
            continue
        diffs = [prices[i + h] - prices[i] for i in range(n - h)]
        if len(diffs) < MIN_SAMPLES_PER_ANCHOR:
            continue
        diffs.sort()
        spread_by_h[h] = [_empirical_quantile(diffs, t) for t in QUANTILES]
        anchors.append(h)

    if len(anchors) < 2:
        def fallback(h):
            return [z * BASE_SIGMA * math.sqrt(h) * vol_ratio for z in Z]
        return fallback

    fits = [_fit_power(anchors, [spread_by_h[h][k] for h in anchors]) for k in range(5)]
    max_anchor = max(anchors)

    def spread_fn(h):
        out = []
        for k in range(5):
            fit = fits[k]
            if fit is None:
                out.append(0.0)
                continue
            a, b, sign = fit
            heff = min(h, max_anchor * 2.5)  # don't extrapolate the exponent too far
            out.append(sign * a * (heff ** b) * vol_ratio * GLOBAL_SHRINK)
        out.sort()
        floor = [z * BASE_SIGMA * math.sqrt(h) for z in Z]
        blended = [(1 - FLOOR_FRAC) * out[i] + FLOOR_FRAC * floor[i] for i in range(5)]
        blended.sort()
        return blended

    return spread_fn


def forecast_market(prices, horizon):
    x0 = prices[-1]
    spread_fn = build_spread_fn(prices)
    rows = []
    for h in range(1, horizon + 1):
        deltas = spread_fn(h)
        row = sorted(price6(x0 + d) for d in deltas)
        rows.append(row)
    return rows


def baseline_forecast_market(prices, horizon):
    """Pure house-baseline shape, used as a last-resort fallback if the
    empirical model raises for some unexpected input."""
    x0 = prices[-1]
    rows = []
    for h in range(1, horizon + 1):
        rows.append(sorted(price6(x0 + z * BASE_SIGMA * math.sqrt(h)) for z in Z))
    return rows


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in files:
        case_id = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = case["horizon_minutes"]
        quantiles = case["quantiles"]
        if quantiles != QUANTILES:
            raise ValueError(f"{fname} asks for quantiles {quantiles}, not {QUANTILES}")
        forecasts = []
        for m in case["markets"]:
            prices = m["prices"]
            if not prices:
                raise ValueError(f"market {m['id']} has no prices")
            # A crash in the empirical model for one odd market must not
            # invalidate the whole case (an invalid case is scored as if
            # every quantile were 0.5 and disqualifies the entry from any
            # prize): fall back to the exact house baseline formula instead.
            try:
                rows = forecast_market(prices, horizon)
            except Exception:
                rows = baseline_forecast_market(prices, horizon)
            forecasts.append({"id": m["id"], "q": rows})
        out_path = os.path.join(OUTPUT_DIR, f"{case_id}.out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}))
            f.write("\n")


if __name__ == "__main__":
    main()
