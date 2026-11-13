"""
Shared forecasting model (pure stdlib). Used by both main.py (the entry) and
backtest.py (dev-only evaluation). Kept in one file so backtest results
reflect exactly what the entry runs.
"""
import math

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
BASE_SIGMA = 0.0025  # house baseline's per-sqrt-minute spread, used as a floor/fallback

HORIZON_ANCHORS = [1, 2, 4, 8, 16, 32, 64, 120, 180, 240]
MIN_SAMPLES_PER_ANCHOR = 20
MIN_HISTORY_FOR_EMPIRICAL = 60  # minutes; below this, just use the baseline shape
FLOOR_FRAC = 0.4  # blend-in fraction of the baseline's spread (safety floor)
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


def build_market_model(prices):
    """
    Returns a callable spread_fn(h) -> list of 5 deltas-from-last-price
    (not yet clamped/rounded), for minute h = 1..horizon.
    Falls back to the baseline's shape when there isn't enough own history.
    """
    n = len(prices)
    x0 = prices[-1]

    if n < MIN_HISTORY_FOR_EMPIRICAL:
        def fallback(h):
            return [z * BASE_SIGMA * math.sqrt(h) for z in Z]
        return fallback

    # level-stickiness: scale empirical spread by the ratio of the current price's
    # Bernoulli variance to the history's average Bernoulli variance (prices near
    # 0/1 move less than prices near 0.5, even within one market's own history).
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
        qs = [_empirical_quantile(diffs, t) for t in QUANTILES]
        spread_by_h[h] = qs
        anchors.append(h)

    if len(anchors) < 2:
        def fallback(h):
            return [z * BASE_SIGMA * math.sqrt(h) * vol_ratio for z in Z]
        return fallback

    fits = []
    for k in range(5):
        ys = [spread_by_h[h][k] for h in anchors]
        fits.append(_fit_power(anchors, ys))

    max_anchor = max(anchors)

    def spread_fn(h):
        out = []
        for k in range(5):
            fit = fits[k]
            if fit is None:
                out.append(0.0)
                continue
            a, b, sign = fit
            # don't extrapolate the power-law exponent too aggressively past the
            # longest anchor we actually measured; cap effective h for the exponent.
            heff = min(h, max_anchor * 2.5)
            val = sign * a * (heff ** b)
            out.append(val * vol_ratio * GLOBAL_SHRINK)
        out.sort()
        # safety floor: blend in a fraction of the baseline's spread so a market
        # that has been dead quiet for 72h isn't forecast as having zero spread
        # (a single real jump outside the training window would then cost a lot).
        floor = [z * BASE_SIGMA * math.sqrt(h) for z in Z]
        floor_frac = FLOOR_FRAC
        blended = [(1 - floor_frac) * out[i] + floor_frac * floor[i] for i in range(5)]
        blended.sort()
        return blended

    return spread_fn


def forecast_market(prices, horizon):
    x0 = prices[-1]
    spread_fn = build_market_model(prices)
    rows = []
    for h in range(1, horizon + 1):
        deltas = spread_fn(h)
        row = sorted(price6(x0 + d) for d in deltas)
        rows.append(row)
    return rows


def baseline_forecast_market(prices, horizon):
    x0 = prices[-1]
    rows = []
    for h in range(1, horizon + 1):
        row = [price6(x0 + z * BASE_SIGMA * math.sqrt(h)) for z in Z]
        rows.append(row)
    return rows
