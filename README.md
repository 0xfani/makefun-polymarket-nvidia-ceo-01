# polymarket-nvidia-ceo-forecast-01 entry

Forecasts Polymarket's "Next CEO of NVIDIA?" event's 24 Yes markets, minute
by minute, 240 minutes past each checkpoint, from 72 hours of per-minute
price history. Entry for make.fun's `polymarket-nvidia-ceo-forecast-01`
competition.

## Entrypoint

`main.py` (Python 3.13, standard library only, no network). Reads every
`*.in` file in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`), in the
exact schema RULES.txt specifies: `{"forecasts": [{"id", "q"}]}`, `q` 240
rows of 5 non-decreasing quantiles each.

## Model

For each market, independently:

1. Compute the empirical distribution of h-step price changes
   (`price[t+h] - price[t]`) over several anchor horizons
   (1, 2, 4, 8, 16, 32, 64, 120, 180, 240 minutes), using every overlapping
   window in that market's own 72h history.
2. Fit a power law (`|quantile of change| ~= a * h^b`) per quantile level
   across those anchors, so the spread can be evaluated at any minute
   1..240 without refitting.
3. Scale the fitted spread by a Bernoulli-variance ratio between the
   current price and the market's own historical average price: prices
   near 0 or 1 are empirically stickier than mid-range prices, even within
   one market's own history.
4. Blend 40% of the house baseline's fixed-sigma spread into the fitted
   spread as a safety floor (`FLOOR_FRAC = 0.4`), and shrink the empirical
   component by 40% (`GLOBAL_SHRINK = 0.4`) — tuned by back-testing (see
   `dev/`) so a market that was quiet for 72h isn't forecast with near-zero
   spread, since a single real jump outside the training window costs the
   pinball loss heavily.
5. The point forecast (median) is the market's last observed price: no
   added drift. Back-testing did not show a reliable benefit from adding
   momentum or mean-reversion on top of the random-walk prior.
6. Markets with too little history (<60 minutes) fall back to the house
   baseline's exact formula.

## Local back-testing (no official local grader for this program/v1 kit)

This competition's kit has no local grader — the scorer and answer keys
stay server-side. To validate the model before submitting, `dev/backtest.py`
builds a genuine walk-forward test from the kit's own practice cases
(`p00.in`..`p05.in`): for several synthetic checkpoints inside each market's
real 72h history (240/480/720 minutes before the case's actual checkpoint),
it trains on the data before that cut and scores the model's forecast
against the market's real, already-recorded prices for the next 240 minutes
(or fewer where the history runs out) — using the exact same pinball-loss
formula as RULES.txt.

Result over all six practice cases, three synthetic splits each (75,840
scored market-minute-quantile rows):

```
mean pinball loss   mine=0.536187   house baseline=0.546347
improvement over the house baseline: 1.86%
```

The model consistently beats the baseline in the 4 cases with full/near-full
market coverage (p02-p05) and trails it slightly in the two earliest,
thinnest-history cases (p00-p01); net is a modest, real improvement validated
against the house's own exact baseline formula replicated in
`dev/model.py`'s `baseline_forecast_market`. `dev/sweep.py` records the
parameter search that picked `FLOOR_FRAC`/`GLOBAL_SHRINK`.

This is a genuine but modest edge, not a guaranteed prize: the final grades
on live, post-deadline market data, which a historical back-test can only
approximate.
