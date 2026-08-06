[README.md](https://github.com/user-attachments/files/30771568/README.md)
# Alpha Signal Research: Short-Term Momentum/Reversal in AAPL

A small, self-contained project that walks through the full quant research
lifecycle on real daily equity data: idea generation → hypothesis testing →
signal construction → backtesting → honest evaluation of what the results
actually mean. 

**This is a small, honest study — not a "the strategy works
great" showcase.** The interesting part is what happens when a
statistically significant signal doesn't survive contact with a real
trading rule, and why.

## Data

Real daily OHLCV data for AAPL, Feb 2015 – Feb 2017 (506 trading days),
from a public dataset. Single-name, single-asset — see **Limitations**
below for how this would extend to a proper cross-sectional, multi-asset
study with a real data vendor.

## Research process

**H1 (short-term reversal):** After an unusually large 5-day price move,
AAPL mean-reverts over the next 5 days. This is a well-documented
microstructure effect and a reasonable first hypothesis.

- Tested via the Spearman Information Coefficient (IC) between a
  z-scored 5-day momentum signal and the 5-day forward return.
- **Result: IC = +0.097 (t = 2.03, p = 0.043) — the opposite sign from
  what H1 predicts.** H1 is rejected. The data instead points to weak
  short-term *momentum continuation*, not reversal, over this horizon.

**H2 (momentum continuation), formed from the H1 result:** 5-day
momentum continues over the next 5 days.

- Same IC, now interpreted correctly: mild, statistically significant
  positive relationship (p < 0.05).
- **Important caveat, stated deliberately rather than glossed over:** H2
  is tested on the same sample used to reject H1. That's in-sample
  re-specification and it inflates apparent significance. A real
  research process would validate H2 on a fresh, held-out sample (a
  different time period or a different name) before trusting it. I did
  not do that here — this is flagged as the single biggest weakness of
  the analysis, on purpose.
- The rolling 60-day IC (`rolling_ic.png`) makes this concrete:
  the "significant" full-sample IC is driven mostly by one strong regime
  in mid-2016, not a stable effect across the whole sample. A real
  research process would treat that as a yellow flag, not a green light.

## From signal to strategy

Rule: go long when the z-scored signal > +1, short when < −1, flat
otherwise; hold each position for exactly 5 trading days (matched to the
horizon the signal is actually about), non-overlapping; 5bps round-trip
cost per trade.

**Result:** the strategy underperforms buy-and-hold (−3.1% vs +3.1%
annualized) despite a 55% trade-level hit rate — average return per trade
is slightly negative once losses (which run larger than wins) and costs
are netted out. 47 trades total.

**Why this matters more than a clean "it works" result:** a
statistically significant IC is a necessary but not sufficient condition
for a profitable strategy. Here, a modest, regime-dependent signal
did not survive translation into a simple, realistic trading rule. That
gap — and knowing how to diagnose it — is a core part of the job, not a
failure of the exercise.

## Limitations & what I'd do next with real infrastructure

- **Single name, single regime.** 2 years of one stock is not enough to
  separate a real effect from noise (N = 47 trades). Next step: run the
  same test across the S&P 500 cross-sectionally and pool the IC.
- **In-sample H2.** Needs a proper train/test split or walk-forward
  validation before it's trustworthy.
- **No regime or volatility conditioning.** The rolling IC plot suggests
  the effect isn't stable — worth testing whether it's conditional on
  volatility regime, earnings proximity, or macro state.
- **Simple cost model.** 5bps flat is a simplification; real
  implementation would model market impact and slippage more carefully.
- **Data.** This environment doesn't have access to a market data vendor,
  so I used a small public dataset. With real infrastructure (e.g. a
  vendor feed + point-in-time universe), this would be redone
  cross-sectionally across hundreds of names.

## Files

```
research.py                          # full pipeline, run with: python3 research.py
aapl.csv                        # raw input data
performance_summary.csv      # strategy vs buy-and-hold stats
hypothesis_test.txt          # H1 and H2 IC/t-stat/p-value results
trade_log.csv                # every individual trade
research_data.csv            # full signal/return dataset
equity_curve.png             # strategy vs benchmark
rolling_ic.png               # signal stability over time
signal_vs_forward_return.png # scatter of signal vs outcome
```

Run it yourself: `pip install pandas numpy scipy matplotlib && python3 research.py`
