"""
Alpha Signal Research: Short-Term Mean-Reversion in AAPL Daily Returns
========================================================================
A small, self-contained research project that follows the full quant
research lifecycle described in a typical junior quant researcher role:

  1. Idea generation      -> hypothesis stated explicitly below
  2. Data analysis         -> exploratory stats on real daily OHLCV data
  3. Hypothesis testing    -> Information Coefficient (IC) + t-stats
  4. Alpha discovery        -> signal construction & selection
  5. Trading strategy       -> simple rule-based long/flat portfolio
  6. Backtesting            -> vectorized backtest with realistic costs
  7. Portfolio analysis     -> Sharpe, drawdown, hit rate, benchmark compare

Data: real daily AAPL OHLCV (2015-2016, ~2 years, 507 trading days),
sourced from a public dataset. This is a single-name, single-asset study;
see README.md for how it would extend to a cross-sectional, multi-asset
version in a production research environment.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 7
np.random.seed(RNG_SEED)

OUT = "outputs"
import os
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv("data/aapl.csv", parse_dates=["Date"]).sort_values("Date")
df = df.rename(columns={
    "AAPL.Open": "open", "AAPL.High": "high", "AAPL.Low": "low",
    "AAPL.Close": "close", "AAPL.Volume": "volume", "AAPL.Adjusted": "adj_close",
})[["Date", "open", "high", "low", "close", "volume", "adj_close"]].reset_index(drop=True)

df["ret_1d"] = df["adj_close"].pct_change()

print(f"Loaded {len(df)} trading days: {df.Date.min().date()} -> {df.Date.max().date()}")

# ----------------------------------------------------------------------
# 2. HYPOTHESIS (H1)
# ----------------------------------------------------------------------
# H1: After a sharp short-term price move (5-day return far from its
#     recent norm), AAPL exhibits short-horizon mean reversion --
#     i.e. the z-scored 5-day return NEGATIVELY predicts the next
#     5-day forward return.
#
# This is a classic, well-documented equity microstructure effect
# (short-term reversal) and a reasonable first hypothesis to test
# with limited single-name data before committing to more complex
# signal research.
# ----------------------------------------------------------------------

LOOKBACK = 5        # signal formation window (trading days)
HORIZON = 5          # forward return window to predict (trading days)
ZSCORE_WINDOW = 60   # rolling window to standardize the signal

df["mom_5d"] = df["adj_close"].pct_change(LOOKBACK)
roll_mean = df["mom_5d"].rolling(ZSCORE_WINDOW).mean()
roll_std = df["mom_5d"].rolling(ZSCORE_WINDOW).std()
df["mom_5d_z"] = (df["mom_5d"] - roll_mean) / roll_std

df["fwd_ret_5d"] = df["adj_close"].shift(-HORIZON) / df["adj_close"] - 1

research = df.dropna(subset=["mom_5d_z", "fwd_ret_5d"]).reset_index(drop=True)
print(f"Usable research sample after warm-up/lookahead trim: {len(research)} obs")

# ----------------------------------------------------------------------
# 3. HYPOTHESIS TEST: INFORMATION COEFFICIENT
# ----------------------------------------------------------------------
# H1 predicts a NEGATIVE IC between mom_5d_z and fwd_ret_5d (reversal).
ic_h1, ic_h1_pvalue = stats.spearmanr(research["mom_5d_z"], research["fwd_ret_5d"])
n = len(research)
t_stat_h1 = ic_h1 * np.sqrt((n - 2) / (1 - ic_h1**2))

print("\n--- H1 Test (short-term reversal): mom_5d_z vs 5-day Forward Return ---")
print(f"Spearman IC:        {ic_h1:.4f}   (H1 predicts this should be NEGATIVE)")
print(f"IC t-stat:           {t_stat_h1:.2f}")
print(f"p-value:              {ic_h1_pvalue:.4f}")
print(f"N observations:       {n}")

if ic_h1 < 0:
    verdict_h1 = "IC is negative, as H1 predicts -> reversal hypothesis SUPPORTED."
else:
    verdict_h1 = ("IC is POSITIVE, the opposite sign from H1's prediction -> reversal "
                  "hypothesis REJECTED. The data instead suggests short-term MOMENTUM "
                  "CONTINUATION over this horizon, not reversal.")
print(f"Verdict: {verdict_h1}")

# ----------------------------------------------------------------------
# 3b. REVISED HYPOTHESIS (H2), FORMED FROM THE H1 RESULT
# ----------------------------------------------------------------------
# H2: 5-day price momentum CONTINUES over the next 5 days (continuation,
#     not reversal). This is the sign-flipped version of H1, motivated
#     directly by the H1 result above.
#
# IMPORTANT CAVEAT (stated deliberately, not glossed over): H2 is being
# tested on the SAME sample used to reject H1. This is in-sample
# re-specification and inflates significance -- in a real research
# setting this signal would need to be validated on a fresh, held-out
# sample (or a different name/period) before being trusted. See
# README.md "Limitations" for how this would be handled properly.
signal_z = research["mom_5d_z"]  # H2 signal: positive momentum -> long
ic_h2, ic_h2_pvalue = stats.spearmanr(signal_z, research["fwd_ret_5d"])
t_stat_h2 = ic_h2 * np.sqrt((n - 2) / (1 - ic_h2**2))

print("\n--- H2 Test (momentum continuation, in-sample, exploratory) ---")
print(f"Spearman IC:        {ic_h2:.4f}")
print(f"IC t-stat:           {t_stat_h2:.2f}")
print(f"p-value:              {ic_h2_pvalue:.4f}")

research["signal_z"] = signal_z

# Rolling IC to check stability of the H2 effect over time (not just a
# single full-sample number)
ROLL_IC_WINDOW = 60
rolling_ic = research["signal_z"].rolling(ROLL_IC_WINDOW).corr(research["fwd_ret_5d"])

# ----------------------------------------------------------------------
# 4. SIGNAL -> PORTFOLIO RULE (based on H2: momentum continuation)
# ----------------------------------------------------------------------
# Simple, transparent rule (deliberately not overfit):
#   signal_z >  +1  -> go long  (recent momentum expected to continue up)
#   signal_z <  -1  -> go short (recent momentum expected to continue down)
#   otherwise        -> flat
ENTRY_Z = 1.0
research["position"] = 0
research.loc[research["signal_z"] > ENTRY_Z, "position"] = 1
research.loc[research["signal_z"] < -ENTRY_Z, "position"] = -1

# ----------------------------------------------------------------------
# 5. EVENT-DRIVEN BACKTEST (non-overlapping, matched to the signal's
#    actual HORIZON)
# ----------------------------------------------------------------------
# The signal is designed to predict a 5-day forward return, so the
# backtest holds each position for exactly HORIZON days rather than
# re-scoring daily -- scoring daily against 1-day returns would test a
# different (and untested) claim than the one the IC analysis supports,
# and would also produce overlapping, autocorrelated trades that are
# harder to interpret cleanly in a project this size.
COST_BPS = 5  # 5 bps per one-way trade, a conservative simplifying assumption

prices = research["adj_close"].values
signal = research["signal_z"].values
dates = research["Date"].values
n_obs = len(research)

trade_log = []
strategy_ret_series = np.zeros(n_obs)
i = 0
while i < n_obs - HORIZON:
    if signal[i] > ENTRY_Z or signal[i] < -ENTRY_Z:
        direction = 1 if signal[i] > ENTRY_Z else -1
        entry_p, exit_p = prices[i], prices[i + HORIZON]
        gross_ret = direction * (exit_p / entry_p - 1)
        net_ret = gross_ret - 2 * (COST_BPS / 1e4)  # entry + exit cost
        strategy_ret_series[i + HORIZON] = net_ret  # booked on exit day
        trade_log.append({
            "entry_date": dates[i], "exit_date": dates[i + HORIZON],
            "direction": direction, "signal_z": signal[i],
            "gross_return": gross_ret, "net_return": net_ret,
        })
        i += HORIZON  # non-overlapping: wait for this trade to close
    else:
        i += 1

research["strategy_ret_net"] = strategy_ret_series
trades = pd.DataFrame(trade_log)

research["equity_strategy"] = (1 + research["strategy_ret_net"]).cumprod()
research["equity_buyhold"] = (1 + research["ret_1d"].fillna(0)).cumprod()

def perf_stats(returns, label, hit_rate_mask=None):
    returns = returns.dropna()
    ann_factor = 252
    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + returns).prod() ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    # Hit rate: for the strategy, only count days with an actual open
    # position -- flat (0-return) days are not "misses" and would
    # otherwise mechanically deflate the hit rate.
    if hit_rate_mask is not None:
        active = returns[hit_rate_mask.reindex(returns.index).fillna(False)]
        hit_rate = (active > 0).mean() if len(active) else np.nan
    else:
        hit_rate = (returns > 0).mean()
    return {
        "label": label, "total_return": total_ret, "annualized_return": ann_ret,
        "annualized_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd,
        "hit_rate": hit_rate, "n_obs": len(returns),
    }

# Trade-level and daily-equivalent performance. Since trades are
# discrete, non-overlapping HORIZON-day events, per-trade stats are the
# most honest way to report hit rate and average return; we also
# annualize the trade-level Sharpe for comparison to buy & hold.
strat_stats = perf_stats(research["strategy_ret_net"], "Signal Strategy (net of costs)")
bh_stats = perf_stats(research["ret_1d"], "Buy & Hold AAPL")

n_trades = len(trades)
if n_trades:
    trade_hit_rate = (trades["net_return"] > 0).mean()
    avg_trade_ret = trades["net_return"].mean()
    trade_sharpe = (trades["net_return"].mean() / trades["net_return"].std()
                     * np.sqrt(252 / HORIZON)) if trades["net_return"].std() > 0 else np.nan
else:
    trade_hit_rate = avg_trade_ret = trade_sharpe = np.nan

exposure_pct = (n_trades * HORIZON) / n_obs * 100

print("\n--- Backtest Results ---")
print(f"{'Signal Strategy (net of costs)':32s}  AnnRet: {strat_stats['annualized_return']*100:6.2f}%   "
      f"AnnVol: {strat_stats['annualized_vol']*100:6.2f}%   MaxDD: {strat_stats['max_drawdown']*100:6.2f}%")
print(f"{'Buy & Hold AAPL':32s}  AnnRet: {bh_stats['annualized_return']*100:6.2f}%   "
      f"AnnVol: {bh_stats['annualized_vol']*100:6.2f}%   Sharpe: {bh_stats['sharpe']:5.2f}   "
      f"MaxDD: {bh_stats['max_drawdown']*100:6.2f}%")
print(f"\nTrade-level stats: N={n_trades}  HitRate={trade_hit_rate*100:.1f}%  "
      f"AvgReturn/Trade={avg_trade_ret*100:.2f}%  Trade-Sharpe(annualized)={trade_sharpe:.2f}")
print(f"Time in market: {exposure_pct:.1f}%")

# ----------------------------------------------------------------------
# 6. SAVE RESULTS
# ----------------------------------------------------------------------
summary = pd.DataFrame([strat_stats, bh_stats]).set_index("label")
summary.to_csv(f"{OUT}/performance_summary.csv")
research.to_csv(f"{OUT}/research_data.csv", index=False)
if n_trades:
    trades.to_csv(f"{OUT}/trade_log.csv", index=False)

with open(f"{OUT}/hypothesis_test.txt", "w") as f:
    f.write("H1: Short-Term Reversal -- mom_5d_z vs 5-day Forward Return\n")
    f.write("=" * 68 + "\n")
    f.write(f"Spearman IC:   {ic_h1:.4f}  (H1 predicts NEGATIVE)\n")
    f.write(f"t-statistic:   {t_stat_h1:.2f}\n")
    f.write(f"p-value:        {ic_h1_pvalue:.4f}\n")
    f.write(f"N:              {n}\n")
    f.write(f"Verdict:        {verdict_h1}\n\n")
    f.write("H2: Momentum Continuation (in-sample, exploratory, sign-flipped from H1)\n")
    f.write("=" * 68 + "\n")
    f.write(f"Spearman IC:   {ic_h2:.4f}\n")
    f.write(f"t-statistic:   {t_stat_h2:.2f}\n")
    f.write(f"p-value:        {ic_h2_pvalue:.4f}\n")
    f.write(f"N:              {n}\n")
    f.write("Caveat:         Tested on the same sample used to reject H1 -- \n")
    f.write("                needs out-of-sample validation before being trusted.\n")

# ----------------------------------------------------------------------
# 7. PLOTS
# ----------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# 7a. Equity curves
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(research["Date"], research["equity_strategy"], label="Signal Strategy (net)", linewidth=1.8)
ax.plot(research["Date"], research["equity_buyhold"], label="Buy & Hold AAPL", linewidth=1.4, alpha=0.7)
ax.set_title("Strategy (5-day event-driven) vs Buy & Hold: Cumulative Growth of $1")
ax.set_ylabel("Growth of $1")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/equity_curve.png", dpi=150)
plt.close(fig)

# 7b. Rolling IC
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(research["Date"], rolling_ic, color="darkorange")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_title(f"Rolling {ROLL_IC_WINDOW}-Day Information Coefficient (Signal vs Fwd Return)")
ax.set_ylabel("IC")
fig.tight_layout()
fig.savefig(f"{OUT}/rolling_ic.png", dpi=150)
plt.close(fig)

# 7c. Signal vs forward return scatter
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(research["signal_z"], research["fwd_ret_5d"] * 100, alpha=0.35, s=18)
z = np.polyfit(research["signal_z"], research["fwd_ret_5d"] * 100, 1)
xs = np.linspace(research["signal_z"].min(), research["signal_z"].max(), 50)
ax.plot(xs, np.poly1d(z)(xs), color="red", linewidth=2)
ax.set_xlabel("5-day Momentum Signal (z-score)")
ax.set_ylabel("5-day Forward Return (%)")
ax.set_title("H2 (Momentum Continuation): Signal vs Forward Return")
fig.tight_layout()
fig.savefig(f"{OUT}/signal_vs_forward_return.png", dpi=150)
plt.close(fig)

print(f"\nSaved outputs to ./{OUT}/  (performance_summary.csv, hypothesis_test.txt, "
      f"research_data.csv, and 3 PNG charts)")
