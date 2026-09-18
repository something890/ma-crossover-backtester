"""
backtest.py
-----------
Turn the day-by-day positions from ``strategy.py`` into money: a daily return
stream and a compounding equity curve, net of transaction costs. Also builds two
reference curves over the exact same window so the strategy can be judged fairly:

  1. benchmark  -- buy-and-hold SPY (the market)
  2. basket     -- buy-and-hold the strategy's OWN five stocks, equal weight

Comparing against both separates two questions that a single comparison blurs
together:
  * "did the market-timing help?"     -> strategy vs. basket
  * "did picking these five names help?" -> basket vs. benchmark

How a single asset is backtested
================================
For each trading day ``t``:

    asset_return[t] = price[t] / price[t-1] - 1          # the stock's own move
    gross_return[t] = position[t] * asset_return[t]      # we only earn it if we held it

``position[t]`` already came out of ``strategy.py`` shifted one day forward, so
it represents "what we were holding at the start of day t, decided from
yesterday's close". Multiplying it by day t's return is therefore lookahead-free.

Transaction costs
=================
    trade[t]    = position[t] - position[t-1]            # +1 enter, -1 exit, 0 hold
    turnover[t] = abs(trade[t])
    cost[t]     = turnover[t] * TRANSACTION_COST         # 0.1% of notional per trade
    net_return[t] = gross_return[t] - cost[t]

The first day of the test window counts as a trade if we start invested.

Equity curve
============
    equity[t] = (1 + net_return).cumprod()              # growth of $1

Combining several stocks
========================
Capital is split equally across the N stocks at the start (1/N each) and never
rebalanced. Each "sleeve" runs its rule on its own stock (in it, or in cash
earning 0%). The combined equity curve is the average of the sleeve equity
curves; the combined daily return is that curve's day-over-day change, which
correctly reflects the weights drifting as sleeves grow apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import (
    TICKERS,
    BENCHMARK,
    SHORT_WINDOW,
    LONG_WINDOW,
    TRANSACTION_COST,
)
from .strategy import generate_signals


# --------------------------------------------------------------------------
# Single-asset backtests
# --------------------------------------------------------------------------
def backtest_asset(
    signals: pd.DataFrame,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Backtest the crossover strategy on one asset.

    Parameters
    ----------
    signals : DataFrame from ``strategy.generate_signals`` (already clipped to
        the desired test window, i.e. no leading rows where the slow MA is NaN).
    transaction_cost : fraction of notional charged per trade.

    Returns a DataFrame with:
        position, asset_return, gross_return, trade, turnover, cost,
        net_return, equity
    """
    price = signals["price"]
    position = signals["position"].astype(float)

    asset_return = price.pct_change().fillna(0.0)
    gross_return = position * asset_return

    # Recompute trades on the clipped window so the entry cost is captured:
    trade = position.diff()
    trade.iloc[0] = position.iloc[0]          # starting invested = an entry trade
    turnover = trade.abs()
    cost = turnover * transaction_cost

    net_return = gross_return - cost
    equity = (1.0 + net_return).cumprod()

    return pd.DataFrame(
        {
            "position": position,
            "asset_return": asset_return,
            "gross_return": gross_return,
            "trade": trade,
            "turnover": turnover,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
        }
    )


def backtest_buy_and_hold(
    price: pd.Series,
    transaction_cost: float = TRANSACTION_COST,
    apply_entry_cost: bool = True,
) -> pd.DataFrame:
    """
    Buy on the first day of the window, hold to the end. One trade total.
    """
    price = price.sort_index()
    asset_return = price.pct_change().fillna(0.0)

    net_return = asset_return.copy()
    if apply_entry_cost and len(net_return) > 0:
        net_return.iloc[0] -= transaction_cost

    equity = (1.0 + net_return).cumprod()

    return pd.DataFrame(
        {
            "asset_return": asset_return,
            "net_return": net_return,
            "equity": equity,
        }
    )


# --------------------------------------------------------------------------
# Helper: combine sleeves into one equal-weight, no-rebalance portfolio
# --------------------------------------------------------------------------
def _combine_equal_weight(
    sleeve_equities: dict[str, pd.Series],
) -> tuple[pd.Series, pd.Series]:
    """
    Average a set of sleeve equity curves (each normalised to start at 1.0) into
    one portfolio equity curve, and derive its daily returns.
    """
    normalised = {t: eq / eq.iloc[0] for t, eq in sleeve_equities.items()}
    equity = pd.DataFrame(normalised).mean(axis=1)
    equity.iloc[0] = 1.0
    returns = equity.pct_change().fillna(0.0)
    return equity, returns


# --------------------------------------------------------------------------
# Portfolio-level backtest
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Everything a later stage (metrics, plotting) needs."""
    per_asset: dict[str, pd.DataFrame]      # ticker -> single-asset strategy backtest
    strategy_returns: pd.Series             # daily net return, timing strategy
    basket_returns: pd.Series               # daily net return, buy & hold same 5 stocks
    benchmark_returns: pd.Series            # daily net return, buy & hold benchmark
    strategy_equity: pd.Series              # growth of $1, timing strategy
    basket_equity: pd.Series                # growth of $1, buy & hold same 5 stocks
    benchmark_equity: pd.Series             # growth of $1, buy & hold benchmark
    start: pd.Timestamp
    end: pd.Timestamp
    tickers: list[str]
    benchmark: str


def run_portfolio_backtest(
    prices: pd.DataFrame,
    tickers: list[str] = TICKERS,
    benchmark: str = BENCHMARK,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
    transaction_cost: float = TRANSACTION_COST,
) -> BacktestResult:
    """
    Run three things over one common window:
      * the crossover timing strategy on ``tickers`` (equal weight, no rebalance)
      * buy-and-hold of those same ``tickers`` (equal weight, no rebalance)
      * buy-and-hold of ``benchmark``
    """
    # 1. Build signals for each stock.
    raw_signals = {
        t: generate_signals(prices[t], short_window, long_window) for t in tickers
    }

    # 2. Common start = first date every sleeve's slow MA is valid.
    start = max(s["sma_long"].first_valid_index() for s in raw_signals.values())
    end = prices.index.max()

    # 3. Timing strategy: clip each sleeve, backtest, collect equity curves.
    per_asset: dict[str, pd.DataFrame] = {}
    strat_sleeves: dict[str, pd.Series] = {}
    hold_sleeves: dict[str, pd.Series] = {}
    for t in tickers:
        clipped = raw_signals[t].loc[start:end]
        res = backtest_asset(clipped, transaction_cost)
        per_asset[t] = res
        strat_sleeves[t] = res["equity"]

        # Buy-and-hold sleeve for the same stock over the same window.
        hold = backtest_buy_and_hold(prices[t].loc[start:end], transaction_cost)
        hold_sleeves[t] = hold["equity"]

    strategy_equity, strategy_returns = _combine_equal_weight(strat_sleeves)
    basket_equity, basket_returns = _combine_equal_weight(hold_sleeves)

    # 4. Benchmark: buy-and-hold over the identical window.
    bench = backtest_buy_and_hold(
        prices[benchmark].loc[start:end], transaction_cost, apply_entry_cost=True
    )
    benchmark_equity = bench["equity"] / bench["equity"].iloc[0]
    benchmark_returns = bench["net_return"].copy()
    benchmark_returns.iloc[0] = 0.0

    return BacktestResult(
        per_asset=per_asset,
        strategy_returns=strategy_returns,
        basket_returns=basket_returns,
        benchmark_returns=benchmark_returns,
        strategy_equity=strategy_equity,
        basket_equity=basket_equity,
        benchmark_equity=benchmark_equity,
        start=start,
        end=end,
        tickers=list(tickers),
        benchmark=benchmark,
    )


if __name__ == "__main__":
    from .data_fetch import fetch_price_history

    prices = fetch_price_history()
    result = run_portfolio_backtest(prices)

    def total_return(equity: pd.Series) -> float:
        return equity.iloc[-1] / equity.iloc[0] - 1.0

    print(f"Backtest window : {result.start.date()} -> {result.end.date()}")
    print(f"Strategy stocks : {', '.join(result.tickers)} (equal weight)")
    print()
    print(f"{'':22s}{'final $1 ->':>13s}{'total return':>16s}")
    for label, eq in [
        ("Timing strategy", result.strategy_equity),
        (f"Buy & hold basket", result.basket_equity),
        (f"Buy & hold {result.benchmark}", result.benchmark_equity),
    ]:
        print(f"{label:22s}{eq.iloc[-1]:>13.3f}{total_return(eq):>15.1%}")
    print()
    print("Per-stock (crossover sleeve, net of costs):")
    print(f"  {'ticker':8s}{'trades':>8s}{'cost drag':>12s}{'timing $1':>12s}{'hold $1':>10s}")
    for t, res in result.per_asset.items():
        n_trades = int(res["turnover"].sum())
        cost_drag = res["cost"].sum()
        timing = res["equity"].iloc[-1] / res["equity"].iloc[0]
        hold = (prices[t].loc[result.end] / prices[t].loc[result.start])
        print(f"  {t:8s}{n_trades:>8d}{cost_drag:>11.2%}{timing:>12.3f}{hold:>10.3f}")
