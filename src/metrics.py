"""
metrics.py
----------
Performance statistics for a daily-return / equity-curve pair.

Every function here is deliberately small and does one thing, so each number in
the final table can be traced back to a formula you can check by hand.

Definitions used
================
total return   : end value / start value - 1
CAGR           : the constant annual growth rate that would take you from the
                 start value to the end value over the actual elapsed time.
                 (1 + total_return) ** (1 / years) - 1
ann. volatility: standard deviation of daily returns, scaled to one year by
                 multiplying by sqrt(252). 252 ~= trading days in a year.
Sharpe ratio   : average daily excess return / daily return volatility, scaled
                 to a year by sqrt(252). "Excess" = return minus the risk-free
                 rate. Higher = more return per unit of wobble. Rough reading:
                 < 1 mediocre, ~1 decent, > 2 very good (and be suspicious).
max drawdown   : the worst peak-to-trough fall in the equity curve, as a
                 percentage of the peak. Always <= 0. This is the "how bad did
                 it hurt at the worst moment" number.
Calmar ratio   : CAGR / |max drawdown|. Return earned per unit of worst-case
                 pain. Higher is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, TRADING_DAYS


def total_return(equity: pd.Series) -> float:
    """End value divided by start value, minus 1."""
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def years_elapsed(equity: pd.Series) -> float:
    """Actual calendar time spanned by the curve, in years."""
    span_days = (equity.index[-1] - equity.index[0]).days
    return span_days / 365.25


def cagr(equity: pd.Series) -> float:
    """
    Compound annual growth rate. Uses real elapsed calendar time (not a
    trading-day count) because CAGR is about how long your money was actually
    at work.
    """
    yrs = years_elapsed(equity)
    if yrs <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / yrs) - 1.0)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """
    Standard deviation of daily returns, annualised.

    ddof=1 -> sample standard deviation (divide by n-1), the usual choice when
    you're estimating from a sample rather than knowing the whole population.
    Multiplying by sqrt(periods_per_year) is the standard way to turn a
    one-day volatility into a one-year figure (variance scales with time, so
    the standard deviation scales with the square root of time).
    """
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Annualised Sharpe ratio from a daily return series.

    daily_rf      = annual risk-free rate / trading days
    excess        = daily returns - daily_rf
    sharpe        = mean(excess) / std(excess) * sqrt(periods_per_year)

    We use the classic "mean over standard deviation of daily excess returns,
    then scale by sqrt(time)" definition rather than (CAGR - rf) / vol, because
    it depends only on the return sample and is the version most readers expect.
    """
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """
    For each day: how far below the highest point seen *so far* the curve is,
    as a fraction of that high. 0 means "at a new high"; -0.20 means "20% below
    the best it had ever been".
    """
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """The single worst value of the drawdown series (a negative number)."""
    return float(drawdown_series(equity).min())


def max_drawdown_dates(equity: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    (peak_date, trough_date) for the worst drawdown -- the prior high-water
    mark, and the bottom of the fall from it.
    """
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return peak, trough


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR divided by the absolute value of max drawdown."""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return np.nan
    return float(cagr(equity) / mdd)


def summarize(
    equity: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float]:
    """All headline stats for one strategy, as an ordered dict."""
    return {
        "Total return": total_return(equity),
        "CAGR": cagr(equity),
        "Ann. volatility": annualized_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns, risk_free_rate),
        "Max drawdown": max_drawdown(equity),
        "Calmar ratio": calmar_ratio(equity),
    }


def comparison_table(result, risk_free_rate: float = RISK_FREE_RATE) -> pd.DataFrame:
    """
    Build the side-by-side stats table for the three curves in a
    ``BacktestResult``: the timing strategy, the buy-and-hold basket of the
    same stocks, and the buy-and-hold benchmark.

    Returns a DataFrame of raw floats (rows = metrics, columns = curves) so it
    can be formatted for the console or reused by the plotting stage.
    """
    columns = {
        "Timing strategy": summarize(
            result.strategy_equity, result.strategy_returns, risk_free_rate
        ),
        "Buy & hold basket": summarize(
            result.basket_equity, result.basket_returns, risk_free_rate
        ),
        f"Buy & hold {result.benchmark}": summarize(
            result.benchmark_equity, result.benchmark_returns, risk_free_rate
        ),
    }
    return pd.DataFrame(columns)


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Human-readable version of :func:`comparison_table`: percentages where a
    percentage makes sense, plain 2dp for the ratios.
    """
    pct_rows = {"Total return", "CAGR", "Ann. volatility", "Max drawdown"}
    out = table.copy().astype(object)
    for row in table.index:
        for col in table.columns:
            val = table.loc[row, col]
            if row in pct_rows:
                out.loc[row, col] = f"{val:.1%}"
            else:
                out.loc[row, col] = f"{val:.2f}"
    return out


if __name__ == "__main__":
    from .data_fetch import fetch_price_history
    from .backtest import run_portfolio_backtest

    prices = fetch_price_history()
    result = run_portfolio_backtest(prices)

    table = comparison_table(result)

    print(f"Backtest window : {result.start.date()} -> {result.end.date()}  "
          f"({years_elapsed(result.strategy_equity):.1f} years)")
    print(f"Risk-free rate  : {RISK_FREE_RATE:.1%} annual")
    print()
    print(format_table(table).to_string())
    print()

    for label, eq in [
        ("Timing strategy", result.strategy_equity),
        ("Buy & hold basket", result.basket_equity),
        (f"Buy & hold {result.benchmark}", result.benchmark_equity),
    ]:
        peak, trough = max_drawdown_dates(eq)
        print(f"{label:20s} worst drawdown {max_drawdown(eq):6.1%}  "
              f"from {peak.date()} to {trough.date()}")
