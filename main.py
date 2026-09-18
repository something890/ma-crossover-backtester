"""
main.py
-------
Single entry point that runs the whole pipeline end to end:

    1. fetch (or load cached) daily price data
    2. build signals and backtest: timing strategy vs. buy-and-hold basket
       vs. buy-and-hold benchmark
    3. compute performance metrics for all three
    4. save the equity-curve / drawdown chart to output/
    5. print a summary report to the console

Run it from the project root with:

    python main.py

Everything it does is a thin call into src/ -- this file has no logic of its
own beyond sequencing and printing. That's deliberate: if you want to reuse
the backtest inside a notebook or another script, import from src.* directly
instead of parsing this file's output.
"""

from __future__ import annotations

import argparse

from src.config import (
    TICKERS,
    BENCHMARK,
    START_DATE,
    SHORT_WINDOW,
    LONG_WINDOW,
    TRANSACTION_COST,
    RISK_FREE_RATE,
)
from src.data_fetch import fetch_price_history
from src.backtest import run_portfolio_backtest
from src.metrics import (
    comparison_table,
    format_table,
    years_elapsed,
    max_drawdown,
    max_drawdown_dates,
)
from src.plot import plot_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a 50/200-day moving-average crossover strategy."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the local CSV cache and re-download prices from Yahoo Finance",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Universe   : {', '.join(TICKERS)}")
    print(f"Benchmark  : {BENCHMARK}")
    print(f"Windows    : {SHORT_WINDOW}-day / {LONG_WINDOW}-day crossover")
    print(f"Cost/trade : {TRANSACTION_COST:.2%}")
    print(f"Start date : {START_DATE}")
    print()

    print("Step 1/4 -- fetching price data...")
    prices = fetch_price_history(force_refresh=args.refresh)
    print(f"  loaded {len(prices)} rows, {prices.index.min().date()} -> "
          f"{prices.index.max().date()}")

    print("Step 2/4 -- running backtest...")
    result = run_portfolio_backtest(prices)
    print(f"  test window: {result.start.date()} -> {result.end.date()} "
          f"({years_elapsed(result.strategy_equity):.1f} years)")

    print("Step 3/4 -- computing metrics...")
    table = comparison_table(result, risk_free_rate=RISK_FREE_RATE)

    print("Step 4/4 -- saving chart...")
    fig_path = plot_backtest(result)
    print(f"  saved to {fig_path}")
    print()

    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(format_table(table).to_string())
    print()

    for label, eq in [
        ("Timing strategy", result.strategy_equity),
        ("Buy & hold basket", result.basket_equity),
        (f"Buy & hold {result.benchmark}", result.benchmark_equity),
    ]:
        peak, trough = max_drawdown_dates(eq)
        print(f"  {label:20s} worst drawdown {max_drawdown(eq):6.1%}  "
              f"({peak.date()} -> {trough.date()})")

    print()
    print("Per-stock breakdown (crossover sleeve, net of costs):")
    print(f"  {'ticker':8s}{'trades':>8s}{'cost drag':>12s}{'timing $1':>12s}{'hold $1':>10s}")
    for t, res in result.per_asset.items():
        n_trades = int(res["turnover"].sum())
        cost_drag = res["cost"].sum()
        timing = res["equity"].iloc[-1] / res["equity"].iloc[0]
        hold = prices[t].loc[result.end] / prices[t].loc[result.start]
        print(f"  {t:8s}{n_trades:>8d}{cost_drag:>11.2%}{timing:>12.3f}{hold:>10.3f}")


if __name__ == "__main__":
    main()
