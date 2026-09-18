"""
plot.py
-------
Draw the result of a backtest as a single figure with two stacked panels:

  top    : equity curves (growth of $1) for the timing strategy, the
           buy-and-hold basket of the same stocks, and the buy-and-hold
           benchmark. Plotted on a LOG y-axis.
  bottom : the drawdown of each curve (percent below its own prior peak).

Why a log scale on the equity panel
===================================
We are compounding over ~11 years, so the curves span a wide range (roughly $1
to $6). On a linear axis the early years get squashed into a flat line at the
bottom and you can only see the recent period. On a log axis, equal *percentage*
moves take equal vertical space, so a 20% year looks the same size whether it
happened in 2016 or 2025 -- which is what you want when judging growth.

Why a separate drawdown panel
=============================
The equity curve alone hides how the ride *felt*. The drawdown panel makes the
depth and length of each slump explicit, and lines the three approaches up so
you can see which one hurt least at the worst moments.

This module never opens a window -- it uses the non-interactive "Agg" backend
and writes a PNG to ``output/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no display needed

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .config import OUTPUT_DIR
from .metrics import drawdown_series, comparison_table, format_table


# Consistent colours across both panels: same curve = same colour.
_COLOURS = {
    "strategy": "#1f77b4",   # blue
    "basket": "#2ca02c",     # green
    "benchmark": "#7f7f7f",  # grey
}


def plot_backtest(result, outpath: Path | str | None = None) -> Path:
    """
    Render the two-panel figure for a ``BacktestResult`` and save it to PNG.

    Returns the path the figure was written to.
    """
    if outpath is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        outpath = OUTPUT_DIR / "equity_curve.png"
    outpath = Path(outpath)

    bench_label = f"Buy & hold {result.benchmark}"

    curves = [
        ("strategy", "Timing strategy", result.strategy_equity),
        ("basket", "Buy & hold basket", result.basket_equity),
        ("benchmark", bench_label, result.benchmark_equity),
    ]

    # Two panels sharing the x-axis; the equity panel gets more vertical room.
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- top panel: equity curves on a log scale --------------------------
    for key, label, eq in curves:
        ax_eq.plot(eq.index, eq.values, label=label, color=_COLOURS[key],
                   linewidth=1.6)

    ax_eq.set_yscale("log")
    # Show the log ticks as plain dollar amounts ($1, $2, $5...) not 10^0 etc.
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax_eq.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"
                                                  if y in (2, 5) else ""))
    ax_eq.set_ylabel("Growth of $1 (log scale)")
    ax_eq.set_title(
        f"Moving-average crossover vs. buy & hold\n"
        f"{result.start.date()} to {result.end.date()}  "
        f"(equal-weight: {', '.join(result.tickers)})",
        fontsize=11,
    )
    ax_eq.grid(True, which="both", alpha=0.25)
    ax_eq.legend(loc="upper left", frameon=False)

    # --- bottom panel: drawdowns ---------------------------------------
    for key, label, eq in curves:
        dd = drawdown_series(eq) * 100.0
        ax_dd.plot(dd.index, dd.values, color=_COLOURS[key], linewidth=1.2)
        ax_dd.fill_between(dd.index, dd.values, 0.0, color=_COLOURS[key],
                           alpha=0.12)

    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(True, alpha=0.25)
    ax_dd.axhline(0.0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    from .data_fetch import fetch_price_history
    from .backtest import run_portfolio_backtest

    prices = fetch_price_history()
    result = run_portfolio_backtest(prices)

    path = plot_backtest(result)
    print(f"Figure written to: {path}")
    print()
    print(format_table(comparison_table(result)).to_string())
