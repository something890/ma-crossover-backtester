"""
strategy.py
-----------
Turn a price series into a day-by-day trading position for the
50-day / 200-day moving-average crossover strategy.

The rule
========
* Compute a fast moving average (default 50 trading days) and a slow moving
  average (default 200 trading days) of the adjusted closing price.
* When the fast MA is **above** the slow MA, the trend is "up" -> we want to
  hold the stock (position = 1).
* When the fast MA is **below** the slow MA, the trend is "down" -> we want to
  be in cash (position = 0).
* The moments where fast crosses slow are the classic signals:
  - fast crosses *above* slow  ->  "golden cross"  -> BUY
  - fast crosses *below* slow  ->  "death cross"   -> SELL

Avoiding lookahead bias (the important bit)
==========================================
A moving average that ends on today's close is not known until the market has
*closed* today. You cannot buy or sell at a price you only learned about after
trading finished. So we split the idea into two columns:

* ``target_position`` -- what the rule says *given data up to and including
  today's close*. This still "peeks" at today.
* ``position``        -- what we are *actually* holding today. It equals
  yesterday's ``target_position`` (``target_position.shift(1)``). We see the
  cross after today's close, then act at tomorrow's open/close.

Every downstream calculation (returns, costs, equity curve) uses ``position``,
never ``target_position``. That one-day shift is what keeps the backtest honest.

During the first ~200 trading days the slow MA doesn't exist yet, so there is
no signal and ``position`` stays 0 (in cash). That "warm-up" period is expected
and is why Stage 1 downloaded history starting well before the test window.
"""

from __future__ import annotations

import pandas as pd

from .config import SHORT_WINDOW, LONG_WINDOW


def generate_signals(
    price: pd.Series,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
) -> pd.DataFrame:
    """
    Build the full signal table for a single instrument.

    Parameters
    ----------
    price : pd.Series
        Adjusted closing price, indexed by date (ascending).
    short_window, long_window : int
        Lengths of the fast and slow moving averages, in trading days.

    Returns
    -------
    pd.DataFrame with columns:
        price            -- the input price
        sma_short        -- fast moving average
        sma_long         -- slow moving average
        target_position  -- 1/0 desired state from *today's* close (peeks at today)
        position         -- 1/0 actually held today = target_position shifted 1 day
        trade            -- change in position: +1 = bought, -1 = sold, 0 = held
    """
    price = price.sort_index()

    # 1. The two moving averages.
    #    min_periods defaults to the window length, so each MA is NaN until it
    #    has enough data -> no accidental signal during the warm-up period.
    sma_short = price.rolling(window=short_window).mean()
    sma_long = price.rolling(window=long_window).mean()

    # 2. Desired state given everything known up to today's close.
    #    NaN > NaN is False, so during warm-up target_position is 0 (in cash).
    target_position = (sma_short > sma_long).astype(int)

    # 3. What we actually hold today: yesterday's desired state.
    #    This single .shift(1) is the anti-lookahead safeguard.
    position = target_position.shift(1).fillna(0).astype(int)

    # 4. Trades happen where the held position changes.
    trade = position.diff().fillna(0).astype(int)

    return pd.DataFrame(
        {
            "price": price,
            "sma_short": sma_short,
            "sma_long": sma_long,
            "target_position": target_position,
            "position": position,
            "trade": trade,
        }
    )


def build_signal_frames(
    prices: pd.DataFrame,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
) -> dict[str, pd.DataFrame]:
    """
    Run :func:`generate_signals` on every column of a price table.

    Returns a dict: ticker -> its signal DataFrame.
    """
    return {
        ticker: generate_signals(prices[ticker], short_window, long_window)
        for ticker in prices.columns
    }


def crossover_events(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Just the rows where a trade happened, with a readable label.
    Handy for eyeballing that the signals make sense.
    """
    events = signals.loc[signals["trade"] != 0, ["price", "sma_short", "sma_long", "trade"]].copy()
    events["action"] = events["trade"].map({1: "BUY  (golden cross)", -1: "SELL (death cross)"})
    return events


if __name__ == "__main__":
    # Quick sanity check: load cached prices, build signals, and show the
    # crossover history + trade count for each ticker.
    from .data_fetch import fetch_price_history

    prices = fetch_price_history()
    frames = build_signal_frames(prices)

    for ticker, sig in frames.items():
        events = crossover_events(sig)
        n_buys = int((events["trade"] == 1).sum())
        n_sells = int((events["trade"] == -1).sum())
        first_signal_date = sig.loc[sig["target_position"].ne(0)].index.min()

        print("=" * 70)
        print(f"{ticker}: {len(events)} trades  ({n_buys} buys, {n_sells} sells)")
        print(f"  slow MA becomes valid ~ {sig['sma_long'].first_valid_index().date()}")
        print(f"  first 'go long' signal ~ {first_signal_date.date() if pd.notna(first_signal_date) else 'never'}")
        print(f"  currently {'IN THE MARKET' if sig['position'].iloc[-1] == 1 else 'in cash'}")
        if not events.empty:
            print("  crossover history:")
            for date, row in events.iterrows():
                print(f"    {date.date()}  {row['action']:20s}  price={row['price']:.2f}")
