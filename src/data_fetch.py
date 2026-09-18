"""
data_fetch.py
-------------
Download historical daily price data from Yahoo Finance (via the `yfinance`
library) and keep a local CSV copy so repeated runs don't hammer the network.

Design choices worth knowing about:

* We download **auto-adjusted** prices (`auto_adjust=True`). That means the
  "Close" column is already adjusted for stock splits and dividends. For a
  backtest this is what you want: a 2-for-1 split shouldn't look like the
  stock lost half its value, and reinvested dividends are part of the real
  return you'd have earned.

* Each symbol is cached to `data/<TICKER>.csv`. If the file exists we read it
  instead of re-downloading. Pass `force_refresh=True` to bypass the cache.

* `fetch_price_history()` returns a single tidy DataFrame: one row per date,
  one column per ticker, values = adjusted closing price.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from .config import DATA_DIR, START_DATE, END_DATE, ALL_SYMBOLS


def _cache_path(ticker: str, data_dir: Path) -> Path:
    """Where the CSV for one ticker lives."""
    return data_dir / f"{ticker.upper()}.csv"


def fetch_one(
    ticker: str,
    start: str = START_DATE,
    end: str | None = END_DATE,
    data_dir: Path = DATA_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily OHLCV data for a single `ticker`.

    Columns: Open, High, Low, Close, Volume  (Close is split/dividend adjusted)
    Index:   DatetimeIndex named "Date"
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, data_dir)

    # 1. Use the cached copy if we have one.
    if path.exists() and not force_refresh:
        cached = pd.read_csv(path, index_col="Date", parse_dates=True)
        return cached

    # 2. Otherwise hit Yahoo Finance.
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,   # adjust Close for splits + dividends
        progress=False,     # no progress bar in the console
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for {ticker!r}. Check the symbol and the date range."
        )

    # yfinance sometimes returns columns as a MultiIndex like ("Close", "AAPL").
    # Flatten it down to just "Close", "Open", etc.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = raw[keep].copy()
    df.index.name = "Date"

    # 3. Save to cache for next time.
    df.to_csv(path)
    return df


def fetch_price_history(
    tickers: Iterable[str] = ALL_SYMBOLS,
    start: str = START_DATE,
    end: str | None = END_DATE,
    data_dir: Path = DATA_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download (or load from cache) every symbol in `tickers` and stitch their
    adjusted closing prices into one DataFrame.

    Returns
    -------
    prices : pd.DataFrame
        index   -> trading dates (sorted ascending)
        columns -> one per ticker
        values  -> adjusted closing price
    """
    closes: dict[str, pd.Series] = {}

    for ticker in tickers:
        df = fetch_one(
            ticker,
            start=start,
            end=end,
            data_dir=data_dir,
            force_refresh=force_refresh,
        )
        closes[ticker.upper()] = df["Close"]

    prices = pd.DataFrame(closes).sort_index()

    # Drop any leading/trailing rows where *every* ticker is missing.
    prices = prices.dropna(how="all")

    return prices


if __name__ == "__main__":
    # Running `python -m src.data_fetch` from the project root gives a quick
    # sanity check that the download + cache round-trip works.
    prices = fetch_price_history()

    print(f"Symbols : {list(prices.columns)}")
    print(f"Rows    : {len(prices)}")
    print(f"Range   : {prices.index.min().date()} -> {prices.index.max().date()}")
    print()
    print("First 3 rows:")
    print(prices.head(3).round(2))
    print()
    print("Last 3 rows:")
    print(prices.tail(3).round(2))
    print()
    print("Missing values per column:")
    print(prices.isna().sum())
