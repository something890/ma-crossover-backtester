"""
config.py
---------
One place for every "knob" in the project. Keeping these values here (instead of
scattering them through the code) means a later stage can change the strategy
parameters, the date range, or the ticker list without hunting through logic.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains this repo (one level up from src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"      # cached price CSVs live here
OUTPUT_DIR = PROJECT_ROOT / "output"  # charts/tables get written here later

# --- Universe --------------------------------------------------------------
# The individual holdings we want to run the strategy on.
TICKERS = ["AAPL", "JNJ", "JPM", "XOM", "BND"]

# The benchmark we compare against. SPY is an ETF that tracks the S&P 500,
# so "buy and hold SPY" is a good stand-in for "buy and hold the market".
BENCHMARK = "SPY"

# Everything we need to download = holdings + benchmark.
ALL_SYMBOLS = TICKERS + [BENCHMARK]

# --- Backtest window -------------------------------------------------------
# Start early enough that the 200-day moving average has plenty of "warm-up"
# data before the first real trading signal. End = today (yfinance default).
START_DATE = "2015-01-01"
END_DATE = None  # None -> up to the most recent available trading day

# --- Strategy parameters -------------------------------------------------
SHORT_WINDOW = 50    # fast moving average (trading days)
LONG_WINDOW = 200    # slow moving average (trading days)

# --- Backtest assumptions ----------------------------------------------
# Cost charged each time we change position (buy or sell), as a fraction of
# the traded value. 0.001 = 0.1%.
TRANSACTION_COST = 0.001

# Annual risk-free rate used in the Sharpe ratio. 0.0 keeps the first version
# simple; a later stage can plug in a real T-bill series if you want.
RISK_FREE_RATE = 0.0

# Trading days per year, used to annualise returns and volatility.
TRADING_DAYS = 252
