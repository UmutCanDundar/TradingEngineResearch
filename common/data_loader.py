import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _csv_path_for(ticker: str) -> str:
    short_name = ticker.split(".")[0].lower()
    return os.path.join(DATA_DIR, f"{short_name}_prices.csv")


def fetch_prices(ticker="GARAN.IS", start="2020-01-01", end="2025-01-01"):
    csv_path = _csv_path_for(ticker)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return df

    data = yf.download(ticker, start=start, end=end, progress=False)
    df = data[["Close"]].copy()
    df.columns = ["close"]
    df = df.dropna()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(csv_path)

    return df