import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from common.data_loader import fetch_prices


def compute_moving_averages(prices, short_window=20, long_window=50):
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    return short_ma, long_ma


def generate_signal(short_ma, long_ma):
    position = pd.Series(0, index=short_ma.index, dtype=int)

    for i in range(len(short_ma)):
        s = short_ma.iloc[i]
        l = long_ma.iloc[i]

        if pd.isna(s) or pd.isna(l):
            continue

        if s > l:
            position.iloc[i] = 1
        elif s < l:
            position.iloc[i] = -1
        else:
            position.iloc[i] = 0

    return position


if __name__ == "__main__":
    df = fetch_prices()
    short_ma, long_ma = compute_moving_averages(df["close"])
    position = generate_signal(short_ma, long_ma)

    print(position.value_counts())
    print()
    print(position.tail(30))
