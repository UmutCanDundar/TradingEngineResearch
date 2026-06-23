import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from common.data_loader import fetch_prices


def compute_bollinger_bands(prices, window=20, num_std=2.0):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    return rolling_mean, upper_band, lower_band


def generate_signal(prices, upper_band, lower_band):
    position = pd.Series(0, index=prices.index, dtype=int)
    current_pos = 0

    for i in range(len(prices)):
        price = prices.iloc[i]
        upper = upper_band.iloc[i]
        lower = lower_band.iloc[i]

        if pd.isna(upper) or pd.isna(lower):
            position.iloc[i] = current_pos
            continue

        if price > upper:
            current_pos = 1
        elif price < lower:
            current_pos = -1

        position.iloc[i] = current_pos

    return position


if __name__ == "__main__":
    df = fetch_prices()
    rolling_mean, upper_band, lower_band = compute_bollinger_bands(df["close"])
    position = generate_signal(df["close"], upper_band, lower_band)

    print(position.value_counts())
    print()
    print(position.tail(30))
