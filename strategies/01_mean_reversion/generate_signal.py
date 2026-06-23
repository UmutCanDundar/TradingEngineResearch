import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from common.data_loader import fetch_prices


def compute_zscore(prices, window=20):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    z = (prices - rolling_mean) / rolling_std
    return z


def generate_signal(z, entry_threshold=1.5, exit_threshold=0.3):
    position = pd.Series(0, index=z.index, dtype=int)
    current_pos = 0

    for i in range(len(z)):
        zi = z.iloc[i]

        if pd.isna(zi):
            position.iloc[i] = current_pos
            continue

        if current_pos == 0:
            if zi > entry_threshold:
                current_pos = -1
            elif zi < -entry_threshold:
                current_pos = 1
        else:
            if abs(zi) < exit_threshold:
                current_pos = 0

        position.iloc[i] = current_pos

    return position


if __name__ == "__main__":
    df = fetch_prices()
    z = compute_zscore(df["close"], window=20)
    position = generate_signal(z)

    print(position.value_counts())
    print()
    print(position.tail(30))