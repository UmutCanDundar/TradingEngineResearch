import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from common.data_loader import fetch_prices


def compute_spread(prices_a, prices_b, window=20):
    log_a = pd.Series(prices_a).apply(lambda x: __import__("math").log(x))
    log_b = pd.Series(prices_b).apply(lambda x: __import__("math").log(x))

    spread = log_a - log_b

    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z = (spread - spread_mean) / spread_std

    return spread, z


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
    df_a = fetch_prices(ticker="GARAN.IS")
    df_b = fetch_prices(ticker="AKBNK.IS")

    common_index = df_a.index.intersection(df_b.index)
    prices_a = df_a.loc[common_index, "close"]
    prices_b = df_b.loc[common_index, "close"]

    spread, z = compute_spread(prices_a, prices_b)
    position = generate_signal(z)

    print(position.value_counts())
    print()
    print(position.tail(30))
