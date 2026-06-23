import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from common.data_loader import fetch_prices
from generate_signal import compute_spread, generate_signal


def run_backtest(prices_a, prices_b, position, transaction_cost_bps=5.0):
    daily_return_a = prices_a.pct_change()
    daily_return_b = prices_b.pct_change()

    strategy_position = position.shift(1).fillna(0)

    position_change = strategy_position.diff().abs().fillna(0)
    cost = position_change * (transaction_cost_bps / 10_000) * 2

    pair_return = daily_return_a - daily_return_b
    strategy_return = strategy_position * pair_return - cost

    df = pd.DataFrame({
        "price": prices_a,
        "price_b": prices_b,
        "position": strategy_position,
        "daily_return": daily_return_a,
        "strategy_return": strategy_return,
    })

    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    df["buy_and_hold"] = (1 + df["daily_return"]).cumprod()

    return df


if __name__ == "__main__":
    df_a = fetch_prices(ticker="GARAN.IS")
    df_b = fetch_prices(ticker="AKBNK.IS")

    common_index = df_a.index.intersection(df_b.index)
    prices_a = df_a.loc[common_index, "close"]
    prices_b = df_b.loc[common_index, "close"]

    spread, z = compute_spread(prices_a, prices_b)
    position = generate_signal(z)

    result = run_backtest(prices_a, prices_b, position)

    print(result.tail(10))
    print()
    print("Final equity (strategy):", result["equity_curve"].iloc[-1])
    print("Final equity (buy&hold, GARAN only):", result["buy_and_hold"].iloc[-1])
