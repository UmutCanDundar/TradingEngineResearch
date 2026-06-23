import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from common.data_loader import fetch_prices
from generate_signal import compute_zscore, generate_signal


def run_backtest(prices, position, transaction_cost_bps=5.0):
    daily_return = prices.pct_change()
    strategy_position = position.shift(1).fillna(0)

    position_change = strategy_position.diff().abs().fillna(0)
    cost = position_change * (transaction_cost_bps / 10_000)

    strategy_return = strategy_position * daily_return - cost

    df = pd.DataFrame({
        "price": prices,
        "position": strategy_position,
        "daily_return": daily_return,
        "strategy_return": strategy_return,
    })

    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    df["buy_and_hold"] = (1 + df["daily_return"]).cumprod()

    return df


if __name__ == "__main__":
    df = fetch_prices()
    prices = df["close"]

    z = compute_zscore(prices, window=20)
    position = generate_signal(z)

    result = run_backtest(prices, position)

    print(result.tail(10))
    print()
    print("Final equity (strategy):", result["equity_curve"].iloc[-1])
    print("Final equity (buy&hold):", result["buy_and_hold"].iloc[-1])