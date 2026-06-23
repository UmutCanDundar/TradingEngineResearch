import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from common.data_loader import fetch_prices
from generate_signal import compute_moving_averages, generate_signal


def run_fixed_backtest(prices, position, transaction_cost_bps=5.0):
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


def compute_kelly_fraction(returns, position):
    active_returns = returns[position.shift(1).fillna(0) != 0]
    wins = active_returns[active_returns > 0]
    losses = active_returns[active_returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    win_rate = len(wins) / len(active_returns)
    loss_rate = 1 - win_rate

    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    win_loss_ratio = avg_win / avg_loss

    kelly_fraction = win_rate - (loss_rate / win_loss_ratio)

    return kelly_fraction


def run_kelly_backtest(prices, position, kelly_fraction, vol_window=20, target_vol=0.15,
                        max_leverage=1.0, transaction_cost_bps=5.0):
    daily_return = prices.pct_change()

    realized_vol = daily_return.rolling(window=vol_window).std() * (252 ** 0.5)
    vol_scale = (target_vol / realized_vol).clip(upper=max_leverage)
    vol_scale = vol_scale.fillna(0)

    raw_position = position.shift(1).fillna(0)
    sized_position = raw_position * vol_scale * max(kelly_fraction, 0.0)

    position_change = sized_position.diff().abs().fillna(0)
    cost = position_change * (transaction_cost_bps / 10_000)

    strategy_return = sized_position * daily_return - cost

    df = pd.DataFrame({
        "price": prices,
        "position": raw_position,
        "sized_position": sized_position,
        "daily_return": daily_return,
        "strategy_return": strategy_return,
    })

    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    df["buy_and_hold"] = (1 + df["daily_return"]).cumprod()

    return df


if __name__ == "__main__":
    df = fetch_prices()
    prices = df["close"]

    short_ma, long_ma = compute_moving_averages(prices)
    position = generate_signal(short_ma, long_ma)

    fixed_result = run_fixed_backtest(prices, position)
    kelly_fraction = compute_kelly_fraction(fixed_result["strategy_return"], position)

    print("Kelly fraction:", kelly_fraction)

    kelly_result = run_kelly_backtest(prices, position, kelly_fraction)

    print(kelly_result.tail(10))
    print()
    print("Final equity (fixed sizing):", fixed_result["equity_curve"].iloc[-1])
    print("Final equity (Kelly + vol-adjusted):", kelly_result["equity_curve"].iloc[-1])
    print("Final equity (buy&hold):", kelly_result["buy_and_hold"].iloc[-1])
