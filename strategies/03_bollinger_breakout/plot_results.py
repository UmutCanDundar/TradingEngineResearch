import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib.pyplot as plt
from common.plot_utils import (
    setup_style,
    shade_positions,
    format_date_axis,
    add_metrics_box,
    COLORS,
)

from common.data_loader import fetch_prices
from generate_signal import compute_bollinger_bands, generate_signal
from backtest import run_backtest


def compute_metrics(df, freq_per_year=252):
    returns = df["strategy_return"].dropna()

    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * (freq_per_year ** 0.5)
    else:
        sharpe = 0.0

    equity = df["equity_curve"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    active = returns[df.loc[returns.index, "position"] != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0.0

    total_return = equity.iloc[-1] - 1.0

    return {
        "Sharpe": round(sharpe, 2),
        "Max DD": f"{max_dd:.1%}",
        "Win Rate": f"{win_rate:.1%}",
        "Total Return": f"{total_return:.1%}",
    }


def plot_bollinger(df, rolling_mean, upper_band, lower_band):
    setup_style()

    fig, (ax_price, ax_equity) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )

    ax_price.plot(df.index, df["price"], color=COLORS["price"], lw=1.0, label="GARAN.IS Close")
    ax_price.plot(df.index, rolling_mean, color=COLORS["signal_band"], lw=0.8, linestyle="--", label="20-day Mean")
    ax_price.plot(df.index, upper_band, color=COLORS["short"], lw=0.8, linestyle=":", label="Upper Band")
    ax_price.plot(df.index, lower_band, color=COLORS["long"], lw=0.8, linestyle=":", label="Lower Band")
    shade_positions(ax_price, df.index, df["position"])

    ax_price.set_title("Bollinger Breakout — Price and Positions", fontsize=12, fontweight="bold")
    ax_price.set_ylabel("Price (TRY)")
    ax_price.legend(loc="upper left", frameon=False, fontsize=8)

    ax_equity.plot(df.index, df["equity_curve"], color=COLORS["strategy"], lw=1.5, label="Strategy")
    ax_equity.plot(df.index, df["buy_and_hold"], color=COLORS["buy_and_hold"], lw=1.5, label="Buy & Hold")

    ax_equity.set_yscale("log")
    ax_equity.set_title("Equity Curve Comparison (log scale)", fontsize=12, fontweight="bold")
    ax_equity.set_ylabel("Growth Multiple (log)")
    ax_equity.legend(loc="upper left", frameon=False)

    metrics = compute_metrics(df)
    add_metrics_box(ax_equity, metrics)

    format_date_axis(ax_equity)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "results", "equity_curve.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Chart saved: {out_path}")
    print("Metrics:", metrics)

    return fig


if __name__ == "__main__":
    df_prices = fetch_prices()
    prices = df_prices["close"]

    rolling_mean, upper_band, lower_band = compute_bollinger_bands(prices)
    position = generate_signal(prices, upper_band, lower_band)
    result = run_backtest(prices, position)

    plot_bollinger(result, rolling_mean, upper_band, lower_band)
