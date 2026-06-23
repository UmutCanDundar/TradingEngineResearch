import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib.pyplot as plt
from common.plot_utils import (
    setup_style,
    format_date_axis,
    add_metrics_box,
    COLORS,
)

from common.data_loader import fetch_prices
from generate_signal import compute_moving_averages, generate_signal
from backtest import run_fixed_backtest, compute_kelly_fraction, run_kelly_backtest


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


def plot_kelly_comparison(fixed_df, kelly_df, kelly_fraction):
    setup_style()

    fig, (ax_position, ax_equity) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3]},
    )

    ax_position.plot(fixed_df.index, fixed_df["position"], color=COLORS["price"], lw=1.0, label="Fixed Position (-1/0/1)")
    ax_position.plot(kelly_df.index, kelly_df["sized_position"], color=COLORS["strategy"], lw=1.2, label="Kelly + Vol-Adjusted Size")
    ax_position.axhline(0, color="#999999", lw=0.6)

    ax_position.set_title(f"Position Sizing Comparison (Kelly fraction = {kelly_fraction:.2f})", fontsize=12, fontweight="bold")
    ax_position.set_ylabel("Position Size")
    ax_position.legend(loc="upper left", frameon=False)

    ax_equity.plot(fixed_df.index, fixed_df["equity_curve"], color=COLORS["price"], lw=1.5, label="Fixed Sizing (Momentum)")
    ax_equity.plot(kelly_df.index, kelly_df["equity_curve"], color=COLORS["strategy"], lw=1.5, label="Kelly + Vol-Adjusted")
    ax_equity.plot(kelly_df.index, kelly_df["buy_and_hold"], color=COLORS["buy_and_hold"], lw=1.5, label="Buy & Hold")

    ax_equity.set_yscale("log")
    ax_equity.set_title("Equity Curve Comparison (log scale)", fontsize=12, fontweight="bold")
    ax_equity.set_ylabel("Growth Multiple (log)")
    ax_equity.legend(loc="upper left", frameon=False)

    metrics = compute_metrics(kelly_df)
    add_metrics_box(ax_equity, metrics)

    format_date_axis(ax_equity)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "results", "equity_curve.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Chart saved: {out_path}")
    print("Kelly fraction:", kelly_fraction)
    print("Metrics (Kelly + vol-adjusted):", metrics)

    return fig


if __name__ == "__main__":
    df_prices = fetch_prices()
    prices = df_prices["close"]

    short_ma, long_ma = compute_moving_averages(prices)
    position = generate_signal(short_ma, long_ma)

    fixed_result = run_fixed_backtest(prices, position)
    kelly_fraction = compute_kelly_fraction(fixed_result["strategy_return"], position)
    kelly_result = run_kelly_backtest(prices, position, kelly_fraction)

    plot_kelly_comparison(fixed_result, kelly_result, kelly_fraction)
