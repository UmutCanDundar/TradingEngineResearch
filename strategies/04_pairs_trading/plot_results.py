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
from generate_signal import compute_spread, generate_signal
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


def plot_pairs(df, z):
    setup_style()

    fig, (ax_spread, ax_equity) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )

    ax_spread.plot(df.index, z, color=COLORS["signal_band"], lw=1.0, label="Spread Z-Score")
    ax_spread.axhline(1.5, color=COLORS["short"], lw=0.8, linestyle="--")
    ax_spread.axhline(-1.5, color=COLORS["long"], lw=0.8, linestyle="--")
    ax_spread.axhline(0, color="#999999", lw=0.6)
    shade_positions(ax_spread, df.index, df["position"])

    ax_spread.set_title("Pairs Trading (GARAN.IS / AKBNK.IS) — Spread Z-Score and Positions", fontsize=12, fontweight="bold")
    ax_spread.set_ylabel("Spread Z-Score")
    ax_spread.legend(loc="upper left", frameon=False)

    ax_equity.plot(df.index, df["equity_curve"], color=COLORS["strategy"], lw=1.5, label="Pairs Strategy")
    ax_equity.plot(df.index, df["buy_and_hold"], color=COLORS["buy_and_hold"], lw=1.5, label="Buy & Hold (GARAN only)")

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
    df_a = fetch_prices(ticker="GARAN.IS")
    df_b = fetch_prices(ticker="AKBNK.IS")

    common_index = df_a.index.intersection(df_b.index)
    prices_a = df_a.loc[common_index, "close"]
    prices_b = df_b.loc[common_index, "close"]

    spread, z = compute_spread(prices_a, prices_b)
    position = generate_signal(z)
    result = run_backtest(prices_a, prices_b, position)

    plot_pairs(result, z)
