import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "strategies", "01_mean_reversion"))

import matplotlib.pyplot as plt
import pandas as pd
from common.plot_utils import setup_style, format_date_axis
from common.data_loader import fetch_prices


def compute_metrics(strategy_return, equity, position, freq_per_year=252):
    returns = strategy_return.dropna()

    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * (freq_per_year ** 0.5)
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    active = returns[position.loc[returns.index] != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0.0

    total_return = equity.iloc[-1] - 1.0

    return {
        "Sharpe": round(sharpe, 2),
        "Max DD": max_dd,
        "Win Rate": win_rate,
        "Total Return": total_return,
    }


import importlib.util
from contextlib import contextmanager


@contextmanager
def _isolated_path(folder):
    sys.path.insert(0, folder)
    try:
        yield
    finally:
        sys.path.remove(folder)


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_all_strategies():
    strategies_dir = os.path.join(os.path.dirname(__file__), "..", "strategies")

    def load_strategy(folder_name, signal_alias, backtest_alias):
        folder = os.path.join(strategies_dir, folder_name)
        with _isolated_path(folder):
            for mod_name in ("generate_signal", "backtest"):
                sys.modules.pop(mod_name, None)
            signal_mod = _load_module(signal_alias, os.path.join(folder, "generate_signal.py"))
            sys.modules["generate_signal"] = signal_mod
            backtest_mod = _load_module(backtest_alias, os.path.join(folder, "backtest.py"))
        sys.modules.pop("generate_signal", None)
        return signal_mod, backtest_mod

    mr_signal, mr_backtest = load_strategy("01_mean_reversion", "mr_signal", "mr_backtest")
    mom_signal, mom_backtest = load_strategy("02_momentum_ma_crossover", "mom_signal", "mom_backtest")
    boll_signal, boll_backtest = load_strategy("03_bollinger_breakout", "boll_signal", "boll_backtest")
    pairs_signal, pairs_backtest = load_strategy("04_pairs_trading", "pairs_signal", "pairs_backtest")
    kelly_signal, kelly_backtest = load_strategy("05_vol_adjusted_sizing", "kelly_signal", "kelly_backtest")

    results = {}

    prices = fetch_prices()["close"]

    z = mr_signal.compute_zscore(prices, window=20)
    position = mr_signal.generate_signal(z)
    df = mr_backtest.run_backtest(prices, position)
    results["Mean Reversion"] = df

    short_ma, long_ma = mom_signal.compute_moving_averages(prices)
    position = mom_signal.generate_signal(short_ma, long_ma)
    df = mom_backtest.run_backtest(prices, position)
    results["Momentum"] = df

    rolling_mean, upper_band, lower_band = boll_signal.compute_bollinger_bands(prices)
    position = boll_signal.generate_signal(prices, upper_band, lower_band)
    df = boll_backtest.run_backtest(prices, position)
    results["Bollinger Breakout"] = df

    prices_a = fetch_prices(ticker="GARAN.IS")["close"]
    prices_b = fetch_prices(ticker="AKBNK.IS")["close"]
    common_idx = prices_a.index.intersection(prices_b.index)
    prices_a = prices_a.loc[common_idx]
    prices_b = prices_b.loc[common_idx]
    spread, z = pairs_signal.compute_spread(prices_a, prices_b)
    position = pairs_signal.generate_signal(z)
    df = pairs_backtest.run_backtest(prices_a, prices_b, position)
    results["Pairs Trading"] = df

    short_ma, long_ma = kelly_signal.compute_moving_averages(prices)
    position = kelly_signal.generate_signal(short_ma, long_ma)
    fixed_df = kelly_backtest.run_fixed_backtest(prices, position)
    kelly_fraction = kelly_backtest.compute_kelly_fraction(fixed_df["strategy_return"], position)
    df = kelly_backtest.run_kelly_backtest(prices, position, kelly_fraction)
    results["Momentum + Kelly"] = df

    return results, prices


def plot_comparison(results, prices):
    setup_style()

    fig, ax = plt.subplots(figsize=(13, 7))

    palette = {
        "Mean Reversion": "#d62728",
        "Momentum": "#1f77b4",
        "Bollinger Breakout": "#9467bd",
        "Pairs Trading": "#2ca02c",
        "Momentum + Kelly": "#17becf",
    }

    for name, df in results.items():
        ax.plot(df.index, df["equity_curve"], label=name, lw=1.4, color=palette.get(name))

    buy_and_hold = (1 + prices.pct_change()).cumprod()
    ax.plot(prices.index, buy_and_hold, label="Buy & Hold", lw=1.8, color="#ff7f0e", linestyle="--")

    ax.set_yscale("log")
    ax.set_title("Strategy Comparison — Equity Curves (log scale)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth Multiple (log)")
    ax.legend(loc="upper left", frameon=False)

    format_date_axis(ax)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "results", "all_strategies_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Chart saved: {out_path}")

    return fig


def build_summary_table(results):
    rows = []
    for name, df in results.items():
        m = compute_metrics(df["strategy_return"], df["equity_curve"], df["position"])
        rows.append({
            "Strategy": name,
            "Sharpe": m["Sharpe"],
            "Max Drawdown": f"{m['Max DD']:.1%}",
            "Win Rate": f"{m['Win Rate']:.1%}",
            "Total Return": f"{m['Total Return']:.1%}",
        })

    summary = pd.DataFrame(rows)
    return summary


if __name__ == "__main__":
    results, prices = run_all_strategies()
    plot_comparison(results, prices)

    summary = build_summary_table(results)
    print(summary.to_string(index=False))

    summary.to_csv(os.path.join(os.path.dirname(__file__), "results", "summary_table.csv"), index=False)