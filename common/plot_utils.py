import matplotlib.pyplot as plt
import matplotlib.dates as mdates


COLORS = {
    "price": "#444444",
    "strategy": "#1f77b4",
    "buy_and_hold": "#ff7f0e",
    "long": "#2ca02c",
    "short": "#d62728",
    "flat": "#cccccc",
    "signal_band": "#9467bd",
}


def setup_style():
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def shade_positions(ax, index, position):
    current = None
    start = None
    n = len(index)

    for i in range(n):
        pos = position.iloc[i]
        if pos != current:
            if current is not None and current != 0:
                color = COLORS["long"] if current == 1 else COLORS["short"]
                ax.axvspan(start, index[i], color=color, alpha=0.08, lw=0)
            current = pos
            start = index[i]

    if current is not None and current != 0:
        color = COLORS["long"] if current == 1 else COLORS["short"]
        ax.axvspan(start, index[-1], color=color, alpha=0.08, lw=0)


def format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def add_metrics_box(ax, metrics: dict, loc="upper right"):
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    text = "\n".join(lines)

    if loc == "upper right":
        x, ha = 0.98, "right"
    else:
        x, ha = 0.02, "left"

    ax.text(
        x, 0.97, text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment=ha,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
