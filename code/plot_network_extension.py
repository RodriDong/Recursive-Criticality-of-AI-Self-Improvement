
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures" / "network_extension"

FILE_30Y = DATA_DIR / "network_extension_summary_30y.csv"
FILE_50Y = DATA_DIR / "network_extension_summary_50y.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colors for network sizes
COLORS = {
    3: "#2878B5",
    5: "#F28E2B",
    10: "#2CA02C",
}

NETWORK_SIZES = [3, 5, 10]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find file: {file_path}\n"
            "Please check the CSV filename and location."
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "n_agents",
        "rho_target",
        "rho_K_mean",
        "rho_M_max_mean",
        "delta_T_mean",
        "success_rate",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {file_path.name}: {missing}"
        )

    return df


df30 = load_data(FILE_30Y)
df50 = load_data(FILE_50Y)

# Convert success rate to percentage if stored as proportions
for df in [df30, df50]:
    if df["success_rate"].max() <= 1:
        df["success_rate"] = df["success_rate"] * 100

print("Loaded 30-year data:", len(df30), "rows")
print("Loaded 50-year data:", len(df50), "rows")


# ============================================================
# 3. HELPER FUNCTION: SAVE PNG AND PDF
# ============================================================

def save_figure(fig, filename):
    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        bbox_inches="tight",
        dpi=300
    )

    fig.savefig(
        OUTPUT_DIR / f"{filename}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {filename}.png and {filename}.pdf")


# ============================================================
# FIGURE 1: AGI-ASI INTERVAL
# ============================================================

def plot_interval():

    fig, ax = plt.subplots(figsize=(10, 6))

    # 30-year results: lines with circle markers
    for n in NETWORK_SIZES:

        data = df30[df30["n_agents"] == n].sort_values(
            "rho_target"
        )

        ax.plot(
            data["rho_K_mean"],
            data["delta_T_mean"],
            color=COLORS[n],
            marker="o",
            linewidth=2,
            markersize=7,
            label=f"{n} agents - 30 years"
        )

    # 50-year results: X markers using the same colors
    for n in NETWORK_SIZES:

        data = df50[df50["n_agents"] == n].sort_values(
            "rho_target"
        )

        ax.scatter(
            data["rho_K_mean"],
            data["delta_T_mean"],
            color=COLORS[n],
            marker="x",
            s=100,
            linewidth=2.5,
            zorder=5
        )

    # Additional legend for 50-year markers
    legend_50y = Line2D(
        [0], [0],
        color="black",
        marker="x",
        linestyle="None",
        markersize=9,
        markeredgewidth=2,
        label="50 years"
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(legend_50y)

    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True
    )

    ax.set_title(
        "AGI-ASI Interval Across Network Sizes"
    )

    ax.set_xlabel("Spectral radius rho(K)")
    ax.set_ylabel("AGI to ASI interval (years)")

    ax.set_xlim(left=0.45)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    save_figure(fig, "fig_network_deltaT")


# ============================================================
# FIGURE 2: SPECTRAL RADIUS AND CRITICALITY
# ============================================================

def plot_criticality():

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate mean, minimum and maximum across network sizes
    grouped = (
        df30.groupby("rho_target")["rho_M_max_mean"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values("rho_target")
    )

    x = grouped["rho_target"]
    y = grouped["mean"]

    # Range across 3, 5 and 10 agents
    ax.fill_between(
        x,
        grouped["min"],
        grouped["max"],
        color="#2878B5",
        alpha=0.18,
        label="Range across network sizes"
    )

    # Mean curve
    ax.plot(
        x,
        y,
        color="#2878B5",
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Mean maximum rho(M)"
    )

    # Criticality threshold
    ax.axhline(
        y=1,
        color="#D62728",
        linestyle="--",
        linewidth=2,
        label="Criticality threshold = 1"
    )

    ax.set_title(
        "Relationship Between Network Spectral Radius and Criticality"
    )

    ax.set_xlabel("Spectral radius rho(K)")
    ax.set_ylabel("Maximum reproduction number rho(M)")

    ax.legend(loc="upper left", frameon=True)

    ax.set_xlim(left=0.45)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    save_figure(fig, "fig_spectral_radius")


# ============================================================
# FIGURE 3: AGI AND ASI ACHIEVEMENT RATE
# ============================================================


def plot_success_rate():

    fig, ax = plt.subplots(figsize=(10, 6))

    # 30-year results: separate lines for each network size
    for n in NETWORK_SIZES:

        data = df30[df30["n_agents"] == n].sort_values(
            "rho_target"
        )

        ax.plot(
            data["rho_K_mean"],
            data["success_rate"],
            color=COLORS[n],
            linestyle="-",
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=f"{n} agents - 30 years"
        )

    # 50-year results: average across network sizes
    data_50y = (
        df50.groupby("rho_target")
        .agg(
            rho_K=("rho_K_mean", "mean"),
            success_rate=("success_rate", "mean")
        )
        .reset_index()
        .sort_values("rho_target")
    )

    ax.plot(
        data_50y["rho_K"],
        data_50y["success_rate"],
        color="black",
        linestyle="--",
        marker="X",
        linewidth=2.2,
        markersize=9,
        label="50 years (all network sizes)"
    )

    ax.set_title("AGI and ASI Achievement Rate")

    ax.set_xlabel("Spectral radius rho(K)")
    ax.set_ylabel("Successful simulations (%)")

    ax.set_ylim(-2, 105)
    ax.set_xlim(left=0.45)

    ax.set_xticks(sorted(
        df30["rho_K_mean"].unique()
    ))

    ax.legend(
        loc="lower right",
        frameon=True,
        ncol=2
    )

    fig.tight_layout()

    save_figure(fig, "fig_success_rate")

# ============================================================
# 4. GENERATE ALL FIGURES
# ============================================================

if __name__ == "__main__":

    plot_interval()
    plot_criticality()
    plot_success_rate()

    print("\n===================================")
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("Output folder:", OUTPUT_DIR)
    print("===================================")
