
import pathlib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
FIGS = ROOT / "figures" / "t3_adaptive_prediction"

FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "ket_qua_t3_adaptive.csv")

thresholds = df["confidence_threshold"] * 100


# ============================================================
# 1. ACCURACY VS CONFIDENCE THRESHOLD
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    df["accuracy"] * 100,
    marker="o",
    label="Accuracy"
)

plt.plot(
    thresholds,
    df["balanced_accuracy"] * 100,
    marker="s",
    label="Balanced Accuracy"
)

plt.plot(
    thresholds,
    df["macro_f1"] * 100,
    marker="^",
    label="Macro-F1"
)

plt.xlabel("Confidence Threshold (%)")
plt.ylabel("Performance (%)")
plt.title("Prediction Performance under Different Confidence Thresholds")
plt.xticks(thresholds)
plt.ylim(85, 100)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(FIGS / "fig_t3_performance.png", dpi=300)
plt.savefig(FIGS / "fig_t3_performance.pdf")
plt.close()


# ============================================================
# 2. AVERAGE OBSERVATION TIME
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    df["average_observation_years"],
    marker="o",
    linewidth=2
)

plt.xlabel("Confidence Threshold (%)")
plt.ylabel("Average Observation Time (Years)")
plt.title("Average Observation Time under Different Thresholds")
plt.xticks(thresholds)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(FIGS / "fig_t3_observation_time.png", dpi=300)
plt.savefig(FIGS / "fig_t3_observation_time.pdf")
plt.close()



# ============================================================
# 3. EARLY STOPPING AND THRESHOLD NOT REACHED RATES
# ============================================================

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    thresholds,
    df["early_stop_rate"] * 100,
    marker="o",
    linewidth=2,
    label="Early Stop Rate"
)

ax1.set_xlabel("Confidence Threshold (%)")
ax1.set_ylabel("Early Stop Rate (%)")
ax1.set_ylim(90, 101)
ax1.set_xticks(thresholds)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()

ax2.plot(
    thresholds,
    df["unresolved_rate"] * 100,
    marker="s",
    linewidth=2,
    linestyle="--",
    color="tab:orange",
    label="Threshold Not Reached Rate"
)

ax2.set_ylabel("Threshold Not Reached Rate (%)")
ax2.set_ylim(0, 2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="center right"
)

plt.title("Adaptive Stopping Behavior under Different Thresholds")
plt.tight_layout()

plt.savefig(FIGS / "fig_t3_stopping_rate.png", dpi=300)
plt.savefig(FIGS / "fig_t3_stopping_rate.pdf")
plt.close()


print("T3 figures saved successfully:")
print(FIGS)
