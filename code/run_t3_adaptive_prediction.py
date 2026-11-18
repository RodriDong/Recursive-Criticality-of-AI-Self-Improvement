
"""
T3 - Adaptive Early Prediction
Confidence-based stopping using Gradient Boosting.
"""

import pathlib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = pathlib.Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

N = 4000
dt = 0.01

WINDOWS = [2, 3, 5, 8, 12]
THRESHOLDS = [0.80, 0.90, 0.95]


# ============================================================
# 2. LOAD ORIGINAL DATASET
# ============================================================

from rsi_dataset import make_dataset

t, x, df = make_dataset(N, seed=0)

x = x.astype(np.float32).astype(float)

y = (df["R_peak"] > 1).astype(int).values

print("Number of trajectories:", N)
print("Class 0 ratio:", round((y == 0).mean(), 4))
print("Class 1 ratio:", round((y == 1).mean(), 4))


# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================

def extract_early_features(W):

    n = int(W / dt)
    xs = x[:, :n + 1]

    vv = np.maximum(
        np.gradient(xs, dt, axis=1),
        1e-9
    )

    i05 = int(0.5 / dt)

    features = pd.DataFrame({
        "v_tb": xs[:, n] / W,

        "gia_toc": np.clip(
            (vv[:, n] - vv[:, i05]) / vv[:, i05],
            -1,
            50
        ),

        "dlnv": np.clip(
            np.log(vv[:, n] / vv[:, i05]) / (W - 0.5),
            -20,
            20
        ),

        "cong": (
            xs[:, n]
            - 2 * xs[:, n // 2]
            + xs[:, 0]
        ) / (W / 2) ** 2
    })

    return features


# ============================================================
# 4. GENERATE OUT-OF-FOLD PROBABILITIES
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0
)

# Predictions for every trajectory and every observation window
oof_probabilities = {}

for W in WINDOWS:

    print(f"\nProcessing window W={W} years...")

    X = extract_early_features(W)

    oof_prob = np.zeros((N, 2))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):

        model = HistGradientBoostingClassifier(
            random_state=0
        )

        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        probabilities = model.predict_proba(
            X.iloc[test_idx]
        )

        oof_prob[test_idx] = probabilities

        print(f"  Fold {fold + 1}/5 completed")

    oof_probabilities[W] = oof_prob


# ============================================================
# 5. ADAPTIVE STOPPING STRATEGY
# ============================================================

results = []
details = []

for threshold in THRESHOLDS:

    print(f"\nEvaluating confidence threshold: {threshold}")

    selected_window = np.full(N, WINDOWS[-1])
    selected_prediction = np.zeros(N, dtype=int)
    selected_confidence = np.zeros(N)

    stopped = np.zeros(N, dtype=bool)

    # Check each observation window from earliest to latest
    for W in WINDOWS:

        probs = oof_probabilities[W]

        predictions = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        # Stop only samples that have not stopped before
        eligible = (
            (~stopped)
            & (confidence >= threshold)
        )

        selected_window[eligible] = W
        selected_prediction[eligible] = predictions[eligible]
        selected_confidence[eligible] = confidence[eligible]

        stopped[eligible] = True

    # Samples that never reached the threshold:
    # use the prediction at the final window (W=12)
    unresolved = ~stopped

    final_probs = oof_probabilities[WINDOWS[-1]]

    selected_prediction[unresolved] = np.argmax(
        final_probs[unresolved],
        axis=1
    )

    selected_confidence[unresolved] = np.max(
        final_probs[unresolved],
        axis=1
    )

    # Evaluation
    accuracy = accuracy_score(
        y,
        selected_prediction
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        selected_prediction
    )

    macro_f1 = f1_score(
        y,
        selected_prediction,
        average="macro"
    )

    average_window = selected_window.mean()

    early_stop_rate = np.mean(
        selected_window < WINDOWS[-1]
    )

    unresolved_rate = unresolved.mean()

    results.append({
        "confidence_threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "average_observation_years": average_window,
        "early_stop_rate": early_stop_rate,
        "unresolved_rate": unresolved_rate
    })

    # Save sample-level details
    for i in range(N):

        details.append({
            "sample_id": i,
            "true_label": y[i],
            "predicted_label": selected_prediction[i],
            "confidence_threshold": threshold,
            "selected_window": selected_window[i],
            "prediction_confidence": selected_confidence[i],
            "stopped_early": selected_window[i] < WINDOWS[-1],
            "reached_threshold": stopped[i]
        })

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"Balanced Accuracy: {balanced_accuracy:.4f} | "
        f"Macro-F1: {macro_f1:.4f}"
    )

    print(
        f"Average observation: {average_window:.2f} years | "
        f"Early stop rate: {early_stop_rate:.2%} | "
        f"Unresolved: {unresolved_rate:.2%}"
    )


# ============================================================
# 6. SAVE RESULTS
# ============================================================

summary = pd.DataFrame(results)

summary.to_csv(
    DATA / "ket_qua_t3_adaptive.csv",
    index=False,
    encoding="utf-8-sig"
)

detail = pd.DataFrame(details)

detail.to_csv(
    DATA / "ket_qua_t3_chi_tiet.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("T3 EXPERIMENT COMPLETED")
print("=" * 60)

print(summary.round(4).to_string(index=False))

print("\nSaved:")
print(DATA / "ket_qua_t3_adaptive.csv")
print(DATA / "ket_qua_t3_chi_tiet.csv")
