
"""
T2 - So sánh các thuật toán Machine Learning
Sử dụng cùng dữ liệu và cách chia fold với thí nghiệm RF ban đầu.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate


# ============================================================
# 1. ĐƯỜNG DẪN PROJECT
# ============================================================

ROOT = pathlib.Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
FIGS = ROOT / "figures" / "t2_model_comparison"

DATA.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. TẠO LẠI BỘ DỮ LIỆU GỐC
# ============================================================

from rsi_dataset import make_dataset

N = 4000
dt = 0.01
T = 60.0

t, x, df = make_dataset(N, seed=0)

# Giữ đúng cách lưu dữ liệu của thí nghiệm cũ
x = x.astype(np.float32).astype(float)

# Nhãn phân loại 2 lớp
y = (df["R_peak"] > 1).astype(int).values

print("Số quỹ đạo:", N)
print("Tỷ lệ lớp 0:", round((y == 0).mean(), 4))
print("Tỷ lệ lớp 1:", round((y == 1).mean(), 4))


# ============================================================
# 3. CẤU HÌNH THỰC NGHIỆM
# ============================================================

WINDOWS = [2, 3, 5, 8, 12]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0
)

SCORING = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro"
]


# ============================================================
# 4. KHAI BÁO CÁC THUẬT TOÁN
# ============================================================

def get_models():

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=0,
            n_jobs=-1
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=0
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale"
            )
        ),

        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1000,
                random_state=0,
                early_stopping=True
            )
        )
    }


# ============================================================
# 5. TRÍCH XUẤT 4 ĐẶC TRƯNG Ở TỪNG CỬA SỔ
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
# 6. HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ============================================================

results = []
fold_results = []

for W in WINDOWS:

    print("\n" + "=" * 55)
    print(f"CỬA SỔ QUAN SÁT: {W} NĂM")
    print("=" * 55)

    X = extract_early_features(W)

    for model_name, model in get_models().items():

        print("Đang chạy:", model_name)

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            return_train_score=False
        )

        accuracy = scores["test_accuracy"]
        balanced = scores["test_balanced_accuracy"]
        macro_f1 = scores["test_f1_macro"]

        results.append({
            "cua_so_nam": W,
            "thuat_toan": model_name,
            "accuracy_mean": accuracy.mean(),
            "accuracy_std": accuracy.std(),
            "balanced_accuracy_mean": balanced.mean(),
            "balanced_accuracy_std": balanced.std(),
            "macro_f1_mean": macro_f1.mean(),
            "macro_f1_std": macro_f1.std()
        })

        for fold in range(5):

            fold_results.append({
                "cua_so_nam": W,
                "thuat_toan": model_name,
                "fold": fold + 1,
                "accuracy": accuracy[fold],
                "balanced_accuracy": balanced[fold],
                "macro_f1": macro_f1[fold]
            })

        print(
            f"Accuracy: {accuracy.mean():.4f} | "
            f"Balanced Accuracy: {balanced.mean():.4f} | "
            f"Macro-F1: {macro_f1.mean():.4f}"
        )


# ============================================================
# 7. LƯU KẾT QUẢ
# ============================================================

summary = pd.DataFrame(results)

summary.to_csv(
    DATA / "ket_qua_t2_so_sanh_thuat_toan.csv",
    index=False,
    encoding="utf-8-sig"
)

detail = pd.DataFrame(fold_results)

detail.to_csv(
    DATA / "ket_qua_t2_chi_tiet_5fold.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nĐÃ LƯU BẢNG KẾT QUẢ T2")

print(
    summary[
        [
            "cua_so_nam",
            "thuat_toan",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean"
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# 8. VẼ BIỂU ĐỒ SO SÁNH ACCURACY
# ============================================================

plt.figure(figsize=(9, 5))

for name in summary["thuat_toan"].unique():

    subset = summary[
        summary["thuat_toan"] == name
    ].sort_values("cua_so_nam")

    plt.plot(
        subset["cua_so_nam"],
        subset["accuracy_mean"],
        marker="o",
        linewidth=1.8,
        label=name
    )

plt.xlabel("Thời gian quan sát (năm)")
plt.ylabel("Accuracy")
plt.title("So sánh Accuracy giữa các thuật toán")
plt.xticks(WINDOWS)
plt.ylim(0, 1.02)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGS / "fig_t2_accuracy.png",
    dpi=300
)

plt.savefig(
    FIGS / "fig_t2_accuracy.pdf"
)

plt.close()


# ============================================================
# 9. VẼ BIỂU ĐỒ SO SÁNH MACRO-F1
# ============================================================

plt.figure(figsize=(9, 5))

for name in summary["thuat_toan"].unique():

    subset = summary[
        summary["thuat_toan"] == name
    ].sort_values("cua_so_nam")

    plt.plot(
        subset["cua_so_nam"],
        subset["macro_f1_mean"],
        marker="o",
        linewidth=1.8,
        label=name
    )

plt.xlabel("Thời gian quan sát (năm)")
plt.ylabel("Macro-F1")
plt.title("So sánh Macro-F1 giữa các thuật toán")
plt.xticks(WINDOWS)
plt.ylim(0, 1.02)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGS / "fig_t2_macro_f1.png",
    dpi=300
)

plt.savefig(
    FIGS / "fig_t2_macro_f1.pdf"
)

plt.close()

print("\nĐÃ HOÀN THÀNH THỰC NGHIỆM T2!")
