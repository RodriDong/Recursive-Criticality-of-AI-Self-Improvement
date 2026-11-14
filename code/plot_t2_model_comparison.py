
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# 1. Xác định thư mục project
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures" / "t2_model_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Đọc bảng kết quả T2
file_path = DATA_DIR / "ket_qua_t2_so_sanh_thuat_toan.csv"
df = pd.read_csv(file_path)

# 3. Các thuật toán cần so sánh
models = [
    "Random Forest",
    "Gradient Boosting",
    "SVM",
    "MLP"
]

# 4. Tên và cột chỉ số
metrics = {
    "accuracy_mean": "Accuracy",
    "balanced_accuracy_mean": "Balanced Accuracy",
    "macro_f1_mean": "Macro-F1"
}

# 5. Vẽ biểu đồ cho từng chỉ số
for metric, metric_name in metrics.items():

    plt.figure(figsize=(9, 5))

    for model in models:
        subset = df[df["thuat_toan"] == model].sort_values("cua_so_nam")

        plt.plot(
            subset["cua_so_nam"],
            subset[metric] * 100,
            marker="o",
            linewidth=2,
            markersize=6,
            label=model
        )

    plt.xlabel("Observation window (years)")
    plt.ylabel(f"{metric_name} (%)")
    plt.title(f"Comparison of ML algorithms - {metric_name}")

    plt.xticks([2, 3, 5, 8, 12])
    plt.ylim(70, 100)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # Lưu PNG và PDF
    filename = metric.replace("_mean", "")

    plt.savefig(
        OUTPUT_DIR / f"fig_t2_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        OUTPUT_DIR / f"fig_t2_{filename}.pdf",
        bbox_inches="tight"
    )

    plt.close()

print("DA VE XONG 3 BIEU DO T2!")
print("Thu muc luu:", OUTPUT_DIR)
