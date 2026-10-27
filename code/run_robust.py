"""Kiểm tra độ bền của kịch bản khác tác giả:  python run_robust.py [--so_seed 5] [--nhanh]

Khắc phục ba hạn chế của phiên bản đầu (chỉ một seed, lưới nhiễu nhỏ, không có độ lệch chuẩn):
  (1) Với mỗi seed s = 0..S-1 ta sinh một tập 4000 quỹ đạo MỚI (make_dataset(seed=s)); kiểm định chéo
      phân tầng 5 lớp dùng random_state = s. Báo cáo trung bình và độ lệch chuẩn GIỮA các seed
      (phương sai do dữ liệu) và độ lệch chuẩn giữa các fold (phương sai do cách chia).
  (2) Ba nhóm thí nghiệm: (a) toàn bộ quỹ đạo, (b) cửa sổ quan sát sớm W = 2, 3, 5, 8, 12 năm,
      (c) nhiễu quan sát: sigma in {0; 0,005; 0,01; 0,02; 0,03; 0,05}, W in {2, 3, 5, 8}; nhiễu Gauss
      được gieo bằng seed s nên mỗi seed là một lần thực nghiệm độc lập (dữ liệu + nhiễu + cách chia fold).
  (3) Ghi từng seed vào CSV ngay khi xong (có thể dừng giữa chừng và dùng kết quả đã có).

Kết quả:  ../data/robust_{toan_quy_dao,cua_so_som,nhieu}.csv  (mỗi dòng một seed)
          ../data/robust_tom_tat_*.csv  (trung bình +- std qua các seed)
Thời gian: khoảng 4-6 phút mỗi seed trên 1 CPU (--nhanh: 2 seed, lưới nhỏ, chừng 3 phút).
"""
import argparse, pathlib, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from rsi_dataset import make_dataset

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--so_seed", type=int, default=5)
ap.add_argument("--nhanh", action="store_true")
ARGS = ap.parse_args()
SEEDS = range(2 if ARGS.nhanh else ARGS.so_seed)
SIGMAS = (0.0, 0.01, 0.03) if ARGS.nhanh else (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
WIN_NOISE = (3, 5) if ARGS.nhanh else (2, 3, 5, 8)
WIN_EARLY = (2, 3, 5, 8, 12)
DT, T = 0.01, 60.0
N_TREE_CLEAN, N_TREE_NOISE = 100, 50


def cv_scores(F, y, seed, n_tree):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = cross_validate(RandomForestClassifier(n_tree, random_state=seed, n_jobs=1), F, y, cv=cv, scoring=["accuracy", "balanced_accuracy"])
    return dict(acc=s["test_accuracy"].mean(), acc_fold_std=s["test_accuracy"].std(), bal=s["test_balanced_accuracy"].mean())


def full_features(t, x):
    """Mười đặc trưng của toàn bộ quỹ đạo (cùng định nghĩa với run_all.py)."""
    N = len(x); v = np.gradient(x, DT, axis=1); F = pd.DataFrame(index=range(N))
    ct = lambda thr: np.where((x >= thr).any(axis=1), t[(x >= thr).argmax(axis=1)], T)
    F["v_dau"] = (x[:, 200] - x[:, 0]) / 2.0; F["v_max"] = v.max(axis=1)
    F["T25"], F["T50"], F["T80"] = ct(.25), ct(.5), ct(.8)
    F["gia_toc_som"] = (v[:, 300] - v[:, 50]) / v[:, 50]; F["v_max_tren_v_dau"] = F.v_max / F.v_dau
    F["t_v_max"] = t[v.argmax(axis=1)]; F["T80_tren_T50"], F["T50_tren_T25"] = F.T80 / F.T50, F.T50 / F.T25
    return F


def early_features(x, W):
    n = int(round(W / DT)); xs = x[:, :n + 1]; vv = np.maximum(np.gradient(xs, DT, axis=1), 1e-9); i05 = int(0.5 / DT)
    sl = pd.DataFrame({"v_tb": xs[:, n] / W})
    sh = pd.DataFrame({"gia_toc": np.clip((vv[:, n] - vv[:, i05]) / vv[:, i05], -1, 50),
                       "dlnv": np.clip(np.log(vv[:, n] / vv[:, i05]) / (W - 0.5), -20, 20),
                       "cong": (xs[:, n] - 2 * xs[:, n // 2] + xs[:, 0]) / (W / 2) ** 2})
    return sl, sh


def noisy_features(x, W, sigma, rng, step=25):
    """Quan sát x(t) mỗi 0,25 năm + nhiễu Gauss; khớp bình phương tối thiểu x ~ b t + c t^2 (x(0) = 0 biết chính xác)."""
    idx = np.arange(0, int(W / 0.25) + 1) * step
    xo = x[:, idx] + rng.normal(0, sigma, (len(x), len(idx))); xo[:, 0] = 0.0; xo = np.clip(xo, 0, 1)
    tt = (idx * DT)[1:]; coef = np.linalg.lstsq(np.stack([tt, tt ** 2], axis=1), xo[:, 1:].T, rcond=None)[0]
    b, c = coef
    sl = pd.DataFrame({"v_tb": xo[:, -1] / W})
    sh = pd.DataFrame({"c": c, "c_tren_b": np.clip(c / np.maximum(np.abs(b), 1e-6) * np.sign(b + 1e-12), -50, 50)})
    return sl, sh


def append(path, rows):
    df = pd.DataFrame(rows); df.to_csv(path, mode="a", header=not path.exists(), index=False)


paths = {k: DATA / f"robust_{k}.csv" for k in ("toan_quy_dao", "cua_so_som", "nhieu")}
for p in paths.values():
    if p.exists(): p.unlink()                      # chạy lại từ đầu
for s in SEEDS:
    t0 = time.time()
    t, x, df = make_dataset(4000, seed=s); x = x.astype(np.float32).astype(float)
    y2 = (df.R_peak > 1).astype(int).values; y3 = np.digitize(df.R_peak.values, [1.0, 2.0]); maj = max(y2.mean(), 1 - y2.mean())

    # (a) toàn bộ quỹ đạo
    F = full_features(t, x)
    SETS = {"Chỉ độ dốc": ["v_dau", "v_max"], "Chỉ mốc thời gian": ["T25", "T50", "T80"],
            "Hình dạng": ["gia_toc_som", "v_max_tren_v_dau", "t_v_max", "T80_tren_T50", "T50_tren_T25"], "Tất cả đặc trưng": list(F.columns)}
    rows = [dict(seed=s, bo=name, bai_toan=lab, majority=maj if lab == "2 lớp" else np.bincount(y3).max() / len(y3), **cv_scores(F[cols], y, s, N_TREE_CLEAN))
            for name, cols in SETS.items() for lab, y in (("2 lớp", y2), ("3 lớp", y3))]
    append(paths["toan_quy_dao"], rows)

    # (b) cửa sổ quan sát sớm (quỹ đạo sạch)
    rows = []
    for W in WIN_EARLY:
        sl, sh = early_features(x, W)
        for name, Fm in (("Độ dốc", sl), ("Hình dạng", sh), ("Độ dốc + hình dạng", pd.concat([sl, sh], axis=1))):
            rows.append(dict(seed=s, W=W, bo=name, majority=maj, **cv_scores(Fm, y2, s, N_TREE_CLEAN)))
    append(paths["cua_so_som"], rows)

    # (c) nhiễu quan sát
    rows = []
    for sg in SIGMAS:
        for W in WIN_NOISE:
            sl, sh = noisy_features(x, W, sg, np.random.default_rng(1000 * s + int(W)))
            for name, Fm in (("Độ dốc", sl), ("Hình dạng", sh), ("Độ dốc + hình dạng", pd.concat([sl, sh], axis=1))):
                rows.append(dict(seed=s, sigma=sg, W=W, bo=name, majority=maj, **cv_scores(Fm, y2, s, N_TREE_NOISE)))
    append(paths["nhieu"], rows)
    print(f"seed {s} xong sau {time.time() - t0:.0f}s", flush=True)

# ---------------- tóm tắt: trung bình +- std giữa các seed
def summarize(path, keys, out):
    d = pd.read_csv(path)
    g = d.groupby(keys, sort=False).agg(acc_tb=("acc", "mean"), acc_std_seed=("acc", "std"), acc_std_fold=("acc_fold_std", "mean"),
                                         bal_tb=("bal", "mean"), bal_std_seed=("bal", "std"), majority=("majority", "mean"), n_seed=("seed", "nunique")).reset_index()
    g.to_csv(DATA / out, index=False); return g

print(summarize(paths["toan_quy_dao"], ["bo", "bai_toan"], "robust_tom_tat_toan_quy_dao.csv").round(3).to_string(index=False))
print(summarize(paths["cua_so_som"], ["W", "bo"], "robust_tom_tat_cua_so_som.csv").round(3).to_string(index=False))
print(summarize(paths["nhieu"], ["sigma", "W", "bo"], "robust_tom_tat_nhieu.csv").round(3).to_string(index=False))
