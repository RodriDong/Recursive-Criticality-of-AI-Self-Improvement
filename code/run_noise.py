"""Kịch bản có nhiễu quan sát:  python run_noise.py
Quan sát x(t) mỗi 0,25 năm với nhiễu Gauss độ lệch chuẩn s; chẩn đoán chế độ (2 lớp) từ cửa sổ W năm đầu.
Đặc trưng (khớp bình phương tối thiểu x ~ b t + c t^2 trên cửa sổ, ép x(0)=0):
  độ dốc    : v_tb = x(W)/W
  hình dạng : c (độ cong), c/b (gia tốc tương đối)
"""
import numpy as np, pandas as pd, pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from rsi_dataset import make_dataset

ROOT = pathlib.Path(__file__).resolve().parent.parent
t, x, df = make_dataset(4000, seed=0)
x = x.astype(np.float32).astype(float)
y = (df.R_peak > 1).astype(int).values
dt, step = 0.01, 25                       # lấy mẫu mỗi 0,25 năm
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows = []
for s in (0.0, 0.01, 0.03):
    for W in (3, 5):
        n = int(W / 0.25)
        idx = np.arange(0, n + 1) * step
        accs = {"Độ dốc": [], "Hình dạng": [], "Độ dốc + hình dạng": []}
        for seed in range(1):
            rng = np.random.default_rng(seed)
            xo = x[:, idx] + rng.normal(0, s, (len(x), len(idx)))
            xo[:, 0] = 0.0
            xo = np.clip(xo, 0, 1)
            tt = (idx * dt)[1:]
            A = np.stack([tt, tt ** 2], axis=1)                   # x ~ b t + c t^2
            coef = np.linalg.lstsq(A, xo[:, 1:].T, rcond=None)[0] # (2, N)
            b, c = coef[0], coef[1]
            F_sl = pd.DataFrame({"v_tb": xo[:, -1] / W})
            F_sh = pd.DataFrame({"c": c, "c_tren_b": np.clip(c / np.maximum(np.abs(b), 1e-6) * np.sign(b + 1e-12), -50, 50)})
            for name, Fm in [("Độ dốc", F_sl), ("Hình dạng", F_sh), ("Độ dốc + hình dạng", pd.concat([F_sl, F_sh], axis=1))]:
                accs[name].append(cross_val_score(RandomForestClassifier(60, random_state=0, n_jobs=2), Fm, y, cv=cv).mean())
        for name, v in accs.items():
            rows.append(dict(nhieu_s=s, cua_so_nam=W, bo=name, acc=np.mean(v), std=np.std(v)))
res = pd.DataFrame(rows); res.to_csv(ROOT / "data" / "ket_qua_co_nhieu.csv", index=False)
print("Mức đoán đa số:", round(max(y.mean(), 1 - y.mean()), 3))
print(res.pivot_table(index=["nhieu_s", "cua_so_nam"], columns="bo", values="acc").round(3))
