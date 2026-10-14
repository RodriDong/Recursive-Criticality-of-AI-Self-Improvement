"""Chạy toàn bộ thực nghiệm của báo cáo:  python run_all.py
Kết quả ghi vào ../data (CSV, JSON) và ../figures (PDF, PNG). Thời gian chạy vài phút.

Các bước
  1. Kịch bản giống tác giả: tái hiện Table 1 (bốn kịch bản) và Table 2 (ba mạng nhiều tác nhân).
  2. Vẽ lại Fig. 3 và Fig. 4 với nhãn tiếng Việt.
  3. Kịch bản khác tác giả: sinh 4000 quỹ đạo, trích đặc trưng, phân lớp, phân cụm,
     kiểm tra với cửa sổ quan sát sớm.
"""
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from rsi_model import simulate, first_crossing, baseline_no_rsi, r_ai, R_REF, X_AGI, X_ASI
from rsi_network import CONFIGS, simulate_network, crossing, spectral_radius, reproduction_numbers
from rsi_dataset import make_dataset

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, FIGS = ROOT / "data", ROOT / "figures"
DATA.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
OUT = {}

# ---------------------------------------------------------------- 1. Giống tác giả
SCEN = [(0.5, "Tăng trưởng mượt", "tab:blue"), (3, "Siêu tới hạn yếu", "tab:green"),
        (6, "Bùng nổ tạm thời", "tab:orange"), (15, "Chuyển pha nhanh", "tab:red")]
PAPER_T1 = {0.5: (21.35, 74.13), 3: (12.90, 24.73), 6: (8.40, 11.08), 15: (4.50, 4.95)}
t_agi0, t_asi0 = baseline_no_rsi()
rows = [dict(kich_ban="Không đệ quy", a=0, T_AGI=t_agi0, T_ASI=t_asi0, T_AGI_bao=24.00, T_ASI_bao=96.00)]
trajs = {}
for a, name, col in SCEN:
    t, x = simulate(a, T=100.0 if a < 3 else 30.0)
    trajs[a] = (t, x)
    rows.append(dict(kich_ban=name, a=a, T_AGI=first_crossing(t, x, X_AGI), T_ASI=first_crossing(t, x, X_ASI),
                     T_AGI_bao=PAPER_T1[a][0], T_ASI_bao=PAPER_T1[a][1]))
t1 = pd.DataFrame(rows); t1["dT"] = t1.T_ASI - t1.T_AGI
t1.to_csv(DATA / "bang1_tai_hien.csv", index=False); print(t1.round(2).to_string(index=False))
OUT["table1_max_abs_diff"] = float(np.nanmax(np.abs(t1[["T_AGI", "T_ASI"]].values - t1[["T_AGI_bao", "T_ASI_bao"]].values)))

PAPER_T2 = {"Mạng phòng lab đóng": (10.40, 16.53), "Hệ sinh thái mở": (9.75, 12.58), "Cạnh tranh toàn cầu": (8.97, 13.42)}
rows, nets = [], {}
for name, c in CONFIGS.items():
    K = np.array(c["K"]); t, x = simulate_network(**c); nets[name] = (t, x)
    ta = min(crossing(t, x[:, i], X_AGI) for i in range(3)); ts = min(crossing(t, x[:, i], X_ASI) for i in range(3))
    rows.append(dict(cau_hinh=name, rho_K=spectral_radius(K), max_Kii=K.diagonal().max(), m_r=c["m"], tau=c["tau"],
                     T_AGI=ta, T_ASI=ts, dT=ts - ta, T_AGI_bao=PAPER_T2[name][0], T_ASI_bao=PAPER_T2[name][1]))
t2 = pd.DataFrame(rows); t2.to_csv(DATA / "bang2_tai_hien.csv", index=False); print(t2.round(3).to_string(index=False))
OUT["table2_max_abs_diff"] = float(np.max(np.abs(t2[["T_AGI", "T_ASI"]].values - t2[["T_AGI_bao", "T_ASI_bao"]].values)))

# ---------------------------------------------------------------- 2. Hình giống tác giả (tiếng Việt)
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
t0 = np.linspace(0, 30, 600)
ax[0].plot(t0, R_REF * t0 / (1 + R_REF * t0), "k--", lw=0.9, label="Không đệ quy")
for a, name, col in SCEN:
    t, x = trajs[a]; m = t <= 30
    ax[0].plot(t[m], x[m], c=col, lw=1.2, label=f"{name} ($a$={a})")
    ax[1].plot(t[m], r_ai(x[m], a), c=col, lw=1.2)
for y, n in [(X_AGI, "AGI"), (X_ASI, "ASI")]:
    ax[0].axhline(y, ls=":", c="gray", lw=0.8); ax[0].text(0.5, y + 0.015, n, color="gray")
ax[1].axhline(1, ls="--", c="tab:blue", lw=0.9); ax[1].text(21, 1.1, r"$\mathcal{R}_{AI}=1$", color="tab:blue")
ax[0].set(xlabel="Năm", ylabel="Năng lực $x(t)$", title="(a) Quỹ đạo năng lực", ylim=(0, 1.02))
ax[1].set(xlabel="Năm", ylabel=r"$\mathcal{R}_{AI}(t)$", title="(b) Số tái sản xuất đệ quy", ylim=(0, 6))
ax[0].legend(fontsize=6, loc="lower right"); [a_.grid(alpha=.3) for a_ in ax]
plt.tight_layout(); plt.savefig(FIGS / "fig3_vi.pdf"); plt.savefig(FIGS / "fig3_vi.png", dpi=170); plt.close()

fig, ax = plt.subplots(2, 3, figsize=(6.4, 4.2), sharex=True)
wt, wx = trajs[3]; m = wt <= 30
for j, (name, c) in enumerate(CONFIGS.items()):
    t, x = nets[name]; yrs = 2026 + t
    for i, (lab, col) in enumerate(zip("ABC", ["tab:blue", "tab:orange", "tab:green"])):
        ax[0, j].plot(yrs, x[:, i], c=col, lw=1.1, label=f"Tác nhân {lab}")
    ax[0, j].plot(2026 + wt[m], wx[m], c="gray", ls="--", lw=0.9, label="Tham chiếu (siêu tới hạn yếu)")
    for y in (X_AGI, X_ASI): ax[0, j].axhline(y, ls=":", c="gray", lw=0.7)
    loc, net = zip(*[reproduction_numbers(c["K"], x[s]) for s in range(len(t))])
    loc = np.array(loc)
    for i, col in enumerate(["tab:blue", "tab:orange", "tab:green"]): ax[1, j].plot(yrs, loc[:, i], c=col, lw=1.0, alpha=.85)
    ax[1, j].plot(yrs, net, "k--", lw=1.3, label=r"$\mathcal{R}_{net}(t)$"); ax[1, j].axhline(1, c="gray", lw=0.7, ls=":")
    ax[0, j].set_title(name, fontsize=7.5); ax[0, j].set_ylim(0, 1.02); ax[1, j].set_ylim(0, 2.5); ax[1, j].set_xlabel("Năm")
    ax[0, j].grid(alpha=.3); ax[1, j].grid(alpha=.3)
ax[0, 0].set_ylabel("Năng lực $x_i(t)$"); ax[1, 0].set_ylabel("Số tái sản xuất")
ax[0, 0].legend(fontsize=5.5, loc="lower right"); ax[1, 0].legend(fontsize=6, loc="upper right")
plt.tight_layout(); plt.savefig(FIGS / "fig4_vi.pdf"); plt.savefig(FIGS / "fig4_vi.png", dpi=170); plt.close()

# ---------------------------------------------------------------- 3. Khác tác giả: khai phá dữ liệu
N = 4000
t, x, df = make_dataset(N, seed=0)
x = x.astype(np.float32).astype(float)            # lưu float32 cho gọn, kết quả báo cáo dùng đúng dữ liệu này
dt, T = 0.01, 60.0
y2 = (df.R_peak > 1).astype(int).values
y3 = np.digitize(df.R_peak.values, [1.0, 2.0])    # 0: cận tới hạn; 1: siêu tới hạn yếu; 2: siêu tới hạn mạnh
OUT.update(n_quy_dao=N, n_diem=int(x.shape[1]), ti_le_2lop=[float(1 - y2.mean()), float(y2.mean())],
           ti_le_3lop=[float(v) for v in np.bincount(y3) / N])
df.to_csv(DATA / "tham_so.csv", index=False)
print(df.describe().T[["min", "50%", "max"]].round(3))

def cross_time(xx, thr):
    ok = xx >= thr; idx = ok.argmax(axis=1)
    return np.where(ok.any(axis=1), t[idx], T)

v = np.gradient(x, dt, axis=1)
F = pd.DataFrame(index=range(N))
F["v_dau"] = (x[:, 200] - x[:, 0]) / 2.0
F["v_max"] = v.max(axis=1)
F["T25"], F["T50"], F["T80"] = cross_time(x, .25), cross_time(x, .5), cross_time(x, .8)
F["gia_toc_som"] = (v[:, 300] - v[:, 50]) / v[:, 50]
F["v_max_tren_v_dau"] = F.v_max / F.v_dau
F["t_v_max"] = t[v.argmax(axis=1)]
F["T80_tren_T50"], F["T50_tren_T25"] = F.T80 / F.T50, F.T50 / F.T25
F.assign(R_peak=df.R_peak, m=df.m, nhan2=y2, nhan3=y3).to_csv(DATA / "dac_trung.csv", index=False)

SETS = {"Chỉ độ dốc": ["v_dau", "v_max"], "Chỉ mốc thời gian": ["T25", "T50", "T80"],
        "Hình dạng": ["gia_toc_som", "v_max_tren_v_dau", "t_v_max", "T80_tren_T50", "T50_tren_T25"],
        "Tất cả đặc trưng": list(F.columns)}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows = []
for name, cols in SETS.items():
    for lab, y in [("2 lớp", y2), ("3 lớp", y3)]:
        s = cross_validate(RandomForestClassifier(300, random_state=0, n_jobs=-1), F[cols], y, cv=cv,
                           scoring=["accuracy", "balanced_accuracy", "f1_macro"])
        rows.append(dict(bo_dac_trung=name, bai_toan=lab, so_dac_trung=len(cols), accuracy=s["test_accuracy"].mean(),
                         bal_acc=s["test_balanced_accuracy"].mean(), f1_macro=s["test_f1_macro"].mean(), std=s["test_accuracy"].std()))
full = pd.DataFrame(rows); full.to_csv(DATA / "ket_qua_phan_lop.csv", index=False); print(full.round(3).to_string(index=False))

imp = pd.Series(RandomForestClassifier(500, random_state=0, n_jobs=-1).fit(F, y2).feature_importances_, F.columns).sort_values(ascending=False)
imp.to_csv(DATA / "do_quan_trong.csv", header=["importance"])
sp = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))
OUT["spearman"] = dict(v_dau_m=sp(F.v_dau, df.m), v_dau_Rpeak=sp(F.v_dau, df.R_peak), v_max_m=sp(F.v_max, df.m), v_max_Rpeak=sp(F.v_max, df.R_peak))
Z = StandardScaler().fit_transform(np.log(F[SETS["Tất cả đặc trưng"]].clip(lower=1e-6).replace([np.inf], 1e6)))
OUT["kmeans_ari"] = {}
for k in (2, 3, 4):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    OUT["kmeans_ari"][k] = dict(ari2=float(adjusted_rand_score(y2, lab)), ari3=float(adjusted_rand_score(y3, lab)))

rows = []
for W in (2, 3, 5, 8, 12):
    n = int(W / dt); xs = x[:, :n + 1]; vv = np.maximum(np.gradient(xs, dt, axis=1), 1e-9); i05 = int(0.5 / dt)
    sl = pd.DataFrame({"v_tb": xs[:, n] / W})
    sh = pd.DataFrame({"gia_toc": np.clip((vv[:, n] - vv[:, i05]) / vv[:, i05], -1, 50),
                       "dlnv": np.clip(np.log(vv[:, n] / vv[:, i05]) / (W - 0.5), -20, 20),
                       "cong": (xs[:, n] - 2 * xs[:, n // 2] + xs[:, 0]) / (W / 2) ** 2})
    for name, Fm in [("Độ dốc", sl), ("Hình dạng", sh), ("Độ dốc + hình dạng", pd.concat([sl, sh], axis=1))]:
        s = cross_validate(RandomForestClassifier(300, random_state=0, n_jobs=-1), Fm, y2, cv=cv, scoring=["accuracy", "balanced_accuracy"])
        rows.append(dict(cua_so_nam=W, bo=name, acc=s["test_accuracy"].mean(), bal=s["test_balanced_accuracy"].mean()))
early = pd.DataFrame(rows); early.to_csv(DATA / "ket_qua_cua_so_som.csv", index=False)
print(early.pivot(index="cua_so_nam", columns="bo", values="acc").round(3))
OUT["majority"] = float(max(y2.mean(), 1 - y2.mean()))

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
for bo, c, mk in [("Độ dốc", "tab:red", "o"), ("Hình dạng", "tab:green", "s"), ("Độ dốc + hình dạng", "tab:blue", "^")]:
    d = early[early.bo == bo]; ax[0].plot(d.cua_so_nam, d.acc, c=c, marker=mk, ms=4, lw=1.2, label=bo)
ax[0].axhline(OUT["majority"], ls="--", c="gray", lw=0.9); ax[0].text(2.1, OUT["majority"] + 0.01, "Đoán lớp đa số", color="gray", fontsize=6.5)
ax[0].set(xlabel="Độ dài cửa sổ quan sát sớm (năm)", ylabel="Độ chính xác (CV 5 lớp)", title="(a) Chẩn đoán chế độ từ cửa sổ sớm", ylim=(0.5, 1.0))
ax[0].legend(fontsize=6.5, loc="lower right"); ax[0].grid(alpha=.3)
ax[1].scatter(df.m, F.v_dau, c=np.where(y2 == 1, "tab:red", "tab:blue"), s=3, alpha=.45)
ax[1].set(xscale="log", yscale="log", xlabel="Hệ số nỗ lực nghiên cứu $m_r$", ylabel="Độ dốc trung bình 2 năm đầu",
          title="(b) Độ dốc sớm phản ánh $m_r$, không phải $\\mathcal{R}_{AI}$")
ax[1].scatter([], [], c="tab:blue", s=10, label="Cận tới hạn"); ax[1].scatter([], [], c="tab:red", s=10, label="Siêu tới hạn")
ax[1].legend(fontsize=6.5); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIGS / "fig_khai_pha_vi.pdf"); plt.savefig(FIGS / "fig_khai_pha_vi.png", dpi=170); plt.close()

OUT["tham_so_tom_tat"] = {c: [float(df[c].min()), float(df[c].max())] for c in ["a", "beta", "k", "tau", "m"]}
json.dump(OUT, open(DATA / "tom_tat.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
