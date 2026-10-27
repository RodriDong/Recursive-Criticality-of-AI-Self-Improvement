"""Tái hiện các hình còn thiếu của bài báo:  python run_fig_con_thieu.py

  Fig. 1  : sơ đồ vòng tự cải tiến tối giản            -> figures/fig1_vi.(pdf|png)
  Fig. 2  : bản đồ pha R_peak theo (beta, a) và (beta, k) -> figures/fig2_vi.(pdf|png)
  Fig. 4a : ba mạng chiến lược (đồ thị có hướng)         -> figures/fig4a_vi.(pdf|png)
  Fig. 5  : độ dư hạ tầng vật lý (quỹ đạo + lưới 61x61)  -> figures/fig5_vi.(pdf|png)

Dữ liệu số kèm theo ghi vào ../data (fig5_*.csv, fig5_tom_tat.json, fig2_tom_tat.json).
Tùy chọn:  python run_fig_con_thieu.py --nhanh   (lưới Fig. 5 thô 21x21, chạy dưới 1 phút)
Thời gian chạy đầy đủ: khoảng vài phút trên 1 CPU.
"""
import argparse, json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

from rsi_network import CONFIGS, spectral_radius
import rsi_physical as ph

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, FIGS = ROOT / "data", ROOT / "figures"
DATA.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
ap = argparse.ArgumentParser(); ap.add_argument("--nhanh", action="store_true"); ARGS = ap.parse_args()


def save(name):
    plt.savefig(FIGS / f"{name}.pdf", bbox_inches="tight"); plt.savefig(FIGS / f"{name}.png", dpi=170, bbox_inches="tight"); plt.close()


# ------------------------------------------------------------------ Fig. 1: sơ đồ vòng đệ quy
def fig1():
    fig, ax = plt.subplots(figsize=(6.8, 2.7)); ax.set_xlim(0, 12.6); ax.set_ylim(-0.3, 5.0); ax.axis("off")
    W, H, Y = 2.1, 1.3, 1.6
    boxes = [(0.1, "Năng lực AI\n$x(t)$", "#dbe9f6"), (2.6, "Độ lợi đệ quy\n$a$", "#e8f5e0"),
             (5.1, "Năng suất\nR&D", "#e8f5e0"), (7.6, "Đóng vận hành $\\chi$\nvà trễ $\\tau$", "#fde9d0"),
             (10.1, "Hệ kế nhiệm\nđã kiểm chứng", "#dbe9f6")]
    for x0, s_, c in boxes:
        ax.add_patch(FancyBboxPatch((x0, Y), W, H, boxstyle="round,pad=0.05", fc=c, ec="0.3", lw=1))
        ax.text(x0 + W / 2, Y + H / 2, s_, ha="center", va="center", fontsize=7)
    arr = lambda p, q, c="0.2", cs="arc3": ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.1, color=c, connectionstyle=cs))
    for x0 in (0.1, 2.6, 5.1, 7.6):
        arr((x0 + W + 0.07, Y + H / 2), (x0 + 2.5 - 0.07, Y + H / 2))
    arr((11.15, Y + H + 0.1), (1.15, Y + H + 0.1), c="tab:green", cs="arc3,rad=0.22")
    ax.text(6.1, 4.75, "Phản hồi: năng lực mới làm tăng năng suất nghiên cứu", ha="center", fontsize=7.5, color="tab:green")
    ax.add_patch(FancyBboxPatch((3.7, -0.15), 4.8, 0.75, boxstyle="round,pad=0.05", fc="#f6dcdc", ec="0.3", lw=1))
    ax.text(6.1, 0.22, "Độ cứng hóa biên $\\sigma$ (gần biên thì khó hơn)", ha="center", va="center", fontsize=6.8)
    arr((6.15, 0.68), (6.15, Y - 0.03), c="tab:red")
    ax.text(6.3, 1.12, "làm giảm", fontsize=6.5, color="tab:red", va="center")
    ax.set_title("Vòng tự cải tiến đệ quy tối giản (vẽ lại theo Fig. 1 của bài báo)", fontsize=8.5)
    save("fig1_vi")


# ------------------------------------------------------------------ Fig. 2: bản đồ pha
def fig2():
    def rpeak(gain_rows, beta, x):
        return np.max(gain_rows * (1 - x[None, :]), axis=1)[:, None] / beta[None, :]
    beta, x = np.linspace(.1, 4, 200), np.linspace(0, 1, 1400)
    sig = lambda k: 1.0 / (1.0 + np.exp(-np.asarray(k)[:, None] * x[None, :]))
    a_vals, k_vals = np.linspace(0, 20, 200), np.linspace(0, 20, 200)
    Ra = rpeak(a_vals[:, None] * sig(np.full(200, 10.0)), beta, x)             # (a, beta), k = 10
    Rk = rpeak(3.0 * sig(k_vals), beta, x)                                     # (k, beta), a = 3
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    def panel(ax, R, Y, ylabel, title):
        L = np.log10(np.maximum(R, 1e-9)); B, YY = np.meshgrid(beta, Y)
        cf = ax.contourf(B, YY, L, levels=np.linspace(-1, 1, 41), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), extend="both")
        vals = [0.2, 0.4, 0.6, 0.8, 2, 3, 4, 5, 6]
        cs = ax.contour(B, YY, L, levels=np.log10(vals), colors="0.25", linewidths=.6)
        ax.clabel(cs, fmt={np.log10(v): f"{v:g}" for v in vals}, fontsize=6)
        cr = ax.contour(B, YY, L, levels=[0], colors="black", linewidths=1.6, linestyles="--")
        ax.clabel(cr, fmt={0: r"$\mathcal{R}_{peak}=1$"}, fontsize=7)
        ax.axvline(2, c="k", ls=":", lw=.8); ax.set_xlim(.1, 4)
        ax.set(xlabel=r"mũ cứng hóa biên $\beta$", ylabel=ylabel, title=title); ax.set_yticks(np.arange(0, 21, 4))
        return cf
    cf = panel(axs[0], Ra, a_vals, r"độ lợi đệ quy $a$", r"(a) $k=10$")
    panel(axs[1], Rk, k_vals, r"độ dốc đóng vận hành $k$", r"(b) $a=3$")
    for a_ in (0.5, 3, 6, 15):
        axs[0].axhline(a_, c="k", ls=":", lw=.8); axs[0].plot(2, a_, "d", ms=5, mfc="none", c="k")
    axs[1].axhline(10, c="k", ls=":", lw=.8); axs[1].plot(2, 10, "d", ms=5, mfc="none", c="k")
    cb = fig.colorbar(cf, ax=axs, pad=.02, shrink=.92); cb.set_label(r"$\log_{10}\mathcal{R}_{peak}$")
    save("fig2_vi")
    # kiểm tra nhanh: R_peak tại bốn kịch bản tham chiếu (a = .5, 3, 6, 15; beta = 2; k = 10)
    xs = np.linspace(0, 1, 100001)
    ref = {a: float(np.max(a / (1 + np.exp(-10 * xs)) * (1 - xs) / 2.0)) for a in (0.5, 3, 6, 15)}
    json.dump({"R_peak_bon_kich_ban": ref}, open(DATA / "fig2_tom_tat.json", "w"), indent=1)
    print("R_peak của 4 kịch bản:", {a: round(v, 3) for a, v in ref.items()})


# ------------------------------------------------------------------ Fig. 4a: ba mạng chiến lược
def fig4a():
    names = {"Mạng phòng lab đóng": ("Phòng lab đóng", "tab:red", ("Lab A", "Lab B", "Lab C")),
             "Hệ sinh thái mở": ("Hệ sinh thái mở", "tab:orange", ("Tác nhân A", "Tác nhân B", "Tác nhân C")),
             "Cạnh tranh toàn cầu": ("Cạnh tranh toàn cầu", "tab:blue", ("Khối A", "Khối B", "Khối C"))}
    pos = {0: (-0.85, -0.45), 1: (0.85, -0.45), 2: (0.0, 0.80)}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.5), constrained_layout=True)
    for ax, (key, cfg) in zip(axs, CONFIGS.items()):
        title, col, nodes = names[key]; K = np.array(cfg["K"]); n = 3
        G = nx.DiGraph(); G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if i != j and K[i, j] > 0: G.add_edge(j, i, weight=K[i, j])          # chuyển giao từ j sang i
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[1250 + 650 * K[i, i] for i in range(n)], node_color="white", edgecolors=col, linewidths=2)
        nx.draw_networkx_labels(G, pos, labels={i: f"{nodes[i]}\n$K_{{ii}}={K[i, i]:.2f}$" for i in range(n)}, ax=ax, font_size=6.5)
        for j, i, d in G.edges(data=True):
            w = d["weight"]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(j, i)], width=0.6 + 6 * w, alpha=min(.95, .25 + 2.2 * w), edge_color=col, arrows=True,
                                   arrowsize=12, connectionstyle="arc3,rad=0.2", min_source_margin=16, min_target_margin=16)
            (x1, y1), (x2, y2) = pos[j], pos[i]; dx, dy = x2 - x1, y2 - y1; ln = np.hypot(dx, dy)
            ax.text((x1 + x2) / 2 - 0.2 * dy / ln, (y1 + y2) / 2 + 0.2 * dx / ln, f"{w:.2f}", fontsize=6, ha="center", va="center", color="0.25",
                    bbox=dict(fc="white", ec="none", alpha=.8, pad=.4))
        ax.set_title(title, fontsize=8.5, pad=28)
        ax.text(0.5, 1.0, rf"$\max_i K_{{ii}}={np.max(np.diag(K)):.2f},\ \rho(\mathbf{{K}})={spectral_radius(K):.2f}$" "\n"
                          rf"$m_r={cfg['m']:.1f},\ \tau={cfg['tau']:.2f}$ năm", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.0, 1.25); ax.set_aspect("equal"); ax.axis("off")
    save("fig4a_vi")


# ------------------------------------------------------------------ Fig. 5: giới hạn hạ tầng
def fig5():
    # (a) ba quỹ đạo (dt = 0.005, như notebook) và quỹ đạo phần mềm không ràng buộc
    free = ph.integrate(dt=0.005, horizon=60.0); t_agi, t_asi = float(free["T_AGI"][0]), float(free["T_ASI"][0])
    PA = [8.0, 2.0, 0.75]
    ra = ph.integrate(PA, [ph.G_PHYS_DEFAULT] * 3, t_agi_free=t_agi, dt=0.005, horizon=60.0)
    reg_a = ph.regime(ra["T_bind"], ra["T_AGI"], ra["T_ASI"])
    rows = [dict(F_AGI="không ràng buộc", T_bind=np.nan, T_AGI=t_agi, T_ASI=t_asi, dT=t_asi - t_agi, che_do="-")]
    for j, F in enumerate(PA):
        rows.append(dict(F_AGI=F, T_bind=ra["T_bind"][j], T_AGI=ra["T_AGI"][j], T_ASI=ra["T_ASI"][j], dT=ra["T_ASI"][j] - ra["T_AGI"][j],
                         che_do=["không chặn trước ASI", "chặn giữa AGI và ASI", "chặn trước AGI"][int(reg_a[j])]))
    tabA = pd.DataFrame(rows); tabA.to_csv(DATA / "fig5_bang_a.csv", index=False); print(tabA.round(2).to_string(index=False))

    # (b) lưới (F_AGI, g_phys): 61x61 như bài báo (hoặc 21x21 nếu --nhanh)
    n = 21 if ARGS.nhanh else 61
    dt_s, hz = 0.02, 140.0
    sw = ph.integrate(dt=dt_s, horizon=hz); ta_s, ts_s = float(sw["T_AGI"][0]), float(sw["T_ASI"][0]); d_soft = ts_s - ta_s
    Hg, Gg = np.geomspace(0.5, 16.0, n), np.linspace(0.15, 0.55, n)
    HH, GG = np.meshgrid(Hg, Gg)
    out = ph.integrate(HH.ravel(), GG.ravel(), t_agi_free=ta_s, dt=dt_s, horizon=hz)
    feas = out["feasible"].reshape(HH.shape)
    shape = lambda v: np.where(feas, v.reshape(HH.shape), np.nan)
    TA, TS, TB = shape(out["T_AGI"]), shape(out["T_ASI"]), shape(out["T_bind"])
    extra = TS - TA - d_soft; reg = np.where(feas, ph.regime(TB, TA, TS), np.nan)
    pd.DataFrame(dict(F_AGI=HH.ravel(), g_phys=GG.ravel(), kha_thi=feas.ravel(), T_AGI=TA.ravel(), T_ASI=TS.ravel(), T_bind=TB.ravel(),
                      tre_them=extra.ravel(), che_do=reg.ravel())).to_csv(DATA / "fig5_luoi.csv", index=False)
    summ = dict(luoi=f"{n}x{n}", T_AGI_mem=ta_s, T_ASI_mem=ts_s, dT_mem=d_soft, ti_le_khong_kha_thi=float(np.mean(~feas)),
                ti_le_che_do={str(c): float(np.nanmean(reg == c)) for c in (0, 1, 2)}, tre_them_max=float(np.nanmax(np.maximum(extra, 0))))
    json.dump(summ, open(DATA / "fig5_tom_tat.json", "w"), indent=1); print(json.dumps(summ, ensure_ascii=False, indent=1))

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
    ax = axs[0]; cols = ["tab:blue", "tab:orange", "tab:red"]
    ax.plot(free["t"], free["x"][0], "--", c="gray", lw=1.6, label="Quỹ đạo phần mềm")
    for j, (F, c) in enumerate(zip(PA, cols)):
        ax.plot(ra["t"], ra["x"][j], c=c, lw=1.4, label=fr"$F_{{AGI}}={F:g}$"); ax.plot(ra["t"], ra["cap"][j], ":", c=c, lw=1, alpha=.8)
    ax.axhline(ph.X_AGI, c="0.4", ls="--", lw=.8); ax.axhline(ph.X_ASI, c="0.4", ls=":", lw=1)
    ax.text(0.5, ph.X_AGI + .03, "AGI", c="tab:blue", fontsize=8); ax.text(0.5, ph.X_ASI + .03, "ASI", c="tab:blue", fontsize=8)
    tk = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44]
    ax.set(xlim=(0, 45), ylim=(0, 1.02), xlabel="Năm", ylabel=r"năng lực $x(t)$", title="(a) Quỹ đạo và trần hạ tầng (nét chấm)")
    ax.set_xticks(tk); ax.set_xticklabels([str(2026 + v) for v in tk], rotation=45); ax.grid(alpha=.3); ax.legend(fontsize=6.5, loc="lower right")
    ax = axs[1]; dl = np.ma.masked_invalid(np.maximum(extra, 0)); top = max(5.0, 5.0 * np.ceil(dl.max() / 5.0))
    cf = ax.contourf(HH, GG, dl, levels=np.linspace(0, top, 41), cmap="YlOrRd", norm=Normalize(0, top), extend="max")
    cl = [v for v in (2, 5, 10, 20, 30, 40, 60) if v < top]
    if cl: ax.clabel(ax.contour(HH, GG, dl, levels=cl, colors="0.25", linewidths=.6), fmt={v: f"{v:g} năm" for v in cl}, fontsize=6)
    if np.nanmin(reg) < 1 or np.nanmax(reg) > 0:
        try:
            cr = ax.contour(HH, GG, reg, levels=[0.5, 1.5], colors="black", linewidths=1.3, linestyles=["--", ":"])
            ax.clabel(cr, fmt={0.5: "chặn trước ASI", 1.5: "chặn trước AGI"}, fontsize=6)
        except Exception as e:                                                  # lưới thô có thể thiếu một đường biên
            print("Cảnh báo: không vẽ được đường biên chế độ:", e)
    for F, c in zip(PA, cols): ax.plot(F, ph.G_PHYS_DEFAULT, "d", ms=6, mfc="none", c=c, mew=1.3)
    ax.set_xscale("log"); ax.set_xticks([.5, 1, 2, 4, 8]); ax.set_xticklabels(["0,5", "1", "2", "4", "8"])
    ax.set(xlim=(Hg.min(), Hg.max()), ylim=(Gg.min(), Gg.max()), xlabel=r"độ dư vật lý tại AGI, $F_{AGI}$",
           ylabel=r"tốc độ tăng tính toán, $g_{phys}$ [năm$^{-1}$]", title="(b) Thời gian AGI→ASI kéo dài thêm"); ax.grid(alpha=.3)
    cb = fig.colorbar(cf, ax=axs, pad=.02, shrink=.92); cb.set_label(r"$\Delta T_{phys}-\Delta T_{soft}$ [năm]")
    save("fig5_vi")


if __name__ == "__main__":
    fig1(); fig2(); fig4a(); fig5()
    print("Đã ghi: fig1_vi, fig2_vi, fig4a_vi, fig5_vi vào", FIGS)
