"""Sinh các bảng LaTeX từ ../data/robust_tom_tat_*.csv:  python tao_bang_tex.py
Kết quả ghi vào ../bang/*.tex (được main.tex nạp bằng \\input), dạng  "trung bình ± độ lệch chuẩn giữa các seed" (đơn vị %).
"""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "bang"; OUT.mkdir(exist_ok=True)
f = lambda v, nd=1: f"{v:.{nd}f}".replace(".", ",")
cell = lambda m, s: f"{f(100 * m)} $\\pm$ {f(100 * s)}"


def write(name, header, rows, spec):
    body = " \\\\\n".join(" & ".join(r) for r in rows) + " \\\\\n"
    (OUT / name).write_text(f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\\bottomrule\n\\end{{tabular}}\n", encoding="utf-8")


# (a) toàn bộ quỹ đạo
d = pd.read_csv(DATA / "robust_tom_tat_toan_quy_dao.csv")
rows = []
for bo in d.bo.unique():
    r2, r3 = d[(d.bo == bo) & (d.bai_toan == "2 lớp")].iloc[0], d[(d.bo == bo) & (d.bai_toan == "3 lớp")].iloc[0]
    rows.append([bo, cell(r2.acc_tb, r2.acc_std_seed), f(100 * r2.bal_tb), cell(r3.acc_tb, r3.acc_std_seed), f(100 * r3.bal_tb)])
write("bang_toan_quy_dao.tex", "\\textbf{Bộ đặc trưng} & \\textbf{2 lớp: độ chính xác} & \\textbf{cân bằng} & \\textbf{3 lớp: độ chính xác} & \\textbf{cân bằng}", rows, "lcccc")

# (b) cửa sổ quan sát sớm
d = pd.read_csv(DATA / "robust_tom_tat_cua_so_som.csv"); rows = []
for W in sorted(d.W.unique()):
    row = [str(W)]
    for bo in ("Độ dốc", "Hình dạng", "Độ dốc + hình dạng"):
        r = d[(d.W == W) & (d.bo == bo)].iloc[0]; row.append(cell(r.acc_tb, r.acc_std_seed))
    rows.append(row)
write("bang_cua_so_som.tex", "\\textbf{Cửa sổ $W$ (năm)} & \\textbf{Độ dốc} & \\textbf{Hình dạng} & \\textbf{Độ dốc + hình dạng}", rows, "cccc")

# (c) nhiễu quan sát
d = pd.read_csv(DATA / "robust_tom_tat_nhieu.csv"); rows = []
for sg in sorted(d.sigma.unique()):
    for k, W in enumerate(sorted(d.W.unique())):
        row = [(f(sg, 3) if k == 0 else ""), str(W)]
        for bo in ("Độ dốc", "Hình dạng", "Độ dốc + hình dạng"):
            r = d[(d.sigma == sg) & (d.W == W) & (d.bo == bo)].iloc[0]; row.append(cell(r.acc_tb, r.acc_std_seed))
        rows.append(row)
write("bang_nhieu.tex", "$\\sigma$ & \\textbf{$W$ (năm)} & \\textbf{Độ dốc} & \\textbf{Hình dạng} & \\textbf{Độ dốc + hình dạng}", rows, "ccccc")
print("Đã ghi 3 bảng vào", OUT)
