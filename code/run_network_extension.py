
import numpy as np
import pandas as pd
import pathlib

from rsi_network import (
    simulate_network,
    crossing,
    spectral_radius,
    reproduction_numbers,
    X_AGI,
    X_ASI
)

# ==========================================
# 1. THIET LAP THI NGHIEM
# ==========================================

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SIZES = [3, 5, 10]
RHO_LEVELS = [0.5, 0.8]

REPEATS = 20

M = 1.0
TAU = 0.5
HORIZON = 50.0
DT = 0.01

rng = np.random.default_rng(42)


# ==========================================
# 2. TAO MA TRAN K NGAU NHIEN
# ==========================================


def generate_K(n, target_rho):
    """
    Tao ma tran K khong am:
    - Kii < 1: moi tac nhan khong tu khu ech dai qua nguong
    - Cac phan tu ngoai duong cheo mo ta tuong tac giua tac nhan
    - Dieu chinh de rho(K) = target_rho
    """

    # Duong cheo nho hon 1
    diagonal = rng.uniform(0.05, 0.25, size=n)
    K = np.diag(diagonal)

    # Cac lien ket giua cac tac nhan
    off_diag = rng.uniform(0.05, 1.0, size=(n, n))
    np.fill_diagonal(off_diag, 0.0)

    # Tim he so scale de dat ban kinh pho muc tieu
    low = 0.0
    high = 1.0

    while spectral_radius(K + high * off_diag) < target_rho:
        high *= 2

    for _ in range(80):
        mid = (low + high) / 2
        candidate = K + mid * off_diag

        if spectral_radius(candidate) < target_rho:
            low = mid
        else:
            high = mid

    K = K + ((low + high) / 2) * off_diag

    return K

# ==========================================
# 3. MO PHONG MOT MANG
# ==========================================

def run_one_network(K, n, rho_target, seed_id):

    t, x = simulate_network(
        K=K,
        m=M,
        tau=TAU,
        horizon=HORIZON,
        dt=DT
    )

    # Thoi diem moi co it nhat mot tac nhan dat nguong
    t_agi_each = [
        crossing(t, x[:, i], X_AGI)
        for i in range(n)
    ]

    t_asi_each = [
        crossing(t, x[:, i], X_ASI)
        for i in range(n)
    ]

    valid_agi = [v for v in t_agi_each if np.isfinite(v)]
    valid_asi = [v for v in t_asi_each if np.isfinite(v)]

    t_agi = min(valid_agi) if valid_agi else np.nan
    t_asi = min(valid_asi) if valid_asi else np.nan

    delta_t = (
        t_asi - t_agi
        if np.isfinite(t_agi) and np.isfinite(t_asi)
        else np.nan
    )

    # Thoi diem tat ca tac nhan dat nguong
    t_agi_all = (
        max(t_agi_each)
        if all(np.isfinite(t_agi_each))
        else np.nan
    )

    t_asi_all = (
        max(t_asi_each)
        if all(np.isfinite(t_asi_each))
        else np.nan
    )

    # Tinh ban kinh pho cua ma tran sinh san mang
    # Lay mau moi 0.1 nam de giam chi phi tinh toan
    rho_m_values = []

    for j in range(0, len(t), 10):
        _, rho_m = reproduction_numbers(K, x[j])
        rho_m_values.append(rho_m)

    rho_m_max = max(rho_m_values)

    return {
        "n_agents": n,
        "rho_target": rho_target,
        "rho_K": spectral_radius(K),
        "max_Kii": np.max(np.diag(K)),
        "seed_id": seed_id,
        "T_AGI_any": t_agi,
        "T_ASI_any": t_asi,
        "delta_T_any": delta_t,
        "T_AGI_all": t_agi_all,
        "T_ASI_all": t_asi_all,
        "rho_M_max": rho_m_max
    }


# ==========================================
# 4. CHAY TOAN BO THI NGHIEM
# ==========================================

results = []

total = len(SIZES) * len(RHO_LEVELS) * REPEATS
count = 0

for n in SIZES:

    for target_rho in RHO_LEVELS:

        for rep in range(REPEATS):

            count += 1

            K = generate_K(n, target_rho)

            result = run_one_network(
                K,
                n,
                target_rho,
                rep
            )

            results.append(result)

            print(
                f"[{count}/{total}] "
                f"n={n}, rho={target_rho}, "
                f"rho_M_max={result['rho_M_max']:.3f}, "
                f"delta_T={result['delta_T_any']:.3f}"
            )


# ==========================================
# 5. LUU KET QUA
# ==========================================

df = pd.DataFrame(results)

df.to_csv(
    DATA / "network_extension_raw.csv",
    index=False
)

# Tong hop theo so tac nhan va muc rho
summary = (
    df.groupby(["n_agents", "rho_target"])
    .agg(
        rho_K_mean=("rho_K", "mean"),
        rho_M_max_mean=("rho_M_max", "mean"),
        T_AGI_mean=("T_AGI_any", "mean"),
        T_ASI_mean=("T_ASI_any", "mean"),
        delta_T_mean=("delta_T_any", "mean"),
        delta_T_std=("delta_T_any", "std"),
        T_AGI_all_mean=("T_AGI_all", "mean"),
        T_ASI_all_mean=("T_ASI_all", "mean"),
        success_rate=("delta_T_any", lambda s: s.notna().mean())
    )
    .reset_index()
)

summary.to_csv(
    DATA / "network_extension_summary.csv",
    index=False
)

print("\n===== KET QUA TONG HOP =====")
print(summary.round(3).to_string(index=False))

print("\nDa luu ket qua tai:", DATA)
