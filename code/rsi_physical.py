"""Giới hạn hạ tầng vật lý (Mục 4.2 của bài báo, Eq. 15-18, Fig. 5).

Cài đặt lại theo notebook của tác giả nhưng vector hóa bằng NumPy (không cần numba):
tích phân đồng thời nhiều cặp (F_AGI, g_phys) bằng Runge-Kutta bậc 4 theo phương pháp bước
(method of steps), nội suy tuyến tính cho giá trị trễ.

Mô hình (kịch bản siêu tới hạn yếu: a = 3, beta = 2, k = 10, tau = 0.5, r = 1/24):
    x'(t) = r (1 - x)^beta exp( Phi(x(t - tau)) ),   Phi(x) = (a/k) ln[(1 + e^{kx}) / 2]
Ràng buộc triển khai   C_req[x(t)] <= C_phys(t)   tương đương trần năng lực
    x(t) <= x0 + (L0 + g_phys * t) / kappa,
    kappa = ln(Q) / (x_ASI - x_AGI),
    L0    = ln(F_AGI) + kappa (x_AGI - x0) - g_phys * T_AGI_mem      (T_AGI_mem: mốc AGI không ràng buộc)
Trần được áp dụng ngay trong từng bước RK4, nên quỹ đạo bị chặn cũng đi vào phản hồi có trễ.
"""
import numpy as np

# Tham số kịch bản siêu tới hạn yếu (Mục 3.2-3.3 của bài báo)
A, K_STEEP, BETA, X_MAX = 3.0, 10.0, 2.0, 1.0
R_REF, TAU, X0 = 1.0 / 24.0, 0.5, 0.0
X_AGI, X_ASI = 0.50, 0.80
Q_DEFAULT, G_PHYS_DEFAULT = 100.0, 0.25            # Q = C_req(ASI) / C_req(AGI); tốc độ tăng tính toán


def _potential(x):
    return (A / K_STEEP) * (np.logaddexp(0.0, K_STEEP * x) - np.log(2.0))


def _rhs(x, xd):
    return R_REF * np.clip(1.0 - x / X_MAX, 0.0, None) ** BETA * np.exp(np.clip(_potential(xd), -50, 50))


def _delayed(xs, i, off, dt):
    """x(t_i + off*dt - tau) bằng nội suy tuyến tính trên lịch sử xs[:, 0..i] (vector theo cột 0)."""
    q = (i + off) - TAU / dt                         # chỉ số (thực) của thời điểm trễ
    if q <= 0.0:
        return np.full(xs.shape[0], X0)
    j = int(np.floor(q)); fr = q - j
    if j >= i:
        return xs[:, i]
    return (1.0 - fr) * xs[:, j] + fr * xs[:, min(j + 1, i)]


def integrate(F=None, g_phys=None, Q=Q_DEFAULT, t_agi_free=None, dt=0.02, horizon=140.0):
    """Tích phân đồng thời nhiều trường hợp.

    F, g_phys : mảng cùng kích thước (độ dư vật lý tại AGI, tốc độ tăng tính toán hiệu dụng);
                None  ->  một quỹ đạo không ràng buộc (phần mềm thuần).
    Trả về dict: t, x (n_case, n_step+1), cap (trần hoặc None), T_AGI, T_ASI, T_bind, feasible.
    """
    n = int(np.ceil(horizon / dt)); t = np.arange(n + 1) * dt
    constrained = F is not None
    if constrained:
        F = np.atleast_1d(np.asarray(F, float)); g = np.atleast_1d(np.asarray(g_phys, float))
        F, g = np.broadcast_arrays(F, g)
        kappa = np.log(Q) / (X_ASI - X_AGI)
        L0 = np.log(F) + kappa * (X_AGI - X0) - g * t_agi_free
        feasible = L0 >= 0.0                          # nếu L0 < 0 thì ngay t = 0 trần đã thấp hơn x0
        ncase = F.size
        cap_fn = lambda tt: np.minimum(X0 + (L0 + g * tt) / kappa, X_MAX * (1 - 1e-10))
    else:
        ncase, feasible = 1, np.ones(1, bool)
        cap_fn = lambda tt: np.full(1, X_MAX * (1 - 1e-10))
    x = np.empty((ncase, n + 1)); x[:, 0] = X0
    t_bind = np.full(ncase, np.nan)
    if constrained:
        t_bind[np.abs(x[:, 0] - cap_fn(0.0)) <= 1e-8] = 0.0
    cmax = X_MAX * (1 - 1e-10)
    for i in range(n):
        ti, tm, tn = t[i], t[i] + dt / 2, t[i + 1]
        xi = x[:, i]
        xd0, xdm, xd1 = _delayed(x, i, 0.0, dt), _delayed(x, i, 0.5, dt), _delayed(x, i, 1.0, dt)
        k1 = _rhs(xi, xd0)
        k2 = _rhs(np.minimum(xi + dt * k1 / 2, cap_fn(tm)), xdm)
        k3 = _rhs(np.minimum(xi + dt * k2 / 2, cap_fn(tm)), xdm)
        k4 = _rhs(np.minimum(xi + dt * k3, cap_fn(tn)), xd1)
        cand = np.clip(xi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0.0, cmax)
        cap = cap_fn(tn)
        x[:, i + 1] = np.minimum(cand, cap)
        if constrained:
            newly = np.isnan(t_bind) & (np.abs(x[:, i + 1] - cap) <= 1e-8)
            t_bind[newly] = tn
    T_AGI, T_ASI = crossing(t, x, X_AGI), crossing(t, x, X_ASI)
    cap_full = cap_fn(t[None, :].T).T if constrained else None
    return dict(t=t, x=x, cap=cap_full, T_AGI=T_AGI, T_ASI=T_ASI, T_bind=t_bind, feasible=feasible, L0=L0 if constrained else None)


def crossing(t, x, thr):
    """Thời điểm đầu tiên x >= thr (nội suy tuyến tính) cho từng hàng của x; nan nếu không đạt."""
    ok = x >= thr; has = ok.any(axis=1); idx = ok.argmax(axis=1)
    out = np.full(x.shape[0], np.nan)
    r = np.arange(x.shape[0])
    i = np.where(has, idx, 1)
    x1, x0_ = x[r, i], x[r, i - 1]
    tt = t[i - 1] + (thr - x0_) * (t[i] - t[i - 1]) / np.where(x1 == x0_, 1, x1 - x0_)
    out[has] = np.where(idx[has] == 0, t[0], tt[has])
    return out


def regime(T_bind, T_AGI, T_ASI):
    """0: không bị chặn trước ASI; 1: chặn giữa AGI và ASI; 2: chặn trước AGI (khớp notebook)."""
    reg = np.zeros(T_bind.shape)
    reg[~np.isnan(T_bind) & ~np.isnan(T_ASI) & (T_bind < T_ASI)] = 1.0
    reg[~np.isnan(T_bind) & ~np.isnan(T_AGI) & (T_bind < T_AGI)] = 2.0
    return reg


if __name__ == "__main__":
    free = integrate(dt=0.005, horizon=60.0)
    ta, ts = float(free["T_AGI"][0]), float(free["T_ASI"][0])
    print(f"Không ràng buộc: T_AGI={ta:.2f}, T_ASI={ts:.2f}, dT={ts - ta:.2f}  (Table 1: 12.90, 24.73, 11.83)")
    Fs = [8.0, 2.0, 0.75]
    res = integrate(Fs, [G_PHYS_DEFAULT] * 3, t_agi_free=ta, dt=0.005, horizon=90.0)
    for j, F in enumerate(Fs):
        print(f"F_AGI={F:5.2f}: T_bind={res['T_bind'][j]:.2f}  T_AGI={res['T_AGI'][j]:.2f}  T_ASI={res['T_ASI'][j]:.2f}  "
              f"dT={res['T_ASI'][j] - res['T_AGI'][j]:.2f}  chế độ={int(regime(res['T_bind'], res['T_AGI'], res['T_ASI'])[j])}")
