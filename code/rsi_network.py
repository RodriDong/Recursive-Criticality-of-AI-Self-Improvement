"""Mô hình mạng nhiều tác nhân (Mục 4.1), cài đặt lại theo notebook của tác giả.

  x_i' = m_r * r * (1 - x_i)^beta * exp( phi_i ),   phi = B @ P(x(t - tau))
  B = sigma* K / g*,  P(x) = (a/k) ln((1+e^{kx})/2)   (a=3: kịch bản siêu tới hạn yếu)
"""
import numpy as np

A, BETA, KSTEEP, X = 3.0, 2.0, 10.0, 1.0
R_REF, X0, X_AGI, X_ASI = 1/24, 0.0, 0.5, 0.8
SIGMA_STAR = BETA / (X - X0)          # = 2
G_STAR = A * 0.5                      # g(x*) = a * chi(0) = 1.5
V_STAR = R_REF                        # x'(0) = r f(0) exp(0)

CONFIGS = {
 "Mạng phòng lab đóng": dict(K=[[1.0,.02,.02],[.02,.9,.02],[.02,.02,.7]], m=1.0, tau=0.30),
 "Hệ sinh thái mở":      dict(K=[[.65,.5,.5],[.5,.55,.5],[.5,.5,.45]],   m=0.8, tau=0.50),
 "Cạnh tranh toàn cầu":  dict(K=[[.9,.03,.3],[.04,.85,.08],[.4,.1,.62]], m=1.2, tau=1.00),
}

def P(x):
    return (A / KSTEEP) * np.log((np.exp(KSTEEP * x) + 1.0) / 2.0)

def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))

def simulate_network(K, m, tau, horizon=30.0, dt=0.005):
    K = np.asarray(K, float); n = K.shape[0]
    B = SIGMA_STAR * K / G_STAR
    steps = int(np.ceil(horizon / dt))
    t = np.linspace(0, steps * dt, steps + 1)
    x = np.zeros((steps + 1, n)); x[0] = X0
    def delayed(q, cur):
        if q <= 0: return np.full(n, X0)
        s = q / dt; j = int(np.floor(s)); fr = s - j
        if j >= cur: return x[cur].copy()
        return (1 - fr) * x[j] + fr * x[min(j + 1, cur)]
    def rhs(time, state, cur):
        phi = B @ P(delayed(time - tau, cur))
        return m * R_REF * np.maximum(1 - state / X, 0) ** BETA * np.exp(np.clip(phi, -50, 50))
    cap = X * (1 - 1e-10)
    for i in range(steps):
        h = dt; xi = x[i].copy(); ti = t[i]; tm = ti + h / 2; tn = ti + h
        k1 = rhs(ti, xi, i)
        k2 = rhs(tm, np.minimum(xi + h/2*k1, cap), i)
        k3 = rhs(tm, np.minimum(xi + h/2*k2, cap), i)
        k4 = rhs(tn, np.minimum(xi + h*k3, cap), i)
        x[i+1] = np.clip(xi + h*(k1 + 2*k2 + 2*k3 + k4)/6, 0, cap)
    return t, x

def crossing(t, y, thr):
    idx = np.flatnonzero(y >= thr)
    if len(idx) == 0: return np.nan
    i = idx[0]
    return t[0] if i == 0 else t[i-1] + (thr - y[i-1])*(t[i]-t[i-1])/(y[i]-y[i-1])

if __name__ == "__main__":
    paper = {"Mạng phòng lab đóng": (10.40, 16.53, 6.13),
             "Hệ sinh thái mở": (9.75, 12.58, 2.82),
             "Cạnh tranh toàn cầu": (8.97, 13.42, 4.45)}
    for name, c in CONFIGS.items():
        K = np.array(c["K"]); t, x = simulate_network(**c)
        ta = min(crossing(t, x[:, i], X_AGI) for i in range(3))
        ts = min(crossing(t, x[:, i], X_ASI) for i in range(3))
        p = paper[name]
        print(f"{name:22s} rho(K)={spectral_radius(K):.3f} maxKii={K.diagonal().max():.2f} | "
              f"T_AGI={ta:5.2f} ({p[0]:5.2f}) T_ASI={ts:5.2f} ({p[1]:5.2f}) dT={ts-ta:5.2f} ({p[2]:4.2f})")


def reproduction_numbers(K, x):
    """R_net(t) và R_AI cục bộ của từng tác nhân tại trạng thái x (vector n tác nhân), theo notebook của tác giả."""
    K = np.asarray(K, float)
    B = SIGMA_STAR * K / G_STAR
    gains = A / (1.0 + np.exp(-KSTEEP * np.asarray(x)))            # g(x_j) = a * chi(x_j)
    sigma_i = BETA / np.maximum(X - np.asarray(x), 1e-12)         # sigma_i(x_i)
    M = (B * gains[None, :]) / sigma_i[:, None]
    return np.diag(M).copy(), spectral_radius(M)
