"""Mô phỏng mô hình đệ quy có trễ (Burtsev, arXiv:2609.00137), Eq. (3), (9), (14).

  x'(t) = r * f(x) * exp( Phi( x(t - tau) ) )
  f(x)  = (1 - x/X)^beta
  chi(x) = 1 / (1 + exp(-k x)),   dPhi/dx = a * chi(x),  Phi(0) = 0
  =>  Phi(x) = (a/k) * [ ln(1 + e^{kx}) - ln 2 ]
  R_AI(x) = a * chi(x) * (1 - x/X) / beta     (X = 1: a*chi*(1-x)/beta)
"""
import numpy as np

# Tham số tham chiếu (Mục 3.1-3.2 của bài báo)
X, BETA, K, TAU, R_REF = 1.0, 2.0, 10.0, 0.5, 1.0 / 24.0
X_AGI, X_ASI = 0.50, 0.80


def chi(x, k=K):
    return 1.0 / (1.0 + np.exp(-k * x))


def phi(x, a, k=K):
    # (a/k) * [ln(1+e^{kx}) - ln 2], dùng logaddexp để ổn định số
    return (a / k) * (np.logaddexp(0.0, k * x) - np.log(2.0))


def r_ai(x, a, beta=BETA, k=K, X=X):
    """Số tái sản xuất đệ quy cục bộ, Eq. (8) với sigma = beta/(X-x)."""
    return a * chi(x, k) * (1.0 - x / X) / beta


def simulate(a, r=R_REF, beta=BETA, k=K, tau=TAU, T=30.0, dt=0.0005, X=X):
    """Tích phân bằng phương pháp Heun (bậc 2). Lịch sử trước t=0: x = 0."""
    n = int(round(T / dt))
    d = max(int(round(tau / dt)), 1)
    x = np.zeros(n + 1)
    f = lambda xx: np.clip(1.0 - xx / X, 0.0, None) ** beta
    for i in range(n):
        xd_i = x[i - d] if i - d >= 0 else 0.0
        xd_j = x[i + 1 - d] if i + 1 - d >= 0 else 0.0
        k1 = r * f(x[i]) * np.exp(phi(xd_i, a, k))
        xp = x[i] + dt * k1
        k2 = r * f(xp) * np.exp(phi(xd_j, a, k))
        x[i + 1] = x[i] + 0.5 * dt * (k1 + k2)
    t = np.linspace(0.0, T, n + 1)
    return t, x


def first_crossing(t, x, y):
    """Thời điểm đầu tiên x >= y (nội suy tuyến tính); nan nếu không đạt."""
    idx = np.argmax(x >= y)
    if x[idx] < y:
        return np.nan
    if idx == 0:
        return t[0]
    t0, t1, x0, x1 = t[idx - 1], t[idx], x[idx - 1], x[idx]
    return t0 + (y - x0) * (t1 - t0) / (x1 - x0)


def baseline_no_rsi(r=R_REF, beta=2.0):
    """Nghiệm giải tích khi beta=2, X=1: T(x) = x / (r (1-x))."""
    T = lambda y: y / (r * (1.0 - y))
    return T(X_AGI), T(X_ASI)


if __name__ == "__main__":
    print("No-RSI (giải tích): T_AGI, T_ASI =", baseline_no_rsi())
    # Giá trị Table 1 của bài báo, để đối chiếu
    paper = {0.5: (21.35, 74.13), 3: (12.90, 24.73), 6: (8.40, 11.08), 15: (4.50, 4.95)}
    for a, (pa, ps) in paper.items():
        T = 30.0 if a >= 3 else 100.0
        t, x = simulate(a, T=T)
        ta, ts = first_crossing(t, x, X_AGI), first_crossing(t, x, X_ASI)
        print(f"a={a:>4}: T_AGI={ta:6.2f} (bài báo {pa:5.2f}) | "
              f"T_ASI={ts:6.2f} (bài báo {ps:5.2f}) | dT={ts - ta:6.2f}")
