"""Sinh tập quỹ đạo tổng hợp từ mô hình (Eq. 3, 9, 14), vector hóa theo lô.
Mỗi mẫu = một bộ tham số (a, beta, k, tau, m) -> một quỹ đạo x(t) trên [0, T]."""
import numpy as np, pandas as pd
R_REF = 1/24

def simulate_batch(a, beta, k, tau, m, T=60.0, dt=0.01):
    N = len(a); n = int(round(T/dt)); d = np.maximum(np.round(tau/dt).astype(int), 1)
    x = np.zeros((N, n+1)); ar = np.arange(N); r = m*R_REF
    Phi = lambda xx: (a/k)*(np.logaddexp(0, k*xx) - np.log(2))
    f = lambda xx: np.clip(1-xx, 0, None)**beta
    for i in range(n):
        ji = i-d; jj = i+1-d
        xi_ = np.where(ji >= 0, x[ar, np.clip(ji, 0, None)], 0.0)
        xj_ = np.where(jj >= 0, x[ar, np.clip(jj, 0, None)], 0.0)
        k1 = r*f(x[:, i])*np.exp(Phi(xi_))
        xp = x[:, i] + dt*k1
        k2 = r*f(xp)*np.exp(Phi(xj_))
        x[:, i+1] = np.minimum(x[:, i] + 0.5*dt*(k1+k2), 1.0)
    return np.linspace(0, T, n+1), x

def r_peak(a, beta, k):
    xs = np.linspace(0, 0.999, 2000)[None, :]
    R = a[:, None]/(1+np.exp(-k[:, None]*xs))*(1-xs)/beta[:, None]
    return R.max(axis=1)

def make_dataset(N=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = dict(a=np.exp(rng.uniform(np.log(0.1), np.log(20), N)),
             beta=rng.uniform(0.5, 4.0, N), k=rng.uniform(2, 20, N),
             tau=rng.uniform(0.1, 2.0, N), m=np.exp(rng.uniform(np.log(0.5), np.log(2.0), N)))
    t, x = simulate_batch(**p)
    df = pd.DataFrame(p); df["R_peak"] = r_peak(p["a"], p["beta"], p["k"])
    return t, x, df
