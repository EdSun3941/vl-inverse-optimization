"""VL inverse optimization on UCI Concrete Compressive Strength (Yeh, 1998).

8 mix-design inputs -> compressive strength (MPa). Inverse: target strength ->
feasible mix recommendation. Frozen MLP surrogate + projected VL updates.
Validation: independent verifier model + k-NN dataset evidence. 20 restarts/scenario.
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnlite import MLP, Adam, train_mlp

OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{OUT}/concrete.csv")
X = df.iloc[:, :8].values.astype(np.float64)
y = df.iloc[:, 8].values.astype(np.float64).reshape(-1, 1)
cols = list(df.columns[:8])
n = len(df)
print("data", X.shape, cols)

perm = np.random.RandomState(0).permutation(n)
ntr, nva = int(0.7 * n), int(0.15 * n)
itr, iva, ite = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

mu_x, sd_x = X[itr].mean(0), X[itr].std(0) + 1e-8
mu_y, sd_y = y[itr].mean(0), y[itr].std(0) + 1e-8
Xs, ys = (X - mu_x) / sd_x, (y - mu_y) / sd_y

t0 = time.time()
sur = MLP([8, 128, 128, 128, 1], seed=1)
v_sur = train_mlp(sur, Xs[itr], ys[itr], Xs[iva], ys[iva], epochs=600, seed=1)
# independent verifier: different architecture, seed, split
pool = np.concatenate([itr, iva])
p2 = np.random.RandomState(7).permutation(pool)
cut = int(0.85 * len(p2))
ver = MLP([8, 96, 96, 96, 96, 1], seed=7)
v_ver = train_mlp(ver, Xs[p2[:cut]], ys[p2[:cut]], Xs[p2[cut:]], ys[p2[cut:]], epochs=600, seed=7)

t_sur = float(np.mean((sur.forward(Xs[ite]) - ys[ite]) ** 2))
t_ver = float(np.mean((ver.forward(Xs[ite]) - ys[ite]) ** 2))
rmse_sur = float(np.sqrt(t_sur) * sd_y[0])
rmse_ver = float(np.sqrt(t_ver) * sd_y[0])
print(f"surrogate test MSE(std) {t_sur:.4f} RMSE {rmse_sur:.2f} MPa | verifier RMSE {rmse_ver:.2f} MPa | train {time.time()-t0:.0f}s")

lo_raw, hi_raw = X[itr].min(0), X[itr].max(0)
AGE, ASH = cols.index("age"), cols.index("ash")

def project(w_std, fixed=None):
    z = w_std * sd_x + mu_x
    z = np.clip(z, lo_raw, hi_raw)
    if fixed:
        for i, v in fixed.items():
            z[i] = v
    return (z - mu_x) / sd_x

def vl_inverse(target_mpa, fixed=None, iters=800, lr=0.05, seed=0):
    rng = np.random.RandomState(seed)
    w = project(rng.randn(8) * 0.5, fixed)
    tgt = (target_mpa - mu_y[0]) / sd_y[0]
    opt = Adam([(8,)], lr=lr)
    for k in range(iters):
        if k and k % 250 == 0:
            opt.lr *= 0.2
        pred = sur.forward(w[None, :], cache=True)        # (1,1)
        dOut = 2.0 * (pred - tgt)
        dX, _, _ = sur.backward(dOut, want_param_grads=False)
        (w,) = opt.step([w], [dX[0]])
        w = project(w, fixed)
    pred = float(sur.forward(w[None, :])[0, 0] * sd_y[0] + mu_y[0])
    vchk = float(ver.forward(w[None, :])[0, 0] * sd_y[0] + mu_y[0])
    x_raw = w * sd_x + mu_x
    return x_raw, pred, vchk

def knn(x_raw, k=5):
    d = np.linalg.norm((X - x_raw) / sd_x, axis=1)
    idx = np.argsort(d)[:k]
    return float(y[idx].mean()), float(d[idx].mean())

scenarios = [
    dict(name="T30_free", target=30.0, fixed=None),
    dict(name="T50_free", target=50.0, fixed=None),
    dict(name="T70_free", target=70.0, fixed=None),
    dict(name="T50_age28", target=50.0, fixed={AGE: 28.0}),
    dict(name="T40_age28_noFlyAsh", target=40.0, fixed={AGE: 28.0, ASH: 0.0}),
]

R = 20
results = []
t1 = time.time()
for sc in scenarios:
    preds, vers = [], []
    best = None
    for r in range(R):
        x_raw, pred, vchk = vl_inverse(sc["target"], sc["fixed"], seed=100 + r)
        preds.append(pred); vers.append(vchk)
        if best is None or abs(pred - sc["target"]) < abs(best["pred"] - sc["target"]):
            ky, kd = knn(x_raw)
            best = dict(x=np.round(x_raw, 2).tolist(), pred=round(pred, 3),
                        verifier=round(vchk, 3), knn_strength=round(ky, 2), knn_dist=round(kd, 3))
    results.append(dict(scenario=sc["name"], target=sc["target"],
                        pred_mean=round(float(np.mean(preds)), 3), pred_std=round(float(np.std(preds)), 3),
                        ver_mean=round(float(np.mean(vers)), 3), ver_std=round(float(np.std(vers)), 3),
                        restarts=R, best=best))
    print(results[-1])

meta = dict(dataset="UCI Concrete Compressive Strength (Yeh, 1998)", n=int(n), columns=cols,
            split="70/15/15", surrogate="MLP 8-128-128-128-1 ReLU",
            verifier="MLP 8-96x4-1 ReLU (independent split/seed)",
            rmse_sur_mpa=round(rmse_sur, 3), rmse_ver_mpa=round(rmse_ver, 3),
            lo=np.round(lo_raw, 1).tolist(), hi=np.round(hi_raw, 1).tolist(),
            inverse_iters=800, restarts=R, runtime_s=round(time.time() - t1, 1))
json.dump(dict(meta=meta, results=results), open(f"{OUT}/concrete_results.json", "w"), indent=1)
print("DONE concrete", meta["runtime_s"], "s")
