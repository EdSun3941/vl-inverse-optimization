"""VL multi-objective aggregation on UCI Energy Efficiency (ENB2012, Tsanas & Xifara 2012).

Two separately trained frozen surrogates (Y1 heating load, Y2 cooling load) share one
virtual layer. Weighted-sum aggregation; reachable + unreachable targets.
Projection handles box bounds AND discrete feasible sets (X5-X8).
Validation: independent joint verifier + k-NN. 20 restarts/scenario.
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnlite import MLP, Adam, train_mlp

OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{OUT}/energy.csv")
X = df.iloc[:, :8].values.astype(np.float64)
Y = df.iloc[:, 8:10].values.astype(np.float64)
cols = list(df.columns[:8])
n = len(df)
print("data", X.shape, "targets", list(df.columns[8:10]))

perm = np.random.RandomState(0).permutation(n)
ntr, nva = int(0.7 * n), int(0.15 * n)
itr, iva, ite = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

mu_x, sd_x = X[itr].mean(0), X[itr].std(0) + 1e-8
mu_y, sd_y = Y[itr].mean(0), Y[itr].std(0) + 1e-8
Xs, Ys = (X - mu_x) / sd_x, (Y - mu_y) / sd_y

t0 = time.time()
f1 = MLP([8, 128, 128, 128, 1], seed=1)
train_mlp(f1, Xs[itr], Ys[itr, :1], Xs[iva], Ys[iva, :1], epochs=800, seed=1)
f2 = MLP([8, 128, 128, 128, 1], seed=2)
train_mlp(f2, Xs[itr], Ys[itr, 1:], Xs[iva], Ys[iva, 1:], epochs=800, seed=2)
pool = np.concatenate([itr, iva])
p2 = np.random.RandomState(7).permutation(pool)
cut = int(0.85 * len(p2))
ver = MLP([8, 96, 96, 96, 96, 2], seed=7)
train_mlp(ver, Xs[p2[:cut]], Ys[p2[:cut]], Xs[p2[cut:]], Ys[p2[cut:]], epochs=800, seed=7)

t1m = float(np.mean((f1.forward(Xs[ite]) - Ys[ite, :1]) ** 2))
t2m = float(np.mean((f2.forward(Xs[ite]) - Ys[ite, 1:]) ** 2))
rmse1 = float(np.sqrt(t1m) * sd_y[0]); rmse2 = float(np.sqrt(t2m) * sd_y[1])
print(f"f1 RMSE {rmse1:.3f} kWh/m2 | f2 RMSE {rmse2:.3f} | train {time.time()-t0:.0f}s")

lo_raw, hi_raw = X[itr].min(0), X[itr].max(0)
# discrete feasible values for X5 height, X6 orientation, X7 glazing area, X8 glazing distribution
DISCRETE = {4: np.array([3.5, 7.0]), 5: np.array([2., 3., 4., 5.]),
            6: np.array([0.0, 0.10, 0.25, 0.40]), 7: np.array([0., 1., 2., 3., 4., 5.])}

def project(w_std):
    z = w_std * sd_x + mu_x
    z = np.clip(z, lo_raw, hi_raw)
    for i, vals in DISCRETE.items():
        z[i] = vals[np.argmin(np.abs(vals - z[i]))]
    return (z - mu_x) / sd_x

def vl_inverse(tgt_raw, alphas=(1.0, 1.0), iters=800, lr=0.05, seed=0):
    rng = np.random.RandomState(seed)
    w = project(rng.randn(8) * 0.5)
    t1s = (tgt_raw[0] - mu_y[0]) / sd_y[0]
    t2s = (tgt_raw[1] - mu_y[1]) / sd_y[1]
    opt = Adam([(8,)], lr=lr)
    for k in range(iters):
        if k and k % 250 == 0:
            opt.lr *= 0.2
        pr1 = f1.forward(w[None, :], cache=True)
        pr2 = f2.forward(w[None, :], cache=True)
        g1, _, _ = f1.backward(2.0 * alphas[0] * (pr1 - t1s), want_param_grads=False)
        g2, _, _ = f2.backward(2.0 * alphas[1] * (pr2 - t2s), want_param_grads=False)
        (w,) = opt.step([w], [g1[0] + g2[0]])
        w = project(w)
    p1 = float(f1.forward(w[None, :])[0, 0] * sd_y[0] + mu_y[0])
    p2_ = float(f2.forward(w[None, :])[0, 0] * sd_y[1] + mu_y[1])
    loss = alphas[0] * (p1 - tgt_raw[0]) ** 2 + alphas[1] * (p2_ - tgt_raw[1]) ** 2
    pv = ver.forward(w[None, :])[0] * sd_y + mu_y
    x_raw = w * sd_x + mu_x
    return x_raw, (p1, p2_), pv.tolist(), float(loss)

def knn(x_raw, k=5):
    d = np.linalg.norm((X - x_raw) / sd_x, axis=1)
    idx = np.argsort(d)[:k]
    return Y[idx].mean(0).round(2).tolist(), float(d[idx].mean())

scenarios = [
    dict(name="reach_low",   target=[15.0, 18.0], alphas=(1, 1)),
    dict(name="reach_high",  target=[35.0, 35.0], alphas=(1, 1)),
    dict(name="unreach_eq",  target=[10.0, 40.0], alphas=(1, 1)),
    dict(name="unreach_w41", target=[10.0, 40.0], alphas=(4, 1)),
]

R = 20
results = []
t1 = time.time()
for sc in scenarios:
    P1, P2, L = [], [], []
    best = None
    for r in range(R):
        x_raw, (p1, p2_), pv, l = vl_inverse(sc["target"], sc["alphas"], seed=100 + r)
        P1.append(p1); P2.append(p2_); L.append(l)
        if best is None or l < best["wloss"]:
            ky, kd = knn(x_raw)
            best = dict(x=np.round(x_raw, 2).tolist(), pred=[round(p1, 2), round(p2_, 2)],
                        verifier=[round(v, 2) for v in pv], knn_Y=ky, knn_dist=round(kd, 3),
                        wloss=round(l, 4))
    results.append(dict(scenario=sc["name"], target=sc["target"], alphas=list(sc["alphas"]),
                        p1_mean=round(float(np.mean(P1)), 3), p1_std=round(float(np.std(P1)), 3),
                        p2_mean=round(float(np.mean(P2)), 3), p2_std=round(float(np.std(P2)), 3),
                        wloss_mean=round(float(np.mean(L)), 4), restarts=R, best=best))
    print(results[-1])

meta = dict(dataset="UCI Energy Efficiency ENB2012 (Tsanas & Xifara, 2012)", n=int(n), columns=cols,
            split="70/15/15", surrogates="2x MLP 8-128-128-128-1 ReLU (separate Y1/Y2)",
            verifier="MLP 8-96x4-2 ReLU (independent split/seed)",
            rmse_heating=round(rmse1, 3), rmse_cooling=round(rmse2, 3),
            discrete_projection="X5 in {3.5,7}, X6 in {2..5}, X7 in {0,.1,.25,.4}, X8 in {0..5}",
            inverse_iters=800, restarts=R, runtime_s=round(time.time() - t1, 1))
json.dump(dict(meta=meta, results=results), open(f"{OUT}/energy_results.json", "w"), indent=1)
print("DONE energy", meta["runtime_s"], "s")
