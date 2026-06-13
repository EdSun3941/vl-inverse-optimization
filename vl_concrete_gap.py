"""Decisive constrained scenario on UCI Concrete: a (constraint, target) pair for
which NO measured training record near the target satisfies the constraint, so
constrained nearest-record retrieval is capped well below the target, while
projected VL reaches it with independent-verifier support.

Constraint: curing age = 3 days (early-strength concrete).
Target:     45 MPa.
In the dataset the maximum strength at age = 3 is 41.6 MPa and there is no age-3
record within 2 MPa of 45, so constrained k-NN cannot return a feasible record
near the target. This isolates VL's advantage: optimization in the constrained
space rather than retrieval from a sparse region.

Reuses the exact surrogate/verifier training setup of vl_concrete_np.py (same
seeds and splits) so the models are identical to the main concrete experiment.
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

perm = np.random.RandomState(0).permutation(n)
ntr, nva = int(0.7 * n), int(0.15 * n)
itr, iva, ite = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]

mu_x, sd_x = X[itr].mean(0), X[itr].std(0) + 1e-8
mu_y, sd_y = y[itr].mean(0), y[itr].std(0) + 1e-8
Xs, ys = (X - mu_x) / sd_x, (y - mu_y) / sd_y

t0 = time.time()
sur = MLP([8, 128, 128, 128, 1], seed=1)
train_mlp(sur, Xs[itr], ys[itr], Xs[iva], ys[iva], epochs=600, seed=1)
pool = np.concatenate([itr, iva])
p2 = np.random.RandomState(7).permutation(pool)
cut = int(0.85 * len(p2))
ver = MLP([8, 96, 96, 96, 96, 1], seed=7)
train_mlp(ver, Xs[p2[:cut]], ys[p2[:cut]], Xs[p2[cut:]], ys[p2[cut:]], epochs=600, seed=7)
rmse_ver = float(np.sqrt(np.mean((ver.forward(Xs[ite]) - ys[ite]) ** 2)) * sd_y[0])

lo_raw, hi_raw = X[itr].min(0), X[itr].max(0)
AGE = cols.index("age")

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
        pred = sur.forward(w[None, :], cache=True)
        dOut = 2.0 * (pred - tgt)
        dX, _, _ = sur.backward(dOut, want_param_grads=False)
        (w,) = opt.step([w], [dX[0]])
        w = project(w, fixed)
    pred = float(sur.forward(w[None, :])[0, 0] * sd_y[0] + mu_y[0])
    vchk = float(ver.forward(w[None, :])[0, 0] * sd_y[0] + mu_y[0])
    return w * sd_x + mu_x, pred, vchk

def knn_inputs(x_raw, k=5):
    d = np.linalg.norm((X - x_raw) / sd_x, axis=1)
    idx = np.argsort(d)[:k]
    return float(y[idx].mean()), float(d[idx].mean())

# ---------------- gap scenario ----------------
TARGET, AGE_FIX = 45.0, 3.0
age = X[:, AGE]

# (a) unconstrained nearest-output k-NN: globally closest record in strength
i_un = int(np.argmin(np.abs(y[:, 0] - TARGET)))
unc = dict(strength=float(y[i_un, 0]), age=float(age[i_un]),
           constraint_satisfied=bool(np.isclose(age[i_un], AGE_FIX)))

# (b) constrained k-NN: pre-filter to age == 3, then nearest in strength to target
elig = np.where(np.isclose(age, AGE_FIX))[0]
j = elig[int(np.argmin(np.abs(y[elig, 0] - TARGET)))]
con = dict(n_eligible=int(elig.size), best_strength=float(y[j, 0]),
           max_strength_at_age=float(y[elig, 0].max()),
           gap_to_target=round(float(TARGET - y[elig, 0].max()), 2),
           n_within_2MPa=int((np.abs(y[elig, 0] - TARGET) < 2).sum()))

# (c) projected VL: 20 restarts, best by |surrogate - target|
R = 20
preds, vers, best = [], [], None
cap = con["max_strength_at_age"]   # 41.6 MPa: the best constrained-retrieval can do
for r in range(R):
    x_raw, pred, vchk = vl_inverse(TARGET, {AGE: AGE_FIX}, seed=100 + r)
    preds.append(pred); vers.append(vchk)
    if best is None or abs(pred - TARGET) < abs(best["pred"] - TARGET):
        ky, kd = knn_inputs(x_raw)
        best = dict(pred=round(pred, 3), verifier=round(vchk, 3),
                    age=round(float(x_raw[AGE]), 2),
                    knn_strength=round(ky, 2), knn_dist=round(kd, 3),
                    mix=np.round(x_raw, 1).tolist())
preds, vers = np.array(preds), np.array(vers)
rate_surrogate_hit = float(np.mean(np.abs(preds - TARGET) < 1.0))   # converged to target
rate_beats_cap = float(np.mean(vers >= cap))                        # verifier beats retrieval cap

out = dict(
    meta=dict(scenario="age=3 d, target=45 MPa (sparse constrained region)",
              verifier_rmse_mpa=round(rmse_ver, 2), restarts=R,
              note="max measured strength at age=3 is %.1f MPa" % con["max_strength_at_age"]),
    target=TARGET,
    unconstrained_knn=unc,
    constrained_knn=con,
    vl=dict(pred_mean=round(float(np.mean(preds)), 3), pred_std=round(float(np.std(preds)), 3),
            ver_mean=round(float(np.mean(vers)), 3), ver_std=round(float(np.std(vers)), 3),
            rate_surrogate_within_1MPa=round(rate_surrogate_hit, 3),
            rate_verifier_beats_retrieval_cap=round(rate_beats_cap, 3),
            best=best),
)
json.dump(out, open(f"{OUT}/concrete_gap_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("train+run %.0fs" % (time.time() - t0))
