"""Full-protocol robotic-arm inverse-kinematics benchmark (Ren et al., 2020 style).

- Ground-truth forward process: 4-DOF planar arm endpoint (Eq. in paper).
- Train frozen MLP surrogate 4->2 on U(-2,2) inputs.
- T=200 test targets drawn from the benchmark test distribution
  x ~ N(0, diag([1/16,1/4,1/4,1/4])), y = f_true(x)  (guaranteed reachable).
- Methods (equal forward-evaluation budget where applicable):
  * VL / neural-adjoint gradient inversion (identical updates; reported once)
    with R=10 restarts per target, best by surrogate loss.
  * Projected VL: same, with box projection to the training domain each step.
  * Warm-start projected VL: initialized from k-NN retrieved inputs.
  * (mu,lambda) evolution strategy on the surrogate, same eval budget.
  * Random search on the surrogate, same eval budget.
  * k-NN retrieval from training data (no optimization).
- Metric: re-simulation MSE = || f_true(x_rec) - y_target ||^2,
  reported as median / mean +- std / 95th pct over the 200 targets, plus runtime.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnlite import MLP, Adam, train_mlp

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.RandomState(0)

L1, L2, L3 = 0.5, 0.5, 1.0

def f_true(X):
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y1 = L1*np.sin(x2) + L2*np.sin(x3-x2) + L3*np.sin(x4-x3-x2) + x1
    y2 = L1*np.cos(x2) + L2*np.cos(x3-x2) + L3*np.cos(x4-x3-x2)
    return np.stack([y1, y2], 1)

# ---- data and frozen surrogate ----
N = 50000
Xtr = rng.uniform(-2, 2, (N, 4))
Ytr = f_true(Xtr)
Xva = rng.uniform(-2, 2, (5000, 4))
Yva = f_true(Xva)

mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-8
mu_y, sd_y = Ytr.mean(0), Ytr.std(0) + 1e-8
s = lambda X: (X - mu_x) / sd_x
sy = lambda Y: (Y - mu_y) / sd_y
inv_s = lambda Z: Z * sd_x + mu_x

t0 = time.time()
sur = MLP([4, 128, 128, 128, 2], seed=1)
vbest = train_mlp(sur, s(Xtr), sy(Ytr), s(Xva), sy(Yva), epochs=120, bs=1024, lr=2e-3,
                  lr_step=40, lr_gamma=0.3, seed=1)
Xte_check = rng.uniform(-2, 2, (5000, 4))
test_loss = float(np.mean((sur.forward(s(Xte_check)) - sy(f_true(Xte_check))) ** 2))
print(f"surrogate val {vbest:.5f} test {test_loss:.5f} (std units) | train {time.time()-t0:.0f}s")

# ---- benchmark targets ----
T = 200
sig = np.sqrt(np.array([1/16, 1/4, 1/4, 1/4]))
Xtest = np.random.RandomState(42).randn(T, 4) * sig
Ytgt = f_true(Xtest)                      # reachable targets
Ytgt_s = sy(Ytgt)

R = 10            # restarts per target (gradient method)
ITERS = 300       # gradient iterations
BUDGET = R * ITERS  # forward-eval budget per target for fair baselines

def resim_mse(Xrec):
    return np.sum((f_true(Xrec) - Ytgt) ** 2, axis=1)   # per-target squared L2

# standardized box bounds of the training domain [-2,2]^4
lo_s, hi_s = s(np.full((1, 4), -2.0))[0], s(np.full((1, 4), 2.0))[0]

def grad_invert(W0, project=False):
    """Batched gradient inversion (VL / neural adjoint). W0: (T*R,4) standardized."""
    W = W0.copy()
    Tgt = np.repeat(Ytgt_s, W.shape[0] // T, axis=0)
    opt = Adam([W.shape], lr=0.05)
    for k in range(ITERS):
        if k and k % 100 == 0:
            opt.lr *= 0.3
        pred = sur.forward(W, cache=True)
        dOut = 2.0 * (pred - Tgt)
        dX, _, _ = sur.backward(dOut, want_param_grads=False)
        (W,) = opt.step([W], [dX])
        if project:
            W = np.clip(W, lo_s, hi_s)
    R_ = W.shape[0] // T
    sur_loss = np.sum((sur.forward(W) - Tgt) ** 2, axis=1).reshape(T, R_)
    pick = sur_loss.argmin(1)
    return inv_s(W.reshape(T, R_, 4)[np.arange(T), pick])

init = np.random.RandomState(7).randn(T * R, 4)

# ---- VL / neural-adjoint, unconstrained (NA-equivalent) ----
t0 = time.time()
X_vl = grad_invert(init, project=False)
t_vl = time.time() - t0
e_vl = resim_mse(X_vl)

# ---- projected VL: box projection to training domain each step ----
t0 = time.time()
X_pvl = grad_invert(init, project=True)
t_pvl = time.time() - t0
e_pvl = resim_mse(X_pvl)

# ---- (mu, lambda) ES on surrogate, same budget ----
t0 = time.time()
MU, LAM = 5, 20
GENS = BUDGET // LAM
es_rng = np.random.RandomState(11)
pop = es_rng.randn(T, MU, 4)
sigma = np.full((T, MU, 4), 0.5)
for g in range(GENS):
    parents = pop[:, es_rng.randint(0, MU, LAM), :]               # (T,LAM,4)
    psig = sigma[:, es_rng.randint(0, MU, LAM), :]
    off = parents + psig * es_rng.randn(T, LAM, 4)
    flat = off.reshape(T * LAM, 4)
    loss = np.sum((sur.forward(flat) - np.repeat(Ytgt_s, LAM, 0)) ** 2, 1).reshape(T, LAM)
    idx = np.argsort(loss, axis=1)[:, :MU]
    pop = np.take_along_axis(off, idx[:, :, None], axis=1)
    sigma = np.maximum(sigma * 0.97, 0.02)
loss_mu = np.sum((sur.forward(pop.reshape(T * MU, 4)) - np.repeat(Ytgt_s, MU, 0)) ** 2, 1).reshape(T, MU)
X_es = inv_s(pop[np.arange(T), loss_mu.argmin(1)])
t_es = time.time() - t0
e_es = resim_mse(X_es)

# ---- random search on surrogate, same budget ----
t0 = time.time()
rs_rng = np.random.RandomState(13)
bestX = np.zeros((T, 4)); bestL = np.full(T, np.inf)
CH = 500
done = 0
while done < BUDGET:
    cand = rs_rng.randn(T, CH, 4)         # standardized-space proposals
    flat = cand.reshape(T * CH, 4)
    loss = np.sum((sur.forward(flat) - np.repeat(Ytgt_s, CH, 0)) ** 2, 1).reshape(T, CH)
    m = loss.min(1); a = loss.argmin(1)
    upd = m < bestL
    bestL[upd] = m[upd]
    bestX[upd] = cand[np.arange(T), a][upd]
    done += CH
X_rs = inv_s(bestX)
t_rs = time.time() - t0
e_rs = resim_mse(X_rs)

# ---- k-NN retrieval from training data ----
t0 = time.time()
d2 = ((Ytr[None, :, :1] - Ytgt[:, None, :1]) ** 2 + (Ytr[None, :, 1:2] - Ytgt[:, None, 1:2]) ** 2)[:, :, 0]
order = np.argsort(d2, axis=1)[:, :R]      # R nearest for warm start reuse
X_nn = Xtr[order[:, 0]]
t_nn = time.time() - t0
e_nn = resim_mse(X_nn)

# ---- retrieval-initialized projected VL (warm start) ----
t0 = time.time()
warm = s(Xtr[order.reshape(-1)])           # (T*R,4) standardized
X_wvl = grad_invert(warm, project=True)
t_wvl = time.time() - t0 + t_nn
e_wvl = resim_mse(X_wvl)

FAIL_THRESH = 0.05   # re-simulation error above which a target counts as a failure

def out_of_domain(Xrec):
    """Count recommendations that fall outside the training box [-2,2]^4."""
    return int(np.sum(np.any(np.abs(Xrec) > 2.0 + 1e-9, axis=1)))

def stats(e, t, Xrec=None):
    d = dict(median=float(np.median(e)), mean=float(np.mean(e)), std=float(np.std(e)),
             p95=float(np.percentile(e, 95)),
             failures=int(np.sum(e > FAIL_THRESH)),   # exact count, not a p95 estimate
             max_error=float(np.max(e)),
             time_s=round(t, 2),
             per_target_error=[float(x) for x in e])  # full per-target log for auditability
    if Xrec is not None:
        d["out_of_domain"] = out_of_domain(Xrec)
    return d

res = {
  "VL / neural adjoint (grad)": stats(e_vl, t_vl, X_vl),
  "Projected VL (box [-2,2])": stats(e_pvl, t_pvl, X_pvl),
  "Warm-start projected VL": stats(e_wvl, t_wvl, X_wvl),
  "Evolution strategy": stats(e_es, t_es, X_es),
  "Random search": stats(e_rs, t_rs, X_rs),
  "k-NN retrieval": stats(e_nn, t_nn, X_nn),
}
for k, v in res.items():
    print(f"{k:28s} median {v['median']:.2e} mean {v['mean']:.2e}+-{v['std']:.1e} "
          f"p95 {v['p95']:.2e} fail {v['failures']} max {v['max_error']:.2f} time {v['time_s']}s")

meta = dict(T=T, restarts=R, iters=ITERS, budget_per_target=BUDGET,
            surrogate="MLP 4-128-128-128-2 ReLU", surrogate_test_mse_std=test_loss,
            failure_threshold=FAIL_THRESH, training_box=[-2.0, 2.0],
            protocol="re-simulation MSE vs reachable targets from N(0,diag(1/16,1/4,1/4,1/4))")
json.dump(dict(meta=meta, results=res), open(f"{OUT}/robotic_benchmark_results.json", "w"), indent=1)
print("DONE")
