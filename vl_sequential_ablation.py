"""Ablation for Algorithm 2 (sequential VL with endogenous states).

Ground truth: VARX process
  Y(t) = a1 Y(t-1) + a2 Y(t-2) + sum_{j=1..4} b_j . X(t-j) + c + eps
Surrogate: MLP that maps the flattened 6-step history
  [X(t-5..t), Y(t-5..t)] (6*3+6 = 24 dims) -> Y(t+1).

Inverse task: choose X(t-5..t) (18 vars) so that Y(t+1) hits a target,
with pre-window history fixed. Endogenous Y(t-4..t) inside the window must
stay consistent: recomputed by rolling the frozen surrogate when X's change.

Methods compared (same total gradient-step budget):
  A. Sequential (Algorithm 2): optimize one X step at a time (newest to
     oldest), re-propagating affected endogenous Y's after each block.
  B. Joint: optimize all 18 vars simultaneously; endogenous Y's refreshed
     by a no-grad surrogate rollout every iteration.
  C. Stable broadcast: one shared 3-dim X across the window (paper's
     equality-constraint variant).

Validation: roll the TRUE VARX process with the optimized X's (fixed
pre-window true history) and compare Y_true(t+1) with the target.
20 random restarts; 3 targets.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnlite import MLP, Adam, train_mlp

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.RandomState(0)

DX, LAG = 3, 6           # X dim, history window length
A = np.array([0.45, -0.2])
B = np.array([[0.8, -0.5, 0.3],
              [0.4, 0.2, -0.3],
              [-0.2, 0.5, 0.1],
              [0.1, -0.1, 0.2]])   # b_j for j=1..4
C0 = 0.05
NOISE = 0.01

def varx_step(Yh, Xh):
    """Yh: (...,>=2) recent Y's [.., Y(t-1), Y(t)]; Xh: (...,>=4, DX) [.., X(t)]. Returns Y(t+1) noiseless."""
    return (A[0] * Yh[..., -1] + A[1] * Yh[..., -2]
            + sum(B[j - 1] @ Xh[..., -j, :].T if Xh.ndim == 3 else B[j - 1] @ Xh[-j] for j in range(1, 5))
            + C0)

# ---- generate training series ----
TT = 60000
X = rng.uniform(-1, 1, (TT, DX))
Y = np.zeros(TT)
for t in range(4, TT - 1):
    Y[t + 1] = (A[0] * Y[t] + A[1] * Y[t - 1]
                + sum(B[j - 1] @ X[t + 1 - j] for j in range(1, 5))
                + C0 + rng.randn() * NOISE)

# build supervised pairs: history of LAG steps ending at t -> Y(t+1)
rows = []
tgts = []
for t in range(LAG + 4, TT - 1):
    h = np.concatenate([X[t - LAG + 1:t + 1].ravel(), Y[t - LAG + 1:t + 1]])
    rows.append(h); tgts.append(Y[t + 1])
H = np.array(rows); G = np.array(tgts).reshape(-1, 1)
n = len(H)
perm = np.random.RandomState(1).permutation(n)
itr, iva, ite = perm[:int(0.8 * n)], perm[int(0.8 * n):int(0.9 * n)], perm[int(0.9 * n):]

mu_h, sd_h = H[itr].mean(0), H[itr].std(0) + 1e-8
mu_g, sd_g = G[itr].mean(0), G[itr].std(0) + 1e-8
Hs, Gs = (H - mu_h) / sd_h, (G - mu_g) / sd_g

t0 = time.time()
sur = MLP([LAG * DX + LAG, 128, 128, 1], seed=2)
train_mlp(sur, Hs[itr], Gs[itr], Hs[iva], Gs[iva], epochs=60, bs=1024, lr=2e-3,
          lr_step=20, lr_gamma=0.3, seed=2)
test_mse = float(np.mean((sur.forward(Hs[ite]) - Gs[ite]) ** 2))
print(f"surrogate test MSE(std) {test_mse:.5f} | train {time.time()-t0:.0f}s")

def sur_pred(Xwin, Ywin):
    """Xwin: (LAG,DX), Ywin: (LAG,) -> predicted Y(t+1) raw."""
    h = np.concatenate([Xwin.ravel(), Ywin])
    return float(sur.forward(((h - mu_h) / sd_h)[None, :])[0, 0] * sd_g[0] + mu_g[0])

def sur_grad_x(Xwin, Ywin, tgt_raw):
    """Gradient of (pred - tgt)^2 wrt Xwin entries (raw units)."""
    h = np.concatenate([Xwin.ravel(), Ywin])
    hs = ((h - mu_h) / sd_h)[None, :]
    pred = sur.forward(hs, cache=True)
    tgt_s = (tgt_raw - mu_g[0]) / sd_g[0]
    dOut = 2.0 * (pred - tgt_s)
    dH, _, _ = sur.backward(dOut, want_param_grads=False)
    g = (dH[0] / sd_h)[:LAG * DX].reshape(LAG, DX)   # chain rule through standardization
    return g, float(pred[0, 0] * sd_g[0] + mu_g[0])

# choose an evaluation window from held-out part of the series
t_eval = TT - 10
X0 = X[t_eval - LAG + 1:t_eval + 1].copy()       # (LAG,DX) original inputs
Ypre = Y[t_eval - LAG - 1:t_eval - LAG + 1].copy()  # true Y just before window: Y(t-LAG), Y(t-LAG+1-1)... fixed

def propagate_endo(Xwin):
    """Recompute endogenous Y inside the window with the FROZEN SURROGATE,
    given fixed pre-window true history. Returns Ywin (LAG,)."""
    # pre-window context: use true X and Y before the window
    Yfull = list(Y[t_eval - LAG - 6:t_eval - LAG + 1])  # ample true context
    Xfull = list(X[t_eval - LAG - 6:t_eval - LAG + 1])
    for i in range(LAG):
        Xfull.append(Xwin[i])
        hX = np.array(Xfull[-LAG:]); hY = np.array(Yfull[-LAG:])
        Yfull.append(sur_pred(hX, hY))
    return np.array(Yfull[-LAG:])

def true_outcome(Xwin):
    """Roll the TRUE VARX with optimized X's (noiseless) and return Y(t+1)."""
    Yfull = list(Y[t_eval - LAG - 6:t_eval - LAG + 1])
    Xfull = list(X[t_eval - LAG - 6:t_eval - LAG + 1])
    for i in range(LAG):
        Xfull.append(Xwin[i])
        ynext = (A[0] * Yfull[-1] + A[1] * Yfull[-2]
                 + sum(B[j - 1] @ np.array(Xfull)[-j] for j in range(1, 5)) + C0)
        Yfull.append(ynext)
    return Yfull[-1]  # Y at window end + ... = Y(t+1) after final step

BUDGET = 600   # total gradient steps per method

def run_sequential(tgt, seed):
    r = np.random.RandomState(seed)
    Xw = X0 + r.randn(LAG, DX) * 0.1
    per = BUDGET // LAG
    for i in range(LAG):                 # newest to oldest per Alg.2 (tau = t - i)
        idx = LAG - 1 - i
        opt = Adam([(DX,)], lr=0.05)
        for k in range(per):
            Yw = propagate_endo(Xw)
            g, _ = sur_grad_x(Xw, Yw, tgt)
            (Xw[idx],) = opt.step([Xw[idx]], [g[idx]])
            Xw[idx] = np.clip(Xw[idx], -1, 1)
    return Xw

def run_joint(tgt, seed):
    r = np.random.RandomState(seed)
    Xw = X0 + r.randn(LAG, DX) * 0.1
    opt = Adam([(LAG, DX)], lr=0.05)
    for k in range(BUDGET):
        Yw = propagate_endo(Xw)
        g, _ = sur_grad_x(Xw, Yw, tgt)
        (Xw,) = opt.step([Xw], [g])
        Xw = np.clip(Xw, -1, 1)
    return Xw

def run_stable(tgt, seed):
    r = np.random.RandomState(seed)
    xs = X0.mean(0) + r.randn(DX) * 0.1
    opt = Adam([(DX,)], lr=0.05)
    for k in range(BUDGET):
        Xw = np.tile(xs, (LAG, 1))
        Yw = propagate_endo(Xw)
        g, _ = sur_grad_x(Xw, Yw, tgt)
        (xs,) = opt.step([xs], [g.sum(0)])
        xs = np.clip(xs, -1, 1)
    return np.tile(xs, (LAG, 1))

targets = [0.5, 0.0, -0.5]
R = 20
methods = {"Sequential (Alg. 2)": run_sequential, "Joint": run_joint, "Stable broadcast": run_stable}
out = {}
for name, fn in methods.items():
    rows = []
    for tgt in targets:
        sur_outs, true_outs, tms = [], [], []
        for rr in range(R):
            t0 = time.time()
            Xw = fn(tgt, 100 + rr)
            tms.append(time.time() - t0)
            Yw = propagate_endo(Xw)
            sur_outs.append(sur_pred(Xw, Yw))
            true_outs.append(true_outcome(Xw))
        rows.append(dict(target=tgt,
                         sur_mean=round(float(np.mean(sur_outs)), 4), sur_std=round(float(np.std(sur_outs)), 4),
                         true_mean=round(float(np.mean(true_outs)), 4), true_std=round(float(np.std(true_outs)), 4),
                         time_s=round(float(np.mean(tms)), 2)))
        print(name, rows[-1])
    out[name] = rows

meta = dict(budget_grad_steps=BUDGET, restarts=R, window=LAG, x_dim=DX,
            surrogate="MLP 24-128-128-1 ReLU on flattened history",
            surrogate_test_mse_std=test_mse,
            truth="VARX a=[0.45,-0.2], 4-lag B, c=0.05, noise 0.01")
json.dump(dict(meta=meta, results=out), open(f"{OUT}/sequential_ablation_results.json", "w"), indent=1)
print("DONE")
