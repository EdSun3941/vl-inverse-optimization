"""Minimal numpy MLP + Adam, with input-gradient support for VL inverse optimization."""
import numpy as np


class MLP:
    def __init__(self, dims, seed=0):
        """dims e.g. [8, 128, 128, 128, 1]"""
        rng = np.random.RandomState(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            # He init
            self.W.append(rng.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i]))
            self.b.append(np.zeros(dims[i + 1]))
        self.L = len(self.W)

    def forward(self, X, cache=False):
        h = X
        hs = [X]
        for i in range(self.L):
            a = h @ self.W[i] + self.b[i]
            h = np.maximum(a, 0.0) if i < self.L - 1 else a
            if cache:
                hs.append(h)
        if cache:
            self._hs = hs
        return h

    def backward(self, dOut, want_param_grads=True):
        """Call after forward(cache=True). Returns (dX, dWs, dbs)."""
        hs = self._hs
        dWs = [None] * self.L
        dbs = [None] * self.L
        d = dOut
        for i in range(self.L - 1, -1, -1):
            if i < self.L - 1:           # relu mask of layer output
                d = d * (hs[i + 1] > 0)
            if want_param_grads:
                dWs[i] = hs[i].T @ d
                dbs[i] = d.sum(0)
            d = d @ self.W[i].T
        return d, dWs, dbs

    def params(self):
        return self.W + self.b

    def set_params(self, ps):
        self.W = [p.copy() for p in ps[:self.L]]
        self.b = [p.copy() for p in ps[self.L:]]


class Adam:
    def __init__(self, shapes, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        out = []
        for p, g, m, v in zip(params, grads, self.m, self.v):
            if self.wd:
                g = g + self.wd * p
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + self.eps))
        return out


def train_mlp(model, Xtr, ytr, Xva, yva, epochs=600, lr=1e-3, bs=128, wd=1e-4,
              lr_step=200, lr_gamma=0.3, seed=0, verbose=False):
    rng = np.random.RandomState(seed)
    shapes = [p.shape for p in model.params()]
    opt = Adam(shapes, lr=lr, wd=wd)
    best, best_ps = np.inf, None
    n = len(Xtr)
    for ep in range(epochs):
        if ep and ep % lr_step == 0:
            opt.lr *= lr_gamma
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = model.forward(Xtr[idx], cache=True)
            dOut = 2.0 * (pred - ytr[idx]) / len(idx)
            _, dWs, dbs = model.backward(dOut)
            new = opt.step(model.params(), dWs + dbs)
            model.W = new[:model.L]
            model.b = new[model.L:]
        vl = float(np.mean((model.forward(Xva) - yva) ** 2))
        if vl < best:
            best, best_ps = vl, [p.copy() for p in model.params()]
        if verbose and ep % 100 == 0:
            print(f"  ep {ep} val {vl:.5f}")
    model.set_params(best_ps)
    return best
