"""Shared pieces for the deep-RL runs of "Trial and Reward" (Modules 4-5).

Pure numpy: a small MLP with hand-written backprop, Adam, and helpers for logging, the
"solved" criterion, curve downsampling and the weights format the scenes load.

Weights format (research/experiments/weights/<name>.npz): arrays W0, b0, W1, b1, ...
layer i computes x @ Wi + bi, the activation sits between layers, nothing after the last.
"""
import json, time
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parent.parent          # research/experiments
LOGS = EXP / "logs"
WEIGHTS = EXP / "weights"


# ------------------------------------------------------------------------------------------ MLP
def _orthogonal(n_in, n_out, gain, rng):
    a = rng.standard_normal((max(n_in, n_out), min(n_in, n_out)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if n_in < n_out:
        q = q.T
    return gain * q[:n_in, :n_out]


class MLP:
    """x @ W0 + b0 -> act -> ... -> x @ WL + bL (no activation after the last layer)."""

    def __init__(self, sizes, activation="relu", rng=None, init="torch", gains=None, dtype=np.float32):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.sizes, self.activation, self.dtype = list(sizes), activation, dtype
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            n_in, n_out = sizes[i], sizes[i + 1]
            if init == "torch":          # PyTorch nn.Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
                bound = 1.0 / np.sqrt(n_in)
                W = rng.uniform(-bound, bound, (n_in, n_out)); b = rng.uniform(-bound, bound, n_out)
            elif init == "orthogonal":   # CleanRL / OpenAI baselines PPO init
                W = _orthogonal(n_in, n_out, gains[i], rng); b = np.zeros(n_out)
            else:
                raise ValueError(init)
            self.W.append(W.astype(dtype)); self.b.append(b.astype(dtype))

    def params(self):
        out = []
        for W, b in zip(self.W, self.b):
            out += [W, b]
        return out

    def _act(self, h):
        if self.activation == "relu":
            return np.maximum(h, 0, out=h)
        return np.tanh(h, out=h)

    def forward(self, x):
        h = np.asarray(x, dtype=self.dtype)
        L = len(self.W)
        for i in range(L):
            h = h @ self.W[i]; h += self.b[i]
            if i < L - 1:
                h = self._act(h)
        return h

    def forward_cache(self, x):
        """Forward pass that keeps every layer's input for backward()."""
        h = np.asarray(x, dtype=self.dtype)
        cache = []
        L = len(self.W)
        for i in range(L):
            cache.append(h)
            h = h @ self.W[i]; h += self.b[i]
            if i < L - 1:
                h = self._act(h)
        return h, cache

    def backward(self, cache, g, input_grad=False, param_grads=True):
        """g = dLoss/dOutput (batch, out). Returns (grads in params() order, dLoss/dInput or None)."""
        L = len(self.W)
        grads = [None] * (2 * L)
        g = np.asarray(g, dtype=self.dtype)
        for i in range(L - 1, -1, -1):
            if param_grads:
                grads[2 * i] = cache[i].T @ g
                grads[2 * i + 1] = g.sum(0)
            if i > 0 or input_grad:
                g = g @ self.W[i].T
                if i > 0:
                    a = cache[i]
                    if self.activation == "relu":
                        np.multiply(g, a > 0, out=g)
                    else:
                        g *= 1.0 - a * a
        return grads, (g if input_grad else None)

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q

    def polyak_from(self, other, tau):
        for p, q in zip(self.params(), other.params()):
            p *= (1.0 - tau); p += tau * q

    def clone(self):
        c = MLP.__new__(MLP)
        c.sizes, c.activation, c.dtype = list(self.sizes), self.activation, self.dtype
        c.W = [w.copy() for w in self.W]; c.b = [b.copy() for b in self.b]
        return c

    def save_npz(self, path):
        np.savez(path, **{f"W{i}": w.astype(np.float32) for i, w in enumerate(self.W)},
                 **{f"b{i}": b.astype(np.float32) for i, b in enumerate(self.b)})


class Adam:
    """Adam (Kingma & Ba) updating the arrays in place; optional global-norm gradient clipping."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_grad_norm=None):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, betas[0], betas[1], eps
        self.max_grad_norm = max_grad_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.tmp = [np.zeros_like(p) for p in params]     # scratch: no allocations in step()
        self.t = 0

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        if self.max_grad_norm is not None:
            norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
            if norm > self.max_grad_norm:
                s = self.max_grad_norm / (norm + 1e-6)
                grads = [g * s for g in grads]
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t; c2 = 1.0 - self.b2 ** self.t
        step = lr * np.sqrt(c2) / c1
        eps = self.eps * np.sqrt(c2)
        for p, g, m, v, tmp in zip(self.params, grads, self.m, self.v, self.tmp):
            np.multiply(g, 1.0 - self.b1, out=tmp); m *= self.b1; m += tmp
            np.multiply(g, g, out=tmp); tmp *= (1.0 - self.b2); v *= self.b2; v += tmp
            np.sqrt(v, out=tmp); tmp += eps; np.divide(m, tmp, out=tmp); tmp *= step
            p -= tmp


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


# ------------------------------------------------------------------------------- bookkeeping
SOLVE_RETURN, SOLVE_WINDOW = 475.0, 100        # CartPole-v1 reward_threshold, gymnasium registry


class EpisodeLog:
    """Training-episode returns in completion order; tracks the 'solved' point."""

    def __init__(self, solve_return=SOLVE_RETURN, window=SOLVE_WINDOW):
        self.returns, self.lengths, self.end_steps, self.end_times = [], [], [], []
        self.solve_return, self.window = solve_return, window
        self.solved_at = None   # dict(episode, env_steps, wall_s)
        self.t0 = time.perf_counter()

    def add(self, ret, length, env_steps):
        self.returns.append(float(ret)); self.lengths.append(int(length)); self.end_steps.append(int(env_steps))
        self.end_times.append(time.perf_counter() - self.t0)
        if self.solved_at is None and self.solve_return is not None and len(self.returns) >= self.window:
            if np.mean(self.returns[-self.window:]) >= self.solve_return:
                self.solved_at = {"episode": len(self.returns), "env_steps": int(env_steps),
                                  "wall_s": round(self.end_times[-1], 2)}

    def avg100(self):
        r = np.asarray(self.returns, float)
        c = np.cumsum(np.insert(r, 0, 0.0))
        n = np.arange(1, len(r) + 1); lo = np.maximum(0, n - self.window)
        return (c[n] - c[lo]) / (n - lo)

    def to_dict(self):
        return {"returns": self.returns, "lengths": self.lengths, "end_steps": self.end_steps,
                "end_wall_s": [round(t, 3) for t in self.end_times], "solved_at": self.solved_at}


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=None, separators=(",", ":")))


def versions():
    import platform, gymnasium
    return {"python": platform.python_version(), "numpy": np.__version__, "gymnasium": gymnasium.__version__,
            "machine": "Apple M4 Max, 64 GB (Mac Studio)", "platform": platform.platform()}


def set_single_thread():
    """Each seed runs in its own process on one core; keep BLAS from oversubscribing."""
    import os
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
