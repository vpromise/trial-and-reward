"""Finite-difference checks of the hand-written gradients (float64): MLP backprop (ReLU and tanh, parameter
and input gradients), the policy-gradient logit gradient, and SAC's reparameterised actor gradient through the
tanh squashing and both critics.   .venv-deep/bin/python scripts/deep_gradcheck.py -> logs/deep_gradcheck.json
"""
import json
import numpy as np
from deep_common import MLP, LOGS, log_softmax
from deep_pg import pg_logit_grad
from deep_sac import SAC, HP as SAC_HP


def to64(net):
    net.W = [w.astype(np.float64) for w in net.W]; net.b = [b.astype(np.float64) for b in net.b]
    net.dtype = np.float64
    return net


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + np.max(np.abs(b)) + 1e-12))


def fd_params(params, loss, idx_per_param=4, h=1e-6, rng=np.random.default_rng(0)):
    num, ana_idx = [], []
    for pi, p in enumerate(params):
        flat = p.reshape(-1)
        for j in rng.choice(flat.size, min(idx_per_param, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + h; lp = loss()
            flat[j] = old - h; lm = loss()
            flat[j] = old
            num.append((lp - lm) / (2 * h)); ana_idx.append((pi, j))
    return np.array(num), ana_idx


results = {}
rng = np.random.default_rng(1)
for act in ("relu", "tanh"):
    net = to64(MLP([5, 16, 16, 3], act, rng))
    x = rng.standard_normal((7, 5)); y = rng.standard_normal((7, 3))
    loss = lambda: 0.5 * float(((net.forward(x) - y) ** 2).sum())
    out, cache = net.forward_cache(x)
    grads, dx = net.backward(cache, out - y, input_grad=True)
    num, ids = fd_params(net.params(), loss)
    ana = np.array([grads[p].reshape(-1)[j] for p, j in ids])
    xs = x.copy(); numx = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += 1e-6; lp = loss(); x[i, j] -= 2e-6; lm = loss(); x[i, j] += 1e-6
            numx[i, j] = (lp - lm) / 2e-6
    results[f"mlp_{act}_params"] = rel_err(num, ana); results[f"mlp_{act}_input"] = rel_err(numx, dx)

# policy-gradient logit gradient: d/dz mean(-A log softmax(z)[a])
z = rng.standard_normal((6, 2)); a = rng.integers(0, 2, 6); A = rng.standard_normal(6)
f = lambda zz: float(np.mean(-A * log_softmax(zz)[np.arange(6), a]))
num = np.zeros_like(z)
for i in range(6):
    for j in range(2):
        z[i, j] += 1e-6; lp = f(z); z[i, j] -= 2e-6; lm = f(z); z[i, j] += 1e-6; num[i, j] = (lp - lm) / 2e-6
results["pg_logit_grad"] = rel_err(num, pg_logit_grad(z.copy(), a, A))

# SAC actor gradient (fixed noise: the rng is reset before every evaluation)
hp = dict(SAC_HP, hidden=[32, 32])
agent = SAC(hp, np.random.default_rng(3))
for n in (agent.actor, agent.q1, agent.q2):
    to64(n)
agent.log_alpha[0] = np.log(0.3)
s = np.random.default_rng(4).standard_normal((9, 3))


def actor_loss():
    agent.rng = np.random.default_rng(99)
    a, logp, _ = agent.sample(s)
    x = np.concatenate([s, a[:, None]], 1)
    q = np.minimum(agent.q1.forward(x)[:, 0], agent.q2.forward(x)[:, 0])
    return float(np.mean(np.exp(agent.log_alpha[0]) * logp - q))


agent.rng = np.random.default_rng(99)
grads, _, _ = agent.actor_grads(s)
num, ids = fd_params(agent.actor.params(), actor_loss, idx_per_param=6)
ana = np.array([grads[p].reshape(-1)[j] for p, j in ids])
results["sac_actor"] = rel_err(num, ana)
results["pass"] = bool(all(v < 1e-5 for k, v in results.items()))
print(json.dumps(results, indent=1))
(LOGS / "deep_gradcheck.json").write_text(json.dumps(results, indent=1))
