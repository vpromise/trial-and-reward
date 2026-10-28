"""How noisy is the policy gradient?  Same policy, same sampled episodes, three advantage estimators (episode 17).

For a fixed CartPole policy (snapshots of the reported REINFORCE run, seed 10: untrained, learning, good) we
  1. fit a state-value net V(s) to Monte Carlo returns of 100 separate episodes of that policy,
  2. sample M = 300 fresh episodes, and for each one compute the policy-gradient estimate
     sum_t A_t grad log pi(a_t|s_t)  (the textbook per-episode estimator; training divides by the episode length,
     which would let a baseline shift the expected gradient, so the measurement uses the plain sum) with
        A_t = G_t                      (REINFORCE, no baseline)
        A_t = G_t - V(s_t)             (REINFORCE with a learned baseline)
        A_t = GAE(lambda = 0.95) from V (actor-critic)
  3. report, across the 300 episodes, the variance of each estimate: summed over all 4,610 policy parameters,
     and for one parameter (the bias of the "push right" logit), plus the mean gradient's norm and its
     agreement (cosine) with the no-baseline mean.

  .venv-deep/bin/python scripts/deep_gradvar.py  -> logs/deep_gradvar.json
"""
import json, time
import numpy as np
import gymnasium as gym
from deep_common import MLP, Adam, LOGS, WEIGHTS
from deep_pg import softmax, returns_to_go, pg_logit_grad, R2

GAMMA, LAM, M, FIT_EPISODES = 0.99, 0.95, 300, 100
VS = 100.0          # V(s) = 100 * net(s), as in training (keeps the tanh units unsaturated)
RUN = "pg_reinforce_s10"


def load_policy(path):
    z = np.load(path); L = len([k for k in z.files if k.startswith("W")])
    net = MLP([4, 64, 64, 2], "tanh", np.random.default_rng(0), init="orthogonal", gains=[R2, R2, 0.01])
    net.W = [z[f"W{i}"] for i in range(L)]; net.b = [z[f"b{i}"] for i in range(L)]
    return net


def episode(env, pi, rng, seed=None):
    o, _ = env.reset(seed=seed); S, A, R = [], [], []; done = False; term = False
    while not done:
        p = softmax(pi.forward(o[None])[0]); a = int(rng.random() < p[1])
        S.append(o); A.append(a); o, r, term, trunc, _ = env.step(a); R.append(r); done = term or trunc
    return np.asarray(S, np.float32), np.asarray(A), np.asarray(R, np.float32), o.astype(np.float32), term


def fit_value(env, pi, rng):
    S, G = [], []
    for k in range(FIT_EPISODES):
        s, a, r, _, _ = episode(env, pi, rng, seed=900_000 + k); S.append(s); G.append(returns_to_go(r, GAMMA))
    S = np.concatenate(S); G = np.concatenate(G)
    vf = MLP([4, 64, 64, 1], "tanh", rng, init="orthogonal", gains=[R2, R2, 1.0])
    opt = Adam(vf.params(), lr=1e-3)
    for it in range(3000):
        idx = rng.integers(0, len(S), 256)
        v, c = vf.forward_cache(S[idx]); g, _ = vf.backward(c, (v[:, 0] - G[idx] / VS)[:, None] / 256); opt.step(g)
    v = VS * vf.forward(S)[:, 0]
    return vf, float(1 - np.var(G - v) / np.var(G)), float(G.mean())


def flat(grads):
    return np.concatenate([g.reshape(-1) for g in grads])


def measure(pi, label, rng):
    env = gym.make("CartPole-v1")
    vf, r2, gmean = fit_value(env, pi, rng)
    est = {"no_baseline": [], "baseline": [], "gae": []}; lens = []
    for k in range(M):
        S, A, R, last, term = episode(env, pi, rng, seed=500_000 + k); lens.append(len(A))
        G = returns_to_go(R, GAMMA); V = VS * vf.forward(S)[:, 0]
        v_next = np.append(V[1:], 0.0 if term else VS * float(vf.forward(last[None])[0, 0]))
        delta = R + GAMMA * v_next - V
        gae = np.zeros_like(delta); acc = 0.0
        for t in range(len(delta) - 1, -1, -1):
            acc = delta[t] + GAMMA * LAM * acc; gae[t] = acc
        logits, cache = pi.forward_cache(S)
        T = len(A)
        for name, adv in (("no_baseline", G), ("baseline", G - V), ("gae", gae)):
            # textbook sum over the episode's steps (pg_logit_grad averages, so scale by T); with a sum, a
            # state-only baseline provably leaves the expected gradient unchanged (a 1/T average would not)
            grads, _ = pi.backward(cache, pg_logit_grad(logits, A, adv * T)); est[name].append(flat(grads))
    nP = len(est["gae"][0])
    bias_right = nP - 1                              # last entry of the flattened params: b2[1]
    out = {"policy": label, "episodes": M, "mean_episode_length": float(np.mean(lens)),
           "value_fit_r2": r2, "mean_return_to_go": gmean, "n_params": nP, "estimators": {}}
    ref = np.mean(est["no_baseline"], 0)
    for name, X in est.items():
        X = np.asarray(X, np.float64); mu = X.mean(0); var = X.var(0, ddof=1)
        out["estimators"][name] = {
            "total_variance": float(var.sum()), "var_bias_push_right": float(var[bias_right]),
            "mean_bias_push_right": float(mu[bias_right]), "mean_grad_norm": float(np.linalg.norm(mu)),
            "mean_sq_norm_per_episode": float((X ** 2).sum(1).mean()),
            "snr": float(mu @ mu / var.sum()),
            "cosine_with_no_baseline_mean": float(mu @ ref / (np.linalg.norm(mu) * np.linalg.norm(ref) + 1e-30))}
    e = out["estimators"]
    out["variance_ratio_no_baseline_over_baseline"] = e["no_baseline"]["total_variance"] / e["baseline"]["total_variance"]
    out["variance_ratio_no_baseline_over_gae"] = e["no_baseline"]["total_variance"] / e["gae"]["total_variance"]
    out["param_variance_ratio_no_baseline_over_baseline"] = e["no_baseline"]["var_bias_push_right"] / e["baseline"]["var_bias_push_right"]
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    run = json.loads((LOGS / f"{RUN}.json").read_text())
    rets = np.array(run["episodes"]["returns"])
    snaps = []
    for s in run["snapshots"]:
        ep = s["episode"]; avg = float(rets[max(0, ep - 100):ep].mean()) if ep else None
        snaps.append((ep, s["step"], avg))
    pick = {"untrained": snaps[0]}
    for label, lo in (("learning", 100.0), ("good", 300.0)):
        cand = [s for s in snaps if s[2] is not None and s[2] >= lo]
        if cand:
            pick[label] = cand[0]
    results = {"run": RUN, "gamma": GAMMA, "gae_lambda": LAM, "stages": []}
    for label, (ep, step, avg) in pick.items():
        pi = load_policy(WEIGHTS / "runs" / "snapshots" / f"{RUN}_ep{ep}_policy.npz")
        r = measure(pi, label, np.random.default_rng(7))
        r.update(snapshot_episode=ep, snapshot_step=step, training_avg100_at_snapshot=avg)
        results["stages"].append(r)
        print(label, ep, avg, "len %.0f" % r["mean_episode_length"], "R2 %.2f" % r["value_fit_r2"],
              "var ratio nb/b %.1f nb/gae %.1f param %.1f" % (r["variance_ratio_no_baseline_over_baseline"],
                                                               r["variance_ratio_no_baseline_over_gae"],
                                                               r["param_variance_ratio_no_baseline_over_baseline"]))
    results["wall_s"] = round(time.perf_counter() - t0, 1)
    (LOGS / "deep_gradvar.json").write_text(json.dumps(results, indent=1))
