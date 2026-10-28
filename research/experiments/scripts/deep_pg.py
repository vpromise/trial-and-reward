"""Policy gradients on CartPole-v1 (episodes 16-17 of "Trial and Reward"), numpy, one seed per process.

Variants:
  reinforce           REINFORCE, one episode per update, advantage = return-to-go G_t (no baseline)
  reinforce_baseline  same, advantage = G_t - V(s_t), V a learned state-value net (fit to G_t)
  a2c_gae             advantage actor-critic: 8 parallel envs x 16 steps per update, GAE(lambda = 0.95)

Same policy network for all (4-64-64-2, tanh, orthogonal init, last layer gain 0.01 so it starts ~50/50).
No advantage normalisation anywhere (normalising would itself be a baseline).

  .venv-deep/bin/python scripts/deep_pg.py --variant reinforce --seed 0   -> logs/pg_reinforce_s0.json
"""
import argparse, json, time
import numpy as np
import gymnasium as gym
from deep_common import MLP, Adam, EpisodeLog, save_json, versions, log_softmax, LOGS, WEIGHTS

R2 = float(np.sqrt(2))
HP_COMMON = dict(env="CartPole-v1", gamma=0.99, hidden=[64, 64], activation="tanh",
                 init="orthogonal (gains sqrt2, sqrt2, 0.01 policy / 1.0 value), zero biases",
                 adam_eps=1e-8, advantage_normalisation=False,
                 value_scale=100.0, value_head="V(s) = 100 * net(s): the net fits targets / 100, so its tanh units "
                                               "do not saturate on returns of up to 1/(1 - gamma) = 100")
HP = {
    "reinforce": dict(HP_COMMON, total_steps=500_000, lr_policy=1e-3, episodes_per_update=1,
                      loss="mean over the episode's steps of -G_t log pi(a_t|s_t)",
                      returns="Monte Carlo return-to-go, no bootstrap at the 500-step time limit"),
    "reinforce_baseline": dict(HP_COMMON, total_steps=500_000, lr_policy=1e-3, lr_value=1e-3,
                               episodes_per_update=1, value_steps_per_update=1,
                               loss="mean over steps of -(G_t - V(s_t)) log pi; V fit by MSE to G_t",
                               returns="Monte Carlo return-to-go, no bootstrap at the 500-step time limit"),
    "a2c_gae": dict(HP_COMMON, total_steps=500_000, lr_policy=1e-3, lr_value=1e-3, n_envs=8, n_steps=16,
                    gae_lambda=0.95, entropy_coef=0.0, max_grad_norm=0.5,
                    loss="mean of -A_GAE log pi; value: mean 0.5 (V - (A + V_old))^2",
                    returns="GAE; time-limit truncation bootstraps with gamma * V(final obs)"),
}
SNAP_EVERY = 100   # episodes (only with --snapshots): policy snapshots for the gradient-variance measurement


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def make_nets(rng):
    pi = MLP([4, 64, 64, 2], "tanh", rng, init="orthogonal", gains=[R2, R2, 0.01])
    vf = MLP([4, 64, 64, 1], "tanh", rng, init="orthogonal", gains=[R2, R2, 1.0])
    return pi, vf


def returns_to_go(rews, gamma):
    G = np.zeros(len(rews), np.float32); acc = 0.0
    for k in range(len(rews) - 1, -1, -1):
        acc = rews[k] + gamma * acc; G[k] = acc
    return G


def pg_logit_grad(logits, acts, adv):
    """d/dlogits of mean_t[-adv_t log softmax(logits_t)[a_t]]."""
    p = softmax(logits); p[np.arange(len(acts)), acts] -= 1.0
    return p * (adv[:, None] / len(acts))


def run_reinforce(variant, seed, hp, snaps):
    rng = np.random.default_rng(seed)
    env = gym.make(hp["env"])
    pi, vf = make_nets(rng)
    opt_pi = Adam(pi.params(), lr=hp["lr_policy"], eps=hp["adam_eps"])
    opt_v = Adam(vf.params(), lr=hp.get("lr_value", 1e-3), eps=hp["adam_eps"])
    log = EpisodeLog(); t = 0; first = True; stats = []
    while t < hp["total_steps"]:
        if snaps is not None and len(log.returns) % SNAP_EVERY == 0:
            snaps.append((len(log.returns), t, pi.clone(), vf.clone()))
        o, _ = env.reset(seed=seed if first else None); first = False
        S, A, Rw = [], [], []
        done = False
        while not done:
            p = softmax(pi.forward(o[None])[0])
            a = int(rng.random() < p[1])
            S.append(o); A.append(a)
            o, r, term, trunc, _ = env.step(a); Rw.append(r); t += 1
            done = term or trunc
        S = np.asarray(S, np.float32); A = np.asarray(A); T = len(A)
        G = returns_to_go(Rw, hp["gamma"])
        log.add(float(np.sum(Rw)), T, t)
        if variant == "reinforce_baseline":
            vs = hp["value_scale"]
            v, vcache = vf.forward_cache(S); v = v[:, 0] * vs
            adv = G - v
            gv, _ = vf.backward(vcache, ((v - G) / vs / T)[:, None]); opt_v.step(gv)
        else:
            adv = G
        logits, cache = pi.forward_cache(S)
        grads, _ = pi.backward(cache, pg_logit_grad(logits, A, adv))
        opt_pi.step(grads)
        if len(log.returns) % 20 == 0:
            stats.append({"episode": len(log.returns), "step": t, "adv_mean": float(adv.mean()),
                          "adv_std": float(adv.std()),
                          "grad_norm": float(np.sqrt(sum(float((g * g).sum()) for g in grads)))})
    return pi, vf, log, stats


def run_a2c(seed, hp):
    rng = np.random.default_rng(seed)
    n, ns, gamma, lam = hp["n_envs"], hp["n_steps"], hp["gamma"], hp["gae_lambda"]
    envs = [gym.make(hp["env"]) for _ in range(n)]
    obs = np.stack([e.reset(seed=seed * 1000 + i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    pi, vf = make_nets(rng)
    opt_pi = Adam(pi.params(), lr=hp["lr_policy"], eps=hp["adam_eps"], max_grad_norm=hp["max_grad_norm"])
    opt_v = Adam(vf.params(), lr=hp["lr_value"], eps=hp["adam_eps"], max_grad_norm=hp["max_grad_norm"])
    log = EpisodeLog(); t = 0; stats = []; vs = hp["value_scale"]
    ep_ret = np.zeros(n); ep_len = np.zeros(n, int)
    while t < hp["total_steps"]:
        S = np.zeros((ns, n, 4), np.float32); A = np.zeros((ns, n), int)
        R = np.zeros((ns, n), np.float32); D = np.zeros((ns, n), np.float32); V = np.zeros((ns, n), np.float32)
        for k in range(ns):
            p = softmax(pi.forward(obs)); a = (rng.random(n) < p[:, 1]).astype(int)
            S[k] = obs; A[k] = a; V[k] = vs * vf.forward(obs)[:, 0]
            nxt = np.empty_like(obs)
            for i, e in enumerate(envs):
                o2, r, term, trunc, _ = e.step(int(a[i]))
                ep_ret[i] += r; ep_len[i] += 1
                rr = r
                if trunc and not term:              # time limit: the state is not terminal, bootstrap it
                    rr += gamma * vs * float(vf.forward(o2[None])[0, 0])
                R[k, i] = rr; D[k, i] = float(term or trunc)
                if term or trunc:
                    log.add(ep_ret[i], ep_len[i], t + i + 1)
                    ep_ret[i] = 0; ep_len[i] = 0
                    o2, _ = e.reset()
                nxt[i] = o2
            obs = nxt; t += n
        last_v = vs * vf.forward(obs)[:, 0]
        adv = np.zeros((ns, n), np.float32); gae = np.zeros(n, np.float32)
        for k in range(ns - 1, -1, -1):
            nv = last_v if k == ns - 1 else V[k + 1]
            nonterm = 1.0 - D[k]
            delta = R[k] + gamma * nv * nonterm - V[k]
            gae = delta + gamma * lam * nonterm * gae; adv[k] = gae
        ret = (adv + V).reshape(-1); advf = adv.reshape(-1); Sf = S.reshape(-1, 4); Af = A.reshape(-1)
        logits, cache = pi.forward_cache(Sf)
        g = pg_logit_grad(logits, Af, advf)
        if hp["entropy_coef"]:
            lp = log_softmax(logits); p = np.exp(lp); H = -(p * lp).sum(1, keepdims=True)
            g += hp["entropy_coef"] * p * (lp + H) / len(Af)
        grads, _ = pi.backward(cache, g); opt_pi.step(grads)
        v, vcache = vf.forward_cache(Sf)
        gv, _ = vf.backward(vcache, (v[:, 0] - ret / vs)[:, None] / len(ret)); opt_v.step(gv)
        v = v * vs
        if (t // (n * ns)) % 20 == 0:
            stats.append({"step": t, "adv_mean": float(advf.mean()), "adv_std": float(advf.std()),
                          "value_loss": float(0.5 * np.mean((v[:, 0] - ret) ** 2)),
                          "grad_norm": float(np.sqrt(sum(float((x * x).sum()) for x in grads)))})
    return pi, vf, log, stats


def run(variant, seed, overrides=None, tag="", snapshots=False):
    hp = dict(HP[variant]); hp.update(overrides or {}); hp["variant"] = variant
    t0 = time.perf_counter(); snaps = [] if snapshots else None
    if variant == "a2c_gae":
        pi, vf, log, stats = run_a2c(seed, hp)
    else:
        pi, vf, log, stats = run_reinforce(variant, seed, hp, snaps)
    wall = time.perf_counter() - t0
    name = f"pg_{variant}{tag}_s{seed}"
    pi.save_npz(WEIGHTS / "runs" / f"{name}_policy.npz"); vf.save_npz(WEIGHTS / "runs" / f"{name}_value.npz")
    snap_index = []
    for ep, st, p, v in (snaps or []):
        p.save_npz(WEIGHTS / "runs" / "snapshots" / f"{name}_ep{ep}_policy.npz")
        snap_index.append({"episode": ep, "step": st})
    out = {"method": f"pg/{variant}", "seed": seed, "hp": hp, "wall_s": round(wall, 2), "env_steps": hp["total_steps"],
           "episodes": log.to_dict(), "train_stats": stats, "snapshots": snap_index, "versions": versions()}
    save_json(LOGS / f"{name}.json", out)
    print(f"{name}: wall {wall:.1f}s episodes {len(log.returns)} solved {log.solved_at} "
          f"final avg100 {log.avg100()[-1]:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="reinforce", choices=list(HP))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--snapshots", action="store_true")
    a = ap.parse_args()
    (WEIGHTS / "runs" / "snapshots").mkdir(parents=True, exist_ok=True)
    run(a.variant, a.seed, json.loads(a.hp) if a.hp else None, a.tag, a.snapshots)
