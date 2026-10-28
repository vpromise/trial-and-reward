"""PPO (Schulman et al. 2017) in numpy, one seed per process: CartPole-v1 (episode 18) and Pendulum-v1 (episode 19).

Variants:
  cartpole_clip     the clipped surrogate, epsilon = 0.2
  cartpole_noclip   identical, but the plain importance-weighted surrogate ratio * A (no clip) -- what the clip prevents
  pendulum_clip     Gaussian policy (state-independent log-std), for contrast with SAC

Logged per update: the fraction of samples whose ratio left [1 - eps, 1 + eps] ("clip fraction", measured the same way
for the no-clip run, where nothing is actually clipped), the approximate KL (k3 estimator: mean((r - 1) - log r),
CleanRL/SB3 style) averaged over the update's minibatches, and the exact KL(pi_old || pi_new) on the whole batch after
the update.

  .venv-deep/bin/python scripts/deep_ppo.py --variant cartpole_clip --seed 0   -> logs/ppo_cartpole_clip_s0.json
"""
import argparse, json, time
import numpy as np
import gymnasium as gym
from deep_common import MLP, Adam, EpisodeLog, save_json, versions, log_softmax, LOGS, WEIGHTS

R2 = float(np.sqrt(2))
HP_CP = dict(env="CartPole-v1", total_steps=300_000, n_envs=8, n_steps=64, epochs=10, minibatches=4,
             lr=3e-4, adam_eps=1e-5, clip_eps=0.2, gamma=0.99, gae_lambda=0.95, entropy_coef=0.0,
             max_grad_norm=0.5, advantage_normalisation="per minibatch", value_clipping=False, lr_anneal=False,
             hidden=[64, 64], activation="tanh", init="orthogonal (sqrt2, sqrt2, 0.01 policy / 1.0 value)",
             networks="separate policy and value MLPs, separate Adam optimisers",
             value_scale=100.0, value_head="V(s) = 100 * net(s): the net fits returns / 100 (keeps tanh units unsaturated)",
             truncation="time-limit truncation bootstraps with gamma * V(final obs)")
HP = {
    "cartpole_clip": dict(HP_CP, clip=True),
    "cartpole_noclip": dict(HP_CP, clip=False),
    # the same pair, reusing each batch harder (20 epochs, lr 1e-3, as in the SB3 zoo CartPole config)
    "cartpole_clip_aggressive": dict(HP_CP, clip=True, epochs=20, lr=1e-3),
    "cartpole_noclip_aggressive": dict(HP_CP, clip=False, epochs=20, lr=1e-3),
    "pendulum_clip": dict(HP_CP, env="Pendulum-v1", total_steps=500_000, n_envs=4, n_steps=512, minibatches=32,
                          lr=1e-3, gamma=0.9, clip=True, log_std_init=0.0, action_clip=2.0,
                          eval_every=20_000, eval_episodes=10, eval_seed_base=200_000,
                          policy="Gaussian: mean from the MLP, log-std a free parameter; env gets clip(a, -2, 2)",
                          init="orthogonal (sqrt2, sqrt2, 0.01 policy-mean / 1.0 value)"),
}


def run(variant, seed, overrides=None, tag=""):
    hp = dict(HP[variant]); hp.update(overrides or {}); hp["variant"] = variant
    cont = hp["env"] == "Pendulum-v1"
    rng = np.random.default_rng(seed)
    n, ns, gamma, lam, eps_c = hp["n_envs"], hp["n_steps"], hp["gamma"], hp["gae_lambda"], hp["clip_eps"]
    envs = [gym.make(hp["env"]) for _ in range(n)]
    od = envs[0].observation_space.shape[0]; ad = 1 if cont else 2
    obs = np.stack([e.reset(seed=seed * 1000 + i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    pi = MLP([od, *hp["hidden"], ad], hp["activation"], rng, init="orthogonal", gains=[R2, R2, 0.01])
    vf = MLP([od, *hp["hidden"], 1], hp["activation"], rng, init="orthogonal", gains=[R2, R2, 1.0])
    log_std = np.full(1, hp.get("log_std_init", 0.0), np.float32)
    pparams = pi.params() + ([log_std] if cont else [])
    opt_pi = Adam(pparams, lr=hp["lr"], eps=hp["adam_eps"], max_grad_norm=hp["max_grad_norm"])
    opt_v = Adam(vf.params(), lr=hp["lr"], eps=hp["adam_eps"], max_grad_norm=hp["max_grad_norm"])
    log = EpisodeLog(solve_return=None if cont else 475.0)
    ep_ret = np.zeros(n); ep_len = np.zeros(n, int); t = 0; updates = []; evals = []; vs = hp["value_scale"]
    eval_env = gym.make(hp["env"]) if cont else None
    t0 = time.perf_counter()

    def logp_and_dist(S, A):
        out, cache = pi.forward_cache(S)
        if cont:
            std = np.exp(log_std[0]); z = (A - out[:, 0]) / std
            return -0.5 * z * z - log_std[0] - 0.5 * np.log(2 * np.pi), out, cache
        lp = log_softmax(out)
        return lp[np.arange(len(A)), A], out, cache

    def evaluate():
        rets = []
        for k in range(hp["eval_episodes"]):
            o, _ = eval_env.reset(seed=hp["eval_seed_base"] + k); G = 0.0; done = False
            while not done:
                a = np.clip(pi.forward(o[None])[0], -hp["action_clip"], hp["action_clip"])
                o, r, term, trunc, _ = eval_env.step(a); G += r; done = term or trunc
            rets.append(G)
        evals.append({"step": t, "return_mean": float(np.mean(rets)), "returns": rets})

    next_eval = hp.get("eval_every", 0)
    while t < hp["total_steps"]:
        S = np.zeros((ns, n, od), np.float32); R = np.zeros((ns, n), np.float32); D = np.zeros((ns, n), np.float32)
        V = np.zeros((ns, n), np.float32); LP = np.zeros((ns, n), np.float32)
        A = np.zeros((ns, n), np.float32 if cont else int)
        for k in range(ns):
            out = pi.forward(obs)
            if cont:
                std = np.exp(log_std[0]); a = out[:, 0] + std * rng.standard_normal(n)
                lp = -0.5 * ((a - out[:, 0]) / std) ** 2 - log_std[0] - 0.5 * np.log(2 * np.pi)
            else:
                l = log_softmax(out); a = (rng.random(n) < np.exp(l[:, 1])).astype(int); lp = l[np.arange(n), a]
            S[k] = obs; A[k] = a; LP[k] = lp; V[k] = vs * vf.forward(obs)[:, 0]
            nxt = np.empty_like(obs)
            for i, e in enumerate(envs):
                act = np.array([np.clip(a[i], -hp["action_clip"], hp["action_clip"])], np.float32) if cont else int(a[i])
                o2, r, term, trunc, _ = e.step(act)
                ep_ret[i] += r; ep_len[i] += 1; rr = r
                if trunc and not term:
                    rr += gamma * vs * float(vf.forward(o2[None].astype(np.float32))[0, 0])
                R[k, i] = rr; D[k, i] = float(term or trunc)
                if term or trunc:
                    log.add(ep_ret[i], ep_len[i], t + i + 1); ep_ret[i] = 0; ep_len[i] = 0
                    o2, _ = e.reset()
                nxt[i] = o2
            obs = nxt; t += n
        last_v = vs * vf.forward(obs)[:, 0]
        adv = np.zeros((ns, n), np.float32); gae = np.zeros(n, np.float32)
        for k in range(ns - 1, -1, -1):
            nv = last_v if k == ns - 1 else V[k + 1]; nonterm = 1.0 - D[k]
            delta = R[k] + gamma * nv * nonterm - V[k]
            gae = delta + gamma * lam * nonterm * gae; adv[k] = gae
        Bsz = ns * n
        Sf = S.reshape(Bsz, od); Af = A.reshape(Bsz); LPf = LP.reshape(Bsz); advf = adv.reshape(Bsz)
        retf = (adv + V).reshape(Bsz)
        old_out = pi.forward(Sf); old_log_std = float(log_std[0])
        mb = Bsz // hp["minibatches"]
        clipfracs, kls, ratio_dev = [], [], []
        for ep in range(hp["epochs"]):
            perm = rng.permutation(Bsz)
            for j in range(hp["minibatches"]):
                idx = perm[j * mb:(j + 1) * mb]
                s, a, lpo, ad_ = Sf[idx], Af[idx], LPf[idx], advf[idx]
                ad_ = (ad_ - ad_.mean()) / (ad_.std() + 1e-8)
                lpn, out, cache = logp_and_dist(s, a)
                logr = lpn - lpo; r = np.exp(logr)
                clipfracs.append(float(np.mean(np.abs(r - 1.0) > eps_c)))
                kls.append(float(np.mean((r - 1.0) - logr)))
                ratio_dev.append(float(np.mean(np.abs(r - 1.0))))
                if hp["clip"]:
                    active = ~(((ad_ > 0) & (r > 1 + eps_c)) | ((ad_ < 0) & (r < 1 - eps_c)))
                    dl = -(ad_ * r * active) / mb          # dLoss/d logp_new
                else:
                    dl = -(ad_ * r) / mb
                if cont:
                    std = np.exp(log_std[0]); zz = (a - out[:, 0]) / std
                    g_out = (dl * zz / std)[:, None]
                    g_logstd = np.array([np.sum(dl * (zz * zz - 1.0))], np.float32)
                    grads, _ = pi.backward(cache, g_out); grads = grads + [g_logstd]
                else:
                    p = np.exp(log_softmax(out)); oh = np.zeros_like(p); oh[np.arange(mb), a] = 1.0
                    grads, _ = pi.backward(cache, dl[:, None] * (oh - p))
                opt_pi.step(grads)
                v, vc = vf.forward_cache(s)
                gv, _ = vf.backward(vc, (v[:, 0] - retf[idx] / vs)[:, None] / mb); opt_v.step(gv)
        # exact KL(pi_old || pi_new) over the whole batch, after the update
        new_out = pi.forward(Sf)
        if cont:
            s0, s1 = np.exp(old_log_std), np.exp(log_std[0])
            kl_exact = float(np.mean(np.log(s1 / s0) + (s0 ** 2 + (old_out[:, 0] - new_out[:, 0]) ** 2) / (2 * s1 ** 2) - 0.5))
            ent = float(0.5 * np.log(2 * np.pi * np.e) + log_std[0])
        else:
            lo, ln = log_softmax(old_out), log_softmax(new_out)
            kl_exact = float(np.mean((np.exp(lo) * (lo - ln)).sum(1)))
            ent = float(np.mean(-(np.exp(ln) * ln).sum(1)))
        vpred = vs * vf.forward(Sf)[:, 0]
        ev = float(1 - np.var(retf - vpred) / (np.var(retf) + 1e-8))
        updates.append({"step": t, "episodes": len(log.returns), "clip_fraction": float(np.mean(clipfracs)),
                        "approx_kl": float(np.mean(kls)), "approx_kl_last_epoch": float(np.mean(kls[-hp["minibatches"]:])),
                        "kl_exact_post": kl_exact, "mean_abs_ratio_minus_1": float(np.mean(ratio_dev)),
                        "entropy": ent, "explained_variance": ev,
                        **({"log_std": float(log_std[0])} if cont else {})})
        if cont and t >= next_eval:
            evaluate(); next_eval += hp["eval_every"]
    if cont and (not evals or evals[-1]["step"] != t):
        evaluate()
    wall = time.perf_counter() - t0
    name = f"ppo_{variant}{tag}_s{seed}"
    pi.save_npz(WEIGHTS / "runs" / f"{name}_policy.npz"); vf.save_npz(WEIGHTS / "runs" / f"{name}_value.npz")
    if cont:
        np.save(WEIGHTS / "runs" / f"{name}_log_std.npy", log_std)
    out = {"method": f"ppo/{variant}", "seed": seed, "hp": hp, "wall_s": round(wall, 2), "env_steps": t,
           "episodes": log.to_dict(), "updates": updates, "evals": evals, "versions": versions()}
    save_json(LOGS / f"{name}.json", out)
    cf = np.mean([u["clip_fraction"] for u in updates]); kl = np.mean([u["approx_kl"] for u in updates])
    extra = f"final eval {evals[-1]['return_mean']:.1f}" if cont else f"solved {log.solved_at} final avg100 {log.avg100()[-1]:.1f}"
    print(f"{name}: wall {wall:.1f}s episodes {len(log.returns)} {extra} clipfrac {cf:.3f} approx_kl {kl:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cartpole_clip", choices=list(HP))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    (WEIGHTS / "runs").mkdir(parents=True, exist_ok=True)
    run(a.variant, a.seed, json.loads(a.hp) if a.hp else None, a.tag)
