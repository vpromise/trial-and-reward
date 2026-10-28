"""Soft actor-critic (Haarnoja et al. 2018, automatic temperature) on Pendulum-v1, numpy, one seed per process.
Episode 19 of "Trial and Reward". Hyperparameters follow CleanRL's sac_continuous_action.py except
learning_starts (1,000 instead of 5,000: Pendulum is short) and the total budget.

Actor: obs -> 256 -> 256 -> [mean, log_std_raw] (ReLU); log_std = -5 + 3.5 (tanh(raw) + 1) in [-5, 2];
u ~ N(mean, std); action = 2 tanh(u)  (Pendulum torque limit is 2); log pi includes the tanh correction.
Critics: two Q(obs, action) nets 256-256 ReLU + Polyak-averaged targets; target uses min of the two.

  .venv-deep/bin/python scripts/deep_sac.py --seed 0   -> logs/sac_pendulum_s0.json
"""
import argparse, json, time
import numpy as np
import gymnasium as gym
from deep_common import MLP, Adam, EpisodeLog, save_json, versions, LOGS, WEIGHTS

HP = dict(env="Pendulum-v1", total_steps=20_000, hidden=[256, 256], activation="relu", gamma=0.99, tau=0.005,
          batch_size=256, buffer_size=1_000_000, learning_starts=1_000, policy_lr=3e-4, q_lr=1e-3, alpha_lr=1e-3,
          policy_frequency=2, target_network_frequency=1, autotune_alpha=True, target_entropy=-1.0, alpha_init=1.0,
          log_std_min=-5.0, log_std_max=2.0, action_scale=2.0, init="torch-default uniform",
          eval_every=1_000, eval_episodes=10, eval_seed_base=200_000,
          exploration="uniform random actions for the first learning_starts steps, then sampled from the policy",
          eval_policy="deterministic: action = 2 tanh(mean)")
LOG2PI = float(np.log(2 * np.pi))


class SAC:
    def __init__(self, hp, rng):
        self.hp, self.rng = hp, rng
        h = hp["hidden"]
        self.actor = MLP([3, *h, 2], "relu", rng)
        self.q1 = MLP([4, *h, 1], "relu", rng); self.q2 = MLP([4, *h, 1], "relu", rng)
        self.q1t = self.q1.clone(); self.q2t = self.q2.clone()
        self.opt_a = Adam(self.actor.params(), lr=hp["policy_lr"])
        self.opt_q = Adam(self.q1.params() + self.q2.params(), lr=hp["q_lr"])
        self.log_alpha = np.array([np.log(hp["alpha_init"])], np.float64)
        self.opt_alpha = Adam([self.log_alpha], lr=hp["alpha_lr"])
        self.scale = hp["action_scale"]; self.lmin, self.lmax = hp["log_std_min"], hp["log_std_max"]

    def _dist(self, out):
        mu, raw = out[:, 0], out[:, 1]
        th = np.tanh(raw)
        ls = self.lmin + 0.5 * (self.lmax - self.lmin) * (th + 1.0)
        return mu, raw, th, ls

    def sample(self, s, with_cache=False):
        out, cache = self.actor.forward_cache(s) if with_cache else (self.actor.forward(s), None)
        mu, raw, th, ls = self._dist(out)
        std = np.exp(ls); eps = self.rng.standard_normal(len(mu)).astype(np.float32)
        u = mu + std * eps; t = np.tanh(u); a = self.scale * t
        sq = self.scale * (1.0 - t * t)
        logp = -0.5 * eps * eps - ls - 0.5 * LOG2PI - np.log(sq + 1e-6)
        return a, logp, dict(cache=cache, mu=mu, th=th, std=std, eps=eps, t=t, sq=sq)

    def act_deterministic(self, s):
        return self.scale * np.tanh(self.actor.forward(s)[:, 0])

    def update_critic(self, s, a, r, s2, d):
        hp = self.hp; alpha = float(np.exp(self.log_alpha[0]))
        a2, logp2, _ = self.sample(s2)
        x2 = np.concatenate([s2, a2[:, None]], 1)
        qn = np.minimum(self.q1t.forward(x2)[:, 0], self.q2t.forward(x2)[:, 0]) - alpha * logp2
        y = r + hp["gamma"] * (1.0 - d) * qn
        x = np.concatenate([s, a[:, None]], 1)
        o1, c1 = self.q1.forward_cache(x); o2, c2 = self.q2.forward_cache(x)
        B = len(y)
        g1, _ = self.q1.backward(c1, (2.0 * (o1[:, 0] - y) / B)[:, None])
        g2, _ = self.q2.backward(c2, (2.0 * (o2[:, 0] - y) / B)[:, None])
        self.opt_q.step(g1 + g2)
        return float(np.mean((o1[:, 0] - y) ** 2) + np.mean((o2[:, 0] - y) ** 2)), float(o1.mean())

    def actor_grads(self, s):
        """Gradients of mean(alpha log pi(a|s) - min Q(s, a)), a = 2 tanh(mean + std eps) (reparameterised)."""
        alpha = float(np.exp(self.log_alpha[0])); B = len(s)
        a, logp, c = self.sample(s, with_cache=True)
        x = np.concatenate([s, a[:, None]], 1)
        o1, k1 = self.q1.forward_cache(x); o2, k2 = self.q2.forward_cache(x)
        use1 = (o1[:, 0] <= o2[:, 0])
        gq = np.full(B, -1.0 / B, np.float32)
        _, dx1 = self.q1.backward(k1, (gq * use1)[:, None], input_grad=True, param_grads=False)
        _, dx2 = self.q2.backward(k2, (gq * ~use1)[:, None], input_grad=True, param_grads=False)
        dL_da = dx1[:, 3] + dx2[:, 3]
        t, sq = c["t"], c["sq"]
        dcorr_du = 2.0 * t * sq / (sq + 1e-6)                 # d/du of -log(scale (1 - tanh(u)^2) + 1e-6)
        dL_du = (alpha / B) * dcorr_du + dL_da * sq           # da/du = scale (1 - tanh^2)
        dL_dmu = dL_du
        dL_dls = -(alpha / B) + dL_du * c["std"] * c["eps"]
        dL_draw = dL_dls * 0.5 * (self.lmax - self.lmin) * (1.0 - c["th"] ** 2)
        grads, _ = self.actor.backward(c["cache"], np.stack([dL_dmu, dL_draw], 1))
        loss = float(np.mean(alpha * logp - np.minimum(o1[:, 0], o2[:, 0])))
        return grads, loss, logp

    def update_actor(self, s):
        grads, loss, _ = self.actor_grads(s)
        self.opt_a.step(grads)
        if self.hp["autotune_alpha"]:
            _, logp, _ = self.sample(s)
            g = -np.exp(self.log_alpha[0]) * np.mean(logp + self.hp["target_entropy"])
            self.opt_alpha.step([np.array([g])])
        return loss


def run(seed, overrides=None, tag=""):
    hp = dict(HP); hp.update(overrides or {})
    rng = np.random.default_rng(seed)
    env = gym.make(hp["env"]); eval_env = gym.make(hp["env"])
    agent = SAC(hp, rng)
    N = hp["buffer_size"]; T = hp["total_steps"]; cap = min(N, T)
    Bs = np.zeros((cap, 3), np.float32); Bs2 = np.zeros((cap, 3), np.float32)
    Ba = np.zeros(cap, np.float32); Br = np.zeros(cap, np.float32); Bd = np.zeros(cap, np.float32)
    ptr = size = 0
    log = EpisodeLog(solve_return=None); evals, stats = [], []
    obs, _ = env.reset(seed=seed); ep_ret = 0.0; ep_len = 0
    t0 = time.perf_counter(); qloss_acc = aloss_acc = 0.0; nq = na = 0

    def evaluate(step):
        rets = []
        for k in range(hp["eval_episodes"]):
            o, _ = eval_env.reset(seed=hp["eval_seed_base"] + k); G = 0.0; done = False
            while not done:
                a = agent.act_deterministic(o[None].astype(np.float32))
                o, r, term, trunc, _ = eval_env.step(a.astype(np.float32)); G += r; done = term or trunc
            rets.append(float(G))
        evals.append({"step": step, "episode": len(log.returns), "return_mean": float(np.mean(rets)), "returns": rets,
                      "wall_s": round(time.perf_counter() - t0, 2)})

    evaluate(0)
    for t in range(1, T + 1):
        if t <= hp["learning_starts"]:
            a = float(rng.uniform(-hp["action_scale"], hp["action_scale"]))
        else:
            a = float(agent.sample(obs[None].astype(np.float32))[0][0])
        o2, r, term, trunc, _ = env.step(np.array([a], np.float32))
        i = ptr % cap
        Bs[i] = obs; Bs2[i] = o2; Ba[i] = a; Br[i] = r; Bd[i] = float(term)
        ptr += 1; size = min(size + 1, cap)
        ep_ret += r; ep_len += 1; obs = o2
        if term or trunc:
            log.add(ep_ret, ep_len, t); obs, _ = env.reset(); ep_ret = 0.0; ep_len = 0
        if t > hp["learning_starts"]:
            idx = rng.integers(0, size, hp["batch_size"])
            ql, qm = agent.update_critic(Bs[idx], Ba[idx], Br[idx], Bs2[idx], Bd[idx]); qloss_acc += ql; nq += 1
            if t % hp["policy_frequency"] == 0:
                for _ in range(hp["policy_frequency"]):   # TD3-style delayed but compensated actor updates
                    aloss_acc += agent.update_actor(Bs[idx]); na += 1
            if t % hp["target_network_frequency"] == 0:
                agent.q1t.polyak_from(agent.q1, hp["tau"]); agent.q2t.polyak_from(agent.q2, hp["tau"])
        if t % 1000 == 0 and nq:
            stats.append({"step": t, "q_loss": qloss_acc / nq, "actor_loss": aloss_acc / max(na, 1),
                          "alpha": float(np.exp(agent.log_alpha[0]))})
            qloss_acc = aloss_acc = 0.0; nq = na = 0
        if t % hp["eval_every"] == 0:
            evaluate(t)
    wall = time.perf_counter() - t0
    name = f"sac_pendulum{tag}_s{seed}"
    agent.actor.save_npz(WEIGHTS / "runs" / f"{name}_actor.npz")
    out = {"method": "sac/pendulum", "seed": seed, "hp": hp, "wall_s": round(wall, 2), "env_steps": T,
           "episodes": log.to_dict(), "evals": evals, "train_stats": stats, "versions": versions(),
           "final_eval_return_mean": evals[-1]["return_mean"],
           "last10_train_return_mean": float(np.mean(log.returns[-10:]))}
    save_json(LOGS / f"{name}.json", out)
    print(f"{name}: wall {wall:.1f}s episodes {len(log.returns)} final eval {evals[-1]['return_mean']:.1f} "
          f"last10 train {np.mean(log.returns[-10:]):.1f} alpha {np.exp(agent.log_alpha[0]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    (WEIGHTS / "runs").mkdir(parents=True, exist_ok=True)
    run(a.seed, json.loads(a.hp) if a.hp else None, a.tag)
