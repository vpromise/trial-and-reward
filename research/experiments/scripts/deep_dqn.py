"""DQN on CartPole-v1 (episodes 13-14 of "Trial and Reward"), numpy, one seed per process.

Variants (everything else identical):
  dqn        replay buffer + target network + linear epsilon decay           (Mnih et al. 2015)
  no_replay  learn from the latest transition only (batch of 1, no buffer)   ablation, ep 13
  no_target  bootstrap targets from the online network itself               ablation, ep 13
  double     Double DQN target: online net picks a', target net evaluates  (van Hasselt et al. 2016), ep 14

Every EVAL_EVERY env steps the greedy policy plays 10 fixed-seed evaluation episodes; we log the network's
predicted max_a Q(s0, a) at each start state and the discounted return (gamma = 0.99) actually achieved from it.

  .venv-deep/bin/python scripts/deep_dqn.py --variant dqn --seed 0      -> logs/dqn_dqn_s0.json
"""
import argparse, time
import numpy as np
import gymnasium as gym
from deep_common import MLP, Adam, EpisodeLog, save_json, versions, LOGS, WEIGHTS

# Config "J" of the pilot sweep (logs/pilot/, seeds 1-5): Atari-DQN-like settings scaled to CartPole.
HP = dict(
    env="CartPole-v1", total_steps=500_000, hidden=[256, 256], activation="relu", gamma=0.99,
    lr=2.5e-4, adam_eps=1e-8, max_grad_norm=10.0, loss="huber(delta=1)", batch_size=128,
    buffer_size=100_000, learning_starts=1_000, train_every=4, gradient_steps=1, target_update_every=1_000,
    eps_start=1.0, eps_end=0.01, eps_decay_steps=20_000,
    eval_every=5_000, eval_episodes=10, eval_seed_base=100_000, init="torch-default uniform",
    truncation="bootstrapped (only 'terminated' stops the target; the 500-step time limit does not)",
)


def run(variant, seed, total_steps=None, overrides=None, tag=""):
    hp = dict(HP); hp.update(overrides or {}); hp["variant"] = variant
    if total_steps:
        hp["total_steps"] = total_steps
    if variant == "no_replay":   # online Q-learning: one update per env step on the newest transition
        hp.update(train_every=1, gradient_steps=1, batch_size=1, buffer_size=1)
    rng = np.random.default_rng(seed)
    env = gym.make(hp["env"]); eval_env = gym.make(hp["env"])
    q = MLP([4, *hp["hidden"], 2], hp["activation"], rng)
    qt = q.clone()
    opt = Adam(q.params(), lr=hp["lr"], eps=hp["adam_eps"], max_grad_norm=hp["max_grad_norm"])
    N = hp["buffer_size"]
    B_obs = np.zeros((N, 4), np.float32); B_next = np.zeros((N, 4), np.float32)
    B_act = np.zeros(N, np.int64); B_rew = np.zeros(N, np.float32); B_term = np.zeros(N, np.float32)
    ptr = size = 0
    gamma = hp["gamma"]
    log = EpisodeLog()
    evals, train_stats = [], []
    loss_acc, q_acc, n_acc = 0.0, 0.0, 0
    best = {"eval_return": -1, "step": 0, "net": q.clone()}
    obs, _ = env.reset(seed=seed); ep_ret = 0.0; ep_len = 0
    t0 = time.perf_counter()

    def evaluate(step):
        nonlocal best
        rets, disc, q0 = [], [], []
        for k in range(hp["eval_episodes"]):
            o, _ = eval_env.reset(seed=hp["eval_seed_base"] + k)
            qs = q.forward(o[None])[0]; q0.append(float(qs.max()))
            G = Gd = 0.0; disc_w = 1.0; done = False
            while not done:
                a = int(np.argmax(q.forward(o[None])[0]))
                o, r, term, trunc, _ = eval_env.step(a)
                G += r; Gd += disc_w * r; disc_w *= gamma; done = term or trunc
            rets.append(G); disc.append(Gd)
        e = {"step": step, "episode": len(log.returns), "return_mean": float(np.mean(rets)),
             "returns": rets, "q_s0_pred": q0, "disc_return_actual": disc,
             "q_s0_pred_mean": float(np.mean(q0)), "disc_return_actual_mean": float(np.mean(disc))}
        evals.append(e)
        if e["return_mean"] >= best["eval_return"]:
            best = {"eval_return": e["return_mean"], "step": step, "net": q.clone()}

    for t in range(1, hp["total_steps"] + 1):
        eps = max(hp["eps_end"], hp["eps_start"] - (hp["eps_start"] - hp["eps_end"]) * t / hp["eps_decay_steps"])
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            a = int(np.argmax(q.forward(obs[None])[0]))
        nobs, r, term, trunc, _ = env.step(a)
        B_obs[ptr] = obs; B_next[ptr] = nobs; B_act[ptr] = a; B_rew[ptr] = r; B_term[ptr] = float(term)
        last = ptr
        ptr = (ptr + 1) % N; size = min(size + 1, N)
        ep_ret += r; ep_len += 1; obs = nobs
        if term or trunc:
            log.add(ep_ret, ep_len, t)
            obs, _ = env.reset(); ep_ret = 0.0; ep_len = 0

        for _ in range(hp["gradient_steps"] if (t >= hp["learning_starts"] and t % hp["train_every"] == 0) else 0):
            idx = np.array([last]) if variant == "no_replay" else rng.integers(0, size, hp["batch_size"])
            s, a_b, r_b, s2, d_b = B_obs[idx], B_act[idx], B_rew[idx], B_next[idx], B_term[idx]
            if variant == "double":
                a_star = np.argmax(q.forward(s2), 1)
                v2 = qt.forward(s2)[np.arange(len(idx)), a_star]
            elif variant == "no_target":
                v2 = q.forward(s2).max(1)
            else:
                v2 = qt.forward(s2).max(1)
            y = r_b + gamma * (1.0 - d_b) * v2
            out, cache = q.forward_cache(s)
            ar = np.arange(len(idx))
            td = out[ar, a_b] - y
            g = np.zeros_like(out); g[ar, a_b] = np.clip(td, -1.0, 1.0) / len(idx)   # Huber gradient
            grads, _ = q.backward(cache, g)
            opt.step(grads)
            ab = np.abs(td)
            loss_acc += float(np.mean(np.where(ab < 1, 0.5 * td * td, ab - 0.5)))
            q_acc += float(out[ar, a_b].mean()); n_acc += 1
        if variant != "no_target" and t % hp["target_update_every"] == 0:
            qt.copy_from(q)
        if t % 1000 == 0 and n_acc:
            train_stats.append({"step": t, "loss": loss_acc / n_acc, "q_batch_mean": q_acc / n_acc, "eps": eps})
            loss_acc = q_acc = 0.0; n_acc = 0
        if t % hp["eval_every"] == 0:
            evaluate(t)

    wall = time.perf_counter() - t0
    name = f"dqn_{variant}{tag}_s{seed}"
    q.save_npz(WEIGHTS / "runs" / f"{name}_final.npz")
    best["net"].save_npz(WEIGHTS / "runs" / f"{name}_best.npz")
    out = {"method": f"dqn/{variant}", "seed": seed, "hp": hp, "wall_s": round(wall, 2),
           "env_steps": hp["total_steps"], "episodes": log.to_dict(), "evals": evals, "train_stats": train_stats,
           "best_eval": {"return_mean": best["eval_return"], "step": best["step"]}, "versions": versions()}
    save_json(LOGS / f"{name}.json", out)
    s = log.solved_at
    print(f"{name}: wall {wall:.1f}s episodes {len(log.returns)} solved {s} final avg100 {log.avg100()[-1]:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dqn", choices=["dqn", "no_replay", "no_target", "double"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--hp", default=None, help="JSON dict of hyperparameter overrides (pilots only)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    (WEIGHTS / "runs").mkdir(parents=True, exist_ok=True)
    import json
    run(a.variant, a.seed, a.steps, json.loads(a.hp) if a.hp else None, a.tag)
