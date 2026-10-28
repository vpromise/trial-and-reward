"""Export trained networks for animation + one seeded episode per policy.

weights/<name>.npz : W0, b0, W1, b1, ... (layer i: x @ Wi + bi; activation between layers, none after the last)
weights/<name>.json: {"activation", "output", "obs", "action_scale", "notes", ...}
trajectories.json  : per policy, one seeded episode's per-step states and actions (<= 500 steps)

All exports come from the first reported seed (10).   .venv-deep/bin/python scripts/deep_export.py
"""
import json
import numpy as np
import gymnasium as gym
from deep_common import MLP, EXP, LOGS, WEIGHTS
from deep_pg import make_nets, softmax

SEED = 10
TRAJ_SEED = 2026          # env.reset(seed=...) and the action-sampling rng for every trajectory
CARTPOLE_OBS = "cartpole: [x, x_dot, theta, theta_dot]"
PENDULUM_OBS = "pendulum: [cos, sin, theta_dot]"


def load(path, activation):
    z = np.load(path); L = len([k for k in z.files if k.startswith("W")])
    net = MLP.__new__(MLP)
    net.W = [z[f"W{i}"].astype(np.float32) for i in range(L)]; net.b = [z[f"b{i}"].astype(np.float32) for i in range(L)]
    net.sizes = [net.W[0].shape[0]] + [w.shape[1] for w in net.W]; net.activation = activation; net.dtype = np.float32
    return net


def eval_cartpole(net, mode, n=100):
    """Mean return over n fresh episodes (reset seeds 3000..3000+n-1)."""
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(1); rets = []
    for k in range(n):
        o, _ = env.reset(seed=3000 + k); G = 0.0; done = False
        while not done:
            out = net.forward(o[None])[0]
            a = int(np.argmax(out)) if mode == "greedy" else int(rng.random() < softmax(out)[1])
            o, r, te, tr, _ = env.step(a); G += r; done = te or tr
        rets.append(G)
    return {"episodes": n, "reset_seeds": f"3000-{3000 + n - 1}", "action_mode": mode,
            "mean_return": round(float(np.mean(rets)), 1), "fraction_reaching_500": round(float(np.mean(np.array(rets) == 500)), 2)}


def write(name, net, meta):
    net.save_npz(WEIGHTS / f"{name}.npz")
    meta = dict(meta, layers=net.sizes)
    (WEIGHTS / f"{name}.json").write_text(json.dumps(meta, indent=1))


def cartpole_episode(policy, mode):
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(TRAJ_SEED)
    o, _ = env.reset(seed=TRAJ_SEED); states = [o.tolist()]; actions = []; probs = []
    done = term = trunc = False
    while not done:
        out = policy.forward(o[None])[0]
        if mode == "greedy":
            a = int(np.argmax(out))
        else:
            p = softmax(out); a = int(rng.random() < p[1]); probs.append(round(float(p[1]), 4))
        o, r, term, trunc, _ = env.step(a); states.append(o.tolist()); actions.append(a); done = term or trunc
    d = {"env": "CartPole-v1", "reset_seed": TRAJ_SEED, "action_mode": mode,
         "state_layout": "[x, x_dot, theta, theta_dot] (= the observation)", "actions_meaning": "0 = push left, 1 = push right",
         "states": [[round(v, 5) for v in s] for s in states], "actions": actions,
         "length": len(actions), "return": float(len(actions)), "terminated": bool(term), "truncated": bool(trunc),
         "note": "states[t] is the state in which actions[t] was taken; states has length + 1 entries (s_0 .. s_T)"}
    if probs:
        d["p_push_right"] = probs
    return d


def pendulum_episode(actor, scale):
    env = gym.make("Pendulum-v1"); o, _ = env.reset(seed=TRAJ_SEED)
    th = [list(map(float, env.unwrapped.state))]; obs = [o.tolist()]; acts = []; G = 0.0; done = False
    while not done:
        a = float(scale * np.tanh(actor.forward(o[None])[0, 0]))
        o, r, term, trunc, _ = env.step(np.array([a], np.float32)); G += r; done = term or trunc
        th.append(list(map(float, env.unwrapped.state))); obs.append(o.tolist()); acts.append(round(a, 5))
    return {"env": "Pendulum-v1", "reset_seed": TRAJ_SEED, "action_mode": "deterministic: torque = 2 tanh(mean)",
            "state_layout": "[theta, theta_dot]; theta in radians, 0 = upright, NOT wrapped to [-pi, pi]",
            "obs_layout": "[cos theta, sin theta, theta_dot]",
            "states": [[round(v, 5) for v in s] for s in th], "obs": [[round(v, 5) for v in s] for s in obs],
            "actions": acts, "length": len(acts), "return": round(G, 3), "terminated": False, "truncated": True,
            "note": "states[t] is the state in which actions[t] (torque, N m, in [-2, 2]) was applied; length + 1 states"}


if __name__ == "__main__":
    traj, chosen = {}, {}
    # --- DQN: final network if its greedy evaluation is still >= 475, otherwise the best evaluated snapshot
    run = json.loads((LOGS / f"dqn_dqn_s{SEED}.json").read_text())
    final_eval = run["evals"][-1]["return_mean"]
    which = "final" if final_eval >= 475 else "best"
    q = load(WEIGHTS / "runs" / f"dqn_dqn_s{SEED}_{which}.npz", "relu")
    note_step = run["hp"]["total_steps"] if which == "final" else run["best_eval"]["step"]
    write("dqn_cartpole", q, {"activation": "relu", "output": "q", "obs": CARTPOLE_OBS, "action_scale": None,
          "actions": ["push left", "push right"], "policy": "greedy: action = argmax of the two Q-values",
          "notes": f"DQN (replay + target net), seed {SEED}, {which} weights after {note_step} env steps "
                   f"(greedy eval at that point: {final_eval if which == 'final' else run['best_eval']['return_mean']} / 500). "
                   "Q-values are discounted returns with gamma = 0.99 (ceiling 1/(1-gamma) = 100)."})
    chosen["dqn_cartpole"] = {"run": f"dqn_dqn_s{SEED}", "weights": which, "env_steps": note_step,
                              "eval": eval_cartpole(q, "greedy")}
    traj["dqn_cartpole"] = cartpole_episode(q, "greedy")

    for name, runname, desc in (("reinforce_cartpole", f"pg_reinforce_s{SEED}", "REINFORCE (no baseline)"),
                                ("ppo_cartpole", f"ppo_cartpole_clip_s{SEED}", "PPO (clip 0.2)")):
        r = json.loads((LOGS / f"{runname}.json").read_text())
        pi = load(WEIGHTS / "runs" / f"{runname}_policy.npz", "tanh")
        last = float(np.mean(r["episodes"]["returns"][-100:]))
        write(name, pi, {"activation": "tanh", "output": "logits", "obs": CARTPOLE_OBS, "action_scale": None,
              "actions": ["push left", "push right"], "policy": "softmax(logits) = action probabilities; sample (or argmax)",
              "notes": f"{desc}, seed {SEED}, final weights after {r['env_steps']} env steps "
                       f"(mean return of its last 100 training episodes: {last:.1f})."})
        chosen[name] = {"run": runname, "weights": "final", "env_steps": r["env_steps"], "last100_train_mean": last,
                        "eval": eval_cartpole(pi, "sampled")}
        traj[name] = cartpole_episode(pi, "sampled")

    rnd = make_nets(np.random.default_rng(SEED))[0]
    write("random_cartpole", rnd, {"activation": "tanh", "output": "logits", "obs": CARTPOLE_OBS, "action_scale": None,
          "actions": ["push left", "push right"], "policy": "softmax(logits) = action probabilities; sample",
          "notes": f"Untrained: the policy network exactly as REINFORCE/PPO initialise it (orthogonal init, gains sqrt2, "
                   f"sqrt2, 0.01; numpy default_rng({SEED})), so its two logits are ~0 and it pushes left/right ~50/50."})
    traj["random_cartpole"] = cartpole_episode(rnd, "sampled")
    chosen["random_cartpole"] = {"run": None, "weights": "initialisation", "eval": eval_cartpole(rnd, "sampled")}

    r = json.loads((LOGS / f"sac_pendulum_s{SEED}.json").read_text())
    actor = load(WEIGHTS / "runs" / f"sac_pendulum_s{SEED}_actor.npz", "relu")
    actor.W[-1] = actor.W[-1][:, :1].copy(); actor.b[-1] = actor.b[-1][:1].copy(); actor.sizes[-1] = 1
    scale = r["hp"]["action_scale"]
    write("sac_pendulum", actor, {"activation": "relu", "output": "mean", "obs": PENDULUM_OBS, "action_scale": scale,
          "squash": "tanh", "policy": f"torque = {scale} * tanh(mean)  (deterministic; training samples u ~ N(mean, std) "
                                      f"then {scale} * tanh(u))",
          "notes": f"SAC actor, seed {SEED}, final weights after {r['env_steps']} env steps; only the mean head is exported "
                   f"(the log-std head is dropped). Deterministic eval return at the end: {r['final_eval_return_mean']:.1f}."})
    chosen["sac_pendulum"] = {"run": f"sac_pendulum_s{SEED}", "weights": "final", "env_steps": r["env_steps"]}
    traj["sac_pendulum"] = pendulum_episode(actor, scale)

    out = {"about": "One seeded episode per exported policy (weights/<name>.npz), for animation without simulating. "
                    "Gymnasium " + gym.__version__ + "; CartPole-v1 tau = 0.02 s per step, Pendulum-v1 dt = 0.05 s.",
           "reset_seed": TRAJ_SEED, "sources": chosen, "policies": traj}
    (EXP / "trajectories.json").write_text(json.dumps(out, separators=(",", ":")))
    for k, v in traj.items():
        print(f"{k}: {v['length']} steps, return {v['return']}", chosen[k].get("eval", ""))
    for k, c in chosen.items():                          # record the 100-episode evaluation in each weights json
        meta = json.loads((WEIGHTS / f"{k}.json").read_text())
        if "eval" in c:
            meta["eval_100_episodes"] = c["eval"]
        (WEIGHTS / f"{k}.json").write_text(json.dumps(meta, indent=1))
