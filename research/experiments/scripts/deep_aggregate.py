"""Collect every reported deep-RL run (seeds 10-14) into research/experiments/deep_rl.json.

Stable layout:
  meta       framework choice + benchmark, versions, machine, seeds, solve criterion, commands
  headline   ~20-40 numbers the scenes assert, nested: headline[method][key]
  methods    per method: hyperparameters, per-seed results, summary statistics
  curves     <= 200 points per method; x in environment steps and in episodes
  overestimation, ppo_updates, gradvar, pendulum: the extra measurements of episodes 14, 17, 18 and 19
Spread: "std" is the sample standard deviation across seeds (ddof = 1); min/max are across seeds.
  .venv-deep/bin/python scripts/deep_aggregate.py
"""
import json, datetime
import numpy as np
from deep_common import EXP, LOGS, versions

SEEDS = [10, 11, 12, 13, 14]
NPTS = 200
GAMMA = 0.99
CEIL = round(1.0 / (1.0 - GAMMA), 6)                    # 100: no discounted CartPole return can exceed this
CEIL500 = (1.0 - GAMMA ** 500) / (1.0 - GAMMA)          # 99.34: the most a 500-step episode can actually collect


def load(prefix):
    return [json.loads((LOGS / f"{prefix}_s{s}.json").read_text()) for s in SEEDS]


def r1(x, nd=1):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r1(v, nd) for v in x]
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def stats(vals, nd=1):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {"mean": r1(v.mean(), nd), "std": r1(v.std(ddof=1) if len(v) > 1 else 0.0, nd),
            "min": r1(v.min(), nd), "max": r1(v.max(), nd), "n": int(len(v))}


def running_mean(R, w):
    c = np.cumsum(np.insert(np.asarray(R, float), 0, 0.0)); n = np.arange(1, len(R) + 1); lo = np.maximum(0, n - w)
    return (c[n] - c[lo]) / (n - lo)


def band(per_seed, nd=1):
    A = np.array(per_seed, float)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        return {"mean": r1(np.nanmean(A, 0), nd), "min": r1(np.nanmin(A, 0), nd), "max": r1(np.nanmax(A, 0), nd)}


def episode_curves(runs, total_steps):
    """Training-episode returns: avg100 (the solve criterion) and a 20-episode moving average,
    sampled on a 200-point env-step grid and on a <=200-point episode grid."""
    grid = np.linspace(0, total_steps, NPTS + 1)[1:].round().astype(int)
    a100, ma20, neps = [], [], []
    E = min(len(r["episodes"]["returns"]) for r in runs)
    xe = np.unique(np.linspace(1, E, min(NPTS, E)).round().astype(int))
    e100 = []
    for r in runs:
        R = r["episodes"]["returns"]; S = np.array(r["episodes"]["end_steps"])
        av, m20 = running_mean(R, 100), running_mean(R, 20)
        idx = np.searchsorted(S, grid, side="right") - 1
        a100.append(np.where(idx >= 0, av[np.maximum(idx, 0)], np.nan))
        ma20.append(np.where(idx >= 0, m20[np.maximum(idx, 0)], np.nan))
        neps.append(idx + 1)
        e100.append(av[xe - 1])
    return {"x_steps": grid.tolist(),
            "avg100": band(a100), "ma20": band(ma20),
            "avg100_per_seed": [r1(a) for a in a100],
            "episodes_completed_mean": r1(np.mean(neps, 0), 0),
            "x_episodes": xe.tolist(), "avg100_by_episode": band(e100),
            "avg100_by_episode_per_seed": [r1(a) for a in e100],
            "note": "avg100 = mean return of the last 100 training episodes (CartPole-v1 'solved' at >= 475); "
                    "ma20 = last 20. Training episodes use the behaviour policy (epsilon-greedy / sampled)."}


def solve_summary(runs):
    sa = [r["episodes"]["solved_at"] for r in runs]
    out = {"solved_seeds": f"{sum(s is not None for s in sa)}/{len(sa)}",
           "n_solved": int(sum(s is not None for s in sa)), "n_seeds": len(sa),
           "steps_to_solve": stats([s and s["env_steps"] for s in sa], 0),
           "episodes_to_solve": stats([s and s["episode"] for s in sa], 0),
           "wall_s_to_solve": stats([s and s["wall_s"] for s in sa], 1),
           "wall_s_total": stats([r["wall_s"] for r in runs], 1),
           "final_avg100": stats([np.mean(r["episodes"]["returns"][-100:]) for r in runs]),
           "best_avg100": stats([running_mean(r["episodes"]["returns"], 100)[99:].max()
                                 if len(r["episodes"]["returns"]) >= 100 else None for r in runs]),
           "per_seed": []}
    for s, r in zip(SEEDS, runs):
        R = r["episodes"]["returns"]; av = running_mean(R, 100); so = r["episodes"]["solved_at"]
        d = {"seed": s, "solved_at": so, "final_avg100": r1(np.mean(R[-100:])), "episodes": len(R),
             "wall_s": r["wall_s"], "env_steps": r["env_steps"]}
        if so:
            d["min_avg100_after_solve"] = r1(av[so["episode"] - 1:].min())
        out["per_seed"].append(d)
    if out["n_solved"]:
        mins = [d["min_avg100_after_solve"] for d in out["per_seed"] if "min_avg100_after_solve" in d]
        out["seeds_dropping_below_300_after_solving"] = int(sum(m < 300 for m in mins))
    return out


def flat_headline(s, extra=None):
    h = {"solved_seeds": s["solved_seeds"], "n_solved": s["n_solved"],
         "steps_to_solve_mean": s["steps_to_solve"]["mean"], "steps_to_solve_std": s["steps_to_solve"]["std"],
         "steps_to_solve_min": s["steps_to_solve"]["min"], "steps_to_solve_max": s["steps_to_solve"]["max"],
         "episodes_to_solve_mean": s["episodes_to_solve"]["mean"], "episodes_to_solve_std": s["episodes_to_solve"]["std"],
         "wall_s_to_solve_mean": s["wall_s_to_solve"]["mean"], "wall_s_total_mean": s["wall_s_total"]["mean"],
         "final_avg100_mean": s["final_avg100"]["mean"], "final_avg100_std": s["final_avg100"]["std"],
         "best_avg100_mean": s["best_avg100"]["mean"]}
    h.update(extra or {})
    return h


def overestimation(runs):
    steps = [e["step"] for e in runs[0]["evals"]]
    Q = np.array([[e["q_s0_pred_mean"] for e in r["evals"]] for r in runs])
    G = np.array([[e["disc_return_actual_mean"] for e in r["evals"]] for r in runs])
    Qall = np.array([[e["q_s0_pred"] for e in r["evals"]] for r in runs])      # seeds x evals x 10 starts
    Gall = np.array([[e["disc_return_actual"] for e in r["evals"]] for r in runs])
    late = np.array(steps) > 0.6 * runs[0]["hp"]["total_steps"]
    good = Gall >= CEIL500 - 1e-6                                              # greedy episode ran the full 500 steps
    lq = Qall[:, late, :]; lg = Gall[:, late, :]; lgood = good[:, late, :]
    return {"x_steps": steps,
            "q_s0_pred": band(Q, 2), "disc_return_actual": band(G, 2), "q_minus_actual": band(Q - G, 2),
            "q_s0_pred_per_seed": r1(Q, 2), "disc_return_actual_per_seed": r1(G, 2),
            "late_window": f"evaluations after {int(0.6 * runs[0]['hp']['total_steps'])} env steps",
            "late_q_s0_pred_mean": r1(lq.mean(), 2), "late_actual_mean": r1(lg.mean(), 2),
            "late_overestimate_mean": r1((lq - lg).mean(), 2),
            "late_overestimate_on_full_episodes_mean": r1((lq - lg)[lgood].mean(), 2) if lgood.any() else None,
            "late_fraction_full_500_step_episodes": r1(lgood.mean(), 3),
            "late_fraction_q_above_ceiling": r1((lq > CEIL).mean(), 3),
            "peak_q_s0_pred_per_seed": r1(Q.max(1), 2), "peak_q_s0_pred_mean": r1(Q.max(1).mean(), 2),
            "final_q_s0_pred_mean": r1(Q[:, -1].mean(), 2), "final_actual_mean": r1(G[:, -1].mean(), 2),
            "value_ceiling": CEIL, "achievable_in_500_steps": r1(CEIL500, 3),
            "note": "Every 5,000 env steps the greedy policy plays the same 10 start states (reset seeds 100000-100009). "
                    "q_s0_pred = the network's max_a Q(s0, a); disc_return_actual = sum_t 0.99^t r_t actually collected "
                    "from s0 (capped by the 500-step limit at 99.34). No policy can be worth more than 1/(1-0.99) = 100."}


def binned(x, ys, nb):
    """Average consecutive entries so a per-update series has <= nb points."""
    k = int(np.ceil(len(x) / nb))
    xs = [int(np.mean(x[i:i + k])) for i in range(0, len(x), k)]
    return xs, [[float(np.mean(y[i:i + k])) for i in range(0, len(x), k)] for y in ys]


def ppo_updates(runs):
    keys = ["clip_fraction", "approx_kl", "kl_exact_post"]
    x = [u["step"] for u in runs[0]["updates"]]
    out = {"x_steps": None}
    for k in keys:
        per = [[u[k] for u in r["updates"]] for r in runs]
        xs, bs = binned(x, per, NPTS)
        out["x_steps"] = xs
        out[k] = band(bs, 5)
    out["note"] = ("per PPO update (batch of 512 samples): clip_fraction = share of samples with |ratio - 1| > 0.2, "
                   "averaged over the update's minibatches (for the no-clip runs nothing is clipped, it measures how far "
                   "ratios wander); approx_kl = mean((r - 1) - log r) over minibatches; kl_exact_post = exact "
                   "KL(old || new) over the batch after the update. Consecutive updates averaged to <= 200 points.")
    return out


def ppo_summary(runs):
    s = solve_summary(runs)
    cf = [np.mean([u["clip_fraction"] for u in r["updates"]]) for r in runs]
    kl = [np.mean([u["approx_kl"] for u in r["updates"]]) for r in runs]
    kx = [np.mean([u["kl_exact_post"] for u in r["updates"]]) for r in runs]
    pre = []
    for r in runs:                                   # the learning phase only: updates before the run solved
        so = r["episodes"]["solved_at"]; lim = so["env_steps"] if so else 1e18
        pre.append(np.mean([u["clip_fraction"] for u in r["updates"] if u["step"] <= lim]))
    s.update(clip_fraction=stats(cf, 4), approx_kl=stats(kl, 5), kl_exact_post=stats(kx, 5),
             clip_fraction_before_solve=stats(pre, 4),
             max_approx_kl=stats([max(u["approx_kl"] for u in r["updates"]) for r in runs], 4))
    return s


def pendulum(runs, kind):
    steps = [e["step"] for e in runs[0]["evals"]]
    E = np.array([[e["return_mean"] for e in r["evals"]] for r in runs])
    tr = [r["episodes"]["returns"] for r in runs]; n = min(len(t) for t in tr)
    T = np.array([t[:n] for t in tr])
    if n > NPTS:
        xe, (Tb,) = binned(list(range(1, n + 1)), [T.mean(0)], NPTS)
        tb = {"mean": r1(Tb), "x_episodes": xe}
        _, Tper = binned(list(range(1, n + 1)), list(T), NPTS)
        tb["per_seed"] = r1(Tper)
    else:
        tb = {"x_episodes": list(range(1, n + 1)), **band(T), "per_seed": r1(T)}
    first = []
    for row in E:
        hit = [s for s, v in zip(steps, row) if v >= -200]
        first.append(hit[0] if hit else None)
    final = E[:, -1]
    return {"curve": {"x_steps": steps, "x_episodes": [s / 200 for s in steps], "eval_return": band(E), "eval_return_per_seed": r1(E),
                      "train_episode_return": tb,
                      "note": f"eval = deterministic policy on 10 fixed start states (reset seeds 200000-200009); {kind}"},
            "summary": {"final_eval_return": stats(final), "final_eval_return_per_seed": r1(final),
                        "best_eval_return": stats(E.max(1)), "steps_to_first_eval_ge_minus200": stats(first, 0),
                        "n_reaching_minus200": int(sum(f is not None for f in first)),
                        "last10_train_return": stats([np.mean(t[-10:]) for t in tr]),
                        "wall_s_total": stats([r["wall_s"] for r in runs], 1), "env_steps": runs[0]["env_steps"]}}


def why_text(bench):
    a, b = bench["results"]["dqn_4-64-64-2_b64"], bench["results"]["sac_3-256-256-1_b256"]
    r = lambda x, y, key: a[x][key] / a[y][key]
    return (f"For CartPole-sized nets (4-64-64-2, batch 64) a numpy training step is "
            f"{r('torch_cpu_1thread', 'numpy', 'train_step_us'):.1f}x faster than PyTorch CPU (1 thread) and "
            f"{r('mlx_cpu', 'numpy', 'train_step_us'):.1f}x faster than MLX (CPU); acting on one observation is "
            f"{r('torch_cpu_1thread', 'numpy', 'act_us'):.1f}x / {r('mlx_cpu', 'numpy', 'act_us'):.1f}x faster "
            f"(MLX on the GPU: {r('mlx_gpu', 'numpy', 'act_us'):.0f}x slower per action, and the GPU was busy with the "
            f"language-model RL job). At SAC's 256-wide nets PyTorch CPU is "
            f"{b['numpy']['train_step_us'] / b['torch_cpu_1thread']['train_step_us']:.1f}x faster per update, but a whole "
            f"SAC run is under a minute, so one framework (the same as rl_reference.py) everywhere.")


def main():
    D = {}
    names = {"dqn": "dqn_dqn", "dqn_no_replay": "dqn_no_replay", "dqn_no_target": "dqn_no_target",
             "double_dqn": "dqn_double", "reinforce": "pg_reinforce", "reinforce_baseline": "pg_reinforce_baseline",
             "a2c_gae": "pg_a2c_gae", "ppo": "ppo_cartpole_clip", "ppo_noclip": "ppo_cartpole_noclip",
             "ppo_aggressive": "ppo_cartpole_clip_aggressive", "ppo_noclip_aggressive": "ppo_cartpole_noclip_aggressive",
             "sac": "sac_pendulum", "ppo_pendulum": "ppo_pendulum_clip"}
    runs = {k: load(v) for k, v in names.items()}
    methods, curves, headline = {}, {}, {}
    for k in ["dqn", "dqn_no_replay", "dqn_no_target", "double_dqn", "reinforce", "reinforce_baseline", "a2c_gae"]:
        methods[k] = {"hp": runs[k][0]["hp"], "log_files": [f"logs/{names[k]}_s{s}.json" for s in SEEDS],
                      **solve_summary(runs[k])}
        curves[k] = episode_curves(runs[k], runs[k][0]["env_steps"])
    for k in ["ppo", "ppo_noclip", "ppo_aggressive", "ppo_noclip_aggressive"]:
        methods[k] = {"hp": runs[k][0]["hp"], "log_files": [f"logs/{names[k]}_s{s}.json" for s in SEEDS],
                      **ppo_summary(runs[k])}
        curves[k] = episode_curves(runs[k], runs[k][0]["env_steps"])
    # greedy evaluations of the DQN family (10 fixed start states every 5k steps)
    for k in ["dqn", "dqn_no_replay", "dqn_no_target", "double_dqn"]:
        E = np.array([[e["return_mean"] for e in r["evals"]] for r in runs[k]])
        methods[k]["greedy_eval_final"] = stats(E[:, -1]); methods[k]["greedy_eval_best"] = stats(E.max(1))
        curves[k]["greedy_eval"] = {"x_steps": [e["step"] for e in runs[k][0]["evals"]], **band(E)}
        qb = [[t["q_batch_mean"] for t in r["train_stats"]] for r in runs[k]]
        curves[k]["q_batch_mean"] = {"x_steps": [t["step"] for t in runs[k][0]["train_stats"]][::5],
                                     **band(np.array(qb)[:, ::5], 2)}
    over = {"dqn": overestimation(runs["dqn"]), "double_dqn": overestimation(runs["double_dqn"])}
    pu = {k: ppo_updates(runs[k]) for k in ["ppo", "ppo_noclip", "ppo_aggressive", "ppo_noclip_aggressive"]}
    pend = {"sac": pendulum(runs["sac"], "SAC: 2 tanh(mean)"), "ppo_pendulum": pendulum(runs["ppo_pendulum"], "PPO: clip(mean, -2, 2)")}
    for k in ("sac", "ppo_pendulum"):
        methods[k] = {"hp": runs[k][0]["hp"], "log_files": [f"logs/{names[k]}_s{s}.json" for s in SEEDS], **pend[k]["summary"]}
        curves[k] = pend[k]["curve"]
    gv = json.loads((LOGS / "deep_gradvar.json").read_text())
    bench = json.loads((LOGS / "deep_bench.json").read_text())

    for k in ["dqn", "dqn_no_replay", "dqn_no_target", "double_dqn", "reinforce", "reinforce_baseline", "a2c_gae"]:
        headline[k] = flat_headline(methods[k])
    for k in ["dqn", "dqn_no_replay", "dqn_no_target", "double_dqn"]:
        headline[k]["greedy_eval_final_mean"] = methods[k]["greedy_eval_final"]["mean"]
        headline[k]["total_steps"] = runs[k][0]["env_steps"]
        qf = [r["train_stats"][-1]["q_batch_mean"] for r in runs[k]]
        headline[k]["q_batch_mean_final_mean"] = float(f"{np.mean(qf):.4g}")
        headline[k]["q_batch_mean_final_max"] = float(f"{np.max(qf):.4g}")
        methods[k]["q_batch_mean_final_per_seed"] = [float(f"{q:.4g}") for q in qf]
    if "seeds_dropping_below_300_after_solving" in methods["dqn"]:
        headline["dqn"]["seeds_dropping_below_300_after_solving"] = methods["dqn"]["seeds_dropping_below_300_after_solving"]
    if "seeds_dropping_below_300_after_solving" in methods["double_dqn"]:
        headline["double_dqn"]["seeds_dropping_below_300_after_solving"] = methods["double_dqn"]["seeds_dropping_below_300_after_solving"]
    for k in ["ppo", "ppo_noclip", "ppo_aggressive", "ppo_noclip_aggressive"]:
        m = methods[k]
        headline[k] = flat_headline(m, {"clip_fraction_mean": m["clip_fraction"]["mean"],
                                        "clip_fraction_std": m["clip_fraction"]["std"],
                                        "approx_kl_mean": m["approx_kl"]["mean"], "approx_kl_std": m["approx_kl"]["std"],
                                        "kl_exact_post_mean": m["kl_exact_post"]["mean"],
                                        "max_approx_kl_mean": m["max_approx_kl"]["mean"],
                                        "total_steps": runs[k][0]["env_steps"]})
    for k in ("dqn", "double_dqn"):
        o = over[k]
        headline.setdefault("overestimation", {}).update({
            f"{k}_late_q_s0_pred_mean": o["late_q_s0_pred_mean"], f"{k}_late_actual_mean": o["late_actual_mean"],
            f"{k}_late_overestimate_mean": o["late_overestimate_mean"],
            f"{k}_late_overestimate_on_full_episodes_mean": o["late_overestimate_on_full_episodes_mean"],
            f"{k}_late_fraction_q_above_ceiling": o["late_fraction_q_above_ceiling"],
            f"{k}_peak_q_s0_pred_mean": o["peak_q_s0_pred_mean"]})
    headline["overestimation"]["value_ceiling"] = CEIL
    headline["overestimation"]["achievable_in_500_steps"] = round(CEIL500, 2)
    stage = {s["policy"]: s for s in gv["stages"]}
    ref = stage.get("learning", gv["stages"][0])
    headline["gradvar"] = {"policy_stage": ref["policy"], "policy_avg100_at_snapshot": r1(ref["training_avg100_at_snapshot"]),
                           "variance_ratio_no_baseline_over_baseline": r1(ref["variance_ratio_no_baseline_over_baseline"], 2),
                           "variance_ratio_no_baseline_over_gae": r1(ref["variance_ratio_no_baseline_over_gae"], 2),
                           "param_variance_ratio_no_baseline_over_baseline": r1(ref["param_variance_ratio_no_baseline_over_baseline"], 2),
                           "estimates_per_estimator": ref["episodes"], "episodes_per_estimate": 1,
                           "value_fit_r2": r1(ref["value_fit_r2"], 2)}
    for st in gv["stages"]:
        headline["gradvar"][f"variance_ratio_no_baseline_over_baseline_{st['policy']}"] = r1(st["variance_ratio_no_baseline_over_baseline"], 2)
        headline["gradvar"][f"variance_ratio_no_baseline_over_gae_{st['policy']}"] = r1(st["variance_ratio_no_baseline_over_gae"], 2)
    for k in ("sac", "ppo_pendulum"):
        s = methods[k]
        headline[k] = {"final_return_mean": s["final_eval_return"]["mean"], "final_return_std": s["final_eval_return"]["std"],
                       "final_return_min": s["final_eval_return"]["min"], "final_return_max": s["final_eval_return"]["max"],
                       "best_return_mean": s["best_eval_return"]["mean"],
                       "steps_to_minus200_mean": s["steps_to_first_eval_ge_minus200"]["mean"],
                       "n_reaching_minus200": s["n_reaching_minus200"],
                       "last10_train_return_mean": s["last10_train_return"]["mean"],
                       "wall_s_mean": s["wall_s_total"]["mean"], "wall_s_std": s["wall_s_total"]["std"],
                       "total_steps": s["env_steps"]}

    D["meta"] = {
        "title": "Trial and Reward - deep RL runs (Modules 4-5, episodes 12-19)",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": "Mac Studio, Apple M4 Max, 64 GB; every seed runs single-threaded in its own process, 10 at a time",
        "framework": {"choice": "numpy (hand-written MLP backprop + Adam, gradient-checked: logs/deep_gradcheck.json)",
                      "why": why_text(bench),
                      "benchmark": bench},
        "versions": versions(), "seeds": SEEDS,
        "seed_policy": "Hyperparameters were chosen in pilot runs on seeds 0-5 (logs/pilot/); every reported number "
                       "comes from fresh seeds 10-14, run once, no seed dropped.",
        "solve_criterion": "CartPole-v1 'solved' = mean return of the last 100 training episodes >= 475 "
                           "(gymnasium reward_threshold); steps_to_solve = total env steps when that first happens.",
        "gamma": GAMMA,
        "commands": ["cd research/experiments && uv venv .venv-deep --python 3.12 && VIRTUAL_ENV=.venv-deep uv pip install gymnasium numpy torch mlx",
                     ".venv-deep/bin/python scripts/deep_bench.py", ".venv-deep/bin/python scripts/deep_gradcheck.py",
                     "nohup .venv-deep/bin/python scripts/deep_run_all.py 10 > logs/deep_run_all.out &",
                     ".venv-deep/bin/python scripts/deep_gradvar.py", ".venv-deep/bin/python scripts/deep_export.py",
                     ".venv-deep/bin/python scripts/deep_aggregate.py"],
    }
    D["headline"] = headline
    D["methods"] = methods
    D["curves"] = curves
    D["overestimation"] = over
    D["ppo_updates"] = pu
    D["gradvar"] = gv
    D["pendulum"] = {k: pend[k]["summary"] for k in pend}
    D["exports"] = {"weights": ["weights/dqn_cartpole", "weights/reinforce_cartpole", "weights/ppo_cartpole",
                                "weights/sac_pendulum", "weights/random_cartpole"],
                    "trajectories": "trajectories.json"}
    (EXP / "deep_rl.json").write_text(json.dumps(D, indent=1))
    print(json.dumps(headline, indent=1))


if __name__ == "__main__":
    main()
