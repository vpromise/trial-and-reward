"""Experiments 1-3: the 10-armed testbed (S&B Figs 2.2-2.4), Rover's five machines (regret), LinUCB on study tips.
Run from the series folder: research/experiments/.venv-classic/bin/python research/experiments/scripts/exp_bandits.py
(any Python with numpy works)."""

import math
import time

import numpy as np

from common import curve, parallel, rl, save

RUNS, STEPS = 2000, 1000

TESTBED = {                                  # name -> run_bandit / bandit_testbed parameters
    "greedy": dict(method="greedy"),
    "eps_0.01": dict(method="eps", epsilon=0.01),
    "eps_0.1": dict(method="eps", epsilon=0.1),
    "optimistic_q5": dict(method="optimistic", q0=5.0, alpha=0.1, epsilon=0.0),       # Fig 2.3
    "realistic_eps_0.1_alpha_0.1": dict(method="eps", epsilon=0.1, q0=0.0, alpha=0.1),   # Fig 2.3's comparison
    "ucb_c2": dict(method="ucb", c=2.0),                                                # Fig 2.4
}
ROVER = {
    "greedy": dict(method="greedy"),
    "eps_0.1": dict(method="eps", epsilon=0.1),
    "ucb1": dict(method="ucb", c=math.sqrt(2)),
    "thompson": dict(method="thompson"),
    "ucb_c0.5": dict(method="ucb", c=0.5),        # extra: a smaller bonus than UCB1's guarantee asks for
}


def single_runs_chunk(job):
    """The readable run_bandit, one Python loop per run, for a block of runs (cross-check of the vectorised runner)."""
    name, params, start, stop = job
    rewards, optimal = np.zeros(STEPS), np.zeros(STEPS)
    for run in range(start, stop):
        bandit = rl.GaussianTestbed(10, seed=run)
        result = rl.run_bandit(bandit, steps=STEPS, seed=100_000 + run, **params)
        rewards += result.rewards
        optimal += result.optimal
    return name, rewards, optimal


def testbed():
    out, vectorised = {}, {}
    for name, params in TESTBED.items():
        t = time.time()
        r = rl.bandit_testbed(runs=RUNS, steps=STEPS, seed=0, **params)
        vectorised[name] = r
        out[name] = dict(params=params, avg_reward=curve(r["mean_reward"]), pct_optimal=curve(100 * r["frac_optimal"]),
                         avg_reward_last100=r["mean_reward"][-100:].mean(), avg_reward_at_1000=r["mean_reward"][-1],
                         pct_optimal_last100=100 * r["frac_optimal"][-100:].mean(),
                         pct_optimal_at_1000=100 * r["frac_optimal"][-1], wall_s=time.time() - t)
    # UCB's spike at step 11 (S&B Exercise 2.8) and the optimistic method's early spike
    ucb = vectorised["ucb_c2"]["mean_reward"]
    out["ucb_c2"]["spike"] = dict(step=int(np.argmax(ucb[:30])) + 1, avg_reward=ucb[:30].max())
    opt = vectorised["optimistic_q5"]["frac_optimal"]
    out["optimistic_q5"]["spike"] = dict(step=int(np.argmax(opt[:30])) + 1, pct_optimal=100 * opt[:30].max())

    jobs = [(name, params, s, s + 125) for name, params in TESTBED.items() for s in range(0, RUNS, 125)]
    sums = {}
    for name, rewards, optimal in parallel(single_runs_chunk, jobs):
        acc = sums.setdefault(name, [np.zeros(STEPS), np.zeros(STEPS)])
        acc[0] += rewards
        acc[1] += optimal
    cross = {name: dict(avg_reward_last100=(acc[0] / RUNS)[-100:].mean(),
                        pct_optimal_last100=100 * (acc[1] / RUNS)[-100:].mean()) for name, acc in sums.items()}
    q_star = np.random.default_rng(0).normal(0, 1, (RUNS, 10))
    return dict(runs=RUNS, steps=STEPS, seed=0, methods=out, cross_check_run_bandit=cross,
                best_possible_avg_reward=dict(expected=rl.expected_max_of_noisy(10), these_2000_problems=q_star.max(1).mean()))


def rover():
    out = {}
    for name, params in ROVER.items():
        r = rl.bandit_testbed(runs=RUNS, steps=STEPS, seed=0, probs=rl.ROVER_ARMS, **params)
        out[name] = dict(params={k: v for k, v in params.items()}, regret=curve(r["regret"]),
                         pct_best=curve(100 * r["frac_optimal"]), avg_reward=curve(r["mean_reward"]),
                         regret_at=dict((str(t), r["regret"][t - 1]) for t in (10, 100, 500, 1000)),
                         pct_best_last100=100 * r["frac_optimal"][-100:].mean(),
                         avg_final_counts=r["final_counts"], most_pulled_share=r["most_pulled"])
    long_steps = 20_000
    long = {}
    for name in ("eps_0.1", "ucb1", "thompson"):
        r = rl.bandit_testbed(runs=RUNS, steps=long_steps, seed=0, probs=rl.ROVER_ARMS, **ROVER[name])
        long[name] = r["regret"]
    behind = long["ucb1"] > long["eps_0.1"]
    crossover = int(np.flatnonzero(behind)[-1]) + 2 if behind.any() and not behind[-1] else None
    long_out = dict(steps=long_steps, regret_at={name: {str(t): reg[t - 1] for t in (1000, 2000, 5000, 10_000, 20_000)}
                                                  for name, reg in long.items()},
                    ucb1_overtakes_eps_at_pull=crossover,
                    regret_curves={name: curve(reg, dense_start=10) for name, reg in long.items()})

    def one_run(name, seed):
        r = rl.run_bandit(rl.BernoulliBandit(rl.ROVER_ARMS), steps=STEPS, seed=seed, **ROVER[name])
        checkpoints = list(range(1, 21)) + [30, 50, 100, 200, 500, 1000]
        wins = np.cumsum(r.rewards[:, None] * (r.actions[:, None] == np.arange(5)), axis=0)
        return dict(seed=seed, actions=r.actions, rewards=r.rewards, regret_final=r.regret[-1],
                    pct_best=100 * r.optimal.mean(), final_counts=r.counts[-1],
                    at_pull={str(t): dict(estimates=r.estimates[t - 1], counts=r.counts[t - 1],
                                          beta_a=1 + wins[t - 1], beta_b=1 + r.counts[t - 1] - wins[t - 1])
                             for t in checkpoints})

    singles = {name: one_run(name, 0) for name in ROVER}
    stuck_seed = next(s for s in range(1000)
                      if rl.run_bandit(rl.BernoulliBandit(rl.ROVER_ARMS), "greedy", STEPS, s).counts[-1].argmax() != 3)
    singles["greedy_stuck"] = one_run("greedy", stuck_seed)
    singles["greedy_stuck"]["note"] = f"first seed from 0 where greedy's most-pulled machine is not machine 4"
    return dict(arms=rl.ROVER_ARMS, best_arm=rl.ROVER_BEST_ARM, runs=RUNS, steps=STEPS, seed=0, methods=out,
                long_horizon=long_out, single_runs=singles)


def study_tip_run(job):
    method, alpha, seed = job
    r = rl.run_study_tips(method, 2000, seed=seed, **({"alpha": alpha} if alpha else {}))
    return job, r["rewards"], r["regret"], r["actions"] == r["best"], np.bincount(r["actions"], minlength=3)


def study_tips(runs=200):
    variants = [("linucb", 1.0), ("linucb", 0.5), ("linucb", 2.0), ("ucb", None), ("best_on_average", None),
                ("oracle", None)]
    results = parallel(study_tip_run, [(m, a, s) for m, a in variants for s in range(runs)], chunksize=10)
    out = {}
    for method, alpha in variants:
        rows = [r for r in results if r[0][:2] == (method, alpha)]
        rewards = np.mean([r[1] for r in rows], 0)
        regret = np.mean([r[2] for r in rows], 0)
        best = np.mean([r[3] for r in rows], 0)
        name = method if method != "linucb" else f"linucb_alpha{alpha:g}"
        out[name] = dict(avg_reward=rewards.mean(), avg_reward_last500=rewards[-500:].mean(),
                         regret_at_2000=regret[-1], regret=curve(regret),
                         pct_best_tip_last500=100 * best[-500:].mean(),
                         tip_share=np.mean([r[4] for r in rows], 0) / 2000,
                         regret_per_run=dict(median=np.median([r[2][-1] for r in rows]),
                                             p10=np.percentile([r[2][-1] for r in rows], 10),
                                             p90=np.percentile([r[2][-1] for r in rows], 90)))
    lin_rows = sorted((r for r in results if r[0][:2] == ("linucb", 1.0)), key=lambda r: r[2][-1])
    typical_seed = lin_rows[len(lin_rows) // 2][0][2]
    theta_at = {str(n): rl.run_study_tips("linucb", n, seed=typical_seed)["theta_hat"] for n in (10, 50, 200, 2000)}
    x = np.random.default_rng(0).random((200_000, 2)) @ rl.STUDY_TIP_THETA.T
    return dict(tips=rl.STUDY_TIPS, features=rl.STUDENT_FEATURES, theta=rl.STUDY_TIP_THETA, students=2000, runs=runs,
                students_x="each feature uniform in [0, 1]", reward="1 if the tip helped, probability theta . x",
                methods=out, example=dict(method="linucb_alpha1", seed=typical_seed, theta_hat_after=theta_at,
                                          note="the median-regret run of the 200"),
                exact=dict(oracle_avg_reward=x.max(1).mean(), tip_avg_reward=x.mean(0),
                           best_tip_share=np.bincount(x.argmax(1), minlength=3) / len(x)))


if __name__ == "__main__":
    started = time.time()
    t = time.time(); tb = testbed(); tb["wall_s"] = time.time() - t
    t = time.time(); rv = rover(); rv["wall_s"] = time.time() - t
    t = time.time(); st = study_tips(); st["wall_s"] = time.time() - t
    save("bandits", dict(testbed=tb, rover_arms=rv, study_tips=st), started)
    for name, m in tb["methods"].items():
        print(f"testbed {name:30s} reward(last100) {m['avg_reward_last100']:.3f}  %opt(last100) {m['pct_optimal_last100']:.1f}"
              f"   single-run cross-check {tb['cross_check_run_bandit'][name]['avg_reward_last100']:.3f} / "
              f"{tb['cross_check_run_bandit'][name]['pct_optimal_last100']:.1f}")
    for name, m in rv["methods"].items():
        print(f"rover {name:10s} regret@1000 {m['regret_at']['1000']:.1f}  %best(last100) {m['pct_best_last100']:.1f}  most pulled {np.round(m['most_pulled_share'], 3)}")
    print("long horizon", rv["long_horizon"]["regret_at"], "UCB1 overtakes eps at", rv["long_horizon"]["ucb1_overtakes_eps_at_pull"])
    for name, m in st["methods"].items():
        print(f"tips {name:18s} reward {m['avg_reward']:.3f} last500 {m['avg_reward_last500']:.3f} regret {m['regret_at_2000']:.1f} best-tip {m['pct_best_tip_last500']:.1f}%")
