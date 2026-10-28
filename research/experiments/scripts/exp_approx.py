"""Experiment 8: maximization bias (S&B Example 6.7, Fig 6.5), Baird's counterexample (S&B §11.2, Fig 11.2), and the
expected max of noisy estimates (episode 14).
Run from the series folder: research/experiments/.venv-classic/bin/python research/experiments/scripts/exp_approx.py"""

import time

import numpy as np

from common import curve, rl, save


def max_bias():
    t = time.time()
    b = rl.maximization_bias(runs=10_000, episodes=300, seed=0)
    out = {}
    for m, left in b.items():
        out[m] = dict(pct_left=curve(100 * left, dense_start=40),
                      pct_left_at={str(e): 100 * left[e - 1] for e in (1, 2, 5, 10, 20, 50, 100, 200, 300)},
                      peak=dict(pct=100 * left.max(), episode=int(left.argmax()) + 1),
                      pct_left_last100=100 * left[-100:].mean())
    return dict(runs=10_000, episodes=300, alpha=0.1, epsilon=0.1, gamma=1.0, b_actions=10,
                b_reward="N(-0.1, 1)", optimal_pct_left=5.0, methods=out, wall_s=time.time() - t)


def baird():
    t = time.time()
    out = {}
    for method in ("td", "dp"):
        w = rl.bairds_counterexample(1000, alpha=0.01, gamma=0.99, method=method, seed=0)
        idx = np.unique(np.linspace(0, 1000, 200).astype(int))
        out[method] = dict(steps=idx.tolist(), weights=[w[idx, i] for i in range(8)], final=w[-1],
                           max_abs_final=np.abs(w[-1]).max(), norm=dict((str(k), np.linalg.norm(w[k])) for k in (0, 100, 500, 1000)))
    w_long = rl.bairds_counterexample(10_000, method="td", seed=0)
    return dict(alpha=0.01, gamma=0.99, w0=rl.BAIRD_W0, features="v(s_i) = 2 w_i + w_8 (i = 1..6), v(s_7) = w_7 + 2 w_8",
                methods=out, td_norm_after_10000=np.linalg.norm(w_long[-1]), wall_s=time.time() - t)


def noisy_max():
    rng = np.random.default_rng(0)
    exact = {str(n): rl.expected_max_of_noisy(n) for n in (1, 2, 3, 4, 5, 8, 10, 20, 50, 100)}
    samples = rng.normal(0, 1, (200_000, 10))
    second = rng.normal(0, 1, (200_000, 10))                 # an independent second estimate of the same 10 zeros
    pick = samples.argmax(1)
    return dict(exact_sigma1=exact, monte_carlo_n10=samples.max(1).mean(),
                double_estimate_n10=second[np.arange(len(pick)), pick].mean(),
                note="10 actions all truly worth 0, each estimate = truth + N(0, 1) noise: max of the estimates averages "
                     "+1.54; scoring the chosen action with an independent second estimate (Double Q) averages 0")


if __name__ == "__main__":
    started = time.time()
    result = dict(maximization_bias=max_bias(), baird=baird(), noisy_max=noisy_max())
    save("approx", result, started)
    for m, v in result["maximization_bias"]["methods"].items():
        print(m, v["peak"], {k: round(x, 1) for k, x in v["pct_left_at"].items()}, round(v["pct_left_last100"], 2))
    for m, v in result["baird"]["methods"].items():
        print("baird", m, np.round(v["final"], 1), {k: round(x, 1) for k, x in v["norm"].items()})
    print("baird td norm @10000", result["baird"]["td_norm_after_10000"])
    print(result["noisy_max"])
