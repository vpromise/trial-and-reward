"""Experiment 4: value iteration and policy iteration on Rover's world (Russell & Norvig's 4x3 grid), plus the
numbers episodes 5-7 show (one backup by hand, a sampled return). Cells are written "col,row" (row 1 = bottom).
Run from the series folder: research/experiments/.venv-classic/bin/python research/experiments/scripts/exp_planning.py"""

import time

import numpy as np

from common import rl, save

W = rl.ROVER_WORLD
key = lambda s: f"{s[0]},{s[1]}"


def as_grid(V):
    """Screen order (top row first), walls null, terminals shown as their reward (as in R&N's figures)."""
    g = W.grid(V)
    return [[None if np.isnan(v) else float(v) for v in row] for row in g]


def arrows(policy):
    rows = []
    for r in range(W.height, 0, -1):
        rows.append("".join("#" if (c, r) in W.walls else ("+" if W.terminals.get((c, r), 0) > 0 else "-")
                            if W.is_terminal((c, r)) else policy[(c, r)] for c in range(1, W.width + 1)))
    return rows


def run():
    t = time.time()
    V, history = rl.value_iteration(W, gamma=1.0, theta=1e-4)
    vi_wall = time.time() - t
    sweeps = len(history) - 1
    deltas = [max(abs(b[s] - a[s]) for s in W.states) for a, b in zip(history, history[1:])]
    policies = [rl.greedy_policy(W, v) for v in history[1:]]
    optimal_from = next(i + 1 for i in range(sweeps) if all(p == rl.RN_POLICY for p in policies[i:]))
    shown = [1, 2, 3, 5, 10, sweeps]
    vi = dict(theta=1e-4, gamma=1.0, sweeps=sweeps, wall_ms=1000 * vi_wall, max_change_per_sweep=deltas,
              policy_optimal_from_sweep=optimal_from,
              after_sweep={str(k): dict(grid=as_grid(history[k]), V={key(s): history[k][s] for s in W.states},
                                        policy=arrows(policies[k - 1])) for k in shown},
              start_value_positive_from_sweep=next(k for k, v in enumerate(history) if v[(1, 1)] > 0.0),
              cell_3_1_policy_by_sweep="".join(p[(3, 1)] for p in policies))
    exact = rl.policy_evaluation(W, rl.RN_POLICY, 1.0, 1e-12)
    vi["final_vs_rn"] = {key(s): dict(ours=V[s], exact=exact[s], rn=u) for s, u in rl.RN_UTILITIES.items()}
    vi["max_abs_error_vs_exact"] = max(abs(V[s] - exact[s]) for s in W.states)

    t = time.time()
    pi, V_pi, rounds = rl.policy_iteration(W)
    pi_info = dict(start_policy="always N", evaluations=len(rounds), improvements=len(rounds) - 1,
                   policies=[arrows(p) for p in rounds], wall_ms=1000 * (time.time() - t),
                   changed_cells=[sum(a[s] != b[s] for s in a) for a, b in zip(rounds, rounds[1:] + [pi])][:-1],
                   final_equals_rn=pi == rl.RN_POLICY)

    Q = rl.q_from_v(W, exact)
    s = (3, 1)                                   # the famous cell: the long way round beats the short way
    by_hand = {}
    for a in rl.ACTIONS:
        terms = [dict(p=p, to=key(s2), r=r, v_next=0.0 if done else exact[s2]) for p, s2, r, done in W.transitions(s, a)]
        by_hand[a] = dict(terms=terms, q=Q[s][a])
    rng = np.random.default_rng(0)
    episodes = [rl.play_episode(W, rl.RN_POLICY, rng) for _ in range(3)]
    sample = [dict(cells=[key(st.s) for st in ep] + [key(ep[-1].s2)], actions=[st.a for st in ep],
                   rewards=[st.r for st in ep], return_=sum(st.r for st in ep)) for ep in episodes]
    return dict(world=dict(layout=rl.ROVER_LAYOUT, step_reward=W.step_reward, slip="0.8 ahead, 0.1 each side",
                           gamma=1.0, start=key(W.start), charger=key(rl.CHARGER), puddle=key(rl.PUDDLE),
                           rock=key(rl.ROCK)),
                value_iteration=vi, policy_iteration=pi_info,
                q_values={key(s): Q[s] for s in Q}, cell_3_1_by_hand=by_hand,
                sampled_episodes_optimal_policy=sample)


if __name__ == "__main__":
    started = time.time()
    result = run()
    save("planning", result, started)
    vi, pi = result["value_iteration"], result["policy_iteration"]
    print("VI sweeps", vi["sweeps"], "policy optimal from sweep", vi["policy_optimal_from_sweep"],
          "max err vs exact", vi["max_abs_error_vs_exact"], "| PI evaluations", pi["evaluations"], pi["changed_cells"])
    for k, v in vi["after_sweep"].items():
        print(k, v["policy"], [[None if x is None else round(x, 3) for x in row] for row in v["grid"]])
