"""Experiments 5-7 (+ episode 8's Monte Carlo on Rover's world): MC vs TD on the random walk (S&B Fig 6.2), the
cliff walk (S&B Example 6.6), the sparse maze (episode 11).
Run from the series folder: research/experiments/.venv-classic/bin/python research/experiments/scripts/exp_td.py"""

import time

import numpy as np

from common import curve, parallel, rl, save

key = lambda s: f"{s[0]},{s[1]}"


# ---------------------------------------------------------------------------- 5. random walk


def random_walk_run(seed, episodes=100):
    """One run: the same 100 episodes fed to every learner; RMS error after each episode (0 = before any)."""
    rng = np.random.default_rng(seed)
    data = [rl.random_walk_episode(rng) for _ in range(episodes)]
    v0 = {s: 0.5 for s in rl.RANDOM_WALK_STATES}
    out = {}
    for alpha in (0.05, 0.1, 0.15):
        _, hist = rl.td0_prediction(data, v0, alpha=alpha, record=True)
        out[f"td_alpha{alpha:g}"] = [rl.rms_error(v) for v in hist]
    for alpha in (0.01, 0.02, 0.03, 0.04):
        _, hist = rl.mc_prediction(data, v0, alpha=alpha, first_visit=False, record=True)
        out[f"mc_alpha{alpha:g}"] = [rl.rms_error(v) for v in hist]
    return out


def random_walk(runs=100):
    rows = parallel(random_walk_run, range(runs), chunksize=10)
    rms = {name: np.mean([r[name] for r in rows], 0) for name in rows[0]}
    rng = np.random.default_rng(0)
    data = [rl.random_walk_episode(rng) for _ in range(100)]
    _, hist = rl.td0_prediction(data, {s: 0.5 for s in rl.RANDOM_WALK_STATES}, alpha=0.1, record=True)
    left = {str(n): [hist[n][s] for s in rl.RANDOM_WALK_STATES] for n in (0, 1, 10, 100)}
    first = data[0]
    return dict(runs=runs, episodes=100, v0=0.5, mc="constant-alpha, every visit", rms_at_0=rms["td_alpha0.1"][0],
                rms_curves={n: dict(x=list(range(101)), y=c) for n, c in rms.items()},
                rms_at_100={n: c[-1] for n, c in rms.items()}, rms_min={n: dict(value=c.min(), episode=int(c.argmin()))
                                                                     for n, c in rms.items()},
                fig_6_2_left=dict(alpha=0.1, seed=0, values_after_episodes=left, true=list(rl.RANDOM_WALK_TRUE.values())),
                first_episode_seed0=dict(states=[st.s for st in first], moves=[st.a for st in first],
                                         reward=sum(st.r for st in first)))


def rover_mc():
    """Episode 8: first-visit Monte Carlo on Rover's world, following the optimal policy, no model."""
    W = rl.ROVER_WORLD
    exact = rl.policy_evaluation(W, rl.RN_POLICY, 1.0, 1e-12)
    rng = np.random.default_rng(0)
    episodes = [rl.play_episode(W, rl.RN_POLICY, rng) for _ in range(1000)]
    V, hist = rl.mc_prediction(episodes, {s: 0.0 for s in W.states}, record=True)
    shown = (1, 10, 100, 1000)
    start_return = [sum(st.r for st in ep) for ep in episodes]
    return dict(policy="Russell & Norvig's optimal policy", seed=0, first_visit=True,
                V_after={str(n): {key(s): hist[n][s] for s in W.states if not W.is_terminal(s)} for n in shown},
                max_abs_error_after={str(n): max(abs(hist[n][s] - exact[s]) for s in W.states if not W.is_terminal(s)
                                                 and any(st.s == s for ep in episodes[:n] for st in ep)) for n in shown},
                start_returns_first10=start_return[:10], mean_start_return=float(np.mean(start_return)),
                exact_start_value=exact[(1, 1)],
                first_episodes=[dict(cells=[key(st.s) for st in ep] + [key(ep[-1].s2)], rewards=[st.r for st in ep])
                                for ep in episodes[:3]])


# ---------------------------------------------------------------------------- 6. cliff walk


def path_kind(path):
    top = max(r for _, r in path)
    return {2: "edge (row 2)", 3: "middle (row 3)", 4: "top (row 4)"}.get(top, f"row {top}")


def cliff_run(job):
    method, seed = job
    run = getattr(rl, method)(rl.CLIFF_WORLD, 500, alpha=0.5, epsilon=0.1, gamma=1.0, seed=seed)
    path = rl.greedy_path(rl.CLIFF_WORLD, run.Q, 60)
    return method, run.rewards, run.lengths, path


def cliff(runs=500):
    methods = ("q_learning", "sarsa", "expected_sarsa")
    rows = parallel(cliff_run, [(m, s) for m in methods for s in range(runs)], chunksize=8)
    out = {}
    for m in methods:
        mine = [r for r in rows if r[0] == m]
        rewards = np.mean([r[1] for r in mine], 0)
        kinds = [path_kind(r[3]) if r[3][-1] == (12, 1) else "does not reach the goal" for r in mine]
        lengths = [len(r[3]) - 1 for r in mine if r[3][-1] == (12, 1)]
        falls = np.mean([np.sum(r[1] <= -100) for r in mine])
        example = mine[0][3]
        out[m] = dict(reward_per_episode=curve(rewards, dense_start=20),
                      avg_reward_episodes_101_500=rewards[100:].mean(), avg_reward_last100=rewards[-100:].mean(),
                      avg_reward_first10=rewards[:10].mean(),
                      smoothed_10=curve(np.convolve(rewards, np.ones(10) / 10, mode="valid"), dense_start=0, x0=10),
                      greedy_path_kinds={k: kinds.count(k) / runs for k in sorted(set(kinds))},
                      greedy_path_length_median=float(np.median(lengths)),
                      example_greedy_path=[key(s) for s in example], example_seed=0,
                      episodes_with_a_fall_per_run=falls)
    return dict(runs=runs, episodes=500, alpha=0.5, epsilon=0.1, gamma=1.0, layout=rl.CLIFF_LAYOUT,
                optimal_return=-13, safe_top_row_return=-17, methods=out)


# ---------------------------------------------------------------------------- 7. sparse maze

MAZE_METHODS = ("sparse", "shaping", "count_bonus", "shaping_negative", "count_bonus_q0", "optimistic")


def maze_run(job):
    method, seed = job
    run, visits = rl.maze_learning(method, 300, seed=seed)
    path = rl.greedy_path(rl.SPARSE_MAZE, run.Q, 200)
    first = rl.first_success(run.rewards)
    return method, run.rewards > 0, run.lengths, first, (len(path) - 1) if path[-1] == rl.MAZE_GOAL else None


def maze(runs=200):
    rows = parallel(maze_run, [(m, s) for m in MAZE_METHODS for s in range(runs)], chunksize=4)
    out = {}
    for m in MAZE_METHODS:
        mine = [r for r in rows if r[0] == m]
        firsts = [r[3] for r in mine if r[3] is not None]
        steps_before = [int(r[2][:r[3]].sum()) for r in mine if r[3] is not None]
        success = np.mean([r[1] for r in mine], 0)
        lengths = np.mean([r[2] for r in mine], 0)
        greedy = [r[4] for r in mine]
        out[m] = dict(first_success_episode=dict(median=np.median(firsts) if firsts else None,
                                                 mean_of_successful=np.mean(firsts) if firsts else None,
                                                 p10=np.percentile(firsts, 10) if firsts else None,
                                                 p90=np.percentile(firsts, 90) if firsts else None,
                                                 never_within_300=sum(r[3] is None for r in mine) / runs),
                      moves_before_first_success_median=np.median(steps_before) if steps_before else None,
                      success_rate=curve(success, dense_start=30), moves_per_episode=curve(lengths, dense_start=30),
                      success_rate_first50=success[:50].mean(), success_rate_last50=success[-50:].mean(),
                      moves_last50=lengths[-50:].mean(),
                      greedy_path_optimal_28=sum(g == 28 for g in greedy) / runs,
                      greedy_path_reaches_goal=sum(g is not None for g in greedy) / runs)
    glow = {}
    for n in (1, 3, 10, 50):
        _, visits = rl.maze_learning("count_bonus", n, seed=0)
        glow[str(n)] = {key(s): v for s, v in visits.items()}
    dist = rl.shortest_path_lengths(rl.SPARSE_MAZE, rl.MAZE_GOAL)
    return dict(runs=runs, episodes=300, max_moves=rl.MAZE_MAX_STEPS, gamma=rl.MAZE_GAMMA, alpha=0.5, epsilon=0.1,
                beta=0.01, layout=rl.MAZE_LAYOUT, shortest_path=dist[rl.SPARSE_MAZE.start],
                random_walk_success_within_200=random_walk_success(),
                methods=out, count_bonus_visits_seed0_after_episodes=glow,
                potential={key(s): rl.maze_potential(s) for s in rl.SPARSE_MAZE.states},
                distance_to_goal={key(s): d for s, d in dist.items()})


def random_walk_success(trials=20_000):
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(trials):
        s = rl.SPARSE_MAZE.start
        for _ in range(rl.MAZE_MAX_STEPS):
            s, _, done = rl.SPARSE_MAZE.step(s, rl.ACTIONS[int(rng.integers(4))], rng)
            if done:
                hits += 1
                break
    return hits / trials


if __name__ == "__main__":
    started = time.time()
    t = time.time(); rw = random_walk(); rw["wall_s"] = time.time() - t
    t = time.time(); mc = rover_mc(); mc["wall_s"] = time.time() - t
    t = time.time(); cl = cliff(); cl["wall_s"] = time.time() - t
    t = time.time(); mz = maze(); mz["wall_s"] = time.time() - t
    save("td", dict(random_walk=rw, rover_mc=mc, cliff=cl, maze=mz), started)
    print("random walk rms@100", {k: round(v, 3) for k, v in rw["rms_at_100"].items()})
    print("rover MC max err", mc["max_abs_error_after"], "mean start return", mc["mean_start_return"])
    for m, v in cl["methods"].items():
        print("cliff", m, round(v["avg_reward_episodes_101_500"], 1), round(v["avg_reward_last100"], 1), v["greedy_path_kinds"])
    for m, v in mz["methods"].items():
        print("maze", m, v["first_success_episode"], v["moves_before_first_success_median"], round(v["success_rate_last50"], 3),
              round(v["moves_last50"], 1), v["greedy_path_optimal_28"])
    print("walltimes", rw["wall_s"], mc["wall_s"], cl["wall_s"], mz["wall_s"])
