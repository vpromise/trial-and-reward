"""Merge scripts/out/*.json into research/experiments/classic.json (with a "headline" block of the numbers scenes
assert) and write research/experiments/classic.md. Run after the exp_*.py scripts and check_gym_ports.py:
    python3 research/experiments/scripts/build_classic.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
load = lambda name: json.loads((HERE / "out" / f"{name}.json").read_text())

bandits, planning, td, approx, mcts, ports = (load(n) for n in ("bandits", "planning", "td", "approx", "mcts", "gym_ports"))
tb, rv, st = bandits["testbed"], bandits["rover_arms"], bandits["study_tips"]
vi, pi = planning["value_iteration"], planning["policy_iteration"]
rw, cl, mz, mc = td["random_walk"], td["cliff"], td["maze"], td["rover_mc"]
mb, bd, nm = approx["maximization_bias"], approx["baird"], approx["noisy_max"]
mt = mcts["matches"]["results"]
T, R, C, M = tb["methods"], rv["methods"], cl["methods"], mz["methods"]


def wdl(entry):
    return {k: entry["as_X"][k] + entry["as_O"][k] for k in ("win", "draw", "loss")}


mid = mcts["positions"]["midgame"]["by_sims"]
headline = {
    # 1. 10-armed testbed, 2,000 runs x 1,000 steps (averages over steps 901-1000)
    "testbed_greedy_pct_optimal": T["greedy"]["pct_optimal_last100"],
    "testbed_eps0.01_pct_optimal": T["eps_0.01"]["pct_optimal_last100"],
    "testbed_eps0.1_pct_optimal": T["eps_0.1"]["pct_optimal_last100"],
    "testbed_optimistic_pct_optimal": T["optimistic_q5"]["pct_optimal_last100"],
    "testbed_ucb_pct_optimal": T["ucb_c2"]["pct_optimal_last100"],
    "testbed_greedy_avg_reward": T["greedy"]["avg_reward_last100"],
    "testbed_eps0.1_avg_reward": T["eps_0.1"]["avg_reward_last100"],
    "testbed_ucb_avg_reward": T["ucb_c2"]["avg_reward_last100"],
    "testbed_best_possible_avg_reward": tb["best_possible_avg_reward"]["expected"],
    # 2. Rover's five machines, 2,000 runs x 1,000 pulls
    "rover_regret_greedy": R["greedy"]["regret_at"]["1000"],
    "rover_regret_eps0.1": R["eps_0.1"]["regret_at"]["1000"],
    "rover_regret_ucb1": R["ucb1"]["regret_at"]["1000"],
    "rover_regret_thompson": R["thompson"]["regret_at"]["1000"],
    "rover_pct_best_thompson": R["thompson"]["pct_best_last100"],
    "rover_greedy_pct_runs_stuck": 100 * (1 - R["greedy"]["most_pulled_share"][3]),
    "rover_ucb1_overtakes_eps_at_pull": rv["long_horizon"]["ucb1_overtakes_eps_at_pull"],
    # 3. study tips, 200 runs x 2,000 students: reward per student
    "tips_linucb_avg_reward": st["methods"]["linucb_alpha1"]["avg_reward"],
    "tips_ucb_avg_reward": st["methods"]["ucb"]["avg_reward"],
    "tips_best_on_average_avg_reward": st["methods"]["best_on_average"]["avg_reward"],
    "tips_oracle_avg_reward": st["methods"]["oracle"]["avg_reward"],
    # 4. Rover's world
    "vi_sweeps": vi["sweeps"],
    "vi_policy_optimal_from_sweep": vi["policy_optimal_from_sweep"],
    "pi_evaluations": pi["evaluations"],
    # 5. random walk, 100 runs: RMS error after 100 episodes
    "rw_rms_td0.1_at_100": rw["rms_at_100"]["td_alpha0.1"],
    "rw_rms_mc0.02_at_100": rw["rms_at_100"]["mc_alpha0.02"],
    # 6. cliff walk, 500 runs: reward per episode, episodes 101-500
    "cliff_q_learning_reward": C["q_learning"]["avg_reward_episodes_101_500"],
    "cliff_sarsa_reward": C["sarsa"]["avg_reward_episodes_101_500"],
    "cliff_expected_sarsa_reward": C["expected_sarsa"]["avg_reward_episodes_101_500"],
    # 7. sparse maze, 200 runs: median episode of the first success
    "maze_first_success_sparse": M["sparse"]["first_success_episode"]["median"],
    "maze_first_success_shaping": M["shaping"]["first_success_episode"]["median"],
    "maze_first_success_count_bonus": M["count_bonus"]["first_success_episode"]["median"],
    "maze_sparse_pct_never_in_300": 100 * M["sparse"]["first_success_episode"]["never_within_300"],
    # 8. overestimation and divergence
    "maxbias_q_learning_peak_pct_left": mb["methods"]["q_learning"]["peak"]["pct"],
    "maxbias_double_q_pct_left_last100": mb["methods"]["double_q"]["pct_left_last100"],
    "noisy_max_of_10": nm["exact_sigma1"]["10"],
    "baird_td_w8_after_1000": bd["methods"]["td"]["final"][7],
    # 9. MCTS
    "mcts1000_vs_minimax_draws_of_200": wdl(mt["mcts1000_vs_minimax"])["draw"],
    "mcts1000_vs_random_wins_of_200": wdl(mt["mcts1000_vs_random"])["win"],
    "mcts_midgame_perfect_move_pct_10_sims": 100 * mid["10"]["over_50_seeds"]["picks_a_perfect_move"],
    "mcts_midgame_perfect_move_pct_1000_sims": 100 * mid["1000"]["over_50_seeds"]["picks_a_perfect_move"],
    # control-env ports
    "gym_port_max_abs_diff": max(ports["cartpole"]["max_abs_state_diff"], ports["pendulum"]["max_abs_state_diff"]),
}
headline = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in headline.items()}

walls = {"exp_bandits.py": bandits["wall_s"], "exp_planning.py": planning["wall_s"], "exp_td.py": td["wall_s"],
         "exp_approx.py": approx["wall_s"], "exp_mcts.py": mcts["wall_s"], "check_gym_ports.py": ports["wall_s"]}
meta = dict(machine=bandits["machine"], wall_s=walls, gym_port_check=dict(gymnasium=ports["gymnasium"],
            numpy=ports["numpy"], python=ports["python"]),
            notes="Cells are 'col,row' with row 1 at the bottom (Russell & Norvig). Curves are {x, y} with at most "
                  "200 points. Bandit arms are numbered from 0 (machine 4 on screen = arm 3).")
classic = dict(headline=headline, meta=meta, testbed=tb, rover_arms=rv, study_tips=st, planning=planning,
               random_walk=rw, rover_mc=mc, cliff=cl, maze=mz, maximization_bias=mb, baird=bd, noisy_max=nm,
               mcts=mcts, gym_ports=ports)
for block in (planning, mcts):
    block.pop("machine", None)
(EXP / "classic.json").write_text(json.dumps(classic, indent=1))
print(f"wrote classic.json ({(EXP / 'classic.json').stat().st_size / 1e3:.0f} kB), {len(headline)} headline numbers")

# ---------------------------------------------------------------------------- classic.md

f1 = lambda x: f"{x:.1f}"
f3 = lambda x: f"{x:.3f}"
m = bandits["machine"]
lines = []
add = lines.append
add(f"""# Classic experiments for "Trial and Reward"

Every number here was computed on this Mac ({m['chip']}, {m['memory_gb']} GB, {m['cores']} cores, {m['workers']} worker
processes) with `rl_reference.py` (Python {m['python']}, NumPy {m['numpy']}). Results are in `classic.json`; scenes read
them with `measured("classic", "headline", "<key>")` or the full blocks (`measured("classic", "cliff", ...)`).
All runs are seeded with `np.random.default_rng`, so re-running gives the same numbers.

## Commands (from the series folder, `~/videos/rl-series`)

```
PY=research/experiments/.venv-classic/bin/python      # uv venv: numpy + gymnasium (gymnasium only for the port check)
$PY research/experiments/scripts/exp_bandits.py         # experiments 1-3   ({walls['exp_bandits.py']:.0f} s)
$PY research/experiments/scripts/exp_planning.py        # experiment 4      ({walls['exp_planning.py']:.1f} s)
$PY research/experiments/scripts/exp_td.py              # experiments 5-7   ({walls['exp_td.py']:.0f} s)
$PY research/experiments/scripts/exp_approx.py          # experiment 8      ({walls['exp_approx.py']:.0f} s)
$PY research/experiments/scripts/exp_mcts.py            # experiment 9      ({walls['exp_mcts.py']:.0f} s)
$PY research/experiments/scripts/check_gym_ports.py     # CartPole / Pendulum ports vs gymnasium ({walls['check_gym_ports.py']:.0f} s)
python3 research/experiments/scripts/build_classic.py   # -> classic.json + this file
python3 rl_reference.py                                 # self-checks (about 1 s)
```

Wall times are for the whole script on {m['workers']} processes; everything in this file took under a minute in total.
The venv can be recreated with `uv venv research/experiments/.venv-classic --python 3.13 && VIRTUAL_ENV=... uv pip install numpy gymnasium`;
any Python with NumPy runs everything except the port check.

## Headline numbers (`measured("classic", "headline", key)`)

| key | value |
|---|---|""")
for k, v in headline.items():
    add(f"| `{k}` | {v} |")

add(f"""
## 1. The 10-armed testbed (Sutton & Barto §2.3, Figures 2.2-2.4)

2,000 random problems (q*(a) ~ N(0, 1), rewards ~ N(q*(a), 1)), 1,000 steps each; every method sees the same 2,000
problems (`bandit_testbed(..., seed=0)`). Numbers are averages over steps 901-1000.

| method | avg reward | % optimal action | readable `run_bandit` cross-check (reward / %) |
|---|---|---|---|""")
names = {"greedy": "greedy (Q1 = 0)", "eps_0.01": "ε = 0.01", "eps_0.1": "ε = 0.1",
         "optimistic_q5": "optimistic greedy Q1 = 5, α = 0.1", "realistic_eps_0.1_alpha_0.1": "realistic ε = 0.1, Q1 = 0, α = 0.1",
         "ucb_c2": "UCB c = 2"}
for k, label in names.items():
    x = tb["cross_check_run_bandit"][k]
    add(f"| {label} | {f3(T[k]['avg_reward_last100'])} | {f1(T[k]['pct_optimal_last100'])} | {f3(x['avg_reward_last100'])} / {f1(x['pct_optimal_last100'])} |")
add(f"""
The best possible average reward is E[max of 10 N(0,1)] = {f3(tb['best_possible_avg_reward']['expected'])}
({f3(tb['best_possible_avg_reward']['these_2000_problems'])} for these 2,000 problems).

**Against the book.** Greedy finds the optimal action in about a third of the problems (S&B: "only approximately one-third")
and its reward levels off near 1.0; ε = 0.1 explores more and reaches about 80% (S&B: "never selected that action more than
91% of the time", which is the ε-greedy ceiling 0.9 + 0.1/10); ε = 0.01 is still climbing at step 1,000 (S&B: it would
eventually win). Optimistic Q1 = 5 beats realistic ε-greedy in the long run after a worse start, with its spike of
{f1(T['optimistic_q5']['spike']['pct_optimal'])}% optimal at step {T['optimistic_q5']['spike']['step']} (Fig. 2.3). UCB c = 2 beats ε = 0.1 on reward, with
the famous spike at step {T['ucb_c2']['spike']['step']} (average reward {f3(T['ucb_c2']['spike']['avg_reward'])}; S&B Exercise 2.8: at step 11 every arm has been
tried once, so all bonuses are equal and UCB pulls the arm with the best single sample, often the best arm; at step 12 that
arm's bonus has shrunk and UCB goes back to exploring). All the shapes of Figures 2.2-2.4 are reproduced.

**On screen:** "Greedy Rover finds the best machine in only a third of the problems." · "Exploring 10% of the time: the best
machine 80% of the time." · "Optimism and confidence bounds both do better: about 86%."

## 2. Rover's five machines (episodes 2-3)

`ROVER_ARMS = (0.25, 0.45, 0.30, 0.60, 0.50)` (Bernoulli; machine 4 = arm 3 is best, machine 5 at 0.50 is the runner-up).
2,000 runs × 1,000 pulls; regret = expected regret, Σ (0.60 − mean of the machine pulled).

| method | regret after 10 / 100 / 500 / 1,000 pulls | % best machine (pulls 901-1000) | share of runs whose most-pulled machine is 1-5 |
|---|---|---|---|""")
for k, label in (("greedy", "greedy"), ("eps_0.1", "ε-greedy 0.1"), ("ucb1", "UCB1 (c = √2)"), ("thompson", "Thompson, Beta(1,1)"),
                 ("ucb_c0.5", "UCB, c = 0.5 (extra)")):
    r = R[k]["regret_at"]
    add(f"| {label} | {f1(r['10'])} / {f1(r['100'])} / {f1(r['500'])} / {f1(r['1000'])} | {f1(R[k]['pct_best_last100'])} | "
        + " · ".join(f"{100 * s:.0f}%" for s in R[k]["most_pulled_share"]) + " |")
lh = rv["long_horizon"]
add(f"""
Longer horizon (2,000 runs × 20,000 pulls), regret:

| pulls | ε-greedy 0.1 | UCB1 | Thompson |
|---|---|---|---|""")
for t in ("1000", "2000", "5000", "10000", "20000"):
    add(f"| {int(t):,} | {f1(lh['regret_at']['eps_0.1'][t])} | {f1(lh['regret_at']['ucb1'][t])} | {f1(lh['regret_at']['thompson'][t])} |")
g0 = rv["single_runs"]["greedy"]
assert g0["rewards"][0] == 1 and g0["final_counts"][g0["actions"][0]] == 1000, "greedy story changed"
add(f"""
**Finding (say it carefully):** over 1,000 pulls UCB1 has *more* regret than ε-greedy ({f1(R['ucb1']['regret_at']['1000'])} vs
{f1(R['eps_0.1']['regret_at']['1000'])}). Its bonus √(2 ln t / n) keeps it checking machine 5 (0.50 vs 0.60, a gap of 0.1):
UCB1 stops favouring a worse machine only after about 2 ln t / gap² pulls of it: about 1,400 for machine 5 at t = 1,000,
more pulls than there have been.
ε-greedy's regret grows linearly (it keeps exploring at random forever), UCB1's like log t, and UCB1 overtakes it for
good at pull {lh['ucb1_overtakes_eps_at_pull']:,}. Thompson sampling is best at every horizon ({f1(R['thompson']['regret_at']['1000'])} at 1,000, {f1(lh['regret_at']['thompson']['20000'])} at 20,000).
A smaller UCB bonus (c = 0.5, no longer UCB1's guarantee) gets {f1(R['ucb_c0.5']['regret_at']['1000'])}. For episode 3: show Thompson vs
UCB1 vs ε-greedy and say "UCB1's regret curve bends over, ε-greedy's keeps climbing; Thompson's is lowest throughout".

Greedy gets stuck: only {100 * R['greedy']['most_pulled_share'][3]:.0f}% of greedy runs end up mostly on machine 4. The seed-0
single run (`single_runs.greedy`, the one to animate) pulls machine {g0['actions'][0] + 1} on its first pull, gets a coin, and
then pulls it all 1,000 times (final counts {[int(c) for c in g0['final_counts']]}): its estimate never drops to 0, the
others never rise above 0. Single seeded runs (`classic.json → rover_arms.single_runs`, seed 0 for every method) hold every
pull's arm and reward, plus estimates, counts and Beta(a, b) parameters after pulls 1-20, 30, 50, 100, 200, 500, 1,000.

## 3. Contextual bandit: study tips (episode 4)

Rover recommends one of `STUDY_TIPS = {tuple(st['tips'])}` to each of 2,000 students. A student is two numbers in [0, 1]
("likes reading", "likes talking"), drawn uniformly; the tip helps (reward 1) with probability θ_tip · x,
θ = (0.8, 0) / (0.45, 0.45) / (0, 0.8). Practice problems are best on average ({st['exact']['tip_avg_reward'][1]:.2f} vs 0.40 and
0.40), but each tip is the best one for some students ({' / '.join(f'{100 * s:.0f}%' for s in st['exact']['best_tip_share'])} of them).
200 runs.

| method | reward per student | last 500 students | regret after 2,000 | best tip given (last 500) |
|---|---|---|---|---|""")
for k, label in (("linucb_alpha1", "LinUCB, α = 1"), ("linucb_alpha0.5", "LinUCB, α = 0.5"), ("linucb_alpha2", "LinUCB, α = 2"),
                 ("ucb", "UCB1, ignores the student"), ("best_on_average", "always practice problems"), ("oracle", "oracle (knows θ)")):
    s = st["methods"][k]
    add(f"| {label} | {f3(s['avg_reward'])} | {f3(s['avg_reward_last500'])} | {f1(s['regret_at_2000'])} | {f1(s['pct_best_tip_last500'])}% |")
ex = st["example"]
add(f"""
**On screen:** "Knowing who is asking is worth {100 * (st['methods']['linucb_alpha1']['avg_reward'] / st['methods']['best_on_average']['avg_reward'] - 1):.0f}% more helped
students than the best single tip." The context-free bandit does worse than simply always giving the best-on-average
tip, because it pays to learn which tip that is. α = 0.5 under-explores: in some runs one tip is starved of data and its
line is never learned (the p90 regret is {f1(st['methods']['linucb_alpha0.5']['regret_per_run']['p90'])}). The fitted lines of a typical run
(seed {ex['seed']}, the median-regret run) after 10 / 50 / 200 / 2,000 students are in `study_tips.example.theta_hat_after`.

## 4. Value iteration on Rover's world (episodes 5-7)

Russell & Norvig's 4×3 world: start (1,1), charger +1 at (4,3), puddle −1 at (4,2), rock at (2,2), −0.04 per step,
0.8 ahead / 0.1 each side, bumping = staying, γ = 1. Synchronous sweeps, stop when no value moves more than θ = 1e-4.

- Value iteration: **{vi['sweeps']} sweeps** ({vi['wall_ms']:.1f} ms). The greedy policy is already optimal from sweep
  {vi['policy_optimal_from_sweep']}; the last cell to flip is (3,1), which goes N (the short way past the puddle) until
  sweep 11 and W (the long way round) from sweep 12. V(start) first becomes positive at sweep
  {vi['start_value_positive_from_sweep']}: the charger's news needs 5 sweeps to reach the start.
- Final values vs Russell & Norvig Fig. 17.3, every cell equal to three decimals (largest error vs the exact solution
  {vi['max_abs_error_vs_exact']:.1e}):

| | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|""")
grid = vi["after_sweep"][str(vi["sweeps"])]["grid"]
for r, row in zip((3, 2, 1), grid):
    add(f"| row {r} | " + " | ".join("rock" if v is None else (f"{v:+.0f}" if (c, r) in ((4, 3), (4, 2)) else f"{v:.3f}")
                                      for c, v in zip((1, 2, 3, 4), row)) + " |")
add(f"""
- Values after sweeps 1, 2, 3, 5, 10 and {vi['sweeps']} (grids, screen order, terminal cells show their reward) and the greedy
  policy after each are in `planning.value_iteration.after_sweep`; the max change per sweep is in `max_change_per_sweep`.
- Policy iteration from "always N": **{pi['evaluations']} evaluations, {pi['improvements']} improvements** (cells changed:
  {pi['changed_cells']}), ending in R&N's Fig. 17.2 policy.
- Episode 6's cell by hand, (3,1) with the exact values: Q(W) = {f3(planning['cell_3_1_by_hand']['W']['q'])},
  Q(N) = {f3(planning['cell_3_1_by_hand']['N']['q'])} (each term: `planning.cell_3_1_by_hand`).

## 5. MC vs TD on the random walk (Sutton & Barto Example 6.2, Figure 6.2)

States A-E, start C, +1 for leaving on the right, true values 1/6 ... 5/6, V starts at 0.5. 100 runs × 100
episodes; within a run every learner sees the same episodes. RMS error = root-mean-square over the 5 states, averaged
over runs (as in the book). MC is constant-α, every visit.

| learner | RMS after 100 episodes | lowest RMS (episode) |
|---|---|---|""")
for k in ("td_alpha0.05", "td_alpha0.1", "td_alpha0.15", "mc_alpha0.01", "mc_alpha0.02", "mc_alpha0.03", "mc_alpha0.04"):
    label = ("TD(0)" if k.startswith("td") else "MC") + ", α = " + k.split("alpha")[1]
    add(f"| {label} | {f3(rw['rms_at_100'][k])} | {f3(rw['rms_min'][k]['value'])} ({rw['rms_min'][k]['episode']}) |")
add(f"""
Starts at {f3(rw['rms_at_0'])} for everyone. **Against the book:** the same picture as Fig. 6.2 right: from the first episode on,
every TD curve is below every MC curve; large α learns fast and then plateaus higher (TD α = 0.15 bottoms out around episode
{rw['rms_min']['td_alpha0.15']['episode']} and rises slightly). Fig. 6.2 left (TD α = 0.1 estimates after 0, 1, 10, 100
episodes, seed 0) is `random_walk.fig_6_2_left`.

Episode 8 (Monte Carlo on Rover's world, optimal policy, first-visit averages): the estimate of the start cell after 1 / 10 /
100 / 1,000 episodes is {', '.join(f3(mc['V_after'][n]['1,1']) for n in ('1', '10', '100', '1000'))} (exact 0.705); the
largest error over the visited cells after 1,000 episodes is {f3(mc['max_abs_error_after']['1000'])}.

## 6. The cliff walk (Sutton & Barto Example 6.6)

4 × 12, −1 per step, −100 and back to the start for stepping into the cliff; ε = 0.1, α = 0.5, γ = 1, 500 runs ×
500 episodes.

| method | reward per episode (episodes 101-500) | last 100 | episodes with a fall (of 500) | learned greedy path |
|---|---|---|---|---|""")
for k, label in (("q_learning", "Q-learning"), ("sarsa", "SARSA"), ("expected_sarsa", "Expected SARSA")):
    c = C[k]
    kinds = ", ".join(f"{100 * v:.0f}% {p}" for p, v in sorted(c["greedy_path_kinds"].items(), key=lambda kv: -kv[1]))
    add(f"| {label} | {f1(c['avg_reward_episodes_101_500'])} | {f1(c['avg_reward_last100'])} | {f1(c['episodes_with_a_fall_per_run'])} | {kinds} |")
add(f"""
**Against the book:** S&B's figure shows SARSA near −25 and Q-learning near −50; we measure {f1(C['sarsa']['avg_reward_episodes_101_500'])} and
{f1(C['q_learning']['avg_reward_episodes_101_500'])}. Q-learning learns the optimal path (13 steps along the edge, return −13) in every run, but
because it keeps exploring with ε = 0.1 next to the edge it falls in about a quarter of its episodes; SARSA learns the
safe road (top row, 17 steps) because it values the actions it will really take, exploratory steps included. Expected SARSA does
better than both online ({f1(C['expected_sarsa']['avg_reward_episodes_101_500'])}; it takes the middle row in every run), as in S&B Fig. 6.3.
Example paths (seed 0): `cliff.methods.<method>.example_greedy_path`. Caveat: in 13% of SARSA runs the greedy path read
off the final Q-table loops instead of reaching the goal; with a constant α = 0.5 SARSA's estimates keep moving with
every exploratory step, and the ε-greedy behaviour still gets home (that is what the reward column measures).

**On screen:** "Q-learning hugs the edge and falls about 125 times in 500 episodes: about −50 per episode. SARSA takes the
safe road: about −27."

## 7. When rewards are rare: the sparse maze (episode 11)

`MAZE_LAYOUT`, 10 × 10, 59 open cells, one reward (+1) at the far corner (10,10), 0 everywhere else, γ = 0.99. The shortest
path is {mz['shortest_path']} moves; an episode ends after {mz['max_moves']} moves; a random walk finishes within 200 moves
{100 * mz['random_walk_success_within_200']:.1f}% of the time. Q-learning, α = 0.5, ε = 0.1, 200 runs × 300 episodes (`maze_learning`).

| variant | first success: median episode (10th-90th pct.) | never in 300 | moves before it (median) | success, first 50 / last 50 episodes | greedy path = 28 |
|---|---|---|---|---|---|""")
labels = {"sparse": "sparse reward only", "shaping": "+ shaping, φ = γ^(d−1) (\"value with no walls\")",
          "count_bonus": "+ novelty bonus 0.01/√n, Q starts at 1", "shaping_negative": "(fails) shaping with φ = −distance/10",
          "count_bonus_q0": "(slow) novelty bonus, Q starts at 0", "optimistic": "(extra) no bonus, Q starts at 1"}
for k, label in labels.items():
    v = M[k]
    fs = v["first_success_episode"]
    med = "never" if fs["median"] is None else f"{fs['median']:.0f} ({fs['p10']:.0f}-{fs['p90']:.0f})"
    moves = "-" if v["moves_before_first_success_median"] is None else f"{v['moves_before_first_success_median']:,.0f}"
    add(f"| {label} | {med} | {100 * fs['never_within_300']:.0f}% | {moves} | {100 * v['success_rate_first50']:.0f}% / {100 * v['success_rate_last50']:.0f}% | {100 * v['greedy_path_optimal_28']:.0f}% |")
add(f"""
**What it shows.** With the sparse reward, Q-learning is a pure random walk until it first stumbles on the charger
(median episode {M['sparse']['first_success_episode']['median']:.0f}, about {M['sparse']['moves_before_first_success_median']:,.0f} moves; {100 * M['sparse']['first_success_episode']['never_within_300']:.0f}% of runs never do in 300 episodes) — and even then it
barely improves ({100 * M['sparse']['success_rate_last50']:.1f}% success in the last 50 episodes): a one-step update moves the news back one cell
per success, and each success needs another lucky walk to the far corner. (That is the case for eligibility traces and
n-step returns.) Potential-based shaping with an optimistic potential finds the charger in episode
{M['shaping']['first_success_episode']['median']:.0f} and learns the 28-move path in every run; the novelty bonus finds it in episode
{M['count_bonus']['first_success_episode']['median']:.0f}.

**Design choices and caveats (important for the script).**
- Shaping is the same as starting Q at φ (Wiewiora, 2003). φ = −distance (the obvious choice) starts every Q *below*
  the value of standing still, and with γ < 1 bumping into a wall then earns (1 − γ)·distance/10 per step: Rover learns to
  bump into walls and never succeeds (0 of 200 runs). φ = γ^(d−1), the value a cell would have if the maze had no
  walls, is optimistic (walls only make the way longer): moving closer earns 0, every other move costs a little.
- A novelty bonus β/√n added to the reward with Q starting at 0 makes tried moves look better than untried ones (they
  earned a bonus); Rover circles where it has been until the bonus fades: first success only at episode
  {M['count_bonus_q0']['first_success_episode']['median']:.0f}, though it then knows the maze ({100 * M['count_bonus_q0']['success_rate_last50']:.0f}% success at the end). Counting
  methods assume unvisited means novel (MBIE-EB style): Q starts at β/(1 − γ) = 1, the value of a bonus collected forever.
  Optimism alone (Q starts at 1, no bonus) gets most of the way (episode {M['optimistic']['first_success_episode']['median']:.0f}).
- Before the first success all Q-values of the bonus variants scale with β, so the first-success numbers do not depend on β.
- Visit counts for the glow (count-bonus run, seed 0, after 1, 3, 10, 50 episodes): `maze.count_bonus_visits_seed0_after_episodes`;
  the potential per cell: `maze.potential`.

## 8. Overestimation and divergence (episodes 12 and 14)

**Maximization bias (S&B Example 6.7, Fig. 6.5).** A → right ends (reward 0); A → left → B, where each of 10 actions
ends with reward ~ N(−0.1, 1). ε = 0.1, α = 0.1, 10,000 runs × 300 episodes. The optimum goes left 5% of the time
(ε/2).

| | episode 1 | 5 | 10 | 20 | 50 | 100 | 300 | peak |
|---|---|---|---|---|---|---|---|---|""")
for k, label in (("q_learning", "Q-learning, % left"), ("double_q", "Double Q-learning, % left")):
    a = mb["methods"][k]["pct_left_at"]
    add(f"| {label} | " + " | ".join(f1(a[e]) for e in ("1", "5", "10", "20", "50", "100", "300"))
        + f" | {f1(mb['methods'][k]['peak']['pct'])} (ep. {mb['methods'][k]['peak']['episode']}) |")
add(f"""
Same shape as Fig. 6.5: Q-learning is drawn left far above 5% early on, Double Q-learning stays near 50% while untrained and then
falls toward 5% ({f1(mb['methods']['double_q']['pct_left_last100'])}% over episodes 201-300, still converging). Caveat: the book doesn't say how
many actions B has; we used 10 as in Zhang Shangtong's reproduction. The peak height depends on that number (with 1
action at B there is no bias).

**The max of noisy estimates (episode 14).** Ten actions all worth exactly 0, each estimated with N(0, 1) noise: the
max of the estimates averages {f3(nm['exact_sigma1']['10'])} (2 actions: {f3(nm['exact_sigma1']['2'])} = 1/√π; 100 actions:
{f3(nm['exact_sigma1']['100'])}); Monte Carlo {f3(nm['monte_carlo_n10'])}. Choosing with one estimate and scoring with an independent second
one (the Double Q trick) averages {nm['double_estimate_n10']:.3f}.

**Baird's counterexample (S&B §11.2, Fig. 11.2).** 7 states, 8 weights, all rewards 0 (so w = 0 is perfect), w0 =
(1,1,1,1,1,1,10,1), γ = 0.99, α = 0.01. Semi-gradient off-policy TD: after 1,000 steps w = ({', '.join(f'{x:.0f}' for x in bd['methods']['td']['final'])}),
|w| {bd['methods']['td']['norm']['0']:.1f} → {bd['methods']['td']['norm']['1000']:.0f}; after 10,000 steps |w| = {bd['td_norm_after_10000']:.1e}. The expected (DP)
update diverges too: w1-w6 = {bd['methods']['dp']['final'][0]:.0f}, w7 = {bd['methods']['dp']['final'][6]:.1f}, w8 = {bd['methods']['dp']['final'][7]:.0f}: w1-w6 and w8 grow
without bound while w7 (the lower state's own weight) drifts down from 10, the divergence S&B Fig. 11.2 shows for both
updates. Weight curves (200 points): `baird.methods.td.weights` and `baird.methods.dp.weights`.

## 9. Monte Carlo tree search on tic-tac-toe (episode 15)

UCT (c = √2), random rollouts, result +1 / 0 / −1 for the player who made each move, final move = most visited.
Perfect play from the empty board is a draw (minimax value 0); 5,478 legal positions.

**Visit counts, seed 0** (squares 0-8 read row by row from the top left):""")
for pos_name, label in (("empty", "empty board (X to move)"), ("midgame", "`X...O...X`, O to move: an edge (1, 3, 5, 7) draws, a corner loses")):
    p = mcts["positions"][pos_name]
    add(f"\n{label}\n\n| sims | visits per square | tree size | most visited | picks a perfect move (50 seeds) |\n|---|---|---|---|---|")
    for n in ("10", "100", "1000"):
        v = p["by_sims"][n]
        add(f"| {n} | " + " ".join(f"{sq}:{c}" for sq, c in v["seed0"]["visits"].items())
            + f" | {v['seed0']['tree_size']} | {v['seed0']['best_move']} | {100 * v['over_50_seeds']['picks_a_perfect_move']:.0f}% |")
add(f"""
From the empty board MCTS learns to prefer the centre, then the corners (every first move draws with perfect play,
but against random continuations the centre wins most). In the mid-game, 10 simulations pick a losing corner
{100 - 100 * mid['10']['over_50_seeds']['picks_a_perfect_move']:.0f}% of the time; by 1,000 every seed picks an edge.

**MCTS(n) against a random player and a perfect (minimax) player**, 100 games as X and 100 as O (win / draw / loss for MCTS):

| | vs random, as X | vs random, as O | vs minimax, as X | vs minimax, as O |
|---|---|---|---|---|""")
for n in ("10", "100", "1000"):
    row = []
    for opp in ("random", "minimax"):
        for side in ("X", "O"):
            e = mt[f"mcts{n}_vs_{opp}"][f"as_{side}"]
            row.append(f"{e['win']} / {e['draw']} / {e['loss']}")
    add(f"| MCTS({n}) | " + " | ".join(row) + " |")
add(f"""
**On screen:** "With 1,000 simulations per move, MCTS never loses to a random player and draws {wdl(mt['mcts1000_vs_minimax'])['draw']} of 200 games against
perfect play" (it lost {wdl(mt['mcts1000_vs_minimax'])['loss']}, all as O).

## Control environments: ports of gymnasium's CartPole-v1 and Pendulum-v1

`cartpole_reset/step/obs` and `pendulum_reset/step/obs` were checked against gymnasium {ports['gymnasium']} (NumPy {ports['numpy']}) step for step:
{ports['cartpole']['episodes']} CartPole episodes ({ports['cartpole']['steps']:,} steps; random pushes and a balancing rule, {ports['cartpole']['episodes_truncated_at_500']} episodes reach the
500-step limit) and {ports['pendulum']['episodes']} Pendulum episodes ({ports['pendulum']['steps']:,} steps, torques beyond ±2 to test the clip, half of them passed as
float32 arrays): the largest state difference is **{max(ports['cartpole']['max_abs_state_diff'], ports['pendulum']['max_abs_state_diff'])}** (bit-identical), observations
identical, rewards and terminated/truncated flags identical, and `cartpole_reset(np.random.default_rng(s))` equals
`env.reset(seed=s)`. Rewards: CartPole 1 per step including the last, Pendulum −(θ² + 0.1 θ̇² + 0.001 u²).

## Machine

{m['chip']}, {m['memory_gb']} GB, {m['os']}; Python {m['python']}, NumPy {m['numpy']}; {m['workers']} worker processes.
""")
(EXP / "classic.md").write_text("\n".join(lines))
print("wrote classic.md")
