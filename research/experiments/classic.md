# Classic experiments for "Trial and Reward"

Every number here was computed on this Mac (Apple M4 Max, 64 GB, 16 cores, 14 worker
processes) with `rl_reference.py` (Python 3.13.15, NumPy 2.5.4). Results are in `classic.json`; scenes read
them with `measured("classic", "headline", "<key>")` or the full blocks (`measured("classic", "cliff", ...)`).
All runs are seeded with `np.random.default_rng`, so re-running gives the same numbers.

## Commands (from the series folder, `~/videos/rl-series`)

```
PY=research/experiments/.venv-classic/bin/python      # uv venv: numpy + gymnasium (gymnasium only for the port check)
$PY research/experiments/scripts/exp_bandits.py         # experiments 1-3   (22 s)
$PY research/experiments/scripts/exp_planning.py        # experiment 4      (0.0 s)
$PY research/experiments/scripts/exp_td.py              # experiments 5-7   (8 s)
$PY research/experiments/scripts/exp_approx.py          # experiment 8      (1 s)
$PY research/experiments/scripts/exp_mcts.py            # experiment 9      (1 s)
$PY research/experiments/scripts/check_gym_ports.py     # CartPole / Pendulum ports vs gymnasium (1 s)
python3 research/experiments/scripts/build_classic.py   # -> classic.json + this file
python3 rl_reference.py                                 # self-checks (about 1 s)
```

Wall times are for the whole script on 14 processes; everything in this file took under a minute in total.
The venv can be recreated with `uv venv research/experiments/.venv-classic --python 3.13 && VIRTUAL_ENV=... uv pip install numpy gymnasium`;
any Python with NumPy runs everything except the port check.

## Headline numbers (`measured("classic", "headline", key)`)

| key | value |
|---|---|
| `testbed_greedy_pct_optimal` | 34.55 |
| `testbed_eps0.01_pct_optimal` | 58.43 |
| `testbed_eps0.1_pct_optimal` | 80.04 |
| `testbed_optimistic_pct_optimal` | 85.59 |
| `testbed_ucb_pct_optimal` | 85.98 |
| `testbed_greedy_avg_reward` | 1.017 |
| `testbed_eps0.1_avg_reward` | 1.367 |
| `testbed_ucb_avg_reward` | 1.486 |
| `testbed_best_possible_avg_reward` | 1.539 |
| `rover_regret_greedy` | 140.7 |
| `rover_regret_eps0.1` | 46.27 |
| `rover_regret_ucb1` | 80.29 |
| `rover_regret_thompson` | 31.28 |
| `rover_pct_best_thompson` | 93.08 |
| `rover_greedy_pct_runs_stuck` | 71.6 |
| `rover_ucb1_overtakes_eps_at_pull` | 12103 |
| `tips_linucb_avg_reward` | 0.5273 |
| `tips_ucb_avg_reward` | 0.4243 |
| `tips_best_on_average_avg_reward` | 0.4492 |
| `tips_oracle_avg_reward` | 0.5398 |
| `vi_sweeps` | 22 |
| `vi_policy_optimal_from_sweep` | 12 |
| `pi_evaluations` | 5 |
| `rw_rms_td0.1_at_100` | 0.0573 |
| `rw_rms_mc0.02_at_100` | 0.0846 |
| `cliff_q_learning_reward` | -49.93 |
| `cliff_sarsa_reward` | -27.42 |
| `cliff_expected_sarsa_reward` | -21.06 |
| `maze_first_success_sparse` | 104.0 |
| `maze_first_success_shaping` | 4.0 |
| `maze_first_success_count_bonus` | 2.0 |
| `maze_sparse_pct_never_in_300` | 21.0 |
| `maxbias_q_learning_peak_pct_left` | 94.26 |
| `maxbias_double_q_pct_left_last100` | 7.08 |
| `noisy_max_of_10` | 1.539 |
| `baird_td_w8_after_1000` | 316.7 |
| `mcts1000_vs_minimax_draws_of_200` | 197 |
| `mcts1000_vs_random_wins_of_200` | 191 |
| `mcts_midgame_perfect_move_pct_10_sims` | 60.0 |
| `mcts_midgame_perfect_move_pct_1000_sims` | 100.0 |
| `gym_port_max_abs_diff` | 0.0 |

## 1. The 10-armed testbed (Sutton & Barto §2.3, Figures 2.2-2.4)

2,000 random problems (q*(a) ~ N(0, 1), rewards ~ N(q*(a), 1)), 1,000 steps each; every method sees the same 2,000
problems (`bandit_testbed(..., seed=0)`). Numbers are averages over steps 901-1000.

| method | avg reward | % optimal action | readable `run_bandit` cross-check (reward / %) |
|---|---|---|---|
| greedy (Q1 = 0) | 1.017 | 34.5 | 1.029 / 36.4 |
| ε = 0.01 | 1.292 | 58.4 | 1.316 / 60.2 |
| ε = 0.1 | 1.367 | 80.0 | 1.359 / 79.5 |
| optimistic greedy Q1 = 5, α = 0.1 | 1.496 | 85.6 | 1.495 / 85.8 |
| realistic ε = 0.1, Q1 = 0, α = 0.1 | 1.350 | 76.2 | 1.339 / 75.5 |
| UCB c = 2 | 1.486 | 86.0 | 1.479 / 85.5 |

The best possible average reward is E[max of 10 N(0,1)] = 1.539
(1.535 for these 2,000 problems).

**Against the book.** Greedy finds the optimal action in about a third of the problems (S&B: "only approximately one-third")
and its reward levels off near 1.0; ε = 0.1 explores more and reaches about 80% (S&B: "never selected that action more than
91% of the time", which is the ε-greedy ceiling 0.9 + 0.1/10); ε = 0.01 is still climbing at step 1,000 (S&B: it would
eventually win). Optimistic Q1 = 5 beats realistic ε-greedy in the long run after a worse start, with its spike of
43.9% optimal at step 11 (Fig. 2.3). UCB c = 2 beats ε = 0.1 on reward, with
the famous spike at step 11 (average reward 1.085; S&B Exercise 2.8: at step 11 every arm has been
tried once, so all bonuses are equal and UCB pulls the arm with the best single sample, often the best arm; at step 12 that
arm's bonus has shrunk and UCB goes back to exploring). All the shapes of Figures 2.2-2.4 are reproduced.

**On screen:** "Greedy Rover finds the best machine in only a third of the problems." · "Exploring 10% of the time: the best
machine 80% of the time." · "Optimism and confidence bounds both do better: about 86%."

## 2. Rover's five machines (episodes 2-3)

`ROVER_ARMS = (0.25, 0.45, 0.30, 0.60, 0.50)` (Bernoulli; machine 4 = arm 3 is best, machine 5 at 0.50 is the runner-up).
2,000 runs × 1,000 pulls; regret = expected regret, Σ (0.60 − mean of the machine pulled).

| method | regret after 10 / 100 / 500 / 1,000 pulls | % best machine (pulls 901-1000) | share of runs whose most-pulled machine is 1-5 |
|---|---|---|---|
| greedy | 1.5 / 14.1 / 70.4 / 140.7 | 28.4 | 12% · 21% · 14% · 28% · 24% |
| ε-greedy 0.1 | 1.5 / 10.5 / 30.0 / 46.3 | 81.5 | 0% · 4% · 0% · 82% · 14% |
| UCB1 (c = √2) | 1.7 / 13.6 / 49.9 / 80.3 | 65.9 | 0% · 0% · 0% · 100% · 0% |
| Thompson, Beta(1,1) | 1.6 / 10.3 / 24.8 / 31.3 | 93.1 | 0% · 0% · 0% · 98% · 2% |
| UCB, c = 0.5 (extra) | 1.5 / 8.8 / 20.4 / 25.8 | 94.0 | 0% · 0% · 0% · 98% · 2% |

Longer horizon (2,000 runs × 20,000 pulls), regret:

| pulls | ε-greedy 0.1 | UCB1 | Thompson |
|---|---|---|---|
| 1,000 | 46.3 | 80.3 | 31.3 |
| 2,000 | 71.5 | 121.0 | 37.3 |
| 5,000 | 130.4 | 187.7 | 44.9 |
| 10,000 | 221.3 | 243.8 | 50.7 |
| 20,000 | 401.2 | 299.9 | 56.8 |

**Finding (say it carefully):** over 1,000 pulls UCB1 has *more* regret than ε-greedy (80.3 vs
46.3). Its bonus √(2 ln t / n) keeps it checking machine 5 (0.50 vs 0.60, a gap of 0.1):
UCB1 stops favouring a worse machine only after about 2 ln t / gap² pulls of it: about 1,400 for machine 5 at t = 1,000,
more pulls than there have been.
ε-greedy's regret grows linearly (it keeps exploring at random forever), UCB1's like log t, and UCB1 overtakes it for
good at pull 12,103. Thompson sampling is best at every horizon (31.3 at 1,000, 56.8 at 20,000).
A smaller UCB bonus (c = 0.5, no longer UCB1's guarantee) gets 25.8. For episode 3: show Thompson vs
UCB1 vs ε-greedy and say "UCB1's regret curve bends over, ε-greedy's keeps climbing; Thompson's is lowest throughout".

Greedy gets stuck: only 28% of greedy runs end up mostly on machine 4. The seed-0
single run (`single_runs.greedy`, the one to animate) pulls machine 3 on its first pull, gets a coin, and
then pulls it all 1,000 times (final counts [0, 0, 1000, 0, 0]): its estimate never drops to 0, the
others never rise above 0. Single seeded runs (`classic.json → rover_arms.single_runs`, seed 0 for every method) hold every
pull's arm and reward, plus estimates, counts and Beta(a, b) parameters after pulls 1-20, 30, 50, 100, 200, 500, 1,000.

## 3. Contextual bandit: study tips (episode 4)

Rover recommends one of `STUDY_TIPS = ('flashcards', 'practice problems', 'explain it to a friend')` to each of 2,000 students. A student is two numbers in [0, 1]
("likes reading", "likes talking"), drawn uniformly; the tip helps (reward 1) with probability θ_tip · x,
θ = (0.8, 0) / (0.45, 0.45) / (0, 0.8). Practice problems are best on average (0.45 vs 0.40 and
0.40), but each tip is the best one for some students (39% / 22% / 39% of them).
200 runs.

| method | reward per student | last 500 students | regret after 2,000 | best tip given (last 500) |
|---|---|---|---|---|
| LinUCB, α = 1 | 0.527 | 0.535 | 26.3 | 85.0% |
| LinUCB, α = 0.5 | 0.504 | 0.510 | 74.2 | 63.7% |
| LinUCB, α = 2 | 0.525 | 0.534 | 31.1 | 83.4% |
| UCB1, ignores the student | 0.424 | 0.429 | 232.6 | 29.6% |
| always practice problems | 0.449 | 0.450 | 181.7 | 22.1% |
| oracle (knows θ) | 0.540 | 0.540 | 0.0 | 100.0% |

**On screen:** "Knowing who is asking is worth 17% more helped
students than the best single tip." The context-free bandit does worse than simply always giving the best-on-average
tip, because it pays to learn which tip that is. α = 0.5 under-explores: in some runs one tip is starved of data and its
line is never learned (the p90 regret is 148.5). The fitted lines of a typical run
(seed 130, the median-regret run) after 10 / 50 / 200 / 2,000 students are in `study_tips.example.theta_hat_after`.

## 4. Value iteration on Rover's world (episodes 5-7)

Russell & Norvig's 4×3 world: start (1,1), charger +1 at (4,3), puddle −1 at (4,2), rock at (2,2), −0.04 per step,
0.8 ahead / 0.1 each side, bumping = staying, γ = 1. Synchronous sweeps, stop when no value moves more than θ = 1e-4.

- Value iteration: **22 sweeps** (0.4 ms). The greedy policy is already optimal from sweep
  12; the last cell to flip is (3,1), which goes N (the short way past the puddle) until
  sweep 11 and W (the long way round) from sweep 12. V(start) first becomes positive at sweep
  5: the charger's news needs 5 sweeps to reach the start.
- Final values vs Russell & Norvig Fig. 17.3, every cell equal to three decimals (largest error vs the exact solution
  7.8e-05):

| | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|
| row 3 | 0.812 | 0.868 | 0.918 | +1 |
| row 2 | 0.762 | rock | 0.660 | -1 |
| row 1 | 0.705 | 0.655 | 0.611 | 0.388 |

- Values after sweeps 1, 2, 3, 5, 10 and 22 (grids, screen order, terminal cells show their reward) and the greedy
  policy after each are in `planning.value_iteration.after_sweep`; the max change per sweep is in `max_change_per_sweep`.
- Policy iteration from "always N": **5 evaluations, 4 improvements** (cells changed:
  [6, 1, 1, 1]), ending in R&N's Fig. 17.2 policy.
- Episode 6's cell by hand, (3,1) with the exact values: Q(W) = 0.611,
  Q(N) = 0.593 (each term: `planning.cell_3_1_by_hand`).

## 5. MC vs TD on the random walk (Sutton & Barto Example 6.2, Figure 6.2)

States A-E, start C, +1 for leaving on the right, true values 1/6 ... 5/6, V starts at 0.5. 100 runs × 100
episodes; within a run every learner sees the same episodes. RMS error = root-mean-square over the 5 states, averaged
over runs (as in the book). MC is constant-α, every visit.

| learner | RMS after 100 episodes | lowest RMS (episode) |
|---|---|---|
| TD(0), α = 0.05 | 0.036 | 0.035 (84) |
| TD(0), α = 0.1 | 0.057 | 0.049 (48) |
| TD(0), α = 0.15 | 0.073 | 0.056 (22) |
| MC, α = 0.01 | 0.097 | 0.097 (100) |
| MC, α = 0.02 | 0.085 | 0.085 (100) |
| MC, α = 0.03 | 0.096 | 0.088 (71) |
| MC, α = 0.04 | 0.108 | 0.099 (51) |

Starts at 0.236 for everyone. **Against the book:** the same picture as Fig. 6.2 right: from the first episode on,
every TD curve is below every MC curve; large α learns fast and then plateaus higher (TD α = 0.15 bottoms out around episode
22 and rises slightly). Fig. 6.2 left (TD α = 0.1 estimates after 0, 1, 10, 100
episodes, seed 0) is `random_walk.fig_6_2_left`.

Episode 8 (Monte Carlo on Rover's world, optimal policy, first-visit averages): the estimate of the start cell after 1 / 10 /
100 / 1,000 episodes is 0.680, 0.544, 0.688, 0.711 (exact 0.705); the
largest error over the visited cells after 1,000 episodes is 0.038.

## 6. The cliff walk (Sutton & Barto Example 6.6)

4 × 12, −1 per step, −100 and back to the start for stepping into the cliff; ε = 0.1, α = 0.5, γ = 1, 500 runs ×
500 episodes.

| method | reward per episode (episodes 101-500) | last 100 | episodes with a fall (of 500) | learned greedy path |
|---|---|---|---|---|
| Q-learning | -49.9 | -49.7 | 125.3 | 100% edge (row 2) |
| SARSA | -27.4 | -26.6 | 35.0 | 86% top (row 4), 13% does not reach the goal, 1% middle (row 3) |
| Expected SARSA | -21.1 | -20.8 | 25.1 | 100% middle (row 3) |

**Against the book:** S&B's figure shows SARSA near −25 and Q-learning near −50; we measure -27.4 and
-49.9. Q-learning learns the optimal path (13 steps along the edge, return −13) in every run, but
because it keeps exploring with ε = 0.1 next to the edge it falls in about a quarter of its episodes; SARSA learns the
safe road (top row, 17 steps) because it values the actions it will really take, exploratory steps included. Expected SARSA does
better than both online (-21.1; it takes the middle row in every run), as in S&B Fig. 6.3.
Example paths (seed 0): `cliff.methods.<method>.example_greedy_path`. Caveat: in 13% of SARSA runs the greedy path read
off the final Q-table loops instead of reaching the goal; with a constant α = 0.5 SARSA's estimates keep moving with
every exploratory step, and the ε-greedy behaviour still gets home (that is what the reward column measures).

**On screen:** "Q-learning hugs the edge and falls about 125 times in 500 episodes: about −50 per episode. SARSA takes the
safe road: about −27."

## 7. When rewards are rare: the sparse maze (episode 11)

`MAZE_LAYOUT`, 10 × 10, 59 open cells, one reward (+1) at the far corner (10,10), 0 everywhere else, γ = 0.99. The shortest
path is 28 moves; an episode ends after 200 moves; a random walk finishes within 200 moves
0.5% of the time. Q-learning, α = 0.5, ε = 0.1, 200 runs × 300 episodes (`maze_learning`).

| variant | first success: median episode (10th-90th pct.) | never in 300 | moves before it (median) | success, first 50 / last 50 episodes | greedy path = 28 |
|---|---|---|---|---|---|
| sparse reward only | 104 (22-238) | 21% | 20,760 | 0% / 1% | 0% |
| + shaping, φ = γ^(d−1) ("value with no walls") | 4 (2-5) | 0% | 763 | 82% / 100% | 100% |
| + novelty bonus 0.01/√n, Q starts at 1 | 2 (1-5) | 0% | 365 | 64% / 100% | 100% |
| (fails) shaping with φ = −distance/10 | never | 100% | - | 0% / 0% | 0% |
| (slow) novelty bonus, Q starts at 0 | 219 (143-251) | 0% | 43,760 | 0% / 94% | 96% |
| (extra) no bonus, Q starts at 1 | 5 (3-8) | 0% | 1,000 | 61% / 100% | 100% |

**What it shows.** With the sparse reward, Q-learning is a pure random walk until it first stumbles on the charger
(median episode 104, about 20,760 moves; 21% of runs never do in 300 episodes) — and even then it
barely improves (0.9% success in the last 50 episodes): a one-step update moves the news back one cell
per success, and each success needs another lucky walk to the far corner. (That is the case for eligibility traces and
n-step returns.) Potential-based shaping with an optimistic potential finds the charger in episode
4 and learns the 28-move path in every run; the novelty bonus finds it in episode
2.

**Design choices and caveats (important for the script).**
- Shaping is the same as starting Q at φ (Wiewiora, 2003). φ = −distance (the obvious choice) starts every Q *below*
  the value of standing still, and with γ < 1 bumping into a wall then earns (1 − γ)·distance/10 per step: Rover learns to
  bump into walls and never succeeds (0 of 200 runs). φ = γ^(d−1), the value a cell would have if the maze had no
  walls, is optimistic (walls only make the way longer): moving closer earns 0, every other move costs a little.
- A novelty bonus β/√n added to the reward with Q starting at 0 makes tried moves look better than untried ones (they
  earned a bonus); Rover circles where it has been until the bonus fades: first success only at episode
  219, though it then knows the maze (94% success at the end). Counting
  methods assume unvisited means novel (MBIE-EB style): Q starts at β/(1 − γ) = 1, the value of a bonus collected forever.
  Optimism alone (Q starts at 1, no bonus) gets most of the way (episode 5).
- Before the first success all Q-values of the bonus variants scale with β, so the first-success numbers do not depend on β.
- Visit counts for the glow (count-bonus run, seed 0, after 1, 3, 10, 50 episodes): `maze.count_bonus_visits_seed0_after_episodes`;
  the potential per cell: `maze.potential`.

## 8. Overestimation and divergence (episodes 12 and 14)

**Maximization bias (S&B Example 6.7, Fig. 6.5).** A → right ends (reward 0); A → left → B, where each of 10 actions
ends with reward ~ N(−0.1, 1). ε = 0.1, α = 0.1, 10,000 runs × 300 episodes. The optimum goes left 5% of the time
(ε/2).

| | episode 1 | 5 | 10 | 20 | 50 | 100 | 300 | peak |
|---|---|---|---|---|---|---|---|---|
| Q-learning, % left | 50.3 | 67.8 | 87.1 | 93.9 | 83.0 | 46.1 | 11.6 | 94.3 (ep. 19) |
| Double Q-learning, % left | 50.5 | 50.0 | 44.4 | 28.5 | 14.2 | 8.8 | 7.2 | 50.5 (ep. 1) |

Same shape as Fig. 6.5: Q-learning is drawn left far above 5% early on, Double Q-learning stays near 50% while untrained and then
falls toward 5% (7.1% over episodes 201-300, still converging). Caveat: the book doesn't say how
many actions B has; we used 10 as in Zhang Shangtong's reproduction. The peak height depends on that number (with 1
action at B there is no bias).

**The max of noisy estimates (episode 14).** Ten actions all worth exactly 0, each estimated with N(0, 1) noise: the
max of the estimates averages 1.539 (2 actions: 0.564 = 1/√π; 100 actions:
2.508); Monte Carlo 1.539. Choosing with one estimate and scoring with an independent second
one (the Double Q trick) averages -0.004.

**Baird's counterexample (S&B §11.2, Fig. 11.2).** 7 states, 8 weights, all rewards 0 (so w = 0 is perfect), w0 =
(1,1,1,1,1,1,10,1), γ = 0.99, α = 0.01. Semi-gradient off-policy TD: after 1,000 steps w = (96, 103, 77, 117, 135, 123, 7, 317),
|w| 10.3 → 416; after 10,000 steps |w| = 5.1e+11. The expected (DP)
update diverges too: w1-w6 = 114, w7 = 6.8, w8 = 335: w1-w6 and w8 grow
without bound while w7 (the lower state's own weight) drifts down from 10, the divergence S&B Fig. 11.2 shows for both
updates. Weight curves (200 points): `baird.methods.td.weights` and `baird.methods.dp.weights`.

## 9. Monte Carlo tree search on tic-tac-toe (episode 15)

UCT (c = √2), random rollouts, result +1 / 0 / −1 for the player who made each move, final move = most visited.
Perfect play from the empty board is a draw (minimax value 0); 5,478 legal positions.

**Visit counts, seed 0** (squares 0-8 read row by row from the top left):

empty board (X to move)

| sims | visits per square | tree size | most visited | picks a perfect move (50 seeds) |
|---|---|---|---|---|
| 10 | 0:1 1:2 2:1 3:1 4:1 5:1 6:1 7:1 8:1 | 11 | 1 | 100% |
| 100 | 0:8 1:14 2:13 3:12 4:12 5:6 6:3 7:8 8:24 | 101 | 8 | 100% |
| 1000 | 0:126 1:58 2:131 3:67 4:307 5:50 6:56 7:55 8:150 | 1001 | 4 | 100% |

`X...O...X`, O to move: an edge (1, 3, 5, 7) draws, a corner loses

| sims | visits per square | tree size | most visited | picks a perfect move (50 seeds) |
|---|---|---|---|---|
| 10 | 1:1 2:1 3:4 5:2 6:1 7:1 | 11 | 3 | 60% |
| 100 | 1:16 2:24 3:21 5:25 6:7 7:7 | 101 | 5 | 84% |
| 1000 | 1:233 2:47 3:245 5:251 6:36 7:188 | 328 | 5 | 100% |

From the empty board MCTS learns to prefer the centre, then the corners (every first move draws with perfect play,
but against random continuations the centre wins most). In the mid-game, 10 simulations pick a losing corner
40% of the time; by 1,000 every seed picks an edge.

**MCTS(n) against a random player and a perfect (minimax) player**, 100 games as X and 100 as O (win / draw / loss for MCTS):

| | vs random, as X | vs random, as O | vs minimax, as X | vs minimax, as O |
|---|---|---|---|---|
| MCTS(10) | 92 / 3 / 5 | 67 / 9 / 24 | 0 / 58 / 42 | 0 / 12 / 88 |
| MCTS(100) | 99 / 1 / 0 | 88 / 11 / 1 | 0 / 100 / 0 | 0 / 49 / 51 |
| MCTS(1000) | 100 / 0 / 0 | 91 / 9 / 0 | 0 / 100 / 0 | 0 / 97 / 3 |

**On screen:** "With 1,000 simulations per move, MCTS never loses to a random player and draws 197 of 200 games against
perfect play" (it lost 3, all as O).

## Control environments: ports of gymnasium's CartPole-v1 and Pendulum-v1

`cartpole_reset/step/obs` and `pendulum_reset/step/obs` were checked against gymnasium 1.4.0 (NumPy 2.5.4) step for step:
300 CartPole episodes (78,433 steps; random pushes and a balancing rule, 150 episodes reach the
500-step limit) and 200 Pendulum episodes (40,000 steps, torques beyond ±2 to test the clip, half of them passed as
float32 arrays): the largest state difference is **0.0** (bit-identical), observations
identical, rewards and terminated/truncated flags identical, and `cartpole_reset(np.random.default_rng(s))` equals
`env.reset(seed=s)`. Rewards: CartPole 1 per step including the last, Pendulum −(θ² + 0.1 θ̇² + 0.001 u²).

## Machine

Apple M4 Max, 64 GB, macOS-27.0.1-arm64-arm-64bit-Mach-O; Python 3.13.15, NumPy 2.5.4; 14 worker processes.
