"""The canonical code and numbers for "Trial and Reward" (reinforcement learning, from slot machines to reasoning
models). Every episode that shows an algorithm, a formula or a number computes it from here, so names, shapes
and values match across the course. Episodes show SHORT excerpts (<= 10 lines), may simplify, but never rename.
`python rl_reference.py` runs the self-checks (textbook values) and prints "all checks passed".

Rules of the house
- numpy only, seeded with np.random.default_rng(seed): the same seed gives the same run.
- Nothing heavy happens at import; the experiments that take minutes live in research/experiments/scripts/.
- Numbers measured on this Mac are in research/experiments/*.json: read them with measured("classic", ...).

Conventions
- Bandit arms are numbered from 0 in code. Rover's machines are numbered from 1 on screen ("machine 4" = arm 3).
- Grid cells are (col, row) as in Russell & Norvig: col 1 is the left column and row 1 the BOTTOM row.
  world.to_screen(cell) gives (row from the top, col from the left), both from 0, for drawing.
- Rewards arrive on transitions: transitions(s, a) -> [(p, s2, r, done)]. Terminal cells have V = 0 and
  their reward is paid on the way in, which makes V(s) equal Russell & Norvig's U(s) in every other cell.
- Action names are the strings "N", "E", "S", "W". Tabular Q is a dict: Q[s][a].

Contents (episode numbers in brackets)
- bandits [2-4]: ROVER_ARMS, BernoulliBandit, GaussianTestbed, incremental_mean, ucb_scores, beta_posterior, beta_pdf,
  run_bandit (one readable run) and bandit_testbed (many runs at once); STUDY_TIPS, linucb_scores, run_study_tips
- planning [5-7]: GridWorld, ROVER_WORLD, RN_UTILITIES, RN_POLICY, backup, policy_evaluation, value_iteration,
  q_from_v, greedy_policy, policy_iteration
- learning from experience [8-11]: random_walk_episode, play_episode, mc_prediction, td0_prediction, rms_error,
  n_step_return, CLIFF_WORLD, q_learning, sarsa, expected_sarsa, greedy_path, SPARSE_MAZE, maze_potential,
  shaped_reward, count_bonus, maze_learning, first_success
- overestimation and divergence [12, 14]: maximization_bias, expected_max_of_noisy, bairds_counterexample
- control [12-19]: cartpole_reset/step/obs, pendulum_reset/step/obs (exact gymnasium ports), mlp_forward, load_policy
- search [15]: TicTacToe, MIDGAME, minimax_value, best_moves, mcts, uct_score, puct_score
- policy gradients and language models [16-26]: softmax, log_softmax, discounted_returns, gae, ppo_clip_objective,
  bradley_terry_prob, dpo_loss, dpo_implicit_reward, grpo_advantages, kl_k1, kl_k3, pass_at_k
- measured(...) for the numbers measured on this Mac
"""

from __future__ import annotations

import json
import math
from collections import namedtuple
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "research" / "experiments"
WEIGHTS = EXPERIMENTS / "weights"


def argmax_random(values, rng):
    """Index of the largest value; ties are broken at random (so an untrained agent does not favour arm 0)."""
    values = np.asarray(values)
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(best[rng.integers(len(best))])


# ============================================================================ Module 1: bandits (episodes 2-4)

ROVER_ARMS = (0.25, 0.45, 0.30, 0.60, 0.50)   # Rover's five slot machines: chance each one pays out 1 coin
ROVER_BEST_ARM = 3                            # machine 4 on screen, pays 60% of the time


class BernoulliBandit:
    """Slot machines that pay 1 with a fixed probability each (and 0 otherwise)."""

    def __init__(self, probs=ROVER_ARMS):
        self.q_star = np.asarray(probs, dtype=float)   # the true mean payout of every arm (hidden from the agent)
        self.k = len(self.q_star)
        self.best_arm = int(np.argmax(self.q_star))

    def pull(self, arm, rng):
        return float(rng.random() < self.q_star[arm])


class GaussianTestbed:
    """One problem of the 10-armed testbed (Sutton & Barto §2.3): true values q*(a) ~ N(0, 1), rewards ~ N(q*(a), 1)."""

    def __init__(self, k=10, seed=0):
        self.q_star = np.random.default_rng(seed).normal(0.0, 1.0, k)
        self.k = k
        self.best_arm = int(np.argmax(self.q_star))

    def pull(self, arm, rng):
        return float(rng.normal(self.q_star[arm], 1.0))


def incremental_mean(q, n, r):
    """The new average after the n-th reward r (n counts r): Q <- Q + (r - Q) / n. No list of rewards needed."""
    return q + (r - q) / n


def ucb_scores(q, n, t, c=2.0):
    """UCB score of every arm at step t (1, 2, ...): Q + c * sqrt(ln t / N). Arms never pulled score +infinity.
    UCB1 (Auer et al., 2002) is c = sqrt(2); Sutton & Barto's Figure 2.4 uses c = 2."""
    q, n = np.asarray(q, dtype=float), np.asarray(n, dtype=float)
    bonus = c * np.sqrt(math.log(t) / np.maximum(n, 1))
    return np.where(n > 0, q + bonus, np.inf)


def beta_posterior(wins, losses):
    """Belief about a coin-paying arm after `wins` 1s and `losses` 0s, starting from Beta(1, 1) (all rates equally
    likely). Returns the Beta parameters (a, b); the mean belief is a / (a + b)."""
    return 1 + np.asarray(wins, dtype=float), 1 + np.asarray(losses, dtype=float)


def beta_pdf(x, a, b):
    """Density of Beta(a, b) at x (for drawing Thompson sampling's belief curves)."""
    x = np.asarray(x, dtype=float)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    with np.errstate(divide="ignore"):
        return np.exp(log_norm + (a - 1) * np.log(x) + (b - 1) * np.log1p(-x))


@dataclass
class BanditRun:
    """One agent playing one bandit, step by step (index t = after t + 1 pulls)."""
    actions: np.ndarray     # (steps,) arm pulled at each step
    rewards: np.ndarray     # (steps,) reward received
    estimates: np.ndarray   # (steps, k) the agent's estimate of every arm after each step (Thompson: posterior mean)
    counts: np.ndarray      # (steps, k) how often every arm has been pulled after each step
    regret: np.ndarray      # (steps,) cumulative expected regret: sum of (best arm's mean - pulled arm's mean)
    best_arm: int

    @property
    def optimal(self):
        return self.actions == self.best_arm


def run_bandit(bandit, method, steps=1000, seed=0, epsilon=None, q0=None, alpha=None, c=None):
    """Play `bandit` for `steps` pulls with one method and return a BanditRun.

    method       what it does                                                         parameters (defaults)
    "greedy"     always pull the best-looking arm                                     q0=0
    "eps"        ε-greedy: a random arm with probability ε, otherwise greedy          epsilon=0.1
    "optimistic" greedy, but every estimate starts high and moves at a fixed step α   q0=5, alpha=0.1
    "ucb"        pull the arm with the highest Q + c * sqrt(ln t / N)                 c=sqrt(2) (UCB1)
    "thompson"   sample a plausible value of every arm from its posterior, pull the best sample
                 (Bernoulli arms: Beta(1, 1) priors; Gaussian arms: N(0, 1) prior, unit noise)
    Estimates are sample averages unless alpha is given. Ties are broken at random.
    """
    rng = np.random.default_rng(seed)
    k = bandit.k
    epsilon = {"eps": 0.1}.get(method, 0.0) if epsilon is None else epsilon
    q0 = {"optimistic": 5.0}.get(method, 0.0) if q0 is None else q0
    alpha = {"optimistic": 0.1}.get(method) if alpha is None else alpha
    c = math.sqrt(2) if c is None else c
    bernoulli = isinstance(bandit, BernoulliBandit)

    Q = np.full(k, float(q0))
    N = np.zeros(k)
    total = np.zeros(k)                                     # sum of rewards per arm (Thompson's posterior)
    actions, rewards = np.zeros(steps, int), np.zeros(steps)
    estimates, counts = np.zeros((steps, k)), np.zeros((steps, k))
    for t in range(1, steps + 1):
        if method == "ucb":
            a = argmax_random(ucb_scores(Q, N, t, c), rng)
        elif method == "thompson":
            if bernoulli:
                a_post, b_post = beta_posterior(total, N - total)
                a = argmax_random(rng.beta(a_post, b_post), rng)
            else:
                a = argmax_random(rng.normal(total / (N + 1), 1 / np.sqrt(N + 1)), rng)
        elif rng.random() < epsilon:
            a = int(rng.integers(k))
        else:
            a = argmax_random(Q, rng)
        r = bandit.pull(a, rng)
        N[a] += 1
        total[a] += r
        Q[a] = Q[a] + alpha * (r - Q[a]) if alpha else incremental_mean(Q[a], N[a], r)
        actions[t - 1], rewards[t - 1], counts[t - 1] = a, r, N
        if method == "thompson":
            estimates[t - 1] = (total + 1) / (N + 2) if bernoulli else total / (N + 1)
        else:
            estimates[t - 1] = Q
    regret = np.cumsum(bandit.q_star.max() - bandit.q_star[actions])
    return BanditRun(actions, rewards, estimates, counts, regret, bandit.best_arm)


def bandit_testbed(method, runs=2000, steps=1000, seed=0, probs=None, k=10, epsilon=None, q0=None, alpha=None,
                   c=None):
    """The same methods as run_bandit, played by `runs` independent agents at once (vectorised; for the 2,000-run
    figures). probs=None: every run gets a fresh 10-armed Gaussian testbed; probs=(...): Bernoulli arms shared by
    all runs. Returns averages over runs per step (reward, fraction optimal, cumulative expected regret), the
    average final pull counts, and the share of runs whose most-pulled arm is each arm."""
    rng = np.random.default_rng(seed)
    epsilon = {"eps": 0.1}.get(method, 0.0) if epsilon is None else epsilon
    q0 = {"optimistic": 5.0}.get(method, 0.0) if q0 is None else q0
    alpha = {"optimistic": 0.1}.get(method) if alpha is None else alpha
    c = math.sqrt(2) if c is None else c
    bernoulli = probs is not None
    if bernoulli:
        q_star = np.tile(np.asarray(probs, dtype=float), (runs, 1))
    else:
        q_star = rng.normal(0.0, 1.0, (runs, k))
    k = q_star.shape[1]
    rows = np.arange(runs)
    best = q_star.argmax(1)
    Q, N, total = np.full((runs, k), float(q0)), np.zeros((runs, k)), np.zeros((runs, k))

    def pick_max(scores):                                   # argmax per row, ties at random
        is_max = scores == scores.max(1, keepdims=True)
        return np.argmax(is_max * rng.random(scores.shape), axis=1)

    mean_reward, frac_optimal, regret = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    for t in range(1, steps + 1):
        if method == "ucb":
            a = pick_max(np.where(N > 0, Q + c * np.sqrt(math.log(t) / np.maximum(N, 1)), np.inf))
        elif method == "thompson":
            sample = rng.beta(1 + total, 1 + N - total) if bernoulli else rng.normal(total / (N + 1), 1 / np.sqrt(N + 1))
            a = pick_max(sample)
        else:
            a = pick_max(Q)
            if epsilon > 0:
                explore = rng.random(runs) < epsilon
                a = np.where(explore, rng.integers(k, size=runs), a)
        means = q_star[rows, a]
        r = (rng.random(runs) < means).astype(float) if bernoulli else rng.normal(means, 1.0)
        N[rows, a] += 1
        total[rows, a] += r
        step = alpha if alpha else 1.0 / N[rows, a]
        Q[rows, a] += step * (r - Q[rows, a])
        mean_reward[t - 1] = r.mean()
        frac_optimal[t - 1] = (a == best).mean()
        regret[t - 1] = (q_star[rows, best] - means).mean()
    return dict(mean_reward=mean_reward, frac_optimal=frac_optimal, regret=np.cumsum(regret),
                final_counts=N.mean(0), most_pulled=np.bincount(N.argmax(1), minlength=k) / runs)


# ---------------------------------------------------------------------------- contextual bandits (episode 4)
# Rover recommends one of three study tips to each student. A student is two numbers in [0, 1]; the chance that
# the tip helps is a straight line in those numbers: P(helped) = theta_tip . x. The best tip depends on the student.

STUDY_TIPS = ("flashcards", "practice problems", "explain it to a friend")
STUDENT_FEATURES = ("likes reading", "likes talking")
STUDY_TIP_THETA = np.array([[0.80, 0.00],      # flashcards: great for readers, useless for talkers
                            [0.45, 0.45],      # practice problems: fine for everyone (the best on average)
                            [0.00, 0.80]])     # explain it to a friend: great for talkers


def linucb_scores(A_list, b_list, x, alpha=1.0):
    """LinUCB (Li et al., 2010, disjoint model): each arm fits a line theta = A^-1 b to what it has seen and scores
    theta . x + alpha * sqrt(x^T A^-1 x) - the predicted payoff plus a bonus where its data is thin."""
    scores = []
    for A, b in zip(A_list, b_list):
        A_inv = np.linalg.inv(A)
        theta = A_inv @ b
        scores.append(theta @ x + alpha * np.sqrt(x @ A_inv @ x))
    return np.array(scores)


def run_study_tips(method, rounds=2000, seed=0, alpha=1.0, c=math.sqrt(2)):
    """Rover recommends a tip to `rounds` students, one at a time. method: "linucb" (uses the student's features),
    "ucb" (UCB1, ignores them), "best_on_average" (always practice problems, the best tip for the average student),
    "oracle" (the best tip for each student - not learnable, the ceiling). Returns per-round arrays."""
    rng = np.random.default_rng(seed)
    n_tips, d = STUDY_TIP_THETA.shape
    A_list, b_list = [np.eye(d) for _ in range(n_tips)], [np.zeros(d) for _ in range(n_tips)]
    Q, N = np.zeros(n_tips), np.zeros(n_tips)
    average_best = int(np.argmax(STUDY_TIP_THETA @ np.full(d, 0.5)))
    actions, rewards, regret, best = (np.zeros(rounds, int), np.zeros(rounds), np.zeros(rounds), np.zeros(rounds, int))
    for t in range(1, rounds + 1):
        x = rng.random(d)                                   # a new student
        p = STUDY_TIP_THETA @ x                             # true chance each tip helps this student
        if method == "linucb":
            a = argmax_random(linucb_scores(A_list, b_list, x, alpha), rng)
        elif method == "ucb":
            a = argmax_random(ucb_scores(Q, N, t, c), rng)
        elif method == "best_on_average":
            a = average_best
        else:
            a = int(np.argmax(p))
        r = float(rng.random() < p[a])
        A_list[a] += np.outer(x, x)
        b_list[a] += r * x
        N[a] += 1
        Q[a] = incremental_mean(Q[a], N[a], r)
        actions[t - 1], rewards[t - 1], regret[t - 1], best[t - 1] = a, r, p.max() - p[a], int(np.argmax(p))
    return dict(actions=actions, rewards=rewards, regret=np.cumsum(regret), best=best,
                theta_hat=np.array([np.linalg.solve(A, b) for A, b in zip(A_list, b_list)]))


# ============================================================================ Module 2: MDPs (episodes 5-7)

ACTIONS = ("N", "E", "S", "W")
MOVES = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
PERPENDICULAR = {"N": ("W", "E"), "E": ("N", "S"), "S": ("E", "W"), "W": ("S", "N")}   # (to the left, to the right)
ARROWS = {"N": "↑", "E": "→", "S": "↓", "W": "←"}


class GridWorld:
    """A grid of cells (col, row), row 1 at the bottom. Rover moves N/E/S/W; with probability `slip` the wheels slip
    and it moves at right angles instead (half each side). Bumping into a wall or the edge means staying put.
    Every move costs `step_reward`; entering a terminal cell pays its reward and ends the episode; entering a
    cliff cell pays `cliff_reward` and sends Rover back to the start (the episode goes on)."""

    actions = ACTIONS

    def __init__(self, width, height, start, terminals, walls=(), cliffs=(), step_reward=0.0, slip=0.0,
                 cliff_reward=-100.0, name="grid"):
        self.width, self.height, self.start, self.name = width, height, tuple(start), name
        self.terminals = {tuple(s): float(r) for s, r in dict(terminals).items()}
        self.walls, self.cliffs = {tuple(s) for s in walls}, {tuple(s) for s in cliffs}
        self.step_reward, self.slip, self.cliff_reward = float(step_reward), float(slip), float(cliff_reward)
        # every cell Rover can be in, in reading order (top row first, left to right)
        self.states = [(c, r) for r in range(height, 0, -1) for c in range(1, width + 1)
                       if (c, r) not in self.walls and (c, r) not in self.cliffs]
        self._model = {(s, a): self._transitions(s, a) for s in self.states for a in ACTIONS}

    @classmethod
    def from_layout(cls, layout, terminal_rewards=None, **kwargs):
        """Build a world from strings, top row first: '.' floor, '#' wall, 'S' start, 'C' cliff, and any symbol in
        `terminal_rewards` (e.g. {"+": 1.0, "-": -1.0}) is a terminal cell paying that reward."""
        terminal_rewards = terminal_rewards or {"G": 1.0}
        height, width = len(layout), len(layout[0])
        cells = {(c + 1, height - r): ch for r, row in enumerate(layout) for c, ch in enumerate(row)}
        pick = lambda chars: [s for s, ch in cells.items() if ch in chars]
        return cls(width, height, start=pick("S")[0], walls=pick("#"), cliffs=pick("C"),
                   terminals={s: terminal_rewards[cells[s]] for s in pick(tuple(terminal_rewards))}, **kwargs)

    def is_terminal(self, s):
        return s in self.terminals

    def move(self, s, direction):
        """Where a move in `direction` lands if the wheels do not slip (walls and edges: stay put)."""
        dc, dr = MOVES[direction]
        s2 = (s[0] + dc, s[1] + dr)
        inside = 1 <= s2[0] <= self.width and 1 <= s2[1] <= self.height
        return s2 if inside and s2 not in self.walls else s

    def _transitions(self, s, a):
        if self.is_terminal(s):
            return [(1.0, s, 0.0, True)]
        outcomes = [(1.0 - self.slip, a)] + ([(self.slip / 2, d) for d in PERPENDICULAR[a]] if self.slip else [])
        result = []
        for p, direction in outcomes:
            s2 = self.move(s, direction)
            if s2 in self.cliffs:
                result.append((p, self.start, self.cliff_reward, False))
            elif s2 in self.terminals:
                result.append((p, s2, self.step_reward + self.terminals[s2], True))
            else:
                result.append((p, s2, self.step_reward, False))
        return result

    def transitions(self, s, a):
        """The model: [(probability, next cell, reward, done)] - intended move first, then the two slips."""
        return self._model[(s, a)]

    def reset(self):
        return self.start

    def step(self, s, a, rng):
        """Sample one move from the model (what the world does when it is not telling Rover its rules)."""
        outcomes = self._model[(s, a)]
        if len(outcomes) > 1:
            u = rng.random()
            for p, s2, r, done in outcomes:
                u -= p
                if u < 0:
                    return s2, r, done
        _, s2, r, done = outcomes[-1]                       # one outcome, or u fell on rounding at the very end
        return s2, r, done

    def to_screen(self, s):
        """(col, row) with row 1 at the bottom -> (row from the top, col from the left), both counted from 0."""
        return self.height - s[1], s[0] - 1

    def from_screen(self, screen_row, screen_col):
        return screen_col + 1, self.height - screen_row

    def grid(self, values, terminal="reward"):
        """A value per cell -> a (height, width) array in screen order; walls are NaN. terminal="reward" shows a
        terminal cell's reward (as Russell & Norvig's figures do), terminal="value" shows the value given."""
        out = np.full((self.height, self.width), np.nan)
        for s in self.states:
            v = self.terminals[s] if (terminal == "reward" and self.is_terminal(s)) else values.get(s, np.nan)
            out[self.to_screen(s)] = v
        for s in self.cliffs:
            out[self.to_screen(s)] = self.cliff_reward if terminal == "reward" else np.nan
        return out

    def show(self, values=None, policy=None):
        """A text picture of the world, values to 3 decimals or policy arrows (for debugging and checks)."""
        lines = []
        for row in range(self.height, 0, -1):
            cells = []
            for col in range(1, self.width + 1):
                s = (col, row)
                if s in self.walls:
                    cells.append("  ##  ")
                elif s in self.cliffs:
                    cells.append(" cliff")
                elif self.is_terminal(s):
                    cells.append(f"{self.terminals[s]:+6.2f}")
                elif policy is not None:
                    cells.append(f"   {ARROWS[policy[s]]}  ")
                elif values is not None:
                    cells.append(f"{values[s]:6.3f}")
                else:
                    cells.append("  S   " if s == self.start else "  .   ")
            lines.append(" ".join(cells))
        return "\n".join(lines)


ROVER_LAYOUT = ("...+",      # row 3: the charger (+1) at (4, 3)
                ".#.-",      # row 2: the rock at (2, 2), the puddle (-1) at (4, 2)
                "S...")      # row 1: Rover starts at (1, 1)
ROVER_WORLD = GridWorld.from_layout(ROVER_LAYOUT, terminal_rewards={"+": 1.0, "-": -1.0}, step_reward=-0.04,
                                    slip=0.2, name="Rover's world (Russell & Norvig 4x3)")
ROVER_GAMMA = 1.0
CHARGER, PUDDLE, ROCK = (4, 3), (4, 2), (2, 2)

# Russell & Norvig (AIMA, Fig. 17.3) utilities and (Fig. 17.2) optimal policy for R = -0.04, gamma = 1
RN_UTILITIES = {(1, 3): 0.812, (2, 3): 0.868, (3, 3): 0.918,
                (1, 2): 0.762, (3, 2): 0.660,
                (1, 1): 0.705, (2, 1): 0.655, (3, 1): 0.611, (4, 1): 0.388}
RN_POLICY = {(1, 3): "E", (2, 3): "E", (3, 3): "E",
             (1, 2): "N", (3, 2): "N",
             (1, 1): "N", (2, 1): "W", (3, 1): "W", (4, 1): "W"}


def backup(world, V, s, a, gamma=1.0):
    """One Bellman backup: the expected reward of doing a in s, plus the discounted value of where Rover lands."""
    return sum(p * (r + (0.0 if done else gamma * V[s2])) for p, s2, r, done in world.transitions(s, a))


def policy_evaluation(world, policy, gamma=1.0, theta=1e-4, record=False):
    """V of a fixed policy (dict cell -> action, or cell -> {action: probability}) by repeated synchronous sweeps
    until no value moves more than theta. record=True also returns the list of V after every sweep."""
    V = {s: 0.0 for s in world.states}
    history = [dict(V)]
    while True:
        new = {}
        for s in world.states:
            choice = None if world.is_terminal(s) else policy[s]
            if choice is None:
                new[s] = 0.0
            elif isinstance(choice, str):
                new[s] = backup(world, V, s, choice, gamma)
            else:
                new[s] = sum(prob * backup(world, V, s, a, gamma) for a, prob in choice.items())
        delta = max(abs(new[s] - V[s]) for s in world.states)
        V = new
        history.append(dict(V))
        if delta < theta:
            return (V, history) if record else V


def value_iteration(world, gamma=1.0, theta=1e-4, record=True):
    """Solve the world: sweep every cell with V(s) <- max_a backup(s, a) until no value moves more than theta.
    Sweeps are synchronous (each sweep reads the previous sweep's values). Returns (V, history) with history[k] =
    V after k sweeps (history[0] is all zeros), or just V if record=False."""
    V = {s: 0.0 for s in world.states}
    history = [dict(V)]
    while True:
        new = {s: 0.0 if world.is_terminal(s) else max(backup(world, V, s, a, gamma) for a in world.actions)
               for s in world.states}
        delta = max(abs(new[s] - V[s]) for s in world.states)
        V = new
        history.append(dict(V))
        if delta < theta:
            return (V, history) if record else V


def q_from_v(world, V, gamma=1.0):
    """Action values from state values: Q(s, a) = backup(s, a), for every non-terminal cell."""
    return {s: {a: backup(world, V, s, a, gamma) for a in world.actions}
            for s in world.states if not world.is_terminal(s)}


def greedy_policy(world, V, gamma=1.0):
    """The action with the best backup in every non-terminal cell (ties: first in N, E, S, W order)."""
    Q = q_from_v(world, V, gamma)
    return {s: max(world.actions, key=lambda a: q[a]) for s, q in Q.items()}


def policy_iteration(world, gamma=1.0, theta=1e-8, start_action="N"):
    """Evaluate the policy, improve it greedily, repeat until it stops changing. Starts from "always go north" (a
    policy that does reach a terminal, so its value is finite with gamma = 1). Returns (policy, V, policies) where
    policies[i] is the policy evaluated in round i; the last one is stable, so len(policies) - 1 improvements
    changed something."""
    policy = {s: start_action for s in world.states if not world.is_terminal(s)}
    policies = [dict(policy)]
    while True:
        V = policy_evaluation(world, policy, gamma, theta)
        new = {}
        for s, q in q_from_v(world, V, gamma).items():
            best = max(world.actions, key=lambda a: q[a])
            new[s] = best if q[best] > q[policy[s]] + 1e-9 else policy[s]   # change only if strictly better
        if new == policy:
            return policy, V, policies
        policy = new
        policies.append(dict(policy))


# ============================================================================ Module 3: learning from experience (8-11)

Transition = namedtuple("Transition", "s a r s2 done")

RANDOM_WALK_STATES = "ABCDE"                                # Sutton & Barto Example 6.2, start at C
RANDOM_WALK_TRUE = {s: (i + 1) / 6 for i, s in enumerate(RANDOM_WALK_STATES)}   # 1/6 ... 5/6


def random_walk_episode(rng):
    """One episode of the five-state random walk: from C, step left or right with equal chance until falling off
    an end. Falling off the right end pays +1, everything else 0. Returns a list of Transitions."""
    i, steps = 2, []
    while True:
        a = "right" if rng.random() < 0.5 else "left"
        j = i + (1 if a == "right" else -1)
        done = j < 0 or j > 4
        steps.append(Transition(RANDOM_WALK_STATES[i], a, 1.0 if j > 4 else 0.0,
                                None if done else RANDOM_WALK_STATES[j], done))
        if done:
            return steps
        i = j


def play_episode(world, policy, rng, max_steps=1000):
    """Let Rover follow `policy` (dict cell -> action, or a function cell -> action) from the start cell."""
    s, steps = world.reset(), []
    for _ in range(max_steps):
        a = policy(s) if callable(policy) else policy[s]
        s2, r, done = world.step(s, a, rng)
        steps.append(Transition(s, a, r, s2, done))
        if done:
            break
        s = s2
    return steps


def mc_prediction(episodes, V, alpha=None, gamma=1.0, first_visit=True, record=False):
    """Monte Carlo prediction: after each episode, move V(s) toward the return G that actually followed s.
    alpha=None: V(s) is the plain average of the returns seen (first_visit decides whether a state revisited in
    one episode counts once); alpha=0.1: constant-alpha MC, V(s) <- V(s) + alpha (G - V(s)).
    V: dict of starting values (it defines the states). record=True also returns V after every episode."""
    V, n = dict(V), {s: 0 for s in V}
    history = [dict(V)]
    for episode in episodes:
        G, returns = 0.0, []
        for step in reversed(episode):                      # returns, computed backwards from the end
            G = step.r + gamma * G
            returns.append((step.s, G))
        returns.reverse()
        seen = set()
        for s, G in returns:
            if first_visit and s in seen:
                continue
            seen.add(s)
            n[s] += 1
            V[s] += (G - V[s]) * (alpha if alpha is not None else 1.0 / n[s])
        history.append(dict(V))
    return (V, history) if record else V


def td0_prediction(episodes, V, alpha=0.1, gamma=1.0, record=False):
    """TD(0): after every single step, move V(s) toward r + gamma V(s'), the reward plus the next guess.
    The TD error is delta = r + gamma V(s') - V(s); V(s) <- V(s) + alpha delta. Same arguments as mc_prediction."""
    V = dict(V)
    history = [dict(V)]
    for episode in episodes:
        for s, a, r, s2, done in episode:
            target = r if done else r + gamma * V[s2]
            V[s] += alpha * (target - V[s])
        history.append(dict(V))
    return (V, history) if record else V


def rms_error(V, true=RANDOM_WALK_TRUE):
    """Root-mean-square error of the estimates, averaged over the states (Sutton & Barto Fig. 6.2, right)."""
    return math.sqrt(sum((V[s] - v) ** 2 for s, v in true.items()) / len(true))


def n_step_return(rewards, bootstrap_value, gamma=1.0):
    """G = r1 + gamma r2 + ... + gamma^(n-1) rn + gamma^n V(s_n): n real rewards, then one guess.
    Pass bootstrap_value = 0 if the episode ended within the n steps."""
    G = bootstrap_value
    for r in reversed(rewards):
        G = r + gamma * G
    return G


CLIFF_LAYOUT = ("............",
                "............",
                "............",
                "SCCCCCCCCCCG")


def CliffWalk():
    """Sutton & Barto Example 6.6: 4 x 12, start bottom-left, goal bottom-right, the cliff in between.
    -1 per step (the step into the goal too); stepping into the cliff costs -100 and sends Rover back to start."""
    return GridWorld.from_layout(CLIFF_LAYOUT, terminal_rewards={"G": 0.0}, step_reward=-1.0, cliff_reward=-100.0,
                                 name="the cliff walk")


CLIFF_WORLD = CliffWalk()

MAZE_LAYOUT = ("....#....G",     # the charger G at the far corner (10, 10): the only reward in the maze
               ".##.#.##.#",
               ".#..#..#..",
               ".#.###.##.",
               ".#...#....",
               ".###.#.###",
               "...#.#....",
               "##.#.####.",
               "...#......",
               "S#...#####")     # Rover starts at (1, 1); the shortest path is 28 moves
MAZE_MAX_STEPS = 200             # an episode ends after 200 moves (a random walk gets there in ~0.5% of them)
MAZE_GAMMA = 0.99


def SparseMaze():
    """Episode 11's maze: 10 x 10, reward +1 at the far corner and 0 everywhere else, no slip."""
    return GridWorld.from_layout(MAZE_LAYOUT, terminal_rewards={"G": 1.0}, step_reward=0.0, name="the sparse maze")


SPARSE_MAZE = SparseMaze()


def shortest_path_lengths(world, goal):
    """Number of moves from every cell to `goal` (breadth-first search over the no-slip moves)."""
    dist, frontier = {goal: 0}, [goal]
    while frontier:
        nxt = []
        for s in frontier:
            for a in ACTIONS:                               # grid moves are reversible: a neighbour is one step back
                s2 = world.move(s, a)
                if s2 not in dist and s2 not in world.cliffs:
                    dist[s2] = dist[s] + 1
                    nxt.append(s2)
        frontier = nxt
    return dist


def epsilon_greedy(q, epsilon, rng):
    """q: dict action -> value. With probability epsilon a random action, otherwise a best one (ties at random)."""
    actions = list(q)
    if rng.random() < epsilon:
        return actions[int(rng.random() * len(actions))]
    best = max(q.values())
    ties = [a for a in actions if q[a] == best]
    return ties[0] if len(ties) == 1 else ties[int(rng.random() * len(ties))]


def expected_q(q, epsilon):
    """The value of the next state under the epsilon-greedy policy itself: sum over a of pi(a) Q(a)."""
    best = max(q.values())
    ties = [a for a in q if q[a] == best]
    return sum((epsilon / len(q) + ((1 - epsilon) / len(ties) if a in ties else 0.0)) * v for a, v in q.items())


def shaped_reward(r, phi_s, phi_s2, gamma):
    """Potential-based shaping (Ng, Harada & Russell, 1999): r + gamma phi(s') - phi(s). It never changes which
    policy is optimal. Use phi(s') = 0 when s' is terminal."""
    return r + gamma * phi_s2 - phi_s


def count_bonus(n, beta=0.1):
    """A novelty bonus that fades with visits: beta / sqrt(n), n = visits so far (including this one)."""
    return beta / math.sqrt(n)


TabularRun = namedtuple("TabularRun", "Q rewards lengths")


def td_control(env, method, episodes=500, alpha=0.5, epsilon=0.1, gamma=1.0, seed=0, q0=0.0, max_steps=None,
               reward_fn=None):
    """Learn Q(s, a) while behaving epsilon-greedily. method: "q_learning", "sarsa" or "expected_sarsa" - they
    differ only in the target the update moves toward. reward_fn(s, r, s2, done) -> the reward to LEARN from
    (shaping, bonuses); the returned per-episode sums are always of the world's own reward."""
    rng = np.random.default_rng(seed)
    Q = {s: {a: (0.0 if env.is_terminal(s) else q0) for a in env.actions} for s in env.states}
    rewards, lengths = [], []
    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q[s], epsilon, rng)
        total, t = 0.0, 0
        while True:
            s2, r, done = env.step(s, a, rng)
            total, t = total + r, t + 1
            r_learn = r if reward_fn is None else reward_fn(s, r, s2, done)
            if method == "sarsa":                           # on-policy: the action Rover will really take next
                a2 = epsilon_greedy(Q[s2], epsilon, rng)
                target = r_learn + (0.0 if done else gamma * Q[s2][a2])
            elif method == "q_learning":                    # off-policy: the best action, whatever Rover does
                target = r_learn + (0.0 if done else gamma * max(Q[s2].values()))
            else:                                           # expected SARSA: average over the policy
                target = r_learn + (0.0 if done else gamma * expected_q(Q[s2], epsilon))
            Q[s][a] += alpha * (target - Q[s][a])
            if done or (max_steps and t >= max_steps):
                break
            if method != "sarsa":
                a2 = epsilon_greedy(Q[s2], epsilon, rng)
            s, a = s2, a2
        rewards.append(total)
        lengths.append(t)
    return TabularRun(Q, np.array(rewards), np.array(lengths))


def q_learning(env, episodes=500, alpha=0.5, epsilon=0.1, gamma=1.0, seed=0, **kwargs):
    """Q-learning (Watkins, 1989): Q(s,a) <- Q(s,a) + alpha (r + gamma max_a' Q(s',a') - Q(s,a))."""
    return td_control(env, "q_learning", episodes, alpha, epsilon, gamma, seed, **kwargs)


def sarsa(env, episodes=500, alpha=0.5, epsilon=0.1, gamma=1.0, seed=0, **kwargs):
    """SARSA: Q(s,a) <- Q(s,a) + alpha (r + gamma Q(s',a') - Q(s,a)), a' = the action actually taken next."""
    return td_control(env, "sarsa", episodes, alpha, epsilon, gamma, seed, **kwargs)


def expected_sarsa(env, episodes=500, alpha=0.5, epsilon=0.1, gamma=1.0, seed=0, **kwargs):
    """Expected SARSA: like SARSA, but the target averages Q(s', .) over the epsilon-greedy policy."""
    return td_control(env, "expected_sarsa", episodes, alpha, epsilon, gamma, seed, **kwargs)


MAZE_GOAL = (10, 10)


def maze_potential(s, goal=MAZE_GOAL, gamma=MAZE_GAMMA):
    """Episode 11's shaping potential: the value cell s would have if the maze had no walls, gamma^(d - 1) with d
    the Manhattan distance to the charger. Optimistic (walls only make the way longer), so it guides without
    trapping. Shaping with phi is the same as starting Q at phi (Wiewiora, 2003): a NEGATIVE potential such as
    -distance starts Q below the value of standing still, and Rover learns to bump into walls."""
    return gamma ** (abs(s[0] - goal[0]) + abs(s[1] - goal[1]) - 1)


def maze_learning(method, episodes=300, seed=0, alpha=0.5, epsilon=0.1, beta=0.01, world=SPARSE_MAZE):
    """Q-learning in the sparse maze (gamma 0.99, 200 moves per episode) with one of:
    "sparse"       the world's reward only (+1 at the charger), Q starts at 0
    "shaping"      + potential-based shaping with maze_potential
    "count_bonus"  + a novelty bonus beta / sqrt(visits to the cell Rover lands in), and moves never tried are
                   assumed to lead somewhere new: Q starts at beta / (1 - gamma), a bonus collected forever
    and the instructive failures: "shaping_negative" (phi = -distance / 10), "count_bonus_q0" (the bonus with Q
    starting at 0), "optimistic" (Q starts at 1, no bonus). Returns (TabularRun, visits per cell)."""
    gamma, visits = MAZE_GAMMA, {}
    distance = lambda s: abs(s[0] - MAZE_GOAL[0]) + abs(s[1] - MAZE_GOAL[1])
    potential = {"shaping": maze_potential, "shaping_negative": lambda s: -distance(s) / 10}.get(method)
    q0 = {"count_bonus": beta / (1 - gamma), "optimistic": 1.0}.get(method, 0.0)

    def reward_fn(s, r, s2, done):
        visits[s2] = visits.get(s2, 0) + 1
        if potential is not None:
            return shaped_reward(r, potential(s), 0.0 if done else potential(s2), gamma)
        if method in ("count_bonus", "count_bonus_q0"):
            return r + count_bonus(visits[s2], beta)
        return r

    run = q_learning(world, episodes, alpha, epsilon, gamma, seed, q0=q0, max_steps=MAZE_MAX_STEPS,
                     reward_fn=reward_fn)
    return run, visits


def first_success(rewards):
    """The episode number (from 1) of the first episode with a positive reward, or None if there was none."""
    hits = np.flatnonzero(np.asarray(rewards) > 0)
    return int(hits[0]) + 1 if len(hits) else None


def greedy_path(env, Q, max_steps=100):
    """The cells Rover visits from the start when it always takes its best action (no exploration, no slip)."""
    s, path = env.reset(), [env.reset()]
    for _ in range(max_steps):
        if env.is_terminal(s):
            break
        a = max(env.actions, key=lambda a: Q[s][a])
        s = env.transitions(s, a)[0][1]
        path.append(s)
    return path


# ============================================================================ Module 4: approximation, overestimation

BAIRD_FEATURES = np.array([[2, 0, 0, 0, 0, 0, 0, 1],
                           [0, 2, 0, 0, 0, 0, 0, 1],
                           [0, 0, 2, 0, 0, 0, 0, 1],
                           [0, 0, 0, 2, 0, 0, 0, 1],
                           [0, 0, 0, 0, 2, 0, 0, 1],
                           [0, 0, 0, 0, 0, 2, 0, 1],
                           [0, 0, 0, 0, 0, 0, 1, 2]], dtype=float)   # v(s) = features[s] . w (S&B Fig. 11.1)
BAIRD_W0 = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)


def bairds_counterexample(steps=1000, alpha=0.01, gamma=0.99, method="td", seed=0):
    """Baird's counterexample (Sutton & Barto §11.2): seven states, eight weights, every reward 0, so the true
    value is 0 and w = 0 would be perfect. The behaviour policy takes the dashed action 6/7 of the time (to one of
    the six upper states) and the solid one 1/7 (to the lower state); the target policy always takes solid.
    method="td": semi-gradient off-policy TD(0) with importance ratio rho (7 for solid, 0 for dashed);
    method="dp": the expected (semi-gradient DP) update over all states. Returns the weights after every step,
    shape (steps + 1, 8). Both diverge: function approximation + bootstrapping + off-policy = the deadly triad."""
    rng = np.random.default_rng(seed)
    X, w = BAIRD_FEATURES, BAIRD_W0.copy()
    history = [w.copy()]
    s = int(rng.integers(7))
    for _ in range(steps):
        if method == "td":
            if rng.random() < 6 / 7:
                s2, rho = int(rng.integers(6)), 0.0         # dashed: the target policy never does this
            else:
                s2, rho = 6, 7.0                            # solid: pi = 1, b = 1/7
            delta = 0.0 + gamma * X[s2] @ w - X[s] @ w
            w = w + alpha * rho * delta * X[s]
            s = s2
        else:
            delta = gamma * X[6] @ w - X @ w                # target policy: every state goes to the lower state
            w = w + alpha * X.T @ delta / 7
        history.append(w.copy())
    return np.array(history)


def maximization_bias(runs=10_000, episodes=300, seed=0, alpha=0.1, epsilon=0.1, n_b_actions=10):
    """Sutton & Barto Example 6.7 (Fig. 6.5). From A, "right" ends the episode with reward 0; "left" goes to B
    (reward 0), where every one of n_b_actions ends the episode with a reward ~ N(-0.1, 1). Going left is worse,
    but the max over B's noisy estimates looks positive. Returns the fraction of runs that went left in each
    episode for Q-learning and Double Q-learning (all runs at once). The optimum with epsilon = 0.1 is 5%."""
    rng = np.random.default_rng(seed)
    rows = np.arange(runs)
    out = {}

    def best(q):                                            # argmax per run, ties at random
        return np.argmax((q == q.max(1, keepdims=True)) * rng.random(q.shape), axis=1)

    def choose(q):                                          # epsilon-greedy per run
        return np.where(rng.random(runs) < epsilon, rng.integers(q.shape[1], size=runs), best(q))

    for method in ("q_learning", "double_q"):
        tables = 1 if method == "q_learning" else 2
        QA = np.zeros((tables, runs, 2))                    # [table, run, action at A]; action 0 = left
        QB = np.zeros((tables, runs, n_b_actions))
        left = np.zeros(episodes)
        for e in range(episodes):
            a = choose(QA.sum(0))                           # Double Q behaves greedily w.r.t. Q1 + Q2
            went_left = a == 0
            left[e] = went_left.mean()
            b = choose(QB.sum(0))
            r = rng.normal(-0.1, 1.0, runs)
            if method == "q_learning":
                target_a = np.where(went_left, QB[0].max(1), 0.0)
                QA[0, rows, a] += alpha * (target_a - QA[0, rows, a])
                QB[0, rows, b] += np.where(went_left, alpha * (r - QB[0, rows, b]), 0.0)
            else:
                first = rng.random(runs) < 0.5               # which table learns this time
                i, j = np.where(first, 0, 1), np.where(first, 1, 0)
                best_b = best(QB[i, rows])                  # one table picks the best action in B...
                target_a = np.where(went_left, QB[j, rows, best_b], 0.0)   # ...the other one scores it
                QA[i, rows, a] += alpha * (target_a - QA[i, rows, a])
                QB[i, rows, b] += np.where(went_left, alpha * (r - QB[i, rows, b]), 0.0)
        out[method] = left
    return out


def expected_max_of_noisy(n, sigma=1.0):
    """E[max of n independent N(0, sigma^2) estimates] - how much the max of n honest-but-noisy zeros overstates
    the truth (0). n = 2: sigma / sqrt(pi) = 0.564 sigma; n = 10: 1.539 sigma. Numerical integration."""
    if n == 1:
        return 0.0
    x = np.linspace(-12.0, 12.0, 24_001)
    pdf = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    integrand = x * n * pdf * cdf ** (n - 1)
    return float(sigma * np.sum((integrand[1:] + integrand[:-1]) / 2) * (x[1] - x[0]))


# ============================================================================ control environments (episodes 12-19)
# Exact numpy ports of gymnasium's CartPole-v1 and Pendulum-v1 (constants, equations, termination). Verified
# against gymnasium 1.4.0: identical (max |difference| = 0) step for step, from reset(seed) through termination or
# truncation (research/experiments/scripts/check_gym_ports.py, results in classic.json["gym_ports"]).
# The state is float64, as gymnasium keeps it internally; gymnasium hands the agent a float32 copy
# (cartpole_obs / pendulum_obs), so trained networks should be fed those.

CARTPOLE = dict(gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, tau=0.02, x_threshold=2.4,
                theta_threshold=12 * 2 * math.pi / 360, max_steps=500)   # length = HALF the pole; 12 degrees


def cartpole_reset(rng):
    """A new episode: cart position, cart velocity, pole angle, pole angular velocity, each uniform in +-0.05.
    cartpole_reset(np.random.default_rng(seed)) equals gymnasium's env.reset(seed=seed)."""
    return rng.uniform(low=-0.05, high=0.05, size=(4,))


def cartpole_step(state, action):
    """Push the cart left (action 0) or right (1) for 0.02 s (Euler integration). Returns (state, reward,
    terminated): reward 1.0 for every step including the last; terminated when |x| > 2.4 or |angle| > 12 degrees.
    gymnasium also truncates the episode after 500 steps (CARTPOLE["max_steps"]), so a return is at most 500."""
    x, x_dot, theta, theta_dot = (float(v) for v in state)
    total_mass = CARTPOLE["masspole"] + CARTPOLE["masscart"]
    polemass_length = CARTPOLE["masspole"] * CARTPOLE["length"]
    force = CARTPOLE["force_mag"] if action == 1 else -CARTPOLE["force_mag"]
    costheta, sintheta = math.cos(theta), math.sin(theta)
    temp = (force + polemass_length * (theta_dot * theta_dot) * sintheta) / total_mass
    thetaacc = (CARTPOLE["gravity"] * sintheta - costheta * temp) / (
        CARTPOLE["length"] * (4.0 / 3.0 - CARTPOLE["masspole"] * (costheta * costheta) / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    tau = CARTPOLE["tau"]
    x, x_dot = x + tau * x_dot, x_dot + tau * xacc
    theta, theta_dot = theta + tau * theta_dot, theta_dot + tau * thetaacc
    terminated = (x < -CARTPOLE["x_threshold"] or x > CARTPOLE["x_threshold"]
                  or theta < -CARTPOLE["theta_threshold"] or theta > CARTPOLE["theta_threshold"])
    return np.array([x, x_dot, theta, theta_dot]), 1.0, terminated


def cartpole_obs(state):
    """What gymnasium gives the agent: the state as float32."""
    return np.asarray(state, dtype=np.float32)


PENDULUM = dict(max_speed=8.0, max_torque=2.0, dt=0.05, g=10.0, m=1.0, l=1.0, max_steps=200)


def angle_normalize(x):
    """An angle folded into [-pi, pi)."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def pendulum_reset(rng):
    """A new episode: angle uniform in [-pi, pi] (0 = upright), angular velocity uniform in [-1, 1].
    pendulum_reset(np.random.default_rng(seed)) equals gymnasium's env.reset(seed=seed)."""
    high = np.array([np.pi, 1.0])
    return rng.uniform(low=-high, high=high)


def pendulum_step(state, torque):
    """Apply a torque (clipped to +-2) for 0.05 s. Returns (state, reward) with reward = -(angle^2 + 0.1 speed^2 +
    0.001 torque^2), computed BEFORE the move; 0 is perfect, about -16.3 the worst. Speed is clipped to +-8.
    Pendulum never terminates; gymnasium truncates after 200 steps (PENDULUM["max_steps"]).
    torque: a number (float64 maths) or, as a gymnasium agent passes it, a float32 array - then the torque terms
    are rounded to float32 exactly as gymnasium (under NumPy 2) rounds them."""
    th, thdot = np.asarray(state, dtype=np.float64)
    g, m, l, dt = PENDULUM["g"], PENDULUM["m"], PENDULUM["l"], PENDULUM["dt"]
    u = np.clip(np.ravel(torque), -PENDULUM["max_torque"], PENDULUM["max_torque"])[0]
    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = np.clip(newthdot, -PENDULUM["max_speed"], PENDULUM["max_speed"])
    newth = th + newthdot * dt
    return np.array([newth, newthdot], dtype=np.float64), float(-costs)


def pendulum_obs(state):
    """What gymnasium gives the agent: (cos angle, sin angle, angular velocity) as float32."""
    return np.array([np.cos(state[0]), np.sin(state[0]), state[1]], dtype=np.float32)


# ---------------------------------------------------------------------------- small networks (trained elsewhere)


def relu(z):
    return np.maximum(z, 0.0)


ACTIVATIONS = {"tanh": np.tanh, "relu": relu}


def mlp_forward(params, x, activation="tanh"):
    """A small multilayer perceptron: h <- act(h @ W + b) for every layer but the last, which stays linear.
    params: [(W0, b0), (W1, b1), ...] with W_i of shape (inputs, outputs). x: (inputs,) or (batch, inputs)."""
    act = ACTIVATIONS[activation]
    h = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = act(h)
    return h


@dataclass
class Policy:
    """A trained network loaded from weights/<name>.npz + .json (see load_policy)."""
    name: str
    params: list
    activation: str
    output: str                      # "logits" (action probabilities), "q" (action values) or "mean" (Gaussian mean)
    meta: dict = field(default_factory=dict)

    def __call__(self, obs):
        """The raw network output for an observation (or a batch of them)."""
        return mlp_forward(self.params, obs, self.activation)

    def act(self, obs, rng=None):
        """An action: the best one for "q"; for "logits" the most likely one, or a sample if rng is given; for
        "mean" the mean itself (squashed and scaled if the .json says "squash": "tanh" and "action_scale": s)."""
        out = self(obs)
        if self.output == "mean":
            if self.meta.get("squash") == "tanh":
                out = np.tanh(out)
            return out * (self.meta.get("action_scale") or 1.0)
        if self.output == "logits" and rng is not None:
            return int(rng.choice(len(out), p=softmax(out)))
        return int(np.argmax(out))


def load_policy(name, folder=None):
    """Load research/experiments/weights/<name>.npz (arrays W0, b0, W1, b1, ...) and <name>.json
    ({"activation": "tanh" | "relu", "output": "logits" | "q" | "mean", "obs": ..., "notes": ...}).
    Layout: y = x @ W + b with W of shape (inputs, outputs). Weights saved the PyTorch way, (outputs, inputs), are
    detected from the first layer (len(b0) == W0.shape[0]) or from "layout": "out_in" in the .json, and transposed."""
    folder = Path(folder) if folder else WEIGHTS
    meta = json.loads((folder / f"{name}.json").read_text())
    with np.load(folder / f"{name}.npz") as arrays:
        n_layers = sum(1 for key in arrays.files if key.startswith("W"))
        raw = [(arrays[f"W{i}"].astype(np.float64), arrays[f"b{i}"].astype(np.float64)) for i in range(n_layers)]
    W0, b0 = raw[0]
    layout = meta.get("layout")
    if layout is None:
        layout = "out_in" if (W0.shape[1] != len(b0) and W0.shape[0] == len(b0)) else "in_out"
    params = [(W.T, b) if layout == "out_in" else (W, b) for W, b in raw]
    for (W, b), (W_next, _) in zip(params, params[1:] + [(None, None)]):
        assert W.shape[1] == len(b) and (W_next is None or W_next.shape[0] == W.shape[1]), f"{name}: bad shapes"
    return Policy(name, params, meta.get("activation", "tanh"), meta.get("output", "logits"), meta)


# ============================================================================ Module 4: search (episode 15)

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


class TicTacToe:
    """Tic-tac-toe on a 9-character string, read row by row from the top left: "X", "O" or "." per square.
    X moves first; squares are numbered 0-8."""

    EMPTY = "........."

    @staticmethod
    def to_move(board):
        return "X" if board.count("X") == board.count("O") else "O"

    @staticmethod
    def legal_moves(board):
        return [i for i, cell in enumerate(board) if cell == "."]

    @staticmethod
    def play(board, move):
        assert board[move] == ".", f"square {move} is taken"
        return board[:move] + TicTacToe.to_move(board) + board[move + 1:]

    @staticmethod
    def winner(board):
        """'X' or 'O' if someone has three in a row, 'draw' if the board is full, None while the game goes on."""
        for i, j, k in LINES:
            if board[i] != "." and board[i] == board[j] == board[k]:
                return board[i]
        return "draw" if "." not in board else None

    @staticmethod
    def show(board):
        return "\n".join(" ".join(board[r * 3:r * 3 + 3]) for r in range(3))


MIDGAME = "X...O...X"   # X took two opposite corners, O the centre; O to move must take an edge: a corner loses


@lru_cache(maxsize=None)
def minimax_value(board):
    """The game's value with perfect play from here, from X's side: +1 X wins, 0 draw, -1 O wins."""
    result = TicTacToe.winner(board)
    if result is not None:
        return {"X": 1, "O": -1, "draw": 0}[result]
    values = [minimax_value(TicTacToe.play(board, m)) for m in TicTacToe.legal_moves(board)]
    return max(values) if TicTacToe.to_move(board) == "X" else min(values)


def best_moves(board):
    """Every move that keeps the best outcome for the player to move (perfect play)."""
    sign = 1 if TicTacToe.to_move(board) == "X" else -1
    values = {m: sign * minimax_value(TicTacToe.play(board, m)) for m in TicTacToe.legal_moves(board)}
    best = max(values.values())
    return [m for m, v in values.items() if v == best]


class MCTSNode:
    """One position in the search tree. N = visits, W = total result for the player who moved INTO this node."""
    __slots__ = ("board", "move", "player", "children", "untried", "N", "W")

    def __init__(self, board, move, player, rng):
        self.board, self.move, self.player = board, move, player
        self.children = []
        moves = TicTacToe.legal_moves(board) if TicTacToe.winner(board) is None else []
        self.untried = [moves[i] for i in rng.permutation(len(moves))]
        self.N, self.W = 0, 0.0


@dataclass
class MCTSResult:
    visits: dict            # move -> visits of that child of the root
    values: dict            # move -> mean result for the player to move at the root (+1 win, 0 draw, -1 loss)
    tree_size: int          # nodes in the tree, root included
    best_move: int          # the most visited move
    root: MCTSNode


def uct_score(child, parent_visits, c):
    """UCT (Kocsis & Szepesvari, 2006): mean result + c * sqrt(ln N_parent / N_child)."""
    return child.W / child.N + c * math.sqrt(math.log(parent_visits) / child.N)


def mcts(board, n_sims=1000, c=math.sqrt(2), seed=0):
    """Monte Carlo tree search with UCT and random rollouts. Each simulation: (1) select children by UCT until a
    node has an untried move, (2) expand that move, (3) play random moves to the end, (4) back the result up the
    path. Results: +1 win, 0 draw, -1 loss for the player who made each node's move."""
    rng = np.random.default_rng(seed)
    root = MCTSNode(board, None, None, rng)
    size = 1
    for _ in range(n_sims):
        node, path = root, [root]
        while not node.untried and node.children:                          # 1. selection
            node = max(node.children, key=lambda ch: uct_score(ch, node.N, c))
            path.append(node)
        if node.untried:                                                   # 2. expansion
            move = node.untried.pop()
            player = TicTacToe.to_move(node.board)
            child = MCTSNode(TicTacToe.play(node.board, move), move, player, rng)
            node.children.append(child)
            path.append(child)
            node, size = child, size + 1
        position = node.board                                              # 3. random rollout
        while TicTacToe.winner(position) is None:
            moves = TicTacToe.legal_moves(position)
            position = TicTacToe.play(position, moves[int(rng.random() * len(moves))])
        result = TicTacToe.winner(position)
        for n in path:                                                     # 4. backup
            n.N += 1
            if n.player is not None and result != "draw":
                n.W += 1.0 if result == n.player else -1.0
    visits = {ch.move: ch.N for ch in root.children}
    values = {ch.move: ch.W / ch.N for ch in root.children}
    best = max(root.children, key=lambda ch: ch.N).move if root.children else None
    return MCTSResult(visits, values, size, best, root)


def puct_score(q, p, n_parent, n_child, c=1.5):
    """AlphaZero's selection rule: Q + c * P * sqrt(N_parent) / (1 + N_child). The network's prior P replaces
    random rollouts' blind exploration; unvisited moves with high prior get looked at first."""
    return q + c * p * math.sqrt(n_parent) / (1 + n_child)


# ============================================================================ Modules 5-6: policy gradients, LM-RL


def softmax(logits, axis=-1):
    """Probabilities from scores: exp(z) / sum exp(z), computed stably."""
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(logits, axis=-1):
    """log softmax(z) = z - log sum exp(z), computed stably."""
    z = np.asarray(logits, dtype=float)
    m = z.max(axis=axis, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=axis, keepdims=True))


def discounted_returns(rewards, gamma=0.99):
    """G_t = r_t + gamma r_(t+1) + gamma^2 r_(t+2) + ... for every step of one episode (computed backwards)."""
    G, out = 0.0, np.zeros(len(rewards))
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    """Generalised advantage estimation (Schulman et al., 2016): delta_t = r_t + gamma V(s_(t+1)) - V(s_t) and
    A_t = delta_t + gamma lam A_(t+1). lam = 0: one-step TD errors; lam = 1: returns minus V. last_value =
    V(s_T) after the last step (0 if the episode ended). Returns the advantages; returns = advantages + values."""
    values = np.append(np.asarray(values, dtype=float), last_value)
    advantages, A = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        A = delta + gamma * lam * A
        advantages[t] = A
    return advantages


def ppo_clip_objective(ratio, advantage, eps=0.2):
    """PPO's clipped surrogate (Schulman et al., 2017), elementwise: min(ratio A, clip(ratio, 1-eps, 1+eps) A).
    ratio = pi_new(a|s) / pi_old(a|s). Once the ratio leaves the band in the helpful direction, the objective goes
    flat - no more push."""
    ratio, advantage = np.asarray(ratio, dtype=float), np.asarray(advantage, dtype=float)
    return np.minimum(ratio * advantage, np.clip(ratio, 1 - eps, 1 + eps) * advantage)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def bradley_terry_prob(r_a, r_b):
    """Bradley-Terry: the chance a person prefers answer a over b, given reward scores: sigmoid(r_a - r_b)."""
    return sigmoid(np.asarray(r_a, dtype=float) - np.asarray(r_b, dtype=float))


def dpo_loss(logp_pi_chosen, logp_pi_rejected, logp_ref_chosen, logp_ref_rejected, beta=0.1):
    """DPO (Rafailov et al., 2023): -log sigmoid(beta [(log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))]).
    Arguments are sequence log-probabilities (summed over tokens). Equals log 2 = 0.693 when pi = ref."""
    margin = beta * ((np.asarray(logp_pi_chosen) - logp_ref_chosen) - (np.asarray(logp_pi_rejected) - logp_ref_rejected))
    return np.logaddexp(0.0, -margin)                      # = -log sigmoid(margin), computed stably


def dpo_implicit_reward(logp_pi, logp_ref, beta=0.1):
    """The reward DPO implicitly assigns an answer: beta (log pi(y) - log ref(y))."""
    return beta * (np.asarray(logp_pi, dtype=float) - logp_ref)


def grpo_advantages(rewards, eps=1e-6, ddof=1):
    """GRPO's group-relative advantages (Shao et al., 2024, DeepSeekMath): (r - mean(r)) / (std(r) + eps) over the
    answers sampled for ONE question. The paper writes only "std" and DeepSeek released no RL training code; the
    open implementations use the SAMPLE std (divide by G - 1): TRL's GRPOTrainer (nanstd with Bessel's correction,
    eps 1e-4) and verl (torch.std, eps 1e-6). So ddof=1 here; ddof=0 gives the population std. For G = 8 the two
    differ by sqrt(8/7) = 1.069. A group where every answer scored the same gets advantage 0 everywhere."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std(ddof=ddof) + eps)


def kl_k1(logp, logp_ref):
    """The plain KL estimator from samples of pi: log pi - log ref (unbiased, but can go negative)."""
    return np.asarray(logp, dtype=float) - logp_ref


def kl_k3(logp, logp_ref):
    """Schulman's k3 estimator of KL(pi || ref) from tokens sampled from pi: (ratio - 1) - log ratio with
    ratio = ref / pi. Unbiased and never negative; the per-token KL in GRPO's loss."""
    log_ratio = np.asarray(logp_ref, dtype=float) - np.asarray(logp, dtype=float)
    return np.expm1(log_ratio) - log_ratio


def pass_at_k(n, c, k):
    """Chen et al. (2021)'s unbiased pass@k from n samples of which c are correct: 1 - C(n-c, k) / C(n, k), the
    chance that at least one of k answers drawn without replacement is correct (computed as a stable product)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


# ============================================================================ measurements on this Mac

_measured_cache = {}


class _Ambiguous:
    def __init__(self, files):
        self.files = files


def _merge(into, data, source, origin, path=()):
    """Deep-merge one file's dict into `into`; a key two files define differently becomes _Ambiguous."""
    for key, value in data.items():
        here = path + (key,)
        if key not in into:
            into[key] = _merge({}, value, source, origin, here) if isinstance(value, dict) else value
            origin[here] = [source]
        elif isinstance(into[key], dict) and isinstance(value, dict):
            _merge(into[key], value, source, origin, here)
        elif isinstance(into[key], _Ambiguous) or into[key] != value:
            origin[here] = origin[here] + [source]
            into[key] = _Ambiguous(origin[here])
    return into


def _measurements():
    files = sorted(EXPERIMENTS.glob("*.json"))
    stamp = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    if _measured_cache.get("stamp") != stamp:
        merged, by_file, origin = {}, {}, {}
        for f in files:
            data = json.loads(f.read_text())
            by_file[f.stem] = data
            _merge(merged, data, f.name, origin)
        merged.update(by_file)                              # measured("classic", ...) always works
        _measured_cache.update(stamp=stamp, data=merged)
    return _measured_cache["data"]


def measured(*keys):
    """A number measured on this Mac, from research/experiments/*.json (classic.json, deep_rl.json, lm_rl.json,
    merged): measured("classic", "headline", "cliff_sarsa_reward") or, when the key is unique, measured("headline",
    "cliff_sarsa_reward"). A key defined differently by two files must be prefixed with the file name."""
    value = _measurements()
    for depth, key in enumerate(keys):
        if not isinstance(value, dict) or key not in value:
            available = sorted(value) if isinstance(value, dict) else type(value).__name__
            raise KeyError(f"{'/'.join(map(str, keys[:depth + 1]))} not measured; available here: {available}")
        value = value[key]
        if isinstance(value, _Ambiguous):
            raise KeyError(f"{'/'.join(map(str, keys[:depth + 1]))} is ambiguous ({', '.join(value.files)}): "
                           f"prefix it with the file name, e.g. measured('classic', ...)")
    return value


# ============================================================================ self-checks


def _self_checks():
    import time
    t0 = time.time()
    ok = lambda name: print(f"  ok  {name}")

    # --- bandits
    rng = np.random.default_rng(1)
    rs = rng.normal(size=50)
    q = 0.0
    for n, r in enumerate(rs, 1):
        q = incremental_mean(q, n, r)
    assert abs(q - rs.mean()) < 1e-12
    s = ucb_scores([0.5, 0.2, 0.0], [10, 2, 0], t=12, c=2)
    assert np.isinf(s[2]) and abs(s[0] - (0.5 + 2 * math.sqrt(math.log(12) / 10))) < 1e-12
    a, b = beta_posterior(3, 1)
    assert (a, b) == (4, 2) and abs(np.sum(beta_pdf(np.linspace(0, 1, 20001)[1:-1], a, b)) / 20000 - 1) < 1e-3
    ok("incremental mean = average; UCB formula; Beta(1,1) posterior integrates to 1")

    bandit = BernoulliBandit(ROVER_ARMS)
    assert bandit.best_arm == ROVER_BEST_ARM == 3
    runs = {m: run_bandit(bandit, m, 1000, seed=7) for m in ("greedy", "eps", "ucb", "thompson")}
    for m, run in runs.items():
        assert run.counts[-1].sum() == 1000 and np.all(np.diff(run.regret) >= -1e-12)
    assert runs["ucb"].counts[4].tolist() == [1, 1, 1, 1, 1]          # UCB tries every machine once first
    small = {m: bandit_testbed(m, runs=400, steps=1000, seed=3, probs=ROVER_ARMS) for m in ("eps", "ucb", "thompson")}
    # at 1,000 pulls UCB1 (c = sqrt 2) still explores the 0.50 machine: it overtakes eps-greedy only after ~10^4
    assert small["thompson"]["regret"][-1] < small["eps"]["regret"][-1] < small["ucb"]["regret"][-1]
    single = np.mean([run_bandit(bandit, "thompson", 300, seed=s).optimal[-100:].mean() for s in range(60)])
    batch = bandit_testbed("thompson", runs=2000, steps=300, seed=0, probs=ROVER_ARMS)["frac_optimal"][-100:].mean()
    assert abs(single - batch) < 0.06, (single, batch)
    tb = bandit_testbed("eps", runs=300, steps=1000, seed=0)
    assert 0.65 < tb["frac_optimal"][-100:].mean() < 0.92 and 1.2 < tb["mean_reward"][-100:].mean() < 1.5
    ok(f"Rover's arms, regret at 1000 pulls: Thompson < eps-greedy < UCB1 "
       f"({small['thompson']['regret'][-1]:.1f} < {small['eps']['regret'][-1]:.1f} < {small['ucb']['regret'][-1]:.1f});"
       f" run_bandit agrees with the vectorised runner; testbed eps=0.1 ≈ S&B Fig. 2.2")

    A_list, b_list = [np.eye(2)] * 3, [np.zeros(2)] * 3
    assert np.allclose(linucb_scores(A_list, b_list, np.array([0.6, 0.8]), 0.5), 0.5)
    lin = [run_study_tips("linucb", 2000, seed=s) for s in range(4)]
    lin_regret = np.mean([run["regret"][-1] for run in lin])
    avg_regret = np.mean([run_study_tips("best_on_average", 2000, seed=s)["regret"][-1] for s in range(4)])
    assert lin_regret < avg_regret / 3 and np.abs(lin[1]["theta_hat"][0] - STUDY_TIP_THETA[0]).max() < 0.1
    ok(f"LinUCB beats always-the-best-tip-on-average (mean regret {lin_regret:.0f} vs {avg_regret:.0f} over "
       f"2,000 students) and recovers the flashcards line")

    # --- MDPs
    world = ROVER_WORLD
    assert len(world.states) == 11 and world.start == (1, 1) and world.to_screen((1, 1)) == (2, 0)
    assert world.to_screen(CHARGER) == (0, 3) and world.from_screen(1, 1) == ROCK and ROCK in world.walls
    out = world.transitions((3, 1), "N")
    assert [(p, s2) for p, s2, _, _ in out] == [(0.8, (3, 2)), (0.1, (2, 1)), (0.1, (4, 1))]   # ahead, left, right
    assert world.transitions((1, 1), "W")[0][1] == (1, 1)                # bump into the edge: stay
    assert abs(world.transitions((3, 3), "E")[0][2] - 0.96) < 1e-12      # -0.04 + 1 on entering the charger
    V, history = value_iteration(world, gamma=1.0, theta=1e-4)
    for s, u in RN_UTILITIES.items():
        assert round(V[s], 3) == u, (s, V[s], u)
    assert greedy_policy(world, V) == RN_POLICY
    pi, V_pi, policies = policy_iteration(world)
    assert pi == RN_POLICY and all(abs(V_pi[s] - V[s]) < 1e-3 for s in RN_UTILITIES)
    V_eval = policy_evaluation(world, RN_POLICY, 1.0, 1e-8)
    assert all(abs(V_eval[s] - u) < 5e-4 for s, u in RN_UTILITIES.items())
    assert abs(backup(world, V_eval, (3, 1), "W") - V_eval[(3, 1)]) < 1e-6
    ok(f"Rover's world = R&N Fig. 17.3 to 3 decimals (value iteration, {len(history) - 1} sweeps at theta=1e-4) "
       f"and Fig. 17.2's policy; policy iteration agrees ({len(policies)} evaluations)")

    # --- model-free prediction and control
    rng = np.random.default_rng(0)
    episodes = [random_walk_episode(rng) for _ in range(3000)]
    v0 = {s: 0.5 for s in RANDOM_WALK_STATES}
    assert abs(rms_error(v0) - 0.2357) < 1e-4
    V_td = td0_prediction(episodes, v0, alpha=0.01)
    V_mc = mc_prediction(episodes, {s: 0.0 for s in RANDOM_WALK_STATES})
    assert rms_error(V_td) < 0.03 and rms_error(V_mc) < 0.02, (rms_error(V_td), rms_error(V_mc))
    assert n_step_return([0, 0, 1], 0.0, 0.9) == 0.81 and n_step_return([1], 2.0, 0.5) == 2.0
    V_rover = mc_prediction([play_episode(world, RN_POLICY, rng) for _ in range(2000)], {s: 0.0 for s in world.states})
    assert abs(V_rover[(1, 1)] - 0.705) < 0.03
    ok("random walk: TD(0) and MC both converge to 1/6..5/6; MC on Rover's world finds U(1,1) ≈ 0.705")

    cliff = CLIFF_WORLD
    assert cliff.transitions((1, 1), "E")[0][1:3] == ((1, 1), -100.0) and cliff.start == (1, 1)
    q_run, s_run = q_learning(cliff, 500, seed=0), sarsa(cliff, 500, seed=0)
    q_path, s_path = greedy_path(cliff, q_run.Q), greedy_path(cliff, s_run.Q)
    assert len(q_path) - 1 == 13 and all(r <= 2 for _, r in q_path)                 # the edge: 13 steps
    assert len(s_path) - 1 > 13 and max(r for _, r in s_path) >= 3                   # a safer road
    assert q_run.rewards[100:].mean() < s_run.rewards[100:].mean()
    ok(f"cliff walk: Q-learning learns the 13-step edge path, SARSA a {len(s_path) - 1}-step safe one; online "
       f"reward {q_run.rewards[100:].mean():.0f} vs {s_run.rewards[100:].mean():.0f} (S&B ≈ -50 vs -25)")
    assert shortest_path_lengths(SPARSE_MAZE, MAZE_GOAL)[SPARSE_MAZE.start] == 28 and len(SPARSE_MAZE.states) == 59
    assert shaped_reward(0.0, -5.0, -4.0, 1.0) == 1.0 and count_bonus(4, 0.2) == 0.1
    first = {m: first_success(maze_learning(m, 40, seed=s)[0].rewards) for m in ("shaping", "count_bonus")
             for s in (0,)}
    sparse_first = [first_success(maze_learning("sparse", 40, seed=s)[0].rewards) for s in range(3)]
    assert all(f is not None and f <= 20 for f in first.values()) and all(f is None or f > 5 for f in sparse_first)
    ok(f"sparse maze: 59 open cells, shortest path 28; first success: shaping ep {first['shaping']}, count bonus "
       f"ep {first['count_bonus']}, sparse {sparse_first} (None = not within 40 episodes)")

    # --- overestimation and divergence
    assert abs(expected_max_of_noisy(2) - 1 / math.sqrt(math.pi)) < 1e-6
    assert abs(expected_max_of_noisy(3) - 1.5 / math.sqrt(math.pi)) < 1e-6
    assert abs(expected_max_of_noisy(10, 2.0) - 2 * 1.538753) < 1e-5
    bias = maximization_bias(runs=2000, episodes=300, seed=0)
    assert bias["q_learning"][:30].max() > 0.85 and 0.04 < bias["double_q"][-100:].mean() < 0.10
    assert np.all(bias["q_learning"][10:150] > bias["double_q"][10:150])
    w_td, w_dp = bairds_counterexample(1000, method="td"), bairds_counterexample(1000, method="dp")
    assert np.abs(w_td[-1]).max() > 10 * np.abs(BAIRD_W0).max() and np.abs(w_dp[-1]).max() > np.abs(w_dp[0]).max()
    ok(f"E[max] of 2 / 10 noisy zeros = 0.564 / 1.539 sigma; max. bias: Q-learning goes left up to "
       f"{bias['q_learning'].max():.0%}, Double Q ends at {bias['double_q'][-100:].mean():.1%}; Baird diverges")

    # --- control environments (reference values recorded from gymnasium 1.4.0 by check_gym_ports.py)
    state = cartpole_reset(np.random.default_rng(0))
    assert np.allclose(state, CARTPOLE_REF["reset_seed0"], rtol=0, atol=1e-12)
    steps, done = 0, False
    while not done:
        state, r, done = cartpole_step(state, steps % 2)
        steps += 1
    assert steps == CARTPOLE_REF["alternate_steps"] and np.allclose(state, CARTPOLE_REF["alternate_final"], rtol=0, atol=1e-9)
    state = pendulum_reset(np.random.default_rng(0))
    total = 0.0
    for t in range(200):
        state, r = pendulum_step(state, 2.0 * math.sin(t / 10))
        total += r
    assert abs(total - PENDULUM_REF["sine_return"]) < 1e-9 and np.allclose(state, PENDULUM_REF["sine_final"], rtol=0, atol=1e-9)
    ok("CartPole-v1 and Pendulum-v1 ports reproduce gymnasium 1.4.0 trajectories to 1e-9")

    if (WEIGHTS / "_selftest.npz").exists():
        policy = load_policy("_selftest")
        x = np.array([0.01, -0.02, 0.03, 0.04])
        h = np.tanh(x @ policy.params[0][0] + policy.params[0][1])
        assert np.allclose(policy(x), h @ policy.params[1][0] + policy.params[1][1]) and policy.act(x) in (0, 1)
        assert policy.output == "logits" and len(policy.params) == 2
        ok("load_policy reads weights/_selftest.npz + .json; mlp_forward = hand-written forward pass")

    # --- search
    seen, frontier = {TicTacToe.EMPTY}, [TicTacToe.EMPTY]
    while frontier:
        board = frontier.pop()
        if TicTacToe.winner(board) is None:
            for m in TicTacToe.legal_moves(board):
                nxt = TicTacToe.play(board, m)
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    assert len(seen) == 5478 and minimax_value(TicTacToe.EMPTY) == 0
    assert sorted(best_moves(MIDGAME)) == [1, 3, 5, 7] and TicTacToe.winner("XXXOO....") == "X"
    result = mcts(MIDGAME, 2000, seed=0)
    assert result.best_move in (1, 3, 5, 7) and sum(result.visits.values()) == 2000
    assert abs(puct_score(0.5, 0.25, 16, 3, 2.0) - 1.0) < 1e-12
    ok(f"tic-tac-toe: 5,478 positions, perfect play draws; MCTS(2000) finds an edge in the mid-game "
       f"(move {result.best_move}, {result.visits[result.best_move]} visits)")

    # --- policy gradient and LM maths
    z = np.array([2.0, 1.0, 0.1])
    assert abs(softmax(z).sum() - 1) < 1e-12 and np.allclose(np.log(softmax(z)), log_softmax(z))
    assert np.allclose(discounted_returns([1, 1, 1], 0.5), [1.75, 1.5, 1.0])
    rew, val = np.array([1.0, 0.0, 2.0]), np.array([0.5, 0.4, 0.3])
    assert np.allclose(gae(rew, val, 0.9, 1.0), discounted_returns(rew, 0.9) - val)
    assert np.allclose(gae(rew, val, 0.9, 0.0), rew + 0.9 * np.append(val[1:], 0) - val)
    assert np.allclose(ppo_clip_objective([0.5, 1.0, 1.5], 1.0), [0.5, 1.0, 1.2])
    assert np.allclose(ppo_clip_objective([0.5, 1.5], -1.0), [-0.8, -1.5])
    assert bradley_terry_prob(1.0, 1.0) == 0.5 and abs(dpo_loss(-5, -7, -5, -7) - math.log(2)) < 1e-12
    adv = grpo_advantages([1, 0, 0, 1, 1, 0, 0, 0])
    assert abs(adv.sum()) < 1e-9 and abs(adv.std(ddof=1) - 1) < 1e-5 and np.all(grpo_advantages([1, 1, 1]) == 0)
    p, p_ref = softmax(np.array([1.0, 0.0, -1.0])), softmax(np.array([0.5, 0.5, 0.0]))
    exact_kl = float(np.sum(p * (np.log(p) - np.log(p_ref))))
    assert abs(float(np.sum(p * kl_k3(np.log(p), np.log(p_ref)))) - exact_kl) < 1e-12
    assert np.all(kl_k3(np.log(p), np.log(p_ref)) >= 0)
    for n, c, k in ((10, 3, 1), (10, 3, 5), (20, 0, 4), (5, 4, 2)):
        assert abs(pass_at_k(n, c, k) - (1 - math.comb(n - c, k) / math.comb(n, k))) < 1e-12
    ok("softmax, returns, GAE (lam 0 and 1), PPO clip, Bradley-Terry, DPO = log 2 at the reference, GRPO, "
       "k3 is an unbiased KL estimate, pass@k = 1 - C(n-c,k)/C(n,k)")

    if (EXPERIMENTS / "classic.json").exists():
        assert isinstance(measured("classic", "headline"), dict)
        try:
            measured("classic", "no such key")
            raise AssertionError("measured() should raise")
        except KeyError:
            pass
        ok("measured() reads research/experiments/*.json")
    print(f"all checks passed ({time.time() - t0:.1f} s)")


# Reference trajectories recorded from gymnasium 1.4.0 (see check_gym_ports.py): CartPole from reset(seed=0)
# pushing left, right, left, ... until it falls; Pendulum from reset(seed=0) with torque 2 sin(t / 10), 200 steps.
CARTPOLE_REF = dict(reset_seed0=[0.013696168732145431, -0.02302132862361297, -0.045902647606380534,
                                 -0.04834723644714709],
                    alternate_steps=39,
                    alternate_final=[-0.06701713760483866, -0.17472681876723511, -0.22520153153077166,
                                     -0.7306654439086074])
PENDULUM_REF = dict(sine_return=-1196.4401238006803, sine_final=[3.10093862708488, -7.626457923274507])

if __name__ == "__main__":
    _self_checks()
