"""Trial and Reward · Episode 11 · When Rewards Are Rare (Module 3 · Learning from Experience)

Sparse rewards make credit assignment and exploration hard. Episode 10's cliff walk paid −1 every move; a long
corridor pays nothing until the charger at its end: a fair-coin walk needs N(N+1) steps to go N cells (90 for 9, 342
for 18, 9,900 for 99: solved here from the hitting-time equations). Rover in a bigger maze (SPARSE_MAZE: 10 × 10,
59 open cells, +1 at the far corner (10,10), 0 everywhere else, γ = 0.99, 200 moves per episode, shortest route 28;
a random walk of 200 moves reaches the charger with probability 0.005, computed exactly here and = classic.json).
Q-learning with every value at 0 is a random walk until its first success: median episode 104, and 21% of 200 runs
never succeed in 300 episodes (measured on this Mac). Credit: one seeded run (seed 21, first success in episode 103,
after 20,400 moves); the one-step update credits only the last move (one cell gets 0.5); after 300 episodes and
4 successes only 4 cells have any value, and over 200 runs the success rate in the last 50 episodes is 0.9%. An
eligibility trace (λ = 0.9, for example) on the same winning walk: every visited cell keeps (γλ)^(moves since its
last visit), and the one reward updates them all at once (α · δ · z with δ = 1). Shaping: the bicycle that rode in
circles; potential-based shaping F = γΦ(s′) − Φ(s) and its theorem; the naive Φ = −distance/10 pays +0.018 for every
bump at the start and never succeeds (0 of 200 runs); Φ = γ^(d−1) (the value with no walls) makes moving closer free:
first success in episode 4. Novelty: a bonus β/√n with a VIOLET glow fading where Rover has been (visit counts of
the measured count-bonus run, seed 0, after 1, 3 and 10 episodes, recomputed and asserted); untried moves must start
optimistic at β/(1 − γ) = 1 (episode 2); starting at 0 delays the first success to episode 219; optimism alone gets
there by episode 5. The race (six variants, 200 runs each). Montezuma's Revenge (cited). Recap: shape, trace, explore.
Module 3 closes: a table of 236 numbers holds this maze; an Atari screen does not fit in a table (Episode 12).

Every number on screen is computed below from rl_reference.py (SPARSE_MAZE, shortest_path_lengths, maze_learning,
first_success, maze_potential, shaped_reward, count_bonus, MAZE_*) and asserted, or loaded with
measured("classic", "maze", …) (research/experiments/classic.md §7: 200 runs × 300 episodes, α = 0.5, ε = 0.1,
β = 0.01; the six variants' first-success medians and percentiles, success rates, greedy-path checks).

Sources (research/notes.md §11, §9, §13, Appendix A, Appendix C row 11; papers as published, read October 2026):
- Ng, Harada & Russell (1999), Policy invariance under reward transformations, ICML: Theorem 1 (F = γΦ(s′) − Φ(s)
  keeps every optimal policy optimal, and vice versa); the bicycle story is theirs, citing Randløv & Alstrøm (1998).
- Wiewiora (2003): shaping with Φ is the same as starting Q at Φ (why the naive potential teaches bumping).
- Sutton & Barto (2018) ch. 7 and 12 (n-step returns, the λ-return, eligibility traces, Fig. 7.4's one-step vs
  n-step picture); §17.4 (sparse reward, "the plateau problem").
- Montezuma's Revenge: Mnih et al. (2015) Extended Data Table 2 (DQN 0; human tester 4,367); Bellemare et al. (2016)
  §6 (pseudo-count bonus: 3,439 at 100M frames; 15 rooms vs 2); Burda et al. (2018) Table 1 (random network
  distillation 8,152, the first above average human without demonstrations); Ecoffet et al. (2021) Extended Data
  Table 3 (Go-Explore without domain knowledge, robustified policy: 43,791) and main text (with domain-knowledge cells
  and simulator restore: 1,731,645). Human baselines differ between papers (notes Appendix A): 4,367 is the DQN
  paper's tester. Curiosity as prediction error: Pathak et al. (2017).
- Atari frames are 210 × 160 pixels (Mnih et al., 2015; notes §13).

Illustrative (said "for example" or labelled): the corridor walk (seed 14, 96 steps); the maze random walk (seed 1,
200 moves, 23 cells, no charger); λ = 0.9 for the trace; the traces are drawn per cell (most recent visit; with every
value tied at 0 each move was greedy, so Watkins's Q(λ) would cut no trace); the bicycle, the Montezuma room and the
pixel grid are drawings. The jitter on the walk trails only keeps overlapping moves apart.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep11"
W = SPARSE_MAZE
GAMMA = MAZE_GAMMA
MZ = measured("classic", "maze")
RUNS, EPISODES, ALPHA, EPS, BETA = MZ["runs"], MZ["episodes"], MZ["alpha"], MZ["epsilon"], MZ["beta"]
GOAL = MAZE_GOAL
START = W.start

# ---------------------------------------------------------------------------- the numbers (all computed here)

assert (W.width, W.height, len(W.states)) == (10, 10, 59) and W.terminals == {GOAL: 1.0}
assert W.step_reward == 0.0 and W.slip == 0.0 and START == (1, 1)
assert (MAZE_MAX_STEPS, GAMMA) == (200, 0.99) == (MZ["max_moves"], MZ["gamma"])
assert (RUNS, EPISODES, ALPHA, EPS, BETA) == (200, 300, 0.5, 0.1, 0.01)
assert tuple(MZ["layout"]) == MAZE_LAYOUT

DIST = shortest_path_lengths(W, GOAL)
assert DIST[START] == 28 == MZ["shortest_path"]                                         # "twenty-eight moves"
ROUTE = [START]
while ROUTE[-1] != GOAL:
    ROUTE.append(next(W.move(ROUTE[-1], a) for a in ACTIONS if DIST.get(W.move(ROUTE[-1], a), 99) == DIST[ROUTE[-1]] - 1))
assert len(ROUTE) - 1 == 28


def corridor_steps(n):
    """Expected steps of a fair-coin walk from cell 0 to cell n (a wall at cell 0: stay put), solved exactly."""
    A, b = np.eye(n), np.ones(n)
    for i in range(n):
        for j in (max(i - 1, 0), i + 1):
            if j < n:
                A[i, j] -= 0.5
    return float(np.linalg.solve(A, b)[0])


HIT = {n: corridor_steps(n) for n in (9, 18, 99)}
assert [round(HIT[n]) for n in HIT] == [n * (n + 1) for n in HIT] == [90, 342, 9900]   # notes §11 worked example
assert round(HIT[18] / HIT[9], 1) == 3.8                                                # "about four times as long"


def corridor_walk(seed, n=9):
    rng = np.random.default_rng(seed)
    x, xs = 0, [0]
    while x < n:
        x = max(0, x + (1 if rng.random() < 0.5 else -1))
        xs.append(x)
    return xs


WALK = corridor_walk(14)                                                                # for example: 96 steps
assert len(WALK) - 1 == 96


def random_walk_reach(world, moves):
    """Probability that uniformly random moves reach the charger within `moves` (exact, by propagation)."""
    idx = {s: i for i, s in enumerate(world.states)}
    P = np.zeros((len(idx), len(idx)))
    for s in world.states:
        for a in ACTIONS:
            for p, s2, _, _ in world.transitions(s, a):
                P[idx[s], idx[s2]] += p / len(ACTIONS)
    x = np.zeros(len(idx))
    x[idx[world.start]] = 1.0
    for _ in range(moves):
        x = np.einsum("i,ij->j", x, P)
    return float(x[idx[GOAL]])


RW_REACH = random_walk_reach(W, MAZE_MAX_STEPS)
assert round(RW_REACH, 3) == 0.005 == MZ["random_walk_success_within_200"]              # "once in two hundred"


def maze_walk(seed, moves=MAZE_MAX_STEPS):
    rng = np.random.default_rng(seed)
    s, cells = START, [START]
    for _ in range(moves):
        s, _, done = W.step(s, ACTIONS[int(rng.random() * len(ACTIONS))], rng)
        cells.append(s)
        if done:
            break
    return cells


RW = maze_walk(1)                                                                       # for example: a failed walk
assert len(RW) - 1 == 200 and GOAL not in RW and len(set(RW)) == 23

# --- the measured race (classic.json maze.methods)
FS = {k: v["first_success_episode"] for k, v in MZ["methods"].items()}
assert FS["sparse"]["median"] == 104 == measured("classic", "headline", "maze_first_success_sparse")
assert FS["shaping"]["median"] == 4 == measured("classic", "headline", "maze_first_success_shaping")
assert FS["count_bonus"]["median"] == 2 == measured("classic", "headline", "maze_first_success_count_bonus")
NEVER = FS["sparse"]["never_within_300"]
assert round(100 * NEVER) == 21 == measured("classic", "headline", "maze_sparse_pct_never_in_300")
assert (round(FS["sparse"]["p10"]), round(FS["sparse"]["p90"])) == (22, 238)
assert FS["shaping_negative"]["never_within_300"] == 1.0 and FS["shaping_negative"]["median"] is None
assert FS["count_bonus_q0"]["median"] == 219 and FS["optimistic"]["median"] == 5
assert (FS["shaping"]["p10"], FS["shaping"]["p90"]) == (2, 5) and FS["shaping"]["never_within_300"] == 0
LAST50 = MZ["methods"]["sparse"]["success_rate_last50"]
assert round(100 * LAST50, 1) == 0.9                                                    # "under one percent"
assert MZ["methods"]["sparse"]["greedy_path_reaches_goal"] == 0.0
assert all(MZ["methods"][k]["greedy_path_optimal_28"] == 1.0 for k in ("shaping", "count_bonus", "optimistic"))
assert MZ["methods"]["sparse"]["moves_before_first_success_median"] == 20760


# --- one sparse run, recorded (seed 21: its first success is the median's neighbour, episode 103)
class Recorder:
    """The maze, unchanged (no slip: stepping draws no random numbers), keeping every episode's cells."""

    def __init__(self, world):
        self.world, self.paths = world, []

    def __getattr__(self, name):
        return getattr(self.world, name)

    def reset(self):
        s = self.world.reset()
        self.paths.append([s])
        return s

    def step(self, s, a, rng):
        s2, r, done = self.world.step(s, a, rng)
        self.paths[-1].append(s2)
        return s2, r, done


SEED = 21
_rec = Recorder(W)
SPARSE_RUN, _ = maze_learning("sparse", seed=SEED, world=_rec)
assert np.array_equal(SPARSE_RUN.rewards, maze_learning("sparse", seed=SEED)[0].rewards)
FIRST = first_success(SPARSE_RUN.rewards)
WIN = _rec.paths[FIRST - 1]                                                             # the first winning walk
MOVES_BEFORE = int(SPARSE_RUN.lengths[:FIRST - 1].sum())
assert FIRST == 103 and len(WIN) - 1 == 156 and WIN[-1] == GOAL and MOVES_BEFORE == 20_400
LAST_CELL = WIN[-2]
_after_first = maze_learning("sparse", episodes=FIRST, seed=SEED)[0]
V_FIRST = {s: max(q.values()) for s, q in _after_first.Q.items() if not W.is_terminal(s)}
assert {s: v for s, v in V_FIRST.items() if v > 0} == {LAST_CELL: ALPHA} == {(9, 10): 0.5}   # one cell, 0.5
SUCCESSES = int((SPARSE_RUN.rewards > 0).sum())
V_END = {s: max(q.values()) for s, q in SPARSE_RUN.Q.items() if not W.is_terminal(s)}
KNOWN = {s: v for s, v in V_END.items() if v > 0}
assert SUCCESSES == 4 and len(KNOWN) == 4 and set(KNOWN) == {(8, 10), (9, 10), (9, 9), (9, 8)}
assert len(SPARSE_RUN.Q) * len(ACTIONS) == 236                                          # the table: 59 × 4

# --- an eligibility trace on the same winning walk (λ = 0.9, for example)
LAMBDA = 0.9
DECAY = GAMMA * LAMBDA
T_END = len(WIN) - 1
LAST_VISIT = {}
for t, s in enumerate(WIN[:-1]):
    LAST_VISIT[s] = t
TRACE = {s: DECAY ** (T_END - 1 - t) for s, t in LAST_VISIT.items()}
DELTA = 1.0 + 0.0 - 0.0                                                                 # r + γ·0 (terminal) − Q (= 0)
TRACE_Q = {s: ALPHA * DELTA * z for s, z in TRACE.items()}
SHOWN = {s: z for s, z in TRACE.items() if z >= 0.01}
assert TRACE[LAST_CELL] == 1.0 and len(SHOWN) == 10 and sum(z > 0.1 for z in TRACE.values()) == 6
HALF_LIFE = math.log(0.5) / math.log(DECAY)
assert round(HALF_LIFE) == 6                                                            # "halves every six moves"

# --- shaping
DIST_M = {s: abs(s[0] - GOAL[0]) + abs(s[1] - GOAL[1]) for s in W.states}


def phi_naive(s):
    return -DIST_M[s] / 10


BUMP_NAIVE = shaped_reward(0.0, phi_naive(START), phi_naive(START), GAMMA)
assert W.move(START, "E") == START and round(BUMP_NAIVE, 3) == 0.018                    # "+0.018 per bump"
PHI = {s: maze_potential(s) for s in W.states if not W.is_terminal(s)}
assert all(round(PHI[s], 4) == MZ["potential"][f"{s[0]},{s[1]}"] for s in PHI)
F_CLOSER = shaped_reward(0.0, PHI[START], PHI[(1, 2)], GAMMA)
F_BUMP = shaped_reward(0.0, PHI[START], PHI[START], GAMMA)
F_AWAY = shaped_reward(0.0, PHI[(1, 2)], PHI[START], GAMMA)
F_ARRIVE = shaped_reward(1.0, PHI[(9, 10)], 0.0, GAMMA)
assert abs(F_CLOSER) < 1e-12 and abs(F_ARRIVE) < 1e-12
assert round(F_BUMP, 3) == -0.008 and round(F_AWAY, 3) == -0.017
PHI_LO, PHI_HI = min(PHI.values()), max(PHI.values())
assert (round(PHI_LO, 2), round(PHI_HI, 2)) == (0.84, 1.0) and PHI[START] == PHI_LO

# --- novelty: the measured count-bonus run's visit counts (seed 0), recomputed
GLOW_AFTER = (1, 3, 10)
VISITS = {k: maze_learning("count_bonus", episodes=k, seed=0)[1] for k in GLOW_AFTER}
_mv = MZ["count_bonus_visits_seed0_after_episodes"]
assert all(VISITS[k] == {tuple(map(int, c.split(","))): n for c, n in _mv[str(k)].items()} for k in GLOW_AFTER)
OPEN = [s for s in W.states if not W.is_terminal(s)]
UNSEEN_AFTER_1 = sum(VISITS[1].get(s, 0) == 0 for s in OPEN)
assert UNSEEN_AFTER_1 == 18 and len(OPEN) == 58
Q0_BONUS = BETA / (1 - GAMMA)
assert abs(Q0_BONUS - 1.0) < 1e-9 and abs(count_bonus(1, BETA) - BETA) < 1e-12
_least_10 = sorted(OPEN, key=lambda s: VISITS[10].get(s, 0))[:2]
assert set(_least_10) == {(9, 10), (9, 9)}                                              # "the far corner shines"

# --- Montezuma's Revenge, cited (research/notes.md §11; papers as published, read October 2026)
MONTEZUMA = [
    ("DQN", "Mnih et al., 2015", 0),
    ("pseudo-count bonus", "Bellemare et al., 2016", 3439),
    ("random network distillation", "Burda et al., 2018", 8152),
    ("Go-Explore", "Ecoffet et al., 2021", 43791),
]
HUMAN = 4367                     # the DQN paper's human tester (Mnih et al., 2015, Extended Data Table 2)
GO_EXPLORE_DOMAIN = 1_731_645    # with domain-knowledge cells and simulator restore (Ecoffet et al., 2021)
ATARI_FRAME = (210, 160)         # Mnih et al., 2015 (notes §13)

# ---------------------------------------------------------------------------- look

CELL = 0.58
MAZE_C = np.array([-3.55, -0.42, 0])
X0 = 0.35                         # the right panel's left edge
XR = 6.5                          # … and its right edge
WALL = "#0A0710"
ZC = 0.84                         # the zoom inset's cell
ZOOM_COLS, ZOOM_ROWS = (6, 7, 8, 9, 10), (10, 9, 8, 7)
ZOOM_C = np.array([3.35, 0.8, 0])
ARROW_ANGLE = {"N": math.pi / 2, "E": 0.0, "S": -math.pi / 2, "W": math.pi}


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.RIGHT)


def sit(mob, y):
    """Put a Text's first glyph on baseline y."""
    return mob.shift((y - mob[0].get_bottom()[1]) * m.UP)


def spark(a, b, color=SIGNAL, width=9, time_width=0.8):
    return m.ShowPassingFlash(m.Line(a, b).set_stroke(color, width), time_width=time_width)


def glow_of(n):
    """The novelty glow of a cell visited n times: proportional to the bonus of the next visit, β/√(n + 1)."""
    return mix(CARD, VIOLET, 0.85 / math.sqrt(n + 1))


def phi_tint(v):
    return mix(CARD, BLUE, 0.12 + 0.62 * (v - PHI_LO) / (PHI_HI - PHI_LO))


def swatch(color, size=0.26):
    return m.Square(size).set_stroke(EDGE, 1.5).set_fill(color, 1)


def maze_view():
    """The sparse maze as cells: walls solid, the charger GOLD (no rocks drawn: there are 41 walls)."""
    g = GridView(W.height, W.width, cell=CELL, center=MAZE_C, walls=[W.to_screen(s) for s in W.walls],
                 specials={W.to_screen(GOAL): ("charger", "", GOLD)}, gap=0.05, icon_scale=0.66)
    g.world = W
    rocks = list(g.marks)[:len(W.walls)]
    g.marks.remove(*rocks)
    for s in W.walls:
        g.squares[W.to_screen(s)].set_fill(WALL, 1).set_stroke(WALL, 0).scale(CELL / (CELL - 0.05))
    return g


class Zoom(m.VGroup):
    """The maze's far corner (columns 6-10, rows 7-10) drawn large, so numbers fit in its cells."""

    def __init__(self):
        super().__init__()
        self.squares, self.centres = {}, {}
        for i, row in enumerate(ZOOM_ROWS):
            for j, col in enumerate(ZOOM_COLS):
                s = (col, row)
                c = ZOOM_C + np.array([(j - 2) * ZC, (1.5 - i) * ZC, 0])
                sq = m.Square(ZC - 0.07).move_to(c)
                if s in W.walls:
                    sq.set_fill(WALL, 1).set_stroke(WALL, 0).scale(ZC / (ZC - 0.07))
                elif s == GOAL:
                    sq.set_stroke(GOLD, 3).set_fill(GOLD_FILL, 1)
                else:
                    sq.set_stroke(EDGE, 2).set_fill(CARD, 1)
                self.squares[s], self.centres[s] = sq, c
                self.add(sq)
        self.charger = icon("charger", ZC * 0.6).move_to(self.centres[GOAL])
        self.add(self.charger)

    def has(self, s):
        return s in self.squares and s not in W.walls and s != GOAL


def jittered(view, cells, seed, amount=0.15):
    rng = np.random.default_rng(seed)
    pts, last = [], None
    for s in cells:
        if s == last:
            continue
        pts.append(view.at(s) + np.array([*rng.uniform(-amount, amount, 2) * CELL, 0]))
        last = s
    return pts


def trail(points, color=EXPLORE, width=2.6, opacity=0.75):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width, opacity=opacity).set_fill(opacity=0)


def follow(bot, points, run_time):
    """Rover driving through `points` at an even pace per move (Create on a corner path draws at the same pace)."""

    def step(_, a):
        f = a * (len(points) - 1)
        j = min(int(f), len(points) - 2)
        t = f - j
        bot.move_to(points[j] * (1 - t) + points[j + 1] * t)

    return m.UpdateFromAlphaFunc(bot, step, run_time=run_time, rate_func=m.linear)


def bicycle(center, color=INK):
    c = np.array(center, dtype=float)
    back, front = c + [-0.42, -0.12, 0], c + [0.42, -0.12, 0]
    wheels = m.VGroup(*[m.Circle(radius=0.27).move_to(p).set_stroke(color, 3) for p in (back, front)])
    seat, crank, bars = c + [-0.12, 0.28, 0], c + [0.02, -0.12, 0], c + [0.3, 0.36, 0]
    frame = m.VMobject().set_points_as_corners([back, seat, crank, back]).set_stroke(color, 3)
    fork = m.VMobject().set_points_as_corners([crank, bars, front]).set_stroke(color, 3)
    top = m.Line(seat, bars).set_stroke(color, 3)
    return m.VGroup(wheels, frame, fork, top)


# ---------------------------------------------------------------------------- the episode


class Ep11WhenRewardsAreRare(Episode):
    number = 11

    def lesson(self):
        self.corridor()
        self.maze()
        self.credit()
        self.shaping()
        self.novelty()
        self.race()
        self.montezuma()
        self.recap()
        self.tables()

    def fade(self, *mobs, run_time=0.6):
        mobs = [mb for mb in mobs if mb is not None]
        if mobs:
            self.play(*[m.FadeOut(mb) for mb in mobs], run_time=run_time)

    def tint(self, colors, zoom=None, run_time=0.8):
        """Fill maze cells (and their zoomed copies) with colours: colors maps a cell → hex colour."""
        anims = [self.g.squares[W.to_screen(s)].animate.set_fill(c, 1) for s, c in colors.items()]
        if zoom is not None:
            anims += [zoom.squares[s].animate.set_fill(c, 1) for s, c in colors.items() if zoom.has(s)]
        return m.AnimationGroup(*anims, run_time=run_time)

    def untint(self, cells, zoom=None, run_time=0.6):
        return self.tint({s: CARD for s in cells}, zoom, run_time)

    # ------------------------------------------------------------------ 1 · one reward, far away

    def corridor(self):
        self.next_section("1 · One reward, far away", notes=(
            "Dense vs sparse. Episode 10's cliff walk (CLIFF_WORLD) pays −1 for every move: its 13-move best route "
            "hears 13 rewards. A corridor of 10 cells, Rover at cell 0, the charger at cell 9 (+1 on arrival), 0 "
            "for every other move. A fair-coin walk (a wall at cell 0: stay put) needs N(N+1) steps on average to go N "
            f"cells: {HIT[9]:.0f} for 9, {HIT[18]:.0f} for 18, {HIT[99]:,.0f} for 99 (solved exactly here; notes §11 "
            "worked example 1). The walk shown is one seeded example (seed 14, 96 steps)."))
        dense_lab = left_at(text("the cliff walk · episode 10 · every move pays", 22, MUTED), -5.5, 2.1)
        chips = m.VGroup(*[reward_chip(-1, 22) for _ in range(12)]).arrange(m.RIGHT, buff=0.12)
        left_at(chips, -5.5, 1.5)

        xs = [-4.5 + k for k in range(10)]
        cy = -0.3
        cells = m.VGroup(*[m.Square(0.94).set_stroke(EDGE, 2).set_fill(CARD, 1).move_to([x, cy, 0]) for x in xs])
        cells[-1].set_stroke(GOLD, 3).set_fill(GOLD_FILL, 1)
        charger = icon("charger", 0.58).move_to(cells[-1])
        corr_lab = left_at(text("a long corridor", 22, MUTED), -4.97, 0.6)
        sparse_pill = pill("sparse reward", GOLD, GOLD_FILL, 22).next_to(corr_lab, m.RIGHT, buff=0.35)
        sparse_pill.set_y(corr_lab.get_y())
        pays = m.VGroup(*[reward_chip(0, 20).move_to([x, cy - 0.88, 0]) for x in xs[1:-1]],
                        reward_chip(1, 20).move_to([xs[-1], cy - 0.88, 0]))
        bot = rover(0.4).move_to(cells[0]).set_z_index(4)

        steps_word = left_at(text("steps", 22, MUTED), 3.75, 0.6)
        base = steps_word[0].get_bottom()[1]

        def counter(i):
            return sit(code(str(i), 24, INK, weight=m.BOLD).next_to(steps_word, m.RIGHT, buff=0.18), base)

        steps_num = counter(0)

        def walk(_, a):
            f = a * (len(WALK) - 1)
            j = min(int(f), len(WALK) - 2)
            t = f - j
            bot.move_to([xs[WALK[j]] * (1 - t) + xs[WALK[j + 1]] * t, cy, 0])
            steps_num.become(counter(min(int(round(f)), len(WALK) - 1)))

        wander = m.UpdateFromAlphaFunc(bot, walk, run_time=4.4, rate_func=m.linear)

        head = cite("average steps to reach the charger, flipping a fair coin each step").move_to([0, -1.95, 0])
        cols = [(-3.8, 9), (0.0, 18), (3.8, 99)]
        nums = [title_text(f"{round(HIT[n]):,}", 52, EXPLORE).move_to([x, -2.55, 0]) for x, n in cols]
        labs = [text(f"{n} cells away", 22, MUTED).move_to([x, -3.2, 0]) for x, n in cols]
        times = [code(f"×{HIT[18] / HIT[9]:.1f}", 22, MUTED).move_to([-1.9, -2.55, 0]),
                 code(f"×{HIT[99] / HIT[18]:.0f}", 22, MUTED).move_to([1.9, -2.55, 0])]

        self.say(
            "[In episode ten's cliff walk], [every move paid minus one]: [a signal after every single step.]",
            m.AnimationGroup(self.new_header("01 · sparse reward", "One reward, far away"), m.FadeIn(dense_lab)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips], lag_ratio=0.15),
            m.Circumscribe(chips, color=RED, buff=0.1),
        )
        self.say(
            "[Now picture a long corridor], [with one charger at the far end], [and nothing at all along the way.]",
            m.AnimationGroup(m.FadeIn(cells), m.FadeIn(bot), m.FadeIn(corr_lab)),
            m.AnimationGroup(m.FadeIn(charger, scale=0.8), m.FadeIn(pays[-1], shift=0.15 * m.UP)),
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in pays[:-1]], lag_ratio=0.12),
        )
        self.say(
            "[That's a sparse reward]: [silence, until the very last step.]",
            m.FadeIn(sparse_pill, shift=0.1 * m.UP),
            m.Circumscribe(pays[-1], color=GOLD, buff=0.08),
        )
        self.say(
            "[With nothing to go on, Rover can only wander], [flipping a coin at every step.]",
            m.AnimationGroup(m.FadeOut(dense_lab), m.FadeOut(chips), m.FadeIn(steps_word), m.FadeIn(steps_num)),
            wander,
        )
        self.say(
            "[Nine cells away, that takes ninety steps on average.] [Twice as far takes about four times as long], "
            "[and ninety-nine cells take nearly ten thousand.]",
            m.AnimationGroup(m.FadeIn(head), m.FadeIn(nums[0], shift=0.15 * m.UP), m.FadeIn(labs[0])),
            m.AnimationGroup(m.FadeIn(times[0]), m.FadeIn(nums[1], shift=0.15 * m.UP), m.FadeIn(labs[1])),
            m.AnimationGroup(m.FadeIn(times[1]), m.FadeIn(nums[2], shift=0.15 * m.UP), m.FadeIn(labs[2])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the maze

    def maze(self):
        self.next_section("2 · Rover in a bigger maze", notes=(
            "SPARSE_MAZE (rl_reference): 10 × 10, 59 open cells, Rover starts at (1,1), the charger at (10,10) pays +1, "
            "every other move 0, no slip, γ = 0.99, episodes end after 200 moves. Shortest route 28 moves (BFS). "
            f"Uniformly random moves reach the charger within 200 moves with probability {RW_REACH:.4f} (computed "
            "exactly here; classic.json: 0.005). The walk drawn is seed 1: 200 moves, 23 distinct cells, no charger. "
            "Q-learning (Episode 10) with Q = 0 everywhere and ties broken at random behaves exactly like a random "
            "walk until the first reward. Measured on this Mac (classic.md §7): 200 runs × 300 episodes, α = 0.5, "
            "ε = 0.1: first success median episode 104 (10th–90th percentile 22–238; median 20,760 moves before it); "
            "21% of runs never succeed within 300 episodes."))
        g = self.g = maze_view()
        bot = self.bot = rover(0.26).move_to(g.at(START)).set_z_index(6)
        route = g.world_path(ROUTE, color=ACTION, width=4).set_z_index(2)
        rw_pts = jittered(g, RW, seed=3)
        rw_trail = trail(rw_pts).set_z_index(3)

        f1 = left_at(text("10 × 10 maze · 59 open cells", 24, INK), X0, 2.15)
        f2 = left_at(text("+1 at the far corner, 0 everywhere else", 24, INK, t2c={"+1": GOLD}), X0, 1.65)
        f3 = left_at(text("shortest route: 28 moves", 24, ACTION), X0, 1.15)
        f4 = left_at(text("200 random moves: 0.5% reach it", 24, EXPLORE), X0, 0.65)
        q1 = left_at(code("Q(s, a) = 0 for every cell and move", 22, VALUE), X0, 0.0)
        q2 = left_at(text("so every choice is a coin flip", 22, MUTED), X0, -0.45)

        TX0, TX1, ty = 0.55, 4.45, -1.85

        def tx(e):
            return TX0 + (TX1 - TX0) * e / EPISODES

        t_title = left_at(cite(f"first success · {RUNS} runs · measured on this Mac"), X0, -0.95)
        axis = m.Line([TX0, ty - 0.3, 0], [TX1 + 0.1, ty - 0.3, 0]).set_stroke(FAINT, 2.5)
        ticks = m.VGroup()
        for e in (0, 100, 200, 300):
            p = np.array([tx(e), ty - 0.3, 0])
            ticks.add(m.Line(p, p + [0, -0.09, 0]).set_stroke(FAINT, 2), code(str(e), 20, MUTED).next_to(p, m.DOWN, buff=0.14))
        sp = FS["sparse"]
        box = m.Rectangle(width=tx(sp["p90"]) - tx(sp["p10"]), height=0.4).set_stroke(VALUE, 2).set_fill(BLUE_FILL, 1)
        box.move_to([(tx(sp["p10"]) + tx(sp["p90"])) / 2, ty, 0])
        med_line = m.Line([tx(sp["median"]), ty - 0.26, 0], [tx(sp["median"]), ty + 0.26, 0]).set_stroke(INK, 5)
        med_lab = text(f"median: episode {sp['median']:.0f}", 22, INK, weight=m.BOLD).move_to([tx(sp["median"]), ty + 0.47, 0])
        never = pill(f"never · {100 * NEVER:.0f}%", RED, RED_FILL, 22, mono=True).move_to([5.55, ty, 0])
        t_cap = left_at(cite(f"box: 10th to 90th percentile ({sp['p10']:.0f} to {sp['p90']:.0f}) · episodes"), X0, -2.9)
        t_par = left_at(cite(f"α {ALPHA} · ε {EPS} · γ {GAMMA} · {MAZE_MAX_STEPS} moves per episode"), X0, -3.25)

        self.say(
            "[Now give Rover a bigger world]: [a ten by ten maze], [with one charger in the far corner.]",
            m.AnimationGroup(self.new_header("02 · the maze", "Rover in a bigger maze"), m.FadeIn(g), m.FadeIn(bot)),
            m.FadeIn(f1, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(g.squares[W.to_screen(GOAL)], color=GOLD, buff=0.04),
                             m.FadeIn(f2, shift=0.1 * m.UP)),
        )
        self.say(
            "[The shortest way there is twenty-eight moves]. [A random walk of two hundred moves finds it once in two hundred tries.]",
            m.AnimationGroup(m.Create(route, run_time=1.8), m.FadeIn(f3, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(rw_trail, rate_func=m.linear, run_time=3.8), follow(bot, rw_pts, 3.8),
                             m.FadeIn(f4, shift=0.1 * m.UP)),
        )
        self.say(
            "[And Q-learning starts out no better]: [with every value at zero, each move is a coin flip.]",
            m.AnimationGroup(m.FadeOut(route), m.FadeOut(rw_trail), bot.animate.move_to(g.at(START)),
                             m.FadeIn(q1, shift=0.1 * m.UP)),
            m.FadeIn(q2, shift=0.1 * m.UP),
        )
        self.say(
            "[Over two hundred runs on this Mac], [its first success came, at the median, in episode one hundred and four],",
            m.AnimationGroup(m.FadeIn(t_title), m.Create(axis), m.FadeIn(ticks)),
            m.AnimationGroup(m.FadeIn(box), m.Create(med_line), m.FadeIn(med_lab, shift=0.1 * m.UP), m.FadeIn(t_cap)),
        )
        self.say(
            "[and in twenty-one percent of runs], [it never came at all, in three hundred episodes.]",
            m.FadeIn(never, shift=0.15 * m.LEFT),
            m.AnimationGroup(m.Circumscribe(never, color=RED, buff=0.08), m.FadeIn(t_par)),
        )
        self.wait(0.3)
        self.fade(f1, f2, f3, f4, q1, q2, t_title, axis, ticks, box, med_line, med_lab, never, t_cap, t_par)

    # ------------------------------------------------------------------ 3 · credit

    def credit(self):
        self.next_section("3 · Which move earned it?", notes=(
            f"One recorded sparse run (maze_learning('sparse', seed={SEED})): first success in episode {FIRST} after "
            f"{MOVES_BEFORE:,} moves (the 200-run median is 104 and 20,760); the winning walk is {len(WIN) - 1} moves. "
            "Before it every Q is 0, so the only non-zero update is the last one: Q((9,10), E) = 0 + α(1 − 0) = 0.5 "
            f"(asserted). After {EPISODES} episodes this run has {SUCCESSES} successes and {len(KNOWN)} cells with any "
            "value: " + ", ".join(f"{s} {v:.2f}" for s, v in sorted(KNOWN.items())) + ". Values spread one cell per "
            "visit (bootstrapping), but Rover rarely revisits the corner. Over 200 runs: success in the last 50 "
            f"episodes {100 * LAST50:.2f}%, and the greedy route reaches the charger in 0 runs (classic.md §7). Trace "
            f"(for example λ = 0.9, γλ = {DECAY:.3f}, halving every {HALF_LIFE:.1f} moves): per cell z = (γλ)^(moves "
            "since its last visit) on the same winning walk; the final TD error δ = 1 updates every cell by α·δ·z "
            "(S&B 2018 ch. 12; with all values tied each move was greedy, so Watkins's Q(λ) cuts no trace). λ = 0 is "
            "one-step TD (Episode 9), λ = 1 behaves like Monte Carlo (Episode 8); the λ-return averages Episode 9's "
            "n-step returns (S&B eq. 12.2). Fig. 7.4 of S&B draws the same one-step vs multi-step contrast."))
        g, bot = self.g, self.bot
        win_pts = jittered(g, WIN, seed=5)
        win_trail = trail(win_pts).set_z_index(3)
        zoom = Zoom()
        corner_tl = g.squares[W.to_screen((6, 10))].get_corner(m.UL) + np.array([-0.025, 0.025, 0])
        corner_br = g.squares[W.to_screen((10, 7))].get_corner(m.DR) + np.array([0.025, -0.025, 0])
        frame = m.Rectangle(width=corner_br[0] - corner_tl[0], height=corner_tl[1] - corner_br[1])
        frame.move_to((corner_tl + corner_br) / 2).set_stroke(INK, 2.5).set_fill(opacity=0).set_z_index(4)
        z_tl = zoom.squares[(6, 10)].get_corner(m.UL) + np.array([-0.035, 0.035, 0])
        z_bl = zoom.squares[(6, 7)].get_corner(m.DL) + np.array([-0.035, -0.035, 0])
        zoom_frame = m.Rectangle(width=5 * ZC, height=4 * ZC).move_to(ZOOM_C).set_stroke(INK, 2.5).set_fill(opacity=0)
        leaders = m.VGroup(m.Line([corner_br[0], corner_tl[1], 0], z_tl), m.Line(corner_br, z_bl)).set_stroke(MUTED, 1.5)
        zoom_lab = left_at(cite("the far corner, up close"), zoom_frame.get_left()[0], zoom_frame.get_bottom()[1] - 0.3)

        r1 = left_at(text(f"this run: first success in episode {FIRST}", 24, INK), X0, -1.95)
        r1b = left_at(cite(f"after {MOVES_BEFORE:,} moves · this walk: {len(WIN) - 1} moves"), X0, -2.42)
        r2 = left_at(text(f"{EPISODES} episodes, {SUCCESSES} successes: {len(KNOWN)} cells know", 24, INK),
                     X0, -1.95)
        stat_num = left_at(title_text(f"{100 * LAST50:.1f}%", 46, REWARD), X0, -2.85)
        stat_lab = m.VGroup(text("success in the last 50 episodes", 22, INK),
                            cite(f"{RUNS} runs · measured on this Mac")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        stat_lab.next_to(stat_num, m.RIGHT, buff=0.35)

        def znum(s, v, color, size=22):
            return code(f"{v:.2f}", size, color, weight=m.BOLD).move_to(zoom.centres[s]).set_z_index(5)

        first_num = znum(LAST_CELL, V_FIRST[LAST_CELL], INK)
        end_nums = {s: znum(s, v, INK) for s, v in KNOWN.items()}
        gc, zc = g.at(GOAL), zoom.centres[GOAL]
        order = [(9, 10), (8, 10), (9, 9), (9, 8)]
        chain = m.Succession(*[m.AnimationGroup(spark(zoom.centres[a] if a != GOAL else zc, zoom.centres[b]), run_time=0.5)
                               for a, b in ((GOAL, (9, 10)), ((9, 10), (8, 10)), ((9, 10), (9, 9)), ((9, 9), (9, 8)))])

        # the trace on the same winning walk
        rule1 = left_at(code("trace z ← γλ·z, and 1 where Rover is", 22, SIGNAL), X0, -1.95)
        rule1b = left_at(cite(f"λ = {LAMBDA}, for example: a trace halves every {HALF_LIFE:.0f} moves"), X0, -2.4)
        rule2 = left_at(code("then Q ← Q + α·δ·z for every cell", 22, VALUE), X0, -2.95)
        z_nums = {s: znum(s, z, INK) for s, z in SHOWN.items() if zoom.has(s)}
        q_nums = {s: znum(s, TRACE_Q[s], INK) for s in z_nums}
        trace_tint = {s: mix(CARD, PINK, 0.9 * z ** 0.7) for s, z in SHOWN.items()}
        value_tint = {s: heat(v, 0.0, 0.5) for s, v in TRACE_Q.items() if s in SHOWN}
        fan = m.AnimationGroup(*[spark(zc, zoom.centres[s], SIGNAL, 7) for s in z_nums])

        # λ dial: one step (episode 9) … the whole walk (episode 8)
        DX0, DX1, dy = X0 + 0.25, XR - 0.25, -2.4
        dial = m.Line([DX0, dy, 0], [DX1, dy, 0]).set_stroke(FAINT, 4)
        ends = m.VGroup(m.Dot([DX0, dy, 0], radius=0.07, color=MUTED), m.Dot([DX1, dy, 0], radius=0.07, color=MUTED))
        dial_l = left_at(m.VGroup(code("λ = 0", 20, MUTED), cite("one step · episode 9")).arrange(m.DOWN, buff=0.08,
                                                                                               aligned_edge=m.LEFT), DX0 - 0.1, dy - 0.45)
        dial_r = right_at(m.VGroup(code("λ = 1", 20, MUTED), cite("whole walk · episode 8")).arrange(m.DOWN, buff=0.08,
                                                                                                aligned_edge=m.RIGHT), DX1 + 0.1, dy - 0.45)
        knob_x = DX0 + LAMBDA * (DX1 - DX0)
        knob = m.Dot([knob_x, dy, 0], radius=0.13, color=SIGNAL).set_z_index(4)
        knob_lab = code(f"λ = {LAMBDA}", 22, SIGNAL, weight=m.BOLD).move_to([knob_x, dy + 0.38, 0])
        dial_parts = m.VGroup(dial, ends, dial_l, dial_r)

        self.say(
            "[Finding the charger is only half the problem]: [this run found it in episode one hundred and three.]",
            m.AnimationGroup(self.new_header("03 · credit", "Which move earned it?"), m.FadeIn(r1, shift=0.1 * m.UP),
                             m.FadeIn(r1b)),
            m.AnimationGroup(m.Create(win_trail, rate_func=m.linear, run_time=3.6), follow(bot, win_pts, 3.6)),
        )
        self.say(
            "[Which of its hundred and fifty-six moves earned the reward?] [The one-step update credits only the last one.]",
            m.AnimationGroup(m.FadeIn(zoom), m.FadeIn(zoom_frame), m.Create(frame), m.Create(leaders), m.FadeIn(zoom_lab),
                             m.FadeOut(bot)),
            m.AnimationGroup(spark(zc, zoom.centres[LAST_CELL]), spark(gc, g.at(LAST_CELL)),
                             self.tint({LAST_CELL: heat(V_FIRST[LAST_CELL], 0.0, 1.0)}, zoom),
                             m.FadeIn(first_num, scale=0.8)),
        )
        self.say(
            "[The news travels back one cell per visit], [but Rover rarely gets back to this corner]: "
            "[after three hundred episodes, only four cells know.]",
            m.AnimationGroup(m.FadeOut(win_trail), chain),
            m.AnimationGroup(m.FadeOut(r1), m.FadeOut(r1b), m.FadeIn(r2, shift=0.1 * m.UP)),
            m.AnimationGroup(self.tint({s: heat(v, 0.0, 1.0) for s, v in KNOWN.items()}, zoom),
                             m.FadeOut(first_num), *[m.FadeIn(n, scale=0.8) for n in end_nums.values()]),
        )
        self.say(
            "[So it barely improves]: [over two hundred runs, under one percent of its last fifty episodes succeed.]",
            m.FadeIn(stat_num, shift=0.15 * m.UP),
            m.FadeIn(stat_lab, shift=0.1 * m.UP),
        )
        win_trail2 = trail(win_pts, opacity=0.45).set_z_index(3)
        self.say(
            "[An eligibility trace spreads the credit]: [every cell Rover passed keeps a memory of the visit, fading with time.]",
            m.AnimationGroup(m.FadeOut(r2), m.FadeOut(stat_num), m.FadeOut(stat_lab), self.untint(KNOWN, zoom),
                             *[m.FadeOut(n) for n in end_nums.values()], m.FadeIn(win_trail2), m.FadeIn(rule1, shift=0.1 * m.UP),
                             m.FadeIn(rule1b)),
            m.AnimationGroup(self.tint(trace_tint, zoom, run_time=1.4),
                             m.LaggedStart(*[m.FadeIn(z_nums[s], scale=0.8) for s in sorted(z_nums, key=lambda c: -SHOWN[c])],
                                           lag_ratio=0.15)),
        )
        self.say(
            "[When the reward arrives, one update reaches them all at once], [most for the latest moves.]",
            m.AnimationGroup(fan, m.FadeIn(rule2, shift=0.1 * m.UP)),
            m.AnimationGroup(self.tint(value_tint, zoom), *[m.Transform(z_nums[s], q_nums[s]) for s in z_nums]),
        )
        self.say(
            "[It blends the multi-step returns of episode nine]: [lambda sets how far back the credit reaches.]",
            m.AnimationGroup(m.FadeOut(rule1), m.FadeOut(rule1b), m.FadeOut(rule2), m.FadeIn(dial_parts)),
            m.AnimationGroup(m.FadeIn(knob, target_position=[DX0, dy, 0], run_time=1.4), m.FadeIn(knob_lab, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        bot.move_to(g.at(START))
        self.fade(win_trail2, zoom, zoom_frame, frame, leaders, zoom_lab, *z_nums.values(), dial_parts, knob, knob_lab)
        self.play(self.untint(SHOWN), run_time=0.5)

    # ------------------------------------------------------------------ 4 · shaping

    def shaping(self):
        self.next_section("4 · Hints that keep the goal", notes=(
            "Reward shaping adds F to the reward. Careless shaping changes the best policy: Ng, Harada & Russell (1999) "
            "tell of a bicycle learner (Randløv & Alstrøm, 1998) rewarded for progress toward the goal and never charged "
            "for riding away: 'the agent learned to ride in tiny circles near the start state'. Potential-based shaping "
            "F = γΦ(s′) − Φ(s); along an episode the payments telescope: Σ γ^t F_t = γ^T Φ(s_T) − Φ(s_0), with Φ = 0 at "
            "the end every route shifts by −Φ(start). Theorem 1 (Ng, Harada & Russell 1999): every optimal policy with F "
            "is optimal without it, and vice versa. It promises the same optimum, not faster learning (notes §11). "
            f"Naive Φ = −distance/10 (classic.md §7): a bump at the start (distance 18) pays γΦ − Φ = +{BUMP_NAIVE:.3f}; "
            "shaping with Φ equals starting Q at Φ (Wiewiora 2003), below the value of standing still, so Rover learns "
            "to bump into walls: 0 of 200 runs succeed. Working Φ = γ^(d−1), d = Manhattan distance to the charger (the "
            f"value of a cell with no walls; maze_potential, = classic.json maze.potential): from {PHI_LO:.2f} at the "
            f"start to {PHI_HI:.2f} beside the charger. Moving closer pays exactly 0 (arriving too: 1 + 0 − 1), a bump "
            f"{F_BUMP:.4f}, a move away {F_AWAY:.4f} near the start. First success: median episode 4 (2–5), 0 runs "
            "never, the greedy route = 28 moves in every run. GOLD arrows: the moves that bring Rover closer (N or E, "
            "where no wall blocks)."))
        g, bot = self.g, self.bot
        def hint_arrows():
            group = m.VGroup()
            for s in OPEN:
                for d in ("N", "E"):
                    s2 = W.move(s, d)
                    if s2 != s and DIST_M[s2] < DIST_M[s]:
                        v = np.array([*GridView.DIRS[d], 0], dtype=float) * 0.36 * CELL
                        p = g.at(s)
                        group.add(m.Arrow(p, p + v, buff=0, color=REWARD, stroke_width=3.5,
                                          max_tip_length_to_length_ratio=0.45).set_z_index(3))
            return group

        arrows, arrows2 = hint_arrows(), hint_arrows()
        s_title = left_at(text("shaping: extra reward for progress", 26, REWARD, weight=m.BOLD), X0, 2.15)

        bike = bicycle([1.2, 0.85, 0])
        loop = m.Arc(radius=0.68, start_angle=0.5, angle=2 * math.pi - 0.9, arc_center=[1.2, 0.85, 0])
        loop.set_stroke(REWARD, 4).add_tip(tip_length=0.2)
        loop.set_color(REWARD)
        bike_txt = m.VGroup(text("paid for heading toward its goal,", 22, INK),
                            text("never charged for riding away:", 22, INK),
                            text("it rode in tiny circles", 22, RED, weight=m.BOLD)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(bike_txt, 2.15, 0.9)
        bike_cite = left_at(cite("Randløv & Alstrøm, 1998 · via Ng et al., 1999"), X0, -0.25)
        bike_all = m.VGroup(bike, loop, bike_txt, bike_cite)

        formula = m.MathTex(r"F", "=", r"\gamma", r"\Phi(s')", "-", r"\Phi(s)", font_size=46, color=INK)
        formula[0].set_color(REWARD)
        formula[3].set_color(VALUE)
        formula[5].set_color(VALUE)
        left_at(formula, X0 + 0.1, 1.3)
        f_cap = left_at(cite("Φ: a potential per cell · s → s′: one move"), X0, 0.7)
        tele = m.MathTex(r"\sum_t \gamma^t F_t", "=", r"\gamma^T \Phi(s_T)", "-", r"\Phi(s_0)", font_size=38, color=INK)
        tele[0].set_color(REWARD)
        left_at(tele, X0 + 0.1, -0.15)
        tele_cap = left_at(text("only the ends remain: every route shifts alike", 22, INK), X0, -0.85)
        thm_box = card(XR - X0, 1.5, VALUE, CARD, radius=0.16, stroke=2).move_to([(X0 + XR) / 2, -2.25, 0])
        thm = m.VGroup(text("With F added, every optimal policy", 22, INK, weight=m.BOLD),
                       text("stays optimal, and vice versa.", 22, INK, weight=m.BOLD),
                       cite("Ng, Harada & Russell, 1999, Theorem 1")).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        thm.move_to(thm_box).align_to(thm_box, m.LEFT).shift(0.3 * m.RIGHT)
        theorem = m.VGroup(thm_box, thm)

        naive = left_at(text("the obvious one: Φ = −distance ÷ 10", 24, RED, weight=m.BOLD), X0, -0.15)
        bump_lab = left_at(text("a bump at the start pays γΦ − Φ =", 22, INK), X0, -0.75)
        bump_chips = m.VGroup(*[pill(f"+{BUMP_NAIVE:.3f}", REWARD, GOLD_FILL, 20, mono=True) for _ in range(3)])
        bump_chips.arrange(m.RIGHT, buff=0.15)
        left_at(bump_chips, X0, -1.35)
        never = left_at(title_text(f"0 of {RUNS}", 46, RED), X0, -2.3)
        never_lab = m.VGroup(text("runs reached the charger", 22, INK), cite("measured on this Mac")).arrange(
            m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(never, m.RIGHT, buff=0.35)
        p0 = g.at(START)
        bumps = m.Succession(*[bot.animate(run_time=0.55, rate_func=m.there_and_back).move_to(p0 + [0.17, 0, 0])
                               for _ in range(3)])

        good = left_at(m.MathTex(r"\Phi(s)", "=", r"\gamma^{\,d(s)\,-\,1}", font_size=44, color=VALUE), X0 + 0.1, -0.15)
        good_cap = left_at(text("a cell's value if the maze had no walls", 22, INK), X0, -0.8)
        legend = m.VGroup(swatch(phi_tint(PHI_LO)), text(f"{PHI_LO:.2f} at the start", 20, MUTED),
                          swatch(phi_tint(PHI_HI)), text(f"{PHI_HI:.2f} next to the charger", 20, MUTED))
        legend.arrange(m.RIGHT, buff=0.15)
        legend[2].shift(0.3 * m.RIGHT)
        legend[3].shift(0.3 * m.RIGHT)
        left_at(legend, X0, -1.3)
        rows_src = [("move closer", F_CLOSER, REWARD), ("bump a wall", F_BUMP, MUTED), ("move away", F_AWAY, MUTED)]
        f_rows = m.VGroup()
        for i, (lab, v, col) in enumerate(rows_src):
            y = -1.95 - 0.42 * i
            f_rows.add(left_at(text(lab, 22, col), X0, y),
                       right_at(code(minus(f"{v:+.3f}") if abs(v) > 1e-12 else "0.000", 22, col, weight=m.BOLD), 3.55, y))
        win = m.VGroup(text("first success", 22, MUTED), title_text(f"episode {FS['shaping']['median']:.0f}", 40, GOOD))
        win.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        left_at(win, 4.0, -2.25)
        win_cite = left_at(cite(f"median · {RUNS} runs"), 4.0, -3.0)
        phi_cells = self.tint({s: phi_tint(PHI[s]) for s in OPEN})

        self.say(
            "[A second fix gives hints along the way]: [extra reward for moving toward the goal. That's reward shaping.]",
            m.AnimationGroup(self.new_header("04 · shaping", "Hints that keep the goal"), m.FadeIn(s_title)),
            m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.02),
        )
        self.say(
            "[Careless hints backfire]: [one bicycle learner, paid for heading toward its goal, learned to ride in tiny circles.]",
            m.AnimationGroup(m.FadeOut(arrows), m.FadeIn(bike), m.FadeIn(bike_cite)),
            m.AnimationGroup(m.Create(loop, run_time=1.6), m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in bike_txt],
                                                                        lag_ratio=0.5)),
        )
        self.say(
            "[The safe kind pays the change in a potential]: [gamma times the potential where Rover lands, minus where it was.]",
            m.AnimationGroup(m.FadeOut(bike_all), m.FadeIn(formula[0:2], shift=0.1 * m.UP), m.FadeIn(bot)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in formula[2:]], lag_ratio=0.25),
                             m.FadeIn(f_cap)),
        )
        self.say(
            "[Summed along any route, the payments collapse to its two ends], "
            "[so the best policy stays the best: a theorem from nineteen ninety-nine.]",
            m.AnimationGroup(m.FadeIn(tele, shift=0.1 * m.UP), m.FadeIn(tele_cap, shift=0.1 * m.UP)),
            m.FadeIn(theorem, shift=0.1 * m.UP),
        )
        self.say(
            "[But the obvious potential, minus the distance, fails here]: [with gamma below one, every bump into a wall pays a little.]",
            m.AnimationGroup(m.FadeOut(tele), m.FadeOut(tele_cap), m.FadeOut(theorem), m.FadeIn(naive, shift=0.1 * m.UP)),
            m.AnimationGroup(bumps, m.FadeIn(bump_lab),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in bump_chips], lag_ratio=0.6, run_time=1.65)),
        )
        self.say(
            "[So Rover learns to bump into walls]: [in two hundred runs, it never once reached the charger.]",
            m.Succession(*[bot.animate(run_time=0.4, rate_func=m.there_and_back).move_to(p0 + [0.17, 0, 0]) for _ in range(3)]),
            m.AnimationGroup(m.FadeIn(never, shift=0.15 * m.UP), m.FadeIn(never_lab)),
        )
        self.say(
            "[The potential that works is each cell's value if the maze had no walls]: "
            "[gamma raised to one less than its distance.]",
            m.AnimationGroup(m.FadeOut(naive), m.FadeOut(bump_lab), m.FadeOut(bump_chips), m.FadeOut(never),
                             m.FadeOut(never_lab), phi_cells, m.FadeIn(good_cap, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(good, shift=0.1 * m.UP), m.FadeIn(legend)),
        )
        self.say(
            "[Now moving closer costs nothing, and every other move costs a little]: [the first success comes in episode four.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in arrows2], lag_ratio=0.01),
                             m.LaggedStart(*[m.FadeIn(r) for r in f_rows], lag_ratio=0.12)),
            m.AnimationGroup(m.FadeIn(win, shift=0.1 * m.UP), m.FadeIn(win_cite)),
        )
        self.wait(0.3)
        self.fade(arrows2, s_title, formula, f_cap, good, good_cap, legend, f_rows, win, win_cite)
        self.play(self.untint(OPEN), run_time=0.5)

    # ------------------------------------------------------------------ 5 · novelty

    def novelty(self):
        self.next_section("5 · A bonus for the unknown", notes=(
            "Count-based exploration bonus: the reward Rover learns from gets + β/√n, n = visits to the cell it lands in "
            "(this visit included; count_bonus, β = 0.01). Glow per cell = the bonus of the NEXT visit, β/√(n + 1), "
            "drawn as a VIOLET tint (0.85/√(n+1) of full strength). Visit counts are those of the measured count-bonus "
            "run (seed 0) after 1, 3 and 10 episodes (classic.json maze.count_bonus_visits_seed0_after_episodes), "
            f"recomputed here with maze_learning and asserted: after 1 episode {UNSEEN_AFTER_1} of 58 open cells are "
            "still unvisited; after 10 the least-visited are (9,10) and (9,9) beside the charger. Untried moves start "
            "at Q = β/(1 − γ) = 1, the value of the bonus collected forever (MBIE-EB style; classic.md §7). Measured "
            "medians, 200 runs: bonus with Q starting at 1: episode 2 (1–5); bonus with Q starting at 0: 219 "
            "(143–251), Rover circles where it has been; optimism alone (Q starts at 1, no bonus): 5 (3–8) — "
            "Episode 3's optimism in the face of uncertainty. Before the first success the bonus variants' Q-values "
            "all scale with β, so these numbers do not depend on β (classic.md)."))
        g, bot = self.g, self.bot
        n_title = left_at(code("bonus = β / √n", 30, EXPLORE, weight=m.BOLD), X0, 2.15)
        n_cap = left_at(cite(f"β = {BETA} · n = visits to the cell, this one included"), X0, 1.62)
        leg = m.VGroup()
        for n, lab in ((0, "unvisited"), (3, "3 visits"), (50, "50 visits")):
            leg.add(m.VGroup(swatch(glow_of(n)), text(lab, 20, MUTED)).arrange(m.RIGHT, buff=0.14))
        leg.arrange(m.RIGHT, buff=0.45)
        left_at(leg, X0, 1.1)
        after = {k: left_at(text(f"after {k} episode{'s' if k > 1 else ''}", 26, INK, weight=m.BOLD), X0, 0.35)
                 for k in GLOW_AFTER}
        after[0] = left_at(text("before the first episode", 26, INK, weight=m.BOLD), X0, 0.35)
        opt = left_at(code(f"Q starts at β / (1 − γ) = {Q0_BONUS:.0f}", 24, VALUE, weight=m.BOLD), X0, -0.45)
        opt_cap = left_at(cite("untried moves look new: a bonus collected forever"), X0, -0.92)

        def row(label, value, color, y):
            return m.VGroup(left_at(text(label, 22, color), X0, y), right_at(code(value, 24, color, weight=m.BOLD), XR, y))

        r_zero = row("bonus, Q starts at 0", f"episode {FS['count_bonus_q0']['median']:.0f}", RED, -1.6)
        r_bonus = row("bonus, Q starts at 1", f"episode {FS['count_bonus']['median']:.0f}", EXPLORE, -2.15)
        r_opt = row("no bonus, Q starts at 1", f"episode {FS['optimistic']['median']:.0f}", VALUE, -2.65)
        r_cite = left_at(cite(f"median first success · {RUNS} runs · optimism: ep. 3"), X0, -3.2)

        glow_all = self.tint({s: glow_of(0) for s in OPEN}, run_time=1.0)

        def glow_after(k, run_time=0.9):
            return self.tint({s: glow_of(VISITS[k].get(s, 0)) for s in OPEN}, run_time=run_time)

        self.say(
            "[A third fix pays for novelty]: [every cell glows with a bonus, beta over the square root of its visit count.]",
            m.AnimationGroup(self.new_header("05 · novelty", "A bonus for the unknown"), glow_all),
            m.AnimationGroup(m.FadeIn(n_title, shift=0.1 * m.UP), m.FadeIn(n_cap), m.FadeIn(leg), m.FadeIn(after[0])),
        )
        self.say(
            "[Where Rover has been, the glow fades]: [after one episode], [after three], [and after ten, the far corner shines brightest.]",
            m.Wait(0.3),
            m.AnimationGroup(glow_after(1), m.Transform(after[0], after[1])),
            m.AnimationGroup(glow_after(3), m.Transform(after[0], after[3])),
            m.AnimationGroup(glow_after(10), m.Transform(after[0], after[10]),
                             m.Circumscribe(m.VGroup(g.squares[W.to_screen((9, 10))], g.squares[W.to_screen((9, 9))]),
                                            color=EXPLORE, buff=0.05)),
        )
        self.say(
            "[One detail matters]: [untried moves must start optimistic], [at beta over one minus gamma: a bonus collected forever.]",
            m.Wait(0.2),
            m.FadeIn(opt, shift=0.1 * m.UP),
            m.FadeIn(opt_cap, shift=0.1 * m.UP),
        )
        self.say(
            "[Start them at zero, and Rover circles where it has been]: "
            "[the first success slips to episode two hundred and nineteen.]",
            m.FadeIn(r_zero[0], shift=0.1 * m.UP),
            m.FadeIn(r_zero[1], shift=0.1 * m.UP),
        )
        self.say(
            "[With optimism, it comes in episode two]. "
            "[To be fair, optimism alone, episode three's idea, gets there by about episode five.]",
            m.FadeIn(r_bonus, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(r_opt, shift=0.1 * m.UP), m.FadeIn(r_cite)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the race

    def race(self):
        self.next_section("6 · The race", notes=(
            "classic.md §7 / classic.json maze.methods: 200 runs × 300 episodes each, Q-learning α = 0.5, ε = 0.1, "
            "γ = 0.99, 200 moves an episode. Bars: median episode of the first success on a log scale, whiskers the "
            "10th–90th percentiles. Sparse 104 (22–238; 21% never in 300); naive shaping Φ = −distance/10 never (0 of "
            "200); bonus with Q starting at 0: 219 (143–251); optimism alone: 5 (3–8); shaping Φ = γ^(d−1): 4 (2–5); "
            "bonus with optimistic start: 2 (1–5). Shaping, the bonus with optimism and optimism alone all end with the "
            "28-move greedy route in 100% of runs; sparse Q-learning's greedy route reaches the charger in 0%."))
        rows = [
            ("sparse reward only", "sparse", MUTED),
            ("shaping, the −distance potential", "shaping_negative", RED),
            ("novelty bonus, Q starts at 0", "count_bonus_q0", EXPLORE),
            ("optimism alone, Q starts at 1", "optimistic", VALUE),
            ("shaping, the no-wall potential", "shaping", REWARD),
            ("novelty bonus + optimism", "count_bonus", EXPLORE),
        ]
        BX0, BX1, LX = -1.75, 3.9, -6.5

        def bx(e):
            return BX0 + (BX1 - BX0) * math.log10(e) / math.log10(300)

        ys = [1.85 - 0.62 * i for i in range(len(rows))]
        labels, bars, whiskers, values = m.VGroup(), [], [], m.VGroup()
        for (lab, key, col), y in zip(rows, ys):
            fs = FS[key]
            labels.add(left_at(text(lab, 22, INK if col == MUTED else col), LX, y))
            if fs["median"] is None:
                bars.append(m.Arrow([BX0, y, 0], [BX1, y, 0], buff=0, color=RED, stroke_width=6,
                                    max_tip_length_to_length_ratio=0.05))
                whiskers.append(None)
                values.add(right_at(code("never", 22, RED, weight=m.BOLD), XR, y))
                continue
            w = max(bx(fs["median"]) - BX0, 0.06)
            fill = "#2E2640" if col == MUTED else FILLS.get(col, CARD)
            bars.append(m.Rectangle(width=w, height=0.34).set_stroke(col, 2).set_fill(fill, 1).move_to([BX0 + w / 2, y, 0]))
            a, b = bx(max(fs["p10"], 1)), bx(fs["p90"])
            whiskers.append(m.VGroup(m.Line([a, y, 0], [b, y, 0]), m.Line([a, y - 0.11, 0], [a, y + 0.11, 0]),
                                     m.Line([b, y - 0.11, 0], [b, y + 0.11, 0])).set_stroke(INK, 2).set_z_index(2))
            extra = f" · {100 * fs['never_within_300']:.0f}% never" if key == "sparse" else ""
            values.add(right_at(code(f"{fs['median']:.0f}{extra}", 22, RED if extra else INK, weight=m.BOLD,
                                     t2c={r"^\d+": INK}), XR, y))
        axis = m.Line([BX0, ys[-1] - 0.45, 0], [BX1 + 0.4, ys[-1] - 0.45, 0]).set_stroke(FAINT, 2.5)
        ticks = m.VGroup()
        for e in (1, 3, 10, 30, 100, 300):
            p = np.array([bx(e), ys[-1] - 0.45, 0])
            ticks.add(m.Line(p, p + [0, -0.09, 0]).set_stroke(FAINT, 2), code(str(e), 20, MUTED).next_to(p, m.DOWN, buff=0.14))
        col_head = right_at(cite("median episode"), XR, 2.4)
        x_lab = cite("episode of the first success, log scale · bar: median · whiskers: 10th to 90th percentile")
        x_lab.move_to([0, ys[-1] - 1.3, 0])
        src = cite(f"{RUNS} runs × {EPISODES} episodes each · Q-learning · measured on this Mac").move_to([0, ys[-1] - 1.72, 0])
        fast = m.SurroundingRectangle(m.VGroup(labels[3:], values[3:]), buff=0.12, corner_radius=0.1).set_stroke(GOOD, 2.5)
        fast_lab = left_at(text("then the 28-move route, in every run", 22, GOOD, weight=m.BOLD), LX, 2.4)

        self.say(
            "[Here's the whole race, two hundred runs each]: [the bare sparse reward waits about a hundred episodes], "
            "[the naive potential forever.]",
            m.AnimationGroup(self.new_header("06 · the race", "Two hundred runs each"), m.FadeIn(labels), m.Create(axis),
                             m.FadeIn(ticks), m.FadeIn(col_head), m.FadeIn(x_lab), m.FadeIn(src)),
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.LEFT), m.Create(whiskers[0]), m.FadeIn(values[0])),
            m.AnimationGroup(m.GrowArrow(bars[1]), m.FadeIn(values[1])),
        )
        self.say(
            "[Optimism, shaping and the bonus cut the wait to a handful of episodes], [and then learn the twenty-eight move route in every run.]",
            m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(bars[i], m.LEFT), m.Create(whiskers[i]), m.FadeIn(values[i]))
                            for i in (2, 3, 4, 5)], lag_ratio=0.35),
            m.AnimationGroup(m.Create(fast), m.FadeIn(fast_lab, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · Montezuma's Revenge

    def montezuma(self):
        self.next_section("7 · Montezuma's Revenge", notes=(
            "Cited scores (research/notes.md §11; papers as published, read October 2026): DQN 0 (Mnih et al. 2015, "
            "Extended Data Table 2; random play 0; human tester 4,367); DQN with a pseudo-count bonus 3,439 at 100M "
            "frames (2,461 at 50M), 15 rooms explored vs 2 without the bonus (Bellemare et al. 2016 §6; RND's table cites "
            "3,700, untraced — we use the primary 3,439, notes Appendix A); random network distillation 8,152 (Burda et "
            "al. 2018 Table 1; 'the first method that achieves better than average human performance on this game "
            "without using demonstrations', their human average 4,753); Go-Explore without domain knowledge, robustified "
            "policy 43,791 (Ecoffet et al. 2021, Extended Data Table 3); with domain-knowledge cells and simulator "
            "restore 1,731,645 (main text). Human baselines differ between papers (4,367 in the DQN paper, 4,753 in "
            "later ones). Curiosity as prediction error: Pathak et al. 2017 (ICM; tested on VizDoom and Mario, not on "
            "Montezuma). The room is a drawing, not a screenshot; the first level has 24 rooms (Bellemare et al. 2016)."))
        # the room (a drawing)
        rc = np.array([-4.4, 0.3, 0])
        room = card(4.2, 3.3, STATE, CARD, radius=0.12, stroke=2.5).move_to(rc)
        plat = m.VGroup(
            m.Line(rc + [-2.0, 0.55, 0], rc + [-0.6, 0.55, 0]), m.Line(rc + [0.6, 0.55, 0], rc + [2.0, 0.55, 0]),
            m.Line(rc + [-0.45, -0.15, 0], rc + [0.45, -0.15, 0]), m.Line(rc + [-2.0, -1.45, 0], rc + [2.0, -1.45, 0]),
        ).set_stroke(STATE, 5)
        ladder = m.VGroup(m.Line(rc + [-1.35, 0.55, 0], rc + [-1.35, -1.45, 0]), m.Line(rc + [-1.05, 0.55, 0], rc + [-1.05, -1.45, 0]),
                          *[m.Line(rc + [-1.35, y, 0], rc + [-1.05, y, 0]) for y in np.arange(0.3, -1.4, -0.3)]).set_stroke(MUTED, 3)
        rope = m.Line(rc + [0.0, 1.35, 0], rc + [0.0, -0.05, 0]).set_stroke(MUTED, 3)
        rope.shift(0.0 * m.RIGHT)
        key = m.VGroup(m.Circle(radius=0.13).set_stroke(GOLD, 4).move_to(rc + [1.55, -1.05, 0]),
                       m.Line(rc + [1.55, -1.18, 0], rc + [1.55, -1.38, 0]).set_stroke(GOLD, 4),
                       m.Line(rc + [1.55, -1.3, 0], rc + [1.66, -1.3, 0]).set_stroke(GOLD, 4))
        doors = m.VGroup(*[m.Rectangle(width=0.16, height=0.6).set_stroke(RED, 2.5).set_fill(RED_FILL, 1).move_to(rc + [x, 0.88, 0])
                           for x in (-1.88, 1.88)])
        mini = rover(0.3).move_to(rc + [-1.7, 0.8, 0])
        hops = m.VGroup(*[m.ArcBetweenPoints(rc + a, rc + b, angle=-1.1).set_stroke(ACTION, 2.5) for a, b in
                          (([-1.45, 0.95, 0], [-0.12, 0.95, 0]), ([0.12, 0.35, 0], [0.75, 0.75, 0]),
                           ([1.35, 0.75, 0], [1.5, -0.85, 0]))])
        room_all = m.VGroup(room, plat, ladder, rope, key, doors, mini)
        room_cap = m.VGroup(cite("Montezuma's Revenge, as a drawing"), cite("24 rooms in level one · Bellemare et al., 2016"))
        room_cap.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(room_cap, -6.5, -1.85)

        # the scores: one row per agent, the human tester's score as a dashed row of its own
        SX0, SX1 = -1.6, 6.2

        def sx(v):
            return SX0 + (SX1 - SX0) * v / 45000

        rows = [MONTEZUMA[0], ("human tester", "Mnih et al., 2015", HUMAN), *MONTEZUMA[1:]]
        ys = [1.95 - 0.9 * i for i in range(len(rows))]       # bar centres; each name sits above its bar
        names, bars, nums = [], [], []
        for i, ((name, src, score), y) in enumerate(zip(rows, ys)):
            person = i == 1
            lab = m.VGroup(text(name, 22, MUTED if person else INK, weight=m.BOLD), cite(src)).arrange(m.RIGHT, buff=0.3)
            sit(lab[1], lab[0][0].get_bottom()[1])
            names.append(left_at(lab, SX0, y + 0.45))
            w = max(sx(score) - SX0, 0.04)
            box = m.Rectangle(width=w, height=0.32).move_to([SX0 + w / 2, y, 0])
            if person:
                box = m.DashedVMobject(box.set_stroke(INK, 2), num_dashes=40)
            else:
                box.set_stroke(GOLD, 2).set_fill(GOLD_FILL, 1)
            bars.append(box)
            num = code(f"{score:,}", 22, INK if person else GOLD, weight=m.BOLD)
            if w > 2.0:
                nums.append(right_at(num, SX0 + w - 0.15, y).set_color(INK).set_z_index(3))
            else:
                nums.append(left_at(num, SX0 + w + 0.18, y))
        foot1 = left_at(cite(f"Go-Explore with hand-made cells and simulator restore: {GO_EXPLORE_DOMAIN:,} · Ecoffet et al., 2021"),
                        -6.5, -2.65)
        foot2 = left_at(cite("game points, as each paper reports them · later papers list the human score as 4,753"),
                        -6.5, -3.1)

        self.say(
            "[The same problem made one Atari game famous]: [Montezuma's Revenge, where the first point needs a long run of precise jumps.]",
            m.AnimationGroup(self.new_header("07 · at scale", "Montezuma's Revenge"), m.FadeIn(room_all), m.FadeIn(room_cap)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(h) for h in hops], lag_ratio=0.6), m.Circumscribe(key, color=GOLD, buff=0.08)),
        )
        self.say(
            "[In twenty fifteen, the DQN scored zero on it]; [a human tester scored four thousand three hundred and sixty-seven.]",
            m.AnimationGroup(m.FadeIn(names[0]), m.FadeIn(bars[0]), m.FadeIn(nums[0])),
            m.AnimationGroup(m.FadeIn(names[1]), m.Create(bars[1]), m.FadeIn(nums[1]), m.FadeIn(foot2)),
        )
        self.say(
            "[Pseudo-counts brought count bonuses to pixels]: [about three thousand four hundred points, and fifteen rooms explored instead of two.]",
            m.FadeIn(names[2]),
            m.AnimationGroup(m.GrowFromEdge(bars[2], m.LEFT), m.FadeIn(nums[2])),
        )
        self.say(
            "[Curiosity pays for surprise]: [random network distillation reached eight thousand one hundred and fifty-two], "
            "[above the human tester.]",
            m.FadeIn(names[3]),
            m.AnimationGroup(m.GrowFromEdge(bars[3], m.LEFT), m.FadeIn(nums[3])),
            m.Circumscribe(m.VGroup(bars[3], nums[3]), color=GOLD, buff=0.08),
        )
        self.say(
            "[And Go-Explore, which remembers good places and goes back before exploring], "
            "[scored forty-three thousand, seven hundred and ninety-one.]",
            m.FadeIn(names[4]),
            m.AnimationGroup(m.GrowFromEdge(bars[4], m.LEFT), m.FadeIn(nums[4]), m.FadeIn(foot1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            "Shape (potential-based shaping keeps the goal), trace (eligibility traces carry the credit back along the "
            "path), explore (a novelty bonus with optimistic starts). In the maze (200 runs, measured): first success "
            f"median {FS['sparse']['median']:.0f} with the sparse reward, {FS['shaping']['median']:.0f} with shaping, "
            f"{FS['count_bonus']['median']:.0f} with the bonus and optimism."))
        halo = glow(BLUE, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Shape", 70, REWARD), title_text("Trace", 70, SIGNAL), title_text("Explore", 70, EXPLORE))
        words.arrange(m.RIGHT, buff=1.4).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text("hints that keep the goal", 24, MUTED), text("credit back along the path", 24, MUTED),
                        text("a bonus for the unknown", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        low = min(s_[0].get_bottom()[1] for s_ in subs)
        for s_ in subs:
            sit(s_, low)
        proof = code(f"first success · episode {FS['sparse']['median']:.0f} → {FS['shaping']['median']:.0f} with shaping "
                     f"→ {FS['count_bonus']['median']:.0f} with the bonus", 22, INK,
                     t2c={r"→ \d+ with shaping": REWARD, r"→ \d+ with the bonus": EXPLORE})
        proof.move_to([0, -1.35, 0])
        proof_cite = cite(f"median of {RUNS} runs · measured on this Mac").next_to(proof, m.DOWN, buff=0.18)
        mascot = rover(0.7).move_to([0, -2.85, 0])

        self.say(
            "[To sum up]: [shape the reward without moving the goal], [trace the credit back along the path], "
            "[and pay for exploring the unknown.]",
            m.AnimationGroup(self.new_header("08 · recap", "Shape, trace, explore"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[In the maze, that took the first success from episode one hundred and four], [down to four, or even two.]",
            m.AnimationGroup(m.FadeIn(proof, shift=0.1 * m.UP), m.FadeIn(proof_cite)),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · module 3 closes

    def tables(self):
        self.next_section("9 · Tables can't hold big worlds", notes=(
            "Module 3 (Episodes 8–11: Monte Carlo, TD, Q-learning and SARSA, sparse rewards) kept one number per state "
            "and move: the maze's Q table has 59 cells × 4 moves = 236 entries (rl_reference's tabular Q, asserted; the "
            f"rows shown are seed {SEED}'s values after 300 episodes). An Atari frame is 210 × 160 pixels (Mnih et al. "
            "2015; notes §13), far too many screens to list. Episode 12: function approximation. The pixel grid is a "
            "drawing (21 × 16 squares, one per 10 × 10 pixels)."))
        chips_ = m.VGroup(*[chip(lab, BLUE, BLUE_FILL, width=2.75, height=0.66, size=22)
                            for lab in ("8 · playing", "9 · guessing", "10 · acting", "11 · exploring")])
        chips_.arrange(m.RIGHT, buff=0.3).move_to([0, 1.95, 0])
        q = SPARSE_RUN.Q
        shown = [START, (1, 2), (9, 9), (9, 10)]

        def qrow(s):
            vals = "  ".join(f"{a} {q[s][a]:.2f}" for a in ACTIONS)
            return f"({s[0]},{s[1]})".ljust(8) + vals

        lines = [qrow(START), qrow((1, 2)), "   ⋮", qrow((9, 9)), qrow((9, 10))]
        assert all(s in q for s in shown)
        table = code_block("\n".join(lines), size=20, title="Q table · the maze, after 300 episodes", accent=VALUE)
        table.move_to([-3.05, -0.55, 0])
        count = left_at(text(f"{len(q)} cells × {len(ACTIONS)} moves = {len(q) * len(ACTIONS)} numbers", 24, VALUE, weight=m.BOLD),
                        table.get_left()[0], -2.15)
        rng = np.random.default_rng(4)
        px = cell_grid(16, 21, size=0.15, gap=0.03, color=CARD, fill=CARD)
        for row_ in px:
            for sq in row_:
                u = rng.random()
                sq.set_fill(TEAL_FILL if u < 0.55 else (STATE if u < 0.62 else CARD), 1).set_stroke(width=0)
        px.move_to([3.45, -0.45, 0])
        screen = card(px.width + 0.4, px.height + 0.4, STATE, "#0C0912", radius=0.12, stroke=2.5).move_to(px)
        px_lab = cite(f"an Atari screen: {ATARI_FRAME[0]} × {ATARI_FRAME[1]} pixels · Mnih et al., 2015").next_to(screen, m.DOWN, buff=0.18)
        final = title_text("Tables can't hold big worlds.", 44, INK).move_to([0, -3.0, 0])

        self.say(
            "[That closes module three]: [Rover learned from experience: by playing, by guessing, by acting, and by exploring.]",
            self.new_header("09 · module 3", "Learning from experience"),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips_], lag_ratio=0.45),
        )
        self.say(
            "[But every method kept a table]: [for this little maze, two hundred and thirty-six numbers, one per cell and move.]",
            m.FadeIn(table, shift=0.1 * m.UP),
            m.FadeIn(count, shift=0.1 * m.UP),
        )
        self.say(
            "[An Atari screen is two hundred and ten by one hundred and sixty pixels]: [far too many screens for any table.]",
            m.AnimationGroup(m.FadeIn(screen), m.LaggedStart(*[m.FadeIn(r) for r in px], lag_ratio=0.08), m.FadeIn(px_lab)),
            m.Write(final, run_time=1.4),
        )
        self.wait(0.6)
        self.clear()
