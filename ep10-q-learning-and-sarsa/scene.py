"""Trial and Reward · Episode 10 · Q-Learning and SARSA (Module 3 · Learning from Experience)

Learn the values of actions, on-policy or off-policy. Episode 9's TD error, now for moves: Rover keeps Q(s, a) for
every move in every cell, acts ε-greedily (Episode 8), and after each move nudges Q(s, a) toward a target. SARSA aims
at the move Rover will really make next (on-policy); Q-learning aims at the best next move, whatever Rover does
(off-policy). The running example is the cliff walk (Sutton & Barto Example 6.6; rl_reference CLIFF_WORLD: 4 × 12,
−1 per step, −100 and back to the start for a step into the cliff). Two Rovers trained with seed 0 (500 episodes,
ε = 0.1, α = 0.5): Q-learning learns the 13-step edge path, SARSA the 17-step top road (its greedy path reaches the
goal; asserted). Acting ε-greedily, the edge Rover falls now and then. Measured on this Mac (classic.md §6, 500 runs ×
500 episodes): reward per episode Q-learning −49.9, SARSA −27.4, expected SARSA −21.1 (the book: ≈ −50 / ≈ −25).
Why: each learner's values for the cell where the routes split. Convergence (Watkins 1989; Watkins & Dayan 1992),
which to use, the recap, and Episode 11.

Every number on screen is computed below from rl_reference.py and asserted, or loaded with measured(...), or cited.

Sources (research/notes.md §10, Appendix A row 10, Appendix C row 10; facts dated October 2026):
- Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018): SARSA eq. 6.7, Q-learning eq. 6.8,
  Example 6.6 Cliff Walking (p. 132: −1 per step, −100 and back to the start, ε = 0.1; the plot: SARSA ≈ −25,
  Q-learning ≈ −50; "if ε were gradually reduced, then both methods would asymptotically converge to the optimal
  policy"). α is not stated in the book; α = 0.5 is the common reproduction's (S. Zhang's), used in our runs.
  The book's cliff figure sits inside Example 6.6, so the screen cites "Example 6.6".
- Watkins (1989), Learning from Delayed Rewards, PhD thesis, Cambridge; Watkins & Dayan (1992), "Q-learning",
  Machine Learning 8:279–292 (converges to the optimal action values with probability 1 if all actions are repeatedly
  sampled in all states, the values are a lookup table, and Σα = ∞, Σα² < ∞).
- Rummery & Niranjan (1994), CUED/F-INFENG/TR 166 ("Modified Connectionist Q-learning"); the name Sarsa: Sutton (1996).
- research/experiments/classic.md §6 + classic.json cliff (500 runs × 500 episodes; headline cliff_*_reward = the
  average reward per episode over episodes 101–500; episodes with a fall per run; greedy path kinds: 13% of SARSA
  runs' final greedy path does not reach the goal — seed 0's does, asserted).

Illustrative (computed, but chosen): seed 0's learned tables (the same runs as classic.json's example paths); the two
targets in section 3 use the next cell's values from seed 0's Q-learning table (equal to the optimal values from value
iteration, asserted) for both learners; the ε-greedy episodes in section 5 replay the final tables with
default_rng(176) (Q-learning) and default_rng(595) (SARSA), three episodes each, no learning.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep10"
W = CLIFF_WORLD
EPS, ALPHA, SEED, EPISODES = 0.1, 0.5, 0, 500
START = W.start
GOAL = next(iter(W.terminals))
assert START == (1, 1) and GOAL == (12, 1) and (W.width, W.height) == (12, 4) and len(W.cliffs) == 10
assert W.transitions(START, "E")[0][1:3] == (START, -100.0) and W.step_reward == -1.0

# ---------------------------------------------------------------------------- the measured runs (classic.json cliff)

CL = measured("classic", "cliff")
assert (CL["runs"], CL["episodes"], CL["alpha"], CL["epsilon"], CL["gamma"]) == (500, 500, ALPHA, EPS, 1.0)
assert tuple(CL["layout"]) == CLIFF_LAYOUT
RUNS = CL["runs"]
QL_R = measured("classic", "headline", "cliff_q_learning_reward")
SA_R = measured("classic", "headline", "cliff_sarsa_reward")
ES_R = measured("classic", "headline", "cliff_expected_sarsa_reward")
assert (round(QL_R, 1), round(SA_R, 1), round(ES_R, 1)) == (-49.9, -27.4, -21.1)
assert round(QL_R) == -50 and round(SA_R) == -27 and round(ES_R) == -21          # "minus fifty / twenty-seven / twenty-one"
METH = CL["methods"]
assert METH["q_learning"]["avg_reward_episodes_101_500"] == QL_R and METH["sarsa"]["avg_reward_episodes_101_500"] == SA_R
FALLS_QL = METH["q_learning"]["episodes_with_a_fall_per_run"]
FALLS_SA = METH["sarsa"]["episodes_with_a_fall_per_run"]
assert round(FALLS_QL) == 125 and round(FALLS_SA) == 35
assert abs(FALLS_QL / CL["episodes"] - 0.25) < 0.01                                 # "about one episode in four"
assert 0.5 < SA_R / QL_R < 0.6                                                       # "loses about half as much"
LOOPS = METH["sarsa"]["greedy_path_kinds"]["does not reach the goal"]
assert round(LOOPS, 2) == 0.13
BOOK = {"sarsa": -25, "q_learning": -50}                                             # read from the book's plot (notes §10)

# ---------------------------------------------------------------------------- seed 0's two Rovers (recomputed here)

QL = q_learning(W, EPISODES, alpha=ALPHA, epsilon=EPS, seed=SEED)
SA = sarsa(W, EPISODES, alpha=ALPHA, epsilon=EPS, seed=SEED)
QL_PATH, SA_PATH = greedy_path(W, QL.Q), greedy_path(W, SA.Q)


def _parse(cells):
    return [tuple(int(v) for v in c.split(",")) for c in cells]


assert METH["q_learning"]["example_seed"] == SEED and METH["sarsa"]["example_seed"] == SEED
assert QL_PATH == _parse(METH["q_learning"]["example_greedy_path"])
assert SA_PATH == _parse(METH["sarsa"]["example_greedy_path"])
assert QL_PATH[-1] == GOAL and len(QL_PATH) - 1 == 13 and max(r for _, r in QL_PATH) == 2     # the edge, 13 steps
assert SA_PATH[-1] == GOAL and len(SA_PATH) - 1 == 17 and max(r for _, r in SA_PATH) == 4     # the top road, 17
assert CL["optimal_return"] == -(len(QL_PATH) - 1) and CL["safe_top_row_return"] == -(len(SA_PATH) - 1)

# the optimal values (Episode 7's value iteration) — Q-learning's table matches them where it matters
V_STAR, _ = value_iteration(W, gamma=1.0)
Q_STAR = q_from_v(W, V_STAR, 1.0)
SPLIT, EDGE_CELL = (1, 2), (2, 2)               # where the routes split; the first cell on the edge
assert QL_PATH[:3] == [START, SPLIT, EDGE_CELL] and SA_PATH[:3] == [START, SPLIT, (1, 3)]
assert all(abs(QL.Q[EDGE_CELL][a] - Q_STAR[EDGE_CELL][a]) < 0.05 for a in ACTIONS)
NEXT = {a: round(QL.Q[EDGE_CELL][a]) for a in ACTIONS}
assert NEXT == {"N": -13, "E": -11, "S": -113, "W": -13}
_s2, R_STEP, _done = W.step(SPLIT, "E", np.random.default_rng(0))
assert _s2 == EDGE_CELL and R_STEP == -1.0 and not _done
T_QL = R_STEP + max(QL.Q[EDGE_CELL].values())
T_SA_E = R_STEP + QL.Q[EDGE_CELL]["E"]
T_SA_S = R_STEP + QL.Q[EDGE_CELL]["S"]
assert (round(T_QL), round(T_SA_E), round(T_SA_S)) == (-12, -12, -114)
P_BEST, P_OTHER = 1 - EPS + EPS / len(ACTIONS), EPS / len(ACTIONS)
assert (round(P_BEST, 3), round(P_OTHER, 3)) == (0.925, 0.025) and round(1 / P_OTHER) == 40   # "one time in forty"

# each learner's values where the routes split (section 7)
SPLIT_QL = {a: round(QL.Q[SPLIT][a]) for a in ACTIONS}
SPLIT_SA = {a: round(SA.Q[SPLIT][a]) for a in ACTIONS}
assert SPLIT_QL == {"N": -14, "E": -12, "S": -14, "W": -13}
assert SPLIT_SA == {"N": -20, "E": -57, "S": -26, "W": -26}
assert max(ACTIONS, key=lambda a: QL.Q[SPLIT][a]) == "E" and max(ACTIONS, key=lambda a: SA.Q[SPLIT][a]) == "N"


# ---------------------------------------------------------------------------- ε-greedy episodes with the final tables

def greedy_move(Q, s):
    return max(W.actions, key=lambda a: Q[s][a])


def rollout(Q, rng, max_steps=200):
    """One ε-greedy episode with a fixed table: [(s, a, s2, r, random_move)], and its total reward."""
    s, steps, total = W.reset(), [], 0.0
    a = epsilon_greedy(Q[s], EPS, rng)
    for _ in range(max_steps):
        s2, r, done = W.step(s, a, rng)
        steps.append((s, a, s2, r, a != greedy_move(Q, s)))
        total += r
        if done:
            break
        s, a = s2, epsilon_greedy(Q[s2], EPS, rng)
    return steps, total


_rq, _rs = np.random.default_rng(176), np.random.default_rng(595)
QL_RUNS = [rollout(QL.Q, _rq) for _ in range(3)]
SA_RUNS = [rollout(SA.Q, _rs) for _ in range(3)]
assert [t for _, t in QL_RUNS] == [-13.0, -119.0, -13.0]
assert [t for _, t in SA_RUNS] == [-17.0, -19.0, -17.0]
_fall = [st for st in QL_RUNS[1][0] if st[3] == W.cliff_reward]
assert len(_fall) == 1 and _fall[0][:2] == ((6, 2), "S") and sum(st[4] for st in QL_RUNS[1][0]) == 1
assert QL_RUNS[1][0][6][3] == W.cliff_reward and all(st[3] == -1.0 for st in QL_RUNS[1][0][:6])   # 6 moves, then the fall
_odd = [st for st in SA_RUNS[1][0] if st[4]]
assert len(_odd) == 1 and _odd[0][:3] == ((6, 4), "S", (6, 3))
assert all(not st[4] for k in (0, 2) for st in QL_RUNS[k][0] + SA_RUNS[k][0])

# ---------------------------------------------------------------------------- look

DICE = recolored("dice", EXPLORE, VIOLET_FILL, prefix=PREFIX)
QL_C, SA_C, ES_C = VALUE, ACTION, EXPLORE   # Q-learning aims at the best VALUE; SARSA at its next ACTION; expected
#                                             SARSA averages over the exploring dice
DIRS = {"N": np.array([0, 1, 0.0]), "E": np.array([1, 0, 0.0]), "S": np.array([0, -1, 0.0]), "W": np.array([-1, 0, 0.0])}
WORD = {"N": "up", "E": "right", "S": "down", "W": "left"}


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def num(v):
    return minus(f"{v:.0f}")


def colored(parts, size=24, weight=m.NORMAL):
    """One mono line from (string, colour) parts joined by spaces; each part keeps its colour."""
    s = " ".join(p for p, _ in parts)
    words = code(s, size, INK, weight=weight)
    glyph_of, n = {}, 0
    for i, ch in enumerate(s):
        if not ch.isspace():
            glyph_of[i] = n
            n += 1
    assert n == len(words)
    pos = 0
    for p, col in parts:
        for i in range(pos, pos + len(p)):
            if i in glyph_of:
                words[glyph_of[i]].set_color(col)
        pos += len(p) + 1
    return words


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.RIGHT)


def cliff_view(cell, center):
    """The cliff walk as GridView draws it: the cliff RED, the charger GOLD."""
    return GridView.from_world(W, cell=cell, center=center)


class Cross(m.VGroup):
    """One cell, zoomed: four triangles (one per move), a small CORAL arrow and a BLUE value in each."""

    def __init__(self, center, size=2.2, values=None, frame=EDGE):
        super().__init__()
        c, h = np.array(center, dtype=float), size / 2
        self.c, self.h = c, h
        tl, tr, br, bl = c + [-h, h, 0], c + [h, h, 0], c + [h, -h, 0], c + [-h, -h, 0]
        corners = {"N": (tl, tr), "E": (tr, br), "S": (br, bl), "W": (bl, tl)}
        self.tris = {a: m.Polygon(p, q, c, stroke_color=EDGE, stroke_width=2, fill_color=CARD, fill_opacity=1)
                     for a, (p, q) in corners.items()}
        self.arrows = {a: m.Arrow(c + 0.1 * h * DIRS[a], c + 0.34 * h * DIRS[a], buff=0, color=ACTION, stroke_width=4,
                                  max_tip_length_to_length_ratio=0.5) for a in ACTIONS}
        self.frame = m.Square(size).move_to(c).set_stroke(frame, 3).set_fill(opacity=0)
        self.vals = {a: self.value(a, values[a] if values else None) for a in ACTIONS}
        self.add(m.VGroup(*self.tris.values()), m.VGroup(*self.arrows.values()), self.frame,
                 m.VGroup(*self.vals.values()))

    def spot(self, a):
        return self.c + (0.7 if a in "NS" else 0.67) * self.h * DIRS[a]

    def value(self, a, v, color=VALUE):
        s = "?" if v is None else num(v)
        return code(s, 22, MUTED if v is None else color, weight=m.BOLD).move_to(self.spot(a))

    def set_values(self, values, run_time=0.8):
        return m.AnimationGroup(*[m.Transform(self.vals[a], self.value(a, values[a])) for a in ACTIONS], run_time=run_time)

    def tint(self, a, color, fill):
        return self.tris[a].animate.set_fill(fill, 1).set_stroke(color, 3)


def takeaway(name, title, desc, color, width=6.4, height=1.3):
    box = card(width, height, color, CARD, stroke=2)
    words = m.VGroup(text(title, 22, INK, weight=m.BOLD), text(desc, 20, MUTED)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
    body = m.VGroup(icon(name, 0.75), words).arrange(m.RIGHT, buff=0.35)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, body)


def flash(point, color, strong=True):
    return m.Flash(point, color=color, line_length=0.2 if strong else 0.14, num_lines=12,
                   flash_radius=0.3 if strong else 0.22, line_stroke_width=4 if strong else 3, run_time=0.45)


def cliff_cell(s, a):
    """The cliff cell a move from s steps into (the world sends Rover home, so ask the map)."""
    return W.move(s, a)


def act(bot, grid, steps, dt=0.085, marks=None):
    """Rover replays one ε-greedy episode: random moves flash VIOLET; a step into the cliff flashes RED, shows −100
    and arcs Rover back to the start."""
    seq = []
    for s, a, s2, r, random_move in steps:
        if r == W.cliff_reward:
            into = grid.at(cliff_cell(s, a))
            chip_ = reward_chip(r, 20).move_to(into + [0, -0.62 * grid.size, 0])
            seq += [m.AnimationGroup(bot.animate(run_time=0.3).move_to(into), flash(grid.at(s), EXPLORE, False), run_time=0.35),
                    m.AnimationGroup(flash(into, RED), m.FadeIn(chip_, shift=0.1 * m.DOWN), run_time=0.45)]
            if marks is not None:
                seq.append(m.FadeIn(marks, run_time=0.2))
            seq += [m.AnimationGroup(bot.animate(run_time=0.55, path_arc=1.1).move_to(grid.at(START)), m.FadeOut(chip_),
                                     run_time=0.55)]
        elif random_move:
            seq.append(m.AnimationGroup(bot.animate(run_time=0.3).move_to(grid.at(s2)), flash(grid.at(s), EXPLORE, False),
                                        run_time=0.4))
            if marks is not None:
                seq.append(m.FadeIn(marks, run_time=0.2))
        else:
            seq.append(bot.animate(run_time=dt).move_to(grid.at(s2)))
    return m.Succession(*seq)


def home(bot, grid, run_time=0.45):
    """From the charger back to the start for the next episode (an arc over the grid)."""
    return bot.animate(run_time=run_time, path_arc=0.9).move_to(grid.at(START))


def route(grid, cells, color=ACTION, width=6, dashed=False, nudge=0.0):
    pts = [grid.at(s) + [0, nudge, 0] for s in cells]
    path = m.VMobject().set_points_as_corners(pts).set_stroke(color, width)
    if dashed:
        path = m.DashedVMobject(path, num_dashes=46).set_stroke(color, width)
    return path


# ---------------------------------------------------------------------------- the episode


class Ep10QLearningAndSARSA(Episode):
    number = 10

    def lesson(self):
        self.the_cliff()
        self.action_values()
        self.two_targets()
        self.in_code()
        self.two_rovers()
        self.measured_runs()
        self.whose_values()
        self.in_the_limit()
        self.recap()

    # ------------------------------------------------------------------ 1 · the cliff

    def the_cliff(self):
        self.next_section("1 · The cliff walk", notes=(
            "Sutton & Barto (2018) Example 6.6, the cliff walk (rl_reference CLIFF_WORLD; GridView.from_world draws the "
            "cliff RED): 4 × 12, start bottom-left, the goal (drawn as Rover's charger) bottom-right, the ten cells "
            "between them are the cliff. −1 per step (into the goal too); a step into the cliff pays −100 and sends Rover "
            "back to the start (the episode goes on); γ = 1. The edge path is 13 steps (return −13, classic.json "
            "optimal_return); the top-row road 17 (−17, safe_top_row_return). Rover's demo walk (up, right, down into "
            "the cliff) is illustrative."))
        g = cliff_view(0.9, (0, 0.45, 0))
        self.g1 = g
        floor = m.VGroup(*[sq for (r, c), sq in g.squares.items() if (r, c) not in g.specials])
        cliffs = m.VGroup(*[g.squares[W.to_screen(s)] for s in sorted(W.cliffs)])
        goal = m.VGroup(g.squares[W.to_screen(GOAL)], g.marks)
        bot = rover(0.42).move_to(g.at(START)).set_z_index(4)
        self.bot = bot
        start_lab = code("start", 20, MUTED).move_to(g.at(START) + [0, -0.75, 0])
        goal_lab = code("goal", 20, REWARD).move_to(g.at(GOAL) + [0, -0.75, 0])
        mid = (g.at((2, 1)) + g.at((11, 1))) / 2
        cliff_lab = text("the cliff: −100, and back to the start", 22, RED).move_to(mid + [0, -0.78, 0])
        src = cite("Sutton & Barto, 2018, Example 6.6").move_to([0, -3.35, 0]).align_to([6.5, 0, 0], m.RIGHT)
        ep9 = pill("episode 9 · learn from the next guess", SIGNAL, PINK_FILL, 20).move_to([0, -2.6, 0])

        def step_chip(s, v):
            return reward_chip(v, 20).move_to(g.at(s) + [0.42, 0.52, 0]).set_z_index(6)

        c1, c2 = step_chip((1, 2), -1), step_chip((2, 2), -1)
        c3 = reward_chip(-100, 20).move_to(g.at((2, 1)) + [0.6, -0.75, 0]).set_z_index(6)

        edge = route(g, QL_PATH, ACTION, 6)
        top = route(g, SA_PATH, ACTION, 5, dashed=True)
        leg_edge = m.VGroup(m.Line([0, 0, 0], [0.6, 0, 0]).set_stroke(ACTION, 6),
                            colored([("along the edge:", INK), ("13 steps", ACTION), ("·", MUTED), ("return −13", VALUE)], 22))
        leg_edge.arrange(m.RIGHT, buff=0.25)
        leg_top = m.VGroup(m.DashedLine([0, 0, 0], [0.6, 0, 0], dash_length=0.09).set_stroke(ACTION, 5),
                           colored([("the top road:", INK), ("17 steps", ACTION), ("·", MUTED), ("return −17", VALUE)], 22))
        leg_top.arrange(m.RIGHT, buff=0.25)
        legs = m.VGroup(leg_edge, leg_top).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        left_at(legs, -5.4, -2.65)
        assert f"{len(QL_PATH) - 1} steps" in "13 steps" and f"{len(SA_PATH) - 1} steps" in "17 steps"

        self.say(
            "[Last time, Rover learned how good each tile was.] [Now it has to choose its own moves.]",
            m.AnimationGroup(self.new_header("01 · the cliff", "A walk along the edge"), m.FadeIn(ep9, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(sq) for sq in floor], lag_ratio=0.02, run_time=1.2),
                             m.FadeIn(bot, shift=0.2 * m.DOWN)),
        )
        self.say(
            "[This is the cliff walk. Rover starts at the bottom left], [the charger waits at the bottom right], "
            "[and in between lies the cliff.]",
            m.AnimationGroup(m.FadeOut(ep9), m.FadeIn(start_lab), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(goal, scale=0.8), m.FadeIn(goal_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(sq, shift=0.1 * m.DOWN) for sq in cliffs], lag_ratio=0.08),
                             m.FadeIn(cliff_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[Every step costs minus one.] [A step into the cliff costs minus one hundred], [and sends Rover back to the start.]",
            m.Succession(bot.animate(run_time=0.4).move_to(g.at((1, 2))), m.FadeIn(c1, shift=0.1 * m.UP, run_time=0.25),
                         m.AnimationGroup(m.FadeOut(c1, run_time=0.3), bot.animate(run_time=0.4).move_to(g.at((2, 2)))),
                         m.FadeIn(c2, shift=0.1 * m.UP, run_time=0.25), m.FadeOut(c2, run_time=0.3)),
            m.Succession(bot.animate(run_time=0.4).move_to(g.at((2, 1))),
                         m.AnimationGroup(flash(g.at((2, 1)), RED), m.FadeIn(c3, shift=0.1 * m.DOWN), run_time=0.5)),
            m.AnimationGroup(bot.animate(run_time=0.8, path_arc=1.2).move_to(g.at(START)), m.FadeOut(c3)),
        )
        self.say(
            "[The shortest way hugs the edge: thirteen steps.] [The top road is safer, but takes seventeen.]",
            m.AnimationGroup(m.Create(edge, run_time=1.4), m.FadeIn(leg_edge, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(top, run_time=1.6), m.FadeIn(leg_top, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.remove(*floor, *cliffs, goal)          # the parts were faded in one by one: keep the grid as one object
        self.add(g)
        self.s1_extra = m.VGroup(start_lab, goal_lab, cliff_lab, src, edge, top, legs)

    # ------------------------------------------------------------------ 2 · action values

    def action_values(self):
        self.next_section("2 · One value per move", notes=(
            "Q(s, a): the value of move a in cell s (Episode 8 learned it by averaging returns). Rover acts ε-greedily "
            f"(epsilon_greedy): with ε = {EPS} and 4 moves, the best move has probability 1 − ε + ε/4 = {P_BEST:.3f}, "
            f"each other move ε/4 = {P_OTHER:.3f}. After every move: Q[s][a] += alpha * (target - Q[s][a]) "
            "(rl_reference td_control) — Episode 9's TD error, for moves. Q starts at 0 (q0 = 0)."))
        big = self.g1
        g = cliff_view(0.55, (-3.1, 1.3, 0))
        self.g2 = g
        bot = self.bot
        here = m.Square(0.55 - 0.02).set_stroke(VALUE, 4).set_fill(opacity=0).move_to(g.at(START)).set_z_index(3)
        self.here = here
        cross = Cross((2.0, 1.3, 0), 2.2, None, frame=VALUE)
        self.cross = cross
        cap1 = text("Rover's cell", 22, INK, weight=m.BOLD)
        cap2 = text("a value for each move", 20, MUTED)
        cap = m.VGroup(cap1, cap2).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(cap, 3.45, 1.3)
        self.cap = cap

        q_lab = m.MathTex("Q(s, a)", font_size=44, color=VALUE)
        q_desc = text("the value of move a in cell s", 22, MUTED)
        ep8 = pill("as in episode 8", VALUE, BLUE_FILL, 20)
        row1 = m.VGroup(q_lab, q_desc, ep8).arrange(m.RIGHT, buff=0.4)
        left_at(row1, -6.4, -0.35)

        dice = icon(DICE, 0.5)
        eps = pill("ε = 0.1", EXPLORE, VIOLET_FILL, 20, mono=True)
        best = colored([("best move", ACTION), (f"{100 * P_BEST:.1f}%", ACTION)], 22)
        other = colored([("·", MUTED), ("each other move", EXPLORE), (f"{100 * P_OTHER:.1f}%", EXPLORE)], 22)
        row2 = m.VGroup(dice, eps, best, other).arrange(m.RIGHT, buff=0.35)
        left_at(row2, -6.4, -1.3)

        src = "Q[s][a] += alpha * (target - Q[s][a])"
        rule = code(src, 26, INK, t2c={r"\(target - Q\[s\]\[a\]\)": SIGNAL, r"Q\[s\]\[a\]": VALUE})
        left_at(rule, -6.4, -2.25)
        td = pill("the TD error · episode 9", SIGNAL, PINK_FILL, 20).next_to(rule, m.RIGHT, buff=0.45)
        target_glyphs = glyphs(rule, "target")
        two = m.VGroup(text("the target?", 24, INK, weight=m.BOLD),
                       pill("SARSA", SA_C, CORAL_FILL, 22), text("or", 24, MUTED), pill("Q-learning", QL_C, BLUE_FILL, 22))
        two.arrange(m.RIGHT, buff=0.35)
        left_at(two, -6.4, -3.2)
        self.two = two
        self.lower2 = m.VGroup(row1, row2, rule, td)

        self.say(
            "[To choose, Rover keeps a value for every move in every cell]: [Q of s and a], "
            "[learned from experience, as in episode eight.]",
            m.AnimationGroup(self.new_header("02 · action values", "One value per move"), m.FadeOut(self.s1_extra),
                             m.ReplacementTransform(big, g), bot.animate.scale(0.3 / 0.42).move_to(g.at(START)),
                             m.Succession(m.Wait(0.6), m.AnimationGroup(m.Create(here), m.FadeIn(cross, scale=0.8),
                                                                        m.FadeIn(cap, shift=0.1 * m.LEFT))),
                             run_time=1.8),
            m.FadeIn(m.VGroup(q_lab, q_desc), shift=0.1 * m.UP),
            m.FadeIn(ep8, shift=0.1 * m.LEFT),
        )
        self.say(
            "[It acts epsilon-greedily]: [usually the move with the best value], [but one move in ten, a random one.]",
            m.AnimationGroup(m.FadeIn(dice, scale=0.6), m.FadeIn(eps, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(best, shift=0.1 * m.UP), m.Circumscribe(cross.arrows["E"], color=ACTION, buff=0.06)),
            m.AnimationGroup(m.Rotate(dice, angle=m.TAU / 2, run_time=0.7), m.FadeIn(other, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.Circumscribe(cross.arrows[a], color=EXPLORE, buff=0.06) for a in "NSW"],
                                           lag_ratio=0.25)),
        )
        self.say(
            "[After every move, it nudges that move's value toward a target]: [the TD error from episode nine, now for moves.]",
            m.FadeIn(rule, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(td, shift=0.1 * m.LEFT),
                             m.Circumscribe(glyphs(rule, "(target - Q[s][a])"), color=SIGNAL, buff=0.06)),
        )
        self.say(
            "[So the only question is the target.] [There are two answers.]",
            m.Circumscribe(target_glyphs, color=INK, buff=0.06),
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in two], lag_ratio=0.2),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ 3 · two targets

    def two_targets(self):
        self.next_section("3 · Two targets", notes=(
            "Rover steps up to (1, 2), then right onto the edge cell (2, 2): r = −1. The next cell's values are seed 0's "
            "Q-learning table after 500 episodes — equal to the optimal values from value iteration (asserted): up "
            f"{NEXT['N']}, right {NEXT['E']}, down (into the cliff) {NEXT['S']}, left {NEXT['W']}. Q-learning's target "
            f"(S&B eq. 6.8): r + max Q(s′, ·) = −1 + (−11) = {T_QL:.0f}, whatever Rover does next (off-policy). SARSA's "
            "(S&B eq. 6.7): r + Q(s′, a′) with a′ drawn ε-greedily, the move Rover will really make (on-policy): right "
            f"with probability {P_BEST:.3f} → {T_SA_E:.0f}; down, into the cliff, with probability {P_OTHER} (one time in "
            f"forty) → {T_SA_S:.0f}. Illustration: both targets use the same table here. Q-learning: Watkins (1989). "
            "SARSA: Rummery & Niranjan (1994); the name: Sutton (1996)."))
        g, bot, here, cross, cap = self.g2, self.bot, self.here, self.cross, self.cap
        cap_b = m.VGroup(text("the next cell", 22, INK, weight=m.BOLD), text("Q-learning, 500 episodes", 20, MUTED),
                         text("= the optimal values", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(cap_b, 3.45, 1.3)
        chip_ = reward_chip(-1, 20).move_to(g.at(EDGE_CELL) + [0.0, 0.62, 0]).set_z_index(6)

        pill_q = pill("Q-learning", QL_C, BLUE_FILL, 22)
        left_at(pill_q, -6.4, -0.4)
        row_q = colored([("target =", INK), ("−1", REWARD), ("+", INK), ("max", QL_C), ("=", INK), ("−1", REWARD), ("+", INK),
                         (f"({num(NEXT['E'])})", VALUE), ("=", INK), (num(T_QL), INK)], 24, weight=m.BOLD)
        left_at(row_q, -3.9, -0.4)
        off = pill("off-policy", QL_C, BLUE_FILL, 20, mono=True)
        left_at(off, 3.35, -0.4)
        off_note = text("the best next move, whatever Rover does", 20, MUTED)
        left_at(off_note, -3.9, -0.92)

        pill_s = pill("SARSA", SA_C, CORAL_FILL, 22)
        left_at(pill_s, -6.4, -1.6)
        row_s = text("first, pick the next move the way Rover acts", 22, INK)
        left_at(row_s, -3.9, -1.6)
        on = pill("on-policy", SA_C, CORAL_FILL, 20, mono=True)
        left_at(on, 3.35, -1.6)
        row_s1 = colored([("right", ACTION), (f"{100 * P_BEST:.1f}%", MUTED), ("→", MUTED), ("target =", INK), ("−1", REWARD),
                          ("+", INK), (f"({num(NEXT['E'])})", VALUE), ("=", INK), (num(T_SA_E), INK)], 24)
        row_s2 = colored([("down ", EXPLORE), (f" {100 * P_OTHER:.1f}%", MUTED), ("→", MUTED), ("target =", INK), ("−1", REWARD),
                          ("+", INK), (f"({num(NEXT['S'])})", VALUE), ("=", INK), (num(T_SA_S), RED)], 24, weight=m.BOLD)
        left_at(row_s1, -3.9, -2.25)
        left_at(row_s2, -3.9, -2.85)
        res_s2 = row_s2[-4:]
        forty = code("1 in 40", 20, EXPLORE).next_to(row_s2, m.RIGHT, buff=0.4)
        on_note = text("the moves it really makes, mistakes included", 20, MUTED)
        left_at(on_note, -3.9, -3.4)
        refs = m.VGroup(right_at(cite("Watkins, 1989"), 6.5, -0.92), right_at(cite("Rummery & Niranjan, 1994"), 6.5, -3.4))
        dice = icon(DICE, 0.42).next_to(pill_s, m.DOWN, buff=0.25).align_to(pill_s, m.LEFT).shift(0.1 * m.RIGHT)

        self.say(
            "[Say Rover steps up, then right, onto the edge.] [Here's what its next moves are worth], "
            "[after five hundred episodes of Q-learning.]",
            m.AnimationGroup(self.new_header("03 · two targets", "The next move, or the best one?"),
                             m.FadeOut(self.lower2), m.FadeOut(self.two),
                             m.Succession(m.AnimationGroup(bot.animate(run_time=0.45).move_to(g.at(SPLIT)),
                                                           here.animate(run_time=0.45).move_to(g.at(SPLIT))),
                                          m.AnimationGroup(bot.animate(run_time=0.45).move_to(g.at(EDGE_CELL)),
                                                           here.animate(run_time=0.45).move_to(g.at(EDGE_CELL))),
                                          m.FadeIn(chip_, shift=0.1 * m.UP, run_time=0.3))),
            m.AnimationGroup(cross.set_values(NEXT), cross.tint("S", RED, RED_FILL), m.FadeOut(cap), m.FadeIn(cap_b)),
            m.Circumscribe(cap_b[1:], color=MUTED, buff=0.08),
        )
        self.say(
            "[The Q-learning target uses the best of them]: [minus one for the step, and minus eleven to go: minus twelve.]",
            m.AnimationGroup(m.FadeIn(pill_q, shift=0.1 * m.RIGHT), m.FadeOut(chip_),
                             m.Circumscribe(m.VGroup(cross.tris["E"], cross.vals["E"]), color=QL_C, buff=0.04)),
            m.FadeIn(row_q, shift=0.1 * m.UP),
        )
        self.say(
            "[It doesn't matter what Rover will really do next]: [it learns about the best move. That's off-policy.]",
            m.FadeIn(off_note, shift=0.1 * m.UP),
            m.FadeIn(off, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Instead, SARSA first picks its next move, the way Rover really acts.] "
            "[Usually it's the move to the right: the same minus twelve.]",
            m.AnimationGroup(m.FadeIn(pill_s, shift=0.1 * m.RIGHT), m.FadeIn(row_s, shift=0.1 * m.UP),
                             m.Succession(m.FadeIn(dice, scale=0.6, run_time=0.3), m.Rotate(dice, angle=m.TAU / 2, run_time=0.6))),
            m.AnimationGroup(m.FadeIn(row_s1, shift=0.1 * m.UP), m.Circumscribe(cross.arrows["E"], color=ACTION, buff=0.06)),
        )
        self.say(
            "[But one time in forty, the move is down, into the cliff], [and the target drops to minus one hundred fourteen.]",
            m.AnimationGroup(m.Rotate(dice, angle=m.TAU / 2, run_time=0.6), m.FadeIn(forty),
                             m.Circumscribe(m.VGroup(cross.tris["S"], cross.vals["S"]), color=EXPLORE, buff=0.04)),
            m.AnimationGroup(m.FadeIn(row_s2, shift=0.1 * m.UP), m.Succession(m.Wait(0.5), m.Circumscribe(res_s2, color=RED, buff=0.06))),
        )
        self.say(
            "[It learns about the moves it really makes, mistakes included]: [that's on-policy.]",
            m.AnimationGroup(m.FadeIn(on_note, shift=0.1 * m.UP), m.FadeIn(refs)),
            m.FadeIn(on, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · in code

    def in_code(self):
        self.next_section("4 · In code", notes=(
            "Excerpts of rl_reference.td_control (its names; the terminal case `0.0 if done` is left out): SARSA picks "
            "a2 = epsilon_greedy(Q[s2], …) BEFORE the update and aims at Q[s2][a2]; Q-learning aims at max(Q[s2].values()) "
            "and picks a2 afterwards. Expected SARSA (S&B eq. 6.9) would aim at expected_q(Q[s2], epsilon). The name: the "
            "quintuple (s, a, r, s′, a′) (S&B §6.4: 'This quintuple gives rise to the name Sarsa'); Rummery & Niranjan "
            "(1994) called it Modified Connectionist Q-learning; 'Sarsa' was introduced by Sutton (1996)."))
        sa_src = ("s2, r, done = env.step(s, a, rng)\n"
                  "a2 = epsilon_greedy(Q[s2], epsilon, rng)\n"
                  "target = r + gamma * Q[s2][a2]\n"
                  "Q[s][a] += alpha * (target - Q[s][a])")
        ql_src = ("s2, r, done = env.step(s, a, rng)\n"
                  "target = r + gamma * max(Q[s2].values())\n"
                  "Q[s][a] += alpha * (target - Q[s][a])\n"
                  "a2 = epsilon_greedy(Q[s2], epsilon, rng)")
        sa = code_block(sa_src, size=20, title="rl_reference.py · sarsa", accent=SA_C, width=7.0)
        ql = code_block(ql_src, size=20, title="rl_reference.py · q_learning", accent=QL_C, width=7.0)
        m.VGroup(sa, ql).arrange(m.DOWN, buff=0.35)
        left_at(sa, -6.5, 1.15)
        left_at(ql, -6.5, -1.75)
        for blk, src in ((sa, sa_src), (ql, ql_src)):
            paint(blk[1], src, {r"\(target - Q\[s\]\[a\]\)": SIGNAL})
        sa_pick = glyphs(sa[1], "a2 = epsilon_greedy(Q[s2], epsilon, rng)")
        sa_aim = glyphs(sa[1], "Q[s2][a2]")
        ql_aim = glyphs(ql[1], "max(Q[s2].values())")
        ql_pick = glyphs(ql[1], "a2 = epsilon_greedy(Q[s2], epsilon, rng)")
        for blk in (sa, ql):
            blk[1].set_z_index(2)                   # the code above its highlight bands

        def mark(part, color, blk):
            """A bar in the card's margin beside the line, and the part recoloured (no shape over the code)."""
            bar = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=0.3).set_stroke(width=0).set_fill(color, 1)
            bar.move_to([blk[0].get_left()[0] + 0.16, part.get_center()[1], 0])
            return m.AnimationGroup(m.FadeIn(bar, shift=0.1 * m.RIGHT), part.animate.set_color(color))

        box_pick, box_aim = mark(sa_pick, SA_C, sa), mark(sa_aim, SA_C, sa)
        box_max, box_late = mark(ql_aim, QL_C, ql), mark(ql_pick, QL_C, ql)

        xs = 3.85
        letters = [("s", STATE), ("a", ACTION), ("r", REWARD), ("s′", STATE), ("a′", ACTION)]
        chips = m.VGroup(*[chip(lab, col, FILLS[col], width=0.78, height=0.72, size=30) for lab, col in letters])
        chips.arrange(m.RIGHT, buff=0.34).move_to([xs, 1.75, 0])
        words = m.VGroup(*[code(w, 20, col) for w, (_, col) in zip(("state", "action", "reward", "state", "action"), letters)])
        for w, c in zip(words, chips):
            w.next_to(c, m.DOWN, buff=0.14)
        on_baseline(words)
        name = title_text("SARSA", 60, SA_C).move_to([xs, 0.35, 0])
        name_src = cite("Rummery & Niranjan, 1994 · named by Sutton, 1996").move_to([xs, -0.3, 0])
        q_name = m.VGroup(code("max", 30, QL_C, weight=m.BOLD), text("the best next move, picked later", 22, INK))
        q_name.arrange(m.RIGHT, buff=0.3).move_to([xs, -1.55, 0])
        q_src = cite("Q-learning · Watkins, 1989").move_to([xs, -2.1, 0])

        self.say(
            "[In code, the two learners differ in a single line.]",
            m.AnimationGroup(self.new_header("04 · in code", "One line apart"),
                             m.FadeIn(sa, shift=0.1 * m.UP), m.FadeIn(ql, shift=0.1 * m.UP)),
        )
        self.say(
            "[Here, SARSA picks its next action before the update, and aims at it.] "
            "[State, action, reward, state, action]: [that's where the name comes from.]",
            m.AnimationGroup(box_pick, m.Succession(m.Wait(0.6), box_aim)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(m.VGroup(c, w), shift=0.1 * m.UP) for c, w in zip(chips, words)],
                                           lag_ratio=0.3)),
            m.AnimationGroup(m.FadeIn(name, shift=0.15 * m.UP), m.FadeIn(name_src)),
        )
        self.say(
            "[And Q-learning aims at the max instead], [choosing its next move only afterwards.]",
            m.AnimationGroup(box_max, m.FadeIn(q_name, shift=0.1 * m.UP), m.FadeIn(q_src)),
            box_late,
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · two Rovers

    def two_rovers(self):
        self.next_section("5 · Two Rovers", notes=(
            f"Seed 0, 500 episodes each, ε = {EPS}, α = {ALPHA} (q_learning / sarsa; the same runs as classic.json's example "
            f"paths). Greedy paths (greedy_path): Q-learning {len(QL_PATH) - 1} steps along the edge (row 2); SARSA "
            f"{len(SA_PATH) - 1} steps along the top row — it reaches the goal (asserted; in {LOOPS:.0%} of SARSA runs the "
            "final greedy path loops instead, classic.md §6). Then each final table acts ε-greedily for three episodes "
            "(no learning; illustrative draws: default_rng(176) for Q-learning, default_rng(595) for SARSA): Q-learning "
            f"{', '.join(num(t) for _, t in QL_RUNS)} — in episode 2 a random step down at (6, 2) falls into the cliff "
            f"(−100, back to the start); SARSA {', '.join(num(t) for _, t in SA_RUNS)} — its random step down at (6, 4) "
            "costs two moves."))
        ga = cliff_view(0.56, (-2.95, 1.1, 0))
        gb = cliff_view(0.56, (-2.95, -1.95, 0))
        ba = rover(0.3).move_to(ga.at(START)).set_z_index(4)
        bb = rover(0.3).move_to(gb.at(START)).set_z_index(4)
        pa = route(ga, QL_PATH, ACTION, 5)
        pb = route(gb, SA_PATH, ACTION, 5)
        x0 = 1.05
        la = pill("Q-learning · off-policy", QL_C, BLUE_FILL, 22)
        left_at(la, x0, 1.95)
        lb = pill("SARSA · on-policy", SA_C, CORAL_FILL, 22)
        left_at(lb, x0, -1.1)
        ra = text(f"best route: the edge, {len(QL_PATH) - 1} steps", 22, INK)
        left_at(ra, x0, 1.38)
        rb = text(f"best route: the top road, {len(SA_PATH) - 1} steps", 22, INK)
        left_at(rb, x0, -1.67)
        slots = [x0 + 0.55 + 1.55 * k for k in range(3)]

        def tally_heads(y):
            return m.VGroup(*[code(f"ep {k + 1}", 20, MUTED).move_to([x, y, 0]) for k, x in enumerate(slots)])

        ha, hb = tally_heads(0.75), tally_heads(-2.3)
        ya, yb = 0.3, -2.75

        def total(t, x, y):
            return code(num(t), 28, RED if t <= W.cliff_reward else INK, weight=m.BOLD).move_to([x, y, 0])

        ta = [total(t, x, ya) for (_, t), x in zip(QL_RUNS, slots)]
        tb = [total(t, x, yb) for (_, t), x in zip(SA_RUNS, slots)]
        eps = pill("both still explore: ε = 0.1", EXPLORE, VIOLET_FILL, 20)
        right_at(eps, 6.5, -3.45)
        fall_at = ga.at(cliff_cell((6, 2), "S"))
        fell = m.VGroup(m.Line(fall_at + [-0.14, -0.14, 0], fall_at + [0.14, 0.14, 0]).set_stroke(INK, 4),
                        m.Line(fall_at + [-0.14, 0.14, 0], fall_at + [0.14, -0.14, 0]).set_stroke(INK, 4)).set_z_index(3)
        odd = m.Circle(radius=0.2).set_stroke(EXPLORE, 3).move_to(gb.at((6, 3))).set_z_index(3)

        self.say(
            "[After five hundred episodes, the Q-learning Rover's best route hugs the edge]: [thirteen steps.]",
            m.AnimationGroup(self.new_header("05 · two rovers", "The edge and the safe road"), m.FadeIn(ga),
                             m.FadeIn(ba, shift=0.1 * m.DOWN), m.FadeIn(la, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(pa, run_time=1.3), m.FadeIn(ra, shift=0.1 * m.UP)),
        )
        self.say(
            "[For SARSA, the best route takes the top road]: [seventeen steps.]",
            m.AnimationGroup(m.FadeIn(gb), m.FadeIn(bb, shift=0.1 * m.DOWN), m.FadeIn(lb, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(pb, run_time=1.5), m.FadeIn(rb, shift=0.1 * m.UP)),
        )
        self.say(
            "[Now watch them act, still exploring one move in ten.]",
            m.AnimationGroup(m.FadeIn(eps, shift=0.1 * m.UP), m.FadeIn(ha), m.FadeIn(hb),
                             pa.animate.set_stroke(opacity=0.35), pb.animate.set_stroke(opacity=0.35)),
        )
        self.say(
            "[Along the edge, a clean run costs thirteen.] [Then one random step down], "
            "[and it falls: minus one hundred, and back to the start.]",
            m.Succession(act(ba, ga, QL_RUNS[0][0]), m.FadeIn(ta[0], shift=0.1 * m.UP, run_time=0.3), home(ba, ga)),
            act(ba, ga, QL_RUNS[1][0][:6]),
            m.Succession(act(ba, ga, QL_RUNS[1][0][6:], marks=fell), m.FadeIn(ta[1], shift=0.1 * m.UP, run_time=0.3),
                         home(ba, ga)),
        )
        self.say(
            "[Up on the top road, a run costs seventeen.] [A random step down there costs two moves], [never a fall.]",
            m.AnimationGroup(m.Succession(act(ba, ga, QL_RUNS[2][0]), m.FadeIn(ta[2], shift=0.1 * m.UP, run_time=0.3)),
                             m.Succession(act(bb, gb, SA_RUNS[0][0]), m.FadeIn(tb[0], shift=0.1 * m.UP, run_time=0.3),
                                          home(bb, gb))),
            m.Succession(act(bb, gb, SA_RUNS[1][0], marks=odd), m.FadeIn(tb[1], shift=0.1 * m.UP, run_time=0.3),
                         home(bb, gb)),
            m.Succession(act(bb, gb, SA_RUNS[2][0]), m.FadeIn(tb[2], shift=0.1 * m.UP, run_time=0.3)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · measured

    def measured_runs(self):
        self.next_section("6 · Measured on this Mac", notes=(
            f"classic.md §6 / classic.json cliff: {RUNS} runs × {CL['episodes']} episodes per learner, ε = {EPS}, "
            f"α = {ALPHA} (the book does not state α; 0.5 is the common reproduction's — Appendix A row 10), γ = 1, Q "
            "starts at 0. Curves: reward per episode averaged over the runs (reward_per_episode; episode 1 ≈ −1,700 is "
            f"off the chart). Average over episodes 101–500: Q-learning {QL_R:.2f}, SARSA {SA_R:.2f}, expected SARSA "
            f"{ES_R:.2f} (headline cliff_*_reward). Episodes with a fall per run: {FALLS_QL:.1f} / {FALLS_SA:.1f} / "
            f"{METH['expected_sarsa']['episodes_with_a_fall_per_run']:.1f}. The book's plot (S&B 2018, Example 6.6): SARSA "
            "≈ −25, Q-learning ≈ −50 (smoothing not stated). Expected SARSA does better than both online, as in S&B "
            "Fig. 6.3. −13 = the edge path's return with no exploring."))
        plot = Plot((0, 500), (-100, 0), width=8.0, height=3.9, center=(-1.35, -0.3, 0),
                    x_ticks=[0, 100, 200, 300, 400], y_ticks=[-100, -75, -50, -25, 0],
                    x_label="episodes (of 500)", y_label="reward per episode")

        def curve(key, color, width=5, opacity=1.0):
            c = METH[key]["reward_per_episode"]
            return plot.line(c["x"], c["y"], color, width).set_stroke(opacity=opacity)

        cq, cs, ce = curve("q_learning", QL_C), curve("sarsa", SA_C), curve("expected_sarsa", ES_C, 3, 0.8)
        best = plot.hline(-13, GREEN, 2)
        best_lab = code("−13: the edge, never exploring", 20, GREEN).next_to(plot.coords(8, -13), m.UP, buff=0.1)
        best_lab.align_to(plot.coords(8, -13), m.LEFT)
        xe = plot.coords(500, 0)[0] + 0.25
        lq = colored([("Q-learning", QL_C), (minus(f"{QL_R:.1f}"), QL_C)], 22, weight=m.BOLD)
        ls = colored([("SARSA", SA_C), (minus(f"{SA_R:.1f}"), SA_C)], 22, weight=m.BOLD)
        le = colored([("expected SARSA", ES_C), (minus(f"{ES_R:.1f}"), ES_C)], 22, weight=m.BOLD)
        left_at(lq, xe, plot.coords(500, QL_R)[1])
        left_at(ls, xe, plot.coords(500, SA_R)[1] - 0.12)
        left_at(le, xe, plot.coords(500, ES_R)[1] + 0.26)
        avg = cite("labels: average of episodes 101–500")
        right_at(avg, 6.5, 2.0)
        falls = m.VGroup(text("episodes with a fall, of 500", 20, MUTED),
                         colored([("Q-learning", QL_C), (f"≈ {FALLS_QL:.0f}", QL_C)], 22),
                         colored([("SARSA", SA_C), (f"≈ {FALLS_SA:.0f}", SA_C)], 22)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        left_at(falls, xe, -1.25)
        meas = cite(f"mean of {RUNS} runs × {CL['episodes']} episodes · ε = 0.1 · α = 0.5 · measured on this Mac")
        right_at(meas, 6.5, 2.4)
        b_sa, b_ql = f"SARSA ≈ {num(BOOK['sarsa'])}", f"Q-learning ≈ {num(BOOK['q_learning'])}"
        book_main = text(f"the book: {b_sa}, {b_ql} · Sutton & Barto, 2018, Example 6.6", 20, MUTED,
                         t2c={b_sa: SA_C, b_ql: QL_C})
        book_alpha = text("· its α is not stated (ours: 0.5)", 20, MUTED)
        book = m.VGroup(book_main, book_alpha).arrange(m.RIGHT, buff=0.15).move_to([0, -3.4, 0])

        self.say(
            "[Now the real runs, measured on this Mac]: [five hundred runs of five hundred episodes, for each learner.]",
            m.AnimationGroup(self.new_header("06 · measured", "Reward per episode"), m.FadeIn(plot)),
            m.FadeIn(meas, shift=0.1 * m.DOWN),
        )
        self.say(
            "[Here, Q-learning settles near minus fifty per episode], [far below a clean run's minus thirteen]: "
            "[it falls in about one episode in four.]",
            m.Succession(m.Create(cq, run_time=2.0), m.AnimationGroup(m.FadeIn(lq, shift=0.1 * m.LEFT), m.FadeIn(avg))),
            m.AnimationGroup(m.Create(best), m.FadeIn(best_lab)),
            m.AnimationGroup(m.FadeIn(falls[:2], shift=0.1 * m.UP)),
        )
        self.say(
            "[And SARSA settles near minus twenty-seven]: [it falls far less often], [and loses about half as much.]",
            m.Succession(m.Create(cs, run_time=2.0), m.FadeIn(ls, shift=0.1 * m.LEFT)),
            m.FadeIn(falls[2], shift=0.1 * m.UP),
            m.Circumscribe(ls, color=SA_C, buff=0.08),
        )
        self.say(
            "[The textbook's figure shows the same picture]: [about minus twenty-five against minus fifty], "
            "[though the book doesn't say which step size it used.]",
            m.FadeIn(book_main, shift=0.1 * m.UP),
            m.LaggedStart(m.Circumscribe(ls, color=SA_C, buff=0.08), m.Circumscribe(lq, color=QL_C, buff=0.08), lag_ratio=0.4),
            m.FadeIn(book_alpha, shift=0.1 * m.UP),
        )
        self.say(
            "[A third learner, expected SARSA, averages over its own dice instead of rolling them]: [about minus twenty-one.]",
            m.Create(ce, run_time=2.0),
            m.FadeIn(le, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · whose values?

    def whose_values(self):
        self.next_section("7 · Whose values?", notes=(
            "Seed 0's final tables at the cell where the two routes split, (1, 2), just above the start (rounded): "
            f"Q-learning {SPLIT_QL} — right (the edge) is best, the optimal values a Rover that never explores would earn "
            f"(value iteration gives right −12, up −14); SARSA {SPLIT_SA} — up is best: its values are those of its own "
            "ε-greedy behaviour, random steps included (they also keep moving: α is a constant 0.5). S&B Example 6.6: "
            "'Sarsa, on the other hand, takes the action selection into account and learns the longer but safer path'."))
        g = cliff_view(0.42, (0, 1.72, 0))
        pq = route(g, QL_PATH, ACTION, 4)
        ps = route(g, SA_PATH, ACTION, 4, dashed=True)
        mark = m.Square(0.42 - 0.02).set_stroke(INK, 4).set_fill(opacity=0).move_to(g.at(SPLIT)).set_z_index(3)
        cq = Cross((-3.3, -1.05, 0), 2.3, SPLIT_QL, frame=QL_C)
        cs = Cross((3.3, -1.05, 0), 2.3, SPLIT_SA, frame=SA_C)
        tq = pill("Q-learning's values", QL_C, BLUE_FILL, 22).next_to(cq, m.UP, buff=0.2)
        ts = pill("SARSA's values", SA_C, CORAL_FILL, 22).next_to(cs, m.UP, buff=0.2)
        nq = text("for a Rover that never slips", 22, MUTED).next_to(cq, m.DOWN, buff=0.22)
        ns = text("for the Rover it really is", 22, MUTED).next_to(cs, m.DOWN, buff=0.22)
        lab_split = code("the routes split here", 20, INK).next_to(g, m.LEFT, buff=0.3).align_to(mark, m.UP)
        vs = m.VGroup(title_text("off-policy", 34, QL_C), text("vs", 24, MUTED), title_text("on-policy", 34, SA_C))
        vs.arrange(m.DOWN, buff=0.2).move_to([0, -1.05, 0])

        self.say(
            "[Why the difference? Look at the cell where the two routes split], [just above the start.]",
            m.AnimationGroup(self.new_header("07 · whose values?", "Priced with or without mistakes"), m.FadeIn(g),
                             m.Create(pq), m.Create(ps)),
            m.AnimationGroup(m.Create(mark), m.FadeIn(lab_split, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[The Q-learning values are for a Rover that never slips]: [right, along the edge, is worth minus twelve]; "
            "[up, minus fourteen.]",
            m.AnimationGroup(m.FadeIn(cq, shift=0.1 * m.UP), m.FadeIn(tq), m.FadeIn(nq)),
            m.AnimationGroup(cq.tint("E", QL_C, BLUE_FILL), m.Circumscribe(cq.vals["E"], color=QL_C, buff=0.06)),
            m.Circumscribe(cq.vals["N"], color=QL_C, buff=0.06),
        )
        self.say(
            "[But the SARSA values include its own random steps]: [along the edge, minus fifty-seven]; [up and away, minus twenty.]",
            m.AnimationGroup(m.FadeIn(cs, shift=0.1 * m.UP), m.FadeIn(ts), m.FadeIn(ns)),
            m.AnimationGroup(cs.tint("E", RED, RED_FILL), m.Circumscribe(cs.vals["E"], color=RED, buff=0.06)),
            m.AnimationGroup(cs.tint("N", SA_C, CORAL_FILL), m.Circumscribe(cs.vals["N"], color=SA_C, buff=0.06)),
        )
        self.say(
            "[Off-policy: the best route for a perfect Rover.] [On-policy: the best route for the Rover it really is.]",
            m.AnimationGroup(m.FadeIn(vs[0], shift=0.1 * m.UP), m.Circumscribe(m.VGroup(tq, cq), color=QL_C, buff=0.1)),
            m.AnimationGroup(m.FadeIn(vs[1:], shift=0.1 * m.UP), m.Circumscribe(m.VGroup(ts, cs), color=SA_C, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · in the limit

    def in_the_limit(self):
        self.next_section("8 · In the limit", notes=(
            "S&B 2018 Example 6.6: 'Of course, if ε were gradually reduced, then both methods would asymptotically converge "
            "to the optimal policy.' Watkins (1989, PhD thesis, Cambridge) introduced Q-learning; Watkins & Dayan (1992, "
            "Machine Learning 8:279–292) proved that tabular Q-learning converges to the optimal action values with "
            "probability 1 if every action is tried in every state again and again and the step sizes satisfy Σα = ∞, "
            "Σα² < ∞ (bounded rewards; a lookup table — not neural networks, Episode 12). SARSA's convergence also needs "
            "the policy to become greedy in the limit (Singh et al., 2000). Which to use (notes §10, misconceptions): it "
            "depends on whether mistakes during learning cost anything; off-policy learning can learn from data another "
            "policy produced — replay (Episode 13) is one use of that."))
        both = m.VGroup(pill("SARSA", SA_C, CORAL_FILL, 22), pill("Q-learning", QL_C, BLUE_FILL, 22))
        both.arrange(m.DOWN, buff=0.45)
        left_at(both, -6.4, 1.55)
        goal_p = pill("the 13-step edge path", ACTION, CORAL_FILL, 22)
        left_at(goal_p, -1.6, 1.55)
        arrs = m.VGroup(*[m.Arrow(p.get_right() + [0.1, 0, 0], goal_p.get_left() + [-0.1, 0, 0], buff=0.05, color=MUTED,
                                  stroke_width=4, max_tip_length_to_length_ratio=0.12) for p in both])
        fade = code("ε → 0", 22, EXPLORE, weight=m.BOLD).move_to((both.get_right() + goal_p.get_left()) / 2 + [0.0, 0.75, 0])
        quote = cite("“if ε were gradually reduced, both methods would converge to the optimal policy”")
        left_at(quote, -6.4, 0.5)
        quote_src = cite("Sutton & Barto, 2018, Example 6.6")
        left_at(quote_src, -6.4, 0.12)

        proof = card(6.0, 2.7, QL_C, CARD, radius=0.16, stroke=2)
        left_at(proof, 0.55, -1.75)
        p_title = text("Q-learning finds the best values", 24, INK, weight=m.BOLD)
        p1 = text("if every move in every cell keeps being tried", 20, MUTED)
        p2 = m.MathTex(r"\sum_t \alpha_t = \infty", r"\qquad", r"\sum_t \alpha_t^2 < \infty", font_size=30, color=INK)
        p2_lab = text("step sizes:", 20, MUTED)
        p3 = text("a lookup table · with probability 1", 20, MUTED)
        body = m.VGroup(p_title, p1, m.VGroup(p2_lab, p2).arrange(m.RIGHT, buff=0.3), p3).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        body.move_to(proof).align_to(proof, m.LEFT).shift(0.35 * m.RIGHT)
        who = cite("Watkins, 1989 · Watkins & Dayan, 1992").next_to(proof, m.DOWN, buff=0.15).align_to(proof, m.RIGHT)

        use = m.VGroup(
            takeaway(recolored("alert", SA_C, CORAL_FILL, prefix=PREFIX), "Mistakes cost while learning?",
                     "on-policy (SARSA) plays it safe", SA_C),
            takeaway(recolored("database", QL_C, BLUE_FILL, prefix=PREFIX), "Moves made by someone else?",
                     "off-policy learns from them · episode 13", QL_C),
        ).arrange(m.DOWN, buff=0.3)
        left_at(use, -6.4, -2.02)

        self.say(
            "[Neither is wrong. If exploring fades over time], [both end up on the edge path], [as the textbook notes.]",
            m.AnimationGroup(self.new_header("08 · in the limit", "Both reach the best"),
                             m.FadeIn(both, shift=0.1 * m.RIGHT), m.FadeIn(fade, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in arrs], lag_ratio=0.3), m.FadeIn(goal_p, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(quote_src)),
        )
        self.say(
            "[As for Q-learning, it comes from Watkins, in nineteen eighty-nine]; "
            "[in nineteen ninety-two, he and Dayan proved it converges.]",
            m.AnimationGroup(m.FadeIn(proof), m.FadeIn(p_title, shift=0.1 * m.UP)),
            m.FadeIn(who, shift=0.1 * m.UP),
        )
        self.say(
            "[If every move in every cell keeps being tried, and the steps shrink just right], "
            "[the table converges to the best values.]",
            m.AnimationGroup(m.FadeIn(p1, shift=0.1 * m.UP), m.FadeIn(body[2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(p3, shift=0.1 * m.UP), m.Circumscribe(p_title, color=QL_C, buff=0.08)),
        )
        self.say(
            "[When mistakes during learning are costly, the on-policy learner is the safer choice.]",
            m.FadeIn(use[0], shift=0.1 * m.UP),
        )
        self.say(
            "[And off-policy learning can reuse old or borrowed moves]: [episode thirteen's replay buffer is built on it.]",
            m.FadeIn(use[1], shift=0.1 * m.UP),
            m.Circumscribe(use[1], color=QL_C, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            "Act (ε-greedy), aim (SARSA: the next move Rover will really make; Q-learning: the best next move), nudge "
            "(Q(s, a) moves a fraction α of the TD error). The cliff walk, measured here: SARSA "
            f"{SA_R:.1f} vs Q-learning {QL_R:.1f} reward per episode ({RUNS} runs). Episode 11: sparse rewards."))
        halo = glow(SIGNAL, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Act", 70, ACTION), title_text("Aim", 70, VALUE), title_text("Nudge", 70, SIGNAL))
        words.arrange(m.RIGHT, buff=1.9).move_to([0, 1.05, 0])
        on_baseline(words)
        subs = m.VGroup(text("mostly greedy", 22, MUTED), text("the next move, or the best", 22, MUTED),
                        text("by the TD error", 22, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        base = max(s_.get_center()[1] for s_ in subs)
        for s_ in subs:
            s_.set_y(base)
        pair = m.VGroup(colored([("next move:", SA_C), ("SARSA, on-policy", INK)], 24),
                        colored([("best move:", QL_C), ("Q-learning, off-policy", INK)], 24)).arrange(m.RIGHT, buff=0.9)
        pair.move_to([0, -0.85, 0])
        proof = colored([("the cliff, reward per episode:", MUTED), (f"SARSA {minus(f'{SA_R:.1f}')}", SA_C), ("·", MUTED),
                         (f"Q-learning {minus(f'{QL_R:.1f}')}", QL_C), ("· measured here", MUTED)], 22)
        proof.move_to([0, -1.75, 0])
        bot = rover(0.55).move_to([-2.3, -2.8, 0])
        nxt = pill("episode 11 · when rewards are rare", REWARD, GOLD_FILL, 22).next_to(bot, m.RIGHT, buff=0.45)

        self.say(
            "[To sum up]: [act, mostly greedy]; [aim at a target]; [and nudge the value of the move you made.]",
            m.AnimationGroup(self.new_header("09 · recap", "Act, aim, nudge"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Aim at the move you'll really make, and it's SARSA, on-policy.] [Aim at the best one, and it's Q-learning, off-policy.]",
            m.FadeIn(pair[0], shift=0.1 * m.UP),
            m.FadeIn(pair[1], shift=0.1 * m.UP),
        )
        self.say(
            "[On the cliff, measured here: minus twenty-seven against minus fifty], [the price of exploring next to an edge.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.Circumscribe(words[0], color=EXPLORE, buff=0.12),
        )
        self.say(
            "[Here, a reward came with every step.] [What if almost none do?]",
            m.FadeIn(bot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()
