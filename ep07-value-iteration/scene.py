"""Trial and Reward · Episode 7 · Value Iteration (Module 2 · Planning in a Known World)

Repeating the Bellman backup until nothing changes solves the world. Episode 6 ended on nine values, each defined by
its neighbours. Here: start from a guess (zero everywhere), apply the backup to every cell at once (one sweep), repeat.
On Rover's world (ROVER_WORLD, Russell & Norvig's 4×3) the first sweep gives −0.04 everywhere except the cell left of
the charger (0.8 × (+1) − 0.04 = 0.76); then the charger's news spreads one cell per sweep (sweeps 2, 3, 5, 10 and 22
in a filmstrip), the start cell turns positive at sweep 5, and after 22 sweeps no value moves by 1e-4: the values
equal Russell & Norvig's Fig. 17.3 to three decimals. Convergence: the backup is a contraction for γ < 1; γ = 1 here
works because the best routes end. The greedy policy, replayed sweep by sweep, stops changing at sweep 12 (the last
flip is (3,1), N → W), ten sweeps before the numbers settle. The famous detour: from the bottom row Rover goes the
long way round, because north passes the puddle (landings: north 0.632, west 0.651, with the 10% slips). Policy
iteration from "always north": 5 evaluations, arrows turned 6, 1, 1, 1. The catch: every backup read the world's
transitions — we needed the map (Module 3 learns without it).

Every number on screen is computed below from rl_reference.py (value_iteration, greedy_policy, backup,
policy_iteration, policy_evaluation, RN_UTILITIES, RN_POLICY) and asserted, or loaded with measured("classic", …)
(research/experiments/classic.md §4: vi_sweeps 22, vi_policy_optimal_from_sweep 12, pi_evaluations 5).

Sources (research/notes.md §5, §6, §7, Appendix A, Appendix C row 7; facts dated October 2026):
- Russell & Norvig, Artificial Intelligence: A Modern Approach, 3rd ed. (2010), Ch. 17: Fig. 17.3 (utilities = the
  values here, to three decimals), Fig. 17.2(a) (the optimal policy, = RN_POLICY). The 4th ed.'s reported tie at (3,1)
  is not reproduced under the 3rd-ed. model (notes §5, Appendix A): West 0.651 vs North 0.632, no tie.
- Sutton & Barto 2018: §4.4, eq. 4.10 (value iteration, "turning the Bellman optimality equation into an update rule"),
  §4.3 ("Policy iteration often converges in surprisingly few iterations", p. 82).
- Denardo (1967), SIAM Review 9(2):165–177: the backup is a γ-contraction in the max norm (notes §6).
- Bertsekas & Tsitsiklis (1996), Neuro-Dynamic Programming: undiscounted episodic (stochastic shortest path) problems
  still converge when the good policies end (notes §6, misconceptions).

Illustrative: the two dots on the number line in the convergence section (a schematic of a contraction with factor ½,
not Rover's numbers). The colour scale tints values blue up to +1 and red down to −0.25 in value iteration (so the
−0.04-a-move costs show), and down to −1.5 in policy iteration (the always-north values).

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

PREFIX = "ep07"
W = ROVER_WORLD

# ---------------------------------------------------------------------------- the numbers (all computed here)

V, HISTORY = value_iteration(W)                      # HISTORY[k] = the values after k sweeps (HISTORY[0] = zeros)
SWEEPS = len(HISTORY) - 1
OPEN = [s for s in W.states if not W.is_terminal(s)]  # the nine cells with a value to find
START, NEXT_TO_CHARGER, DETOUR, UNDER_PUDDLE = W.start, (3, 3), (3, 1), (4, 1)
CHANGE = {k: max(abs(HISTORY[k][s] - HISTORY[k - 1][s]) for s in W.states) for k in range(1, SWEEPS + 1)}
ERROR = {k: max(abs(HISTORY[k][s] - V[s]) for s in W.states) for k in range(SWEEPS + 1)}
POLICIES = {k: greedy_policy(W, HISTORY[k]) for k in range(1, SWEEPS + 1)}
FLIPS = {k: [s for s in OPEN if POLICIES[k][s] != POLICIES[k - 1][s]] for k in range(2, SWEEPS + 1)}
FLIP_SWEEPS = [k for k in FLIPS if FLIPS[k]]
OPTIMAL_FROM = min(k for k in POLICIES if all(POLICIES[j] == RN_POLICY for j in range(k, SWEEPS + 1)))
FIRST_POSITIVE = min(k for k in range(SWEEPS + 1) if HISTORY[k][START] > 0)

assert len(OPEN) == 9 and W.slip == 0.2 and W.step_reward == -0.04 and ROVER_GAMMA == 1.0
assert SWEEPS == 22 == measured("classic", "headline", "vi_sweeps")                  # "twenty-two sweeps"
assert OPTIMAL_FROM == 12 == measured("classic", "headline", "vi_policy_optimal_from_sweep")
assert all(round(V[s], 3) == RN_UTILITIES[s] for s in RN_UTILITIES)                    # R&N Fig. 17.3, three decimals
assert POLICIES[SWEEPS] == greedy_policy(W, V) == RN_POLICY                            # R&N Fig. 17.2(a)
assert FLIP_SWEEPS == [2, 3, 8, 12] and FLIPS[12] == [DETOUR]                         # "sweeps two, three and eight"
assert POLICIES[11][DETOUR] == "N" and POLICIES[12][DETOUR] == "W"                    # "from north to west"
assert CHANGE[SWEEPS] < 1e-4 < CHANGE[SWEEPS - 1]                                       # "not even a ten-thousandth"
assert round(CHANGE[1], 2) == 0.76 and round(CHANGE[10], 2) == 0.04                    # "four hundredths"
assert all(0.45 < CHANGE[k] / CHANGE[k - 1] < 0.6 for k in range(18, SWEEPS + 1))       # "roughly halves"
assert round(ERROR[12], 3) == 0.026                                                     # "off by up to 0.026"
assert FIRST_POSITIVE == 5 == measured("classic", "planning", "value_iteration", "start_value_positive_from_sweep")

# --- sweep one, by hand: everything pays −0.04 except the cell left of the charger
H1 = HISTORY[1]
assert all(round(H1[s], 3) == -0.04 for s in OPEN if s != NEXT_TO_CHARGER) and round(H1[NEXT_TO_CHARGER], 3) == 0.76
_east = W.transitions(NEXT_TO_CHARGER, "E")
assert [(p, s2) for p, s2, _, _ in _east] == [(0.8, CHARGER), (0.1, NEXT_TO_CHARGER), (0.1, (3, 2))]
SWEEP1_TERMS = [0.8 * W.terminals[CHARGER], 0.1 * HISTORY[0][NEXT_TO_CHARGER], 0.1 * HISTORY[0][(3, 2)], W.step_reward]
assert abs(sum(SWEEP1_TERMS) - backup(W, HISTORY[0], NEXT_TO_CHARGER, "E")) < 1e-12
assert abs(sum(SWEEP1_TERMS) - H1[NEXT_TO_CHARGER]) < 1e-12


def news(k):
    """Cells whose value turns positive in sweep k, each with the neighbour its best move (on the old values) heads to."""
    out = []
    for s in OPEN:
        if HISTORY[k - 1][s] <= 0 < HISTORY[k][s]:
            best = max(W.actions, key=lambda a: backup(W, HISTORY[k - 1], s, a))
            out.append((W.move(s, best), s))
    return out


assert news(1) == [(CHARGER, NEXT_TO_CHARGER)]
assert sorted(s for _, s in news(2)) == [(2, 3), (3, 2)]                                # "left along the top, and down"
assert sorted(s for _, s in news(3)) == [(1, 3), (3, 1)]                                # "the top-left corner, the bottom row"
STILL_PAYING = [s for s in OPEN if HISTORY[3][s] < 0]
assert sorted(STILL_PAYING) == [(1, 1), (1, 2), (2, 1), (4, 1)] and all(round(HISTORY[3][s], 2) == -0.12 for s in STILL_PAYING)
FILM = (1, 2, 3, 5, 10, SWEEPS)                                                          # the sweeps the notes record
assert all(round(HISTORY[k][s], 4) == round(measured("classic", "planning", "value_iteration", "after_sweep", str(k), "V",
                                                     f"{s[0]},{s[1]}"), 4) for k in FILM for s in OPEN)

# --- the detour at (3,1): weigh the landings with the textbook's three-decimal values
R = RN_UTILITIES


def landing(s, a, values):
    """The slip-weighted value of where move a from s lands (the backup without the −0.04 every move pays)."""
    return sum(p * values[s2] for p, s2, _, _ in W.transitions(s, a))


LAND_N, LAND_W = landing(DETOUR, "N", R), landing(DETOUR, "W", R)
TERMS = {"N": [(0.8, (3, 2)), (0.1, (2, 1)), (0.1, UNDER_PUDDLE)], "W": [(0.8, (2, 1)), (0.1, DETOUR), (0.1, (3, 2))]}
assert all([(p, s2) for p, s2, _, _ in W.transitions(DETOUR, a)] == TERMS[a] for a in "NW")
assert round(LAND_N, 3) == 0.632 and round(LAND_W, 3) == 0.651 and round(LAND_W - LAND_N, 3) == 0.019   # notes §5
assert abs(LAND_N - (backup(W, V, DETOUR, "N") - W.step_reward)) < 1e-3
assert abs(LAND_W - (backup(W, V, DETOUR, "W") - W.step_reward)) < 1e-3
assert round(LAND_W + W.step_reward, 3) == R[DETOUR]                                    # 0.651 − 0.04 = 0.611
assert RN_POLICY[DETOUR] == RN_POLICY[UNDER_PUDDLE] == "W"


def route(s, moves=None):
    """The cells visited when every move works, following `moves` (a string) or else the optimal arrows."""
    cells = [s]
    for i in range(30):
        if W.is_terminal(cells[-1]) or (moves is not None and i == len(moves)):
            break
        cells.append(W.move(cells[-1], moves[i] if moves is not None else RN_POLICY[cells[-1]]))
    return cells


LONG = route(DETOUR)
SHORT = route(DETOUR, "NNE")
assert LONG == [DETOUR, (2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), CHARGER] and SHORT[-1] == CHARGER
EXTRA = (len(LONG) - 1) - (len(SHORT) - 1)
assert (len(SHORT) - 1, EXTRA) == (3, 4) and round(-EXTRA * W.step_reward, 2) == 0.16    # "three moves", "four extra"
assert W.terminals[CHARGER] - W.terminals[PUDDLE] == 2.0                                   # "two whole points"
assert [p for p, s2, _, _ in W.transitions((3, 2), "N") if s2 == PUDDLE] == [0.1]          # "one slip in ten"

# --- policy iteration from "always north"
PI_POLICY, PI_V, PI_POLICIES = policy_iteration(W)
PI_VALUES = [policy_evaluation(W, p, theta=1e-8) for p in PI_POLICIES]
TURNED = [sum(a[s] != b[s] for s in a) for a, b in zip(PI_POLICIES, PI_POLICIES[1:])] + [0]
assert len(PI_POLICIES) == 5 == measured("classic", "headline", "pi_evaluations")
assert TURNED == [6, 1, 1, 1, 0] and TURNED[:4] == measured("classic", "planning", "policy_iteration", "changed_cells")
assert all(a == "N" for a in PI_POLICIES[0].values()) and PI_POLICY == RN_POLICY == PI_POLICIES[-1]
START_PI = [pv[START] for pv in PI_VALUES]
assert [round(v, 3) for v in START_PI] == [-1.466, 0.508, 0.691, 0.705, 0.705]            # "minus one point four seven"
assert all(round(PI_VALUES[-1][s], 3) == R[s] for s in OPEN)

# --- the catch: the map every backup reads
MAP_ROWS = W.transitions(DETOUR, "N")
assert [(p, s2, round(r, 2), d) for p, s2, r, d in MAP_ROWS] == [(0.8, (3, 2), -0.04, False), (0.1, (2, 1), -0.04, False),
                                                                (0.1, UNDER_PUDDLE, -0.04, False)]

# ---------------------------------------------------------------------------- look

CELL = 1.4
GRID_C = np.array([-3.75, -0.5, 0])
GL, GR = GRID_C[0] - 2 * CELL, GRID_C[0] + 2 * CELL            # the grid's left and right edges
LO_VI, LO_PI = -0.25, -1.5                                      # red reaches full strength at these values
NUM_DOWN = 0.3                                                  # numbers sit this many cells below centre under arrows
ARROW_UP, ARROW_LEN = 0.14, 0.42
ANGLE = {"N": math.pi / 2, "E": 0.0, "S": -math.pi / 2, "W": math.pi}
MAP = "ep07-map"
READOUT_Y = 2.15

VI_SRC = """V = {s: 0.0 for s in world.states}
while True:
    new = {s: max(backup(world, V, s, a)
                  for a in world.actions)
           for s in world.states}
    delta = max(abs(new[s] - V[s])
                for s in world.states)
    V = new
    if delta < theta:          # 1e-4
        return V"""

BACKUP_SRC = """def backup(world, V, s, a, gamma=1.0):
    return sum(p * (r + (0.0 if done else gamma * V[s2]))
               for p, s2, r, done in world.transitions(s, a))"""


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def f3(v):
    return minus(f"{v:.3f}")


def f_change(c):
    return f"{c:.3f}" if c >= 0.001 else f"{c:.5f}"


def tick(center, size=0.26, color=GOOD, width=5):
    """A check mark drawn with two strokes."""
    c = np.array(center, dtype=float)
    pts = [c + size * np.array(p) for p in ((-0.5, 0.0, 0), (-0.15, -0.38, 0), (0.55, 0.45, 0))]
    return m.VMobject().set_points_as_corners(pts).set_stroke(color, width).set_fill(opacity=0)


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.RIGHT)


def sit(mob, y):
    """Put a Text's first glyph on baseline y (digits and most first letters sit on the baseline)."""
    return mob.shift((y - mob[0].get_bottom()[1]) * m.UP)


def thumb(k, center, cell=0.5, gap=0.05):
    """A small picture of the grid after k sweeps: the tint only (blue for value, red for moves paid for)."""
    c = np.array(center, dtype=float)
    top_left = c + np.array([-2 * cell, 1.5 * cell, 0])
    squares = m.VGroup()
    for row in range(W.height):
        for col in range(W.width):
            s = W.from_screen(row, col)
            sq = m.Square(cell - gap).move_to(top_left + np.array([(col + 0.5) * cell, -(row + 0.5) * cell, 0]))
            if s in W.walls:
                sq.set_stroke(EDGE, 1.5).set_fill("#2A2238", 1)
            elif s == CHARGER:
                sq.set_stroke(GOLD, 2).set_fill(GOLD_FILL, 1)
            elif s == PUDDLE:
                sq.set_stroke(RED, 2).set_fill(RED_FILL, 1)
            else:
                sq.set_stroke(EDGE, 1.5).set_fill(heat(HISTORY[k][s], LO_VI, 1.0), 1)
            squares.add(sq)
    label = code(f"sweep {k}", 20, MUTED).next_to(squares, m.DOWN, buff=0.16)
    return m.VGroup(squares, label)


def flash(a, b, color=VALUE, width=9):
    return m.ShowPassingFlash(m.Line(a, b).set_stroke(color, width), time_width=0.8)


def ring_of(square, color, width=4):
    return m.SurroundingRectangle(square, buff=0.0, corner_radius=0.04).set_stroke(color, width)


def loop_arc(a, b, color=MUTED):
    arc = m.ArcBetweenPoints(a, b, angle=-1.1).set_stroke(color, 3)
    arc.add_tip(tip_length=0.16)
    arc.set_color(color)
    return arc


# ---------------------------------------------------------------------------- the episode


class Ep07ValueIteration(Episode):
    number = 7

    def lesson(self):
        self.puzzle()
        self.first_sweep()
        self.spreading()
        self.settling()
        self.arrows_first()
        self.detour()
        self.policy_iter()
        self.recap()
        self.catch()

    # ------------------------------------------------------------------ helpers on the shared grid

    def values_to(self, values, lo=LO_VI, run_time=0.8):
        return self.grid.world_values(values, fmt="{:.3f}", lo=lo, hi=1.0, size=20, offset=self.num_down,
                                      run_time=run_time)

    def counter_to(self, k):
        num = title_text(str(k), 40, INK).next_to(self.count_word, m.RIGHT, buff=0.2)
        sit(num, self.count_base)
        val = right_at(code(f_change(CHANGE[k]), 22, VALUE, weight=m.BOLD), GR, READOUT_Y)
        sit(val, self.count_base)
        return m.AnimationGroup(m.Transform(self.count_num, num), m.Transform(self.change_val, val))

    def sweep_to(self, k, run_time=0.8, arrows=False, timeline=False):
        parts = [self.values_to(HISTORY[k], run_time=run_time), self.counter_to(k)]
        if arrows:
            parts += [r for s, d in POLICIES[k].items() if (r := self.turn(s, d)) is not None]
        if timeline:
            flipped = k in FLIP_SWEEPS
            parts.append(self.boxes[k - 1].animate.set_fill(CORAL_FILL if flipped else BLUE_FILL, 1)
                         .set_stroke(ACTION if flipped else VALUE, 2))
        return m.AnimationGroup(*parts, run_time=run_time)

    def make_arrows(self, policy):
        self.arrow_of, self.arrow_dir = {}, {}
        for s, d in policy.items():
            arrow_ = self.grid.world_arrows({s: d}, length=ARROW_LEN, offset=ARROW_UP, width=5)[0]
            self.arrow_of[s], self.arrow_dir[s] = arrow_, d
        return m.VGroup(*self.arrow_of.values())

    def turn(self, s, d):
        """Rotate cell s's arrow to direction d (None if it already points there)."""
        old = self.arrow_dir[s]
        if old == d:
            return None
        angle = (ANGLE[d] - ANGLE[old] + math.pi) % (2 * math.pi) - math.pi
        if abs(angle + math.pi) < 1e-9:
            angle = math.pi
        self.arrow_dir[s] = d
        centre = self.grid.at(s) + np.array([0, ARROW_UP * CELL, 0])
        return m.Rotate(self.arrow_of[s], angle=angle, about_point=centre)

    def fade(self, *mobs, run_time=0.6):
        """A silent beat between sections that share the grid: fade out exactly the objects faded in."""
        mobs = [mb for mb in mobs if mb is not None]
        if mobs:
            self.play(*[m.FadeOut(mb) for mb in mobs], run_time=run_time)

    # ------------------------------------------------------------------ 1 · start from a guess

    def puzzle(self):
        self.next_section("1 · Start from a guess", notes=(
            "Episode 6 ended here: nine values in Rover's world (ROVER_WORLD, Russell & Norvig 4x3), each defined by its "
            "neighbours through the Bellman optimality equation. Value iteration: start from V = 0 and apply the backup "
            "V_(k+1)(s) = max_a Σ p(s′|s,a)[r + γ V_k(s′)] (γ = 1 here) to every cell at once (one synchronous sweep; "
            "S&B 2018 eq. 4.10, §4.4: 'value iteration is obtained simply by turning the Bellman optimality equation into "
            "an update rule'), and repeat until no value moves by more than 1e-4. Code: an excerpt of "
            "rl_reference.value_iteration (terminal cells are set to 0 there; their backup is 0 anyway)."))
        g = self.grid = GridView.from_world(W, cell=CELL, center=GRID_C)
        self.num_down = 0.0
        qs = m.VGroup(*[code("?", 30, MUTED, weight=m.BOLD).move_to(g.at(s)) for s in OPEN])
        pairs = [(a, b) for a in OPEN for b in OPEN if a < b and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
        web = m.VGroup()
        for a, b in pairs:
            pa, pb = g.at(a), g.at(b)
            d = (pb - pa) / np.linalg.norm(pb - pa)
            mid = (pa + pb) / 2
            web.add(m.DoubleArrow(mid - d * 0.3, mid + d * 0.3, buff=0, color=VALUE, stroke_width=3, tip_length=0.12,
                                  max_tip_length_to_length_ratio=0.3))

        formula = m.MathTex(r"V_{k+1}(s)", "=", r"\max_{a}", r"\sum_{s'} p(s' \mid s, a)", r"\big[", "r", "+",
                            r"V_{k}(s')", r"\big]", font_size=34, color=INK)
        for i, col in ((0, VALUE), (2, ACTION), (3, EXPLORE), (5, REWARD), (7, VALUE)):
            formula[i].set_color(col)
        formula.move_to([3.1, 2.1, 0])
        olds = m.VGroup(text("new value", 20, VALUE).next_to(formula[0], m.DOWN, buff=0.16),
                        text("old values", 20, VALUE).next_to(formula[7], m.DOWN, buff=0.16))
        olds[0].align_to(olds[1], m.UP)
        vi = code_block(VI_SRC, size=20, title="value_iteration · rl_reference.py", accent=EDGE)
        vi.next_to(olds, m.DOWN, buff=0.3).set_x(3.1)
        loop_line = glyphs(vi[1], "while True:")
        sb = cite("Sutton & Barto, 2018, §4.4").next_to(vi, m.DOWN, buff=0.12).align_to(vi, m.RIGHT)
        scan = m.Square(CELL - 0.02).set_stroke(STATE, 6).set_fill(opacity=0).move_to(g.at(OPEN[0]))
        scan_moves = m.Succession(m.FadeIn(scan, run_time=0.15),
                                  *[scan.animate(run_time=0.17).move_to(g.at(s)) for s in OPEN[1:]],
                                  m.FadeOut(scan, run_time=0.15))

        self.say(
            "[Episode six ended on a puzzle]: [nine values, each one defined by its neighbours.]",
            m.AnimationGroup(self.new_header("01 · the puzzle", "Start from a guess"), m.FadeIn(g), m.FadeIn(qs)),
            m.LaggedStart(*[m.GrowFromCenter(a) for a in web], lag_ratio=0.1),
        )
        self.say(
            "[Here's the way out]: [start from a guess, zero in every cell.]",
            m.FadeOut(web),
            m.AnimationGroup(m.FadeOut(qs), self.values_to(HISTORY[0])),
        )
        self.say(
            "[Now apply the Bellman backup to every cell], [each one using its neighbours' old values]: [that's one sweep.]",
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in formula], lag_ratio=0.08),
            m.AnimationGroup(m.FadeIn(olds), m.Circumscribe(formula[7], color=VALUE, buff=0.06)),
            scan_moves,
        )
        self.say(
            "[Then sweep again, and again, until nothing changes]: [that's value iteration.]",
            m.AnimationGroup(m.FadeIn(vi, shift=0.1 * m.UP), m.FadeIn(sb)),
            m.Circumscribe(m.VGroup(*loop_line), color=STATE, buff=0.08),
        )
        self.wait(0.3)
        self.fade(formula, olds, vi, sb)

    # ------------------------------------------------------------------ 2 · the first sweep

    def first_sweep(self):
        self.next_section("2 · The first sweep", notes=(
            "Sweep 1 from V = 0: every non-terminal cell's best backup is −0.04 (no neighbour is worth anything yet; the "
            "best moves avoid the puddle), except (3,3): moving east, 0.8 × (+1) + 0.1 × 0 (bump the wall, stay) + 0.1 × 0 "
            f"(slip south to (3,2)) − 0.04 = {H1[NEXT_TO_CHARGER]:.3f} (backup(world, V0, (3,3), 'E'), asserted). The "
            "readout above the grid is the biggest change of any value in the sweep (the stopping test)."))
        g = self.grid
        self.count_word = left_at(text("sweep", 28, MUTED), GL, READOUT_Y)
        self.count_base = self.count_word[0].get_bottom()[1]
        self.count_num = title_text("1", 40, INK).next_to(self.count_word, m.RIGHT, buff=0.2)
        sit(self.count_num, self.count_base)
        self.change_val = sit(right_at(code(f_change(CHANGE[1]), 22, VALUE, weight=m.BOLD), GR, READOUT_Y), self.count_base)
        widest = max(code(f_change(c), 22, VALUE, weight=m.BOLD).width for c in CHANGE.values())
        self.change_label = right_at(text("biggest change", 22, MUTED), GR - widest - 0.22, READOUT_Y)
        sit(self.change_label, self.count_base)
        readout = self.readout = m.VGroup(self.count_word, self.count_num, self.change_label, self.change_val)

        others = {s: H1[s] for s in OPEN if s != NEXT_TO_CHARGER}
        scan = m.Square(CELL - 0.02).set_stroke(STATE, 6).set_fill(opacity=0).move_to(g.at(OPEN[0]))
        scan_moves = m.Succession(m.FadeIn(scan, run_time=0.12),
                                  *[scan.animate(run_time=0.13).move_to(g.at(s)) for s in OPEN[1:]],
                                  m.FadeOut(scan, run_time=0.12))
        c = g.at(NEXT_TO_CHARGER)
        east = m.Arrow(c + [-0.35, 0.3, 0], c + [0.35, 0.3, 0], buff=0, color=ACTION, stroke_width=6,
                       max_tip_length_to_length_ratio=0.35)

        x0, ys = 0.45, [-0.05, -0.5, -0.95, -1.4]
        head = left_at(code("(3,3) moving east, sweep 1", 22, MUTED), x0, 0.5)
        srcs = ["0.8 × (+1)    0.800", "0.1 × 0       0.000", "0.1 × 0       0.000", "every move   −0.040"]
        rules = {r"^0\.[18]": EXPLORE, r"\(\+1\)": REWARD, r"(?<=× )0(?= )": VALUE, r"−0\.040": REWARD, r"every move": MUTED}
        rows = m.VGroup(*[left_at(code(s, 24, INK, t2c=rules), x0, y) for s, y in zip(srcs, ys)])
        why = m.VGroup(*[left_at(text(w, 20, MUTED), x0 + rows[0].width + 0.45, y)
                         for w, y in zip(("plugs in", "bump, stay", "slip south"), ys)])
        line = m.Line([x0, -1.72, 0], [x0 + rows[0].width + 0.1, -1.72, 0]).set_stroke(MUTED, 2)
        total = code(f"{H1[NEXT_TO_CHARGER]:.3f}", 30, VALUE, weight=m.BOLD).move_to([0, -2.15, 0]).align_to(rows[0], m.RIGHT)
        guesses = left_at(cite("every neighbour is still the guess, 0"), x0, 1.75)
        every = left_at(text("so each cell just pays for its move: −0.040", 22, MUTED), x0, 1.3)

        self.say(
            "[Sweep one.] [Every neighbour is still worth zero], [so each cell just pays for its move: minus zero point zero four.]",
            m.AnimationGroup(self.new_header("02 · sweep one", "The first sweep"), m.FadeIn(readout)),
            m.AnimationGroup(scan_moves, m.FadeIn(guesses)),
            m.AnimationGroup(self.values_to(others), m.FadeIn(every, shift=0.1 * m.UP)),
        )
        self.say(
            "[Except beside the charger]: [eight times in ten, the move east plugs Rover in], [so its value jumps to zero point seven six.]",
            m.AnimationGroup(m.GrowArrow(east), m.FadeIn(head)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in rows], lag_ratio=0.25),
                             m.FadeIn(why), flash(g.at(CHARGER), c, REWARD)),
            m.AnimationGroup(m.Create(line), m.FadeIn(total, shift=0.1 * m.UP),
                             self.values_to({NEXT_TO_CHARGER: H1[NEXT_TO_CHARGER]})),
        )
        self.wait(0.3)
        self.fade(east, head, rows, why, line, total, every, guesses)

    # ------------------------------------------------------------------ 3 · value spreads out

    def spreading(self):
        self.next_section("3 · Value spreads out", notes=(
            "Synchronous sweeps (each reads the previous sweep's values). Newly positive cells per sweep, with the "
            "neighbour their best move heads to (the light-blue flashes): sweep 2 " + ", ".join(f"{s}" for _, s in news(2)) +
            "; sweep 3 " + ", ".join(f"{s}" for _, s in news(3)) + f". After 3 sweeps {STILL_PAYING} still sit at −0.12 "
            "(three moves paid for, no news yet). The start cell (1,1) is first positive after sweep "
            f"{FIRST_POSITIVE} ({HISTORY[5][START]:.3f}); notes §7 worked example 1. Biggest change at sweep 10: "
            f"{CHANGE[10]:.4f}. Filmstrip: the values after sweeps 1, 2, 3, 5, 10 (and 22 in the next section), the same "
            "grids as classic.json planning.value_iteration.after_sweep (asserted). Tint: blue up to +1, red down to −0.25."))
        g = self.grid
        xs, ys = [0.75, 3.15, 5.55], [1.15, -1.3]
        thumbs = [thumb(k, (xs[i % 3], ys[i // 3], 0)) for i, k in enumerate(FILM)]
        sw_blue = m.Square(0.26).set_stroke(EDGE, 1.5).set_fill(heat(0.8, LO_VI, 1.0), 1)
        sw_red = m.Square(0.26).set_stroke(EDGE, 1.5).set_fill(heat(-0.12, LO_VI, 1.0), 1)
        legend = m.VGroup(m.VGroup(sw_blue, text("value from the charger", 20, MUTED)).arrange(m.RIGHT, buff=0.15),
                          m.VGroup(sw_red, text("moves paid for, no news yet", 20, MUTED)).arrange(m.RIGHT, buff=0.15))
        legend.arrange(m.RIGHT, buff=0.4)
        right_at(legend, 6.55, -3.2)

        def flashes(k):
            return m.AnimationGroup(*[flash(g.at(a), g.at(b)) for a, b in news(k)])

        rings = m.VGroup(*[ring_of(g.squares[W.to_screen(s)], RED) for s in STILL_PAYING])
        start_ring = ring_of(g.squares[W.to_screen(START)], GOOD)

        self.say(
            "[Sweep two]: [the news spreads one cell further, left along the top, and down beside the puddle.]",
            m.AnimationGroup(self.new_header("03 · spreading", "Value spreads out"), m.FadeIn(thumbs[0], shift=0.1 * m.UP)),
            m.AnimationGroup(self.sweep_to(2), flashes(2), m.FadeIn(thumbs[1], shift=0.1 * m.UP)),
        )
        self.say(
            "[Sweep three: it reaches the top-left corner, and turns into the bottom row.]",
            m.AnimationGroup(self.sweep_to(3), flashes(3), m.FadeIn(thumbs[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[Cells it hasn't reached yet just keep paying for moves]: [after three sweeps, minus zero point one two.]",
            m.LaggedStart(*[m.Create(r) for r in rings], lag_ratio=0.2),
            m.FadeIn(legend, shift=0.1 * m.UP),
        )
        self.say(
            "[By sweep five, even the start cell is worth something]: [the charger's news took five sweeps to get there.]",
            m.Succession(m.AnimationGroup(m.FadeOut(rings), self.sweep_to(4, run_time=0.7), flashes(4)),
                         m.AnimationGroup(self.sweep_to(5, run_time=0.7), flashes(5), m.FadeIn(thumbs[3], shift=0.1 * m.UP))),
            m.Create(start_ring),
        )
        self.say(
            "[By sweep ten, the picture has nearly settled]: [the biggest change is down to about four hundredths.]",
            m.Succession(m.FadeOut(start_ring, run_time=0.3), *[self.sweep_to(k, run_time=0.45) for k in range(6, 11)],
                         m.FadeIn(thumbs[4], shift=0.1 * m.UP, run_time=0.4)),
            m.Circumscribe(self.change_val, color=VALUE, buff=0.08),
        )
        self.strip = (thumbs, legend)
        self.wait(0.3)

    # ------------------------------------------------------------------ 4 · until nothing changes

    def settling(self):
        self.next_section("4 · Until nothing changes", notes=(
            f"Biggest change per sweep (log scale): {CHANGE[1]:.2f} after sweep 1 … {CHANGE[SWEEPS - 1]:.6f} after sweep "
            f"{SWEEPS - 1}, {CHANGE[SWEEPS]:.6f} after sweep {SWEEPS} < 1e-4: stop (classic.md §4: 22 sweeps, 0.4 ms). Near "
            "the end it roughly halves each sweep (ratios 0.45–0.6 from sweep 18 on, asserted). Contraction: for γ < 1, "
            "‖TV − TU‖∞ ≤ γ‖V − U‖∞ (Denardo 1967), so any two guesses approach each other and the fixed point is unique. "
            "The number-line dots are a schematic (factor ½). Rover's world has γ = 1: it still converges because the best "
            "routes end and wandering forever costs −0.04 a move without end (stochastic shortest path; Bertsekas & "
            "Tsitsiklis 1996; notes §6). Final values vs Russell & Norvig 3rd ed. (2010) Fig. 17.3: every cell equal to "
            "three decimals (asserted against RN_UTILITIES)."))
        g = self.grid
        thumbs, legend = self.strip
        strip = m.VGroup(*thumbs, legend)

        # the plot of the biggest change per sweep, log scale (labels in powers of ten)
        def make_plot(center):
            return Plot((0, 22), (-4.5, 0.0), width=4.7, height=2.2, center=center, x_ticks=[1, 5, 10, 15, 22],
                        x_label="sweep", y_label="biggest change in a sweep (log scale)", size=20)

        probe = make_plot((0, 0, 0))
        y_labels = ["1", "0.1", "0.01", "0.001", "0.0001"]
        widest = max(code(s, 20).width for s in y_labels)
        left_extent = probe.origin[0] - 0.25 - widest
        plot = make_plot((0.05 - left_extent, 2.45 - probe.get_top()[1], 0))
        ylabs = m.VGroup()
        for i, s in enumerate(y_labels):
            p = plot.coords(0, -i)
            ylabs.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2), code(s, 20, MUTED).next_to(p, m.LEFT, buff=0.16))
        ks = list(range(1, SWEEPS + 1))
        curve = plot.line(ks, [math.log10(CHANGE[k]) for k in ks], VALUE, 4)
        end_dot = plot.dot(SWEEPS, math.log10(CHANGE[SWEEPS]), VALUE, 0.08)
        stop_line = plot.hline(-4, GOOD, 2.5)
        stop_lab = text("stop below 0.0001", 20, GOOD).next_to(plot.coords(1, -4), m.UP, buff=0.12)
        stop_lab.align_to(plot.coords(1.5, 0), m.LEFT)
        chart = m.VGroup(plot, ylabs)

        law = m.MathTex(r"\lVert TV - TU \rVert", r"\le", r"\gamma", r"\lVert V - U \rVert", font_size=40, color=INK)
        law[2].set_color(VALUE)
        law.move_to([3.2, -1.95, 0])
        law_cite = cite("a contraction · Denardo, 1967").next_to(law, m.DOWN, buff=0.12)
        # schematic: two guesses on a line, the gap halving each step (illustrative)
        nl_y, target = -2.95, 3.4
        number_line = m.Line([1.0, nl_y, 0], [5.8, nl_y, 0]).set_stroke(FAINT, 3)
        star = m.Line([target, nl_y - 0.16, 0], [target, nl_y + 0.16, 0]).set_stroke(GOOD, 4)
        dots = [m.Dot([x, nl_y, 0], radius=0.1, color=c).set_z_index(3) for x, c in ((1.2, VALUE), (5.6, EXPLORE))]
        steps, pos = [], [1.2, 5.6]
        for _ in range(4):
            pos = [target + 0.5 * (p - target) for p in pos]
            steps.append(m.AnimationGroup(*[d.animate(run_time=0.45).move_to([p, nl_y, 0]) for d, p in zip(dots, pos)]))
        schematic = m.VGroup(number_line, star, *dots)
        schem_lab = cite("two guesses (schematic)").next_to(number_line, m.DOWN, buff=0.26)
        gamma_note = m.VGroup(text("γ = 1 here: it settles because the best routes end", 22, INK),
                              cite("stochastic shortest path · Bertsekas & Tsitsiklis, 1996"))
        gamma_note.arrange(m.DOWN, buff=0.12).move_to([3.2, -3.0, 0])

        # Russell & Norvig's table
        rn_box = card(5.6, 3.1, EDGE, CARD).move_to([3.2, 0.35, 0])
        rn_title = cite("Russell & Norvig, 2010, Fig. 17.3").move_to(rn_box.get_top() + [0, -0.38, 0])
        cols = [1.25, 2.55, 3.85, 5.15]
        rn_cells = m.VGroup()
        for row in range(W.height):
            for col in range(W.width):
                s = W.from_screen(row, col)
                y = 0.75 - 0.72 * row
                if s in W.walls:
                    cell_t = text("rock", 20, FAINT)
                elif s in W.terminals:
                    r = W.terminals[s]
                    cell_t = code(minus(f"{r:+g}"), 24, GOLD if r > 0 else RED, weight=m.BOLD)
                else:
                    cell_t = code(f"{R[s]:.3f}", 24, INK)
                rn_cells.add(cell_t.move_to([cols[col], y, 0]))
        rn_sub = text(f"after {SWEEPS} sweeps: every cell, to three decimals", 22, GOOD).move_to([3.2, -1.75, 0])
        ticks = m.VGroup(*[tick(g.at(s) + [0.45, 0.45, 0], 0.18, width=4) for s in OPEN])

        self.say(
            "[Keep sweeping, and the changes keep shrinking], [until at sweep twenty-two, no value moves by even one ten-thousandth.]",
            m.AnimationGroup(self.new_header("04 · convergence", "Until nothing changes"),
                             m.Succession(*[self.sweep_to(k, run_time=0.3) for k in range(11, SWEEPS + 1)])),
            m.AnimationGroup(m.Circumscribe(self.change_val, color=GOOD, buff=0.08), m.FadeIn(thumbs[5], shift=0.1 * m.UP)),
        )
        self.say(
            "[Plot the biggest change in each sweep]: [near the end, it roughly halves every sweep], [and we stop below this line.]",
            m.AnimationGroup(m.FadeOut(strip), m.FadeIn(chart)),
            m.AnimationGroup(m.Create(curve, run_time=1.8), m.FadeIn(end_dot)),
            m.AnimationGroup(m.Create(stop_line), m.FadeIn(stop_lab)),
        )
        self.say(
            "[Why must it settle?] [With a discount below one, the backup is a contraction]: "
            "[each sweep shrinks the gap between any two guesses], [so they all end at the same answer.]",
            m.Wait(0.3),
            m.AnimationGroup(m.FadeIn(law, shift=0.1 * m.UP), m.FadeIn(law_cite)),
            m.Succession(m.AnimationGroup(m.FadeIn(schematic), m.FadeIn(schem_lab), run_time=0.5), *steps[:2]),
            m.Succession(*steps[2:]),
        )
        self.say(
            "[Rover's world has no discount], [but its best routes always end, and that's enough.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(schematic, schem_lab)), m.Circumscribe(law[2], color=VALUE, buff=0.06)),
            m.FadeIn(gamma_note, shift=0.1 * m.UP),
        )
        self.say(
            "[And where it settles is the table in Russell and Norvig's textbook]: [every cell matches, to three decimals.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(chart, curve, end_dot, stop_line, stop_lab, law, law_cite, gamma_note)),
                             m.FadeIn(rn_box), m.FadeIn(rn_title), m.FadeIn(rn_cells)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(t) for t in ticks], lag_ratio=0.1), m.FadeIn(rn_sub, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.fade(rn_box, rn_title, rn_cells, rn_sub, ticks)

    # ------------------------------------------------------------------ 5 · arrows settle first

    def arrows_first(self):
        self.next_section("5 · Arrows settle first", notes=(
            "Replay of the same sweeps; after each, greedy_policy(world, V_k) (ties: N, E, S, W order). Arrows turn at "
            f"sweeps {FLIP_SWEEPS}: sweep 2 (1,3) N→E; sweep 3 (2,1) N→E and (4,1) S→W; sweep 8 (2,1) E→W; sweep 12 (3,1) "
            f"N→W, and from then on the policy equals R&N's Fig. 17.2(a) (classic.md §4: optimal from sweep 12). At sweep "
            f"12 the values are still off by up to {ERROR[12]:.3f} from the converged ones (notes §7); the numbers settle "
            "at sweep 22. Timeline boxes: CORAL = an arrow turned in that sweep, BLUE = only the values moved."))
        g = self.grid
        self.num_down = NUM_DOWN
        arrows = self.make_arrows(POLICIES[1])
        w_box, gap, tx0, ty = 0.26, 0.04, 0.0, 1.4
        self.boxes = m.VGroup(*[m.Square(w_box).set_stroke(EDGE, 2).set_fill(CARD, 1)
                                .move_to([tx0 + w_box / 2 + i * (w_box + gap), ty, 0]) for i in range(SWEEPS)])
        tl_title = left_at(text("sweep by sweep", 22, MUTED), tx0, ty + 0.48)
        key = m.VGroup(m.Square(0.22).set_stroke(ACTION, 2).set_fill(CORAL_FILL, 1),
                       text("an arrow turned", 20, MUTED)).arrange(m.RIGHT, buff=0.14)
        right_at(key, self.boxes[-1].get_right()[0], ty + 0.48)
        sit(key[1], tl_title[0].get_bottom()[1])
        key[0].set_y(key[1].get_y())
        tl_nums = m.VGroup(*[code(str(k), 20, MUTED).next_to(self.boxes[k - 1], m.DOWN, buff=0.14)
                             for k in (1, OPTIMAL_FROM, SWEEPS)])
        mark22 = text("values settle", 22, VALUE, weight=m.BOLD).next_to(tl_nums[2], m.DOWN, buff=0.16)
        mark22.align_to(self.boxes[-1], m.RIGHT)
        mark12 = text("arrows final", 22, ACTION, weight=m.BOLD).next_to(tl_nums[1], m.DOWN, buff=0.16)
        sit(mark12, mark22[0].get_bottom()[1])
        log_title = left_at(text("arrows that turned", 20, MUTED), tx0, 0.0)
        rules = {r"^sweep\s+\d+": MUTED, r"\(\d,\d\)": STATE, r"[NESW]→[NESW]": ACTION}
        self.log_rows = {}
        for i, k in enumerate(FLIP_SWEEPS):
            src = f"sweep {k:<2}  " + ", ".join(f"({c},{r}) {POLICIES[k - 1][(c, r)]}→{POLICIES[k][(c, r)]}"
                                                 for c, r in FLIPS[k])
            self.log_rows[k] = left_at(code(src, 22, INK, t2c=rules), tx0, -0.42 - 0.42 * i)
        ring = ring_of(g.squares[W.to_screen(DETOUR)], ACTION)
        off = m.VGroup(text(f"at sweep {OPTIMAL_FROM}, values still off by up to", 22, MUTED),
                       code(f"{ERROR[OPTIMAL_FROM]:.3f}", 24, VALUE, weight=m.BOLD)).arrange(m.RIGHT, buff=0.18)
        sit(off[1], off[0][0].get_bottom()[1])
        left_at(off, tx0, -2.3)
        best = left_at(text("the arrows: already the best policy", 24, GOOD, weight=m.BOLD), tx0, -2.85)

        def step(k, run_time):
            parts = [self.sweep_to(k, run_time=run_time, arrows=True, timeline=True)]
            if k in self.log_rows:
                parts.append(m.FadeIn(self.log_rows[k], shift=0.1 * m.UP, run_time=run_time))
            return m.AnimationGroup(*parts, run_time=run_time)

        self.say(
            "[Now replay the sweeps from the start], [and after each one, point every cell at its best move.]",
            m.AnimationGroup(self.new_header("05 · the policy", "Arrows settle first"),
                             m.FadeIn(tl_title), m.FadeIn(key), m.FadeIn(self.boxes), m.FadeIn(tl_nums[0]),
                             self.sweep_to(1, timeline=True)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.1), m.FadeIn(log_title)),
        )
        self.say(
            "[A few arrows turn as the news arrives]: [at sweeps two, three and eight.]",
            m.Succession(*[step(k, 0.5) for k in range(2, 9)]),
            m.Circumscribe(m.VGroup(*[self.log_rows[k] for k in FLIP_SWEEPS[:3]]), color=ACTION, buff=0.1),
        )
        self.say(
            "[The last one turns at sweep twelve]: [in the bottom row, from north to west.]",
            m.Succession(*[step(k, 0.5) for k in range(9, OPTIMAL_FROM + 1)]),
            m.AnimationGroup(m.Create(ring), m.FadeIn(tl_nums[1]), m.FadeIn(mark12, shift=0.1 * m.UP)),
        )
        self.say(
            "[After that, no arrow ever moves again], [though the numbers keep changing for ten more sweeps.]",
            m.FadeOut(ring),
            m.AnimationGroup(m.Succession(*[step(k, 0.32) for k in range(OPTIMAL_FROM + 1, SWEEPS + 1)]),
                             m.FadeIn(tl_nums[2]), m.FadeIn(mark22, shift=0.1 * m.UP)),
        )
        self.say(
            "[At sweep twelve, the values were still off by up to zero point zero two six], [but the arrows were already the best policy.]",
            m.FadeIn(off, shift=0.1 * m.UP),
            m.FadeIn(best, shift=0.1 * m.UP),
        )
        self.policy_arrows = arrows
        self.wait(0.3)
        self.fade(tl_title, key, self.boxes, tl_nums, mark12, mark22, log_title, *self.log_rows.values(), off, best,
                  self.readout)

    # ------------------------------------------------------------------ 6 · the long way round

    def detour(self):
        self.next_section("6 · The long way round", notes=(
            "Russell & Norvig 3rd ed. (2010) Fig. 17.2(a): from (3,1) and (4,1) the optimal policy goes West, the long way "
            f"round ({len(LONG) - 1} moves from (3,1) if no wheel slips) instead of North past the puddle ({len(SHORT) - 1} "
            "moves). Landing values with the textbook's three-decimal utilities (notes §5, §6; Gao 2021 lecture notes give "
            f"the same): North 0.8 × 0.660 + 0.1 × 0.655 + 0.1 × 0.388 = {LAND_N:.4f}; West 0.8 × 0.655 + 0.1 × 0.660 + "
            f"0.1 × 0.611 = {LAND_W:.4f}; both pay −0.04, so V(3,1) = 0.651 − 0.04 = 0.611. North's east slip lands under "
            "the puddle (0.388), and from (3,2) a north move slips into the puddle one time in ten. The extra moves cost "
            f"{EXTRA} × 0.04 = {EXTRA * 0.04:.2f}; the puddle instead of the charger is a swing of 2. The 4th edition "
            "reportedly shows a tie at (3,1); the 3rd-ed. model gives none (notes Appendix A)."))
        g = self.grid
        long_off = np.array([0.5, -0.17, 0])
        long_path = m.VMobject().set_points_as_corners([g.at(s) + long_off for s in LONG[:-1]] +
                                                       [g.at(CHARGER) + long_off + np.array([-0.95, 0, 0])])
        long_path.set_stroke(ACTION, 6)
        bot = rover(0.36).move_to(g.at(DETOUR) + long_off).set_z_index(4)
        drive = m.Succession(m.FadeIn(bot, scale=0.8, run_time=0.3),
                             *[bot.animate(run_time=0.36).move_to(g.at(s) + long_off) for s in LONG[1:-1]],
                             m.FadeOut(bot, shift=0.3 * m.RIGHT, run_time=0.35))
        p0 = g.at(DETOUR) + np.array([-0.5, -0.17, 0])
        p1 = g.at(NEXT_TO_CHARGER) + np.array([-0.5, 0.47, 0])
        p2 = g.at(CHARGER) + np.array([-0.62, 0.47, 0])
        short = m.DashedVMobject(m.VMobject().set_points_as_corners([p0, p1, p2]), num_dashes=22).set_stroke(INK, 4)
        c32 = g.at((3, 2))
        slip = m.Arrow(c32 + [0.15, 0.05, 0], g.at(PUDDLE) + [-0.32, 0.05, 0], buff=0, color=RED, stroke_width=5,
                       max_tip_length_to_length_ratio=0.3)
        slip_lab = code("0.1", 20, RED, weight=m.BOLD).move_to(c32 + [0.42, 0.33, 0])
        rings = [ring_of(g.squares[W.to_screen(s)], ACTION) for s in (DETOUR, UNDER_PUDDLE)]
        puddle_ring = ring_of(g.squares[W.to_screen(PUDDLE)], RED)

        xs = (1.75, 4.95)
        heads = m.VGroup(text("north", 26, ACTION, weight=m.BOLD).move_to([xs[0], 1.75, 0]),
                         text("west", 26, ACTION, weight=m.BOLD).move_to([xs[1], 1.75, 0]))
        on_baseline(heads)
        rules = {r"^0\.[18]": EXPLORE, r"(?<=× )0\.\d{3}": VALUE}
        cols = []
        for x, a in zip(xs, ("N", "W")):
            rows = m.VGroup(*[code(f"{p:.1f} × {R[s2]:.3f}", 24, INK, t2c=rules).move_to([x, 1.15 - 0.5 * i, 0])
                              for i, (p, s2) in enumerate(TERMS[a])])
            line = m.Line([x - 1.05, -0.3, 0], [x + 1.05, -0.3, 0]).set_stroke(MUTED, 2)
            tot = code(f"{(LAND_N if a == 'N' else LAND_W):.3f}", 32, VALUE, weight=m.BOLD).move_to([x, -0.72, 0])
            cols.append(m.VGroup(rows, line, tot))
        toll = cite("each then pays −0.04 for the move").move_to([3.35, -1.3, 0])
        cost_rows = [left_at(text(f"{EXTRA} extra moves:  −{-EXTRA * W.step_reward:.2f}", 22, REWARD), 0.6, -1.95),
                     left_at(text("puddle instead of charger:  a swing of 2", 22, RED), 0.6, -2.42)]
        wins = left_at(text(f"west wins, by {LAND_W - LAND_N:.3f}", 26, GOOD, weight=m.BOLD), 0.6, -3.05)
        rn = right_at(cite("Russell & Norvig, 2010, Fig. 17.2"), 6.6, 2.4)

        self.say(
            "[Look at those last arrows]: [from the bottom row, Rover takes the long way round], [away from the puddle.]",
            m.AnimationGroup(self.new_header("06 · the detour", "The long way round"), m.Create(rings[0]), m.Create(rings[1])),
            m.AnimationGroup(m.FadeOut(rings[0]), m.FadeOut(rings[1]), m.Create(long_path, run_time=2.4), drive),
            m.Create(puddle_ring),
        )
        self.say(
            "[The short way, north, is only three moves], [but it passes the puddle, where one slip in ten tips Rover in.]",
            m.AnimationGroup(m.FadeOut(puddle_ring), m.Create(short, run_time=1.2)),
            m.AnimationGroup(m.GrowArrow(slip), m.FadeIn(slip_lab)),
        )
        self.say(
            "[Weigh where each move lands]: [north is worth zero point six three two on average], [and west, zero point six five one.]",
            m.FadeIn(heads, shift=0.1 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in cols[0][0]], lag_ratio=0.2),
                             m.Create(cols[0][1]), m.FadeIn(cols[0][2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in cols[1][0]], lag_ratio=0.2),
                             m.Create(cols[1][1]), m.FadeIn(cols[1][2], shift=0.1 * m.UP), m.FadeIn(toll)),
        )
        self.say(
            "[Each extra move costs only zero point zero four], [but a slip into the puddle swings the result by two whole points]: "
            "[with ten percent slips, west wins.]",
            m.FadeIn(cost_rows[0], shift=0.1 * m.UP),
            m.FadeIn(cost_rows[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(wins, shift=0.1 * m.UP), m.FadeIn(rn), m.Circumscribe(cols[1][2], color=GOOD, buff=0.1)),
        )
        self.wait(0.3)
        self.fade(long_path, short, slip, slip_lab, heads, *cols, toll, *cost_rows, wins, rn)

    # ------------------------------------------------------------------ 7 · policy iteration

    def policy_iter(self):
        self.next_section("7 · Policy iteration", notes=(
            "policy_iteration(ROVER_WORLD) (Howard 1960; S&B 2018 §4.3) from 'always north' (a policy that does end, so its "
            "value is finite with γ = 1): evaluate (policy_evaluation, θ = 1e-8; the start cell is worth "
            f"{START_PI[0]:.3f}), improve greedily (change a move only if strictly better), repeat. Arrows turned per "
            f"improvement: {TURNED[:4]}; {len(PI_POLICIES)} evaluations, the last changes nothing; the result equals value "
            "iteration's policy and R&N's Fig. 17.2(a) (classic.md §4: 5 evaluations, 4 improvements). Start-cell value "
            f"after each evaluation: {', '.join(f'{v:.3f}' for v in START_PI)}. Quote: 'Policy iteration often converges in "
            "surprisingly few iterations' (S&B 2018, §4.3, p. 82). Tint here: red down to −1.5."))
        g = self.grid
        blank = m.AnimationGroup(
            *[m.Transform(g.numbers[W.to_screen(s)], code("?", 24, MUTED, weight=m.BOLD).move_to(g.numbers[W.to_screen(s)]))
              for s in OPEN],
            *[g.squares[W.to_screen(s)].animate.set_fill(CARD, 1) for s in OPEN])
        north = m.AnimationGroup(*[r for s in OPEN if (r := self.turn(s, "N")) is not None])

        ev = chip("evaluate", VALUE, BLUE_FILL, width=2.0, height=0.66, size=24).move_to([1.2, 1.95, 0])
        im = chip("improve", ACTION, CORAL_FILL, width=2.0, height=0.66, size=24).move_to([5.1, 1.95, 0])
        top = loop_arc(ev.get_right() + [0.1, 0.12, 0], im.get_left() + [-0.1, 0.12, 0])
        bottom = loop_arc(im.get_left() + [-0.1, -0.12, 0], ev.get_right() + [0.1, -0.12, 0])
        loop = m.VGroup(ev, im, top, bottom)
        cx = (0.75, 3.05, 5.15)
        hdr = m.VGroup(text("evaluation", 20, MUTED).move_to([cx[0], 0.95, 0]),
                       text("start cell", 20, MUTED).move_to([cx[1], 0.95, 0]),
                       text("arrows turned", 20, MUTED).move_to([cx[2], 0.95, 0]))
        on_baseline(hdr)
        rule = m.Line([0.0, 0.7, 0], [6.5, 0.7, 0]).set_stroke(EDGE, 2)
        rows = []
        for i, (v0, t) in enumerate(zip(START_PI, TURNED)):
            y = 0.3 - 0.5 * i
            rows.append(m.VGroup(code(str(i + 1), 24, MUTED).move_to([cx[0], y, 0]),
                                 code(f3(v0), 24, VALUE, weight=m.BOLD).move_to([cx[1], y, 0]),
                                 code(str(t) if t else "0 · stop", 24, ACTION if t else GOOD, weight=m.BOLD)
                                 .move_to([cx[2], y, 0])))
        done = tick(rows[-1][2].get_right() + [0.35, 0.02, 0], 0.26)
        same = left_at(text("= value iteration's policy", 22, GOOD), 0.4, -2.25)
        cost = left_at(text("a full solve per round, but few rounds", 22, INK), 0.4, -2.75)
        quote = cite("“Policy iteration often converges in surprisingly few iterations” · Sutton & Barto, 2018, §4.3")
        quote.move_to([0, -3.35, 0])
        start_ring = ring_of(g.squares[W.to_screen(START)], RED)

        def evaluate(i):
            return self.values_to(PI_VALUES[i], lo=LO_PI, run_time=0.7)

        def improve(i):
            turns = [r for s, d in PI_POLICIES[i].items() if (r := self.turn(s, d)) is not None]
            return m.AnimationGroup(*turns, run_time=0.7)

        def round_(i):
            return m.Succession(m.AnimationGroup(evaluate(i), m.FadeIn(rows[i], shift=0.1 * m.UP), run_time=0.7),
                                m.AnimationGroup(improve(i + 1), m.Indicate(im, color=ACTION, scale_factor=1.06), run_time=0.7))

        self.say(
            "[There's a second way to solve the world]: [policy iteration.] [Start from a simple rule: always go north.]",
            m.AnimationGroup(self.new_header("07 · policy iteration", "Evaluate, improve, repeat"), blank),
            m.FadeIn(loop, shift=0.1 * m.UP),
            north,
        )
        self.say(
            "[First, evaluate it, as in episode six]: [always heading north, the start cell is worth minus one point four seven.]",
            m.AnimationGroup(m.Indicate(ev, color=VALUE, scale_factor=1.06), m.FadeIn(hdr), m.Create(rule), evaluate(0)),
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.1 * m.UP), m.Create(start_ring)),
        )
        self.say(
            "[Then improve it: point every cell at its best move under those values]: [six arrows turn.]",
            m.AnimationGroup(m.FadeOut(start_ring), m.Indicate(im, color=ACTION, scale_factor=1.06)),
            improve(1),
        )
        self.say(
            "[Evaluate and improve again]: [one more arrow turns], [then another], [then a last one.]",
            m.ShowPassingFlash(top.copy().set_stroke(INK, 5), time_width=0.6),
            round_(1),
            round_(2),
            round_(3),
        )
        self.say(
            "[The fifth evaluation changes nothing]: [these are the same arrows value iteration found.]",
            m.AnimationGroup(evaluate(4), m.FadeIn(rows[4], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(done), m.FadeIn(same, shift=0.1 * m.UP)),
        )
        self.say(
            "[Each round is a full evaluation, so it costs more than a sweep]: [but you need surprisingly few of them.]",
            m.FadeIn(cost, shift=0.1 * m.UP),
            m.FadeIn(quote),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            f"Sweep (apply the backup to every cell), settle (repeat until no value moves by 1e-4: {SWEEPS} sweeps), act "
            f"(the greedy policy, final from sweep {OPTIMAL_FROM}). Values = Russell & Norvig Fig. 17.3 to three decimals."))
        halo = glow(STATE, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Sweep", 70, VALUE), title_text("Settle", 70, GOOD), title_text("Act", 70, ACTION))
        words.arrange(m.RIGHT, buff=1.6).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text("back up every cell", 24, MUTED), text("repeat until nothing changes", 24, MUTED),
                        text("take the best move", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        base = min(s_[0].get_bottom()[1] for s_ in subs)
        for s_ in subs:
            sit(s_, base)
        proof = code(f"{SWEEPS} sweeps · arrows final at {OPTIMAL_FROM} · = Russell & Norvig, Fig. 17.3", 22, INK,
                     t2c={rf"^{SWEEPS} sweeps": VALUE, rf"arrows final at {OPTIMAL_FROM}": ACTION})
        proof.move_to([0, -1.35, 0])
        mascot = rover(0.7).move_to([0, -2.75, 0])

        self.say(
            "[To sum up]: [sweep the backup over every cell], [repeat until it settles], [then act on the best move.]",
            m.AnimationGroup(self.new_header("08 · recap", "Sweep, settle, act"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Twenty-two sweeps, and Rover's world is solved]: [the arrows were right by sweep twelve, and the values match the textbook.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · we needed the map

    def catch(self):
        self.next_section("9 · We needed the map", notes=(
            "Every backup calls world.transitions(s, a): the model p(s′, r | s, a), here the 0.8 / 0.1 / 0.1 slips and the "
            "rewards, which Rover was given, never measured. Dynamic programming plans with a known model; Module 3 "
            "(episodes 8–11: Monte Carlo, TD, Q-learning and SARSA) learns values from experience without it (notes §7). "
            f"world.transitions((3,1), 'N') = {[(p, s2, round(r, 2)) for p, s2, r, _ in MAP_ROWS]}."))
        blk = code_block(BACKUP_SRC, size=20, title="backup · rl_reference.py", accent=EDGE).move_to([0, 1.6, 0])
        model = glyphs(blk[1], "world.transitions(s, a)")
        hl = m.Line(model.get_corner(m.DL) + [0, -0.09, 0], model.get_corner(m.DR) + [0, -0.09, 0]).set_stroke(STATE, 4)
        lines = [f"{p:.1f}  →  ({s2[0]},{s2[1]})   {minus(f'{r:.2f}')}" for p, s2, r, _ in MAP_ROWS]
        rules = {r"^\d\.\d": EXPLORE, r"\(\d,\d\)": STATE, r"−0\.04": REWARD}
        rows = m.VGroup(*[code(s, 24, INK, t2c=rules) for s in lines]).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        cap = code("world.transitions((3,1), \"N\")", 20, STATE)
        inner = m.VGroup(cap, rows).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)
        box = card(inner.width + 0.7, inner.height + 0.6, STATE, CARD)
        inner.move_to(box)
        tcard = m.VGroup(box, inner).move_to([-3.3, -1.45, 0])
        sub = cite("where a move lands, how often, and what it pays").next_to(tcard, m.DOWN, buff=0.15)

        bot = rover(0.85).move_to([2.6, -1.3, 0])
        the_map = icon(MAP, 0.9).next_to(bot, m.RIGHT, buff=0.4)
        gift = cite("handed over, not learned").next_to(m.VGroup(bot, the_map), m.DOWN, buff=0.25)
        cross = m.VGroup(m.Line(the_map.get_corner(m.UL), the_map.get_corner(m.DR)),
                         m.Line(the_map.get_corner(m.DL), the_map.get_corner(m.UR))).set_stroke(RED, 6)
        next_mod = pill("MODULE 3 · LEARNING FROM EXPERIENCE", BLUE, BLUE_FILL, 20, mono=True).move_to([3.0, -3.2, 0])
        final = title_text("We needed the map.", 54, INK).move_to([0, 1.55, 0])

        self.say(
            "[There's a catch.] [Every backup read the world's rules]: [where each move lands, and how often.]",
            m.AnimationGroup(self.new_header("09 · the catch", "We needed the map"), m.FadeIn(blk, shift=0.1 * m.UP)),
            m.Create(hl),
            m.AnimationGroup(m.FadeIn(tcard, target_position=hl), m.FadeIn(sub)),
        )
        self.say(
            "[Rover never learned those slips by trying]: [we handed it the map.]",
            m.FadeIn(bot, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(the_map, target_position=tcard, path_arc=-0.6), m.FadeIn(gift)),
        )
        self.say(
            "[Out in a real world, nobody hands you the map.] [In module three, Rover learns from what actually happens.]",
            m.AnimationGroup(m.Create(cross), m.FadeOut(gift)),
            m.FadeIn(next_mod, shift=0.1 * m.UP),
        )
        self.say(
            "[Value iteration solved Rover's world], [but we needed the map.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(blk, hl)), m.Circumscribe(tcard, color=STATE, buff=0.1)),
            m.Write(final, run_time=1.4),
        )
        self.wait(0.6)
        self.clear()
