"""Trial and Reward · Episode 8 · Learning by Playing (Module 3 · Learning from Experience)

Module 3 opens by taking the map away. Episodes 5–7 had the world's model (world.transitions: every outcome of a move
and its chance), and Episode 7 solved Rover's world with it. Now Rover can only play: model-free learning. Monte Carlo
prediction plays an episode, walks back along its path adding up the rewards (every cell gets the sum of what
followed it), and keeps, in every cell, the average of those returns. The running example is the measured run of
research/experiments/classic.md §5 (classic.json rover_mc): Russell & Norvig's optimal arrows, seed 0, 1,000 episodes,
first-visit averages. Its first episode IS Episode 5's eight-move drive (seed 0, 0.68), its fourth slides into the
puddle (−1.24), and the start cell's estimate reads 0.680, 0.544, 0.688, 0.711 after 1 / 10 / 100 / 1,000 episodes,
settling toward the planned 0.705. Then: two cells are never visited (coverage), control needs Q and exploration
(exploring starts, ε-soft policies), the trade (no model, no bootstrapping, but wait for the end and high variance:
one return's spread 0.25, an average of 100 still 0.025), Blackjack in one line, and the step to Episode 9.

Every number on screen is computed below from rl_reference.py and asserted, or loaded with measured(...), or cited.

Sources (research/notes.md §8, Appendix C row 8; facts dated October 2026):
- Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), ch. 5: first-visit and every-visit MC both
  converge to v_π (§5.1, p. 93); "Monte Carlo methods do not bootstrap" (§5.1, p. 95); Example 5.1 Blackjack (DP needs
  every probability, "often complex and error-prone"; sample games are easy); exploring starts (§5.3); ε-soft and
  ε-greedy policies (§5.4: greedy action 1 − ε + ε/|A|, others ε/|A|).
- Russell & Norvig 2010, Fig. 17.2–17.3 (the 4×3 world, its optimal policy RN_POLICY and utilities, U(1,1) = 0.705).
- research/experiments/classic.md §5 and classic.json rover_mc (measured on this Mac, one seeded run, recomputed here
  with mc_prediction and asserted equal).

Illustrative: the four exploring starts (cells and first moves drawn with default_rng(12)); everything else is
computed from rl_reference.

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

PREFIX = "ep08"
W = ROVER_WORLD
START = W.start
NEAR = (3, 3)                                   # the cell beside the charger
OPEN = [s for s in W.states if not W.is_terminal(s)]

# ---------------------------------------------------------------------------- the map (Episodes 5–7)

V_PLAN, _SWEEPS = value_iteration(W)            # what Episode 7 computes with the map
EXACT = policy_evaluation(W, RN_POLICY, 1.0, 1e-12)
assert greedy_policy(W, V_PLAN) == RN_POLICY
assert all(round(V_PLAN[s], 3) == RN_UTILITIES[s] for s in RN_UTILITIES)               # R&N Fig. 17.3
assert round(EXACT[START], 3) == 0.705 and abs(EXACT[START] - V_PLAN[START]) < 1e-3     # "zero point seven zero five"
MAP = W.transitions(START, "N")
assert [(p, s2) for p, s2, _, _ in MAP] == [(0.8, (1, 2)), (0.1, (1, 1)), (0.1, (2, 1))]
assert all(np.isclose(r, -0.04) and not d for _, _, r, d in MAP)

# ---------------------------------------------------------------------------- the Monte Carlo run (measured: rover_mc)

N_EP = 1000
_rng = np.random.default_rng(0)
EPISODES = [play_episode(W, RN_POLICY, _rng) for _ in range(N_EP)]
V_MC, HIST = mc_prediction(EPISODES, {s: 0.0 for s in W.states}, record=True)        # first-visit averages
MC = measured("classic", "rover_mc")
assert MC["seed"] == 0 and MC["first_visit"] is True
for _n, _vals in MC["V_after"].items():
    for _k, _v in _vals.items():
        assert abs(HIST[int(_n)][tuple(int(x) for x in _k.split(","))] - _v) < 1e-4, (_n, _k)
RETURNS = [sum(t.r for t in ep) for ep in EPISODES]
assert [round(g, 2) for g in RETURNS[:10]] == MC["start_returns_first10"]
assert abs(np.mean(RETURNS) - MC["mean_start_return"]) < 1e-4 and abs(EXACT[START] - MC["exact_start_value"]) < 1e-4
EST = [HIST[n][START] for n in range(N_EP + 1)]                                       # EST[n] after n episodes
assert [round(EST[n], 3) for n in (1, 10, 100, 1000)] == [0.680, 0.544, 0.688, 0.711]  # classic.md §5
assert [f"{EST[n]:.2f}" for n in (1, 2, 3, 4, 10, 100, 1000)] == ["0.68", "0.68", "0.71", "0.22", "0.54", "0.69", "0.71"]
assert min(EST[1:]) == EST[4] and min(EST[1:]) > 0 and max(EST[1:]) < 1


def visited(n):
    return {t.s for ep in EPISODES[:n] for t in ep}


NEVER = [s for s in OPEN if s not in visited(N_EP)]
assert NEVER == [(3, 1), (4, 1)]                                                      # "except two cells"
MAX_GAP = max(abs(HIST[N_EP][s] - EXACT[s]) for s in visited(N_EP))
assert abs(MAX_GAP - MC["max_abs_error_after"]["1000"]) < 1e-4 and round(MAX_GAP, 3) == 0.038 and MAX_GAP < 0.04

# --- episode 1 of the run = Episode 5's eight-move drive (seed 0)
EP1 = EPISODES[0]
assert EP1 == play_episode(W, RN_POLICY, np.random.default_rng(0))
assert len(EP1) == 8 and EP1[-1].s2 == CHARGER and round(RETURNS[0], 2) == 0.68
G_BACK, _g = [], 0.0
for _t in reversed(EP1):                         # walking back from the end: G = r + G (gamma = 1)
    _g = _t.r + _g
    G_BACK.append(_g)
G_BACK.reverse()                                 # G_BACK[k] = the return from step k
assert [round(g, 2) for g in G_BACK] == [0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96]
VISITS = [k for k, t in enumerate(EP1) if t.s == NEAR]
assert VISITS == [4, 5, 7] and [round(G_BACK[k], 2) for k in VISITS] == [0.84, 0.88, 0.96]   # "three times"
EVERY = float(np.mean([G_BACK[k] for k in VISITS]))
assert abs(mc_prediction(EPISODES[:1], {s: 0.0 for s in W.states}, first_visit=False)[NEAR] - EVERY) < 1e-12
assert round(EVERY, 2) == 0.89 and abs(HIST[1][NEAR] - G_BACK[VISITS[0]]) < 1e-12              # first-visit: 0.84
assert all(abs(HIST[1][t.s] - G_BACK[[u.s for u in EP1].index(t.s)]) < 1e-12 for t in EP1)


def slipped(t):
    return t.s2 != W.move(t.s, t.a)


assert sum(slipped(t) for t in EP1) == 2                                             # "slips and all"
# --- episodes 2–4
assert [round(g, 2) for g in RETURNS[:4]] == [0.68, 0.68, 0.76, -1.24] and EPISODES[3][-1].s2 == PUDDLE
assert all(ep[-1].s2 == CHARGER for ep in EPISODES[:3])
assert sorted(visited(3)) == sorted(visited(1)) and (2, 1) not in visited(10) and (2, 1) in visited(100)

# --- one return's spread, exactly (second moment of the return under RN_POLICY, solved like policy evaluation)
_M = {s: 0.0 for s in W.states}
for _ in range(3000):
    _M = {s: 0.0 if W.is_terminal(s) else sum(p * (r * r + (0.0 if d else 2 * r * EXACT[s2] + _M[s2]))
                                              for p, s2, r, d in W.transitions(s, RN_POLICY[s])) for s in W.states}
SD_ONE = math.sqrt(_M[START] - EXACT[START] ** 2)
SD_100 = SD_ONE / math.sqrt(100)
assert round(SD_ONE, 2) == 0.25 and round(SD_100, 3) == 0.025          # notes §8 / Appendix C: 0.251 → 0.025 (1/√n)

# --- the same cell, two of Episode 6's drives
LUCKY = play_episode(W, RN_POLICY, np.random.default_rng(11))
UNLUCKY = play_episode(W, RN_POLICY, np.random.default_rng(9))
assert len(LUCKY) == 5 and round(sum(t.r for t in LUCKY), 2) == 0.80 and LUCKY[-1].s2 == CHARGER
assert len(UNLUCKY) == 8 and round(sum(t.r for t in UNLUCKY), 2) == -1.32 and UNLUCKY[-1].s2 == PUDDLE

# --- control: Q at the start, exploring starts (illustrative draws), epsilon-soft
assert all(t.a == "N" for ep in EPISODES for t in ep if t.s == START)    # every visit to the start moved north ...
Q_START_N = HIST[N_EP][START]                                            # ... so Q(start, N) is the start's average
assert f"{Q_START_N:.2f}" == "0.71"
ES_SEED = 12
_es = np.random.default_rng(ES_SEED)
ES = list(zip([OPEN[i] for i in _es.choice(len(OPEN), 4, replace=False)], [ACTIONS[j] for j in _es.integers(4, size=4)]))
assert ES == [((2, 3), "N"), ((3, 1), "S"), ((4, 1), "E"), ((1, 2), "E")] and set(NEVER) <= {s for s, _ in ES}
EPS = 0.1
P_BEST, P_OTHER = 1 - EPS + EPS / len(ACTIONS), EPS / len(ACTIONS)
assert (round(P_BEST, 3), round(P_OTHER, 3)) == (0.925, 0.025)            # "ninety-two and a half", "two and a half"

# ---------------------------------------------------------------------------- look

CLOCK = recolored("clock", RED, RED_FILL, prefix=PREFIX)
MAP_ICON, CARDS_ICON = "ep08-map", "ep08-cards"

CELL = 1.25
GRID_C = (-3.85, -0.45, 0.0)
LIFT = np.array([0.0, 0.14, 0.0])
NUM_OFF = np.array([0.0, -0.38, 0.0])
RX = 3.0                                          # the right panel's centre
XL = -0.5                                         # the right panel's left edge
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def vec(direction):
    dc, dr = MOVES[direction]
    return np.array([dc, dr, 0.0])


def baseline(words, s):
    chars = [c for c in s if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    """Put the baseline of a Text at height y."""
    return words.shift((y - baseline(words, words.text)) * m.UP)


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def make_grid():
    return GridView.from_world(W, cell=CELL, center=GRID_C)


def spot(grid, s):
    return grid.at(s) + LIFT


def bump_direction(t):
    """Which way Rover bumped when a move left it in place (a direction among aim and slips that stays; when two
    would, as at (1,2) between the edge and the rock, the episode doesn't say which: take the first)."""
    stays = [d for d in (t.a, *PERPENDICULAR[t.a]) if W.move(t.s, d) == t.s]
    assert stays
    return stays[0]


def bump_mark(grid, s, d):
    c, v = grid.at(s), vec(d)
    perp = np.array([-v[1], v[0], 0.0])
    mid = c + 0.5 * v
    return m.Line(mid - 0.26 * perp, mid + 0.26 * perp).set_stroke(EXPLORE, 7).set_z_index(3)


def drive(grid, bot, ep, step=0.36, offset=(0.0, 0.0, 0.0), rows=None, trail=True):
    """Rover follows one recorded episode: a glide per move (with a trail segment), a nudge and a VIOLET mark
    against the side it bumped into when a slip left it in place. `rows[k]` fades in with move k."""
    off = np.array(offset)
    seq, marks = [], m.VGroup()
    for k, t in enumerate(ep):
        parts = [m.FadeIn(rows[k], shift=0.1 * m.LEFT)] if rows is not None else []
        if t.s2 == t.s:
            d = bump_direction(t)
            parts.append(bot.animate(rate_func=m.there_and_back).move_to(spot(grid, t.s) + 0.22 * vec(d)))
            if trail:
                mark = bump_mark(grid, t.s, d)
                marks.add(mark)
                parts.append(m.Create(mark))
        else:
            parts.append(bot.animate.move_to(spot(grid, t.s2)))
            if trail:
                seg = m.Line(grid.at(t.s) + off, grid.at(t.s2) + off).set_stroke(ACTION, 5, opacity=0.9).set_z_index(2)
                marks.add(seg)
                parts.append(m.Create(seg))
        seq.append(m.AnimationGroup(*parts, run_time=step))
    return m.Succession(*seq), marks


def static_trail(grid, ep, offset=(0.0, 0.0, 0.0), color=ACTION, width=5):
    off = np.array(offset)
    group = m.VGroup()
    for t in ep:
        if t.s2 == t.s:
            group.add(bump_mark(grid, t.s, bump_direction(t)))
        else:
            group.add(m.Line(grid.at(t.s) + off, grid.at(t.s2) + off).set_stroke(color, width, opacity=0.9).set_z_index(2))
    return group


def cell_number(grid, s, v, size=22, color=INK, fmt="{:.2f}", off=NUM_OFF):
    return code(minus(fmt.format(v)), size, color).move_to(grid.at(s) + off)


def set_values(grid, nums, values, cells, run_time=0.4, fmt="{:.2f}"):
    """Tint the cells by value (kit.heat) and write or update their numbers (nums: cell → Text on screen)."""
    anims = []
    for s in cells:
        v = values[s]
        anims.append(grid.squares[W.to_screen(s)].animate.set_fill(heat(v), 1))
        new = cell_number(grid, s, v, fmt=fmt)
        if s in nums:
            anims.append(m.Transform(nums[s], new))
        else:
            nums[s] = new
            anims.append(m.FadeIn(new))
    return m.AnimationGroup(*anims, run_time=run_time)


def code_lines(lines, size=20):
    """Python output in Menlo: probabilities VIOLET, cells TEAL, rewards GOLD, True RED, False MUTED."""
    return code("\n".join(lines), size, INK, t2c={
        r">>>": MUTED,
        r"\"N\"": ACTION,
        r"(?<=\()0\.\d(?=,)": EXPLORE,
        r"\(\d, \d\)(?=, -)": STATE,
        r"-\d\.\d+": REWARD,
        r"False": MUTED,
    })


def wide_card(color, title, desc, glyph, y, width=12.4, height=1.3):
    box = card(width, height, color, CARD, radius=0.18, stroke=2.5).move_to([0, y, 0])
    glyph.move_to(box.get_left() + [0.65, 0, 0])
    words = m.VGroup(text(title, 28, INK, weight=m.BOLD), text(desc, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
    words.next_to(glyph, m.RIGHT, buff=0.35)
    return m.VGroup(box, glyph, words)


# ---------------------------------------------------------------------------- the episode


class Ep08LearningByPlaying(Episode):
    number = 8

    def lesson(self):
        self.no_map()
        self.one_episode()
        self.average()
        self.settle()
        self.coverage()
        self.control()
        self.trade()
        self.noise()
        self.recap()
        self.next_step()

    # ------------------------------------------------------------------ 0 · take the map away

    def no_map(self):
        self.next_section("0 · Take the map away", notes=(
            "Episodes 5–7 had the world's model: world.transitions((1, 1), 'N') = "
            f"{[(p, s2, round(r, 2), d) for p, s2, r, d in MAP]} (Russell & Norvig 4x3: 0.8 ahead, 0.1 each side, bumping = "
            "staying, −0.04 a move). Episode 7 (value iteration) computed every value from it: "
            f"V(1,1) = {V_PLAN[START]:.3f} (R&N Fig. 17.3). Without the model Rover only sees, move after move, its cell "
            "(state), its move (action) and the reward: model-free learning (S&B 2018 ch. 5; notes §8)."))
        grid = make_grid()
        bot = rover(0.42).move_to(spot(grid, START)).set_z_index(4)
        self.grid, self.bot = grid, bot

        lines = code_lines([">>> world.transitions((1, 1), \"N\")",
                            "[(0.8, (1, 2), -0.04, False),",
                            " (0.1, (1, 1), -0.04, False),",
                            " (0.1, (2, 1), -0.04, False)]"])
        top = m.VGroup(icon(MAP_ICON, 0.62),
                       m.VGroup(text("the map", 28, STATE, weight=m.BOLD),
                                text("every outcome of a move, and its chance", 20, MUTED)).arrange(m.DOWN, buff=0.08,
                                                                                                    aligned_edge=m.LEFT))
        top.arrange(m.RIGHT, buff=0.3)
        inner = m.VGroup(top, lines).arrange(m.DOWN, buff=0.35, aligned_edge=m.LEFT)
        box = card(inner.width + 0.8, inner.height + 0.7, STATE, CARD, radius=0.2, stroke=2.5)
        inner.move_to(box)
        map_card = m.VGroup(box, inner).move_to([RX, 0.9, 0])
        ep7 = pill("Episode 7 · value iteration", STATE, TEAL_FILL, 20, mono=True).move_to([RX, -1.0, 0])
        ep7_sub = text("every value worked out from the map", 22, MUTED).next_to(ep7, m.DOWN, buff=0.22)
        plan_nums = {s: cell_number(grid, s, V_PLAN[s], size=20, fmt="{:.3f}") for s in OPEN}
        unknown = {s: code("?", 22, MUTED, weight=m.BOLD).move_to(grid.at(s) + NUM_OFF) for s in OPEN}
        cost = reward_chip(W.step_reward, 20).move_to(spot(grid, START) + [0, 0.62, 0]).set_z_index(5)

        chips = m.VGroup(chip("state", STATE, TEAL_FILL, width=1.7, size=26),
                         chip("action", ACTION, CORAL_FILL, width=1.7, size=26),
                         chip("reward", REWARD, GOLD_FILL, width=1.7, size=26)).arrange(m.RIGHT, buff=0.35)
        chips.move_to([RX, -0.55, 0])
        sees = text("all Rover gets, move after move", 22, MUTED).next_to(chips, m.DOWN, buff=0.3)
        name = title_text("Model-free learning", 44, VALUE).move_to([RX, 1.25, 0])
        name_sub = text("no model of the world: learn from what happens", 22, MUTED).next_to(name, m.DOWN, buff=0.25)

        self.say(
            "[So far, Rover has had a map]: [every move's outcomes, and their chances, written down.]",
            m.AnimationGroup(self.new_header("00 · no map", "Take the map away"), m.FadeIn(grid),
                             m.FadeIn(bot, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(map_card[0]), m.FadeIn(top, shift=0.1 * m.UP),
                             m.AddTextLetterByLetter(lines, time_per_char=0.012)),
        )
        self.say(
            "[With the map, episode seven worked out every cell's value], [without turning a wheel.]",
            m.AnimationGroup(m.FadeIn(ep7, shift=0.1 * m.UP), m.FadeIn(ep7_sub),
                             *[grid.squares[W.to_screen(s)].animate.set_fill(heat(V_PLAN[s]), 1) for s in OPEN],
                             m.LaggedStart(*[m.FadeIn(plan_nums[s]) for s in OPEN], lag_ratio=0.1)),
            m.Circumscribe(grid.squares[W.to_screen(START)], color=VALUE, buff=0.04),
        )
        self.say(
            "[Now take the map away.] [Rover can't look up a single chance]: [all it can do is play, and see what happens.]",
            m.AnimationGroup(m.FadeOut(map_card, shift=1.2 * m.RIGHT), m.FadeOut(m.VGroup(ep7, ep7_sub), shift=1.2 * m.RIGHT)),
            m.AnimationGroup(*[m.Transform(plan_nums[s], unknown[s]) for s in OPEN],
                             *[grid.squares[W.to_screen(s)].animate.set_fill(CARD, 1) for s in OPEN]),
            m.AnimationGroup(bot.animate(rate_func=m.there_and_back, run_time=0.8).shift(0.25 * m.UP),
                             m.Succession(m.FadeIn(cost, shift=0.2 * m.UP, run_time=0.5), m.Wait(0.4),
                                          m.FadeOut(cost, shift=0.2 * m.UP, run_time=0.4))),
        )
        self.say(
            "[Learning without a model of the world] [is called model-free learning.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips], lag_ratio=0.25),
                             m.FadeIn(sees)),
            m.AnimationGroup(m.FadeIn(name, shift=0.15 * m.UP), m.FadeIn(name_sub)),
        )
        self.wait(0.4)
        self.clear(grid, bot)

    # ------------------------------------------------------------------ 1 · one episode, one return

    def one_episode(self):
        self.next_section("1 · One episode, one return", notes=(
            "Episode 1 of the measured run (play_episode(ROVER_WORLD, RN_POLICY, default_rng(0)), the first of 1,000) is "
            f"Episode 5's drive: cells {[t.s for t in EP1] + [EP1[-1].s2]}, two slips at (3,3) (a bump north, then south), "
            f"rewards −0.04 × 7 and +0.96, return {RETURNS[0]:.2f}. Walking back, G = r + γG with γ = 1 (S&B §5.1, notes §8): "
            f"{[round(g, 2) for g in G_BACK]}. (3,3) is visited at steps 5, 6 and 8 with returns 0.84, 0.88, 0.96: "
            f"first-visit MC keeps 0.84, every-visit averages {EVERY:.3f} (mc_prediction first_visit=False). Both converge "
            "to v_π (S&B §5.1, p. 93)."))
        grid, bot = self.grid, self.bot
        cols = {"#": XL + 0.2, "aim": XL + 1.05, "cell": XL + 2.35, "reward": XL + 3.95, "return": XL + 5.7}
        heads = m.VGroup(*[code(k, 20, FAINT).move_to([x, 2.15, 0]) for k, x in cols.items()])
        ys = [1.6 - 0.42 * k for k in range(len(EP1))]
        rows, cells_txt, returns = [], [], []
        for k, t in enumerate(EP1):
            y = ys[k]
            v = vec(t.a) * 0.15
            c = np.array([cols["aim"], y, 0])
            cell_t = code(f"({t.s[0]},{t.s[1]})", 22, STATE).move_to([cols["cell"], y, 0])
            row = m.VGroup(code(str(k + 1), 20, MUTED).move_to([cols["#"], y, 0]),
                           m.Arrow(c - v, c + v, buff=0, color=ACTION, stroke_width=5, max_tip_length_to_length_ratio=0.45),
                           cell_t,
                           code(minus(f"{t.r:+.2f}"), 22, GOLD if t.r > 0 else RED).move_to([cols["reward"], y, 0]))
            rows.append(row)
            cells_txt.append(cell_t)
            returns.append(code(f"{G_BACK[k]:.2f}", 22, VALUE, weight=m.BOLD).move_to([cols["return"], y, 0]))
        rule = m.Line([XL - 0.1, -1.75, 0], [XL + 6.4, -1.75, 0]).set_stroke(EDGE, 2)
        total_lab = left_at(text("the drive's return", 22, MUTED), XL - 0.1, -2.2)
        total = code(f"{RETURNS[0]:.2f}", 30, VALUE, weight=m.BOLD).move_to([cols["return"], -2.2, 0])
        sit(total_lab, baseline(total, total.text))
        rule_g = m.VGroup(code("G = r + G", 22, VALUE, weight=m.BOLD), text("walking back from the end", 20, MUTED))
        rule_g.arrange(m.RIGHT, buff=0.3)
        left_at(rule_g, XL - 0.1, -2.8)
        sit(rule_g[1], baseline(rule_g[0], rule_g[0].text))

        run, trail = drive(grid, bot, EP1, step=0.36, rows=rows)
        dot = m.Dot(grid.at(CHARGER), radius=0.12, color=VALUE).set_z_index(5)
        nums = {}
        back = []
        for k in range(len(EP1) - 1, -1, -1):
            s = EP1[k].s
            back.append(m.AnimationGroup(dot.animate.move_to(grid.at(s)), m.FadeIn(returns[k], shift=0.1 * m.LEFT),
                                         set_values(grid, nums, {s: G_BACK[k]}, [s], run_time=0.42), run_time=0.42))
        self.nums = nums
        assert set(nums) == visited(1)

        rings = m.VGroup(*[m.SurroundingRectangle(returns[k], buff=0.07, corner_radius=0.06).set_stroke(VALUE, 3)
                           for k in VISITS])
        cell_rings = m.VGroup(*[m.SurroundingRectangle(cells_txt[k], buff=0.07, corner_radius=0.06).set_stroke(STATE, 3)
                                for k in VISITS])
        first = m.VGroup(code("first-visit", 22, VALUE, weight=m.BOLD), code(f"{G_BACK[VISITS[0]]:.2f}", 24, INK, weight=m.BOLD),
                         text("keeps the first return", 20, MUTED))
        every = m.VGroup(code("every-visit", 22, VALUE, weight=m.BOLD), code(f"{EVERY:.2f}", 24, INK, weight=m.BOLD),
                         text("averages all three", 20, MUTED))
        for row_, y in ((first, -2.2), (every, -2.75)):
            left_at(row_[0], XL - 0.1, y)
            row_[1].move_to([XL + 2.6, y, 0])
            left_at(row_[2], XL + 3.3, y)
            sit(row_[1], baseline(row_[0], row_[0].text))
            sit(row_[2], baseline(row_[0], row_[0].text))
        both = cite("both converge to the true value · Sutton & Barto, 2018, §5.1")
        left_at(both, XL - 0.1, -3.3)

        self.say(
            "[Here's the eight-move drive from episode five again], [slips and all.]",
            m.AnimationGroup(self.new_header("01 · play", "One episode, one return"), m.FadeIn(heads),
                             m.Succession(m.Wait(0.5), run)),
            m.Circumscribe(grid.squares[W.to_screen(NEAR)], color=EXPLORE, buff=0.04),
        )
        self.say(
            "[It reaches the charger with a return of zero point six eight.]",
            m.AnimationGroup(m.Flash(grid.at(CHARGER), color=GOLD, flash_radius=0.55, line_length=0.18), m.Create(rule),
                             m.FadeIn(total_lab), m.FadeIn(total, shift=0.1 * m.UP)),
        )
        self.say(
            "[Now walk back along the path from the end], "
            "[adding up the rewards as you go: each cell gets the sum of everything that followed it.]",
            m.AnimationGroup(m.FadeOut(trail), m.FadeIn(dot, scale=0.5), m.FadeIn(rule_g, shift=0.1 * m.UP)),
            m.Succession(*back),
        )
        self.say(
            "[That's the return from each cell]: [one sample of what the cell is worth.]",
            m.AnimationGroup(m.FadeOut(dot), m.Circumscribe(m.VGroup(*returns), color=VALUE, buff=0.1)),
            m.LaggedStart(*[m.Indicate(nums[s], color=VALUE, scale_factor=1.2) for s in nums], lag_ratio=0.12),
        )
        self.say(
            "[Rover passed the cell beside the charger three times], [so it got three returns]: "
            "[zero point eight four, zero point eight eight and zero point nine six.]",
            m.AnimationGroup(m.Circumscribe(grid.squares[W.to_screen(NEAR)], color=STATE, buff=0.04),
                             m.LaggedStart(*[m.Create(r) for r in cell_rings], lag_ratio=0.3)),
            m.Wait(0.1),
            m.LaggedStart(*[m.Create(r) for r in rings], lag_ratio=0.45),
        )
        self.say(
            "[First-visit Monte Carlo keeps only the first]; [every-visit averages all three.] "
            "[Both converge to the true value.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(total_lab, total, rule_g)), m.FadeIn(first, shift=0.1 * m.UP),
                             m.Circumscribe(nums[NEAR], color=VALUE, buff=0.06)),
            m.FadeIn(every, shift=0.1 * m.UP),
            m.FadeIn(both),
        )
        self.wait(0.3)
        self.clear(grid, bot, *nums.values())

    # ------------------------------------------------------------------ 2 · many episodes, one average

    def average(self):
        self.next_section("2 · Many episodes, one average", notes=(
            "The measured run continues (classic.json rover_mc, seed 0, first-visit, mc_prediction): returns from the "
            f"start in episodes 1–4: {[round(g, 2) for g in RETURNS[:4]]} (episode 4 ends in the puddle). The start's "
            f"average after 1, 2, 3, 4 episodes: {[round(EST[n], 3) for n in (1, 2, 3, 4)]}; after 10 / 100 / 1,000: "
            f"{EST[10]:.3f} / {EST[100]:.3f} / {EST[N_EP]:.4f} (classic.md §5: 0.680, 0.544, 0.688, 0.711). (2,1) is first "
            "visited in episode 15 (a slip east from the start). Code: mc_prediction, simplified (rl_reference.py keeps "
            "the first visit with a 'seen' set)."))
        grid, bot, nums = self.grid, self.bot, self.nums
        src = "\n".join([
            "G, first = 0.0, {}",
            "for step in reversed(episode):",
            "    G = step.r + gamma * G",
            "    first[step.s] = G",
            "for s, G in first.items():",
            "    n[s] += 1",
            "    V[s] += (G - V[s]) / n[s]",
        ])
        block = code_block(src, 20, width=6.6, title="mc_prediction, simplified")
        block.move_to([RX, 1.0, 0])
        body = block[1]
        lines = src.split("\n")

        def line_glyphs(i):
            start = sum(len(l.replace(" ", "")) for l in lines[:i])
            return m.VGroup(*body[start:start + len(lines[i].replace(" ", ""))])

        g_line, avg_line = line_glyphs(2), line_glyphs(6)
        avg_tag = text("a running average", 20, VALUE).next_to(block, m.DOWN, buff=0.12).align_to(block, m.RIGHT).shift(0.1 * m.LEFT)

        lab = left_at(text("returns from the start cell", 22, MUTED), XL, -1.2)
        ret_chips = [code(minus(f"{RETURNS[i]:.2f}"), 26, VALUE, weight=m.BOLD) for i in range(4)]
        for i, c in enumerate(ret_chips):
            left_at(c, XL + 1.45 * i, -1.8)
        avg_lab = left_at(text("average", 22, MUTED), XL, -2.5)
        avgs = {n: code(f"{EST[n]:.2f}", 32, VALUE, weight=m.BOLD) for n in (3, 4)}
        for c in avgs.values():
            left_at(c, XL + 1.45, -2.5)
            sit(c, baseline(avg_lab, avg_lab.text))

        def back_to_start():
            return bot.animate(run_time=0.3).move_to(spot(grid, START))

        runs = [drive(grid, bot, EPISODES[i], step=0.17, trail=False)[0] for i in (1, 2, 3)]

        # the fast-forward table
        ff_cols = [XL + 3.2, XL + 4.6, XL + 6.0]
        ff_head = left_at(text("episodes", 22, MUTED), XL, -1.55)
        ff_lab = left_at(text("start cell", 22, MUTED), XL, -2.25)
        ff_n = [code(f"{n:,}", 24, INK).move_to([x, -1.55, 0]) for n, x in zip((10, 100, 1000), ff_cols)]
        ff_v = [code(f"{EST[n]:.2f}", 28, VALUE, weight=m.BOLD).move_to([x, -2.25, 0]) for n, x in zip((10, 100, 1000), ff_cols)]
        for t_ in ff_n:
            sit(t_, baseline(ff_head, ff_head.text))
        for t_ in ff_v:
            sit(t_, baseline(ff_lab, ff_lab.text))
        plan = cite(f"planned with the map: {V_PLAN[START]:.3f}")
        left_at(plan, XL, -2.95)

        self.say(
            "[Play again, and again]: [each new episode adds a return to every cell it passes],",
            m.AnimationGroup(self.new_header("02 · average", "Many episodes, one average"),
                             m.FadeIn(block, shift=0.1 * m.UP), m.FadeIn(lab), m.FadeIn(ret_chips[0])),
            m.Succession(back_to_start(), runs[0], set_values(grid, nums, HIST[2], visited(2)),
                         m.FadeIn(ret_chips[1], shift=0.1 * m.LEFT, run_time=0.3),
                         back_to_start(), runs[1], set_values(grid, nums, HIST[3], visited(3)),
                         m.FadeIn(ret_chips[2], shift=0.1 * m.LEFT, run_time=0.3)),
        )
        self.say(
            "[and each cell's estimate is simply the average of all its returns so far.]",
            m.AnimationGroup(avg_line.animate.set_color(VALUE), g_line.animate.set_color(VALUE),
                             m.FadeIn(avg_tag, shift=0.1 * m.UP), m.FadeIn(avg_lab), m.FadeIn(avgs[3], shift=0.1 * m.UP)),
        )
        self.say(
            "[The fourth episode slides into the puddle]: [minus one point two four], "
            "[and the start's average drops to zero point two two.]",
            m.Succession(back_to_start(), runs[2], m.Flash(grid.at(PUDDLE), color=RED, flash_radius=0.55, line_length=0.18)),
            m.FadeIn(ret_chips[3], shift=0.1 * m.LEFT),
            m.AnimationGroup(m.Transform(avgs[3], avgs[4]), set_values(grid, nums, HIST[4], visited(4), run_time=0.8)),
        )
        self.say(
            "[Keep playing.] [After ten episodes, the start reads zero point five four]; "
            "[after a hundred, zero point six nine]; [after a thousand, zero point seven one.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(lab, *ret_chips, avg_lab, avgs[3])), m.FadeOut(bot),
                             m.FadeIn(ff_head), m.FadeIn(ff_lab)),
            m.AnimationGroup(m.FadeIn(ff_n[0]), m.FadeIn(ff_v[0], shift=0.1 * m.UP), set_values(grid, nums, HIST[10], visited(10))),
            m.AnimationGroup(m.FadeIn(ff_n[1]), m.FadeIn(ff_v[1], shift=0.1 * m.UP), set_values(grid, nums, HIST[100], visited(100))),
            m.AnimationGroup(m.FadeIn(ff_n[2]), m.FadeIn(ff_v[2], shift=0.1 * m.UP), set_values(grid, nums, HIST[N_EP], visited(N_EP)),
                             m.FadeIn(plan)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · the average settles

    def settle(self):
        self.next_section("3 · The average settles", notes=(
            "The start cell's first-visit average after every episode of the measured run (classic.json rover_mc: one "
            "seeded run, 1,000 episodes, recomputed here and asserted). Dip: episode 4 (the puddle), "
            f"{EST[4]:.2f}. Dashed: the planned value {EXACT[START]:.4f} (policy evaluation with the map; R&N 0.705). Band: "
            f"0.705 ± σ/√n with σ = {SD_ONE:.4f}, the standard deviation of one return from the start (computed exactly from "
            "the model's second moment; notes §8 simulated 0.251), i.e. how far an average of n returns typically strays."))
        plot = Plot((1, 1500), (0, 1), width=10.2, height=3.9, center=(0.25, -0.65, 0), x_ticks=[1, 10, 100, 1000],
                    y_ticks=[0, 0.25, 0.5, 0.75, 1.0], x_label="episodes played", y_label="the start cell's estimate",
                    log_x=True, x_fmt="{:,}", y_fmt="{:g}")
        ns = np.arange(1, N_EP + 1)
        curve = plot.line(ns, EST[1:], VALUE, 4).set_z_index(2)
        plan_line = m.DashedLine(plot.coords(1, EXACT[START]), plot.coords(N_EP, EXACT[START]), dash_length=0.1)
        plan_line.set_stroke(INK, 2.5).set_z_index(1)
        bx = np.logspace(0, 3, 160)
        band = plot.band(bx, EXACT[START] - SD_ONE / np.sqrt(bx), EXACT[START] + SD_ONE / np.sqrt(bx), EXPLORE, 0.2)
        right = plot.coords(1500, 0)[0]
        key1 = m.VGroup(m.DashedLine([0, 0, 0], [0.55, 0, 0], dash_length=0.1).set_stroke(INK, 2.5),
                        text(f"{EXACT[START]:.3f} · planned with the map (Episode 7)", 22, INK)).arrange(m.RIGHT, buff=0.2)
        key2 = m.VGroup(m.Rectangle(width=0.55, height=0.24).set_stroke(EXPLORE, 1.5).set_fill(EXPLORE, 0.3),
                        text("how far an average of that many returns typically strays", 20, EXPLORE)).arrange(m.RIGHT, buff=0.2)
        for key_, yv in ((key1, 0.98), (key2, 0.88)):
            key_.move_to([0, plot.coords(1, yv)[1], 0]).align_to([right, 0, 0], m.RIGHT)
        key1.align_to(key2, m.LEFT)
        dip = plot.dot(4, EST[4], RED, 0.09)
        dip_lab = text("episode 4: the puddle", 20, RED).next_to(dip, m.RIGHT, buff=0.18)
        end = plot.dot(N_EP, EST[N_EP], VALUE, 0.09)
        end_lab = code(f"{EST[N_EP]:.3f}", 22, VALUE, weight=m.BOLD).next_to(end, m.DOWN, buff=0.22)
        src = cite("measured on this Mac · one seeded run, 1,000 episodes · first-visit")
        src.move_to([0, 2.45, 0]).align_to([right, 0, 0], m.RIGHT)

        self.say(
            "[Here's the start cell's estimate after every one of those episodes], [measured on this Mac.]",
            m.AnimationGroup(self.new_header("03 · settle", "The average settles"), m.FadeIn(plot),
                             m.Create(curve, run_time=3.0)),
            m.FadeIn(src, shift=0.1 * m.DOWN),
        )
        self.say(
            "[It swings wildly at first], [then settles near zero point seven zero five]: "
            "[the value episode seven planned with the map.]",
            m.AnimationGroup(m.FadeIn(dip, scale=0.5), m.FadeIn(dip_lab, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(plan_line), m.FadeIn(end, scale=0.5), m.FadeIn(end_lab)),
            m.FadeIn(key1, shift=0.1 * m.UP),
        )
        self.say(
            "[The violet band shows how far an average of that many returns typically strays]: "
            "[it narrows as the episodes pile up.]",
            m.AnimationGroup(m.FadeIn(band), m.FadeIn(key2, shift=0.1 * m.UP)),
            m.Circumscribe(m.VGroup(end, end_lab), color=EXPLORE, buff=0.12),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · only where Rover goes

    def coverage(self):
        self.next_section("4 · Only where Rover goes", notes=(
            f"After 1,000 episodes (measured run): the largest gap between Monte Carlo and the planned values over the "
            f"cells Rover visited is {MAX_GAP:.4f} (classic.json max_abs_error_after['1000'] = 0.03767), at (3,2). "
            f"Never visited: {NEVER}: under RN_POLICY (R&N Fig. 17.2) the start heads north, (2,1) (reached only by a slip) "
            "heads west, and no slip from a visited cell lands in (3,1) or (4,1), so their averages never start "
            f"(planned values {EXACT[(3, 1)]:.3f} and {EXACT[(4, 1)]:.3f})."))
        grid = make_grid()
        mc_off, plan_off = np.array([0, -0.02, 0]), np.array([0, -0.37, 0])
        seen = visited(N_EP)
        mc_nums = m.VGroup()
        for s in OPEN:
            if s in seen:
                grid.squares[W.to_screen(s)].set_fill(heat(HIST[N_EP][s]), 1)
                mc_nums.add(cell_number(grid, s, HIST[N_EP][s], size=24, off=mc_off))
        blanks = m.VGroup(*[code("?", 24, MUTED, weight=m.BOLD).move_to(grid.at(s) + mc_off) for s in NEVER])
        plan_nums = m.VGroup(*[cell_number(grid, s, EXACT[s], size=20, color=MUTED, fmt="{:.3f}", off=plan_off) for s in OPEN])
        arrows = grid.world_arrows(RN_POLICY, length=0.3, offset=0.31, width=4).set_z_index(3)

        leg1 = m.VGroup(code(f"{HIST[N_EP][START]:.2f}", 24, INK, weight=m.BOLD), text("Monte Carlo, 1,000 episodes", 22, INK))
        leg2 = m.VGroup(code(f"{EXACT[START]:.3f}", 20, MUTED), text("planned with the map", 22, MUTED))
        for row_, y in ((leg1, 1.3), (leg2, 0.75)):
            row_[0].move_to([XL + 0.5, y, 0])
            left_at(row_[1], XL + 1.4, y)
            sit(row_[1], baseline(row_[0], row_[0].text))
        gap = m.VGroup(text("largest gap, any cell Rover visited", 22, MUTED),
                       code(f"{MAX_GAP:.3f}", 40, GOOD, weight=m.BOLD)).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        left_at(gap, XL, -0.45)
        rings = m.VGroup(*[m.DashedVMobject(m.Square(CELL - 0.2), num_dashes=24).set_stroke(EXPLORE, 3).move_to(grid.at(s))
                           for s in NEVER]).set_z_index(2)
        never = m.VGroup(text("never visited", 24, EXPLORE, weight=m.BOLD),
                         text("no returns, so no average", 20, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(never, XL, -2.1)

        self.say(
            "[Across the grid, every cell Rover visited] [ends up within zero point zero four of the plan.]",
            m.AnimationGroup(self.new_header("04 · coverage", "Only where Rover goes"), m.FadeIn(grid),
                             m.FadeIn(mc_nums), m.FadeIn(blanks), m.FadeIn(leg1, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(plan_nums, shift=0.05 * m.UP), m.FadeIn(leg2, shift=0.1 * m.UP),
                             m.FadeIn(gap, shift=0.1 * m.UP)),
        )
        self.say(
            "[Except two cells]: [Rover's arrows lead up and around, never down here], [so their averages never even start.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(r) for r in rings], lag_ratio=0.3)),
            m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.08),
            m.AnimationGroup(m.FadeIn(never, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.Indicate(b, color=EXPLORE, scale_factor=1.4) for b in blanks], lag_ratio=0.3)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · control: Q and exploration

    def control(self):
        self.next_section("5 · To choose, learn Q", notes=(
            "Control without a model (S&B 2018 §5.2–5.4, notes §8): picking a move from V needs p(s′ | s, a), so learn "
            f"Q(s, a), the average return after move a in cell s. In the measured run every visit to the start moved north, "
            f"so Q(start, N) is the start's average, {Q_START_N:.4f}; east, south and west were never tried. Exploring "
            f"starts (S&B §5.3): every episode begins from a random cell and first move (the four draws shown are "
            f"illustrative, default_rng({ES_SEED}): {ES}). ε-soft (§5.4): every move keeps probability ≥ ε/|A|; "
            f"ε-greedy with ε = 0.1 and 4 moves gives the best move 1 − ε + ε/4 = {P_BEST:.3f} and each other ε/4 = "
            f"{P_OTHER:.3f}."))
        grid = make_grid()
        bot = rover(0.34).move_to(grid.at(START)).set_z_index(4)
        arrows = grid.world_arrows(RN_POLICY, length=0.3, offset=0.31, width=4).set_z_index(3)
        c0 = grid.at(START)
        compass = m.VGroup(*[m.Arrow(c0 + 0.27 * vec(a), c0 + 0.57 * vec(a), buff=0, color=ACTION, stroke_width=7,
                                     max_tip_length_to_length_ratio=0.5) for a in ACTIONS]).set_z_index(3)

        no_map = icon(MAP_ICON, 0.6)
        strike = m.Line(no_map.get_corner(m.DL) + [-0.05, -0.05, 0], no_map.get_corner(m.UR) + [0.05, 0.05, 0]).set_stroke(RED, 6)
        no_map_g = m.VGroup(no_map, strike)
        v_words = m.VGroup(text("a cell's value alone can't pick a move:", 22, INK),
                           text("where each move lands was in the map", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        top_row = m.VGroup(no_map_g, v_words).arrange(m.RIGHT, buff=0.35)
        left_at(top_row, XL, 1.9)

        q_title = m.VGroup(m.MathTex("Q(s, a)", font_size=44, color=VALUE),
                           text("the average return after move a in cell s", 22, MUTED)).arrange(m.RIGHT, buff=0.35)
        left_at(q_title, XL, 1.9)
        q_cards = m.VGroup()
        for a in ACTIONS:
            box = card(1.35, 1.05, VALUE if a == "N" else EDGE, CARD, radius=0.14, stroke=2.5)
            v = vec(a) * 0.17
            arr = m.Arrow(-v, v, buff=0, color=ACTION, stroke_width=5, max_tip_length_to_length_ratio=0.45)
            val = code(f"{Q_START_N:.2f}", 26, VALUE, weight=m.BOLD) if a == "N" else code("?", 28, MUTED, weight=m.BOLD)
            inner = m.VGroup(arr, val).arrange(m.DOWN, buff=0.15)
            q_cards.add(m.VGroup(box, inner.move_to(box)))
        q_cards.arrange(m.RIGHT, buff=0.3)
        left_at(q_cards, XL, 0.7)
        q_where = cite("at the start cell, after 1,000 episodes").next_to(q_cards, m.DOWN, buff=0.18).align_to(q_cards, m.LEFT)
        untried = text("never tried", 22, RED, weight=m.BOLD).next_to(q_cards[1:], m.UP, buff=0.12)

        dice = icon("dice", 0.62)
        es_title = m.VGroup(dice, m.VGroup(text("exploring starts", 28, EXPLORE, weight=m.BOLD),
                                           text("a random cell and a random first move", 20, MUTED)).arrange(
            m.DOWN, buff=0.08, aligned_edge=m.LEFT)).arrange(m.RIGHT, buff=0.3)
        left_at(es_title, XL, -1.25)
        es_marks, es_moves = [], []
        for s, a in ES:
            sq = m.DashedVMobject(m.Square(CELL - 0.2), num_dashes=24).set_stroke(EXPLORE, 3).move_to(grid.at(s)).set_z_index(2)
            p = grid.at(s) + [0.36, -0.36, 0]
            v = vec(a) * 0.14
            es_marks.append(sq)
            es_moves.append(m.Arrow(p - v, p + v, buff=0, color=ACTION, stroke_width=4, max_tip_length_to_length_ratio=0.45).set_z_index(3))
        sim = m.VGroup(pill("fine in a simulator", GOOD, GOOD_FILL, 20), pill("hard for a real robot", RED, RED_FILL, 20))
        sim.arrange(m.RIGHT, buff=0.3)
        left_at(sim, XL, -2.35)

        bars = Bars([1.0, 0.0, 0.0, 0.0], labels=list(ACTIONS), top=1.0, bottom=0.0, width=3.4, height=1.6,
                    center=(XL + 1.75, -2.05, 0), color=ACTION, gap=0.35, fmt="{:.3f}", size=20)
        for lab_ in bars.labels:
            lab_.set_color(ACTION)
        soft_title = m.VGroup(text("ε-soft:", 26, EXPLORE, weight=m.BOLD), text("every move keeps a chance", 22, MUTED))
        soft_title.arrange(m.RIGHT, buff=0.2)
        sit(soft_title[1], baseline(soft_title[0], soft_title[0].text))
        left_at(soft_title, XL, -0.6)
        f_best = code(f"1 − ε + ε/4 = {P_BEST:.3f}", 22, INK, t2c={r"ε": EXPLORE, r"0\.925": ACTION})
        f_other = code(f"ε/4 = {P_OTHER:.3f}", 22, INK, t2c={r"ε": EXPLORE, r"0\.025": ACTION})
        eps_pill = pill("ε-greedy, ε = 0.1", EXPLORE, VIOLET_FILL, 20, mono=True)
        side = m.VGroup(eps_pill, f_best, f_other).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        left_at(side, XL + 3.9, -2.0)

        self.say(
            "[So far, Rover only judged one fixed set of arrows.] [To improve them, it has to compare moves],",
            m.AnimationGroup(self.new_header("05 · control", "To choose, learn Q"), m.FadeIn(grid),
                             m.FadeIn(bot, shift=0.1 * m.UP), m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.06)),
            m.AnimationGroup(m.FadeOut(arrows), m.LaggedStart(*[m.GrowArrow(a) for a in compass], lag_ratio=0.2)),
        )
        self.say(
            "[and without the map], [a cell's value can't say where each move would land.]",
            m.FadeIn(no_map_g, shift=0.1 * m.UP),
            m.FadeIn(v_words, shift=0.1 * m.UP),
        )
        self.say(
            "[So Rover averages the returns after each cell and move instead]: [that's Q of s and a.]",
            m.AnimationGroup(m.FadeOut(top_row, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in q_cards], lag_ratio=0.2),
                             m.FadeIn(q_where)),
            m.FadeIn(q_title, shift=0.1 * m.UP),
        )
        self.say(
            "[But if Rover always takes its best-looking move], [the others are never tried], [and their averages never start.]",
            m.AnimationGroup(m.Circumscribe(q_cards[0], color=VALUE, buff=0.06), compass[0].animate.set_stroke(width=8)),
            m.AnimationGroup(*[q.animate.set_opacity(0.45) for q in q_cards[1:]],
                             *[a.animate.set_opacity(0.3) for a in compass[1:]]),
            m.AnimationGroup(m.FadeIn(untried, shift=0.1 * m.DOWN),
                             m.LaggedStart(*[m.Circumscribe(q[1][1], color=RED, buff=0.06) for q in q_cards[1:]], lag_ratio=0.25)),
        )
        hops = []
        for (s, a), sq, mv in zip(ES, es_marks, es_moves):
            hops.append(m.AnimationGroup(bot.animate(run_time=0.35).move_to(grid.at(s)), m.Create(sq, run_time=0.35)))
            hops.append(m.GrowArrow(mv, run_time=0.3))
        self.say(
            "[One fix is exploring starts]: [begin each episode in a random cell, with a random first move.]",
            m.AnimationGroup(m.FadeOut(compass),
                             m.Succession(m.FadeIn(es_title, shift=0.1 * m.UP, run_time=0.6),
                                          m.Rotate(dice, angle=m.TAU / 2, run_time=0.6))),
            m.Succession(*hops),
        )
        self.say(
            "[Easy in a simulator], [though you can't drop a real robot just anywhere.]",
            m.FadeIn(sim[0], shift=0.1 * m.UP),
            m.FadeIn(sim[1], shift=0.1 * m.UP),
        )
        self.say(
            "[The other fix is an epsilon-soft policy]: [every move keeps at least a small chance.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(es_title, sim, *es_marks, *es_moves)),
                             bot.animate.move_to(grid.at(START)),
                             m.FadeIn(soft_title, shift=0.1 * m.UP), m.FadeIn(bars.zero), m.FadeIn(bars.labels),
                             m.FadeIn(bars.bars), m.FadeIn(bars.numbers)),
            bars.set([P_BEST, P_OTHER, P_OTHER, P_OTHER], run_time=1.2),
        )
        self.say(
            "[With epsilon-greedy at zero point one, the best move gets ninety-two and a half percent], "
            "[and each other move two and a half.]",
            m.AnimationGroup(m.FadeIn(eps_pill, shift=0.1 * m.UP), m.FadeIn(f_best, shift=0.1 * m.UP),
                             m.Circumscribe(m.VGroup(bars.bars[0], bars.numbers[0]), color=ACTION, buff=0.06)),
            m.AnimationGroup(m.FadeIn(f_other, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.Circumscribe(m.VGroup(bars.bars[i], bars.numbers[i]), color=EXPLORE, buff=0.06)
                                             for i in (1, 2, 3)], lag_ratio=0.2)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the trade

    def trade(self):
        self.next_section("6 · No map, no guesses", notes=(
            "Strengths and costs of Monte Carlo (S&B 2018 ch. 5; notes §8): it needs only episodes, played or simulated; "
            "Example 5.1 Blackjack: DP would need every probability of the next cards ('often complex and error-prone'), "
            "while generating sample games is easy. 'Monte Carlo methods do not bootstrap' (§5.1, p. 95): every estimate "
            "uses only real returns, so the sample averages are unbiased. Cost: it learns only when an episode ends, so it "
            "needs episodic tasks."))
        good1 = wide_card(GOOD, "No model needed", "only episodes, played or simulated", icon("check", 0.72), 1.65)
        bj = m.VGroup(icon(CARDS_ICON, 0.68),
                      m.VGroup(text("blackjack: odds are hard to work out,", 20, INK),
                               text("but practice hands are easy to deal", 20, INK),
                               cite("Sutton & Barto, 2018, Example 5.1")).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT))
        bj.arrange(m.RIGHT, buff=0.25)
        bj.move_to(good1[0]).align_to(good1[0], m.RIGHT).shift(0.35 * m.LEFT)
        good2 = wide_card(GOOD, "Unbiased", "no estimate built on another estimate", icon("check", 0.72), 0.1)
        quote = m.VGroup(text("“Monte Carlo methods do not bootstrap”", 22, INK),
                         cite("Sutton & Barto, 2018, §5.1")).arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT)
        quote.move_to(good2[0]).align_to(good2[0], m.RIGHT).shift(0.4 * m.LEFT)
        bad = wide_card(RED, "Waits for the end", "learns nothing until the episode is over", icon(CLOCK, 0.72), -1.45)
        bad_note = text("tasks must end", 22, RED, weight=m.BOLD).move_to(bad[0]).align_to(bad[0], m.RIGHT).shift(0.4 * m.LEFT)
        mascot = rover(0.5).move_to([0, -2.85, 0])

        self.say(
            "[What Monte Carlo gets right]: [it needs no model of the world, only episodes.]",
            m.AnimationGroup(self.new_header("06 · the trade", "No map, no guesses"), m.FadeIn(good1[0]), m.FadeIn(good1[1]),
                             m.FadeIn(good1[2][0], shift=0.1 * m.UP)),
            m.FadeIn(good1[2][1], shift=0.1 * m.UP),
        )
        self.say(
            "[In the textbook by Sutton and Barto, the classic example is blackjack]: [working out the odds is hard], "
            "[but dealing practice hands is easy.]",
            m.AnimationGroup(m.FadeIn(bj[0], scale=0.7), m.FadeIn(bj[1][2])),
            m.FadeIn(bj[1][0], shift=0.1 * m.LEFT),
            m.FadeIn(bj[1][1], shift=0.1 * m.LEFT),
        )
        self.say(
            "[And it never builds on other estimates, only on real returns], [so its averages are unbiased.]",
            m.AnimationGroup(m.FadeIn(good2, shift=0.1 * m.UP), m.FadeIn(quote)),
            m.Circumscribe(good2[2][0], color=GOOD, buff=0.08),
        )
        self.say(
            "[The cost: Rover learns nothing until an episode ends], [so the task has to end.]",
            m.FadeIn(bad, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(bad_note, shift=0.1 * m.LEFT), m.FadeIn(mascot, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · noise

    def noise(self):
        self.next_section("7 · Returns are noisy", notes=(
            "Two drives from the start with the same arrows (Episode 6's seeds 11 and 9): 5 moves, +0.80; 8 moves into the "
            f"puddle, −1.32. One return's standard deviation from the start: {SD_ONE:.4f} (exact, from the model's second "
            f"moment; notes §8 simulated 0.251); an average of n returns strays by σ/√n: {SD_100:.4f} for n = 100 (notes "
            "Appendix C: 0.251 → 0.025). Ten times the precision needs a hundred times the episodes."))
        grid = make_grid()
        bot = rover(0.42).move_to(spot(grid, START)).set_z_index(4)
        t_good = static_trail(grid, LUCKY, offset=(-0.12, 0.12, 0))
        t_bad = static_trail(grid, UNLUCKY, offset=(0.12, -0.12, 0), color=RED)
        rows = m.VGroup()
        for i, (ep, ic) in enumerate(((LUCKY, "charger"), (UNLUCKY, "puddle"))):
            y = 1.55 - 0.65 * i
            g = sum(t.r for t in ep)
            rows.add(m.VGroup(m.Line([0, 0, 0], [0.5, 0, 0]).set_stroke(ACTION if i == 0 else RED, 5).move_to([XL + 0.25, y, 0]),
                              icon(ic, 0.42).move_to([XL + 1.0, y, 0]),
                              left_at(code(f"{len(ep)} moves", 22, MUTED), XL + 1.5, y),
                              code(minus(f"{g:+.2f}"), 30, VALUE, weight=m.BOLD).move_to([XL + 4.6, y, 0])))
        same = left_at(text("same cell, same arrows", 22, MUTED), XL, 2.2)

        lo_x, hi_x, ax_y = XL + 0.1, XL + 6.3, -2.55
        to_x = lambda v: lo_x + (v - 0.4) / 0.6 * (hi_x - lo_x)
        axis = m.Line([lo_x, ax_y, 0], [hi_x, ax_y, 0]).set_stroke(FAINT, 3)
        ticks = m.VGroup()
        for v in (0.4, 0.6, 0.8, 1.0):
            ticks.add(m.Line([to_x(v), ax_y, 0], [to_x(v), ax_y - 0.1, 0]).set_stroke(FAINT, 2),
                      code(f"{v:g}", 20, MUTED).move_to([to_x(v), ax_y - 0.35, 0]))
        mark = m.DashedLine([to_x(EXACT[START]), ax_y, 0], [to_x(EXACT[START]), 0.15, 0], dash_length=0.08).set_stroke(INK, 2)
        mark_lab = code(f"{EXACT[START]:.3f}", 20, INK).next_to(mark, m.UP, buff=0.1)

        def spread(sd, y):
            bar = m.RoundedRectangle(corner_radius=0.08, width=to_x(EXACT[START] + sd) - to_x(EXACT[START] - sd), height=0.22)
            return bar.set_stroke(EXPLORE, 2).set_fill(VIOLET_FILL, 1).move_to([to_x(EXACT[START]), y, 0])

        wide, narrow = spread(SD_ONE, -0.55), spread(SD_100, -1.75)
        wide_lab = m.VGroup(text("one return", 22, INK), code(f"± {SD_ONE:.2f}", 22, EXPLORE, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        narrow_lab = m.VGroup(text("average of 100", 22, INK), code(f"± {SD_100:.3f}", 22, EXPLORE, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        for lab_, bar_ in ((wide_lab, wide), (narrow_lab, narrow)):
            sit(lab_[1], baseline(lab_[0], lab_[0].text))
            lab_.next_to(bar_, m.UP, buff=0.14)
            left_at(lab_, to_x(EXACT[START]) + 0.25, lab_.get_y())
        typical = cite("typical spread · one standard deviation").move_to([0, ax_y - 0.78, 0]).align_to([hi_x + 0.3, 0, 0], m.RIGHT)
        rule = text("×10 precision = ×100 episodes", 26, INK, weight=m.BOLD,
                    t2c={"×10": EXPLORE, "×100": EXPLORE}).move_to([GRID_C[0], -3.05, 0])

        self.say(
            "[Worse, single returns are noisy]: [same cell, same arrows], [plus zero point eight in one drive], "
            "[minus one point three two in another.]",
            m.AnimationGroup(self.new_header("07 · noise", "One return is a rough guess"), m.FadeIn(grid),
                             m.FadeIn(bot, shift=0.1 * m.UP)),
            m.FadeIn(same),
            m.AnimationGroup(m.Create(t_good, run_time=1.2), m.FadeIn(rows[0], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(t_bad, run_time=1.4), m.FadeIn(rows[1], shift=0.1 * m.LEFT),
                             m.Flash(grid.at(PUDDLE), color=RED, flash_radius=0.55, line_length=0.18)),
        )
        self.say(
            "[One return misses the value by about zero point two five, typically]; "
            "[an average of a hundred still misses by zero point zero two five.]",
            m.AnimationGroup(m.Create(axis), m.FadeIn(ticks), m.Create(mark), m.FadeIn(mark_lab),
                             m.GrowFromCenter(wide), m.FadeIn(wide_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromCenter(narrow), m.FadeIn(narrow_lab, shift=0.1 * m.UP), m.FadeIn(typical)),
        )
        self.say(
            "[Ten times the precision costs a hundred times the episodes.]",
            m.FadeIn(rule, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            "Play (whole episodes, no map), sum (the rewards that followed each cell, walking back: G = r + γG), average "
            f"(first-visit means in every cell). Measured: the start cell's average {EST[N_EP]:.3f} after 1,000 episodes vs "
            f"the planned {EXACT[START]:.3f}."))
        halo = glow(VALUE, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Play", 72, ACTION), title_text("Sum", 72, REWARD), title_text("Average", 72, VALUE))
        words.arrange(m.RIGHT, buff=1.6).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text("whole episodes", 24, MUTED), text("the rewards that followed", 24, MUTED),
                        text("in every cell", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        base = max(baseline(s_, s_.text) for s_ in subs)
        for s_ in subs:
            sit(s_, base)
        proof = code(f"start cell: {EST[N_EP]:.3f} after 1,000 episodes · planned {EXACT[START]:.3f}", 24, INK,
                     t2c={r"0\.711": VALUE, r"0\.705": MUTED})
        proof.move_to([0, -1.3, 0])
        mascot = rover(0.7).move_to([0, -2.7, 0])

        self.say(
            "[To sum up]: [play], [sum], [average.]",
            m.AnimationGroup(self.new_header("08 · recap", "Play, sum, average"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Play whole episodes, sum the rewards back along each path], [and average them in every cell]: [no map required.]",
            m.Circumscribe(m.VGroup(words[0], words[1]), color=INK, buff=0.15),
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · next: must Rover wait?

    def next_step(self):
        self.next_section("9 · Must Rover wait?", notes=(
            "Monte Carlo updates only when the episode ends (the BLUE update at the end). Episode 9 (TD learning) updates "
            "after every step from the next estimate: the PINK sparks are its TD errors (the learning signal). The drive "
            "is Episode 6's lucky run (seed 11): five moves, −0.04 each, +1 at the charger."))
        xs = [-5.0 + 2.0 * i for i in range(6)]
        y = 0.35
        nodes = m.VGroup(*[m.Circle(radius=0.3).set_stroke(STATE, 3).set_fill(BG, 1).move_to([x, y, 0]) for x in xs[:5]])
        goal = icon("charger", 0.7).move_to([xs[5], y, 0])
        links = m.VGroup(*[m.Arrow([a + 0.32, y, 0], [b - 0.38, y, 0], buff=0, color=ACTION, stroke_width=5,
                                   max_tip_length_to_length_ratio=0.2) for a, b in zip(xs[:5], xs[1:])])
        rewards = m.VGroup(*[code(minus(f"{t.r:+.2f}"), 20, GOLD if t.r > 0 else RED).move_to([(a + b) / 2, y - 0.42, 0])
                             for t, a, b in zip(LUCKY, xs[:5], xs[1:])])
        bot = rover(0.5).move_to([xs[0], y + 1.05, 0])
        learn_end = m.VGroup(m.Circle(radius=0.55).set_stroke(VALUE, 4).move_to(goal),
                             text("Monte Carlo learns here", 22, VALUE, weight=m.BOLD).move_to([xs[5] - 0.6, y - 1.35, 0]))
        back_arc = m.ArcBetweenPoints([xs[5] - 0.62, y + 0.32, 0], [xs[0], y + 0.42, 0], angle=0.45).set_stroke(VALUE, 3)
        back_arc.add_tip(tip_length=0.18)
        back_arc.set_color(VALUE)
        sparks = [m.Flash([(a + b) / 2, y + 0.05, 0], color=SIGNAL, flash_radius=0.3, line_length=0.16, num_lines=10)
                  for a, b in zip(xs[:5], xs[1:])]
        ticks_ = m.VGroup(*[m.Dot([(a + b) / 2, y - 0.85, 0], radius=0.08, color=SIGNAL) for a, b in zip(xs[:5], xs[1:])])
        every = text("learn after every step?", 26, SIGNAL, weight=m.BOLD).move_to([0, -1.75, 0])
        ep9 = pill("Episode 9 · learning from a guess", VALUE, BLUE_FILL, 20, mono=True).next_to(every, m.DOWN, buff=0.4)

        rides = [bot.animate(run_time=0.4).move_to([x, y + 1.05, 0]) for x in xs[1:]]
        self.say(
            "[One catch]: [Rover learns nothing until the episode is over.]",
            m.AnimationGroup(self.new_header("09 · next", "Must Rover wait?"), m.FadeIn(nodes), m.FadeIn(goal),
                             m.FadeIn(links), m.FadeIn(rewards), m.FadeIn(bot, shift=0.1 * m.UP)),
            m.Succession(*rides, m.AnimationGroup(m.Create(learn_end[0]), m.FadeIn(learn_end[1]), m.Create(back_arc))),
        )
        self.say(
            "[What if it could learn after every single step], [from its own next guess?]",
            m.AnimationGroup(learn_end.animate.set_opacity(0.35), back_arc.animate.set_stroke(opacity=0.3),
                             m.LaggedStart(*sparks, lag_ratio=0.25), m.FadeIn(ticks_), m.FadeIn(every, shift=0.1 * m.UP)),
            m.FadeIn(ep9, shift=0.1 * m.UP),
        )
        self.wait(0.5)
        self.clear()
