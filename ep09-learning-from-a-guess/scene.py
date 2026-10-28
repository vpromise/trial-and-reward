"""Trial and Reward · Episode 9 · Learning from a Guess (Module 3 · Learning from Experience)

Update a guess from the next guess, without waiting for the end. The running example is the five-state random walk
(Sutton & Barto Example 6.2: tiles A–E, start in the middle, a fair coin each step, +1 for falling off the right end,
true values 1/6 … 5/6), with Rover walking it. Episode 8's Monte Carlo waits for the walk to end and pulls every
visited tile toward the return; TD(0) learns after every step from the gap between two neighbouring guesses, the
TD error δ = r + γV(s′) − V(s), drawn as a PINK spark. On the first seeded walk only A moves (0.5 → 0.45); the news
from the charger then creeps back one tile at a time. The rule, bootstrapping, a code excerpt; the race measured on
this Mac (100 runs × 100 walks: RMS error after 100 walks TD α 0.1 ≈ 0.057 vs MC α 0.02 ≈ 0.085, the picture of
S&B Example 6.2); bias vs variance from the middle tile; n-step returns as a dial; TD-Gammon (Tesauro 1995, Table 1).

Sources (facts dated October 2026):
- rl_reference.py: RANDOM_WALK_STATES / RANDOM_WALK_TRUE, random_walk_episode, td0_prediction, mc_prediction,
  rms_error, n_step_return, measured().
- research/experiments/classic.md §5 + classic.json random_walk (rms_curves, rms_at_100, fig_6_2_left,
  first_episode_seed0) and headline rw_rms_td0.1_at_100 / rw_rms_mc0.02_at_100: 100 runs × 100 walks, V starts at 0.5,
  MC constant-α every visit, the same walks for every learner.
- research/notes.md §9: Example 6.2 setup and true values, "a guess from a guess" (S&B §6.2), "no one has been able to
  prove mathematically that one method converges faster" (S&B §6.2), Exercise 6.3 (the first walk moves one value),
  n-step returns (S&B eq. 7.1, Fig. 7.2: 19-state walk, an intermediate n — about 4 — best), TD-Gammon (Tesauro 1992
  first published; Tesauro 1995 Table 1: −13/51, −7/38, −1/40 points; "best previous commercial program −0.66"; the
  1-point loss to Bill Robertie in 40 games). Appendix A: "−0.08 ppg" appears in no source and is not used.
- Sutton, R. S. (1988), Learning to predict by the methods of temporal differences, Machine Learning 3:9–44.

Illustrative choices (all numbers are still computed): the walks shown are seed 0's (the same walks as classic.json's
fig_6_2_left); Monte Carlo's single-walk illustration uses first-visit, step 0.1 (the measured race uses every-visit
constant-α MC); the walk-7 n-step targets use the TD values after walk 6 of the same run.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep09"

# ---------------------------------------------------------------------------- the numbers (all computed here)

S = RANDOM_WALK_STATES                                   # "ABCDE"
TRUE = [RANDOM_WALK_TRUE[s] for s in S]
assert S == "ABCDE" and [round(6 * v, 9) for v in TRUE] == [1, 2, 3, 4, 5]
V0 = {s: 0.5 for s in S}
ALPHA = 0.1
assert round(rms_error(V0), 4) == 0.2357                 # where every curve starts (notes §9)

# seed 0's walks: the same 100 walks as classic.json's fig_6_2_left
_rng = np.random.default_rng(0)
WALKS = [random_walk_episode(_rng) for _ in range(100)]
_, HIST = td0_prediction(WALKS, V0, alpha=ALPHA, record=True)
LEFT = measured("classic", "random_walk", "fig_6_2_left")
assert LEFT["alpha"] == ALPHA and LEFT["seed"] == 0
for _n, _vals in LEFT["values_after_episodes"].items():
    assert all(abs(HIST[int(_n)][s] - v) < 1e-3 for s, v in zip(S, _vals))
assert measured("classic", "random_walk", "first_episode_seed0")["states"] == list("CBCDEDCBA")


def path(walk):
    return "".join(t.s for t in walk)


# --- walk 1: nine steps, out to E and back, off the left end (return 0)
W1 = WALKS[0]
assert path(W1) == "CBCDEDCBA" and len(W1) == 9 and W1[-1].done and sum(t.r for t in W1) == 0.0


def td_trace(walk, V):
    """The TD errors of one walk, step by step (the same rule as td0_prediction), and the values after it."""
    V, deltas = dict(V), []
    for t in walk:
        target = t.r if t.done else t.r + V[t.s2]
        deltas.append(target - V[t.s])
        V[t.s] += ALPHA * (target - V[t.s])
    return deltas, V


D1, V1 = td_trace(W1, V0)
assert all(abs(d) < 1e-12 for d in D1[:-1]) and D1[-1] == -0.5                  # only the last step sparks
assert V1 == HIST[1] and round(V1["A"], 3) == 0.45 and all(V1[s] == 0.5 for s in "BCDE")
MC1 = mc_prediction([W1], V0, alpha=ALPHA, first_visit=True)
assert all(round(MC1[s], 3) == 0.45 for s in S)                                  # Monte Carlo moved all five

# --- walk 3: C D E, off the right end: the first +1
W3 = WALKS[2]
assert path(W3) == "CDE" and W3[-1].r == 1.0 and path(WALKS[1]) == "CBABABABA" and WALKS[1][-1].r == 0.0
V2 = HIST[2]
D3, V3 = td_trace(W3, V2)
assert D3[:2] == [0.0, 0.0] and D3[2] == 0.5 and round(V3["E"], 3) == 0.55 and V3 == HIST[3]

# --- walk 6: C → D (a tiny spark), D → E: E's better guess reaches D
W6 = WALKS[5]
assert path(W6)[:3] == "CDE" and len(W6) == 25 and W6[-1].r == 1.0
V5 = HIST[5]
assert round(V5["E"], 3) == 0.55 and V5["D"] == 0.5
D6, _ = td_trace(W6[:2], V5)
V6_MID = dict(V5)
for _t, _d in zip(W6[:2], D6):
    V6_MID[_t.s] += ALPHA * _d
assert abs(D6[0]) < 0.01 and round(D6[1], 3) == 0.05 and round(V6_MID["D"], 3) == 0.505
assert round(V6_MID["C"], 3) == round(V5["C"], 3) == 0.496                     # the tiny spark changes no digit

# --- the guesses settle, then drift: walks 10, 81, 100
RMS10, RMS81, RMS100 = (rms_error(HIST[k]) for k in (10, 81, 100))
assert (round(RMS10, 3), round(RMS81, 3), round(RMS100, 3)) == (0.126, 0.007, 0.066)
assert min(range(101), key=lambda k: rms_error(HIST[k])) == 81

# --- the race, measured on this Mac
RW = measured("classic", "random_walk")
CURVES = RW["rms_curves"]
RUNS, RW_EPISODES = RW["runs"], RW["episodes"]
TD_END = measured("classic", "headline", "rw_rms_td0.1_at_100")
MC_END = measured("classic", "headline", "rw_rms_mc0.02_at_100")
assert RUNS == 100 and RW_EPISODES == 100 and RW["v0"] == 0.5 and RW["mc"] == "constant-alpha, every visit"
assert round(TD_END, 3) == 0.057 and round(MC_END, 3) == 0.085
assert abs(CURVES["td_alpha0.1"]["y"][-1] - TD_END) < 1e-3 and abs(CURVES["mc_alpha0.02"]["y"][-1] - MC_END) < 1e-3
assert round(RW["rms_at_0"], 3) == round(rms_error(V0), 3) == 0.236
TD_KEYS = [k for k in CURVES if k.startswith("td")]
MC_KEYS = [k for k in CURVES if k.startswith("mc")]
assert len(TD_KEYS) == 3 and len(MC_KEYS) == 4
assert min(MC_KEYS, key=lambda k: CURVES[k]["y"][-1]) == "mc_alpha0.02"         # MC's best step size at 100
assert all(CURVES[t]["y"][k] < CURVES[c]["y"][k] for t in TD_KEYS for c in MC_KEYS for k in range(1, 101))

# --- bias and variance, from the middle tile (exact, from the true values)
P_RIGHT = RANDOM_WALK_TRUE["C"]
MC_SWING = (P_RIGHT * (1 - P_RIGHT)) ** 0.5                                      # outcomes 0 or 1
TD_TARGETS = (RANDOM_WALK_TRUE["B"], RANDOM_WALK_TRUE["D"])                       # one step, then a perfect guess
TD_SWING = (TD_TARGETS[1] - TD_TARGETS[0]) / 2
assert P_RIGHT == 0.5 and MC_SWING == 0.5 and abs(TD_SWING - 1 / 6) < 1e-12 and abs(TD_SWING / MC_SWING - 1 / 3) < 1e-12
assert abs(sum(TD_TARGETS) / 2 - P_RIGHT) < 1e-12                                 # the same average
EARLY_TARGET_B = 0.5 * (0 + V0["A"]) + 0.5 * (0 + V0["C"])                        # from B, all guesses at 0.5
BIAS_B = EARLY_TARGET_B - RANDOM_WALK_TRUE["B"]
assert EARLY_TARGET_B == 0.5 and round(BIAS_B, 2) == 0.17

# --- n-step returns on walk 7 (C D C D E, then +1), with the guesses after walk 6
W7 = WALKS[6]
assert path(W7) == "CDCDE" and [t.r for t in W7] == [0, 0, 0, 0, 1.0]
V6 = HIST[6]
W7_STATES = [t.s for t in W7]
W7_REWARDS = [t.r for t in W7]


def n_target(n):
    boot = 0.0 if n >= len(W7) else V6[W7_STATES[n]]
    return n_step_return(W7_REWARDS[:n], boot)


NSTEP = {n: n_target(n) for n in (1, 2, 4, 5)}
assert [round(NSTEP[n], 2) for n in (1, 2, 4, 5)] == [0.51, 0.50, 0.58, 1.00]

# --- TD-Gammon, Tesauro 1995 Table 1 (points won per game against world-class humans)
TDG = [("1.0", 300_000, -13, 51), ("2.0", 800_000, -7, 38), ("2.1", 1_500_000, -1, 40)]
TDG_PRINTED = [-0.25, -0.18, -0.02]
assert [int(100 * pts / games) / 100 for _, _, pts, games in TDG] == TDG_PRINTED  # Tesauro truncates to 2 places
EARLIER_BEST = -0.66                                                               # "best previous commercial program"

# ---------------------------------------------------------------------------- look

DICE = recolored("dice", EXPLORE, VIOLET_FILL, prefix=PREFIX)
CLOCK = recolored("clock", VALUE, BLUE_FILL, prefix=PREFIX)

BAR_C, BAR_W, BAR_H = (0.0, 0.7, 0.0), 7.0, 1.9
NUM_Y, TILE_Y, TILE = 1.98, -1.25, 0.9
SEQ_Y, LED1_Y, LED2_Y = -2.25, -2.85, -3.3
TD_C, MC_C = SIGNAL, VALUE        # TD's curves and targets in the spark's PINK, Monte Carlo's in the returns' BLUE


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def fmt3(v):
    return minus(f"{v:.3f}")


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


def spark_arc(p_from, p_to, strong=True, bend=1.1):
    """The TD error as a PINK arc from the next guess to the guess it corrects, bulging upward."""
    d = np.array(p_to) - np.array(p_from)
    angle = (-bend if d[0] > 0 else bend)
    arc = m.ArcBetweenPoints(np.array(p_from, dtype=float), np.array(p_to, dtype=float), angle=angle)
    arc.set_stroke(SIGNAL, 6 if strong else 3)
    arc.add_tip(tip_length=0.2 if strong else 0.14)
    arc.set_color(SIGNAL)
    return arc


def spark(arc, point, strong=True):
    """Draw the arc, then a flash where it lands."""
    return m.Succession(
        m.Create(arc, run_time=0.5),
        m.Flash(point, color=SIGNAL, line_length=0.24 if strong else 0.14, num_lines=12, flash_radius=0.3 if strong else 0.2,
                line_stroke_width=4 if strong else 2, run_time=0.6),
    )


def flicker(p_from, p_to):
    """A faint spark that changes nothing (δ = 0)."""
    return m.ShowPassingFlash(spark_arc(p_from, p_to, strong=False, bend=1.0).set_stroke(SIGNAL, 3, opacity=0.6),
                              time_width=0.8)


class Chain:
    """The random walk: five TEAL tiles A–E with Rover's guesses as BLUE bars above (numbers in a row on top), the
    true values as dashed marks, and the two ends: nothing on the left, the charger (+1) on the right."""

    def __init__(self, values):
        self.bars = Bars([values[s] for s in S], labels=list(S), top=1.0, bottom=0.0, width=BAR_W, height=BAR_H,
                         center=BAR_C, color=VALUE, gap=0.45, show_values=False, size=22)
        for lab in self.bars.labels:
            lab.set_color(STATE)
        self.xs = list(self.bars.xs)
        self.step = self.xs[1] - self.xs[0]
        self.nums = m.VGroup(*[self._num(i, values[s]) for i, s in enumerate(S)])
        self.tiles = m.VGroup(*[card(TILE, TILE, STATE, CARD, radius=0.12, stroke=2.5).move_to([x, TILE_Y, 0])
                                for x in self.xs])
        lx, rx = self.xs[0] - self.step, self.xs[-1] + self.step
        left_box = card(TILE, TILE, MUTED, CARD, radius=0.12, stroke=2.5).move_to([lx, TILE_Y, 0])
        self.left = m.VGroup(left_box, code("0", 26, MUTED, weight=m.BOLD).move_to(left_box))
        right_box = card(TILE, TILE, REWARD, GOLD_FILL, radius=0.12, stroke=3).move_to([rx, TILE_Y, 0])
        self.right = m.VGroup(right_box, icon("charger", 0.4).move_to(right_box).shift(0.13 * m.UP),
                              code("+1", 20, REWARD, weight=m.BOLD).move_to(right_box).shift(0.27 * m.DOWN))
        self.markers = m.VGroup(*[self.bars.marker(i, v, INK) for i, v in enumerate(TRUE)]).set_z_index(2)
        self.base = m.VGroup(self.bars.labels, self.tiles)          # what L1 shows

    def _num(self, i, v):
        return code(f"{v:.3f}", 22, VALUE).move_to([self.xs[i], NUM_Y, 0])

    def set(self, values, run_time=0.8):
        vals = [values[s] for s in S]
        return m.AnimationGroup(self.bars.set(vals, run_time=run_time),
                                *[m.Transform(self.nums[i], self._num(i, v), run_time=run_time) for i, v in enumerate(vals)])

    def top(self, i):
        """Just above guess i's number (the end of a spark)."""
        return np.array([self.xs[i], NUM_Y + 0.26, 0])

    def at(self, i):
        """Where Rover stands on tile i (−1 and 5 are the two ends)."""
        x = self.xs[0] + i * self.step
        return np.array([x, TILE_Y + 0.02, 0])

    def end_top(self, side):
        box = self.left[0] if side < 0 else self.right[0]
        return box.get_top() + np.array([0, 0.08, 0])

    def side_of(self, i, side):
        """Beside guess i's number (where a spark from an end lands)."""
        return np.array([self.xs[i] + side * 0.5, NUM_Y, 0])

    def everything(self):
        return m.VGroup(self.bars, self.nums, self.tiles, self.left, self.right, self.markers)


def legend():
    swatch = m.Rectangle(width=0.32, height=0.22).set_stroke(VALUE, 2).set_fill(BLUE_FILL, 1)
    dash = m.DashedLine([0, 0, 0], [0.34, 0, 0], dash_length=0.08).set_stroke(INK, 3)
    row1 = m.VGroup(swatch, text("Rover's guess", 20, MUTED)).arrange(m.RIGHT, buff=0.18)
    row2 = m.VGroup(dash, text("true value", 20, MUTED)).arrange(m.RIGHT, buff=0.18)
    g = m.VGroup(row1, row2).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
    return g.move_to([0, 0.75, 0]).align_to([4.4, 0, 0], m.LEFT)


def hops(bot, chain, walk, step=0.3, extra=None):
    """Rover follows a walk tile by tile (not the final fall); `extra(k)` adds an animation to hop k."""
    seq = []
    for k, t in enumerate(walk):
        if t.done:
            break
        j = S.index(t.s2)
        anim = bot.animate(run_time=step).move_to(chain.at(j))
        if extra is not None:
            anim = m.AnimationGroup(anim, extra(k), run_time=step)
        seq.append(anim)
    return m.Succession(*seq)


def fall(bot, chain, side, run_time=0.6):
    """Rover steps off an end and vanishes into it."""
    edge_x = chain.at(-1 if side < 0 else 5)[0] + (0.25 if side < 0 else -0.25)
    return m.Succession(bot.animate(run_time=run_time * 0.5).move_to([edge_x, TILE_Y + 0.02, 0]),
                        m.FadeOut(bot, shift=0.3 * side * m.RIGHT, scale=0.6, run_time=run_time * 0.5))


def board(width=5.4, height=2.7):
    """A backgammon board as a flat card: 24 points (triangles), a bar, a few checkers. No logos."""
    box = card(width, height, EDGE, CARD, radius=0.16, stroke=2.5)
    bar = m.Rectangle(width=0.28, height=height - 0.2).set_stroke(EDGE, 2).set_fill("#2A2238", 1).move_to(box)
    tri = m.VGroup()
    w = (width - 0.28 - 0.5) / 12
    x0 = box.get_left()[0] + 0.25
    for k in range(12):
        x = x0 + (k + 0.5) * w + (0.28 if k >= 6 else 0)
        col = mix(CARD, CORAL, 0.35) if k % 2 == 0 else mix(CARD, TEAL, 0.35)
        top_y, bot_y = box.get_top()[1] - 0.1, box.get_bottom()[1] + 0.1
        tri.add(m.Polygon([x - w / 2 + 0.02, top_y, 0], [x + w / 2 - 0.02, top_y, 0], [x, top_y - 0.43 * (height - 0.2), 0],
                          stroke_width=0, fill_color=col, fill_opacity=1))
        col2 = mix(CARD, TEAL, 0.35) if k % 2 == 0 else mix(CARD, CORAL, 0.35)
        tri.add(m.Polygon([x - w / 2 + 0.02, bot_y, 0], [x + w / 2 - 0.02, bot_y, 0], [x, bot_y + 0.43 * (height - 0.2), 0],
                          stroke_width=0, fill_color=col2, fill_opacity=1))
    checkers = m.VGroup()
    r = w * 0.42

    def stack(k, n, top, color):
        x = x0 + (k + 0.5) * w + (0.28 if k >= 6 else 0)
        for i in range(n):
            y = (box.get_top()[1] - 0.1 - r - 2 * r * i) if top else (box.get_bottom()[1] + 0.1 + r + 2 * r * i)
            checkers.add(m.Circle(radius=r * 0.95).set_stroke(BG, 1.5).set_fill(color, 1).move_to([x, y, 0]))

    for k, n, top, color in ((0, 2, True, INK), (5, 4, False, INK), (7, 3, False, INK), (11, 3, True, INK),
                             (11, 4, False, CORAL), (6, 4, True, CORAL), (4, 3, True, CORAL), (0, 2, False, CORAL)):
        stack(k, n, top, color)
    return m.VGroup(box, tri, bar, checkers)


# ---------------------------------------------------------------------------- the episode


class Ep09LearningFromAGuess(Episode):
    number = 9

    def series_intro(self):
        """The kit's intro card, unchanged, except that the title is spoken in lower case: the narrator's four seeded
        takes of "… Learning from a Guess." were all heard as "Guest" by both Whisper models (98%); new text, new takes."""
        import kit
        original = kit.spoken_title
        kit.spoken_title = lambda title: original(title).lower() if title == EPISODES[self.number][0] else original(title)
        try:
            super().series_intro()
        finally:
            kit.spoken_title = original

    def lesson(self):
        self.the_walk()
        self.the_wait()
        self.the_spark()
        self.the_rule()
        self.news_travels()
        self.the_race()
        self.bias_variance()
        self.n_steps()
        self.backgammon()
        self.recap()

    # ------------------------------------------------------------------ 1 · the walk

    def the_walk(self):
        self.next_section("1 · Five tiles and a coin", notes=(
            "Sutton & Barto (2018) Example 6.2, the random walk (rl_reference RANDOM_WALK_STATES, random_walk_episode): "
            "tiles A–E, every walk starts in C, each step left or right with probability 1/2, +1 for falling off the "
            "right end, 0 otherwise, γ = 1. A tile's value is the chance of ending on the right: 1/6 … 5/6 "
            "(RANDOM_WALK_TRUE). Rover's guesses start at 0.5 (as in the book and classic.json, v0 = 0.5)."))
        ch = Chain(V0)
        self.chain = ch
        bot = rover(0.38).move_to(ch.at(2)).set_z_index(3)
        self.bot = bot
        dice = icon(DICE, 0.5).move_to([0, 0.75, 0])
        a_left = m.CurvedArrow(np.array([ch.xs[2] - 0.15, -0.15, 0]), np.array([ch.xs[1] + 0.15, -0.15, 0]),
                               angle=0.9, color=EXPLORE, stroke_width=4, tip_length=0.18)
        a_right = m.CurvedArrow(np.array([ch.xs[2] + 0.15, -0.15, 0]), np.array([ch.xs[3] - 0.15, -0.15, 0]),
                                angle=-0.9, color=EXPLORE, stroke_width=4, tip_length=0.18)
        halves = m.VGroup(code("1/2", 20, EXPLORE, weight=m.BOLD).move_to([(ch.xs[1] + ch.xs[2]) / 2, 0.38, 0]),
                          code("1/2", 20, EXPLORE, weight=m.BOLD).move_to([(ch.xs[2] + ch.xs[3]) / 2, 0.38, 0]))
        coin = m.VGroup(a_left, a_right, halves)
        fracs = m.VGroup(*[code(f"{k + 1}/6", 20, INK).move_to([ch.xs[k], ch.bars.level(v) + 0.22, 0])
                           for k, v in enumerate(TRUE)])
        src = cite("Sutton & Barto, 2018, Example 6.2").move_to([0, -3.3, 0]).align_to([6.5, 0, 0], m.RIGHT)
        self.src = src
        lg = legend()
        self.lg = lg

        self.say(
            "[A tiny world from the textbook]: [five tiles in a row], [with Rover in the middle.]",
            m.AnimationGroup(self.new_header("01 · the walk", "Five tiles and a coin"), m.FadeIn(src)),
            m.LaggedStart(*[m.FadeIn(m.VGroup(t, lab), shift=0.15 * m.UP) for t, lab in zip(ch.tiles, ch.bars.labels)],
                          lag_ratio=0.15),
            m.FadeIn(bot, shift=0.2 * m.DOWN),
        )
        self.say(
            "[Each step, Rover flips a coin: left or right], [until it falls off an end.] "
            "[The right end pays plus one; the left, nothing.]",
            m.AnimationGroup(m.FadeIn(dice, scale=0.6), m.Create(a_left), m.Create(a_right), m.FadeIn(halves)),
            m.Succession(bot.animate(run_time=0.35).move_to(ch.at(3)), bot.animate(run_time=0.35).move_to(ch.at(2))),
            m.AnimationGroup(m.FadeIn(ch.right, shift=0.15 * m.LEFT), m.FadeIn(ch.left, shift=0.15 * m.RIGHT)),
        )
        self.say(
            "[So a tile's value is its chance of ending on the right]: [from one sixth up to five sixths.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(dice, coin)), m.Circumscribe(ch.right, color=REWARD, buff=0.06)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(mk) for mk in ch.markers], lag_ratio=0.15),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in fracs], lag_ratio=0.15)),
        )
        self.say(
            "[Rover doesn't know that: its guesses all start at one half.]",
            m.AnimationGroup(m.FadeOut(fracs), m.FadeIn(ch.bars.zero),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in ch.bars.bars], lag_ratio=0.12),
                             m.LaggedStart(*[m.FadeIn(n, shift=0.1 * m.UP) for n in ch.nums], lag_ratio=0.12),
                             m.FadeIn(lg), run_time=1.6),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ 2 · the wait

    def the_wait(self):
        self.next_section("2 · Monte Carlo waits", notes=(
            "Seed 0's first walk (random_walk_episode, default_rng(0); classic.json first_episode_seed0): C B C D E D C "
            "B A, then off the left end: 9 steps, return 0. Monte Carlo (episode 8) can only update after the end: "
            "mc_prediction([walk], V0, alpha=0.1, first_visit=True) moves every visited tile from 0.5 to 0.45 — all five, "
            "even E beside the charger (illustration; the measured race uses every-visit constant-α MC). Walk 6 of the "
            f"same seed is {len(W6)} steps long: {len(W6)} steps before Monte Carlo learns anything from it."))
        ch, bot = self.chain, self.bot
        ep8 = pill("ep 8 · Monte Carlo", VALUE, BLUE_FILL, 20).move_to([0, 1.98, 0]).align_to([6.5, 0, 0], m.RIGHT)
        row = token_row(list(path(W1)), color=STATE, size=22, gap=0.1)
        row.move_to([0.2, SEQ_Y, 0])
        row_lab = code("walk 1", 20, MUTED).next_to(row, m.LEFT, buff=0.35)
        self.row, self.row_lab = row, row_lab
        g_lab = code("return 0", 22, VALUE, weight=m.BOLD).next_to(row, m.RIGHT, buff=0.35)
        back = m.Line(row.get_right() + [0.1, -0.38, 0], row.get_left() + [-0.1, -0.38, 0]).set_stroke(VALUE, 7)
        mc_note = cite("Monte Carlo · first visit · step 0.1").move_to([0, LED1_Y, 0])

        dots = m.VGroup(*[m.Dot(radius=0.07, color=STATE) for _ in range(len(W6))]).arrange(m.RIGHT, buff=0.19)
        dots.move_to([-0.2, SEQ_Y, 0])
        clock = icon(CLOCK, 0.42).next_to(dots, m.LEFT, buff=0.35)
        long_lab = text(f"walk 6 · {len(W6)} steps, then one update", 22, MUTED).move_to([0, LED1_Y, 0])
        more = m.VGroup(*[m.Dot(radius=0.04, color=STATE) for _ in range(3)]).arrange(m.RIGHT, buff=0.08)
        never = m.VGroup(more, code("∞", 32, RED, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        never.next_to(dots, m.RIGHT, buff=0.25)
        never_lab = text("no end, no update", 22, RED).move_to([0, LED2_Y, 0])

        def add_chip(k):
            return m.FadeIn(row[k + 1], shift=0.1 * m.UP)

        self.say(
            "[Episode eight would play a whole walk first]: [nine steps, out to the far tile and back], "
            "[then off the left end.]",
            m.AnimationGroup(self.new_header("02 · the wait", "Monte Carlo waits"), m.FadeIn(ep8, shift=0.1 * m.DOWN),
                             m.FadeIn(row_lab), m.FadeIn(row[0], shift=0.1 * m.UP)),
            hops(bot, ch, W1, step=0.3, extra=add_chip),
            fall(bot, ch, -1),
        )
        self.say(
            "[Only then can Monte Carlo learn]: [the return was zero, so every tile on the path gets pulled down], "
            "[even the one beside the charger.]",
            m.AnimationGroup(m.FadeIn(g_lab, shift=0.1 * m.LEFT), m.ShowPassingFlash(back, time_width=0.6, run_time=1.0)),
            m.AnimationGroup(ch.set(MC1, run_time=1.0), m.FadeIn(mc_note)),
            m.Circumscribe(m.VGroup(ch.bars.bars[4], ch.nums[4]), color=VALUE, buff=0.08),
        )
        self.say(
            "[A longer walk means a longer wait], [and a task that never ends never learns at all.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(row, row_lab, g_lab, mc_note)),
                             m.LaggedStart(*[m.FadeIn(d) for d in dots], lag_ratio=0.08), m.FadeIn(clock),
                             m.FadeIn(long_lab), run_time=1.6),
            m.AnimationGroup(m.FadeIn(never, shift=0.1 * m.LEFT), m.FadeIn(never_lab, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.mc_gone = m.VGroup(ep8, dots, clock, long_lab, never, never_lab)

    # ------------------------------------------------------------------ 3 · the spark

    def the_spark(self):
        self.next_section("3 · The spark", notes=(
            "TD(0) on the same walk, same starting guesses 0.5, α = 0.1 (td0_prediction): after every step the target is "
            "r + γV(s′) (γ = 1, V(end) = 0) and the TD error δ = r + γV(s′) − V(s) (S&B eq. 6.5) is drawn as a PINK "
            "arc from the next guess to the guess it corrects. Steps 1–8 have δ = 0 (neighbours agree at 0.5); the "
            "last step, A → off the left end: δ = 0 + 0 − 0.5 = −0.5, V(A) = 0.5 + 0.1 × (−0.5) = 0.45 (notes §9 "
            "worked example 2, S&B Exercise 6.3). After walk 1, TD moved one guess, Monte Carlo five. Temporal-"
            "difference learning: Sutton (1988)."))
        ch, bot, row, row_lab = self.chain, self.bot, self.row, self.row_lab
        tdp = pill("TD · every step", SIGNAL, PINK_FILL, 20).move_to([0, 1.98, 0]).align_to([6.5, 0, 0], m.RIGHT)
        sutton = cite("temporal-difference learning · Sutton, 1988").move_to([0, 2.62, 0]).align_to([6.5, 0, 0], m.RIGHT)
        bot.move_to(ch.at(2))
        hl = m.SurroundingRectangle(row[0:2], buff=0.06, corner_radius=0.1).set_stroke(SIGNAL, 3)

        def hl_at(k):
            return m.SurroundingRectangle(row[k:k + 2], buff=0.06, corner_radius=0.1).set_stroke(SIGNAL, 3)

        before = colored([("before:", MUTED), ("V(C) = 0.500", VALUE)], 22)
        after = colored([("after:", MUTED), ("r", REWARD), ("+", INK), ("V(B)", VALUE), ("=", INK), ("0", REWARD),
                         ("+", INK), ("0.500", VALUE)], 22)
        m.VGroup(before, after).arrange(m.RIGHT, buff=1.0).move_to([0, LED1_Y, 0])
        delta0 = colored([("TD error", SIGNAL), ("δ = after − before =", INK), ("0.000", SIGNAL)], 22, weight=m.BOLD)
        delta0.move_to([0, LED2_Y, 0])
        arc0 = spark_arc(ch.top(1), ch.top(2), strong=False)

        steps = W1[1:8]                        # B→C … B→A: seven hops, every δ zero

        def flick(k):
            t = steps[k]
            i, j = S.index(t.s), S.index(t.s2)
            return m.AnimationGroup(flicker(ch.top(j), ch.top(i)), m.Transform(hl, hl_at(k + 1)))

        before_a = colored([("before:", MUTED), ("V(A) = 0.500", VALUE)], 22)
        after_a = colored([("after:", MUTED), ("r", REWARD), ("+", INK), ("V(end)", VALUE), ("=", INK), ("0", REWARD),
                           ("+", INK), ("0", VALUE)], 22)
        m.VGroup(before_a, after_a).arrange(m.RIGHT, buff=1.0).move_to([0, LED1_Y, 0])
        delta_a = colored([("TD error", SIGNAL), ("δ =", INK), ("0", REWARD), ("+", INK), ("0", VALUE), ("−", INK),
                           ("0.500", VALUE), ("=", INK), (minus(f"{D1[-1]:.3f}"), SIGNAL)], 22, weight=m.BOLD)
        delta_a.move_to([0, LED2_Y, 0])
        update_a = colored([("V(A) ←", VALUE), ("0.500", VALUE), ("+", INK), ("0.1", INK), ("×", INK),
                            (minus(f"({D1[-1]:.3f})"), SIGNAL), ("=", INK), (f"{V1['A']:.3f}", VALUE)], 22)
        update_a.move_to([0, LED1_Y, 0])
        arc_a = spark_arc(ch.end_top(-1), ch.side_of(0, -1), strong=True, bend=0.6)
        tally = colored([("after walk 1:", MUTED), ("TD moved 1 guess", SIGNAL), ("·", MUTED),
                         ("Monte Carlo moved 5", VALUE)], 22)
        tally.move_to([0, LED2_Y + 0.2, 0])

        self.say(
            "[Temporal-difference learning doesn't wait.] [Same walk, same guesses], [but Rover learns after every step.]",
            m.AnimationGroup(self.new_header("03 · the spark", "Learn from the next guess"),
                             m.FadeOut(m.VGroup(self.mc_gone, self.src)), m.FadeIn(sutton)),
            m.AnimationGroup(ch.set(V0), m.FadeIn(row_lab), m.FadeIn(row, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.DOWN), m.FadeIn(tdp, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[First step: left from the middle.] [Before it, Rover guessed one half.] "
            "[After it: a reward of zero, plus the new tile's guess, one half.]",
            m.AnimationGroup(bot.animate(run_time=0.5).move_to(ch.at(1)), m.Create(hl)),
            m.AnimationGroup(m.Circumscribe(ch.nums[2], color=VALUE, buff=0.08), m.FadeIn(before, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(ch.nums[1], color=VALUE, buff=0.08), m.FadeIn(after, shift=0.1 * m.UP)),
        )
        self.say(
            "[The gap between them is the TD error], [the pink spark.] [Here it's zero: the guesses agree, nothing changes.]",
            m.FadeIn(delta0, shift=0.1 * m.UP),
            spark(arc0, ch.top(2), strong=False),
            m.Circumscribe(delta0[-5:], color=SIGNAL, buff=0.08),
        )
        self.say(
            "[Step after step, neighbours agree], [until Rover falls off the left end.]",
            m.Succession(m.FadeOut(m.VGroup(before, after, delta0, arc0), run_time=0.4),
                         hops(bot, ch, steps, step=0.32, extra=flick)),
            fall(bot, ch, -1),
        )
        self.say(
            "[Now the next guess is zero: the walk is over.] [The error is minus one half], "
            "[and that guess moves a tenth of the way, to zero point four five.]",
            m.AnimationGroup(m.Circumscribe(ch.left, color=MUTED, buff=0.06), m.FadeIn(before_a, shift=0.1 * m.UP),
                             m.FadeIn(after_a, shift=0.1 * m.UP)),
            m.AnimationGroup(spark(arc_a, ch.side_of(0, -1), strong=True), m.FadeIn(delta_a, shift=0.1 * m.UP)),
            m.AnimationGroup(ch.set(V1), m.FadeOut(m.VGroup(before_a, after_a)), m.FadeIn(update_a, shift=0.1 * m.UP)),
        )
        self.say(
            "[One walk, one guess moved]: [the tile next to real news.] [Monte Carlo moved all five.]",
            m.Circumscribe(m.VGroup(ch.bars.bars[0], ch.nums[0]), color=SIGNAL, buff=0.08),
            m.Circumscribe(ch.left, color=SIGNAL, buff=0.06),
            m.AnimationGroup(m.FadeOut(m.VGroup(update_a, delta_a, arc_a, hl)), m.FadeIn(tally, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · the rule

    def the_rule(self):
        self.next_section("4 · A guess from a guess", notes=(
            "TD(0), S&B eq. 6.2 and 6.5: δ = r + γV(s′) − V(s), V(s) ← V(s) + α δ, with V(terminal) = 0. 'They learn "
            "a guess from a guess — they bootstrap' (S&B §6.2). Model-free like Monte Carlo (episode 8), bootstrapping "
            "like dynamic programming (episode 7), which needs the map p(s′, r | s, a). The code is an excerpt of "
            "rl_reference.td0_prediction (its names)."))
        f1 = m.MathTex(r"\delta", "=", "r", "+", r"\gamma", r"V(s')", "-", r"V(s)", font_size=60, color=INK)
        f1[0].set_color(SIGNAL)
        f1[2].set_color(REWARD)
        f1[5].set_color(VALUE)
        f1[7].set_color(VALUE)
        f1.move_to([-3.0, 1.95, 0])
        brace = m.Brace(f1[2:6], m.DOWN, buff=0.12, color=MUTED)
        target = text("target: reward + next guess", 22, INK, t2c={"reward": REWARD, "next guess": VALUE})
        target.next_to(brace, m.DOWN, buff=0.12)
        gamma = cite("γ = 1 here").next_to(target, m.RIGHT, buff=0.4)
        f2 = m.MathTex(r"V(s)", r"\leftarrow", r"V(s)", "+", r"\alpha", r"\delta", font_size=56, color=INK)
        f2[0].set_color(VALUE)
        f2[2].set_color(VALUE)
        f2[5].set_color(SIGNAL)
        f2.move_to([3.4, 1.95, 0])
        old = text("the old guess", 22, VALUE).next_to(f1[7], m.UP, buff=0.18)
        alpha_lab = cite("α = 0.1: a tenth of the way").next_to(f2, m.DOWN, buff=0.4)

        src = ("for s, a, r, s2, done in episode:\n"
               "    target = r if done else r + gamma * V[s2]\n"
               "    V[s] += alpha * (target - V[s])")
        block = code_block(src, size=20, accent=EDGE)
        paint(block[1], src, {r"\(target - V\[s\]\)": SIGNAL})
        block.move_to([0, -2.5, 0]).align_to([-6.5, 0, 0], m.LEFT)
        lines = m.VGroup(*block[1][len("for s, a, r, s2, done in episode:".replace(" ", "")):])
        where = code("rl_reference.py · td0_prediction", 20, MUTED).next_to(block, m.RIGHT, buff=0.45).shift(0.2 * m.UP)
        every = cite("run after every single step").next_to(where, m.DOWN, buff=0.15).align_to(where, m.LEFT)

        hx = [-6.5, -0.55, 1.25]
        heads = m.VGroup(code("needs a map", 20, FAINT).move_to([hx[1], 0.3, 0]),
                         code("waits", 20, FAINT).move_to([hx[2], 0.3, 0]))
        rows_src = [("ep 7 · value iteration", "yes", "no", STATE), ("ep 8 · Monte Carlo", "no", "yes", VALUE),
                    ("TD(0)", "no", "no", SIGNAL)]
        rows = []
        for k, (name, a, b, col) in enumerate(rows_src):
            y = -0.2 - 0.5 * k
            rows.append(m.VGroup(code(name, 22, col, weight=m.BOLD).move_to([0, y, 0]).align_to([hx[0], 0, 0], m.LEFT),
                                 code(a, 22, RED if a == "yes" else GREEN).move_to([hx[1], y, 0]),
                                 code(b, 22, RED if b == "yes" else GREEN).move_to([hx[2], y, 0])))
        boot = m.VGroup(text("bootstrapping", 26, INK, weight=m.BOLD), text("a guess from a guess", 22, MUTED))
        boot.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([0, -0.1, 0]).align_to([2.7, 0, 0], m.LEFT)
        sb = cite("Sutton & Barto, 2018, §6.2").next_to(boot, m.DOWN, buff=0.22).align_to(boot, m.LEFT)

        self.say(
            "[Here's the rule.] [The target: the reward, plus gamma times the next guess.] [Gamma is one here.]",
            self.new_header("04 · the rule", "A guess from a guess"),
            m.AnimationGroup(m.FadeIn(f1[2:6], shift=0.1 * m.UP), m.GrowFromCenter(brace), m.FadeIn(target)),
            m.FadeIn(gamma),
        )
        self.say(
            "[Subtract the old guess: that's the TD error, delta.] [Then move a fraction alpha of the way.]",
            m.AnimationGroup(m.FadeIn(f1[6:], shift=0.1 * m.UP), m.FadeIn(old, shift=0.1 * m.DOWN),
                             m.FadeIn(f1[:2], shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(f2, shift=0.1 * m.UP), m.FadeIn(alpha_lab)),
        )
        self.say(
            "[A guess from a guess: that's called bootstrapping.] [Like episode seven, but with no map], "
            "[like episode eight, but with no wait.]",
            m.AnimationGroup(m.FadeIn(boot, shift=0.1 * m.UP), m.FadeIn(sb), m.FadeIn(heads)),
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.1 * m.UP), m.FadeIn(rows[2][0], shift=0.1 * m.UP),
                             m.FadeIn(rows[2][1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.1 * m.UP), m.FadeIn(rows[2][2], shift=0.1 * m.UP)),
        )
        self.say(
            "[The code is just two lines], [run after every step.]",
            m.FadeIn(block, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(lines, color=SIGNAL, buff=0.08), m.FadeIn(where), m.FadeIn(every)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · the news travels back

    def news_travels(self):
        self.next_section("5 · The news travels back", notes=(
            "Same TD(0) run (seed 0, α = 0.1). Walk 3 (C D E, off the right end): δ = 1 + 0 − 0.5 = +0.5, V(E) = 0.55. "
            f"Walk 6 starts C D E: C → D gives δ = {D6[0]:+.4f} (no visible change); D → E: δ = 0 + 0.55 − 0.5 = +0.05, "
            f"V(D) = 0.505. After 10 walks RMS error {RMS10:.3f}; the lowest of the run is walk 81 ({RMS81:.3f}); after "
            f"100 walks {RMS100:.3f}: with a constant α the estimates keep fluctuating (S&B Example 6.2: 'the values "
            "fluctuate indefinitely'). Values after 0/1/10/100 walks match classic.json fig_6_2_left (asserted)."))
        ch = Chain(V2)
        bot = rover(0.38).move_to(ch.at(2)).set_z_index(3)
        counter = code("walk 3", 24, INK, weight=m.BOLD).move_to([0, 2.05, 0]).align_to([6.5, 0, 0], m.RIGHT)
        err = code(f"RMS error {rms_error(V2):.3f}", 20, MUTED).next_to(counter, m.DOWN, buff=0.14).align_to([6.5, 0, 0], m.RIGHT)

        def walk_lab(k):
            return code(f"walk {k}", 24, INK, weight=m.BOLD).move_to(counter, aligned_edge=m.RIGHT)

        def err_lab(k):
            return code(f"RMS error {rms_error(HIST[k]):.3f}", 20, MUTED).move_to(err, aligned_edge=m.RIGHT)

        arc_e = spark_arc(ch.end_top(1), ch.side_of(4, 1), strong=True, bend=0.6)
        delta_e = colored([("TD error", SIGNAL), ("δ =", INK), ("1", REWARD), ("+", INK), ("0", VALUE), ("−", INK),
                           ("0.500", VALUE), ("=", INK), (f"+{D3[2]:.3f}", SIGNAL)], 22, weight=m.BOLD)
        delta_e.move_to([0, LED1_Y, 0])
        arc_d = spark_arc(ch.top(4), ch.top(3), strong=True)
        delta_d = colored([("TD error", SIGNAL), ("δ =", INK), ("0", REWARD), ("+", INK), (fmt3(V5["E"]), VALUE),
                           ("−", INK), (fmt3(V5["D"]), VALUE), ("=", INK), (f"+{D6[1]:.3f}", SIGNAL)], 22, weight=m.BOLD)
        delta_d.move_to([0, LED1_Y, 0])
        creep = m.VMobject().set_points_as_corners([ch.top(4) + [0.3, 0.12, 0], ch.top(0) + [-0.3, 0.12, 0]])
        creep.set_stroke(SIGNAL, 6)
        drift = pill("step size fixed at 0.1", SIGNAL, PINK_FILL, 20).move_to([0, LED2_Y, 0])
        src = cite("Sutton & Barto, 2018, Example 6.2").move_to([0, -3.3, 0]).align_to([6.5, 0, 0], m.RIGHT)

        def run_to(ks, step=0.3):
            return m.Succession(*[m.AnimationGroup(ch.set(HIST[k], run_time=step), m.Transform(counter, walk_lab(k)),
                                                   m.Transform(err, err_lab(k)), run_time=step) for k in ks])

        self.say(
            "[Two walks later, Rover reaches the charger]: [a bright spark, plus one half.]",
            m.AnimationGroup(
                self.new_header("05 · a hundred walks", "The news travels back"),
                m.Succession(m.AnimationGroup(m.FadeIn(ch.everything()),
                                              m.FadeIn(counter), m.FadeIn(err), m.FadeIn(bot), run_time=0.8),
                             hops(bot, ch, W3, step=0.3), fall(bot, ch, 1))),
            m.AnimationGroup(spark(arc_e, ch.side_of(4, 1), strong=True), m.FadeIn(delta_e, shift=0.1 * m.UP),
                             m.Succession(m.Wait(0.5), ch.set(V3)), m.Transform(err, err_lab(3))),
        )
        bot.move_to(ch.at(2))
        self.say(
            "[Three walks on, its neighbour finds that better guess]: [a small spark, plus zero point zero five.] "
            "[The news creeps back, one tile at a time.]",
            m.Succession(m.AnimationGroup(m.FadeOut(m.VGroup(arc_e, delta_e)), ch.set(V5), m.Transform(counter, walk_lab(6)),
                                          m.Transform(err, err_lab(5)), m.FadeIn(bot, shift=0.15 * m.DOWN), run_time=0.6),
                         bot.animate(run_time=0.35).move_to(ch.at(3)),
                         m.AnimationGroup(bot.animate(run_time=0.35).move_to(ch.at(4)))),
            m.AnimationGroup(spark(arc_d, ch.top(3), strong=True), m.FadeIn(delta_d, shift=0.1 * m.UP),
                             m.Succession(m.Wait(0.5), ch.set(V6_MID))),
            m.ShowPassingFlash(creep, time_width=0.5, run_time=1.4),
        )
        self.say(
            "[After ten walks, the guesses lean toward the truth.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(arc_d, delta_d, bot)), run_to(range(7, 11), step=0.3), m.FadeIn(src)),
        )
        self.say(
            "[Around walk eighty, they almost touch it.] [But a fixed step size never settles]: "
            "[by walk one hundred, they've drifted again.]",
            run_to(list(range(15, 81, 5)) + [81], step=0.22),
            m.FadeIn(drift, shift=0.1 * m.UP),
            run_to([85, 90, 95, 100], step=0.35),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the race

    def the_race(self):
        self.next_section("6 · The race", notes=(
            f"Measured on this Mac (classic.md §5, classic.json random_walk): {RUNS} runs × {RW_EPISODES} walks, V starts at "
            f"0.5, the same walks for every learner; RMS error over the five tiles, averaged over runs. Start "
            f"{RW['rms_at_0']:.4f} for all. After 100 walks: TD α 0.1 {TD_END:.4f} (rw_rms_td0.1_at_100), MC α 0.02 "
            f"{MC_END:.4f} (rw_rms_mc0.02_at_100; MC's best α at 100). Faint: TD α 0.05 / 0.15 "
            f"({CURVES['td_alpha0.05']['y'][-1]:.3f} / {CURVES['td_alpha0.15']['y'][-1]:.3f}), MC α 0.01 / 0.03 / 0.04. "
            "From walk 1 on every TD curve is below every MC curve (asserted), as in S&B Example 6.2 (right graph; the "
            "book: 'The TD method was consistently better than the MC method on this task'). S&B §6.2: 'no one has been "
            "able to prove mathematically that one method converges faster than the other.' (classic.md calls the book's "
            "graph Fig. 6.2; in the 2nd edition Figure 6.2 is the batch version, so the screen cites Example 6.2.)"))
        plot = Plot((0, 106), (0, 0.265), width=8.4, height=4.4, center=(-1.0, -0.5, 0),
                    x_ticks=[0, 25, 50, 75, 100], y_ticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25], y_fmt="{:.2f}",
                    x_label="walks", y_label="RMS error of the five guesses")

        def curve(key, color, width=5, opacity=1.0):
            c = CURVES[key]
            return plot.line(c["x"], c["y"], color, width).set_stroke(opacity=opacity)

        td = curve("td_alpha0.1", TD_C)
        mc = curve("mc_alpha0.02", MC_C)
        faint = m.VGroup(*[curve(k, TD_C if k.startswith("td") else MC_C, 3, 0.35)
                           for k in TD_KEYS + MC_KEYS if k not in ("td_alpha0.1", "mc_alpha0.02")])
        start = plot.dot(0, RW["rms_at_0"], INK, radius=0.09)
        start_lab = code(f"{RW['rms_at_0']:.3f}", 20, INK, weight=m.BOLD).next_to(start, m.UR, buff=0.08)
        l_td = code(f"TD α = 0.1 · {TD_END:.3f}", 20, TD_C, weight=m.BOLD).next_to(plot.coords(100, TD_END), m.RIGHT, buff=0.18)
        l_mc = code(f"MC α = 0.02 · {MC_END:.3f}", 20, MC_C, weight=m.BOLD).next_to(plot.coords(100, MC_END), m.RIGHT, buff=0.18)
        e_td = plot.dot(100, TD_END, TD_C, radius=0.07)
        e_mc = plot.dot(100, MC_END, MC_C, radius=0.07)
        meas = cite(f"{RUNS} runs × {RW_EPISODES} walks · measured on this Mac").move_to([0, 2.3, 0]).align_to([6.5, 0, 0], m.RIGHT)
        faint_lab = cite("faint: other step sizes · TD 0.05, 0.15 · MC 0.01, 0.03, 0.04").move_to([0, 1.9, 0]).align_to([6.5, 0, 0], m.RIGHT)
        book = cite("same picture: Sutton & Barto, 2018, Example 6.2").move_to([0, 1.5, 0]).align_to([6.5, 0, 0], m.RIGHT)
        proof = text("TD faster in general? never proved · §6.2", 20, MUTED).move_to([0, 1.1, 0]).align_to([6.5, 0, 0], m.RIGHT)

        self.say(
            "[Now race the two, measured on this Mac]: [a hundred runs of a hundred walks.]",
            m.AnimationGroup(self.new_header("06 · the race", "Monte Carlo against TD"), m.FadeIn(plot)),
            m.FadeIn(meas, shift=0.1 * m.DOWN),
        )
        self.say(
            "[Both start with an error of about zero point two four.]",
            m.AnimationGroup(m.FadeIn(start, scale=0.5), m.FadeIn(start_lab, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Monte Carlo, at its best step size here, ends near zero point zero eight five.]",
            m.Succession(m.Create(mc, run_time=2.2),
                         m.AnimationGroup(m.FadeIn(e_mc, scale=0.5), m.FadeIn(l_mc, shift=0.1 * m.LEFT))),
        )
        self.say(
            "[With alpha at zero point one, TD is lower from the first walk], [and ends near zero point zero five seven.]",
            m.Create(td, run_time=2.4),
            m.AnimationGroup(m.FadeIn(e_td, scale=0.5), m.FadeIn(l_td, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[As in the textbook, every TD curve here stays below every Monte Carlo curve.] "
            "[Still, no one has proved TD is always faster.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(f) for f in faint], lag_ratio=0.2, run_time=2.0),
                             m.FadeIn(faint_lab), m.FadeIn(book, shift=0.1 * m.UP)),
            m.FadeIn(proof, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · bias and variance

    def bias_variance(self):
        self.next_section("7 · Bias and variance", notes=(
            "Targets from the middle tile C (exact, from RANDOM_WALK_TRUE): Monte Carlo's is the walk's outcome, 0 or 1 "
            f"with probability 1/2 each: mean {P_RIGHT}, standard deviation {MC_SWING}. TD's is r + V(next), with perfect "
            f"guesses 1/3 or 2/3: mean 0.5, standard deviation {TD_SWING:.3f}, a third of the swing. TD's bias: early on "
            f"every guess is 0.5, so from B its target is always 0.5 while V(B) = 1/3 (bias {BIAS_B:+.3f}); Monte Carlo's "
            "target from B is still 0 or 1 with mean 1/3 (unbiased). Notes §9: MC target unbiased, high variance; TD "
            "target biased (V(s′) is a guess), low variance."))
        x0, x1 = -1.6, 4.4

        def xpos(v):
            return x0 + (x1 - x0) * v

        names = {0: "0", 1 / 3: "1/3", 0.5: "1/2", 2 / 3: "2/3", 1: "1"}

        def number_line(y, shown):
            ln = m.Line([x0 - 0.2, y, 0], [x1 + 0.2, y, 0]).set_stroke(EDGE, 3)
            ticks = m.VGroup()
            for v, lab in names.items():
                ticks.add(m.Line([xpos(v), y - 0.08, 0], [xpos(v), y + 0.08, 0]).set_stroke(FAINT, 2))
                if v in shown:
                    ticks.add(code(lab, 20, MUTED).move_to([xpos(v), y - 0.36, 0]))
            return m.VGroup(ln, ticks)

        ys = [1.55, 0.05, -1.55]
        lx = -6.5
        labels = [
            m.VGroup(text("Monte Carlo's target", 24, MC_C, weight=m.BOLD), text("from the middle tile", 20, MUTED)),
            m.VGroup(text("TD's target", 24, TD_C, weight=m.BOLD), text("from the middle, perfect guesses", 20, MUTED)),
            m.VGroup(text("TD's target, early", 24, TD_C, weight=m.BOLD), text("from tile B, every guess 0.5", 20, MUTED)),
        ]
        for lab, y in zip(labels, ys):
            lab.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([0, y, 0]).align_to([lx, 0, 0], m.LEFT)
        lines = [number_line(ys[0], (0, 1)), number_line(ys[1], (1 / 3, 2 / 3)), number_line(ys[2], (0, 0.5, 1))]
        mc_dots = m.VGroup(*[m.Dot([xpos(v), ys[0], 0], radius=0.17, color=MC_C) for v in (0, 1)])
        td_dots = m.VGroup(*[m.Dot([xpos(v), ys[1], 0], radius=0.17, color=TD_C) for v in TD_TARGETS])
        early = m.Dot([xpos(EARLY_TARGET_B), ys[2], 0], radius=0.17, color=TD_C)

        def swing(a, b, y, color, label):
            arr = m.DoubleArrow([xpos(a), y + 0.42, 0], [xpos(b), y + 0.42, 0], buff=0.0, color=color, stroke_width=3,
                                tip_length=0.14, max_tip_length_to_length_ratio=0.2)
            lab = code(label, 22, color, weight=m.BOLD).move_to([0, y, 0]).align_to([4.9, 0, 0], m.LEFT)
            return m.VGroup(arr, lab)

        mc_swing = swing(0, 1, ys[0], MC_C, f"swing ± {MC_SWING:.2f}")
        td_swing = swing(TD_TARGETS[0], TD_TARGETS[1], ys[1], TD_C, f"swing ± {TD_SWING:.2f}")
        mean_line = m.DashedLine([xpos(0.5), ys[1] - 0.15, 0], [xpos(0.5), ys[0] + 0.72, 0], dash_length=0.08).set_stroke(INK, 2)
        mean_lab = code("average 1/2 = true value", 20, INK).move_to([xpos(0.5), ys[0] + 0.92, 0])
        truth_b = m.DashedLine([xpos(1 / 3), ys[2] - 0.2, 0], [xpos(1 / 3), ys[2] + 0.5, 0], dash_length=0.08).set_stroke(INK, 3)
        truth_lab = code("true 1/3", 20, INK).next_to(truth_b, m.UP, buff=0.06)
        bias_arr = m.Arrow([xpos(1 / 3), ys[2] + 0.32, 0], [xpos(0.5), ys[2] + 0.32, 0], buff=0.03, color=RED,
                           stroke_width=4, max_tip_length_to_length_ratio=0.3)
        bias_lab = code(f"bias +{BIAS_B:.2f}", 22, RED, weight=m.BOLD).move_to([0, ys[2], 0]).align_to([4.9, 0, 0], m.LEFT)
        verdict = m.VGroup(pill("Monte Carlo: unbiased, noisy", MC_C, BLUE_FILL, 22),
                           pill("TD: a little biased, calm", TD_C, PINK_FILL, 22)).arrange(m.RIGHT, buff=0.6)
        verdict.move_to([0, -3.1, 0])

        self.say(
            "[Why? Look at what each one aims for, from the middle tile.]",
            m.AnimationGroup(self.new_header("07 · bias and variance", "Why TD wins here"),
                             m.FadeIn(lines[0]), m.FadeIn(labels[0]), m.FadeIn(lines[1]), m.FadeIn(labels[1])),
        )
        self.say(
            "[Monte Carlo aims at the real outcome: zero or one.] [Right on average, but it swings by a half.]",
            m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in mc_dots], lag_ratio=0.4),
            m.AnimationGroup(m.Create(mean_line), m.FadeIn(mean_lab), m.GrowFromCenter(mc_swing)),
        )
        self.say(
            "[And TD aims at one step plus the next guess]: [with perfect guesses, a third or two thirds.] "
            "[Same average, a third of the swing.]",
            m.Circumscribe(labels[1][0], color=TD_C, buff=0.08),
            m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in td_dots], lag_ratio=0.4),
            m.GrowFromCenter(td_swing),
        )
        self.say(
            "[The catch: TD trusts its guesses.] [Early on, from the second tile, it aims at one half, not one third]: "
            "[that's bias.]",
            m.AnimationGroup(m.FadeIn(lines[2]), m.FadeIn(labels[2])),
            m.AnimationGroup(m.FadeIn(early, scale=0.5), m.Create(truth_b), m.FadeIn(truth_lab)),
            m.AnimationGroup(m.GrowArrow(bias_arr), m.FadeIn(bias_lab, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[So Monte Carlo is unbiased but noisy]; [TD is a little biased, but calm.]",
            m.FadeIn(verdict[0], shift=0.1 * m.UP),
            m.FadeIn(verdict[1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · n steps

    def n_steps(self):
        self.next_section("8 · n-step returns", notes=(
            "n-step return (S&B eq. 7.1): G = r1 + γ r2 + … + γ^(n−1) rn + γ^n V(s_n), n real rewards then one guess "
            "(rl_reference.n_step_return). Walk 7 of seed 0 (C D C D E, then +1), guesses after walk 6 of the TD run: "
            + ", ".join(f"n = {n}: {NSTEP[n]:.3f}" for n in (1, 2, 4)) + f"; to the end (Monte Carlo): {NSTEP[5]:.3f}. "
            "n = 1 is TD(0), n = ∞ Monte Carlo. S&B Example 7.1 / Fig. 7.2 (19-state walk, RMS over the first 10 "
            "episodes, 100 repetitions): 'methods with an intermediate value of n worked best' — n = 4 lowest (read from "
            "the figure, notes §9)."))
        xs = [-5.4, -3.4, -1.4, 0.6, 2.6]
        end_x, y = 4.6, 1.15
        nodes = m.VGroup()
        for x, s in zip(xs, W7_STATES):
            c = m.Circle(radius=0.36).set_stroke(STATE, 3).set_fill(CARD, 1).move_to([x, y, 0])
            nodes.add(m.VGroup(c, code(s, 24, INK, weight=m.BOLD).move_to(c)))
        goal_box = card(0.8, 0.8, REWARD, GOLD_FILL, radius=0.12, stroke=3).move_to([end_x, y, 0])
        goal = m.VGroup(goal_box, icon("charger", 0.46).move_to(goal_box))
        pts = xs + [end_x]
        links = m.VGroup(*[m.Arrow([pts[k] + 0.42, y, 0], [pts[k + 1] - 0.42, y, 0], buff=0, color=MUTED, stroke_width=3,
                                   max_tip_length_to_length_ratio=0.15) for k in range(5)])
        chips = m.VGroup(*[reward_chip(r, 20).move_to([(pts[k] + pts[k + 1]) / 2, y + 0.55, 0])
                           for k, r in enumerate(W7_REWARDS)])
        walk_lab = code("walk 7", 20, MUTED).move_to([0, y + 1.15, 0]).align_to([-6.5, 0, 0], m.LEFT)
        name_lab = colored([("n-step return:", SIGNAL), ("n real rewards, then one guess", MUTED)], 20)
        name_lab.move_to([0, y + 1.15, 0]).align_to([6.5, 0, 0], m.RIGHT)

        def bracket(n):
            right = pts[n] if n < 5 else end_x
            br = m.Brace(m.VGroup(m.Dot([pts[0], y - 0.4, 0], radius=0.01), m.Dot([right, y - 0.4, 0], radius=0.01)),
                         m.DOWN, buff=0.05, color=SIGNAL)
            if n < 5:
                ring = m.Circle(radius=0.48).set_stroke(VALUE, 4).move_to([pts[n], y, 0])
                lab = colored([(f"n = {n}:", SIGNAL), (f"{n} reward{'s' if n > 1 else ''} + guess at {W7_STATES[n]}", INK),
                               ("=", INK), (f"{NSTEP[n]:.2f}", VALUE)], 22)
            else:
                ring = m.SurroundingRectangle(goal_box, buff=0.08, corner_radius=0.14).set_stroke(REWARD, 4)
                lab = colored([("n = ∞:", SIGNAL), ("every reward to the end", INK), ("=", INK), (f"{NSTEP[5]:.2f}", VALUE)], 22)
            lab.next_to(br, m.DOWN, buff=0.15)
            return m.VGroup(br, ring, lab)

        b = {n: bracket(n) for n in (1, 2, 4, 5)}
        cur = b[1].copy()
        tx0, tx1, ty = -4.2, 4.2, -1.75
        track = m.Line([tx0, ty, 0], [tx1, ty, 0]).set_stroke(EDGE, 5)
        stops = {1: tx0, 2: tx0 + 2.1, 3: tx0 + 4.2, 4: tx0 + 6.3, 5: tx1}
        tick_labels = m.VGroup(*[code("∞" if n == 5 else str(n), 34 if n == 5 else 22, MUTED).move_to([x, ty - 0.42, 0])
                                 for n, x in stops.items()])
        ticks = m.VGroup(*[m.Line([x, ty - 0.12, 0], [x, ty + 0.12, 0]).set_stroke(FAINT, 3) for x in stops.values()])
        knob = m.Circle(radius=0.17).set_stroke(SIGNAL, 4).set_fill(PINK_FILL, 1).move_to([tx0, ty, 0]).set_z_index(2)
        left_end = m.VGroup(code("TD(0)", 22, TD_C, weight=m.BOLD), text("leans on guesses", 20, MUTED))
        left_end.arrange(m.DOWN, buff=0.08).move_to([tx0, ty - 1.0, 0])
        right_end = m.VGroup(code("Monte Carlo", 22, MC_C, weight=m.BOLD), text("waits for the end", 20, MUTED))
        right_end.arrange(m.DOWN, buff=0.08).move_to([tx1, ty - 1.0, 0])
        dial_lab = code("n", 24, SIGNAL, weight=m.BOLD).next_to(track, m.LEFT, buff=0.35)
        fig = cite("19-tile walk: about 4 steps did best · Sutton & Barto, 2018, Fig. 7.2").move_to([0, -3.35, 0])

        def go(n, run_time=0.8):
            return m.AnimationGroup(m.Transform(cur, b[n].copy()), knob.animate.move_to([stops[n], ty, 0]), run_time=run_time)

        self.say(
            "[You don't have to pick a side.] [Take a few real rewards, then one guess]: [a multi-step return.]",
            m.AnimationGroup(self.new_header("08 · n steps", "A dial between them"), m.FadeIn(nodes), m.FadeIn(goal),
                             m.FadeIn(links), m.FadeIn(chips), m.FadeIn(walk_lab)),
            m.AnimationGroup(m.FadeIn(track), m.FadeIn(ticks), m.FadeIn(tick_labels), m.FadeIn(dial_lab),
                             m.FadeIn(knob, scale=0.5), m.FadeIn(cur), m.FadeIn(name_lab, shift=0.1 * m.DOWN)),
            m.Succession(go(2), go(4)),
        )
        self.say(
            "[One step is TD; all the way to the end is Monte Carlo]; [in between, a dial.]",
            m.Succession(go(1), m.FadeIn(left_end, shift=0.1 * m.UP, run_time=0.4), go(5),
                         m.FadeIn(right_end, shift=0.1 * m.UP, run_time=0.4)),
            m.Succession(go(2, 0.6), go(4, 0.6)),
        )
        self.say(
            "[On a longer walk, the textbook found a setting in the middle worked best.]",
            m.AnimationGroup(m.FadeIn(fig, shift=0.1 * m.UP), m.Circumscribe(knob, color=SIGNAL, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · backgammon

    def backgammon(self):
        self.next_section("9 · TD-Gammon", notes=(
            "TD-Gammon (Gerald Tesauro, IBM): first published 1992 (notes §1); a neural-network value function trained by "
            "TD(λ) self-play (Tesauro 1995, S&B §16.1). Tesauro 1995 Table 1, against world-class humans: TD-Gammon 1.0, "
            "300,000 training games, −13 points in 51 games (−0.25 per game); 2.0, 800,000, −7 in 38 (−0.18); 2.1, "
            "1,500,000, −1 in 40 against Bill Robertie (−0.02): 'Robertie actually trailed the entire session, and only in "
            "the very last game was he able to pull ahead for an extremely narrow 1-point victory.' The best previous "
            "commercial program: −0.66 points per game on this scale (Tesauro 1995). The '−0.08 ppg' sometimes quoted "
            "appears in neither Tesauro 1995 nor S&B (notes Appendix A) and is not used."))
        bd = board(4.6, 2.3).move_to([-4.1, 1.2, 0])
        dice = icon(DICE, 0.42).next_to(bd, m.DOWN, buff=0.22).align_to(bd, m.LEFT)
        name = code("TD-Gammon · Tesauro, IBM · 1992", 20, MUTED).next_to(dice, m.RIGHT, buff=0.25)
        net = NeuralNet(layers=(6, 5, 1), width=1.1, height=1.3, center=(-0.95, 1.2, 0), color=VALUE, radius=0.07)
        g2 = pill("guess after the move", VALUE, BLUE_FILL, 20).move_to([-2.2, -2.45, 0])
        g1 = pill("guess now", VALUE, BLUE_FILL, 20).next_to(g2, m.LEFT, buff=0.6)
        arc = spark_arc(g2.get_top() + [-0.4, 0.08, 0], g1.get_top() + [0.25, 0.08, 0], strong=True, bend=1.0)
        tdl = code("TD(λ) self-play", 20, SIGNAL, weight=m.BOLD).next_to(arc, m.UP, buff=0.08)

        x_zero, scale = 6.2, 4.2 / 0.66
        head = text("points per game against world-class players", 22, INK).move_to([0, 2.05, 0]).align_to([0.35, 0, 0], m.LEFT)
        rows_data = [("best program before", None, EARLIER_BEST, MUTED)] + \
            [(f"TD-Gammon {v} · {g:,} games", None, p, REWARD) for (v, g, _, _), p in zip(TDG, TDG_PRINTED)]
        rows = []
        for k, (lab, _, val, col) in enumerate(rows_data):
            yk = 1.35 - 0.95 * k
            w = max(abs(val) * scale, 0.06)
            bar = m.Rectangle(width=w, height=0.34).set_stroke(col, 2).set_fill(FILLS.get(col, CARD), 1)
            bar.move_to([x_zero - w / 2, yk - 0.2, 0])
            name_k = text(lab, 20, MUTED if col == MUTED else INK).move_to([0, yk + 0.17, 0]).align_to([0.35, 0, 0], m.LEFT)
            num = code(minus(f"{val:+.2f}"), 22, col, weight=m.BOLD).next_to(bar, m.LEFT, buff=0.15)
            rows.append(m.VGroup(bar, name_k, num))
        zero = m.Line([x_zero, 1.55, 0], [x_zero, -1.95, 0]).set_stroke(INK, 2)
        even = code("even", 24, INK).next_to(zero, m.DOWN, buff=0.1)
        rob = text("vs Bill Robertie: lost by 1 point in 40 games", 20, REWARD).move_to([0, -2.45, 0]).align_to([0.35, 0, 0], m.LEFT)
        tes = cite("Tesauro, 1995, Table 1").move_to([0, -3.3, 0]).align_to([6.5, 0, 0], m.RIGHT)

        self.say(
            "[The first famous success was backgammon.] [In nineteen ninety-two, TD-Gammon], "
            "[a small neural network, learned by playing itself.]",
            m.AnimationGroup(self.new_header("09 · backgammon", "The first big win"), m.FadeIn(bd, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(dice, scale=0.6), m.FadeIn(name)),
            m.AnimationGroup(m.FadeIn(net), net.forward(VALUE)),
        )
        self.say(
            "[After every move, it nudged its guess toward the next one]: [the TD error was its teacher.]",
            m.AnimationGroup(m.FadeIn(g1, shift=0.1 * m.UP), m.FadeIn(g2, shift=0.1 * m.UP)),
            m.AnimationGroup(spark(arc, g1.get_top() + [0.25, 0.08, 0], strong=True), m.FadeIn(tdl)),
        )
        self.say(
            "[After three hundred thousand games, it lost a quarter point per game to world-class players]; "
            "[after one and a half million, two hundredths.]",
            m.AnimationGroup(m.FadeIn(head), m.Create(zero), m.FadeIn(even), m.FadeIn(rows[0]), m.FadeIn(tes),
                             m.FadeIn(rows[1][1:]), m.GrowFromEdge(rows[1][0], m.RIGHT)),
            m.AnimationGroup(m.FadeIn(rows[2][1:]), m.GrowFromEdge(rows[2][0], m.RIGHT),
                             m.FadeIn(rows[3][1:]), m.GrowFromEdge(rows[3][0], m.RIGHT)),
        )
        self.say(
            "[In a forty-game match, a top player beat it by a single point.]",
            m.AnimationGroup(m.FadeIn(rob, shift=0.1 * m.UP), m.Circumscribe(rows[3], color=REWARD, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 10 · recap

    def recap(self):
        self.next_section("10 · Recap", notes=(
            "Step (one move: the reward and the next tile's guess), compare (the TD error δ = r + γV(s′) − V(s), PINK), "
            f"nudge (V(s) moves a fraction α of the way). The race, measured here: RMS after 100 walks TD α 0.1 {TD_END:.3f} "
            f"vs MC α 0.02 {MC_END:.3f} ({RUNS} runs). Episode 10: TD for actions (Q-learning and SARSA)."))
        halo = glow(SIGNAL, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Step", 70, STATE), title_text("Compare", 70, SIGNAL), title_text("Nudge", 70, VALUE))
        words.arrange(m.RIGHT, buff=1.4).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text("reward + next guess", 24, MUTED), text("the TD error δ", 24, MUTED),
                        text("a fraction α of the way", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        base = max(s_.get_center()[1] for s_ in subs)
        for s_ in subs:
            s_.set_y(base)
        proof = colored([("random walk, 100 walks:", MUTED), (f"TD {TD_END:.3f}", TD_C), ("·", MUTED),
                         (f"Monte Carlo {MC_END:.3f}", MC_C), ("· measured here", MUTED)], 24)
        proof.move_to([0, -1.2, 0])
        bot = rover(0.6).move_to([-1.9, -2.6, 0])
        nxt = pill("episode 10 · TD for actions", ACTION, CORAL_FILL, 22).next_to(bot, m.RIGHT, buff=0.45)

        self.say(
            "[To sum up]: [take one step], [compare the new guess with the old], [and nudge.]",
            m.AnimationGroup(self.new_header("10 · recap", "Step, compare, nudge"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[On the random walk, that beat Monte Carlo]: [zero point zero five seven against zero point zero eight five.]",
            m.Circumscribe(words[1], color=SIGNAL, buff=0.12),
            m.FadeIn(proof, shift=0.1 * m.UP),
        )
        self.say(
            "[So far, Rover only judged tiles on a walk it didn't choose.] [Next, TD learns which move to make.]",
            m.FadeIn(bot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()
