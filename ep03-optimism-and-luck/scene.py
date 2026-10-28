"""Trial and Reward · Episode 3 · Optimism and Luck (Module 1 · Trial and Reward)

Two smarter ways to explore than episode 2's coin flip. Optimism: UCB1 scores every machine by its estimate plus a
bonus sqrt(2 ln t / N) — a confidence bar around each estimate — and pulls the highest TOP (a real seeded run on
Rover's five machines: pull 20, then pull by pull, 100, 1,000). Luck: Thompson sampling keeps a Beta(1 + wins,
1 + losses) belief per machine, draws one guess from each and plays the highest draw (the real seeded run's own
draws); the curves sharpen. Then the measured comparison on Rover's machines (2,000 runs x 1,000 pulls): Thompson
lowest, epsilon-greedy next, UCB1 LAST — told honestly (machines 4 and 5 are close; UCB1 overtakes epsilon-greedy
only at pull 12,103; logarithmic regret is a long-run promise) — and the 10-armed testbed, where UCB (c = 2) does
beat epsilon-greedy, with its spike at step 11.

Sources (facts dated October 2026):
- rl_reference.py: ROVER_ARMS, BernoulliBandit, ucb_scores, beta_posterior, beta_pdf, run_bandit, argmax_random.
- research/experiments/classic.md / classic.json (measured on this Mac, Apple M4 Max): rover_arms (methods: regret
  curves, regret_at, pct_best_last100; long_horizon: regret curves to 20,000 pulls, ucb1_overtakes_eps_at_pull;
  single_runs seed 0: every pull's arm and reward) and testbed (avg_reward curves, pct_optimal_last100, spike).
- research/notes.md §3 (UCB1 = Q + sqrt(2 ln t / N), Auer, Cesa-Bianchi & Fischer 2002, Fig. 1; Beta(1 + wins,
  1 + losses); Thompson 1933; Chapelle & Li 2011: Thompson beat UCB in their simulations and is robust to delayed
  feedback), §2 (Lai & Robbins 1985: regret must grow at least logarithmically), Appendix A (say which c a curve
  uses: c = sqrt(2) on Rover's machines, c = 2 on the testbed), Appendix C (UCB1 and Beta formulas).

Illustrative: none of the data. The random lever pulls in the hook are a seeded animation (no rewards shown).
Belief curves are drawn scaled to the same peak height (shape only).

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

PREFIX = "ep03"

# ---------------------------------------------------------------------------- the numbers (computed or measured)

C_UCB1 = math.sqrt(2)                                   # UCB1 (Auer et al. 2002); S&B's testbed figure uses c = 2
RA = measured("classic", "rover_arms")
METHODS, LONG, SINGLE = RA["methods"], RA["long_horizon"], RA["single_runs"]
HEADLINE = measured("classic", "headline")
TB = measured("classic", "testbed")
RUNS, PULLS = RA["runs"], RA["steps"]

assert tuple(RA["arms"]) == tuple(ROVER_ARMS) == (0.25, 0.45, 0.30, 0.60, 0.50) and ROVER_BEST_ARM == 3
assert RUNS == 2000 and PULLS == 1000 and TB["runs"] == 2000 and TB["steps"] == 1000

# --- the hook: one epsilon-greedy run (seed 0), where its 1,000 pulls went
EPS_COUNTS = [int(c) for c in SINGLE["eps_0.1"]["final_counts"]]
assert EPS_COUNTS == [25, 31, 22, 894, 28]          # machine 1 (pays 25%) got about as many pulls as machine 5 (50%)

# --- optimism: the seeded UCB1 run, replayed pull by pull from its recorded arms and rewards
UCB_A, UCB_R = SINGLE["ucb1"]["actions"], SINGLE["ucb1"]["rewards"]
assert list(run_bandit(BernoulliBandit(), "ucb", steps=1000, seed=0).actions) == UCB_A   # the run is reproducible


def ucb_state(p):
    """After p pulls of the UCB1 run: pulls N, sample averages Q and the bonus for the next choice (t = p + 1)."""
    N, total = np.zeros(5), np.zeros(5)
    for a, r in zip(UCB_A[:p], UCB_R[:p]):
        N[a] += 1
        total[a] += r
    Q = np.where(N > 0, total / np.maximum(N, 1), 0.0)
    bonus = C_UCB1 * np.sqrt(math.log(p + 1) / np.maximum(N, 1))
    assert np.allclose(ucb_scores(Q, N, p + 1, c=C_UCB1), Q + bonus)
    return N, Q, bonus


for _p in ("20", "100", "1000"):
    _N, _Q, _ = ucb_state(int(_p))
    assert np.allclose(_Q, SINGLE["ucb1"]["at_pull"][_p]["estimates"], atol=1e-4)
    assert np.allclose(_N, SINGLE["ucb1"]["at_pull"][_p]["counts"])

N20, Q20, B20 = ucb_state(20)
TOP20 = Q20 + B20
assert N20[3] == 2 and Q20[3] == 0 and N20[2] == 5 and N20[4] == 5    # "paid nothing in two pulls", "five times"
assert int(np.argmax(TOP20)) == 3 and Q20[3] == Q20.min()               # worst estimate, highest top
assert UCB_A[20] == 3                                                   # "so Rover pulls machine four next"
STEPS = list(range(20, 26))                                             # pulls 21..26 shown one by one
for _p in STEPS + [26]:
    _N, _Q, _B = ucb_state(_p)
    assert UCB_A[_p] in np.flatnonzero(np.isclose(_Q + _B, (_Q + _B).max()))   # always the highest top (ties: random)
N100, Q100, B100 = ucb_state(100)
N1000, Q1000, B1000 = ucb_state(1000)
TOP100, TOP1000 = Q100 + B100, Q1000 + B1000
assert np.ptp(TOP100) < 0.05 and np.ptp(TOP1000) < 0.03                 # "nearly lined up", "about the same height"
assert np.ptp(B100) < np.ptp(B20) and B100.max() < B20.min()            # "every bar has shrunk"
assert N1000[3] == 467 and N1000[4] == 272 and int(np.argmin(B1000)) == 3   # "four hundred sixty-seven", "two seventy-two"

# --- luck: the seeded Thompson run, replayed with the same random stream to recover its draws
TS_A, TS_R = SINGLE["thompson"]["actions"], SINGLE["thompson"]["rewards"]


def thompson_replay(steps=1000, seed=0):
    """run_bandit's Thompson loop for Rover's machines, keeping every round's draws."""
    bandit, rng = BernoulliBandit(), np.random.default_rng(seed)
    N, wins, draws, actions = np.zeros(5), np.zeros(5), [], []
    for _ in range(steps):
        a_post, b_post = beta_posterior(wins, N - wins)
        sample = rng.beta(a_post, b_post)
        arm = argmax_random(sample, rng)
        r = bandit.pull(arm, rng)
        N[arm] += 1
        wins[arm] += r
        draws.append(sample)
        actions.append(arm)
    return draws, actions


DRAWS, _ts_actions = thompson_replay()
assert _ts_actions == TS_A == list(run_bandit(BernoulliBandit(), "thompson", steps=1000, seed=0).actions)


def ts_beliefs(p):
    """Beta(a, b) of every machine after p pulls of the Thompson run."""
    N, wins = np.zeros(5), np.zeros(5)
    for a, r in zip(TS_A[:p], TS_R[:p]):
        N[a] += 1
        wins[a] += r
    return beta_posterior(wins, N - wins)


for _p in ("10", "100", "1000"):
    _a, _b = ts_beliefs(int(_p))
    assert np.allclose(_a, SINGLE["thompson"]["at_pull"][_p]["beta_a"]) and np.allclose(_b, SINGLE["thompson"]["at_pull"][_p]["beta_b"])

A10, B10 = ts_beliefs(10)
assert (A10[3], B10[3]) == (4, 3)                                   # "won three times and lost twice": Beta(4, 3)
assert [r for a, r in zip(TS_A[:10], TS_R[:10]) if a == 3] == [1, 0, 1, 1, 0]   # notes §3 C: the same 1, 0, 1, 1, 0
DRAW11 = DRAWS[10]                                                   # the draws that chose pull 11
assert int(np.argmax(DRAW11)) == 4 and TS_A[10] == 4                 # machine five's wide belief drew highest
assert round(DRAW11[4], 2) == 0.70 and round(DRAW11[3], 2) == 0.40
GRID = np.linspace(0.0005, 0.9995, 241)


def prob_best(a, b, n=200_001):
    """P(machine i has the highest rate | Beta(a_i, b_i) beliefs), by numerical integration."""
    x = np.linspace(0, 1, n)[1:-1]
    dx = x[1] - x[0]
    pdfs = [beta_pdf(x, ai, bi) for ai, bi in zip(a, b)]
    cdfs = [np.cumsum(f) * dx for f in pdfs]
    out = []
    for i in range(len(a)):
        prod = pdfs[i].copy()
        for j in range(len(a)):
            if j != i:
                prod *= cdfs[j]
        out.append(float(prod.sum() * dx))
    return np.array(out)


PBEST10 = prob_best(A10, B10)
assert abs(PBEST10.sum() - 1) < 1e-3 and round(PBEST10[3], 2) == 0.51 and int(np.argmax(PBEST10)) == 3   # "about half"
A100, B100 = ts_beliefs(100)
A1000, B1000 = ts_beliefs(1000)
TS_N1000 = (A1000 + B1000 - 2).astype(int)
assert list(TS_N1000) == [9, 30, 16, 870, 75]                        # "eight hundred seventy", "machine one just nine"

# --- the measured comparison: 2,000 runs x 1,000 pulls, and the long horizon
REG = {k: METHODS[k]["regret"] for k in ("eps_0.1", "ucb1", "thompson")}
R_EPS, R_UCB, R_TS = (HEADLINE[f"rover_regret_{k}"] for k in ("eps0.1", "ucb1", "thompson"))
assert (R_EPS, R_UCB, R_TS) == (METHODS["eps_0.1"]["regret_at"]["1000"], METHODS["ucb1"]["regret_at"]["1000"],
                                METHODS["thompson"]["regret_at"]["1000"])
assert round(R_EPS) == 46 and round(R_TS) == 31 and round(R_UCB) == 80 and R_TS < R_EPS < R_UCB
PCT_BEST_TS = HEADLINE["rover_pct_best_thompson"]
assert round(PCT_BEST_TS) == 93
GAP = ROVER_ARMS[3] - ROVER_ARMS[4]
N_NEEDED = 2 * math.log(1000) / GAP ** 2                             # bonus sqrt(2 ln t / N) < gap  <=>  N > 2 ln t / gap^2
assert round(GAP, 2) == 0.10 and round(N_NEEDED, -2) == 1400 and N_NEEDED > 1000   # "about fourteen hundred"
LREG = LONG["regret_curves"]
OVERTAKE = HEADLINE["rover_ucb1_overtakes_eps_at_pull"]
assert OVERTAKE == LONG["ucb1_overtakes_eps_at_pull"] == 12103 and LONG["steps"] == 20000 and round(OVERTAKE, -3) == 12000   # "about twelve thousand pulls in"
_x = np.asarray(LREG["ucb1"]["x"])
_diff = np.asarray(LREG["ucb1"]["y"]) - np.asarray(LREG["eps_0.1"]["y"])
_cross = _x[np.flatnonzero((_diff[:-1] > 0) & (_diff[1:] <= 0))]
assert len(_cross) == 1 and abs(_cross[0] - OVERTAKE) < 300          # the sampled curves cross where the data says
Y_CROSS = float(np.interp(OVERTAKE, _x, LREG["ucb1"]["y"]))
LONG_AT = LONG["regret_at"]
assert all(LONG_AT["thompson"][k] < min(LONG_AT["eps_0.1"][k], LONG_AT["ucb1"][k]) for k in LONG_AT["thompson"])
assert round(LONG_AT["thompson"]["20000"]) == 57                     # "about fifty-seven coins"
_tx = np.asarray(REG["thompson"]["x"])
_t_low = np.asarray(REG["thompson"]["y"]) < np.minimum(REG["eps_0.1"]["y"], REG["ucb1"]["y"])
assert _t_low[_tx >= 45].all() and not _t_low[0]                      # "pulls ahead early" (a near-tie for ~44 pulls)
_lx = np.asarray(LREG["thompson"]["x"])
_lt_low = np.asarray(LREG["thompson"]["y"]) < np.minimum(LREG["eps_0.1"]["y"], LREG["ucb1"]["y"])
assert _lt_low[_lx >= 100].all()                                      # "lowest from a hundred pulls to twenty thousand"
assert all(METHODS["thompson"]["regret_at"][k] <= min(METHODS["eps_0.1"]["regret_at"][k], METHODS["ucb1"]["regret_at"][k])
           for k in ("100", "500", "1000"))                           # lowest at every horizon tested here

# --- the 10-armed testbed (Sutton & Barto §2.3; UCB with c = 2 as in their Fig. 2.4)
TBM = TB["methods"]
assert TBM["ucb_c2"]["params"]["c"] == 2.0 and TBM["eps_0.1"]["params"]["epsilon"] == 0.1
PCT_UCB, PCT_EPS = HEADLINE["testbed_ucb_pct_optimal"], HEADLINE["testbed_eps0.1_pct_optimal"]
REW_UCB, REW_EPS = HEADLINE["testbed_ucb_avg_reward"], HEADLINE["testbed_eps0.1_avg_reward"]
PCT_OPT = HEADLINE["testbed_optimistic_pct_optimal"]
BEST_REW = HEADLINE["testbed_best_possible_avg_reward"]
SPIKE = TBM["ucb_c2"]["spike"]
assert round(PCT_UCB) == 86 and round(PCT_EPS) == 80 and round(PCT_OPT, 1) == 85.6      # "eighty-six … eighty … 85.6"
assert round(REW_UCB, 2) == 1.49 and round(REW_EPS, 2) == 1.37 and round(BEST_REW, 2) == 1.54
assert SPIKE["step"] == 11 and TBM["ucb_c2"]["avg_reward"]["x"][10] == 11
_ucb_y = TBM["ucb_c2"]["avg_reward"]["y"]
assert _ucb_y[10] == SPIKE["avg_reward"] and _ucb_y[10] > max(_ucb_y[:10]) + 0.9 and _ucb_y[10] > max(_ucb_y[11:15])   # a spike
assert TBM["optimistic_q5"]["params"]["q0"] == 5.0

# ---------------------------------------------------------------------------- look

EPS_C, UCB_C, TS_C = INK, VALUE, EXPLORE      # the three explorers' curves (labelled on screen)
DICE = recolored("dice", EXPLORE, VIOLET_FILL, prefix=PREFIX)

# the UCB chart: five estimate bars with confidence bars, on the left; the rule on the right
CH_X, CH_W, CH_H, CH_Y0, CH_TOP = -2.1, 6.6, 3.6, -1.65, 2.0
RX = 4.45


def cite(s, size=20):
    return text(s, size, MUTED)


class UCBChart:
    """The seeded UCB1 run as five BLUE estimate bars with VIOLET confidence bars; `.state(p)` animates to pull p."""

    def __init__(self, p):
        N, Q, B = ucb_state(p)
        self.bars = Bars(Q, labels=[str(i + 1) for i in range(5)], top=CH_TOP, bottom=0, width=CH_W, height=CH_H,
                         center=(CH_X, CH_Y0 + CH_H / 2, 0), color=VALUE, gap=0.35, show_values=False, size=22)
        self.xs, self.bw = self.bars.xs, self.bars.bar_w
        left = CH_X - CH_W / 2 - 0.15
        self.axis = m.Line([left, CH_Y0, 0], [left, self.bars.level(CH_TOP) + 0.1, 0]).set_stroke(FAINT, 2.5)
        self.ticks = m.VGroup()
        for v in (0, 0.5, 1.0, 1.5, 2.0):
            y = self.bars.level(v)
            self.ticks.add(m.Line([left - 0.09, y, 0], [left, y, 0]).set_stroke(FAINT, 2),
                           code(f"{v:g}", 20, MUTED).next_to([left - 0.09, y, 0], m.LEFT, buff=0.12))
        self.n_title = code("N", 22, MUTED).move_to([left - 0.3, CH_Y0 - 0.85, 0])
        self.q_title = code("Q", 22, VALUE).move_to([left - 0.3, CH_Y0 - 1.35, 0])
        self.n_cells = m.VGroup(*[self._n(i, n) for i, n in enumerate(N)])
        self.q_cells = m.VGroup(*[self._q(i, q) for i, q in enumerate(Q)])
        self.whiskers = m.VGroup(*[self._whisker(i, q, b) for i, (q, b) in enumerate(zip(Q, B))])
        self.tops = m.VGroup(*[self._top(i, q + b) for i, (q, b) in enumerate(zip(Q, B))])
        self.frame = m.VGroup(self.bars.zero, self.bars.labels, self.axis, self.ticks, self.n_title, self.q_title, self.n_cells)

    def _n(self, i, n):
        return code(f"{n:.0f}", 22, MUTED).move_to([self.xs[i], CH_Y0 - 0.85, 0])

    def _q(self, i, q):
        return code(f"{q:.2f}", 22, VALUE).move_to([self.xs[i], CH_Y0 - 1.35, 0])

    def _whisker(self, i, q, b):
        lo, hi = max(0.0, q - b), min(CH_TOP, q + b)
        y_lo, y_hi = self.bars.level(lo), self.bars.level(hi)
        x = self.xs[i]
        band = m.Rectangle(width=self.bw * 0.42, height=max(y_hi - y_lo, 0.03)).move_to([x, (y_lo + y_hi) / 2, 0])
        band.set_stroke(EXPLORE, 2.5).set_fill(EXPLORE, 0.2)
        cap = m.Line([x - self.bw * 0.4, y_hi, 0], [x + self.bw * 0.4, y_hi, 0]).set_stroke(EXPLORE, 6)
        foot = m.Line([x - self.bw * 0.16, y_lo, 0], [x + self.bw * 0.16, y_lo, 0]).set_stroke(EXPLORE, 3)
        return m.VGroup(band, cap, foot).set_z_index(2)

    def _top(self, i, top):
        return code(f"{top:.2f}", 20, EXPLORE).move_to([self.xs[i], self.bars.level(min(top, CH_TOP)) + 0.24, 0])

    def top_y(self, i, top):
        return self.bars.level(min(top, CH_TOP))

    def state(self, p, run_time=0.8):
        N, Q, B = ucb_state(p)
        anims = [self.bars.set(Q, run_time=run_time)]
        anims += [m.Transform(self.whiskers[i], self._whisker(i, q, b)) for i, (q, b) in enumerate(zip(Q, B))]
        anims += [m.Transform(self.tops[i], self._top(i, q + b)) for i, (q, b) in enumerate(zip(Q, B))]
        anims += [m.Transform(self.n_cells[i], self._n(i, n)) for i, n in enumerate(N)]
        anims += [m.Transform(self.q_cells[i], self._q(i, q)) for i, q in enumerate(Q)]
        return m.AnimationGroup(*anims, run_time=run_time)

    def level_line(self, v, color=EXPLORE):
        y = self.bars.level(v)
        return m.DashedLine([CH_X - CH_W / 2, y, 0], [CH_X + CH_W / 2, y, 0], dash_length=0.1).set_stroke(color, 2)


def next_marker(chart, i, top):
    """A CORAL arrow pointing down at machine i's top: the next pull."""
    y = chart.top_y(i, top)
    x = chart.xs[i]
    arr = m.Arrow([x, y + 1.05, 0], [x, y + 0.42, 0], buff=0, color=ACTION, stroke_width=6,
                  max_tip_length_to_length_ratio=0.35)
    lab = code("next pull", 20, ACTION, weight=m.BOLD).next_to(arr, m.RIGHT, buff=0.12).shift(0.15 * m.UP)
    return m.VGroup(arr, lab)


def marker_center(chart, i, top, marker):
    """Where `marker` (from next_marker) must move so it points at machine i's top."""
    ref = next_marker(chart, i, top)
    return ref.get_center()


class BeliefPanels:
    """Five small plots, one Beta belief per machine (VIOLET area, scaled to the same peak height)."""

    XS = (-5.0, -2.5, 0.0, 2.5, 5.0)

    def __init__(self, y=0.05, width=1.8, height=1.7):
        self.plots = [Plot((0, 1), (0, 1.12), width=width, height=height, center=(x, y, 0), x_ticks=[0, 1], size=20)
                      for x in self.XS]
        self.shapes = m.VGroup(*[self.shape(i, 1, 1) for i in range(5)])
        self.axes = m.VGroup(*self.plots)
        bottom = min(p.get_bottom()[1] for p in self.plots)
        self.row = [bottom - 0.32 - 0.5 * k for k in range(3)]
        self.labels = m.VGroup(*[self.label(i, 1, 1) for i in range(5)])

    def norm(self, a, b):
        ys = beta_pdf(GRID, a, b)
        return ys / ys.max()

    def shape(self, i, a, b):
        ys = self.norm(a, b)
        plot = self.plots[i]
        return m.VGroup(plot.area(GRID, ys, EXPLORE, 0.3), plot.line(GRID, ys, EXPLORE, 3))

    def height_at(self, i, a, b, x):
        return float(np.interp(x, GRID, self.norm(a, b)))

    def label(self, i, a, b):
        return code(f"Beta({a:g}, {b:g})", 20, EXPLORE).move_to([self.XS[i], self.row[0], 0])

    def to(self, i, a, b):
        return m.AnimationGroup(m.Transform(self.shapes[i], self.shape(i, a, b)), m.Transform(self.labels[i], self.label(i, a, b)))

    def to_all(self, a, b, run_time=1.0):
        return m.AnimationGroup(*[self.to(i, ai, bi) for i, (ai, bi) in enumerate(zip(a, b))], run_time=run_time)


def regret_curve(plot, curve, color, upto=None, width=4):
    xs, ys = np.asarray(curve["x"], float), np.asarray(curve["y"], float)
    if upto is not None:
        keep = xs <= upto
        xs, ys = xs[keep], ys[keep]
    return plot.line(xs, ys, color, width)


# ---------------------------------------------------------------------------- the episode


class Ep03OptimismAndLuck(Episode):
    number = 3

    def lesson(self):
        self.problem()
        self.optimism()
        self.shrinking()
        self.luck()
        self.sharpen()
        self.thousand_pulls()
        self.long_run()
        self.testbed()
        self.recap()

    # ------------------------------------------------------------------ 1 · the problem

    def problem(self):
        self.next_section("1 · Exploring blindly", notes=(
            "Episode 2's epsilon-greedy (epsilon = 0.1) picks a random machine one pull in ten, whatever it already knows. "
            f"Where one seeded run's 1,000 pulls went (classic.json rover_arms.single_runs.eps_0.1, seed 0): {EPS_COUNTS}. "
            "Machine 1 (pays 25%) got about as many pulls as machine 5 (50%), the close rival. True rates: ROVER_ARMS. "
            "The five lever pulls during the first line are a seeded animation, not data."))
        xs = [-4.0, -2.0, 0.0, 2.0, 4.0]
        machines = m.VGroup()
        for i, x in enumerate(xs):
            sm = SlotMachine(str(i + 1), size=0.8)
            sm.shift(np.array([x, 1.35, 0]) - sm.cabinet.get_center())
            machines.add(sm)
        bot = rover(0.72).move_to([-5.85, 1.05, 0])
        dice = icon(DICE, 0.55)
        eps = pill("ε = 0.1 · one pull in ten is random", EXPLORE, VIOLET_FILL, 24, mono=True)
        eps_row = m.VGroup(dice, eps).arrange(m.RIGHT, buff=0.25).move_to([0, -1.2, 0])
        order = [int(i) for i in np.random.default_rng(3).integers(5, size=5)]
        pulls = [m.AnimationGroup(machines[i].pull(0.5), m.Circumscribe(machines[i].cabinet, color=EXPLORE, buff=0.06, run_time=0.5))
                 for i in order]

        counts = Bars(EPS_COUNTS, top=1000, bottom=0, width=9.1, height=1.9, center=(0, -1.75, 0), color=ACTION,
                      gap=0.9, fmt="{:.0f}", size=22)
        caption = cite("where 1,000 pulls went · one ε-greedy run (seed 0)").move_to([0, -3.2, 0])
        rates = m.VGroup(*[code(f"pays {p:.0%}", 22, REWARD).move_to([sm.cabinet.get_x(), 0.2, 0])
                           for sm, p in zip(machines, ROVER_ARMS)])
        bad = pill("known to be bad", RED, RED_FILL, 20).move_to([xs[0], -1.45, 0])

        hero = title_text("Two smarter ways to explore", 56).move_to([0, 1.5, 0])

        def idea(word, sub, x):
            box = card(5.0, 2.3, EXPLORE, CARD)
            body = m.VGroup(title_text(word, 52, EXPLORE), text(sub, 26, MUTED)).arrange(m.DOWN, buff=0.2)
            return m.VGroup(box, body.move_to(box)).move_to([x, -1.1, 0])

        opt = idea("Optimism", "trust the best case", -2.9)
        luck = idea("Luck", "sample your beliefs", 2.9)
        frame1 = m.VGroup(machines, bot, counts, caption, rates, bad)

        self.say(
            "[In episode two], [Rover learned to explore at random]: [one pull in ten goes to any machine at all.]",
            m.AnimationGroup(self.new_header("01 · the problem", "Exploring blindly"),
                             m.LaggedStart(*[m.FadeIn(sm, shift=0.2 * m.UP) for sm in machines], lag_ratio=0.12),
                             m.FadeIn(bot, shift=0.2 * m.RIGHT)),
            m.FadeIn(eps_row, shift=0.15 * m.UP),
            m.Succession(*pulls),
        )
        self.say(
            "[Here's where a thousand pulls went, in one run.] [Machine four, the best, got most of them.]",
            m.AnimationGroup(m.FadeOut(eps_row), m.FadeIn(counts.zero), m.FadeIn(caption),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in counts.bars], lag_ratio=0.15),
                             m.LaggedStart(*[m.FadeIn(n) for n in counts.numbers], lag_ratio=0.15)),
            m.Circumscribe(m.VGroup(counts.bars[3], counts.numbers[3]), color=ACTION, buff=0.08),
        )
        self.say(
            "[But machine one pays only a quarter of the time], [yet it got about as many pulls as machine five, the close rival.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.DOWN) for r in rates], lag_ratio=0.1),
                             m.Circumscribe(m.VGroup(rates[0], counts.numbers[0]), color=RED, buff=0.08)),
            m.AnimationGroup(m.Circumscribe(m.VGroup(rates[4], counts.numbers[4]), color=ACTION, buff=0.08)),
        )
        self.say(
            "[Random pulls ignore what Rover already knows.] [Today, two smarter ways to explore]: [optimism], [and luck.]",
            m.AnimationGroup(m.FadeIn(bad, shift=0.1 * m.DOWN), machines[0].pull(0.6)),
            m.AnimationGroup(m.FadeOut(frame1, shift=0.2 * m.UP), m.FadeIn(hero, shift=0.2 * m.UP)),
            m.FadeIn(opt, shift=0.2 * m.UP),
            m.FadeIn(luck, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · optimism

    def optimism(self):
        self.next_section("2 · Optimism: UCB1", notes=(
            "A real seeded run (run_bandit(BernoulliBandit(), 'ucb', seed=0), classic.json single_runs.ucb1), replayed from "
            f"its recorded arms and rewards. After 20 pulls: N = {N20.astype(int).tolist()}, Q = {np.round(Q20, 2).tolist()}, "
            f"tops Q + sqrt(2 ln 21 / N) = {np.round(TOP20, 2).tolist()} (ucb_scores, c = sqrt(2)). Machine 4 has the worst "
            "estimate (0 of 2) and the highest top, and the run pulls it next. UCB1: Auer, Cesa-Bianchi & Fischer 2002, "
            "Fig. 1 (notes §3). VIOLET = uncertainty, BLUE = estimates, CORAL = the action."))
        ch = UCBChart(20)
        self.ucb = ch
        counter = code("after 20 pulls", 32, INK, weight=m.BOLD).move_to([RX, 2.05, 0])
        run_cite = cite("one run · seed 0 · measured here").next_to(counter, m.DOWN, buff=0.15)
        self.counter, self.run_cite = counter, run_cite
        arrow4 = next_marker(ch, 3, TOP20[3])
        self.arrow = arrow4
        best_est = m.VGroup(ch.q_cells[2], ch.q_cells[4])

        name = m.VGroup(code("UCB1", 32, INK, weight=m.BOLD), text("upper confidence bound", 22, MUTED)).arrange(m.DOWN, buff=0.08)
        name.move_to([RX, 0.85, 0])
        formula = m.MathTex(r"\mathrm{top}", "=", "Q(a)", "+", r"\sqrt{\frac{2\ln t}{N(a)}}", font_size=46, color=INK)
        formula[2].set_color(VALUE)
        formula[4].set_color(EXPLORE)
        formula.move_to([RX, -0.45, 0])
        n_row = code("N(a): pulls of machine a", 20, MUTED)
        t_row = code("t: number of this pull", 20, MUTED)
        legend = m.VGroup(n_row, t_row).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([RX, -1.7, 0])
        auer = cite("Auer, Cesa-Bianchi & Fischer, 2002").move_to([RX, -2.6, 0])
        self.rule = m.VGroup(name, formula, legend, auer)

        self.say(
            "[The first idea is optimism.] [Here's Rover twenty pulls into a real run], [with its estimate of each machine.]",
            m.AnimationGroup(self.new_header("02 · optimism", "Trust the best case"), m.FadeIn(counter, shift=0.15 * m.UP),
                             m.FadeIn(run_cite)),
            m.FadeIn(ch.frame),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in ch.bars.bars], lag_ratio=0.12),
                             m.FadeIn(ch.q_cells), m.FadeIn(ch.q_title)),
        )
        self.say(
            "[Around each estimate goes a bar of uncertainty]: [tall for a machine pulled twice], [shorter for one pulled five times.]",
            m.LaggedStart(*[m.GrowFromCenter(w) for w in ch.whiskers], lag_ratio=0.15),
            m.Circumscribe(m.VGroup(ch.whiskers[3], ch.n_cells[3]), color=EXPLORE, buff=0.08),
            m.Circumscribe(m.VGroup(ch.whiskers[4], ch.n_cells[4]), color=EXPLORE, buff=0.08),
        )
        self.say(
            "[Machine four has paid nothing in its two pulls: the worst estimate.] [Yet its bar reaches the highest.]",
            m.Circumscribe(ch.q_cells[3], color=RED, buff=0.08),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in ch.tops], lag_ratio=0.1),
                             m.Indicate(ch.whiskers[3][1], color=EXPLORE, scale_factor=1.15)),
        )
        self.say(
            "[So Rover pulls machine four next]: [not the best estimate], [but the best case.]",
            m.AnimationGroup(m.GrowArrow(arrow4[0]), m.FadeIn(arrow4[1], shift=0.1 * m.LEFT)),
            m.Circumscribe(best_est, color=VALUE, buff=0.08),
            m.Circumscribe(ch.tops[3], color=ACTION, buff=0.08),
        )
        self.say(
            "[That rule is UCB one]: [each bar's top is the estimate], [plus a bonus.]",
            m.AnimationGroup(m.FadeIn(name, shift=0.15 * m.UP), m.FadeIn(auer)),
            m.FadeIn(formula[:3], shift=0.1 * m.UP),
            m.FadeIn(formula[3:], shift=0.1 * m.UP),
        )
        self.say(
            "[The bonus shrinks every time a machine is pulled], [and slowly grows while it's left alone.]",
            m.AnimationGroup(m.FadeIn(n_row, shift=0.1 * m.UP), m.Circumscribe(formula[4], color=EXPLORE, buff=0.08)),
            m.FadeIn(t_row, shift=0.1 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ 3 · the bars shrink

    def shrinking(self):
        self.next_section("3 · Bars that shrink", notes=(
            "The same seeded UCB1 run, pull by pull from pull 21 to 26 (the arrow always sits on the highest top; ties are "
            f"broken at random), then at pull 100 (tops {np.round(TOP100, 2).tolist()}) and pull 1,000 (N = "
            f"{N1000.astype(int).tolist()}, tops {np.round(TOP1000, 2).tolist()}): every top ends at about the same height, so "
            "UCB1 keeps checking every machine."))
        ch, arrow = self.ucb, self.arrow

        def counter_to(p):
            return m.Transform(self.counter, code(f"after {p} pulls", 32, INK, weight=m.BOLD).move_to(self.counter))

        def step(p):
            """Pull p + 1 happens (the arrow's machine), then the arrow moves to the next highest top."""
            N, Q, B = ucb_state(p + 1)
            nxt = UCB_A[p + 1]
            return m.Succession(
                m.AnimationGroup(ch.state(p + 1, run_time=0.45), counter_to(p + 1)),
                arrow.animate(run_time=0.3).move_to(marker_center(ch, nxt, Q[nxt] + B[nxt], arrow)),
            )

        level100 = ch.level_line(float(TOP100.mean()))
        level1000 = ch.level_line(float(TOP1000.mean()))
        self.say(
            "[Whichever top is highest gets pulled, and pulling it shrinks that bar], [so sooner or later, another machine gets its turn.]",
            m.AnimationGroup(self.new_header("03 · a real run", "Bars that shrink"), m.Succession(*[step(p) for p in STEPS[:3]])),
            m.Succession(*[step(p) for p in STEPS[3:]]),
        )
        self.say(
            "[By pull one hundred], [every bar has shrunk], [and the tops have nearly lined up.]",
            m.AnimationGroup(m.FadeOut(arrow), counter_to(100)),
            ch.state(100, run_time=1.2),
            m.Create(level100),
        )
        self.say(
            "[After a thousand pulls], [machine four has the narrowest bar], [and the most pulls: four hundred sixty-seven.]",
            m.AnimationGroup(m.FadeOut(level100), counter_to(1000), ch.state(1000, run_time=1.2)),
            m.Circumscribe(ch.whiskers[3], color=EXPLORE, buff=0.1),
            m.Circumscribe(ch.n_cells[3], color=INK, buff=0.08),
        )
        self.say(
            "[But every top still reaches about the same height], [so Rover keeps checking them all]: [machine five got two hundred seventy-two pulls.]",
            m.Create(level1000),
            m.LaggedStart(*[m.Indicate(w[1], color=EXPLORE, scale_factor=1.2) for w in ch.whiskers], lag_ratio=0.15),
            m.Circumscribe(ch.n_cells[4], color=INK, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · luck

    def luck(self):
        self.next_section("4 · Luck: Thompson sampling", notes=(
            "A real seeded run (run_bandit(BernoulliBandit(), 'thompson', seed=0)); its draws are recovered by replaying the "
            "same random stream (the replay's pulls equal the recorded ones). Beliefs Beta(1 + wins, 1 + losses) "
            "(beta_posterior), drawn with beta_pdf, scaled to the same peak height. First ten pulls: machines "
            f"{[a + 1 for a in TS_A[:10]]}, rewards {[int(r) for r in TS_R[:10]]}; machine 4 = 1, 0, 1, 1, 0 → Beta(4, 3) "
            f"(notes §3 C). Pull 11's draws: {np.round(DRAW11, 2).tolist()} → machine 5. Chance each machine is best under "
            f"the pull-10 beliefs (numerical integration): {np.round(PBEST10, 3).tolist()}. Thompson 1933; Chapelle & Li 2011."))
        panels = BeliefPanels()
        self.panels = panels
        machines = m.VGroup()
        for i, x in enumerate(panels.XS):
            sm = SlotMachine(str(i + 1), size=0.55)
            sm.shift(np.array([x, 2.0, 0]) - sm.cabinet.get_center())
            machines.add(sm)
        self.machines = machines
        bot = rover(0.45).move_to([-6.1, 1.85, 0])
        self.bot = bot
        formula = code("belief = Beta(1 + wins, 1 + losses)", 22, EXPLORE).move_to([0, -3.2, 0]).align_to([-6.2, 0, 0], m.LEFT)
        self.formula = formula
        name = m.VGroup(text("Thompson sampling", 24, INK, weight=m.BOLD), cite("Thompson, 1933"))
        name.arrange(m.RIGHT, buff=0.3).move_to([0, -3.2, 0]).align_to([6.2, 0, 0], m.RIGHT)
        self.ts_name = name

        steps = []
        wins, n = np.zeros(5), np.zeros(5)
        for a, r in zip(TS_A[:10], TS_R[:10]):
            n[a] += 1
            wins[a] += r
            chip_ = machines[a].payout(r, 20)
            steps.append(m.Succession(
                m.AnimationGroup(machines[a].pull(0.42), m.FadeIn(chip_, run_time=0.25), panels.to(a, 1 + wins[a], 1 + n[a] - wins[a]),
                                 run_time=0.42),
                m.FadeOut(chip_, run_time=0.16),
            ))

        marks, draw_labels = m.VGroup(), m.VGroup()
        for i, s in enumerate(DRAW11):
            plot = panels.plots[i]
            h = panels.height_at(i, A10[i], B10[i], s)
            stem = m.Line(plot.coords(s, 0), plot.coords(s, h)).set_stroke(INK, 3)
            dot = m.Dot(plot.coords(s, 0), radius=0.075, color=INK).set_z_index(3)
            marks.add(m.VGroup(stem, dot))
            draw_labels.add(code(f"draw {s:.2f}", 22, INK).move_to([panels.XS[i], panels.row[1], 0]))
        win = int(np.argmax(DRAW11))
        ring = m.Circle(radius=0.17).set_stroke(ACTION, 4).move_to(marks[win][1]).set_z_index(4)
        win_label = code(f"draw {DRAW11[win]:.2f}", 22, ACTION, weight=m.BOLD).move_to(draw_labels[win])
        chance = m.VGroup(*[code(f"P(best) {p:.0%}", 20, MUTED).move_to([x, panels.row[2], 0]) for x, p in zip(panels.XS, PBEST10)])
        self.luck_extras = m.VGroup(marks, draw_labels, ring, chance)

        self.say(
            "[The second idea is luck.] [Rover keeps a belief about each machine]: [a curve over every payout rate it might have.]",
            m.AnimationGroup(self.new_header("04 · luck", "Sample your beliefs"),
                             m.LaggedStart(*[m.FadeIn(sm, shift=0.15 * m.UP) for sm in machines], lag_ratio=0.1),
                             m.FadeIn(bot, shift=0.15 * m.RIGHT)),
            m.LaggedStart(*[m.FadeIn(p) for p in panels.plots], lag_ratio=0.1),
            m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.UP) for s in panels.shapes], lag_ratio=0.1),
        )
        self.say(
            "[Before any pulls, every curve is flat]: [all rates from zero to one look equally likely.]",
            m.LaggedStart(*[m.FadeIn(lab) for lab in panels.labels], lag_ratio=0.1),
            m.LaggedStart(*[m.Indicate(s, color=EXPLORE, scale_factor=1.05) for s in panels.shapes], lag_ratio=0.12),
        )
        self.say(
            "[Each win pushes a machine's curve to the right], [and each loss pushes it to the left.]",
            m.Succession(*steps[:5]),
            m.Succession(*steps[5:]),
        )
        self.say(
            "[After ten pulls, machine four has won three times and lost twice]: [its belief is Beta of four and three.]",
            m.Circumscribe(m.VGroup(panels.plots[3], panels.shapes[3]), color=EXPLORE, buff=0.1),
            m.AnimationGroup(m.Indicate(panels.labels[3], color=EXPLORE), m.FadeIn(formula, shift=0.1 * m.UP)),
        )
        self.say(
            "[To choose, Rover draws one random guess from each belief]: [machine five's wide curve drew highest], [so Rover plays machine five.]",
            m.LaggedStart(*[m.AnimationGroup(m.Create(mk[0]), m.FadeIn(mk[1]), m.FadeIn(lab)) for mk, lab in zip(marks, draw_labels)],
                          lag_ratio=0.2),
            m.AnimationGroup(m.Create(ring), m.Transform(draw_labels[win], win_label)),
            m.AnimationGroup(machines[win].pull(0.6), m.Circumscribe(machines[win].cabinet, color=ACTION, buff=0.06)),
        )
        self.say(
            "[That's Thompson sampling, from nineteen thirty-three]: [each machine is played about as often as it's likely to be the best.]",
            m.FadeIn(name, shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in chance], lag_ratio=0.12),
        )
        self.say(
            "[Machine four is the favourite, about half the time], [but every uncertain machine still gets its chances.]",
            m.Circumscribe(chance[3], color=EXPLORE, buff=0.08),
            m.LaggedStart(*[m.Indicate(chance[i], color=EXPLORE) for i in (1, 2, 4)], lag_ratio=0.25),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ 5 · the beliefs sharpen

    def sharpen(self):
        self.next_section("5 · Beliefs sharpen", notes=(
            f"The same seeded Thompson run. Pull 100: Beta a = {A100.astype(int).tolist()}, b = {B100.astype(int).tolist()}. "
            f"Pull 1,000: a = {A1000.astype(int).tolist()}, b = {B1000.astype(int).tolist()}; pulls per machine "
            f"{TS_N1000.tolist()} (the epsilon-greedy run of section 1 gave machine 1 {EPS_COUNTS[0]}). Gold dashed lines: "
            "the true payout rates ROVER_ARMS, hidden from Rover."))
        panels = self.panels
        golds = m.VGroup(*[panels.plots[i].vline(p, REWARD, 3) for i, p in enumerate(ROVER_ARMS)])
        gold_note = cite("dashed gold: each machine's true payout rate, hidden from Rover").move_to([0, panels.row[2], 0])
        pulls_row = m.VGroup(*[code(f"{n} pulls", 22, ACTION).move_to([x, panels.row[1], 0]) for x, n in zip(panels.XS, TS_N1000)])
        self.say(
            "[As the pulls add up, the curves sharpen.] [By pull one hundred], [the curve of machine four is already narrow.]",
            m.AnimationGroup(self.new_header("05 · luck, continued", "Beliefs that sharpen"),
                             m.FadeOut(self.luck_extras), m.FadeOut(self.ts_name)),
            panels.to_all(A100, B100, run_time=1.2),
            m.Circumscribe(m.VGroup(panels.plots[3], panels.shapes[3]), color=EXPLORE, buff=0.1),
        )
        self.say(
            "[After a thousand pulls], [machine four's curve sits right on its true payout rate, in gold], [while the bad machines stay vague.]",
            panels.to_all(A1000, B1000, run_time=1.2),
            m.AnimationGroup(m.LaggedStart(*[m.Create(g) for g in golds], lag_ratio=0.12), m.FadeIn(gold_note)),
            m.LaggedStart(*[m.Indicate(panels.shapes[i], color=EXPLORE, scale_factor=1.05) for i in (0, 2)], lag_ratio=0.3),
        )
        self.say(
            "[Rover stopped wasting pulls on them]: [machine four got eight hundred seventy], [and machine one just nine, where random exploring gave it twenty-five.]",
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in pulls_row], lag_ratio=0.12),
            m.Circumscribe(pulls_row[3], color=ACTION, buff=0.08),
            m.Circumscribe(pulls_row[0], color=ACTION, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · measured: 1,000 pulls

    def thousand_pulls(self):
        self.next_section("6 · Measured: a thousand pulls", notes=(
            f"Measured on this Mac (classic.md §2): {RUNS:,} runs x {PULLS:,} pulls on ROVER_ARMS, expected regret "
            f"(sum of 0.60 minus the pulled machine's rate), mean over runs. After 1,000 pulls: epsilon-greedy (0.1) {R_EPS}, "
            f"UCB1 (c = sqrt(2)) {R_UCB}, Thompson {R_TS}; Thompson pulls machine 4 {PCT_BEST_TS}% of pulls 901-1000. "
            f"Why UCB1 trails: its bonus sqrt(2 ln t / N) only drops below machine 5's gap of {GAP:.2f} after N > 2 ln t / gap^2 "
            f"= {N_NEEDED:,.0f} pulls of machine 5 at t = 1,000 (classic.md, 'about 1,400'). Honest finding, as in the notes."))
        plot = Plot((0, 1000), (0, 90), width=5.6, height=3.9, center=(-3.1, -0.25, 0), x_ticks=[0, 500, 1000],
                    y_ticks=[0, 20, 40, 60, 80], x_label="pulls", y_label="regret: coins lost")
        eps = regret_curve(plot, REG["eps_0.1"], EPS_C)
        ts = regret_curve(plot, REG["thompson"], TS_C)
        ucb = regret_curve(plot, REG["ucb1"], UCB_C)

        def end_label(s, v, color):
            return code(s, 20, color, weight=m.BOLD).next_to(plot.coords(1000, v), m.RIGHT, buff=0.18)

        l_eps = end_label(f"ε-greedy {R_EPS:.1f}", R_EPS, EPS_C)
        l_ts = end_label(f"Thompson {R_TS:.1f}", R_TS, TS_C)
        l_ucb = end_label(f"UCB1 {R_UCB:.1f}", R_UCB, UCB_C)
        src = m.VGroup(cite(f"{RUNS:,} runs × {PULLS:,} pulls · mean"), cite("measured on this Mac"))
        src.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([4.6, 2.2, 0]).align_to([2.75, 0, 0], m.LEFT)
        ts_best = text("Thompson, pulls 901–1,000:", 20, TS_C).next_to(src, m.DOWN, buff=0.3).align_to(src, m.LEFT)
        ts_best2 = code(f"{PCT_BEST_TS:.1f}% on machine 4", 22, TS_C, weight=m.BOLD).next_to(ts_best, m.DOWN, buff=0.12).align_to(src, m.LEFT)

        # why UCB1 trails: machines 4 and 5 are close
        why_box = card(3.75, 3.15, EDGE, CARD).move_to([4.5, -1.75, 0])
        why_title = text("why UCB1 keeps checking", 20, INK, weight=m.BOLD).move_to(why_box.get_top() + [0, -0.36, 0])
        rows = m.VGroup()
        for k in (3, 4):
            lab = code(f"machine {k + 1}", 20, MUTED)
            bar = m.Rectangle(width=1.6 * ROVER_ARMS[k], height=0.26).set_stroke(REWARD, 2).set_fill(GOLD_FILL, 1)
            val = code(f"{ROVER_ARMS[k]:.2f}", 20, REWARD)
            rows.add(m.VGroup(lab, bar, val).arrange(m.RIGHT, buff=0.18))
        rows.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).next_to(why_title, m.DOWN, buff=0.28).align_to(why_box, m.LEFT).shift(0.3 * m.RIGHT)
        rows[1][2].next_to(rows[1][1], m.RIGHT, buff=0.18)
        gap_lab = code(f"gap {GAP:.2f}", 20, INK).next_to(rows, m.DOWN, buff=0.16).align_to(rows, m.LEFT)
        ineq = m.MathTex(r"\sqrt{2\ln t \,/\, N} < 0.1", font_size=34, color=EXPLORE)
        ineq.next_to(gap_lab, m.DOWN, buff=0.2).align_to(rows, m.LEFT)
        need = code(f"N > {N_NEEDED:,.0f} at t = 1,000", 20, INK, weight=m.BOLD).next_to(ineq, m.DOWN, buff=0.16).align_to(rows, m.LEFT)
        why = m.VGroup(why_box, why_title, rows, gap_lab)

        self.say(
            "[So which explorer loses the fewest coins?] [Each one played Rover's machines for a thousand pulls, two thousand times, on this Mac.]",
            m.AnimationGroup(self.new_header("06 · measured", "A thousand pulls"), m.FadeIn(plot)),
            m.FadeIn(src, shift=0.1 * m.UP),
        )
        self.say(
            "[Epsilon-greedy, from episode two], [ends near forty-six coins of regret.]",
            m.Create(eps, run_time=1.6),
            m.FadeIn(l_eps, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Thompson sampling pulls ahead early and ends lowest]: [about thirty-one], [and by the end it plays machine four ninety-three percent of the time.]",
            m.Create(ts, run_time=1.6),
            m.FadeIn(l_ts, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeIn(ts_best, shift=0.1 * m.UP), m.FadeIn(ts_best2)),
        )
        self.say(
            "[And UCB one?] [About eighty]: [worse than exploring at random!]",
            m.Create(ucb, run_time=1.6),
            m.FadeIn(l_ucb, shift=0.1 * m.LEFT),
            m.Circumscribe(m.VGroup(l_ucb, l_eps), color=UCB_C, buff=0.1),
        )
        self.say(
            "[Here's why.] [Machines four and five are close]: [six in ten, against five in ten.]",
            m.FadeIn(why_box),
            m.AnimationGroup(m.FadeIn(why_title), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in rows], lag_ratio=0.3)),
            m.FadeIn(gap_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[To rule out machine five, the bonus must shrink below that gap]: [about fourteen hundred pulls of machine five], [more than the whole run.]",
            m.FadeIn(ineq, shift=0.1 * m.UP),
            m.FadeIn(need, shift=0.1 * m.UP),
            m.Circumscribe(need, color=UCB_C, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · the long run

    def long_run(self):
        self.next_section("7 · The long run", notes=(
            f"Measured on this Mac: {RUNS:,} runs x 20,000 pulls (classic.json rover_arms.long_horizon). Regret at "
            f"1,000 / 2,000 / 5,000 / 10,000 / 20,000: epsilon-greedy {list(LONG_AT['eps_0.1'].values())}, UCB1 "
            f"{list(LONG_AT['ucb1'].values())}, Thompson {list(LONG_AT['thompson'].values())}. UCB1 overtakes epsilon-greedy "
            f"for good at pull {OVERTAKE:,} (rover_ucb1_overtakes_eps_at_pull). Logarithmic regret: UCB1's guarantee holds for "
            "every n (Auer et al. 2002, Theorem 1); no consistent method grows slower than log T (Lai & Robbins 1985, read "
            "via Auer et al. 2002 and Ren & Zhang 2024, notes §2). Chapelle & Li (NeurIPS 2011) found Thompson sampling "
            "with lower regret than UCB in their simulations and robust to delayed feedback (notes §3)."))
        plot = Plot((0, 21000), (0, 420), width=8.2, height=3.9, center=(-1.75, -0.25, 0),
                    x_ticks=[0, 5000, 10000, 15000, 20000], y_ticks=[0, 100, 200, 300, 400], x_fmt="{:,.0f}",
                    x_label="pulls", y_label="regret: coins lost")
        eps = regret_curve(plot, LREG["eps_0.1"], EPS_C)
        ucb = regret_curve(plot, LREG["ucb1"], UCB_C)
        ts = regret_curve(plot, LREG["thompson"], TS_C)
        first = plot.vline(1000, FAINT, 2)
        first_lab = cite("1,000").next_to(plot.coords(1000, 420), m.RIGHT, buff=0.1)
        last = {k: LREG[k]["y"][-1] for k in LREG}

        def end_label(s, v, color):
            return code(s, 20, color, weight=m.BOLD).next_to(plot.coords(20000, v), m.RIGHT, buff=0.15)

        l_eps = end_label(f"ε-greedy {LONG_AT['eps_0.1']['20000']:.0f}", last["eps_0.1"], EPS_C)
        l_ucb = end_label(f"UCB1 {LONG_AT['ucb1']['20000']:.0f}", last["ucb1"], UCB_C)
        l_ts = end_label(f"Thompson {LONG_AT['thompson']['20000']:.0f}", last["thompson"], TS_C)
        cross = plot.dot(OVERTAKE, Y_CROSS, INK, radius=0.09)
        cross_line = m.DashedLine(plot.coords(OVERTAKE, 0), plot.coords(OVERTAKE, Y_CROSS), dash_length=0.08).set_stroke(INK, 2)
        cross_lab = code(f"pull {OVERTAKE:,}", 20, INK, weight=m.BOLD).next_to(cross, m.UP + m.LEFT, buff=0.12)
        log_lab = code("∝ log t", 22, UCB_C).next_to(plot.coords(17000, float(np.interp(17000, LREG["ucb1"]["x"], LREG["ucb1"]["y"]))),
                                                    m.DOWN + m.RIGHT, buff=0.05)
        lin_lab = code("straight line", 20, EPS_C).next_to(plot.coords(15500, float(np.interp(15500, LREG["eps_0.1"]["x"], LREG["eps_0.1"]["y"]))),
                                                          m.UP + m.LEFT, buff=0.12)
        lai = cite("Lai & Robbins, 1985 · Auer et al., 2002").move_to([0, -3.3, 0]).align_to([-6.2, 0, 0], m.LEFT)
        chapelle = cite("same in practice: Chapelle & Li, 2011").move_to([0, -3.3, 0]).align_to([6.2, 0, 0], m.RIGHT)
        src = cite(f"{RUNS:,} runs · measured on this Mac").move_to([4.7, 2.2, 0])

        self.say(
            "[The promise of UCB one is about the long run], [so let's play twenty thousand pulls.]",
            m.AnimationGroup(self.new_header("07 · the long run", "Logarithmic regret"), m.FadeIn(plot), m.FadeIn(src)),
            m.AnimationGroup(m.Create(first), m.FadeIn(first_lab)),
        )
        self.say(
            "[Epsilon-greedy explores at random forever], [so its regret keeps climbing in a straight line.]",
            m.Create(eps, run_time=1.8),
            m.AnimationGroup(m.FadeIn(l_eps, shift=0.1 * m.LEFT), m.FadeIn(lin_lab)),
        )
        self.say(
            "[Meanwhile, UCB one's curve bends over, growing like a logarithm], [and it crosses below for good, about twelve thousand pulls in.]",
            m.AnimationGroup(m.Create(ucb, run_time=2.4), m.FadeIn(l_ucb), m.FadeIn(log_lab)),
            m.AnimationGroup(m.Create(cross_line), m.FadeIn(cross, scale=0.5), m.FadeIn(cross_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[No method can beat that logarithmic growth]: [a classic result from nineteen eighty-five.]",
            m.Circumscribe(log_lab, color=UCB_C, buff=0.08),
            m.FadeIn(lai, shift=0.1 * m.UP),
        )
        self.say(
            "[Thompson sampling stays lowest at every horizon tested here]: [about fifty-seven coins after twenty thousand pulls.]",
            m.AnimationGroup(m.Create(ts, run_time=1.8), m.FadeIn(chapelle, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(l_ts, shift=0.1 * m.LEFT), m.Circumscribe(l_ts, color=TS_C, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · the testbed

    def testbed(self):
        self.next_section("8 · The testbed", notes=(
            f"Measured on this Mac: Sutton & Barto's 10-armed testbed (§2.3), {TB['runs']:,} problems x {TB['steps']:,} steps, "
            f"every method on the same problems. Averages over steps 901-1000: UCB (c = 2, as in S&B Fig. 2.4) {PCT_UCB}% "
            f"optimal, reward {REW_UCB}; epsilon-greedy 0.1 {PCT_EPS}%, {REW_EPS}; best possible {BEST_REW} (E[max of 10 "
            f"N(0, 1)]). UCB's spike at step {SPIKE['step']} (reward {SPIKE['avg_reward']}): every arm tried once, then the "
            f"best single sample (S&B Exercise 2.8). Optimistic greedy Q1 = 5, alpha = 0.1: {PCT_OPT}% (S&B Fig. 2.3)."))
        plot = Plot((0, 1000), (0, 1.8), width=6.2, height=3.9, center=(-2.85, -0.25, 0), x_ticks=[0, 500, 1000],
                    y_ticks=[0, 0.5, 1.0, 1.5], x_label="steps", y_label="average reward")
        ucb_d, eps_d = TBM["ucb_c2"]["avg_reward"], TBM["eps_0.1"]["avg_reward"]
        eps = plot.line(eps_d["x"], eps_d["y"], EPS_C, 3)
        ucb = plot.line(ucb_d["x"], ucb_d["y"], UCB_C, 3)
        best = plot.hline(BEST_REW, REWARD, 2)
        best_lab = code(f"best possible {BEST_REW:.2f}", 20, REWARD).next_to(plot.coords(500, BEST_REW), m.UP, buff=0.12)
        tip = plot.coords(SPIKE["step"], SPIKE["avg_reward"])
        spike_lab = code(f"step {SPIKE['step']}: {SPIKE['avg_reward']:.3f}", 20, UCB_C, weight=m.BOLD).move_to(plot.coords(330, 0.6))
        spike_arrow = arrow(spike_lab.get_left() + [-0.05, 0.05, 0], tip + [0.06, -0.04, 0], UCB_C, buff=0.08)

        x0 = 1.7
        head1 = cite("best arm chosen · steps 901–1,000").move_to([0, 2.15, 0]).align_to([x0, 0, 0], m.LEFT)

        def stat_row(label, value, color, y):
            lab = code(label, 22, color).move_to([0, y, 0]).align_to([x0, 0, 0], m.LEFT)
            val = code(value, 24, color, weight=m.BOLD).move_to([0, y, 0]).align_to([6.3, 0, 0], m.RIGHT)
            return m.VGroup(lab, val)

        r_ucb = stat_row("UCB, c = 2", f"{PCT_UCB:.1f}%", UCB_C, 1.6)
        r_eps = stat_row("ε-greedy, ε = 0.1", f"{PCT_EPS:.1f}%", EPS_C, 1.1)
        r_opt = stat_row("optimistic, Q₁ = 5", f"{PCT_OPT:.1f}%", EXPLORE, 0.6)
        head2 = cite("average reward · same steps").move_to([0, -0.15, 0]).align_to([x0, 0, 0], m.LEFT)
        r_ucb2 = stat_row("UCB, c = 2", f"{REW_UCB:.2f}", UCB_C, -0.7)
        r_eps2 = stat_row("ε-greedy, ε = 0.1", f"{REW_EPS:.2f}", EPS_C, -1.2)
        src = m.VGroup(cite(f"{TB['runs']:,} problems · measured on this Mac"), cite("Sutton & Barto, 2018, Figs. 2.3–2.4"))
        src.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([0, -2.4, 0]).align_to([x0, 0, 0], m.LEFT)
        rule = m.Line([x0, 1.85, 0], [6.3, 1.85, 0]).set_stroke(EDGE, 1.5)

        self.say(
            "[On episode two's ten-armed testbed, the story turns around.]",
            m.AnimationGroup(self.new_header("08 · the testbed", "Where UCB shines"), m.FadeIn(plot),
                             m.AnimationGroup(m.Create(best), m.FadeIn(best_lab)), m.FadeIn(src)),
        )
        self.say(
            "[With a bonus weight of two], [UCB beats epsilon-greedy]: [it picks the best arm eighty-six percent of the time, against eighty.]",
            m.AnimationGroup(m.Create(eps, run_time=1.5), m.FadeIn(head1), m.FadeIn(rule), m.FadeIn(r_eps)),
            m.AnimationGroup(m.Create(ucb, run_time=1.5), m.FadeIn(r_ucb)),
            m.AnimationGroup(m.Circumscribe(m.VGroup(r_ucb, r_eps), color=UCB_C, buff=0.08),
                             m.FadeIn(m.VGroup(head2, r_ucb2, r_eps2), shift=0.1 * m.UP)),
        )
        self.say(
            "[See the spike at step eleven?] [By then, every arm has been tried exactly once.]",
            m.AnimationGroup(m.FadeIn(spike_lab, shift=0.1 * m.UP), m.GrowArrow(spike_arrow)),
            m.Circumscribe(spike_lab, color=UCB_C, buff=0.08),
        )
        self.say(
            "[So UCB picks the arm with the best first result], [then the bonuses send it exploring again.]",
            m.Indicate(spike_arrow, color=UCB_C),
            m.ShowPassingFlash(ucb.copy().set_stroke(INK, 6), time_width=0.4, run_time=1.6),
        )
        self.say(
            "[Even simpler optimism works here]: [start every estimate at five], [and plain greedy reaches eighty-five point six percent.]",
            m.Circumscribe(m.VGroup(r_ucb, r_eps), color=EXPLORE, buff=0.08),
            m.FadeIn(r_opt[0], shift=0.1 * m.UP),
            m.FadeIn(r_opt[1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            "Optimism (UCB1: Q + sqrt(2 ln t / N), pull the highest top), luck (Thompson sampling: draw from Beta(1 + wins, "
            f"1 + losses), play the highest draw), patience (logarithmic regret is a long-run promise). Rover's machines, "
            f"1,000 pulls, {RUNS:,} runs: Thompson {R_TS}, epsilon-greedy {R_EPS}, UCB1 {R_UCB}; UCB1 ahead after pull "
            f"{OVERTAKE:,}. Testbed: UCB (c = 2) {PCT_UCB}% vs epsilon-greedy {PCT_EPS}%. Next: episode 4, bandits with context."))
        halo = glow(EXPLORE, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Optimism", 62, VALUE), title_text("Luck", 62, EXPLORE), title_text("Patience", 62, REWARD))
        words.arrange(m.RIGHT, buff=0.9).move_to([0, 0.9, 0])
        on_baseline(words)
        subs = m.VGroup(
            m.MathTex(r"Q + \sqrt{2\ln t \,/\, N}", font_size=36, color=VALUE),
            code("Beta(1 + wins, 1 + losses)", 20, EXPLORE),
            code("regret ∝ log t", 22, REWARD),
        )
        for s, w in zip(subs, words):
            s.next_to(w, m.DOWN, buff=0.4)
        base = max(s.get_center()[1] for s in subs)
        for s in subs:
            s.set_y(base)
        proof = code(f"Rover's machines, 1,000 pulls: Thompson {R_TS:.1f} · ε-greedy {R_EPS:.1f} · UCB1 {R_UCB:.1f}", 20, MUTED)
        proof.move_to([0, -1.55, 0])
        proof2 = code(f"10-armed testbed: UCB {PCT_UCB:.0f}% vs ε-greedy {PCT_EPS:.0f}% best arm", 20, MUTED).next_to(proof, m.DOWN, buff=0.2)
        mascot = rover(0.7).move_to([0, -3.0, 0])

        self.say(
            "[To sum up]: [optimism], [luck], [and patience.]",
            m.AnimationGroup(self.new_header("09 · recap", "Optimism, luck, patience"), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.2 * m.UP),
            m.FadeIn(words[1], shift=0.2 * m.UP),
            m.FadeIn(words[2], shift=0.2 * m.UP),
        )
        self.say(
            "[Add a bonus for what you haven't checked], [or sample from your beliefs], [and judge an explorer over the long run.]",
            m.FadeIn(subs[0], shift=0.1 * m.UP),
            m.FadeIn(subs[1], shift=0.1 * m.UP),
            m.FadeIn(subs[2], shift=0.1 * m.UP),
        )
        self.say(
            "[On Rover's machines, Thompson sampling came out lowest from a hundred pulls to twenty thousand]; [on the testbed, UCB beat exploring at random.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(proof2, shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()
