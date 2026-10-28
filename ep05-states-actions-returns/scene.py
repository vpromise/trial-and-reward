"""Trial and Reward · Episode 5 · States, Actions, Returns (Module 2 · Planning in a Known World)

Module 2 opens: Rover leaves the slot machines (Episodes 2–4: every pull starts fresh, there is no state) for its
grid world, where every move changes where it stands and so what can happen next. That is a Markov decision
process. The episode builds the pieces on Russell & Norvig's 4×3 grid (GridView.from_world(ROVER_WORLD)): the eleven
states, the four actions, the rewards (−0.04 a move, +1 charger, −1 puddle, both terminal) and the transitions — an
intended move splitting 80/10/10, drawn from world.transitions(s, a), ten sampled tries from world.step, and the
risky cell beside the puddle. Then the Markov property in one picture (two trips into the same cell, the same split
afterwards), an episode (Episode 1's seeded drive replayed, its rewards summed: 1 − 5 × 0.04 = 0.80) against a
wandering drive (seed 0: two slips, eight moves, 0.68), discounting (weights γ^k shrinking at γ = 0.9; why: an
uncertain future, sooner is better, finite sums; Rover's world uses γ = 1), the recursion G_t = r + γ G_(t+1) that
Episode 6 builds on, and the policy as arrows (and the loop that plays an episode). It ends on the question
Episode 6 answers: a thousand drives scatter their returns; how good is a cell, on average?

Every number on screen is computed below from rl_reference.py and asserted, or cited with its source.

Sources (research/notes.md §5, §0 row 7, Appendix A rows for §5, Appendix C row 5; facts dated October 2026):
- Russell & Norvig, Artificial Intelligence: A Modern Approach, 3rd ed. (2010), Ch. 17, Fig. 17.1 (the 4×3 world:
  0.8 intended, 0.1 each side, bumping = staying, −0.04 per step, γ = 1), Fig. 17.2(a) (the optimal policy = RN_POLICY).
- Sutton & Barto 2018, §3.1 (the Markov property, quoted on screen, p. 49), eqs. 3.7–3.9 (episodic return, discounted
  return, G_t = R_(t+1) + γ G_(t+1)), eq. 3.10 (+1 forever = 1/(1 − γ)), §1.7 (Bellman 1957 introduced MDPs).
- Bellman, R. (1957). A Markovian Decision Process. Journal of Mathematics and Mechanics 6(5):679–684.

Computed here (rl_reference): world.transitions((3, 1), "N") and ((3, 2), "N"); ten world.step tries from (3, 1),
seed 0 (7 north, 1 left, 2 right); Episode 1's drive play_episode(ROVER_WORLD, RN_POLICY, default_rng(11)) and a
second drive with seed 0; discounted_returns; the chance of reaching the charger under RN_POLICY and under uniformly
random moves (absorbing-chain linear solve on world.transitions: 98.6% and 35.4%, notes §5 worked example 3–4);
1,000 drives with seeds 0–999 (returns histogram, mean ≈ 0.69, exact value 0.705 is Episode 6/7's).
Illustrative: the slot-machine pulls in the hook (seeded BernoulliBandit pulls, values not discussed); the two trips
in the Markov picture (possible paths, not sampled); Rover's battery level (a made-up example of a missing state
variable); "money now rather than next year" (an analogy).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep05"
W = ROVER_WORLD

# ---------------------------------------------------------------------------- the world (Russell & Norvig 4x3)

NON_TERMINAL = [s for s in W.states if not W.is_terminal(s)]
assert (W.width, W.height) == (4, 3) and W.start == (1, 1)                        # "four columns by three rows"
assert len(W.states) == 11 and len(NON_TERMINAL) == 9                             # "the eleven cells"
assert W.walls == {ROCK} == {(2, 2)} and CHARGER == (4, 3) and PUDDLE == (4, 2)
assert W.terminals == {CHARGER: 1.0, PUDDLE: -1.0} and W.step_reward == -0.04      # "plus one", "minus one", "0.04"
assert W.slip == 0.2 and W.actions == ("N", "E", "S", "W") and ROVER_GAMMA == 1.0

# ---------------------------------------------------------------------------- slippery wheels


def fmt_outcome(p, s2, r, done):
    return f"({p:g}, ({s2[0]}, {s2[1]}), {r:g}, {done})"


SPLIT_CELL = (3, 1)
SPLIT = W.transitions(SPLIT_CELL, "N")
assert [(p, s2) for p, s2, _, _ in SPLIT] == [(0.8, (3, 2)), (0.1, (2, 1)), (0.1, (4, 1))]   # "eighty … ten, ten"
assert all(np.isclose(r, -0.04) and not d for _, _, r, d in SPLIT)
assert PERPENDICULAR["N"] == ("W", "E")                                            # slips: left, then right
RISK_CELL = (3, 2)
RISK = W.transitions(RISK_CELL, "N")
assert [(p, s2, d) for p, s2, _, d in RISK] == [(0.8, (3, 3), False), (0.1, (3, 2), False), (0.1, PUDDLE, True)]
assert np.isclose(RISK[2][2], -1.04) and W.move(RISK_CELL, "W") == RISK_CELL       # "slides right into it"; the rock
TRY_SEED = 0
_rng = np.random.default_rng(TRY_SEED)
TRIES = [W.step(SPLIT_CELL, "N", _rng)[0] for _ in range(10)]
TALLY = Counter(TRIES)
assert (TALLY[(3, 2)], TALLY[(2, 1)], TALLY[(4, 1)]) == (7, 1, 2)                 # "most go north, a few sideways"

SPLIT_CODE = [">>> world = ROVER_WORLD", '>>> world.transitions((3, 1), "N")'] + [
    ("[" if i == 0 else " ") + fmt_outcome(*o) + ("," if i < 2 else "]") for i, o in enumerate(SPLIT)]
RISK_CODE = [">>> world = ROVER_WORLD", '>>> world.transitions((3, 2), "N")'] + [
    ("[" if i == 0 else " ") + fmt_outcome(*o) + ("," if i < 2 else "]") for i, o in enumerate(RISK)]
assert SPLIT_CODE[2] == "[(0.8, (3, 2), -0.04, False)," and RISK_CODE[4] == " (0.1, (4, 2), -1.04, True)]"

# ---------------------------------------------------------------------------- the Markov picture (two possible trips)

TRIP_A = [(1, 1), (2, 1), (3, 1)]
TRIP_B = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1)]
for _trip in (TRIP_A, TRIP_B):
    assert all(any(W.move(a, d) == b for d in ACTIONS) for a, b in zip(_trip, _trip[1:]))   # every step is a real move

# ---------------------------------------------------------------------------- two drives (episodes)


def drive(seed):
    steps = play_episode(W, RN_POLICY, np.random.default_rng(seed))
    trail = [steps[0].s] + [t.s2 for t in steps]
    rewards = [t.r for t in steps]
    slips = [t.s2 != W.move(t.s, t.a) for t in steps]
    return steps, trail, rewards, slips


DRIVE1, TRAIL1, REW1, SLIP1 = drive(11)                                            # Episode 1's drive
G1 = discounted_returns(REW1, gamma=ROVER_GAMMA)
assert TRAIL1 == [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3)] and len(DRIVE1) == 5 and not any(SLIP1)
assert np.allclose(REW1, [-0.04] * 4 + [0.96])                                     # "plus one on top" of −0.04
assert np.isclose(G1[0], 1 - 5 * 0.04) and f"{G1[0]:.2f}" == "0.80"                # "zero point eight"
DRIVE2, TRAIL2, REW2, SLIP2 = drive(0)
G2 = discounted_returns(REW2, gamma=ROVER_GAMMA)
assert len(DRIVE2) == 8 and sum(SLIP2) == 2 and TRAIL2[-1] == CHARGER              # "slip twice", "eight moves"
assert [t.a for t in DRIVE2] == [RN_POLICY[t.s] for t in DRIVE2]                   # "aiming the same way"
assert np.isclose(G2[0], 1 - 8 * 0.04) and f"{G2[0]:.2f}" == "0.68"                # "zero point six eight"
assert len(DRIVE2) - len(DRIVE1) == 3                                              # "three extra moves"


def bump_direction(t):
    """Which way Rover bumped when a move left it in place (the one direction among aim and slips that stays)."""
    stays = [d for d in (t.a, *PERPENDICULAR[t.a]) if W.move(t.s, d) == t.s]
    assert len(stays) == 1
    return stays[0]


# ---------------------------------------------------------------------------- discounting

GAMMA_D = 0.9
WEIGHTS = GAMMA_D ** np.arange(8)
assert [f"{w:.2f}" for w in WEIGHTS[:3]] == ["1.00", "0.90", "0.81"]               # "one, zero point nine, …"
G1_09 = discounted_returns(REW1, gamma=GAMMA_D)[0]
TERMS = [r * GAMMA_D ** k for k, r in enumerate(REW1)]
assert np.isclose(sum(TERMS), G1_09) and round(G1_09, 2) == 0.49                   # "zero point four nine"
assert [f"{x:+.3f}" for x in TERMS] == ["-0.040", "-0.036", "-0.032", "-0.029", "+0.630"]
FOREVER = 1 / (1 - GAMMA_D)
assert np.isclose(FOREVER, 10) and np.isclose(sum(GAMMA_D ** k for k in range(3000)), 10)   # "adds up to just ten"
assert np.isclose(G1[0], REW1[0] + ROVER_GAMMA * G1[1]) and f"{G1[1]:.2f}" == "0.84"         # 0.80 = −0.04 + 0.84

# ---------------------------------------------------------------------------- policies


def reach_charger(policy):
    """P(reach the charger) and the expected number of moves from the start, solved exactly from the model."""
    idx = {s: i for i, s in enumerate(NON_TERMINAL)}
    n = len(NON_TERMINAL)
    P, b = np.zeros((n, n)), np.zeros(n)
    for s in NON_TERMINAL:
        choice = policy[s] if isinstance(policy[s], dict) else {policy[s]: 1.0}
        for a, pa in choice.items():
            for p, s2, _, done in W.transitions(s, a):
                if done:
                    b[idx[s]] += pa * p * (s2 == CHARGER)
                else:
                    P[idx[s], idx[s2]] += pa * p
    A = np.eye(n) - P
    return np.linalg.solve(A, b)[idx[W.start]], np.linalg.solve(A, np.ones(n))[idx[W.start]]


assert set(RN_POLICY) == set(NON_TERMINAL)
RANDOM_POLICY = {s: {a: 0.25 for a in ACTIONS} for s in NON_TERMINAL}
P_RN, MOVES_RN = reach_charger(RN_POLICY)
P_RANDOM, MOVES_RANDOM = reach_charger(RANDOM_POLICY)
assert round(100 * P_RN, 1) == 98.6 and round(MOVES_RN, 2) == 6.68                 # notes §5 worked example 3
assert round(100 * P_RANDOM) == 35 and round(MOVES_RANDOM, 1) == 32.4              # "thirty-five percent"

# ---------------------------------------------------------------------------- a thousand drives

MANY = 1000
MANY_G = [float(np.round(sum(t.r for t in play_episode(W, RN_POLICY, np.random.default_rng(seed))), 2))
          for seed in range(MANY)]
assert MANY_G[11] == 0.80 and MANY_G[0] == 0.68                                    # both drives are among them
MANY_MEAN = float(np.mean(MANY_G))
COUNTS = Counter(MANY_G)
PERFECT, IN_PUDDLE = COUNTS[0.80], sum(g < 0 for g in MANY_G)
assert round(MANY_MEAN, 1) == 0.7 and round(MANY_MEAN, 2) == 0.69                  # "about zero point seven"
assert PERFECT == 319 and round(PERFECT / MANY, 1) == 0.3 and IN_PUDDLE == 19      # "about a third", "a few"
assert max(COUNTS.values()) == PERFECT and min(MANY_G) > -1.6 and max(MANY_G) == 0.80

# ---------------------------------------------------------------------------- the hook's slot machines (illustrative)

_hrng = np.random.default_rng(5)
HOOK_PULLS = [(arm, BernoulliBandit(ROVER_ARMS).pull(arm, _hrng)) for arm in (1, 3)]

# ---------------------------------------------------------------------------- look

DICE = recolored("dice", EXPLORE, VIOLET_FILL, prefix=PREFIX)
CLOCK = recolored("clock", REWARD, GOLD_FILL, prefix=PREFIX)

GRID_C = (-3.4, -0.3, 0.0)
CELL = 1.3
LIFT = np.array([0.0, 0.08, 0.0])
RX = 3.35                     # the right panel's centre
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words, s):
    """The baseline of a Text made from s: the median bottom of its flat-bottomed letters."""
    chars = [c for c in s if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, s, y):
    """Put the baseline of Text `words` (made from s) at height y."""
    return words.shift((y - baseline(words, s)) * m.UP)


def vec(direction):
    dc, dr = MOVES[direction]
    return np.array([dc, dr, 0.0])


def code_lines(lines, size=20):
    """Python output in Menlo: probabilities VIOLET, cells TEAL, rewards GOLD, True RED, False MUTED."""
    src = "\n".join(lines)
    return code(src, size, INK, t2c={
        r">>>": MUTED,
        r"ROVER_WORLD": STATE,
        r"\"N\"": ACTION,
        r"(?<=\()0\.\d(?=,)": EXPLORE,
        r"\(\d, \d\)(?=, -)": STATE,
        r"-\d\.\d+": REWARD,
        r"True": RED,
        r"False": MUTED,
    })


class Board:
    """Rover's world on the left of the frame: the GridView, its column and row numbers, and helpers."""

    def __init__(self):
        self.grid = GridView.from_world(W, cell=CELL, center=GRID_C)
        g = self.grid
        cols = m.VGroup(*[code(str(c), 20, MUTED).move_to(g.at((c, 1)) + [0, -0.95, 0]) for c in range(1, 5)])
        rows = m.VGroup(*[code(str(r), 20, MUTED).move_to(g.at((1, r)) + [-0.95, 0, 0]) for r in range(1, 4)])
        self.numbers = m.VGroup(cols, rows)
        self.group = m.VGroup(self.grid, self.numbers)

    def at(self, s):
        return self.grid.at(s)

    def spot(self, s):
        return self.grid.at(s) + LIFT

    def square(self, s):
        return self.grid.squares[W.to_screen(s)]

    def branch(self, s, s2, p, label_offset=(0.0, 0.02), end_back=0.42, size=22):
        """An outcome of a move: a VIOLET arrow from s into s2, as thick as it is likely, and its probability."""
        a, b = self.at(s), self.at(s2)
        d = (b - a) / np.linalg.norm(b - a)
        arr = m.Arrow(a + 0.4 * d, b - end_back * d, buff=0, color=EXPLORE, stroke_width=3 + 8 * p,
                      max_tip_length_to_length_ratio=0.3)
        lab = code(f"{p:g}", size, EXPLORE, weight=m.BOLD).move_to(b + np.array([*label_offset, 0.0][:3]))
        return m.VGroup(arr, lab)

    def bounce(self, s, direction, p, label_offset):
        """A slip into a wall or the rock: a U-turn arrow against that side of the cell, and its probability."""
        c, d = self.at(s), vec(direction)
        perp = np.array([-d[1], d[0], 0.0])
        arc = m.CurvedArrow(c + 0.4 * d + 0.14 * perp, c + 0.4 * d - 0.14 * perp, angle=-0.9 * np.pi,
                            color=EXPLORE, stroke_width=4, tip_length=0.13)
        lab = code(f"{p:g}", 20, EXPLORE, weight=m.BOLD).move_to(c + np.array([*label_offset, 0.0][:3]))
        return m.VGroup(arc, lab)

    def trail(self, cells, color=ACTION, offset=(0, 0, 0), width=6):
        pts = [self.at(s) + np.array(offset) for s in cells]
        return m.VMobject().set_points_as_corners(pts).set_stroke(color, width).set_z_index(3)

    def tick(self, s, direction, color=EXPLORE):
        """A short bar on the side of cell s that Rover bumped into."""
        c, d = self.at(s), vec(direction)
        perp = np.array([-d[1], d[0], 0.0])
        mid = c + 0.56 * d
        return m.Line(mid - 0.3 * perp, mid + 0.3 * perp).set_stroke(color, 7).set_z_index(3)

    def policy_arrows(self, policy, shift=(0.3, -0.3, 0), length=0.5, color=ACTION, width=5):
        group = m.VGroup()
        for s, a in policy.items():
            p = self.at(s) + np.array(shift)
            v = vec(a) * length / 2
            group.add(m.Arrow(p - v, p + v, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.4))
        return group.set_z_index(2)

    def compasses(self, cells, shift=(0.3, -0.3, 0), color=ACTION):
        """Uniformly random moves: four short thin arrows in every cell."""
        group = m.VGroup()
        for s in cells:
            p = self.at(s) + np.array(shift)
            for a in ACTIONS:
                v = vec(a)
                group.add(m.Arrow(p + 0.05 * v, p + 0.27 * v, buff=0, color=color, stroke_width=3,
                                  max_tip_length_to_length_ratio=0.45))
        return group.set_z_index(2)


def row_term(term, color, desc, y, left, desc_x, mono_desc=False, desc_color=INK):
    t = code(term, 22, color, weight=m.BOLD).move_to([0, y, 0]).align_to([left, 0, 0], m.LEFT)
    d = (code(desc, 22, desc_color) if mono_desc else text(desc, 22, desc_color)).move_to([0, y, 0])
    d.align_to([desc_x, 0, 0], m.LEFT)
    sit(d, desc, baseline(t, term))
    return m.VGroup(t, d)


def battery(level=0.4, color=GREEN):
    body = m.RoundedRectangle(corner_radius=0.05, width=0.5, height=0.26).set_stroke(INK, 2.5).set_fill(CARD, 1)
    nub = m.Rectangle(width=0.05, height=0.11).set_stroke(width=0).set_fill(INK, 1).next_to(body, m.RIGHT, buff=0)
    juice = m.Rectangle(width=0.42 * level, height=0.18).set_stroke(width=0).set_fill(color, 1)
    juice.align_to(body, m.LEFT).shift(0.04 * m.RIGHT).set_y(body.get_y())
    return m.VGroup(body, nub, juice)


# ---------------------------------------------------------------------------- the episode


class Ep05StatesActionsReturns(Episode):
    number = 5

    def lesson(self):
        self.board = Board()
        self.bot = rover(0.42).set_z_index(4)
        self.hook()
        self.pieces()
        self.slippery()
        self.markov()
        self.episodes()
        self.discount()
        self.policy()
        self.next_question()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Episodes 2–4 (bandits): Rover's choices leave no trace, every pull starts fresh: no state. The two pulls are "
            f"illustrative (BernoulliBandit(ROVER_ARMS), seed 5: {HOOK_PULLS}). In the grid every move changes the state "
            "(the TEAL readout), and with it what can happen next (the cells reachable from (3, 1)). Name: Markov decision "
            "process (Bellman 1957, 'A Markovian Decision Process', via Sutton & Barto 2018 §1.7)."))
        B, bot = self.board, self.bot
        machines = m.VGroup(*[SlotMachine(str(i + 1), 0.8) for i in range(5)]).arrange(m.RIGHT, buff=0.55)
        machines.move_to([0, 0.95, 0])
        big = rover(0.75).move_to([machines[1].cabinet.get_x(), -1.0, 0])
        chips = [machines[arm].payout(r, 22) for arm, r in HOOK_PULLS]
        fresh = m.VGroup(code("state:", 24, STATE, weight=m.BOLD), text("none, every pull starts fresh", 26, INK))
        fresh.arrange(m.RIGHT, buff=0.2)
        sit(fresh[1], fresh[1].text, baseline(fresh[0], "state:"))
        fresh.move_to([0, -2.3, 0])
        eps = cite("Episodes 2–4 · bandits").next_to(fresh, m.DOWN, buff=0.25)
        stage = m.VGroup(machines, big, fresh, eps)

        bot.move_to(B.spot((1, 1)))
        panel = card(4.2, 1.7, STATE, CARD, radius=0.2, stroke=2.5).move_to([RX, 0.55, 0])
        p_lab = code("state", 24, STATE, weight=m.BOLD).move_to(panel.get_top() + [0, -0.38, 0])
        p_val = [code(f"({s[0]}, {s[1]})", 40, INK, weight=m.BOLD).move_to(panel.get_center() + [0, -0.22, 0])
                 for s in ((1, 1), (2, 1), (3, 1))]
        p_sub = text("where Rover stands", 22, MUTED).next_to(panel, m.DOWN, buff=0.22)
        nexts = m.VGroup(*[m.DashedVMobject(m.Square(1.08), num_dashes=22).set_stroke(EXPLORE, 3).move_to(B.at(s))
                           for s in ((2, 1), (3, 2), (4, 1))]).set_z_index(2)
        n_lab = text("what can happen next depends on it", 22, EXPLORE).next_to(p_sub, m.DOWN, buff=0.45)

        def move_to(s, k):
            return m.AnimationGroup(bot.animate.move_to(B.spot(s)), m.Transform(p_val[0], p_val[k]), run_time=0.7)

        self.say(
            "[Until now, Rover played slot machines]: [every pull started from scratch], [whatever came before.]",
            m.AnimationGroup(self.new_header("00 · hook", "From slot machines to a world"),
                             m.LaggedStart(*[m.FadeIn(sm, shift=0.2 * m.UP) for sm in machines], lag_ratio=0.1),
                             m.FadeIn(big, shift=0.2 * m.UP)),
            m.Succession(
                m.AnimationGroup(machines[HOOK_PULLS[0][0]].pull(0.6), m.FadeIn(chips[0], scale=0.6, run_time=0.6)),
                m.FadeOut(chips[0], run_time=0.3),
                big.animate(run_time=0.6).move_to([machines[HOOK_PULLS[1][0]].cabinet.get_x(), -1.0, 0]),
                m.AnimationGroup(machines[HOOK_PULLS[1][0]].pull(0.6), m.FadeIn(chips[1], scale=0.6, run_time=0.6)),
            ),
            m.AnimationGroup(m.FadeIn(fresh, shift=0.15 * m.UP), m.FadeIn(eps)),
        )
        self.say(
            "[Now Rover drives around a grid]: [each move changes where it stands], [and so what can happen next.]",
            m.AnimationGroup(m.FadeOut(stage, shift=0.2 * m.DOWN), m.FadeOut(chips[1]), m.FadeIn(B.group),
                             m.FadeIn(bot, shift=0.15 * m.UP), m.FadeIn(panel), m.FadeIn(p_lab), m.FadeIn(p_val[0]),
                             m.FadeIn(p_sub)),
            m.Succession(move_to((2, 1), 1), move_to((3, 1), 2)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(n) for n in nexts], lag_ratio=0.2), m.FadeIn(n_lab, shift=0.1 * m.UP)),
        )
        halo = glow(STATE, radius=2.6).move_to([RX, 0.2, 0])
        name = title_text("Markov decision process", 34, INK).move_to([RX, 0.55, 0])
        short = pill("MDP", STATE, TEAL_FILL, 26, mono=True).next_to(name, m.UP, buff=0.35)
        what = text("a world where actions change what happens next", 22, MUTED).next_to(name, m.DOWN, buff=0.3)
        src = cite("Bellman, 1957 · Sutton & Barto, 2018, §1.7").next_to(what, m.DOWN, buff=0.45)
        readout = m.VGroup(panel, p_lab, p_val[0], p_sub, n_lab)
        self.say(
            "[A world like this is called a Markov decision process]: [an MDP, for short.]",
            m.AnimationGroup(m.FadeOut(readout, shift=0.15 * m.UP), m.FadeOut(nexts), m.FadeIn(halo, scale=0.6),
                             m.FadeIn(name, shift=0.15 * m.UP), m.FadeIn(what)),
            m.AnimationGroup(m.FadeIn(short, scale=0.7), m.FadeIn(src)),
        )
        self.wait(0.4)
        self.clear(B.group, bot)

    # ------------------------------------------------------------------ 1 · the pieces

    def pieces(self):
        self.next_section("1 · The pieces", notes=(
            "Rover's world = Russell & Norvig 3rd ed. (2010), Ch. 17, Fig. 17.1: 4 columns × 3 rows, (col, row) with (1, 1) "
            "bottom-left, the rock at (2, 2), charger +1 at (4, 3), puddle −1 at (4, 2), start (1, 1). States: the 11 cells "
            "Rover can occupy (9 non-terminal). Actions N, E, S, W. Rewards: −0.04 on every move, +1 / −1 on the move that "
            "enters a terminal cell, which ends the episode (rl_reference convention = R&N's numbers at γ = 1, notes §5). "
            "Transitions p(s′ | s, a): next section."))
        B, bot = self.board, self.bot
        box = card(6.6, 4.5, STATE, CARD, radius=0.22, stroke=2.5).move_to([RX - 0.1, -0.3, 0])
        left, desc_x = box.get_left()[0] + 0.4, box.get_left()[0] + 2.25
        title = title_text("Rover's world as an MDP", 30).move_to([0, box.get_top()[1] - 0.5, 0]).align_to([left, 0, 0], m.LEFT)
        rule = m.Line([left, box.get_top()[1] - 0.92, 0], [box.get_right()[0] - 0.4, box.get_top()[1] - 0.92, 0]).set_stroke(EDGE, 2)
        ys = [0.65, -0.1, -0.85, -1.85]
        r_states = row_term("states", STATE, "11 cells Rover can be in", ys[0], left, desc_x)
        r_actions = row_term("actions", ACTION, "N · E · S · W", ys[1], left, desc_x, mono_desc=True, desc_color=ACTION)
        r_rewards = row_term("rewards", REWARD, "−0.04 a move, ±1 at the end", ys[2], left, desc_x)
        r_end = text("both end the episode", 20, MUTED).move_to([0, ys[2] - 0.42, 0]).align_to([desc_x, 0, 0], m.LEFT)
        r_trans = row_term("transitions", EXPLORE, "where a move really lands", ys[3], left, desc_x)
        r_q = code("p(s′ | s, a)", 22, EXPLORE).next_to(r_trans[1], m.DOWN, buff=0.16).align_to([desc_x, 0, 0], m.LEFT)
        src = cite("Russell & Norvig, 2010, Fig. 17.1").move_to([GRID_C[0], -3.15, 0])

        overlays = m.VGroup(*[m.Square(CELL - 0.16).set_stroke(STATE, 5).set_fill(STATE, 0.18).move_to(B.at(s))
                              for s in W.states]).set_z_index(2)
        compass = m.VGroup(*[m.Arrow(B.at((1, 1)) + LIFT * 0 + 0.42 * vec(a), B.at((1, 1)) + 0.76 * vec(a), buff=0,
                                     color=ACTION, stroke_width=6, max_tip_length_to_length_ratio=0.45)
                             for a in ACTIONS]).set_z_index(5)
        cost = reward_chip(W.step_reward, 20).move_to(B.at((1, 1)) + [0, 0.25, 0]).set_z_index(6)

        self.say(
            "[It's the classic grid from Russell and Norvig's textbook]: [four columns by three rows], [with Rover starting in the corner.]",
            m.AnimationGroup(self.new_header("01 · the pieces", "Rover's world as an MDP"), m.FadeIn(src),
                             m.FadeIn(box), m.FadeIn(title), m.Create(rule)),
            m.LaggedStart(m.Indicate(B.numbers[0], color=INK, scale_factor=1.15),
                          m.Indicate(B.numbers[1], color=INK, scale_factor=1.15), lag_ratio=0.5),
            bot.animate.move_to(B.spot((1, 1))),
        )
        self.say(
            "[First, the states]: [the eleven cells Rover can stand in.]",
            m.FadeIn(r_states, shift=0.1 * m.LEFT),
            m.LaggedStart(*[m.FadeIn(o, scale=0.85) for o in overlays], lag_ratio=0.12),
        )
        self.say(
            "[Then the actions]: [north, east, south or west.]",
            m.AnimationGroup(m.FadeOut(overlays), m.FadeIn(r_actions, shift=0.1 * m.LEFT)),
            m.LaggedStart(*[m.GrowArrow(a) for a in compass], lag_ratio=0.25),
        )
        self.say(
            "[The rewards]: [every move costs zero point zero four], [the charger pays plus one, the puddle minus one], "
            "[and both of them end the episode.]",
            m.AnimationGroup(m.FadeOut(compass), m.FadeIn(r_rewards, shift=0.1 * m.LEFT)),
            m.Succession(m.FadeIn(cost, shift=0.25 * m.UP, run_time=0.5), m.FadeOut(cost, shift=0.25 * m.UP, run_time=0.5)),
            m.AnimationGroup(m.Circumscribe(B.square(CHARGER), color=GOLD, buff=0.05),
                             m.Circumscribe(B.square(PUDDLE), color=RED, buff=0.05)),
            m.AnimationGroup(m.FadeIn(r_end, shift=0.1 * m.UP),
                             m.Flash(B.at(CHARGER), color=GOLD, flash_radius=0.55, line_length=0.18),
                             m.Flash(B.at(PUDDLE), color=RED, flash_radius=0.55, line_length=0.18)),
        )
        self.say(
            "[And the transitions]: [where each move really lands.] [That's where it gets interesting.]",
            m.FadeIn(r_trans, shift=0.1 * m.LEFT),
            m.FadeIn(r_q, shift=0.1 * m.UP),
            m.Circumscribe(m.VGroup(r_trans, r_q), color=EXPLORE, buff=0.1),
        )
        self.wait(0.3)
        self.clear(B.group, bot)

    # ------------------------------------------------------------------ 2 · slippery wheels

    def slippery(self):
        self.next_section("2 · Slippery wheels", notes=(
            "The transition model (R&N Fig. 17.1): the intended move happens with probability 0.8, a slip to either side "
            "with 0.1 each; bumping into a wall or the rock means staying put. From (3, 1) aiming N (world.transitions): "
            f"{SPLIT}. Ten tries with world.step, seed {TRY_SEED}: {TRIES} → north {TALLY[(3, 2)]}, left {TALLY[(2, 1)]}, "
            f"right {TALLY[(4, 1)]}. Beside the puddle, (3, 2) aiming N: {RISK} (−1.04 = −0.04 for the move − 1 for the "
            "puddle; True = the episode is over)."))
        B, bot = self.board, self.bot
        a0 = B.at(SPLIT_CELL)
        aim = m.Arrow(a0 + 0.4 * vec("N"), B.at((3, 2)) - 0.3 * vec("N"), buff=0, color=ACTION, stroke_width=7,
                      max_tip_length_to_length_ratio=0.3).set_z_index(3)
        aim_lab = code("aim: N", 22, ACTION, weight=m.BOLD).move_to(B.at((3, 2)) + [0.0, 0.3, 0])
        north = B.branch(SPLIT_CELL, (3, 2), 0.8).set_z_index(3)
        left = B.branch(SPLIT_CELL, (2, 1), 0.1).set_z_index(3)
        right = B.branch(SPLIT_CELL, (4, 1), 0.1).set_z_index(3)

        # ten tries: dots fly from Rover to where each try landed; the tally counts them
        slots = {(3, 2): [(-0.46 + 0.16 * (k % 3), 0.5 - 0.14 * (k // 3)) for k in range(9)],
                 (2, 1): [(-0.46 + 0.16 * k, 0.46) for k in range(3)],
                 (4, 1): [(0.46 - 0.16 * k, 0.46) for k in range(3)]}
        used = Counter()
        dots = []
        for s2 in TRIES:
            dx, dy = slots[s2][used[s2]]
            used[s2] += 1
            dots.append(m.Dot(B.at(s2) + [dx, dy, 0], radius=0.055, color=EXPLORE).set_z_index(4))
        heads = ["north", "left", "right"]
        cells = [(3, 2), (2, 1), (4, 1)]
        xs = [RX - 1.8, RX, RX + 1.8]
        t_title = text(f"10 tries from the same cell · world.step, seed {TRY_SEED}", 22, MUTED).move_to([RX, 1.7, 0])
        t_heads = m.VGroup(*[code(h, 22, MUTED).move_to([x, 1.15, 0]) for h, x in zip(heads, xs)])
        running = {c: 0 for c in cells}
        counts = [code("0", 36, EXPLORE, weight=m.BOLD).move_to([x, 0.62, 0]) for x in xs]
        flights = []
        for s2, dot in zip(TRIES, dots):
            running[s2] += 1
            k = cells.index(s2)
            new = code(str(running[s2]), 36, EXPLORE, weight=m.BOLD).move_to([xs[k], 0.62, 0])
            flights.append(m.AnimationGroup(m.FadeIn(dot, target_position=bot.get_center(), run_time=0.32),
                                            m.Transform(counts[k], new, run_time=0.32)))
        assert running == {(3, 2): 7, (2, 1): 1, (4, 1): 2}

        out = code_lines(SPLIT_CODE)
        out_box = card(out.width + 0.7, out.height + 0.6, EDGE, CARD, radius=0.16, stroke=2).move_to([RX, -1.05, 0])
        out.move_to(out_box).align_to(out_box, m.LEFT).shift(0.32 * m.RIGHT)
        legend = m.VGroup(*[code(w, 20, c) for w, c in (("probability", EXPLORE), ("next cell", STATE),
                                                        ("reward", REWARD), ("episode over?", MUTED))])
        legend.arrange(m.RIGHT, buff=0.32).next_to(out_box, m.DOWN, buff=0.25)
        out_risk = code_lines(RISK_CODE).move_to(out).align_to(out, m.LEFT)

        r_north = B.branch(RISK_CELL, (3, 3), 0.8).set_z_index(3)
        r_puddle = B.branch(RISK_CELL, PUDDLE, 0.1, (-CELL + 0.33, -0.42), size=20).set_z_index(3)
        r_rock = B.bounce(RISK_CELL, "W", 0.1, (-0.34, 0.37)).set_z_index(3)
        tries = m.VGroup(*dots)
        tally = m.VGroup(t_title, t_heads, *counts)

        self.say(
            "[Rover's wheels slip.] [Say it stands here], [and tries to go north.]",
            self.new_header("02 · transitions", "Slippery wheels"),
            bot.animate.move_to(B.spot(SPLIT_CELL)),
            m.AnimationGroup(m.GrowArrow(aim), m.FadeIn(aim_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[Eighty percent of the time, it goes where it meant to]; [ten percent each, it slides off to the left or the right.]",
            m.AnimationGroup(m.ReplacementTransform(aim, north[0]), m.FadeOut(aim_lab), m.FadeIn(north[1], scale=0.7)),
            m.LaggedStart(m.AnimationGroup(m.GrowArrow(left[0]), m.FadeIn(left[1], scale=0.7)),
                          m.AnimationGroup(m.GrowArrow(right[0]), m.FadeIn(right[1], scale=0.7)), lag_ratio=0.5),
        )
        self.say(
            "[Ten tries from the same cell]: [most go north, and a few slide sideways.]",
            m.AnimationGroup(m.FadeIn(t_title), m.FadeIn(t_heads), *[m.FadeIn(c) for c in counts]),
            m.Succession(*flights),
        )
        self.say(
            "[The world's model lists every outcome]: [its probability, the next cell, the reward], [and whether the episode is over.]",
            m.AnimationGroup(m.FadeIn(out_box), m.AddTextLetterByLetter(out, time_per_char=0.012)),
            m.LaggedStart(*[m.FadeIn(w, shift=0.1 * m.UP) for w in legend[:3]], lag_ratio=0.3),
            m.FadeIn(legend[3], shift=0.1 * m.UP),
        )
        self.say(
            "[Next to the puddle, the same move is risky]: [one try out of ten slides right into it.]",
            m.AnimationGroup(m.FadeOut(tries), m.FadeOut(tally), m.FadeOut(north), m.FadeOut(left), m.FadeOut(right),
                             bot.animate.move_to(B.spot(RISK_CELL)), m.Transform(out, out_risk),
                             m.FadeIn(r_north[0]), m.FadeIn(r_north[1])),
            m.AnimationGroup(m.GrowArrow(r_puddle[0]), m.FadeIn(r_puddle[1], scale=0.7),
                             m.Flash(B.at(PUDDLE), color=RED, flash_radius=0.55, line_length=0.2)),
        )
        last = sum(len(line.replace(" ", "")) for line in RISK_CODE[:4])
        risky_row = m.VGroup(*out_risk[last:])
        self.say(
            "[A slip into the rock just leaves Rover where it is]: [the episode goes on.] [The puddle ends it.]",
            m.AnimationGroup(m.Create(r_rock[0]), m.FadeIn(r_rock[1], scale=0.7)),
            m.Circumscribe(r_rock, color=EXPLORE, buff=0.06),
            m.Circumscribe(risky_row, color=RED, buff=0.06),
        )
        self.wait(0.3)
        self.clear(B.group, bot)

    # ------------------------------------------------------------------ 3 · the Markov property

    def markov(self):
        self.next_section("3 · Markov", notes=(
            "The Markov property (Sutton & Barto 2018 §3.1, quoted): the next state and reward depend only on the current "
            "state and action, not on the path that led there. Two possible trips into (3, 1) (illustrative paths, every "
            f"step a real move): {TRIP_A} and {TRIP_B}; afterwards a move north splits {[p for p, *_ in SPLIT]} either way "
            "(world.transitions((3, 1), 'N') has no memory). It is a property of how the state is described: a battery "
            "level that mattered would have to be part of the state (illustrative)."))
        B, bot = self.board, self.bot
        tA = B.trail(TRIP_A, ACTION)
        tB = B.trail(TRIP_B, INK, width=5)
        start_dot = m.Dot(B.at((1, 1)), radius=0.09, color=INK).set_z_index(4)
        north = B.branch(SPLIT_CELL, (3, 2), 0.8, (0.3, 0.12)).set_z_index(5)
        left = B.branch(SPLIT_CELL, (2, 1), 0.1, (-0.05, 0.3)).set_z_index(5)
        right = B.branch(SPLIT_CELL, (4, 1), 0.1).set_z_index(5)
        split = m.VGroup(north, left, right)

        x0 = 0.35
        rowA = m.VGroup(m.Line([0, 0, 0], [0.5, 0, 0]).set_stroke(ACTION, 6), text("trip 1 · 2 moves", 24, INK))
        rowB = m.VGroup(m.Line([0, 0, 0], [0.5, 0, 0]).set_stroke(INK, 5), text("trip 2 · 6 moves", 24, INK))
        for row, y in ((rowA, 1.5), (rowB, 0.9)):
            row.arrange(m.RIGHT, buff=0.25).move_to([0, y, 0]).align_to([x0, 0, 0], m.LEFT)
        thenA = code("N: 0.8 · 0.1 · 0.1", 22, EXPLORE).next_to(rowA, m.RIGHT, buff=0.35)
        thenB = code("N: 0.8 · 0.1 · 0.1", 22, EXPLORE).next_to(rowB, m.RIGHT, buff=0.35)
        thenB.align_to(thenA, m.LEFT)
        sit(thenA, thenA.text, baseline(rowA[1], rowA[1].text))
        sit(thenB, thenB.text, baseline(rowB[1], rowB[1].text))

        name = title_text("The Markov property", 32, STATE).move_to([RX, 0.2, 0])
        f1 = m.MathTex("p(", "s_{t+1}", r"\mid", "s_t", ",", "a_t", ",", r"s_{t-1}, a_{t-1}, \ldots, s_0", ")",
                       font_size=36, color=INK)
        f2 = m.MathTex("=", "p(", "s_{t+1}", r"\mid", "s_t", ",", "a_t", ")", font_size=36, color=INK)
        for f, parts in ((f1, ((1, STATE), (3, STATE), (5, ACTION), (7, MUTED))), (f2, ((2, STATE), (4, STATE), (6, ACTION)))):
            for k, c in parts:
                f[k].set_color(c)
        f1[0].set_color(EXPLORE)
        f2[1].set_color(EXPLORE)
        f1.move_to([RX, -0.55, 0])
        strike = m.Line(f1[7].get_corner(m.DL) + [0, -0.1, 0], f1[7].get_corner(m.DR) + [0, -0.1, 0]).set_stroke(RED, 4)
        hist = text("the path that led there", 20, RED).next_to(strike, m.DOWN, buff=0.12)
        f2.next_to(hist, m.DOWN, buff=0.3).align_to(f1, m.LEFT).shift(0.2 * m.RIGHT)

        quote = m.VGroup(text("“The state must include information about all", 22, INK),
                         text("aspects of the past agent–environment", 22, INK),
                         text("interaction that make a difference", 22, INK),
                         text("for the future.”", 22, INK)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        quote.move_to([RX, 1.55, 0])
        q_src = cite("Sutton & Barto, 2018, §3.1").next_to(quote, m.DOWN, buff=0.15).align_to(quote, m.RIGHT)

        chip_y = -2.75
        s_lab = code("state", 22, STATE, weight=m.BOLD)
        s_val = code("(3, 1)", 26, INK, weight=m.BOLD)
        plain = m.VGroup(s_lab, s_val).arrange(m.RIGHT, buff=0.3)
        plain_box = card(plain.width + 0.6, 0.66, STATE, TEAL_FILL, radius=0.14, stroke=2.5)
        plain.move_to(plain_box)
        state_chip = m.VGroup(plain_box, plain).move_to([RX - 0.6, chip_y, 0])
        b_val = m.VGroup(code("(3, 1)", 26, INK, weight=m.BOLD), code("+", 26, MUTED), battery(0.4),
                         code("40%", 24, GREEN, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        rich = m.VGroup(code("state", 22, STATE, weight=m.BOLD), b_val).arrange(m.RIGHT, buff=0.3)
        rich_box = card(rich.width + 0.6, 0.66, STATE, TEAL_FILL, radius=0.14, stroke=2.5)
        rich.move_to(rich_box)
        state_rich = m.VGroup(rich_box, rich).move_to([RX - 0.6, chip_y, 0])
        ex = cite("for example").next_to(state_rich, m.RIGHT, buff=0.25)

        self.say(
            "[Two different trips can end in the same cell]: [one along the bottom], [one all the way around the top.]",
            m.AnimationGroup(self.new_header("03 · markov", "Only the current cell matters"), m.FadeOut(bot),
                             m.FadeIn(start_dot, scale=0.5)),
            m.AnimationGroup(m.Create(tA, run_time=1.0), m.FadeIn(rowA, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(tB, run_time=1.6), m.FadeIn(rowB, shift=0.1 * m.LEFT)),
        )
        bot.move_to(B.spot(SPLIT_CELL))
        self.say(
            "[From here, a move north splits the same way after either trip]: [eighty, ten, ten.]",
            m.AnimationGroup(tA.animate.set_stroke(opacity=0.35), tB.animate.set_stroke(opacity=0.35),
                             m.FadeIn(bot, scale=0.8), *[m.GrowArrow(b[0]) for b in split]),
            m.AnimationGroup(*[m.FadeIn(b[1], scale=0.7) for b in split], m.FadeIn(thenA, shift=0.1 * m.LEFT),
                             m.FadeIn(thenB, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[That's the Markov property]: [what happens next depends only on the current state and action], "
            "[not on the path that led there.]",
            m.FadeIn(name, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(f1, shift=0.1 * m.UP), m.FadeIn(f2, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(hist, shift=0.1 * m.DOWN), m.Create(strike), f1[7].animate.set_color(FAINT)),
        )
        self.say(
            "[So the state has to carry everything that matters for the future.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(rowA, rowB, thenA, thenB)), m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(q_src)),
        )
        self.say(
            "[If Rover's battery mattered too], [it would have to be part of the state.]",
            m.FadeIn(state_chip, shift=0.1 * m.UP),
            m.AnimationGroup(m.ReplacementTransform(state_chip, state_rich), m.FadeIn(ex)),
        )
        self.wait(0.4)
        self.clear(B.group)

    # ------------------------------------------------------------------ 4 · episodes and returns

    def episodes(self):
        self.next_section("4 · An episode", notes=(
            "An episode runs from the start until a terminal cell (episodic task, Sutton & Barto §3.4). Drive 1 is Episode 1's: "
            f"play_episode(ROVER_WORLD, RN_POLICY, default_rng(11)): {TRAIL1}, rewards {[round(r, 2) for r in REW1]}, no slip; "
            f"return with γ = 1 (S&B eq. 3.7) G = {G1[0]:.2f} = 1 − 5 × 0.04. Drive 2, the same policy, seed 0: {TRAIL2}, "
            f"slips at moves {[k + 1 for k, sl in enumerate(SLIP2) if sl]} (a bump into the top wall, then a slide south), "
            f"8 moves, G = {G2[0]:.2f} = 1 − 8 × 0.04."))
        B, bot = self.board, self.bot
        row_y = [0.85 - 0.38 * i for i in range(8)]
        cols = {1: 0.45, 2: 3.55}

        def table(n, steps, slips, sub):
            x0 = cols[n]
            head = code(f"drive {n}", 24, ACTION, weight=m.BOLD).move_to([0, 1.85, 0]).align_to([x0, 0, 0], m.LEFT)
            note = text(sub, 20, MUTED).next_to(head, m.DOWN, buff=0.14).align_to(head, m.LEFT)
            aims, rewards, tags = [], [], []
            for k, (t, sl) in enumerate(zip(steps, slips)):
                y = row_y[k]
                v = vec(t.a) * 0.15
                c = np.array([x0 + 0.6, y, 0])
                aims.append(m.VGroup(code(str(k + 1), 20, MUTED).move_to([x0 + 0.1, y, 0]),
                                     m.Arrow(c - v, c + v, buff=0, color=ACTION, stroke_width=5,
                                             max_tip_length_to_length_ratio=0.45)))
                rewards.append(code(f"{t.r:+.2f}".replace("-", "−"), 22, GOLD if t.r > 0 else RED, weight=m.BOLD)
                               .move_to([x0 + 1.45, y, 0]))
                tags.append(code("slip", 20, EXPLORE, weight=m.BOLD).move_to([x0 + 2.35, y, 0]) if sl else None)
            rule = m.Line([x0, -2.08, 0], [x0 + 2.7, -2.08, 0]).set_stroke(EDGE, 2)
            lab = code("return", 22, VALUE).move_to([0, -2.45, 0]).align_to([x0, 0, 0], m.LEFT)
            totals, acc = [], 0.0
            for t in steps:
                acc += t.r
                totals.append(code(f"{acc:.2f}".replace("-", "−"), 30, VALUE, weight=m.BOLD)
                              .move_to([0, -2.45, 0]).align_to([x0 + 1.15, 0, 0], m.LEFT))
            zero = code("0.00", 30, VALUE, weight=m.BOLD).move_to(totals[0]).align_to(totals[0], m.LEFT)
            sit(lab, "return", baseline(zero, "0.00"))
            for tt in totals:
                sit(tt, tt.text, baseline(zero, "0.00"))
            return dict(head=m.VGroup(head, note), aims=aims, rewards=rewards, tags=tags, rule=rule, lab=lab,
                        zero=zero, totals=totals)

        T1 = table(1, DRIVE1, SLIP1, "from Episode 1")
        T2 = table(2, DRIVE2, SLIP2, "same aims, new dice")
        assert T1["totals"][-1].text == "0.80" and T2["totals"][-1].text == "0.68"
        sum1 = code("= 1 − 5 × 0.04", 22, MUTED).next_to(T1["totals"][-1], m.DOWN, buff=0.2).align_to(T1["lab"], m.LEFT)

        def run(trail, steps, table_, offset=(0, 0, 0)):
            anims, segs = [], m.VGroup()
            for k, t in enumerate(steps):
                parts = [m.FadeIn(table_["aims"][k], shift=0.1 * m.LEFT)]
                if table_["tags"][k] is not None:
                    parts.append(m.FadeIn(table_["tags"][k], scale=0.7))
                if t.s2 == t.s:
                    d = bump_direction(t)
                    mark = B.tick(t.s, d)
                    segs.add(mark)
                    parts += [bot.animate(rate_func=m.there_and_back).move_to(B.spot(t.s) + 0.22 * vec(d)), m.Create(mark)]
                else:
                    seg = m.Line(B.at(t.s) + np.array(offset), B.at(t.s2) + np.array(offset)).set_stroke(ACTION, 6).set_z_index(3)
                    segs.add(seg)
                    parts += [bot.animate.move_to(B.spot(t.s2)), m.Create(seg)]
                anims.append(m.AnimationGroup(*parts, run_time=0.5))
            return m.Succession(*anims), segs

        drive1, segs1 = run(TRAIL1, DRIVE1, T1)
        drive2, segs2 = run(TRAIL2, DRIVE2, T2, offset=(0.14, -0.14, 0))

        def pay(table_, upto):
            steps = []
            for k in range(upto):
                steps.append(m.AnimationGroup(m.FadeIn(table_["rewards"][k], shift=0.1 * m.LEFT),
                                              m.Transform(table_["zero"], table_["totals"][k]), run_time=0.4))
            return steps

        bot.move_to(B.spot((1, 1)))
        self.say(
            "[In episode one, Rover drove from the start to the charger]: [five moves, without a single slip.]",
            m.AnimationGroup(self.new_header("04 · an episode", "One drive, one return"), m.FadeIn(bot, scale=0.8),
                             m.FadeIn(T1["head"], shift=0.1 * m.DOWN)),
            drive1,
        )
        self.say(
            "[Each move cost zero point zero four], [and the last one paid plus one on top.]",
            m.AnimationGroup(m.Create(T1["rule"]), m.FadeIn(T1["lab"]),
                             m.Succession(m.FadeIn(T1["zero"], run_time=0.3), *pay(T1, 4))),
            m.Succession(*pay(T1, 5)[4:], m.Flash(B.at(CHARGER), color=GOLD, flash_radius=0.55, line_length=0.2)),
        )
        self.say(
            "[An episode ends at a terminal cell.] [Add up its rewards, and that's its return]: [zero point eight.]",
            m.Circumscribe(B.square(CHARGER), color=GOLD, buff=0.05),
            m.FadeIn(sum1, shift=0.1 * m.UP),
            m.Circumscribe(T1["zero"], color=VALUE, buff=0.1),
        )
        self.say(
            "[Drive again, aiming the same way in every cell], [and this time the wheels slip twice: eight moves.]",
            m.AnimationGroup(*[sg.animate.set_stroke(opacity=0.3) for sg in segs1], bot.animate.move_to(B.spot((1, 1))),
                             m.FadeIn(T2["head"], shift=0.1 * m.DOWN)),
            drive2,
        )
        self.say(
            "[Three extra moves, three extra costs]: [this return is only zero point six eight.]",
            m.AnimationGroup(m.Create(T2["rule"]), m.FadeIn(T2["lab"]),
                             m.Succession(m.FadeIn(T2["zero"], run_time=0.3), *pay(T2, 8)),
                             m.LaggedStart(*[m.Circumscribe(tag, color=EXPLORE, buff=0.06) for tag in T2["tags"] if tag is not None],
                                           lag_ratio=0.5)),
            m.Circumscribe(T2["zero"], color=VALUE, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · discounting

    def discount(self):
        self.next_section("5 · Discount", notes=(
            "Discounted return (Sutton & Barto eq. 3.8): G = r1 + γ r2 + γ² r3 + …, 0 ≤ γ ≤ 1. Weights at γ = 0.9: "
            f"{[round(w, 3) for w in WEIGHTS]}. Why discount (notes §5): the future is uncertain (equivalently a constant "
            "chance 1 − γ that the episode stops each step), sooner is preferred (the money analogy is illustrative), and "
            f"continuing tasks need a finite sum: +1 forever = 1/(1 − γ) = {FOREVER:g} (eq. 3.10). Drive 1 at γ = 0.9: "
            f"{[round(x, 4) for x in TERMS]} → {G1_09:.4f}. Rover's world uses γ = 1 (episodic; R&N; notes Appendix A: aima-python's "
            "default 0.9 gives other numbers). Recursion G_t = r_(t+1) + γ G_(t+1) (eq. 3.9): 0.80 = −0.04 + 0.84 for drive 1."))
        formula = m.MathTex("G", "=", "r_1", "+", r"\gamma", "r_2", "+", r"\gamma^2", "r_3", "+", r"\gamma^3", "r_4",
                            "+", r"\cdots", font_size=46, color=INK)
        formula[0].set_color(VALUE)
        for k in (2, 5, 8, 11):
            formula[k].set_color(REWARD)
        for k in (4, 7, 10):
            formula[k].set_color(VALUE)
        formula.move_to([0, 1.8, 0])
        gp = pill("γ = 0.9", VALUE, BLUE_FILL, 22, mono=True).move_to([-3.4, 0.95, 0])
        bars = Bars(WEIGHTS, labels=[str(k + 1) for k in range(len(WEIGHTS))], top=1.0, bottom=0.0, width=5.9,
                    height=2.3, center=(-3.35, -0.85, 0), color=VALUE, gap=0.24, fmt="{:.2f}", size=20)
        caption = text("the weight of reward 1, 2, 3 …", 22, MUTED).move_to([-3.4, -2.8, 0])

        def reason(name, title, desc, y, glyph=None):
            box = card(5.9, 1.2, EDGE, CARD, radius=0.18, stroke=2).move_to([RX + 0.1, y, 0])
            pic = glyph if glyph is not None else icon(name, 0.62)
            pic.move_to(box.get_left() + [0.62, 0, 0])
            words = m.VGroup(text(title, 24, INK, weight=m.BOLD), text(desc, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            words.next_to(pic, m.RIGHT, buff=0.4)
            return m.VGroup(box, pic, words)

        inf = m.MathTex(r"\infty", font_size=56, color=VALUE)
        r1 = reason(DICE, "The future is uncertain", "a reward far away may never arrive", 0.85)
        r2 = reason(CLOCK, "Sooner is better", "like money now rather than next year", -0.5)
        r3 = reason(None, "Sums stay finite", "+1 forever: 1 + 0.9 + 0.81 + … = 10", -1.85, glyph=inf)
        reasons = m.VGroup(r1, r2, r3)

        d_lab = text("drive 1, with γ = 0.9", 22, MUTED).move_to([RX, 0.95, 0])
        terms = f"{TERMS[0]:.3f}".replace("-", "−") + " " + " ".join(("+ " if x > 0 else "− ") + f"{abs(x):.3f}" for x in TERMS[1:])
        assert terms == "−0.040 − 0.036 − 0.032 − 0.029 + 0.630"
        d_terms = code(terms, 20, INK, t2c={r"0\.630": GOLD}).move_to([RX, 0.5, 0])
        d_val = code(f"= {G1_09:.2f}", 30, VALUE, weight=m.BOLD).move_to([RX, 0.0, 0])
        d_was = text(f"instead of {G1[0]:.2f}", 20, MUTED).next_to(d_val, m.RIGHT, buff=0.3)
        sit(d_was, d_was.text, baseline(d_val, d_val.text))
        g1 = pill("Rover's world: γ = 1", STATE, TEAL_FILL, 24, mono=True).move_to([RX, -0.95, 0])
        g1_sub = text("episodes end, so every reward counts in full", 20, MUTED).next_to(g1, m.DOWN, buff=0.18)
        rec = m.MathTex("G_t", "=", "r_{t+1}", "+", r"\gamma", "G_{t+1}", font_size=40, color=INK)
        rec[0].set_color(VALUE)
        rec[2].set_color(REWARD)
        rec[4].set_color(VALUE)
        rec[5].set_color(VALUE)
        rec.move_to([RX - 0.8, -2.3, 0])
        rec_ex = code(f"{G1[0]:.2f} = {REW1[0]:.2f} + {G1[1]:.2f}".replace("-", "−"), 22, INK,
                      t2c={r"0\.80|0\.84": VALUE, r"−0\.04": RED}).next_to(rec, m.DOWN, buff=0.2).align_to(rec, m.LEFT)
        ep6 = pill("Episode 6", STATE, TEAL_FILL, 20, mono=True).next_to(rec, m.RIGHT, buff=0.5)
        ones = [1.0] * len(WEIGHTS)

        self.say(
            "[So far, every reward counted in full.] [In general, we discount]: [each step of delay multiplies a reward by gamma.]",
            self.new_header("05 · discount", "Sooner counts more"),
            m.FadeIn(formula, shift=0.15 * m.UP),
            m.LaggedStart(*[m.Indicate(formula[k], color=VALUE, scale_factor=1.3) for k in (4, 7, 10)], lag_ratio=0.3),
        )
        self.say(
            "[With gamma at zero point nine], [the weights shrink]: [one, zero point nine, zero point eight one, and on down.]",
            m.FadeIn(gp, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(bars.zero), m.FadeIn(bars.labels), m.FadeIn(caption),
                             m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(b, m.DOWN), m.FadeIn(n))
                                             for b, n in zip(bars.bars, bars.numbers)], lag_ratio=0.15)),
            m.LaggedStart(*[m.Circumscribe(m.VGroup(bars.bars[k], bars.numbers[k]), color=VALUE, buff=0.06)
                            for k in range(3)], lag_ratio=0.35),
        )
        self.say(
            "[Why discount?] [First, the future is uncertain]: [a reward far away may never arrive.]",
            m.Wait(0.1),
            m.FadeIn(r1[0:2], shift=0.1 * m.LEFT),
            m.FadeIn(r1[2], shift=0.1 * m.LEFT),
        )
        self.say(
            "[Second, sooner is better], [like money now rather than next year.]",
            m.FadeIn(r2[0:2], shift=0.1 * m.LEFT),
            m.FadeIn(r2[2], shift=0.1 * m.LEFT),
        )
        self.say(
            "[Third, when a task never ends], [discounting keeps the sum finite]: [plus one forever adds up to just ten.]",
            m.FadeIn(r3[0:2], shift=0.1 * m.LEFT),
            m.FadeIn(r3[2], shift=0.1 * m.LEFT),
            m.Circumscribe(r3[2][1], color=VALUE, buff=0.06),
        )
        self.say(
            "[With gamma at zero point nine], [Rover's first drive would be worth only zero point four nine.]",
            m.AnimationGroup(m.FadeOut(reasons, shift=0.1 * m.RIGHT), m.FadeIn(d_lab), m.FadeIn(d_terms, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(d_val, scale=0.8), m.FadeIn(d_was)),
        )
        self.say(
            "[But Rover's world doesn't need a discount]: [its episodes end], [so gamma is one, and every reward counts in full.]",
            m.FadeIn(g1, scale=0.85),
            m.FadeIn(g1_sub, shift=0.1 * m.UP),
            m.AnimationGroup(bars.set(ones, run_time=1.0), m.Transform(gp, pill("γ = 1", VALUE, BLUE_FILL, 22, mono=True).move_to(gp))),
        )
        self.say(
            "[One more thing]: [each return is one reward plus the next return.] [Episode six is built on exactly that.]",
            m.Wait(0.1),
            m.AnimationGroup(m.FadeIn(rec, shift=0.1 * m.UP), m.FadeIn(rec_ex, shift=0.1 * m.UP)),
            m.FadeIn(ep6, shift=0.1 * m.LEFT),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the policy

    def policy(self):
        self.next_section("6 · Policy", notes=(
            "A policy is a rule from cell to action (deterministic here; π(a | s) in general). The arrows are RN_POLICY, Russell "
            "& Norvig Fig. 17.2(a) (optimal for −0.04, γ = 1): note the bottom row going the long way round. Both drives "
            "followed it; only world.step's random slips differed. The loop is play_episode, simplified (rl_reference.py), "
            f"plus the running return. Reaching the charger from the start, solved exactly from world.transitions: these "
            f"arrows {P_RN:.4f} (in {MOVES_RN:.2f} moves on average), uniformly random moves {P_RANDOM:.4f} (in "
            f"{MOVES_RANDOM:.1f} moves); notes §5 worked example: 98.63% and 35.4%."))
        B = self.board
        arrows = B.policy_arrows(RN_POLICY)
        rand = B.compasses(NON_TERMINAL)
        t1 = B.trail(TRAIL1, ACTION).set_stroke(opacity=0.5)
        t2 = B.trail([s for s, s2 in zip(TRAIL2, TRAIL2[1:]) if s != s2] + [TRAIL2[-1]], ACTION, offset=(0.14, -0.14, 0))
        bump = B.tick((3, 3), bump_direction(DRIVE2[4]))
        trails = m.VGroup(t1, t2, bump)

        look = code_lines([">>> RN_POLICY[(1, 1)]", "'N'", ">>> RN_POLICY[(3, 1)]", "'W'"], 22)
        paint(look, look.text, {r"'[NW]'": ACTION, r"\(\d, \d\)": STATE, r">>>": MUTED})
        look_title = text("a policy: cell → action", 24, INK, weight=m.BOLD)
        look_box = card(4.6, look.height + look_title.height + 1.0, ACTION, CARD, radius=0.18, stroke=2.5).move_to([RX, 0.6, 0])
        look_title.move_to(look_box.get_top() + [0, -0.42, 0])
        look.next_to(look_title, m.DOWN, buff=0.3)
        look_card = m.VGroup(look_box, look_title, look)
        src = cite("Russell & Norvig, 2010, Fig. 17.2").move_to([GRID_C[0], -3.15, 0])

        loop_src = "\n".join([
            "s, G = world.reset(), 0.0",
            "while True:",
            "    a = policy[s]",
            "    s, r, done = world.step(s, a, rng)",
            "    G += r",
            "    if done:",
            "        break",
        ])
        loop = code_block(loop_src, 20, width=6.2, title="play_episode, simplified", accent=EDGE)
        loop.move_to([RX, 0.45, 0])
        body = loop[1]
        lines = loop_src.split("\n")

        def line_glyphs(i):
            start = sum(len(l.replace(" ", "")) for l in lines[:i])
            return m.VGroup(*body[start:start + len(lines[i].replace(" ", ""))])

        rule_line, dice_line = line_glyphs(2), line_glyphs(3)
        tag_rule = m.VGroup(m.Dot(radius=0.07, color=ACTION), code("the rule: policy[s]", 20, ACTION, weight=m.BOLD))
        tag_dice = m.VGroup(m.Dot(radius=0.07, color=EXPLORE), code("the dice: world.step", 20, EXPLORE, weight=m.BOLD))
        for tg in (tag_rule, tag_dice):
            tg.arrange(m.RIGHT, buff=0.15)
        tag_rule.next_to(loop, m.DOWN, buff=0.3).align_to(loop, m.LEFT).shift(0.3 * m.RIGHT)
        tag_dice.next_to(tag_rule, m.DOWN, buff=0.2).align_to(tag_rule, m.LEFT)

        bars_y = [-0.05, -1.25]
        cmp_title = text("reaching the charger from the start", 24, INK, weight=m.BOLD).move_to([RX, 1.45, 0])
        cmp_sub = cite("exact, solved from the world's model").next_to(cmp_title, m.DOWN, buff=0.15)
        full = 4.4
        x_left = RX - 2.6
        cmp = m.VGroup()
        for (lab, p, color), y in zip((("these arrows", P_RN, ACTION), ("random moves", P_RANDOM, MUTED)), bars_y):
            name = text(lab, 22, MUTED).move_to([0, y + 0.38, 0]).align_to([x_left, 0, 0], m.LEFT)
            track = m.Rectangle(width=full, height=0.34).set_stroke(EDGE, 2).set_fill(CARD, 1)
            track.move_to([x_left + full / 2, y, 0])
            bar = m.Rectangle(width=full * p, height=0.34).set_stroke(color, 2).set_fill(FILLS.get(color, CARD), 1)
            bar.align_to(track, m.LEFT).set_y(y)
            val = code(f"{100 * p:.1f}%", 24, color if color != MUTED else INK, weight=m.BOLD).next_to(track, m.RIGHT, buff=0.2)
            cmp.add(m.VGroup(name, track, bar, val))

        self.say(
            "[Where did Rover's aims come from?] [A policy]: [a rule that names an action for every cell.]",
            m.AnimationGroup(self.new_header("06 · policy", "A rule for every cell"), m.FadeIn(B.group)),
            m.AnimationGroup(m.FadeIn(look_card, shift=0.1 * m.UP), m.FadeIn(src)),
            m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.1),
        )
        self.say(
            "[Both drives followed these same arrows]; [only the dice were different.]",
            m.AnimationGroup(m.Create(t1), m.Create(t2, run_time=1.4), m.Create(bump)),
            m.Circumscribe(bump, color=EXPLORE, buff=0.12),
        )
        self.say(
            "[In code, an episode is a short loop]: [the policy picks the move], [and the world rolls the dice.]",
            m.AnimationGroup(m.FadeOut(look_card, shift=0.1 * m.UP), m.FadeIn(loop, shift=0.1 * m.UP)),
            m.AnimationGroup(rule_line.animate.set_color(ACTION), m.FadeIn(tag_rule, shift=0.1 * m.UP)),
            m.AnimationGroup(dice_line.animate.set_color(EXPLORE), m.FadeIn(tag_dice, shift=0.1 * m.UP)),
        )
        self.say(
            "[Change the rule, and the episodes change.] [A Rover that drives at random reaches the charger only thirty-five percent of the time.]",
            m.AnimationGroup(m.FadeOut(trails), m.FadeOut(m.VGroup(loop, tag_rule, tag_dice)),
                             m.FadeIn(cmp_title), m.FadeIn(cmp_sub), m.FadeIn(cmp[0], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.ReplacementTransform(arrows, rand), m.FadeIn(cmp[1], shift=0.1 * m.LEFT),
                             m.Circumscribe(cmp[1][3], color=INK, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · how good is a cell?

    def next_question(self):
        self.next_section("7 · How good is a cell?", notes=(
            f"{MANY} drives from the start with RN_POLICY (play_episode, seeds 0–{MANY - 1}; drive 2 is seed 0, drive 1 seed 11). "
            f"Returns: {PERFECT} perfect 5-move drives (0.80; 0.8^5 = 0.328 expected), {IN_PUDDLE} ended in the puddle; mean "
            f"{MANY_MEAN:.3f}. The expected return is the cell's value, V(1, 1) = 0.705 exactly (R&N Fig. 17.3; Episodes 6–7 "
            "compute it, Episode 8 estimates it by averaging drives like these)."))
        s1 = m.VGroup(code("drive 1", 24, ACTION, weight=m.BOLD), code(f"{G1[0]:.2f}", 40, VALUE, weight=m.BOLD)).arrange(m.DOWN, buff=0.15)
        s2 = m.VGroup(code("drive 2", 24, ACTION, weight=m.BOLD), code(f"{G2[0]:.2f}", 40, VALUE, weight=m.BOLD)).arrange(m.DOWN, buff=0.15)
        pair = m.VGroup(s1, s2).arrange(m.RIGHT, buff=1.6).move_to([0, 1.2, 0])
        ask = title_text("How good is the start cell?", 50, INK).move_to([0, -0.6, 0])
        bot = rover(0.6).next_to(ask, m.DOWN, buff=0.5)

        plot = Plot((-1.6, 1.0), (0, 340), width=10.4, height=2.9, center=(0.3, -0.85, 0),
                    x_ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1.0], y_ticks=[0, 100, 200, 300], x_label="return",
                    y_label="drives", x_fmt="{:g}")
        hist = m.VGroup()
        for v, n in sorted(COUNTS.items()):
            lo, hi = plot.coords(v - 0.015, 0), plot.coords(v + 0.015, n)
            hist.add(m.Rectangle(width=hi[0] - lo[0], height=max(hi[1] - lo[1], 0.02)).set_stroke(VALUE, 1.5)
                     .set_fill(BLUE_FILL, 1).move_to((lo + hi) / 2))
        many = text(f"{MANY:,} drives from the start, same arrows", 22, MUTED)
        many.next_to(plot.coords(1.0, 340), m.UP, buff=0.05).align_to(plot.coords(1.0, 0), m.RIGHT)
        perfect = text(f"{PERFECT} perfect", 20, VALUE).next_to(hist[-1], m.RIGHT, buff=0.15).align_to(hist[-1], m.UP)
        puddle_bars = m.VGroup(*[b for (v, _), b in zip(sorted(COUNTS.items()), hist) if v < 0])
        puddle = text(f"puddle: {IN_PUDDLE}", 20, RED).next_to(puddle_bars, m.UP, buff=0.2)
        mean_line = plot.vline(MANY_MEAN, VALUE, 3).set_z_index(3)
        block_x = plot.coords(-1.0, 0)[0]
        mean_lab = code("average ≈ 0.7", 24, VALUE, weight=m.BOLD).move_to(plot.coords(0, 250)).align_to([block_x, 0, 0], m.LEFT)
        value_lab = text("the value of the start cell", 26, INK, weight=m.BOLD).next_to(mean_lab, m.DOWN, buff=0.18).align_to(mean_lab, m.LEFT)
        links = cite("Episode 6 computes it; Episode 8 averages drives").next_to(value_lab, m.DOWN, buff=0.15)
        links.align_to(value_lab, m.LEFT)
        pointer = arrow(mean_lab.get_right() + [0.15, 0, 0], [plot.coords(MANY_MEAN, 0)[0] - 0.08, mean_lab.get_y(), 0], VALUE, buff=0)

        self.say(
            "[Same arrows, two drives]: [zero point eight, then zero point six eight.] [So how good is the start cell, really?]",
            m.AnimationGroup(self.new_header("07 · next", "How good is a cell?"), m.FadeIn(s1[0])),
            m.AnimationGroup(m.FadeIn(s1[1], scale=0.8), m.FadeIn(s2, scale=0.9)),
            m.AnimationGroup(m.FadeIn(ask, shift=0.15 * m.UP), m.FadeIn(bot, shift=0.15 * m.UP)),
        )
        self.say(
            "[Play a thousand drives from the start], [and the returns spread out]: [about a third are perfect], [and a few end in the puddle.]",
            m.AnimationGroup(m.FadeOut(ask, shift=0.15 * m.UP), m.FadeOut(bot), m.FadeOut(s1), m.FadeOut(s2),
                             m.FadeIn(plot), m.FadeIn(many)),
            m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in hist], lag_ratio=0.06),
            m.FadeIn(perfect, shift=0.1 * m.LEFT),
            m.FadeIn(puddle, shift=0.1 * m.DOWN),
        )
        self.say(
            "[Their average, about zero point seven, is what we're after]: [the return Rover can expect from a cell], [its value.]",
            m.AnimationGroup(m.Create(mean_line), m.FadeIn(mean_lab, shift=0.1 * m.LEFT), m.GrowArrow(pointer)),
            m.FadeIn(value_lab, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(links), m.Circumscribe(value_lab, color=VALUE, buff=0.1)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "States (where Rover is: all that matters, the Markov property), actions (what it tries; the world's transitions "
            "decide where it lands, 0.8 / 0.1 / 0.1), returns (rewards added up, discounted by γ; γ = 1 in Rover's world). "
            "An MDP: states, actions, transitions, rewards, discount."))
        halo = glow(STATE, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("States", STATE), ("Actions", ACTION), ("Returns", VALUE))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("where Rover is", "what it tries", "the rewards, added up")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.9
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.6 * m.UP)
        parts = [("states", STATE), ("actions", ACTION), ("transitions", EXPLORE), ("rewards", REWARD), ("discount γ", VALUE)]
        mdp = m.VGroup(code("MDP =", 22, INK, weight=m.BOLD),
                       *[pill(p, c, FILLS[c], 20, mono=True) for p, c in parts]).arrange(m.RIGHT, buff=0.18)
        mdp.next_to(row, m.DOWN, buff=0.75)
        tagline = text("A world where actions change what happens next.", 30, MUTED).next_to(mdp, m.DOWN, buff=0.5)
        mascot = rover(0.42).next_to(tagline, m.DOWN, buff=0.35)

        self.say(
            "[To sum up]: [states], [actions], [returns.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[A world where actions change what happens next], [and every reward adds to the return.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in mdp], lag_ratio=0.12), m.Write(tagline)),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, mdp, tagline, mascot)], run_time=0.9)
