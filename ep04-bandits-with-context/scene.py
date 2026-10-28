"""Trial and Reward · Episode 4 · Bandits with Context (Module 1 · Trial and Reward)

The best action can depend on who is asking. Rover recommends one of three study tips (STUDY_TIPS) to each
student; a student is two numbers, "likes reading" and "likes talking" (the context, TEAL). For the example, each
tip helps with a chance that is a straight line in those numbers (STUDY_TIP_THETA). Lined up from pure readers to
pure talkers, each tip is a line and the best tip switches as the student changes (a context marker sweeps, the
chosen arm switches). LinUCB learns one line per tip (least squares, theta = A^-1 b) and adds a bonus where its
data is thin (linucb_scores); the real seeded run (seed 130, the measured median-regret run) after 10, 50, 200 and
2,000 students; Rover frozen after 200 students choosing for three real students. Measured on this Mac (200 runs x
2,000 students): LinUCB 0.527 helped per student, context-free UCB1 0.424, always practice problems 0.449, the
oracle 0.540. Yahoo! front-page news (Li et al., 2010): 12.5% click lift, an offline replay estimate on over 33
million events vs context-free epsilon-greedy. Offline replay, done here on a random log of Rover's own. A/B test vs
bandit (traffic share, computed). Module 1 closes: in a bandit, Rover's actions don't change what comes next.

Sources (facts dated October 2026):
- rl_reference.py: STUDY_TIPS, STUDENT_FEATURES, STUDY_TIP_THETA, linucb_scores, run_study_tips, argmax_random,
  bandit_testbed (the A/B sketch), ROVER_WORLD (the closing grid), measured().
- research/experiments/classic.md §3 / classic.json: study_tips (200 runs x 2,000 students; methods avg_reward,
  regret curves, tip_share, pct_best_tip_last500; exact best_tip_share; example = seed 130, theta_hat_after).
- research/notes.md §4: Li, Chu, Langford & Schapire, WWW 2010 (arXiv:1003.0146): abstract "over 33 million
  events", "12.5% click lift compared to a standard context-free bandit algorithm"; Table 1, linucb (disjoint),
  deployment bucket, 100% data: relative CTR 1.795 vs epsilon-greedy 1.596 (random = 1); users reduced to 6 numbers
  (5 user clusters + a constant). Appendix A: quote the abstract's "over 33 million" (section 5.2.1 says about 36
  million). Replay evaluation: Li et al. 2010 Algorithm 3 / Li, Chu, Langford & Wang 2011 ("provably unbiased");
  keeps about 1/K of a uniformly random log. A/B tests vs bandits: Scott 2015 (opportunity cost; the weekday/weekend
  caveat, section 4.2). Sutton & Barto 2018 §2.9 (associative search: if actions also affect the next situation,
  "then we have the full reinforcement learning problem").

Illustrative (said "for example"): the payoff lines themselves (STUDY_TIP_THETA is the course's example world, not
data about real students); the A/B sketch's two versions (helping 50% and 60%), whose traffic curve is computed with
bandit_testbed (Thompson sampling, 2,000 runs). Everything else is computed below and asserted, or cited.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep04"

# ---------------------------------------------------------------------------- the example world (rl_reference)

N_TIPS, N_FEATURES = STUDY_TIP_THETA.shape
FLASH, PRACTICE, FRIEND = 0, 1, 2
SHORT = ("flashcards", "practice", "friend")
assert (N_TIPS, N_FEATURES) == (3, 2)
assert STUDY_TIPS == ("flashcards", "practice problems", "explain it to a friend")
assert STUDENT_FEATURES == ("likes reading", "likes talking")
THETA = STUDY_TIP_THETA
assert np.allclose(THETA, [[0.8, 0.0], [0.45, 0.45], [0.0, 0.8]])

AVG = THETA @ np.full(N_FEATURES, 0.5)                       # the average student (0.5, 0.5)
assert np.allclose(AVG, [0.40, 0.45, 0.40]) and int(np.argmax(AVG)) == PRACTICE   # "forty-five percent, against forty"
# For uniform students, flashcards are best where talking < 7/9 reading: an area of 7/18; explaining, the mirror image;
# practice problems, the wedge between: 4/18.
SHARE = np.array([7, 4, 7]) / 18
ST = measured("classic", "study_tips")
assert np.allclose(ST["exact"]["best_tip_share"], SHARE, atol=6e-4)
assert [round(100 * v) for v in SHARE] == [39, 22, 39]       # "twenty-two percent", "thirty-nine percent each"


def slice_x(s):
    """A student on the line from a pure reader (s = 0: reading 1, talking 0) to a pure talker (s = 1)."""
    return np.array([1.0 - s, s])


def best_on_slice(s):
    return int(np.argmax(THETA @ slice_x(s)))


def envelope(s):
    return float(np.max(THETA @ slice_x(s)))


CROSS = (7 / 16, 9 / 16)                                      # flashcards = practice, practice = explaining
assert np.isclose(THETA[FLASH] @ slice_x(CROSS[0]), 0.45) and np.isclose(THETA[FRIEND] @ slice_x(CROSS[1]), 0.45)
S_READER, S_STOPS = 0.12, (0.30, 0.50, 0.88)                  # where the context marker stops
assert best_on_slice(S_READER) == FLASH and [best_on_slice(s) for s in S_STOPS] == [FLASH, PRACTICE, FRIEND]

# ---------------------------------------------------------------------------- the seeded LinUCB run (seed 130)

EX = ST["example"]
SEED, ROUNDS, ALPHA = EX["seed"], ST["students"], 1.0
assert EX["method"] == "linucb_alpha1" and SEED == 130 and ROUNDS == 2000 and ST["runs"] == 200
SNAP_AT = (10, 50, 200, 2000)


def linucb_run(seed=SEED, rounds=ROUNDS, alpha=ALPHA):
    """run_study_tips("linucb") step by step on the same random stream, keeping every student, choice and reward,
    and each tip's (A, b) after 10, 50, 200 and 2,000 students."""
    rng = np.random.default_rng(seed)
    A_list = [np.eye(N_FEATURES) for _ in range(N_TIPS)]
    b_list = [np.zeros(N_FEATURES) for _ in range(N_TIPS)]
    xs, acts, rewards, snaps = [], [], [], {}
    for t in range(1, rounds + 1):
        x = rng.random(N_FEATURES)
        p = THETA @ x
        a = argmax_random(linucb_scores(A_list, b_list, x, alpha), rng)
        r = float(rng.random() < p[a])
        A_list[a] += np.outer(x, x)
        b_list[a] += r * x
        xs.append(x)
        acts.append(a)
        rewards.append(r)
        if t in SNAP_AT:
            snaps[t] = ([A.copy() for A in A_list], [b.copy() for b in b_list])
    return np.array(xs), np.array(acts), np.array(rewards), snaps


XS, ACTS, RS, SNAPS = linucb_run()
_ref = run_study_tips("linucb", ROUNDS, seed=SEED, alpha=ALPHA)
assert (ACTS == _ref["actions"]).all() and (RS == _ref["rewards"]).all()     # the same run as the measured example
for _t in SNAP_AT:
    _theta = [np.linalg.solve(A, b) for A, b in zip(*SNAPS[_t])]
    assert np.allclose(_theta, EX["theta_hat_after"][str(_t)], atol=1e-4)


def fitted(t, k):
    A, b = SNAPS[t][0][k], SNAPS[t][1][k]
    return np.linalg.solve(A, b), np.linalg.inv(A)


SERVED = {t: np.bincount(ACTS[:t], minlength=N_TIPS) for t in SNAP_AT}
assert SERVED[10].tolist() == [6, 3, 1] and SERVED[2000].tolist() == [781, 271, 948]
assert np.allclose(fitted(10, FRIEND)[0], 0.0)                 # one student, no help: a flat line at zero
_g60 = (np.arange(60) + 0.5) / 60
_X = np.array([[a, b] for a in _g60 for b in _g60])               # students on a grid over the square
_OWN = np.argmax(_X @ THETA.T, axis=1)                             # each student's truly best tip


def own_error(t):
    """Mean |fitted - true| chance of helping, for each tip, over the students that tip is truly best for."""
    return np.array([np.abs(_X[_OWN == k] @ (fitted(t, k)[0] - THETA[k])).mean() for k in range(N_TIPS)])


OWN_ERR = {t: own_error(t) for t in SNAP_AT}
assert (OWN_ERR[2000] < OWN_ERR[10]).all() and OWN_ERR[2000].max() < 0.04   # "lands near the truth for the students it serves"

# the hook: two early students of the run, both given flashcards
TALKER, READER = 4, 6                                          # students 5 and 7
assert ACTS[TALKER] == FLASH and RS[TALKER] == 0.0 and XS[TALKER][1] > 0.9 and XS[TALKER][0] < 0.2   # "nothing"
assert ACTS[READER] == FLASH and RS[READER] == 1.0 and XS[READER][0] > 0.79 and XS[READER][1] < 0.05  # "plus one"
assert READER - TALKER == 2                                    # "two students later"

# one decision: Rover's data frozen after 200 students; three students from later in the same run
A200, B200 = SNAPS[200]
DECIDE = (200, 210, 211)                                       # students 201 (a reader), 211 (a talker), 212 (between)


def scores_200(x):
    s = linucb_scores(A200, B200, x, ALPHA)
    parts = []
    for A, b in zip(A200, B200):
        A_inv = np.linalg.inv(A)
        parts.append((float((A_inv @ b) @ x), float(ALPHA * np.sqrt(x @ A_inv @ x))))
    assert np.allclose(s, [line + bonus for line, bonus in parts])
    return s, parts


DECISIONS = [scores_200(XS[i]) for i in DECIDE]
assert [int(np.argmax(s)) for s, _ in DECISIONS] == [FLASH, FRIEND, PRACTICE]
assert [int(np.argmax(THETA @ XS[i])) for i in DECIDE] == [FLASH, FRIEND, PRACTICE]   # each is the right tip
assert int(np.argmax(DECISIONS[0][0])) == ACTS[200]                                    # as Rover chose in the run
_close = np.sort(DECISIONS[2][0])
assert _close[-1] - _close[-2] < 0.02                                                  # "by a hair"

# ---------------------------------------------------------------------------- measured on this Mac

HL = measured("classic", "headline")
METH = ST["methods"]
R_LIN, R_UCB = HL["tips_linucb_avg_reward"], HL["tips_ucb_avg_reward"]
R_AVG, R_ORC = HL["tips_best_on_average_avg_reward"], HL["tips_oracle_avg_reward"]
assert (METH["linucb_alpha1"]["avg_reward"], METH["ucb"]["avg_reward"]) == (R_LIN, R_UCB)
assert (METH["best_on_average"]["avg_reward"], METH["oracle"]["avg_reward"]) == (R_AVG, R_ORC)
assert (round(R_LIN, 3), round(R_UCB, 3), round(R_AVG, 3), round(R_ORC, 3)) == (0.527, 0.424, 0.449, 0.540)
assert [round(100 * v) for v in (R_UCB, R_AVG, R_LIN, R_ORC)] == [42, 45, 53, 54]     # spoken percentages
LIFT = R_LIN / R_AVG - 1
assert round(100 * LIFT) == 17                                                         # "seventeen percent more"
UCB_SHARE = METH["ucb"]["tip_share"]
assert UCB_SHARE[PRACTICE] < 0.5 < UCB_SHARE[PRACTICE] + 0.02   # it gives the best-on-average tip only half the time
REG = {k: METH[k]["regret"] for k in ("linucb_alpha1", "ucb", "best_on_average")}
REG_END = {k: METH[k]["regret_at_2000"] for k in REG}
assert REG_END["linucb_alpha1"] < 30 < 180 < REG_END["best_on_average"] < REG_END["ucb"]

# ---------------------------------------------------------------------------- Yahoo! (notes §4, Table 1)

CTR = {"random": 1.0, "eps": 1.596, "linucb": 1.795}       # relative CTR, deployment bucket, 100% data
YAHOO_LIFT = CTR["linucb"] / CTR["eps"] - 1
assert round(100 * YAHOO_LIFT, 1) == 12.5                    # "twelve point five percent"

# ---------------------------------------------------------------------------- offline replay, on Rover's own log


def rover200(x):
    """Rover frozen after 200 students: the tip with the highest LinUCB score."""
    return int(np.argmax(linucb_scores(A200, B200, x, ALPHA)))


LOG_N = 30_000
_lrng = np.random.default_rng(4)
LOG_X = _lrng.random((LOG_N, N_FEATURES))
LOG_TIP = _lrng.integers(N_TIPS, size=LOG_N)                  # tips handed out uniformly at random
LOG_R = (_lrng.random(LOG_N) < np.einsum("ij,ij->i", THETA[LOG_TIP], LOG_X)).astype(float)
LOG_PICK = np.array([rover200(x) for x in LOG_X])
KEPT = LOG_PICK == LOG_TIP
REPLAY = float(LOG_R[KEPT].mean())
_g = (np.arange(150) + 0.5) / 150                             # the policy's true help rate: a 150 x 150 midpoint grid
LIVE = float(np.mean([THETA[rover200(np.array([a, b]))] @ np.array([a, b]) for a in _g for b in _g]))
LOG_OWN = float(LOG_R.mean())
assert abs(KEPT.mean() - 1 / 3) < 0.02                        # "about one row in three"
assert abs(REPLAY - LIVE) < 0.01                              # "within one point"
assert LOG_OWN < 0.43 < LIVE
LOG_ROWS = 7                                                  # the rows drawn on screen: the log's first seven
assert KEPT[:LOG_ROWS].sum() >= 2 and (~KEPT[:LOG_ROWS]).sum() >= 2

# ---------------------------------------------------------------------------- A/B test vs bandit (computed sketch)

AB_P = (0.50, 0.60)                                           # for example: two versions of a tip
AB_RUNS, AB_STEPS = 2000, 1000
AB = bandit_testbed("thompson", runs=AB_RUNS, steps=AB_STEPS, seed=0, probs=AB_P)
AB_SHARE = AB["frac_optimal"]                                 # share of visitors sent to the better version
WORSE_AB = AB_STEPS * 0.5
WORSE_BANDIT = AB_STEPS * (1 - AB_SHARE.mean())
assert 100 < WORSE_BANDIT < 125                               # "about one visitor in nine"
assert np.isclose(AB["regret"][-1], WORSE_BANDIT * (AB_P[1] - AB_P[0]))
assert AB_SHARE[-1] > 0.95

# ---------------------------------------------------------------------------- look

STUDENT = recolored("user", STATE, TEAL_FILL, prefix=PREFIX)
ARTICLE = recolored("doc", ACTION, CORAL_FILL, prefix=PREFIX)
TIP_ICONS = ("ep04-flashcards", "ep04-practice", "ep04-friend")
LIN_C, UCB_C, AVG_C = STATE, VALUE, INK                       # the strategies' curves (labelled on screen)
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words, s):
    chars = [c for c in s if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def tip_label(k, size=24, icon_h=0.5, color=ACTION, name=None, weight=m.BOLD):
    pic = icon(TIP_ICONS[k], icon_h)
    words = text(name or SHORT[k], size, color, weight=weight)
    return m.VGroup(pic, words).arrange(m.RIGHT, buff=0.18)


def tip_card(k, width=4.2, height=1.0):
    box = card(width, height, ACTION, CARD, radius=0.16, stroke=2.5)
    lab = tip_label(k, 22, 0.55, INK, name=STUDY_TIPS[k])
    lab.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    return m.VGroup(box, lab)


TRACK = 1.05


def feature_row(label, value, left, y, size=20):
    """One feature of a student: a mono label, a TEAL bar in a track, the value. Positioned absolutely."""
    lab = code(label, size, MUTED)
    lab.move_to([left, y, 0]).align_to([left, 0, 0], m.LEFT)
    x0 = left + 1.15
    track = m.RoundedRectangle(corner_radius=0.09, width=TRACK, height=0.18, stroke_color=EDGE, stroke_width=1.5,
                               fill_color=BG, fill_opacity=1).move_to([x0 + TRACK / 2, y, 0])
    w = max(TRACK * float(value), 0.04)
    bar = m.RoundedRectangle(corner_radius=0.09, width=w, height=0.18, stroke_width=0, fill_color=STATE,
                             fill_opacity=1).move_to([x0 + w / 2, y, 0])
    val = code(f"{value:.2f}", size, STATE).move_to([x0 + TRACK + 0.42, y, 0])
    return m.VGroup(lab, track, bar, val)


def student_card(x, name, center, width=3.6, vec=True):
    """A student: icon, name, the two features as bars and, below, the context vector x."""
    c = np.array([*center, 0.0][:3], dtype=float)
    pic = icon(STUDENT, 0.7)
    title = code(name, 22, INK, weight=m.BOLD)
    head = m.VGroup(pic, title).arrange(m.RIGHT, buff=0.25)
    h = 2.75 if vec else 2.2
    box = card(width, h, STATE, CARD, radius=0.2, stroke=2.5).move_to(c)
    top = c[1] + h / 2
    head.move_to([c[0], top - 0.55, 0])
    left = c[0] - width / 2 + 0.3
    rows = m.VGroup(feature_row("reading", x[0], left, top - 1.25), feature_row("talking", x[1], left, top - 1.7))
    parts = [box, head, rows]
    if vec:
        parts.append(code(f"x = [{x[0]:.2f}, {x[1]:.2f}]", 22, STATE).move_to([c[0], top - 2.3, 0]))
    group = m.VGroup(*parts)
    group.box, group.head, group.rows = box, head, rows
    group.vec = parts[3] if vec else None
    return group


def place(plot, origin):
    """Move a kit Plot so its axes' origin sits at `origin` (and keep .coords right)."""
    delta = np.array([*origin, 0.0][:3], dtype=float) - plot.origin
    plot.shift(delta)
    plot.origin = plot.origin + delta
    return plot


def dashed(mob, n=24):
    return m.DashedVMobject(mob, num_dashes=n)


def recap_rows(words, subs):
    for s, w in zip(subs, words):
        s.next_to(w, m.DOWN, buff=0.4)
    base = max(s.get_center()[1] for s in subs)
    for s in subs:
        s.set_y(base)


# ---------------------------------------------------------------------------- the episode


class Ep04BanditsWithContext(Episode):
    number = 4

    def lesson(self):
        self.hook()
        self.context()
        self.lines()
        self.linucb()
        self.decision()
        self.measured_runs()
        self.yahoo()
        self.replay()
        self.ab_test()
        self.module_close()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("0 · Hook", notes=(
            "Episodes 2-3 looked for the one best machine (ROVER_ARMS; machine 4 pays 0.60). Now Rover recommends one of "
            f"STUDY_TIPS = {STUDY_TIPS} to each student. The two students are students 5 and 7 of the seeded LinUCB run "
            f"(run_study_tips('linucb', seed=130), the measured median-regret run): student 5 = {np.round(XS[TALKER], 2).tolist()} "
            f"(reading, talking) got flashcards and reward {RS[TALKER]:g}; student 7 = {np.round(XS[READER], 2).tolist()} got "
            f"flashcards and reward {RS[READER]:g}."))
        machines = m.VGroup(*[SlotMachine(str(i + 1), 0.72) for i in range(5)]).arrange(m.RIGHT, buff=0.5).move_to([0, 0.75, 0])
        best = pill("best", REWARD, GOLD_FILL, 20).next_to(machines[3], m.UP, buff=0.22)
        bot = rover(0.8).move_to([0, -2.0, 0])
        desk = [0.2, -0.35, 0]
        tips = m.VGroup(*[tip_card(k) for k in range(N_TIPS)]).arrange(m.DOWN, buff=0.3).move_to([4.35, -0.35, 0])
        tips_cap = code("three study tips", 22, MUTED).next_to(tips, m.UP, buff=0.3)
        newcomer = m.VGroup(icon(STUDENT, 1.0), code("a student", 22, MUTED)).arrange(m.DOWN, buff=0.2).move_to([-4.6, -0.35, 0])
        talker = student_card(XS[TALKER], "student 5", (-4.75, 1.0), vec=False)
        reader = student_card(XS[READER], "student 7", (-4.75, -1.55), vec=False)
        given = [icon(TIP_ICONS[FLASH], 0.55).move_to([talker.get_right()[0] + 0.75, talker.get_y() + 0.35, 0]),
                 icon(TIP_ICONS[FLASH], 0.55).move_to([reader.get_right()[0] + 0.75, reader.get_y() + 0.35, 0])]
        chips = [reward_chip(RS[TALKER], 24).move_to(given[0].get_center() + [0, -0.75, 0]),
                 reward_chip(RS[READER], 24).move_to(given[1].get_center() + [0, -0.75, 0])]
        source = tips[FLASH][1][0]

        self.say(
            "[In episodes two and three, Rover hunted for the one best machine.] [Today it has a new job]: "
            "[recommending study tips to students.]",
            m.AnimationGroup(self.new_header("00 · hook", "Who is asking?"), m.FadeIn(machines, lag_ratio=0.1),
                             m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(best, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeOut(machines, shift=0.3 * m.DOWN), m.FadeOut(best, shift=0.3 * m.DOWN), bot.animate.move_to(desk)),
            m.FadeIn(newcomer, shift=0.3 * m.RIGHT),
        )
        self.say(
            "[There are three tips]: [flashcards], [practice problems], [and explaining it to a friend.]",
            m.FadeIn(tips_cap),
            *[m.FadeIn(tips[k], shift=0.2 * m.LEFT) for k in range(N_TIPS)],
        )
        self.say(
            "[Early on, Rover gives flashcards to a student who loves to talk]: [nothing.]",
            m.AnimationGroup(m.FadeOut(newcomer), m.FadeIn(talker, shift=0.2 * m.RIGHT),
                             m.Succession(m.Wait(0.6), m.FadeIn(given[0], target_position=source, path_arc=0.6, run_time=1.2))),
            m.FadeIn(chips[0], scale=0.6),
        )
        self.say(
            "[Two students later, the same tip goes to a keen reader]: [plus one!]",
            m.AnimationGroup(m.FadeIn(reader, shift=0.2 * m.RIGHT),
                             m.Succession(m.Wait(0.6), m.FadeIn(given[1], target_position=source, path_arc=0.6, run_time=1.2))),
            m.AnimationGroup(m.FadeIn(chips[1], scale=0.6),
                             m.Flash(chips[1].get_center(), color=GOLD, flash_radius=0.45, line_length=0.2)),
        )
        scene_bits = m.VGroup(bot, tips, tips_cap, talker, reader, *given, *chips)
        halo = glow(STATE, radius=3.8).move_to([0, 0.1, 0])
        big = title_text("The best action can depend", 60, INK).move_to([0, 0.55, 0])
        small = title_text("on who is asking", 60, STATE).next_to(big, m.DOWN, buff=0.3)
        self.say(
            "[Same tip, different student, different result.] [The best action can depend on who is asking.]",
            m.LaggedStart(m.Circumscribe(m.VGroup(talker, given[0], chips[0]), color=STATE, buff=0.08),
                          m.Circumscribe(m.VGroup(reader, given[1], chips[1]), color=STATE, buff=0.08), lag_ratio=0.5),
            m.AnimationGroup(m.FadeOut(scene_bits, shift=0.3 * m.DOWN), m.FadeIn(halo, scale=0.6),
                             m.FadeIn(big, shift=0.2 * m.UP), m.FadeIn(small, shift=0.2 * m.UP), lag_ratio=0.15),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 1 · context

    def context(self):
        self.next_section("1 · Context", notes=(
            f"The context: two numbers per student, {STUDENT_FEATURES}, each uniform in [0, 1] (classic.md §3). For example "
            f"(STUDY_TIP_THETA, illustrative): P(helped) = theta_tip . x with theta = {THETA.tolist()}. Average student "
            f"(0.5, 0.5): {np.round(AVG, 2).tolist()}. Best tip for each share of uniform students: exactly 7/18, 4/18, 7/18 "
            f"= {np.round(100 * SHARE, 1).tolist()}% (classic.json study_tips.exact.best_tip_share agrees). Student 7 is "
            "from the seeded run (hook)."))
        stu = student_card(XS[READER], "student 7", (-4.75, 0.55))
        ctx_cap = code("the context", 20, STATE).next_to(stu, m.DOWN, buff=0.22)
        cols = {"reading": 1.75, "talking": 3.15, "avg": 4.5, "best": 5.85}
        head_y, row_y = 1.9, [1.1, 0.25, -0.6]
        name_x = -2.4
        heads = {
            "reading": code("reading", 20, MUTED).move_to([cols["reading"], head_y, 0]),
            "talking": code("talking", 20, MUTED).move_to([cols["talking"], head_y, 0]),
            "avg": m.VGroup(code("average", 20, MUTED), code("student", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to([cols["avg"], head_y + 0.12, 0]),
            "best": m.VGroup(code("best tip", 20, MUTED), code("for", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to([cols["best"], head_y + 0.12, 0]),
        }
        rule = m.Line([name_x, head_y - 0.38, 0], [6.5, head_y - 0.38, 0]).set_stroke(EDGE, 2)
        names = m.VGroup(*[tip_label(k, 20, 0.4, ACTION, name=STUDY_TIPS[k]).move_to([0, y, 0]).align_to([name_x, 0, 0], m.LEFT)
                           for k, y in zip(range(N_TIPS), row_y)])
        w_read = m.VGroup(*[code(f"{THETA[k][0]:.2f}", 24, INK).move_to([cols["reading"], y, 0]) for k, y in enumerate(row_y)])
        w_talk = m.VGroup(*[code(f"{THETA[k][1]:.2f}", 24, INK).move_to([cols["talking"], y, 0]) for k, y in enumerate(row_y)])
        avg = m.VGroup(*[code(f"{AVG[k]:.0%}", 24, REWARD, weight=m.BOLD).move_to([cols["avg"], y, 0]) for k, y in enumerate(row_y)])
        best = m.VGroup(*[code(f"{SHARE[k]:.0%}", 24, INK, weight=m.BOLD).move_to([cols["best"], y, 0]) for k, y in enumerate(row_y)])
        for k in range(N_TIPS):
            ref = baseline(names[k][1], STUDY_TIPS[k])
            for col in (w_read, w_talk, avg, best):
                s = col[k].text if hasattr(col[k], "text") else ""
                col[k].shift((ref - baseline(col[k], s)) * m.UP)
        formula = m.MathTex(r"\text{chance it helps}", "=", r"w_{\text{read}}", r"\cdot", r"\text{reading}", "+",
                            r"w_{\text{talk}}", r"\cdot", r"\text{talking}", font_size=36, color=INK)
        formula[0].set_color(REWARD)
        formula[4].set_color(STATE)
        formula[8].set_color(STATE)
        formula.move_to([2.15, -1.75, 0])
        example = pill("for example", MUTED, CARD, 20).next_to(formula, m.DOWN, buff=0.3)
        hi22 = m.SurroundingRectangle(best[PRACTICE], color=REWARD, buff=0.1, corner_radius=0.08)

        self.say(
            "[Before each choice, Rover sees a context]: [two numbers about the student]: "
            "[how much they like reading, and how much they like talking.]",
            m.AnimationGroup(self.new_header("01 · context", "A student is two numbers"), m.FadeIn(stu.box),
                             m.FadeIn(stu.head, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in stu.rows], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(stu.vec, target_position=stu.rows), m.FadeIn(ctx_cap)),
        )
        self.say(
            "[For example, say each tip's chance of helping is a straight line in those two numbers]: "
            "[one weight for reading], [one for talking.]",
            m.AnimationGroup(m.FadeIn(formula, shift=0.1 * m.UP), m.FadeIn(example), m.FadeIn(rule),
                             m.LaggedStart(*[m.FadeIn(n, shift=0.1 * m.RIGHT) for n in names], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeIn(heads["reading"]), m.LaggedStart(*[m.FadeIn(v) for v in w_read], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeIn(heads["talking"]), m.LaggedStart(*[m.FadeIn(v) for v in w_talk], lag_ratio=0.2)),
        )
        self.say(
            "[For the average student, practice problems win]: [forty-five percent, against forty.]",
            m.AnimationGroup(m.FadeIn(heads["avg"]), m.LaggedStart(*[m.FadeIn(v, shift=0.1 * m.UP) for v in avg], lag_ratio=0.2)),
            m.Circumscribe(avg[PRACTICE], color=REWARD, buff=0.1),
        )
        self.say(
            "[Yet practice problems are the best tip for only twenty-two percent of students]; "
            "[the other two are best for thirty-nine percent each.]",
            m.AnimationGroup(m.FadeIn(heads["best"]), m.FadeIn(best[PRACTICE], shift=0.1 * m.UP), m.Create(hi22)),
            m.AnimationGroup(m.FadeIn(best[FLASH], shift=0.1 * m.UP), m.FadeIn(best[FRIEND], shift=0.1 * m.UP)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 2 · one line per tip

    def lines(self):
        self.next_section("2 · One line per tip", notes=(
            "Students lined up from a pure reader x = (1, 0) to a pure talker x = (0, 1): x(s) = (1 - s, s). On that line "
            "each tip's chance of helping, theta_tip . x(s), is a straight line: flashcards 0.8 -> 0, practice 0.45 flat, "
            f"explaining 0 -> 0.8 (for example). The best tip switches at s = {CROSS[0]} and {CROSS[1]}. A context-free "
            "bandit can at best give everyone practice problems (the best tip on average); the gold area between the upper "
            "envelope and that flat line is the reward that knowing the student adds on this line."))
        plot = Plot((0, 1), (0, 1), width=5.6, height=3.5, y_ticks=[0, 0.5, 1], y_fmt="{:.0%}",
                    y_label="chance the tip helps")
        place(plot, (-5.75, -2.0))
        ends = [code("pure reader", 20, STATE).next_to(plot.coords(0, 0), m.DOWN, buff=0.22),
                code("pure talker", 20, STATE).next_to(plot.coords(1, 0), m.DOWN, buff=0.22)]
        mid = code("a bit of both", 20, MUTED).next_to(plot.coords(0.5, 0), m.DOWN, buff=0.22)
        for w in (*ends, mid):
            w.set_y(plot.coords(0, 0)[1] - 0.36)
        truth = [plot.line([0, 1], [THETA[k] @ slice_x(0), THETA[k] @ slice_x(1)], REWARD, 5) for k in range(N_TIPS)]
        tags = []
        for k in range(N_TIPS):
            end = plot.coords(1, THETA[k] @ slice_x(1))
            lab = tip_label(k, 22, 0.42, REWARD)
            lab.move_to(end).align_to([end[0] + 0.62, 0, 0], m.LEFT).shift(0.18 * m.UP if k == FLASH else 0 * m.UP)
            tags.append(lab)

        # the context card on the right
        cx, top = 4.75, 2.05
        box = card(3.6, 4.2, STATE, CARD, radius=0.2, stroke=2.5).move_to([cx, top - 2.1, 0])
        head = m.VGroup(icon(STUDENT, 0.6), code("a student", 22, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.22)
        head.move_to([cx, top - 0.5, 0])
        pick_cap = code("best tip", 20, MUTED).move_to([cx, top - 2.7, 0])
        frame = m.VGroup(box, head, pick_cap)
        left = cx - 1.5

        def dynamic(s, s0=S_READER):
            x = slice_x(s)
            rows = m.VGroup(feature_row("reading", x[0], left, top - 1.2), feature_row("talking", x[1], left, top - 1.65))
            vec = code(f"x = [{x[0]:.2f}, {x[1]:.2f}]", 22, STATE).move_to([cx, top - 2.15, 0])
            k = best_on_slice(s)
            pick = tip_label(k, 26, 0.5, ACTION).move_to([cx, top - 3.25, 0])
            vline = m.DashedLine(plot.coords(s, 0), plot.coords(s, 1), dash_length=0.09).set_stroke(STATE, 3)
            foot = m.Dot(plot.coords(s, 0), radius=0.09, color=STATE)
            pts = [s0] + [c for c in CROSS if s0 < c < s] + [max(s, s0 + 1e-3)]
            trace = m.VMobject().set_points_as_corners([plot.coords(p, envelope(p)) for p in pts]).set_stroke(ACTION, 8)
            dot = m.Dot(plot.coords(s, envelope(s)), radius=0.12, color=ACTION).set_z_index(3)
            return m.VGroup(rows, vec, pick, vline, foot, trace, dot)

        probe = dynamic(S_READER)

        def sweep(s_from, s_to, run_time=1.6):
            def step(mob, alpha):
                mob.become(dynamic(s_from + (s_to - s_from) * alpha))
            return m.UpdateFromAlphaFunc(probe, step, run_time=run_time, rate_func=m.smooth)

        flat_tag = m.VGroup(title_text("No context", 32, INK), text("one tip for everyone:", 22, MUTED),
                            tip_label(PRACTICE, 26, 0.5, ACTION)).arrange(m.DOWN, buff=0.22).move_to([cx, 0.1, 0])
        gain = plot.band(list(np.linspace(0, 1, 81)), [0.45] * 81, [max(envelope(s), 0.45) for s in np.linspace(0, 1, 81)],
                         REWARD, opacity=0.38)
        gain_tag = text("what knowing the student adds", 22, REWARD, weight=m.BOLD).move_to(plot.coords(0.5, 0.92))

        self.say(
            "[Now line the students up], [from pure readers on the left], [to pure talkers on the right.]",
            m.AnimationGroup(self.new_header("02 · lines", "One line per tip"), m.FadeIn(plot.axes)),
            m.FadeIn(ends[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(ends[1], shift=0.1 * m.UP), m.FadeIn(mid)),
        )
        self.say(
            "[Each tip's chance of helping is a straight line]: [flashcards fall], [explaining rises], [and practice stays flat.]",
            m.AnimationGroup(m.FadeIn(plot.ticks), m.FadeIn(plot.y_label)),
            m.AnimationGroup(m.Create(truth[FLASH]), m.FadeIn(tags[FLASH], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(truth[FRIEND]), m.FadeIn(tags[FRIEND], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(truth[PRACTICE]), m.FadeIn(tags[PRACTICE], shift=0.1 * m.LEFT)),
        )
        self.say(
            "[A reader arrives.] [Rover reads the context], [and picks the highest line: flashcards.]",
            m.AnimationGroup(m.FadeIn(frame), m.FadeIn(probe[0]), m.FadeIn(probe[1]), m.FadeIn(probe[3]), m.FadeIn(probe[4])),
            m.Circumscribe(probe[1], color=STATE, buff=0.08),
            m.AnimationGroup(m.FadeIn(probe[5]), m.FadeIn(probe[6], scale=0.5), m.FadeIn(probe[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[As the student changes, the best tip switches]: [to practice problems], [then to explaining it to a friend.]",
            sweep(S_READER, S_STOPS[0], 1.2),
            sweep(S_STOPS[0], S_STOPS[1], 1.3),
            sweep(S_STOPS[1], S_STOPS[2], 1.6),
        )
        self.say(
            "[A bandit that ignores the student gives everyone the same tip]: [at best, practice problems, the flat line.]",
            m.AnimationGroup(m.FadeOut(probe), m.FadeOut(frame), m.FadeIn(flat_tag, shift=0.1 * m.UP)),
            m.AnimationGroup(truth[PRACTICE].animate.set_stroke(width=9), m.Circumscribe(tags[PRACTICE], color=REWARD, buff=0.08)),
        )
        self.say(
            "[The gold area above it is what knowing the student is worth.]",
            m.AnimationGroup(m.FadeIn(gain), m.FadeIn(gain_tag, shift=0.1 * m.UP)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 3 · LinUCB

    def linucb(self):
        self.next_section("3 · LinUCB", notes=(
            "LinUCB (Li, Chu, Langford & Schapire 2010, disjoint model; linucb_scores): each tip keeps A = I + sum x x^T and "
            "b = sum r x over the students it served, fits theta = A^-1 b (ridge least squares) and scores "
            "theta . x + alpha sqrt(x^T A^-1 x), alpha = 1. Panels: the seeded run (seed 130, the median-regret run of 200, "
            f"classic.json study_tips.example) after {SNAP_AT} students, drawn along the reader-to-talker line; served "
            f"counts {[SERVED[t].tolist() for t in SNAP_AT]}. Gold dashed: the example's true lines. The fitted lines get "
            "close, not exact: at 2,000 students flashcards' line is (0.68, 0.18) against (0.80, 0) - its students are "
            "mostly readers, so it knows little about the corners it never serves. Mean |fitted - true| over the students "
            f"each tip is truly best for: after 10 students {np.round(OWN_ERR[10], 3).tolist()}, after 2,000 "
            f"{np.round(OWN_ERR[2000], 3).tolist()}."))
        origins = [(-5.9, 0.25), (-1.55, 0.25), (2.8, 0.25)]
        S = list(np.linspace(0, 1, 41))
        panels = []
        for k, org in enumerate(origins):
            plot = Plot((0, 1), (0, 1.4), width=2.9, height=1.85, y_ticks=[0, 1] if k == 0 else None)
            place(plot, org)
            truth = dashed(plot.line([0, 1], [THETA[k] @ slice_x(0), THETA[k] @ slice_x(1)], REWARD, 3), 18)
            ends = m.VGroup(code("reader", 20, MUTED).next_to(plot.coords(0, 0), m.DOWN, buff=0.16),
                            code("talker", 20, MUTED).next_to(plot.coords(1, 0), m.DOWN, buff=0.16))
            ends[1].align_to(ends[0], m.DOWN)
            title = tip_label(k, 22, 0.4, ACTION)
            title.move_to([plot.coords(0.5, 0)[0], org[1] - 0.82, 0])
            panels.append(dict(plot=plot, truth=truth, ends=ends, title=title, org=org))

        def fit_line(p, t, k):
            theta, _ = fitted(t, k)
            return p["plot"].line([0, 1], [theta @ slice_x(0), theta @ slice_x(1)], VALUE, 5).set_z_index(2)

        def bonus_band(p, t, k):
            theta, A_inv = fitted(t, k)
            lo = [theta @ slice_x(s) for s in S]
            hi = [v + ALPHA * np.sqrt(slice_x(s) @ A_inv @ slice_x(s)) for v, s in zip(lo, S)]
            band = p["plot"].band(S, lo, hi, EXPLORE, opacity=0.3)
            edge = p["plot"].line(S, hi, EXPLORE, 2.5)
            return m.VGroup(band, edge)

        def served(p, t, k):
            n = SERVED[t][k]
            return code(f"{n:,} served", 20, MUTED).next_to(p["title"], m.RIGHT, buff=0.25)

        fits = [fit_line(p, 10, k) for k, p in enumerate(panels)]
        bands = [bonus_band(p, 10, k) for k, p in enumerate(panels)]
        counts = [served(p, 10, k) for k, p in enumerate(panels)]
        for p, c in zip(panels, counts):
            m.VGroup(p["title"], c).move_to([p["plot"].coords(0.5, 0)[0], p["org"][1] - 0.82, 0])
        counts = [served(p, 10, k) for k, p in enumerate(panels)]

        src = ("for A, b in zip(A_list, b_list):  # each tip\n"
               "    A_inv = np.linalg.inv(A)\n"
               "    theta = A_inv @ b   # its line\n"
               "    bonus = alpha * np.sqrt(x @ A_inv @ x)\n"
               "    scores.append(theta @ x + bonus)")
        block = code_block(src, 20, title="rl_reference.py · linucb_scores (excerpt)")
        paint(block[1], src, {r"\btheta\b": VALUE, r"\bbonus\b": EXPLORE})
        block.move_to([-2.4, -2.35, 0])
        last_line = glyphs(block[1], "scores.append(theta @ x + bonus)")

        rx = 2.15
        name = pill("LinUCB · Li et al., 2010", ACTION, CORAL_FILL, 22)
        name.move_to([0, -1.25, 0]).align_to([rx, 0, 0], m.LEFT)
        bot = rover(0.42)
        counter = title_text("after 10 students", 30, INK)
        count_row = m.VGroup(bot, counter).arrange(m.RIGHT, buff=0.25)
        count_row.move_to([0, -1.95, 0]).align_to([rx, 0, 0], m.LEFT)

        def counter_to(t):
            new = title_text(f"after {t:,} students", 30, INK)
            new.move_to(counter.get_center()).align_to(counter, m.LEFT)
            return m.Transform(counter, new)

        legend_rows = []
        sw_truth = dashed(m.Line([0, 0, 0], [0.55, 0, 0]).set_stroke(REWARD, 3), 4)
        sw_fit = m.Line([0, 0, 0], [0.55, 0, 0]).set_stroke(VALUE, 5)
        sw_band = m.Rectangle(width=0.55, height=0.22).set_stroke(EXPLORE, 2).set_fill(EXPLORE, 0.3)
        for sw, words, col in ((sw_truth, "true line, hidden from Rover", REWARD), (sw_fit, "Rover's fitted line", VALUE),
                               (sw_band, "bonus, as in UCB (episode 3)", EXPLORE)):
            legend_rows.append(m.VGroup(sw, text(words, 20, col)).arrange(m.RIGHT, buff=0.2))
        legend = m.VGroup(*legend_rows).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        legend.move_to([rx, -2.95, 0]).align_to(count_row, m.LEFT)

        def state(t, run_time=1.0):
            anims = []
            for k, p in enumerate(panels):
                anims += [m.Transform(fits[k], fit_line(p, t, k)), m.Transform(bands[k], bonus_band(p, t, k)),
                          m.Transform(counts[k], served(p, t, k).move_to(counts[k]).align_to(counts[k], m.LEFT))]
            anims.append(counter_to(t))
            return m.AnimationGroup(*anims, run_time=run_time)

        frames = m.VGroup(*[m.VGroup(p["plot"], p["ends"], p["title"]) for p in panels])
        self.say(
            "[But Rover doesn't know these lines.] [It has to learn them, one student at a time.]",
            m.AnimationGroup(self.new_header("03 · LinUCB", "Learning the lines"), m.FadeIn(frames),
                             *[m.FadeIn(p["truth"]) for p in panels], m.FadeIn(legend_rows[0])),
            m.FadeIn(count_row, shift=0.1 * m.UP),
        )
        self.say(
            "[Each tip fits its own line to the students it has served]: [a small least-squares fit.]",
            m.AnimationGroup(*[m.Create(f) for f in fits], *[m.FadeIn(c) for c in counts]),
            m.FadeIn(legend_rows[1], shift=0.1 * m.UP),
        )
        self.say(
            "[And like UCB in episode three], [each adds a bonus where its data is thin.]",
            m.FadeIn(legend_rows[2], shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(b) for b in bands], lag_ratio=0.25),
        )
        self.say(
            "[Each tip's score is its line plus its bonus], [and Rover picks the highest.] [This is LinUCB, from twenty ten.]",
            m.FadeIn(block, shift=0.1 * m.UP),
            m.Circumscribe(last_line, color=ACTION, buff=0.06),
            m.FadeIn(name, shift=0.1 * m.UP),
        )
        self.say(
            "[After fifty students], [two hundred], [and two thousand, the bonuses shrink, and each line lands near the truth for the students it serves.]",
            state(50),
            state(200),
            state(2000, run_time=2.0),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 4 · one decision

    def decision(self):
        (s1, p1), (s2, p2), (s3, p3) = DECISIONS
        self.next_section("4 · One decision", notes=(
            "Rover's LinUCB data frozen after 200 students of the seeded run (seed 130); three later students of the same "
            f"run: 201 = {np.round(XS[200], 2).tolist()}, 211 = {np.round(XS[210], 2).tolist()}, 212 = "
            f"{np.round(XS[211], 2).tolist()}. Scores (line + bonus, alpha = 1, linucb_scores): {np.round(s1, 3).tolist()}, "
            f"{np.round(s2, 3).tolist()}, {np.round(s3, 3).tolist()} -> flashcards, explaining, practice problems; each is "
            "the truly best tip for that student. Same data, different context, different choice."))
        cards = [student_card(XS[i], f"student {i + 1}", (-4.85, 0.1)) for i in DECIDE]
        into = arrow([-3.0, 0.1, 0], [-2.25, 0.1, 0], STATE)
        row_y = [1.3, 0.1, -1.1]
        x_lab, x0, unit = -1.95, 0.55, 3.3
        labels = m.VGroup(*[tip_label(k, 22, 0.42, ACTION).move_to([0, y, 0]).align_to([x_lab, 0, 0], m.LEFT)
                            for k, y in zip(range(N_TIPS), row_y)])
        axis = m.Line([x0, row_y[0] + 0.5, 0], [x0, row_y[-1] - 0.5, 0]).set_stroke(FAINT, 2)

        def bars(parts):
            group = m.VGroup()
            for k, ((line, bonus), y) in enumerate(zip(parts, row_y)):
                a, b = x0, x0 + unit * line
                blue = m.Rectangle(width=max(abs(b - a), 0.02), height=0.36).set_stroke(VALUE, 2).set_fill(BLUE_FILL, 1)
                blue.move_to([(a + b) / 2, y, 0])
                c = b + unit * bonus
                violet = m.Rectangle(width=max(c - b, 0.02), height=0.36).set_stroke(EXPLORE, 2).set_fill(VIOLET_FILL, 1)
                violet.move_to([(b + c) / 2, y, 0])
                val = code(f"{line + bonus:.2f}", 22, INK).move_to([max(c, x0) + 0.5, y, 0])
                group.add(m.VGroup(blue, violet, val))
            return group

        def highlight(parts, k):
            right = max(x0 + unit * (sum(parts[j])) for j in range(N_TIPS)) + 1.0
            box = m.RoundedRectangle(corner_radius=0.12, width=right - x_lab + 0.25, height=0.72)
            box.set_stroke(ACTION, 3).set_fill(opacity=0).move_to([(x_lab - 0.2 + right) / 2, row_y[k], 0])
            tag = pill("pick", ACTION, CORAL_FILL, 20).next_to(box, m.RIGHT, buff=0.15)
            return m.VGroup(box, tag)

        b1, b2, b3 = bars(p1), bars(p2), bars(p3)
        h1, h2, h3 = highlight(p1, FLASH), highlight(p2, FRIEND), highlight(p3, PRACTICE)
        right_edge = max(h.get_right()[0] for h in (h1, h2, h3))
        assert right_edge < 6.55, right_edge
        note = cite("Rover's lines after 200 students · three later students of the same run").move_to([0.9, 2.35, 0])
        key = m.VGroup(
            m.Rectangle(width=0.4, height=0.24).set_stroke(VALUE, 2).set_fill(BLUE_FILL, 1), text("line", 22, VALUE),
            m.Rectangle(width=0.4, height=0.24).set_stroke(EXPLORE, 2).set_fill(VIOLET_FILL, 1), text("bonus", 22, EXPLORE),
            text("= the score", 22, INK),
        ).arrange(m.RIGHT, buff=0.15)
        key[2].shift(0.25 * m.RIGHT)
        key[3].shift(0.25 * m.RIGHT)
        key[4].shift(0.4 * m.RIGHT)
        key.move_to([1.9, -2.2, 0])

        self.say(
            "[Here's Rover after two hundred students.] [A keen reader walks in]: [flashcards score highest.]",
            m.AnimationGroup(self.new_header("04 · one choice", "Same data, another student"), m.FadeIn(note),
                             m.FadeIn(labels, lag_ratio=0.2), m.Create(axis), m.FadeIn(key)),
            m.AnimationGroup(m.FadeIn(cards[0], shift=0.2 * m.RIGHT), m.GrowArrow(into)),
            m.AnimationGroup(*[m.GrowFromEdge(r[0], m.LEFT) for r in b1], *[m.FadeIn(r[1]) for r in b1],
                             *[m.FadeIn(r[2]) for r in b1], m.Succession(m.Wait(0.6), m.FadeIn(h1))),
        )
        self.say(
            "[Next, a talker]: [now explaining it to a friend comes out on top.]",
            m.AnimationGroup(m.FadeOut(cards[0], shift=0.2 * m.DOWN), m.FadeIn(cards[1], shift=0.2 * m.DOWN), pulse(into, STATE)),
            m.AnimationGroup(m.Transform(b1, b2), m.Transform(h1, h2)),
        )
        self.say(
            "[Then a student in between gets practice problems, by a hair.] [Same lines, same data]: [only the student changed.]",
            m.Succession(m.AnimationGroup(m.FadeOut(cards[1], shift=0.2 * m.DOWN), m.FadeIn(cards[2], shift=0.2 * m.DOWN),
                                          pulse(into, STATE)),
                         m.AnimationGroup(m.Transform(b1, b3), m.Transform(h1, h3))),
            m.Circumscribe(m.VGroup(labels, b1), color=VALUE, buff=0.12),
            m.Circumscribe(cards[2], color=STATE, buff=0.08),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 5 · measured

    def measured_runs(self):
        self.next_section("5 · Measured", notes=(
            f"Measured on this Mac (classic.md §3): {ST['runs']} runs x {ST['students']:,} students, mean helped per student: "
            f"LinUCB (alpha = 1) {R_LIN}, UCB1 ignoring the student {R_UCB}, always practice problems {R_AVG}, the oracle "
            f"{R_ORC}. UCB1 gives practice problems only {UCB_SHARE[PRACTICE]:.0%} of the time (tip_share {UCB_SHARE}): the "
            f"three tips are close for the average student (0.40 / 0.45 / 0.40), so it keeps testing. Regret after 2,000: "
            f"{REG_END}. LinUCB vs always practice: +{100 * LIFT:.1f}% (classic.md: 'worth 17% more helped students'). "
            f"LinUCB gives the best tip {METH['linucb_alpha1']['pct_best_tip_last500']}% of the last 500 students."))
        base_y, scale, bw = -2.55, 5.6, 1.05
        xs = [-5.2, -3.6, -2.0]
        vals = [R_UCB, R_AVG, R_LIN]
        names = [("UCB1", "no context"), ("always", "practice"), ("LinUCB", "with context")]
        colors = [UCB_C, AVG_C, LIN_C]
        floor = m.Line([-6.0, base_y, 0], [-1.2, base_y, 0]).set_stroke(FAINT, 2)
        bars, nums, labs = [], [], []
        for x, v, (a, b), col in zip(xs, vals, names, colors):
            h = scale * v
            bar = m.Rectangle(width=bw, height=h).set_stroke(REWARD, 2).set_fill(GOLD_FILL, 1).move_to([x, base_y + h / 2, 0])
            bars.append(bar)
            nums.append(code(f"{v:.1%}", 22, REWARD, weight=m.BOLD).next_to(bar, m.UP, buff=0.15))
            labs.append(m.VGroup(code(a, 20, col, weight=m.BOLD), text(b, 20, MUTED)).arrange(m.DOWN, buff=0.08)
                        .move_to([x, base_y - 0.5, 0]))
        oracle_y = base_y + scale * R_ORC
        oracle = m.DashedLine([-5.85, oracle_y, 0], [xs[2] + bw / 2 + 0.05, oracle_y, 0], dash_length=0.1).set_stroke(REWARD, 2.5)
        oracle_lab = code(f"oracle {R_ORC:.1%}", 20, REWARD).next_to(oracle, m.UP, buff=0.12).align_to(oracle, m.LEFT)
        chart_cap = text("students helped, per student served", 20, MUTED).move_to([-3.6, 1.85, 0])
        source = cite(f"{ST['runs']} runs × {ST['students']:,} students · mean · measured on this Mac").move_to([-3.0, 2.35, 0])
        # +17%: the gap from always-practice up to LinUCB, at the LinUCB bar's right
        y_a, y_b = bars[1].get_top()[1], bars[2].get_top()[1]
        ax = xs[2] + bw / 2 + 0.3
        guide = m.DashedLine([xs[1] + bw / 2, y_a, 0], [ax + 0.1, y_a, 0], dash_length=0.08).set_stroke(REWARD, 2)
        lift_arrow = m.Arrow([ax, y_a, 0], [ax, y_b, 0], buff=0, color=REWARD, stroke_width=5,
                             max_tip_length_to_length_ratio=0.35)
        lift = title_text(f"+{100 * LIFT:.0f}%", 36, REWARD).next_to(lift_arrow, m.RIGHT, buff=0.15)
        lift_arrow = m.VGroup(guide, lift_arrow)

        plot = Plot((0, 2000), (0, 250), width=3.0, height=3.3, x_ticks=[0, 1000, 2000], y_ticks=[0, 100, 200],
                    x_fmt="{:,.0f}", x_label="students", y_label="regret: helps missed")
        place(plot, (1.0, -1.95))

        def curve(key, color):
            xs_, ys_ = REG[key]["x"], REG[key]["y"]
            return plot.line([0] + list(xs_), [0] + list(ys_), color, 4)

        curves = {k: curve(k, c) for k, c in (("linucb_alpha1", LIN_C), ("ucb", UCB_C), ("best_on_average", AVG_C))}

        def end_label(key, s, color):
            v = REG_END[key]
            lab = code(s.format(v), 20, color, weight=m.BOLD)
            return lab.next_to(plot.coords(2000, v), m.RIGHT, buff=0.55)

        l_lin = end_label("linucb_alpha1", "LinUCB {:.0f}", LIN_C)
        l_ucb = end_label("ucb", "UCB1 {:.0f}", UCB_C)
        l_avg = end_label("best_on_average", "practice {:.0f}", AVG_C)

        self.say(
            "[Does context pay?] [Each strategy served two thousand students, two hundred times over, on this Mac.]",
            m.AnimationGroup(self.new_header("05 · measured", "Two thousand students"), m.Create(floor), m.FadeIn(chart_cap),
                             m.FadeIn(m.VGroup(*labs), lag_ratio=0.2)),
            m.FadeIn(source, shift=0.1 * m.UP),
        )
        self.say(
            "[Ignoring the student, UCB helps forty-two percent]: [less than always giving practice problems, at forty-five], "
            "[because with tips this close, it keeps testing the other two.]",
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.DOWN), m.FadeIn(nums[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.DOWN), m.FadeIn(nums[1], shift=0.1 * m.UP)),
            m.Circumscribe(m.VGroup(bars[0], nums[0], labs[0]), color=UCB_C, buff=0.1),
        )
        self.say(
            "[With context, LinUCB helps fifty-three percent], [close to the fifty-four percent of an oracle that knows every line.]",
            m.AnimationGroup(m.GrowFromEdge(bars[2], m.DOWN), m.FadeIn(nums[2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(oracle), m.FadeIn(oracle_lab)),
        )
        self.say(
            "[Its regret flattens out], [while both context-free strategies keep losing, student after student.]",
            m.AnimationGroup(m.FadeIn(plot), m.Create(curves["linucb_alpha1"], run_time=1.6), m.FadeIn(l_lin)),
            m.AnimationGroup(m.Create(curves["ucb"], run_time=1.8), m.Create(curves["best_on_average"], run_time=1.8),
                             m.FadeIn(l_ucb), m.FadeIn(l_avg)),
        )
        self.say(
            "[Knowing who is asking helps seventeen percent more students than the best single tip.]",
            m.AnimationGroup(m.Create(lift_arrow), m.FadeIn(lift, shift=0.1 * m.UP)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 6 · Yahoo!

    def yahoo(self):
        self.next_section("6 · Yahoo! news", notes=(
            "Li, Chu, Langford & Schapire, 'A Contextual-Bandit Approach to Personalized News Article Recommendation', "
            "WWW 2010 (arXiv:1003.0146; notes §4). Yahoo! Front Page Today Module, the featured story slot; articles are "
            "the arms; users reduced from 1,193 binary features to 6 numbers (5 user clusters + a constant). Abstract: "
            "'over 33 million events' (§5.2.1 says about 36 million; Appendix A: quote the abstract); '12.5% click lift "
            "compared to a standard context-free bandit algorithm'. Table 1: linucb (disjoint), deployment bucket, 100% "
            f"data: relative CTR {CTR['linucb']} vs epsilon-greedy {CTR['eps']} (random = 1 by definition): "
            f"+{100 * YAHOO_LIFT:.2f}%. An offline replay estimate, not a live A/B result."))
        win = card(5.6, 4.6, EDGE, CARD, radius=0.18, stroke=2).move_to([-3.75, -0.25, 0])
        bar_h = 0.45
        top = win.get_top()[1]
        bar_line = m.Line([win.get_left()[0], top - bar_h, 0], [win.get_right()[0], top - bar_h, 0]).set_stroke(EDGE, 2)
        dots = m.VGroup(*[m.Dot(radius=0.06, color=c) for c in (RED, GOLD, GREEN)]).arrange(m.RIGHT, buff=0.12)
        dots.move_to([win.get_left()[0] + 0.45, top - bar_h / 2, 0])
        page = code("front page · Today module", 20, MUTED).move_to([win.get_x() + 0.3, top - bar_h / 2, 0])
        slot = card(4.6, 1.65, ACTION, CORAL_FILL, radius=0.14, stroke=2.5).move_to([win.get_x(), top - bar_h - 0.6 - 0.825, 0])
        slot_cap = code("featured story", 20, ACTION).move_to([0, top - bar_h - 0.3, 0]).align_to(slot, m.LEFT).shift(0.1 * m.RIGHT)
        arts = m.VGroup(*[icon(ARTICLE, 0.95) for _ in range(4)]).arrange(m.RIGHT, buff=0.35).move_to(slot)
        arm_cap = text("each article: an arm", 20, ACTION).next_to(slot, m.DOWN, buff=0.18)
        visitor = icon(STUDENT, 0.75).move_to([win.get_left()[0] + 0.85, win.get_bottom()[1] + 0.85, 0])
        cells = m.VGroup(*[m.Square(0.3).set_stroke(STATE, 2).set_fill(TEAL_FILL if i < 5 else STATE, 1) for i in range(6)])
        cells.arrange(m.RIGHT, buff=0.08).next_to(visitor, m.RIGHT, buff=0.35).shift(0.15 * m.UP)
        cells_cap = text("each visitor: 6 numbers", 20, STATE).next_to(cells, m.DOWN, buff=0.12).align_to(cells, m.LEFT)
        cite_paper = cite("Li, Chu, Langford & Schapire, 2010").next_to(win, m.DOWN, buff=0.2)

        # relative click rate (random = 1)
        x0, unit, rx = 0.9, 2.6, 0.9
        rows = [("random", CTR["random"], MUTED), ("ε-greedy, no context", CTR["eps"], INK), ("LinUCB", CTR["linucb"], LIN_C)]
        ys = [0.95, 0.2, -0.55]
        chart = []
        for (lab, v, col), y in zip(rows, ys):
            name = code(lab, 20, col, weight=m.BOLD).move_to([x0, y + 0.32, 0]).align_to([x0, 0, 0], m.LEFT)
            bar = m.Rectangle(width=unit * v, height=0.3).set_stroke(REWARD, 2).set_fill(GOLD_FILL, 1)
            bar.move_to([x0 + unit * v / 2, y - 0.05, 0])
            val = code(f"{v:.3f}", 20, REWARD).next_to(bar, m.RIGHT, buff=0.15)
            chart.append(m.VGroup(name, bar, val))
        chart_cap = text("click rate, relative to random", 20, MUTED).move_to([x0, 1.85, 0]).align_to([x0, 0, 0], m.LEFT)
        lift = title_text(f"+{100 * YAHOO_LIFT:.1f}%", 48, REWARD).move_to([x0, -1.75, 0]).align_to([x0, 0, 0], m.LEFT)
        lift_sub = m.VGroup(text("clicks, LinUCB vs context-free ε-greedy", 20, INK),
                            text("offline replay · over 33 million events", 20, MUTED))
        lift_sub.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).next_to(lift, m.DOWN, buff=0.18).align_to(lift, m.LEFT)
        table = cite("Table 1 · deployment bucket · all data").next_to(lift_sub, m.DOWN, buff=0.12).align_to(lift, m.LEFT)

        self.say(
            "[The same idea once picked the news on Yahoo's front page], [in a well-known paper from twenty ten.]",
            m.AnimationGroup(self.new_header("06 · in the wild", "Yahoo! front-page news"), m.FadeIn(win), m.FadeIn(bar_line),
                             m.FadeIn(dots), m.FadeIn(page), m.FadeIn(slot), m.FadeIn(slot_cap)),
            m.FadeIn(cite_paper, shift=0.1 * m.UP),
        )
        self.say(
            "[Each article was an arm], [and each visitor a context]: [six numbers summing up who they were.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(a, shift=0.15 * m.UP) for a in arts], lag_ratio=0.2), m.FadeIn(arm_cap)),
            m.FadeIn(visitor, shift=0.15 * m.RIGHT),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in cells], lag_ratio=0.12), m.FadeIn(cells_cap)),
        )
        self.say(
            "[Replayed on over thirty-three million logged events], [LinUCB's click rate came out twelve point five percent higher], "
            "[than context-free epsilon-greedy.]",
            m.AnimationGroup(m.FadeIn(chart_cap), m.FadeIn(chart[0][0]), m.GrowFromEdge(chart[0][1], m.LEFT), m.FadeIn(chart[0][2]),
                             m.FadeIn(chart[1][0]), m.GrowFromEdge(chart[1][1], m.LEFT), m.FadeIn(chart[1][2])),
            m.AnimationGroup(m.FadeIn(chart[2][0]), m.GrowFromEdge(chart[2][1], m.LEFT), m.FadeIn(chart[2][2]),
                             m.FadeIn(lift, shift=0.1 * m.UP), m.FadeIn(lift_sub), m.FadeIn(table)),
            m.Circumscribe(chart[1], color=INK, buff=0.08),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 7 · offline replay

    def replay(self):
        self.next_section("7 · Replay", notes=(
            "Offline replay (Li et al. 2010, Algorithm 3; Li, Chu, Langford & Wang 2011: 'provably unbiased'): log events "
            "whose arm was chosen uniformly at random; stream them through the new policy; keep an event only when the "
            "policy picks the logged arm (each survives with probability 1/K); the kept events are distributed as a live "
            f"run. Done here on Rover's own example: {LOG_N:,} students, tips at random (seed 4); the new policy is Rover "
            f"frozen after 200 students (LinUCB scores). Kept {KEPT.sum():,} ({KEPT.mean():.1%}); replay estimate "
            f"{REPLAY:.3f} vs the policy's true help rate {LIVE:.3f} (150 x 150 grid over students); the log's own rate "
            f"{LOG_OWN:.3f}. The table shows the log's first {LOG_ROWS} rows."))
        cols = {"student": -4.65, "tip": -2.2, "helped": -0.55, "pick": 1.15, "keep": 2.45}
        head_y = 1.75
        heads = m.VGroup(code("student x", 20, STATE).move_to([cols["student"], head_y, 0]),
                         code("random tip", 20, ACTION).move_to([cols["tip"], head_y, 0]),
                         code("helped", 20, REWARD).move_to([cols["helped"], head_y, 0]),
                         code("new policy", 20, ACTION).move_to([cols["pick"], head_y, 0]),
                         code("keep", 20, GREEN).move_to([cols["keep"], head_y, 0]))
        rule = m.Line([-6.1, head_y - 0.3, 0], [3.0, head_y - 0.3, 0]).set_stroke(EDGE, 2)
        rows = []
        for i in range(LOG_ROWS):
            y = head_y - 0.72 - 0.5 * i
            x = LOG_X[i]
            stu = code(f"[{x[0]:.2f}, {x[1]:.2f}]", 20, STATE).move_to([cols["student"], y, 0])
            tip = code(SHORT[LOG_TIP[i]], 20, ACTION).move_to([cols["tip"], y, 0])
            got = code(f"{LOG_R[i]:.0f}", 20, REWARD if LOG_R[i] else MUTED).move_to([cols["helped"], y, 0])
            pick = code(SHORT[LOG_PICK[i]], 20, ACTION).move_to([cols["pick"], y, 0])
            if KEPT[i]:
                mark = m.VMobject().set_points_as_corners([[-0.13, 0.0, 0], [-0.03, -0.11, 0], [0.15, 0.12, 0]])
                mark.set_stroke(GREEN, 5).move_to([cols["keep"], y, 0])
            else:
                mark = m.Line([-0.1, 0, 0], [0.1, 0, 0]).set_stroke(FAINT, 4).move_to([cols["keep"], y, 0])
            rows.append(dict(log=m.VGroup(stu, tip, got), pick=pick, mark=mark, kept=bool(KEPT[i])))
        more = code(f"… {LOG_N - LOG_ROWS:,} more rows", 20, MUTED).move_to([cols["student"], head_y - 0.72 - 0.5 * LOG_ROWS, 0])
        more.align_to(rows[0]["log"][0], m.LEFT)
        est_pill = pill("offline replay estimate", EXPLORE, VIOLET_FILL, 22).move_to([0.0, 2.35, 0])

        rx = 4.85
        box = card(3.0, 4.3, EDGE, CARD, radius=0.18, stroke=2).move_to([rx, -0.3, 0])

        def stat_rows(label, value, color, y, size=30):
            lab = text(label, 20, MUTED).move_to([rx, y + 0.3, 0])
            val = title_text(value, size, color).move_to([rx, y - 0.12, 0])
            return m.VGroup(lab, val)

        s_log = stat_rows("logged students", f"{LOG_N:,}", INK, 1.2)
        s_kept = stat_rows("rows kept, about 1 in 3", f"{KEPT.sum():,}", GREEN, 0.2)
        s_rep = stat_rows("replay says it helps", f"{REPLAY:.1%}", EXPLORE, -0.8)
        s_live = stat_rows("a live run would give", f"{LIVE:.1%}", REWARD, -1.8)
        dim_log = [r["log"] for r in rows if not r["kept"]]
        dim_pick = [r["pick"] for r in rows if not r["kept"]]

        self.say(
            "[That lift is an offline replay estimate, not a live test.] [Here's how a replay works.]",
            m.AnimationGroup(self.new_header("07 · replay", "Grading a policy on old logs"), m.FadeIn(est_pill, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(heads[:3]), m.Create(rule)),
        )
        self.say(
            "[First, hand out tips at random for a while], [and log every student, tip and result.]",
            m.LaggedStart(*[m.FadeIn(r["log"][:2], shift=0.1 * m.UP) for r in rows], m.FadeIn(more), lag_ratio=0.15),
            m.LaggedStart(*[m.FadeIn(r["log"][2]) for r in rows], lag_ratio=0.1),
        )
        self.say(
            "[Then stream the log past a new policy], [and keep only the rows where it picks the same tip.]",
            m.AnimationGroup(m.FadeIn(heads[3]), m.LaggedStart(*[m.FadeIn(r["pick"], shift=0.2 * m.LEFT) for r in rows], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(heads[4]), m.LaggedStart(*[m.FadeIn(r["mark"], scale=0.5) for r in rows], lag_ratio=0.1),
                             *[d.animate.set_opacity(0.3) for d in dim_log], m.FadeIn(box), m.FadeIn(s_log), m.FadeIn(s_kept)),
        )
        self.say(
            "[Because the log was random], [the kept rows look just like a live run]: [here, the replay lands within one point of the truth.]",
            m.AnimationGroup(m.Circumscribe(s_kept, color=GREEN, buff=0.08), *[d.animate.set_opacity(0.3) for d in dim_pick]),
            m.FadeIn(s_rep, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(s_live, shift=0.1 * m.UP), m.Circumscribe(m.VGroup(s_rep, s_live), color=REWARD, buff=0.1)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 8 · A/B test or bandit

    def ab_test(self):
        self.next_section("8 · A/B test vs bandit", notes=(
            f"For example: two versions of a tip helping {AB_P[0]:.0%} and {AB_P[1]:.0%} of students. A fixed 50/50 test "
            f"sends half of its {AB_STEPS:,} visitors to the worse version ({WORSE_AB:.0f}). A bandit (Thompson sampling, "
            f"bandit_testbed, {AB_RUNS:,} runs, seed 0, computed) sends {WORSE_BANDIT:.0f} on average; its share to the better "
            f"version climbs from {AB_SHARE[0]:.0%} to {AB_SHARE[-1]:.0%}. Scott 2015 (Applied Stochastic Models in Business "
            "and Industry): the cost of experimenting is mostly opportunity cost; bandits are more exposed to time-varying "
            "effects (weekday vs weekend users, §4.2); a fixed test answers 'which is better' with known power."))
        plot = Plot((0, AB_STEPS), (0, 0.7), width=6.0, height=3.5, x_ticks=[0, 500, 1000], y_ticks=[0, 0.25, 0.5],
                    y_fmt="{:.0%}", x_fmt="{:,.0f}", x_label="visitors", y_label="share sent to the worse version")
        place(plot, (-5.9, -1.95))
        ab_line = plot.line([0, AB_STEPS], [0.5, 0.5], INK, 5)
        ab_lab = code("A/B test", 20, INK, weight=m.BOLD).next_to(plot.coords(AB_STEPS, 0.5), m.UP, buff=0.15).align_to(
            plot.coords(AB_STEPS, 0), m.RIGHT)
        cost = plot.band([0, AB_STEPS], [0.0, 0.0], [0.5, 0.5], EXPLORE, opacity=0.22)
        cost_lab = text("the price of exploring", 22, EXPLORE, weight=m.BOLD).move_to(plot.coords(AB_STEPS * 0.6, 0.33))
        steps = np.arange(1, AB_STEPS + 1)
        pick = np.unique(np.concatenate([np.arange(0, 60), np.arange(60, AB_STEPS, 10), [AB_STEPS - 1]]))
        worse = 1 - AB_SHARE
        bandit = plot.line(list(steps[pick]), list(worse[pick]), ACTION, 5)
        bandit_area = plot.area(list(steps[pick]), list(worse[pick]), ACTION, opacity=0.3)
        bandit_lab = code("bandit", 20, ACTION, weight=m.BOLD).next_to(plot.coords(AB_STEPS * 0.88, worse[879]), m.UP, buff=0.15)
        rx = 4.25
        example = m.VGroup(text("for example: two versions", 20, MUTED),
                           m.VGroup(code("A", 22, INK, weight=m.BOLD), text(f"helps {AB_P[0]:.0%}", 20, REWARD)).arrange(m.RIGHT, buff=0.2),
                           m.VGroup(code("B", 22, INK, weight=m.BOLD), text(f"helps {AB_P[1]:.0%}", 20, REWARD)).arrange(m.RIGHT, buff=0.2))
        example.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([rx, 1.75, 0])
        worse_cap = text("visitors given the worse version", 20, MUTED).move_to([rx, 0.6, 0])
        w_ab = m.VGroup(code("A/B test", 22, INK), title_text(f"{WORSE_AB:.0f}", 36, EXPLORE)).arrange(m.RIGHT, buff=0.35)
        w_bd = m.VGroup(code("bandit", 22, ACTION), title_text(f"{WORSE_BANDIT:.0f}", 36, ACTION)).arrange(m.RIGHT, buff=0.35)
        for w, y in ((w_ab, 0.0), (w_bd, -0.65)):
            w.move_to([rx, y, 0])
            w[0].align_to([rx - 1.5, 0, 0], m.LEFT)
            w[1].align_to([rx + 1.4, 0, 0], m.RIGHT)
        comp = m.VGroup(cite(f"Thompson sampling · {AB_RUNS:,} runs"), cite("mean · computed here")).arrange(m.DOWN, buff=0.06)
        comp.move_to([rx, -1.3, 0])
        caveat_box = card(4.35, 1.6, EDGE, CARD, radius=0.16, stroke=2).move_to([rx, -2.65, 0])
        caveat = m.VGroup(text("a fixed split: a cleaner answer,", 20, INK), text("steadier from weekday to weekend", 20, INK),
                          cite("Scott, 2015")).arrange(m.DOWN, buff=0.1).move_to(caveat_box)

        self.say(
            "[The classic alternative is an A-B test]: [split the traffic fifty-fifty until the test ends.]",
            m.AnimationGroup(self.new_header("08 · A/B test", "Who pays for exploring?"), m.FadeIn(plot), m.FadeIn(example)),
            m.AnimationGroup(m.Create(ab_line), m.FadeIn(ab_lab)),
        )
        self.say(
            "[So half the visitors get the worse version the whole time]: [the price of exploring, paid in full.]",
            m.AnimationGroup(m.FadeIn(cost), m.FadeIn(cost_lab)),
            m.AnimationGroup(m.FadeIn(worse_cap), m.FadeIn(w_ab, shift=0.1 * m.UP)),
        )
        self.say(
            "[A bandit shifts traffic toward the winner as the evidence comes in]: [only about one visitor in nine got the weaker version.]",
            m.AnimationGroup(m.Create(bandit, run_time=2.2), m.FadeIn(bandit_area), m.FadeIn(bandit_lab)),
            m.AnimationGroup(m.FadeIn(w_bd, shift=0.1 * m.UP), m.FadeIn(comp)),
        )
        self.say(
            "[But a fixed split gives a cleaner answer], [and copes better when users change from weekday to weekend.]",
            m.AnimationGroup(m.FadeIn(caveat_box), m.FadeIn(caveat[0])),
            m.AnimationGroup(m.FadeIn(caveat[1]), m.FadeIn(caveat[2])),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 9 · closing Module 1

    def module_close(self):
        self.next_section("9 · Module 1 closes", notes=(
            "Module 1 (Episodes 1-4): the reward loop, explore or exploit, optimism and luck, context. In every bandit, the "
            "action affects only the immediate reward, never the next situation: the next student arrives whatever tip "
            "Rover gave (notes §4; Sutton & Barto §2.9: if actions also affect the next situation, 'then we have the full "
            "reinforcement learning problem'). In Rover's grid world (ROVER_WORLD) a move changes the cell Rover is in: "
            "the state. Module 2 (Episodes 5-7) adds states."))
        chips = m.VGroup(*[pill(f"{n} · {t}", REWARD, GOLD_FILL, 20) for n, t in
                           ((1, "the reward loop"), (2, "explore or exploit"), (3, "optimism and luck"), (4, "context"))])
        chips.arrange(m.RIGHT, buff=0.25)
        if chips.width > 12.6:
            chips.scale(12.6 / chips.width)
        chips.move_to([0, 2.05, 0])
        left = card(6.1, 3.75, EDGE, CARD, radius=0.2, stroke=2).move_to([-3.3, -0.85, 0])
        right = card(6.1, 3.75, STATE, CARD, radius=0.2, stroke=2).move_to([3.3, -0.85, 0])
        l_title = title_text("A bandit", 30).move_to([-3.3, 0.6, 0])
        r_title = title_text("A grid world", 30).move_to([3.3, 0.6, 0])
        bot = rover(0.6).move_to([-5.45, -0.9, 0])
        queue_x = [-3.95, -3.1, -2.25, -1.4]
        queue = [icon(STUDENT, 0.62).move_to([x, -0.9, 0]) for x in queue_x]
        newcomer = icon(STUDENT, 0.62).move_to([queue_x[-1], -0.9, 0])
        handed = icon(TIP_ICONS[FLASH], 0.42).move_to([-4.75, -0.35, 0])
        left_group = m.VGroup(left, l_title, bot, handed, *queue)
        l_cap = text("the next student comes anyway", 22, INK).move_to([-3.3, -2.25, 0])
        cells = cell_grid(ROVER_WORLD.height, ROVER_WORLD.width, size=0.62, gap=0.07, color=EDGE, fill=BG).move_to([3.3, -0.85, 0])

        def cell_at(state_):
            r, c = ROVER_WORLD.to_screen(state_)
            return cells[r][c]

        cell_at(CHARGER).set_fill(GOLD_FILL, 1).set_stroke(GOLD, 2.5)
        cell_at(PUDDLE).set_fill(RED_FILL, 1).set_stroke(RED, 2.5)
        cell_at(ROCK).set_fill("#2A2238", 1)
        grid = m.VGroup(cells, icon("charger", 0.4).move_to(cell_at(CHARGER)), icon("puddle", 0.4).move_to(cell_at(PUDDLE)),
                        icon("rock", 0.4).move_to(cell_at(ROCK)))
        start = ROVER_WORLD.start
        after = (start[0], start[1] + 1)
        assert after in ROVER_WORLD.states and after != ROCK
        r_bot = rover(0.3).move_to(cell_at(start)).set_z_index(4)
        mark = m.Square(0.62).set_stroke(STATE, 3).set_fill(STATE, 0.35).move_to(cell_at(start)).set_z_index(3)
        r_cap = text("each move changes the state", 22, INK).move_to([3.3, -2.25, 0])
        nxt = pill("Module 2 · states", STATE, TEAL_FILL, 24).move_to([0, -3.2, 0])

        def serve(enter):
            first = queue[0]
            return m.Succession(
                handed.animate(run_time=0.6).move_to(first.get_center() + np.array([0, 0.62, 0])),
                m.AnimationGroup(m.FadeOut(first, shift=0.4 * m.UP), m.FadeOut(handed, shift=0.4 * m.UP),
                                 *[q.animate.move_to([x, -0.9, 0]) for q, x in zip(queue[1:], queue_x)],
                                 m.FadeIn(enter, shift=0.4 * m.LEFT), run_time=0.8),
            )

        self.say(
            "[That closes Module one]: [the reward loop], [exploring], [optimism and luck], [and now context.]",
            self.new_header("09 · module 1", "Choices that don't change the world"),
            *[m.FadeIn(c, shift=0.15 * m.UP) for c in chips],
        )
        self.say(
            "[In all of it, Rover's choice never changed what came next]: [whatever tip it gives, the next student walks in all the same.]",
            m.FadeIn(left_group),
            m.Succession(serve(newcomer), m.Wait(0.1), m.FadeIn(l_cap, shift=0.1 * m.UP)),
        )
        self.say(
            "[In a grid world, a move changes where Rover stands], [and what it can do next.] [Module two adds states.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(r_title), m.FadeIn(grid), m.FadeIn(mark), m.FadeIn(r_bot),
                             m.Succession(m.Wait(0.8), m.AnimationGroup(r_bot.animate.move_to(cell_at(after).get_center()),
                                                                       mark.animate.move_to(cell_at(after).get_center()), run_time=0.8))),
            m.AnimationGroup(m.FadeIn(r_cap, shift=0.1 * m.UP), m.Indicate(mark, color=STATE, scale_factor=1.15)),
            m.FadeIn(nxt, shift=0.15 * m.UP),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 10 · recap

    def recap(self):
        self.next_section("10 · Recap", notes=(
            "Context (who is asking: x), lines (each tip's payoff as theta . x, fitted by least squares), optimism (a bonus "
            f"alpha sqrt(x^T A^-1 x) where data is thin). Measured: LinUCB {R_LIN:.1%} helped vs the best single tip "
            f"{R_AVG:.1%} vs UCB1 {R_UCB:.1%}; Yahoo! +12.5% (offline replay). Next: Episode 5, States, Actions, Returns."))
        halo = glow(STATE, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Context", 62, STATE), title_text("Lines", 62, VALUE), title_text("Optimism", 62, EXPLORE))
        words.arrange(m.RIGHT, buff=0.9).move_to([0, 0.9, 0])
        subs = m.VGroup(code("who is asking: x", 22, STATE), code("one line per tip", 22, VALUE),
                        code("a bonus where data is thin", 22, EXPLORE))
        recap_rows(words, subs)
        proof = code(f"2,000 students, measured: LinUCB {R_LIN:.1%} helped · best single tip {R_AVG:.1%}", 20, MUTED)
        proof.move_to([0, -1.55, 0])
        mascot = rover(0.7).move_to([0, -2.85, 0])

        self.say(
            "[To sum up]: [context], [lines], [and optimism.]",
            m.AnimationGroup(self.new_header("10 · recap", "Context, lines, optimism"), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.2 * m.UP),
            m.FadeIn(words[1], shift=0.2 * m.UP),
            m.FadeIn(words[2], shift=0.2 * m.UP),
        )
        self.say(
            "[See who is asking], [predict each tip with a line], [and add a bonus where you're unsure.]",
            m.FadeIn(subs[0], shift=0.1 * m.UP),
            m.FadeIn(subs[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(subs[2], shift=0.1 * m.UP), m.FadeIn(proof), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()
