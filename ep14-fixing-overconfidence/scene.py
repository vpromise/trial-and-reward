"""Trial and Reward · Episode 14 · Fixing Overconfidence (Module 4 · Deep Value Learning)

Max over noisy estimates overestimates; several fixes stack. Episode 13's DQN learned CartPole on this Mac, but late in
training it said the start was worth about 103, although no CartPole policy can be worth more than 1/(1 − 0.99) = 100.
Why: the max over ten honest-but-noisy estimates of zero comes out 1.54 σ too high on average (expected_max_of_noisy);
Q-learning's target is built on that max, and bootstrapping copies the error back (maximization bias, S&B §6.7). The
textbook's toy (S&B Example 6.7, computed here with maximization_bias: 10,000 runs, state B with 10 actions — the book
says "many"): Q-learning goes left up to ≈ 94% early on (the optimum with ε = 0.1 is 5%); Double Q-learning (two
tables: one chooses, the other judges) ≈ 7.1% over episodes 201–300. Double DQN reuses the frozen target network as the
judge; on CartPole (deep_rl.json overestimation block, 5 seeds each) it cut the late overestimate from +8.8 to +1.7,
broke the ceiling in 34% of checks instead of 79%, solved sooner (175k vs 252k steps) and finished higher (444 vs 283).
Dueling networks and prioritized replay; Rainbow's six ingredients, its median score and its ablations (cited).

Every number on screen is computed below from rl_reference.py and asserted, or loaded with measured(...), or cited.

Sources (research/notes.md §14, Appendix C row 14; facts dated October 2026):
- Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), §6.7 "Maximization Bias and Double
  Learning", Example 6.7 / Fig. 6.5 (ε = 0.1, α = 0.1, γ = 1, 10,000 runs; B's rewards ~ N(−0.1, 1); "many" actions in
  B — 10 here, as in rl_reference.maximization_bias); eq. 6.10 (Double Q-learning).
- van Hasselt (2010), "Double Q-learning", NeurIPS. van Hasselt, Guez & Silver (2016), "Deep Reinforcement Learning
  with Double Q-learning", AAAI ("overestimations were observed for DQN in all 49 tested Atari games").
- Wang et al. (2016), "Dueling Network Architectures for Deep RL", ICML: Q = V + (A − mean A) (eq. 9).
- Schaul et al. (2016), "Prioritized Experience Replay", ICLR: P(i) = p_i^α / Σ p_k^α, importance weights.
- Hessel et al. (2018), "Rainbow", AAAI: six components; Table 2 median human-normalised scores, 57 Atari games,
  no-ops regime (DQN 79%, DDQN 117%, Noisy DQN 118%, Prioritized DDQN 140%, Dueling DDQN 151%, Distributional DQN
  164%, Rainbow 223%); ablations (Fig. 3 and text): prioritized replay and multi-step "the two most crucial",
  distributional next, Noisy Nets helped, dueling and Double Q made little difference to the median (the authors'
  hypothesis: clipping the value distribution to [−10, 10] already counteracts overestimation).
- research/experiments/deep_rl.md "Episode 14 — overestimation, measured" + deep_rl.json overestimation / headline:
  every 5,000 steps the greedy policy played the same 10 start states; prediction = max_a Q(s0, a); late = evaluations
  after 300k env steps; 5 seeds per method (seeds 10–14).

Illustrative (computed, but chosen): the noisy bars are draws from default_rng(99) (section 2) and default_rng(1516)
(section 4), picked so that four draws look typical (their maxima average 1.56, the judge's readings 0.00); the
dueling numbers (V = 2, A = [1, 0, −1, 2]) and the four replay memories (|δ| = 2, 1, 0.5, 0.1) are notes §14's worked
examples; the replay picks are a draw from default_rng(3) with those probabilities.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep14"
K = 10                                        # actions per state in every illustration (and in B)

# ---------------------------------------------------------------------------- the max of noisy estimates

EMAX = expected_max_of_noisy(K)
assert abs(EMAX - 1.538753) < 1e-5 and f"{EMAX:.2f}" == "1.54"          # "one point five four standard deviations"

DRAWS = np.random.default_rng(99).normal(0.0, 1.0, (4, K))              # four draws of ten noisy zeros (illustrative)
MAX_I, MAX_V = DRAWS.argmax(1), DRAWS.max(1)
assert len(set(MAX_I.tolist())) == 4 and np.abs(DRAWS).max() < 2.6 and MAX_V.min() > 0.5
assert abs(MAX_V.mean() - EMAX) < 0.05

_two = np.random.default_rng(1516)
SET1, SET2 = _two.normal(0.0, 1.0, (4, K)), _two.normal(0.0, 1.0, (4, K))
PICK = SET1.argmax(1)
VERDICT = SET2[np.arange(4), PICK]
assert len(set(PICK.tolist())) == 4 and max(np.abs(SET1).max(), np.abs(SET2).max()) < 2.6
assert abs(VERDICT.mean()) < 0.01 and (VERDICT > 0).sum() == 2 and abs(SET1.max(1).mean() - EMAX) < 0.1

# ---------------------------------------------------------------------------- the toy (S&B Example 6.7), computed

MB = maximization_bias()                       # 10,000 runs × 300 episodes, ε = 0.1, α = 0.1, 10 actions in B, seed 0
RUNS, EPISODES, EPS = 10_000, 300, 0.1
QL_LEFT, DQ_LEFT = MB["q_learning"], MB["double_q"]
assert len(QL_LEFT) == EPISODES
PEAK, PEAK_EP = float(QL_LEFT.max()), int(QL_LEFT.argmax()) + 1
assert round(100 * PEAK) == 94 and PEAK_EP == 19                         # "up to ninety-four percent"
QL_LATE, DQ_LATE = float(QL_LEFT[200:].mean()), float(DQ_LEFT[200:].mean())
assert f"{100 * DQ_LATE:.1f}" == "7.1" and round(100 * QL_LATE) == 13     # "about seven percent" (episodes 201–300)
OPT = EPS / 2                                                             # left only when exploring picks it
assert OPT == 0.05
assert np.all(QL_LEFT[10:150] > DQ_LEFT[10:150])

# ---------------------------------------------------------------------------- CartPole, measured on this Mac

OV = measured("deep_rl", "overestimation")
OD, ODD = OV["dqn"], OV["double_dqn"]
CEIL = OD["value_ceiling"]
assert CEIL == 100.0 and abs(1 / (1 - 0.99) - CEIL) < 1e-9 and ODD["value_ceiling"] == CEIL
PRED, ACTUAL, GAP = OD["late_q_s0_pred_mean"], OD["late_actual_mean"], OD["late_overestimate_mean"]
PRED2, ACTUAL2, GAP2 = ODD["late_q_s0_pred_mean"], ODD["late_actual_mean"], ODD["late_overestimate_mean"]
assert (f"{PRED:.1f}", f"{ACTUAL:.1f}", f"{GAP:.1f}") == ("102.8", "94.0", "8.8")
assert (f"{PRED2:.1f}", f"{ACTUAL2:.1f}", f"{GAP2:.1f}") == ("99.5", "97.8", "1.7")
assert round(PRED) == 103                                                 # "about one hundred and three"
ABOVE, ABOVE2 = OD["late_fraction_q_above_ceiling"], ODD["late_fraction_q_above_ceiling"]
assert (round(100 * ABOVE), round(100 * ABOVE2)) == (79, 34)
assert OD["late_window"] == ODD["late_window"] == "evaluations after 300000 env steps"
XS = np.array(OD["x_steps"]) / 1000                                       # thousands of env steps
assert XS[0] == 5 and XS[-1] == 500 and OD["x_steps"] == ODD["x_steps"]   # "every five thousand steps"
assert "10 start states" in OD["note"]
Q_DQN, Q_DDQN = OD["q_s0_pred"], ODD["q_s0_pred"]
CROSS = int(np.argmax(np.array(Q_DQN["mean"]) > CEIL))                    # where DQN's mean prediction breaks 100
assert XS[CROSS] == 340 and all(v > CEIL for v in Q_DQN["mean"][CROSS:])
assert Q_DQN["mean"][0] < 5 and Q_DDQN["mean"][0] < 5                     # "both start near zero"
HL, HL2 = measured("deep_rl", "headline", "dqn"), measured("deep_rl", "headline", "double_dqn")
assert HL["n_solved"] == HL2["n_solved"] == 5
STEPS, STEPS2 = HL["steps_to_solve_mean"] / 1000, HL2["steps_to_solve_mean"] / 1000
FINAL, FINAL2 = HL["final_avg100_mean"], HL2["final_avg100_mean"]
assert (round(STEPS), round(STEPS2), round(FINAL), round(FINAL2)) == (252, 175, 283, 444)
SEEDS = 5
TRAJ = measured("trajectories", "policies", "dqn_cartpole")
assert TRAJ["length"] == 500 and len(TRAJ["states"]) >= 300

# ---------------------------------------------------------------------------- dueling and prioritized replay (notes §14)

V_S, ADV = 2.0, np.array([1.0, 0.0, -1.0, 2.0])
Q_DUEL = V_S + ADV - ADV.mean()
assert ADV.mean() == 0.5 and Q_DUEL.tolist() == [2.5, 1.5, 0.5, 3.5]
assert int(np.argmax(Q_DUEL)) == int(np.argmax(ADV))                      # the greedy action is unchanged
DELTA = np.array([2.0, 1.0, 0.5, 0.1])
ALPHA_PER = 0.6
P_PER = DELTA ** ALPHA_PER / (DELTA ** ALPHA_PER).sum()
assert [round(p, 3) for p in P_PER] == [0.442, 0.292, 0.193, 0.073]
PICKS = np.random.default_rng(3).choice(4, size=6, p=P_PER)               # six replays, illustrative
assert PICKS.tolist().count(0) >= 2 and 3 not in PICKS.tolist()

# ---------------------------------------------------------------------------- Rainbow (cited: Hessel et al. 2018, Table 2)

RAINBOW_PARTS = ["double Q", "prioritized replay", "dueling heads", "multi-step returns", "distributional values",
                 "noisy networks"]
MEDIANS = [("DQN", 79), ("Double DQN", 117), ("Noisy DQN", 118), ("Prioritized DDQN", 140), ("Dueling DDQN", 151),
           ("Distributional DQN", 164), ("Rainbow", 223)]                 # % of human, 57 games, no-ops
ABLATION = {"prioritized replay": ("most crucial", GOLD), "multi-step returns": ("most crucial", GOLD),
            "distributional values": ("next", INK), "noisy networks": ("helps", INK),
            "dueling heads": ("little", MUTED), "double Q": ("little", MUTED)}

# ---------------------------------------------------------------------------- look

ONE_C, TWO_C = VALUE, GREEN        # the single-max learners (Q-learning, DQN) and their double versions
LOCK = recolored("lock", VALUE, BLUE_FILL, prefix=PREFIX)
PAD = recolored("gamepad", REWARD, GOLD_FILL, prefix=PREFIX)


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


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


def part(words, parts, k):
    """The glyphs of part k of a colored() line."""
    start = sum(len(p) + 1 for p, _ in parts[:k])
    s = " ".join(p for p, _ in parts)
    skip = sum(c.isspace() for c in s[:start])
    width = sum(not c.isspace() for c in parts[k][0])
    return words[start - skip: start - skip + width]


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.RIGHT)


def noisy_bars(center, width, height, labels=True):
    """Ten BLUE estimate bars on a ±2.6σ scale, all starting at zero."""
    return Bars([0.0] * K, labels=[str(i + 1) for i in range(K)] if labels else None, top=2.6, bottom=-2.6,
                width=width, height=height, color=VALUE, show_values=False, gap=0.18, center=center)


def sigma_band(bars):
    """The VIOLET ±1σ band of noise behind the bars."""
    x0, x1 = bars.xs[0] - bars.bar_w / 2 - 0.15, bars.xs[-1] + bars.bar_w / 2 + 0.15
    y0, y1 = bars.level(-1), bars.level(1)
    band = m.Rectangle(width=x1 - x0, height=y1 - y0).set_stroke(width=0).set_fill(EXPLORE, 0.14)
    return band.move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0]).set_z_index(-2)


def truth_line(bars):
    x0, x1 = bars.xs[0] - bars.bar_w / 2 - 0.15, bars.xs[-1] + bars.bar_w / 2 + 0.15
    return m.DashedLine([x0, bars.level(0), 0], [x1, bars.level(0), 0], dash_length=0.1).set_stroke(INK, 2.5).set_z_index(2)


def outline(bars, i, v, color, width=4):
    r = bars._rect(i, v)
    box = m.Rectangle(width=bars.bar_w + 0.12, height=max(abs(bars.level(v) - bars.level(0)), 0.02) + 0.12)
    return box.move_to(r).set_stroke(color, width).set_fill(opacity=0).set_z_index(3)


def pointer(bars, i, v, color=ACTION):
    top = max(bars.level(max(v, 0.0)), bars.level(0))
    return m.Arrow([bars.xs[i], top + 0.72, 0], [bars.xs[i], top + 0.1, 0], buff=0, color=color, stroke_width=6,
                   max_tip_length_to_length_ratio=0.4)


def reading(bars, i, v, color=RED, prefix="max"):
    top = max(bars.level(max(v, 0.0)), bars.level(0))
    return left_at(code(f"{prefix} {minus(f'{v:+.2f}')}", 22, color, weight=m.BOLD), bars.xs[i] + 0.22, top + 0.48)


class Tally(m.VGroup):
    """A vertical σ axis (`level` maps σ to scene y), collecting one dot per draw to the right of the axis."""

    def __init__(self, level, x, title, color):
        super().__init__()
        self.level, self.x = level, x
        self.axis = m.Line([x, level(-2.4), 0], [x, level(2.4), 0]).set_stroke(FAINT, 3)
        self.ticks = m.VGroup()
        for t in (-2, -1, 0, 1, 2):
            y = level(t)
            lab = "0" if t == 0 else minus(f"{t:+d}σ")
            self.ticks.add(m.Line([x - 0.08, y, 0], [x + 0.08, y, 0]).set_stroke(FAINT, 2),
                           right_at(code(lab, 20, MUTED), x - 0.18, y))
        self.title = text(title, 20, color).next_to(self.axis, m.UP, buff=0.18)
        self.add(self.axis, self.ticks, self.title)

    def dot(self, k, v, color):
        return m.Dot([self.x + 0.3 + 0.2 * k, self.level(v), 0], radius=0.085, color=color).set_z_index(4)

    def line(self, v, color, x_to):
        y = self.level(v)
        return m.DashedLine([self.x + 0.12, y, 0], [x_to, y, 0], dash_length=0.08).set_stroke(color, 3)


def node_box(label, color=STATE, size=0.85):
    box = card(size, size, color, FILLS.get(color, CARD), radius=0.16, stroke=3)
    return m.VGroup(box, code(label, 28, INK, weight=m.BOLD).move_to(box))


def end_box(height=0.85):
    box = card(0.62, height, EDGE, CARD, radius=0.12, stroke=2.5)
    return m.VGroup(box, code("end", 20, MUTED).move_to(box))


def hbar(x0, y, length, height, color):
    return m.Rectangle(width=max(length, 0.02), height=height).set_stroke(color, 2).set_fill(FILLS.get(color, CARD), 1) \
        .move_to([x0 + max(length, 0.02) / 2, y, 0])


# ---------------------------------------------------------------------------- the episode


class Ep14FixingOverconfidence(Episode):
    number = 14

    def lesson(self):
        self.the_ceiling()
        self.luckiest_bar()
        self.the_toy()
        self.double_learning()
        self.double_dqn()
        self.measured_runs()
        self.more_fixes()
        self.rainbow()
        self.recap()

    # ------------------------------------------------------------------ 1 · the ceiling

    def the_ceiling(self):
        self.next_section("1 · Too good to be true", notes=(
            "Episode 13's DQN on CartPole, trained on this Mac (trajectories.json dqn_cartpole: the exported greedy policy "
            "balancing 500 steps; the first 300 replayed). Reward +1 per step, γ = 0.99 ⇒ no policy is worth more than "
            f"1/(1 − 0.99) = {CEIL:.0f} (a full 500-step episode collects 99.34). deep_rl.json overestimation.dqn: the mean "
            f"prediction max_a Q(s0, a) over all evaluations after 300k env steps, 10 start states, {SEEDS} seeds = "
            f"{PRED:.2f}. The gauge's axis is zoomed (80–110) so the 2.8 points over the ceiling are visible."))
        cp = CartPoleView(center=(-3.4, -0.85, 0), units_per_m=0.85, pole_units=1.6)
        states = TRAJ["states"][:300]
        cp.set_state(states[0][0], states[0][2])
        src = cite("episode 13 · trained on this Mac").move_to([-3.4, -1.45, 0])
        ask = m.MathTex(r"Q(s_0) = \;?", font_size=48, color=VALUE).move_to([-3.4, 1.75, 0])
        plus = reward_chip(1, 22).move_to([-1.3, 0.45, 0])
        per = text("per step", 20, REWARD).next_to(plus, m.DOWN, buff=0.12)
        sum_tex = m.MathTex(r"1 + 0.99 + 0.99^2 + \cdots", r"= \frac{1}{1 - 0.99}", r"= 100", font_size=36, color=INK)
        sum_tex[2].set_color(RED)
        sum_tex.move_to([-3.4, -2.7, 0])

        xg = 1.75

        def gy(v):
            return -2.3 + (v - 80) * 3.9 / 30

        axis = m.Line([xg, gy(80), 0], [xg, gy(111.5), 0]).set_stroke(FAINT, 3)
        ticks = m.VGroup()
        for v in (80, 90, 100, 110):
            ticks.add(m.Line([xg - 0.08, gy(v), 0], [xg, gy(v), 0]).set_stroke(FAINT, 2),
                      right_at(code(str(v), 20, MUTED), xg - 0.18, gy(v)))
        zone = m.Rectangle(width=4.3, height=gy(110) - gy(100)).set_stroke(width=0).set_fill(RED, 0.1)
        zone.move_to([xg + 2.15, (gy(110) + gy(100)) / 2, 0])
        zone_lab = right_at(text("impossible", 22, RED), xg + 4.15, gy(107.5))
        ceil_line = m.DashedLine([xg, gy(100), 0], [xg + 4.3, gy(100), 0], dash_length=0.1).set_stroke(RED, 3)
        ceil_lab = right_at(text("the ceiling: 100", 22, RED, weight=m.BOLD), xg + 4.15, gy(100) - 0.3)
        col_x, col_w = xg + 0.45, 0.46
        column = m.Rectangle(width=col_w, height=gy(min(PRED, CEIL)) - gy(80)).set_stroke(VALUE, 2).set_fill(BLUE_FILL, 1)
        column.move_to([col_x, (gy(80) + gy(min(PRED, CEIL))) / 2, 0])
        over = m.Rectangle(width=col_w, height=gy(PRED) - gy(CEIL)).set_stroke(RED, 2).set_fill(RED, 0.85)
        over.move_to([col_x, (gy(PRED) + gy(CEIL)) / 2, 0])
        pred_lab = left_at(colored([("predicted", VALUE), (f"{PRED:.1f}", VALUE)], 26, weight=m.BOLD), col_x + 0.42, gy(PRED) + 0.05)
        caption = cite(f"after 300k steps · {SEEDS} seeds · zoomed axis").move_to([xg + 2.1, -2.75, 0])

        self.say(
            "[In episode thirteen, a deep Q-network learned to play CartPole], [trained right here, on this Mac.]",
            m.AnimationGroup(self.new_header("01 · the ceiling", "Too good to be true"), m.FadeIn(cp),
                             m.Succession(m.Wait(0.5), cp.replay(states, 6.0))),
            m.FadeIn(src, shift=0.1 * m.UP),
        )
        self.say(
            "[So, what does it think the starting position is worth?]",
            m.FadeIn(ask, shift=0.15 * m.DOWN),
        )
        self.say(
            "[With one point per step, discounted by zero point nine nine], [no policy can ever be worth more than one hundred.]",
            m.AnimationGroup(m.FadeIn(plus, shift=0.15 * m.UP), m.FadeIn(per), m.FadeIn(sum_tex[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(sum_tex[1:], shift=0.1 * m.UP), m.Create(axis), m.FadeIn(ticks),
                             m.Create(ceil_line), m.FadeIn(ceil_lab), m.FadeIn(zone), m.FadeIn(zone_lab)),
        )
        self.say(
            "[Yet late in training, the network predicted about one hundred and three.]",
            m.Succession(m.GrowFromEdge(column, m.DOWN, run_time=1.4),
                         m.AnimationGroup(m.GrowFromEdge(over, m.DOWN), m.FadeIn(pred_lab, shift=0.1 * m.LEFT),
                                          m.FadeIn(caption), run_time=0.7)),
        )
        self.say(
            "[It was sure of reward it could never collect.] [Today: where that overconfidence comes from, and how to fix it.]",
            m.AnimationGroup(m.Circumscribe(m.VGroup(over, pred_lab), color=RED, buff=0.1),
                             m.Indicate(sum_tex[2], color=RED, scale_factor=1.15)),
            m.Circumscribe(ask, color=VALUE, buff=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the luckiest bar

    def luckiest_bar(self):
        self.next_section("2 · The max picks the luckiest", notes=(
            f"Ten actions whose true values are all 0; each estimate = 0 + N(0, 1) noise (illustrative draws from "
            f"default_rng(99); their maxima: {', '.join(f'{v:.2f}' for v in MAX_V)}). E[max of 10 independent N(0, 1)] = "
            f"{EMAX:.3f} σ (expected_max_of_noisy(10); notes §14 worked example 1: 0.564 σ for 2, 2.508 σ for 100) — "
            "every estimate is unbiased, the max is not. Q-learning's target r + γ max_a′ Q(s′, a′) inherits the bias, "
            "and bootstrapping passes it backwards (S&B 2018 §6.7: 'We call this maximization bias')."))
        bars = noisy_bars((-1.9, 0.25, 0), 7.4, 3.8)
        band = sigma_band(bars)
        truth = truth_line(bars)
        bot = rover(0.4).move_to([-6.1, 1.75, 0])
        cap_a = colored([("10 actions, each truly worth", MUTED), ("0", INK)], 20)
        cap_b = colored([("·", MUTED), ("noise ±1σ", EXPLORE)], 20)
        caps = m.VGroup(cap_a, cap_b).arrange(m.RIGHT, buff=0.2).move_to([-1.9, -2.48, 0])
        tally = Tally(bars.level, 3.55, "the max, draw by draw", RED)

        hls = [outline(bars, i, v, RED) for i, v in zip(MAX_I, MAX_V)]
        ptrs = [pointer(bars, i, v) for i, v in zip(MAX_I, MAX_V)]
        reads = [reading(bars, i, v) for i, v in zip(MAX_I, MAX_V)]
        dots = [tally.dot(k, v, RED) for k, v in enumerate(MAX_V)]
        hl, ptr, rd = hls[0], ptrs[0], reads[0]
        avg = tally.line(EMAX, RED, tally.x + 1.0)
        avg_big = title_text(f"+{EMAX:.2f}σ", 40, RED)
        left_at(avg_big, tally.x + 1.12, bars.level(EMAX) + 0.08)
        avg_sub = text("on average", 20, MUTED).next_to(avg_big, m.DOWN, buff=0.1).align_to(avg_big, m.LEFT)

        parts = [("target =", INK), ("r", REWARD), ("+ γ ·", INK), ("max", RED), ("Q(s′, a′)", VALUE)]
        target = colored(parts, 26, weight=m.BOLD)
        left_at(target, -6.3, -3.18)
        bias = pill("maximization bias", RED, RED_FILL, 22).next_to(target, m.RIGHT, buff=0.5)
        src = right_at(cite("Sutton & Barto, 2018, §6.7"), 6.5, -3.18)

        def redraw(k):
            return m.AnimationGroup(bars.set(DRAWS[k], run_time=0.7), m.Transform(hl, hls[k]), m.Transform(ptr, ptrs[k]),
                                    m.Transform(rd, reads[k]), m.Succession(m.Wait(0.5), m.FadeIn(dots[k], scale=0.4)),
                                    run_time=1.1)

        self.say(
            "[Picture ten actions, each worth exactly zero.]",
            m.AnimationGroup(self.new_header("02 · maximization bias", "The max picks the luckiest"),
                             m.FadeIn(bars), m.Create(truth), m.FadeIn(cap_a, shift=0.1 * m.UP)),
        )
        self.say(
            "[But Rover only has estimates]: [the truth, plus some noise.]",
            m.FadeIn(bot, shift=0.15 * m.DOWN),
            m.AnimationGroup(m.FadeIn(band), m.FadeIn(cap_b, shift=0.1 * m.UP), bars.set(DRAWS[0], run_time=1.0)),
        )
        self.say(
            "[The max picks whichever estimate got luckiest], [and reports it as the value.]",
            m.AnimationGroup(m.GrowArrow(ptr), m.Create(hl)),
            m.AnimationGroup(m.FadeIn(rd, shift=0.1 * m.LEFT), m.FadeIn(tally),
                             m.Succession(m.Wait(0.4), m.FadeIn(dots[0], scale=0.4))),
        )
        self.say(
            "[New noise, a new lucky winner], [and again, the max is too high.]",
            m.Succession(redraw(1), redraw(2)),
            redraw(3),
        )
        self.say(
            "[On average, the max of ten honest but noisy zeros] [is one point five four standard deviations too high.]",
            m.Create(avg),
            m.AnimationGroup(m.FadeIn(avg_big, shift=0.1 * m.LEFT), m.FadeIn(avg_sub)),
        )
        self.say(
            "[The target of Q-learning is built on exactly this max], "
            "[and bootstrapping copies the error backwards. Sutton and Barto call it maximization bias.]",
            m.AnimationGroup(m.FadeIn(target, shift=0.1 * m.UP),
                             m.Succession(m.Wait(0.6), m.Circumscribe(part(target, parts, 3), color=RED, buff=0.06))),
            m.AnimationGroup(m.FadeIn(bias, shift=0.1 * m.LEFT), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · the toy

    def toy_plot(self):
        return Plot((0, EPISODES), (0, 100), width=7.6, height=2.0, center=(-0.95, -1.72, 0),
                    x_ticks=[0, 100, 200], y_ticks=[0, 50, 100], x_label="episode (of 300)",
                    y_label=f"% of runs going left · {RUNS:,} runs, computed here", y_fmt="{:g}%")

    def toy_curve(self, plot, ys, color, width=4):
        return plot.line(np.arange(1, EPISODES + 1), 100 * np.asarray(ys), color, width)

    @staticmethod
    def end_label(plot, mob, slot):
        """Labels at the right end of the toy plot, stacked in fixed slots (0 = lowest) clear of the axis tip."""
        return left_at(mob, plot.coords(EPISODES, 0)[0] + 0.3, plot.coords(0, 0)[1] + 0.3 + 0.31 * slot)

    def the_toy(self):
        self.next_section("3 · A detour that looks lucky", notes=(
            "Sutton & Barto (2018) Example 6.7 / Fig. 6.5, computed with rl_reference.maximization_bias (seed 0): from A, "
            "right ends with reward 0; left goes to B (reward 0), where each of B's actions ends with a reward ~ N(−0.1, 1). "
            "The book says 'many' actions; 10 here (notes §14 worked example 3 uses 10 too). ε = 0.1, α = 0.1, γ = 1, Q starts "
            f"at 0, {RUNS:,} runs × {EPISODES} episodes. Q-learning goes left in {100 * PEAK:.1f}% of runs at episode "
            f"{PEAK_EP} (the book's figure: a peak near 95% around episode 20), still {100 * QL_LATE:.1f}% over episodes 201–300. "
            f"The optimum is ε/2 = {100 * OPT:.0f}% (left only when exploration picks it)."))
        y = 1.55
        tl, B, A, tr = end_box(1.15).move_to([-6.0, y, 0]), node_box("B").move_to([-2.6, y, 0]), \
            node_box("A").move_to([0.55, y, 0]), end_box().move_to([3.45, y, 0])
        bot = rover(0.36).next_to(A, m.UP, buff=0.12)
        go_right = arrow(A.get_right(), tr.get_left(), MUTED)
        lab_right = code("right · 0", 20, MUTED).next_to(go_right, m.DOWN, buff=0.12)
        go_left = arrow(A.get_left(), B.get_right(), MUTED)
        lab_left = code("left · 0", 20, MUTED).next_to(go_left, m.DOWN, buff=0.12)
        fan = m.VGroup(*[m.Line(B.get_left() + [-0.05, 0, 0], tl.get_right() + [0.05, d, 0]).set_stroke(ACTION, 2.5)
                         for d in np.linspace(-0.45, 0.45, K)])
        lab_fan = code(f"{K} actions", 20, ACTION, weight=m.BOLD).move_to([-4.6, y + 0.68, 0])
        lab_rew = text("random reward, −0.1 on average", 20, REWARD).move_to([-4.3, y - 0.8, 0])
        book = cite("the book says “many” actions · here, 10")
        left_at(book, 1.35, y - 0.8)
        src = right_at(cite("Sutton & Barto, 2018, Example 6.7"), 6.5, y + 0.85)
        wrong = pill("always a mistake", RED, RED_FILL, 20).move_to([-2.05, y + 0.78, 0])
        looks = pill("max looks positive", RED, RED_FILL, 20).move_to([-2.05, y + 0.78, 0])

        plot = self.toy_plot()
        self.plot3 = plot
        ql = self.toy_curve(plot, QL_LEFT, ONE_C, 4)
        peak = plot.dot(PEAK_EP, 100 * PEAK, ONE_C)
        peak_lab = left_at(code(f"peak {100 * PEAK:.0f}% · episode {PEAK_EP}", 20, ONE_C, weight=m.BOLD),
                           plot.coords(62, 0)[0], plot.coords(0, 100 * PEAK)[1])
        ql_lab = self.end_label(plot, colored([("Q-learning", ONE_C), (f"{100 * QL_LATE:.0f}%", ONE_C)], 20, weight=m.BOLD), 2)
        opt = plot.hline(100 * OPT, GOLD, 2.5)
        opt_lab = self.end_label(plot, colored([("5%:", GOLD), ("best possible", GOLD)], 20), 0)
        late = cite("labels: episodes 201–300")
        right_at(late, 6.5, plot.coords(0, 100)[1] + 0.25)
        self.toy_keep = (ql, ql_lab, opt, opt_lab)

        self.say(
            "[Here's a tiny world from the textbook. Rover starts in state A.] [Going right ends the episode, with nothing.]",
            m.AnimationGroup(self.new_header("03 · the toy", "A detour that looks lucky"), m.FadeIn(A, scale=0.8),
                             m.FadeIn(bot, shift=0.15 * m.DOWN), m.FadeIn(src)),
            m.AnimationGroup(m.GrowArrow(go_right), m.FadeIn(lab_right), m.FadeIn(tr, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Going left leads to state B, which has ten actions]: [each ends with a random reward, minus zero point one on average.]",
            m.AnimationGroup(m.GrowArrow(go_left), m.FadeIn(lab_left), m.FadeIn(B, shift=0.1 * m.RIGHT),
                             m.Succession(m.Wait(0.5), m.AnimationGroup(m.LaggedStart(*[m.Create(f) for f in fan], lag_ratio=0.08),
                                                                        m.FadeIn(lab_fan), m.FadeIn(tl)))),
            m.FadeIn(lab_rew, shift=0.1 * m.UP),
        )
        self.say(
            "[So left is always a mistake.] [The book doesn't say how many actions; we used ten.]",
            m.FadeIn(wrong, shift=0.1 * m.DOWN),
            m.FadeIn(book, shift=0.1 * m.UP),
        )
        self.say(
            "[Yet the max over ten noisy estimates in B looks positive], [so Q-learning heads left]: "
            "[in ten thousand runs, up to ninety-four percent of the time.]",
            m.AnimationGroup(m.FadeOut(wrong), m.FadeIn(looks, shift=0.1 * m.DOWN),
                             m.LaggedStart(*[m.Indicate(f, color=EXPLORE, scale_factor=1.0) for f in fan], lag_ratio=0.1)),
            m.AnimationGroup(go_left.animate.set_color(ACTION).set_stroke(width=8), bot.animate.move_to(A.get_top() + [-0.75, 0.3, 0]),
                             m.FadeIn(plot)),
            m.AnimationGroup(m.Create(ql, run_time=2.2), m.Succession(m.Wait(0.5), m.AnimationGroup(m.FadeIn(peak), m.FadeIn(peak_lab)))),
        )
        self.say(
            "[The best an epsilon-greedy Rover can do is five percent]: [it takes hundreds of episodes to unlearn the detour.]",
            m.AnimationGroup(m.Create(opt), m.FadeIn(opt_lab, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(ql_lab, shift=0.1 * m.LEFT), m.FadeIn(late), m.Circumscribe(ql_lab, color=ONE_C, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · double learning

    def double_learning(self):
        self.next_section("4 · One chooses, the other judges", notes=(
            "Double learning: two independent sets of estimates; set 1 picks argmax, set 2 says what that action is worth. "
            "Set 2's noise is independent of the choice, so E[Q2(argmax Q1)] = 0 exactly (notes §14 worked example 2). "
            f"Illustrative draws from default_rng(1516): set 1's maxima {', '.join(f'{v:.2f}' for v in SET1.max(1))}; set 2's "
            f"verdicts on those actions {', '.join(minus(f'{v:+.2f}') for v in VERDICT)} (mean {VERDICT.mean():+.3f}). Double "
            "Q-learning (van Hasselt 2010; S&B eq. 6.10): two tables, a coin decides which one learns each step; it chooses "
            "with its own argmax and evaluates with the other table. On the toy (maximization_bias, same 10,000 runs): "
            f"{100 * DQ_LATE:.1f}% left over episodes 201–300 vs the optimum {100 * OPT:.0f}%."))
        r1 = noisy_bars((-2.25, 1.05, 0), 6.6, 1.9, labels=False)
        r2 = noisy_bars((-2.25, -1.6, 0), 6.6, 1.9, labels=True)
        b1, b2 = sigma_band(r1), sigma_band(r2)
        t1, t2 = truth_line(r1), truth_line(r2)
        lab1 = m.VGroup(text("estimates 1", 22, INK, weight=m.BOLD), text("choose", 22, ACTION, weight=m.BOLD))
        lab2 = m.VGroup(text("estimates 2", 22, INK, weight=m.BOLD), text("judge", 22, VALUE, weight=m.BOLD))
        for lab, row in ((lab1, r1), (lab2, r2)):
            lab.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            left_at(lab, 1.55, row.level(0))
        tally = Tally(lambda v: -0.35 + 0.7 * v, 5.0, "the judge, draw by draw", VALUE)
        picks = [outline(r1, i, SET1[k][i], ACTION) for k, i in enumerate(PICK)]
        judged = [outline(r2, i, SET2[k][i], VALUE) for k, i in enumerate(PICK)]
        ptrs = [pointer(r1, i, SET1[k][i]) for k, i in enumerate(PICK)]
        links = [m.DashedLine([r1.xs[i], r1.level(-2.6) - 0.05, 0], [r2.xs[i], r2.level(2.6), 0], dash_length=0.07)
                 .set_stroke(ACTION, 3) for i in PICK]
        dots = [tally.dot(k, v, VALUE) for k, v in enumerate(VERDICT)]
        pk, jd, pt, ln = picks[0], judged[0], ptrs[0], links[0]
        max_line = tally.line(EMAX, RED, 6.5)
        max_lab = right_at(code(f"max +{EMAX:.2f}", 20, RED), 6.5, tally.level(EMAX) + 0.22)
        zero_line = tally.line(0.0, VALUE, 6.5)
        zero_lab = right_at(code("judge 0", 20, VALUE, weight=m.BOLD), 6.5, tally.level(0) - 0.24)

        def redraw(k):
            return m.AnimationGroup(r1.set(SET1[k], run_time=0.7), r2.set(SET2[k], run_time=0.7), m.Transform(pk, picks[k]),
                                    m.Transform(pt, ptrs[k]), m.Transform(ln, links[k]), m.Transform(jd, judged[k]),
                                    m.Succession(m.Wait(0.55), m.FadeIn(dots[k], scale=0.4)), run_time=1.15)

        self.say(
            "[The fix: keep two independent sets of estimates.]",
            m.AnimationGroup(self.new_header("04 · double learning", "One chooses, the other judges"),
                             m.FadeIn(m.VGroup(r1, b1, t1)), m.FadeIn(m.VGroup(r2, b2, t2)),
                             m.Succession(m.Wait(0.3), m.AnimationGroup(r1.set(SET1[0]), r2.set(SET2[0]))),
                             m.FadeIn(lab1[0]), m.FadeIn(lab2[0])),
        )
        self.say(
            "[One picks the action that looks best]; [the other says what it's worth.]",
            m.AnimationGroup(m.FadeIn(lab1[1], shift=0.1 * m.LEFT), m.GrowArrow(pt), m.Create(pk)),
            m.AnimationGroup(m.Create(ln), m.Create(jd), m.FadeIn(lab2[1], shift=0.1 * m.LEFT), m.FadeIn(tally),
                             m.Succession(m.Wait(0.5), m.FadeIn(dots[0], scale=0.4))),
        )
        self.say(
            "[The judge's noise has nothing to do with the choice], [so on average, the verdict is exactly right.]",
            m.Succession(redraw(1), redraw(2), redraw(3)),
            m.AnimationGroup(m.Create(zero_line), m.FadeIn(zero_lab), m.Create(max_line), m.FadeIn(max_lab)),
        )
        self.wait(0.3)
        self.clear()

        parts = [r"Q_1(s,a) \leftarrow Q_1(s,a) + \alpha \big[\, r + \gamma\,", r"Q_2\big(", r"s', \arg\max_{a'} Q_1(s',a')",
                 r"\big)", r"- Q_1(s,a) \,\big]"]
        rule = m.MathTex(*parts, font_size=38, color=INK)
        rule[1].set_color(VALUE)
        rule[3].set_color(VALUE)
        rule[2].set_color(ACTION)
        rule.move_to([0, 1.75, 0])
        if rule.width > 12.6:
            rule.scale(12.6 / rule.width)
        ch = text("table 1 chooses", 20, ACTION, weight=m.BOLD).next_to(rule[2], m.DOWN, buff=0.18)
        jg = text("table 2 judges", 20, VALUE, weight=m.BOLD).next_to(rule[1], m.DOWN, buff=0.18)
        if jg.get_right()[0] > ch.get_left()[0] - 0.25:
            jg.shift((ch.get_left()[0] - 0.25 - jg.get_right()[0]) * m.RIGHT)
        coin = left_at(text("two tables · each step, a coin picks which one learns", 20, MUTED), -6.3, 0.62)
        src = right_at(cite("van Hasselt, 2010 · Sutton & Barto, 2018, eq. 6.10"), 6.5, 0.62)
        plot = self.toy_plot()
        ql_old, ql_lab_old, opt_old, opt_lab_old = self.toy_keep
        ql = self.toy_curve(plot, QL_LEFT, ONE_C, 3).set_stroke(opacity=0.45)
        dq = self.toy_curve(plot, DQ_LEFT, TWO_C, 4)
        opt = plot.hline(100 * OPT, GOLD, 2.5)
        ql_lab = ql_lab_old.copy().set_opacity(0.6)
        dq_lab = self.end_label(plot, colored([("Double Q", TWO_C), (f"{100 * DQ_LATE:.1f}%", TWO_C)], 20, weight=m.BOLD), 1)
        opt_lab = opt_lab_old.copy()
        late = right_at(cite("labels: episodes 201–300"), 6.5, plot.coords(0, 100)[1] + 0.25)

        self.say(
            "[That's Double Q-learning, from twenty ten]: [two tables, and each step, one of them learns.]",
            m.AnimationGroup(m.FadeIn(rule, shift=0.1 * m.UP), m.FadeIn(src),
                             m.Succession(m.Wait(0.8), m.AnimationGroup(m.FadeIn(ch, shift=0.1 * m.UP), m.FadeIn(jg, shift=0.1 * m.UP)))),
            m.FadeIn(coin, shift=0.1 * m.UP),
        )
        self.say(
            "[On the toy, it barely takes the bait]: [about seven percent left over the last hundred episodes.]",
            m.AnimationGroup(m.FadeIn(m.VGroup(plot, ql, ql_lab, opt, opt_lab, late)), m.Create(dq, run_time=2.0)),
            m.AnimationGroup(m.FadeIn(dq_lab, shift=0.1 * m.LEFT), m.Circumscribe(dq_lab, color=TWO_C, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · double DQN

    def double_dqn(self):
        self.next_section("5 · Double DQN", notes=(
            "A DQN already has two networks (Episode 13): the online network, trained every few steps, and the target "
            "network, a frozen copy refreshed every 1,000 steps in our runs. DQN target: y = r + γ max_a Q_target(s′, a) — the "
            "frozen copy both chooses and evaluates. Double DQN (van Hasselt, Guez & Silver 2016): y = r + γ Q_target(s′, "
            "argmax_a Q_online(s′, a)); no new network, only the target changes. Code: research/experiments/scripts/"
            "deep_dqn.py, the 'double' branch, verbatim names (q = online, qt = target, idx = the sampled batch). The paper: "
            "'overestimations were observed for DQN in all 49 tested Atari games'."))
        on = NeuralNet(layers=(4, 6, 6, 2), width=2.4, height=1.25, center=(-4.85, 1.45, 0), color=EXPLORE)
        tg = NeuralNet(layers=(4, 6, 6, 2), width=2.4, height=1.25, center=(-4.85, -1.15, 0), color=EXPLORE)
        on_lab = text("online network", 22, INK, weight=m.BOLD).next_to(on, m.DOWN, buff=0.22)
        tg_lab = text("target network", 22, INK, weight=m.BOLD).next_to(tg, m.DOWN, buff=0.22)
        on_sub = text("learns as it plays", 20, MUTED).next_to(on_lab, m.DOWN, buff=0.08)
        tg_sub = text("a frozen copy · episode 13", 20, MUTED).next_to(tg_lab, m.DOWN, buff=0.08)
        on_role = text("chooses", 22, ACTION, weight=m.BOLD).move_to(on_sub)
        tg_role = text("judges", 22, VALUE, weight=m.BOLD).move_to(tg_sub)
        lock = icon(LOCK, 0.42).move_to(tg.get_corner(m.UR) + [0.2, 0.15, 0])

        x0 = -2.2
        p1 = pill("DQN", ONE_C, BLUE_FILL, 20)
        left_at(p1, x0, 2.15)
        n1 = text("the frozen copy chooses and judges", 20, MUTED).next_to(p1, m.RIGHT, buff=0.3)
        f1_parts = [("y =", INK), ("r", REWARD), ("+ γ ·", INK), ("max", RED), ("Q_target(s′, ·)", VALUE)]
        f1 = left_at(colored(f1_parts, 23, weight=m.BOLD), x0, 1.52)
        p2 = pill("Double DQN", TWO_C, GREEN_FILL, 20)
        left_at(p2, x0, 0.72)
        n2 = m.VGroup(text("the online net chooses", 20, MUTED, t2c={"chooses": ACTION}),
                      text("· the frozen copy judges", 20, MUTED, t2c={"judges": VALUE})).arrange(m.RIGHT, buff=0.15)
        n2[1].shift((n2[0][0].get_bottom()[1] - n2[1][1].get_bottom()[1]) * m.UP)   # share the baseline of "the"
        n2.next_to(p2, m.RIGHT, buff=0.3)
        f2_parts = [("y =", INK), ("r", REWARD), ("+ γ ·", INK), ("Q_target(s′,", VALUE), ("argmax Q(s′, ·)", ACTION),
                    (")", VALUE)]
        f2 = left_at(colored(f2_parts, 23, weight=m.BOLD), x0, 0.09)
        src_code = ('a_star = np.argmax(q.forward(s2), 1)\n'
                    'v2 = qt.forward(s2)[np.arange(len(idx)), a_star]\n'
                    'y = r_b + gamma * (1.0 - d_b) * v2')
        blk = code_block(src_code, size=19, title="deep_dqn.py · the Double DQN target", accent=TWO_C)
        left_at(blk, x0, -1.5)
        blk[1].set_z_index(2)
        paint(blk[1], src_code, {r"q\.forward\(s2\)": ACTION, r"qt\.forward\(s2\)": VALUE})
        atari = m.VGroup(icon(PAD, 0.42), text("the DQN overestimated in all 49 Atari games tested", 22, INK),
                         cite("van Hasselt, Guez & Silver, 2016"))
        atari.arrange(m.RIGHT, buff=0.3)
        left_at(atari, -6.3, -3.12)

        self.say(
            "[A deep Q-network already has a second set of estimates]: [the frozen target network from episode thirteen.]",
            m.AnimationGroup(self.new_header("05 · double DQN", "Two networks, two jobs"),
                             m.FadeIn(on), m.FadeIn(on_lab), m.FadeIn(on_sub), m.Succession(m.Wait(0.4), on.forward())),
            m.AnimationGroup(m.FadeIn(tg), m.FadeIn(tg_lab), m.FadeIn(tg_sub), m.FadeIn(lock, scale=0.6)),
        )
        self.say(
            "[The usual target lets that frozen network both choose the next action and judge it.]",
            m.AnimationGroup(m.FadeIn(p1, shift=0.1 * m.RIGHT), m.FadeIn(n1), m.FadeIn(f1, shift=0.1 * m.UP),
                             m.Succession(m.Wait(1.0), m.Circumscribe(part(f1, f1_parts, 3), color=RED, buff=0.06))),
        )
        self.say(
            "[The Double DQN splits the jobs]: [the online network chooses], [the frozen one judges.]",
            m.AnimationGroup(m.FadeIn(p2, shift=0.1 * m.RIGHT), m.FadeIn(f2, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeOut(on_sub), m.FadeIn(on_role), on.columns.animate.set_stroke(ACTION),
                             m.FadeIn(n2[0]), m.Circumscribe(part(f2, f2_parts, 4), color=ACTION, buff=0.06)),
            m.AnimationGroup(m.FadeOut(tg_sub), m.FadeIn(tg_role), tg.columns.animate.set_stroke(VALUE),
                             m.FadeIn(n2[1]), m.Circumscribe(part(f2, f2_parts, 3), color=VALUE, buff=0.06)),
        )
        self.say(
            "[No new network: only the target changes, in two lines of code.]",
            m.FadeIn(blk, shift=0.1 * m.UP),
        )
        self.say(
            "[Its authors found overestimation in all forty-nine Atari games they tested.]",
            m.FadeIn(atari, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · measured on CartPole

    def measured_runs(self):
        self.next_section("6 · Measured on this Mac", notes=(
            f"deep_rl.json overestimation (Episode 14 block of deep_rl.md): every 5,000 env steps the greedy policy played the "
            f"same 10 start states; predicted = max_a Q(s0, a), collected = the discounted return (γ = 0.99) it then earned "
            f"(at most 99.34 in 500 steps). Curves: mean over {SEEDS} seeds, band = min–max. Late window = evaluations after "
            f"300k steps: DQN predicted {PRED:.2f}, collected {ACTUAL:.2f}, overestimate {GAP:.2f}; Double DQN {PRED2:.2f} / "
            f"{ACTUAL2:.2f} / {GAP2:.2f}; predictions above 100: {100 * ABOVE:.1f}% vs {100 * ABOVE2:.1f}%. Early on both "
            "UNDERestimate (Q starts near 0). Headline: solved (mean of the last 100 training episodes ≥ 475) in 5/5 seeds each "
            f"after {STEPS:.0f}k ± {HL['steps_to_solve_std'] / 1000:.0f}k vs {STEPS2:.0f}k ± {HL2['steps_to_solve_std'] / 1000:.0f}k "
            f"env steps; final avg100 {FINAL:.1f} ± {HL['final_avg100_std']:.1f} vs {FINAL2:.1f} ± {HL2['final_avg100_std']:.1f}. "
            "Five seeds: an honest but small sample."))
        plot = Plot((0, 500), (0, 120), width=5.3, height=3.2, center=(-3.45, -0.35, 0), x_ticks=[0, 250, 500],
                    y_ticks=[0, 50, 100], x_label="environment steps", y_label="predicted value of the start",
                    x_fmt="{:g}k")
        zero_k = plot.ticks[1]                                   # "0k" reads oddly: plain "0"
        plot.ticks.remove(zero_k)
        plot.ticks.add(code("0", 20, MUTED).move_to(zero_k.get_center()))
        top = right_at(cite(f"trained on this Mac · {SEEDS} seeds each · mean, band = min–max"), 6.5, 2.35)
        ceil = plot.hline(CEIL, RED, 2.5)
        ceil_lab = left_at(code("ceiling 100", 20, RED, weight=m.BOLD), plot.coords(12, 0)[0], plot.coords(0, CEIL)[1] + 0.2)
        band1 = plot.band(XS, Q_DQN["min"], Q_DQN["max"], ONE_C, 0.16)
        band2 = plot.band(XS, Q_DDQN["min"], Q_DDQN["max"], TWO_C, 0.16)
        c1 = plot.line(XS, Q_DQN["mean"], ONE_C, 4)
        c1_red = plot.line(XS[CROSS - 1:], Q_DQN["mean"][CROSS - 1:], RED, 5)
        c2 = plot.line(XS, Q_DDQN["mean"], TWO_C, 4)
        xe = plot.coords(500, 0)[0] + 0.2
        l1 = left_at(code("DQN", 20, ONE_C, weight=m.BOLD), xe, plot.coords(0, Q_DQN["mean"][-1])[1] + 0.16)
        l2 = left_at(code("Double", 20, TWO_C, weight=m.BOLD), xe, plot.coords(0, Q_DDQN["mean"][-1])[1] - 0.2)
        climb = left_at(text("both start near 0 and climb", 20, MUTED), plot.coords(150, 0)[0], plot.coords(0, 32)[1])

        # the dumbbell: predicted vs collected, late in training
        xa = 1.75

        def dy(v):
            return -2.5 + (v - 90) * 4.1 / 15

        d_axis = m.Line([xa, dy(90), 0], [xa, dy(105.5), 0]).set_stroke(FAINT, 3)
        d_ticks = m.VGroup()
        for v in (90, 95, 100, 105):
            d_ticks.add(m.Line([xa - 0.08, dy(v), 0], [xa, dy(v), 0]).set_stroke(FAINT, 2),
                        right_at(code(str(v), 20, MUTED), xa - 0.18, dy(v)))
        d_ceil = m.DashedLine([xa, dy(CEIL), 0], [6.35, dy(CEIL), 0], dash_length=0.1).set_stroke(RED, 2.5)
        d_ceil_lab = right_at(code("ceiling", 20, RED, weight=m.BOLD), 6.35, dy(CEIL) + 0.22)
        legend = m.VGroup(m.Dot(radius=0.09, color=VALUE), text("predicted", 20, MUTED), m.Dot(radius=0.09, color=REWARD),
                          text("collected", 20, MUTED)).arrange(m.RIGHT, buff=0.15)
        legend[2].shift(0.25 * m.RIGHT)
        legend[3].shift(0.25 * m.RIGHT)
        left_at(legend, 2.35, 1.85)
        late = cite("after 300k steps · 10 start states")
        left_at(late, 2.35, -3.25)

        def dumbbell(x, pred, actual, gap, name, color):
            seg = m.Line([x, dy(actual), 0], [x, dy(pred), 0]).set_stroke(RED, 6)
            p = m.Dot([x, dy(pred), 0], radius=0.11, color=VALUE).set_z_index(3)
            a = m.Dot([x, dy(actual), 0], radius=0.11, color=REWARD).set_z_index(3)
            pl = right_at(code(f"{pred:.1f}", 20, VALUE), x - 0.2, dy(pred))
            al = right_at(code(f"{actual:.1f}", 20, REWARD), x - 0.2, dy(actual))
            gl = left_at(title_text(f"+{gap:.1f}", 32, RED), x + 0.22, (dy(pred) + dy(actual)) / 2)
            nm = code(name, 20, color, weight=m.BOLD).move_to([x, -2.85, 0])
            return m.VGroup(seg, p, a, pl, al, gl, nm)

        db1 = dumbbell(3.1, PRED, ACTUAL, GAP, "DQN", ONE_C)
        db2 = dumbbell(5.4, PRED2, ACTUAL2, GAP2, "Double DQN", TWO_C)

        # the scoreboard (replaces the curves)
        rows = [("predictions above the ceiling", ABOVE, ABOVE2, 1.0, lambda v: f"{100 * v:.0f}%", "of checks"),
                ("steps to solve, mean of 5 seeds", STEPS, STEPS2, 300.0, lambda v: f"{v:.0f}k", "fewer is better"),
                ("final average score", FINAL, FINAL2, 500.0, lambda v: f"{v:.0f}", "of 500")]
        board = []
        for k, (title, a, b, scale, fmt, note) in enumerate(rows):
            y = 1.55 - 1.7 * k
            head = left_at(text(title, 22, INK, weight=m.BOLD), -6.35, y + 0.45)
            nt = text(note, 20, MUTED).next_to(head, m.RIGHT, buff=0.25).align_to(head, m.DOWN)
            bars = []
            for j, (v, color, tag) in enumerate(((a, ONE_C, "DQN"), (b, TWO_C, "Double"))):
                yy = y - 0.42 * j
                tg_ = right_at(code(tag, 20, color), -4.95, yy)
                bar = hbar(-4.75, yy, 3.9 * v / scale, 0.3, color)
                val = left_at(code(fmt(v), 20, color, weight=m.BOLD), -4.75 + 3.9 * v / scale + 0.15, yy)
                bars.append(m.VGroup(tg_, bar, val))
            board.append((m.VGroup(head, nt), bars))

        self.say(
            "[Back to CartPole, trained on this Mac, five seeds each]: "
            "[every five thousand steps, both networks predicted the value of ten start positions.]",
            m.AnimationGroup(self.new_header("06 · measured", "CartPole, scored honestly"), m.FadeIn(plot), m.FadeIn(top)),
            m.AnimationGroup(m.FadeIn(band1), m.FadeIn(band2), m.Create(c1, run_time=2.2), m.Create(c2, run_time=2.2),
                             m.Create(ceil), m.FadeIn(ceil_lab)),
        )
        self.say(
            "[Both start near zero and climb, as news of the reward spreads.] [But the DQN keeps climbing, past the ceiling.]",
            m.FadeIn(climb, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(c1_red, run_time=1.2), m.FadeIn(l1), m.FadeIn(l2)),
        )
        self.say(
            "[After three hundred thousand steps, the DQN predicted one hundred and two point eight], "
            "[but collected ninety-four]: [eight point eight too high.]",
            m.AnimationGroup(m.Create(d_axis), m.FadeIn(d_ticks), m.Create(d_ceil), m.FadeIn(d_ceil_lab), m.FadeIn(legend),
                             m.FadeIn(late), m.FadeIn(db1[6]), m.FadeIn(db1[1], scale=0.5), m.FadeIn(db1[3])),
            m.AnimationGroup(m.FadeIn(db1[2], scale=0.5), m.FadeIn(db1[4])),
            m.AnimationGroup(m.Create(db1[0]), m.FadeIn(db1[5], shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The Double DQN predicted ninety-nine point five, and collected ninety-seven point eight]: [only one point seven too high.]",
            m.AnimationGroup(m.FadeIn(db2[6]), m.FadeIn(db2[1], scale=0.5), m.FadeIn(db2[3]), m.FadeIn(db2[2], scale=0.5),
                             m.FadeIn(db2[4])),
            m.AnimationGroup(m.Create(db2[0]), m.FadeIn(db2[5], shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The DQN's guesses broke the ceiling in seventy-nine percent of checks]; [the Double DQN's, in thirty-four: better, not perfect.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(plot, band1, band2, c1, c1_red, c2, ceil, ceil_lab, l1, l2, climb)),
                             m.FadeIn(board[0][0]), m.FadeIn(board[0][1][0][0]),
                             m.GrowFromEdge(board[0][1][0][1], m.LEFT), m.FadeIn(board[0][1][0][2])),
            m.AnimationGroup(m.FadeIn(board[0][1][1][0]), m.GrowFromEdge(board[0][1][1][1], m.LEFT), m.FadeIn(board[0][1][1][2])),
        )
        self.say(
            "[With more honest targets, the Double DQN also solved CartPole sooner]: "
            "[after a hundred and seventy-five thousand steps, instead of two hundred and fifty-two thousand.]",
            m.AnimationGroup(m.FadeIn(board[1][0]), m.FadeIn(board[1][1][0][0]), m.GrowFromEdge(board[1][1][0][1], m.LEFT),
                             m.FadeIn(board[1][1][0][2])),
            m.AnimationGroup(m.FadeIn(board[1][1][1][0]), m.GrowFromEdge(board[1][1][1][1], m.LEFT), m.FadeIn(board[1][1][1][2])),
        )
        self.say(
            "[And it stayed better]: [a final average of four hundred and forty-four, against two hundred and eighty-three.]",
            m.AnimationGroup(m.FadeIn(board[2][0]), m.FadeIn(board[2][1][0][0]), m.GrowFromEdge(board[2][1][0][1], m.LEFT),
                             m.FadeIn(board[2][1][0][2])),
            m.AnimationGroup(m.FadeIn(board[2][1][1][0]), m.GrowFromEdge(board[2][1][1][1], m.LEFT), m.FadeIn(board[2][1][1][2])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · dueling and prioritized replay

    def more_fixes(self):
        self.next_section("7 · Dueling and prioritized replay", notes=(
            "Dueling networks (Wang et al. 2016, eq. 9): two streams after a shared trunk, V(s) and A(s, a), recombined as "
            f"Q = V + (A − mean A). Worked example (notes §14): V = {V_S:g}, A = {ADV.tolist()} → mean A = {ADV.mean():g}, Q = "
            f"{Q_DUEL.tolist()}; the greedy action is unchanged, only the offset is pinned. V learns from every transition. "
            "Prioritized replay (Schaul et al. 2016): P(i) = p_i^α / Σ_k p_k^α with p = |δ| (+ a small ε, left out here), "
            f"α = {ALPHA_PER}: |δ| = {DELTA.tolist()} → P = {[round(p, 3) for p in P_PER]} vs 0.25 each uniformly. Importance "
            "weights w_i = (1/(N·P(i)))^β, normalised by their max, β annealed to 1, undo the sampling bias. The six "
            f"highlighted replays are a draw from default_rng(3) with these P: {PICKS.tolist()}."))
        divider = m.DashedLine([0, 2.3, 0], [0, -3.35, 0], dash_length=0.1).set_stroke(EDGE, 2)
        # left: dueling
        t_l = left_at(text("dueling network", 26, INK, weight=m.BOLD), -6.35, 2.2)
        src_l = left_at(cite("Wang et al., 2016"), -6.35, 1.78)
        trunk = NeuralNet(layers=(4, 5, 5), width=1.5, height=1.5, center=(-5.35, 0.55, 0), color=EXPLORE)
        v_box = card(1.8, 0.6, VALUE, BLUE_FILL, radius=0.12, stroke=2.5).move_to([-2.55, 1.2, 0])
        v_lab = code(f"V(s) = {V_S:g}", 22, INK, weight=m.BOLD).move_to(v_box)
        adv = Bars(ADV.tolist(), labels=None, top=2.4, bottom=-1.4, width=1.8, height=1.2, color=ACTION, gap=0.14,
                   fmt="{:g}", size=20, center=(-2.55, -0.2, 0))
        fork = trunk.get_right() + [0.2, 0, 0]
        wires = m.VGroup(arrow(fork, v_box.get_left() + [-0.05, 0, 0], MUTED, buff=0.05),
                         arrow(fork, [adv.get_left()[0] - 0.1, adv.level(0.5), 0], MUTED, buff=0.05))
        v_say = text("how good the state is", 20, VALUE).next_to(v_box, m.UP, buff=0.14)
        a_say = text("how much each action adds", 20, ACTION).move_to([-2.55, -1.12, 0])
        f_parts = [("Q =", VALUE), ("V", VALUE), ("+ (", INK), ("A", ACTION), ("− mean", INK), ("A", ACTION), (")", INK)]
        form = colored(f_parts, 24, weight=m.BOLD).move_to([-3.4, -1.85, 0])
        q_row = colored([("Q =", VALUE), ("[" + ", ".join(f"{q:g}" for q in Q_DUEL) + "]", VALUE)], 24, weight=m.BOLD)
        q_row.move_to([-3.4, -2.5, 0])
        mean_note = code(f"mean A = {ADV.mean():g} · same best action", 20, MUTED).move_to([-3.4, -3.1, 0])

        # right: prioritized replay
        t_r = left_at(text("prioritized replay", 26, INK, weight=m.BOLD), 0.45, 2.2)
        src_r = right_at(cite("Schaul et al., 2016"), 6.5, 2.2)
        ys = [1.3, 0.55, -0.2, -0.95]
        mems = m.VGroup(*[m.VGroup(card(1.45, 0.52, EDGE, CARD, radius=0.1, stroke=2),
                                   code("s a r s′", 20, MUTED)).move_to([1.3, y, 0]) for y in ys])
        for mem in mems:
            mem[1].move_to(mem[0])
        h_err = code("|TD error|", 20, SIGNAL, weight=m.BOLD).move_to([3.05, 1.85, 0])
        h_p = code("chance", 20, INK, weight=m.BOLD).move_to([5.65, 1.85, 0])
        errs = m.VGroup(*[hbar(2.3, y, 0.75 * d, 0.3, SIGNAL) for d, y in zip(DELTA, ys)])
        err_n = m.VGroup(*[left_at(code(f"{d:g}", 20, SIGNAL), 2.3 + 0.75 * d + 0.12, y) for d, y in zip(DELTA, ys)])
        probs = m.VGroup(*[code(f"{p:.2f}", 22, INK, weight=m.BOLD).move_to([5.65, y, 0]) for p, y in zip(P_PER, ys)])
        uni = left_at(code("uniform: 0.25 each", 20, MUTED), 0.6, -1.65)
        weights = m.VGroup(left_at(text("importance weights undo the bias", 20, INK), 0.6, -2.35),
                           left_at(code("w = (1 / (N·P))^β", 22, SIGNAL, weight=m.BOLD), 0.6, -2.9))
        flashes = [m.Indicate(mems[i], color=SIGNAL, scale_factor=1.08) for i in PICKS]

        self.say(
            "[Two more fixes change other parts. A dueling network splits its head in two]: "
            "[how good the state is], [and how much each action adds.]",
            m.AnimationGroup(self.new_header("07 · two more fixes", "Split the head, replay the surprises"),
                             m.FadeIn(t_l), m.FadeIn(trunk), m.Create(divider), m.FadeIn(src_l),
                             m.Succession(m.Wait(0.8), m.LaggedStart(*[m.GrowArrow(w) for w in wires], lag_ratio=0.3))),
            m.AnimationGroup(m.FadeIn(v_box), m.FadeIn(v_lab), m.FadeIn(v_say)),
            m.AnimationGroup(m.FadeIn(adv), m.FadeIn(a_say)),
        )
        self.say(
            "[Adding them back gives the Q-values], [so the state's value learns from every move, even when the action hardly matters.]",
            m.AnimationGroup(m.FadeIn(form, shift=0.1 * m.UP), m.Succession(m.Wait(0.6), m.AnimationGroup(
                m.FadeIn(q_row, shift=0.1 * m.UP), m.FadeIn(mean_note)))),
            m.Circumscribe(m.VGroup(v_box, v_lab), color=VALUE, buff=0.08),
        )
        self.say(
            "[Prioritized replay replays surprising memories more often]: [those with a large TD error.]",
            m.AnimationGroup(m.FadeIn(t_r), m.FadeIn(src_r), m.LaggedStart(*[m.FadeIn(x, shift=0.1 * m.RIGHT) for x in mems],
                                                                           lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(h_err), m.LaggedStart(*[m.GrowFromEdge(e, m.LEFT) for e in errs], lag_ratio=0.15),
                             m.FadeIn(err_n), m.FadeIn(h_p), m.FadeIn(probs), m.FadeIn(uni),
                             m.Succession(m.Wait(0.8), *flashes)),
        )
        self.say(
            "[Importance weights then cancel the bias this sampling adds.]",
            m.FadeIn(weights, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · Rainbow

    def rainbow(self):
        self.next_section("8 · Rainbow", notes=(
            "Hessel et al. (2018), Rainbow: Double Q, prioritized replay, dueling, multi-step returns (n = 3), distributional RL "
            "(C51, 51 atoms) and Noisy Nets in one agent. Table 2, median human-normalised score, 57 Atari games, no-ops "
            "regime: " + ", ".join(f"{n} {v}%" for n, v in MEDIANS) + " (human starts: Rainbow 153%). Ablations (Fig. 3 and "
            "§'Ablation studies'): prioritized replay and multi-step learning 'were the two most crucial components' (removing "
            "either caused a large drop in median performance); distributional ranked immediately below; removing Noisy Nets "
            "hurt overall; dueling: 'no significant difference'; Double Q: 'limited' difference — the authors suspect that "
            "clipping the value distribution to [−10, 10] already counteracts overestimation. The six blocks are a list, not "
            "a measurement: their sizes are equal."))
        bw, bh, gap = 3.0, 0.5, 0.09
        x_s = -4.75
        blocks = []
        for k, name in enumerate(RAINBOW_PARTS):
            y = -2.55 + k * (bh + gap)
            box = card(bw, bh, EXPLORE, VIOLET_FILL, radius=0.1, stroke=2)
            blocks.append(m.VGroup(box, text(name, 20, INK, weight=m.BOLD).move_to(box)).move_to([x_s, y, 0]))
        stack = m.VGroup(*blocks)
        name = title_text("Rainbow", 48, INK, gradient=(GOLD, CORAL, PINK, VIOLET, BLUE)).next_to(stack, m.UP, buff=0.3)
        year = cite("Hessel et al., 2018").next_to(stack, m.DOWN, buff=0.2)
        tags = m.VGroup()
        for blk, nm in zip(blocks, RAINBOW_PARTS):
            word, col = ABLATION[nm]
            tags.add(left_at(code(word, 20, col, weight=m.BOLD), x_s + bw / 2 + 0.2, blk.get_center()[1]))
        abl_head = left_at(text("taking one out:", 20, MUTED), x_s + bw / 2 + 0.2, blocks[-1].get_center()[1] + 0.6)

        # median scores
        x_lab, x_bar, unit = 2.05, 2.25, 0.016
        chart_head = right_at(cite("median score, % of a human · 57 Atari games"), 6.5, 2.2)
        rows = m.VGroup()
        for k, (nm, v) in enumerate(MEDIANS):
            y = 1.5 - 0.66 * k
            col = INK if nm == "Rainbow" else MUTED
            lab = right_at(code(nm, 20, col, weight=m.BOLD if nm == "Rainbow" else m.NORMAL), x_lab, y)
            bar = hbar(x_bar, y, unit * v, 0.36, REWARD)
            if nm == "Rainbow":
                bar.set_stroke(GOLD, 3.5)
            val = left_at(code(f"{v}%", 20, REWARD, weight=m.BOLD), x_bar + unit * max(v, 100) + 0.12, y)  # clear of the human line
            rows.add(m.VGroup(lab, bar, val))
        human = m.DashedLine([x_bar + unit * 100, 1.85, 0], [x_bar + unit * 100, -2.75, 0], dash_length=0.08).set_stroke(INK, 2)
        human_lab = code("human", 20, INK).next_to(human, m.DOWN, buff=0.1)
        dqn_row, rb_row = rows[0], rows[-1]

        self.say(
            "[In twenty eighteen, the Rainbow agent stacked six ideas]: "
            "[double Q-learning, prioritized replay, dueling heads, multi-step returns, distributional values, and noisy networks.]",
            m.AnimationGroup(self.new_header("08 · rainbow", "Six fixes, stacked"), m.FadeIn(name, shift=0.1 * m.DOWN),
                             m.FadeIn(year)),
            m.LaggedStart(*[m.FadeIn(b, shift=0.3 * m.DOWN) for b in blocks], lag_ratio=0.45),
        )
        self.say(
            "[On fifty-seven Atari games, its median score reached two hundred and twenty-three percent of a human's], "
            "[against seventy-nine for the DQN.]",
            m.AnimationGroup(m.FadeIn(chart_head), m.Create(human), m.FadeIn(human_lab),
                             m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2]))
                                             for r in rows], lag_ratio=0.25),
                             m.Succession(m.Wait(2.2), m.Circumscribe(rb_row, color=REWARD, buff=0.08))),
            m.Circumscribe(dqn_row, color=MUTED, buff=0.08),
        )
        self.say(
            "[Then the authors took pieces out. Prioritized replay and multi-step returns mattered most.]",
            m.AnimationGroup(m.FadeIn(abl_head), m.FadeIn(tags[1]), m.FadeIn(tags[3]),
                             blocks[1][0].animate.set_stroke(GOLD, 3.5), blocks[3][0].animate.set_stroke(GOLD, 3.5),
                             m.FadeIn(tags[4]), m.FadeIn(tags[5])),
        )
        self.say(
            "[Double Q-learning and dueling mattered least]: [the authors suspect other parts already kept the values in check.]",
            m.AnimationGroup(m.FadeIn(tags[0]), m.FadeIn(tags[2]), blocks[0].animate.set_opacity(0.5),
                             blocks[2].animate.set_opacity(0.5)),
            m.Circumscribe(m.VGroup(blocks[4], tags[4]), color=EXPLORE, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            "Luck (a max over noisy estimates picks the luckiest: +1.54 σ for ten zeros), split (one estimate chooses, another "
            "judges: Double Q-learning, Double DQN), stack (Rainbow's six fixes; replay priorities and multi-step mattered "
            f"most). CartPole, measured here: overestimate +{GAP:.1f} → +{GAP2:.1f} ({SEEDS} seeds). Episode 15: search and "
            "self-play."))
        halo = glow(EXPLORE, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Luck", 70, EXPLORE), title_text("Split", 70, ACTION), title_text("Stack", 70, REWARD))
        words.arrange(m.RIGHT, buff=1.9).move_to([0, 1.05, 0])
        on_baseline(words)
        subs = m.VGroup(text("the max picks the luckiest", 22, MUTED), text("one chooses, one judges", 22, MUTED),
                        text("the fixes that matter", 22, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        base = max(s_.get_center()[1] for s_ in subs)
        for s_ in subs:
            s_.set_y(base)
        proof = colored([("CartPole, overestimate:", MUTED), (f"DQN +{GAP:.1f}", ONE_C), ("→", MUTED),
                         (f"Double DQN +{GAP2:.1f}", TWO_C), (f"· {SEEDS} seeds, measured here", MUTED)], 22)
        proof.move_to([0, -1.2, 0])
        bot = rover(0.55).move_to([-2.4, -2.6, 0])
        nxt = pill("episode 15 · search and self-play", EXPLORE, VIOLET_FILL, 22).next_to(bot, m.RIGHT, buff=0.45)

        self.say(
            "[To sum up]: [a max over noisy estimates picks luck]; [split choosing from judging]; [and stack the fixes that matter.]",
            m.AnimationGroup(self.new_header("09 · recap", "Luck, split, stack"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[On CartPole, measured here, splitting the jobs cut the overestimate from eight point eight to one point seven.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
        )
        self.say(
            "[So far, Rover only learned values from experience.] [Next, it searches ahead, and plays against itself.]",
            m.FadeIn(bot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()
