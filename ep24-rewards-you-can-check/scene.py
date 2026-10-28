"""Trial and Reward · Episode 24 · Rewards You Can Check (Module 6 · Teaching Language Models with Reward)

Correct answers and passing tests are rewards no one can argue with. Episode 21's reward came from a learned judge
(a reward model, which can be fooled: Episode 25); a maths answer or a test suite needs no judge, only a program.
That is RL with verifiable rewards (the term from Lambert et al. 2024, Tülu 3). The running example is OUR GRPO RUN
on this Mac (research/experiments/lm_rl.md + lm_rl.json; introduced in Episode 20): its checker (the last number in
the last \\boxed{} equals the reference) stamps a held-out problem (GSM8K test 788, John's school trip: the base model
boxes 75 → RED, 0; the trained model boxes 100 → GREEN, +1) and a right-but-unboxed answer (test 776 → 0). Then
DeepSeek-R1-Zero (pure RL with GRPO on a base model, rule-based accuracy + format rewards, no neural reward model):
AIME 2024 pass@1 15.6% → 71.0% in the Jan 2025 preprint (arXiv v1; the Nature 2025 version reports 77.9% after a
longer run), majority vote over 64 → 86.7%, answers growing from hundreds of tokens to thousands, the 'aha moment'
(and the debate about it). OUR answers did NOT grow (309 → 305 tokens). Our results on 500 held-out problems (8
samples each at T = 0.8): pass@1 38.3% → 46.2% (+7.9 ± 0.9), the gain mostly maths, not format; pass@k with the
unbiased estimator (rl_reference.pass_at_k) on John's 8 samples, then the pass@k curves: the gain shrinks from +7.9 at
k = 1 to +4.8 at k = 8 (Yue et al. 2025 find the curves cross at large k). Greedy accuracy is shaky (46–48% for the
same weights depending on the generation batch size; 14% of problems flip): that's why the headline is pass@1 from
samples. Recap: check, sample, sharpen. Episode 26 replays the run with test-1028 (Tom's bedroom set), so this
episode uses other examples.

Every number on screen is computed below from rl_reference.py and asserted, loaded with measured(...) (or from the
run's own evaluation log, research/experiments/logs/lm_grpo_eval.jsonl, cross-checked against lm_rl.json), or cited.

Sources (research/notes.md §24, §26, Appendix A row 24, Appendix C row 24; read October 2026):
- research/experiments/lm_rl.md + lm_rl.json (this Mac Studio, Apple M4 Max, 64 GB, MLX; run 2026-10-25, one seed).
- Lambert et al. 2024, "Tülu 3" (arXiv 2411.15124): "We refer to this algorithm as Reinforcement Learning with
  Verifiable Rewards (RLVR); it obtains a constant reward value if a completion is successful." (α = 10).
- DeepSeek-AI 2025, "DeepSeek-R1" arXiv v1 (22 Jan 2025): §2.2.2 rule-based accuracy + format rewards, no neural reward
  model ("the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process");
  §2.2.4 AIME 2024 pass@1 15.6% → 71.0%, majority voting 86.7% (Table 2: cons@64); Fig. 3 response length ("from
  generating hundreds to thousands of reasoning tokens"); Table 3, the aha moment. Nature version (Guo et al. 2025,
  via arXiv v2): 77.9%; R1-Zero ran 10,400 steps with outputs up to 65,536 tokens.
- Liu et al. 2025 (Dr. GRPO): "DeepSeek-V3-Base already exhibit 'Aha moment'".
- Chen et al. 2021 (Codex): the unbiased pass@k estimator 1 − C(n−c, k)/C(n, k).
- Yue et al. 2025 ("Does RL really incentivize reasoning…", arXiv 2504.13837): RLVR models win at small k, base models
  overtake at large k; e.g. Minerva with a 32B model, the base model ahead by about 9% at k = 128.

Illustrative (labelled on screen): the unit-test terminal in section 1; the order of John's 8 samples in section 6
(only the count, 5 of 8 right, is real); the R1-Zero length curve in section 4 (a sketch of the shape of the paper's
Fig. 3, no numbers); the crossing curves in the small inset of section 6 (a schematic of Yue et al.'s finding).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import math
from decimal import ROUND_HALF_UP, Decimal
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep24"
LM = "lm_rl"

# ---------------------------------------------------------------------------- our run (measured on this Mac)

H = measured(LM, "headline")
SETUP = measured(LM, "setup")
CURVES = measured(LM, "curves")
EX = {e["id"]: e for e in measured(LM, "examples")}
JOHN, LAPTOP = EX["test-788"], EX["test-776"]
KS = (1, 2, 4, 8)


def pct(x, places=1):
    """A fraction as a percentage, rounded half up (51.65 → 51.7, as lm_rl.md prints it)."""
    q = Decimal(repr(round(100 * x, 9))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{q}%"


N_TEST, N_SAMPLES = H["test_questions"], measured(LM, "setup", "args", "eval_samples")
assert (N_TEST, N_SAMPLES, H["eval_temperature"]) == (500, 8, 0.8)
assert N_TEST * N_SAMPLES == 4000                                                       # "all four thousand"

# the run's evaluation log: per held-out problem, how many of its 8 samples were right
EVAL_LOG = EXPERIMENTS / "logs" / "lm_grpo_eval.jsonl"
_EVALS = [json.loads(line) for line in EVAL_LOG.read_text().splitlines() if line.strip()]
_E0, _E1 = _EVALS[0], _EVALS[-1]
assert (_E0["step"], _E1["step"]) == (0, H["steps"]) and _E0["n_samples"] == N_SAMPLES
IDS = [q["id"] for q in _E0["per_question"]]
assert IDS == [q["id"] for q in _E1["per_question"]] and len(IDS) == N_TEST
C0 = np.array([q["c"] for q in _E0["per_question"]])
C1 = np.array([q["c"] for q in _E1["per_question"]])
# the estimator, averaged over the 500 problems, reproduces the headline (lm_rl.json keeps 4 decimals)
P_BEFORE = {k: float(np.mean([pass_at_k(8, int(c), k) for c in C0])) for k in KS}
P_AFTER = {k: float(np.mean([pass_at_k(8, int(c), k) for c in C1])) for k in KS}
for k in KS:
    assert abs(P_BEFORE[k] - H["pass_at_k_before"][str(k)]) < 1e-4
    assert abs(P_AFTER[k] - H["pass_at_k_after"][str(k)]) < 1e-4


# the headline, as said
assert pct(P_BEFORE[1]) == "38.3%" and pct(P_AFTER[1]) == "46.2%"
assert [pct(P_BEFORE[k]) for k in KS] == ["38.3%", "51.7%", "63.0%", "71.6%"]
assert [pct(P_AFTER[k]) for k in KS] == ["46.2%", "58.1%", "68.4%", "76.4%"]
GAIN = {k: float(pct(P_AFTER[k])[:-1]) - float(pct(P_BEFORE[k])[:-1]) for k in KS}      # of the shown values
assert [round(GAIN[k], 1) for k in KS] == [7.9, 6.4, 5.4, 4.8]                      # "the gain shrinks"
assert all(GAIN[a] > GAIN[b] for a, b in zip(KS, KS[1:]))
SE = H["paired_se"]
assert round(100 * SE["pass1_gain"]["se"], 1) == 0.9 and round(100 * SE["pass8_gain"]["se"], 1) == 1.8
assert abs(SE["pass1_gain"]["gain"] - (P_AFTER[1] - P_BEFORE[1])) < 1e-4              # "give or take one"
GAP_BEFORE = float(pct(P_BEFORE[8])[:-1]) - float(pct(P_BEFORE[1])[:-1])
GAP_AFTER = float(pct(P_AFTER[8])[:-1]) - float(pct(P_AFTER[1])[:-1])
assert (round(GAP_BEFORE, 1), round(GAP_AFTER, 1)) == (33.3, 30.2)

_D = (C1 - C0) / 8
assert abs(_D.std(ddof=1) / math.sqrt(N_TEST) - SE["pass1_gain"]["se"]) < 2e-4        # the paired standard error

# John's trip (test-788), the episode's example
J_BEFORE, J_AFTER = int(C0[IDS.index("test-788")]), int(C1[IDS.index("test-788")])
assert (J_BEFORE, J_AFTER) == (5, 7)                                                    # "five of its eight"
J_PASS = {k: pass_at_k(8, J_BEFORE, k) for k in KS}
assert abs(J_PASS[1] - 5 / 8) < 1e-12 and abs(J_PASS[2] - (1 - 3 / 28)) < 1e-12 and J_PASS[4] == 1.0   # "three in twenty-eight"
assert math.comb(8 - J_BEFORE, 2) == 3 and math.comb(8, 2) == 28 and round(100 * J_PASS[2]) == 89
assert 8 - J_BEFORE < 4                                                                  # "only three are wrong"

# the checker (the run's own regexes, lm/common.py extract_answer + reward_correct)
BOXED = re.compile(SETUP["boxed_regex"])
NUMBER = re.compile(SETUP["number_regex"])


def reward_correct(answer, ref):
    boxes = BOXED.findall(answer)
    if not boxes:
        return 0.0
    nums = NUMBER.findall(boxes[-1])
    return 1.0 if nums and abs(float(nums[-1].replace(",", "")) - ref) < 1e-4 else 0.0


JB, JT = JOHN["base"]["text"], JOHN["trained"]["text"]
assert JOHN["ref"] == 100 and reward_correct(JB, 100) == 0.0 == JOHN["base"]["correct"] and JOHN["base"]["answer"] == 75
assert reward_correct(JT, 100) == 1.0 == JOHN["trained"]["correct"]
assert "\\frac{300}{2} = 150" in JB and "\\frac{150}{2} = 75" in JB and "\\boxed{75}" in JB      # "halved twice"
assert "\\frac{300}{2} = 150" in JT and "150 - 50 = 100" in JT and "\\boxed{100}" in JT
assert (JOHN["base"]["tokens"], JOHN["trained"]["tokens"]) == (252, 182)
J_Q = " ".join(JOHN["question"].split())
J_LINES = ["John is raising money for a school trip. He has applied for help from the school, which",
           "has decided to cover half the cost of the trip. How much money is John missing if he has",
           "$50 and the trip costs $300?"]
assert " ".join(J_LINES) == J_Q
LB = LAPTOP["base"]["text"]
assert LAPTOP["ref"] == 800 and not BOXED.findall(LB) and reward_correct(LB, 800) == 0.0          # right, unboxed: 0
assert LB.rstrip().endswith("So, you have to pay $800 after a 20% discount.")
assert LAPTOP["question"] == "The price of a laptop is $1000. If you get a 20% discount, how much do you have to pay?"

# format vs maths (training answers, steps 0-24 vs 175-199)
FVM = H["train_samples_format_vs_maths"]
F0, F1 = FVM["steps_0_24"], FVM["steps_175_199"]
FMT = [(name, F0[key], F1[key]) for name, key in (("reward (boxed)", "strict_acc"), ("last number", "last_number_acc"),
                                                     ("boxed at all", "boxed_rate"))]
assert [pct(a) for _, a, _ in FMT] == ["47.6%", "48.1%", "91.6%"] and [pct(b) for _, _, b in FMT] == ["57.0%", "56.2%", "94.8%"]
assert round(100 * (F1["last_number_acc"] - F0["last_number_acc"])) == 8            # "still rose eight points"
assert round(100 * (F1["strict_acc"] - F0["strict_acc"])) == 9

# lengths: did NOT grow
assert (round(H["greedy_len_before"]), round(H["greedy_len_after"])) == (309, 305)
STEPS = np.array(CURVES["step"], dtype=float)
REWARD_MA = np.array(CURVES["reward_ma10"], dtype=float)
LEN_MA = np.array(CURVES["mean_len_ma10"], dtype=float)
assert len(STEPS) == 200 and LEN_MA[10:].min() > 250 and LEN_MA[10:].max() < 350      # "stayed near three hundred"
assert (round(H["train_reward_first25"], 3), round(H["train_reward_last25"], 3)) == (0.476, 0.57)   # "the reward climbed"
assert H["max_new_tokens"] == 512 and H["steps"] == 200                                  # "capped at 512"

# greedy accuracy vs generation batch size
GB = H["greedy_by_batch_size"]
BATCHES = GB["batch_sizes"]
G_BASE = [GB["base"][str(b)] for b in BATCHES]
G_TRAIN = [GB["trained"][str(b)] for b in BATCHES]
assert BATCHES == [16, 32, 64, 128, 256]
assert (pct(min(G_BASE)), pct(max(G_BASE))) == ("46.2%", "48.2%")                       # "forty-six to forty-eight"
FLIP = 1 - GB["stable_fraction_base"]
assert round(100 * FLIP) == 14                                                           # "about fourteen percent"
G_GAINS = [t - b for t, b in zip(G_TRAIN, G_BASE)]
assert (round(100 * min(G_GAINS), 1), round(100 * max(G_GAINS), 1)) == (1.6, 4.2)

# ---------------------------------------------------------------------------- cited (notes §24)

R1_START, R1_END, R1_MAJ, R1_NATURE = 15.6, 71.0, 86.7, 77.9      # AIME 2024, R1 arXiv v1 (Jan 2025); Nature: 77.9
R1_VOTES, R1_SAMPLES = 64, 16
R1_STEPS, R1_MAXLEN = 10_400, 65_536                               # Nature / arXiv v2 §2
YUE_K, YUE_GAP = 128, 9

# ---------------------------------------------------------------------------- look

RM_ICON = recolored("llm", REWARD, GOLD_FILL, prefix=PREFIX)
CHECK = recolored("check", REWARD, GOLD_FILL, prefix=PREFIX)
POLICY_ICON = recolored("llm", ACTION, CORAL_FILL, prefix=PREFIX)
BASE_ICON = recolored("llm", MUTED, CARD, prefix=PREFIX)
BEFORE_C, AFTER_C = MUTED, ACTION          # the base model MUTED, the trained policy CORAL


def cite(s, size=20):
    return text(s, size, MUTED)


def left_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.RIGHT)


def stamp(ok, word=None, size=22):
    """The checker's verdict: a thick-edged GREEN or RED stamp."""
    color, fill = (GREEN, GREEN_FILL) if ok else (RED, RED_FILL)
    words = code(word or ("RIGHT" if ok else "WRONG"), size, color, weight=m.BOLD)
    box = m.RoundedRectangle(corner_radius=0.08, width=words.width + 0.42, height=words.height + 0.32,
                             stroke_color=color, stroke_width=4, fill_color=fill, fill_opacity=1)
    return m.VGroup(box, words.move_to(box))


def stamp_in(mob, run_time=0.45):
    return m.FadeIn(mob, scale=1.7, run_time=run_time)


def mark(ok, size=0.34, color=None):
    """A drawn tick (GREEN) or cross (RED), centred at the origin."""
    s = size
    if ok:
        pts = [[-0.5 * s, 0.0, 0], [-0.12 * s, -0.38 * s, 0], [0.55 * s, 0.4 * s, 0]]
        return m.VMobject().set_points_as_corners(pts).set_stroke(color or GREEN, 6).set_fill(opacity=0)
    a = m.Line([-0.4 * s, -0.4 * s, 0], [0.4 * s, 0.4 * s, 0])
    b = m.Line([-0.4 * s, 0.4 * s, 0], [0.4 * s, -0.4 * s, 0])
    return m.VGroup(a, b).set_stroke(color or RED, 6)


def answer_chip(s, size=20, color=ACTION, fill=CORAL_FILL):
    """A model's answer (or its end) as a CORAL card with mono text."""
    words = code(s, size, INK)
    box = card(words.width + 0.5, words.height + 0.36, color, fill, radius=0.12, stroke=2.2)
    return m.VGroup(box, words.move_to(box))


def msg_card(role, lines, color, width=11.8, size=20):
    """A chat message: the role in its colour, the text, a stripe on the left (as in Episode 20)."""
    label = code(role, 22, color, weight=m.BOLD)
    body = m.VGroup(*[text(ln, size, INK) for ln in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    inner = m.VGroup(label, body).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def model_card(icon_name, title, subs, color, width=3.5, height=2.2, icon_h=0.62):
    box = card(width, height, color, CARD)
    body = m.VGroup(icon(icon_name, icon_h), text(title, 22, INK, weight=m.BOLD),
                    *[cite(s) for s in subs]).arrange(m.DOWN, buff=0.1)
    body.move_to(box)
    assert body.width < width - 0.2 and body.height < height - 0.15, (title, body.width, body.height)
    return m.VGroup(box, body)


def heat_cell(c, n=8):
    """A held-out problem's square: the more of its n samples right, the brighter the GREEN."""
    return mix(CARD, GREEN, 0.92 * c / n)


def formula(parts, colors, size=40):
    f = m.MathTex(*parts, font_size=size, color=INK)
    for part, col in zip(f, colors):
        part.set_color(col)
    return f


def dots_line(plot, xs, ys, color, width=4, radius=0.075):
    line = plot.line(xs, ys, color, width)
    dots = m.VGroup(*[plot.dot(x, y, color, radius) for x, y in zip(xs, ys)])
    return m.VGroup(line, dots)


# ---------------------------------------------------------------------------- the episode


class Ep24RewardsYouCanCheck(Episode):
    number = 24

    def lesson(self):
        self.checkable()
        self.the_checker()
        self.r1_zero()
        self.longer_answers()
        self.our_accuracy()
        self.pass_at_k_section()
        self.greedy_section()
        self.recap()

    # ------------------------------------------------------------------ 1 · a reward no one can argue with

    def checkable(self):
        self.next_section("1 · A reward no one can argue with", notes=(
            "Episode 21's reward model is a learned judge of people's taste, and it can be over-optimised (Episode 25). "
            "A maths answer can be checked by a program (the final number equals the reference), code by its unit "
            "tests (the terminal is an illustration). RLVR: 'We refer to this algorithm as Reinforcement Learning with "
            "Verifiable Rewards (RLVR); it obtains a constant reward value if a completion is successful.' (Lambert et "
            "al. 2024, Tülu 3, arXiv 2411.15124; there the constant is α = 10, and PPO is the optimiser). Notes §24: "
            "rewarding correct final answers is older (STaR 2022, RL from execution feedback 2024); the NAME is Tülu 3's."))
        judge_box = card(4.0, 2.75, REWARD, CARD).move_to([-4.6, 0.95, 0])
        judge_body = m.VGroup(icon(RM_ICON, 0.85), text("a reward model", 24, REWARD, weight=m.BOLD),
                              cite("learned from people's choices"), cite("episode 21")).arrange(m.DOWN, buff=0.12)
        judge_body.move_to(judge_box)
        judge = m.VGroup(judge_box, judge_body)
        fooled = pill("it can be fooled · episode 25", RED, RED_FILL, 20).move_to([-4.6, -0.95, 0])

        ys = (1.95, 0.95)
        chips = [left_at(answer_chip("… \\boxed{100}"), -2.1, ys[0]), left_at(answer_chip("… \\boxed{75}"), -2.1, ys[1])]
        ck_box = card(2.25, 2.15, REWARD, CARD).move_to([2.45, 1.45, 0])
        ck_body = m.VGroup(m.VGroup(icon(CHECK, 0.42), text("checker", 22, REWARD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.12),
                           code("= 100 ?", 24, INK, weight=m.BOLD), cite("the reference")).arrange(m.DOWN, buff=0.14)
        ck_body.move_to(ck_box)
        ck = m.VGroup(ck_box, ck_body)
        ins = [arrow(c.get_right(), [ck_box.get_left()[0], y, 0], MUTED, buff=0.08) for c, y in zip(chips, ys)]
        stamps = [stamp(True).move_to([5.3, ys[0], 0]), stamp(False).move_to([5.3, ys[1], 0])]
        outs = [arrow([ck_box.get_right()[0], y, 0], s.get_left(), REWARD, buff=0.08) for s, y in zip(stamps, ys)]
        assert chips[0].get_right()[0] < ck_box.get_left()[0] - 0.5

        term = terminal(["$ pytest", "test_total ...... PASSED", "test_empty ...... PASSED"], width=4.0,
                        title="for example")
        left_at(term, -2.1, -0.95)
        pass_stamp = stamp(True, "PASS").move_to([5.3, -0.95, 0])
        t_arrow = arrow([term.get_right()[0], -0.95, 0], pass_stamp.get_left(), REWARD, buff=0.1)
        assert term.get_top()[1] < chips[1].get_bottom()[1] - 0.25 and term.get_left()[0] > fooled.get_right()[0] + 0.2

        rlvr = m.VGroup(title_text("RLVR", 66, REWARD), text("reinforcement learning", 24, INK),
                        text("with verifiable rewards", 24, INK)).arrange(m.DOWN, buff=0.12).move_to([-4.55, 0.85, 0])
        rlvr_src = cite("Lambert et al., 2024 · Tülu 3").next_to(rlvr, m.DOWN, buff=0.22)
        quote = text("“… it obtains a constant reward value if a completion is successful.”", 22, INK).move_to([0, -2.65, 0])
        q_src = cite("Lambert et al., 2024, Tülu 3: the paper that named it · their constant: 10").move_to([0, -3.1, 0])
        assert quote.get_top()[1] < term.get_bottom()[1] - 0.2 and quote.width < 12.8

        self.say(
            "[Episode twenty-one's reward came from a learned judge]: [a model of people's taste, and models can be fooled.]",
            m.AnimationGroup(self.new_header("01 · checkable rewards", "A reward no one can argue with"),
                             m.FadeIn(judge, shift=0.15 * m.UP)),
            m.FadeIn(fooled, shift=0.1 * m.UP),
        )
        self.say(
            "[A maths answer needs no judge.] [Either its final number matches the reference], [or it doesn't.]",
            m.AnimationGroup(m.FadeIn(chips[0], shift=0.1 * m.RIGHT), m.FadeIn(chips[1], shift=0.1 * m.RIGHT),
                             m.FadeIn(ck, scale=0.9)),
            m.Succession(m.GrowArrow(ins[0], run_time=0.4), m.GrowArrow(outs[0], run_time=0.4), stamp_in(stamps[0])),
            m.Succession(m.GrowArrow(ins[1], run_time=0.4), m.GrowArrow(outs[1], run_time=0.4), stamp_in(stamps[1])),
        )
        self.say(
            "[Code works the same way]: [run its unit tests], [and they pass or fail.]",
            m.FadeIn(term[:4], shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(row, shift=0.05 * m.RIGHT) for row in term[4]], lag_ratio=0.45, run_time=1.4),
            m.AnimationGroup(m.GrowArrow(t_arrow), stamp_in(pass_stamp)),
        )
        self.say(
            "[That's reinforcement learning with verifiable rewards]: [the reward is a program, not a model.]",
            m.AnimationGroup(m.FadeOut(judge), m.FadeOut(fooled), m.FadeIn(rlvr, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Circumscribe(ck, color=REWARD, buff=0.1), m.Circumscribe(term, color=REWARD, buff=0.08)),
        )
        self.say(
            "[The name comes from a twenty twenty-four paper]: "
            "[a fixed reward when an answer checks out, and zero when it doesn't.]",
            m.FadeIn(rlvr_src, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(q_src)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · our run's checker

    def the_checker(self):
        self.next_section("2 · A program stamps every answer", notes=(
            "Our run's checker (research/experiments/lm/common.py extract_answer + reward_correct, shown simplified: the "
            "real one also strips commas and '\\!' in the box). Regexes from lm_rl.json setup: BOXED "
            f"{SETUP['boxed_regex']} and NUMBER {SETUP['number_regex']}; re-implemented here and re-scoring every answer "
            "shown. GSM8K test problem 788 (held out; lm_rl.json examples): reference 100. The BASE model's greedy answer "
            "(252 tokens) computes 300/2 = 150 as 'the total cost', then halves again, 150/2 = 75 'school's contribution', "
            "150 − 75 = 75, boxes 75 → 0. The TRAINED model (182 tokens): 300/2 = 150, 300 − 150 = 150, 150 − 50 = 100, "
            "boxes 100 → +1. Answers shown as short verbatim fragments ('\\frac{300}{2}' written 300/2). Test 776 (a $1000 "
            "laptop at 20% off, reference 800): the base answer ends 'So, you have to pay $800 after a 20% discount.' with "
            "no \\boxed{}: the checker gives 0 (the trained model boxes 800). These texts were re-decoded at batch 64 "
            "(lm_rl.md). Nothing reads the reasoning: its strength (no judge to fool) and its loophole (Episode 25)."))
        src = "\n".join([
            "def reward_correct(text, ref):",
            "    boxes = BOXED.findall(text)".ljust(40) + "# every \\boxed{...}",
            "    if not boxes:",
            "        return 0.0".ljust(40) + "# no box: no reward",
            "    a = NUMBER.findall(boxes[-1])[-1]".ljust(40) + "# its last number",
            "    return 1.0 if abs(float(a) - ref) < 1e-4 else 0.0"])
        blk = code_block(src, size=20, title="our run's checker · lm/common.py, simplified", accent=REWARD)
        blk.move_to([0, 0.45, 0])
        body = blk[1]
        line_box = glyphs(body, "boxes = BOXED.findall(text)")
        line_last = glyphs(body, "a = NUMBER.findall(boxes[-1])[-1]")
        line_cmp = glyphs(body, "return 1.0 if abs(float(a) - ref) < 1e-4 else 0.0")
        regex_note = cite("the run's regular expressions · the reward is 1 or 0, nothing in between")
        regex_note.next_to(blk, m.DOWN, buff=0.3)

        q = msg_card("user", J_LINES, STATE, width=12.6, size=20)
        q.move_to([0, 2.55 - q.height / 2, 0])
        y_lab0 = q.get_bottom()[1] - 0.32
        q_src = right_at(cite("GSM8K test problem 788 · held out, never trained on"), 6.4, y_lab0)
        ys = tuple(y_lab0 - 0.53 - 1.15 * i for i in range(3))
        specs = (("… 300/2 = 150 … 150/2 = 75 … \\boxed{75}", "before training · 252 tokens", "75 = 100 ?", False, "WRONG", 0),
                 ("… 300/2 = 150 … 150 - 50 = 100 … \\boxed{100}", "after training · 182 tokens", "100 = 100 ?", True, "RIGHT", 1),
                 ("… pay $800 after a 20% discount.", "another problem: a $1000 laptop, 20% off · reference 800",
                  "no box", False, "NO BOX", 0))
        chips_ = [left_at(answer_chip(sp[0]), -6.5, y) for sp, y in zip(specs, ys)]
        pills_ = [pill(sp[2], REWARD, GOLD_FILL, 22, mono=True) for sp in specs]
        stamps_ = [stamp(sp[3], sp[4]) for sp in specs]
        x_pill = max(c.get_right()[0] for c in chips_) + 0.58 + max(p.width for p in pills_) / 2
        x_stamp = x_pill + max(p.width for p in pills_) / 2 + 0.58 + max(st.width for st in stamps_) / 2
        x_rew = x_stamp + max(st.width for st in stamps_) / 2 + 0.52
        rows = []
        for (s, lab, cmp_s, ok, word, r), y, c, p, st in zip(specs, ys, chips_, pills_, stamps_):
            label = left_at(cite(lab), -6.5, y + 0.53)
            p.move_to([x_pill, y, 0])
            st.move_to([x_stamp, y, 0])
            rc = reward_chip(r, 24).move_to([x_rew, y, 0])
            a1 = arrow(c.get_right(), p.get_left(), MUTED, buff=0.1)
            a2 = arrow(p.get_right(), st.get_left(), MUTED, buff=0.1)
            assert c.get_right()[0] < p.get_left()[0] - 0.45 and rc.get_right()[0] < 6.55, (s, rc.get_right()[0])
            rows.append(dict(chip=c, label=label, pill=p, stamp=st, reward=rc, a1=a1, a2=a2, text=s))
        assert rows[0]["label"].get_right()[0] < q_src.get_left()[0] - 0.4 and rows[0]["label"].get_top()[1] < q.get_bottom()[1] - 0.1
        assert all(a["chip"].get_bottom()[1] > b["label"].get_top()[1] + 0.12 for a, b in zip(rows, rows[1:]))
        strength = pill("its strength: no judge to fool", GREEN, GREEN_FILL, 20)
        loophole = pill("its loophole: episode 25", RED, RED_FILL, 20)
        m.VGroup(strength, loophole).arrange(m.RIGHT, buff=0.4).move_to([0, rows[2]["chip"].get_bottom()[1] - 0.45, 0])
        assert strength.get_bottom()[1] > -3.55
        middles, finals = [], []
        for row in rows[:2]:
            words = row["chip"][1]
            box_s = row["text"][row["text"].index("\\boxed"):]
            fin = glyphs(words, box_s)
            finals.append(fin)
            middles.append(m.VGroup(*[g for g in words if g not in fin]))

        self.say(
            "[Our run's checker is a few lines of code]: [find the last boxed answer], "
            "[and compare its number with the reference.]",
            m.AnimationGroup(self.new_header("02 · the checker", "A program stamps every answer"),
                             m.FadeIn(blk, shift=0.15 * m.UP)),
            m.Circumscribe(line_box, color=REWARD, buff=0.06),
            m.AnimationGroup(m.Circumscribe(m.VGroup(line_last, line_cmp), color=REWARD, buff=0.06), m.FadeIn(regex_note)),
        )
        self.say(
            "[Here's a problem the model never trained on]: [the school pays half of John's three-hundred-dollar trip], "
            "[and he has fifty. What's missing?]",
            m.AnimationGroup(m.FadeOut(blk), m.FadeOut(regex_note), m.FadeIn(q[0]), m.FadeIn(q[1]), m.FadeIn(q[2][0]),
                             m.FadeIn(q_src)),
            m.FadeIn(q[2][1][:2], shift=0.08 * m.RIGHT),
            m.FadeIn(q[2][1][2], shift=0.08 * m.RIGHT),
        )

        def check_row(row):
            return m.Succession(m.AnimationGroup(m.GrowArrow(row["a1"]), m.FadeIn(row["pill"]), run_time=0.5),
                                m.AnimationGroup(m.GrowArrow(row["a2"]), stamp_in(row["stamp"]), m.FadeIn(row["reward"], scale=0.7),
                                                 run_time=0.5))

        r0, r1, r2 = rows
        self.say(
            "[Before training, the model halved the cost twice], [and boxed seventy-five]: [red, reward zero.]",
            m.AnimationGroup(m.FadeIn(r0["label"]), m.FadeIn(r0["chip"], shift=0.1 * m.RIGHT)),
            m.Circumscribe(finals[0], color=ACTION, buff=0.05),
            check_row(r0),
        )
        self.say(
            "[After training, half of three hundred is one fifty, minus fifty]: [one hundred. Green, plus one.]",
            m.AnimationGroup(m.FadeIn(r1["label"]), m.FadeIn(r1["chip"], shift=0.1 * m.RIGHT)),
            check_row(r1),
        )
        self.say(
            "[The checker is strict]: [this laptop answer says eight hundred dollars, which is right], "
            "[but it never boxed it: zero.]",
            m.FadeIn(r2["label"]),
            m.FadeIn(r2["chip"], shift=0.1 * m.RIGHT),
            check_row(r2),
        )
        self.say(
            "[It never reads the reasoning in between], [only the final number]: [that's its strength], "
            "[and, in episode twenty-five, its loophole.]",
            m.AnimationGroup(*[mid.animate.set_color(FAINT) for mid in middles]),
            m.AnimationGroup(*[m.Circumscribe(f, color=REWARD, buff=0.05) for f in finals]),
            m.FadeIn(strength, shift=0.1 * m.UP),
            m.FadeIn(loophole, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · DeepSeek-R1-Zero

    def r1_zero(self):
        self.next_section("3 · DeepSeek-R1-Zero", notes=(
            "DeepSeek-AI, 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning', arXiv "
            "2501.12948 v1, 22 Jan 2025. R1-Zero: GRPO (Episode 23) applied directly to DeepSeek-V3-Base, no supervised "
            "fine-tuning; two rule-based rewards (§2.2.2): accuracy (is the final answer right; boxed for maths, compiler "
            "tests for code) and format (thinking inside <think> </think>); no neural reward model: 'we find that the "
            "neural reward model may suffer from reward hacking in the large-scale reinforcement learning process'. "
            f"AIME 2024 pass@1 (averaged over {R1_SAMPLES} samples per question): {R1_START}% → {R1_END}% (§2.2.4, "
            f"Table 2, Jan 2025 preprint = arXiv v1, the version cited here); majority vote over {R1_VOTES} samples "
            f"(cons@64): {R1_MAJ}%. The later Nature version (Guo et al., Nature 645, Sept 2025; arXiv v2) reports "
            f"{R1_NATURE}% after a longer run: we do not mix the two (notes Appendix A row 24). Majority voting: "
            "'AI, Explained' Episode 43."))
        xs, y = (-4.65, 0.0, 4.65), 1.4
        cards = [model_card(BASE_ICON, "a base model", ["pretrained only", "no fine-tuning"], MUTED),
                 model_card("trophy", "reinforcement learning", ["GRPO · episode 23", "on its own answers"], REWARD),
                 model_card(POLICY_ICON, "R1-Zero", ["DeepSeek-AI", "January 2025"], ACTION)]
        for c, x in zip(cards, xs):
            c.move_to([x, y, 0])
        links = [arrow(cards[i][0].get_right(), cards[i + 1][0].get_left(), MUTED, buff=0.08) for i in range(2)]
        r_acc = pill("accuracy: is the final answer right?", REWARD, GOLD_FILL, 22)
        r_fmt = pill("format: thinking inside <think> … </think>", REWARD, GOLD_FILL, 22)
        rew = m.VGroup(r_acc, r_fmt).arrange(m.RIGHT, buff=0.45).move_to([0, -0.4, 0])
        assert rew.width < 12.8
        top = cards[1][0].get_bottom()
        forks = m.VGroup(*[m.Line(top + [0, -0.04, 0], p.get_top() + [0, 0.06, 0]).set_stroke(REWARD, 2.5) for p in (r_acc, r_fmt)])
        no_rm = m.VGroup(text("“the neural reward model may suffer from reward hacking", 22, INK),
                         text("in the large-scale reinforcement learning process”", 22, INK)).arrange(m.DOWN, buff=0.1)
        no_rm.move_to([0, -1.45, 0])
        no_rm_src = cite("DeepSeek-AI, 2025, §2.2.2 · so: no learned reward model at all").next_to(no_rm, m.DOWN, buff=0.16)
        assert no_rm.width < 12.8

        # results
        x0, span = -2.2, 7.0
        def bar(pct_, color, fill, yb):
            return m.Rectangle(width=span * pct_ / 100, height=0.42).set_stroke(color, 2).set_fill(fill, 1).move_to(
                [x0 + span * pct_ / 200, yb, 0])
        yb1, yb2 = -1.1, -1.8
        lab1 = right_at(code("AIME 2024 · pass@1", 20, MUTED), x0 - 0.2, yb1)
        lab2 = right_at(code("majority of 64", 20, MUTED), x0 - 0.2, yb2)
        b_start = bar(R1_START, MUTED, CARD_HI, yb1)
        b_end = bar(R1_END, ACTION, CORAL_FILL, yb1)
        b_maj = bar(R1_MAJ, ACTION, CORAL_FILL, yb2)
        v_start = code(f"{R1_START}%", 22, MUTED, weight=m.BOLD).next_to(b_start, m.RIGHT, buff=0.15)
        v_end = code(f"{R1_END}%", 22, ACTION, weight=m.BOLD).next_to(b_end, m.RIGHT, buff=0.15)
        v_maj = code(f"{R1_MAJ}%", 22, ACTION, weight=m.BOLD).next_to(b_maj, m.RIGHT, buff=0.15)
        start_note = cite("at the start").next_to(v_start, m.RIGHT, buff=0.2)
        from_note = cite(f"from {R1_START}%").next_to(v_end, m.RIGHT, buff=0.25)
        assert from_note.get_right()[0] < 6.5
        assert v_maj.get_right()[0] < 6.5 and lab1.get_left()[0] > -6.6
        res_src1 = cite(f"DeepSeek-AI, Jan 2025 preprint (arXiv v1), Table 2 · pass@1 averaged over {R1_SAMPLES} samples")
        res_src1.move_to([0, -2.3, 0])
        res_src2 = cite(f"the later Nature version reports {R1_NATURE}% after a longer run").move_to([0, -2.66, 0])
        ep43 = pill("majority voting · AI, Explained, episode 43", MUTED, CARD, 20).move_to([0, -3.18, 0])
        assert ep43.get_bottom()[1] > -3.55

        self.say(
            "[In January twenty twenty-five, the DeepSeek team pushed this idea all the way], [with a model called R1-Zero.]",
            m.AnimationGroup(self.new_header("03 · DeepSeek-R1-Zero", "Pure RL on a base model"),
                             m.FadeIn(cards[0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(cards[1], shift=0.15 * m.UP), m.GrowArrow(links[1]),
                             m.FadeIn(cards[2], shift=0.15 * m.UP)),
        )
        self.say(
            "[They took a base model, with no fine-tuning on worked examples], "
            "[and ran GRPO from episode twenty-three], [straight on it.]",
            m.Circumscribe(cards[0], color=MUTED, buff=0.08),
            m.Circumscribe(cards[1], color=REWARD, buff=0.08),
            m.AnimationGroup(pulse(links[0], ACTION), pulse(links[1], ACTION)),
        )
        self.say(
            "[Two simple rules gave the reward]: [is the final answer right], [and is the thinking wrapped in think tags?]",
            m.Create(forks),
            m.FadeIn(r_acc, shift=0.1 * m.DOWN),
            m.FadeIn(r_fmt, shift=0.1 * m.DOWN),
        )
        self.say(
            "[No neural reward model at all]: [they found one could be hacked at this scale.]",
            m.FadeIn(no_rm, shift=0.1 * m.UP),
            m.FadeIn(no_rm_src),
        )
        self.say(
            "[On a hard maths competition from twenty twenty-four], [single-try accuracy rose from fifteen point six percent], "
            "[to seventy-one.]",
            m.AnimationGroup(m.FadeOut(no_rm), m.FadeOut(no_rm_src), m.FadeIn(lab1), m.FadeIn(res_src1)),
            m.AnimationGroup(m.GrowFromEdge(b_start, m.LEFT), m.FadeIn(v_start), m.FadeIn(start_note)),
            m.AnimationGroup(m.GrowFromEdge(b_end, m.LEFT, run_time=1.2), m.FadeOut(start_note), m.FadeIn(v_end),
                             m.FadeOut(v_start), m.FadeIn(from_note), m.FadeIn(res_src2)),
        )
        self.say(
            "[With a majority vote over sixty-four answers, as in AI, Explained episode forty-three], "
            "[it reached eighty-six point seven.]",
            m.AnimationGroup(m.FadeIn(lab2), m.FadeIn(ep43, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(b_maj, m.LEFT, run_time=1.0), m.FadeIn(v_maj)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · longer answers

    def longer_answers(self):
        self.next_section("4 · Longer answers, and an aha moment", notes=(
            "R1-Zero's average response length on the training set grew during RL (DeepSeek-AI 2025, Fig. 3: 'DeepSeek-R1-"
            "Zero naturally learns to solve reasoning tasks with more thinking time'; the text: 'from generating hundreds "
            "to thousands of reasoning tokens'). The curve here is a SKETCH of that shape, with no numbers (the paper's "
            "values can only be read off the plot). The aha moment: 'Wait, wait. Wait. That's an aha moment I can flag "
            "here.' (an R1-Zero output, Table 3 in v1). Debate (notes §24): Liu et al. 2025 (Dr. GRPO) report that "
            "'DeepSeek-V3-Base already exhibit \"Aha moment\"'; Yue et al. 2025 find RLVR's reasoning paths already in "
            "the base model's samples. OUR run (lm_rl.json curves, one run, 10-step moving averages): the training reward "
            f"rose (mean of steps 1–25 {H['train_reward_first25']:.3f} → steps 176–200 {H['train_reward_last25']:.3f}) "
            f"while the mean training-answer length stayed flat ({H['train_len_first10']:.1f} → {H['train_len_last10']:.1f}"
            f" tokens, first vs last 10 steps); held-out greedy answers {H['greedy_len_before']:.1f} → "
            f"{H['greedy_len_after']:.1f} tokens. R1-Zero ran {R1_STEPS:,} steps with outputs up to {R1_MAXLEN:,} tokens "
            "(Nature version / arXiv v2 §2); ours 200 steps, at most 512 new tokens. We do not claim R1-style length "
            "growth: it needs far longer runs and bigger models."))
        sk = Plot((0, 1), (0, 1), width=5.2, height=2.7, center=(-3.25, 0.25, 0), x_label="training steps",
                  y_label="R1-Zero: average answer length")
        rng = np.random.default_rng(24)
        t = np.linspace(0, 1, 90)
        shape = 0.08 + 0.78 * t ** 1.15 + 0.035 * np.sin(31 * t) * t + rng.normal(0, 0.012, t.size)
        sketch = sk.line(t, np.clip(shape, 0.03, 0.97), ACTION, 4)
        lo = text("hundreds of tokens", 20, ACTION).move_to(sk.coords(0.44, 0.07))
        hi = text("thousands", 20, ACTION).next_to(sk.coords(0.97, float(shape[-1])), m.UP, buff=0.18)
        hi.align_to(sk.coords(1.0, 0), m.RIGHT)
        sk_src = cite("a sketch of the shape · DeepSeek-AI, 2025, Fig. 3").next_to(sk, m.DOWN, buff=0.18)

        lines = ["“Wait, wait. Wait. That's an aha", "moment I can flag here.”"]
        qbody = m.VGroup(*[text(s, 26, INK) for s in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        q_label = code("R1-Zero, mid-answer", 20, ACTION, weight=m.BOLD)
        inner = m.VGroup(q_label, qbody).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
        qbox = card(5.6, inner.height + 0.5, EDGE, CARD, radius=0.14, stroke=1.5)
        stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=qbox.height - 0.18, stroke_width=0,
                                    fill_color=ACTION, fill_opacity=1)
        stripe.move_to(qbox).align_to(qbox, m.LEFT).shift(0.1 * m.RIGHT)
        inner.move_to(qbox).align_to(qbox, m.LEFT).shift(0.35 * m.RIGHT)
        quote = m.VGroup(qbox, stripe, inner).move_to([3.45, 1.0, 0])
        q_src = cite("DeepSeek-AI, 2025, Table 3").next_to(quote, m.DOWN, buff=0.15).align_to(quote, m.LEFT)
        debate = pill("already in the base model? · Liu et al., 2025", EXPLORE, VIOLET_FILL, 20).move_to([3.45, -1.0, 0])
        live = cite("a live debate · Yue et al., 2025, too").next_to(debate, m.DOWN, buff=0.15)
        assert quote.get_right()[0] < 6.5 and debate.get_right()[0] < 6.5 and sk.get_right()[0] < quote.get_left()[0] - 0.2

        # phase B: our run
        rp = Plot((0, 200), (0.4, 0.65), width=4.6, height=2.3, center=(-3.35, 0.3, 0), x_ticks=[0, 100, 200],
                  y_ticks=[0.4, 0.5, 0.6], x_label="training step", y_label="reward · 10-step average", y_fmt="{:.1f}")
        lp = Plot((0, 200), (0, 600), width=4.6, height=2.3, center=(3.3, 0.3, 0), x_ticks=[0, 100, 200],
                  y_ticks=[0, 300, 600], x_label="training step", y_label="answer length, tokens")
        r_curve = rp.line(STEPS[9:], REWARD_MA[9:], REWARD, 4)              # from the first full 10-step window
        l_curve = lp.line(STEPS[9:], LEN_MA[9:], ACTION, 4)
        flat = lp.hline(300, MUTED, 2)
        held = code(f"held-out answers: {H['greedy_len_before']:.0f} → {H['greedy_len_after']:.0f} tokens", 20, ACTION)
        held.next_to(lp, m.DOWN, buff=0.2)
        ours_src = cite("trained on this Mac · one run · lm_rl.json").next_to(rp, m.DOWN, buff=0.2)
        ours_pill = pill("our run · 200 steps · at most 512 tokens", ACTION, CORAL_FILL, 20)
        r1_pill = pill(f"R1-Zero · {R1_STEPS:,} steps · up to {R1_MAXLEN:,} tokens", MUTED, CARD, 20)
        cmp_row = m.VGroup(ours_pill, r1_pill).arrange(m.RIGHT, buff=0.4).move_to([0, -2.55, 0])
        r1_cite = cite("R1-Zero's run: the Nature version (Guo et al., 2025)").move_to([0, -3.1, 0])
        assert cmp_row.width < 12.8 and held.get_bottom()[1] > cmp_row.get_top()[1] + 0.15
        assert ours_src.get_bottom()[1] > cmp_row.get_top()[1] + 0.15

        self.say(
            "[Something else grew during training]: [the answers.] [They went from hundreds of tokens to thousands.]",
            m.AnimationGroup(self.new_header("04 · longer reasoning", "Longer answers, and an aha moment"),
                             m.FadeIn(sk), m.FadeIn(sk_src)),
            m.FadeIn(lo, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(sketch, run_time=2.2), m.FadeIn(hi, shift=0.1 * m.UP)),
        )
        self.say(
            "[And mid-answer, the model began to stop and re-check its own work.]",
            m.AnimationGroup(m.FadeIn(quote, shift=0.15 * m.LEFT), m.FadeIn(q_src)),
        )
        self.say(
            "[The authors called it an aha moment.] [Others found the same habit in the base model already], "
            "[so that part is still debated.]",
            m.Circumscribe(quote, color=ACTION, buff=0.08),
            m.FadeIn(debate, shift=0.1 * m.UP),
            m.FadeIn(live),
        )
        self.say(
            "[Did our answers grow too? No.] [Over two hundred steps the reward climbed], "
            "[but the length stayed near three hundred tokens.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(sk, sk_src, sketch, lo, hi, quote, q_src, debate, live)),
                             m.FadeIn(rp), m.FadeIn(lp), m.FadeIn(ours_src)),
            m.Create(r_curve, run_time=2.0),
            m.AnimationGroup(m.Create(l_curve, run_time=2.0), m.FadeIn(flat), m.FadeIn(held)),
        )
        self.say(
            "[Our run was tiny by comparison]: [two hundred steps, and answers capped at five hundred and twelve tokens.]",
            m.FadeIn(r1_pill, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(ours_pill, shift=0.1 * m.UP), m.FadeIn(r1_cite)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · our accuracy

    def our_accuracy(self):
        self.next_section("5 · Problems it never saw", notes=(
            f"Our run's evaluation (lm_rl.json headline; per-problem counts from the run's log "
            f"research/experiments/logs/lm_grpo_eval.jsonl, steps 0 and 200): {N_TEST} held-out GSM8K test problems, "
            f"{N_SAMPLES} samples each at temperature {H['eval_temperature']} (max 512 tokens), every sample scored by the "
            "checker. One square per problem, in the log's order; brightness = the fraction of its 8 samples that are "
            "right. pass@1 = the mean of those fractions (the unbiased estimator at k = 1): "
            f"{pct(P_BEFORE[1], 2)} → {pct(P_AFTER[1], 2)}; paired SE {100 * SE['pass1_gain']['se']:.2f} points "
            "(recomputed here from the per-problem differences). Format vs maths (lm_rl.json "
            "train_samples_format_vs_maths: the 1,600 training answers of steps 0–24 vs 175–199; different questions, so "
            f"indicative): reward {pct(F0['strict_acc'])} → {pct(F1['strict_acc'])}, format-free last-number accuracy "
            f"{pct(F0['last_number_acc'])} → {pct(F1['last_number_acc'])}, boxed {pct(F0['boxed_rate'])} → "
            f"{pct(F1['boxed_rate'])}: nearly all of the gain is getting the number right. One run, one seed."))
        cols, rows_n, size, gap = 25, 20, 0.165, 0.045
        assert cols * rows_n == N_TEST
        pitch = size + gap
        gx, gy = -3.55, -0.15
        squares = m.VGroup()
        for i in range(N_TEST):
            r, c = divmod(i, cols)
            sq = m.Square(size).set_stroke(EDGE, 1).set_fill(CARD, 1)
            sq.move_to([gx + (c - (cols - 1) / 2) * pitch, gy - (r - (rows_n - 1) / 2) * pitch, 0])
            squares.add(sq)
        grid_rows = [squares[r * cols:(r + 1) * cols] for r in range(rows_n)]
        legend = m.VGroup(*[m.Square(size).set_stroke(EDGE, 1).set_fill(heat_cell(c), 1) for c in range(9)])
        legend.arrange(m.RIGHT, buff=0.05)
        l_lo = cite("0").next_to(legend, m.LEFT, buff=0.15)
        l_hi = cite("8 of 8 samples right").next_to(legend, m.RIGHT, buff=0.15)
        leg = m.VGroup(l_lo, legend, l_hi).move_to([gx, squares.get_bottom()[1] - 0.38, 0])
        g_src = cite(f"{N_TEST} held-out problems · {N_SAMPLES} samples each at T = 0.8").move_to([gx, leg.get_y() - 0.45, 0])
        assert g_src.get_bottom()[1] > -3.55 and squares.get_top()[1] < 2.5 and squares.get_left()[0] > -6.5

        px = 3.55
        p_title = code("pass@1", 26, INK, weight=m.BOLD).move_to([px, 2.05, 0])
        p_sub = cite("the share of single tries that are right").next_to(p_title, m.DOWN, buff=0.12)
        s_before = title_text(pct(P_BEFORE[1]), 56, BEFORE_C).move_to([px - 1.55, 0.75, 0])
        s_after = title_text(pct(P_AFTER[1]), 56, AFTER_C).move_to([px + 1.55, 0.75, 0])
        s_arrow = arrow(s_before.get_right(), s_after.get_left(), MUTED, buff=0.15)
        lb = cite("before").next_to(s_before, m.DOWN, buff=0.12)
        la = cite("after 200 steps").next_to(s_after, m.DOWN, buff=0.12)
        gain = code(f"+{GAIN[1]:.1f} ± {100 * SE['pass1_gain']['se']:.1f} points", 26, GREEN, weight=m.BOLD)
        gain.move_to([px, -0.3, 0])
        gain_src = cite("paired standard error · one run").next_to(gain, m.DOWN, buff=0.1)
        assert s_after.get_right()[0] < 6.5 and s_before.get_left()[0] > squares.get_right()[0] + 0.4

        head = cite("on training answers, first 25 steps → last 25")
        trows = [code(f"{name:<15}  {pct(a)} → {pct(b)}   +{100 * (b - a):.1f}", 20, INK,
                      t2c={r"\+\d+\.\d": GREEN if name != "boxed at all" else MUTED}) for name, a, b in FMT]
        table = m.VGroup(head, *trows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        left_at(table, 0.05, -2.05)
        t_src = cite("different questions early and late · indicative").next_to(table, m.DOWN, buff=0.14)
        assert table.get_right()[0] < 6.5 and table.get_top()[1] < gain_src.get_bottom()[1] - 0.15, (table.get_right(), table.get_top(), gain_src.get_bottom(), table.width)
        assert t_src.get_bottom()[1] > -3.55

        self.say(
            "[So, did our run learn? We test it on five hundred problems it never trained on.]",
            m.AnimationGroup(self.new_header("05 · our run", "Problems it never saw"),
                             m.LaggedStart(*[m.FadeIn(rw) for rw in grid_rows], lag_ratio=0.08, run_time=1.6)),
        )
        self.say(
            "[Each gets eight samples at temperature zero point eight], [and the checker stamps all four thousand]: "
            "[the brighter the square, the more of its eight are right.]",
            m.FadeIn(g_src),
            m.LaggedStart(*[m.AnimationGroup(*[sq.animate.set_fill(heat_cell(C0[r * cols + j]), 1)
                                               for j, sq in enumerate(grid_rows[r])]) for r in range(rows_n)],
                          lag_ratio=0.15, run_time=2.4),
            m.FadeIn(leg, shift=0.1 * m.UP),
        )
        self.say(
            "[Pass at one is the share of single tries that are right]: [thirty-eight point three percent before training.]",
            m.AnimationGroup(m.FadeIn(p_title), m.FadeIn(p_sub)),
            m.AnimationGroup(m.FadeIn(s_before, shift=0.1 * m.UP), m.FadeIn(lb)),
        )
        self.say(
            "[After two hundred steps]: [forty-six point two.] [That's seven point nine points better, give or take one.]",
            m.LaggedStart(*[m.AnimationGroup(*[sq.animate.set_fill(heat_cell(C1[r * cols + j]), 1)
                                               for j, sq in enumerate(grid_rows[r])]) for r in range(rows_n)],
                          lag_ratio=0.15, run_time=2.0),
            m.AnimationGroup(m.GrowArrow(s_arrow), m.FadeIn(s_after, shift=0.1 * m.UP), m.FadeIn(la)),
            m.AnimationGroup(m.FadeIn(gain, shift=0.1 * m.UP), m.FadeIn(gain_src)),
        )
        self.say(
            "[Is it just neater formatting? Mostly not]: [read the last number and ignore the box], "
            "[and accuracy still rose eight points.]",
            m.AnimationGroup(m.FadeIn(head), m.FadeIn(trows[0]), m.FadeIn(t_src)),
            m.FadeIn(trows[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(trows[1], color=GREEN, buff=0.06), m.FadeIn(trows[2], shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · pass at k

    def pass_at_k_section(self):
        self.next_section("6 · pass@k", notes=(
            "pass@k = the chance that at least one of k answers is right. Unbiased estimator from n samples with c right "
            "(Chen et al. 2021, the Codex paper): 1 − C(n−c, k)/C(n, k) (rl_reference.pass_at_k). John's trip "
            f"(test-788): the base model got {J_BEFORE} of its 8 samples right (the run's eval log; the ORDER drawn is "
            f"illustrative): pass@1 = 5/8, pass@2 = 1 − 3/28 = {J_PASS[2]:.4f}, pass@4 = 1 (only 3 are wrong); after "
            f"training {J_AFTER} of 8. Averaged over the 500 problems, the estimator reproduces lm_rl.json's headline "
            "(asserted). The curves: pass@k at k = 1, 2, 4, 8 before and after 200 steps (measured here, one run); gains "
            f"{', '.join(f'+{GAIN[k]:.1f}' for k in KS)} points; pass@8 gain ± {100 * SE['pass8_gain']['se']:.1f} (paired "
            f"SE); the gap between 1 try and 8 narrowed from {GAP_BEFORE:.1f} to {GAP_AFTER:.1f} points. Yue et al. 2025 "
            "(arXiv 2504.13837): RLVR models win at small k, base models overtake at large k (e.g. Minerva, 32B, the base "
            f"model ahead by about {YUE_GAP}% at k = {YUE_K}); the inset is a schematic of that shape, not data. Our run "
            "does not reach large k: at k ≤ 8 pass@8 still rose. DeepSeekMath (Shao et al. 2024) saw the same direction."))
        n = 8
        right = [True, False, True, True, False, True, True, False]              # 5 of 8 (the order: for example)
        assert sum(right) == J_BEFORE
        tiles, marks = [], []
        for i, ok in enumerate(right):
            box = card(1.0, 1.0, ACTION, CORAL_FILL, radius=0.12, stroke=2.5)
            lab = code(f"#{i + 1}", 20, MUTED).move_to(box.get_top() + [0, -0.22, 0])
            tiles.append(m.VGroup(box, lab).move_to([(i - 3.5) * 1.3, 1.65, 0]))
            marks.append(mark(ok).move_to(box.get_center() + [0, -0.12, 0]))
        t_src = cite("John's trip · the base model's n = 8 samples, c = 5 right · the order is for example")
        t_src.move_to([0, 0.88, 0])

        f = formula([r"\mathrm{pass@}k", "=", "1", "-", r"\frac{\binom{n-c}{k}}{\binom{n}{k}}"],
                    [ACTION, INK, INK, INK, INK], 40)
        f.move_to([-3.6, -0.05, 0])
        rows_s = ["k = 1    5/8          = 62.5%", "k = 2    1 − 3/28     = 89.3%", "k = 4    1 − 0/70     = 100%"]
        assert rows_s[0].endswith(pct(J_PASS[1])) and rows_s[1].endswith(pct(J_PASS[2])) and J_PASS[4] == 1
        assert math.comb(3, 4) == 0 and math.comb(8, 4) == 70
        krows = [left_at(code(s, 22, INK, t2c={r"= \d+\.?\d*%": ACTION}), -6.2, y) for s, y in zip(rows_s, (-1.15, -1.75, -2.35))]
        src = ("def pass_at_k(n, c, k):\n"
               "    # at least one right in k picks\n"
               "    if n - c < k:\n"
               "        return 1.0\n"
               "    return 1 - comb(n-c, k) / comb(n, k)")
        blk = code_block(src, size=20, title="rl_reference.py, simplified")
        right_at(blk, 6.45, -1.55)
        blk_src = cite("Chen et al., 2021 · the Codex paper").next_to(blk, m.DOWN, buff=0.14)
        mean_line = left_at(code("mean of 500: " + " · ".join(pct(P_BEFORE[k]) for k in KS), 20, MUTED), -6.2, -3.25)
        assert blk.get_left()[0] > krows[1].get_right()[0] + 0.3 and blk.get_right()[0] < 6.5
        assert mean_line.get_top()[1] < blk.get_bottom()[1] - 0.12 and mean_line.get_right()[0] < blk_src.get_left()[0] - 0.3
        assert blk_src.get_bottom()[1] > -3.5 and mean_line.get_bottom()[1] > -3.5

        # phase B: the curves
        pl = Plot((0.6, 12), (0.3, 0.8), width=5.8, height=3.5, center=(-2.75, 0.15, 0), x_ticks=list(KS),
                  y_ticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], x_label="k, the number of tries", y_label="pass@k",
                  log_x=True, y_fmt="{:.0%}")
        before = dots_line(pl, KS, [P_BEFORE[k] for k in KS], BEFORE_C)
        after = dots_line(pl, KS, [P_AFTER[k] for k in KS], AFTER_C)
        key = m.VGroup(*[m.VGroup(m.Line([0, 0, 0], [0.45, 0, 0]).set_stroke(col, 4), text(name, 22, col)).arrange(m.RIGHT, buff=0.15)
                         for name, col in (("after 200 steps", AFTER_C), ("before", BEFORE_C))])
        key.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(key, pl.coords(0.6, 0.8)[0] + 0.35, pl.coords(1, 0.745)[1])
        lab_b, lab_a = key[1], key[0]
        vb = [code(pct(P_BEFORE[k]), 20, BEFORE_C).next_to(pl.coords(k, P_BEFORE[k]), m.LEFT if k == 1 else m.RIGHT, buff=0.14)
              for k in (1, 8)]
        va = [code(pct(P_AFTER[k]), 20, AFTER_C).next_to(pl.coords(k, P_AFTER[k]), m.LEFT if k == 1 else m.RIGHT, buff=0.14)
              for k in (1, 8)]
        assert vb[0].get_left()[0] > pl.coords(0.6, 0.3)[0] + 0.05, (vb[0].get_left(), pl.coords(0.6, 0.3))
        assert key.get_bottom()[1] > pl.coords(2, P_AFTER[2])[1] + 0.3 and key.get_right()[0] < pl.coords(4, 0.3)[0], key.get_corner(m.DR)

        def gap_mark(k):
            a, b = pl.coords(k, P_BEFORE[k]), pl.coords(k, P_AFTER[k])
            return m.Line(a + [0, 0.1, 0], b - [0, 0.1, 0]).set_stroke(GREEN, 4)

        g1, g8 = gap_mark(1), gap_mark(8)
        gy = pl.x_label.get_y() - 0.42
        gains = m.VGroup(*[code(f"+{GAIN[k]:.1f}", 22, GREEN, weight=m.BOLD).move_to([pl.coords(k, 0.3)[0], gy, 0]) for k in KS])
        g_name = text("points", 20, GREEN).next_to(gains[-1], m.RIGHT, buff=0.25)
        gain_row = m.VGroup(g_name, gains)
        pl_src = cite("measured here · one run · 8 samples per problem").move_to([-2.75, gy - 0.45, 0])
        gaps_note = m.VGroup(cite(f"1 try vs 8 tries: the gap"), cite(f"narrowed from {GAP_BEFORE:.1f} to {GAP_AFTER:.1f} points"))
        gaps_note.arrange(m.DOWN, buff=0.05).move_to([4.6, -2.95, 0])
        assert pl_src.get_bottom()[1] > -3.55 and gains[0].get_left()[0] > -6.6 and g_name.get_right()[0] < 2.4, (pl_src.get_bottom(), gains[0].get_left(), g_name.get_right())
        assert va[1].get_right()[0] < 1.7

        ib = card(3.75, 3.3, EDGE, CARD, radius=0.16, stroke=2).move_to([4.6, 0.25, 0])
        i_title = text("at large k", 22, INK, weight=m.BOLD).move_to(ib.get_top() + [0, -0.35, 0])
        ox, oy, iw, ih = ib.get_left()[0] + 0.4, ib.get_bottom()[1] + 0.65, 2.9, 1.95
        i_axes = m.VGroup(m.Line([ox, oy, 0], [ox + iw, oy, 0]), m.Line([ox, oy, 0], [ox, oy + ih, 0])).set_stroke(FAINT, 2.5)
        u = np.linspace(0, 1, 60)
        rl_y = 0.42 + 0.33 * (1 - np.exp(-u / 0.25))
        base_y = 0.12 + 0.85 * (1 - np.exp(-u / 0.55))
        cross = int(np.argmax(base_y > rl_y))
        assert 0.4 < u[cross] < 0.9
        def ipath(vals, color):
            return m.VMobject().set_points_smoothly([[ox + iw * a, oy + ih * min(b, 0.97), 0] for a, b in zip(u, vals)]).set_stroke(color, 4)
        i_rl, i_base = ipath(rl_y, AFTER_C), ipath(base_y, BEFORE_C)
        i_k = m.VGroup(code("1", 20, MUTED).next_to([ox, oy, 0], m.DOWN, buff=0.1),
                       code("k in the hundreds", 20, MUTED).next_to([ox + iw, oy, 0], m.DOWN, buff=0.1).align_to([ox + iw, 0, 0], m.RIGHT))
        i_lab = m.VGroup(text("RL-trained", 20, AFTER_C).move_to([ox + 0.95, oy + ih * 0.88, 0]),
                         text("base", 20, BEFORE_C).move_to([ox + 2.35, oy + ih * 0.4, 0]))
        i_src = m.VGroup(cite("a schematic of Yue et al., 2025"), cite(f"e.g. a 32B model at k = {YUE_K}:"),
                         cite(f"the base model ≈ {YUE_GAP}% ahead")).arrange(m.DOWN, buff=0.06)
        i_src.next_to(ib, m.DOWN, buff=0.15)
        inset = m.VGroup(ib, i_title, i_axes, i_rl, i_base, i_k, i_lab)
        assert ib.get_left()[0] > va[1].get_right()[0] + 0.3 and i_src.get_right()[0] < 6.55
        assert i_src.get_bottom()[1] > -3.55

        self.say(
            "[Pass at k asks a looser question]: [is at least one of k tries right?]",
            m.AnimationGroup(self.new_header("06 · pass at k", "The first try catches up"),
                             m.LaggedStart(*[m.FadeIn(tl, shift=0.1 * m.DOWN) for tl in tiles], lag_ratio=0.1)),
            m.FadeIn(f, shift=0.1 * m.UP),
        )
        self.say(
            "[Back to John's trip. Before training], [five of its eight samples were right]: [pass at one is five out of eight.]",
            m.FadeIn(t_src),
            m.LaggedStart(*[m.AnimationGroup(stamp_in(mk), tl[0].animate.set_stroke(GREEN if ok else RED))
                            for mk, tl, ok in zip(marks, tiles, right)], lag_ratio=0.2, run_time=1.8),
            m.FadeIn(krows[0], shift=0.1 * m.RIGHT),
        )
        wrong_tiles = m.VGroup(*[tl for tl, ok in zip(tiles, right) if not ok])
        self.say(
            "[Pick two at random]: [both are wrong only three times in twenty-eight], [so pass at two is eighty-nine percent.]",
            m.AnimationGroup(m.Circumscribe(tiles[2], color=ACTION, buff=0.06), m.Circumscribe(tiles[4], color=ACTION, buff=0.06)),
            m.LaggedStart(*[m.Circumscribe(tl, color=RED, buff=0.06) for tl in wrong_tiles], lag_ratio=0.3),
            m.FadeIn(krows[1], shift=0.1 * m.RIGHT),
        )
        self.say(
            "[Pick four], [and one is always right: only three are wrong.]",
            m.Circumscribe(m.VGroup(*tiles[4:]), color=ACTION, buff=0.08),
            m.AnimationGroup(m.FadeIn(krows[2], shift=0.1 * m.RIGHT), m.Circumscribe(tiles[5], color=GREEN, buff=0.06)),
        )
        self.say(
            "[That count is the unbiased estimator from the Codex paper], [averaged over all five hundred problems.]",
            m.AnimationGroup(m.FadeIn(blk, shift=0.1 * m.LEFT), m.FadeIn(blk_src)),
            m.FadeIn(mean_line, shift=0.1 * m.UP),
        )
        phase_a = m.VGroup(*tiles, *marks, t_src, f, *krows, blk, blk_src, mean_line)
        self.say(
            "[Before training, pass at one was thirty-eight percent], [and pass at eight was seventy-two.]",
            m.AnimationGroup(m.FadeOut(phase_a), m.FadeIn(pl), m.FadeIn(pl_src),
                             m.Create(before[0]), m.FadeIn(before[1][0]), m.FadeIn(vb[0])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in before[1][1:]], lag_ratio=0.3),
                             m.FadeIn(vb[1]), m.FadeIn(lab_b)),
        )
        self.say(
            "[After training, every k went up], [but the gain shrinks]: [almost eight points at one try], [under five at eight.]",
            m.AnimationGroup(m.Create(after[0]), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in after[1]], lag_ratio=0.25),
                             m.FadeIn(lab_a)),
            m.AnimationGroup(m.FadeIn(va[0]), m.FadeIn(va[1])),
            m.AnimationGroup(m.GrowFromCenter(g1), m.FadeIn(g_name), m.FadeIn(gains[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromCenter(g8), m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.UP) for g in gains[1:]], lag_ratio=0.3)),
        )
        self.say(
            "[Training mostly made the first try more like the best of eight.]",
            m.FadeIn(gaps_note, shift=0.1 * m.UP),
        )
        self.say(
            "[A twenty twenty-five study pushed k into the hundreds]: [there, the base models catch up, and overtake.]",
            m.AnimationGroup(m.FadeIn(ib), m.FadeIn(i_title), m.Create(i_axes), m.FadeIn(i_k), m.FadeIn(i_src)),
            m.AnimationGroup(m.Create(i_rl, run_time=1.6), m.Create(i_base, run_time=1.6), m.FadeIn(i_lab)),
        )
        self.say(
            "[So reinforcement learning may mostly sharpen what a model can already sample.] "
            "[Our run is one small, honest data point.]",
            m.Circumscribe(inset, color=EXPLORE, buff=0.06),
            m.Circumscribe(m.VGroup(pl, before, after), color=ACTION, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · why samples, not greedy

    def greedy_section(self):
        self.next_section("7 · Why the headline uses samples", notes=(
            "lm_rl.json greedy_by_batch_size: the SAME weights decoded greedily (temperature 0) on this Mac at generation "
            f"batch sizes {BATCHES}: base {[pct(v) for v in G_BASE]}, trained {[pct(v) for v in G_TRAIN]}; only "
            f"{pct(GB['stable_fraction_base'])} of problems (base; trained {pct(GB['stable_fraction_trained'])}) keep the "
            f"same correctness across batch sizes, so {pct(FLIP)} flip (bf16 batched decoding with left padding is not "
            "batch-invariant: floating-point sums depend on their order; 'Tokens per Second' Episode 17). The greedy gain "
            f"at one batch size ranges from +{100 * min(G_GAINS):.1f} to +{100 * max(G_GAINS):.1f} points. Hence the "
            f"headline: pass@1 from {N_TEST * N_SAMPLES:,} samples, +{GAIN[1]:.1f} ± {100 * SE['pass1_gain']['se']:.1f}. "
            "DeepSeek's R1 paper also reports pass@1 averaged over 4–64 samples (temperature 0.6, top-p 0.95), avoiding "
            "greedy decoding for its 'higher repetition rates and significant variability' (notes §24)."))
        pl = Plot((16, 256), (0.44, 0.52), width=5.6, height=3.3, center=(-2.75, 0.2, 0), x_ticks=BATCHES,
                  y_ticks=[0.44, 0.46, 0.48, 0.50, 0.52], x_label="generation batch size", y_label="greedy accuracy",
                  log_x=True, y_fmt="{:.0%}")
        base = dots_line(pl, BATCHES, G_BASE, BEFORE_C)
        trained = dots_line(pl, BATCHES, G_TRAIN, AFTER_C)
        lab_b = text("base", 22, BEFORE_C).next_to(pl.coords(256, G_BASE[-1]), m.RIGHT, buff=0.22)
        lab_t = text("trained", 22, AFTER_C).next_to(pl.coords(256, G_TRAIN[-1]), m.RIGHT, buff=0.22)
        band = pl.band([16, 256], [min(G_BASE)] * 2, [max(G_BASE)] * 2, BEFORE_C, 0.16)
        rng_lab = code(f"{pct(min(G_BASE))} – {pct(max(G_BASE))}", 22, INK, weight=m.BOLD)
        rng_lab.next_to(pl.coords(16, max(G_BASE)), m.UP + 0.6 * m.RIGHT, buff=0.18)
        pl_src = cite("the same weights, decoded greedily on this Mac · one run").move_to([-2.75, -2.55, 0])
        g_note = cite(f"the greedy gain: +{100 * min(G_GAINS):.1f} to +{100 * max(G_GAINS):.1f} points, "
                      "depending on batch size").move_to([-2.75, -2.95, 0])
        assert lab_t.get_right()[0] < 1.6

        rx = 4.05
        flip = title_text(f"{round(100 * FLIP)}%", 72, RED).move_to([rx, 1.55, 0])
        flip_t = m.VGroup(text("of problems flip", 22, INK), text("between right and wrong", 22, INK),
                          text("from batch size alone", 22, INK)).arrange(m.DOWN, buff=0.06)
        flip_t.next_to(flip, m.DOWN, buff=0.18)
        tps = pill("Tokens per Second · episode 17", MUTED, CARD, 20).next_to(flip_t, m.DOWN, buff=0.3)
        tps_sub = cite("batched sums add up in a different order").next_to(tps, m.DOWN, buff=0.12)
        head = m.VGroup(code(f"pass@1 from {N_TEST * N_SAMPLES:,} samples", 22, INK, weight=m.BOLD),
                        code(f"+{GAIN[1]:.1f} ± {100 * SE['pass1_gain']['se']:.1f} points", 26, GREEN, weight=m.BOLD))
        head.arrange(m.DOWN, buff=0.12)
        hbox = card(head.width + 0.6, head.height + 0.45, GREEN, CARD, radius=0.16, stroke=2.5)
        headline = m.VGroup(hbox, head.move_to(hbox)).move_to([rx, -1.95, 0])
        ds = m.VGroup(cite("DeepSeek's papers, too, report"), cite("pass@1 averaged over samples")).arrange(m.DOWN, buff=0.05)
        ds.next_to(headline, m.DOWN, buff=0.18)
        for t_ in (flip_t, tps, tps_sub, headline, ds):
            assert t_.get_right()[0] < 6.5, (t_.get_right(), t_.width)
        assert tps_sub.get_bottom()[1] > headline.get_top()[1] + 0.15 and flip.get_top()[1] < 2.55
        assert headline.get_left()[0] > pl.get_right()[0] + 0.3 and ds.get_bottom()[1] > -3.5

        self.say(
            "[Why not grade the single most likely answer, the greedy one?] [Because on this Mac, it isn't stable.]",
            m.AnimationGroup(self.new_header("07 · measuring honestly", "Why the headline uses samples"), m.FadeIn(pl),
                             m.FadeIn(pl_src)),
            m.AnimationGroup(m.Create(base[0]), m.FadeIn(base[1]), m.FadeIn(lab_b)),
        )
        self.say(
            "[The same weights, decoded greedily], [score from forty-six to forty-eight percent], "
            "[depending only on the batch size.]",
            m.Circumscribe(base, color=MUTED, buff=0.1),
            m.AnimationGroup(m.FadeIn(band), m.FadeIn(rng_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(trained[0]), m.FadeIn(trained[1]), m.FadeIn(lab_t), m.FadeIn(g_note)),
        )
        self.say(
            "[About fourteen percent of problems flip between right and wrong]: "
            "[batched sums add up in a different order, as in Tokens per Second, episode seventeen.]",
            m.AnimationGroup(m.FadeIn(flip, scale=0.8), m.FadeIn(flip_t, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(tps, shift=0.1 * m.UP), m.FadeIn(tps_sub)),
        )
        self.say(
            "[Four thousand samples average that noise away]: [so our headline is pass at one, up seven point nine points.]",
            m.FadeIn(headline, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(headline, color=GREEN, buff=0.08), m.FadeIn(ds)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            "Check (a program gives the reward: RLVR), sample (measure with many tries: pass@1 from samples, pass@k), "
            f"sharpen (RL mostly raised the first try: our pass@1 {pct(P_BEFORE[1])} → {pct(P_AFTER[1])}, pass@8 "
            f"{pct(P_BEFORE[8])} → {pct(P_AFTER[8])})."))
        halo = glow(REWARD, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Check", 56, REWARD), title_text("Sample", 56, ACTION), title_text("Sharpen", 56, SIGNAL))
        words.arrange(m.RIGHT, buff=1.4).move_to([0, 1.1, 0])
        on_baseline(words)
        subs = m.VGroup(*[m.VGroup(text(a, 22, MUTED), text(b, 22, MUTED)).arrange(m.DOWN, buff=0.08)
                          for a, b in (("a program gives", "the reward"), ("measure with", "many tries"),
                                       ("the first try", "catches up"))])
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), 0, 0]).align_to([0, words[1].get_bottom()[1] - 0.3, 0], m.UP)
        assert all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        res = code(f"pass@1  {pct(P_BEFORE[1])} → {pct(P_AFTER[1])}     pass@8  {pct(P_BEFORE[8])} → {pct(P_AFTER[8])}",
                   24, INK, t2c={r"\d+\.\d%(?=\s+pass)|\d+\.\d%$": ACTION})
        res.move_to([0, -1.55, 0])
        res_src = cite("our run · 500 held-out problems · trained on this Mac").next_to(res, m.DOWN, buff=0.15)
        model = icon(POLICY_ICON, 0.7)
        ck = icon(CHECK, 0.6)
        pair = m.VGroup(model, m.Arrow([0, 0, 0], [0.8, 0, 0], buff=0, color=MUTED, stroke_width=5,
                                       max_tip_length_to_length_ratio=0.3), ck)
        pair.arrange(m.RIGHT, buff=0.3).move_to([0, -2.95, 0])

        self.say(
            "[To sum up]: [check], [sample], [sharpen.]",
            m.AnimationGroup(self.new_header("08 · recap", "Check, sample, sharpen"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[A checker gives a reward no one can argue with], [samples measure the model honestly], "
            "[and reinforcement learning sharpens its first try.]",
            m.AnimationGroup(m.Circumscribe(words[0], color=REWARD, buff=0.12), m.FadeIn(pair, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Circumscribe(words[1], color=ACTION, buff=0.12), m.FadeIn(res_src)),
            m.AnimationGroup(m.Circumscribe(words[2], color=SIGNAL, buff=0.12), m.FadeIn(res, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()
