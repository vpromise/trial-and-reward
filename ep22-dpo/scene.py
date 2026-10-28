"""Trial and Reward · Episode 22 · Direct Preference Optimization (Module 6 · Teaching Language Models with Reward)

The preference objective can be optimised directly, without a reward model. Episode 21 fitted a reward model to
preference pairs, then ran PPO against it on a KL leash (two training runs, fresh samples every step, four models in
memory). DPO (Rafailov et al. 2023) skips both: the derivation in four tex lines — the KL-regularised objective
(Episode 20) → its closed-form optimum π* = π_ref·exp(r/β)/Z (a 3-answer toy, notes §22 worked example 2, computed
here) → solved for the reward: the implicit reward β·log π/π_ref + β·log Z → plugged into Bradley–Terry (Episode 21):
the constant cancels, leaving −log σ(β[log π_θ/π_ref (chosen) − log π_θ/π_ref (rejected)]). The running example is the
REAL pair from our run on this Mac (research/experiments/lm_rl.md, "Episode 22 — DPO on one pair", lm_rl.json dpo):
the trucks question, chosen 'x + 2x = 300 → 100' (240 tokens) vs rejected '2x = 300 → 150' (54 tokens). The base model
(the reference) finds the wrong answer more likely (−28.3 vs −41.5 nats; per token the right one is likelier, −0.17 vs
−0.52); after GRPO (which never saw a pair) the wrong answer fell 9.3 nats, the right one 3.7; the DPO loss at
β = 0.05 / 0.1 / 0.5 is 0.563 / 0.452 / 0.059 vs ln 2 = 0.693 at the start (rl_reference.dpo_loss and
dpo_implicit_reward on the recorded log-probs, asserted against lm_rl.json). A short DPO run on 239 pairs (58 steps,
6 min) pulled the pair's log-probs apart, drove the batch loss to 0.007 and overfit: held-out greedy accuracy 43.4% vs
47.4% for the base decoded the same way. What DPO gives up (on-policy exploration), the paper's results and the
disagreement after it, four variants (IPO, KTO, ORPO, SimPO). Recap: solve, invert, compare. Episode 23: GRPO.

Every number on screen is computed below from rl_reference.py and asserted, loaded with measured(...), or cited.

Sources (research/notes.md §22, Appendix A row 22, Appendix C row 22; read October 2026):
- Rafailov, Sharma, Mitchell, Ermon, Manning & Finn 2023, "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model" (NeurIPS 2023; equation numbers from arXiv v3): eq. 3 (objective), eq. 4 (optimum, proof
  App. A.1), eq. 5 (reward from the policy), eq. 6 (Bradley–Terry, Z cancels), eq. 7 (the loss); App. B (β = 0.1
  default); §6.1 IMDb ("DPO produces by far the most efficient frontier"), §6.2 TL;DR (GPT-4-judged win rate ≈ 61% vs
  PPO 57% at temperature 0), human study (DPO preferred 58% over PPO); caveat: TL;DR/HH numbers are GPT-4-judged.
- Later evidence (Appendix A row 22: "present as open"): Llama Team 2024 §4.1.4 (chose DPO: "required less compute
  for large-scale models and performed better, especially on instruction following benchmarks like IFEval"); Xu et al.
  2024 ("PPO is able to surpass other alignment methods in all cases"); Ivison et al. 2024 ("PPO outperforms DPO by
  up to 2.5% in math and 1.2% in general domains"). Chosen likelihood falling: Rafailov, Hejna, Park & Finn 2024 ("From r
  to Q*") §5.4, citing Pal et al. 2024.
- Variants (notes §22 table): IPO — Azar et al., arXiv Oct 2023, AISTATS 2024; KTO — Ethayarajh et al., ICML 2024;
  ORPO — Hong, Lee & Thorne, EMNLP 2024; SimPO — Meng, Xia & Chen, NeurIPS 2024.
- research/experiments/lm_rl.md + lm_rl.json (measured("lm_rl", ...)): dpo.showcase (question, answers, token
  counts), dpo.pair_scores (base / GRPO-trained / DPO-trained log-probs, deltas, losses and margins at β 0.05/0.1/0.5),
  dpo.curve (58 steps: batch loss, reward accuracy, the showcase pair's log-probs during training), dpo.n_pairs 239,
  n_questions 400, samples_per_question 4, beta_train 0.1, lr 1e-5, epochs 2, pairs_per_step 8, train_seconds,
  heldout_greedy_acc_after_dpo 0.434 (500 problems); headline.greedy_by_batch_size.base["64"] = 0.474 (the same decode
  batch size as the DPO eval, per lm_rl.md). One run, one seed (lm_rl.md caveats 2 and 7).

Illustrative (said "for example" or labelled on screen): the 3-answer toy of section 2 (π_ref = 0.5/0.3/0.2, rewards
1/2/0, β = 1: notes §22 worked example 2, recomputed and asserted here). Excerpts of the two answers are verbatim
pieces of the recorded texts (LaTeX delimiters dropped).

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

PREFIX = "ep22"
LM = "lm_rl"

# ---------------------------------------------------------------------------- the toy optimum (notes §22 example 2)

TOY_REF = np.array([0.5, 0.3, 0.2])
TOY_R = np.array([1.0, 2.0, 0.0])
TOY_BETA = 1.0
_toy_w = TOY_REF * np.exp(TOY_R / TOY_BETA)
TOY_Z = float(_toy_w.sum())
TOY_OPT = _toy_w / TOY_Z
assert [round(float(p), 3) for p in TOY_OPT] == [0.360, 0.587, 0.053] and round(TOY_Z, 3) == 3.776
assert np.allclose(TOY_BETA * np.log(TOY_OPT / TOY_REF), TOY_R - TOY_BETA * math.log(TOY_Z))   # reward, up to a constant
_toy_obj = float(TOY_OPT @ TOY_R - TOY_BETA * np.sum(TOY_OPT * np.log(TOY_OPT / TOY_REF)))
assert abs(_toy_obj - TOY_BETA * math.log(TOY_Z)) < 1e-12 and round(_toy_obj, 3) == 1.329         # the optimum's value

# ---------------------------------------------------------------------------- our pair (measured on this Mac)

DPO = measured(LM, "dpo")
SHOW = DPO["showcase"]
QUESTION = SHOW["question"]
assert QUESTION.startswith("On a highway, the total number of vehicles passing is 300. If the number of cars")
assert SHOW["ref_answer"] == 100
assert "x + 2x = 300" in SHOW["chosen"] and "3x = 300" in SHOW["chosen"] and "x = 100" in SHOW["chosen"]
assert "\\boxed{100}" in SHOW["chosen"]
assert "$2x = 300$" in SHOW["rejected"] and "\\boxed{150}" in SHOW["rejected"]
N_W, N_L = SHOW["chosen_tokens"], SHOW["rejected_tokens"]
assert (N_W, N_L) == (240, 54)                                       # said: "fifty-four tokens against two hundred forty"
assert DPO["n_questions"] == 400 and DPO["samples_per_question"] == 4  # said: "answered it four times"

BASE, GR, DT = DPO["pair_scores"]
assert BASE["policy"].startswith("base") and GR["policy"].startswith("GRPO") and DT["policy"].startswith("DPO")
REF_W, REF_L = BASE["ref_logp_chosen"], BASE["ref_logp_rejected"]
assert BASE["logp_chosen"] == REF_W and BASE["logp_rejected"] == REF_L
assert GR["ref_logp_chosen"] == REF_W and GR["ref_logp_rejected"] == REF_L
assert (round(REF_W, 1), round(REF_L, 1)) == (-41.5, -28.3) and REF_L > REF_W   # said: the wrong one is more likely
TOK_W, TOK_L = REF_W / N_W, REF_L / N_L
assert (round(TOK_W, 2), round(TOK_L, 2)) == (-0.17, -0.52) and TOK_W > TOK_L    # said: per token, the right one
BETAS = (0.05, 0.1, 0.5)
LN2 = math.log(2)
for _b in BETAS:
    assert abs(float(dpo_loss(REF_W, REF_L, REF_W, REF_L, beta=_b)) - LN2) < 1e-12   # π = π_ref: ln 2 for every β
    assert abs(BASE[f"dpo_loss_beta_{_b}"] - LN2) < 1e-9
assert round(LN2, 3) == 0.693

G_W, G_L = GR["logp_chosen"], GR["logp_rejected"]
D_W, D_L = float(dpo_implicit_reward(G_W, REF_W, 1.0)), float(dpo_implicit_reward(G_L, REF_L, 1.0))   # Δ = log π − log π_ref
assert abs(D_W - GR["delta_chosen"]) < 1e-9 and abs(D_L - GR["delta_rejected"]) < 1e-9
assert (round(G_W, 1), round(G_L, 1)) == (-45.3, -37.6)
assert (round(D_W, 1), round(D_L, 1)) == (-3.7, -9.3)               # said: "three point seven", "nine point three"
assert round(D_L / D_W, 1) == 2.5                                     # said: "two and a half times harder"
GAP = D_W - D_L
assert round(GAP, 1) == 5.6                                           # said: "a gap of five point six"
MARGIN = {b: float(dpo_implicit_reward(G_W, REF_W, b) - dpo_implicit_reward(G_L, REF_L, b)) for b in BETAS}
LOSS = {b: float(dpo_loss(G_W, G_L, REF_W, REF_L, beta=b)) for b in BETAS}
for _b in BETAS:
    assert abs(MARGIN[_b] - GR[f"implicit_reward_margin_beta_{_b}"]) < 1e-6
    assert abs(LOSS[_b] - GR[f"dpo_loss_beta_{_b}"]) < 1e-9
    assert abs(LOSS[_b] - float(dpo_loss(MARGIN[_b], 0.0, 0.0, 0.0, beta=1.0))) < 1e-12   # the curve: loss of the margin
assert [round(LOSS[b], 3) for b in BETAS] == [0.563, 0.452, 0.059]   # said: 0.56, 0.45, 0.06
assert [round(MARGIN[b], 2) for b in BETAS] == [0.28, 0.56, 2.80]     # said: "zero point five six" (β 0.1)
assert all(LOSS[b] < LN2 for b in BETAS)
WEIGHT = {b: float(sigmoid(-MARGIN[b])) for b in BETAS}               # σ(r̂_l − r̂_w), the gradient's weight
assert round(float(sigmoid(0.0)), 2) == 0.5 and round(WEIGHT[0.5], 2) == 0.06

# the short DPO run
T_W, T_L = DT["logp_chosen"], DT["logp_rejected"]
assert (round(T_W, 1), round(T_L, 1)) == (-48.9, -72.0)              # said: "minus seventy-two against minus forty-nine"
assert DPO["n_pairs"] == 239 and DPO["beta_train"] == 0.1 and DPO["lr"] == 1e-5      # said: "two hundred thirty-nine"
assert DPO["pairs_per_step"] == 8 and DPO["epochs"] == 2
CURVE = DPO["curve"]
STEPS = len(CURVE)
assert STEPS == 58 == DPO["epochs"] * (DPO["n_pairs"] // DPO["pairs_per_step"])      # said: "fifty-eight steps"
assert [c["step"] for c in CURVE] == list(range(STEPS))
MINUTES = DPO["train_seconds"] / 60
assert round(MINUTES) == 6                                            # said: "six minutes"
C_W = [c["show_logp_chosen"] for c in CURVE]
C_L = [c["show_logp_rejected"] for c in CURVE]
LOSS_C = [c["loss"] for c in CURVE]
EP1 = STEPS // 2 - 1                                                  # step 28: the last step of the first pass
assert (round(C_W[0], 1), round(C_L[0], 1)) == (-41.9, -28.2)
assert (round(C_W[EP1], 1), round(C_L[EP1], 1)) == (-38.6, -45.0) and C_W[EP1] > C_W[0] and C_L[EP1] < C_L[0]
assert abs(C_W[-1] - T_W) < 1e-3 and abs(C_L[-1] - T_L) < 1e-3 and C_W[-1] < C_W[EP1] and C_L[-1] < C_L[EP1]
assert abs(C_W[0] - REF_W) < 0.35 and abs(C_L[0] - REF_L) < 0.05     # step 0 scored inside a training batch
CROSS = next(i for i in range(STEPS) if C_W[i] > C_L[i])
assert CROSS == 19 and all(C_W[i] > C_L[i] for i in range(CROSS, STEPS))
assert round(LOSS_C[0], 3) == 0.693 and round(LOSS_C[EP1], 3) == 0.484 and round(LOSS_C[-1], 3) == 0.007
assert CURVE[EP1]["reward_acc"] == 0.75 and CURVE[-1]["reward_acc"] == 1.0         # said: "every pair … ranked right"
ACC_DPO = DPO["heldout_greedy_acc_after_dpo"]
ACC_BASE = measured(LM, "headline", "greedy_by_batch_size", "base", "64")
ACC_GRPO = measured(LM, "headline", "greedy_by_batch_size", "trained", "64")
assert (ACC_DPO, ACC_BASE, ACC_GRPO) == (0.434, 0.474, 0.49) and DPO["heldout_n"] == 500  # said: 47.4 → 43.4
assert measured(LM, "headline", "model") == "Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------- cited (notes §22)

TLDR_DPO, TLDR_PPO, HUMAN_DPO = 61, 57, 58        # Rafailov et al. 2023 §6.2 (GPT-4 judge; human study)
DEFAULT_BETA = 0.1                                # Rafailov et al. 2023 App. B
assert DEFAULT_BETA == DPO["beta_train"]
IVISON_MATH = 2.5                                 # Ivison et al. 2024: "up to 2.5% in math"

# ---------------------------------------------------------------------------- look

POLICY_ICON = recolored("llm", ACTION, CORAL_FILL, prefix=PREFIX)
RM_ICON = recolored("llm", REWARD, GOLD_FILL, prefix=PREFIX)
VALUE_ICON = recolored("llm", VALUE, BLUE_FILL, prefix=PREFIX)
REF_ICON = recolored("llm", MUTED, CARD, prefix=PREFIX)
DICE = recolored("dice", ACTION, CORAL_FILL, prefix=PREFIX)
REF_C = MUTED          # the frozen reference model and its log-probs
BETA_C = EXPLORE       # β and the KL leash (VIOLET), as in Episodes 20–21
LOSS_COL = SIGNAL      # the loss / the learning signal (PINK)


def cite(s, size=20):
    return text(s, size, MUTED)


def left_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.RIGHT)


def formula(parts, colors, size=38):
    f = m.MathTex(*parts, font_size=size, color=INK)
    for part, col in zip(f, colors):
        part.set_color(col)
    return f


def colored(parts, size=22, weight=m.NORMAL):
    """One mono line from (string, colour) parts joined by spaces; each part keeps its colour."""
    s = " ".join(p for p, _ in parts)
    words = code(s, size, INK, weight=weight)
    glyph_of, n = {}, 0
    for i, ch in enumerate(s):
        if not ch.isspace():
            glyph_of[i] = n
            n += 1
    assert n == len(words), s
    pos = 0
    for p, col in parts:
        for i in range(pos, pos + len(p)):
            if i in glyph_of:
                words[glyph_of[i]].set_color(col)
        pos += len(p) + 1
    return words


def rich(parts, size=20, weight=m.NORMAL):
    """One Avenir line from (string, colour) parts joined by spaces (t2c on literal substrings)."""
    s = " ".join(p for p, _ in parts)
    return text(s, size, INK, weight=weight, t2c={p: c for p, c in parts if c != INK})


def tag(label, color, fill, size=22):
    """A pill whose word sits on the common baseline (kit's chip), in the pill's colour."""
    c = chip(label, color, fill, width=text(label, size, weight=m.BOLD).width + 0.6, height=0.52, size=size)
    c[0].set_stroke(color, 2).set_fill(fill, 1)
    c[1].set_color(color)
    return c


def model_card(icon_name, name, color, width=2.4, height=1.6):
    box = card(width, height, color, CARD, radius=0.18, stroke=2.5)
    body = m.VGroup(icon(icon_name, 0.58), text(name, 22, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.14)
    body.move_to(box)
    return m.VGroup(box, *body)


def badge(label, r=0.25):
    disc = m.Circle(radius=r).set_stroke(LOSS_COL, 2.5).set_fill(PINK_FILL, 1)
    return m.VGroup(disc, title_text(label, 24, LOSS_COL).move_to(disc))


def spring(a, b, waves=6, amp=0.09):
    """The KL leash as Episodes 20–21 draw it: a VIOLET spring between two points."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = b - a
    n = np.array([-d[1], d[0], 0.0]) / (np.linalg.norm(d) + 1e-9)
    ts = np.linspace(0, 1, 16 * waves + 1)
    pts = [a + t * d + amp * math.sin(2 * math.pi * waves * t) * n for t in ts]
    return m.VMobject().set_points_smoothly(pts).set_stroke(BETA_C, 4)


def strike(mob, color=RED, pad=0.08):
    return m.Line(mob.get_left() + [-pad, 0, 0], mob.get_right() + [pad, 0, 0]).set_stroke(color, 4)


def hollow(point, color, r=0.12):
    return m.Circle(radius=r).set_stroke(color, 3).set_fill(BG, 1).move_to(point).set_z_index(3)


def solid(point, color, r=0.12):
    return m.Dot(point, radius=r, color=color).set_z_index(4)


def minus(s):
    return s.replace("-", "−")


class LogAxis:
    """A horizontal axis for sequence log-probabilities (nats): .x(v) is the scene x of value v."""

    def __init__(self, lo, hi, x0, x1, y, ticks, size=20):
        self.lo, self.hi, self.x0, self.x1, self.y = lo, hi, x0, x1, y
        self.line = m.Arrow([x0 - 0.15, y, 0], [x1 + 0.35, y, 0], buff=0, color=FAINT, stroke_width=3,
                            max_tip_length_to_length_ratio=0.03)
        self.ticks = m.VGroup()
        for t in ticks:
            p = np.array([self.x(t), y, 0])
            self.ticks.add(m.Line(p + [0, 0.07, 0], p + [0, -0.07, 0]).set_stroke(FAINT, 2),
                           code(minus(f"{t:g}"), size, MUTED).next_to(p, m.DOWN, buff=0.26))
        self.group = m.VGroup(self.line, self.ticks)

    def x(self, v):
        return self.x0 + (v - self.lo) / (self.hi - self.lo) * (self.x1 - self.x0)

    def at(self, v, y):
        return np.array([self.x(v), y, 0])


# ---------------------------------------------------------------------------- the episode


class Ep22DirectPreferenceOptimization(Episode):
    number = 22

    def lesson(self):
        self.shortcut()
        self.derivation()
        self.real_pair()
        self.through_the_loss()
        self.short_run()
        self.the_trade()
        self.variants()
        self.recap()

    # ------------------------------------------------------------------ 1 · the shortcut

    def shortcut(self):
        self.next_section("1 · The shortcut", notes=(
            "Episode 21's RLHF pipeline (InstructGPT): preference pairs → a reward model (training run 1) → PPO against it "
            "with a KL penalty to the frozen reference (training run 2), sampling fresh answers at every step; four "
            "models in memory (policy, value network, reward model, reference: Hu et al. 2024, OpenRLHF). DPO (Rafailov "
            "et al. 2023, NeurIPS): 'eliminating the need for sampling from the LM during fine-tuning' — one "
            "classification-style loss on the pairs, applied to the policy, with a frozen reference (whose log-probs can "
            "be precomputed). Notes §22, 'What disappears'."))
        y = 0.25
        pairs_box = card(2.3, 1.75, EDGE, CARD).move_to([-5.2, y, 0])
        p_stack = m.VGroup(tag("chosen", GREEN, GREEN_FILL, 20), tag("rejected", RED, RED_FILL, 20))
        p_stack.arrange(m.DOWN, buff=0.12).move_to(pairs_box).shift(0.2 * m.UP)
        p_lab = text("preference pairs", 20, MUTED).next_to(p_stack, m.DOWN, buff=0.14)
        pairs = m.VGroup(pairs_box, p_stack, p_lab)
        rm = model_card(RM_ICON, "reward model", REWARD, 2.5, 1.75).move_to([-1.9, y, 0])
        pol = model_card(POLICY_ICON, "policy", ACTION, 2.2, 1.75).move_to([1.35, y, 0])
        val = model_card(VALUE_ICON, "value network", VALUE, 2.6, 1.75).move_to([4.75, y, 0])
        ref = model_card(REF_ICON, "reference", REF_C, 2.2, 1.45).move_to([1.35, -2.35, 0])
        frozen = text("frozen copy", 20, MUTED).next_to(ref, m.RIGHT, buff=0.2)
        a1 = arrow(pairs_box.get_right(), rm[0].get_left(), MUTED)
        a2 = arrow(rm[0].get_right(), pol[0].get_left(), REWARD)
        leash = spring(pol[0].get_bottom() + [0, -0.06, 0], ref[0].get_top() + [0, 0.06, 0])
        kl = code("KL", 22, BETA_C, weight=m.BOLD).next_to(leash, m.RIGHT, buff=0.18)
        run1 = pill("training run 1", MUTED, CARD, 20).move_to([rm.get_x(), 1.62, 0])
        run2 = pill("training run 2 · PPO", ACTION, CORAL_FILL, 20).move_to([pol.get_x(), 1.62, 0])
        loop = m.CurvedArrow(pol[0].get_bottom() + [-0.55, -0.05, 0], rm[0].get_bottom() + [0.45, -0.05, 0], angle=-1.3,
                             color=ACTION, stroke_width=4, tip_length=0.2)
        loop_lab = text("fresh samples, every step", 20, ACTION).move_to([-1.65, -1.5, 0])
        critic = text("PPO's critic", 20, MUTED).next_to(val, m.DOWN, buff=0.15)
        assert loop_lab.get_right()[0] < ref[0].get_left()[0] - 0.2
        four = left_at(text("four models in memory", 24, INK, weight=m.BOLD), -6.3, -2.4)
        four_sub = left_at(cite("policy · value · reward model · reference"), -6.3, -2.85)
        two = left_at(text("two models: policy + reference", 24, INK, weight=m.BOLD), -6.3, -2.4)
        two_sub = left_at(cite("no reward model · no sampling · Rafailov et al., 2023"), -6.3, -2.85)
        assert two.get_right()[0] < ref[0].get_left()[0] - 0.3
        arc = m.CurvedArrow(pairs_box.get_top() + [0, 0.08, 0], pol[0].get_top() + [-0.2, 0.08, 0], angle=-0.4 * np.pi,
                            color=LOSS_COL, stroke_width=5, tip_length=0.24)
        one_loss = pill("one loss, on the pairs", LOSS_COL, PINK_FILL, 20).move_to([rm.get_x(), 1.62, 0])
        dimmed = m.VGroup(rm, val, critic, a1, a2, loop, loop_lab)

        self.say(
            "[Episode twenty-one learned from people's choices in two stages]: [first a reward model], "
            "[then PPO against it, on a KL leash.]",
            m.AnimationGroup(self.new_header("01 · the shortcut", "Skip the reward model"), m.FadeIn(pairs, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(rm, scale=0.9)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(pol, scale=0.9), m.FadeIn(ref, shift=0.1 * m.UP), m.FadeIn(frozen),
                             m.Create(leash), m.FadeIn(kl)),
        )
        self.say(
            "[That takes two training runs], [fresh samples at every step], [and four models in memory.]",
            m.LaggedStart(m.FadeIn(run1, shift=0.1 * m.DOWN), m.FadeIn(run2, shift=0.1 * m.DOWN), lag_ratio=0.4),
            m.AnimationGroup(m.Create(loop), m.FadeIn(loop_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(val, scale=0.9), m.FadeIn(critic), m.FadeIn(four, shift=0.1 * m.UP), m.FadeIn(four_sub)),
        )
        self.say(
            "[In twenty twenty-three, a paper found a shortcut]: [no reward model, no sampling], "
            "[just one loss on the pairs, applied to the policy itself.]",
            m.AnimationGroup(m.FadeOut(four), m.FadeOut(four_sub), m.FadeIn(two_sub)),
            m.AnimationGroup(m.FadeOut(run1), m.FadeOut(run2), dimmed.animate.set_opacity(0.16), m.FadeIn(two, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(arc, run_time=1.2), m.FadeIn(one_loss, scale=0.85),
                             m.Circumscribe(pol[0], color=LOSS_COL, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the derivation

    def derivation(self):
        self.next_section("2 · Four lines of algebra", notes=(
            "Rafailov et al. 2023 (arXiv v3). Line 1, eq. 3: max_π E_{x, y~π}[r(x, y)] − β D_KL[π(y|x) ‖ π_ref(y|x)] (Episode "
            "20's leash). Line 2, eq. 4: π*(y|x) = π_ref(y|x) exp(r(x, y)/β) / Z(x), Z(x) = Σ_y π_ref(y|x) exp(r(x, y)/β) "
            "(proof App. A.1: the objective equals −β·KL(π ‖ π*) + β log Z, Gibbs' inequality). Toy (ILLUSTRATION, notes "
            f"§22 worked example 2): π_ref = 0.5/0.3/0.2, rewards 1/2/0, β = 1 → π* = "
            + "/".join(f"{p:.3f}" for p in TOY_OPT) + f", Z = {TOY_Z:.3f}; the optimum's objective value is β ln Z = "
            f"{_toy_obj:.3f}; β ln(π*/π_ref) = r − β ln Z (the reward back, up to a constant). For a language model Z sums "
            "over every possible text: intractable. Line 3, eq. 5: r(x, y) = β log π*(y|x)/π_ref(y|x) + β log Z(x) — the "
            "implicit reward plus a prompt-only constant ('Your Language Model is Secretly a Reward Model'). Line 4, eq. 6: "
            "Bradley–Terry (Episode 21) depends only on r(x, y_w) − r(x, y_l), so β log Z(x) cancels (x omitted on screen "
            "for space). The loss, eq. 7: replace π* with the trainable π_θ and maximise the likelihood of the observed "
            "preferences: L_DPO = −E[log σ(β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x))]; log π(y|x) is "
            "the SUM of the answer's token log-probs (App. B)."))
        big = title_text("Direct Preference Optimization", 50).move_to([0.45, 0.75, 0])
        sub = text("“Your Language Model is Secretly a Reward Model”", 26, MUTED).next_to(big, m.DOWN, buff=0.35)
        src = cite("Rafailov, Sharma, Mitchell, Ermon, Manning & Finn · NeurIPS 2023").next_to(sub, m.DOWN, buff=0.2)
        x0 = -5.55
        ys = [2.2, 1.12, -0.1, -1.33]
        badges = m.VGroup(*[badge(str(k + 1)).move_to([-6.15, yy, 0]) for k, yy in enumerate(ys)])

        row1 = formula([r"\max_{\pi}\;\mathbb{E}_{y \sim \pi}\big[", r"r(x, y)", r"\big]", "-", r"\beta",
                        r"\,\mathrm{KL}\big(", r"\pi", r"\,\|\,", r"\pi_{\mathrm{ref}}", r"\big)"],
                       [INK, REWARD, INK, INK, BETA_C, BETA_C, ACTION, BETA_C, REF_C, BETA_C])
        row2 = formula([r"\pi^{*}(y \mid x)", "=", r"\frac{1}{Z(x)}", r"\,\pi_{\mathrm{ref}}(y \mid x)", r"\,\exp\big(",
                        r"r(x, y)", r"/", r"\beta", r"\big)"],
                       [ACTION, INK, REF_C, REF_C, INK, REWARD, INK, BETA_C, INK])
        row3 = formula([r"r(x, y)", "=", r"\beta", r"\log \frac{\pi^{*}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}", "+",
                        r"\beta \log Z(x)"],
                       [REWARD, INK, BETA_C, REWARD, INK, REF_C])
        row4 = formula([r"P(y_w \succ y_l)", "=", r"\sigma\Big(", r"\beta \log \frac{\pi^{*}(y_w)}{\pi_{\mathrm{ref}}(y_w)}",
                        r"+\beta \log Z", "-", r"\beta \log \frac{\pi^{*}(y_l)}{\pi_{\mathrm{ref}}(y_l)}", r"-\beta \log Z",
                        r"\Big)"],
                       [GREEN, INK, INK, GREEN, REF_C, INK, RED, REF_C, INK])
        row4b = formula([r"P(y_w \succ y_l)", "=", r"\sigma\Big(", r"\beta \log \frac{\pi^{*}(y_w)}{\pi_{\mathrm{ref}}(y_w)}",
                         "-", r"\beta \log \frac{\pi^{*}(y_l)}{\pi_{\mathrm{ref}}(y_l)}", r"\Big)"],
                        [GREEN, INK, INK, GREEN, INK, RED, INK])
        loss = formula([r"\mathcal{L}_{\mathrm{DPO}}", "=", r"-\log \sigma\Big(",
                        r"\beta \log \frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}", "-",
                        r"\beta \log \frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}", r"\Big)"],
                       [LOSS_COL, INK, LOSS_COL, GREEN, INK, RED, LOSS_COL])
        for f, idx in ((row4, (3, 6)), (row4b, (3, 5)), (loss, (3, 5))):
            for i in idx:
                f[i][0].set_color(BETA_C)
        for f, yy in zip((row1, row2, row3, row4), ys):
            left_at(f, x0, yy)
        left_at(row4b, x0, ys[3])
        loss_box = card(loss.width + 0.6, loss.height + 0.36, LOSS_COL, CARD, radius=0.16, stroke=2.5)
        left_at(loss_box, x0 - 0.3, -2.72)
        loss.move_to(loss_box)
        loss_lab = right_at(text("a loss on log-prob ratios", 20, LOSS_COL, weight=m.BOLD), 6.4, -2.5)
        loss_src = right_at(cite("Rafailov et al., 2023, eq. 7"), 6.4, -2.95)
        assert loss_box.get_right()[0] < loss_lab.get_left()[0] - 0.2 and row4.get_right()[0] < 6.4
        cap1 = right_at(cite("episode 20's objective"), 6.4, ys[0])
        cap2 = right_at(cite("the best policy"), 6.4, ys[1])
        cap3 = right_at(colored([("implicit reward", REWARD), ("+ a constant", MUTED)], 20), 6.4, ys[2] + 0.2)
        cap3b = right_at(cite("(the constant depends only on the prompt)"), 6.4, ys[2] - 0.22)
        assert cap3b.get_left()[0] > row3.get_right()[0] + 0.3 and cap2.get_left()[0] > row2.get_right()[0] + 0.3

        # the toy: three answers, the reference re-weighted by e^(r/β)
        bars = Bars(list(TOY_REF), labels=[f"r = {r:g}" for r in TOY_R], top=0.6, bottom=0.0, width=3.1, height=1.5,
                    color=REF_C, center=(4.3, 0.25, 0), size=20, show_values=False)
        ghosts = m.VGroup(*[bars.marker(i, v, REF_C) for i, v in enumerate(TOY_REF)])
        nums_ref = m.VGroup(*[code(f"{v:.2f}", 20, REF_C).move_to([bars.xs[i], bars.level(v) + 0.24, 0])
                              for i, v in enumerate(TOY_REF)])
        nums_opt = m.VGroup(*[code(f"{p:.2f}", 20, ACTION, weight=m.BOLD).move_to([bars.xs[i], bars.level(max(v, p)) + 0.24, 0])
                              for i, (v, p) in enumerate(zip(TOY_REF, TOY_OPT))])
        for lab in bars.labels:
            lab.set_color(REWARD)
        bars.colors = [ACTION] * 3
        to_opt = bars.set(list(TOY_OPT), run_time=1.4)
        sw_ref = m.Square(0.22).set_stroke(REF_C, 2).set_fill(CARD, 1)
        sw_opt = m.Square(0.22).set_stroke(ACTION, 2.5).set_fill(CORAL_FILL, 1)
        leg_ref = m.VGroup(sw_ref, code("π_ref", 20, REF_C)).arrange(m.RIGHT, buff=0.12)
        leg_opt = m.VGroup(sw_opt, code("π*", 20, ACTION)).arrange(m.RIGHT, buff=0.12)
        legs = m.VGroup(leg_ref, leg_opt).arrange(m.RIGHT, buff=0.35)
        legs.move_to([4.3, 1.62, 0])
        toy_cite = cite("for example · three answers · β = 1").move_to([4.0, -1.22, 0])
        z_read = colored([("Z =", REF_C), (f"{TOY_Z:.2f}", INK), ("· makes them sum to 1", MUTED)], 20).move_to([4.0, -1.65, 0])
        z_hard = rich([("a language model:", MUTED), ("every possible text", INK)], 20).move_to([4.0, -2.05, 0])
        toy = m.VGroup(bars, ghosts, nums_ref, legs, toy_cite, z_read, z_hard)
        assert bars.get_left()[0] > row2.get_right()[0] + 0.5 and max(t.get_right()[0] for t in toy[4:]) < 6.45

        z4 = (row4[4], row4[7])
        cuts = m.VGroup(*[strike(z, RED, 0.04) for z in z4])

        self.say(
            "[They called it Direct Preference Optimization.] [Its whole idea fits in four lines of algebra.]",
            m.AnimationGroup(self.new_header("02 · the derivation", "Four lines of algebra"), m.FadeIn(big, shift=0.1 * m.UP),
                             m.FadeIn(sub), m.FadeIn(src)),
            m.LaggedStart(*[m.FadeIn(b, scale=0.6) for b in badges], lag_ratio=0.25),
        )
        self.say(
            "[Line one is the objective from episode twenty]: [maximise the reward], "
            "[minus beta times the KL divergence from the reference.]",
            m.AnimationGroup(m.FadeOut(big), m.FadeOut(sub), m.FadeOut(src), m.FadeIn(cap1)),
            m.FadeIn(row1[0:3], shift=0.1 * m.RIGHT),
            m.FadeIn(row1[3:], shift=0.1 * m.RIGHT),
        )
        self.say(
            "[Line two: this objective has an exact best policy.] [Start from the reference], "
            "[and re-weight each answer by e to the reward over beta.]",
            m.FadeIn(row2, shift=0.1 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(bars), m.FadeIn(nums_ref), m.FadeIn(leg_ref), m.FadeIn(toy_cite)),
            m.AnimationGroup(to_opt, *[m.Transform(a, b) for a, b in zip(nums_ref, nums_opt)], m.FadeIn(ghosts),
                             m.FadeIn(leg_opt, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The term Z sums over every possible answer]: [for a language model, far too many to compute.]",
            m.AnimationGroup(m.Circumscribe(row2[2], color=REF_C, buff=0.06), m.FadeIn(z_read, shift=0.1 * m.UP)),
            m.FadeIn(z_hard, shift=0.1 * m.UP),
        )
        self.say(
            "[Line three turns it around and solves for the reward]: [beta times the log of the ratio to the reference], "
            "[plus beta log Z.]",
            m.AnimationGroup(m.FadeOut(toy), m.FadeIn(row3[0:2], shift=0.1 * m.RIGHT), m.FadeIn(cap2)),
            m.FadeIn(row3[2:4], shift=0.1 * m.RIGHT),
            m.FadeIn(row3[4:], shift=0.1 * m.RIGHT),
        )
        self.say(
            "[So any policy secretly defines a reward]: [its implicit reward, plus a constant that only depends on the prompt.]",
            m.Circumscribe(row3[2:4], color=REWARD, buff=0.06),
            m.AnimationGroup(m.FadeIn(cap3, shift=0.1 * m.LEFT), m.FadeIn(cap3b)),
        )
        self.say(
            "[Line four puts two answers' rewards into Bradley-Terry, from episode twenty-one]: [the chosen one's], "
            "[minus the rejected one's.]",
            m.FadeIn(row4[0:3], shift=0.1 * m.RIGHT),
            m.FadeIn(row4[3:5], shift=0.1 * m.RIGHT),
            m.FadeIn(row4[5:], shift=0.1 * m.RIGHT),
        )
        keep4 = ((0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (8, 6))
        self.say(
            "[Both answers share the same prompt, so they share the constant], [and it cancels.]",
            m.AnimationGroup(*[m.Circumscribe(z, color=REF_C, buff=0.05) for z in z4]),
            m.Succession(m.AnimationGroup(*[m.Create(c_) for c_ in cuts], run_time=0.5),
                         m.AnimationGroup(*[m.FadeOut(z) for z in z4], m.FadeOut(cuts),
                                          *[m.Transform(row4[i], row4b[j]) for i, j in keep4], run_time=1.0)),
        )
        row4.remove(*z4)
        self.remove(*z4)
        self.say(
            "[Now put our own policy in place of the best one, and fit the pairs]: "
            "[minus the log sigmoid of that difference is the DPO loss.]",
            m.AnimationGroup(m.FadeIn(loss_box), m.FadeIn(loss[0:3], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(loss[3:], shift=0.1 * m.UP), m.FadeIn(loss_lab, shift=0.1 * m.LEFT), m.FadeIn(loss_src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · one real pair

    def real_pair(self):
        self.next_section("3 · One real pair", notes=(
            "From our run on this Mac (lm_rl.md, 'Episode 22 — DPO on one pair'; lm_rl.json dpo.showcase / pair_scores). "
            "Pairs: the starting model (Qwen2.5-0.5B-Instruct, the base = DPO's reference) sampled 4 answers (T = 1.0) to "
            "each of 400 training problems the GRPO run never used (train[2000:2400]); 239 problems had at least one right "
            f"and one wrong answer. The showcase problem: '{QUESTION}' (reference answer 100). Chosen ({N_W} tokens): "
            f"'x + 2x = 300 … 3x = 300 … x = 100 … \\boxed{{100}}'. Rejected ({N_L} tokens): '… we have $2x = 300$ … x = "
            f"\\boxed{{150}}'. Sequence log-probs under the base model (nats, summed over the answer's tokens): chosen "
            f"{REF_W:.3f}, rejected {REF_L:.3f}; per token {TOK_W:.3f} vs {TOK_L:.3f} (derived: sum ÷ tokens). At the start "
            "of DPO the policy IS the reference, so both log ratios are 0 and the loss is ln 2 for every β."))
        qbox = card(13.0, 1.4, EDGE, CARD, radius=0.12, stroke=1.5).move_to([0, 1.95, 0])
        stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=qbox.height - 0.16, stroke_width=0,
                                    fill_color=STATE, fill_opacity=1).move_to(qbox).align_to(qbox, m.LEFT).shift(0.1 * m.RIGHT)
        role = left_at(code("user", 22, STATE, weight=m.BOLD), -6.15, 2.38)
        where = right_at(cite("a training problem GRPO never used · our run on this Mac"), 6.3, 2.38)
        q_words = QUESTION.split()
        half = min(range(1, len(q_words)), key=lambda k: abs(len(" ".join(q_words[:k])) - len(QUESTION) / 2))
        q1 = left_at(text(" ".join(q_words[:half]), 20, INK), -6.15, 1.95)
        q2 = left_at(text(" ".join(q_words[half:]), 20, INK), -6.15, 1.58)
        assert q2.width < 12.3 and q1.width < 12.3

        def answer_card(cx, color, fill, title, tokens, lines):
            box = card(6.1, 2.25, color, CARD, radius=0.16, stroke=2.5).move_to([cx, -0.15, 0])
            head = left_at(tag(title, color, fill, 20), cx - 2.8, 0.6)
            toks = right_at(code(f"{tokens} tokens", 20, MUTED), cx + 2.8, 0.6)
            body = m.VGroup(*[code(s, 22, INK, t2c=t2c) for s, t2c in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
            body.next_to(head, m.DOWN, buff=0.26).align_to(head, m.LEFT)
            return box, head, toks, body

        ch = answer_card(-3.25, GREEN, GREEN_FILL, "chosen · right", N_W,
                         [("x + 2x = 300", None), ("3x = 300   x = 100", None), (r"… \boxed{100}", {r"\d+": GREEN})])
        rj = answer_card(3.25, RED, RED_FILL, "rejected · wrong", N_L,
                         [("we have 2x = 300", None), (r"x = \boxed{150}", {r"150": RED})])
        per_w = right_at(code(minus(f"{TOK_W:.2f} per token"), 20, GREEN), -3.25 + 2.8, 0.2)
        per_l = right_at(code(minus(f"{TOK_L:.2f} per token"), 20, RED), 3.25 + 2.8, 0.2)
        excerpt = right_at(cite("excerpts, verbatim"), 6.3, -1.55)

        ax = LogAxis(-50, 0, -5.4, 0.9, -2.3, [-50, -40, -30, -20, -10, 0])
        ax_lab = left_at(cite("log-probability of the whole answer, starting model (nats)"), -6.3, -3.25)
        likely = text("more likely", 20, MUTED).move_to([0.2, -1.85, 0])
        likely_arrow = m.Arrow(likely.get_right() + [0.08, 0, 0], likely.get_right() + [0.5, 0, 0], buff=0, color=MUTED,
                               stroke_width=3, max_tip_length_to_length_ratio=0.4)
        dw = solid(ax.at(REF_W, -2.3), GREEN, 0.14)
        dl = solid(ax.at(REF_L, -2.3), RED, 0.14)
        vw = code(minus(f"{REF_W:.1f}"), 24, GREEN, weight=m.BOLD).next_to(dw, m.UP, buff=0.16)
        vl = code(minus(f"{REF_L:.1f}"), 24, RED, weight=m.BOLD).next_to(dl, m.UP, buff=0.16)
        is_ref = left_at(text("the starting model = reference", 22, INK), 1.95, -2.0)
        zero = left_at(colored([("both log ratios:", MUTED), ("0", INK)], 22), 1.95, -2.5)
        start = left_at(colored([("loss = ln 2 =", MUTED), (f"{LN2:.3f}", LOSS_COL)], 22, weight=m.BOLD), 1.95, -3.0)
        assert likely_arrow.get_right()[0] < is_ref.get_left()[0] - 0.3 and is_ref.get_right()[0] < 6.45

        self.say(
            "[Now a real pair, from our run on this Mac]: [three hundred vehicles pass on a highway], "
            "[with twice as many cars as trucks. How many trucks?]",
            m.AnimationGroup(self.new_header("03 · one real pair", "Trucks on a highway"), m.FadeIn(qbox), m.FadeIn(stripe),
                             m.FadeIn(role), m.FadeIn(where)),
            m.FadeIn(q1, shift=0.1 * m.RIGHT),
            m.FadeIn(q2, shift=0.1 * m.RIGHT),
        )
        self.say(
            "[The starting model answered it four times.] [One answer wrote x plus two x equals three hundred], "
            "[and found one hundred: right.]",
            m.AnimationGroup(m.FadeIn(ch[0]), m.FadeIn(ch[2])),
            m.AnimationGroup(m.FadeIn(ch[3][:2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(ch[3][2], shift=0.1 * m.UP), m.FadeIn(ch[1], scale=0.85)),
        )
        self.say(
            "[Another wrote two x equals three hundred], [and boxed a hundred and fifty: wrong.]",
            m.AnimationGroup(m.FadeIn(rj[0]), m.FadeIn(rj[2]), m.FadeIn(rj[3][0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(rj[3][1], shift=0.1 * m.UP), m.FadeIn(rj[1], scale=0.85), m.FadeIn(excerpt)),
        )
        self.say(
            "[The catch: the starting model finds the wrong answer more likely], "
            "[minus twenty-eight point three against minus forty-one point five.]",
            m.AnimationGroup(m.FadeIn(ax.group), m.FadeIn(ax_lab), m.FadeIn(likely), m.GrowArrow(likely_arrow)),
            m.AnimationGroup(m.FadeIn(dl, scale=0.5), m.FadeIn(vl), m.FadeIn(dw, scale=0.5), m.FadeIn(vw)),
        )
        self.say(
            "[Partly because it is short]: [fifty-four tokens against two hundred forty. Per token, the right answer is the likelier one.]",
            m.AnimationGroup(m.Circumscribe(rj[2], color=RED, buff=0.06), m.Circumscribe(ch[2], color=GREEN, buff=0.06)),
            m.AnimationGroup(m.FadeIn(per_w, shift=0.1 * m.UP), m.FadeIn(per_l, shift=0.1 * m.UP)),
        )
        self.say(
            "[This starting model is also the reference.] [So both log ratios start at zero], "
            "[and the loss at log two: zero point six nine three.]",
            m.FadeIn(is_ref, shift=0.1 * m.UP),
            m.FadeIn(zero, shift=0.1 * m.UP),
            m.FadeIn(start, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · through the loss

    def through_the_loss(self):
        self.next_section("4 · Through the loss", notes=(
            "The GRPO-trained model (lm_rl's main run: 200 steps, reward = the checker's 1 or 0, never any pairs; Episodes "
            "23 and 26) scored on the same pair (lm_rl.json dpo.pair_scores[1]): log π(chosen) "
            f"{G_W:.3f}, log π(rejected) {G_L:.3f}; Δ = log π − log π_ref = {D_W:.3f} and {D_L:.3f} nats "
            f"(rl_reference.dpo_implicit_reward with β = 1); the wrong answer fell {D_L / D_W:.2f}× as far. Δ_w − Δ_l = "
            f"{GAP:.3f}. Implicit-reward margin β(Δ_w − Δ_l) and loss −log σ(margin) (rl_reference.dpo_loss), asserted "
            "against lm_rl.json: "
            + "; ".join(f"β = {b:g}: margin {MARGIN[b]:.4f}, loss {LOSS[b]:.4f}" for b in BETAS)
            + f". At the start (π = π_ref) every loss is ln 2 = {LN2:.4f}. The curve: −log σ(m) for m in [−2, 3] (201 "
            "points). Its slope is −σ(−m): the gradient's weight σ(r̂_l − r̂_w), 'higher weight when reward estimate is "
            f"wrong' (Rafailov et al. 2023 §4): 0.50 at the start, {WEIGHT[0.1]:.2f} at β = 0.1, {WEIGHT[0.5]:.3f} at "
            "β = 0.5. β is the KL strength (the inverse temperature of the optimal policy), not a learning rate (notes §22, "
            "misconceptions)."))
        f = formula([r"\mathcal{L}", "=", r"-\log\sigma\big(", r"\beta", r"\big[", r"\Delta_w", "-", r"\Delta_l", r"\big]",
                     r"\big)"], [LOSS_COL, INK, LOSS_COL, BETA_C, INK, GREEN, INK, RED, INK, LOSS_COL], 40)
        left_at(f, -6.3, 2.15)
        d_def = formula([r"\Delta", "=", r"\log \pi_\theta(y)", "-", r"\log \pi_{\mathrm{ref}}(y)"],
                        [INK, INK, ACTION, INK, REF_C], 34)
        right_at(d_def, 6.4, 2.15)
        note = left_at(cite("after GRPO: a checker's 1 or 0, never a pair"), -6.3, 1.55)

        ax = LogAxis(-50, -25, -5.0, -0.45, -0.6, [-50, -40, -30])
        yw, yl = 0.75, -0.05
        lab_w = left_at(code("chosen", 20, GREEN, weight=m.BOLD), -6.3, yw)
        lab_l = left_at(code("rejected", 20, RED, weight=m.BOLD), -6.3, yl)
        r_w, r_l = hollow(ax.at(REF_W, yw), GREEN), hollow(ax.at(REF_L, yl), RED)
        g_w, g_l = solid(ax.at(G_W, yw), GREEN), solid(ax.at(G_L, yl), RED)
        arr_w = m.Arrow(ax.at(REF_W, yw), ax.at(G_W, yw), buff=0.14, color=LOSS_COL, stroke_width=5,
                        max_tip_length_to_length_ratio=0.35)
        arr_l = m.Arrow(ax.at(REF_L, yl), ax.at(G_L, yl), buff=0.14, color=LOSS_COL, stroke_width=5,
                        max_tip_length_to_length_ratio=0.2)
        t_w = code(minus(f"Δ {D_W:.1f}"), 20, GREEN, weight=m.BOLD).next_to(arr_w, m.UP, buff=0.12)
        t_l = code(minus(f"Δ {D_L:.1f}"), 20, RED, weight=m.BOLD).next_to(arr_l, m.UP, buff=0.12)
        ax_lab = left_at(cite("log-probability (nats)"), -5.0, -1.3)
        leg = left_at(m.VGroup(hollow(m.ORIGIN, MUTED, 0.1), code("start = reference", 20, MUTED), solid(m.ORIGIN, INK, 0.1),
                               code("after GRPO", 20, INK)).arrange(m.RIGHT, buff=0.14), -6.3, -1.75)
        leg[2].shift(0.25 * m.RIGHT)
        leg[3].shift(0.25 * m.RIGHT)
        gap = left_at(colored([("Δw − Δl =", INK), (minus(f"{D_W:.1f}"), GREEN), ("+", INK), (f"{-D_L:.1f}", RED), ("=", INK),
                               (f"{GAP:.1f}", INK)], 24, weight=m.BOLD), -6.3, -2.4)
        harder = left_at(text(f"the wrong answer fell {D_L / D_W:.1f}× as far", 22, RED), -6.3, -2.95)

        plot = Plot((-2, 3), (0, 2.4), width=4.9, height=2.2, center=(3.6, 0.08, 0), x_ticks=[-2, -1, 0, 1, 2, 3],
                    y_ticks=[0, 2], x_label="implicit-reward margin")
        loss_y = text("loss", 22, LOSS_COL, weight=m.BOLD).next_to(plot.coords(-2, 2.4), m.LEFT, buff=0.22)
        ms = np.linspace(-2, 3, 201)
        curve = plot.line(ms, dpo_loss(ms, 0.0, 0.0, 0.0, beta=1.0), LOSS_COL, 5)
        p0 = plot.coords(0, LN2)
        start = m.Dot(p0, radius=0.09, color=MUTED).set_z_index(3)
        start_lab = code("ln 2", 20, MUTED).next_to(p0, m.UR, buff=0.16)

        def bdot(b):
            return m.Dot(plot.coords(MARGIN[b], LOSS[b]), radius=0.1, color=BETA_C).set_z_index(4)

        def slide(dot, m0, m1, seconds=1.2):
            def step(_, alpha):
                mm = m0 + (m1 - m0) * alpha
                dot.move_to(plot.coords(mm, float(dpo_loss(mm, 0.0, 0.0, 0.0, beta=1.0))))
            return m.UpdateFromAlphaFunc(dot, step, run_time=seconds, rate_func=m.smooth)

        d1 = m.Dot(p0, radius=0.1, color=BETA_C).set_z_index(4)
        d05 = m.Dot(plot.coords(MARGIN[0.1], LOSS[0.1]), radius=0.1, color=BETA_C).set_z_index(4)
        d5 = m.Dot(plot.coords(MARGIN[0.1], LOSS[0.1]), radius=0.1, color=BETA_C).set_z_index(4)
        lab5 = code("β 0.5", 20, BETA_C, weight=m.BOLD).next_to(bdot(0.5), m.UP, buff=0.16)

        cols = (1.95, 3.75, 5.55)
        t_head = m.VGroup(*[code(h, 20, c_).move_to([x, -1.95, 0]) for h, c_, x in
                            zip(("β", "margin", "loss"), (BETA_C, REWARD, LOSS_COL), cols)])

        def row(b, y):
            return m.VGroup(code(f"{b:g}", 22, BETA_C, weight=m.BOLD).move_to([cols[0], y, 0]),
                            code(f"{MARGIN[b]:.2f}", 22, REWARD).move_to([cols[1], y, 0]),
                            code(f"{LOSS[b]:.3f}", 22, LOSS_COL, weight=m.BOLD).move_to([cols[2], y, 0]))

        rows = {b: row(b, y) for b, y in zip((0.1, 0.05, 0.5), (-2.4, -2.85, -3.3))}
        n_head = {b: 2 for b in BETAS}
        tangent = m.Line(plot.coords(-0.75, LN2 + 0.375), plot.coords(0.75, LN2 - 0.375)).set_stroke(LOSS_COL, 3, opacity=0.9)
        wrong_lab = m.VGroup(text("ranked wrong", 20, RED), text("big push", 20, RED)).arrange(m.DOWN, buff=0.06)
        wrong_lab.move_to(plot.coords(-1.1, 0.38))
        right_lab = text("learned: little push", 20, GREEN).move_to(plot.coords(2.0, 1.25))
        assert note.get_right()[0] < plot.get_left()[0] - 0.3, (note.get_right()[0], plot.get_left()[0])
        assert t_head.get_top()[1] < plot.get_bottom()[1] - 0.15, (t_head.get_top()[1], plot.get_bottom()[1])
        assert d_def.get_bottom()[1] > plot.get_top()[1] + 0.1, (d_def.get_bottom()[1], plot.get_top()[1])

        self.say(
            "[Next, the same model after our GRPO run], [which learned from a checker's right or wrong, and never saw a pair.]",
            m.AnimationGroup(self.new_header("04 · through the loss", "What GRPO already learned"), m.FadeIn(f),
                             m.FadeIn(d_def), m.FadeIn(ax.group), m.FadeIn(ax_lab), m.FadeIn(lab_w), m.FadeIn(lab_l),
                             m.FadeIn(r_w, scale=0.6), m.FadeIn(r_l, scale=0.6), m.FadeIn(leg)),
            m.FadeIn(note, shift=0.1 * m.UP),
        )
        self.say(
            "[Both answers became less likely]: [the right one by three point seven], [and the wrong one by nine point three.]",
            m.AnimationGroup(m.FadeIn(g_w, target_position=r_w), m.FadeIn(g_l, target_position=r_l)),
            m.AnimationGroup(m.GrowArrow(arr_w), m.FadeIn(t_w, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arr_l), m.FadeIn(t_l, shift=0.1 * m.UP)),
        )
        self.say(
            "[So the wrong one fell two and a half times further]: [a gap of five point six for the right one.]",
            m.AnimationGroup(m.Circumscribe(t_l, color=RED, buff=0.06), m.FadeIn(harder, shift=0.1 * m.UP)),
            m.FadeIn(gap, shift=0.1 * m.UP),
        )
        self.say(
            "[With beta at zero point one, the implicit reward margin is zero point five six], "
            "[and the loss falls from zero point six nine to zero point four five.]",
            m.AnimationGroup(m.FadeIn(plot), m.FadeIn(loss_y), m.Create(curve), m.FadeIn(start, scale=0.5), m.FadeIn(start_lab),
                             m.FadeIn(t_head), m.FadeIn(rows[0.1][:n_head[0.1]], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Succession(m.FadeIn(d1, scale=0.5, run_time=0.3), slide(d1, 0.0, MARGIN[0.1], 1.4)),
                             m.FadeIn(rows[0.1][n_head[0.1]:])),
        )
        self.say(
            "[A smaller beta, zero point zero five, gives zero point five six], "
            "[and a larger one, zero point five, gives just zero point zero six.]",
            m.AnimationGroup(slide(d05, MARGIN[0.1], MARGIN[0.05], 1.0), m.FadeIn(rows[0.05], shift=0.1 * m.UP)),
            m.AnimationGroup(slide(d5, MARGIN[0.1], MARGIN[0.5], 1.4), m.FadeIn(rows[0.5], shift=0.1 * m.UP),
                             m.FadeIn(lab5)),
        )
        self.say(
            "[Beta is the exchange rate]: [a larger beta means the policy has to move less], "
            "[so it stays closer to the reference.]",
            m.Circumscribe(f[3], color=BETA_C, buff=0.08),
            m.Circumscribe(rows[0.5], color=BETA_C, buff=0.08),
            m.AnimationGroup(m.Circumscribe(r_w, color=REF_C, buff=0.08), m.Circumscribe(r_l, color=REF_C, buff=0.08)),
        )
        self.say(
            "[The slope of this curve is each pair's weight in the gradient]: [steep where a pair is still ranked wrong], "
            "[and nearly flat once it is learned.]",
            m.Create(tangent),
            m.FadeIn(wrong_lab, shift=0.1 * m.UP),
            m.FadeIn(right_lab, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · a short DPO run

    def short_run(self):
        self.next_section("5 · A short DPO run", notes=(
            f"lm_rl.md 'Short DPO run' (lm/dpo.py, logs/lm_dpo.json): β = {DPO['beta_train']}, lr {DPO['lr']:g}, LoRA as "
            f"in the GRPO run, {DPO['pairs_per_step']} pairs per step, {DPO['epochs']} epochs = {STEPS} steps "
            f"({DPO['n_pairs']} // {DPO['pairs_per_step']} = {DPO['n_pairs'] // DPO['pairs_per_step']} per epoch), "
            f"{MINUTES:.1f} min on this Mac. Left: the showcase pair's log-probs during training (dpo.curve "
            "show_logp_chosen / show_logp_rejected; step 0 was scored inside a training batch: −41.86 / −28.25 vs the "
            "standalone −41.54 / −28.27, bf16 batched numerics). They cross at step "
            f"{CROSS}; end of pass 1 (step {EP1}): {C_W[EP1]:.1f} / {C_L[EP1]:.1f}; end: {C_W[-1]:.1f} / {C_L[-1]:.1f}. "
            f"Right: the batch loss (8 pairs per batch, noisy): {LOSS_C[0]:.3f} → {LOSS_C[EP1]:.3f} at the end of pass 1 "
            f"(75% of the batch ranked right) → {LOSS_C[-1]:.3f} at the last step (100%). Held-out greedy accuracy on the "
            f"500 test problems: {100 * ACC_DPO:.1f}% after DPO vs {100 * ACC_BASE:.1f}% for the base at the same decode "
            f"batch size (64; the GRPO model: {100 * ACC_GRPO:.1f}%); greedy decoding is not batch-invariant (lm_rl.md "
            "caveat 1), one run. lm_rl.md: 'The tiny offline run overfits its 239 pairs and lowers the likelihood of good "
            "answers too. That is a known DPO failure mode, not a fair DPO-vs-GRPO comparison.' Rafailov, Hejna, Park & "
            "Finn 2024 §5.4: 'the likelihood of the chosen responses actually decrease over time (Pal et al., 2024)'."))
        plot = Plot((0, STEPS - 1), (-75, -25), width=5.5, height=3.1, center=(-3.1, -0.05, 0), x_ticks=[0, 20, 40],
                    y_ticks=[-70, -50, -30], x_label="training step", y_label="log-probability of each answer (nats)")
        xs = list(range(STEPS))
        p1w = plot.line(xs[:EP1 + 1], C_W[:EP1 + 1], GREEN, 5)
        p1l = plot.line(xs[:EP1 + 1], C_L[:EP1 + 1], RED, 5)
        p2w = plot.line(xs[EP1:], C_W[EP1:], GREEN, 5)
        p2l = plot.line(xs[EP1:], C_L[EP1:], RED, 5)
        s_w, s_l = plot.dot(0, C_W[0], GREEN, 0.09), plot.dot(0, C_L[0], RED, 0.09)
        s_lab_w = left_at(code("chosen · right", 20, GREEN, weight=m.BOLD), plot.coords(2, 0)[0], plot.coords(0, -61)[1])
        s_lab_l = left_at(code("rejected · wrong", 20, RED, weight=m.BOLD), plot.coords(2, 0)[0], plot.coords(0, -68)[1])
        divider = plot.vline(EP1 + 0.5, BETA_C, 2)
        pass2 = code("pass 2", 20, BETA_C).next_to(plot.coords(EP1 + 0.5, -30), m.RIGHT, buff=0.12)
        _k = CROSS - 1
        _t = (C_L[_k] - C_W[_k]) / ((C_W[CROSS] - C_W[_k]) - (C_L[CROSS] - C_L[_k]))
        assert 0 <= _t <= 1
        ring = m.Circle(radius=0.22).set_stroke(INK, 2.5).move_to(plot.coords(_k + _t, C_W[_k] + _t * (C_W[CROSS] - C_W[_k])))
        cross_lab = code("they cross", 20, INK).move_to(plot.coords(_k + _t, -26.6))
        e_w = code(minus(f"{C_W[-1]:.1f}"), 22, GREEN, weight=m.BOLD).next_to(plot.coords(STEPS - 1, C_W[-1]), m.RIGHT, buff=0.12)
        e_l = code(minus(f"{C_L[-1]:.1f}"), 22, RED, weight=m.BOLD).next_to(plot.coords(STEPS - 1, C_L[-1]), m.RIGHT, buff=0.12)
        e_l.shift(0.28 * m.UP)

        cx = 1.75
        set1a = left_at(code(f"{DPO['n_pairs']} pairs", 22, INK, weight=m.BOLD), cx, 2.25)
        set1b = left_at(cite(f"its own answers · {DPO['samples_per_question']} per problem"), cx, 1.85)
        set2 = left_at(rich([(f"β = {DPO['beta_train']:g}", BETA_C), (f"· LoRA · {DPO['pairs_per_step']} pairs per step", MUTED)],
                            20), cx, 1.4)
        set3 = left_at(rich([(f"{DPO['epochs']} passes = {STEPS} steps", INK), (f"· {MINUTES:.0f} min · this Mac", MUTED)], 20),
                       cx, 1.0)
        assert max(t.get_right()[0] for t in (set1b, set2, set3)) < 6.45, [t.get_right()[0] for t in (set1b, set2, set3)]
        lp = Plot((0, STEPS - 1), (0, 0.85), width=3.1, height=0.8, center=(4.3, -0.12, 0), x_ticks=[], y_ticks=[0],
                  y_label="loss, per batch of 8 pairs", y_fmt="{:g}")
        lcurve = lp.line(xs, LOSS_C, LOSS_COL, 3)
        l0 = code(f"{LOSS_C[0]:.3f}", 20, LOSS_COL).next_to(lp.coords(0, LOSS_C[0]), m.UR, buff=0.15)
        l1 = code(f"{LOSS_C[-1]:.3f}", 20, LOSS_COL, weight=m.BOLD).next_to(lp.coords(STEPS - 1, LOSS_C[-1]), m.UP, buff=0.3)
        ranked = right_at(code("last batch: 100% ranked right", 20, GREEN), 6.4, -1.2)
        assert l1.get_right()[0] < 6.6 and lp.get_top()[1] < set3.get_bottom()[1] - 0.1, (l1.get_right()[0], lp.get_top()[1])

        acc_cap = left_at(cite("held-out greedy accuracy · 500 problems"), cx, -1.68)
        x_bar, scale = 3.75, 3.9

        def acc_row(label, acc, color, fill, y):
            lab = left_at(text(label, 20, color, weight=m.BOLD), cx, y)
            bar = m.Rectangle(width=acc * scale, height=0.36).set_stroke(color, 2).set_fill(fill, 1)
            left_at(bar, x_bar, y)
            val = code(f"{100 * acc:.1f}%", 22, color, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.12)
            return m.VGroup(lab, bar, val)

        acc_b = acc_row("starting model", ACC_BASE, MUTED, mix(CARD, MUTED, 0.3), -2.12)
        acc_d = acc_row("after DPO", ACC_DPO, RED, RED_FILL, -2.62)
        assert acc_b[2].get_right()[0] < 6.6 and acc_b[0].get_right()[0] < x_bar - 0.1, (acc_b[2].get_right()[0], acc_b[0].get_right()[0])
        known = left_at(cite("known: the chosen answer's likelihood can fall too · Rafailov et al., 2024"), -6.3, -3.0)
        fair = left_at(cite("one tiny run on 239 pairs, one seed: not a verdict on DPO versus GRPO"), -6.3, -3.36)
        assert known.get_right()[0] < cx - 0.2 or known.get_top()[1] < acc_d.get_bottom()[1] - 0.03, (known.get_right()[0], known.get_top()[1], acc_d.get_bottom()[1])

        self.say(
            "[Then we ran DPO itself on this Mac]: [two hundred thirty-nine pairs from the starting model's own answers], "
            "[fifty-eight steps in six minutes.]",
            m.AnimationGroup(self.new_header("05 · a short DPO run", "Pulled apart, then too far"), m.FadeIn(plot)),
            m.AnimationGroup(m.FadeIn(set1a, shift=0.1 * m.UP), m.FadeIn(set1b, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(set2, shift=0.1 * m.UP), m.FadeIn(set3, shift=0.1 * m.UP)),
        )
        self.say(
            "[Watch our trucks pair.] [In the first pass, the right answer rose and the wrong one fell], [until they crossed.]",
            m.AnimationGroup(m.FadeIn(s_w, scale=0.5), m.FadeIn(s_l, scale=0.5), m.FadeIn(s_lab_w), m.FadeIn(s_lab_l)),
            m.AnimationGroup(m.Create(p1w, run_time=2.6), m.Create(p1l, run_time=2.6)),
            m.AnimationGroup(m.Create(ring), m.FadeIn(cross_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[In the second pass, both fell], [the wrong one much faster: minus seventy-two against minus forty-nine.]",
            m.AnimationGroup(m.Create(divider), m.FadeIn(pass2), m.Create(p2w, run_time=2.4), m.Create(p2l, run_time=2.4)),
            m.AnimationGroup(m.FadeIn(e_l, shift=0.1 * m.LEFT), m.FadeIn(e_w, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The loss fell from zero point six nine to zero point zero zero seven]: [every pair in the last batch ranked right.]",
            m.AnimationGroup(m.FadeIn(lp), m.Create(lcurve, run_time=2.0), m.FadeIn(l0), m.FadeIn(l1)),
            m.FadeIn(ranked, shift=0.1 * m.UP),
        )
        self.say(
            "[But on five hundred problems it had never seen], "
            "[its accuracy fell from forty-seven point four percent to forty-three point four.]",
            m.AnimationGroup(m.FadeIn(acc_cap), m.FadeIn(acc_b[0]), m.GrowFromEdge(acc_b[1], m.LEFT), m.FadeIn(acc_b[2])),
            m.AnimationGroup(m.FadeIn(acc_d[0]), m.GrowFromEdge(acc_d[1], m.LEFT), m.FadeIn(acc_d[2])),
        )
        self.say(
            "[The tiny offline run overfit its pairs, and pushed the right answers down too]: [a known DPO failure], "
            "[and not a fair contest with GRPO.]",
            m.Circumscribe(e_w, color=GREEN, buff=0.08),
            m.FadeIn(known, shift=0.1 * m.UP),
            m.FadeIn(fair, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the trade

    def the_trade(self):
        self.next_section("6 · What DPO gives up", notes=(
            "Notes §22 'What disappears': no reward-model training run, no sampling during training, no value network, no "
            "PPO; two models (the trained policy and a frozen reference; its log-probs can be precomputed). What it gives "
            "up: DPO is OFFLINE — it learns from a fixed dataset of pairs, usually sampled from another model, and never "
            "gets feedback on its own new answers (no on-policy exploration). Results, Rafailov et al. 2023: TL;DR "
            f"summarisation (GPT-J, GPT-4 as judge vs reference summaries) DPO ≈ {TLDR_DPO}% win rate at temperature 0 vs "
            f"PPO {TLDR_PPO}% at its best temperature (§6.2); humans preferred DPO samples {HUMAN_DPO}% of the time over PPO "
            "(§6.4); IMDb: 'DPO produces by far the most efficient frontier' (§6.1); caveat: GPT-4-judged, β barely tuned. "
            "Later evidence disagrees (Appendix A row 22, present as open): Llama 3 chose DPO (Llama Team 2024 §4.1.4: "
            "less compute, better on IFEval); Xu et al. 2024: 'PPO is able to surpass other alignment methods in all "
            f"cases'; Ivison et al. 2024 (NeurIPS): PPO ahead of DPO 'by up to {IVISON_MATH}% in math and 1.2% in general "
            "domains'."))
        xl, xr = -6.3, 0.55
        t_rm = left_at(text("what it removes", 26, GREEN, weight=m.BOLD), xl, 2.2)
        items = [(RM_ICON, "a reward model"), (DICE, "sampling while training"), (VALUE_ICON, "a value network")]
        rows = []
        for k, (ic, s) in enumerate(items):
            yy = 1.5 - 0.62 * k
            ico = icon(ic, 0.46).move_to([xl + 0.25, yy, 0])
            words = left_at(text(s, 24, INK), xl + 0.7, yy)
            c0 = ico.get_center()
            x_mark = m.VGroup(m.Line(c0 + [-0.26, -0.26, 0], c0 + [0.26, 0.26, 0]), m.Line(c0 + [-0.26, 0.26, 0], c0 + [0.26, -0.26, 0]))
            rows.append(m.VGroup(ico, words, x_mark.set_stroke(RED, 4).set_z_index(2)))
        keeps = left_at(rich([("keeps:", MUTED), ("the policy", ACTION), ("+", MUTED), ("a frozen reference", REF_C)], 22,
                             weight=m.BOLD), xl, -0.4)
        pre = left_at(cite("the reference's log-probs can be computed once"), xl, -0.82)
        t_gu = left_at(text("what it gives up", 26, RED, weight=m.BOLD), xr, 2.2)
        g1 = left_at(text("exploring on its own", 24, INK, weight=m.BOLD), xr, 1.5)
        g2 = left_at(text("it learns from a fixed set of pairs", 22, MUTED), xr, 0.95)
        g3 = left_at(text("its own new answers are never scored", 22, MUTED), xr, 0.5)
        toks = token_row(["its", "own", "new", "answer"], ACTION, size=22)
        left_at(toks, xr, -0.25)
        q_mark = pill("no score", RED, RED_FILL, 22).next_to(toks, m.RIGHT, buff=0.5)
        dash = m.DashedLine(toks.get_right() + [0.08, 0, 0], q_mark.get_left() + [-0.08, 0, 0], dash_length=0.08)
        dash.set_stroke(MUTED, 2)
        assert q_mark.get_right()[0] < 6.45

        def tile(x, title, src, lines):
            box = card(6.15, 1.85, EDGE, CARD, radius=0.16, stroke=2).move_to([x, -2.3, 0])
            head = left_at(text(title, 22, INK, weight=m.BOLD), x - 2.8, -1.67)
            s_ = right_at(cite(src), x + 2.8, -1.67)
            body = m.VGroup(*[rich(l_, 20) for l_ in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
            body.next_to(head, m.DOWN, buff=0.22).align_to(head, m.LEFT)
            return box, head, s_, body

        tl = tile(-3.2, "the paper", "Rafailov et al., 2023", [
            [("summaries:", MUTED), ("DPO", LOSS_COL), (f"≈ {TLDR_DPO}% wins", INK), ("vs PPO", ACTION), (f"{TLDR_PPO}%", INK)],
            [("people preferred DPO", MUTED), (f"{HUMAN_DPO}%", INK), ("of the time", MUTED)],
            [("judge: GPT-4 · β = 0.1, barely tuned", MUTED)]])
        tr = tile(3.2, "later studies: split", "2024", [
            [("Llama 3 chose DPO:", INK), ("less compute", MUTED)],
            [("tuned PPO ahead:", ACTION), (f"up to {IVISON_MATH:g} points in maths", MUTED)],
            [("Xu et al. · Ivison et al. · an open question", MUTED)]])
        assert all(t[3].get_right()[0] < t[0].get_right()[0] - 0.1 for t in (tl, tr))

        self.say(
            "[What DPO removes]: [the reward model], [the sampling, and the value network.] "
            "[Two models remain: the policy and a frozen reference.]",
            m.AnimationGroup(self.new_header("06 · the trade", "No reward model, no exploring"), m.FadeIn(t_rm)),
            m.AnimationGroup(m.FadeIn(rows[0][:2], shift=0.1 * m.RIGHT), m.Create(rows[0][2])),
            m.AnimationGroup(*[m.AnimationGroup(m.FadeIn(r_[:2], shift=0.1 * m.RIGHT), m.Create(r_[2])) for r_ in rows[1:]]),
            m.AnimationGroup(m.FadeIn(keeps, shift=0.1 * m.UP), m.FadeIn(pre)),
        )
        self.say(
            "[What it gives up is exploration]: [it learns only from a fixed set of pairs], "
            "[and never gets feedback on its own new answers.]",
            m.AnimationGroup(m.FadeIn(t_gu), m.FadeIn(g1, shift=0.1 * m.UP)),
            m.FadeIn(g2, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(g3, shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(t_, shift=0.1 * m.RIGHT) for t_ in toks],
                                                                           lag_ratio=0.2),
                             m.Create(dash), m.FadeIn(q_mark)),
        )
        self.say(
            "[In the original paper it matched or beat PPO]: [on a summarization task, about sixty-one percent wins against fifty-seven.]",
            m.AnimationGroup(m.FadeIn(tl[0]), m.FadeIn(tl[1]), m.FadeIn(tl[2])),
            m.LaggedStart(*[m.FadeIn(l_, shift=0.1 * m.UP) for l_ in tl[3]], lag_ratio=0.3),
        )
        self.say(
            "[Later studies are split]: [Llama three chose DPO], [while carefully tuned PPO did better in others: an open question.]",
            m.AnimationGroup(m.FadeIn(tr[0]), m.FadeIn(tr[1]), m.FadeIn(tr[2])),
            m.FadeIn(tr[3][0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(tr[3][1], shift=0.1 * m.UP), m.FadeIn(tr[3][2], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · variants

    def variants(self):
        self.next_section("7 · Variants", notes=(
            "Notes §22 variants table. IPO: Azar et al., 'A General Theoretical Paradigm to Understand Learning from Human "
            "Preferences', arXiv Oct 2023, AISTATS 2024 — a squared loss regressing the log-ratio gap h_π to the target "
            "1/(2τ) instead of pushing it to ∞ (DPO overfits when preferences are near-deterministic). KTO: Ethayarajh et "
            "al., ICML 2024 — a thumbs-up/down per answer, no pairs (prospect theory). ORPO: Hong, Lee & Thorne, EMNLP 2024 "
            "— no reference model and no separate SFT stage: SFT loss + λ·odds-ratio term. SimPO: Meng, Xia & Chen, NeurIPS "
            "2024 — no reference model: the reward is the LENGTH-AVERAGED log-likelihood (β/|y|) log π plus a margin γ. "
            f"On our trucks pair the base model's per-token log-probs are {TOK_W:.3f} (right) vs {TOK_L:.3f} (wrong): a "
            "length-averaged score already ranks the right answer first, while the sum ranked it second (derived here)."))
        rows = [("IPO", "Azar et al., 2023", "pulls the gap to a fixed target, instead of pushing it forever"),
                ("KTO", "Ethayarajh et al., 2024", "no pairs: a thumbs up or down on single answers"),
                ("ORPO", "Hong, Lee & Thorne, 2024", "no reference model: a preference term added to fine-tuning"),
                ("SimPO", "Meng, Xia & Chen, 2024", "no reference model: log-probability averaged over the length")]
        groups = []
        for k, (name, src, desc) in enumerate(rows):
            yy = 1.75 - 1.05 * k
            nm = tag(name, LOSS_COL, PINK_FILL, 24)
            left_at(nm, -6.3, yy)
            d_ = left_at(text(desc, 26, INK), -4.05, yy + 0.15)
            s_ = left_at(cite(src), -4.05, yy - 0.3)
            groups.append(m.VGroup(nm, d_, s_))
            assert d_.get_right()[0] < 6.45
        call = left_at(colored([("our trucks pair, per token:", MUTED), (minus(f"{TOK_W:.2f}"), GREEN), ("right", GREEN),
                                ("vs", MUTED), (minus(f"{TOK_L:.2f}"), RED), ("wrong", RED)], 22), -4.05, -2.75)

        self.say(
            "[Variants came quickly.] [IPO, from twenty twenty-three, pulls the gap toward a fixed target "
            "instead of pushing it forever.]",
            self.new_header("07 · variants", "Four relatives"),
            m.FadeIn(groups[0], shift=0.1 * m.RIGHT),
        )
        self.say(
            "[The next, KTO, needs no pairs at all]: [just a thumbs up or down on single answers.]",
            m.AnimationGroup(m.FadeIn(groups[1][0], shift=0.1 * m.RIGHT), m.FadeIn(groups[1][2])),
            m.FadeIn(groups[1][1], shift=0.1 * m.RIGHT),
        )
        self.say(
            "[And two from twenty twenty-four drop the reference]: [ORPO folds the preference into fine-tuning], "
            "[and SimPO averages over length, which already ranks our trucks pair right.]",
            m.AnimationGroup(m.FadeIn(groups[2][0], shift=0.1 * m.RIGHT), m.FadeIn(groups[2][2]),
                             m.FadeIn(groups[3][0], shift=0.1 * m.RIGHT), m.FadeIn(groups[3][2])),
            m.FadeIn(groups[2][1], shift=0.1 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(groups[3][1], shift=0.1 * m.RIGHT), m.FadeIn(call, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            "Solve (the KL-leashed objective has the closed-form optimum π* ∝ π_ref·e^{r/β}), invert (r = β log π/π_ref + "
            "β log Z: the implicit reward), compare (Bradley–Terry uses only the difference, so the constant cancels and "
            "the policy is fitted to the pairs directly). Our pair: the GRPO-trained model already had a DPO loss of "
            f"{LOSS[0.1]:.3f} < ln 2 at β = 0.1; the short DPO run fitted its {DPO['n_pairs']} pairs (batch loss "
            f"{LOSS_C[-1]:.3f}) and overfit (held-out greedy {100 * ACC_DPO:.1f}% vs {100 * ACC_BASE:.1f}%). Episode 23: "
            "GRPO."))
        halo = glow(LOSS_COL, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Solve", 66, ACTION), title_text("Invert", 66, REWARD), title_text("Compare", 66, GREEN))
        words.arrange(m.RIGHT, buff=1.3).move_to([0, 1.1, 0])
        on_baseline(words)
        subs = m.VGroup(tex(r"\pi^{*} \propto \pi_{\mathrm{ref}}\, e^{r/\beta}", 34, MUTED),
                        tex(r"r = \beta \log (\pi / \pi_{\mathrm{ref}}) + c", 32, MUTED),
                        text("the constant cancels", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), words[0].get_bottom()[1] - 0.6, 0])
        assert all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        proof = colored([("our pair:", MUTED), ("after GRPO, loss", INK), (f"{LOSS[0.1]:.3f} < ln 2", LOSS_COL), ("·", MUTED),
                         ("a short DPO run overfit", RED)], 22).move_to([0, -1.2, 0])
        model = icon(POLICY_ICON, 0.62).move_to([-3.0, -2.45, 0])
        nxt = pill("episode 23 · a group instead of a critic", LOSS_COL, PINK_FILL, 22).next_to(model, m.RIGHT, buff=0.4)

        self.say(
            "[To sum up]: [solve the leashed objective], [invert it for the implicit reward], "
            "[and compare two answers, so the constant cancels.]",
            m.AnimationGroup(self.new_header("08 · recap", "Solve, invert, compare"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[On our trucks pair, GRPO had already pushed the loss below log two], "
            "[and a short DPO run fit its pairs, then overfit.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(model, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.1 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()
