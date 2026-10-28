"""Trial and Reward · Episode 21 · Learning from Human Preferences (Module 6 · Teaching Language Models with Reward)

People compare answers; a reward model learns their taste. Episode 20 made the language model a policy and graded
its maths answers with a checker (our GRPO run on this Mac: the boxed number equals the reference, or it doesn't).
This episode asks what to do when no checker exists. The running example is InstructGPT's own (Ouyang et al. 2022,
Fig. 2): the prompt "Explain the moon landing to a 6 year old" and its sampled answers "People went to the moon…"
(chosen) and "Explain gravity…" (rejected), then the step-3 prompt "Write a story about frogs" → "Once upon a time…".
One choice is one preference pair. Bradley–Terry turns two hidden scores into a win probability, σ(r_A − r_B)
(rl_reference.bradley_terry_prob, drawn as a curve; notes §21 worked example r = 1.2 vs 0.4 → 0.690, loss 0.371,
wrong way round 1.171). A reward model is trained on such pairs (a few illustrative gradient steps on the two scores);
labelers rank 4–9 answers, every two make a pair (C(4,2) = 6 … C(9,2) = 36). Then PPO (Episode 18) optimises the policy
against the reward model with a KL penalty to the frozen SFT model (Episode 20's leash; InstructGPT β = 0.02). Four
models in memory: policy and value network (trained), reward model and reference (frozen); a 7B policy's model states
take ≈ 252 GB (derived, notes §21). History: Christiano et al. 2017 → Stiennon et al. 2020 → InstructGPT 2022 (12,725 /
33,207 / 31,144 prompts, about 40 labelers, a 6B reward model, β = 0.02, 60 vs 3,640 petaflop/s-days = 1.65%, and the
headline: the 1.3B model preferred over the 175B GPT-3). Costs in one line each: labels are expensive (labelers agree
72.6 ± 1.5%), the reward model can be fooled (Episode 25); our own run used a checker instead (Episode 24). Recap:
compare, score, optimise. 'AI, Explained' Episode 30 is the short version.

Every number on screen is computed below from rl_reference.py and asserted, loaded with measured(...), or cited.

Sources (research/notes.md §21, Appendix A rows 21, Appendix C row 21; read October 2026):
- Ouyang et al. 2022, "Training language models to follow instructions with human feedback" (arXiv 2203.02155,
  NeurIPS 2022), checked against the PDF: abstract ("outputs from the 1.3B parameter InstructGPT model are preferred to
  outputs from the 175B GPT-3, despite having 100x fewer parameters."); §1 ("These models have the same architecture,
  and differ only by the fact that InstructGPT is fine-tuned on our human data."); Fig. 2 (the moon-landing prompt,
  answers A "Explain gravity…", B "Explain war…", C "Moon is natural satellite of…", D "People went to the moon…",
  ranked D > C > A = B; step 3 "Write a story about frogs" → "Once upon a time…"); §3.4 ("a team of about 40
  contractors"; labelers agree 72.6 ± 1.5%); §3.5 (K = 4 to 9 responses → C(K,2) comparisons; a 6B reward model);
  Table 6 (SFT 11,295 + 1,430 = 12,725; RM 6,623 + 26,584 = 33,207; PPO 31,144 prompts); App. C.4 (β = 0.02; value
  function initialised from the RM); §5.1 (SFT 4.9, PPO-ptx 60, GPT-3 3,640 petaflop/s-days, 175B models).
- Bradley & Terry 1952 (as cited by Christiano et al. 2017 and DPO); Christiano et al. 2017 §2.2.3 (the Elo
  analogy); §3.2 (the Hopper backflip "trained using 900 queries in less than an hour"); OpenAI + DeepMind.
- Stiennon et al. 2020 §4.1 ("our 1.3B human feedback model significantly outperforming a supervised model 10× its
  size (61% versus 43% raw preference score against reference summaries)"), §4.3 ("eventually the reward model
  becomes anti-correlated with human preferences").
- Four models: Hu et al. 2024 (OpenRLHF) §2.1 "actor, critic, reward, reference". Memory [DERIVED, notes §21 worked
  example 3]: mixed-precision Adam 16 bytes per trained parameter (Rajbhandari et al. 2020, ZeRO §3.1), frozen BF16 2
  bytes; 7B: 112 + 112 + 14 + 14 = 252 GB of model states (no activations, no KV cache).
- research/experiments/lm_rl.md + lm_rl.json (measured("lm_rl", ...)): the model, the reward rule (the checker), the
  machine (64 GB), example test-1028 (reference 200; two answers boxed 300 and 200, scored 0 and 1).

Illustrative (said "for example" or labelled): the hidden scores 1.2 and 0.4 (notes §21 worked example); the gradient
steps on the two scores (5 steps of plain gradient descent at learning rate 1 on −log σ(r_A − r_B), applied to the
scores themselves — a real reward model moves its weights); the moving dot on the Bradley–Terry curve.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep21"
LM = "lm_rl"

# ---------------------------------------------------------------------------- Bradley–Terry (computed)

R_A, R_B = 1.2, 0.4                                                 # for example (notes §21 worked example)
P_AB = float(bradley_terry_prob(R_A, R_B))
LOSS = -math.log(P_AB)
P_BA = float(bradley_terry_prob(R_B, R_A))
LOSS_WRONG = -math.log(P_BA)
assert round(P_AB, 3) == 0.690 and round(100 * P_AB) == 69                 # said: "sixty-nine percent"
assert round(LOSS, 3) == 0.371 and round(LOSS, 2) == 0.37                   # said: "about zero point three seven"
assert round(P_BA, 3) == 0.310 and round(LOSS_WRONG, 3) == 1.171            # said: "one point one seven"
assert float(bradley_terry_prob(0.7, 0.7)) == 0.5                           # said: "a coin flip"
P_GAP1, P_GAP4 = float(bradley_terry_prob(1, 0)), float(bradley_terry_prob(4, 0))
assert round(100 * P_GAP1) == 73 and round(100 * P_GAP4) == 98             # said: "seventy-three", "ninety-eight"
SHIFT = 5
assert abs(float(bradley_terry_prob(R_A + SHIFT, R_B + SHIFT)) - P_AB) < 1e-12   # said: "add five … nothing changes"
assert abs(math.log(P_GAP1 / (1 - P_GAP1)) - 1) < 1e-12                     # the gap is the log-odds

# a few gradient steps on −log σ(r_A − r_B), applied to the two scores (illustration)
STEPS, LR = 5, 1.0
TRAJ = [(R_A, R_B)]
for _ in range(STEPS):
    _a, _b = TRAJ[-1]
    _push = 1.0 - float(bradley_terry_prob(_a, _b))                 # −∂loss/∂r_A = ∂loss/∂r_B
    TRAJ.append((_a + LR * _push, _b - LR * _push))
LOSSES = [-math.log(float(bradley_terry_prob(a, b))) for a, b in TRAJ]
assert all(x > y for x, y in zip(LOSSES, LOSSES[1:]))                      # said: "the loss falls"
assert all(a1 > a0 and b1 < b0 for (a0, b0), (a1, b1) in zip(TRAJ, TRAJ[1:]))   # chosen up, rejected down
assert round(LOSSES[-1], 3) == 0.080 and round(TRAJ[-1][0] - TRAJ[-1][1], 2) == 2.49

PAIRS4, PAIRS9 = math.comb(4, 2), math.comb(9, 2)
assert (PAIRS4, PAIRS9) == (6, 36)                                         # said: "nine answers give thirty-six"

# ---------------------------------------------------------------------------- InstructGPT (cited, notes §21)

SFT_PROMPTS = 11_295 + 1_430
RM_PROMPTS = 6_623 + 26_584
PPO_PROMPTS = 31_144
assert (SFT_PROMPTS, RM_PROMPTS) == (12_725, 33_207)                       # Table 6 sums
assert round(SFT_PROMPTS, -3) == 13_000 and RM_PROMPTS // 1000 == 33 and PPO_PROMPTS // 1000 == 31
BETA = 0.02
RM_B = 6
LABELERS = 40
AGREE, AGREE_SE = 72.6, 1.5
assert round(AGREE) == 73                                                  # said: "about seventy-three percent"
PF_GPT3, PF_SFT, PF_PPO = 3_640, 4.9, 60                                   # petaflop/s-days, 175B models (§5.1)
PF_PCT = 100 * PF_PPO / PF_GPT3
assert round(PF_PCT, 2) == 1.65                                            # said: "about one point six five percent"
SIZE_RATIO = 175 / 1.3
assert SIZE_RATIO > 100 and round(SIZE_RATIO) == 135                       # said: "over a hundred times smaller"
QUOTE = ("outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, "
         "despite having 100x fewer parameters.")
BACKFLIP_QUERIES = 900                                                     # Christiano et al. 2017 §3.2
assert BACKFLIP_QUERIES == 900
SUMMARY_WIN, SUMMARY_SUP = 61, 43                                          # Stiennon et al. 2020 §4.1

# ---------------------------------------------------------------------------- four models in memory (derived)

POLICY_PARAMS = 7e9
TRAINED_BYTES, FROZEN_BYTES = 16, 2                                        # ZeRO §3.1: 2Ψ + 2Ψ + 12Ψ; BF16
MEM = {"policy": POLICY_PARAMS * TRAINED_BYTES / 1e9, "value": POLICY_PARAMS * TRAINED_BYTES / 1e9,
       "reward": POLICY_PARAMS * FROZEN_BYTES / 1e9, "reference": POLICY_PARAMS * FROZEN_BYTES / 1e9}
MEM_TOTAL = sum(MEM.values())
assert [round(v) for v in MEM.values()] == [112, 112, 14, 14] and round(MEM_TOTAL) == 252   # notes §21 example 3
MAC_GB = int(re.search(r"(\d+) GB", measured(LM, "machine")).group(1))
assert MAC_GB == 64 and 3.5 < MEM_TOTAL / MAC_GB < 4                       # said: "nearly four times"

# ---------------------------------------------------------------------------- our run (measured on this Mac)

MODEL = measured(LM, "headline", "model")
assert MODEL == "Qwen2.5-0.5B-Instruct"
assert measured(LM, "setup", "reward").startswith("1 if the last number inside the last \\boxed{...} equals")
EX = measured(LM, "examples")[0]
assert EX["id"] == "test-1028" and EX["ref"] == 200
assert (EX["base"]["answer"], EX["base"]["correct"]) == (300, 0) and (EX["trained"]["answer"], EX["trained"]["correct"]) == (200, 1)
assert EX["question"].startswith("Tom buys a bedroom set for $3000.") and EX["question"].endswith("pay per month?")

# ---------------------------------------------------------------------------- look

USER = recolored("user", INK, CARD_HI, prefix=PREFIX)        # the person who chooses (neutral)
POLICY_ICON = recolored("llm", ACTION, CORAL_FILL, prefix=PREFIX)
RM_ICON = recolored("llm", REWARD, GOLD_FILL, prefix=PREFIX)
VALUE_ICON = recolored("llm", VALUE, BLUE_FILL, prefix=PREFIX)
REF_ICON = "llm"                                             # the course's llm icon is VIOLET: the leash's colour
LOCK = recolored("lock", MUTED, CARD, prefix=PREFIX)
LEASH = EXPLORE                                              # the KL leash (VIOLET), as in Episode 20


def cite(s, size=20):
    return text(s, size, MUTED)


def left_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.RIGHT)


def formula(parts, colors, size=42):
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


def moon(r=0.32):
    disc = m.Circle(radius=r).set_stroke(STATE, 3).set_fill(TEAL_FILL, 1)
    craters = m.VGroup(*[
        m.Circle(radius=k * r).set_stroke(STATE, 2).set_fill(opacity=0).move_to(disc.get_center() + np.array([dx, dy, 0]) * r)
        for k, dx, dy in ((0.24, -0.3, 0.25), (0.16, 0.38, 0.12), (0.18, 0.02, -0.42))
    ])
    return m.VGroup(disc, craters)


def badge(letter, color, fill, r=0.24):
    disc = m.Circle(radius=r).set_stroke(color, 2.5).set_fill(fill, 1)
    return m.VGroup(disc, title_text(letter, 24, color).move_to(disc))


def answer_chip(letter, body, color, width=4.3, height=0.72, size=22):
    box = card(width, height, color, CARD, radius=0.14, stroke=2.5)
    row = m.VGroup(title_text(letter, 28, color), text(body, size, INK)).arrange(m.RIGHT, buff=0.25)
    row.move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
    return m.VGroup(box, row)


def tag(label, color, fill, size=22):
    """A pill whose word sits on the common baseline (kit's chip), in the pill's colour."""
    c = chip(label, color, fill, width=text(label, size, weight=m.BOLD).width + 0.6, height=0.56, size=size)
    c[0].set_stroke(color, 2).set_fill(fill, 1)
    c[1].set_color(color)
    return c


def model_card(icon_name, name, color, width=2.4, height=1.6, top=None):
    """A model as a card: icon over its name (centred, or from `top` below the card's top edge)."""
    box = card(width, height, color, CARD, radius=0.18, stroke=2.5)
    body = m.VGroup(icon(icon_name, 0.58), text(name, 22, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.14)
    body.move_to(box)
    if top is not None:
        body.align_to(box, m.UP).shift(top * m.DOWN)
    return m.VGroup(box, *body)


def bt_marker(plot, g):
    """The point (gap, chance) on the Bradley–Terry curve with dashed guides and its percentage."""
    p = float(bradley_terry_prob(g, 0.0))
    pt = plot.coords(g, p)
    v = m.DashedLine(plot.coords(g, plot.y0), pt, dash_length=0.08).set_stroke(MUTED, 2)
    h = m.DashedLine(plot.coords(plot.x0, p), pt, dash_length=0.08).set_stroke(MUTED, 2)
    dot = m.Dot(pt, radius=0.09, color=GREEN).set_z_index(3)
    read = code(f"{100 * p:.0f}%", 22, GREEN, weight=m.BOLD).next_to(pt, m.UL, buff=0.16)
    return m.VGroup(v, h, dot, read)


def slide_bt(plot, marker, g0, g1, seconds=1.0):
    def step(_, alpha):
        marker.become(bt_marker(plot, g0 + (g1 - g0) * alpha))
    return m.UpdateFromAlphaFunc(marker, step, run_time=seconds, rate_func=m.smooth)


def spring(a, b, waves=7, amp=0.09):
    """The KL leash as Episode 20 draws it: a VIOLET spring between two points."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = b - a
    n = np.array([-d[1], d[0], 0.0]) / (np.linalg.norm(d) + 1e-9)
    ts = np.linspace(0, 1, 16 * waves + 1)
    pts = [a + t * d + amp * math.sin(2 * math.pi * waves * t) * n for t in ts]
    return m.VMobject().set_points_smoothly(pts).set_stroke(LEASH, 4)


def vertical_arrow(x, y0, y1, color=SIGNAL):
    return m.Arrow([x, y0, 0], [x, y1, 0], buff=0, color=color, stroke_width=6, max_tip_length_to_length_ratio=0.3)


# ---------------------------------------------------------------------------- the episode


class Ep21LearningFromHumanPreferences(Episode):
    number = 21

    def lesson(self):
        self.two_answers()
        self.bradley_terry()
        self.reward_model()
        self.rl_stage()
        self.four_models()
        self.history()
        self.instructgpt()
        self.the_catch()
        self.recap()

    # ------------------------------------------------------------------ 1 · two answers

    def two_answers(self):
        self.next_section("1 · Two answers", notes=(
            f"Episode 20's checker, on our run's example {EX['id']} (lm_rl.json examples[0]): reference {EX['ref']:.0f}; "
            f"two answers from the run boxed {EX['base']['answer']:.0f} (reward {EX['base']['correct']:.0f}) and "
            f"{EX['trained']['answer']:.0f} (reward {EX['trained']['correct']:.0f}). Reward rule (setup.reward): "
            "1 if the last number inside the last \\boxed{...} equals the reference, else 0. The moon-landing prompt and the "
            "two answers are InstructGPT's own (Ouyang et al. 2022, Fig. 2: answers there labelled D 'People went to the "
            "moon…' and A 'Explain gravity…', truncated as in the figure; here renamed A and B). Notes §21: 'It is hard for "
            "a person to give 7.3/10. It is easy to say A is better than B.' Headline (abstract): the 1.3B InstructGPT "
            "preferred to the 175B GPT-3."))
        q = message("user", "Tom buys a bedroom set for $3000 … How much does he have to pay per month?", width=9.6)
        q.move_to([0, 1.3, 0])
        lab20 = cite("episode 20 · two answers from our run on this Mac").next_to(q, m.UP, buff=0.18)
        rows = []
        for k, (ans, r) in enumerate(((EX["base"]["answer"], EX["base"]["correct"]),
                                      (EX["trained"]["answer"], EX["trained"]["correct"]))):
            y = -0.05 - 1.2 * k
            s = f"… \\boxed{{{ans:.0f}}}"
            c = chip(s, ACTION, CORAL_FILL, width=code(s, 22).width + 0.55, mono=True, size=26).move_to([-4.1, y, 0])
            chip_r = reward_chip(r, 24).move_to([4.0, y, 0])
            rows.append((c, chip_r))
        checker = pill(f"checker: = {EX['ref']:.0f} ?", MUTED, CARD, 24, mono=True).move_to([0, -0.65, 0])
        links = m.VGroup()
        for c, chip_r in rows:
            links.add(arrow(c.get_right(), checker.get_left() + [0, 0.12 * np.sign(c.get_y() - checker.get_y()), 0], MUTED))
            links.add(arrow(checker.get_right() + [0, 0.12 * np.sign(c.get_y() - checker.get_y()), 0], chip_r.get_left(), MUTED))
        recap20 = m.VGroup(q, lab20, *[x for row in rows for x in row], checker, links)

        prompt = message("user", "Explain the moon landing to a 6 year old", width=6.4)
        orb = moon(0.34)
        prompt.move_to([0, 1.6, 0])
        orb.next_to(prompt, m.RIGHT, buff=0.4)
        top = m.VGroup(prompt, orb)
        ans_a = message("assistant", "People went to the moon…", width=5.2).move_to([-3.25, -0.3, 0])
        ans_b = message("assistant", "Explain gravity…", width=5.2).move_to([3.25, -0.3, 0])
        bad_a = badge("A", ACTION, CORAL_FILL).move_to(ans_a[0].get_corner(m.UR) + [-0.4, -0.38, 0])
        bad_b = badge("B", ACTION, CORAL_FILL).move_to(ans_b[0].get_corner(m.UR) + [-0.4, -0.38, 0])
        down_a = arrow(prompt.get_bottom() + [-1.0, 0, 0], ans_a.get_top() + [0.9, 0, 0], ACTION)
        down_b = arrow(prompt.get_bottom() + [1.0, 0, 0], ans_b.get_top() + [-0.9, 0, 0], ACTION)
        src = right_at(cite("Ouyang et al., 2022, Fig. 2"), 6.5, -3.3)

        no_chk = pill("checker", MUTED, CARD, 22, mono=True)
        strike = m.Line(no_chk.get_left() + [-0.1, -0.1, 0], no_chk.get_right() + [0.1, 0.1, 0]).set_stroke(RED, 4)
        no_check = m.VGroup(no_chk, strike).move_to([-3.25, -1.55, 0])
        no_score = code("score: 7 / 10 ?", 24, MUTED).move_to([3.25, -1.55, 0])
        chosen = tag("chosen", GREEN, GREEN_FILL).move_to([-3.25, -1.55, 0])
        rejected = tag("rejected", RED, RED_FILL).move_to([3.25, -1.55, 0])
        person = icon(USER, 0.85).move_to([0, -2.45, 0])
        picks = text("a person picks in seconds", 22, INK).next_to(person, m.RIGHT, buff=0.3)
        pick_arrow = arrow(person.get_left() + [-0.1, 0.1, 0], chosen.get_bottom() + [0.5, -0.05, 0], GREEN)

        rec_box = card(10.0, 2.55, EDGE, CARD, radius=0.2, stroke=2.5).move_to([0, 0.45, 0])
        rec_title = left_at(code("one comparison", 20, MUTED), -4.65, 1.4)
        rec_rows = m.VGroup()
        for k, (key, val, col) in enumerate((("prompt", "Explain the moon landing to a 6 year old", STATE),
                                             ("chosen", "People went to the moon…", GREEN),
                                             ("rejected", "Explain gravity…", RED))):
            y = 0.8 - 0.62 * k
            rec_rows.add(m.VGroup(left_at(code(key, 22, col, weight=m.BOLD), -4.65, y), left_at(text(val, 24, INK), -2.4, y)))
        record = m.VGroup(rec_box, rec_title)
        shadows = m.VGroup(*[card(10.0, 2.55, EDGE, BG, radius=0.2, stroke=2).move_to([0.18 * k, 0.45 + 0.16 * k, 0])
                             for k in (2, 1)])
        shadows.set_z_index(-1)
        many = right_at(cite("collect thousands of these"), 5.18, 2.2)
        teaser = m.VGroup(text("InstructGPT 1.3B", 28, ACTION, weight=m.BOLD), text("preferred over", 28, GREEN),
                          text("GPT-3 175B", 28, MUTED, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3).move_to([0, -2.05, 0])
        on_baseline(teaser)
        teaser_src = cite("Ouyang et al., 2022, abstract").next_to(teaser, m.DOWN, buff=0.25)

        self.say(
            "[In episode twenty, a checker graded our model's answers]: [the boxed number matches the reference, or it doesn't.]",
            m.AnimationGroup(self.new_header("01 · two answers", "Which one is better?"), m.FadeIn(q, shift=0.1 * m.DOWN),
                             m.FadeIn(lab20), *[m.FadeIn(c, shift=0.1 * m.RIGHT) for c, _ in rows]),
            m.AnimationGroup(m.FadeIn(checker), m.LaggedStart(*[m.GrowArrow(a) for a in links], lag_ratio=0.15),
                             m.LaggedStart(*[m.FadeIn(r, scale=0.7) for _, r in rows], lag_ratio=0.4)),
        )
        self.say(
            "[But what's the right answer to this prompt]: [explain the moon landing to a six-year-old?]",
            m.FadeOut(recap20, shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(prompt, shift=0.1 * m.DOWN), m.FadeIn(orb, scale=0.6)),
        )
        self.say(
            "[Two answers come back.] [One begins: people went to the moon.] [The other wanders off: explain gravity.]",
            m.AnimationGroup(m.GrowArrow(down_a), m.GrowArrow(down_b), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(ans_a, shift=0.1 * m.DOWN), m.FadeIn(bad_a, scale=0.6)),
            m.AnimationGroup(m.FadeIn(ans_b, shift=0.1 * m.DOWN), m.FadeIn(bad_b, scale=0.6)),
        )
        self.say(
            "[No checker can grade this], [and a score out of ten is hard to give.] [But which one is better? You can tell in a second.]",
            m.FadeIn(no_check, shift=0.1 * m.UP),
            m.FadeIn(no_score, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(no_check), m.FadeOut(no_score), m.FadeIn(person, shift=0.1 * m.UP), m.FadeIn(picks),
                             ans_a[0].animate.set_stroke(GREEN, 3.5), ans_b[0].animate.set_stroke(RED, 3.5),
                             m.FadeIn(chosen, scale=0.8), m.FadeIn(rejected, scale=0.8), m.GrowArrow(pick_arrow)),
        )
        hook = m.VGroup(top, ans_a, ans_b, bad_a, bad_b, down_a, down_b, chosen, rejected, person, picks, pick_arrow)
        self.say(
            "[That one choice is the data]: [a prompt, a chosen answer, and a rejected one.]",
            m.AnimationGroup(m.FadeOut(hook, scale=0.9), m.FadeIn(record, scale=0.95)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in rec_rows], lag_ratio=0.35),
        )
        self.say(
            "[In twenty twenty-two, choices like this one taught a small model to beat one over a hundred times its size], "
            "[at least in people's eyes.]",
            m.AnimationGroup(m.FadeIn(shadows, shift=0.15 * m.DOWN), m.FadeIn(many),
                             m.FadeIn(teaser, shift=0.1 * m.UP), m.FadeIn(teaser_src)),
            m.Circumscribe(teaser, color=GREEN, buff=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · Bradley–Terry

    def bradley_terry(self):
        self.next_section("2 · Bradley–Terry", notes=(
            "Bradley & Terry (1952): P(A ≻ B) = exp r_A / (exp r_A + exp r_B) = σ(r_A − r_B); "
            "rl_reference.bradley_terry_prob. The scores 1.2 and 0.4 are an EXAMPLE (notes §21 worked example): gap 0.8 → "
            f"{P_AB:.4f}. The curve: σ(gap) for gaps −4 … 4 (161 points). Gap 0 → 0.5, gap 1 → {P_GAP1:.4f} "
            f"(odds e : 1, the gap is the log-odds), gap 4 → {P_GAP4:.4f}. Shift invariance: σ((r_A + 5) − (r_B + 5)) = "
            "σ(r_A − r_B), so only differences are learned (InstructGPT re-centres its reward model so labeler "
            "demonstrations average 0). Christiano et al. 2017 §2.2.3: 'Just as the difference in Elo points of two chess "
            "players estimates the probability of one player defeating the other …'."))
        row_a = answer_chip("A", "People went to the moon…", GREEN, width=4.6).move_to([-4.2, 1.45, 0])
        row_b = answer_chip("B", "Explain gravity…", RED, width=4.6).move_to([-4.2, 0.55, 0])
        s_a = pill(f"{R_A:.2f}", REWARD, GOLD_FILL, 24, mono=True).move_to([-1.05, 1.45, 0])
        s_b = pill(f"{R_B:.2f}", REWARD, GOLD_FILL, 24, mono=True).move_to([-1.05, 0.55, 0])
        s_a2 = pill(f"{R_A + SHIFT:.2f}", REWARD, GOLD_FILL, 24, mono=True).move_to(s_a)
        s_b2 = pill(f"{R_B + SHIFT:.2f}", REWARD, GOLD_FILL, 24, mono=True).move_to(s_b)
        plus = m.VGroup(*[code(f"+{SHIFT}", 22, REWARD, weight=m.BOLD).next_to(s, m.RIGHT, buff=0.15) for s in (s_a, s_b)])
        score_lab = code("reward", 20, REWARD).next_to(s_a, m.UP, buff=0.22)
        example = left_at(cite("for example"), -6.3, 2.15)

        def gap_line(a, b):
            return left_at(colored([("gap", MUTED), (f"{a:.2f} − {b:.2f} =", INK), (f"{a - b:.2f}", REWARD)], 24), -6.3, -0.45)

        gap = gap_line(R_A, R_B)
        gap2 = gap_line(R_A + SHIFT, R_B + SHIFT)
        chance = left_at(colored([("chance A is picked:", MUTED), (f"{P_AB:.2f}", GREEN)], 24), -6.3, -1.15)
        chess = left_at(cite("like chess ratings · Christiano et al., 2017"), -6.3, -3.25)

        f = formula([r"P(A \succ B)", "=", r"\sigma(", "r_A", "-", "r_B", ")"], [GREEN, INK, INK, REWARD, INK, REWARD, INK], 46)
        f.move_to([3.35, 1.95, 0])
        bt_src = cite("Bradley & Terry, 1952").next_to(f, m.DOWN, buff=0.2)
        plot = Plot((-4, 4), (0, 1), width=5.2, height=2.5, center=(3.35, -1.2, 0), x_ticks=[-4, -2, 0, 2, 4],
                    y_ticks=[0, 0.5, 1], x_label="gap between the scores", y_label="chance A is picked", x_fmt="{:g}")
        xs = np.linspace(-4, 4, 161)
        curve = plot.line(xs, bradley_terry_prob(xs, 0.0), GREEN, 5)
        marker = bt_marker(plot, R_A - R_B)

        self.say(
            "[Give each answer a hidden score: its reward.] [For example, one point two for A], [and zero point four for B.]",
            m.AnimationGroup(self.new_header("02 · Bradley–Terry", "From choices to chances"),
                             m.FadeIn(row_a, shift=0.1 * m.RIGHT), m.FadeIn(row_b, shift=0.1 * m.RIGHT), m.FadeIn(score_lab)),
            m.AnimationGroup(m.FadeIn(s_a, scale=0.7), m.FadeIn(example)),
            m.FadeIn(s_b, scale=0.7),
        )
        self.say(
            "[The Bradley-Terry model, from nineteen fifty-two], [turns the gap between two scores into a chance]: "
            "[the sigmoid of the difference.]",
            m.AnimationGroup(m.FadeIn(f[0:2], shift=0.1 * m.DOWN), m.FadeIn(bt_src)),
            m.AnimationGroup(m.FadeIn(gap, shift=0.1 * m.UP), m.FadeIn(plot)),
            m.AnimationGroup(m.FadeIn(f[2:], shift=0.1 * m.DOWN), m.Create(curve, run_time=1.6)),
        )
        self.say(
            "[Here the gap is zero point eight], [so people should pick A about sixty-nine percent of the time.]",
            m.AnimationGroup(m.Circumscribe(gap, color=REWARD, buff=0.08), m.Create(marker[0]), m.FadeIn(marker[2], scale=0.5)),
            m.AnimationGroup(m.Create(marker[1]), m.FadeIn(marker[3]), m.FadeIn(chance, shift=0.1 * m.UP)),
        )
        g = R_A - R_B
        self.say(
            "[Equal scores: a coin flip.] [A gap of one: about seventy-three percent.] [A gap of four: ninety-eight.]",
            slide_bt(plot, marker, g, 0.0, 1.2),
            slide_bt(plot, marker, 0.0, 1.0, 1.0),
            slide_bt(plot, marker, 1.0, 4.0, 1.2),
        )
        self.say(
            "[As with chess ratings, only the gap matters]: [add five to both scores, and nothing changes.]",
            m.AnimationGroup(slide_bt(plot, marker, 4.0, g, 1.2), m.FadeIn(chess)),
            m.AnimationGroup(m.Transform(s_a, s_a2), m.Transform(s_b, s_b2), m.FadeIn(plus, shift=0.1 * m.LEFT),
                             m.Transform(gap, gap2), m.Circumscribe(chance, color=GREEN, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · the reward model

    def reward_model(self):
        self.next_section("3 · The reward model", notes=(
            "A reward model r_φ(x, y): a language model with a scalar head reads the prompt and ONE answer and returns one "
            "number. Loss per pair (Stiennon 2020; InstructGPT eq. 1): −log σ(r_w − r_l). On the example scores 1.2 / 0.4: "
            f"P = {P_AB:.3f}, loss {LOSS:.3f}; wrong way round P = {P_BA:.3f}, loss {LOSS_WRONG:.3f} (notes §21 worked "
            f"example). Training (ILLUSTRATION): {STEPS} steps of gradient descent at learning rate {LR:g} applied to the two "
            "scores themselves (−∂loss/∂r_A = ∂loss/∂r_B = 1 − σ(gap)): scores "
            + " → ".join(f"({a:.2f}, {b:.2f})" for a, b in TRAJ) + "; loss "
            + " → ".join(f"{x:.3f}" for x in LOSSES) + ". A real reward model moves its weights, and every pair moves the "
            "shared network. Rankings: InstructGPT labelers ranked K = 4 to 9 answers per prompt, giving C(K,2) pairs, "
            f"C(4,2) = {PAIRS4}, C(9,2) = {PAIRS9} (§3.5); the four answers and the ranking D > C > A = B are Fig. 2's "
            "(the tie is drawn dashed). InstructGPT batches all C(K,2) pairs of one prompt together (one epoch only: more "
            "overfit)."))
        in_a = chip("prompt + A", GREEN, CARD, width=2.1, height=0.62, size=24, mono=True).move_to([-5.35, 1.95, 0])
        in_b = chip("prompt + B", RED, CARD, width=2.1, height=0.62, size=24, mono=True).move_to([-5.35, 1.05, 0])
        rm = model_card(RM_ICON, "reward model", REWARD, width=2.45, height=1.6).move_to([-2.3, 1.5, 0])
        out_a = code(f"{R_A:.2f}", 28, REWARD, weight=m.BOLD).move_to([0.25, 1.95, 0])
        out_b = code(f"{R_B:.2f}", 28, REWARD, weight=m.BOLD).move_to([0.25, 1.05, 0])
        ins = m.VGroup(arrow(in_a.get_right(), rm[0].get_left() + [0, 0.3, 0], MUTED),
                       arrow(in_b.get_right(), rm[0].get_left() + [0, -0.3, 0], MUTED))
        outs = m.VGroup(arrow(rm[0].get_right() + [0, 0.3, 0], out_a.get_left(), REWARD),
                        arrow(rm[0].get_right() + [0, -0.3, 0], out_b.get_left(), REWARD))

        lf = formula([r"\mathrm{loss}", "=", r"-\log\sigma(", "r_A", "-", "r_B", ")"],
                     [SIGNAL, INK, INK, REWARD, INK, REWARD, INK], 38)
        left_at(lf, 1.45, 1.95)
        lv = left_at(colored([("= −log", MUTED), (f"{P_AB:.2f}", GREEN), ("=", MUTED), (f"{LOSS:.3f}", SIGNAL)], 24), 1.65, 1.2)
        wrong = left_at(colored([("wrong way: −log", MUTED), (f"{P_BA:.2f}", RED), ("=", MUTED),
                                 (f"{LOSS_WRONG:.3f}", RED)], 22), 1.65, 0.6)
        n_wrong = len(f"wrong way: −log {P_BA:.2f}".replace(" ", ""))

        def step_line(k, loss):
            return left_at(colored([(f"step {k} · loss", MUTED), (f"{loss:.3f}", SIGNAL)], 24), 1.65, 0.6)

        steps = step_line(0, LOSSES[0])
        bars = Bars([R_A, R_B], labels=["A · chosen", "B · rejected"], top=3.0, bottom=-1.0, width=3.6, height=2.3,
                    color=REWARD, center=(-3.6, -1.85, 0), size=20)
        bars.labels[0].set_color(GREEN)
        bars.labels[1].set_color(RED)
        bars_lab = cite("the two scores · for example").move_to([-3.6, -0.4, 0])
        up = vertical_arrow(-5.65, -2.55, -1.55)
        down = vertical_arrow(-1.55, -1.55, -2.55)
        mini = Plot((-1, 3), (0, 1), width=4.0, height=1.65, center=(3.55, -1.95, 0), x_ticks=[-1, 0, 1, 2, 3],
                    y_ticks=[0, 1], x_label="gap", y_label="chance A is picked")
        mxs = np.linspace(-1, 3, 101)
        mcurve = mini.line(mxs, bradley_terry_prob(mxs, 0.0), GREEN, 4)
        mdot = mini.dot(R_A - R_B, P_AB, GREEN, 0.1)

        def train(seconds):
            def step(_, alpha):
                x = alpha * STEPS
                j = min(int(np.floor(x)), STEPS - 1)
                t = x - j
                a = (1 - t) * TRAJ[j][0] + t * TRAJ[j + 1][0]
                b = (1 - t) * TRAJ[j][1] + t * TRAJ[j + 1][1]
                bars.bars[0].become(bars._rect(0, a))
                bars.bars[1].become(bars._rect(1, b))
                bars.numbers[0].become(bars._number(0, a))
                bars.numbers[1].become(bars._number(1, b))
                gp = a - b
                mdot.move_to(mini.coords(gp, float(bradley_terry_prob(gp, 0.0))))
                k = int(round(x))
                steps.become(step_line(k, -math.log(float(bradley_terry_prob(a, b)))))
            return m.UpdateFromAlphaFunc(bars, step, run_time=seconds, rate_func=m.linear)

        # rankings: Fig. 2's four answers, best first; every two make a pair
        items = [("1", "People went to the moon…"), ("2", "Moon is natural satellite of…"), ("3", "Explain gravity…"),
                 ("3", "Explain war…")]
        rank_rows, anchors = m.VGroup(), []
        for k, (rank, body) in enumerate(items):
            y = -0.45 - 0.72 * k
            box = card(4.4, 0.56, ACTION, CARD, radius=0.12, stroke=2).move_to([-3.45, y, 0])
            num = code(rank, 22, INK, weight=m.BOLD).move_to([-6.0, y, 0])
            words = left_at(text(body, 20, INK), box.get_left()[0] + 0.25, y)
            rank_rows.add(m.VGroup(num, box, words))
            anchors.append(box.get_right() + [0.08, 0, 0])
        arcs = m.VGroup()
        for i in range(4):
            for j in range(i + 1, 4):
                arc = m.ArcBetweenPoints(anchors[i], anchors[j], angle=-0.8 * np.pi)
                if (i, j) == (2, 3):
                    arc = m.DashedVMobject(arc.set_stroke(MUTED, 2.5), num_dashes=8)
                else:
                    arc.set_stroke(INK, 2.5)
                arcs.add(arc)
        assert len(arcs) == PAIRS4
        rank_head = left_at(text("one prompt, four answers, ranked", 22, INK, weight=m.BOLD), -6.3, 0.2)
        four = left_at(title_text(f"4 answers → {PAIRS4} pairs", 36, INK), 1.3, -1.0)
        nine = left_at(title_text(f"9 answers → {PAIRS9} pairs", 36, INK), 1.3, -1.8)
        rank_src = left_at(cite("Ouyang et al., 2022: Fig. 2, §3.5"), 1.3, -2.6)

        self.say(
            "[So we train a reward model to find those scores.] [It reads the prompt and one answer], [and returns a single number.]",
            m.AnimationGroup(self.new_header("03 · the reward model", "A model of their taste"), m.FadeIn(rm, scale=0.9)),
            m.AnimationGroup(m.FadeIn(in_a, shift=0.1 * m.RIGHT), m.FadeIn(in_b, shift=0.1 * m.RIGHT),
                             *[m.GrowArrow(a) for a in ins]),
            m.AnimationGroup(*[m.GrowArrow(a) for a in outs], m.FadeIn(out_a, shift=0.1 * m.RIGHT),
                             m.FadeIn(out_b, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[Its loss on our pair is minus the log of that sixty-nine percent]: [about zero point three seven.]",
            m.FadeIn(lf, shift=0.1 * m.DOWN),
            m.FadeIn(lv, shift=0.1 * m.UP),
        )
        self.say(
            "[Had it scored the pair the wrong way round], [the loss would be one point one seven.]",
            m.FadeIn(wrong[:n_wrong], shift=0.1 * m.UP),
            m.FadeIn(wrong[n_wrong:], shift=0.1 * m.UP),
        )
        self.say(
            "[Each gradient step lifts the chosen score and lowers the rejected one], "
            "[so the pair climbs the curve, and the loss falls.]",
            m.AnimationGroup(m.FadeOut(wrong), m.FadeIn(steps), m.FadeIn(bars), m.FadeIn(bars_lab), m.GrowArrow(up),
                             m.GrowArrow(down), m.FadeIn(mini), m.FadeIn(mcurve), m.FadeIn(mdot, scale=0.5)),
            train(3.2),
        )
        self.say(
            "[The InstructGPT labelers ranked four to nine answers at a time.] [Every two answers make a pair], "
            "[so nine answers give thirty-six.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(bars, bars_lab, up, down, mini, mcurve, mdot)), m.FadeIn(rank_head),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in rank_rows], lag_ratio=0.2)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(a) for a in arcs], lag_ratio=0.2, run_time=1.6),
                             m.FadeIn(four, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(nine, shift=0.1 * m.UP), m.FadeIn(rank_src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · the RL stage

    def rl_stage(self):
        self.next_section("4 · Reinforcement learning against the reward model", notes=(
            "InstructGPT step 3 (Fig. 2): a new prompt ('Write a story about frogs'), the policy writes 'Once upon a "
            "time…', the frozen reward model scores the whole answer once (Episode 20: one reward at the end), PPO "
            "(Episode 18) updates the policy. The KL penalty (Episode 20's leash) to the frozen SFT model: InstructGPT eq. "
            f"2, objective = E[r_θ(x, y) − β log(π_φ^RL(y|x) / π^SFT(y|x))] (+ γ·pretraining term in PPO-ptx), β = {BETA}; "
            "their φ/θ swapped relative to our notation. Without the leash the policy drifts toward text the reward model "
            "overrates (Episode 25)."))
        prompt = message("user", "Write a story about frogs", width=3.75).move_to([-4.5, 1.55, 0])
        pol = model_card(POLICY_ICON, "policy", ACTION, width=2.3, height=1.6).move_to([-0.45, 1.55, 0])
        toks = token_row(["Once", "upon", "a", "time"], ACTION, size=22).move_to([3.3, 1.55, 0])
        rm = model_card(RM_ICON, "reward model", REWARD, width=2.6, height=1.6).move_to([4.95, -0.55, 0])
        score = pill("score r", REWARD, GOLD_FILL, 24, mono=True).move_to([1.95, -0.55, 0])
        ref = model_card(REF_ICON, "reference", LEASH, width=2.6, height=1.6).move_to([-4.5, -0.55, 0])
        frozen_ref = code("frozen copy", 20, MUTED).next_to(ref, m.DOWN, buff=0.12)
        a1 = arrow(prompt.get_right(), pol[0].get_left(), STATE)
        a2 = arrow(pol[0].get_right(), toks.get_left(), ACTION)
        a3 = arrow(toks.get_bottom() + [1.0, 0, 0], rm[0].get_top() + [-0.3, 0, 0], ACTION)
        a4 = arrow(rm[0].get_left(), score.get_right(), REWARD)
        ppo = m.Arrow(score.get_left() + [-0.05, 0.05, 0], pol[0].get_bottom() + [0.55, 0, 0], buff=0.08, color=SIGNAL,
                      stroke_width=6, max_tip_length_to_length_ratio=0.15)
        ppo_lab = pill("PPO · episode 18", SIGNAL, PINK_FILL, 20).move_to([-0.6, -0.75, 0])
        leash = spring(pol[0].get_left() + [-0.05, -0.45, 0], ref[0].get_top() + [0.7, 0.05, 0])
        kl = code("KL", 22, LEASH, weight=m.BOLD).next_to(leash.get_center(), m.DR, buff=0.22)

        R = formula(["R", "=", r"r_\phi(x, y)", "-", r"\beta", r"\log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}"],
                    [INK, INK, REWARD, INK, LEASH, LEASH], 40).move_to([-1.2, -2.25, 0])
        sub_r = text("reward model", 20, REWARD).next_to(R[2], m.DOWN, buff=0.35)
        sub_kl = text("KL leash · episode 20", 20, LEASH).next_to(R[5], m.DOWN, buff=0.12)
        sub_r.set_y(sub_kl.get_y())
        beta = pill(f"β = {BETA} · InstructGPT", LEASH, VIOLET_FILL, 22).move_to([4.55, -2.25, 0])
        src = right_at(cite("Ouyang et al., 2022: Fig. 2, eq. 2"), 6.5, -3.25)

        self.say(
            "[Now the reward model stands in for the people.] [A new prompt arrives: write a story about frogs.]",
            m.AnimationGroup(self.new_header("04 · the RL stage", "Chase the score, on a leash"), m.FadeIn(rm, scale=0.9)),
            m.AnimationGroup(m.FadeIn(prompt, shift=0.1 * m.RIGHT), m.GrowArrow(a1), m.FadeIn(pol, scale=0.9), m.FadeIn(src)),
        )
        self.say(
            "[The policy writes its answer, token by token], [and the reward model scores the whole answer, once.]",
            m.AnimationGroup(m.GrowArrow(a2), m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.RIGHT) for t in toks], lag_ratio=0.3)),
            m.AnimationGroup(m.GrowArrow(a3), m.Succession(m.Wait(0.5), m.AnimationGroup(m.GrowArrow(a4), m.FadeIn(score, scale=0.7)))),
        )
        self.say(
            "[Then PPO, from episode eighteen], [pushes the policy toward answers that score higher.]",
            m.AnimationGroup(m.GrowArrow(ppo), m.FadeIn(ppo_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(pulse(ppo, SIGNAL), m.Circumscribe(pol[0], color=SIGNAL, buff=0.06)),
        )
        self.say(
            "[Left alone, it would drift toward odd text the reward model happens to love.]",
            m.Wiggle(pol, scale_value=1.06, rotation_angle=0.04 * np.pi, run_time=1.6),
        )
        self.say(
            "[So a frozen copy of the starting model holds it], [on the KL leash from episode twenty.]",
            m.AnimationGroup(m.FadeIn(ref, scale=0.9), m.FadeIn(frozen_ref)),
            m.AnimationGroup(m.Create(leash), m.FadeIn(kl)),
        )
        self.say(
            "[The reward becomes the score minus beta times the drift]: [InstructGPT used a beta of zero point zero two.]",
            m.AnimationGroup(m.FadeIn(R, shift=0.1 * m.UP), m.FadeIn(sub_r), m.FadeIn(sub_kl)),
            m.FadeIn(beta, scale=0.85),
        )
        self.wait(0.3)
        self.s4 = dict(pol=pol, rm=rm, ref=ref, rest=m.VGroup(prompt, toks, score, frozen_ref, a1, a2, a3, a4, ppo, ppo_lab,
                                                                leash, kl, R, sub_r, sub_kl, beta, src))

    # ------------------------------------------------------------------ 5 · four models in memory

    def four_models(self):
        self.next_section("5 · Four models in memory", notes=(
            "During RLHF with PPO four models sit in memory (Hu et al. 2024, OpenRLHF §2.1: 'actor, critic, reward, "
            "reference'): the policy (trained), the value network = PPO's critic (trained; InstructGPT initialised its 6B "
            "value function from the reward model, App. C.4), the reward model (frozen) and the reference = the SFT model "
            "(frozen, for the KL). Memory [DERIVED, notes §21 worked example 3]: a 7B policy; trained models with "
            f"mixed-precision Adam at {TRAINED_BYTES} bytes per parameter (ZeRO §3.1: 2Ψ + 2Ψ + 12Ψ), frozen ones in BF16 at "
            f"{FROZEN_BYTES}: " + " + ".join(f"{v:.0f}" for v in MEM.values()) + f" = {MEM_TOTAL:.0f} GB of model states "
            "only (no activations, no KV cache for generation), all four the policy's size for the sketch (InstructGPT "
            f"used one 6B reward model and value function for every policy size). This Mac: {MAC_GB} GB unified memory "
            f"(lm_rl.json machine): {MEM_TOTAL / MAC_GB:.2f}×."))
        s4 = self.s4
        xs = (-4.95, -1.65, 1.65, 4.95)
        W, H, Y = 3.0, 2.5, 1.05
        pol = model_card(POLICY_ICON, "policy", ACTION, W, H, top=0.25).move_to([xs[0], Y, 0])
        val = model_card(VALUE_ICON, "value network", VALUE, W, H, top=0.25).move_to([xs[1], Y, 0])
        rm = model_card(RM_ICON, "reward model", REWARD, W, H, top=0.25).move_to([xs[2], Y, 0])
        ref = model_card(REF_ICON, "reference", LEASH, W, H, top=0.25).move_to([xs[3], Y, 0])
        subs = [text(s, 20, MUTED) for s in ("writes the answers", "PPO's critic", "scores the answers", "anchors the leash")]
        tags = []
        sub_y = min(c[2].get_bottom()[1] for c in (pol, val, rm, ref)) - 0.28
        for k, (c, s) in enumerate(zip((pol, val, rm, ref), subs)):
            s.move_to([xs[k], sub_y, 0])
            if k < 2:
                tg = pill("trained", SIGNAL, PINK_FILL, 20)
            else:
                tg = m.VGroup(icon(LOCK, 0.32), pill("frozen", MUTED, CARD, 20)).arrange(m.RIGHT, buff=0.12)
            tags.append(tg.move_to([xs[k], c[0].get_bottom()[1] + 0.42, 0]))
        init = m.CurvedArrow(rm[0].get_bottom() + [-0.5, -0.06, 0], val[0].get_bottom() + [0.5, -0.06, 0], angle=-0.55 * np.pi,
                             color=VALUE, stroke_width=4, tip_length=0.18)
        init_lab = cite("starts as a copy of the reward model").move_to([0, -1.05, 0])

        x0, x1, yb = -6.0, 6.0, -2.25
        unit = (x1 - x0) / MEM_TOTAL
        segs, seg_labs = m.VGroup(), m.VGroup()
        cols = (ACTION, VALUE, REWARD, LEASH)
        names = ("policy", "value network", "", "")
        x = x0
        for (key, gb), col, nm in zip(MEM.items(), cols, names):
            w = gb * unit
            seg = m.Rectangle(width=w, height=0.5).set_stroke(col, 2.5).set_fill(FILLS[col], 1).move_to([x + w / 2, yb, 0])
            segs.add(seg)
            if nm:
                seg_labs.add(left_at(text(f"{nm} · {gb:.0f} GB", 20, INK), seg.get_left()[0] + 0.25, yb))
            else:
                seg_labs.add(code(f"{gb:.0f}", 20, col, weight=m.BOLD).next_to(seg, m.UP, buff=0.12))
            x += w
        seg_labs.set_z_index(2)
        mem_title = left_at(text("a 7-billion-parameter policy, in memory", 22, INK, weight=m.BOLD), -6.3, -1.6)
        mac_x = x0 + MAC_GB * unit
        mac_line = m.DashedLine([mac_x, yb - 0.45, 0], [mac_x, yb + 0.42, 0], dash_length=0.08).set_stroke(INK, 3)
        mac_lab = code(f"this Mac: {MAC_GB} GB", 20, INK, weight=m.BOLD).next_to(mac_line, m.DOWN, buff=0.1)
        total = right_at(code(f"≈ {MEM_TOTAL:.0f} GB", 22, INK, weight=m.BOLD), x1, yb - 0.6)
        assume = cite("model states only: 16 bytes per trained parameter, 2 per frozen one · Rajbhandari et al., 2020")
        assume.move_to([0, -3.3, 0])

        self.say(
            "[Count what sits in memory during this stage]: [four whole models.]",
            m.AnimationGroup(self.new_header("05 · in memory", "Four models at once"), m.FadeOut(s4["rest"]),
                             m.ReplacementTransform(s4["pol"], pol), m.ReplacementTransform(s4["rm"], rm),
                             m.ReplacementTransform(s4["ref"], ref), run_time=1.4),
            m.FadeIn(val, shift=0.15 * m.UP),
        )
        self.say(
            "[The policy learns], [and so does PPO's critic, the value network], [which InstructGPT started from the reward model.]",
            m.AnimationGroup(m.FadeIn(subs[0]), m.FadeIn(tags[0], scale=0.8)),
            m.AnimationGroup(m.FadeIn(subs[1]), m.FadeIn(tags[1], scale=0.8)),
            m.AnimationGroup(m.Create(init), m.FadeIn(init_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[The reward model and the reference stay frozen]: [one scores the answers, the other anchors the leash.]",
            m.AnimationGroup(m.FadeIn(tags[2], scale=0.8), m.FadeIn(tags[3], scale=0.8)),
            m.AnimationGroup(m.FadeIn(subs[2], shift=0.1 * m.UP), m.FadeIn(subs[3], shift=0.1 * m.UP)),
        )
        self.say(
            "[For a seven-billion-parameter policy, that's about two hundred fifty gigabytes], [nearly four times the memory of this Mac.]",
            m.AnimationGroup(m.FadeIn(mem_title), m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in segs], lag_ratio=0.5),
                             m.FadeIn(seg_labs), m.FadeIn(total), m.FadeIn(assume)),
            m.AnimationGroup(m.Create(mac_line), m.FadeIn(mac_lab, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · history

    def history_card(self, cx, year, icon_name, title, lines, src, color):
        box = card(4.0, 4.3, color, CARD, radius=0.2, stroke=2.5).move_to([cx, -0.45, 0])
        top, left = box.get_top()[1], box.get_left()[0] + 0.3
        yr = left_at(pill(year, SIGNAL, PINK_FILL, 20, mono=True), left, top - 0.42)
        ic = icon(icon_name, 0.55).move_to([box.get_right()[0] - 0.55, top - 0.45, 0])
        head = left_at(text(title, 26, INK, weight=m.BOLD), left, top - 1.05)
        body = m.VGroup(*lines).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        assert body.width < box.width - 0.5, title
        left_at(body, left, 0).align_to([0, top - 1.45, 0], m.UP)
        foot = left_at(cite(src), left, box.get_bottom()[1] + 0.35)
        return m.VGroup(box, yr, ic, head), body, foot

    def history(self):
        self.next_section("6 · A short history", notes=(
            "Christiano, Leike, Brown, Martic, Legg & Amodei, 'Deep reinforcement learning from human preferences' "
            "(NeurIPS 2017; OpenAI + DeepMind): Atari and simulated robots (MuJoCo); §3.2 'The Hopper robot performing a "
            f"sequence of backflips … trained using {BACKFLIP_QUERIES} queries in less than an hour.' Stiennon et al., "
            "'Learning to summarize from human feedback' (NeurIPS 2020): 'our 1.3B human feedback model significantly "
            f"outperforming a supervised model 10× its size ({SUMMARY_WIN}% versus {SUMMARY_SUP}% raw preference score "
            "against reference summaries)' (§4.1). Ziegler et al. 2019 (GPT-2 style and summaries) not drawn. Ouyang et "
            "al. 2022: InstructGPT (GPT-3 fine-tuned to follow instructions). ChatGPT (30 Nov 2022) used 'the same methods "
            "as InstructGPT, but with slight differences in the data collection setup' (not drawn)."))
        c1, b1, f1 = self.history_card(-4.45, "2017", "robot-arm", "Robots and Atari", [
            text("a simulated robot", 22, INK), text("learns to backflip", 22, INK),
            code(f"{BACKFLIP_QUERIES} human choices", 22, GREEN, weight=m.BOLD), text("in under an hour", 22, INK),
        ], "Christiano et al., 2017", EDGE)
        c2, b2, f2 = self.history_card(0.0, "2020", "doc", "Summaries", [
            text("a 1.3B model beats", 22, INK), text("a supervised model", 22, INK), text("10× its size", 22, INK),
            code(f"{SUMMARY_WIN}% vs {SUMMARY_SUP}% preferred", 22, GREEN, weight=m.BOLD),
        ], "Stiennon et al., 2020", EDGE)
        c3, b3, f3 = self.history_card(4.45, "2022", POLICY_ICON, "Instructions", [
            text("GPT-3, fine-tuned on", 22, INK), text("human data, learns to", 22, INK), text("follow instructions:", 22, INK),
            text("InstructGPT", 26, ACTION, weight=m.BOLD),
        ], "Ouyang et al., 2022", ACTION)

        self.say(
            "[The idea is older than chatbots.]",
            m.AnimationGroup(self.new_header("06 · a short history", "From backflips to instructions"),
                             m.FadeIn(c1, shift=0.1 * m.UP)),
        )
        self.say(
            "[In twenty seventeen, researchers at OpenAI and DeepMind taught a simulated robot to backflip], "
            "[from nine hundred human choices, in under an hour.]",
            m.AnimationGroup(m.FadeIn(b1[0:2], shift=0.1 * m.UP), m.FadeIn(f1)),
            m.FadeIn(b1[2:], shift=0.1 * m.UP),
        )
        self.say(
            "[In twenty twenty, the same recipe learned to summarise], "
            "[and a one point three billion parameter model beat a supervised one ten times its size.]",
            m.AnimationGroup(m.FadeIn(c2, shift=0.1 * m.UP), m.FadeIn(f2)),
            m.FadeIn(b2, shift=0.1 * m.UP),
        )
        self.say(
            "[And in twenty twenty-two, it taught GPT-3 to follow instructions]: [InstructGPT.]",
            m.AnimationGroup(m.FadeIn(c3, shift=0.1 * m.UP), m.FadeIn(b3[0:3], shift=0.1 * m.UP), m.FadeIn(f3)),
            m.FadeIn(b3[3], scale=0.8),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · InstructGPT

    def stage_card(self, cx, title, number, sub, color):
        box = card(3.95, 1.85, color, CARD, radius=0.2, stroke=2.5).move_to([cx, 1.35, 0])
        left, top = box.get_left()[0] + 0.3, box.get_top()[1]
        head = left_at(text(title, 21, INK, weight=m.BOLD), left, top - 0.38)
        num = left_at(code(number, 26, color, weight=m.BOLD), left, top - 0.92)
        s = left_at(text(sub, 20, MUTED), left, top - 1.42)
        assert max(head.width, num.width, s.width) < 3.45, title
        return m.VGroup(box, head), num, s

    def instructgpt(self):
        self.next_section("7 · InstructGPT", notes=(
            f"Ouyang et al. 2022. Labelers: 'a team of about {LABELERS} contractors on Upwork and through ScaleAI' (§3.4; "
            f"§1 says 40). Table 6 training prompts: SFT 11,295 labeler + 1,430 customer = {SFT_PROMPTS:,} (labelers wrote "
            f"the demonstrations); RM 6,623 + 26,584 = {RM_PROMPTS:,} (labelers ranked K = 4–9 sampled answers); PPO "
            f"{PPO_PROMPTS:,} customer prompts, no human labels. Reward model: 6B for every policy size (§3.5). β = {BETA} "
            f"(App. C.4). Compute (§5.1, 175B models): SFT {PF_SFT}, PPO-ptx {PF_PPO}, GPT-3 pretraining {PF_GPT3:,} "
            f"petaflop/s-days → {PF_PCT:.3f}% (said 'about one point six five percent'). Headline (abstract, exact): "
            f"'{QUOTE}' Appendix A: abstract '100x', intro 'over 100x', actual 175/1.3 = {SIZE_RATIO:.1f}× → 'over 100× "
            "smaller'. Squares drawn with AREA proportional to parameter count. Caveat (notes misconceptions): preferred by "
            "labelers on the API prompt distribution; GPT-3 still wins some public NLP benchmarks (the 'alignment tax'), "
            "bias did not improve. §1: 'These models have the same architecture, and differ only by the fact that "
            "InstructGPT is fine-tuned on our human data.'"))
        crew = m.VGroup(icon(USER, 0.48), text(f"about {LABELERS} labelers wrote and ranked all the human data", 22, INK))
        left_at(crew.arrange(m.RIGHT, buff=0.25), -6.3, 0.05)
        s1, n1, d1 = self.stage_card(-4.35, "1 · supervised fine-tuning", f"{SFT_PROMPTS:,} prompts", "labelers write answers", ACTION)
        s2, n2, d2 = self.stage_card(0.0, "2 · reward model", f"{RM_PROMPTS:,} prompts", f"labelers rank · a {RM_B}B model", REWARD)
        s3, n3, d3 = self.stage_card(4.35, "3 · PPO", f"{PPO_PROMPTS:,} prompts", f"no labels · β = {BETA}", SIGNAL)
        src = right_at(cite("Ouyang et al., 2022: Table 6, §3"), 6.5, 0.05)

        # compute, petaflop/s-days (175B models)
        cx0, cw = -2.2, 8.2
        c_title = left_at(text("compute for the 175B models, petaflop/s-days", 22, INK, weight=m.BOLD), -6.3, -0.85)
        c_src = right_at(cite("§5.1"), 6.5, -0.85)
        crow = []
        for k, (lab, v, col, val) in enumerate((("pretraining GPT-3", PF_GPT3, MUTED, f"{PF_GPT3:,}"),
                                                ("supervised fine-tuning", PF_SFT, ACTION, f"{PF_SFT}"),
                                                ("PPO-ptx stage", PF_PPO, SIGNAL, f"{PF_PPO} · {PF_PCT:.2f}%"))):
            y = -1.5 - 0.62 * k
            w = max(v / PF_GPT3 * cw, 0.04)
            bar = m.Rectangle(width=w, height=0.38).set_stroke(col, 2).set_fill(FILLS.get(col, CARD), 1).move_to([cx0 + w / 2, y, 0])
            name = right_at(code(lab, 20, col if col != MUTED else INK), cx0 - 0.25, y)
            num = code(val, 22, col if col != MUTED else INK, weight=m.BOLD)
            if k == 0:
                num.move_to(bar).align_to(bar, m.RIGHT).shift(0.25 * m.LEFT)
            else:
                num.next_to(bar, m.RIGHT, buff=0.2)
            crow.append(m.VGroup(name, bar, num))
        compute = m.VGroup(c_title, c_src, *crow)

        # the headline: squares with area ∝ parameters
        side = 2.3
        big = m.Square(side).set_stroke(MUTED, 2.5).set_fill(CARD_HI, 1).move_to([-4.85, -1.95, 0])
        big_lab = m.VGroup(text("GPT-3", 24, INK, weight=m.BOLD), code("175B", 22, MUTED)).arrange(m.DOWN, buff=0.1).move_to(big)
        small_side = side * math.sqrt(1.3 / 175)
        small = m.Square(small_side).set_stroke(ACTION, 2).set_fill(ACTION, 1)
        small.move_to([-3.05, big.get_bottom()[1] + small_side / 2, 0])
        small_lab = left_at(m.VGroup(text("InstructGPT", 24, ACTION, weight=m.BOLD), code("1.3B", 22, ACTION))
                            .arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT), -2.75, -2.6)
        preferred = left_at(pill("preferred", GREEN, GREEN_FILL, 22), -2.85, -1.55)
        area = left_at(cite("area ∝ parameters"), -6.0, -3.35)
        q_lines = m.VGroup(text("“outputs from the 1.3B parameter InstructGPT", 20, INK),
                           text("model are preferred to outputs from the 175B", 20, INK),
                           text("GPT-3, despite having 100x fewer parameters.”", 20, INK))
        q_lines.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        q_lines.move_to([0, -1.3, 0]).align_to([1.2, 0, 0], m.LEFT)
        q_src = left_at(cite("Ouyang et al., 2022, abstract"), 1.2, -2.05)
        ratio = left_at(colored([("175 ÷ 1.3 =", MUTED), (f"{SIZE_RATIO:.0f}× smaller", INK)], 22, weight=m.BOLD), 1.2, -2.5)
        caveat = m.VGroup(cite("judged by labelers, on prompts sent to the API;"), cite("GPT-3 still won some public benchmarks"))
        caveat.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(caveat, 1.2, -3.1)
        assert " ".join(t.text for t in q_lines).strip("“”") == QUOTE

        self.say(
            "[About forty hired labelers did the human part.] [First they wrote answers to about thirteen thousand prompts], "
            "[for supervised fine-tuning.]",
            m.AnimationGroup(self.new_header("07 · InstructGPT", "Three stages, about forty people"), m.FadeIn(crew, shift=0.1 * m.UP),
                             m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(s1, shift=0.1 * m.UP), m.FadeIn(n1, shift=0.1 * m.UP)),
            m.FadeIn(d1, shift=0.1 * m.UP),
        )
        self.say(
            "[Then they ranked answers to thirty-three thousand prompts], [to train a reward model with six billion parameters.]",
            m.AnimationGroup(m.FadeIn(s2, shift=0.1 * m.UP), m.FadeIn(n2, shift=0.1 * m.UP)),
            m.FadeIn(d2, shift=0.1 * m.UP),
        )
        self.say(
            "[Finally, PPO ran on thirty-one thousand more prompts], [with no human labels at all.]",
            m.AnimationGroup(m.FadeIn(s3, shift=0.1 * m.UP), m.FadeIn(n3, shift=0.1 * m.UP)),
            m.FadeIn(d3, shift=0.1 * m.UP),
        )
        self.say(
            "[Reinforcement learning on the largest model cost sixty petaflop-days], "
            "[against three thousand six hundred forty to pretrain GPT-3]: [about one point six five percent.]",
            m.AnimationGroup(m.FadeIn(c_title), m.FadeIn(c_src), m.FadeIn(crow[2][0]), m.GrowFromEdge(crow[2][1], m.LEFT),
                             m.FadeIn(crow[1][0]), m.GrowFromEdge(crow[1][1], m.LEFT), m.FadeIn(crow[1][2])),
            m.AnimationGroup(m.FadeIn(crow[0][0]), m.GrowFromEdge(crow[0][1], m.LEFT, run_time=1.6), m.FadeIn(crow[0][2])),
            m.FadeIn(crow[2][2], shift=0.1 * m.LEFT),
        )
        self.say(
            "[And the headline]: [people preferred the answers of the one point three billion parameter InstructGPT] "
            "[to those of the one hundred seventy-five billion parameter GPT-3.]",
            m.AnimationGroup(m.FadeOut(compute), m.FadeIn(q_lines, shift=0.1 * m.UP), m.FadeIn(q_src)),
            m.AnimationGroup(m.FadeIn(small, scale=0.3), m.FadeIn(small_lab), m.FadeIn(preferred, scale=0.8)),
            m.AnimationGroup(m.GrowFromEdge(big, m.DOWN), m.FadeIn(big_lab), m.FadeIn(area)),
        )
        self.say(
            "[That's a model over a hundred times smaller], [with the same design, fine-tuned on human data.]",
            m.AnimationGroup(m.FadeIn(ratio, shift=0.1 * m.UP), m.Circumscribe(small, color=ACTION, buff=0.08)),
            m.FadeIn(caveat, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · the catch

    def info_card(self, cx, name, title, lines, color):
        box = card(4.1, 4.3, color, CARD, radius=0.2, stroke=2.5).move_to([cx, -0.45, 0])
        top, left = box.get_top()[1], box.get_left()[0] + 0.3
        ic = icon(name, 0.55).move_to([left + 0.28, top - 0.48, 0])
        head = text(title, 24, INK, weight=m.BOLD).next_to(ic, m.RIGHT, buff=0.2)
        body = m.VGroup(*lines).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        assert body.width < box.width - 0.5, title
        left_at(body, left, 0).align_to([0, top - 1.0, 0], m.UP)
        return m.VGroup(box, ic, head), body

    def the_catch(self):
        self.next_section("8 · Costs and failure modes", notes=(
            f"Labels: about {LABELERS} hired labelers (Ouyang et al. 2022 §3.4); 'training labelers agree with each-other "
            f"{AGREE} ± {AGREE_SE}% of the time, while for held-out labelers this number is 77.3 ± 1.3%'. The reward model "
            "can be fooled: Stiennon et al. 2020 §4.3, 'Under light optimization, the models improve (according to "
            "labelers). However, as we optimize further, true preferences fall off compared to the prediction, and "
            "eventually the reward model becomes anti-correlated with human preferences.' → Episode 25 (reward hacking). "
            f"Our run (lm_rl.json): {MODEL}, GRPO on GSM8K, reward = the checker ('{measured(LM, 'setup', 'reward')}'): no "
            "labelers, no reward model → Episode 24 (rewards you can check)."))
        c1, b1 = self.info_card(-4.5, USER, "Labels cost", [
            text("people, not GPUs:", 22, INK),
            text(f"about {LABELERS} hired labelers,", 22, INK),
            text("who agreed with each", 22, INK),
            text("other only", 22, INK),
            code(f"{AGREE} ± {AGREE_SE}%", 24, INK, weight=m.BOLD),
            text("of the time", 22, INK),
            cite("Ouyang et al., 2022, §3.4"),
        ], MUTED)
        c2, b2 = self.info_card(0.0, "alert", "A judge to fool", [
            text("a reward model is only", 22, INK),
            text("a model; optimise hard,", 22, INK),
            text("“eventually the reward model", 20, MUTED),
            text("becomes anti-correlated", 20, MUTED),
            text("with human preferences”", 20, MUTED),
            cite("Stiennon et al., 2020, §4.3"),
            pill("episode 25 · hacking", PINK, PINK_FILL, 20),
        ], RED)
        c3, b3 = self.info_card(4.5, "check", "A checker instead", [
            code(MODEL, 20, ACTION, weight=m.BOLD),
            text("on grade-school maths", 22, INK),
            text("reward 1 if the boxed", 22, INK),
            text("number is right, else 0", 22, INK),
            text("no labels, no reward model", 22, INK),
            cite("trained on this Mac"),
            pill("episode 24 · checkers", PINK, PINK_FILL, 20),
        ], GREEN)

        self.say(
            "[The costs? Labels are slow and expensive], "
            "[and labelers agreed with each other only about seventy-three percent of the time.]",
            m.AnimationGroup(self.new_header("08 · the catch", "Costs, and a judge that can be fooled"),
                             m.FadeIn(c1, shift=0.1 * m.UP), m.FadeIn(b1[0:2], shift=0.1 * m.UP)),
            m.FadeIn(b1[2:], shift=0.1 * m.UP),
        )
        self.say(
            "[And a reward model is only a model]: "
            "[push too hard, and the policy finds answers it overrates. That's episode twenty-five.]",
            m.AnimationGroup(m.FadeIn(c2, shift=0.1 * m.UP), m.FadeIn(b2[0:2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(b2[2:6], shift=0.1 * m.UP), m.Succession(m.Wait(1.6), m.FadeIn(b2[6], scale=0.8))),
        )
        self.say(
            "[Our run on this Mac skipped people and reward model alike]: "
            "[a checker compares the boxed number, as episode twenty-four will show.]",
            m.AnimationGroup(m.FadeIn(c3, shift=0.1 * m.UP), m.FadeIn(b3[0:2], shift=0.1 * m.UP), m.FadeIn(b3[5])),
            m.AnimationGroup(m.FadeIn(b3[2:5], shift=0.1 * m.UP), m.Succession(m.Wait(1.4), m.FadeIn(b3[6], scale=0.8))),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            "Compare (people pick one of two answers: preference pairs), score (a reward model fitted with Bradley–Terry, "
            "σ(r_A − r_B)), optimise (PPO against the reward model, minus β·KL to the frozen reference; four models in "
            f"memory). InstructGPT: the 1.3B model preferred over the 175B GPT-3 ({SIZE_RATIO:.0f}× larger). The short "
            "version: 'AI, Explained' Episode 30 (RLHF and DPO). Next: Episode 22, Direct Preference Optimization (no "
            "reward model, no RL loop)."))
        halo = glow(GREEN, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Compare", 58, GREEN), title_text("Score", 58, REWARD), title_text("Optimise", 58, SIGNAL))
        words.arrange(m.RIGHT, buff=0.9).move_to([0, 1.0, 0])
        assert words.width < 11.8
        on_baseline(words)
        subs = m.VGroup(text("two answers, one pick", 22, MUTED), text("a reward model learns", 22, MUTED),
                        text("PPO, on a KL leash", 22, MUTED))
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), words[0].get_bottom()[1] - 0.5, 0])
        on_baseline(subs)
        assert all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        proof = m.VGroup(text("InstructGPT 1.3B", 26, ACTION, weight=m.BOLD), text("preferred over", 26, GREEN),
                         text("GPT-3 175B", 26, MUTED, weight=m.BOLD)).arrange(m.RIGHT, buff=0.28).move_to([0, -1.2, 0])
        on_baseline(proof)
        model = icon(POLICY_ICON, 0.62).move_to([-2.95, -2.35, 0])
        nxt = pill("episode 22 · skip the reward model", PINK, PINK_FILL, 22).next_to(model, m.RIGHT, buff=0.4)
        short = cite("the short version: AI, Explained · episode 30").move_to([0, -3.25, 0])

        self.say(
            "[To sum up]: [people compare two answers], [a reward model learns to score them], "
            "[and PPO optimises against it, on a leash.]",
            m.AnimationGroup(self.new_header("09 · recap", "Compare, score, optimise"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[That's how a model over a hundred times smaller won people over.] "
            "[For the short version, see episode thirty of AI, Explained.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(model, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.1 * m.LEFT), m.FadeIn(short)),
        )
        self.wait(0.3)
        self.clear()
