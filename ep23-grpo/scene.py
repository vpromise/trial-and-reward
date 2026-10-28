"""Trial and Reward · Episode 23 · GRPO (Module 6 · Teaching Language Models with Reward)

Compare a group of answers to each other instead of training a critic. PPO for language models (Episodes 18, 21) gets
its advantages from a critic, a value network as large as the policy (four models in memory, Episode 21); GRPO (Shao
et al. 2024, DeepSeekMath) drops the critic (NOT the reward model) and takes the baseline from a group of G answers to
the same question: advantage = (r − group mean) / (group std + ε), the same for every token of an answer, then
Episode 18's clipped ratio plus a k3 KL term to the frozen reference (Episode 20), averaged over each answer's tokens
and then over the group. The running example is OUR GRPO RUN on this Mac (research/experiments/lm_rl.md, lm_rl.json):
one real group (step 4, GSM8K training problem 3690, Sandy's phone bill: 4 of 8 right → advantages ±1 with the
population std and ε = 1e-4; the sample std of TRL and verl would give ±0.935), the late group (step 191, 5 of 8
right → +0.77 / −1.29), the share of all-equal groups (31 % in the first ten steps, 35 % in the last ten), the loss
in ten lines of the run's code, and one real update (step 100: ratios exactly 1, then 0.68–1.54; 24 of 9,703 tokens
outside the band, 5 clipped). DeepSeekMath's own numbers (G = 64, β = 0.04, GSM8K 82.9 % → 88.2 %), the memory saved
(derived), and the 2025 follow-ups Dr. GRPO and DAPO in one line each. The accuracy results of our run are kept for
Episodes 24 and 26.

Every number on screen is computed below from rl_reference.py and asserted, or loaded with measured(...), or cited.

Sources (research/notes.md §23, §18, §21; Appendix A rows 23; Appendix C row 23; facts dated October 2026):
- research/experiments/lm_rl.md + lm_rl.json (this Mac Studio, Apple M4 Max, 64 GB, MLX; 2026-10-25; one run, one
  seed): setup (prompt, checker regexes, advantage = population std + 1e-4, loss, k3), groups.early / groups.late
  (verbatim answers, rewards, advantages, token counts), curves (zero_adv_groups per step, peak_mem_gb, clip_frac),
  headline (zero_adv_groups_first10 0.3125, zero_adv_groups_last10 0.35, clip_frac_mean 0.0005, G, β, ε), update
  (step 100, two minibatches: ratio stats, histogram, clip fraction). The loss excerpt is lm_rl.md's on-screen version
  of research/experiments/lm/grpo.py (comments shortened; every code line is asserted against both files below).
- Shao et al. 2024, DeepSeekMath (arXiv 2402.03300): GRPO, eq. 3 (two averages, KL inside), eq. 4 (k3); §4.2 "we sample
  64 outputs", "The KL coefficient is 0.04", a learned reward model; Table 5 GSM8K 82.9 → 88.2 (DeepSeekMath-Instruct
  7B → DeepSeekMath-RL 7B, chain of thought); "GRPO foregoes the critic model, instead estimating the baseline from
  group scores". Notes §23 "Implementations": TRL GRPOTrainer (nanstd, Bessel's correction, eps 1e-4) and verl
  (torch.std, eps 1e-6) use the SAMPLE std. rl_reference.grpo_advantages (ddof=1 default; ddof=0 = population).
- Liu et al. 2025, Dr. GRPO (arXiv 2503.20783): length bias from 1/|o_i|, difficulty bias from ÷ std; drops both, β = 0.
  Yu et al. 2025, DAPO (arXiv 2503.14476): Clip-Higher (ε_low 0.2, ε_high 0.28), Dynamic Sampling (drop groups whose
  accuracy is 0 or 1), Token-Level Policy Gradient Loss, Overlong Reward Shaping; no KL term.
- Memory [DERIVED, notes §23 worked example, "illustrative memory accounting, not from the paper"]: 7B models, 16 bytes
  per trained parameter (mixed-precision Adam, ZeRO: Rajbhandari et al. 2020), 2 per frozen BF16 one: PPO 112 + 112 +
  14 + 14 = 252 GB; GRPO with a reward model 140 GB (−44 %); model states only. Episode 21 showed the four models.
- Agents, Animated, Episode 30: coding agents trained with group-relative advantages (sixteen rollouts per task).

Illustrative (said "for example" or labelled on screen): the all-right group of section 4 (eight rewards of 1); the
critic drawn as a small network; the eight chips standing for "a group of answers" in section 1. The answers' text is
real but cut to one short verbatim phrase each (LaTeX markup \\( \\) and \\text{} removed, \\times shown as ×).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep23"
LM = "lm_rl"

# ---------------------------------------------------------------------------- the numbers (all checked here)

H = measured(LM, "headline")
SETUP = measured(LM, "setup")
CURVES = measured(LM, "curves")
EARLY = measured(LM, "groups", "early")
LATE = measured(LM, "groups", "late")
UPD = measured(LM, "update")

G = H["G"]
assert G == 8 and H["beta_kl"] == 0.04 and H["clip_eps"] == 0.2 and H["steps"] == 200
CLIP = H["clip_eps"]
BETA = H["beta_kl"]

# the checker: the run's own regexes (lm_rl.md, "Reward (the checker)")
BOXED = re.compile(SETUP["boxed_regex"])
NUMBER = re.compile(SETUP["number_regex"])


def checker(answer, reference):
    """1 if the last number inside the last \\boxed{...} equals the reference, else 0."""
    boxes = BOXED.findall(answer)
    if not boxes:
        return 0.0
    numbers = NUMBER.findall(boxes[-1])
    if not numbers:
        return 0.0
    return float(abs(float(numbers[-1].replace(",", "")) - reference) < 1e-4)


# the early group (step 4): 4 of 8 right → ±1 with the population std and ε = 1e-4
EPS_ADV = 1e-4
assert "std_population" in SETUP["advantage"] and "1e-4" in SETUP["advantage"]
assert EARLY["step"] == 4 and EARLY["id"] == "train-3690" and EARLY["ref"] == 340
R_EARLY = np.array(EARLY["rewards"], dtype=float)
assert list(R_EARLY) == [1, 0, 0, 1, 0, 1, 0, 1] and R_EARLY.sum() == 4             # "four answers ... reward one"
assert [checker(c, EARLY["ref"]) for c in EARLY["completions"]] == list(R_EARLY)   # the checker agrees with the run
A_EARLY = grpo_advantages(R_EARLY, eps=EPS_ADV, ddof=0)
assert np.allclose(A_EARLY, EARLY["advantages"], atol=1e-4)
assert np.allclose(np.abs(A_EARLY), 1.0, atol=3e-4)                                 # "plus one ... minus one"
MEAN_E, STD_E = float(R_EARLY.mean()), float(R_EARLY.std())
assert MEAN_E == 0.5 and STD_E == 0.5                                               # "zero point five ... too"
STD_SAMPLE = float(R_EARLY.std(ddof=1))
A_SAMPLE = grpo_advantages(R_EARLY)                                                 # ddof=1, as TRL and verl
assert round(STD_SAMPLE, 3) == 0.535 and np.allclose(np.abs(A_SAMPLE), abs(A_SAMPLE[0]))
assert round(abs(A_SAMPLE[0]), 3) == 0.935 == round(abs(grpo_advantages(R_EARLY, eps=1e-4)[0]), 3)  # "0.935"
BOX_E = [BOXED.findall(c)[-1] for c in EARLY["completions"]]
assert BOX_E == ["340", "10S", "120", "340", "200", "340", "100", "340"]           # "a formula, or a wrong number"
TOK_E = EARLY["tokens"]
assert (min(TOK_E), max(TOK_E)) == (145, 305)
Q_LINES = ["Sandy's monthly phone bill expense is equal to ten times her age now.",
           "In two years, Sandy will be three times as old as Kim.",
           "If Kim is currently 10 years old, calculate Sandy's monthly phone bill expense."]
assert " ".join(Q_LINES) == EARLY["question"]
assert (10 + 2) * 3 - 2 == 34 and 34 * 10 == EARLY["ref"]                           # Kim 12 → Sandy 36 → 34 → 340


def plain(s):
    """The answer text without LaTeX markup, as drawn (\\( \\) dropped, \\times as ×, \\text{} unwrapped)."""
    s = s.replace("\\( ", "").replace(" \\)", "").replace("\\times", "×")
    return re.sub(r"\\text\{([^{}]*)\}", r"\1", s)


SNIPS = ["Now that we know Sandy is currently 34 years old",
         "the problem does not provide Sandy's current age",
         "so Sandy will be 12 years old in two years",
         "Substituting S = 34 into this expression",
         "and she is currently 20 years old",
         "So, Sandy is currently 34 years old.",
         "10 × Kim's age = 10 × 10 = 100",
         "Now that we know Sandy's current age is 34"]
assert all(s in plain(c) for s, c in zip(SNIPS, EARLY["completions"]))              # verbatim, markup removed

# the late group (step 191): 5 of 8 right → +0.77 / −1.29
assert LATE["step"] == 191 and LATE["ref"] == 48
R_LATE = np.array(LATE["rewards"], dtype=float)
assert list(R_LATE) == [1, 0, 1, 1, 1, 0, 1, 0]
assert [checker(c, LATE["ref"]) for c in LATE["completions"]] == list(R_LATE)
A_LATE = grpo_advantages(R_LATE, eps=EPS_ADV, ddof=0)
assert np.allclose(A_LATE, LATE["advantages"], atol=1e-4)
assert f"{A_LATE.max():+.2f}" == "+0.77" and f"{A_LATE.min():+.2f}" == "-1.29"      # said "seven seven", "two nine"
assert LATE["question"].startswith("Hannah has three brothers.")

# for example: a group that all agrees
R_ALL = np.ones(G)
A_ALL = grpo_advantages(R_ALL, eps=EPS_ADV, ddof=0)
assert np.all(A_ALL == 0.0)                                                         # "so is every advantage"

# how often a whole group agreed (curves.zero_adv_groups: fraction of the 8 groups per step)
Z = np.array(CURVES["zero_adv_groups"], dtype=float)
assert len(Z) == 200
Z_FIRST, Z_LAST = float(Z[:10].mean()), float(Z[-10:].mean())
assert Z_FIRST == H["zero_adv_groups_first10"] == 0.3125 and Z_LAST == H["zero_adv_groups_last10"] == 0.35
N10 = 10 * H["questions_per_step"]
assert N10 == 80 and round(Z_FIRST * N10) == 25 and round(Z_LAST * N10) == 28
assert round(100 * Z_FIRST) == 31 and round(100 * Z_LAST) == 35                     # "thirty-one ... thirty-five"
assert abs(Z.mean() - 1 / 3) < 0.01                                                 # "about a third"

# one update (step 100): two minibatches
MB1, MB2 = UPD["minibatches"]
assert UPD["step"] == 100
EDGES = MB2["ratio_hist"]["edges"]
C1, C2 = MB1["ratio_hist"]["counts"], MB2["ratio_hist"]["counts"]
assert EDGES == MB1["ratio_hist"]["edges"] and len(EDGES) == 12 and len(C2) == 11
N1, N2 = sum(C1), sum(C2)
assert (N1, N2) == (11_690, 9_703) and C1[5] == N1                                 # all of minibatch 1 at 0.99–1.01
assert MB1["ratio_min"] == MB1["ratio_max"] == 1.0 and MB1["clip_frac"] == 0.0     # "exactly one"
assert f"{MB2['ratio_min']:.2f}" == "0.68" and f"{MB2['ratio_max']:.2f}" == "1.54"
OUTSIDE = C2[0] + C2[1] + C2[9] + C2[10]                                           # ratio < 0.8 or ≥ 1.2
assert (EDGES[2], EDGES[9]) == (1 - CLIP, 1 + CLIP) and OUTSIDE == 24
assert abs(OUTSIDE / N2 - MB2["outside_band"]) < 1e-9 and f"{100 * OUTSIDE / N2:.2f}" == "0.25"   # "a quarter of a percent"
CLIPPED = MB2["clip_frac"] * N2
assert abs(CLIPPED - 5) < 1e-6                                                      # "just five of them"
CLIPPED = 5
assert f"{100 * H['clip_frac_mean']:.2f}" == "0.05" and abs(np.mean(CURVES["clip_frac"]) - H["clip_frac_mean"]) < 1e-4

# the loss, as the run's code shows it (lm_rl.md's on-screen excerpt of lm/grpo.py; comments shortened)
LOSS = """a = (r - r.mean()) / (r.std() + 1e-4)    # population std

def grpo_loss(model, tokens, mask, old_lp, ref_lp, adv, n_seq):
    lp = token_logprobs(model, tokens)       # log pi_theta, per token
    ratio = mx.exp(lp - old_lp)              # pi_theta / pi_old
    A = adv[:, None]
    surr = mx.minimum(ratio * A, mx.clip(ratio, 1 - eps, 1 + eps) * A)
    kl = mx.exp(ref_lp - lp) - (ref_lp - lp) - 1       # k3, to pi_ref
    per_token = surr - beta * kl
    per_seq = (per_token * mask).sum(1) / mask.sum(1)  # 1/|o_i| sum_t
    return -per_seq.sum() / n_seq                      # 1/G sum_i"""
EXPERIMENTS = Path(__file__).resolve().parent.parent / "research" / "experiments"
_MD = (EXPERIMENTS / "lm_rl.md").read_text()
_PY = (EXPERIMENTS / "lm" / "grpo.py").read_text().replace("args.", "").replace("C.token_logprobs", "token_logprobs")
LOSS_LINES = [ln.split("#")[0].rstrip() for ln in LOSS.splitlines() if ln.strip()]
assert len(LOSS_LINES) == 10 and len(LOSS.splitlines()) <= 12                      # "ten lines"
assert all(ln.strip() in _MD for ln in LOSS_LINES)
assert all(ln.strip() in _PY for ln in LOSS_LINES[1:-1])                            # and in the run's own trainer
assert "per_seq.sum() / n_seq" in _PY and "/ (mx.sqrt(((r - r.mean()) ** 2).mean()) + 1e-4)   # population std" in _PY

# DeepSeekMath (Shao et al. 2024; notes §23) — cited
DSM_G, DSM_BETA = 64, 0.04
DSM_BEFORE, DSM_AFTER = 82.9, 88.2                                                 # GSM8K, Table 5, 7B
assert DSM_BETA == BETA and DSM_AFTER > DSM_BEFORE

# memory [DERIVED, notes §23 worked example]: 7B models, 16 bytes per trained parameter, 2 per frozen one
P7 = 7e9
TRAINED, FROZEN = 16, 2
MEM_PPO = {"policy": P7 * TRAINED / 1e9, "critic": P7 * TRAINED / 1e9, "reward": P7 * FROZEN / 1e9,
           "reference": P7 * FROZEN / 1e9}
MEM_GRPO = {k: v for k, v in MEM_PPO.items() if k != "critic"}
PPO_GB, GRPO_GB = sum(MEM_PPO.values()), sum(MEM_GRPO.values())
assert [round(v) for v in MEM_PPO.values()] == [112, 112, 14, 14] and (round(PPO_GB), round(GRPO_GB)) == (252, 140)
SAVED = round(100 * (1 - GRPO_GB / PPO_GB))
assert SAVED == 44                                                                  # "about 250 ... a hundred forty"
OUR_MEM = float(np.median(CURVES["peak_mem_gb"]))
assert round(OUR_MEM) == 25 and H["params_base_M"] == 494.0                        # "≈ 25 GB", measured

# ---------------------------------------------------------------------------- look

LLM_POL = recolored("llm", ACTION, CORAL_FILL, prefix=PREFIX)
LLM_RM = recolored("llm", REWARD, GOLD_FILL, prefix=PREFIX)
LLM_REF = "llm"                                         # the course's llm icon is VIOLET: the leash's colour
CHECK = recolored("check", REWARD, GOLD_FILL, prefix=PREFIX)
WRONG_DIM = 0.4


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def left_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.RIGHT)


def formula(parts, colors, size=40):
    f = m.MathTex(*parts, font_size=size, color=INK)
    for part, col in zip(f, colors):
        part.set_color(col)
    return f


def val_chip(label, color, fill, w=None, h=0.42, size=20):
    """A value in a small rounded pill (mono): rewards GOLD, advantages PINK, answer numbers CORAL."""
    words = code(label, size, color, weight=m.BOLD)
    box = card(w or words.width + 0.34, h, color, fill, radius=h / 2, stroke=2)
    return m.VGroup(box, words.move_to(box))


def reward_pill(r, w=0.62):
    return val_chip("+1" if r > 0 else "0", REWARD if r > 0 else MUTED, GOLD_FILL if r > 0 else CARD, w=w)


def adv_pill(a, w=1.08):
    if abs(a) < 1e-9:
        return val_chip("0.00", MUTED, CARD, w=w)
    return val_chip(minus(f"{a:+.2f}"), SIGNAL, PINK_FILL, w=w)


def push_arrow(x, y, up, length=0.36, color=SIGNAL):
    a, b = ([x, y - length / 2, 0], [x, y + length / 2, 0]) if up else ([x, y + length / 2, 0], [x, y - length / 2, 0])
    return m.Arrow(a, b, buff=0, color=color, stroke_width=6, max_tip_length_to_length_ratio=0.4)


def model_card(art, title, sub, color, w=2.95, h=2.45):
    """A model as a card: an icon (or a drawing) over its name and a MUTED line."""
    box = card(w, h, color, CARD)
    words = m.VGroup(text(title, 22, INK, weight=m.BOLD), cite(sub)).arrange(m.DOWN, buff=0.1)
    art.move_to(box.get_center() + [0, 0.42, 0])
    words.move_to(box.get_center() + [0, -0.62, 0])
    return m.VGroup(box, art, words)


# ---------------------------------------------------------------------------- the episode


class Ep23GRPO(Episode):
    number = 23

    def lesson(self):
        self.the_critic()
        self.eight_answers()
        self.compare()
        self.agreement()
        self.the_update()
        self.one_update()
        self.deepseekmath()
        self.what_next()
        self.recap()

    # ------------------------------------------------------------------ 1 · the price of a critic

    def the_critic(self):
        self.next_section("1 · The price of a critic", notes=(
            "PPO (Episode 18) learns from advantages: how much better an action did than expected. For a language model "
            "the expected part comes from a critic, a value network 'typically another model of comparable size as the "
            "policy model' (Shao et al. 2024, §4.1.1), trained from a reward that only the last token receives. With the "
            "reward model and the frozen reference that is Episode 21's four models in memory (Hu et al. 2024: 'actor, "
            "critic, reward, reference'). GRPO (Shao et al. 2024, DeepSeekMath, arXiv 2402.03300): 'GRPO foregoes the "
            "critic model, instead estimating the baseline from group scores, significantly reducing training "
            "resources.' It removes the critic, NOT the reward model: DeepSeekMath trained against a learned reward model "
            "(our run uses a checker, Episode 24). The critic is drawn as a small network and the group as eight chips "
            "(illustration)."))
        xs = (-4.95, -1.65, 1.65, 4.95)
        Y = 0.95
        net = NeuralNet(layers=(3, 5, 5, 1), width=1.8, height=0.9, color=VALUE, radius=0.075)
        pol = model_card(icon(LLM_POL, 0.85), "policy", "writes the answers", ACTION).move_to([xs[0], Y, 0])
        critic = model_card(net, "critic", "predicts the value", VALUE).move_to([xs[1], Y, 0])
        rm = model_card(icon(LLM_RM, 0.85), "reward model", "scores the answers", REWARD).move_to([xs[2], Y, 0])
        ref = model_card(icon(LLM_REF, 0.85), "reference", "anchors the leash", EXPLORE).move_to([xs[3], Y, 0])
        c_box, c_net, c_words = critic
        size_t = cite("as big as the policy").move_to([xs[1], -0.6, 0])

        prefix = text("advantage = reward −", 32, INK, weight=m.BOLD, t2c={"advantage": SIGNAL, "reward": REWARD})
        expected = text("expected", 32, VALUE, weight=m.BOLD)
        average = text("the group's average", 32, REWARD, weight=m.BOLD)
        f1 = on_baseline(m.VGroup(prefix, expected).arrange(m.RIGHT, buff=0.15)).move_to([0, -1.35, 0])
        f2 = on_baseline(m.VGroup(prefix.copy(), average).arrange(m.RIGHT, buff=0.15)).move_to([0, -1.35, 0])
        on_baseline(m.VGroup(f1[0], f2[0]))
        on_baseline(m.VGroup(f1[0], f2[1]))
        assert f2.width < 12.6
        ep18 = cite("episode 18 · PPO").move_to([xs[0], -0.6, 0])
        ep21 = pill("episode 21 · four models in memory", MUTED, CARD, 22).move_to([0, -2.4, 0])

        # the group that replaces the critic
        chips = m.VGroup(*[m.RoundedRectangle(corner_radius=0.07, width=0.5, height=0.3, stroke_color=ACTION,
                                              stroke_width=2, fill_color=CORAL_FILL, fill_opacity=1) for _ in range(G)])
        chips.arrange_in_grid(rows=2, cols=4, buff=(0.12, 0.14)).move_to([xs[1], Y + 0.42, 0])
        g_words = m.VGroup(text("the group", 22, INK, weight=m.BOLD), cite("8 answers"))
        g_words.arrange(m.DOWN, buff=0.1).move_to(c_words)
        g_box = card(2.95, 2.45, ACTION, CARD).move_to(c_box)
        name = pill("GRPO · Group Relative Policy Optimization", SIGNAL, PINK_FILL, 24).move_to([0, -2.4, 0])
        src = cite("Shao et al., 2024 · DeepSeekMath").move_to([0, -3.2, 0])
        gone = text("no critic", 22, RED, weight=m.BOLD).move_to([xs[1], -0.6, 0])
        kept = text("kept: DeepSeekMath trained one", 20, REWARD).move_to([xs[2] + 0.2, -0.6, 0])
        assert kept.get_left()[0] > gone.get_right()[0] + 0.2 and kept.get_right()[0] < 6.5

        self.say(
            "[Episode eighteen's PPO learns from advantages]: [how much better an action did than expected.]",
            m.AnimationGroup(self.new_header("01 · the critic", "The price of a critic"), m.FadeIn(pol, shift=0.15 * m.UP),
                             m.FadeIn(ep18)),
            m.FadeIn(f1, shift=0.1 * m.UP),
        )
        self.say(
            "[For a language model, the expected part comes from a critic]: [a value network as big as the policy], "
            "[trained from one reward at the end of each answer.]",
            m.AnimationGroup(m.FadeIn(c_box), m.FadeIn(c_net), m.FadeIn(c_words),
                             m.Circumscribe(expected, color=VALUE, buff=0.08)),
            m.AnimationGroup(m.FadeIn(size_t), c_net.forward(VALUE, run_time=1.4)),
            m.Circumscribe(prefix, color=REWARD, buff=0.1),
        )
        self.say(
            "[Add a reward model and a frozen reference], [and that's episode twenty-one's four models in memory.]",
            m.LaggedStart(m.FadeIn(rm, shift=0.15 * m.UP), m.FadeIn(ref, shift=0.15 * m.UP), lag_ratio=0.4),
            m.FadeIn(ep21, shift=0.1 * m.UP),
        )
        self.say(
            "[In twenty twenty-four, the DeepSeekMath paper asked]: [why predict how good an answer is], "
            "[when you can sample a few and compare?]",
            m.FadeIn(src),
            m.Circumscribe(c_box, color=VALUE, buff=0.08),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeOut(e) for e in c_net.edges], lag_ratio=0.25, run_time=1.0),
                m.LaggedStart(*[m.FadeOut(n, scale=0.2) for col in c_net.columns for n in col], lag_ratio=0.06,
                              run_time=1.4),
                m.FadeOut(c_words), m.FadeOut(size_t), m.ReplacementTransform(c_box, g_box),
                m.Succession(m.Wait(0.8), m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in chips], lag_ratio=0.12)),
                m.Succession(m.Wait(0.8), m.FadeIn(g_words)),
                prefix.animate.move_to(f2[0].get_center()), m.FadeOut(expected, shift=0.3 * m.UP),
                m.Succession(m.Wait(0.5), m.FadeIn(average, shift=0.3 * m.UP)), run_time=2.4),
        )
        self.remove(c_net, expected)
        f1.remove(expected)
        self.say(
            "[That's Group Relative Policy Optimization, or GRPO.] [The critic is gone], [but the reward model stays.]",
            m.AnimationGroup(m.FadeOut(ep21), m.FadeIn(name, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(gone), m.Circumscribe(g_box, color=ACTION, buff=0.08)),
            m.AnimationGroup(m.FadeIn(kept), m.Circumscribe(rm[0], color=REWARD, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · one question, eight answers

    def eight_answers(self):
        self.next_section("2 · One question, eight answers", notes=(
            "A real group of our run (lm_rl.json groups.early): step 4 of 200, GSM8K training problem 3690, the run's "
            "prompt (system: '" + SETUP["prompt_system"] + "'). Eight answers sampled at temperature 1 from the current "
            f"policy, {min(TOK_E)}–{max(TOK_E)} tokens each ({TOK_E}); each row shows one short verbatim phrase (LaTeX "
            f"markup removed) and the last boxed answer {BOX_E}. Reference 340 (Kim 10 + 2 = 12, Sandy 36 in two years, "
            "34 now, 10 × 34 = 340). The checker (the run's regexes, re-run here and asserted equal to the run's rewards): "
            "1 if the last number in the last \\boxed{} equals the reference, else 0."))
        q_rows = m.VGroup(*[text(s, 23, INK) for s in Q_LINES]).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        q_box = card(q_rows.width + 0.9, q_rows.height + 0.36, EDGE, CARD, radius=0.14, stroke=1.5)
        stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=q_box.height - 0.16, stroke_width=0,
                                    fill_color=STATE, fill_opacity=1)
        stripe.move_to(q_box).align_to(q_box, m.LEFT).shift(0.1 * m.RIGHT)
        q_rows.move_to(q_box).align_to(q_box, m.LEFT).shift(0.45 * m.RIGHT)
        qcard = m.VGroup(q_box, stripe, q_rows).move_to([0, 0.7, 0])
        assert qcard.width < 12.8
        q_lab = text("the question: the state", 22, STATE, weight=m.BOLD)
        left_at(q_lab, q_box.get_left()[0], q_box.get_bottom()[1] - 0.4)
        SCALE = 0.87
        q_target = qcard.copy().scale(SCALE)
        q_target.move_to([0, 2.62 - q_target.height / 2, 0])
        assert 23 * SCALE >= 20

        ys = [0.58 - 0.49 * k for k in range(G)]
        X_IDX, X_SNIP, X_BOX, X_REW = -5.55, -5.0, 2.85, 5.35
        idx = m.VGroup(*[val_chip(str(k + 1), ACTION, CORAL_FILL, w=0.46).move_to([X_IDX, y, 0]) for k, y in enumerate(ys)])
        snips = m.VGroup(*[left_at(text("… " + s + " …", 20, INK), X_SNIP, y) for s, y in zip(SNIPS, ys)])
        boxes = m.VGroup(*[left_at(code("\\boxed{" + b + "}", 20, INK), X_BOX, y) for b, y in zip(BOX_E, ys)])
        rews = m.VGroup(*[reward_pill(r).move_to([X_REW, y, 0]) for r, y in zip(R_EARLY, ys)])
        assert max(s.get_right()[0] for s in snips) < X_BOX - 0.4 and q_target.get_bottom()[1] > 1.3
        g_lab = code(f"G = {G}", 20, ACTION, weight=m.BOLD).move_to([X_IDX, 1.0, 0])
        a_lab = left_at(cite(f"real answers, cut short · {min(TOK_E)} to {max(TOK_E)} tokens each"), X_SNIP, 1.0)
        b_lab = left_at(cite("boxed"), X_BOX, 1.0)
        r_lab = text("reward", 20, REWARD).move_to([X_REW, 1.0, 0])
        assert a_lab.get_right()[0] < b_lab.get_left()[0] - 0.3
        src = cite("GSM8K training problem 3690 · step 4 of 200 · trained on this Mac").move_to([0, -3.33, 0])
        right = [k for k in range(G) if R_EARLY[k] > 0]
        wrong = [k for k in range(G) if R_EARLY[k] == 0]

        self.say(
            "[Here's a real group from our run on this Mac], [early in training, at step four.]",
            m.AnimationGroup(self.new_header("02 · eight answers", "One question, eight answers"),
                             m.FadeIn(qcard, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(src), m.FadeIn(q_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[Sandy's phone bill is ten times her age.] [In two years, she'll be three times as old as Kim], [who is ten now.]",
            m.Circumscribe(q_rows[0], color=STATE, buff=0.06),
            m.Circumscribe(q_rows[1], color=STATE, buff=0.06),
            m.Circumscribe(q_rows[2], color=STATE, buff=0.06),
        )
        self.say(
            "[The model samples eight answers to this one question]: [that's the group.]",
            m.AnimationGroup(m.FadeOut(q_lab), qcard.animate.scale(SCALE).move_to(q_target.get_center()),
                             m.LaggedStart(*[m.FadeIn(m.VGroup(idx[k], snips[k], boxes[k]), shift=0.1 * m.DOWN)
                                             for k in range(G)], lag_ratio=0.15),
                             m.FadeIn(a_lab), run_time=2.2),
            m.AnimationGroup(m.FadeIn(g_lab), m.Circumscribe(idx, color=ACTION, buff=0.08)),
        )
        self.say(
            "[A checker reads each boxed number.] [Four answers box three hundred forty]: [reward one.]",
            m.AnimationGroup(m.FadeIn(b_lab), m.LaggedStart(*[m.Circumscribe(b, color=REWARD, buff=0.05) for b in boxes],
                                                            lag_ratio=0.12)),
            m.AnimationGroup(*[boxes[k].animate.set_color(GREEN) for k in right]),
            m.AnimationGroup(m.FadeIn(r_lab), *[m.FadeIn(rews[k], scale=0.6) for k in right]),
        )
        self.say(
            "[The other four box a formula, or a wrong number]: [reward zero.]",
            m.AnimationGroup(*[boxes[k].animate.set_color(RED) for k in wrong]),
            m.AnimationGroup(*[m.FadeIn(rews[k], scale=0.6) for k in wrong]),
        )
        self.wait(0.4)
        self.s2 = dict(idx=idx, snips=snips, boxes=boxes, rews=rews, qcard=qcard, labels=m.VGroup(g_lab, a_lab, b_lab, r_lab),
                       src=src, ys=ys, right=right, wrong=wrong)

    # ------------------------------------------------------------------ 3 · compare within the group

    def compare(self):
        self.next_section("3 · Better than the group?", notes=(
            "Group-relative advantage (Shao et al. 2024, §4.1.2, outcome supervision): A_i = (r_i − mean(r)) / std(r), "
            "the same for every token of answer i. Our run (lm_rl.json setup.advantage): population std + 1e-4. "
            f"Rewards {list(R_EARLY.astype(int))}: mean {MEAN_E}, population std {STD_E} → advantages "
            f"{A_EARLY[0]:+.4f} / {A_EARLY[1]:+.4f} (rl_reference.grpo_advantages(rewards, eps=1e-4, ddof=0), equal to the "
            f"run's logged advantages). TRL's GRPOTrainer (nanstd with Bessel's correction, eps 1e-4) and verl (torch.std, "
            f"eps 1e-6) use the SAMPLE std {STD_SAMPLE:.4f} → ±{abs(A_SAMPLE[0]):.4f} (notes §23, 'Implementations', "
            "source read 2026-10-25). The number line: rewards (GOLD) → minus the mean → divided by the std → advantages "
            "(PINK). The arrows show only the direction of the update for each answer (every token of a right answer "
            "becomes likelier, of a wrong one less likely). The baseline is the group's mean: no critic network."))
        s2 = self.s2
        idx, boxes, rews, ys = s2["idx"], s2["boxes"], s2["rews"], s2["ys"]
        right, wrong = s2["right"], s2["wrong"]
        X_IDX, X_BOX, X_REW, X_ADV, X_ARR = -6.1, -5.62, -3.45, -2.05, -1.2
        title = left_at(text("step 4 · Sandy's phone bill · 4 of 8 right", 22, INK, weight=m.BOLD), -6.4, 1.5)
        b_lab = left_at(cite("boxed"), X_BOX, 1.0)
        r_lab = text("reward", 20, REWARD).move_to([X_REW, 1.0, 0])
        a_lab = text("advantage", 20, SIGNAL).move_to([X_ADV, 1.0, 0])
        assert r_lab.get_right()[0] < a_lab.get_left()[0] - 0.3
        advs = m.VGroup(*[adv_pill(a).move_to([X_ADV, y, 0]) for a, y in zip(A_EARLY, ys)])
        ups = m.VGroup(*[push_arrow(X_ARR, ys[k], True) for k in right])
        downs = m.VGroup(*[push_arrow(X_ARR, ys[k], False) for k in wrong])

        # the number line (right half): x = NC + v · NS
        NC, NS, NY = 2.85, 2.4, 0.5

        def nx(v):
            return NC + v * NS

        axis = m.Line([nx(-1.35), NY, 0], [nx(1.35), NY, 0]).set_stroke(FAINT, 3)
        ticks = m.VGroup()
        for v, lab in ((-1, "−1"), (0, "0"), (1, "+1")):
            ticks.add(m.Line([nx(v), NY - 0.08, 0], [nx(v), NY + 0.08, 0]).set_stroke(FAINT, 3),
                      code(lab, 20, MUTED).move_to([nx(v), NY - 0.32, 0]))
        assert nx(-1.35) > X_ARR + 0.5 and nx(1.35) < 6.3

        def stack(v, j):
            return np.array([nx(v), NY + 0.24 + 0.26 * j, 0])

        order = {k: (j, R_EARLY[k]) for j, k in enumerate(right)}
        order.update({k: (j, R_EARLY[k]) for j, k in enumerate(wrong)})
        dots = m.VGroup(*[m.Dot(stack(order[k][1], order[k][0]), radius=0.1, color=REWARD) for k in range(G)])
        dots.set_z_index(3)
        mean_line = m.DashedLine([nx(MEAN_E), NY - 0.15, 0], [nx(MEAN_E), NY + 1.3, 0], dash_length=0.08).set_stroke(INK, 2.5)
        mean_lab = code(f"mean {MEAN_E:.1f}", 20, INK, weight=m.BOLD).move_to([nx(MEAN_E), NY + 1.55, 0])
        mean_lab0 = code("mean 0", 20, INK, weight=m.BOLD).move_to([nx(0), NY + 1.55, 0])
        SY = NY - 0.75

        def spread(a, b, label):
            arr = m.DoubleArrow([nx(a), SY, 0], [nx(b), SY, 0], buff=0, color=INK, stroke_width=3,
                                max_tip_length_to_length_ratio=0.15, tip_length=0.14)
            return m.VGroup(arr, code(label, 20, INK).next_to(arr, m.DOWN, buff=0.1))

        std0 = spread(MEAN_E, MEAN_E + STD_E, f"std {STD_E:.1f}")
        std1 = spread(0, STD_E, f"std {STD_E:.1f}")
        std2 = spread(0, 1, "std 1")
        f_adv = formula([r"A_i", "=", r"\big(", r"r_i", "-", r"\mathrm{mean}(r)", r"\big)", r"\,/\,", r"\big(",
                         r"\mathrm{std}(r)", "+", r"\epsilon", r"\big)"],
                        [SIGNAL, INK, INK, REWARD, INK, REWARD, INK, INK, INK, INK, INK, INK, INK], 38)
        f_adv.move_to([NC, -1.25, 0])
        assert f_adv.get_left()[0] > X_ARR + 0.3 and f_adv.get_right()[0] < 6.4
        row_a = left_at(m.VGroup(cite("our run · population std, ε = 0.0001"),
                                 code(f"{STD_E:.3f} → ±{abs(A_EARLY[0]):.3f}", 20, SIGNAL, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25),
                        -0.3, -2.1)
        row_b0 = cite("TRL, verl · sample std")
        row_b1 = code(f"{STD_SAMPLE:.3f} → ±{abs(A_SAMPLE[0]):.3f}", 20, SIGNAL, weight=m.BOLD)
        row_b = left_at(m.VGroup(row_b0, row_b1).arrange(m.RIGHT, buff=0.25), -0.3, -2.6)
        row_b1.align_to(row_a[1], m.LEFT)
        assert row_a[1].get_right()[0] < 6.4 and row_b1.get_right()[0] < 6.4
        base = pill("the baseline: the group's own mean · no critic", REWARD, GOLD_FILL, 20).move_to([NC - 0.15, -3.25, 0])
        assert base.get_right()[0] < 6.5 and base.get_left()[0] > X_ARR + 0.2
        src = cite("trained on this Mac · lm_rl.json").move_to([-3.9, -3.3, 0])
        assert src.get_right()[0] < base.get_left()[0] - 0.2

        self.say(
            "[Now compare each answer with its own group.] [The rewards average zero point five], "
            "[and their spread, the standard deviation, is zero point five too.]",
            m.AnimationGroup(self.new_header("03 · compare", "Better than the group?"),
                             m.FadeOut(s2["snips"]), m.FadeOut(s2["qcard"]), m.FadeOut(s2["labels"]), m.FadeOut(s2["src"]),
                             idx.animate.set_x(X_IDX), boxes.animate.shift((X_BOX - boxes.get_left()[0]) * m.RIGHT),
                             rews.animate.set_x(X_REW), m.FadeIn(title), m.FadeIn(b_lab), m.FadeIn(r_lab), m.FadeIn(src),
                             m.Create(axis), m.FadeIn(ticks),
                             m.Succession(m.Wait(0.7), m.LaggedStart(*[m.FadeIn(dots[k], target_position=np.array([X_REW, ys[k], 0]))
                                                                       for k in range(G)], lag_ratio=0.1)),
                             run_time=2.4),
            m.AnimationGroup(m.Create(mean_line), m.FadeIn(mean_lab, shift=0.1 * m.DOWN)),
            m.FadeIn(std0, shift=0.1 * m.UP),
        )
        self.say(
            "[Subtract the mean], [divide by the spread]: [that's each answer's advantage.]",
            m.AnimationGroup(*[d.animate.move_to(stack(order[k][1] - MEAN_E, order[k][0])) for k, d in enumerate(dots)],
                             mean_line.animate.set_x(nx(0)), m.Transform(mean_lab, mean_lab0), m.Transform(std0, std1),
                             run_time=1.2),
            m.AnimationGroup(*[d.animate.move_to(stack(A_EARLY[k], order[k][0])).set_color(SIGNAL)
                               for k, d in enumerate(dots)], m.Transform(std0, std2), run_time=1.3),
            m.FadeIn(f_adv, shift=0.1 * m.UP),
        )
        self.say(
            "[So each right answer gets plus one], [and each wrong one minus one.]",
            m.AnimationGroup(m.FadeIn(a_lab), *[m.FadeIn(advs[k], target_position=dots[k]) for k in right]),
            m.AnimationGroup(*[m.FadeIn(advs[k], target_position=dots[k]) for k in wrong]),
        )
        self.say(
            "[That's because our run divided by the population spread.] [Two popular libraries use the sample spread "
            "instead]: [then it's plus or minus zero point nine three five.]",
            m.FadeIn(row_a, shift=0.1 * m.UP),
            m.FadeIn(row_b0, shift=0.1 * m.UP),
            m.FadeIn(row_b1, shift=0.1 * m.LEFT),
        )
        dim = m.VGroup(*[m.VGroup(idx[k], boxes[k], rews[k], advs[k]) for k in wrong])
        self.say(
            "[Every token of an answer shares its advantage]: [the update makes the right answers likelier], "
            "[and the wrong ones less likely.]",
            m.Circumscribe(advs, color=SIGNAL, buff=0.08),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in ups], lag_ratio=0.15),
                             *[idx[k][0].animate.set_fill(mix(CORAL_FILL, CORAL, 0.45), 1) for k in right]),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in downs], lag_ratio=0.15),
                             dim.animate.set_opacity(WRONG_DIM)),
        )
        self.say(
            "[No critic needed]: [the group's own average is the baseline.]",
            m.FadeIn(base, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(m.VGroup(mean_line, mean_lab), color=REWARD, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · when the group agrees

    def agreement(self):
        self.next_section("4 · When the group agrees", notes=(
            f"Late group (lm_rl.json groups.late): step 191, GSM8K training problem {LATE['id'].split('-')[1]}, 'Hannah "
            f"has three brothers…' (reference 48), rewards {list(R_LATE.astype(int))} → {A_LATE.max():+.4f} for a right "
            f"answer, {A_LATE.min():+.4f} for a wrong one (grpo_advantages, eps 1e-4, ddof 0; equal to the run's log). The "
            "closed form (notes §23): A₊ = √((1−p)/p), A₋ = −√(p/(1−p)), p = share right: the rarer outcome gets the "
            "bigger push. All eight right (FOR EXAMPLE): std 0 → every advantage 0 → no policy gradient; only the KL term "
            "acts (DAPO's dynamic sampling drops such groups). Measured (curves.zero_adv_groups, 8 groups per step): "
            f"{Z_FIRST:.4f} of the groups in steps 0–9 ({round(Z_FIRST * N10)} of {N10}) and {Z_LAST:.2f} in steps "
            f"190–199 ({round(Z_LAST * N10)} of {N10}); over all 200 steps {Z.mean():.3f}, about one in three."))
        XS = [-2.55 + 1.2 * j for j in range(G)]
        LX = -3.3

        def strip(rewards, advs, y_r, y_a):
            r = m.VGroup(*[reward_pill(v).move_to([x, y_r, 0]) for v, x in zip(rewards, XS)])
            a = m.VGroup(*[adv_pill(v).move_to([x, y_a, 0]) for v, x in zip(advs, XS)])
            labs = m.VGroup(right_at(text("reward", 20, REWARD), LX, y_r), right_at(text("advantage", 20, SIGNAL), LX, y_a))
            return r, a, labs

        r1, a1, l1 = strip(R_LATE, A_LATE, 1.75, 1.2)
        t1 = m.VGroup(text("step 191 · a real group", 22, INK, weight=m.BOLD),
                      cite("“Hannah has three brothers…” · 5 of 8 right")).arrange(m.RIGHT, buff=0.3)
        left_at(t1, -6.4, 2.35)
        assert XS[-1] + 0.6 < 6.5 and t1.get_right()[0] < 6.4
        r2, a2, l2 = strip(R_ALL, A_ALL, -0.05, -0.6)
        t2 = m.VGroup(text("all 8 right", 22, INK, weight=m.BOLD), cite("for example")).arrange(m.RIGHT, buff=0.3)
        left_at(t2, -6.4, 0.55)
        nosig = left_at(cite("spread 0 → every advantage 0 · only the leash still pulls"), XS[0] - 0.5, -1.1)
        wrong_late = [j for j in range(G) if R_LATE[j] == 0]
        right_late = [j for j in range(G) if R_LATE[j] > 0]

        m_title = left_at(text("groups where all 8 answers scored the same", 22, INK, weight=m.BOLD), -6.4, -1.75)
        meters, m_labs, m_vals = [], [], []
        for (lab, frac, n), y in zip((("first 10 steps", Z_FIRST, round(Z_FIRST * N10)),
                                      ("last 10 steps", Z_LAST, round(Z_LAST * N10))), (-2.3, -2.85)):
            m_labs.append(right_at(text(lab, 20, MUTED), LX, y))
            mt = Meter(width=6.0, fraction=frac, color=mix(CARD, SIGNAL, 0.6), height=0.28)
            mt.move_to([XS[0] - 0.5 + 3.0, y, 0])
            meters.append(mt)
            m_vals.append(left_at(code(f"{round(100 * frac)}% · {n} of {N10}", 20, INK, weight=m.BOLD),
                                  mt.get_right()[0] + 0.2, y))
        assert m_vals[0].get_right()[0] < 6.5
        src = cite("trained on this Mac · one run · 8 groups per step").move_to([0, -3.35, 0])

        self.say(
            "[Late in training, at step one hundred ninety-one], [five of eight answers to another question were right.]",
            m.AnimationGroup(self.new_header("04 · agreement", "When the whole group agrees"), m.FadeIn(t1),
                             m.FadeIn(l1[0])),
            m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in r1], lag_ratio=0.12),
        )
        self.say(
            "[Now a right answer gets plus zero point seven seven], [a wrong one minus one point two nine]: "
            "[the rarer outcome gets the bigger push.]",
            m.AnimationGroup(m.FadeIn(l1[1]), *[m.FadeIn(a1[j], shift=0.1 * m.DOWN) for j in right_late]),
            m.AnimationGroup(*[m.FadeIn(a1[j], shift=0.1 * m.DOWN) for j in wrong_late]),
            m.LaggedStart(*[m.Circumscribe(a1[j], color=SIGNAL, buff=0.06) for j in wrong_late], lag_ratio=0.2),
        )
        self.say(
            "[But what if all eight are right?] [Then the spread is zero, and so is every advantage]: "
            "[that group teaches nothing.]",
            m.AnimationGroup(m.FadeIn(t2), m.FadeIn(l2[0]), m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in r2], lag_ratio=0.08)),
            m.AnimationGroup(m.FadeIn(l2[1]), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.DOWN) for c in a2], lag_ratio=0.08)),
            m.FadeIn(nosig, shift=0.1 * m.UP),
        )
        self.say(
            "[In our run, about a third of the groups were like that]: [thirty-one percent in the first ten steps], "
            "[and thirty-five percent in the last ten.]",
            m.AnimationGroup(m.FadeIn(m_title), m.FadeIn(src), m.FadeIn(meters[0].track), m.FadeIn(meters[1].track),
                             m.FadeIn(m_labs[0]), m.FadeIn(m_labs[1])),
            m.AnimationGroup(m.GrowFromEdge(meters[0].bar, m.LEFT), m.FadeIn(m_vals[0])),
            m.AnimationGroup(m.GrowFromEdge(meters[1].bar, m.LEFT), m.FadeIn(m_vals[1])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · the update

    def the_update(self):
        self.next_section("5 · The clip, plus a leash", notes=(
            "The GRPO objective (Shao et al. 2024, eq. 3): J = (1/G) Σ_i (1/|o_i|) Σ_t { min(ρ_{i,t} A_i, clip(ρ_{i,t}, "
            "1 − ε, 1 + ε) A_i) − β KL[π_θ ‖ π_ref] }, ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t}); "
            "outcome rewards: A_i is the same for every token t. The clip is PPO's (Episode 18, Schulman et al. 2017). The "
            "KL is estimated per token with k3 = π_ref/π_θ − log(π_ref/π_θ) − 1 (eq. 4; Schulman 2020) and sits in the "
            "loss, not the reward (Episode 20). Our run: ε = 0.2, β = 0.04, reference = the frozen base model. The code "
            "is the on-screen excerpt from lm_rl.md of research/experiments/lm/grpo.py (MLX; mx = mlx.core; comments "
            "shortened; asserted line by line against both files). The two averages (per answer, then per group) are "
            "DeepSeekMath's sequence-level averaging, not the token-level average of DAPO / Dr. GRPO."))
        J = formula([r"J", "=", r"\frac{1}{G}\sum_{i=1}^{G}", r"\frac{1}{|o_i|}\sum_{t}", r"\Big[", r"\min\big(",
                     r"\rho_{i,t}", r"\,A_i", r",\;", r"\operatorname{clip}(\rho_{i,t},\,1-\epsilon,\,1+\epsilon)",
                     r"\,A_i", r"\big)", r"-", r"\beta", r"\,\mathrm{KL}", r"\Big]"],
                    [INK, INK, INK, INK, INK, INK, ACTION, SIGNAL, INK, EXPLORE, SIGNAL, INK, INK, EXPLORE, EXPLORE, INK], 38)
        J.move_to([0, 1.75, 0])
        assert J.width < 12.6
        jb = J.get_bottom()[1] - 0.08

        def brace(parts, color):
            b = m.Brace(parts, m.DOWN, buff=0.1, color=color)
            return b.shift((jb - b.get_top()[1]) * m.UP)

        b_avg = brace(J[2:4], MUTED)
        b_clip = brace(J[5:12], EXPLORE)
        b_kl = brace(J[13:15], EXPLORE)
        l_avg = m.VGroup(cite("per token,"), cite("then per answer")).arrange(m.DOWN, buff=0.05).next_to(b_avg, m.DOWN, buff=0.1)
        l_clip = m.VGroup(text("episode 18's clipped ratio", 20, EXPLORE),
                          code(f"ε = {CLIP:g}", 20, EXPLORE)).arrange(m.DOWN, buff=0.06).next_to(b_clip, m.DOWN, buff=0.1)
        l_kl = m.VGroup(text("episode 20's leash", 20, EXPLORE),
                        code(f"k3 · β = {BETA:g}", 20, EXPLORE)).arrange(m.DOWN, buff=0.06).next_to(b_kl, m.DOWN, buff=0.1)
        assert l_avg.get_right()[0] < l_clip.get_left()[0] - 0.15 and l_clip.get_right()[0] < l_kl.get_left()[0] - 0.15
        assert l_kl.get_right()[0] < 6.5

        rho = formula([r"\rho_{i,t}", "=", r"\pi_\theta(o_{i,t})", r"\,/\,", r"\pi_{\mathrm{old}}(o_{i,t})"],
                      [ACTION, INK, ACTION, INK, MUTED], 36)
        left_at(rho, -5.6, -0.75)
        rho_t = left_at(cite("a token's new probability ÷ its old one"), rho.get_right()[0] + 0.4, -0.75)
        clip_t = left_at(m.VGroup(code(f"clip: {1 - CLIP:g} ≤ ρ ≤ {1 + CLIP:g}", 22, EXPLORE, weight=m.BOLD),
                                  cite("past it, the objective goes flat · episode 18")).arrange(m.RIGHT, buff=0.35), -5.6, -1.45)
        a_t = left_at(m.VGroup(formula([r"A_i"], [SIGNAL], 36), cite("one advantage per answer, for every one of its tokens")).arrange(m.RIGHT, buff=0.35),
                      -5.6, -2.15)
        kl_t = left_at(m.VGroup(code("KL", 22, EXPLORE, weight=m.BOLD),
                                cite("to the frozen starting model, estimated token by token (k3)")).arrange(m.RIGHT, buff=0.35),
                       -5.6, -2.85)
        for t_ in (rho_t, clip_t, a_t, kl_t):
            assert t_.get_right()[0] < 6.4
        src = cite("Shao et al., 2024, eq. 3").move_to([5.0, 2.55, 0])

        cb = code_block(LOSS, 20, title="our run · research/experiments/lm/grpo.py (MLX)", accent=EDGE)
        cb.move_to([0, -0.35, 0])
        assert cb.width < 12.6 and cb.get_bottom()[1] > -3.4 and cb.get_top()[1] < 2.6
        body = cb[1]
        hl = [glyphs(body, s) for s in ("a = (r - r.mean()) / (r.std() + 1e-4)", "ratio = mx.exp(lp - old_lp)",
                                        "mx.clip(ratio, 1 - eps, 1 + eps)", "kl = mx.exp(ref_lp - lp) - (ref_lp - lp) - 1")]
        hl_cols = (SIGNAL, ACTION, EXPLORE, EXPLORE)

        self.say(
            "[With advantages in hand], [the update is episode eighteen's clipped objective, on every token.]",
            m.AnimationGroup(self.new_header("05 · the update", "The clip, plus a leash"),
                             m.FadeIn(J[5:12], shift=0.1 * m.UP), m.FadeIn(src)),
            m.AnimationGroup(m.GrowFromCenter(b_clip), m.FadeIn(l_clip)),
        )
        self.say(
            "[The ratio compares a token's new probability with its old one], "
            "[and the clip keeps it between zero point eight and one point two.]",
            m.AnimationGroup(m.FadeIn(rho), m.FadeIn(rho_t), m.Circumscribe(J[6], color=ACTION, buff=0.06)),
            m.AnimationGroup(m.FadeIn(clip_t, shift=0.1 * m.UP), m.Circumscribe(J[9], color=EXPLORE, buff=0.06)),
        )
        self.say(
            "[Then it subtracts beta times a KL estimate to the frozen starting model]: "
            "[episode twenty's leash, with beta at zero point zero four.]",
            m.AnimationGroup(m.FadeIn(J[12:15], shift=0.1 * m.UP), m.FadeIn(J[4]), m.FadeIn(J[15]), m.FadeIn(kl_t)),
            m.AnimationGroup(m.GrowFromCenter(b_kl), m.FadeIn(l_kl)),
        )
        self.say(
            "[It averages over each answer's tokens], [then over the group.]",
            m.AnimationGroup(m.FadeIn(J[3], shift=0.1 * m.UP), m.FadeIn(a_t, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(J[0:3], shift=0.1 * m.UP), m.GrowFromCenter(b_avg), m.FadeIn(l_avg)),
        )
        self.say(
            "[In our run's code, the loss takes ten lines]: [the advantage], [the ratio], [the clip], [and the leash.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(J, b_avg, b_clip, b_kl, l_avg, l_clip, l_kl, rho, rho_t, clip_t, a_t, kl_t, src)),
                             m.FadeIn(cb, shift=0.1 * m.UP), run_time=1.2),
            *[m.Circumscribe(m.VGroup(*g), color=c, buff=0.06) for g, c in zip(hl, hl_cols)],
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · one real update

    def one_update(self):
        self.next_section("6 · One real update", notes=(
            "lm_rl.json update (step 100 of our run): the 64 answers of a step are split into 2 minibatches of 32 (4 groups "
            f"each), one AdamW step each. Minibatch 1 ({N1:,} answer tokens) is scored by the same weights that sampled it: "
            f"ratio π_θ/π_old = 1.000 for every token, nothing clipped. Minibatch 2 ({N2:,} tokens) comes after one AdamW "
            f"step: ratio mean {MB2['ratio_mean']:.4f}, min {MB2['ratio_min']:.4f}, max {MB2['ratio_max']:.4f}; "
            f"{OUTSIDE} tokens ({100 * MB2['outside_band']:.2f} %) outside [0.8, 1.2]; the clip changed the objective for "
            f"{MB2['clip_frac']:.6f} of the tokens = {CLIPPED} tokens (ratio > 1.2 with A > 0, or < 0.8 with A < 0). "
            f"Histogram bin edges {EDGES[:-1]} | ∞, counts {C2}; bar heights on a log scale (log10(1 + count)). Over the "
            f"whole run the clip touched {100 * H['clip_frac_mean']:.2f} % of tokens on average (clip_frac_mean). "
            "DeepSeekMath took a single update per batch (notes §23), where the ratio is 1 and the clip cannot act."))
        NB = len(C2)
        BW, GAP = 0.86, 0.1
        X0 = -5.0
        Y0, HMAX = -1.45, 2.85
        LOGMAX = np.log10(1 + 12_000)
        xs = [X0 + BW / 2 + i * (BW + GAP) for i in range(NB)]
        assert xs[-1] + BW / 2 < 6.4

        def height(c):
            return max(HMAX * np.log10(1 + c) / LOGMAX, 0.02)

        def bar(i, c):
            hgt = height(c)
            r = m.Rectangle(width=BW, height=hgt).set_stroke(ACTION, 2 if c else 0).set_fill(CORAL_FILL if c else EDGE, 1)
            return r.move_to([xs[i], Y0 + hgt / 2, 0])

        def count_lab(i, c):
            return code(f"{c:,}", 20, INK).move_to([xs[i], Y0 + height(c) + 0.22, 0])

        lo_x = xs[2] - BW / 2 - GAP / 2
        hi_x = xs[8] + BW / 2 + GAP / 2
        band = m.Rectangle(width=hi_x - lo_x, height=HMAX + 0.3).set_stroke(width=0).set_fill(EXPLORE, 0.16)
        band.move_to([(lo_x + hi_x) / 2, Y0 + (HMAX + 0.3) / 2, 0])
        band_lab = code(f"clip band {1 - CLIP:g} to {1 + CLIP:g}", 20, EXPLORE, weight=m.BOLD).move_to([band.get_x(), Y0 + HMAX + 0.55, 0])
        base = m.Line([X0 - 0.15, Y0, 0], [xs[-1] + BW / 2 + 0.15, Y0, 0]).set_stroke(FAINT, 3)
        edge_labs = m.VGroup(*[code(f"{EDGES[i]:g}", 20, MUTED).move_to([xs[i] - BW / 2 - GAP / 2, Y0 - 0.3, 0])
                               for i in range(1, NB)])
        x_lab = cite("the ratio: a token's new probability ÷ its old one").move_to([0.2, Y0 - 0.78, 0])
        y_lab = left_at(cite("tokens · log scale"), X0 - 0.1, 1.85)
        bars1 = m.VGroup(*[bar(i, c) for i, c in enumerate(C1)])
        bars2 = m.VGroup(*[bar(i, c) for i, c in enumerate(C2)])
        lab1 = count_lab(5, C1[5])
        labs2 = m.VGroup(*[count_lab(i, c) for i, c in enumerate(C2) if c])
        assert all(a.get_right()[0] < b.get_left()[0] - 0.05 for a, b in zip(labs2, labs2[1:]))
        assert band_lab.get_bottom()[1] > max(l.get_top()[1] for l in labs2) + 0.05 and band_lab.get_top()[1] < 2.15

        info1 = left_at(text(f"step 100 · first half: {N1:,} tokens, scored by the weights that sampled them", 22, INK),
                        -6.4, 2.4)
        info2 = left_at(text(f"second half: {N2:,} tokens, after one gradient step", 22, INK), -6.4, 2.4)
        one = left_at(code("every ratio = 1.000", 22, ACTION, weight=m.BOLD), xs[6] + 0.2, 0.9)
        rng_t = code(f"ratios {MB2['ratio_min']:.2f} to {MB2['ratio_max']:.2f}", 22, ACTION, weight=m.BOLD)
        left_at(rng_t, -6.4, -2.75)
        stat = m.VGroup(text(f"outside the band: {OUTSIDE} tokens ({100 * OUTSIDE / N2:.2f}%)", 22, INK),
                        text(f"clipped: {CLIPPED}", 22, EXPLORE, weight=m.BOLD)).arrange(m.RIGHT, buff=0.4)
        left_at(stat, -6.4, -3.25)
        assert stat.get_right()[0] < 6.4
        run_t = right_at(cite(f"whole run: {100 * H['clip_frac_mean']:.2f}% of tokens clipped"), 6.4, -2.75)
        assert run_t.get_left()[0] > rng_t.get_right()[0] + 0.3
        src = right_at(cite("trained on this Mac"), 6.4, 2.4)
        assert info1.get_right()[0] < src.get_left()[0] - 0.2

        self.say(
            "[Here's one real update from our run, at step one hundred.]",
            m.AnimationGroup(self.new_header("06 · one update", "One real update"), m.Create(base), m.FadeIn(edge_labs),
                             m.FadeIn(x_lab), m.FadeIn(y_lab), m.FadeIn(band), m.FadeIn(band_lab), m.FadeIn(src)),
        )
        self.say(
            "[The first half of the batch is scored by the same weights that sampled it]: [every ratio is exactly one.]",
            m.FadeIn(info1, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(bars1, shift=0.1 * m.UP), m.FadeIn(lab1), m.FadeIn(one, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[After one gradient step, the second half spreads out], [from zero point six eight to one point five four.]",
            m.AnimationGroup(m.FadeTransform(info1, info2), m.FadeOut(one), m.FadeOut(lab1),
                             *[m.Transform(a, b) for a, b in zip(bars1, bars2)], run_time=1.6),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(l) for l in labs2], lag_ratio=0.08), m.FadeIn(rng_t, shift=0.1 * m.UP)),
        )
        self.say(
            "[Only a quarter of a percent of its tokens left the clip band], "
            "[and the clip changed the objective for just five of them.]",
            m.AnimationGroup(m.FadeIn(stat[0], shift=0.1 * m.UP),
                             *[m.Circumscribe(bars1[i], color=INK, buff=0.05) for i in (1, 9, 10)]),
            m.FadeIn(stat[1], shift=0.1 * m.LEFT),
        )
        self.say(
            "[The clip is a safety net], [and in our run it rarely had to act.]",
            m.Circumscribe(band, color=EXPLORE, buff=0.04),
            m.FadeIn(run_t, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · DeepSeekMath

    def deepseekmath(self):
        self.next_section("7 · Where GRPO came from", notes=(
            "Shao et al. 2024, 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models', arXiv "
            "2402.03300 (v1 5 Feb 2024). §4.2: 'For each question, we sample 64 outputs', 'The KL coefficient is 0.04', "
            "learning rate 1e-6, 'a single update following each exploration stage'; the reward came from a LEARNED reward "
            "model (DeepSeekMath-Base 7B, retrained iteratively). Table 5 (chain of thought, top-1): GSM8K 82.9 % "
            "(DeepSeekMath-Instruct 7B) → 88.2 % (DeepSeekMath-RL 7B); MATH 46.8 → 51.7. Memory [DERIVED, notes §23 worked "
            "example, illustrative, not from the paper]: 7B models, full fine-tuning, mixed-precision Adam at 16 bytes per "
            "trained parameter (Rajbhandari et al. 2020), 2 bytes per frozen BF16 parameter, model states only (no "
            f"activations, no KV cache): PPO {' + '.join(f'{v:.0f}' for v in MEM_PPO.values())} = {PPO_GB:.0f} GB; GRPO "
            f"with a reward model {GRPO_GB:.0f} GB (−{SAVED} %). Our run: the policy (0.5B + a LoRA adapter) and a frozen "
            f"copy of the base model as the reference; the reward is a regex checker, not a model; peak GPU memory per step "
            f"≈ {OUR_MEM:.1f} GB (median over the 200 steps, curves.peak_mem_gb; includes sampling)."))
        # left: the paper
        lc = card(5.5, 5.25, SIGNAL, CARD).move_to([-3.75, -0.25, 0])
        lx = lc.get_left()[0] + 0.35
        top = lc.get_top()[1]
        p_title = left_at(title_text("DeepSeekMath", 34, INK), lx, top - 0.45)
        p_src = left_at(cite("Shao et al., 2024"), lx, top - 0.9)
        facts = m.VGroup(code(f"G = {DSM_G}", 24, ACTION, weight=m.BOLD), text("answers per question", 20, MUTED))
        facts.arrange(m.RIGHT, buff=0.25)
        left_at(facts, lx, top - 1.55)
        beta_t = m.VGroup(code(f"β = {DSM_BETA:g}", 24, EXPLORE, weight=m.BOLD), text("the KL leash, in the loss", 20, MUTED))
        beta_t.arrange(m.RIGHT, buff=0.25)
        left_at(beta_t, lx, top - 2.1)
        rm_t = left_at(text("a learned reward model: kept", 20, REWARD), lx, top - 2.6)
        g_lab = left_at(text("GSM8K · 7B model · top-1", 20, INK, weight=m.BOLD), lx, top - 3.25)
        BX, BS = lx + 1.05, 3.2 / 100
        bars, vals = [], []
        for (lab, v, col), y in zip((("before", DSM_BEFORE, MUTED), ("after", DSM_AFTER, GREEN)), (top - 3.75, top - 4.3)):
            r = m.Rectangle(width=v * BS, height=0.34).set_stroke(col, 2).set_fill(FILLS[col], 1)
            left_at(r, BX, y)
            bars.append(m.VGroup(right_at(text(lab, 20, MUTED), BX - 0.15, y), r))
            vals.append(left_at(code(f"{v:.1f}%", 20, col, weight=m.BOLD), r.get_right()[0] + 0.12, y))
        assert vals[1].get_right()[0] < lc.get_right()[0] - 0.1
        t5 = left_at(cite("Table 5"), lx, top - 4.85)
        assert t5.get_bottom()[1] > lc.get_bottom()[1] + 0.05

        # right: memory, 7B, model states
        rx0 = -0.35
        m_title = left_at(text("model state in memory · a 7B model", 22, INK, weight=m.BOLD), rx0, 2.3)
        m_sub = left_at(cite("derived · 16 bytes per trained parameter, 2 per frozen one"), rx0, 1.9)
        XB, UNIT = 0.75, 4.2 / PPO_GB
        cols = {"policy": ACTION, "critic": VALUE, "reward": REWARD, "reference": EXPLORE}

        def mem_bar(mem, y, name):
            segs, x = m.VGroup(), XB
            for k, gb in mem.items():
                w = gb * UNIT
                segs.add(m.Rectangle(width=w, height=0.5).set_stroke(cols[k], 2.5).set_fill(FILLS[cols[k]], 1)
                         .move_to([x + w / 2, y, 0]))
                x += w
            labs = m.VGroup()
            for seg, k in zip(segs, mem):
                if k in ("policy", "critic"):
                    labs.add(text(f"{k} {mem[k]:.0f}", 20, INK).move_to(seg))
            small = m.VGroup(*[s for s, k in zip(segs, mem) if k in ("reward", "reference")])
            labs.add(code("14 + 14", 20, MUTED).next_to(small, m.UP, buff=0.1))
            labs.set_z_index(2)
            row = right_at(code(name, 22, INK, weight=m.BOLD), XB - 0.15, y)
            total = left_at(code(f"{sum(mem.values()):.0f} GB", 22, INK, weight=m.BOLD), x + 0.15, y)
            return m.VGroup(row, segs, labs, total)

        ppo = mem_bar(MEM_PPO, 0.95, "PPO")
        grpo = mem_bar(MEM_GRPO, -0.15, "GRPO")
        assert ppo[3].get_right()[0] < 6.4 and ppo[0].get_left()[0] > lc.get_right()[0] + 0.15
        cut = code(f"−{SAVED}%", 22, SIGNAL, weight=m.BOLD).next_to(grpo[3], m.RIGHT, buff=0.2)
        assert cut.get_right()[0] < 6.5

        # our run: two models and a checker
        oy = -1.65
        o_lab = right_at(code("ours", 22, INK, weight=m.BOLD), XB - 0.15, oy)

        def mini(art, title, color, x):
            box = card(1.75, 1.05, color, CARD, radius=0.14, stroke=2)
            body = m.VGroup(art, text(title, 20, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.08).move_to(box)
            return m.VGroup(box, body).move_to([x, oy, 0])

        o_pol = mini(icon(LLM_POL, 0.45), "policy", ACTION, XB + 0.9)
        o_ref = mini(icon(LLM_REF, 0.45), "reference", EXPLORE, XB + 2.8)
        o_chk = mini(icon(CHECK, 0.45), "a checker", REWARD, XB + 4.7)
        o_chk[0].set_stroke(REWARD, 2, opacity=0.6)
        assert o_chk.get_right()[0] < 6.5
        o_sub = left_at(cite(f"≈ {OUR_MEM:.0f} GB peak, measured on this Mac · 0.5B"), rx0, oy - 0.9)
        o_ep = left_at(cite("the checker, not a model: episode 24"), rx0, oy - 1.35)
        assert o_ep.get_bottom()[1] > -3.55 and o_sub.get_right()[0] < 6.4, (o_ep.get_bottom(), o_sub.get_right())

        self.say(
            "[It comes from the DeepSeekMath paper, in twenty twenty-four]: [sixty-four answers per question], "
            "[and a beta of zero point zero four.]",
            m.AnimationGroup(self.new_header("07 · DeepSeekMath", "Where GRPO came from"), m.FadeIn(lc), m.FadeIn(p_title),
                             m.FadeIn(p_src)),
            m.FadeIn(facts, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(beta_t, shift=0.1 * m.UP), m.FadeIn(rm_t, shift=0.1 * m.UP)),
        )
        self.say(
            "[Their seven-billion-parameter model went from eighty-two point nine to eighty-eight point two percent], "
            "[on grade-school maths.]",
            m.AnimationGroup(m.FadeIn(g_lab), m.FadeIn(bars[0][0]), m.GrowFromEdge(bars[0][1], m.LEFT), m.FadeIn(vals[0]),
                             m.Succession(m.Wait(0.6), m.AnimationGroup(m.FadeIn(bars[1][0]), m.GrowFromEdge(bars[1][1], m.LEFT),
                                                                        m.FadeIn(vals[1])))),
            m.FadeIn(t5),
        )
        self.say(
            "[For a model that size, PPO's four models hold about two hundred fifty gigabytes]; "
            "[without the critic, about a hundred forty.]",
            m.AnimationGroup(m.FadeIn(m_title), m.FadeIn(m_sub), m.FadeIn(ppo[0]),
                             m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in ppo[1]], lag_ratio=0.4),
                             m.FadeIn(ppo[2]), m.FadeIn(ppo[3])),
            m.AnimationGroup(m.FadeIn(grpo[0]), m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in grpo[1]], lag_ratio=0.4),
                             m.FadeIn(grpo[2]), m.FadeIn(grpo[3]), m.FadeIn(cut, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Our run needs even less]: [a checker replaces the reward model, as episode twenty-four will show], "
            "[so only the policy and its frozen reference remain.]",
            m.AnimationGroup(m.FadeIn(o_lab), m.FadeIn(o_sub)),
            m.AnimationGroup(m.FadeIn(o_chk, shift=0.1 * m.UP), m.FadeIn(o_ep)),
            m.AnimationGroup(m.FadeIn(o_pol, shift=0.1 * m.UP), m.FadeIn(o_ref, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · what came next

    def what_next(self):
        self.next_section("8 · Two fixes, a year later", notes=(
            "Dr. GRPO: Liu et al. 2025, 'Understanding R1-Zero-Like Training: A Critical Perspective' (arXiv 2503.20783): "
            "'Response-level length bias' from dividing by |o_i| (long wrong answers are penalised less per token) and "
            "'Question-level difficulty bias' from dividing by std (too-easy and too-hard questions get higher weights); "
            "the fix: drop both (a constant normaliser), advantage = r − mean, and β = 0. DAPO: Yu et al. 2025 (ByteDance "
            "Seed / Tsinghua AIR, arXiv 2503.14476): Clip-Higher (ε_low 0.2, ε_high 0.28), Dynamic Sampling ('over-sample "
            "and filter out prompts with the accuracy equal to 1 and 0'), Token-Level Policy Gradient Loss (1/Σ|o_i|), "
            "Overlong Reward Shaping; no KL term; AIME 2024 50 points with Qwen2.5-32B. 'Agents, Animated' Episode 30 "
            "trains a coding agent with group-relative advantages over sixteen rollouts in sandboxes."))

        def fix_card(cx, title, src, rows, color):
            box = card(6.1, 3.9, color, CARD).move_to([cx, 0.45, 0])
            lx, top = box.get_left()[0] + 0.35, box.get_top()[1]
            t = left_at(title_text(title, 34, INK), lx, top - 0.5)
            s = right_at(cite(src), box.get_right()[0] - 0.35, top - 0.5)
            items = []
            for k, (head, sub) in enumerate(rows):
                y = top - 1.25 - 0.9 * k
                dot = m.Dot([lx + 0.1, y, 0], radius=0.07, color=SIGNAL)
                h_ = left_at(text(head, 22, INK, weight=m.BOLD), lx + 0.35, y)
                s_ = left_at(cite(sub), lx + 0.35, y - 0.37)
                assert max(h_.get_right()[0], s_.get_right()[0]) < box.get_right()[0] - 0.2, head
                items.append(m.VGroup(dot, h_, s_))
            return m.VGroup(box, t, s), items

        c1, i1 = fix_card(-3.25, "Dr. GRPO", "Liu et al., 2025", [
            ("no division by the spread", "it favours very easy and very hard questions"),
            ("no division by the length", "it goes easy on long wrong answers"),
            ("and no KL leash: β = 0", "advantage = reward − the group's mean")], SIGNAL)
        c2, i2 = fix_card(3.25, "DAPO", "Yu et al., 2025", [
            ("clip higher: up to 1.28", "rare tokens may grow faster"),
            ("dynamic sampling", "skip groups that all agree, sample more"),
            ("token-level loss", "every token in the batch counts the same")], SIGNAL)
        sister = pill("Agents, Animated · episode 30: coding agents, trained with group-relative advantages", MUTED, CARD, 20)
        sister.move_to([0, -2.4, 0])
        assert sister.width < 12.8
        src = cite("both 2025 · DAPO also drops the KL term").move_to([0, -3.2, 0])

        self.say(
            "[In twenty twenty-five, Doctor GRPO stopped dividing by the spread], "
            "[which over-weights very easy and very hard questions], "
            "[and by each answer's length, which goes easy on long wrong answers.]",
            m.AnimationGroup(self.new_header("08 · what came next", "Two fixes, a year later"), m.FadeIn(c1),
                             m.FadeIn(i1[0][0]), m.FadeIn(i1[0][1])),
            m.FadeIn(i1[0][2], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(i1[1], shift=0.1 * m.UP), m.Succession(m.Wait(1.2), m.FadeIn(i1[2], shift=0.1 * m.UP))),
        )
        self.say(
            "[A second recipe, from the same year, raises the clip's upper edge], [skips groups that all agree], "
            "[and averages over every token in the batch.]",
            m.AnimationGroup(m.FadeIn(c2), m.FadeIn(i2[0], shift=0.1 * m.UP), m.FadeIn(src)),
            m.FadeIn(i2[1], shift=0.1 * m.UP),
            m.FadeIn(i2[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Our sister course, Agents, Animated, trains coding agents with this same group trick], [in episode thirty.]",
            m.FadeIn(sister, shift=0.1 * m.UP),
            m.Circumscribe(sister, color=MUTED, buff=0.06),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            "Sample (a group of G answers to one question; G = 8 in our run, 64 in DeepSeekMath), compare (advantage = "
            "(reward − the group's mean) / the group's std: no critic), clip (Episode 18's clipped ratio, plus Episode 20's "
            "KL leash to the frozen reference). Episode 24: rewards you can check."))
        halo = glow(SIGNAL, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Sample", 64, ACTION), title_text("Compare", 64, SIGNAL), title_text("Clip", 64, EXPLORE))
        words.arrange(m.RIGHT, buff=1.3).move_to([0, 1.2, 0])
        on_baseline(words)
        subs = m.VGroup(*[m.VGroup(text(a, 22, MUTED), text(b, 22, MUTED)).arrange(m.DOWN, buff=0.08)
                          for a, b in (("a group of answers", "to one question"), ("each with its group,", "no critic"),
                                       ("a small step,", "on a leash"))])
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), 0, 0]).align_to([0, words[1].get_bottom()[1] - 0.3, 0], m.UP)
        assert all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        res = formula([r"A_i", "=", r"\big(", r"r_i", "-", r"\mathrm{mean}(r)", r"\big)", r"\,/\,", r"\mathrm{std}(r)"],
                      [SIGNAL, INK, INK, REWARD, INK, REWARD, INK, INK, INK], 40).move_to([0, -1.45, 0])
        row = m.VGroup(*[adv_pill(a, w=1.08) for a in A_EARLY]).arrange(m.RIGHT, buff=0.15).move_to([0, -2.6, 0])
        assert row.width < 12.6

        self.say(
            "[To sum up]: [sample], [compare], [clip.]",
            m.AnimationGroup(self.new_header("09 · recap", "Sample, compare, clip"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Sample a group of answers to one question], [compare each one with its group instead of asking a critic], "
            "[and take a small step, clipped and leashed.]",
            m.AnimationGroup(m.Circumscribe(words[0], color=ACTION, buff=0.12),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in row], lag_ratio=0.1)),
            m.AnimationGroup(m.Circumscribe(words[1], color=SIGNAL, buff=0.12), m.FadeIn(res, shift=0.1 * m.UP)),
            m.Circumscribe(words[2], color=EXPLORE, buff=0.12),
        )
        self.wait(0.3)
        self.clear()
