"""Trial and Reward · Episode 20 · The Model Is the Policy (Module 6 · Teaching Language Models with Reward)

Module 6 opens by handing Rover's job to a small language model. The loop of Episode 1 stays: the state is the prompt
plus the tokens written so far, each token is an action (CORAL chips, kit token_row), the world just appends the
token, and a completion is one episode with one reward at the very end. The running example is OUR GRPO RUN on this
Mac (research/experiments/lm_rl.md, lm_rl.json): Qwen2.5-0.5B-Instruct + LoRA on GSM8K with the \\boxed{} prompt and an
exact-match checker. One real held-out question (test-1028, Tom's bedroom set): the starting model's own answer
(181 tokens, verbatim) boxes 300 and the checker pays 0; a second real answer of the same model (test-1280, Anna's
puzzle, 173 tokens) boxes 7 and earns +1, which flows back to every token (γ = 1, so every token's return is the
final reward). Policy gradient on tokens: log π(y|x) = Σ_t log π(y_t | x, y_<t), so Episode 16's REINFORCE pushes
every token of an answer by the same reward; a reward of 0 teaches plain REINFORCE nothing, a baseline (Episode 17)
pushes it down, GRPO's group baseline is Episode 23. The pipeline: pretraining → supervised fine-tuning → RL
(InstructGPT). The KL leash: reward − β·KL(π_θ ‖ π_ref), k3 per token (rl_reference.kl_k3); our run's β = 0.04 kept the
KL near 0.02 per token. The run is introduced (model, adapter, data, G = 8, 200 steps, 71.7 min on this Mac) but its
results (accuracy before/after) are saved for Episodes 24 and 26.

Every number on screen is computed below from rl_reference.py and asserted, or loaded with measured(...), or cited.

Sources (research/notes.md §20, §21, §23 worked example "k3 on one token", Appendix A row 23 and Appendix C row 20;
facts dated October 2026):
- research/experiments/lm_rl.md + lm_rl.json (this Mac Studio, Apple M4 Max, 64 GB, MLX; run 2026-10-25, one seed):
  headline (model, params_base_M 494, trainable_params_M 8.8, G 8, questions_per_step 8, steps 200,
  train_questions_seen 1,600, completions_sampled 12,800, test_questions 500, beta_kl 0.04, clip_eps 0.2, lr 2e-5,
  wall_time_train_min 71.7, train_len_first10 307.9), setup (prompt_system, boxed_regex, number_regex: the checker is
  re-implemented here with the run's own regexes and re-scores both answers), curves (kl, kl_ma10,
  gen_tokens_per_s, peak_mem_gb per step), examples (test-1028 and test-1280: the BASE model's greedy answers,
  verbatim, with their token counts).
- Stiennon et al. 2020, footnote 8: each token a time step, the reward only for the whole summary, γ = 1.
- Ouyang et al. 2022 (InstructGPT), Fig. 2: demonstrations → SFT, comparisons → reward model, PPO with a per-token KL
  penalty to the SFT model (β = 0.02). Ziegler et al. 2019 eq. 2: R = r − β log(π/ρ). DPO eq. 3: max E[r] − β KL.
- Williams 1992 (REINFORCE), Episode 16. Schulman 2020, "Approximating KL Divergence": k3 = (ratio − 1) − log ratio,
  ratio = π_ref/π_θ, unbiased and never negative. Shao et al. 2024 (DeepSeekMath): β = 0.04, KL as a loss term.
- Why RL after imitation (notes §20): SFT can only push probability up (ORPO, Hong et al. 2024, Fig. 3); exposure bias,
  a model trained on human prefixes never practises on its own (Ranzato et al. 2016).

Illustrative (said "for example" or labelled on screen): the next-token probabilities in section 2 (top four
candidates for the first token); the five token probabilities in section 4 (0.95, 0.90, 0.80, 0.60, 0.90 — the
multiset of notes §20's worked example); the baseline 0.5 (the mean of the two rewards) in section 4; the "drifting"
picture in section 6; the token probabilities 0.50 / 0.40 in the k3 example (notes §23 worked example). The answers
are drawn as whitespace-separated words ("for example"); the token counts (181, 173) are the run's real counts.

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

PREFIX = "ep20"
LM = "lm_rl"
END = "<|im_end|>"

# ---------------------------------------------------------------------------- the numbers (all checked here)

H = measured(LM, "headline")
SETUP = measured(LM, "setup")
CURVES = measured(LM, "curves")
EX = {e["id"]: e for e in measured(LM, "examples")}
TOM, ANNA = EX["test-1028"], EX["test-1280"]

# the run, as introduced in section 7
assert H["model"] == "Qwen2.5-0.5B-Instruct" and H["params_base_M"] == 494.0 and H["trainable_params_M"] == 8.8
TRAIN_PCT = 100 * H["trainable_params_M"] / H["params_base_M"]
assert round(TRAIN_PCT, 1) == 1.8 and TRAIN_PCT < 2                                     # "under two percent"
assert "rank 16" in H["finetune"]
assert H["G"] == 8 and H["questions_per_step"] == 8 and H["steps"] == 200              # "eight answers each, 200 steps"
assert H["train_questions_seen"] == 1600 == H["steps"] * H["questions_per_step"]        # "sixteen hundred problems"
assert H["completions_sampled"] == 12800 == H["train_questions_seen"] * H["G"]          # "twelve thousand eight hundred"
assert H["test_questions"] == 500
assert H["beta_kl"] == 0.04 and H["clip_eps"] == 0.2 and H["lr"] == 2e-5
assert H["wall_time_train_min"] == 71.7 and round(H["wall_time_train_min"]) == 72      # "seventy-two minutes"
TOKS = float(np.mean(CURVES["gen_tokens_per_s"]))
assert abs(TOKS - 2150) < 25                                                             # "≈ 2,150 tokens/s"
MEM = float(np.median(CURVES["peak_mem_gb"]))
assert round(MEM) == 25                                                                  # "≈ 25 GB"
AVG_LEN = H["train_len_first10"]
assert abs(AVG_LEN - 300) < 15 and abs(H["greedy_len_before"] - 300) < 15                # "about three hundred tokens"

# the prompt and the checker (the run's own regexes)
SYSTEM = SETUP["prompt_system"]
assert SYSTEM == "Please reason step by step, and put your final answer within \\boxed{}."
BOXED = re.compile(SETUP["boxed_regex"])
NUMBER = re.compile(SETUP["number_regex"])


def checker(answer, reference):
    """1 if the last number inside the last \\boxed{...} equals the reference, else 0 (lm_rl.md, 'The checker')."""
    boxes = BOXED.findall(answer)
    if not boxes:
        return 0.0
    numbers = NUMBER.findall(boxes[-1])
    if not numbers:
        return 0.0
    return float(abs(float(numbers[-1].replace(",", "")) - reference) < 1e-4)


TOM_TEXT, ANNA_TEXT = TOM["base"]["text"], ANNA["base"]["text"]
assert TOM["ref"] == 200 and TOM["base"]["answer"] == 300 and TOM["base"]["tokens"] == 181
assert TOM["base"]["finish"] == "stop" and TOM["base"]["correct"] == 0 == checker(TOM_TEXT, TOM["ref"])   # reward 0
assert ANNA["ref"] == 7 and ANNA["base"]["answer"] == 7 and ANNA["base"]["tokens"] == 173
assert ANNA["base"]["finish"] == "stop" and ANNA["base"]["correct"] == 1 == checker(ANNA_TEXT, ANNA["ref"])  # reward 1
MISTAKE = "Tom has to pay 10% of the $3000 bedroom set each month."
assert MISTAKE in TOM_TEXT and "0.10 \\times 3000 = 300" in TOM_TEXT                    # "ten percent of the full price"


def as_words(answer):
    """An answer split at whitespace, for drawing (not the real tokenizer: "for example"); a bare symbol such as a
    bullet "-" or "=" is joined to the word after it, so no chip is a lone dash."""
    out = []
    for w in answer.split():
        if out and not re.search(r"[A-Za-z0-9\\\[\]()]", out[-1]):
            out[-1] = out[-1] + " " + w
        else:
            out.append(w)
    return out


TOM_WORDS, ANNA_WORDS = as_words(TOM_TEXT), as_words(ANNA_TEXT)                           # drawn as words, "for example"
TOM_BOX, ANNA_BOX = "\\(\\boxed{300}\\)", "\\(\\boxed{7}\\)"
assert TOM_WORDS.count(TOM_BOX) == 1 and ANNA_WORDS.count(ANNA_BOX) == 1
assert TOM_WORDS[:10] == ["To", "determine", "how", "much", "Tom", "has", "to", "pay", "per", "month,"]
QUESTION = " ".join(TOM["question"].split())                                             # double spaces closed up
Q_LINES = ["Tom buys a bedroom set for $3000. He sells his old bedroom for $1000 and uses that",
           "to pay for part of the bedroom set. He then has to pay 10% a month for the bedroom set.",
           "How much does he have to pay per month?"]
assert " ".join(Q_LINES) == QUESTION

# section 2: the first token's top four candidates (illustration)
NEXT = {"To": 0.62, "First": 0.17, "Let's": 0.12, "Tom": 0.04}
assert sum(NEXT.values()) < 1 and max(NEXT, key=NEXT.get) == TOM_WORDS[0]

# section 4: five tokens of Anna's answer with made-up probabilities (notes §20 worked example's multiset)
i_box = ANNA_WORDS.index(ANNA_BOX)
FIVE = ANNA_WORDS[i_box - 3:i_box + 2]
assert FIVE == ["it", "took", "Anna", ANNA_BOX, "hours"]
P5 = np.array([0.95, 0.90, 0.80, 0.60, 0.90])
LOG5 = np.log(P5)
P_ALL, LOG_ALL = float(np.prod(P5)), float(LOG5.sum())
assert round(P_ALL, 2) == 0.37 and round(LOG_ALL, 2) == -1.00 and abs(math.log(P_ALL) - LOG_ALL) < 1e-12
j_box = TOM_WORDS.index(TOM_BOX)
FIVE_TOM = TOM_WORDS[j_box - 4:j_box + 1]
assert FIVE_TOM == ["Tom", "has", "to", "pay", TOM_BOX]
R_RIGHT, R_WRONG = checker(ANNA_TEXT, 7), checker(TOM_TEXT, 200)
BASE = (R_RIGHT + R_WRONG) / 2                                                            # a baseline, e.g. the mean
ADV_RIGHT, ADV_WRONG = R_RIGHT - BASE, R_WRONG - BASE
assert (R_RIGHT, R_WRONG, BASE, ADV_RIGHT, ADV_WRONG) == (1.0, 0.0, 0.5, 0.5, -0.5)

# section 6: k3 on one token (notes §23 worked example) and the run's KL
P_POL, P_REF = 0.50, 0.40
K3 = float(kl_k3(np.log(P_POL), np.log(P_REF)))
assert round(K3, 4) == 0.0231 and K3 >= 0                                                 # "about zero point zero two"
KL = np.array(CURVES["kl"], dtype=float)
KL_MA = np.array(CURVES["kl_ma10"], dtype=float)
STEPS = np.array(CURVES["step"], dtype=float)
assert len(KL) == len(STEPS) == 200 and STEPS[0] == 0 and STEPS[-1] == 199
assert 0.015 < KL[10:].mean() < 0.025 and abs(H["kl_last10"] - 0.0162) < 1e-9            # "near zero point zero two"
assert KL_MA.max() < 0.03

# ---------------------------------------------------------------------------- look

LLM = recolored("llm", CORAL, CORAL_FILL, prefix=PREFIX)
LLM_REF = recolored("llm", MUTED, CARD, prefix=PREFIX)
CHECK = recolored("check", GOLD, GOLD_FILL, prefix=PREFIX)
BOOK = recolored("book", MUTED, CARD, prefix=PREFIX)
PEOPLE = recolored("user", MUTED, CARD, prefix=PREFIX)
DICE = recolored("dice", CORAL, CORAL_FILL, prefix=PREFIX)


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.RIGHT)


def formula(parts, colors, size=44):
    f = m.MathTex(*parts, font_size=size, color=INK)
    for part, col in zip(f, colors):
        part.set_color(col)
    return f


def tint(chip_, color, fill):
    chip_[0].set_stroke(color).set_fill(fill, 1)
    return chip_


def tokens(words, prompt=True, end=True, max_width=13.0):
    """Word chips (CORAL actions), the prompt chip first (TEAL, the state), the end token last."""
    row = token_row((["prompt"] if prompt else []) + list(words) + ([END] if end else []), ACTION, size=21,
                    gap=0.07, max_width=max_width)
    if prompt:
        tint(row[0], STATE, TEAL_FILL)
    return row


def frame(mobs, color=STATE, buff=0.1):
    return m.SurroundingRectangle(m.VGroup(*mobs), buff=buff, corner_radius=0.12).set_stroke(color, 2.5)


def msg_card(role, lines, color, width=11.8, size=20):
    """A chat message of the prompt: the role in its colour, the text, a stripe on the left (kit message())."""
    label = code(role, 22, color, weight=m.BOLD)
    body = m.VGroup(*[text(ln, size, INK) for ln in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    inner = m.VGroup(label, body).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    group = m.VGroup(box, stripe, inner)
    group.body = body
    group.label = label
    return group


def model_card(name, title, sub, color, fill=CARD, width=2.9, height=2.35, icon_h=0.95):
    box = card(width, height, color, fill)
    body = m.VGroup(icon(name, icon_h), text(title, 22, color, weight=m.BOLD), cite(sub)).arrange(m.DOWN, buff=0.12)
    body.move_to(box)
    return m.VGroup(box, body)


def leash(a, b, waves=7, amp=0.09):
    """A VIOLET spring between two points."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = b - a
    n = np.array([-d[1], d[0], 0.0]) / (np.linalg.norm(d) + 1e-9)
    ts = np.linspace(0, 1, 16 * waves + 1)
    pts = [a + t * d + amp * math.sin(2 * math.pi * waves * t) * n for t in ts]
    return m.VMobject().set_points_smoothly(pts).set_stroke(EXPLORE, 4)


def push(chip_, length, color=SIGNAL, down=False):
    """A PINK push above a token chip: up (more likely) or down (less likely)."""
    x, y = chip_.get_x(), chip_.get_top()[1] + 0.1
    a, b = ([x, y, 0], [x, y + length, 0]) if not down else ([x, y + length, 0], [x, y, 0])
    return m.Arrow(a, b, buff=0, color=color, stroke_width=6, max_tip_length_to_length_ratio=0.35)


# ---------------------------------------------------------------------------- the episode


class Ep20TheModelIsThePolicy(Episode):
    number = 20

    def lesson(self):
        self.new_learner()
        self.tokens_are_actions()
        self.reward_at_the_end()
        self.reinforce_on_tokens()
        self.pipeline()
        self.leash_section()
        self.our_run()
        self.recap()

    # ------------------------------------------------------------------ 1 · a new learner

    def new_learner(self):
        self.next_section("1 · A new learner", notes=(
            "Episode 1's loop (kit agent_world_loop) with Rover; Rover hands over to a small language model (the kit's llm "
            "icon in CORAL: the model is the policy). The arrows are relabelled for text (notes §20, 'same loop, new "
            "world'): the action is the next token, the state is the text so far (the world just appends the token: a "
            "deterministic transition, Ramamurthy et al. 2023 §3.1), and the reward arrives once, at the end (Stiennon et "
            "al. 2020, footnote 8). The module's running example is a real GRPO run on this Mac Studio (Apple M4 Max), "
            "research/experiments/lm_rl.md, used in Episodes 20–26."))
        loop = agent_world_loop(center=(0, -0.05, 0), width=8.8, size=1.1)
        bot = loop.agent[1]
        model = icon(LLM, 1.05).move_to(bot)
        wc = loop.world[1].get_center()
        widths = [1.55, 1.2, 1.45, 0.95]
        lines = m.VGroup(*[m.RoundedRectangle(corner_radius=0.06, width=w, height=0.13, stroke_width=0,
                                              fill_color=STATE, fill_opacity=0.85) for w in widths])
        lines.arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        newest = m.RoundedRectangle(corner_radius=0.06, width=0.4, height=0.13, stroke_width=0, fill_color=ACTION,
                                    fill_opacity=1).next_to(lines[3], m.RIGHT, buff=0.08)
        page = m.VGroup(lines, newest).move_to(wc)
        labs = []
        for grp, words, color in ((loop.action, "next token", ACTION), (loop.state, "the text so far", STATE),
                                  (loop.reward, "reward, at the end", REWARD)):
            labs.append(code(words, 22, color, weight=m.BOLD).move_to(grp[1]))
            assert labs[-1].width < grp[0].get_length() + 0.2
        worlds = m.VGroup(*[pill(s, MUTED, CARD, 20) for s in ("slot machines", "a grid world", "a cart and a pole")])
        worlds.arrange(m.RIGHT, buff=0.3).move_to([0, -2.8, 0])
        spot = loop.agent[0].get_corner(m.DL) + np.array([-0.62, 0.42, 0])
        run_pill = pill("Module 6 · a real run, trained on this Mac", PINK, PINK_FILL, 22).move_to([0, -2.7, 0])
        eps = cite("the running example of episodes 20 to 26").next_to(run_pill, m.DOWN, buff=0.15)

        self.say(
            "[For nineteen episodes, Rover has been our learner]: [slot machines, a grid world, a cart and a pole.]",
            m.AnimationGroup(self.new_header("01 · a new learner", "Same loop, new learner"), m.FadeIn(loop)),
            m.LaggedStart(*[m.FadeIn(w, shift=0.15 * m.UP) for w in worlds], lag_ratio=0.3),
        )
        self.say(
            "[Now Rover hands over], [to a small language model.]",
            bot.animate.scale(0.5).move_to(spot),
            m.FadeIn(model, scale=0.7),
        )
        self.say(
            "[The loop stays the same]: [its action is the next token], [the world is the text so far], "
            "[and the reward comes at the end.]",
            m.Circumscribe(m.VGroup(loop.action, loop.state, loop.reward), color=MUTED, buff=0.12),
            m.AnimationGroup(m.FadeTransform(loop.action[1], labs[0]), pulse(loop.action[0], ACTION)),
            m.AnimationGroup(m.FadeOut(loop.world[1]), m.FadeIn(page), m.FadeTransform(loop.state[1], labs[1]),
                             pulse(loop.state[0], STATE)),
            m.AnimationGroup(m.FadeTransform(loop.reward[1], labs[2]), pulse(loop.reward[0], REWARD)),
        )
        self.say(
            "[And this module's example is real]: [we trained such a model ourselves, here on this Mac.]",
            m.AnimationGroup(m.FadeOut(worlds, shift=0.15 * m.DOWN), m.FadeIn(run_pill, shift=0.15 * m.UP)),
            m.FadeIn(eps, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · tokens are actions

    def tokens_are_actions(self):
        self.next_section("2 · Every token is an action", notes=(
            "A real held-out question of our run (GSM8K test problem 1028, lm_rl.json examples[0]) under the run's prompt "
            "(Qwen2.5 chat template; system = Qwen2.5-Math's instruction, verbatim: '" + SYSTEM + "'). The answer is the "
            "BASE model's own greedy answer (temperature 0: it takes the top token each time; training samples at "
            "temperature 1), verbatim: 181 tokens, finish 'stop', ending '\\(\\boxed{300}\\) dollars per month.'. Drawn as "
            f"its {len(TOM_WORDS)} whitespace-separated words (for example; the real tokenizer splits differently), plus "
            "the chat template's end-of-message token <|im_end|>. The bars are an illustration (the top four candidates "
            f"for the first token: {NEXT}). State = prompt + tokens so far; action = next token; transition = append "
            "(notes §20)."))
        wall = tokens(TOM_WORDS)
        wall.move_to([0, 2.3 - wall.height / 2, 0])
        assert wall.get_left()[0] > -6.6 and wall.get_right()[0] < 6.6 and wall.get_bottom()[1] > -2.3
        K = 10
        first = wall[:K + 1]
        assert all(abs(c.get_y() - wall[0].get_y()) < 1e-6 for c in first)
        row_y = -1.0
        D = row_y - wall[0].get_y()
        row = m.VGroup(*[c.copy().shift(D * m.UP) for c in first])
        rest = wall[K + 1:]
        boxed = wall[1 + TOM_WORDS.index(TOM_BOX)]
        end_chip = wall[-1]

        # the prompt (two messages) and the state outline
        sys_card = msg_card("system", [SYSTEM], MUTED)
        user_card = msg_card("user", Q_LINES, STATE)
        prompt = m.VGroup(sys_card, user_card).arrange(m.DOWN, buff=0.16)
        prompt.move_to([0, 2.3 - prompt.height / 2, 0])
        p_frame = frame([prompt], buff=0.12)
        p_lab = left_at(text("the state", 24, STATE, weight=m.BOLD), -5.9, p_frame.get_bottom()[1] - 0.35)
        src = right_at(cite("GSM8K test problem 1028 · a held-out question of our run"), 5.9, p_lab.get_y())
        user_final, src_final = user_card.get_center(), src.get_center()
        lift = (prompt.get_center()[1] - user_card.get_center()[1]) * m.UP     # alone, the question sits mid-block
        user_card.shift(lift)
        src.shift(lift)

        # the policy, its next-token probabilities, the dice
        pol = model_card(LLM, "the policy", "the model", ACTION, width=2.7, height=2.25).move_to([-4.95, 1.2, 0])
        ax = row[0].get_x()
        in_arrow = arrow([ax, row_y + 0.36, 0], [ax, pol.get_bottom()[1] - 0.04, 0], STATE, buff=0.05)
        bars = Bars(list(NEXT.values()), labels=list(NEXT), top=1.0, bottom=0.0, width=3.6, height=1.6,
                    color=ACTION, gap=0.3, fmt="{:.2f}", size=20, center=(1.55, 0.95, 0))
        b_title = cite("the first token: its top four · illustration").move_to([1.55, 2.25, 0])
        out_arrow = arrow([pol.get_right()[0] + 0.05, 1.0, 0], [bars.get_left()[0] - 0.05, 1.0, 0], ACTION)
        dice = icon(DICE, 0.7).move_to([4.75, 1.05, 0])
        d_lab = text("sample one", 20, ACTION).next_to(dice, m.DOWN, buff=0.18)
        s_lab = left_at(text("the state: the prompt + every token so far", 22, STATE), -6.4, row_y - 0.62)
        greedy = left_at(cite("this answer: greedy (always the top token) · in training, every token is sampled"),
                         -6.4, row_y - 1.15)
        s_frame = frame(row[:1])
        w_frame = frame([wall])

        count = m.VGroup(code("181 tokens", 22, ACTION, weight=m.BOLD),
                         cite("· the real count · drawn here as words, for example"))
        count.arrange(m.RIGHT, buff=0.18)
        left_at(count, -6.4, wall.get_bottom()[1] - 0.45)
        end_lab = cite("end token")
        if end_chip.get_right()[0] + 0.25 + end_lab.width < 6.5:
            end_lab.next_to(end_chip, m.RIGHT, buff=0.22)
        else:
            end_lab.next_to(end_chip, m.DOWN, buff=0.12)
        episode = text("one answer = one episode", 26, INK, weight=m.BOLD)
        left_at(episode, -6.4, count.get_y() - 0.6)

        self.say(
            "[Here's a real question from our run.] [Tom buys a bedroom set for three thousand dollars], "
            "[and puts a thousand from his old one toward it.]",
            m.AnimationGroup(self.new_header("02 · tokens are actions", "Every token is an action"),
                             m.FadeIn(user_card[0]), m.FadeIn(user_card[1]), m.FadeIn(user_card.label), m.FadeIn(src)),
            m.FadeIn(user_card.body[0], shift=0.08 * m.RIGHT),
            m.FadeIn(user_card.body[1], shift=0.08 * m.RIGHT),
        )
        self.say(
            "[Then he pays ten percent a month.] [How much is that per month?]",
            m.Circumscribe(user_card.body[1], color=STATE, buff=0.06),
            m.FadeIn(user_card.body[2], shift=0.08 * m.RIGHT),
        )
        self.say(
            "[With an instruction to reason step by step, and box the answer], [this prompt is the state.]",
            m.AnimationGroup(user_card.animate.move_to(user_final), src.animate.move_to(src_final),
                             m.FadeIn(sys_card, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.Create(p_frame), m.FadeIn(p_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[The policy is the model.] [It reads the state], [and gives every possible next token a probability.]",
            m.AnimationGroup(m.FadeTransform(prompt, row[0]), m.Transform(p_frame, s_frame), m.FadeOut(src),
                             m.FadeTransform(p_lab, s_lab), m.FadeIn(pol, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.GrowArrow(in_arrow), m.Circumscribe(pol, color=ACTION, buff=0.08)),
            m.AnimationGroup(m.GrowArrow(out_arrow), m.FadeIn(bars, shift=0.15 * m.UP), m.FadeIn(b_title)),
        )
        self.say(
            "[Its action is one token, drawn from those probabilities]: [here, the word “To”.]",
            m.AnimationGroup(m.FadeIn(d_lab), m.Succession(m.FadeIn(dice, scale=0.6, run_time=0.4),
                                                           m.Rotate(dice, angle=2 * math.pi, run_time=0.9))),
            m.AnimationGroup(m.Circumscribe(m.VGroup(bars.bars[0], bars.labels[0]), color=ACTION, buff=0.06),
                             m.FadeIn(row[1], target_position=bars.labels[0]), m.Transform(p_frame, frame(row[:2])),
                             m.FadeIn(greedy)),
        )
        self.say(
            "[Now the state is the prompt plus every token so far.] [The policy picks again], "
            "[and again]: [each token, one action.]",
            m.Circumscribe(p_frame, color=STATE, buff=0.06),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(row[i], shift=0.1 * m.DOWN) for i in range(2, 6)], lag_ratio=0.35),
                             m.Transform(p_frame, frame(row[:6]))),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(row[i], shift=0.1 * m.DOWN) for i in range(6, K + 1)], lag_ratio=0.3),
                             m.Transform(p_frame, frame(row[:K + 1]))),
            m.LaggedStart(*[m.Circumscribe(row[i], color=ACTION, buff=0.04) for i in range(1, K + 1)], lag_ratio=0.12),
        )
        self.say(
            "[The whole answer runs a hundred eighty-one tokens], [and ends with a boxed number: three hundred.]",
            m.Succession(
                m.AnimationGroup(m.FadeOut(m.VGroup(pol, in_arrow, out_arrow, bars, b_title, dice, d_lab, s_lab, greedy)),
                                 row.animate.shift(-D * m.UP), m.Transform(p_frame, w_frame), run_time=0.9),
                m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.05 * m.DOWN) for c in rest], lag_ratio=0.03,
                                               run_time=2.6), m.FadeIn(count))),
            m.Circumscribe(boxed, color=INK, buff=0.06),
        )
        self.say(
            "[A special end token closes it.] [One answer, from the first token to the last, is one episode.]",
            m.AnimationGroup(m.Circumscribe(end_chip, color=ACTION, buff=0.06), m.FadeIn(end_lab)),
            m.FadeIn(episode, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · one reward, at the end

    def reward_at_the_end(self):
        self.next_section("3 · One reward, at the end", notes=(
            "The checker is the run's reward (lm_rl.md, 'Reward (the checker)'), re-implemented in this file with the "
            "run's own regexes: 1 if the last number inside the last \\boxed{...} equals the reference (|diff| < 1e-4), "
            "else 0; no partial credit, no format bonus. Tom (test-1028): boxed 300, reference 200 → 0; the base answer "
            f"says, verbatim, '{MISTAKE}' (0.10 × 3000 = 300). Anna (test-1280): the same base model's greedy answer, "
            "173 tokens, boxes 7 = 7 → +1. γ = 1 and one reward at the end, so the return from every token on is that "
            "same reward (Stiennon et al. 2020, footnote 8; InstructGPT: no discount). Our answers average about 300 "
            f"tokens (training samples, steps 1–10: {AVG_LEN:.1f}; greedy before training: {H['greedy_len_before']:.1f}). "
            "Credit assignment with rare rewards: Episode 11. Anna's answer drawn as its words, for example."))
        rows_y = (0.8, -1.3)
        specs = ((TOM_BOX, "Tom's bedroom set · 181 tokens"), (ANNA_BOX, "Anna's puzzle · 173 tokens · the same model"))
        rows, labels = [], []
        for (box_word, lab), y in zip(specs, rows_y):
            r = token_row(["prompt", "To", "MMM", box_word, END], ACTION, size=21, gap=0.1, max_width=30)
            tint(r[0], STATE, TEAL_FILL)
            tint(r[2], EDGE, CARD)
            r[2].remove(r[2][1])
            r[2].add(m.VGroup(*[m.Dot(radius=0.045, color=MUTED) for _ in range(3)]).arrange(m.RIGHT, buff=0.1).move_to(r[2][0]))
            left_at(r, -6.4, y)
            rows.append(r)
            labels.append(left_at(cite(lab), -6.4, y + 0.5))
        ck_box = card(2.9, 3.95, REWARD, CARD).move_to([3.05, -0.1, 0])
        ck_title = m.VGroup(icon(CHECK, 0.42), text("checker", 24, REWARD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.15)
        ck_title.move_to([3.05, ck_box.get_top()[1] - 0.38, 0])
        assert all(r.get_right()[0] < ck_box.get_left()[0] - 0.6 for r in rows)
        cmp_t = code("300 ≠ 200", 24, RED, weight=m.BOLD).move_to([3.05, rows_y[0], 0])
        cmp_a = code("7 = 7", 24, GREEN, weight=m.BOLD).move_to([3.05, rows_y[1], 0])
        ref_t = cite("answer · reference").next_to(cmp_t, m.DOWN, buff=0.14)
        in_t = arrow(rows[0].get_right() + [0.05, 0, 0], [ck_box.get_left()[0], rows_y[0], 0], MUTED)
        in_a = arrow(rows[1].get_right() + [0.05, 0, 0], [ck_box.get_left()[0], rows_y[1], 0], MUTED)
        rw_t = reward_chip(R_WRONG, 30).move_to([5.75, rows_y[0], 0])
        rw_a = reward_chip(R_RIGHT, 30).move_to([5.75, rows_y[1], 0])
        out_t = arrow([ck_box.get_right()[0], rows_y[0], 0], rw_t.get_left(), REWARD)
        out_a = arrow([ck_box.get_right()[0], rows_y[1], 0], rw_a.get_left(), REWARD)
        rw_head = text("reward", 22, REWARD, weight=m.BOLD).move_to([5.75, ck_title.get_y(), 0])
        rule = code("last \\boxed{} number = the reference ?  1 : 0", 20, MUTED).move_to([1.6, -2.62, 0])
        rule_src = cite("the run's checker · no partial credit").next_to(rule, m.DOWN, buff=0.14)
        quote = left_at(text("“" + MISTAKE + "”", 20, INK), -6.4, rows_y[0] - 0.6)
        wrong = left_at(text("10% of the full price, not of the $2,000 left to pay", 20, RED), -6.4, rows_y[0] - 1.02)
        assert quote.get_right()[0] < ck_box.get_left()[0] - 0.3 and wrong.get_bottom()[1] > labels[1].get_top()[1] + 0.12

        # Anna's whole answer, and the reward flowing back
        wall = tokens(ANNA_WORDS)
        wall.move_to([0, 2.3 - wall.height / 2, 0])
        assert wall.get_left()[0] > -6.6 and wall.get_bottom()[1] > -2.3
        last = wall[-1]
        plus = reward_chip(R_RIGHT, 24)
        if last.get_right()[0] + 0.3 + plus.width < 6.5:
            plus.next_to(last, m.RIGHT, buff=0.3)
        else:
            plus.next_to(last, m.DOWN, buff=0.12).align_to(last, m.RIGHT)
        cap1 = m.VGroup(code("173 tokens", 22, ACTION, weight=m.BOLD), cite("· one reward, at the very end"))
        cap1.arrange(m.RIGHT, buff=0.18)
        left_at(cap1, -6.4, wall.get_bottom()[1] - 0.45)
        cap2 = m.VGroup(code("γ = 1:", 22, REWARD, weight=m.BOLD), cite("every token's return is the same +1"))
        cap2.arrange(m.RIGHT, buff=0.18)
        left_at(cap2, -6.4, cap1.get_y() - 0.55)
        ep11 = pill("episode 11 · rare rewards, credit", MUTED, CARD, 20).move_to([0, cap1.get_y(), 0])
        ep11.align_to([6.4, 0, 0], m.RIGHT)
        avg = code(f"≈ {round(AVG_LEN, -2):.0f} tokens per answer", 22, ACTION, weight=m.BOLD)
        right_at(avg, 6.4, cap2.get_y())
        assert ep11.get_left()[0] > cap1.get_right()[0] + 0.3 and avg.get_left()[0] > cap2.get_right()[0] + 0.3
        flow = [c for c in reversed(wall[1:])]

        self.say(
            "[At the end, a checker scores the answer]: [it takes the last boxed number], "
            "[and compares it with the right one.]",
            m.AnimationGroup(self.new_header("03 · reward at the end", "One reward, at the very end"),
                             m.FadeIn(rows[0]), m.FadeIn(labels[0])),
            m.AnimationGroup(m.FadeIn(ck_box), m.FadeIn(ck_title), m.GrowArrow(in_t), m.FadeIn(rule)),
            m.AnimationGroup(m.FadeIn(cmp_t), m.FadeIn(ref_t), m.FadeIn(rule_src)),
        )
        self.say(
            "[Three hundred is not two hundred], [so the reward is zero.]",
            m.Circumscribe(cmp_t, color=RED, buff=0.08),
            m.AnimationGroup(m.GrowArrow(out_t), m.FadeIn(rw_t, scale=0.7), m.FadeIn(rw_head)),
        )
        self.say(
            "[The model took ten percent of the full price], [not of the two thousand left to pay.]",
            m.FadeIn(quote, shift=0.1 * m.UP),
            m.FadeIn(wrong, shift=0.1 * m.UP),
        )
        self.say(
            "[Another question, the same model]: [Anna's puzzle.] [It boxes seven, and that's right]: [plus one.]",
            m.AnimationGroup(m.FadeIn(rows[1]), m.FadeIn(labels[1])),
            m.GrowArrow(in_a),
            m.FadeIn(cmp_a),
            m.AnimationGroup(m.GrowArrow(out_a), m.FadeIn(rw_a, scale=0.7)),
        )
        gone = m.VGroup(*rows, *labels, ck_box, ck_title, cmp_t, cmp_a, ref_t, in_t, in_a, rw_t, rw_a, out_t, out_a,
                        rw_head, rule, rule_src, quote, wrong)
        self.say(
            "[Here is that answer in full]: [a hundred seventy-three tokens], [and one reward at the very end.]",
            m.AnimationGroup(m.FadeOut(gone), m.LaggedStart(*[m.FadeIn(c) for c in wall], lag_ratio=0.015, run_time=2.0)),
            m.FadeIn(cap1),
            m.FadeIn(plus, scale=0.6),
        )
        self.say(
            "[So which tokens earned it?] [With no discount], [the reward flows back to every token, equally.]",
            m.Circumscribe(plus, color=REWARD, buff=0.08),
            m.FadeIn(cap2, shift=0.1 * m.UP),
            m.LaggedStart(*[c[0].animate.set_stroke(REWARD).set_fill(GOLD_FILL, 1) for c in flow], lag_ratio=0.03,
                          run_time=3.2),
        )
        self.say(
            "[That's the credit problem of episode eleven, at its hardest]: "
            "[our answers run about three hundred tokens, for one reward.]",
            m.FadeIn(ep11, shift=0.1 * m.UP),
            m.FadeIn(avg, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · REINFORCE on tokens

    def reinforce_on_tokens(self):
        self.next_section("4 · REINFORCE on tokens", notes=(
            "The chain rule of probability: π(y|x) = Π_t π(y_t | x, y_<t), so log π(y|x) = Σ_t log π(y_t | x, y_<t) "
            "(notes §20 key formula 2). Five consecutive words of Anna's answer with MADE-UP probabilities "
            f"{list(P5)} (the multiset of notes §20's worked example): product {P_ALL:.4f}, sum of logs {LOG_ALL:.4f}. "
            "REINFORCE (Williams 1992; Episode 16) on a whole answer with one reward R: θ ← θ + α R ∇ log π(y|x) = "
            "α R Σ_t ∇ log π(y_t | x, y_<t): every token of the answer is pushed by the same R. With 0/1 rewards a wrong "
            "answer (R = 0) gives plain REINFORCE no gradient at all. A baseline (Episode 17; Williams' own b) of 0.5, "
            f"the mean of the two rewards here, gives advantages {ADV_RIGHT:+.1f} and {ADV_WRONG:+.1f}: the wrong "
            "answer's tokens are pushed down. GRPO (Episode 23) takes the baseline from a group of G = 8 answers to the "
            "same question (our run). Arrow lengths ∝ the push (illustration)."))
        # phase A: the probability of a whole answer
        five = token_row(FIVE, ACTION, size=22, gap=0.6, max_width=30)
        p_row = m.VGroup(*[code(f"{p:.2f}", 22, ACTION) for p in P5])
        l_row = m.VGroup(*[code(minus(f"{v:.2f}"), 22, ACTION) for v in LOG5])
        col_w = [max(c.width, p.width, v.width) for c, p, v in zip(five, p_row, l_row)]
        GAP = 0.6
        x = -0.7 - (sum(col_w) + GAP * (len(col_w) - 1)) / 2
        edges = []
        for c, p, v, w in zip(five, p_row, l_row, col_w):
            c.move_to([x + w / 2, 0.25, 0])
            p.move_to([x + w / 2, 0.95, 0])
            v.move_to([x + w / 2, -0.45, 0])
            edges.append((x, x + w))
            x += w + GAP
        mids = [(a[1] + b[0]) / 2 for a, b in zip(edges, edges[1:])]
        times = m.VGroup(*[code("×", 30, MUTED).move_to([xm, 0.95, 0]) for xm in mids])
        plus = m.VGroup(*[code("+", 28, MUTED).move_to([xm, -0.45, 0]) for xm in mids])
        rx = edges[-1][1] + 0.45
        p_res = left_at(code(f"= {P_ALL:.2f}", 24, ACTION, weight=m.BOLD), rx, 0.95)
        l_res = left_at(code(minus(f"= {LOG_ALL:.2f}"), 24, ACTION, weight=m.BOLD), rx, -0.45)
        p_name = right_at(text("chance", 22, MUTED), five.get_left()[0] - 0.4, 0.95)
        l_name = right_at(text("log", 22, MUTED), five.get_left()[0] - 0.4, -0.45)
        ill = cite("five tokens of Anna's answer · made-up probabilities, for example").move_to([-0.6, -1.1, 0])
        assert p_res.get_right()[0] < 6.5 and p_name.get_left()[0] > -6.6

        f1 = formula([r"\log \pi_\theta(y \mid x)", "=", r"\sum_t", r"\log \pi_\theta(", r"y_t", r"\mid", r"x,\, y_{<t}", r")"],
                     [ACTION, INK, INK, ACTION, ACTION, INK, STATE, ACTION], 48)
        f1.move_to([0, -2.15, 0])
        b_act = m.Brace(f1[4], m.DOWN, buff=0.1, color=ACTION)
        b_st = m.Brace(f1[6], m.DOWN, buff=0.1, color=STATE)
        for b_ in (b_act, b_st):                       # below the whole formula (the sum's subscript hangs lower)
            b_.shift((f1.get_bottom()[1] - 0.08 - b_.get_top()[1]) * m.UP)
        t_act = text("the action", 20, ACTION).next_to(b_act, m.DOWN, buff=0.06)
        t_st = text("the state", 20, STATE).next_to(b_st, m.DOWN, buff=0.06)
        t_st.align_to(t_act, m.UP)
        if t_act.get_right()[0] > t_st.get_left()[0] - 0.2:
            t_act.align_to(b_act, m.RIGHT)
            t_st.align_to(b_st, m.LEFT)

        # phase B: REINFORCE on the two answers
        f2 = formula([r"\theta", r"\leftarrow", r"\theta", "+", r"\alpha", r"\,R", r"\,\sum_t \nabla_\theta",
                      r"\log \pi_\theta(y_t \mid x, y_{<t})"], [ACTION, INK, ACTION, INK, INK, REWARD, SIGNAL, ACTION], 44)
        f2.move_to([-0.9, 1.95, 0])
        f2_cite = m.VGroup(code("REINFORCE", 22, ACTION, weight=m.BOLD), cite("episode 16 · Williams, 1992"))
        f2_cite.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(f2_cite, f2.get_right()[0] + 0.45, 1.95)
        assert f2_cite.get_right()[0] < 6.5
        y_r, y_w = 0.05, -1.95
        right_row = token_row(FIVE, ACTION, size=21, gap=0.33, max_width=30)
        wrong_row = token_row(FIVE_TOM, ACTION, size=21, gap=0.33, max_width=30)
        left_at(right_row, -3.9, y_r)
        left_at(wrong_row, -3.9, y_w)
        r_chip = reward_chip(R_RIGHT, 28).move_to([-5.4, y_r, 0])
        w_chip = reward_chip(R_WRONG, 28).move_to([-5.4, y_w, 0])
        r_lab = cite("Anna: right").next_to(r_chip, m.DOWN, buff=0.15)
        w_lab = cite("Tom: wrong").next_to(w_chip, m.DOWN, buff=0.15)
        SCALE = 0.95
        ups = m.VGroup(*[push(c, SCALE * R_RIGHT) for c in right_row])
        ups_half = m.VGroup(*[push(c, SCALE * ADV_RIGHT) for c in right_row])
        downs = m.VGroup(*[push(c, SCALE * abs(ADV_WRONG), down=True) for c in wrong_row])
        same = left_at(text("the same push for every token", 22, SIGNAL), right_row.get_left()[0], y_r - 0.62)
        none = left_at(text("× 0: no push at all", 22, MUTED), wrong_row.get_right()[0] + 0.45, y_w)
        base_t = left_at(code("baseline b = 0.5", 22, SIGNAL, weight=m.BOLD), right_row.get_right()[0] + 0.45, y_r - 0.05)
        adv_r = left_at(code("1 − 0.5 = +0.5", 22, SIGNAL), right_row.get_right()[0] + 0.45, y_r - 0.5)
        adv_w = left_at(code("0 − 0.5 = −0.5", 22, SIGNAL), wrong_row.get_right()[0] + 0.45, y_w)
        ep_pills = m.VGroup(pill("episode 17 · baselines", MUTED, CARD, 20),
                            pill("episode 23 · a group of 8 answers", PINK, PINK_FILL, 20))
        ep_pills.arrange(m.RIGHT, buff=0.3).move_to([0.9, -3.05, 0])
        b_ill = m.VGroup(cite("for example: the mean"), cite("of the two rewards")).arrange(m.DOWN, buff=0.05, aligned_edge=m.LEFT)
        left_at(b_ill, adv_r.get_left()[0], y_r - 1.05)
        for t_ in (same, none, base_t, adv_r, adv_w):
            assert t_.get_right()[0] < 6.5, t_.text

        self.say(
            "[How likely was a whole answer? Take five of its tokens, for example]: [multiply their chances], "
            "[or add up their logs.]",
            m.AnimationGroup(self.new_header("04 · REINFORCE on tokens", "Policy gradient, token by token"),
                             m.FadeIn(five, shift=0.1 * m.UP), m.FadeIn(ill)),
            m.AnimationGroup(m.FadeIn(p_name), m.LaggedStart(*[m.FadeIn(p) for p in p_row], lag_ratio=0.2),
                             m.FadeIn(times), m.FadeIn(p_res, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(l_name), m.LaggedStart(*[m.FadeIn(v) for v in l_row], lag_ratio=0.2),
                             m.FadeIn(plus), m.FadeIn(l_res, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[So the log-probability of an answer is a sum over its tokens]: [each one an action], [in its own state.]",
            m.FadeIn(f1, shift=0.15 * m.UP),
            m.AnimationGroup(m.GrowFromCenter(b_act), m.FadeIn(t_act)),
            m.AnimationGroup(m.GrowFromCenter(b_st), m.FadeIn(t_st)),
        )
        self.say(
            "[Now episode sixteen's REINFORCE applies as it is]: [push up that log-probability], "
            "[in proportion to the reward.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(five, p_row, l_row, times, plus, p_res, l_res, p_name, l_name, ill, f1,
                                                b_act, t_act, b_st, t_st)),
                             m.FadeIn(f2, shift=0.1 * m.UP), m.FadeIn(f2_cite, shift=0.1 * m.LEFT)),
            m.Circumscribe(f2[6:], color=SIGNAL, buff=0.06),
            m.Circumscribe(f2[5], color=REWARD, buff=0.06),
        )
        self.say(
            "[A right answer earns one]: [every one of its tokens gets the same push up.]",
            m.AnimationGroup(m.FadeIn(right_row, shift=0.1 * m.UP), m.FadeIn(r_chip, scale=0.7), m.FadeIn(r_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in ups], lag_ratio=0.12), m.FadeIn(same)),
        )
        self.say(
            "[A wrong answer earns zero], [and plain REINFORCE learns nothing from it.]",
            m.AnimationGroup(m.FadeIn(wrong_row, shift=0.1 * m.UP), m.FadeIn(w_chip, scale=0.7), m.FadeIn(w_lab)),
            m.FadeIn(none, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Subtract a baseline, as in episode seventeen], [and the wrong answer's tokens get pushed down.] "
            "[Episode twenty-three takes that baseline from a group of answers.]",
            m.AnimationGroup(m.FadeOut(same), m.FadeIn(base_t), m.FadeIn(adv_r), m.Transform(ups, ups_half), m.FadeIn(b_ill)),
            m.AnimationGroup(m.FadeOut(none), m.FadeIn(adv_w), m.LaggedStart(*[m.GrowArrow(a) for a in downs], lag_ratio=0.12)),
            m.FadeIn(ep_pills, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · the pipeline

    def pipeline(self):
        self.next_section("5 · Pretraining, fine-tuning, reward", notes=(
            "The pipeline as in InstructGPT (Ouyang et al. 2022, Fig. 2): pretraining (next-token prediction on web-scale "
            "text) → supervised fine-tuning on human demonstrations → RL against a reward (there a reward model trained "
            "on human comparisons, Episode 21; DPO folds the last two steps into one loss, Episode 22; RLVR swaps the "
            "reward model for a checker, Episode 24). Why RL after imitation (notes §20): SFT can only push probability "
            "UP (ORPO, Hong et al. 2024, Fig. 3: chosen and rejected log-probabilities rise together during SFT); "
            "rewards can push bad outputs down, and RL trains on the model's own samples (exposure bias, Ranzato et al. "
            "2016). Our run starts from Qwen2.5-0.5B-Instruct, which its makers already pretrained and post-trained to "
            "follow instructions; the run adds RL with a checker as the reward (lm_rl.md)."))
        xs = (-4.5, 0.0, 4.5)
        y = 0.4
        specs = ((BOOK, "Pretraining", ("predict the next token", "of a vast amount of text"), EDGE),
                 (PEOPLE, "Fine-tuning", ("supervised: copy good", "answers people wrote"), EDGE),
                 ("trophy", "Reinforcement learning", ("its own answers,", "scored by a reward"), REWARD))
        nodes = []
        for x, (ic, title, subs, color) in zip(xs, specs):
            box = card(3.85, 2.75, color, CARD).move_to([x, y, 0])
            body = m.VGroup(icon(ic, 0.8), text(title, 22, INK, weight=m.BOLD),
                            m.VGroup(*[cite(s) for s in subs]).arrange(m.DOWN, buff=0.06)).arrange(m.DOWN, buff=0.14)
            body.move_to(box)
            assert body.width < 3.6, title
            nodes.append(m.VGroup(box, body))
        links = [arrow(nodes[i][0].get_right(), nodes[i + 1][0].get_left(), MUTED, buff=0.06) for i in range(2)]
        n_cite = cite("InstructGPT's recipe · Ouyang et al., 2022, Fig. 2").move_to([0, -3.25, 0])
        sub3 = cite("the reward: people's preferences (episode 21), or a checker, like ours (episode 24)")
        sub3.move_to([0, -2.6, 0])
        push_y = -1.6
        up2 = m.Arrow([xs[1] - 1.45, push_y - 0.3, 0], [xs[1] - 1.45, push_y + 0.3, 0], buff=0, color=SIGNAL,
                      stroke_width=6, max_tip_length_to_length_ratio=0.35)
        up2_t = m.VGroup(text("pushes good answers up", 20, SIGNAL), cite("only ever up"))
        up2_t.arrange(m.DOWN, buff=0.05, aligned_edge=m.LEFT)
        left_at(up2_t, xs[1] - 1.15, push_y)
        both = m.VGroup(m.Arrow([0, -0.3, 0], [0, 0.3, 0], buff=0, color=SIGNAL, stroke_width=6, max_tip_length_to_length_ratio=0.35),
                        m.Arrow([0.32, 0.3, 0], [0.32, -0.3, 0], buff=0, color=SIGNAL, stroke_width=6, max_tip_length_to_length_ratio=0.35))
        both.move_to([2.41, push_y, 0])
        both_t = m.VGroup(text("and its own mistakes down", 20, SIGNAL), cite("learning from its own answers"))
        both_t.arrange(m.DOWN, buff=0.05, aligned_edge=m.LEFT)
        left_at(both_t, 2.8, push_y)
        assert both_t.get_right()[0] < 6.5 and up2_t.get_right()[0] < both.get_left()[0] - 0.3
        assert sub3.width < 12.6 and both_t.get_bottom()[1] > sub3.get_top()[1] + 0.2
        brace = m.Brace(m.VGroup(nodes[0], nodes[1]), m.UP, buff=0.1, color=MUTED)
        made = cite("done by its makers, before our run").next_to(brace, m.UP, buff=0.06)
        ours = pill("our run adds this", PINK, PINK_FILL, 20).move_to([xs[2], made.get_y(), 0])
        assert made.get_top()[1] < 2.8 and ours.get_top()[1] < 2.8

        self.say(
            "[Where does a model like this come from?] [Three stages.]",
            self.new_header("05 · the pipeline", "Pretraining, fine-tuning, then reward"),
            m.LaggedStart(*[m.FadeIn(n[0], shift=0.15 * m.UP) for n in nodes], *[m.GrowArrow(a) for a in links],
                          lag_ratio=0.15),
        )
        self.say(
            "[First, pretraining]: [predict the next token of a vast amount of text.]",
            m.FadeIn(nodes[0][1][:2], shift=0.1 * m.UP),
            m.FadeIn(nodes[0][1][2], shift=0.1 * m.UP),
        )
        self.say(
            "[Second, supervised fine-tuning]: [copy good answers written by people], [and learn to follow instructions.]",
            m.FadeIn(nodes[1][1][:2], shift=0.1 * m.UP),
            m.FadeIn(nodes[1][1][2], shift=0.1 * m.UP),
            m.FadeIn(n_cite),
        )
        self.say(
            "[Third, reinforcement learning]: [the model answers on its own], [and a reward says how it did.]",
            m.FadeIn(nodes[2][1][:2], shift=0.1 * m.UP),
            m.FadeIn(nodes[2][1][2], shift=0.1 * m.UP),
            m.FadeIn(sub3, shift=0.1 * m.UP),
        )
        self.say(
            "[Copying can only push good answers up.] [A reward can also push the model's own mistakes down.]",
            m.AnimationGroup(m.GrowArrow(up2), m.FadeIn(up2_t)),
            m.AnimationGroup(m.GrowArrow(both[0]), m.GrowArrow(both[1]), m.FadeIn(both_t)),
        )
        self.say(
            "[Our model arrives with the first two stages done]; [our run adds the third.]",
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(made)),
            m.AnimationGroup(m.FadeIn(ours, shift=0.1 * m.DOWN), m.Circumscribe(nodes[2][0], color=PINK, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the KL leash

    def leash_section(self):
        self.next_section("6 · The KL leash", notes=(
            "The KL-regularised objective (DPO eq. 3; Ziegler et al. 2019 eq. 2 writes it per answer as R = r − β log(π/ρ)): "
            "maximise E[r] − β KL(π_θ ‖ π_ref). Reasons in the sources (notes §20): an entropy bonus, keeping the policy "
            "where the reward is valid, keeping text fluent (Stiennon et al. 2020 §3.4; Ziegler et al. 2019 §2). The "
            "drifting picture is an illustration. In InstructGPT the reference is the SFT model and β = 0.02 (a per-token "
            "penalty in the reward); in our run the reference is the frozen starting model (no adapter) and the KL is a "
            "separate term in the loss, as in DeepSeekMath (β = 0.04, Shao et al. 2024), estimated per token with k3 "
            "(Schulman 2020): ratio = π_ref/π_θ, k3 = ratio − 1 − log ratio, never negative. One token, for example "
            f"(notes §23 worked example): π_θ = {P_POL}, π_ref = {P_REF} → ratio 0.8, k3 = {K3:.4f} (rl_reference.kl_k3). "
            f"Measured here (lm_rl.json curves.kl, one run, 200 steps): KL rose to about 0.02 per token within 10 steps and "
            f"stayed there (mean over steps 10–199: {KL[10:].mean():.4f}; last 10 steps: {H['kl_last10']}); plotted: the "
            "10-step moving average (curves.kl_ma10)."))
        pol = model_card(LLM, "the policy", "learning", ACTION, width=2.7, height=2.25)
        ref = model_card(LLM_REF, "a frozen copy", "the starting model", MUTED, width=2.7, height=2.25)
        start = np.array([0.0, 0.95, 0])
        pol.move_to(start)
        ref.move_to([2.9, 0.95, 0])
        far, near = np.array([-3.9, 0.95, 0]), np.array([-2.6, 0.95, 0])
        pull = arrow([-4.3, -0.75, 0], [-5.9, -0.75, 0], REWARD)
        pull_t = text("reward pulls", 20, REWARD).next_to(pull, m.RIGHT, buff=0.15)
        odd = pill("odd text that still scores", RED, RED_FILL, 20).move_to([-3.9, -1.4, 0])
        odd_ill = cite("for example").next_to(odd, m.DOWN, buff=0.1)

        def tie(p):
            return leash(np.array(p) + [1.35 + 0.05, 0, 0], ref.get_left() + [-0.05, 0, 0])

        rope_far = tie(far)
        rope = tie(near)
        rope_t = text("the KL leash", 22, EXPLORE, weight=m.BOLD).move_to([(near[0] + 1.35 + ref.get_left()[0]) / 2, 2.35, 0])
        obj = formula([r"\max_\theta", r"\;\mathbb{E}\big[", r"\,r\,", r"\big]", "-", r"\beta", r"\,\mathrm{KL}(",
                       r"\pi_\theta", r"\,\|\,", r"\pi_{\mathrm{ref}}", ")"],
                      [INK, INK, REWARD, INK, INK, EXPLORE, EXPLORE, ACTION, EXPLORE, MUTED, EXPLORE], 48)
        obj.move_to([0, -1.55, 0])
        ob_r = m.Brace(obj[2], m.DOWN, buff=0.16, color=REWARD)
        ob_k = m.Brace(obj[5:], m.DOWN, buff=0.16, color=EXPLORE)
        ob_y = min(ob_r.get_bottom()[1], ob_k.get_bottom()[1]) - 0.24
        ob_r_t = right_at(text("reward", 20, REWARD), ob_r.get_right()[0] + 0.2, ob_y)
        ob_k_t = left_at(text("how far the policy drifted from the copy", 20, EXPLORE), ob_k.get_left()[0], ob_y)
        assert ob_r_t.get_right()[0] < ob_k_t.get_left()[0] - 0.2 and ob_k_t.get_right()[0] < 6.5
        ob_cite = cite("Ziegler et al., 2019 · Ouyang et al., 2022").move_to([0, -3.2, 0])

        # phase B: one token's fee, and the leash in our run
        obj_top = obj.copy().scale(0.8).move_to([0, 1.95, 0])
        tok = token_row(["pay"], ACTION, size=22)[0]
        bx = -4.2
        b_pol = m.Rectangle(width=3.0 * P_POL, height=0.32).set_stroke(ACTION, 2).set_fill(CORAL_FILL, 1)
        b_ref = m.Rectangle(width=3.0 * P_REF, height=0.32).set_stroke(MUTED, 2).set_fill(CARD, 1)
        left_at(b_pol, bx, 0.75)
        left_at(b_ref, bx, 0.3)
        v_pol = code(f"{P_POL:.2f}", 20, ACTION).next_to(b_pol, m.RIGHT, buff=0.15)
        v_ref = code(f"{P_REF:.2f}", 20, MUTED).next_to(b_ref, m.RIGHT, buff=0.15)
        n_pol = right_at(text("policy", 20, ACTION), bx - 0.15, 0.75)
        n_ref = right_at(text("copy", 20, MUTED), bx - 0.15, 0.3)
        tok.move_to([n_pol.get_left()[0] - 0.55, 0.52, 0])
        k3a = left_at(code("ratio = 0.40 / 0.50 = 0.8", 20, INK), -6.3, -0.45)
        k3b = left_at(code("fee = ratio − 1 − log ratio", 20, EXPLORE), -6.3, -0.95)
        k3c = left_at(code(f"    = {K3:.3f}", 24, EXPLORE, weight=m.BOLD), -6.3, -1.5)
        k3_src = m.VGroup(cite("the k3 estimate · Schulman, 2020"), cite("never negative · for example"))
        k3_src.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        left_at(k3_src, -6.3, -2.3)
        plot = Plot((0, 200), (0, 0.03), width=5.0, height=2.2, center=(3.3, -0.72, 0), x_ticks=[0, 100, 200],
                    y_ticks=[0, 0.01, 0.02, 0.03], x_label="training step", y_label="KL to the copy, per token",
                    y_fmt="{:.2f}")
        curve = plot.line(STEPS, KL_MA, EXPLORE, 4)
        p_src = m.VGroup(cite("trained on this Mac · one run · 10-step average"),
                         cite("β = 0.04, as in DeepSeekMath (Shao et al., 2024)"))
        p_src.arrange(m.DOWN, buff=0.06).next_to(plot, m.DOWN, buff=0.12)
        assert p_src.get_bottom()[1] > -3.55 and plot.get_top()[1] < obj_top.get_bottom()[1] - 0.2
        assert plot.get_left()[0] > k3a.get_right()[0] + 0.3

        self.say(
            "[There's one danger.] [Chase the reward alone], [and the model can drift into strange text that happens to score.]",
            m.AnimationGroup(self.new_header("06 · the KL leash", "A leash to where it started"), m.FadeIn(pol, scale=0.9)),
            m.AnimationGroup(m.GrowArrow(pull), m.FadeIn(pull_t), pol.animate.move_to(far)),
            m.AnimationGroup(m.FadeIn(odd, shift=0.1 * m.UP), m.FadeIn(odd_ill)),
        )
        self.say(
            "[So we keep a frozen copy of the starting model], [and tie the policy to it with a leash.]",
            m.AnimationGroup(m.FadeIn(ref, target_position=start), m.FadeOut(odd), m.FadeOut(odd_ill)),
            m.AnimationGroup(m.Create(rope_far), m.FadeIn(rope_t)),
        )
        self.say(
            "[The goal becomes the reward], [minus beta times the KL divergence]: [how far the policy has drifted from that copy.]",
            m.AnimationGroup(m.FadeIn(obj[:4], shift=0.1 * m.UP), m.GrowFromCenter(ob_r), m.FadeIn(ob_r_t)),
            m.AnimationGroup(m.FadeIn(obj[4:], shift=0.1 * m.UP), m.FadeIn(ob_cite),
                             pol.animate.move_to(near), m.Transform(rope_far, rope)),
            m.AnimationGroup(m.GrowFromCenter(ob_k), m.FadeIn(ob_k_t)),
        )
        self.say(
            "[It's measured token by token. For example], [the policy gives a token fifty percent, the copy forty]: "
            "[a fee of about zero point zero two.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(pol, ref, rope_far, rope_t, pull, pull_t, ob_r, ob_r_t, ob_k, ob_k_t, ob_cite)),
                             m.Transform(obj, obj_top), m.FadeIn(tok, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(n_pol), m.GrowFromEdge(b_pol, m.LEFT), m.FadeIn(v_pol),
                             m.FadeIn(n_ref), m.GrowFromEdge(b_ref, m.LEFT), m.FadeIn(v_ref), m.FadeIn(k3a)),
            m.AnimationGroup(m.FadeIn(k3b), m.FadeIn(k3c), m.FadeIn(k3_src)),
        )
        self.say(
            "[A bigger beta is a shorter leash.] [Ours was zero point zero four], "
            "[and the drift soon settled near zero point zero two per token.]",
            m.Circumscribe(obj[5], color=EXPLORE, buff=0.08),
            m.AnimationGroup(m.FadeIn(plot), m.FadeIn(p_src)),
            m.Create(curve, run_time=2.4),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · our run

    def our_run(self):
        self.next_section("7 · Our run", notes=(
            "research/experiments/lm_rl.md + lm_rl.json (this Mac Studio, Apple M4 Max, 64 GB, MLX; 2026-10-25; one run, "
            f"one seed): {H['model']} ({H['params_base_M']:.0f}M parameters), {H['finetune']}: "
            f"{H['trainable_params_M']}M trainable ({TRAIN_PCT:.2f}%); GSM8K (Cobbe et al. 2021; files: 7,473 train / "
            "1,319 test): 1,600 training problems used once each (8 per step × 200 steps), a fixed 500 of the 1,319 test "
            f"problems held out; G = {H['G']} answers per question → {H['completions_sampled']:,} answers, 400 AdamW updates; "
            f"clip ε = {H['clip_eps']}, β = {H['beta_kl']} (k3, frozen base), lr {H['lr']}; {H['wall_time_train_min']} min "
            f"of training (+{H['wall_time_eval_min']} min of evaluation), {H['sec_per_step']} s per step; sampling "
            f"≈ {TOKS:,.0f} tokens/s (mean over steps of curves.gen_tokens_per_s); peak GPU memory ≈ {MEM:.1f} GB "
            f"(median over steps; max {H['peak_mem_gb']}). The results (pass@1 before/after etc.) are deliberately NOT "
            "shown: Episodes 24 and 26. Group-relative advantages: Episode 23. Sister course: 'Agents, Animated' "
            "Episode 30 trained agents with RL in the same way (sixteen rollouts, group-relative advantages)."))
        mc = card(4.1, 4.3, ACTION, CARD).move_to([-4.3, 0.15, 0])
        m_icon = icon(LLM, 1.1)
        m_name = code("Qwen2.5-0.5B-Instruct", 20, INK, weight=m.BOLD)
        m_params = m.VGroup(title_text(f"{H['params_base_M']:.0f}M", 44, ACTION), text("parameters", 22, MUTED))
        m_params.arrange(m.RIGHT, buff=0.15, aligned_edge=m.DOWN)
        m_body = m.VGroup(m_icon, m_name, m_params).arrange(m.DOWN, buff=0.2)
        m_body.move_to(mc).shift(0.45 * m.UP)
        sep = m.Line([mc.get_left()[0] + 0.3, 0, 0], [mc.get_right()[0] - 0.3, 0, 0]).set_stroke(EDGE, 2)
        sep.set_y(m_body.get_bottom()[1] - 0.22)
        adapter = m.VGroup(text("+ a small adapter (LoRA)", 20, MUTED),
                           code(f"{H['trainable_params_M']}M trainable · {TRAIN_PCT:.1f}%", 20, SIGNAL, weight=m.BOLD))
        adapter.arrange(m.DOWN, buff=0.1).next_to(sep, m.DOWN, buff=0.2)
        assert adapter.get_bottom()[1] > mc.get_bottom()[1] + 0.1 and m_name.width < 3.8

        rx = -1.6
        rows_spec = (("1,600 maths problems", "GSM8K, grade school · 500 more held out, never trained on", STATE),
                     ("8 questions × 8 answers per step", "200 steps · 12,800 answers, each one checked", ACTION),
                     (f"{H['wall_time_train_min']} minutes of training",
                      f"this Mac (M4 Max) · ≈ {round(TOKS / 50) * 50:,.0f} tokens/s · ≈ {MEM:.0f} GB", INK))
        info = []
        for (big, small, color), y in zip(rows_spec, (1.65, 0.45, -0.75)):
            g = m.VGroup(title_text(big, 30, color), cite(small)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            left_at(g, rx, y)
            assert g.get_right()[0] < 6.5, big
            info.append(g)
        assert f"{H['completions_sampled']:,}" in rows_spec[1][1] and f"{H['steps']}" in rows_spec[1][1]
        nexts = m.VGroup(pill("how the 8 answers are compared · episode 23", PINK, PINK_FILL, 20),
                         pill("did it learn? · episodes 24 and 26", PINK, PINK_FILL, 20)).arrange(m.RIGHT, buff=0.3)
        nexts.move_to([0, -2.55, 0])
        assert nexts.width < 12.8
        src = cite("trained on this Mac · one run · lm_rl.md · agents trained this way: Agents, Animated, episode 30")
        src.move_to([0, -3.25, 0])
        assert src.width < 12.8

        self.say(
            "[All of this is our own run]: [the running example for this whole module.]",
            m.AnimationGroup(self.new_header("07 · our run", "The module's running example"), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(mc), m.FadeIn(m_icon, scale=0.7)),
        )
        self.say(
            "[The model is small and open]: [about half a billion parameters.] [We train only a small adapter], "
            "[under two percent of them.]",
            m.FadeIn(m_name, shift=0.1 * m.UP),
            m.FadeIn(m_params, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(sep), m.FadeIn(adapter[0])),
            m.FadeIn(adapter[1], shift=0.1 * m.UP),
        )
        self.say(
            "[It practised on sixteen hundred grade-school maths problems], [eight answers each], [for two hundred steps.]",
            m.FadeIn(info[0], shift=0.1 * m.LEFT),
            m.FadeIn(info[1][0], shift=0.1 * m.LEFT),
            m.FadeIn(info[1][1], shift=0.1 * m.LEFT),
        )
        self.say(
            "[That's twelve thousand eight hundred answers], [in seventy-two minutes of training on this Mac.]",
            m.Circumscribe(info[1][1], color=ACTION, buff=0.06),
            m.FadeIn(info[2], shift=0.1 * m.LEFT),
        )
        self.say(
            "[How the eight answers are compared is episode twenty-three.] "
            "[Whether it learned, we'll see in episodes twenty-four and twenty-six.]",
            m.FadeIn(nexts[0], shift=0.1 * m.UP),
            m.FadeIn(nexts[1], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            "Tokens (each one an action of the policy, the state is the prompt plus the tokens so far), reward (once, at "
            "the end, for the whole answer: every token gets the same return; log π(y|x) = Σ_t log π(y_t | x, y_<t)), "
            "leash (reward − β·KL to a frozen copy of the starting model; β = 0.04 in our run)."))
        halo = glow(ACTION, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Tokens", 54, ACTION), title_text("Reward", 54, REWARD), title_text("Leash", 54, EXPLORE))
        words.arrange(m.RIGHT, buff=1.5).move_to([0, 1.1, 0])
        on_baseline(words)
        subs = m.VGroup(*[m.VGroup(text(a, 22, MUTED), text(b, 22, MUTED)).arrange(m.DOWN, buff=0.08)
                          for a, b in (("each one an action", "of the policy"), ("once, at the end,", "for every token"),
                                       ("stay near", "where it started"))])
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), 0, 0]).align_to([0, words[1].get_bottom()[1] - 0.3, 0], m.UP)
        assert all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        res = formula([r"\log \pi_\theta(y \mid x)", "=", r"\sum_t", r"\log \pi_\theta(y_t \mid x, y_{<t})"],
                      [ACTION, INK, INK, ACTION], 40)
        res.move_to([0, -1.5, 0])
        model = icon(LLM, 0.7)
        bot = rover(0.5)
        pair = m.VGroup(bot, m.Arrow([0, 0, 0], [0.8, 0, 0], buff=0, color=MUTED, stroke_width=5,
                                     max_tip_length_to_length_ratio=0.3), model)
        pair.arrange(m.RIGHT, buff=0.3).move_to([0, -2.85, 0])

        self.say(
            "[To sum up]: [tokens], [reward], [leash.]",
            m.AnimationGroup(self.new_header("08 · recap", "Tokens, reward, leash"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Each token is an action of the policy], [one reward at the end trains them all], "
            "[and a leash keeps the model near where it started.]",
            m.AnimationGroup(m.FadeIn(res, shift=0.1 * m.UP), m.Circumscribe(words[0], color=ACTION, buff=0.12)),
            m.Circumscribe(words[1], color=REWARD, buff=0.12),
            m.AnimationGroup(m.Circumscribe(words[2], color=EXPLORE, buff=0.12), m.FadeIn(pair, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()
