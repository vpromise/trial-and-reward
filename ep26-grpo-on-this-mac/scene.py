"""Trial and Reward · Episode 26 · GRPO on This Mac (Module 6 · Teaching Language Models with Reward) — the FINALE

The whole loop, trained for real, and what every piece did. Built on OUR GRPO RUN on this Mac Studio (Apple M4 Max,
64 GB, MLX; research/experiments/lm_rl.md + lm_rl.json; one run, one seed, 2026-10-25): Qwen2.5-0.5B-Instruct (494M
parameters, frozen) with a LoRA adapter (rank 16, 8.8M trainable), GSM8K (1,600 training problems; a fixed 500 of the
1,319 test problems held out), the run's prompt and checker (1 if the last number in the last \\boxed{} equals the
reference, else 0), G = 8 answers × 8 questions per step, clip ε = 0.2, KL β = 0.04 (k3) to the frozen base, lr 2e-5,
200 steps. Sections: the run's card (Rover hands over to the model); Episode 1's loop again, now with a language model,
one real step (step 108) through sample a group → check → advantages → update; where the time went per step (run log:
sampling 9.2 s, log-probs 5.0 s, update 7.4 s of 21.5 s; ≈ 2,150 tokens/s sampling; 71.7 min; ≈ 25 GB), against
"over 90 %" on big clusters (OpenRLHF); the training reward curve (0.476 → 0.570, first vs last 25 steps) with the
held-out pass@1 at the five evaluations (38.3 % → 46.2 %); test-1028 before and after, verbatim lines (Episode 20 showed
the base answer; Episode 24 left the reveal for this episode); the regression test-1280 and the 60 / 39 greedy flips;
one seed. Then the course revisited: one card per module (kit.MODULES colours) with its episodes, key idea, and the two
questions (credit, exploration) answered six ways. Recap: act, reward, learn (Episode 1's words). The kit's
next-episode outro is replaced by a course-closing card.

Every number on screen is computed below from rl_reference.py and asserted, loaded with measured(...) (or read from the
run's own logs in research/experiments/logs/, cross-checked against lm_rl.json), or cited.

Sources (research/notes.md §26, §23, §24, Appendix A rows 23 and 26, Appendix C row 26; read October 2026):
- research/experiments/lm_rl.md + lm_rl.json: headline (model, params, G, β, ε, lr, steps, train_questions_seen,
  completions_sampled, wall_time_train_min 71.7, wall_time_eval_min 37.4, sec_per_step 21.5, train_reward_first25 /
  last25, pass_at_k_before / after, paired_se, heldout_flips), setup (prompt, regexes), curves (reward, reward_ma10,
  sample_s, logprob_s, update_s, gen_tokens_per_s, peak_mem_gb), evals (pass@1 at steps 0, 50, 100, 150, 200),
  examples (test-1028 and test-1280: greedy answers, verbatim, re-decoded at batch 64; token counts).
- research/experiments/logs/lm_grpo.jsonl (per step: seconds, new tokens), lm_grpo_groups.jsonl (step 108: all 64
  rewards and advantages), lm_grpo_eval.jsonl (per held-out problem: greedy correctness and samples right, steps 0 and
  200), lm_grpo_config.json.
- Hu et al. 2024, OpenRLHF (arXiv 2405.11143): "the inference phase often accounts for over 90% of the total RLHF (or
  RLVR) runtime" (notes §26). Cobbe et al. 2021 (GSM8K). Shao et al. 2024 (DeepSeekMath, GRPO).
- The course: COURSE.md (modules, episodes), kit.MODULES / EPISODES. Sister course: "Tokens per Second" (inference).

Illustrative: none of the data. The Anna question is shortened on screen (labelled); answers are shown as verbatim
lines with LaTeX markup removed (\\times as ×, \\[ \\] dropped, \\frac{10}{2} as 10/2, \\text{ hours} as " hours").

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep26"
LM = "lm_rl"
LOGS = EXPERIMENTS / "logs"


def pct(x, places=1):
    """A fraction as a percentage, rounded half up (as lm_rl.md prints it)."""
    q = Decimal(repr(round(100 * x, 9))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{q}%"


# ---------------------------------------------------------------------------- the run (measured on this Mac)

H = measured(LM, "headline")
SETUP = measured(LM, "setup")
CURVES = measured(LM, "curves")
EVALS = measured(LM, "evals")
EX = {e["id"]: e for e in measured(LM, "examples")}
CONFIG = json.loads((LOGS / "lm_grpo_config.json").read_text())

G, Q_PER_STEP, STEPS = H["G"], H["questions_per_step"], H["steps"]
assert (G, Q_PER_STEP, STEPS, H["completions_per_step"]) == (8, 8, 200, 64)            # "eight answers to each of eight"
assert (H["clip_eps"], H["beta_kl"], H["lr"]) == (0.2, 0.04, 2e-5)                    # "zero point two", "zero point zero four"
assert H["model"] == "Qwen2.5-0.5B-Instruct" and H["params_base_M"] == 494.0           # "half a billion parameters"
assert H["trainable_params_M"] == 8.8 and CONFIG["trainable_params"] == 8_798_208      # "about nine million"
assert H["train_questions_seen"] == 1600 == STEPS * Q_PER_STEP                          # "sixteen hundred problems"
assert H["test_questions"] == 500 and "1,319" in SETUP["data"]                          # "five hundred others"
assert H["completions_sampled"] == 12_800 == STEPS * Q_PER_STEP * G
assert SETUP["args"]["lora_rank"] == 16 and SETUP["args"]["temp"] == 1.0 and SETUP["args"]["minibatches"] == 2

# the checker (the run's own regexes)
BOXED = re.compile(SETUP["boxed_regex"])
NUMBER = re.compile(SETUP["number_regex"])


def checker(answer, reference):
    """1 if the last number inside the last \\boxed{...} equals the reference, else 0."""
    boxes = BOXED.findall(answer)
    nums = NUMBER.findall(boxes[-1]) if boxes else []
    return float(bool(nums) and abs(float(nums[-1].replace(",", "")) - reference) < 1e-4)


# the run log: where the time went (per step)
RUN = [json.loads(line) for line in (LOGS / "lm_grpo.jsonl").read_text().splitlines() if line.strip()]
assert len(RUN) == STEPS and [r["step"] for r in RUN] == list(range(STEPS))
SAMPLE_S = float(np.mean([r["sample_s"] for r in RUN]))
LOGP_S = float(np.mean([r["logprob_s"] for r in RUN]))
UPDATE_S = float(np.mean([r["update_s"] for r in RUN]))
STEP_S = SAMPLE_S + LOGP_S + UPDATE_S
assert abs(SAMPLE_S - np.mean(CURVES["sample_s"])) < 1e-3 and abs(UPDATE_S - np.mean(CURVES["update_s"])) < 1e-3
assert f"{STEP_S:.1f}" == "21.5" == f"{H['sec_per_step']:.1f}"                         # "twenty-one and a half seconds"
assert (f"{SAMPLE_S:.1f}", f"{LOGP_S:.1f}", f"{UPDATE_S:.1f}") == ("9.2", "5.0", "7.4")  # "nine ... five ... seven"
SHARES = [round(100 * s / STEP_S) for s in (SAMPLE_S, LOGP_S, UPDATE_S)]
assert SHARES == [43, 23, 34] and sum(SHARES) == 100
TRAIN_MIN = sum(r["sample_s"] + r["logprob_s"] + r["update_s"] for r in RUN) / 60
assert f"{TRAIN_MIN:.1f}" == "71.7" == f"{H['wall_time_train_min']:.1f}" and round(TRAIN_MIN) == 72   # "seventy-two minutes"
EVAL_MIN = H["wall_time_eval_min"]
assert EVAL_MIN == 37.4
NEW_TOK = float(np.mean([r["gen_tokens"] for r in RUN]))
assert round(NEW_TOK, -2) == 19_500                                                   # "about twenty thousand"
TOK_S = float(np.mean(CURVES["gen_tokens_per_s"]))                                    # mean of the 200 steps' rates
TOK_S_ALL = sum(r["gen_tokens"] for r in RUN) / sum(r["sample_s"] for r in RUN)       # all tokens ÷ all sampling time
assert round(TOK_S / 50) * 50 == 2150 and abs(TOK_S_ALL - H["gen_tokens_per_s"]) < 1   # "≈ 2,150", headline 2,127
assert min(TOK_S, TOK_S_ALL) > 2000                                                   # "over two thousand a second"
MEM = float(np.median(CURVES["peak_mem_gb"]))
MEM_MAX = float(np.max(CURVES["peak_mem_gb"]))
assert round(MEM) == 25 and f"{MEM_MAX:.1f}" == "28.8" == f"{H['peak_mem_gb']:.1f}"     # "about twenty-five gigabytes"
BIG_CLUSTER = 90                                                                      # Hu et al. 2024: "over 90%"

# one real step (step 108 of the log, steps counted from 0): 8 questions × 8 answers
GROUP_LOG = [json.loads(line) for line in (LOGS / "lm_grpo_groups.jsonl").read_text().splitlines() if line.strip()]
STEP_SHOWN = 108
_S = GROUP_LOG[STEP_SHOWN]
assert _S["step"] == STEP_SHOWN and len(_S["groups"]) == Q_PER_STEP
R_STEP = np.array([g["rewards"] for g in _S["groups"]], dtype=float)                  # (8 questions, 8 answers)
assert R_STEP.shape == (Q_PER_STEP, G)
assert all(checker(c, g["ref"]) == r for g in _S["groups"] for c, r in zip(g["completions"], g["rewards"]))
A_STEP = np.array([grpo_advantages(r, eps=1e-4, ddof=0) for r in R_STEP])             # population std, as the run
assert np.allclose(A_STEP, [g["advantages"] for g in _S["groups"]], atol=1e-3)
RIGHT_STEP = int(R_STEP.sum())
assert RIGHT_STEP == 31 and abs(RIGHT_STEP / 64 - RUN[STEP_SHOWN]["reward"]) < 1e-9      # "31 of 64 right"
FLAT = [i for i in range(Q_PER_STEP) if np.all(A_STEP[i] == 0)]
assert len(FLAT) == 3 and [len(set(R_STEP[i])) for i in FLAT] == [1, 1, 1]              # "about one group in three"
TOK_STEP = sum(sum(g["tokens"]) for g in _S["groups"])
assert TOK_STEP == RUN[STEP_SHOWN]["gen_tokens"] == 20_250                            # "about twenty thousand new tokens"
Z_ALL = float(np.mean(CURVES["zero_adv_groups"]))
assert abs(Z_ALL - 1 / 3) < 0.01                                                      # "about a third", over the run

# the reward curve
STEP_X = np.array(CURVES["step"], dtype=float)
TRAIN_R = np.array(CURVES["reward"], dtype=float)
TRAIN_MA = np.array(CURVES["reward_ma10"], dtype=float)
assert len(TRAIN_R) == STEPS and np.allclose(TRAIN_R, [r["reward"] for r in RUN], atol=1e-4)
F25, L25 = float(TRAIN_R[:25].mean()), float(TRAIN_R[-25:].mean())
assert f"{F25:.3f}" == "0.476" == f"{H['train_reward_first25']:.3f}"                  # "zero point four seven six"
assert f"{L25:.3f}" == "0.570" == f"{H['train_reward_last25']:.3f}"                   # "zero point five seven"
assert TRAIN_R.min() > 0.2 and TRAIN_R.max() < 0.85
P1 = [(e["step"], e["pass_at_k"]["1"]) for e in EVALS]
assert [s for s, _ in P1] == [0, 50, 100, 150, 200]
assert pct(P1[0][1]) == "38.3%" == pct(H["pass_at_k_before"]["1"]) and pct(P1[-1][1]) == "46.2%" == pct(H["pass_at_k_after"]["1"])
GAIN = float(pct(P1[-1][1])[:-1]) - float(pct(P1[0][1])[:-1])
SE1 = H["paired_se"]["pass1_gain"]["se"]
assert round(GAIN, 1) == 7.9 and f"{100 * SE1:.1f}" == "0.9"
assert (P1[1][1] - P1[0][1]) / (P1[-1][1] - P1[0][1]) > 0.7                             # "most of that in the first fifty"

# before and after: test-1028 (Tom's bedroom set) and the regression test-1280 (Anna's puzzle)
TOM, ANNA = EX["test-1028"], EX["test-1280"]
TB, TT = TOM["base"]["text"], TOM["trained"]["text"]
assert TOM["ref"] == 200 and TOM["base"]["answer"] == 300 and TOM["trained"]["answer"] == 200
assert (TOM["base"]["tokens"], TOM["trained"]["tokens"]) == (181, 203)                  # real token counts
assert checker(TB, 200) == 0.0 == TOM["base"]["correct"] and checker(TT, 200) == 1.0 == TOM["trained"]["correct"]
TOM_BEFORE = ["3000 - 1000 = 2000", "Tom has to pay 10% of the $3000 bedroom set each month.",
              "0.10 \\times 3000 = 300", "\\boxed{300}"]
TOM_AFTER = ["3000 - 1000 = 2000", "Tom has to pay 10% of the remaining amount per month.",
             "0.10 \\times 2000 = 200", "\\boxed{200}"]
assert all(s in TB for s in TOM_BEFORE) and all(s in TT for s in TOM_AFTER)            # verbatim
Q_TOM = ["Tom buys a bedroom set for $3000. He sells his old bedroom for $1000 and uses that",
         "to pay for part of the bedroom set. He then has to pay 10% a month for the bedroom set.",
         "How much does he have to pay per month?"]
assert " ".join(Q_TOM) == " ".join(TOM["question"].split())

AB, AT = ANNA["base"]["text"], ANNA["trained"]["text"]
assert ANNA["ref"] == 7 and ANNA["base"]["answer"] == 7 and ANNA["trained"]["answer"] == 5
assert (ANNA["base"]["tokens"], ANNA["trained"]["tokens"]) == (173, 135)
assert checker(AB, 7) == 1.0 and checker(AT, 7) == 0.0
assert "5 \\text{ hours} + 2 \\text{ hours} = 7 \\text{ hours}" in AB and "\\boxed{7}" in AB
assert "Anna took half the time Steve took." in AT and "\\frac{10}{2} = 5 \\text{ hours}" in AT and "\\boxed{5}" in AT
assert "2 hours more than half Steve's time" in ANNA["question"] and "10 hours" in ANNA["question"]
assert "Anna took 2 hours more than half of Steve's time" in AB and "Anna took half the time Steve took" in AT

# the held-out flips (greedy, the run's eval log at steps 0 and 200)
EVAL_LOG = [json.loads(line) for line in (LOGS / "lm_grpo_eval.jsonl").read_text().splitlines() if line.strip()]
_E0, _E1 = EVAL_LOG[0], EVAL_LOG[-1]
assert (_E0["step"], _E1["step"]) == (0, STEPS)
IDS = [q["id"] for q in _E0["per_question"]]
assert IDS == [q["id"] for q in _E1["per_question"]] and len(IDS) == 500
G0 = [q["greedy_correct"] == 1.0 for q in _E0["per_question"]]
G1 = [q["greedy_correct"] == 1.0 for q in _E1["per_question"]]
KIND = ["up" if (not a and b) else "down" if (a and not b) else "both" if a else "none" for a, b in zip(G0, G1)]
COUNT = {k: KIND.count(k) for k in ("up", "down", "both", "none")}
assert COUNT == {"up": 60, "down": 39, "both": 192, "none": 209}                        # "sixty ... thirty-nine"
assert COUNT["up"] == H["heldout_flips"]["wrong_to_right"] and COUNT["down"] == H["heldout_flips"]["right_to_wrong"]
C_OF = {q["id"]: q["c"] for q in _E0["per_question"]}, {q["id"]: q["c"] for q in _E1["per_question"]}
TOM_C = (C_OF[0]["test-1028"], C_OF[1]["test-1028"])
ANNA_C = (C_OF[0]["test-1280"], C_OF[1]["test-1280"])
assert TOM_C == (1, 2) and ANNA_C == (2, 4)                                          # their 8 samples, before → after
assert KIND[IDS.index("test-1028")] == "up" and KIND[IDS.index("test-1280")] == "down"

# the course
assert len(MODULES) == 6 and len(EPISODES) == 26
SPANS = {}
for n, (_, mod) in EPISODES.items():
    lo, hi = SPANS.get(mod, (n, n))
    SPANS[mod] = (min(lo, n), max(hi, n))
assert SPANS == {1: (1, 4), 2: (5, 7), 3: (8, 11), 4: (12, 15), 5: (16, 19), 6: (20, 26)}
NAMES = {1: "Trial and Reward", 2: "Planning in a Known World", 3: "Learning from Experience", 4: "Deep Value Learning",
         5: "Learning the Policy", 6: "Teaching Language Models with Reward"}
REVISIT = {  # module → (key idea, credit, exploration); COURSE.md, one line each
    1: ("estimate each lever from its payouts", "the reward belongs to the lever just pulled",
        "ε-greedy, optimism (UCB), Thompson sampling"),
    2: ("Bellman backups, swept until stable", "values flow back from the charger",
        "nothing to explore: the map is known"),
    3: ("learn from what actually happened", "returns, then TD errors between neighbours",
        "ε-greedy moves, bonuses for the unvisited"),
    4: ("networks learn values; search plans", "TD targets from a frozen target network",
        "ε that decays; search tries moves ahead"),
    5: ("push up actions that beat a baseline", "advantage = return − the critic's guess",
        "the policy's own randomness"),
    6: ("the model is the policy", "one advantage per answer, against its group",
        "a group of answers, sampled at temperature 1"),
}

# ---------------------------------------------------------------------------- look

LLM = recolored("llm", ACTION, CORAL_FILL, prefix=PREFIX)
CHECK = recolored("check", REWARD, GOLD_FILL, prefix=PREFIX)
BOOK = recolored("book", STATE, TEAL_FILL, prefix=PREFIX)
GEAR = recolored("gear", SIGNAL, PINK_FILL, prefix=PREFIX)
BEFORE_C, AFTER_C = MUTED, ACTION          # the base model MUTED, the trained policy CORAL (as in Episode 24)
LOGP_C = EXPLORE                           # scoring under the frozen reference: the leash's colour (Episode 23)


def cite(s, size=20):
    return text(s, size, MUTED)


def left_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.RIGHT)


def val_chip(label, color, fill, w=None, h=0.42, size=20):
    words = code(label, size, color, weight=m.BOLD)
    box = card(w or words.width + 0.34, h, color, fill, radius=h / 2, stroke=2)
    return m.VGroup(box, words.move_to(box))


def badge(n, color, r=0.2):
    disc = m.Circle(radius=r).set_stroke(color, 2).set_fill(color, 1)
    return m.VGroup(disc, code(str(n), 20, BG, weight=m.BOLD).move_to(disc))


def boxed_chip(words, color=EDGE, h=0.52, pad=0.4):
    """A line of an answer in a small dark card (mono or sans words already built)."""
    box = card(words.width + pad, h, color, CARD, radius=0.1, stroke=2)
    return m.VGroup(box, words.move_to(box))


def ring(labels, colors, radius, center, size=22):
    """A cycle of steps around a circle (like kit.loop_ring), plus plain arcs to flash along."""
    n = len(labels)
    c = np.array(center, dtype=float)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    nodes = m.VGroup()
    for label, color, a in zip(labels, colors, angles):
        words = text(label, size, INK, weight=m.BOLD)
        nodes.add(chip(label, color, FILLS[color], width=words.width + 0.5, height=0.62, size=size)
                  .move_to(c + radius * np.array([np.cos(a), np.sin(a), 0])))

    def clearance(ch, a):
        reach = (ch.width / 2 + 0.14) * abs(np.sin(a)) + (ch.height / 2 + 0.14) * abs(np.cos(a))
        return min(reach / radius, 0.45 * 2 * np.pi / n)

    arcs, paths = m.VGroup(), []
    for i in range(n):
        a0, a1 = angles[i], angles[(i + 1) % n]
        if a1 > a0:
            a1 -= 2 * np.pi
        start, end = a0 - clearance(nodes[i], a0), a1 + clearance(nodes[(i + 1) % n], a1)
        arc = m.Arc(radius=radius, start_angle=start, angle=end - start, arc_center=c)
        paths.append(arc.copy())
        arcs.add(arc.set_stroke(MUTED, 4).add_tip(tip_length=0.2))
    return nodes, arcs, paths


def apart(a, b, buff=0.05):
    """True if the bounding boxes of a and b are at least `buff` apart (in x or in y)."""
    return (a.get_right()[0] + buff <= b.get_left()[0] or b.get_right()[0] + buff <= a.get_left()[0]
            or a.get_top()[1] + buff <= b.get_bottom()[1] or b.get_top()[1] + buff <= a.get_bottom()[1])


def flash(path, color, run_time=0.7):
    return m.ShowPassingFlash(path.copy().set_stroke(color, 9), time_width=0.6, run_time=run_time)


def plain(s):
    """An answer line without LaTeX markup, as drawn."""
    s = s.replace("\\times", "×").replace("\\frac{10}{2}", "10/2").replace(" \\text{ hours}", " hours")
    return s.replace("\\text{ hours}", " hours")


# ---------------------------------------------------------------------------- the episode


class Ep26GRPOOnThisMac(Episode):
    number = 26

    def lesson(self):
        self.our_run()
        self.the_loop()
        self.the_time()
        self.the_curve()
        self.before_after()
        self.honestly()
        self.the_course()
        self.recap()

    # ------------------------------------------------------------------ 1 · our run

    def our_run(self):
        self.next_section("1 · Our run", notes=(
            "The run (lm_rl.json headline / setup; logs/lm_grpo_config.json): "
            f"{H['model']} ({H['params_base_M']:.0f}M parameters, frozen, bf16) with a LoRA adapter (rank "
            f"{SETUP['args']['lora_rank']}, scale 2.0, all linear layers of all 24 blocks: {CONFIG['trainable_params']:,} "
            f"trainable parameters, '8.8M'). Task: GSM8K (Cobbe et al. 2021), {H['train_questions_seen']:,} training "
            f"problems used (steps × 8 questions), {H['test_questions']} of the 1,319 test problems held out. Prompt: "
            f"system '{SETUP['prompt_system']}'. Reward: {SETUP['reward']}. GRPO (Shao et al. 2024; Episode 23): G = {G} "
            f"answers × {Q_PER_STEP} questions per step, clip ε = {H['clip_eps']}, KL β = {H['beta_kl']} (k3) to the "
            f"frozen base, lr {H['lr']:g}, {STEPS} steps, two AdamW updates per step. Hardware: Mac Studio, Apple M4 Max, "
            "64 GB unified memory, MLX 0.32.4 / mlx-lm 0.32.2. One run, one seed (seed 0). Rover's lever pull is the "
            "course's opening image (Episode 1)."))
        machine = SlotMachine("1", size=1.3).move_to([-1.3, 0.2, 0])
        bot = rover(1.1).move_to([1.3, -0.05, 0]).align_to(machine.cabinet, m.DOWN)
        coin = machine.payout(1, 24)
        model = icon(LLM, 1.25).move_to(bot).set_z_index(2)    # stays above the policy tile it moves into
        mac = pill("trained on this Mac · Apple M4 Max · 64 GB · MLX", MUTED, CARD, 22).move_to([0, -2.0, 0])

        W, HT = 6.1, 1.95
        centres = [(-3.25, 1.3), (3.25, 1.3), (-3.25, -1.0), (3.25, -1.0)]

        def tile(k, art, title, rows, color):
            box = card(W, HT, color, CARD, stroke=2.2).move_to([*centres[k], 0])
            lx = box.get_left()[0]
            art.move_to([lx + 0.75, box.get_y(), 0])
            head = text(title, 26, color, weight=m.BOLD)
            body = m.VGroup(head, *rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
            left_at(body, lx + 1.45, box.get_y())
            assert body.get_right()[0] < box.get_right()[0] - 0.15, (title, body.get_right()[0] - box.get_right()[0])
            assert body.height < HT - 0.25, title
            return dict(box=box, art=art, head=head, rows=list(rows))

        policy = tile(0, icon(LLM, 0.85), "the policy", [
            code(H["model"], 20, INK),
            text(f"{H['params_base_M']:.0f} million parameters, frozen", 20, MUTED),
            text(f"a LoRA adapter learns: {H['trainable_params_M']:g} million", 20, ACTION)], ACTION)
        task = tile(1, icon(BOOK, 0.8), "the task", [
            text("GSM8K grade-school maths", 20, INK),
            text(f"{H['train_questions_seen']:,} training problems", 20, MUTED),
            text(f"{H['test_questions']} more held out for the test", 20, MUTED)], STATE)
        reward = tile(2, icon(CHECK, 0.8), "the reward", [
            text("1 if the last boxed number", 20, INK),
            text("equals the reference, else 0", 20, INK),
            cite("a program · no format bonus")], REWARD)
        recipe = tile(3, icon(GEAR, 0.8), "GRPO · episode 23", [
            code(f"G = {G} answers × {Q_PER_STEP} questions", 20, INK),
            code(f"clip ε = {H['clip_eps']:g} · KL β = {H['beta_kl']:g}", 20, INK),
            code(f"lr {H['lr']:g} · {STEPS} steps", 20, INK)], SIGNAL)
        foot = cite("Cobbe et al., 2021 (GSM8K) · Shao et al., 2024 (GRPO) · one run, one seed").move_to([0, -2.72, 0])
        mac_final = mac.copy().scale(0.92).move_to([0, -3.25, 0])
        assert mac_final.get_bottom()[1] > -3.55 and foot.get_bottom()[1] > mac_final.get_top()[1] + 0.05

        def show(t, *which):
            parts = []
            if "frame" in which:
                parts += [m.FadeIn(t["box"]), m.FadeIn(t["head"], shift=0.1 * m.UP)]
                if "art" in which:
                    parts.append(m.FadeIn(t["art"], scale=0.7))
            for i in which:
                if isinstance(i, int):
                    parts.append(m.FadeIn(t["rows"][i], shift=0.08 * m.RIGHT))
            return m.AnimationGroup(*parts)

        self.say(
            "[Twenty-five episodes ago], [Rover pulled its first lever.] [Today, the same loop trains a real language "
            "model], [right here on this Mac.]",
            m.AnimationGroup(self.new_header("01 · our run", "A real run on this Mac"), m.FadeIn(machine, shift=0.2 * m.UP),
                             m.FadeIn(bot, shift=0.2 * m.UP)),
            m.Succession(machine.pull(0.8), m.FadeIn(coin, scale=0.6, run_time=0.4)),
            m.AnimationGroup(m.FadeOut(machine, shift=0.2 * m.DOWN), m.FadeOut(coin), m.FadeTransform(bot, model), run_time=1.2),
            m.FadeIn(mac, shift=0.15 * m.UP),
        )
        self.say(
            "[The policy is a small open model with half a billion parameters.] [They stay frozen]; "
            "[an adapter of about nine million is what learns.]",
            m.AnimationGroup(m.FadeIn(policy["box"]), m.FadeIn(policy["head"], shift=0.1 * m.UP),
                             model.animate.scale(0.85 / 1.25).move_to(policy["art"]), mac.animate.move_to(mac_final).scale(0.92),
                             m.FadeIn(policy["rows"][0], shift=0.08 * m.RIGHT), run_time=1.3),
            show(policy, 1),
            show(policy, 2),
        )
        self.say(
            "[The task is grade-school maths]: [it trains on sixteen hundred problems], "
            "[and five hundred others are kept for the test.]",
            m.AnimationGroup(show(task, "frame", "art"), show(task, 0)),
            show(task, 1),
            show(task, 2),
        )
        self.say(
            "[The reward is a checker]: [one if the last boxed number is right], [and zero otherwise.]",
            show(reward, "frame", "art"),
            show(reward, 0),
            m.AnimationGroup(show(reward, 1), show(reward, 2)),
        )
        self.say(
            "[The recipe is episode twenty-three's GRPO]: [eight answers to each of eight questions], "
            "[a clip of zero point two, and a leash of beta zero point zero four], [for two hundred steps.]",
            show(recipe, "frame", "art"),
            show(recipe, 0),
            show(recipe, 1),
            m.AnimationGroup(show(recipe, 2), m.FadeIn(foot)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 2 · the loop, again

    def the_loop(self):
        self.next_section("2 · The loop, again", notes=(
            "Episode 1's agent–environment loop (kit.agent_world_loop), then the same loop with the language model as the "
            "agent: actions are whole answers, the state is the question, the reward comes from the checker. One turn = "
            f"one GRPO step. Shown: a real step of our run, step {STEP_SHOWN} of 0–199 (logs/lm_grpo_groups.jsonl): "
            f"{Q_PER_STEP} questions × {G} answers sampled at temperature 1, {TOK_STEP:,} new tokens (logs/lm_grpo.jsonl), "
            f"{RIGHT_STEP} of 64 right (the checker re-run here agrees with every logged reward). Rows of the grid = "
            f"questions; GOLD = reward 1. Advantages per group: (r − mean) / (population std + 1e-4) "
            "(rl_reference.grpo_advantages, ddof=0, equal to the logged advantages); bright PINK = positive, outlined = "
            f"negative; {len(FLAT)} of the 8 groups all agree (std 0 → advantage 0 for all 8 answers; over the whole run "
            f"{Z_ALL:.2f} of groups). Update: the 64 answers in 2 minibatches of 32, one AdamW step each (lr 2e-5), clipped "
            "ratio (ε 0.2) and k3 KL to the frozen base (β 0.04), only the LoRA adapter changes. 200 steps = 12,800 "
            "answers to 1,600 problems."))
        C = np.array([-3.2, -0.3, 0])
        loop = agent_world_loop(center=tuple(C + [0, 0.1, 0]), width=6.4, size=0.85)
        rover_m, agent_lab = loop.agent[1], loop.agent[2]
        grid_m, world_lab = loop.world[1], loop.world[2]
        act_lab, st_lab = loop.action[1], loop.state[1]
        pol = icon(LLM, 0.8).move_to(rover_m)
        pol_lab = text("policy", 22, MUTED).move_to(agent_lab)
        chk = icon(CHECK, 0.75).move_to(grid_m)
        chk_lab = text("checker", 22, MUTED).move_to(world_lab)
        ans_lab = code("answers", 22, ACTION, weight=m.BOLD).next_to(loop.action[0], m.UP, buff=0.1)
        q_lab = code("question", 22, STATE, weight=m.BOLD).next_to(loop.state[0], m.UP, buff=0.1)
        ep1 = cite("the loop of episode 1").move_to([C[0], -2.65, 0])

        nodes, arcs, paths = ring(["sample a group", "check", "advantages", "update"], [ACTION, REWARD, SIGNAL, SIGNAL],
                                  2.05, C)
        assert nodes.get_left()[0] > -6.5 and nodes.get_top()[1] < 2.55 and nodes.get_bottom()[1] > -3.0
        cell, gap = 0.2, 0.06
        cells = m.VGroup()
        for i in range(Q_PER_STEP):
            for j in range(G):
                sq = m.Square(cell).set_stroke(ACTION, 1.5).set_fill(CORAL_FILL, 1)
                sq.move_to(C + [(j - 3.5) * (cell + gap), 0.12 + (3.5 - i) * (cell + gap), 0])
                cells.add(sq)
        g_lab = cite(f"step {STEP_SHOWN} · 64 answers").move_to(C + [0, -1.12, 0])
        assert g_lab.width < 3.0 and cells.get_left()[0] > nodes[3].get_right()[0] + 0.15
        assert cells.get_right()[0] < nodes[1].get_left()[0] - 0.15 and g_lab.get_bottom()[1] > nodes[2].get_top()[1] + 0.05
        assert cells.get_top()[1] < nodes[0].get_bottom()[1] - 0.1

        def cell_of(i, j):
            return cells[i * G + j]

        checked = [cell_of(i, j).animate.set_fill(GOLD if R_STEP[i, j] else CARD, 1).set_stroke(GOLD if R_STEP[i, j] else EDGE, 1.5)
                   for i in range(Q_PER_STEP) for j in range(G)]

        def adv_look(i, j):
            a = A_STEP[i, j]
            if a > 0:
                return cell_of(i, j).animate.set_fill(PINK, 1).set_stroke(PINK, 1.5)
            if a < 0:
                return cell_of(i, j).animate.set_fill(PINK_FILL, 1).set_stroke(PINK, 1.5)
            return cell_of(i, j).animate.set_fill(CARD, 1).set_stroke(FAINT, 1.5)

        flat_marks = m.VGroup(*[code("0", 20, MUTED, weight=m.BOLD).next_to(cell_of(i, G - 1), m.RIGHT, buff=0.12) for i in FLAT])
        assert all(apart(z, nd) for z in flat_marks for nd in nodes) and flat_marks.get_right()[0] < 0.5

        # the right-hand panel: what happened in this step
        PX = 0.75
        specs = [
            (ACTION, f"{Q_PER_STEP} questions × {G} answers", f"{TOK_STEP:,} new tokens, at temperature 1"),
            (REWARD, f"{RIGHT_STEP} of 64 right", "the checker: 1 or 0, nothing in between"),
            (SIGNAL, f"{len(FLAT)} of {Q_PER_STEP} groups all agree: 0", "the rest: (reward − mean) ÷ spread"),
            (SIGNAL, "2 AdamW steps on the adapter", f"clip {H['clip_eps']:g} · leash β = {H['beta_kl']:g} · base frozen"),
        ]
        rows = []
        for k, (color, head, sub) in enumerate(specs):
            y = 1.85 - 1.0 * k
            idx = val_chip(str(k + 1), color, FILLS[color], w=0.46).move_to([PX + 0.23, y, 0])
            h_ = left_at(text(head, 22, INK, weight=m.BOLD), PX + 0.65, y)
            s_ = left_at(cite(sub), PX + 0.65, y - 0.4)
            assert max(h_.get_right()[0], s_.get_right()[0]) < 6.45, head
            rows.append((idx, h_, s_))
        total = left_at(code(f"× {STEPS} steps = {H['completions_sampled']:,} answers", 22, INK, weight=m.BOLD), PX, -2.5)
        src = left_at(cite("trained on this Mac · logs/lm_grpo_groups.jsonl"), PX, -3.05)
        assert total.get_right()[0] < 6.45 and src.get_bottom()[1] > -3.5

        self.say(
            "[Remember episode one's loop?] [Rover acts], [the world answers with a reward], [and Rover learns.]",
            m.AnimationGroup(self.new_header("02 · the loop", "The loop, one more time"), m.FadeIn(loop.agent, shift=0.15 * m.RIGHT),
                             m.FadeIn(loop.world, shift=0.15 * m.LEFT), m.FadeIn(ep1)),
            m.AnimationGroup(m.GrowArrow(loop.action[0]), m.FadeIn(act_lab), pulse(loop.action[0], ACTION)),
            m.AnimationGroup(m.GrowArrow(loop.reward[0]), m.FadeIn(loop.reward[1]), m.GrowArrow(loop.state[0]), m.FadeIn(st_lab)),
            m.Flash(rover_m[3][1].get_center(), color=GOLD, flash_radius=0.3, line_length=0.18),
        )
        self.say(
            "[Now the agent is the language model], [its actions are whole answers], [and the reward comes from the checker.]",
            m.AnimationGroup(m.FadeTransform(rover_m, pol), m.FadeTransform(agent_lab, pol_lab), run_time=1.0),
            m.AnimationGroup(m.FadeTransform(act_lab, ans_lab), m.FadeTransform(st_lab, q_lab), pulse(loop.action[0], ACTION)),
            m.AnimationGroup(m.FadeTransform(grid_m, chk), m.FadeTransform(world_lab, chk_lab), pulse(loop.reward[0], REWARD)),
        )
        for part, parent in ((rover_m, loop.agent), (agent_lab, loop.agent), (grid_m, loop.world), (world_lab, loop.world),
                             (act_lab, loop.action), (st_lab, loop.state)):
            parent.remove(part)
        lm_loop = m.VGroup(loop, pol, pol_lab, chk, chk_lab, ans_lab, q_lab, ep1)
        self.say(
            "[One training step turns this loop once.] [First, sample a group]: [eight answers to each of eight questions], "
            "[about twenty thousand new tokens.]",
            m.AnimationGroup(m.FadeOut(lm_loop), m.FadeIn(arcs), m.LaggedStart(*[m.FadeIn(nd, scale=0.85) for nd in nodes], lag_ratio=0.2),
                             run_time=1.4),
            m.AnimationGroup(m.Circumscribe(nodes[0], color=ACTION, buff=0.06), m.FadeIn(rows[0][0]), m.FadeIn(rows[0][1])),
            m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in cells], lag_ratio=0.02, run_time=1.4),
            m.AnimationGroup(m.FadeIn(rows[0][2], shift=0.08 * m.UP), m.FadeIn(g_lab)),
        )
        self.say(
            "[Then the checker scores all sixty-four]: [one, or zero.]",
            m.AnimationGroup(flash(paths[0], REWARD), m.Circumscribe(nodes[1], color=REWARD, buff=0.06),
                             m.FadeIn(rows[1][0]), m.FadeIn(rows[1][1])),
            m.AnimationGroup(m.LaggedStart(*checked, lag_ratio=0.015, run_time=1.2), m.FadeIn(rows[1][2], shift=0.08 * m.UP)),
        )
        self.say(
            "[Each answer is compared with its own group]: [that's its advantage.] "
            "[When a whole group agrees, every advantage is zero, and it teaches nothing.]",
            m.AnimationGroup(flash(paths[1], SIGNAL), m.Circumscribe(nodes[2], color=SIGNAL, buff=0.06),
                             m.FadeIn(rows[2][0]), m.FadeIn(rows[2][1])),
            m.AnimationGroup(*[adv_look(i, j) for i in range(Q_PER_STEP) for j in range(G) if i not in FLAT],
                             m.FadeIn(rows[2][2], shift=0.08 * m.UP), run_time=1.0),
            m.AnimationGroup(*[adv_look(i, j) for i in FLAT for j in range(G)], m.LaggedStart(*[m.FadeIn(z) for z in flat_marks], lag_ratio=0.2)),
        )
        self.say(
            "[Then two small, clipped steps update the adapter], [and the loop turns again]: [two hundred times.]",
            m.AnimationGroup(flash(paths[2], SIGNAL), m.Circumscribe(nodes[3], color=SIGNAL, buff=0.06),
                             m.FadeIn(rows[3][0]), m.FadeIn(rows[3][1]), m.FadeIn(rows[3][2], shift=0.08 * m.UP)),
            m.Succession(flash(paths[3], ACTION, 0.5), *[flash(p, col, 0.45) for p, col in zip(paths, (REWARD, SIGNAL, SIGNAL, ACTION))]),
            m.AnimationGroup(m.FadeIn(total, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 3 · where the time went

    def the_time(self):
        self.next_section("3 · Where the time went", notes=(
            f"Per step, mean over the 200 steps of logs/lm_grpo.jsonl: sampling {SAMPLE_S:.2f} s (the 64 answers, "
            f"{NEW_TOK:,.0f} new tokens on average, at {TOK_S:,.0f} tokens/s averaged over steps; all tokens ÷ all sampling "
            f"time = {TOK_S_ALL:,.0f}), log-probs {LOGP_S:.2f} s (every answer token scored under π_old and the frozen "
            f"reference π_ref, no gradient), update {UPDATE_S:.2f} s (two AdamW steps, 32 answers each, forward + backward "
            f"through the LoRA adapter); total {STEP_S:.2f} s = {SHARES[0]}% / {SHARES[1]}% / {SHARES[2]}%. 200 steps = "
            f"{TRAIN_MIN:.1f} min of training (+ {EVAL_MIN} min of evaluation; wall clock 121 min with one crash and "
            f"resume). Peak GPU memory per step: median {MEM:.1f} GB, max {MEM_MAX:.1f} GB (the GPU was shared with the "
            "narration server during part of the run, lm_rl.md caveat 3). At scale it is different: 'the inference phase "
            "often accounts for over 90% of the total RLHF (or RLVR) runtime' (Hu et al. 2024, OpenRLHF, arXiv "
            "2405.11143); HybridFlow reports up to 81.2% for NeMo-Aligner (notes §26). Here a 0.5B model with ~300-token "
            "answers, batched, makes the update cost about as much as sampling: measure, don't assume. Sampling is "
            "decoding: what the sister course 'Tokens per Second' is about (memory-bound decode, batching)."))
        X0, X1 = -6.0, 6.0
        span = X1 - X0
        scale = span / STEP_S
        BY, BH = 1.2, 0.78
        parts = [(SAMPLE_S, ACTION, "sampling", f"64 answers · ≈ {round(NEW_TOK, -2):,.0f} new tokens"),
                 (LOGP_S, LOGP_C, "scoring", "old policy +\nfrozen reference"),
                 (UPDATE_S, SIGNAL, "update", "two AdamW steps")]
        segs, names, secs, descs = [], [], [], []
        x = X0
        for (s, color, name, desc), share in zip(parts, SHARES):
            w = s * scale
            seg = m.Rectangle(width=w, height=BH).set_stroke(color, 2.5).set_fill(FILLS[color], 1).move_to([x + w / 2, BY, 0])
            segs.append(seg)
            names.append(text(name, 24, INK, weight=m.BOLD).move_to(seg))
            secs.append(code(f"{s:.1f} s · {share}%", 22, color, weight=m.BOLD).move_to([x + w / 2, BY + BH / 2 + 0.3, 0]))
            descs.append(m.VGroup(*[cite(t_) for t_ in desc.split("\n")]).arrange(m.DOWN, buff=0.06)
                         .next_to(seg, m.DOWN, buff=0.15))
            assert descs[-1].width < w - 0.1 and names[-1].width < w - 0.2, name
            x += w
        assert abs(x - X1) < 1e-9
        outline = m.Rectangle(width=span, height=BH).set_stroke(MUTED, 2).set_fill(opacity=0).move_to([0, BY, 0])
        top_lab = left_at(text(f"one training step · {STEP_S:.1f} s on average", 24, INK, weight=m.BOLD), X0, 2.3)
        top_src = right_at(cite("measured on this Mac · mean of 200 steps"), X1, 2.3)
        assert top_lab.get_right()[0] < top_src.get_left()[0] - 0.3

        # big clusters, long reasoning (cited)
        CY, CH = -0.8, 0.42
        c_lab = left_at(cite(f"big GPU clusters, long reasoning: sampling is often over {BIG_CLUSTER}% · Hu et al., 2024 (OpenRLHF)"),
                        X0, CY + 0.5)
        c_bar = m.Rectangle(width=span * BIG_CLUSTER / 100, height=CH).set_stroke(ACTION, 2).set_fill(CORAL_FILL, 1)
        left_at(c_bar, X0, CY)
        c_rest = m.Rectangle(width=span * (100 - BIG_CLUSTER) / 100, height=CH).set_stroke(EDGE, 2).set_fill(CARD, 1)
        left_at(c_rest, c_bar.get_right()[0], CY)
        c_txt = text(f"sampling > {BIG_CLUSTER}%", 20, INK, weight=m.BOLD).move_to(c_bar)
        assert c_lab.get_right()[0] < X1 and c_lab.get_top()[1] < min(d.get_bottom()[1] for d in descs) - 0.1

        # the totals
        SY = -2.2

        def total(x, value, unit, label, color):
            num = title_text(value, 46, color)
            u = code(unit, 22, MUTED).next_to(num, m.RIGHT, buff=0.12).align_to(num, m.DOWN)
            lab = cite(label)
            grp = m.VGroup(m.VGroup(num, u), lab).arrange(m.DOWN, buff=0.14)
            return grp.move_to([x, SY, 0])

        t_min = total(-4.3, f"{TRAIN_MIN:.1f}", "min", f"of training · tests took {EVAL_MIN} more", INK)
        t_tok = total(0.0, f"≈ {round(TOK_S / 50) * 50:,}", "tokens/s", "while sampling", ACTION)
        t_mem = total(4.3, f"≈ {MEM:.0f}", "GB", f"peak memory (max {MEM_MAX:.1f}) of 64", INK)
        assert t_min.get_left()[0] > -6.5 and t_mem.get_right()[0] < 6.5
        assert t_min.get_right()[0] < t_tok.get_left()[0] - 0.2 and t_tok.get_right()[0] < t_mem.get_left()[0] - 0.2
        link = pill("Tokens per Second · our course on fast inference", MUTED, CARD, 20).move_to([0, -3.2, 0])

        self.say(
            "[Where did the time go?] [One step took twenty-one and a half seconds, on average.]",
            m.AnimationGroup(self.new_header("03 · the clock", "Where the time went"), m.FadeIn(top_lab), m.FadeIn(top_src)),
            m.Create(outline),
        )
        self.say(
            "[Nine seconds went to sampling]: [the model writing its sixty-four answers], [at over two thousand tokens a second.]",
            m.AnimationGroup(m.GrowFromEdge(segs[0], m.LEFT), m.FadeIn(names[0]), m.FadeIn(secs[0], shift=0.1 * m.DOWN)),
            m.FadeIn(descs[0], shift=0.1 * m.UP),
            m.FadeIn(t_tok, shift=0.1 * m.UP),
        )
        self.say(
            "[Five went to scoring every token under the old policy and the frozen reference], "
            "[and seven to the two updates.]",
            m.AnimationGroup(m.GrowFromEdge(segs[1], m.LEFT), m.FadeIn(names[1]), m.FadeIn(secs[1], shift=0.1 * m.DOWN),
                             m.FadeIn(descs[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(segs[2], m.LEFT), m.FadeIn(names[2]), m.FadeIn(secs[2], shift=0.1 * m.DOWN),
                             m.FadeIn(descs[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[Two hundred steps made seventy-two minutes of training], [in about twenty-five gigabytes of memory.]",
            m.FadeIn(t_min, shift=0.1 * m.UP),
            m.FadeIn(t_mem, shift=0.1 * m.UP),
        )
        self.say(
            "[On big clusters with long reasoning, sampling often eats over ninety percent of the time]: "
            "[that's why fast inference, our sister course Tokens per Second, matters here.]",
            m.AnimationGroup(m.FadeIn(c_lab), m.GrowFromEdge(c_bar, m.LEFT), m.FadeIn(c_rest), m.FadeIn(c_txt),
                             ),
            m.AnimationGroup(m.FadeIn(link, shift=0.1 * m.UP), m.Circumscribe(segs[0], color=ACTION, buff=0.05)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 4 · the reward curve

    def the_curve(self):
        self.next_section("4 · Did it learn?", notes=(
            "Training reward per step (lm_rl.json curves.reward = the share of that step's 64 fresh answers the checker "
            "accepted; every step brings 8 new questions, so single steps are noisy), its 10-step moving average "
            f"(curves.reward_ma10), and the 25-step block means of steps 0–24 ({F25:.4f}) and 175–199 ({L25:.4f}). CORAL "
            f"dots: held-out pass@1 at the five evaluations (500 GSM8K test problems never trained on, 8 samples each at "
            f"T = 0.8, the unbiased estimator; lm_rl.json evals): {', '.join(pct(p) for _, p in P1)}. Gain "
            f"+{GAIN:.1f} ± {100 * SE1:.1f} points (paired SE, lm_rl.json paired_se). Most of it by step 50 "
            f"(+{100 * (P1[1][1] - P1[0][1]):.1f}). Episode 24 took the evaluation apart (pass@k, greedy vs sampled). "
            "One run, one seed."))
        pl = Plot((0, 200), (0.2, 1.0), width=6.6, height=4.3, center=(-2.55, -0.3, 0), x_ticks=[0, 50, 100, 150, 200],
                  y_ticks=[0.2, 0.4, 0.6, 0.8], x_label="training step", y_label="share of answers right", y_fmt="{:.1f}")
        assert pl.get_left()[0] > -6.6 and pl.get_bottom()[1] > -3.55
        raw = pl.line(STEP_X, TRAIN_R, mix(BG, GOLD, 0.42), width=2)
        smooth = pl.line(STEP_X, TRAIN_MA, GOLD, 5)
        g1 = m.Line(pl.coords(0, F25), pl.coords(24, F25)).set_stroke(INK, 4)
        g2 = m.Line(pl.coords(175, L25), pl.coords(199, L25)).set_stroke(INK, 4)
        top_y = pl.coords(0, 0.93)[1]
        l1 = code(f"first 25 steps: {F25:.3f}", 20, INK, weight=m.BOLD)
        left_at(l1, pl.coords(0, 0)[0] + 0.1, top_y)
        l2 = code(f"last 25: {L25:.3f}", 20, INK, weight=m.BOLD)
        right_at(l2, pl.coords(200, 0)[0], top_y)
        lead1 = m.DashedLine(pl.coords(12, F25), [pl.coords(12, 0)[0], l1.get_bottom()[1] - 0.06, 0], dash_length=0.06).set_stroke(INK, 1.5)
        lead2 = m.DashedLine(pl.coords(187, L25), [pl.coords(187, 0)[0], l2.get_bottom()[1] - 0.06, 0], dash_length=0.06).set_stroke(INK, 1.5)
        assert l1.get_right()[0] < l2.get_left()[0] - 0.3 and l1.get_bottom()[1] > pl.coords(0, 0.83)[1]
        pdots = m.VGroup(*[pl.dot(s, p, ACTION, 0.1) for s, p in P1])
        pline = pl.line([s for s, _ in P1], [p for _, p in P1], ACTION, 3)

        # the right-hand panel
        RX = 2.15
        Y0, Y1 = 2.05, 0.95
        leg = m.VGroup(
            m.VGroup(m.Line([RX, Y0, 0], [RX + 0.45, Y0, 0]).set_stroke(GOLD, 5), left_at(text("training reward", 20, INK), RX + 0.63, Y0)),
            m.VGroup(m.Dot([RX + 0.22, Y1, 0], radius=0.1, color=ACTION), left_at(text("held-out pass@1", 20, INK), RX + 0.63, Y1)),
        )
        leg_sub = m.VGroup(cite("64 new answers a step"), cite("bold: a 10-step average")).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        leg_sub.next_to(leg[0][1], m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        assert leg_sub.get_bottom()[1] > Y1 + 0.2
        big0 = title_text(pct(P1[0][1]), 40, BEFORE_C)
        big1 = title_text(pct(P1[-1][1]), 40, AFTER_C)
        left_at(big0, RX, 0.15)
        arr = arrow(big0.get_right() + [0.1, 0, 0], big0.get_right() + [0.65, 0, 0], MUTED, buff=0)
        left_at(big1, arr.get_right()[0] + 0.12, 0.15)
        b0_lab = cite("before").next_to(big0, m.DOWN, buff=0.12)
        b1_lab = cite(f"after {STEPS} steps").next_to(big1, m.DOWN, buff=0.12)
        gain = code(f"+{GAIN:.1f} ± {100 * SE1:.1f} points", 24, GREEN, weight=m.BOLD)
        left_at(gain, RX, -1.05)
        g_src = left_at(cite("500 problems never trained on"), RX, -1.5)
        ep24 = pill("episode 24 · pass@k", MUTED, CARD, 20)
        left_at(ep24, RX, -2.35)
        src = left_at(cite("trained on this Mac · one seed"), RX, -3.0)
        for mob in (leg, leg_sub, big1, gain, g_src, ep24, src):
            assert mob.get_right()[0] < 6.5, (mob, mob.get_right()[0])
        assert big0.get_left()[0] > pl.get_right()[0] + 0.15 and leg_sub.get_left()[0] > pl.get_right()[0] + 0.15
        early = m.Rectangle(width=pl.coords(50, 0)[0] - pl.coords(0, 0)[0], height=pl.coords(0, 0.8)[1] - pl.coords(0, 0.2)[1])
        early.set_stroke(width=0).set_fill(GREEN, 0.08).move_to((pl.coords(0, 0.2) + pl.coords(50, 0.8)) / 2).set_z_index(-1)

        self.say(
            "[Here's the training reward at every step]: [noisy, because each step brings eight new questions.]",
            m.AnimationGroup(self.new_header("04 · the curve", "Did it learn?"), m.FadeIn(pl), m.FadeIn(leg[0]), m.FadeIn(leg_sub)),
            m.Create(raw, run_time=2.4),
        )
        self.say(
            "[Averaged over ten steps, it climbs]: [from zero point four seven six in the first twenty-five steps], "
            "[to zero point five seven in the last twenty-five.]",
            m.Create(smooth, run_time=2.2),
            m.AnimationGroup(m.Create(g1), m.Create(lead1), m.FadeIn(l1, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.Create(g2), m.Create(lead2), m.FadeIn(l2, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[On five hundred problems it never saw], [single tries went from thirty-eight point three percent right], "
            "[to forty-six point two.]",
            m.AnimationGroup(m.FadeIn(leg[1]), m.FadeIn(pdots[0], scale=0.5), m.FadeIn(g_src)),
            m.FadeIn(big0, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(pline), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in pdots[1:]], lag_ratio=0.25),
                             m.GrowArrow(arr), m.FadeIn(big1, shift=0.1 * m.UP), m.FadeIn(b0_lab), m.FadeIn(b1_lab),
                             m.FadeIn(gain, shift=0.1 * m.UP)),
        )
        self.say(
            "[Most of that came in the first fifty steps.] [Episode twenty-four took these numbers apart.]",
            m.AnimationGroup(m.FadeIn(early), m.Circumscribe(m.VGroup(pdots[0], pdots[1]), color=ACTION, buff=0.12)),
            m.AnimationGroup(m.FadeIn(ep24, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 5 · one question, before and after

    def before_after(self):
        self.next_section("5 · Before and after", notes=(
            "GSM8K test problem 1028, held out (lm_rl.json examples[0]): greedy answers of the base model and of the "
            f"trained model (re-decoded at batch 64; their correctness agrees with the eval log). Base: {TOM['base']['tokens']} "
            f"tokens, computes 3000 − 1000 = 2000, then 'Tom has to pay 10% of the $3000 bedroom set each month', 0.10 × "
            f"3000 = 300 → \\boxed{{300}}, reward 0. Trained: {TOM['trained']['tokens']} tokens, 'Tom has to pay 10% of the "
            "remaining amount per month', 0.10 × 2000 = 200 → \\boxed{200}, reward 1 (the checker re-run here). Lines "
            "verbatim, LaTeX markup removed. Episode 20 showed the base answer token by token. Sampled 8 times at T = 0.8 "
            f"(eval log): right {TOM_C[0]} → {TOM_C[1]} times — the greedy flip is the clearest case, not a typical one."))
        q_rows = m.VGroup(*[text(s, 20, INK) for s in Q_TOM]).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        q_box = card(12.6, q_rows.height + 0.4, EDGE, CARD, radius=0.14, stroke=1.5)
        stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=q_box.height - 0.16, stroke_width=0,
                                    fill_color=STATE, fill_opacity=1)
        stripe.move_to(q_box).align_to(q_box, m.LEFT).shift(0.1 * m.RIGHT)
        q_rows.move_to(q_box).align_to(q_box, m.LEFT).shift(0.4 * m.RIGHT)
        qcard = m.VGroup(q_box, stripe, q_rows).move_to([0, 2.5 - (q_rows.height + 0.4) / 2, 0])
        assert q_rows.width < 12.0
        q_src = right_at(cite("GSM8K test problem 1028 · held out · episode 20"), q_box.get_right()[0] - 0.3, q_rows[2].get_y())
        assert q_src.get_left()[0] > q_rows[2].get_right()[0] + 0.4

        XC = (-3.25, 3.25)
        YS = (0.0, -0.78, -1.56, -2.34)
        HY = 0.62

        def column(lines, x, color, wrong):
            ok_col, bad_col = (GREEN, RED)
            chips = []
            for k, (s, y) in enumerate(zip(lines, YS)):
                words = text(s.split("pay ", 1)[1].rstrip("."), 20, INK) if k == 1 else code(plain(s), 20, INK)
                stroke = (bad_col if wrong else ok_col) if k in (1, 3) else EDGE
                chips.append(boxed_chip(words, stroke).move_to([x, y, 0]))
            arrows = m.VGroup(*[arrow(a.get_bottom(), b.get_top(), MUTED, buff=0.05) for a, b in zip(chips, chips[1:])])
            return chips, arrows

        b_chips, b_arr = column(TOM_BEFORE, XC[0], BEFORE_C, True)
        a_chips, a_arr = column(TOM_AFTER, XC[1], AFTER_C, False)
        assert all(c.width < 6.2 for c in b_chips + a_chips)
        b_head = m.VGroup(text("before training", 24, BEFORE_C, weight=m.BOLD), cite(f"{TOM['base']['tokens']} tokens"))
        b_head.arrange(m.RIGHT, buff=0.25).move_to([XC[0], HY, 0])
        a_head = m.VGroup(text(f"after {STEPS} steps", 24, AFTER_C, weight=m.BOLD), cite(f"{TOM['trained']['tokens']} tokens"))
        a_head.arrange(m.RIGHT, buff=0.25).move_to([XC[1], HY, 0])
        on_baseline(b_head)
        on_baseline(a_head)
        r0 = reward_chip(0, 24).next_to(b_chips[3], m.RIGHT, buff=0.3)
        r1 = reward_chip(1, 24).next_to(a_chips[3], m.RIGHT, buff=0.3)
        foot = cite(f"greedy answers, lines verbatim · sampled 8 times: right {TOM_C[0]} → {TOM_C[1]} times").move_to([0, -3.15, 0])
        assert qcard.get_bottom()[1] > HY + 0.35

        def walk(chips, arrows, k0, k1):
            anims = []
            for k in range(k0, k1):
                if k > 0:
                    anims.append(m.GrowArrow(arrows[k - 1]))
                anims.append(m.FadeIn(chips[k], shift=0.1 * m.DOWN))
            return m.LaggedStart(*anims, lag_ratio=0.35)

        self.say(
            "[Remember Tom's bedroom set from episode twenty?] [Three thousand dollars, minus a thousand dollars from his old one], "
            "[then ten percent a month.]",
            m.AnimationGroup(self.new_header("05 · before and after", "One question, before and after"),
                             m.FadeIn(qcard, shift=0.15 * m.UP), m.FadeIn(q_src)),
            m.Circumscribe(q_rows[0], color=STATE, buff=0.06),
            m.Circumscribe(q_rows[1], color=STATE, buff=0.06),
        )
        self.say(
            "[Before training, the model even computes the two thousand], [then takes ten percent of the full three "
            "thousand]: [three hundred, and the checker pays zero.]",
            m.AnimationGroup(m.FadeIn(b_head), walk(b_chips, b_arr, 0, 1)),
            walk(b_chips, b_arr, 1, 3),
            m.AnimationGroup(walk(b_chips, b_arr, 3, 4), m.Succession(m.Wait(0.4), m.FadeIn(r0, scale=0.6))),
        )
        self.say(
            "[After two hundred steps, the same question]: [ten percent of the remaining amount]: "
            "[two hundred, and a reward of one.]",
            m.AnimationGroup(m.FadeIn(a_head), walk(a_chips, a_arr, 0, 1)),
            walk(a_chips, a_arr, 1, 3),
            m.AnimationGroup(walk(a_chips, a_arr, 3, 4), m.Succession(m.Wait(0.4), m.FadeIn(r1, scale=0.6))),
        )
        self.say(
            "[About the same length: a hundred eighty-one tokens, and two hundred three.] "
            "[Only one step of the reasoning changed.]",
            m.AnimationGroup(m.Circumscribe(b_head[1], color=MUTED, buff=0.06), m.Circumscribe(a_head[1], color=MUTED, buff=0.06),
                             m.FadeIn(foot)),
            m.AnimationGroup(m.Circumscribe(b_chips[1], color=RED, buff=0.06), m.Circumscribe(a_chips[1], color=GREEN, buff=0.06)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 6 · honestly

    def honestly(self):
        self.next_section("6 · Honestly", notes=(
            "GSM8K test problem 1280 (lm_rl.json examples[3], chosen by rule as the shortest right → wrong pair): 'Steve put "
            "together a puzzle that took 10 hours of hard work to complete. Anna put together the same puzzle in 2 hours "
            "more than half Steve's time. How long did it take Anna to finish the difficult puzzle?' (shortened on screen). "
            f"Base (greedy, {ANNA['base']['tokens']} tokens): 10/2 = 5, 5 hours + 2 hours = 7 hours → \\boxed{{7}}, right. "
            f"Trained ({ANNA['trained']['tokens']} tokens): 'Anna took half the time Steve took.' 10/2 = 5 → \\boxed{{5}}, "
            f"wrong. Its 8 samples at T = 0.8 were right {ANNA_C[0]} → {ANNA_C[1]} times. The grid: the 500 held-out "
            "problems in the eval log's order, greedy answers at steps 0 and 200 (logs/lm_grpo_eval.jsonl, batch 256): "
            f"{COUNT['up']} wrong → right, {COUNT['down']} right → wrong, {COUNT['both']} right both times, "
            f"{COUNT['none']} wrong both times (= lm_rl.json heldout_flips). Greedy decoding is not batch-invariant here "
            "(86% of problems keep their correctness across batch sizes; lm_rl.md caveat 1), so single flips are noisy. "
            "One run, one seed: no run-to-run variance was measured (caveat 2). Episode 25: a reward with a gap gets gamed "
            "(the length-bonus run)."))
        LX0, LX1 = -6.4, 1.15
        lines = ["Steve's puzzle took 10 hours. Anna took 2 hours more",
                 "than half Steve's time. How long did she take?"]
        q_rows = m.VGroup(*[text(s_, 20, INK) for s_ in lines]).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        q_box = card(LX1 - LX0, q_rows.height + 0.4, EDGE, CARD, radius=0.14, stroke=1.5)
        stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=q_box.height - 0.16, stroke_width=0,
                                    fill_color=STATE, fill_opacity=1)
        stripe.move_to(q_box).align_to(q_box, m.LEFT).shift(0.1 * m.RIGHT)
        q_rows.move_to(q_box).align_to(q_box, m.LEFT).shift(0.4 * m.RIGHT)
        qcard = m.VGroup(q_box, stripe, q_rows)
        left_at(qcard, LX0, 2.45 - q_box.height / 2)
        assert q_rows.get_right()[0] < LX1 - 0.2
        q_src = left_at(cite("GSM8K test problem 1280 · held out · shortened"), LX0, qcard.get_bottom()[1] - 0.28)

        def answer_rows(y, label, color, tokens, first, math, box, ok):
            head = m.VGroup(text(label, 24, color, weight=m.BOLD), cite(f"{tokens} tokens")).arrange(m.RIGHT, buff=0.25)
            on_baseline(head)
            left_at(head, LX0, y)
            c1 = left_at(boxed_chip(first), LX0, y - 0.58)
            c2 = left_at(boxed_chip(math), LX0, y - 1.2)
            c3 = boxed_chip(box, GREEN if ok else RED).next_to(c2, m.RIGHT, buff=0.18)
            r = reward_chip(1 if ok else 0, 22).next_to(c3, m.RIGHT, buff=0.22)
            assert max(c1.get_right()[0], r.get_right()[0]) < LX1, (label, c1.get_right()[0], r.get_right()[0])
            return head, c1, c2, c3, r

        before = answer_rows(0.62, "before", BEFORE_C, ANNA["base"]["tokens"],
                             text("Anna took 2 hours more than half of Steve's time", 20, INK),
                             code(plain("5 \\text{ hours} + 2 \\text{ hours} = 7 \\text{ hours}"), 20, INK),
                             code("\\boxed{7}", 20, INK), True)
        after = answer_rows(-1.22, f"after {STEPS} steps", AFTER_C, ANNA["trained"]["tokens"],
                            text("Anna took half the time Steve took", 20, INK),
                            code(plain("\\frac{10}{2} = 5 \\text{ hours}"), 20, INK),
                            code("\\boxed{5}", 20, INK), False)
        assert before[0].get_top()[1] < q_src.get_bottom()[1] - 0.2 and after[2].get_bottom()[1] > -2.85
        assert before[2].get_bottom()[1] > after[0].get_top()[1] + 0.15

        # the 500 held-out problems
        cols, rows_n, size, gap = 25, 20, 0.155, 0.04
        pitch = size + gap
        gx, gy = 3.97, 0.32
        look = {"both": (mix(CARD, GREEN, 0.38), mix(EDGE, GREEN, 0.4)), "none": (CARD, EDGE),
                "up": (GREEN, GREEN), "down": (RED, RED)}
        squares = m.VGroup()
        for i in range(500):
            r, c = divmod(i, cols)
            sq = m.Square(size).set_stroke(EDGE, 1).set_fill(CARD, 1)
            sq.move_to([gx + (c - (cols - 1) / 2) * pitch, gy - (r - (rows_n - 1) / 2) * pitch, 0])
            squares.add(sq)
        assert squares.get_left()[0] > LX1 + 0.38 and squares.get_right()[0] < 6.45 and squares.get_top()[1] < 2.35

        def paint_kind(kinds, run_time=1.2):
            idx = [i for i, k in enumerate(KIND) if k in kinds]
            return m.LaggedStart(*[squares[i].animate.set_fill(look[KIND[i]][0], 1).set_stroke(look[KIND[i]][1], 1) for i in idx],
                                 lag_ratio=0.01, run_time=run_time)

        def legend_item(kind, words):
            sw = m.Square(size).set_fill(look[kind][0], 1).set_stroke(look[kind][1], 1)
            return m.VGroup(sw, code(str(COUNT[kind]), 20, INK, weight=m.BOLD), cite(words)).arrange(m.RIGHT, buff=0.1)

        L_up, L_down = legend_item("up", "wrong → right"), legend_item("down", "right → wrong")
        L_both, L_none = legend_item("both", "right twice"), legend_item("none", "wrong twice")
        x_l, x_r = squares.get_left()[0], squares.get_left()[0] + 2.5
        y1, y2 = squares.get_bottom()[1] - 0.35, squares.get_bottom()[1] - 0.78
        left_at(L_both, x_l, y1)
        left_at(L_none, x_l, y2)
        left_at(L_up, x_r, y1)
        left_at(L_down, x_r, y2)
        assert L_up.get_right()[0] < 6.45 and L_both.get_right()[0] < x_r - 0.15 and L_down.get_bottom()[1] > -2.75
        g_title = cite("500 held-out problems · greedy answers").move_to([gx, squares.get_top()[1] + 0.25, 0])
        assert g_title.get_top()[1] < 2.75 and g_title.get_left()[0] > LX1 + 0.2
        seed = pill("one run · one seed", EXPLORE, VIOLET_FILL, 22).move_to([gx, -3.12, 0])
        ep25 = left_at(cite("a reward with a gap gets gamed: episode 25"), LX0, -3.12)
        assert ep25.get_right()[0] < seed.get_left()[0] - 0.2 and seed.get_right()[0] < 6.5

        self.say(
            "[But training isn't magic.] [On Anna's puzzle, the base model was right]: [half of ten, plus two, is seven.]",
            m.AnimationGroup(self.new_header("06 · honestly", "What got worse"), m.FadeIn(qcard, shift=0.15 * m.UP), m.FadeIn(q_src)),
            m.AnimationGroup(m.FadeIn(before[0], shift=0.1 * m.UP), m.FadeIn(before[1], shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.LaggedStart(m.FadeIn(before[2], shift=0.1 * m.RIGHT), m.FadeIn(before[3], shift=0.1 * m.RIGHT),
                                           lag_ratio=0.5), m.Succession(m.Wait(0.6), m.FadeIn(before[4], scale=0.6))),
        )
        self.say(
            "[After training, it drops the plus two], [and answers five.]",
            m.AnimationGroup(m.FadeIn(after[0], shift=0.1 * m.UP), m.FadeIn(after[1], shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.LaggedStart(m.FadeIn(after[2], shift=0.1 * m.RIGHT), m.FadeIn(after[3], shift=0.1 * m.RIGHT),
                                           lag_ratio=0.5), m.Succession(m.Wait(0.6), m.FadeIn(after[4], scale=0.6))),
        )
        self.say(
            "[Across all five hundred problems], [sixty answers went from wrong to right], [and thirty-nine from right to wrong.]",
            m.AnimationGroup(m.FadeIn(g_title), m.LaggedStart(*[m.FadeIn(sq) for sq in squares], lag_ratio=0.002, run_time=1.0),
                             m.Succession(m.Wait(0.6), m.AnimationGroup(paint_kind({"both", "none"}, 0.9),
                                                                        m.FadeIn(L_both), m.FadeIn(L_none)))),
            m.AnimationGroup(paint_kind({"up"}), m.FadeIn(L_up, shift=0.1 * m.UP)),
            m.AnimationGroup(paint_kind({"down"}), m.FadeIn(L_down, shift=0.1 * m.UP)),
        )
        self.say(
            "[And this is one run, with one seed]: [we never measured how much a second run would differ.]",
            m.FadeIn(seed, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(seed, color=EXPLORE, buff=0.06), m.FadeIn(ep25)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 7 · the course revisited

    def the_course(self):
        self.next_section("7 · The course revisited", notes=(
            "One card per module (kit.MODULES colours, COURSE.md): its episodes, its key idea, and the course's two "
            "questions (Episode 1): credit, which action was good? (PINK) and exploration, what haven't I tried? (VIOLET). "
            "Module 1 (Ep 1–4): sample averages per lever; a bandit's reward belongs to the action just taken; ε-greedy, "
            "UCB1 and Thompson sampling. Module 2 (5–7): Bellman backups, value iteration sweeps; credit flows back from "
            "the charger through the values; with a known model there is nothing to explore. Module 3 (8–11): Monte Carlo "
            "returns, then TD errors; ε-greedy behaviour, count-based and curiosity bonuses (Ep 11). Module 4 (12–15): "
            "function approximation, DQN's frozen target network, ε decaying; MCTS tries moves ahead (UCT/PUCT bonus). "
            "Module 5 (16–19): policy gradients, advantage = return − a learned baseline (critic); exploration from the "
            "stochastic policy itself (entropy in SAC, Gaussian noise). Module 6 (20–26): the language model is the "
            "policy; GRPO's group-relative advantage, one per answer, shared by its tokens; a group sampled at "
            "temperature 1 is the exploration."))
        W, HT = 6.45, 1.8
        xs, ys = (-3.33, 3.33), (1.5, -0.42, -2.34)
        cards = []
        for k in range(1, 7):
            name, accent, tint = MODULES[k]
            box = card(W, HT, accent, CARD, stroke=2.5).move_to([xs[(k - 1) % 2], ys[(k - 1) // 2], 0])
            lx, top = box.get_left()[0] + 0.25, box.get_top()[1]
            b = badge(k, accent).move_to([lx + 0.18, top - 0.38, 0])
            assert NAMES[k].upper() == name
            title = left_at(title_text(NAMES[k], 20, INK), lx + 0.48, top - 0.38)
            lo, hi = SPANS[k]
            eps = code(f"ep {lo}–{hi}", 20, accent, weight=m.BOLD)
            left_at(eps, lx, top - 0.82)
            idea_s, credit_s, explore_s = REVISIT[k]
            idea = left_at(text(idea_s, 20, INK), eps.get_right()[0] + 0.25, top - 0.82)
            rows = []
            for j, (s, color) in enumerate(((credit_s, SIGNAL), (explore_s, EXPLORE))):
                y = top - 1.18 - 0.33 * j
                rows.append(m.VGroup(m.Dot([lx + 0.1, y, 0], radius=0.07, color=color),
                                     left_at(text(s, 20, MUTED), lx + 0.32, y)))
            for mob in (title, idea, *rows):
                assert mob.get_right()[0] < box.get_right()[0] - 0.15, (k, mob)
            assert rows[1].get_bottom()[1] > box.get_bottom()[1] + 0.08, k
            cards.append(dict(box=box, head=m.VGroup(b, title, eps, idea), credit=rows[0], explore=rows[1], accent=accent))
        assert cards[0]["box"].get_top()[1] < 2.55 and cards[-1]["box"].get_bottom()[1] > -3.5
        legend = m.VGroup(pill("credit", SIGNAL, PINK_FILL, 22), pill("exploration", EXPLORE, VIOLET_FILL, 22)).arrange(m.RIGHT, buff=0.25)
        legend.move_to([0, 3.19, 0]).align_to([6.6, 0, 0], m.RIGHT)

        def open_card(c):
            return m.AnimationGroup(m.FadeIn(c["head"], shift=0.08 * m.UP), m.Circumscribe(c["box"], color=c["accent"], buff=0.04))

        self.say(
            "[Twenty-six episodes, six modules], [and in every one, the same two questions]: [which action was good], "
            "[and what haven't I tried?]",
            m.AnimationGroup(self.new_header("07 · the course", "Two questions, six ways"),
                             m.LaggedStart(*[m.FadeIn(c["box"], scale=0.95) for c in cards], lag_ratio=0.12)),
            m.FadeIn(legend, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.Circumscribe(legend[0], color=SIGNAL, buff=0.05)),
            m.AnimationGroup(m.Circumscribe(legend[1], color=EXPLORE, buff=0.05)),
        )
        lines = [
            "[Module one, slot machines]: [the reward belongs to the lever just pulled], "
            "[and exploring means trying the uncertain levers.]",
            "[Module two, a known world]: [Bellman backups carry value back from the charger], "
            "[and with the map known, nothing needs exploring.]",
            "[Module three, learning from experience]: [returns, then TD errors, share out the credit], "
            "[and random moves and novelty bonuses explore.]",
            "[Module four, deep values]: [a network learns from a frozen target], [and search tries moves ahead.]",
            "[Module five, the policy itself]: [credit is an advantage over the critic's guess], "
            "[and the policy's own randomness explores.]",
            "[And module six, language models]: [each answer gets one advantage against its group], "
            "[and sampling the group is the exploring.]",
        ]
        for c, line in zip(cards, lines):
            self.say(line, open_card(c), m.FadeIn(c["credit"], shift=0.08 * m.RIGHT), m.FadeIn(c["explore"], shift=0.08 * m.RIGHT))
        self.say(
            "[Six worlds], [one loop], [and two questions, answered six ways.]",
            m.LaggedStart(*[m.Circumscribe(c["box"], color=c["accent"], buff=0.04) for c in cards], lag_ratio=0.12),
            m.LaggedStart(*[m.Indicate(c["credit"][0], color=SIGNAL, scale_factor=1.8) for c in cards], lag_ratio=0.1),
            m.LaggedStart(*[m.Indicate(c["explore"][0], color=EXPLORE, scale_factor=1.8) for c in cards], lag_ratio=0.1),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            "Episode 1's three words for the whole course: act (the policy tries an action: a lever, a move, an answer), "
            "reward (the world scores it: a coin, the charger, a checker), learn (do more of what paid). Reinforcement "
            "learning is learning what to do from the consequences of doing it (Sutton & Barto 2018, §1.1)."))
        halo = glow(GOLD, radius=4.0)
        words = m.VGroup(*[title_text(w, 88, c) for w, c in (("Act", ACTION), ("Reward", REWARD), ("Learn", SIGNAL))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("write an answer", "a checker scores it", "do more of what paid")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 1.0
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
        assert row.width < 12.8
        tagline = text("Learning what to do, from the consequences of doing it.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)
        pair = m.VGroup(rover(0.5), icon(LLM, 0.55)).arrange(m.RIGHT, buff=0.6).next_to(tagline, m.DOWN, buff=0.45)

        self.say(
            "[To sum up the whole course, in episode one's three words]: [act], [reward], [learn.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[A robot at a slot machine, or a model doing maths]: [learning what to do, from the consequences of doing it.]",
            m.LaggedStart(m.FadeIn(pair[0], shift=0.2 * m.UP), m.FadeIn(pair[1], shift=0.2 * m.UP), lag_ratio=0.4),
            m.Write(tagline),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline, pair)], run_time=0.9)

    # ------------------------------------------------------------------ the course-closing card

    def series_outro(self):
        self.next_section("The end", notes="The course's closing card: Trial and Reward, from slot machines to reasoning "
                          "models; the six modules in their colours (kit.MODULES); Rover.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        halo = glow(GOLD, radius=4.3).move_to(0.2 * m.UP)
        brand = title_text("That's Trial and Reward", 64, INK, gradient=(GOLD, CORAL))
        sub = text("from slot machines to reasoning models", 30, MUTED)
        top = m.VGroup(brand, sub).arrange(m.DOWN, buff=0.3)
        mods = m.VGroup(*[pill(f"{k} · {name}", color, fill, 22, mono=True) for k, (name, color, fill) in MODULES.items()])
        rows = m.VGroup(m.VGroup(mods[0], mods[1]).arrange(m.RIGHT, buff=0.25),
                        m.VGroup(mods[2], mods[3]).arrange(m.RIGHT, buff=0.25),
                        m.VGroup(mods[4], mods[5]).arrange(m.RIGHT, buff=0.25))
        grid = rows.arrange(m.DOWN, buff=0.18)
        thanks = title_text("Thanks for watching!", 42, INK)
        mascot = rover(0.6)
        foot = m.VGroup(mascot, thanks).arrange(m.RIGHT, buff=0.4)
        group = m.VGroup(top, grid, foot).arrange(m.DOWN, buff=0.45).move_to(0.05 * m.UP)
        assert grid.width < 12.8 and group.height < 7.0 and brand.width < TITLE_WIDTH

        self.say(
            "[That's Trial and Reward]: [from slot machines to reasoning models.] [Thanks for watching!]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(brand, shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(sub, shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in mods], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.Write(thanks)),
        )
        self.wait(1.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)
