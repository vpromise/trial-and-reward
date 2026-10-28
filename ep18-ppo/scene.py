"""Trial and Reward · Episode 18 · Proximal Policy Optimization (Module 5 · Learning the Policy)

Reuse each batch several times without stepping too far. The running example stays CartPole (Episodes 12–17): a
policy trained with PPO on this Mac balances for all 500 steps, sampling every move (trajectories.json, replayed).
Episodes 16–17 play a batch, take one gradient step and throw the batch away; playing is the slow part, so reuse the
batch — but after one step the data comes from an older policy, and a step too far wrecks the policy that plays the
next batch (illustrated by a policy collapsed to "always push right", simulated here with cartpole_step). The
importance ratio ρ = π_new/π_old re-weights old samples (example 0.50 → 0.60, ρ = 1.2); the surrogate Ê[ρÂ] has
Episode 16's gradient at ρ = 1 and no brake. TRPO (Schulman et al. 2015) fences the step with an average KL ≤ 0.01
(the example step: KL 0.020, TRPO would stop near 0.57; second-order solver). PPO (Schulman et al. 2017) clips the
ratio to [0.8, 1.2] and keeps the smaller term: ppo_clip_objective plotted against the ratio for Â > 0 and Â < 0,
the ratio sliding inside the band, the objective flat outside it, the pessimistic sloped side. The paper's Table 1
(ε = 0.2 best: 0.82; 0.76 at 0.1, 0.70 at 0.3, −0.39 without clipping). The loop on this Mac: 8 CartPoles × 64 steps
= 512 samples, 10 passes × 4 minibatches = 40 gradient steps per batch. Measured (5 seeds × 300k steps): solved 5/5
in 84k ± 24k steps and never fell back; without the clip 5/5 but 4 of 5 fell back (final 387 ± 154); the clip touched
3.7% of samples, KL per update 0.003 vs 0.105 (35×); 20 passes at lr 1e-3: 58k vs 234k steps (KL 2.7 per update).
Why PPO became the default: simple, OpenAI Five (clip 0.2, beat OG 2–0 on 13 April 2019), RLHF (Episode 21); its
price, a critic (GRPO, Episode 23). Recap: reuse, ratio, clip. Episode 19: continuous control.

Every number on screen is computed below from rl_reference.py and asserted, loaded with measured(...), or cited.

Sources (facts dated October 2026):
- research/experiments/deep_rl.md + deep_rl.json (measured("deep_rl", ...)): headline.ppo / ppo_noclip /
  ppo_aggressive / ppo_noclip_aggressive (solved seeds, steps to solve mean ± sample std, final avg100, clip_fraction,
  approx_kl), methods.*.hp (8 envs × 64 steps, 10 epochs × 4 minibatches, Adam 3e-4, clip 0.2, γ 0.99, GAE λ 0.95,
  4-64-64 tanh policy and value nets, 300k steps; aggressive: 20 epochs, lr 1e-3), methods.*.per_seed
  (min_avg100_after_solve), curves.*.avg100 (mean and per seed), ppo_updates.*.approx_kl (per update, consecutive
  updates averaged to ≤ 200 points). Seeds 10–14. "Solved" = mean return of the last 100 training episodes ≥ 475.
- research/experiments/trajectories.json: ppo_cartpole (run ppo_cartpole_clip_s10, final weights, reset seed 2026,
  actions sampled, 500 steps, truncated); p_push_right recomputed here with load_policy + softmax.
- research/notes.md §18: TRPO (Schulman, Levine, Moritz, Jordan & Abbeel, ICML 2015; δ = 0.01 "for all experiments");
  PPO (Schulman, Wolski, Dhariwal, Radford & Klimov, arXiv 1707.06347, 2017): Eq. 7 L^CLIP, Table 1 (7 simulated
  robotics tasks × 3 seeds, 1M steps, normalised random = 0 / best = 1: no clipping −0.39, ε 0.1 0.76, ε 0.2 0.82,
  ε 0.3 0.70; checked against the PDF, page 6), "TRPO is relatively complicated, and is not compatible with
  architectures that include noise (such as dropout) or parameter sharing"; OpenAI blog 20 Jul 2017 ("PPO has become
  the default reinforcement learning algorithm at OpenAI because of its ease of use and good performance"); Berner et
  al. 2019 (OpenAI Five: beat OG 2–0 on 13 April 2019, PPO clipping 0.2, GAE λ 0.95, batches up to 120 × 16 × 1,536 =
  2,949,120 timesteps, one run from 30 June 2018 to 22 April 2019 = ten months, 770 ± 50 PFlop/s·days, "PPO was the
  first to show initial learning progress"); Ziegler et al. 2019, Stiennon et al. 2020, Ouyang et al. 2022 trained
  their policies with PPO. Appendix C row 18 checked (ε = 0.2: 0.82; OG 2–0 on 13 April 2019).

Illustrative (said "for example" or labelled on screen): the batch sketch of section 1 (4 × 10 squares); the
collapsed policy "always push right" (simulated from reset seed 2026 with cartpole_step: the pole passes 12° after 9
steps); the ratio example 0.50 → 0.60 (its KL and TRPO's stopping point are computed); the TRPO landscape sketch; the
minibatch picks in section 6 (default_rng(18)).

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

PREFIX = "ep18"
DEEP = "deep_rl"
SOLVED = 475
SEEDS = 5

# ---------------------------------------------------------------------------- measured on this Mac

HP = measured(DEEP, "methods", "ppo", "hp")
HPN = measured(DEEP, "methods", "ppo_noclip", "hp")
HPA = measured(DEEP, "methods", "ppo_aggressive", "hp")
HPNA = measured(DEEP, "methods", "ppo_noclip_aggressive", "hp")
assert (HP["n_envs"], HP["n_steps"], HP["epochs"], HP["minibatches"]) == (8, 64, 10, 4)
assert HP["clip"] and not HPN["clip"] and HPA["clip"] and not HPNA["clip"] and HP["clip_eps"] == 0.2
assert HP["lr"] == 3e-4 and HP["gamma"] == 0.99 and HP["gae_lambda"] == 0.95 and HP["hidden"] == [64, 64]
assert HP["total_steps"] == 300_000 and HPN["epochs"] == 10 and HPN["lr"] == HP["lr"]
assert (HPA["epochs"], HPA["lr"]) == (20, 1e-3) and (HPNA["epochs"], HPNA["lr"]) == (20, 1e-3)
EPS = HP["clip_eps"]
N_ENVS, N_STEPS, EPOCHS, N_MB = HP["n_envs"], HP["n_steps"], HP["epochs"], HP["minibatches"]
BATCH = N_ENVS * N_STEPS
MB_SIZE = BATCH // N_MB
GRAD_STEPS = EPOCHS * N_MB
assert (BATCH, MB_SIZE, GRAD_STEPS) == (512, 128, 40)             # said: "five hundred twelve", "forty steps"

RUNS = ("ppo", "ppo_noclip", "ppo_aggressive", "ppo_noclip_aggressive")
H = {k: measured(DEEP, "headline", k) for k in RUNS}
assert all(H[k]["solved_seeds"] == "5/5" for k in RUNS)


def k_steps(h):
    return round(h["steps_to_solve_mean"] / 1000), round(h["steps_to_solve_std"] / 1000)


PPO_K, NC_K, AG_K, NCA_K = (k_steps(H[k]) for k in RUNS)
assert (PPO_K, NC_K, AG_K, NCA_K) == ((84, 24), (107, 24), (58, 6), (234, 76))   # said: 84k, 58k, 234k
assert round(H["ppo_noclip_aggressive"]["steps_to_solve_mean"] / H["ppo_aggressive"]["steps_to_solve_mean"]) == 4
FINAL = (round(H["ppo"]["final_avg100_mean"]), round(H["ppo"]["final_avg100_std"]))
NC_FINAL = (round(H["ppo_noclip"]["final_avg100_mean"]), round(H["ppo_noclip"]["final_avg100_std"]))
assert FINAL == (500, 0) and NC_FINAL == (387, 154)                # said: "about three hundred eighty-seven"
CLIP_PCT = round(100 * H["ppo"]["clip_fraction_mean"], 1)
assert CLIP_PCT == 3.7 and CLIP_PCT < 4                            # said: "three point seven", "under four percent"
KL_CLIP, KL_NC = H["ppo"]["approx_kl_mean"], H["ppo_noclip"]["approx_kl_mean"]
assert round(KL_CLIP, 3) == 0.003 and round(KL_NC, 3) == 0.105
FURTHER = round(KL_NC / KL_CLIP)
assert FURTHER == 35                                               # said: "about thirty-five times further"
KL_AG, KL_NCA = H["ppo_aggressive"]["approx_kl_mean"], H["ppo_noclip_aggressive"]["approx_kl_mean"]
assert round(KL_AG, 4) == 0.0067 and round(KL_NCA, 1) == 2.7
NC_OUT = round(100 * H["ppo_noclip"]["clip_fraction_mean"])
assert NC_OUT == 14                                                # no clip: 14% of ratios outside [0.8, 1.2]

PER = {k: measured(DEEP, "methods", k, "per_seed") for k in ("ppo", "ppo_noclip")}
assert len(PER["ppo"]) == len(PER["ppo_noclip"]) == SEEDS
assert all(s["min_avg100_after_solve"] >= SOLVED for s in PER["ppo"])          # said: "none ever fell back"
NC_DIPPED = [i for i, s in enumerate(PER["ppo_noclip"]) if s["min_avg100_after_solve"] < SOLVED]
NC_ENDED_LOW = [i for i, s in enumerate(PER["ppo_noclip"]) if s["final_avg100"] < SOLVED]
assert len(NC_DIPPED) == 4 and len(NC_ENDED_LOW) == 2                         # said: "four of the five fell back"

CURVES = {k: measured(DEEP, "curves", k) for k in ("ppo", "ppo_noclip")}
X_K = [x / 1000 for x in CURVES["ppo"]["x_steps"]]
assert CURVES["ppo_noclip"]["x_steps"] == CURVES["ppo"]["x_steps"] and round(X_K[-1]) == 300
SEED_C = {k: CURVES[k]["avg100_per_seed"] for k in CURVES}
for k in CURVES:                                                    # per-seed curves line up with per_seed results
    assert all(abs(c[-1] - s["final_avg100"]) < 1 for c, s in zip(SEED_C[k], PER[k]))
UPD = {k: measured(DEEP, "ppo_updates", k) for k in ("ppo", "ppo_noclip")}
U_X_K = [x / 1000 for x in UPD["ppo"]["x_steps"]]
assert UPD["ppo_noclip"]["x_steps"] == UPD["ppo"]["x_steps"]
assert abs(np.mean(UPD["ppo"]["approx_kl"]["mean"]) - KL_CLIP) < 1e-4
assert abs(np.mean(UPD["ppo_noclip"]["approx_kl"]["mean"]) - KL_NC) < 0.005
KL_LOG = {k: [math.log10(v) for v in UPD[k]["approx_kl"]["mean"]] for k in UPD}
assert -3 < min(min(v) for v in KL_LOG.values()) and max(max(v) for v in KL_LOG.values()) < 0

# ---------------------------------------------------------------------------- the trained policy (replayed, not simulated)

POL = measured("trajectories", "policies", "ppo_cartpole")
SRC = measured("trajectories", "sources", "ppo_cartpole")
GOOD = np.array(POL["states"])
assert POL["length"] == 500 and POL["truncated"] and len(GOOD) == 501 and POL["action_mode"] == "sampled"
assert SRC["run"] == "ppo_cartpole_clip_s10" and SRC["weights"] == "final"
assert np.abs(cartpole_reset(np.random.default_rng(2026)) - GOOD[0]).max() < 1e-4
NET = load_policy("ppo_cartpole")
P_RIGHT = np.array([softmax(NET(cartpole_obs(s)))[1] for s in GOOD[:-1]])
assert np.abs(P_RIGHT - np.array(POL["p_push_right"])).max() < 1e-3            # the replay is this network

# a policy that collapsed to "always push right" (illustration, simulated with the exact CartPole port)
_s = cartpole_reset(np.random.default_rng(2026))
COLLAPSE = [_s]
while True:
    _s, _, _done = cartpole_step(_s, 1)
    COLLAPSE.append(_s)
    if _done:
        break
COLLAPSE = np.array(COLLAPSE)
assert len(COLLAPSE) - 1 == 9 and abs(COLLAPSE[-1][2]) > CARTPOLE["theta_threshold"]

# ---------------------------------------------------------------------------- the ratio, TRPO, the clip (computed)

OLD, NEW = np.array([0.5, 0.5]), np.array([0.4, 0.6])                # for example: push left, push right
RHO = NEW / OLD
assert np.allclose(RHO, [0.8, 1.2])                                 # said: "a ratio of one point two"
KL_EX = float(np.sum(OLD * np.log(OLD / NEW)))                      # KL(old ‖ new) at this state
DELTA = 0.01                                                        # TRPO's δ (Schulman et al., 2015)
assert round(KL_EX, 3) == 0.020 and KL_EX > DELTA                   # said: "zero point zero two"
TRPO_P = (1 + math.sqrt(1 - math.exp(-2 * DELTA))) / 2              # KL((½, ½) ‖ (1 − p, p)) = δ
assert abs(np.sum(OLD * np.log(OLD / np.array([1 - TRPO_P, TRPO_P]))) - DELTA) < 1e-12 and round(TRPO_P, 2) == 0.57

R = np.linspace(0.0, 2.0, 401)
OBJ = {+1: ppo_clip_objective(R, 1.0, EPS), -1: ppo_clip_objective(R, -1.0, EPS)}
assert np.allclose(OBJ[+1][R >= 1 + EPS], 1 + EPS) and np.allclose(OBJ[-1][R <= 1 - EPS], -(1 - EPS))  # the flat parts
assert np.allclose(OBJ[+1][R < 1 - EPS], R[R < 1 - EPS]) and np.allclose(OBJ[-1][R > 1 + EPS], -R[R > 1 + EPS])
assert np.allclose(OBJ[+1][(R > 0.8) & (R < 1.2)], R[(R > 0.8) & (R < 1.2)])
# notes §18 worked example: one sample at a time
assert [float(ppo_clip_objective(r, a)) for r, a in ((1.0, 2), (1.5, 2), (0.7, 2), (0.7, -1), (1.5, -1))] == \
    [2.0, 2.4, 1.4, -0.8, -1.5]

# ---------------------------------------------------------------------------- cited (notes §18)

TABLE1 = [("no clipping", -0.39), ("clip ε = 0.1", 0.76), ("clip ε = 0.2", 0.82), ("clip ε = 0.3", 0.70)]
assert max(TABLE1, key=lambda t: t[1])[0] == "clip ε = 0.2" and min(TABLE1, key=lambda t: t[1])[1] < 0
FIVE_BATCH = 120 * 16 * 1536
assert FIVE_BATCH == 2_949_120 and round(FIVE_BATCH / 1e6) == 3      # said: "about three million"

# ---------------------------------------------------------------------------- look

CLOCK = recolored("clock", MUTED, CARD, prefix=PREFIX)
PAD = recolored("gamepad", STATE, TEAL_FILL, prefix=PREFIX)
LLM = recolored("llm", ACTION, CORAL_FILL, prefix=PREFIX)
CODE = recolored("code", INK, CARD, prefix=PREFIX)
CLIP_C, NOCLIP_C = VIOLET, RED            # the two runs: with the clip (the band's colour), without it
BATCH_C = mix(CARD, TEAL, 0.45)           # a played sample
OLD_C = mix(CARD, TEAL, 0.2)              # a sample from an older policy
BAD_C = mix(CARD, RED, 0.55)              # a sample from a wrecked policy


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
    assert n == len(words)
    pos = 0
    for p, col in parts:
        for i in range(pos, pos + len(p)):
            if i in glyph_of:
                words[glyph_of[i]].set_color(col)
        pos += len(p) + 1
    return words


def play(view, states, seconds, counter=None, make=None):
    """Replay a trajectory on a CartPoleView, with an optional step counter that follows it."""
    states = np.asarray(states, dtype=float)
    last = len(states) - 1

    def step(_, alpha):
        i = alpha * last
        j = int(np.floor(i))
        k = min(j + 1, last)
        f = i - j
        view.set_state((1 - f) * states[j][0] + f * states[k][0], (1 - f) * states[j][2] + f * states[k][2])
        if counter is not None:
            counter.become(make(int(round(i))))

    return m.UpdateFromAlphaFunc(view, step, run_time=seconds, rate_func=m.linear)


def topple(view, x, theta0, theta1=-1.35, seconds=0.6):
    """The pole keeps falling after the episode ends."""
    def step(_, alpha):
        view.set_state(x, theta0 + (theta1 - theta0) * alpha)
    return m.UpdateFromAlphaFunc(view, step, run_time=seconds, rate_func=m.rush_into)


def tint(host, cells, color, seconds=1.0, stroke=None):
    """Cells of `host` (already on screen) change colour one after another (a batch played, emptied, replaced)."""
    n = len(cells)

    def step(_, alpha):
        for k, cl in enumerate(cells):
            if k <= alpha * n - 0.5:
                cl.set_fill(color, 1)
                if stroke:
                    cl.set_stroke(stroke, 1.5)

    return m.UpdateFromAlphaFunc(host, step, run_time=seconds, rate_func=m.linear)


def objective(r, a):
    return float(ppo_clip_objective(r, a, EPS))


def grad_arrow(plot, r, a, length=1.0):
    """The direction that raises one sample's clipped objective, drawn at the ratio r (None where it is flat)."""
    h = 1e-4
    slope = (objective(r + h, a) - objective(r - h, a)) / (2 * h)
    p0 = plot.coords(r, objective(r, a))
    if abs(slope) < 1e-6:
        return m.Arrow(p0, p0 + [0.01, 0, 0], buff=0, color=SIGNAL, stroke_width=5).set_opacity(0)
    dx = 0.2 * np.sign(slope)
    p1 = plot.coords(r + dx, objective(r, a) + slope * dx)
    v = (p1 - p0) / np.linalg.norm(p1 - p0) * length
    return m.Arrow(p0, p0 + v, buff=0.12, color=SIGNAL, stroke_width=6, max_tip_length_to_length_ratio=0.3)


def slide(plot, dot, arrow_, readout, a, r0, r1, seconds):
    """Move one sample's ratio from r0 to r1 along its clipped objective, with the gradient arrow riding along."""
    def step(_, alpha):
        r = r0 + (r1 - r0) * alpha
        dot.move_to(plot.coords(r, objective(r, a)))
        arrow_.become(grad_arrow(plot, r, a))
        readout.become(ratio_readout(r).move_to(readout.get_center()))

    return m.UpdateFromAlphaFunc(dot, step, run_time=seconds, rate_func=m.smooth)


def backward(net, color=SIGNAL, run_time=1.0):
    """A gradient step: a PINK flash sweeping from the outputs back to the inputs."""
    r = net.columns[0][0].width / 2
    flashes = []
    for a, b in reversed(list(zip(net.columns[:-1], net.columns[1:]))):
        lines = m.VGroup(*[m.Line(q.get_center(), p.get_center(), buff=r).set_stroke(color, 3) for p in a for q in b])
        flashes.append(m.ShowPassingFlash(lines, time_width=0.6))
    return m.LaggedStart(*flashes, lag_ratio=0.7, run_time=run_time)


def ratio_readout(r):
    return code(f"ratio {r:.2f}", 20, ACTION, weight=m.BOLD)


class Panel:
    """One CartPole in a card: title, subtitle, a step counter."""

    def __init__(self, cx, cy, w, h, title, sub, color):
        self.box = card(w, h, color, CARD, radius=0.2, stroke=2.5).move_to([cx, cy, 0])
        top, self.left = cy + h / 2, cx - w / 2 + 0.35
        self.right_x, self.count_y, self.top = cx + w / 2 - 0.35, cy - h / 2 + 0.42, top
        self.title = left_at(text(title, 26, INK, weight=m.BOLD), self.left, top - 0.42)
        self.sub = left_at(text(sub, 20, MUTED), self.left, top - 0.85)
        self.view = CartPoleView(center=(cx, cy - 1.25, 0), units_per_m=1.0, pole_units=1.7)
        self.counter = self.count(0)

    def count(self, n):
        return left_at(code(f"step {n}", 22, INK), self.left, self.count_y)

    def everything(self):
        return m.VGroup(self.box, self.title, self.sub, self.view, self.counter)


# ---------------------------------------------------------------------------- the episode


class Ep18ProximalPolicyOptimization(Episode):
    number = 18

    def lesson(self):
        self.one_batch()
        self.the_ratio()
        self.trust_region()
        self.the_clip()
        self.why_point_two()
        self.the_loop()
        self.on_this_mac()
        self.step_size()
        self.the_default()
        self.recap()

    # ------------------------------------------------------------------ 1 · one batch, used once

    def one_batch(self):
        self.next_section("1 · One batch, used once", notes=(
            f"Left: the policy trained with PPO (clip 0.2) on this Mac — run {SRC['run']}, final weights after "
            f"{SRC['env_steps']:,} env steps — replayed from trajectories.json (reset seed 2026, every move sampled from "
            "its softmax; 500 steps, CartPole-v1's limit, truncated). Episodes 16–17 (REINFORCE, actor-critic) use each "
            "batch for one gradient step. Right: a sketch of a batch (4 × 10 squares, not the real 512). The 'wrecked' "
            "policy is an ILLUSTRATION: a policy collapsed to 'always push right', simulated here from the same reset "
            f"seed with cartpole_step — the pole passes 12° after {len(COLLAPSE) - 1} steps. Notes §18: 'a too-large "
            "step can wreck the policy. The next batch is then collected by the wrecked policy.'"))
        pan = Panel(-3.35, -0.3, 6.3, 4.9, "a policy trained on this Mac", "every move is sampled", ACTION)
        pan.view.set_state(GOOD[0][0], GOOD[0][2])
        bot = rover(0.42).move_to([pan.right_x - 0.25, pan.top - 0.55, 0])
        held = right_at(pill(f"{len(GOOD) - 1} steps · the limit", GREEN, GREEN_FILL, 20), pan.right_x, pan.count_y)
        fell = right_at(pill(f"fell · {len(COLLAPSE) - 1} steps", RED, RED_FILL, 20), pan.right_x, pan.count_y)
        new_pol = right_at(pill("updated policy", ACTION, CORAL_FILL, 20), pan.right_x, pan.top - 1.3)
        wrecked = right_at(pill("wrecked: always right", RED, RED_FILL, 20), pan.right_x, pan.top - 1.3)
        ill = left_at(cite("for example"), pan.left, pan.top - 1.3)

        gx = 3.25
        grid = cell_grid(4, 10, size=0.28, gap=0.07, color=EDGE, fill=CARD).move_to([gx, 1.1, 0])
        cells = [grid[r][c] for r in range(4) for c in range(10)]
        lab1 = left_at(text("1 · play a batch", 22, INK, weight=m.BOLD), grid.get_left()[0], 2.15)
        lab_old = left_at(text("played by the old policy", 22, MUTED, weight=m.BOLD), grid.get_left()[0], 2.15)
        lab_bad = left_at(text("next batch: the wrecked policy", 22, RED, weight=m.BOLD), grid.get_left()[0], 2.15)
        step_p = pill("2 · one gradient step", SIGNAL, PINK_FILL, 22).move_to([gx, -0.55, 0])
        down1 = arrow(grid.get_bottom(), step_p.get_top(), MUTED)
        lab3 = text("3 · throw the batch away", 22, MUTED).move_to([gx, -1.75, 0])
        down2 = arrow(step_p.get_bottom(), lab3.get_top(), MUTED)
        clock = icon(CLOCK, 0.42).next_to(lab1, m.RIGHT, buff=0.45)
        slow = cite("the slow part").next_to(clock, m.RIGHT, buff=0.15)
        again = m.CurvedArrow(step_p.get_right() + [0.05, 0.1, 0], grid.get_right() + [0.1, -0.3, 0], angle=1.6,
                              color=INK, stroke_width=4, tip_length=0.2)
        reuse = text("reuse?", 22, INK, weight=m.BOLD).next_to(again, m.RIGHT, buff=0.1)

        self.say(
            "[Here's a policy trained on this Mac], [balancing the pole for all five hundred steps], "
            "[rolling the dice on every move.]",
            m.AnimationGroup(self.new_header("01 · one batch", "Played once, then thrown away"),
                             m.FadeIn(pan.everything(), shift=0.1 * m.UP), m.FadeIn(bot, shift=0.1 * m.DOWN)),
            m.Succession(play(pan.view, GOOD, 6.0, pan.counter, pan.count), m.FadeIn(held, shift=0.1 * m.LEFT, run_time=0.4)),
            m.Circumscribe(pan.sub, color=ACTION, buff=0.08),
        )
        self.say(
            "[Episodes sixteen and seventeen learned like this]: [play a batch of moves], [take one gradient step], "
            "[and throw the batch away.]",
            m.AnimationGroup(m.FadeIn(lab1, shift=0.1 * m.UP), m.FadeIn(grid)),
            tint(grid, cells, BATCH_C, 1.2, STATE),
            m.AnimationGroup(m.GrowArrow(down1), m.FadeIn(step_p, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.GrowArrow(down2), m.FadeIn(lab3, shift=0.1 * m.DOWN), tint(grid, cells[::-1], CARD, 0.9, EDGE)),
        )
        self.say(
            "[Playing is the slow part.] [So why not learn from each batch several times?]",
            m.AnimationGroup(m.FadeIn(clock, scale=0.7), m.FadeIn(slow), tint(grid, cells, BATCH_C, 2.0, STATE)),
            m.AnimationGroup(m.Create(again), m.FadeIn(reuse, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The catch: after one step, the policy has changed], [so the batch was played by an older one.]",
            m.AnimationGroup(m.FadeIn(new_pol, shift=0.1 * m.LEFT), m.Circumscribe(step_p, color=SIGNAL, buff=0.08)),
            m.AnimationGroup(m.FadeOut(lab1), m.FadeOut(clock), m.FadeOut(slow), m.FadeIn(lab_old),
                             tint(grid, cells, OLD_C, 0.8, EDGE)),
        )
        self.say(
            "[And one step too far can wreck the policy], [which then plays the next batch itself.]",
            m.AnimationGroup(m.FadeOut(held), m.FadeOut(new_pol), m.FadeIn(wrecked), m.FadeIn(ill),
                             m.Succession(play(pan.view, COLLAPSE, 1.3, pan.counter, pan.count),
                                          topple(pan.view, COLLAPSE[-1][0], COLLAPSE[-1][2]),
                                          m.FadeIn(fell, shift=0.1 * m.LEFT, run_time=0.4))),
            m.AnimationGroup(m.FadeOut(lab_old), m.FadeIn(lab_bad), tint(grid, cells, BAD_C, 1.4, RED)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the ratio

    def the_ratio(self):
        self.next_section("2 · The importance ratio", notes=(
            "ρ = π_new(a|s) / π_old(a|s) (notes §18 writes ρ so that r stays the reward; PPO's paper calls it r_t(θ)). "
            "The probabilities 0.50/0.50 → 0.40/0.60 are an EXAMPLE (ratios 0.80 and 1.20, computed). The surrogate "
            "objective L(θ) = Ê[ρ_t Â_t] (PPO Eq. 6, 'CPI'); Â is the advantage of Episode 17 (GAE). At θ = θ_old every "
            "ratio is 1 and ∇ρ = ∇π_θ/π_old = ∇ log π_θ, so the surrogate's gradient is Episode 16's policy gradient. "
            "The faded line: the estimate is only trustworthy near ρ = 1 (an illustration of trust, not a measured "
            "quantity); ρÂ itself grows without bound."))
        rho_eq = tex(r"\rho\;=", 46, ACTION)
        num = tex(r"\pi_{\mathrm{new}}(a \mid s)", 42, ACTION)
        den = tex(r"\pi_{\mathrm{old}}(a \mid s)", 42, MUTED)
        frac_w = max(num.width, den.width) + 0.2
        fbar = m.Line([0, 0, 0], [frac_w, 0, 0]).set_stroke(INK, 3)
        left_at(rho_eq, -6.2, 1.75)
        left_at(fbar, rho_eq.get_right()[0] + 0.25, 1.75)
        num.next_to(fbar, m.UP, buff=0.12)
        den.next_to(fbar, m.DOWN, buff=0.12)
        name = left_at(cite("the importance ratio"), fbar.get_right()[0] + 0.3, 1.75)

        base, scale = -2.0, 2.4
        old_bars, new_bars, old_nums, new_nums, labels, ratios = (m.VGroup() for _ in range(6))
        for i, (lab, po, pn) in enumerate(zip(("push left", "push right"), OLD, NEW)):
            cx = -4.75 + 2.75 * i
            ob = m.Rectangle(width=0.75, height=po * scale).set_stroke(MUTED, 2).set_fill(CARD, 1)
            ob.move_to([cx - 0.42, base + po * scale / 2, 0])
            nb = m.Rectangle(width=0.75, height=pn * scale).set_stroke(ACTION, 2.5).set_fill(CORAL_FILL, 1)
            nb.move_to([cx + 0.42, base + pn * scale / 2, 0])
            old_bars.add(ob)
            new_bars.add(nb)
            old_nums.add(code(f"{po:.2f}", 20, MUTED).next_to(ob, m.UP, buff=0.12))
            new_nums.add(code(f"{pn:.2f}", 20, ACTION, weight=m.BOLD).next_to(nb, m.UP, buff=0.12))
            labels.add(code(lab, 20, MUTED).move_to([cx, base - 0.3, 0]))
            ratios.add(code(f"ρ = {pn / po:.2f}", 22, ACTION, weight=m.BOLD).move_to([cx, base - 0.78, 0]))
        floor = m.Line([-6.0, base, 0], [-1.25, base, 0]).set_stroke(FAINT, 2)
        sw_old = m.Square(0.24).set_stroke(MUTED, 2).set_fill(CARD, 1)
        sw_new = m.Square(0.24).set_stroke(ACTION, 2.5).set_fill(CORAL_FILL, 1)
        leg_old = m.VGroup(sw_old, code("old policy", 20, MUTED)).arrange(m.RIGHT, buff=0.12)
        leg_new = m.VGroup(sw_new, code("new policy", 20, ACTION)).arrange(m.RIGHT, buff=0.12)
        left_at(m.VGroup(leg_old, leg_new).arrange(m.RIGHT, buff=0.45), -6.2, 0.45)
        example = left_at(cite("for example"), -1.85, 0.45)

        sur = formula([r"L(\theta)", "=", r"\hat{\mathbb{E}}\big[", r"\rho", r"\,\hat{A}", r"\big]"],
                      [INK, INK, INK, ACTION, SIGNAL, INK], 46).move_to([3.4, 1.8, 0])
        sur_lab = code("Â: the advantage · episode 17", 20, SIGNAL).next_to(sur, m.DOWN, buff=0.22)
        judge = cite("the surrogate: old samples, scored for the new policy").next_to(sur_lab, m.DOWN, buff=0.12)
        start = text("before the first step, every ratio is 1", 22, INK).move_to([3.4, -0.05, 0])
        ident = formula([r"\nabla_\theta\, \rho", "=", r"\nabla_\theta \log \pi_\theta(a \mid s)"],
                        [SIGNAL, INK, ACTION], 40).move_to([3.4, -0.85, 0])
        ep16 = pill("episode 16's policy gradient", ACTION, CORAL_FILL, 20).move_to([3.4, -1.75, 0])

        mini = Plot((0, 2.5), (0, 2.5), width=4.4, height=1.9, center=(3.4, -1.35, 0), x_ticks=[0, 1, 2],
                    y_ticks=[0, 1, 2], x_label="ratio ρ", y_label="ρ × Â, for Â = +1")
        segs = m.VGroup()
        rs = np.linspace(0, 2.5, 51)
        for a, b in zip(rs[:-1], rs[1:]):
            mid = (a + b) / 2
            segs.add(m.Line(mini.coords(a, a), mini.coords(b, b)).set_stroke(SIGNAL, 5, opacity=max(0.15, math.exp(-1.6 * abs(mid - 1)))))
        one = mini.dot(1, 1, ACTION, 0.08)
        trust = code("trusted near 1", 20, ACTION).next_to(mini.coords(1, 1), m.RIGHT, buff=0.25).shift(0.2 * m.DOWN)
        rising = m.Arrow(mini.coords(2.1, 1.7), mini.coords(2.45, 2.45) + [0.05, 0.25, 0], buff=0, color=RED,
                         stroke_width=5, max_tip_length_to_length_ratio=0.3)
        no_lim = text("no limit", 22, RED, weight=m.BOLD).next_to(rising, m.LEFT, buff=0.15).shift(0.15 * m.UP)

        self.say(
            "[The fix is a ratio.] [For each sample, divide the new policy's chance of that move] [by the old policy's chance.]",
            m.AnimationGroup(self.new_header("02 · the ratio", "Re-weight the old batch"), m.FadeIn(rho_eq), m.FadeIn(name)),
            m.FadeIn(num, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.GrowFromCenter(fbar), m.FadeIn(den, shift=0.1 * m.UP)),
        )
        self.say(
            "[For example, the old policy pushed right half the time], [the new one sixty percent]: [a ratio of one point two.]",
            m.AnimationGroup(m.FadeIn(floor), m.FadeIn(labels), m.FadeIn(leg_old), m.FadeIn(example),
                             *[m.GrowFromEdge(b, m.DOWN) for b in old_bars], m.FadeIn(old_nums)),
            m.AnimationGroup(m.FadeIn(leg_new), *[m.GrowFromEdge(b, m.DOWN) for b in new_bars], m.FadeIn(new_nums)),
            m.LaggedStart(m.FadeIn(ratios[1], shift=0.1 * m.UP), m.FadeIn(ratios[0], shift=0.1 * m.UP), lag_ratio=0.5),
        )
        self.say(
            "[Weight each advantage from episode seventeen by its ratio]: [the old batch now scores the new policy. That's the surrogate.]",
            m.AnimationGroup(m.FadeIn(sur, shift=0.1 * m.DOWN), m.FadeIn(sur_lab)),
            m.FadeIn(judge, shift=0.1 * m.UP),
        )
        self.say(
            "[Before the first step, every ratio is one], [and the gradient is exactly the policy gradient of episode sixteen.]",
            m.FadeIn(start, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(ident, shift=0.1 * m.UP), m.FadeIn(ep16, shift=0.1 * m.UP)),
        )
        self.say(
            "[But the further the policy drifts, the less the old batch can be trusted], [and the surrogate just keeps rising.]",
            m.AnimationGroup(m.FadeOut(start), m.FadeOut(ident), m.FadeOut(ep16), m.FadeIn(mini), m.Create(segs, run_time=2.2),
                             m.FadeIn(one, scale=0.5), m.FadeIn(trust)),
            m.AnimationGroup(m.GrowArrow(rising), m.FadeIn(no_lim, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · trust regions

    def trust_region(self):
        self.next_section("3 · Trust regions (TRPO)", notes=(
            "TRPO (Schulman, Levine, Moritz, Jordan & Abbeel, ICML 2015): maximise the surrogate L_θold(θ) subject to the "
            f"average KL D̄_KL(θold, θ) ≤ δ; 'We used δ = 0.01 for all experiments.' The example step (0.50/0.50 → "
            f"0.40/0.60) has KL(old ‖ new) = {KL_EX:.4f} at that state, over δ; a step that stays at KL = δ there stops at "
            f"p(right) = {TRPO_P:.4f} (computed). Solver: a natural-gradient step via conjugate gradient with "
            "Fisher-vector products, then a line search. The PPO paper: TRPO 'is relatively complicated, and is not "
            "compatible with architectures that include noise (such as dropout) or parameter sharing'. The landscape on "
            "the left is an ILLUSTRATION (contours of a made-up surrogate)."))
        O = np.array([-5.0, -1.9, 0])
        P = np.array([-2.0, 0.85, 0])
        u = (P - O) / np.linalg.norm(P - O)
        radius = 1.05
        rings = m.VGroup(*[m.Ellipse(width=2 * a, height=2 * b).move_to(P).set_stroke(FAINT, 2)
                           for a, b in ((0.4, 0.27), (0.85, 0.57), (1.35, 0.9), (1.85, 1.25))])
        peak = m.Dot(P, radius=0.07, color=MUTED)
        peak_lab = left_at(cite("the surrogate's peak"), -6.3, 2.05)
        peak_link = m.DashedLine(peak_lab.get_right() + [0.1, -0.05, 0], P + [-0.1, 0.05, 0], dash_length=0.08)
        peak_link.set_stroke(MUTED, 1.5)
        greedy = m.Arrow(O, P - 0.25 * u, buff=0.1, color=SIGNAL, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        old_dot = m.Dot(O, radius=0.09, color=ACTION).set_z_index(3)
        old_lab = tex(r"\theta_{\mathrm{old}}", 32, ACTION).move_to(O + [-0.42, -0.36, 0]).set_z_index(3)
        fence = m.Circle(radius=radius).move_to(O).set_stroke(VIOLET, 3).set_fill(VIOLET, 0.14)
        new_pt = O + radius * u
        new_dot = m.Dot(new_pt, radius=0.09, color=ACTION).set_z_index(3)
        new_lab = tex(r"\theta_{\mathrm{new}}", 32, ACTION).move_to(new_pt + [0.62, -0.22, 0]).set_z_index(3)
        fence_lab = text("trust region: KL ≤ δ", 22, VIOLET, weight=m.BOLD).move_to([O[0], -3.25, 0])
        ill = cite("illustration").move_to([-1.6, -3.25, 0])

        x0 = 0.5
        name = left_at(text("Trust Region Policy Optimization", 26, INK, weight=m.BOLD), x0, 2.15)
        src = left_at(cite("Schulman et al., 2015"), x0, 1.72)
        max_w = left_at(text("maximise", 22, MUTED), x0, 0.95)
        max_f = formula([r"\hat{\mathbb{E}}\big[", r"\rho", r"\,\hat{A}", r"\big]"], [INK, ACTION, SIGNAL, INK], 38)
        max_f.next_to(max_w, m.RIGHT, buff=0.25)
        sub_w = left_at(text("subject to", 22, MUTED), x0, 0.25)
        sub_f = formula([r"\hat{\mathbb{E}}\big[\mathrm{KL}(\pi_{\mathrm{old}} \,\|\, \pi_{\mathrm{new}})\big]", r"\le",
                         r"\delta = 0.01"], [INK, INK, VIOLET], 36)
        sub_f.next_to(sub_w, m.RIGHT, buff=0.25)
        alike = left_at(text("KL = 0 when two policies act alike", 22, INK), x0, -0.65)
        ours = left_at(colored([("our step 0.50 → 0.60:", MUTED), ("KL", INK), (f"= {KL_EX:.3f}", RED)], 22), x0, -1.1)
        stop = left_at(colored([("TRPO would stop near", MUTED), (f"{TRPO_P:.2f}", VIOLET)], 20), x0, -1.5)
        catch = m.VGroup(text("the catch: second-order maths", 22, INK, weight=m.BOLD),
                         text("conjugate gradients and a line search", 20, MUTED),
                         text("awkward with dropout or shared networks", 20, MUTED))
        catch.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(catch, x0, -2.55)

        self.say(
            "[In twenty fifteen, Trust Region Policy Optimization, or TRPO], [drew a fence around the old policy.]",
            m.AnimationGroup(self.new_header("03 · trust regions", "A fence around the old policy"), m.FadeIn(name),
                             m.FadeIn(src), m.FadeIn(rings), m.FadeIn(peak), m.FadeIn(peak_lab), m.FadeIn(peak_link),
                             m.FadeIn(old_dot, scale=0.5), m.FadeIn(old_lab), m.GrowArrow(greedy)),
            m.AnimationGroup(m.GrowFromCenter(fence), m.FadeIn(fence_lab, shift=0.1 * m.UP), m.FadeIn(ill),
                             greedy.animate.set_opacity(0.35), m.FadeIn(new_dot, scale=0.5), m.FadeIn(new_lab)),
        )
        self.say(
            "[Maximise the surrogate], [but keep the average KL divergence from the old policy under zero point zero one.]",
            m.AnimationGroup(m.FadeIn(max_w), m.FadeIn(max_f, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(sub_w), m.FadeIn(sub_f, shift=0.1 * m.LEFT), m.Circumscribe(fence, color=VIOLET)),
        )
        self.say(
            "[The KL is zero when two policies act alike.] [Our step from fifty to sixty percent scores zero point zero two]: "
            "[over the fence.]",
            m.FadeIn(alike, shift=0.1 * m.UP),
            m.FadeIn(ours, shift=0.1 * m.UP),
            m.FadeIn(stop, shift=0.1 * m.UP),
        )
        self.say(
            "[It works, but enforcing the fence takes second-order machinery], [and it's awkward with big shared networks.]",
            m.AnimationGroup(m.FadeIn(catch[0], shift=0.1 * m.UP), m.FadeIn(catch[1], shift=0.1 * m.UP)),
            m.FadeIn(catch[2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · the clip

    def clip_plot(self, a, cx):
        y_range, y_ticks = ((0, 2.2), [0, 1, 2]) if a > 0 else ((-2.2, 0.2), [-2, -1, 0])
        plot = Plot((0, 2), y_range, width=4.6, height=2.35, center=(cx, -1.05, 0), x_ticks=[0, 1, 2], y_ticks=y_ticks,
                    x_label="ratio ρ", y_label="objective")
        lo, hi = plot.coords(1 - EPS, y_range[0]), plot.coords(1 + EPS, y_range[1])
        band = m.Rectangle(width=hi[0] - lo[0], height=hi[1] - lo[1]).move_to((lo + hi) / 2)
        band.set_stroke(width=0).set_fill(VIOLET, 0.2)
        edges = m.VGroup(plot.vline(1 - EPS, VIOLET, 2), plot.vline(1 + EPS, VIOLET, 2))
        marks = m.VGroup(code(f"{1 - EPS:g}", 20, VIOLET).next_to(plot.coords(1 - EPS, y_range[1]), m.UP, buff=0.1),
                         code(f"{1 + EPS:g}", 20, VIOLET).next_to(plot.coords(1 + EPS, y_range[1]), m.UP, buff=0.1))
        marks[0].shift(0.12 * m.LEFT)
        marks[1].shift(0.12 * m.RIGHT)
        plain = m.DashedVMobject(plot.line(R, R * a, MUTED, 3), num_dashes=40)
        curve = plot.line(R, OBJ[a], SIGNAL, 6)
        if a > 0:
            wrong = plot.line(R[R <= 1 - EPS], OBJ[a][R <= 1 - EPS], SIGNAL, 11)
        else:
            wrong = plot.line(R[R >= 1 + EPS], OBJ[a][R >= 1 + EPS], SIGNAL, 11)
        return plot, band, edges, marks, plain, curve, wrong

    def the_clip(self):
        self.next_section("4 · The clip", notes=(
            "PPO (Schulman et al. 2017, Eq. 7): L^CLIP(θ) = Ê[min(ρ_t Â_t, clip(ρ_t, 1 − ε, 1 + ε) Â_t)], ε = 0.2. Both "
            "curves are rl_reference.ppo_clip_objective(ratio, ±1, 0.2) on 401 ratios in [0, 2] (asserted: flat at 1.2 for "
            "ratios ≥ 1.2 when Â > 0, flat at −0.8 for ratios ≤ 0.8 when Â < 0). Dashed: the unclipped ρÂ. The arrow is "
            "the direction that raises the sample's objective (its gradient); it vanishes on the flat parts. Thick pink: "
            "the 'wrong way' sides where the unclipped term is the minimum, so the gradient still flows — the bound is "
            "pessimistic. Notes §18 worked example (asserted): ρ = 1.5, Â = +2 → 2.4 (no gradient); ρ = 0.7, Â = +2 → 1.4 "
            "(flows); ρ = 0.7, Â = −1 → −0.8 (no gradient); ρ = 1.5, Â = −1 → −1.5 (flows). 'We only ignore the change in "
            "probability ratio when it would make the objective improve, and we include it when it makes the objective "
            "worse.' (§3)"))
        f = formula([r"L^{\mathrm{CLIP}}", "=", r"\hat{\mathbb{E}}\Big[\min\big(", r"\rho", r"\hat{A}", ",",
                     r"\operatorname{clip}(\rho,\, 1-\epsilon,\, 1+\epsilon)", r"\,\hat{A}", r"\big)\Big]"],
                    [INK, INK, INK, ACTION, SIGNAL, INK, VIOLET, SIGNAL, INK], 40).move_to([-1.3, 2.1, 0])
        eps_lab = code(f"ε = {EPS:g}", 22, VIOLET, weight=m.BOLD).next_to(f, m.RIGHT, buff=0.45)
        src = right_at(cite("Schulman et al., 2017"), 6.5, 1.75)
        L = self.clip_plot(+1, -3.4)
        Rt = self.clip_plot(-1, 3.4)
        t_left = text("a good move: Â > 0", 22, GREEN, weight=m.BOLD).move_to([-2.5, 1.15, 0])
        t_right = text("a bad move: Â < 0", 22, RED, weight=m.BOLD).move_to([4.3, 1.15, 0])

        def rider(plot, a, r0=1.0):
            dot = m.Dot(plot.coords(r0, objective(r0, a)), radius=0.1, color=ACTION).set_z_index(4)
            arr = grad_arrow(plot, r0, a)
            read = ratio_readout(r0)
            return dot, arr, read

        dl, al, rl = rider(L[0], +1)
        dr, ar, rr = rider(Rt[0], -1)
        left_at(rl, L[0].coords(0.05, 0)[0], L[0].coords(0, 2.0)[1])
        right_at(rr, Rt[0].coords(1.95, 0)[0], Rt[0].coords(0, 0.0)[1] + 0.05)
        flat_l = text("flat · no push", 20, SIGNAL, weight=m.BOLD).next_to(L[0].coords(1.62, 1.2), m.DOWN, buff=0.3)
        flat_r = text("flat · no push", 20, SIGNAL, weight=m.BOLD).next_to(Rt[0].coords(0.4, -0.8), m.DOWN, buff=0.3)
        back_l = text("pushed back", 20, SIGNAL, weight=m.BOLD).move_to(L[0].coords(0.4, 1.4))
        back_r = text("pushed back", 20, SIGNAL, weight=m.BOLD).move_to(Rt[0].coords(1.62, -0.7))

        self.say(
            "[In twenty seventeen, Proximal Policy Optimization, or PPO], [replaced the fence with a clip.]",
            m.AnimationGroup(self.new_header("04 · the clip", "Clip the ratio"), m.FadeIn(src), m.FadeIn(L[0]), m.FadeIn(Rt[0]),
                             m.FadeIn(t_left), m.FadeIn(t_right)),
            m.AnimationGroup(m.FadeIn(L[1]), m.FadeIn(Rt[1]), m.Create(L[2]), m.Create(Rt[2])),
        )
        self.say(
            "[Take the ratio times the advantage], [and again with the ratio clipped between zero point eight and one point two]: "
            "[keep the smaller.]",
            m.AnimationGroup(m.FadeIn(f[:6], shift=0.1 * m.DOWN), m.Create(L[4]), m.Create(Rt[4])),
            m.AnimationGroup(m.FadeIn(f[6:8], shift=0.1 * m.DOWN), m.FadeIn(eps_lab), m.FadeIn(L[3]), m.FadeIn(Rt[3])),
            m.AnimationGroup(m.FadeIn(f[8]), m.Create(L[5], run_time=1.4), m.Create(Rt[5], run_time=1.4)),
        )
        self.say(
            "[For a good move, with a positive advantage], [the objective rises as the new policy makes that move likelier.]",
            m.AnimationGroup(m.Circumscribe(t_left, color=GREEN, buff=0.08), m.FadeIn(dl, scale=0.5), m.FadeIn(al), m.FadeIn(rl)),
            slide(L[0], dl, al, rl, +1, 1.0, 1.17, 2.2),
        )
        self.say(
            "[Past one point two, it goes flat]: [no gradient, no more push.]",
            slide(L[0], dl, al, rl, +1, 1.17, 1.6, 1.8),
            m.FadeIn(flat_l, shift=0.1 * m.UP),
        )
        self.say(
            "[For a bad move, it's the mirror image]: [push it down to zero point eight], [then flat.]",
            m.AnimationGroup(m.Circumscribe(t_right, color=RED, buff=0.08), m.FadeIn(dr, scale=0.5), m.FadeIn(ar), m.FadeIn(rr)),
            slide(Rt[0], dr, ar, rr, -1, 1.0, 0.83, 1.6),
            m.AnimationGroup(slide(Rt[0], dr, ar, rr, -1, 0.83, 0.45, 1.4), m.FadeIn(flat_r, shift=0.1 * m.UP)),
        )
        self.say(
            "[But if a step went the wrong way], [the unclipped term still counts], [so that mistake gets corrected.]",
            m.AnimationGroup(m.Create(L[6]), m.Create(Rt[6])),
            m.AnimationGroup(m.FadeIn(back_l, shift=0.1 * m.UP), m.FadeIn(back_r, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(L[6], color=SIGNAL, buff=0.1), m.Circumscribe(Rt[6], color=SIGNAL, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · why 0.2

    def why_point_two(self):
        self.next_section("5 · Why 0.2", notes=(
            "PPO paper, Table 1 (checked against the PDF, p. 6): 7 simulated robotics tasks (HalfCheetah, Hopper, "
            "InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d; MuJoCo, OpenAI Gym) × 3 seeds = 21 runs, "
            "1M timesteps each; each run scored by its last 100 episodes, shifted and scaled per environment so that a "
            "random policy = 0 and the best result = 1, averaged. No clipping or penalty −0.39 ('because for one environment "
            "(half cheetah) it leads to a very negative score, which is worse than the initial random policy'); clipping "
            "ε = 0.1: 0.76, 0.2: 0.82, 0.3: 0.70; adaptive KL 0.68–0.74, fixed KL 0.62–0.72 (not drawn)."))
        x_zero, unit = -0.8, 5.0
        title = left_at(text("the PPO paper's test: 7 simulated robot tasks × 3 seeds", 24, INK, weight=m.BOLD), -6.3, 2.15)
        sub = left_at(cite("average normalised score: a random policy = 0, the best run = 1"), -6.3, 1.7)
        src = right_at(cite("Schulman et al., 2017, Table 1"), 6.5, -3.2)
        axis = m.Line([x_zero, 1.2, 0], [x_zero, -2.55, 0]).set_stroke(FAINT, 2)
        zero_lab = code("0", 20, MUTED).move_to([x_zero, -2.8, 0])
        one = m.Line([x_zero + unit, -2.6, 0], [x_zero + unit, -2.45, 0]).set_stroke(FAINT, 2)
        one_lab = code("1", 20, MUTED).move_to([x_zero + unit, -2.8, 0])
        rows = []
        for k, (lab, v) in enumerate(TABLE1):
            y = 0.75 - 0.9 * k
            col = RED if v < 0 else (VIOLET if v == 0.82 else MUTED)
            fill = RED_FILL if v < 0 else (VIOLET_FILL if v == 0.82 else CARD)
            w = abs(v) * unit
            bar = m.Rectangle(width=w, height=0.48).set_stroke(col, 2.5).set_fill(fill, 1)
            bar.move_to([x_zero + np.sign(v) * w / 2, y, 0])
            name = left_at(code(lab, 22, VIOLET if v > 0 else RED, weight=m.BOLD), -6.3, y)
            val = code(f"{v:.2f}".replace("-", "−"), 22, col if v != 0.70 and v != 0.76 else INK, weight=m.BOLD)
            if v > 0:
                val.next_to(bar, m.RIGHT, buff=0.15)
            else:
                val.next_to(bar, m.LEFT, buff=0.15)
            rows.append(m.VGroup(bar, name, val))
        best = pill("best", VIOLET, VIOLET_FILL, 20).next_to(rows[2][2], m.RIGHT, buff=0.25)
        below = text("below random", 20, RED).next_to(rows[0][0], m.DOWN, buff=0.12).align_to(rows[0][0], m.LEFT)

        self.say(
            "[Why zero point two?] [On seven simulated robot tasks in the paper], [it scored best: zero point eight two.]",
            m.AnimationGroup(self.new_header("05 · why 0.2", "The paper's own test"), m.FadeIn(title), m.FadeIn(sub),
                             m.FadeIn(src), m.Create(axis), m.FadeIn(zero_lab), m.Create(one), m.FadeIn(one_lab)),
            m.AnimationGroup(*[m.AnimationGroup(m.FadeIn(r_[1]), m.GrowFromEdge(r_[0], m.LEFT), m.FadeIn(r_[2]))
                               for r_ in rows[1:]]),
            m.AnimationGroup(m.Circumscribe(rows[2][0], color=VIOLET, buff=0.06), m.FadeIn(best, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Clipping at zero point one or zero point three did a little worse], "
            "[and no clipping at all scored minus zero point three nine]: [below a random policy.]",
            m.AnimationGroup(m.Circumscribe(rows[1][2], color=MUTED, buff=0.06), m.Circumscribe(rows[3][2], color=MUTED, buff=0.06)),
            m.AnimationGroup(m.FadeIn(rows[0][1]), m.GrowFromEdge(rows[0][0], m.RIGHT), m.FadeIn(rows[0][2])),
            m.FadeIn(below, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the loop

    def the_loop(self):
        self.next_section("6 · The loop", notes=(
            f"This Mac's PPO (methods.ppo.hp): {N_ENVS} CartPoles × {N_STEPS} steps = {BATCH} samples per update; "
            f"advantages by GAE (γ = {HP['gamma']}, λ = {HP['gae_lambda']}, Episode 17), normalised per minibatch; "
            f"{EPOCHS} epochs × {N_MB} minibatches of {MB_SIZE} = {GRAD_STEPS} Adam steps (lr {HP['lr']:g}, ε = {EPS:g}) on "
            "each batch; then θ_old ← θ and the batch is discarded (PPO paper, Algorithm 1). The paper's MuJoCo setting: "
            "horizon 2,048, 10 epochs, minibatch 64. Which squares light up for each minibatch: a random shuffle per pass "
            "(default_rng(18)), an illustration of the same rule."))
        size, gap = 0.11, 0.03
        grid = cell_grid(N_ENVS, N_STEPS, size=size, gap=gap, color=EDGE, fill=CARD).move_to([0.45, 1.0, 0])
        cells = [grid[r][c] for r in range(N_ENVS) for c in range(N_STEPS)]
        cols = [m.VGroup(*[grid[r][c] for r in range(N_ENVS)]) for c in range(N_STEPS)]
        envs = right_at(text(f"{N_ENVS} CartPoles", 22, STATE, weight=m.BOLD), grid.get_left()[0] - 0.25, 1.0)
        steps = text(f"{N_STEPS} steps each", 22, INK).next_to(grid, m.UP, buff=0.2)
        total = right_at(code(f"= {BATCH} samples", 22, STATE, weight=m.BOLD), grid.get_right()[0], 0.12)
        adv = left_at(code(f"advantages Â: GAE, λ = {HP['gae_lambda']:g} · episode 17", 20, SIGNAL), grid.get_left()[0], 0.12)
        scan = m.Line(grid.get_corner(m.UL) + [0, 0.05, 0], grid.get_corner(m.DL) + [0, -0.05, 0]).set_stroke(SIGNAL, 6)

        rng = np.random.default_rng(18)
        passes = [rng.permutation(BATCH) for _ in range(EPOCHS)]
        hi, lo = mix(CARD, TEAL, 0.95), mix(CARD, TEAL, 0.22)

        def counters(k, done):
            ep = (k // N_MB) + 1
            st = done
            line = colored([("pass", MUTED), (f"{min(ep, EPOCHS):>2} of {EPOCHS}", INK), ("    minibatch", MUTED),
                            (f"{(k % N_MB) + 1} of {N_MB}", STATE), ("    gradient steps", MUTED), (f"{st:>2}", SIGNAL)],
                           22, weight=m.BOLD)
            return left_at(line, -4.6, -0.75)

        count = counters(0, 0)
        mb_note = left_at(cite(f"{N_MB} minibatches of {MB_SIZE}, reshuffled every pass"), -4.6, -1.3)

        def highlight(k):
            idx = set(passes[min(k // N_MB, EPOCHS - 1)][(k % N_MB) * MB_SIZE:((k % N_MB) + 1) * MB_SIZE].tolist())
            for i, cl in enumerate(cells):
                cl.set_fill(hi if i in idx else lo, 1)

        def first_mb():
            def step(_, alpha):
                if alpha > 0.05:
                    highlight(0)
            return m.UpdateFromAlphaFunc(grid, step, run_time=0.6)

        state = {"k": -1}

        def epochs(seconds):
            def step(_, alpha):
                k = min(int(alpha * GRAD_STEPS), GRAD_STEPS - 1)
                if k != state["k"]:
                    state["k"] = k
                    highlight(k)
                    count.become(counters(k, k + 1))
                if alpha >= 1:
                    for cl in cells:
                        cl.set_fill(BATCH_C, 1)
            return m.UpdateFromAlphaFunc(grid, step, run_time=seconds, rate_func=m.linear)

        big = colored([(f"{GRAD_STEPS} gradient steps", SIGNAL), ("from one batch", INK)], 30, weight=m.BOLD).move_to([0, -2.05, 0])
        clipped = pill(f"every ratio clipped to [{1 - EPS:g}, {1 + EPS:g}]", VIOLET, VIOLET_FILL, 20).move_to([0, -2.85, 0])
        renew = formula([r"\theta_{\mathrm{old}}", r"\leftarrow", r"\theta"], [ACTION, INK, ACTION], 40).move_to([-3.6, -2.05, 0])
        thrown = cite("batch thrown away").move_to([0.6, -2.05, 0])
        net = NeuralNet(layers=(4, 6, 6, 2), width=1.7, height=1.15, center=(5.2, -2.45, 0), color=ACTION, radius=0.07)
        net_lab = cite("the policy").next_to(net, m.UP, buff=0.15)
        mac = right_at(cite("trained on this Mac"), 6.5, 2.4)

        x_start, x_end = grid.get_left()[0], grid.get_right()[0]

        def sweep(seconds):
            def step(_, alpha):
                scan.set_x(x_start + (x_end - x_start) * alpha)
            return m.UpdateFromAlphaFunc(scan, step, run_time=seconds, rate_func=m.linear)

        def fill_cols(color, seconds):
            def step(_, alpha):
                upto = alpha * N_STEPS
                for c, col in enumerate(cols):
                    if c < upto:
                        col.set_fill(color, 1).set_stroke(STATE if color != CARD else EDGE, 1)
            return m.UpdateFromAlphaFunc(grid, step, run_time=seconds, rate_func=m.linear)

        self.say(
            "[Here's the loop, as trained on this Mac.] [Eight CartPoles play sixty-four steps each]: [five hundred twelve samples.]",
            m.AnimationGroup(self.new_header("06 · the loop", "Ten passes over one batch"), m.FadeIn(grid), m.FadeIn(envs),
                             m.FadeIn(steps), m.FadeIn(mac)),
            fill_cols(BATCH_C, 2.2),
            m.FadeIn(total, shift=0.1 * m.UP),
        )
        self.say(
            "[Each sample gets its advantage once, as in episode seventeen], [then the batch is shuffled into four minibatches.]",
            m.AnimationGroup(m.FadeIn(adv), m.Succession(m.FadeIn(scan, run_time=0.2), sweep(1.6))),
            m.AnimationGroup(m.FadeOut(scan), m.FadeIn(count), m.FadeIn(mb_note), first_mb(), m.FadeIn(net), m.FadeIn(net_lab)),
        )
        self.say(
            "[One gradient step per minibatch, ten passes over the batch]: [forty steps from one batch, every ratio clipped.]",
            m.AnimationGroup(epochs(5.0), m.Succession(*[backward(net, SIGNAL, 1.0) for _ in range(5)])),
            m.AnimationGroup(m.FadeIn(big, shift=0.1 * m.UP), m.FadeIn(clipped, shift=0.1 * m.UP)),
        )
        self.say(
            "[Then the new policy becomes the old one], [the batch is thrown away], [and the carts play again.]",
            m.AnimationGroup(m.FadeOut(big), m.FadeOut(clipped), m.FadeIn(renew, shift=0.1 * m.UP)),
            m.AnimationGroup(fill_cols(CARD, 0.8), m.FadeIn(thrown)),
            fill_cols(BATCH_C, 1.8),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · on this Mac

    def run_legend(self, rows, y0=1.75):
        group = m.VGroup()
        for k, (name, color, lines) in enumerate(rows):
            sw = m.Line([0, 0, 0], [0.4, 0, 0]).set_stroke(color, 5)
            nm = text(name, 22, color, weight=m.BOLD)
            head = m.VGroup(sw, nm).arrange(m.RIGHT, buff=0.15)
            subs = m.VGroup(*[code(s, 20, INK) for s in lines]).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            item = m.VGroup(head, subs).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            left_at(item, 2.75, y0 - 1.55 * k)
            subs.align_to(nm, m.LEFT)
            group.add(item)
        return group

    def on_this_mac(self):
        self.next_section("7 · Trained on this Mac", notes=(
            f"deep_rl.json, CartPole-v1, {SEEDS} seeds (10–14) per variant, 300,032 steps each. Score = mean return of the "
            f"last 100 training episodes (sampled moves); solved at ≥ {SOLVED}. Thick: the mean over seeds "
            "(curves.*.avg100.mean); thin: each seed. Clip 0.2: 5/5 solved after "
            f"{H['ppo']['steps_to_solve_mean']:,.0f} ± {H['ppo']['steps_to_solve_std']:,.0f} steps (min "
            f"{H['ppo']['steps_to_solve_min']:,.0f}, max {H['ppo']['steps_to_solve_max']:,.0f}), "
            f"{H['ppo']['episodes_to_solve_mean']:.0f} episodes, {H['ppo']['wall_s_to_solve_mean']} s; no seed's average "
            "fell below 475 after solving; final 500 ± 0. No clip (identical, objective = ratio × advantage): 5/5 solved "
            f"after {H['ppo_noclip']['steps_to_solve_mean']:,.0f} ± {H['ppo_noclip']['steps_to_solve_std']:,.0f}; "
            f"{len(NC_DIPPED)} of 5 fell below 475 after solving, {len(NC_ENDED_LOW)} ended below it (final per seed "
            + ", ".join(f"{s['final_avg100']:.0f}" for s in PER["ppo_noclip"]) + f"); final {H['ppo_noclip']['final_avg100_mean']}"
            f" ± {H['ppo_noclip']['final_avg100_std']} (mean ± sample std)."))
        plot = Plot((0, 315), (0, 525), width=7.3, height=4.2, center=(-2.25, -0.55, 0), x_ticks=[0, 100, 200, 300],
                    y_ticks=[0, 100, 200, 300, 400], x_label="thousand environment steps",
                    y_label="score: average of the last 100 episodes")
        solved = plot.hline(SOLVED, INK, 2)
        p475 = plot.coords(0, SOLVED)
        solved_lab = m.VGroup(m.Line(p475, p475 + [-0.09, 0, 0]).set_stroke(INK, 2),
                              code(f"{SOLVED}", 20, INK, weight=m.BOLD).next_to(p475, m.LEFT, buff=0.16),
                              code("solved", 20, INK).next_to(plot.coords(8, SOLVED), m.DOWN, buff=0.1)
                              .align_to(plot.coords(8, 0), m.LEFT))
        meas = right_at(cite(f"trained on this Mac · {SEEDS} seeds each · mean and each seed"), 6.5, 2.35)
        clip_mean = plot.line(X_K, CURVES["ppo"]["avg100"]["mean"], CLIP_C, 6)
        clip_seeds = m.VGroup(*[plot.line(X_K, c, CLIP_C, 2).set_stroke(opacity=0.35) for c in SEED_C["ppo"]])
        nc_mean = plot.line(X_K, CURVES["ppo_noclip"]["avg100"]["mean"], NOCLIP_C, 6)
        nc_seeds = m.VGroup(*[plot.line(X_K, c, NOCLIP_C, 2).set_stroke(opacity=0.35) for c in SEED_C["ppo_noclip"]])
        nc_fell = m.VGroup(*[plot.line(X_K, SEED_C["ppo_noclip"][i], NOCLIP_C, 2.5).set_stroke(opacity=0.8) for i in NC_DIPPED])
        ends = m.VGroup(*[plot.dot(X_K[-1], SEED_C["ppo_noclip"][i][-1], RED, 0.08) for i in NC_ENDED_LOW])
        leg = self.run_legend([
            ("clip ε = 0.2", CLIP_C, [f"5 of 5 solved", f"{PPO_K[0]}k ± {PPO_K[1]}k steps", f"final {FINAL[0]} ± {FINAL[1]}"]),
            ("no clip", NOCLIP_C, [f"5 of 5 solved", f"{NC_K[0]}k ± {NC_K[1]}k steps", f"final {NC_FINAL[0]} ± {NC_FINAL[1]}"]),
        ], y0=1.55)
        never = code("never fell back", 20, GREEN, weight=m.BOLD).next_to(leg[0], m.DOWN, buff=0.1).align_to(leg[0][1], m.LEFT)
        leg[1].shift(0.35 * m.DOWN)
        fell = code(f"{len(NC_DIPPED)} of 5 fell back", 20, RED, weight=m.BOLD).next_to(leg[1], m.DOWN, buff=0.1)
        fell.align_to(leg[1][1], m.LEFT)

        self.say(
            "[Here it is, trained on this Mac]: [five seeds each, three hundred thousand steps.]",
            m.AnimationGroup(self.new_header("07 · on this Mac", "Solved, and it stays solved"), m.FadeIn(plot),
                             m.Create(solved), m.FadeIn(solved_lab)),
            m.FadeIn(meas, shift=0.1 * m.DOWN),
        )
        self.say(
            "[With the clip, every seed reached the solved line], [after about eighty-four thousand steps on average], "
            "[and none ever fell back.]",
            m.AnimationGroup(m.Create(clip_mean, run_time=2.4), m.LaggedStart(*[m.Create(s) for s in clip_seeds], lag_ratio=0.1,
                                                                                run_time=2.4), m.FadeIn(leg[0][0])),
            m.FadeIn(leg[0][1][:2], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(leg[0][1][2:], shift=0.1 * m.UP), m.FadeIn(never, shift=0.1 * m.UP)),
        )
        self.say(
            "[Without the clip, every seed got there too], [but four of the five fell back]: "
            "[the final average was about three hundred eighty-seven.]",
            m.AnimationGroup(m.Create(nc_mean, run_time=2.4), m.LaggedStart(*[m.Create(s) for s in nc_seeds], lag_ratio=0.1,
                                                                              run_time=2.4),
                             m.FadeIn(leg[1][0]), m.FadeIn(leg[1][1][:2])),
            m.AnimationGroup(clip_seeds.animate.set_stroke(opacity=0.15), m.Create(nc_fell, run_time=1.4),
                             m.FadeIn(fell, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(ends, scale=0.5), m.FadeIn(leg[1][1][2:], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · step size

    def step_size(self):
        self.next_section("8 · How far each update moves", notes=(
            "ppo_updates.*.approx_kl: per update (512 samples), mean over minibatches of (ρ − 1) − log ρ (Schulman's k3 "
            "estimator of KL(old ‖ new)), consecutive updates averaged to ≤ 200 points, mean of 5 seeds; log scale. Run "
            f"averages (headline): clip {KL_CLIP:.5f}, no clip {KL_NC:.5f} → ×{KL_NC / KL_CLIP:.1f}; exact KL after the "
            f"update {H['ppo']['kl_exact_post_mean']} vs {H['ppo_noclip']['kl_exact_post_mean']}. Clip fraction (share of "
            f"samples with |ρ − 1| > 0.2): {100 * H['ppo']['clip_fraction_mean']:.2f}% ± {100 * H['ppo']['clip_fraction_std']:.2f}; "
            f"without the clip {NC_OUT}% of ratios wander outside the band. Heavy reuse (20 epochs, Adam 1e-3): clip "
            f"{H['ppo_aggressive']['steps_to_solve_mean']:,.0f} ± {H['ppo_aggressive']['steps_to_solve_std']:,.0f} steps to "
            f"solve (KL {KL_AG:.4f}); no clip {H['ppo_noclip_aggressive']['steps_to_solve_mean']:,.0f} ± "
            f"{H['ppo_noclip_aggressive']['steps_to_solve_std']:,.0f} (KL {KL_NCA:.2f}). Bars: mean, whiskers ± sample std. "
            "deep_rl.md caveats: CartPole is easy, five seeds; the robust finding is the 35× larger policy steps."))
        plot = Plot((0, 315), (-3, 0.2), width=7.3, height=4.0, center=(-2.25, -0.6, 0), x_ticks=[0, 100, 200, 300],
                    x_label="thousand environment steps", y_label="KL per update, old → new (log scale)")
        ticks = m.VGroup()
        for e, lab in ((-3, "0.001"), (-2, "0.01"), (-1, "0.1"), (0, "1")):
            p = plot.coords(0, e)
            ticks.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2), code(lab, 20, MUTED).next_to(p, m.LEFT, buff=0.16))
        plot.add(ticks)
        kl_c = plot.line(U_X_K, KL_LOG["ppo"], CLIP_C, 4)
        kl_n = plot.line(U_X_K, KL_LOG["ppo_noclip"], NOCLIP_C, 4)
        meas = right_at(cite(f"trained on this Mac · {SEEDS} seeds · mean"), 6.5, 2.35)
        leg = self.run_legend([
            ("clip ε = 0.2", CLIP_C, [f"KL ≈ {KL_CLIP:.3f} per update", f"{CLIP_PCT}% of samples clipped"]),
            ("no clip", NOCLIP_C, [f"KL ≈ {KL_NC:.3f} per update", f"{NC_OUT}% outside the band"]),
        ], y0=1.55)
        times = title_text(f"{FURTHER}× further", 44, RED).next_to(leg[1], m.DOWN, buff=0.4).align_to(leg[1][1], m.LEFT)

        # heavy reuse: steps to solve, four runs
        x0, unit = -0.6, 4.3 / 300
        bars = m.VGroup()
        rows = [("clip · 10 passes", CLIP_C, H["ppo"], f"{KL_CLIP:.3f}"),
                ("no clip · 10 passes", NOCLIP_C, H["ppo_noclip"], f"{KL_NC:.3f}"),
                ("clip · 20 passes, lr 1e-3", CLIP_C, H["ppo_aggressive"], f"{KL_AG:.4f}"),
                ("no clip · 20 passes, lr 1e-3", NOCLIP_C, H["ppo_noclip_aggressive"], f"{KL_NCA:.1f}")]
        for k, (lab, col, h, kl) in enumerate(rows):
            y = 1.25 - 1.05 * k - (0.3 if k >= 2 else 0)
            mean, sd = h["steps_to_solve_mean"] / 1000, h["steps_to_solve_std"] / 1000
            w = mean * unit
            bar = m.Rectangle(width=w, height=0.42).set_stroke(col, 2.5).set_fill(FILLS[col], 1).move_to([x0 + w / 2, y, 0])
            whisk = m.VGroup(m.Line([x0 + (mean - sd) * unit, y, 0], [x0 + (mean + sd) * unit, y, 0]).set_stroke(INK, 2),
                             *[m.Line([x0 + v * unit, y - 0.12, 0], [x0 + v * unit, y + 0.12, 0]).set_stroke(INK, 2)
                               for v in (mean - sd, mean + sd)])
            name = left_at(code(lab, 20, col, weight=m.BOLD), -6.3, y + 0.17)
            klab = left_at(code(f"KL ≈ {kl} per update", 20, MUTED), -6.3, y - 0.2)
            val = code(f"{round(mean)}k ± {round(sd)}k", 20, INK, weight=m.BOLD).next_to(whisk, m.RIGHT, buff=0.15)
            bars.add(m.VGroup(bar, whisk, name, klab, val))
        b_axis = m.Line([x0, 1.75, 0], [x0, -2.9, 0]).set_stroke(FAINT, 2)
        b_title = left_at(text("steps to solve · thousand · mean ± std", 22, INK, weight=m.BOLD), x0, 2.15)
        split = m.DashedLine([-6.3, -0.75, 0], [6.3, -0.75, 0], dash_length=0.1).set_stroke(EDGE, 2)
        honest = pill("CartPole is easy, and five seeds is few", MUTED, CARD, 20).move_to([0, -3.35, 0])
        b_meas = left_at(cite(f"trained on this Mac · {SEEDS} seeds each"), -6.3, -2.85)

        self.say(
            "[Why the difference?] [Measure how far each update moves the policy]: [the KL divergence between old and new.]",
            m.AnimationGroup(self.new_header("08 · step size", "How far each update moves"), m.FadeIn(meas)),
            m.FadeIn(plot),
            m.Circumscribe(plot.y_label, color=MUTED, buff=0.06),
        )
        self.say(
            "[With the clip, about zero point zero zero three per update], "
            "[and the clip touched just three point seven percent of the samples.]",
            m.AnimationGroup(m.Create(kl_c, run_time=2.2), m.FadeIn(leg[0][0]), m.FadeIn(leg[0][1][0])),
            m.FadeIn(leg[0][1][1], shift=0.1 * m.UP),
        )
        self.say(
            "[Without it, each update moved the policy about thirty-five times further.]",
            m.AnimationGroup(m.Create(kl_n, run_time=2.2), m.FadeIn(leg[1]), m.Succession(m.Wait(1.2), m.FadeIn(times, scale=0.8))),
        )
        self.say(
            "[Reuse each batch harder, with twenty passes and a higher learning rate.] "
            "[With the clip, it solved in fifty-eight thousand steps]; [without it, two hundred thirty-four thousand.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(plot, kl_c, kl_n, leg, times, meas)), m.FadeIn(b_title), m.Create(b_axis),
                             m.FadeIn(b_meas),
                             m.FadeIn(bars[0][2:]), m.GrowFromEdge(bars[0][0], m.LEFT), m.FadeIn(bars[0][1]),
                             m.FadeIn(bars[1][2:]), m.GrowFromEdge(bars[1][0], m.LEFT), m.FadeIn(bars[1][1]),
                             m.Create(split)),
            m.AnimationGroup(m.FadeIn(bars[2][2:]), m.GrowFromEdge(bars[2][0], m.LEFT), m.FadeIn(bars[2][1])),
            m.AnimationGroup(m.FadeIn(bars[3][2:]), m.GrowFromEdge(bars[3][0], m.LEFT), m.FadeIn(bars[3][1])),
        )
        self.say(
            "[To be honest, this task is easy and five seeds is few]: [the solid finding is the size of each step.]",
            m.FadeIn(honest, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(bars[1][3], color=RED, buff=0.06), m.Circumscribe(bars[3][3], color=RED, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · the default

    def info_card(self, cx, name, title, lines, color):
        box = card(4.1, 4.6, color, CARD, radius=0.2, stroke=2.5).move_to([cx, -0.55, 0])
        top, left = box.get_top()[1], box.get_left()[0] + 0.3
        ic = icon(name, 0.55).move_to([left + 0.28, top - 0.48, 0])
        head = text(title, 24, INK, weight=m.BOLD).next_to(ic, m.RIGHT, buff=0.2)
        body = m.VGroup(*lines).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        assert body.width < box.width - 0.5, title
        left_at(body, left, 0).align_to([0, top - 1.0, 0], m.UP)
        return m.VGroup(box, ic, head), body

    def the_default(self):
        self.next_section("9 · Why PPO became the default", notes=(
            "OpenAI blog 'Proximal Policy Optimization', 20 Jul 2017 (Wayback): 'PPO has become the default reinforcement "
            "learning algorithm at OpenAI because of its ease of use and good performance.' The PPO paper: TRPO 'is "
            "relatively complicated, and is not compatible with architectures that include noise (such as dropout) or "
            "parameter sharing'. OpenAI Five (Berner et al. 2019, arXiv 1912.06680): 'On April 13th, 2019, OpenAI Five "
            "became the first AI system to defeat the world champions at an esports game', beating OG 2–0; Table 2: PPO "
            f"clipping 0.2, GAE λ 0.95; batch 120 × 16 × 1,536 = {FIVE_BATCH:,} timesteps at most ('batch sizes of 1 to 3 "
            "million timesteps'); one training run from 30 June 2018 to 22 April 2019 (ten months; 180 days of "
            "training because of restarts); 770 ± 50 PFlop/s·days (Appendix A: paper over blog); footnote 6: 'PPO was "
            "the first to show initial learning progress.' RLHF with PPO: Ziegler et al. 2019, Stiennon et al. 2020, "
            "Ouyang et al. 2022 (InstructGPT) → Episode 21. Its cost: a value network (critic) as large as the policy, "
            "which GRPO removes (Episode 23)."))
        c1, b1 = self.info_card(-4.5, CODE, "Simple", [
            text("plain first-order steps", 22, INK),
            text("a few lines of code", 22, INK),
            text("fine with shared networks", 22, INK),
            text("“the default reinforcement", 20, MUTED),
            text("learning algorithm at OpenAI”", 20, MUTED),
            cite("OpenAI blog, July 2017"),
        ], ACTION)
        c2, b2 = self.info_card(0.0, PAD, "OpenAI Five", [
            text("Dota 2: beat the world", 22, INK),
            text("champions, OG", 22, INK),
            code("2–0 · 13 April 2019", 22, GREEN, weight=m.BOLD),
            code(f"clip {EPS:g} · GAE λ 0.95", 20, VIOLET),
            text("batches of up to", 22, INK),
            code(f"{FIVE_BATCH:,} steps", 22, INK, weight=m.BOLD),
            text("one run, ten months", 22, INK),
            cite("Berner et al., 2019"),
        ], STATE)
        c3, b3 = self.info_card(4.5, LLM, "Language models", [
            text("from human preferences:", 22, INK),
            cite("Ziegler 2019 · Stiennon 2020"),
            text("InstructGPT 2022: all PPO", 20, INK),
            pill("episode 21", PINK, PINK_FILL, 20),
            text("its price: a critic", 22, INK),
            text("as big as the policy", 22, INK),
            pill("episode 23 · GRPO", PINK, PINK_FILL, 20),
        ], PINK)
        first = cite("“PPO was the first to show initial learning progress” · Berner et al., 2019").move_to([0, -3.35, 0])

        self.say(
            "[So why did PPO become the default?] [Plain gradient steps and a few lines of code], "
            "[and it works with big shared networks.]",
            m.AnimationGroup(self.new_header("09 · the default", "Why everyone used it"), m.FadeIn(c1, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(b1[0], shift=0.1 * m.UP), m.FadeIn(b1[1], shift=0.1 * m.UP)),
            m.FadeIn(b1[2], shift=0.1 * m.UP),
        )
        self.say(
            "[By July twenty seventeen, OpenAI called it their default reinforcement learning algorithm.]",
            m.FadeIn(b1[3:], shift=0.1 * m.UP),
        )
        self.say(
            "[In twenty nineteen, OpenAI Five, trained with this same clip], [beat the world champions at Dota two], "
            "[two games to nil.]",
            m.AnimationGroup(m.FadeIn(c2, shift=0.1 * m.UP), m.FadeIn(b2[3], shift=0.1 * m.UP)),
            m.FadeIn(b2[0:2], shift=0.1 * m.UP),
            m.FadeIn(b2[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Its batches held up to about three million steps], [in one training run that spanned ten months.]",
            m.FadeIn(b2[4:6], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(b2[6:], shift=0.1 * m.UP), m.FadeIn(first)),
        )
        self.say(
            "[Then, when language models learned from human preferences], [this was the algorithm doing the learning]: "
            "[episode twenty-one.]",
            m.AnimationGroup(m.FadeIn(c3, shift=0.1 * m.UP), m.FadeIn(b3[0], shift=0.1 * m.UP)),
            m.FadeIn(b3[1:3], shift=0.1 * m.UP),
            m.FadeIn(b3[3], shift=0.1 * m.UP),
        )
        self.say(
            "[Its price is a critic as big as the policy], [which episode twenty-three's GRPO removes.]",
            m.FadeIn(b3[4:6], shift=0.1 * m.UP),
            m.FadeIn(b3[6], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 10 · recap

    def recap(self):
        self.next_section("10 · Recap", notes=(
            f"Reuse (each batch of {BATCH} samples for {EPOCHS} passes, {GRAD_STEPS} gradient steps), ratio (π_new/π_old "
            f"re-weights old samples), clip (to [{1 - EPS:g}, {1 + EPS:g}], keep the smaller term). On this Mac: 5/5 solved "
            f"CartPole in {PPO_K[0]}k ± {PPO_K[1]}k steps, every seed stayed solved (final {FINAL[0]} ± {FINAL[1]}); the clip "
            f"touched {CLIP_PCT}% of samples per update. Episode 19: continuous control (Gaussian policies, SAC)."))
        halo = glow(VIOLET, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Reuse", 70, STATE), title_text("Ratio", 70, ACTION), title_text("Clip", 70, VIOLET))
        words.arrange(m.RIGHT, buff=1.5).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text(f"{EPOCHS} passes per batch", 24, MUTED), text("new ÷ old", 24, MUTED),
                        text(f"to [{1 - EPS:g}, {1 + EPS:g}]", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), words[0].get_bottom()[1] - 0.5, 0])
        on_baseline(subs)
        assert all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        proof = colored([("CartPole, this Mac:", MUTED), ("5/5 solved, stayed solved", GREEN), ("·", MUTED),
                         (f"clip touched {CLIP_PCT}%", VIOLET)], 24).move_to([0, -1.25, 0])
        bot = rover(0.6).move_to([-2.9, -2.6, 0])
        nxt = pill("episode 19 · actions that are numbers", ACTION, CORAL_FILL, 22).next_to(bot, m.RIGHT, buff=0.45)

        self.say(
            "[To sum up]: [reuse each batch], [re-weight it with the ratio], [and clip the ratio so no step goes too far.]",
            m.AnimationGroup(self.new_header("10 · recap", "Reuse, ratio, clip"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[On this Mac, the clip touched under four percent of samples], [and all five runs stayed solved.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.1 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()
