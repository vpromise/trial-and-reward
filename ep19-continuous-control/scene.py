"""Trial and Reward · Episode 19 · Continuous Control (Module 5 · Learning the Policy)

Robots need actions that are numbers, not choices. Until now Rover's actions were choices (push left or right, a
lever); a robot arm needs a torque per joint. The running example is the pendulum swing-up (Pendulum-v1, the exact
numpy port in rl_reference.py): a torque in [−2, 2] N·m at the pivot; the motor is weak (it adds at most 6 rad/s²
against gravity's 15·sin θ, so a full push lifts the pendulum only 48.8°, and it can hold it only within 23.6° of
upright), so the policy must pump; reward r = −(θ² + 0.1·θ̇² + 0.001·u²). A Q-network's max over infinitely many
torques is the problem; two ways out. (1) A Gaussian policy: the network outputs a mean and a spread, Rover samples a
torque, the spread is built-in exploration; the policy gradient of Episode 16 with the Gaussian score (notes §19
worked example C: μ 0.5 → 0.66, σ 1 → 0.93), and a seeded toy run of many updates in which the bell slides to the
good torques and narrows (Plot.area). (2) Deterministic: the actor outputs one torque and climbs the critic's slope
(DPG, Silver et al. 2014); DDPG (Lillicrap et al. 2015) adds Episode 13's replay and slow targets plus exploration
noise; TD3 (Fujimoto et al. 2018) uses two critics and the smaller (Episode 14). Soft actor-critic (Haarnoja et al.
2018): a stochastic actor learned off-policy, objective reward + temperature × entropy; the temperature auto-tuned
(Haarnoja et al. 2018b), as in our run. Measured on this Mac (5 seeds): the SAC actor replayed (trajectories.json,
torque = 2·tanh(mean), up within 1.4 s, return −118.2 for this start); SAC −147.4 ± 1.6 after 20k steps (56 s; better
than −200 after 4.4k steps) vs PPO −222 ± 22 after 502k steps (29 s). Sim-to-real: domain randomization (Tobin et
al. 2017), OpenAI's Rubik's-cube hand (Akkaya et al. 2019) and the cost of randomization (Dactyl 2018). Recap:
spread, slope, randomize. Episode 20: the model is the policy (Module 6).

Every number on screen is computed below from rl_reference.py and asserted, loaded with measured(...), or cited.

Sources (facts dated October 2026):
- rl_reference.py: PENDULUM, pendulum_reset / pendulum_step / pendulum_obs (gymnasium 1.4.0 Pendulum-v1, verified
  step for step), angle_normalize, load_policy("sac_pendulum") (SAC actor, seed 10, final weights after 20k env steps;
  mean head only; torque = 2·tanh(mean)).
- research/experiments/deep_rl.md "Episode 19 — Pendulum swing-up" + deep_rl.json (measured("deep_rl", ...)):
  headline.sac / headline.ppo_pendulum, methods.*.hp, curves.sac / curves.ppo_pendulum (evaluation = deterministic
  policy on 10 fixed start states, reset seeds 200000–200009; mean, band = min–max over seeds 10–14).
- research/experiments/trajectories.json policies.sac_pendulum (reset seed 2026, deterministic, 200 steps, return
  −118.194; the rewards are recomputed here with pendulum_step and match).
- research/notes.md §19: Pendulum-v1 table (reward, torque range, 200 steps), worked examples A (15·sin θ vs 6,
  23.6°) and C (one Gaussian policy-gradient step), DPG (Silver, Lever, Heess, Degris, Wierstra & Riedmiller, ICML
  2014), DDPG (Lillicrap et al., arXiv 2015 / ICLR 2016: replay, soft target updates, Ornstein–Uhlenbeck noise), TD3
  (Fujimoto, van Hoof & Meger, ICML 2018: two critics, the minimum; "far preferable to overestimation bias"), SAC
  (Haarnoja, Zhou, Abbeel & Levine, ICML 2018, Eq. 1; abstract: "to succeed at the task while acting as randomly as
  possible"; automatic temperature with target entropy −dim(A) in Haarnoja et al. 2018b), Tobin et al. IROS 2017
  ("With enough variability in the simulator, the real world may appear to the model as just another variation."),
  OpenAI / Akkaya et al. 2019 (ADR; "roughly 13 thousand years" of experience; 60% of scrambles needing 15 face
  rotations, 20% of those needing 26; the solution sequence from Kociemba's solver), OpenAI / Andrychowicz et al.
  2018 (Dactyl: about 3 years of simulated experience without randomization vs about 100 years fully randomized).
  Appendix C row 19 checked (reward formula, u ∈ [−2, 2], 200 steps, Kociemba).

Illustrative (said "for example", "say", or labelled on screen): the arm and its joint torques (a smooth reach, torque
readouts from a made-up inertia + gravity law); the 14 torque samples (default_rng(29) from N(0.5, 1)); the toy
Gaussian run of section 2 (reward −(u − 1.2)² for one state, 32 samples per update, step 0.05, default_rng(0) —
computed here, but the reward is invented); the critic's curves of section 3 and the hops up the slope; the two
entropy bells (σ = 1 and 0.3; their entropies are computed); the temperature slider's motion; the simulator and real
arms (lengths jittered with default_rng(19)); the cube's sticker colours.

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

PREFIX = "ep19"
DEEP = "deep_rl"
SEEDS = 5
SAC_C, PPO_C = ACTION, MUTED          # the two measured learners: SAC (our policy, its torque) and PPO for contrast

# ---------------------------------------------------------------------------- Pendulum-v1 (rl_reference port)

P = PENDULUM
assert (P["max_torque"], P["max_steps"], P["dt"], P["max_speed"]) == (2.0, 200, 0.05, 8.0)
G_ACC = 3 * P["g"] / (2 * P["l"])                         # gravity: 15·sin θ rad/s²
M_ACC = 3.0 / (P["m"] * P["l"] ** 2) * P["max_torque"]   # the motor: at most 6 rad/s²
assert (G_ACC, M_ACC) == (15.0, 6.0)
HOLD_DEG = math.degrees(math.asin(M_ACC / G_ACC))
assert f"{HOLD_DEG:.1f}" == "23.6"                        # notes §19 worked example A
assert P["max_steps"] * P["dt"] == 10.0
HANG = P["max_steps"] * -(math.pi ** 2)                   # hanging still with no torque, one episode
assert round(HANG) == -1974
_, R_TEST = pendulum_step(np.array([0.5, 1.0]), 1.0)
assert abs(R_TEST + (0.5 ** 2 + 0.1 * 1.0 ** 2 + 0.001 * 1.0 ** 2)) < 1e-12   # r = −(θ² + 0.1 θ̇² + 0.001 u²)


def run_pendulum(act, s0, steps):
    """Play Pendulum-v1 from state s0 with act(state) -> torque. Returns states (steps + 1), torques, rewards."""
    s = np.array(s0, dtype=float)
    states, torques, rewards = [s.copy()], [], []
    for _ in range(steps):
        u = act(s)
        s, r = pendulum_step(s, u)
        torques.append(float(np.ravel(u)[0]))
        rewards.append(r)
        states.append(s.copy())
    return np.array(states), np.array(torques), np.array(rewards)


# a full push from hanging still: it rises 48.8° and falls back
PUSH_S, _, _ = run_pendulum(lambda s: P["max_torque"], (math.pi, 0.0), 34)
PUSH_DEG = np.degrees(np.abs(PUSH_S[:, 0] - math.pi))
PUSH_PEAK, PUSH_PEAK_STEP = float(PUSH_DEG.max()), int(PUSH_DEG.argmax())
assert f"{PUSH_PEAK:.1f}" == "48.8" and PUSH_PEAK < 50 and PUSH_PEAK_STEP == 17 and PUSH_DEG[-1] < 1   # "less than fifty"

# the trained SAC actor (exported mean head): torque = 2·tanh(mean)
NET = load_policy("sac_pendulum")
assert NET.output == "mean" and NET.meta["squash"] == "tanh" and NET.meta["action_scale"] == P["max_torque"]
assert NET.meta["layers"] == [3, 256, 256, 1]


def sac_act(s):
    with np.errstate(all="ignore"):
        return np.asarray(NET.act(pendulum_obs(s)), dtype=np.float32)


def sac_mean(s):
    with np.errstate(all="ignore"):
        return float(NET(pendulum_obs(s))[0])


# the same network started hanging straight down (computed here): it pumps, then swings up
PUMP_S, PUMP_U, PUMP_R = run_pendulum(sac_act, (math.pi, 0.0), P["max_steps"])
PUMP_TH = np.unwrap(PUMP_S[:, 0])
PUMP_N = angle_normalize(PUMP_TH)
PUMP_UP = int(np.argmax(np.abs(PUMP_N) < 0.1))
assert PUMP_UP == 80 and np.all(np.abs(PUMP_N[PUMP_UP:]) < 0.1)
PUMP_DEV = np.degrees(angle_normalize(PUMP_TH - math.pi))
PUMP_EXT = [(i, round(float(PUMP_DEV[i]), 1)) for i in range(1, PUMP_UP)
            if (PUMP_DEV[i] - PUMP_DEV[i - 1]) * (PUMP_DEV[i + 1] - PUMP_DEV[i]) < 0]
assert PUMP_EXT == [(15, 46.0), (31, -92.6)]                 # "rock one way, then the other", then over the top
PUMP_SHOW = 92

# the recorded episode (trajectories.json, replayed, not simulated)
TRAJ = measured("trajectories", "policies", "sac_pendulum")
TSRC = measured("trajectories", "sources", "sac_pendulum")
assert TRAJ["env"] == "Pendulum-v1" and TRAJ["length"] == P["max_steps"] and TRAJ["truncated"] and not TRAJ["terminated"]
assert TRAJ["reset_seed"] == 2026 and "2 tanh(mean)" in TRAJ["action_mode"]
assert TSRC["run"] == "sac_pendulum_s10" and TSRC["env_steps"] == 20_000 and TSRC["weights"] == "final"
T_S, T_U = np.array(TRAJ["states"]), np.array(TRAJ["actions"])
assert len(T_S) == 201 and len(T_U) == 200
assert np.abs(pendulum_reset(np.random.default_rng(2026)) - T_S[0]).max() < 1e-4
T_R = np.array([pendulum_step(s, np.array([u], dtype=np.float32))[1] for s, u in zip(T_S[:-1], T_U)])
T_RET = np.concatenate([[0.0], np.cumsum(T_R)])
assert abs(T_RET[-1] - TRAJ["return"]) < 1e-3 and f"{TRAJ['return']:.1f}" == "-118.2"   # "minus one hundred eighteen"
T_MEAN = np.array([sac_mean(s) for s in T_S[:-1]])
assert np.abs(P["max_torque"] * np.tanh(T_MEAN) - T_U).max() < 5e-4                    # torque = 2·tanh(mean)
T_N = angle_normalize(T_S[:, 0])
T_UP = int(np.argmax(np.abs(T_N) < 0.1))
assert T_UP == 28 and np.all(np.abs(T_N[T_UP:]) < 0.1) and T_UP * P["dt"] < 1.5          # "within a second and a half"
assert np.all(T_U[:13] < -1.95)                                                          # "one hard swing"
T_START = math.degrees(abs(T_N[0]))
assert round(T_START) == 116                                                             # this start: partway up
T_TH = np.unwrap(T_S[:, 0])

# ---------------------------------------------------------------------------- measured on this Mac

H_SAC, H_PPO = measured(DEEP, "headline", "sac"), measured(DEEP, "headline", "ppo_pendulum")
HP_SAC, HP_PPO = measured(DEEP, "methods", "sac", "hp"), measured(DEEP, "methods", "ppo_pendulum", "hp")
assert (H_SAC["final_return_mean"], H_SAC["final_return_std"]) == (-147.4, 1.6)
assert H_SAC["final_return_max"] < -145 and H_SAC["final_return_min"] > -150                # every seed better than −150
SAC_STEPS = H_SAC["total_steps"]
assert SAC_STEPS == 20_000 and H_SAC["n_reaching_minus200"] == SEEDS
SAC_200 = H_SAC["steps_to_minus200_mean"]
assert SAC_200 == 4400 and SAC_200 / P["max_steps"] == 22                                # "4,400 steps", 22 episodes
SAC_WALL, PPO_WALL = H_SAC["wall_s_mean"], H_PPO["wall_s_mean"]
assert (round(SAC_WALL), round(PPO_WALL)) == (56, 29)
PPO_STEPS = H_PPO["total_steps"]
assert round(PPO_STEPS / 1000) == 502 and abs(PPO_STEPS - 500_000) < 2_000                # "about half a million"
PPO_RET, PPO_SD = H_PPO["final_return_mean"], H_PPO["final_return_std"]
assert (round(PPO_RET), round(PPO_SD)) == (-222, 22) and H_PPO["n_reaching_minus200"] == SEEDS
SAC_MS, PPO_MS = 1000 * SAC_WALL / SAC_STEPS, 1000 * PPO_WALL / PPO_STEPS
PER_STEP = SAC_MS / PPO_MS
assert (f"{SAC_MS:.1f}", f"{PPO_MS:.2f}", round(PER_STEP)) == ("2.8", "0.06", 49)
assert HP_SAC["action_scale"] == 2.0 and HP_SAC["autotune_alpha"] and HP_SAC["target_entropy"] == -1.0
assert HP_SAC["learning_starts"] == 1000 and HP_SAC["hidden"] == [256, 256] and HP_SAC["total_steps"] == SAC_STEPS
assert HP_PPO["clip"] and HP_PPO["clip_eps"] == 0.2 and HP_PPO["gamma"] == 0.9
C_SAC, C_PPO = measured(DEEP, "curves", "sac"), measured(DEEP, "curves", "ppo_pendulum")
assert C_SAC["x_steps"][0] == 0 and C_SAC["x_steps"][-1] == SAC_STEPS and C_PPO["x_steps"][-1] == PPO_STEPS
assert C_SAC["eval_return"]["mean"][-1] == H_SAC["final_return_mean"]
assert abs(C_PPO["eval_return"]["mean"][-1] - PPO_RET) < 0.05
assert "10 fixed start states" in C_SAC["note"] and "2 tanh(mean)" in C_SAC["note"]
SAC_X = np.array(C_SAC["x_steps"][1:], dtype=float)          # log axis: from the first 1,000 steps
SAC_Y = {k: C_SAC["eval_return"][k][1:] for k in ("mean", "min", "max")}
PPO_X = np.array(C_PPO["x_steps"], dtype=float)
PPO_Y = {k: C_PPO["eval_return"][k] for k in ("mean", "min", "max")}
assert min(min(SAC_Y["min"]), min(PPO_Y["min"])) > -1600 and max(max(SAC_Y["max"]), max(PPO_Y["max"])) < 0

# ---------------------------------------------------------------------------- the Gaussian policy (notes §19 C)

MU0, SD0, U_GOOD, ADV, LR = 0.5, 1.0, 1.3, 2.0, 0.1
G_MU = (U_GOOD - MU0) / SD0 ** 2                      # ∂ln π/∂μ
G_LS = (U_GOOD - MU0) ** 2 / SD0 ** 2 - 1             # ∂ln π/∂log σ
MU1 = MU0 + LR * ADV * G_MU
SD1 = SD0 * math.exp(LR * ADV * G_LS)
assert (round(G_MU, 2), round(G_LS, 2), round(MU1, 2), round(SD1, 3)) == (0.8, -0.36, 0.66, 0.931)
SAMPLES = np.random.default_rng(29).normal(MU0, SD0, 14)        # illustrative draws from the starting bell
assert np.sum(np.abs(SAMPLES - MU0) > 1.5) == 2 and np.sum(np.abs(SAMPLES - MU0) < 1) >= 9 and np.abs(SAMPLES).max() < 2.6

BEST_U, TOY_N, TOY_LR, TOY_UPDATES = 1.2, 32, 0.05, 60


def toy_run(seed=0):
    """Gaussian REINFORCE on one made-up state whose reward is −(u − 1.2)², from (μ, σ) = (0.66, 0.93); baseline =
    the batch mean (Episode 17). Returns the (μ, σ) path, start included."""
    rng = np.random.default_rng(seed)
    mu, ls = MU1, math.log(SD1)
    path = [(mu, SD1)]
    for _ in range(TOY_UPDATES):
        sd = math.exp(ls)
        u = rng.normal(mu, sd, TOY_N)
        a = -(u - BEST_U) ** 2
        a = a - a.mean()
        mu, ls = mu + TOY_LR * np.mean(a * (u - mu) / sd ** 2), ls + TOY_LR * np.mean(a * ((u - mu) ** 2 / sd ** 2 - 1))
        path.append((mu, math.exp(ls)))
    return np.array(path)


TOY = toy_run()
assert abs(TOY[-1, 0] - BEST_U) < 0.05 and TOY[-1, 1] < 0.3 and np.all(np.diff(TOY[::10, 1]) < 0)


def density(x, mu, sd):
    return np.exp(-((np.asarray(x) - mu) ** 2) / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi))


def entropy(sd):
    """Differential entropy of a 1-D Gaussian, in nats."""
    return 0.5 * math.log(2 * math.pi * math.e * sd ** 2)


SD_WIDE, SD_NARROW = 1.0, 0.3
H_WIDE, H_NARROW = entropy(SD_WIDE), entropy(SD_NARROW)
assert (f"{H_WIDE:.2f}", f"{H_NARROW:.2f}") == ("1.42", "0.21")

# ---------------------------------------------------------------------------- the critic's slope (illustration)


def q_guess(u):
    return 0.1 + 1.7 * np.exp(-((u - 1.1) ** 2) / 2.5)


def dq_guess(u):
    return -1.7 * (u - 1.1) / 1.25 * np.exp(-((u - 1.1) ** 2) / 2.5)


def q_guess2(u):
    return q_guess(u) + 0.08 + 0.22 * np.sin(2.4 * u)


HOPS = [-1.3]
for _ in range(6):
    HOPS.append(float(HOPS[-1] + 1.0 * dq_guess(HOPS[-1])))      # u ← u + η·dQ/du
assert abs(HOPS[-1] - 1.1) < 0.05 and q_guess(HOPS[-1]) > q_guess(HOPS[0]) + 1.4

# ---------------------------------------------------------------------------- cited numbers (notes §19)

RUBIK_YEARS, RUBIK_15, RUBIK_26 = 13_000, 60, 20
DACTYL_PLAIN, DACTYL_RAND = 3, 100

DICE = recolored("dice", STATE, TEAL_FILL, prefix=PREFIX)

# ---------------------------------------------------------------------------- helpers


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


def hbar(x0, y, length, height, color):
    return m.Rectangle(width=max(length, 0.03), height=height).set_stroke(color, 2).set_fill(FILLS.get(color, CARD), 1) \
        .move_to([x0 + max(length, 0.03) / 2, y, 0])


class Gauge(m.VGroup):
    """A torque gauge, −2 … +2 N·m: a CORAL bar from 0 to the torque, a knob at its end. `.set(u)` moves it."""

    def __init__(self, center, width=3.4, label=True):
        super().__init__()
        c = np.array(center, dtype=float)
        self.half = width / 2
        self.track = m.Line(c - [self.half, 0, 0], c + [self.half, 0, 0]).set_stroke(EDGE, 6)
        marks = m.VGroup()
        for u in (-2, -1, 0, 1, 2):
            p = c + [u / 2 * self.half, 0, 0]
            marks.add(m.Line(p + [0, -0.12, 0], p + [0, 0.12, 0]).set_stroke(FAINT, 2))
            if u in (-2, 0, 2):
                marks.add(code(minus(f"{u:+d}") if u else "0", 20, MUTED).move_to(p + [0, -0.36, 0]))
        self.marks = marks
        self.bar = m.Line(c, c + [0.01, 0, 0]).set_stroke(ACTION, 9)
        self.knob = m.Dot(c, radius=0.11, color=ACTION).set_z_index(3)
        self.add(self.track, marks, self.bar, self.knob)
        if label:
            self.label = right_at(code("torque", 20, ACTION), c[0] - self.half - 0.3, c[1])
            self.add(self.label)
        self.set(0.0)

    def at(self, u):
        return self.track.get_center() + np.array([u / 2 * self.half, 0, 0])

    def set(self, u):
        u = float(np.clip(u, -2, 2))
        end = self.at(u if abs(u) > 0.01 else 0.01)
        self.bar.put_start_and_end_on(self.at(0), end)
        self.knob.move_to(end)
        return self


def swing(view, thetas, seconds, on_step=None, rate=m.linear):
    """Replay angles on a PendulumView; on_step(n) is called with the step shown (for gauges and counters)."""
    th = np.asarray(thetas, dtype=float)
    last = len(th) - 1

    def step(_, alpha):
        i = alpha * last
        j = int(np.floor(i))
        k = min(j + 1, last)
        f = i - j
        view.set_angle((1 - f) * th[j] + f * th[k])
        if on_step:
            on_step(min(int(round(i)), last))

    return m.UpdateFromAlphaFunc(view, step, run_time=seconds, rate_func=rate)


class Arm(m.VGroup):
    """A two-joint robot arm on a base plate (CORAL links, INK fingers). `.set_q(q1, q2)` poses it (radians, q1 from
    the horizontal, q2 relative to link 1); `.solve(target)` the elbow-up angles that put the fingertips on a point;
    `.move(q, seconds)` a smooth reach (it reads the current pose when it starts)."""

    def __init__(self, base, l1, l2, q=(1.75, -2.2), color=ACTION, opacity=1.0, stroke=9):
        super().__init__()
        self.l1, self.l2 = l1, l2
        b = np.array(base, dtype=float)
        self.plate = m.RoundedRectangle(corner_radius=0.05, width=0.8, height=0.18, stroke_width=0,
                                        fill_color=MUTED, fill_opacity=opacity).move_to(b + [0, -0.11, 0])
        self.link1 = m.Line(b, b + [l1, 0, 0]).set_stroke(color, stroke, opacity)
        self.link2 = m.Line(b, b + [l2, 0, 0]).set_stroke(color, stroke * 0.8, opacity)
        self.f1 = m.Line(b, b + [0.2, 0, 0]).set_stroke(INK, 4, opacity)
        self.f2 = m.Line(b, b + [0.2, 0, 0]).set_stroke(INK, 4, opacity)
        self.j0 = m.Circle(radius=0.12).set_stroke(color, 3, opacity).set_fill(CORAL_FILL, opacity)
        self.j1 = m.Circle(radius=0.11).set_stroke(color, 3, opacity).set_fill(CORAL_FILL, opacity)
        self.add(self.plate, self.link1, self.link2, self.f1, self.f2, self.j0, self.j1)
        self.set_q(*q)

    def base(self):
        return self.plate.get_center() + np.array([0, 0.11, 0])

    def set_q(self, q1, q2):
        b = self.base()
        p1 = b + self.l1 * np.array([math.cos(q1), math.sin(q1), 0])
        a = q1 + q2
        d = np.array([math.cos(a), math.sin(a), 0])
        p2 = p1 + self.l2 * d
        self.link1.put_start_and_end_on(b, p1)
        self.link2.put_start_and_end_on(p1, p2)
        self.j0.move_to(b)
        self.j1.move_to(p1)
        for f, s in ((self.f1, 1), (self.f2, -1)):
            r = np.array([math.cos(a + s * 0.6), math.sin(a + s * 0.6), 0])
            f.put_start_and_end_on(p2 - 0.04 * d, p2 + 0.26 * r)
        self.q = (q1, q2)
        return self

    def tip(self):
        return self.link2.get_end()

    def solve(self, target):
        rel = np.array(target, dtype=float) - self.base()
        x, y = rel[0], rel[1]
        c2 = float(np.clip((x * x + y * y - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1, 1))
        q2 = -math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(self.l2 * math.sin(q2), self.l1 + self.l2 * math.cos(q2))
        return q1, q2

    def move(self, q_to, seconds, on_step=None):
        start = {}

        def step(_, alpha):
            if "q" not in start:
                start["q"] = self.q
            s = 10 * alpha ** 3 - 15 * alpha ** 4 + 6 * alpha ** 5     # minimum-jerk: starts and stops at rest
            self.set_q(*(a + (b - a) * s for a, b in zip(start["q"], q_to)))
            if on_step:
                on_step(alpha)

        return m.UpdateFromAlphaFunc(self, step, run_time=seconds, rate_func=m.linear)


BELL_XS = np.linspace(-3.0, 3.0, 181)


def bell(plot, mu, sd):
    """The Gaussian policy on a plot: VIOLET spread (the area under the curve), CORAL curve and mean line, and a VIOLET
    bracket one σ either side of the mean, at the height where the curve has fallen to e^(−1/2) of its peak."""
    ys = density(BELL_XS, mu, sd)
    area = plot.area(BELL_XS, ys, EXPLORE, 0.3)
    curve = plot.line(BELL_XS, ys, ACTION, 4)
    peak = float(density(mu, mu, sd))
    mean_line = m.Line(plot.coords(mu, 0), plot.coords(mu, peak)).set_stroke(ACTION, 3)
    h = peak * math.exp(-0.5)
    a, b = plot.coords(mu - sd, h), plot.coords(mu + sd, h)
    bracket = m.VGroup(m.Line(a, b), m.Line(a + [0, -0.1, 0], a + [0, 0.1, 0]), m.Line(b + [0, -0.1, 0], b + [0, 0.1, 0]))
    bracket.set_stroke(EXPLORE, 4)
    return m.VGroup(area, curve, mean_line, bracket)


def readout(mu, sd):
    return colored([("mean", ACTION), (f"{mu:.2f}", ACTION), ("·", MUTED), ("spread", EXPLORE), (f"{sd:.2f}", EXPLORE)],
                   22, weight=m.BOLD)


def morph(plot, group, path, seconds, read=None, read_at=None, counter=None, count_at=None, rate=m.smooth):
    """Animate a bell() group along a path of (μ, σ); optionally keep a readout and an update counter in step."""
    path = np.asarray(path, dtype=float)
    last = len(path) - 1

    def step(_, alpha):
        i = alpha * last
        j = int(np.floor(i))
        k = min(j + 1, last)
        mu, sd = (1 - (i - j)) * path[j] + (i - j) * path[k]
        group.become(bell(plot, mu, sd))
        if read is not None:
            read.become(left_at(readout(mu, sd), *read_at))
        if counter is not None:
            counter.become(left_at(code(f"update {int(round(i)):>2} of {last}", 20, INK), *count_at))

    return m.UpdateFromAlphaFunc(group, step, run_time=seconds, rate_func=rate)


def mini_bell(cx, cy, sd, width=2.6, height=1.3, peak_max=1.4):
    """A small bell over torque without axes labels (for the entropy comparison)."""
    xs = np.linspace(-2.6, 2.6, 121)
    ys = density(xs, 0.0, sd)
    sx, sy = width / 5.2, height / peak_max
    pts = [np.array([cx + x * sx, cy + y * sy, 0]) for x, y in zip(xs, ys)]
    base = m.Line([cx - width / 2, cy, 0], [cx + width / 2, cy, 0]).set_stroke(FAINT, 2.5)
    area = m.Polygon(*pts, [cx + width / 2, cy, 0], [cx - width / 2, cy, 0], stroke_width=0, fill_color=EXPLORE,
                     fill_opacity=0.3)
    curve = m.VMobject().set_points_as_corners(pts).set_stroke(ACTION, 4)
    return m.VGroup(base, area, curve)


def cube(cx, cy, s=0.62, seed=7):
    """A Rubik's cube in a simple oblique view: a 3×3 front face, top and right faces as parallelograms."""
    rng = np.random.default_rng(seed)
    colors = [GOLD, RED, BLUE, GREEN, INK, CORAL]
    group = m.VGroup()
    dx = np.array([0.5 * s, 0.32 * s, 0])            # one step "back" into the picture
    o = np.array([cx - 1.5 * s - 0.25 * s, cy - 1.5 * s - 0.16 * s, 0])
    for r in range(3):
        for c in range(3):
            p = o + np.array([c * s, r * s, 0])
            group.add(m.Polygon(p, p + [s, 0, 0], p + [s, s, 0], p + [0, s, 0]))
    top0 = o + np.array([0, 3 * s, 0])
    for r in range(3):
        for c in range(3):
            p = top0 + np.array([c * s, 0, 0]) + r * dx
            group.add(m.Polygon(p, p + [s, 0, 0], p + [s, 0, 0] + dx, p + dx))
    right0 = o + np.array([3 * s, 0, 0])
    for r in range(3):
        for c in range(3):
            p = right0 + np.array([0, r * s, 0]) + c * dx
            group.add(m.Polygon(p, p + dx, p + dx + [0, s, 0], p + [0, s, 0]))
    for k, sticker in enumerate(group):
        col = colors[int(rng.integers(len(colors)))]
        shade = 1.0 if k < 9 else (0.8 if k < 18 else 0.62)
        sticker.set_fill(mix(BG, col, shade), 1).set_stroke(BG, 3)
    return group


# ---------------------------------------------------------------------------- the episode


class Ep19ContinuousControl(Episode):
    number = 19

    def lesson(self):
        self.numbers()
        self.gaussian()
        self.deterministic()
        self.soft()
        self.measured_runs()
        self.sim_to_real()
        self.recap()

    # ------------------------------------------------------------------ 1 · actions that are numbers

    def numbers(self):
        self.next_section("1 · Choices, or numbers?", notes=(
            "Until now actions were choices: CartPole's push left / right (Episodes 12–18), Rover's five machines "
            "(Episodes 2–4). A robot arm needs a torque per joint (the arm and its torque readouts are an illustration). "
            "Pendulum-v1 (notes §19 table; rl_reference PENDULUM): torque u ∈ [−2, 2] N·m at the pivot, 200 steps of "
            f"0.05 s; reward r = −(θ² + 0.1·θ̇² + 0.001·u²), θ normalised, 0 = upright; hanging still scores {HANG:,.1f} per "
            f"episode. Gravity's angular acceleration is {G_ACC:g}·sin θ rad/s², the motor adds at most {M_ACC:g} (notes "
            f"§19 worked example A): it can hold the pendulum only within {HOLD_DEG:.1f}° of upright; a full push from "
            f"hanging still (simulated here with pendulum_step) lifts it {PUSH_PEAK:.1f}° at step {PUSH_PEAK_STEP} and back. "
            "The pumping pendulum is the trained SAC actor (section 5) started hanging straight down, computed here with "
            f"load_policy + pendulum_step: extremes {PUMP_EXT[0][1]}° and {PUMP_EXT[1][1]}° from the bottom, upright "
            f"(|θ| < 0.1) from step {PUMP_UP} ({PUMP_UP * P['dt']:.1f} s); the first {PUMP_SHOW} steps are replayed."))
        # the two kinds of action
        lp = card(5.9, 4.4, EDGE, CARD).move_to([-3.4, -0.15, 0])
        rp = card(5.9, 4.4, ACTION, CARD).move_to([3.4, -0.15, 0])
        lt = left_at(text("choices", 26, INK, weight=m.BOLD), -6.0, 1.7)
        rt = left_at(text("numbers", 26, INK, weight=m.BOLD), 0.8, 1.7)
        bot = rover(0.5).move_to([-1.15, 1.55, 0])
        row1 = m.VGroup(chip("push left", ACTION, CORAL_FILL, width=2.2, size=24),
                        chip("push right", ACTION, CORAL_FILL, width=2.2, size=24)).arrange(m.RIGHT, buff=0.3)
        row1.move_to([-3.4, 0.55, 0])
        cap1 = cite("CartPole · episodes 12 to 18").move_to([-3.4, -0.15, 0])
        levers = m.VGroup(*[chip(str(k), REWARD if k == 4 else EDGE, GOLD_FILL if k == 4 else CARD, width=0.66,
                                 height=0.62, size=24) for k in range(1, 6)]).arrange(m.RIGHT, buff=0.2)
        levers.move_to([-3.4, -1.0, 0])
        cap2 = cite("Rover's slot machines · episodes 2 to 4").move_to([-3.4, -1.7, 0])
        arm = Arm((1.95, -1.7, 0), 1.55, 1.25)
        target = arm.base() + np.array([2.05, 1.35, 0])
        goal = m.VGroup(m.Circle(radius=0.2).set_stroke(GOLD, 3), m.Dot(radius=0.08, color=GOLD)).move_to(target)
        q_goal = arm.solve(target)
        q_start = arm.q
        illus = right_at(cite("illustration"), 6.0, -1.95)

        def torque_text(k, tau):
            return right_at(colored([(f"joint {k}", MUTED), (minus(f"{tau:+.2f}"), ACTION), ("N·m", MUTED)], 22,
                                    weight=m.BOLD), 6.0, 1.68 - 0.45 * (k - 1))

        tq = [torque_text(1, 0.0), torque_text(2, 0.0)]

        def arm_torques(alpha):
            for k in (1, 2):
                dq = q_goal[k - 1] - q_start[k - 1]
                accel = 60 * alpha - 180 * alpha ** 2 + 120 * alpha ** 3      # the reach's acceleration profile
                hold = 0.9 * math.cos(arm.q[0]) if k == 1 else 0.4 * math.cos(arm.q[0] + arm.q[1])
                tau = 0.25 * dq * accel + hold
                tq[k - 1].become(torque_text(k, tau))

        self.say(
            "[Until now, Rover's actions were choices]: [push left or push right, pull lever one or lever four.]",
            m.AnimationGroup(self.new_header("01 · actions as numbers", "Choices, or numbers?"), m.FadeIn(lp),
                             m.FadeIn(lt), m.FadeIn(bot, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in row1], lag_ratio=0.3),
                             m.FadeIn(cap1),
                             m.Succession(m.Wait(0.8), m.AnimationGroup(
                                 m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in levers], lag_ratio=0.15),
                                 m.FadeIn(cap2)))),
        )
        self.say(
            "[A robot arm needs numbers instead]: [a torque for every joint, many times a second.]",
            m.AnimationGroup(m.FadeIn(rp), m.FadeIn(rt), m.FadeIn(arm), m.FadeIn(goal, scale=0.5), m.FadeIn(illus),
                             m.FadeIn(tq[0]), m.FadeIn(tq[1])),
            arm.move(q_goal, 2.6, on_step=arm_torques),
        )

        # the pendulum
        pc = np.array([-3.3, 0.55, 0])
        pend = PendulumView(center=pc, length=1.45)
        motor = m.Arc(radius=0.42, start_angle=math.radians(200), angle=math.radians(280), arc_center=pc)
        motor.set_stroke(ACTION, 4).add_tip(tip_length=0.18)
        motor_lab = right_at(text("motor", 22, ACTION, weight=m.BOLD), -5.05, 0.55)
        prov = rover(0.45).move_to([-6.1, 1.75, 0])
        gauge = Gauge((-3.3, -1.95, 0), width=3.4)
        top = pc + np.array([0, 1.45, 0])
        goal_dot = m.VGroup(m.Circle(radius=0.24).set_stroke(GOLD, 3), m.Dot(radius=0.07, color=GOLD)).move_to(top)
        goal_lab = left_at(text("goal: upright", 22, GOLD, weight=m.BOLD), -2.7, top[1] + 0.3)

        # right column: the weak motor, the reward, why a Q-network is stuck
        acc_t = left_at(text("angular acceleration, rad/s²", 20, MUTED), 0.5, 0.3)
        g_lab = right_at(code("gravity", 20, STATE), 1.95, -0.15)
        g_bar = hbar(2.1, -0.15, 0.2 * G_ACC, 0.26, STATE)
        g_val = left_at(code(f"up to {G_ACC:g}", 20, STATE, weight=m.BOLD), 2.1 + 0.2 * G_ACC + 0.15, -0.15)
        m_lab = right_at(code("motor", 20, ACTION), 1.95, -0.55)
        m_bar = hbar(2.1, -0.55, 0.2 * M_ACC, 0.26, ACTION)
        m_val = left_at(code(f"at most {M_ACC:g}", 20, ACTION, weight=m.BOLD), 2.1 + 0.2 * M_ACC + 0.15, -0.55)
        hold = left_at(text(f"it can only hold within {HOLD_DEG:.1f}° of upright", 20, MUTED), 0.5, -0.98)
        arc = m.Arc(radius=1.8, start_angle=-math.pi / 2, angle=math.radians(PUSH_PEAK), arc_center=pc)
        arc.set_stroke(INK, 3)
        end = pc + 1.8 * np.array([math.cos(-math.pi / 2 + math.radians(PUSH_PEAK)),
                                   math.sin(-math.pi / 2 + math.radians(PUSH_PEAK)), 0])
        arc_lab = left_at(code(f"{PUSH_PEAK:.1f}°", 22, INK, weight=m.BOLD), end[0] + 0.2, end[1] - 0.1)
        push_lab = code("full push", 20, MUTED).next_to(arc_lab, m.DOWN, buff=0.08).align_to(arc_lab, m.LEFT)
        pump_note = left_at(cite("section 5's network, from hanging · computed here"), -6.4, -3.1)

        r_parts = [r"r", r"= -\big(", r"\theta^2", r"+ 0.1\,", r"\dot\theta^2", r"+ 0.001\,", r"u^2", r"\big)"]
        r_eq = formula(r_parts, [REWARD, INK, STATE, INK, STATE, INK, ACTION, INK], 42)
        left_at(r_eq, 0.7, 2.0)
        r_labs = m.VGroup(code("angle", 20, STATE), code("speed", 20, STATE), code("torque", 20, ACTION))
        for lab, k in zip(r_labs, (2, 4, 6)):
            lab.move_to([r_eq[k].get_x(), r_eq.get_bottom()[1] - 0.25, 0])
        best = left_at(colored([("best: 0, upright and still", REWARD), ("·", MUTED), ("200 steps", MUTED)], 20), 0.7, 0.85)

        qbars = Bars([0.55, 0.85], labels=["left", "right"], top=1.0, bottom=0.0, width=1.5, height=1.0, color=VALUE,
                     show_values=False, center=(1.45, -2.3, 0), gap=0.25)
        easy = code("max: easy", 20, VALUE, weight=m.BOLD).move_to([1.45, -1.55, 0])
        us = np.linspace(-2, 2, 41)
        qs = 0.3 + 0.62 * np.exp(-((us - 0.7) ** 2) / 0.9)
        comb = m.VGroup(*[m.Rectangle(width=0.045, height=0.95 * q).set_stroke(width=0).set_fill(VALUE, 0.85)
                          .move_to([4.75 + 0.7 * u, -2.8 + 0.95 * q / 2, 0]) for u, q in zip(us, qs)])
        comb_base = m.Line([3.25, -2.8, 0], [6.25, -2.8, 0]).set_stroke(FAINT, 2)
        comb_lab = code("torque −2 … +2", 20, MUTED).move_to([4.75, -3.12, 0])
        hard = code("max over ∞ torques?", 20, RED, weight=m.BOLD).move_to([4.75, -1.55, 0])
        q_arrow = arrow([2.5, -2.3, 0], [3.15, -2.3, 0], MUTED)

        def push_step(n):
            gauge.set(P["max_torque"])

        def pump_step(n):
            gauge.set(PUMP_U[min(n, len(PUMP_U) - 1)])

        self.say(
            "[Our example has a single joint]: [a pendulum, with a motor at its pivot.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(lp, lt, bot, row1, cap1, levers, cap2)),
                             m.FadeOut(m.VGroup(rp, rt, arm, goal, illus, *tq))),
            m.AnimationGroup(m.FadeIn(pend), m.FadeIn(prov, shift=0.15 * m.DOWN),
                             m.Succession(m.Wait(0.4), m.AnimationGroup(m.Create(motor), m.FadeIn(motor_lab)))),
        )
        self.say(
            "[The motor's torque can be any number from minus two to plus two], "
            "[and the goal is to hold the pendulum upright.]",
            m.AnimationGroup(m.FadeIn(gauge), m.Succession(m.Wait(0.5), m.UpdateFromAlphaFunc(
                gauge, lambda g, a: g.set(2 * math.sin(2 * math.pi * a)), run_time=2.4, rate_func=m.linear))),
            m.AnimationGroup(m.FadeIn(goal_dot, scale=0.5), m.FadeIn(goal_lab, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[But the motor is weak against gravity]: "
            "[pushing as hard as it can, the pendulum rises less than fifty degrees.]",
            m.AnimationGroup(m.FadeIn(acc_t), m.FadeIn(g_lab), m.GrowFromEdge(g_bar, m.LEFT), m.FadeIn(g_val),
                             m.FadeIn(m_lab), m.GrowFromEdge(m_bar, m.LEFT), m.FadeIn(m_val),
                             m.Succession(m.Wait(0.8), m.FadeIn(hold))),
            m.AnimationGroup(swing(pend, PUSH_S[:, 0], 2.8, push_step),
                             m.Succession(m.Wait(1.4), m.AnimationGroup(m.Create(arc), m.FadeIn(arc_lab),
                                                                        m.FadeIn(push_lab)))),
        )
        self.say(
            "[So it has to pump: rock one way, then the other, and swing over the top to catch it there.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(arc, arc_lab, push_lab)), m.FadeIn(pump_note),
                             swing(pend, PUMP_TH[:PUMP_SHOW + 1], 5.6, pump_step)),
        )
        self.say(
            "[Every step, the reward is minus the angle squared], [minus a little for speed and for torque]: "
            "[zero means upright and still.]",
            m.AnimationGroup(m.FadeIn(r_eq[:3], shift=0.1 * m.UP), m.FadeIn(r_labs[0])),
            m.AnimationGroup(m.FadeIn(r_eq[3:], shift=0.1 * m.UP), m.FadeIn(r_labs[1:])),
            m.AnimationGroup(m.FadeIn(best, shift=0.1 * m.UP), m.Circumscribe(goal_dot, color=GOLD, buff=0.08)),
        )
        self.say(
            "[A Q-network can't just take the max here]: [there are infinitely many torques to compare.]",
            m.AnimationGroup(m.FadeIn(qbars), m.FadeIn(easy, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.GrowArrow(q_arrow), m.FadeIn(comb_base),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in comb], lag_ratio=0.03),
                             m.FadeIn(comb_lab), m.Succession(m.Wait(0.8), m.FadeIn(hard, shift=0.1 * m.DOWN))),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · a Gaussian policy

    def gaussian(self):
        self.next_section("2 · A bell curve over torque", notes=(
            "Way out 1: a Gaussian policy (S&B eq. 13.19; notes §16 formula 6, §19). The network outputs a mean μ(s) and a "
            "spread σ (log σ, often state-independent); Rover samples u ~ N(μ, σ²). Exploration is built in. The policy "
            "gradient of Episode 16 is unchanged; only the score changes: ∂ln π/∂μ = (u − μ)/σ², ∂ln π/∂log σ = (u − μ)²/σ² "
            f"− 1. Worked example (notes §19 C): μ = {MU0}, σ = {SD0}, u = {U_GOOD}, advantage {ADV:+g}, step {LR}: "
            f"∂/∂μ = {G_MU:.1f} → μ = {MU1:.2f}; ∂/∂log σ = {G_LS:.2f} → σ = {SD1:.4f} (a good torque inside one σ narrows "
            "the bell; outside it would widen it). The 14 dots are illustrative draws from N(0.5, 1) (default_rng(29)). "
            f"The toy run is illustrative but computed here: one state whose reward is −(u − {BEST_U})², {TOY_N} samples per "
            f"update, advantage = reward − batch mean, step {TOY_LR}, default_rng(0), {TOY_UPDATES} updates from "
            f"(μ, σ) = ({MU1:.2f}, {SD1:.2f}) to ({TOY[-1, 0]:.2f}, {TOY[-1, 1]:.2f})."))
        net = NeuralNet(layers=(3, 6, 6, 2), width=2.2, height=1.9, center=(-4.1, 0.75, 0), color=ACTION)
        in_labs = m.VGroup(*[right_at(code(s, 20, STATE), net.columns[0][k].get_left()[0] - 0.2, net.columns[0][k].get_y())
                             for k, s in enumerate(("cos θ", "sin θ", "speed"))])
        out_labs = m.VGroup(left_at(code("mean", 22, ACTION, weight=m.BOLD), net.columns[-1][0].get_right()[0] + 0.2,
                                    net.columns[-1][0].get_y()),
                            left_at(code("spread", 22, EXPLORE, weight=m.BOLD), net.columns[-1][1].get_right()[0] + 0.2,
                                    net.columns[-1][1].get_y()))
        net_lab = text("the policy network", 20, MUTED).next_to(net, m.DOWN, buff=0.25)
        bot = rover(0.42).move_to([-4.1, -1.35, 0])
        plot = Plot((-3, 3), (0, 1.8), width=6.0, height=2.9, center=(2.75, 0.0, 0), x_ticks=[-2, -1, 0, 1, 2],
                    x_label="torque")
        b0 = bell(plot, MU0, SD0)
        read_at = (0.15, 2.15)
        read = left_at(readout(MU0, SD0), *read_at)
        explore = left_at(text("the spread = exploration, built in", 22, EXPLORE, weight=m.BOLD), read_at[0], 1.78)
        dots = m.VGroup(*[m.Dot(plot.coords(u, 0), radius=0.075, color=ACTION).set_z_index(3) for u in SAMPLES])
        good = m.Dot(plot.coords(U_GOOD, 0), radius=0.11, color=ACTION).set_z_index(4)
        good_ring = m.Circle(radius=0.2).set_stroke(SIGNAL, 3).move_to(good).set_z_index(4)
        call = colored([("u = 1.3", ACTION), ("·", MUTED), ("advantage +2", SIGNAL)], 20, weight=m.BOLD)
        call.move_to(plot.coords(U_GOOD, 1.62))
        call_line = m.DashedLine(plot.coords(U_GOOD, 1.5), plot.coords(U_GOOD, 0.06), dash_length=0.07).set_stroke(SIGNAL, 2)

        f_mu = formula([r"\mu", r"\leftarrow", r"\mu", r"+\,\alpha\,", r"A", r"\,\frac{u-\mu}{\sigma^2}",
                        rf"= {MU0:g} + {LR:g}\cdot {ADV:g}\cdot {G_MU:.1f} = {MU1:.2f}"],
                       [ACTION, INK, ACTION, INK, SIGNAL, INK, ACTION], 30)
        left_at(f_mu, -6.4, -2.2)
        f_sd = formula([r"\log\sigma", r"\leftarrow \log\sigma + \alpha\,", r"A", r"\Big(\frac{(u-\mu)^2}{\sigma^2} - 1\Big)",
                        rf"\;\Rightarrow\; \sigma = e^{{\,{LR:g}\cdot {ADV:g}\cdot({G_LS:.2f})}} = {SD1:.2f}"],
                       [EXPLORE, INK, SIGNAL, INK, EXPLORE], 30)
        left_at(f_sd, -6.4, -3.1)
        src16 = right_at(cite("episode 16 · Sutton & Barto, 2018, §13.7"), 6.5, -2.2)
        best_line = plot.vline(BEST_U, GOLD, 2.5)
        best_lab = code(f"pays best: {BEST_U}", 20, GOLD, weight=m.BOLD).next_to(plot.coords(BEST_U, 1.72), m.RIGHT, buff=0.15)
        toy_cap = left_at(cite(f"for example: one state, reward −(u − {BEST_U})² · {TOY_N} samples per update · computed here"),
                          -6.4, -2.6)
        count_at = (-6.4, -3.2)
        counter = left_at(code(f"update  0 of {TOY_UPDATES}", 20, INK), *count_at)

        self.say(
            "[The first way out]: [the policy network outputs a bell curve over torque]: [a mean, and a spread.]",
            m.AnimationGroup(self.new_header("02 · gaussian policy", "A bell curve over torque"), m.FadeIn(net),
                             m.FadeIn(in_labs), m.FadeIn(net_lab)),
            m.AnimationGroup(m.FadeIn(plot), net.forward(ACTION), m.FadeIn(out_labs),
                             m.Succession(m.Wait(0.5), m.AnimationGroup(m.FadeIn(b0[0]), m.Create(b0[1])))),
            m.AnimationGroup(m.Create(b0[2]), m.Create(b0[3]), m.FadeIn(read, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[To act, Rover samples from it]: [most torques land near the mean, and a few wander off.]",
            m.FadeIn(bot, shift=0.15 * m.UP),
            m.LaggedStart(*[m.FadeIn(d, shift=0.35 * m.DOWN, scale=0.5) for d in dots], lag_ratio=0.18),
        )
        self.say(
            "[That spread is exploration, built right in.]",
            m.AnimationGroup(m.FadeIn(explore, shift=0.1 * m.LEFT), m.Circumscribe(b0[3], color=EXPLORE, buff=0.08)),
        )
        self.say(
            "[Episode sixteen's policy gradient works unchanged.] [Say a torque of one point three did well]: "
            "[the mean moves toward it, from zero point five to zero point six six.]",
            m.AnimationGroup(m.FadeIn(f_mu[:6], shift=0.1 * m.UP), m.FadeIn(src16)),
            m.AnimationGroup(dots.animate.set_opacity(0.3), m.FadeIn(good, scale=0.4), m.Create(good_ring),
                             m.Create(call_line), m.FadeIn(call, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(f_mu[6], shift=0.1 * m.LEFT),
                             morph(plot, b0, [(MU0, SD0), (MU1, SD0)], 1.6, read, read_at)),
        )
        self.say(
            "[It landed inside the spread, so the spread narrows too]: [from one to zero point nine three.]",
            m.AnimationGroup(m.FadeIn(f_sd[:4], shift=0.1 * m.UP), m.Circumscribe(b0[3], color=EXPLORE, buff=0.08)),
            m.AnimationGroup(m.FadeIn(f_sd[4], shift=0.1 * m.LEFT),
                             morph(plot, b0, [(MU1, SD0), (MU1, SD1)], 1.4, read, read_at)),
        )
        self.say(
            "[Repeat that over many samples], [and the bell slides toward good torques, and narrows, as Rover grows sure.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(dots, good, good_ring, call, call_line, f_mu, f_sd, src16, explore)),
                             m.Succession(m.Wait(0.4), m.AnimationGroup(m.FadeIn(toy_cap), m.FadeIn(counter),
                                                                        m.Create(best_line), m.FadeIn(best_lab)))),
            morph(plot, b0, TOY, 4.4, read, read_at, counter, count_at, rate=m.linear),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · deterministic policy gradients

    def deterministic(self):
        self.next_section("3 · Follow the critic uphill", notes=(
            "Way out 2: a deterministic actor a = μθ(s) plus a learned critic Q(s, a), differentiable in a; the actor "
            "follows ∇_a Q (DPG theorem, Silver et al., ICML 2014, Theorem 1 / Eq. 9: ∇θ J = E[∇θ μθ(s) ∇_a Q(s, a) at "
            "a = μθ(s)]). DDPG (Lillicrap et al., arXiv Sept 2015, ICLR 2016) = DPG + DQN's replay buffer and target "
            "networks with slow soft updates (τ = 0.001) + temporally correlated Ornstein–Uhlenbeck exploration noise. "
            "TD3 (Fujimoto, van Hoof & Meger, ICML 2018): two critics and the minimum for the target (an underestimate, "
            "'far preferable to overestimation bias'), delayed actor updates, target-policy smoothing — the continuous "
            "cousin of Episode 14's maximization bias. The critic's curves and the hops up the slope are an illustration "
            "(Q(u) = 0.1 + 1.7·exp(−(u − 1.1)²/2.5), six gradient steps u ← u + dQ/du from u = −1.3)."))
        actor = NeuralNet(layers=(3, 5, 5, 1), width=1.5, height=1.35, center=(-5.45, 1.25, 0), color=ACTION)
        critic = NeuralNet(layers=(4, 5, 5, 1), width=1.5, height=1.35, center=(-1.35, 1.25, 0), color=VALUE)
        a_lab = text("actor", 22, ACTION, weight=m.BOLD).next_to(actor, m.DOWN, buff=0.2)
        c_lab = text("critic", 22, VALUE, weight=m.BOLD).next_to(critic, m.DOWN, buff=0.2)

        def u_chip(u):
            return chip(minus(f"u = {u:+.2f}"), ACTION, CORAL_FILL, width=1.75, height=0.6, size=22, mono=True).move_to([-3.4, 1.25, 0])

        uc = u_chip(HOPS[0])
        a1 = arrow(actor.get_right() + [0.05, 0, 0], [-4.32, 1.25, 0], MUTED)
        a2 = arrow([-2.48, 1.25, 0], critic.get_left() + [-0.05, 0, 0], MUTED)
        plot = Plot((-2, 2), (0, 2.2), width=4.4, height=1.6, center=(3.85, 0.95, 0), x_ticks=[-2, 0, 2],
                    x_label="torque u", y_label="the critic's value")
        us = np.linspace(-2, 2, 121)
        qc = plot.line(us, q_guess(us), VALUE, 4)
        qc2 = plot.line(us, q_guess2(us), VALUE, 3).set_stroke(opacity=0.45)
        qmin = plot.line(us, np.minimum(q_guess(us), q_guess2(us)), VALUE, 6)
        illus = right_at(cite("illustration"), 6.5, 2.35)
        dot = m.Dot(plot.coords(HOPS[0], q_guess(HOPS[0])), radius=0.11, color=ACTION).set_z_index(4)

        def slope_arrow(u):
            p0 = plot.coords(u, q_guess(u))
            d = plot.coords(u + 0.3, q_guess(u) + 0.3 * dq_guess(u)) - p0
            n = np.linalg.norm(d)
            length = 0.25 + 0.55 * min(abs(dq_guess(u)) / 1.0, 1.0)
            return m.Arrow(p0, p0 + d / n * length, buff=0.12, color=SIGNAL, stroke_width=6,
                           max_tip_length_to_length_ratio=0.35).set_z_index(3)

        slope = slope_arrow(HOPS[0])

        def climb(seconds):
            path = np.array(HOPS)
            last = len(path) - 1

            def step(_, alpha):
                i = alpha * last
                j = int(np.floor(i))
                k = min(j + 1, last)
                f = i - j
                f = f * f * (3 - 2 * f)
                u = (1 - f) * path[j] + f * path[k]
                dot.move_to(plot.coords(u, q_guess(u)))
                slope.become(slope_arrow(u))
                uc.become(u_chip(u))

            return m.UpdateFromAlphaFunc(dot, step, run_time=seconds, rate_func=m.linear)

        dpg = formula([r"\nabla_\theta J", r"= \mathbb{E}\big[\,", r"\nabla_\theta \mu_\theta(s)", r"\;",
                       r"\nabla_u Q(s,u)", r"\,\big]"], [SIGNAL, INK, ACTION, INK, VALUE, INK], 34)
        dpg.move_to([-3.4, -0.8, 0])

        cards = []
        specs = [("DPG", "2014", "follow the critic's slope", "Silver et al., 2014"),
                 ("DDPG", "2015", "+ replay, slow targets, noise", "Lillicrap et al., 2015"),
                 ("TD3", "2018", "two critics: trust the smaller", "Fujimoto et al., 2018")]
        for k, (name, year, line, src) in enumerate(specs):
            cx = -4.45 + 4.45 * k
            box = card(4.0, 1.75, ACTION, CARD, radius=0.18, stroke=2.5).move_to([cx, -2.4, 0])
            nm = code(name, 26, ACTION, weight=m.BOLD)
            yr = pill(year, MUTED, CARD, 20, mono=True)
            head = m.VGroup(nm, yr).arrange(m.RIGHT, buff=0.25)
            left_at(head, cx - 1.75, -1.85)
            ln = left_at(text(line, 22, INK), cx - 1.75, -2.4)
            sr = left_at(cite(src), cx - 1.75, -2.9)
            cards.append(m.VGroup(box, head, ln, sr))
        links = [arrow(cards[k][0].get_right(), cards[k + 1][0].get_left(), MUTED, buff=0.06) for k in range(2)]
        noise = m.VGroup(*[m.Dot(plot.coords(1.1 + d, 0.06), radius=0.06, color=EXPLORE) for d in (-0.5, -0.25, 0.15, 0.4)])
        noise_lab = code("noise", 20, EXPLORE, weight=m.BOLD).next_to(plot.coords(1.1, 0.0), m.UP, buff=0.32)

        self.say(
            "[The second way out drops the bell curve]: [the actor outputs one exact torque], "
            "[and a critic, as in episode seventeen, learns how good it is.]",
            m.AnimationGroup(self.new_header("03 · deterministic", "Follow the critic uphill"), m.FadeIn(actor),
                             m.FadeIn(a_lab)),
            m.AnimationGroup(actor.forward(ACTION), m.GrowArrow(a1), m.FadeIn(uc, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(critic), m.FadeIn(c_lab), m.FadeIn(plot), m.FadeIn(illus),
                             m.Succession(m.Wait(0.4), m.AnimationGroup(m.Create(qc), m.FadeIn(dot, scale=0.5)))),
        )
        self.say(
            "[The actor then turns its knob uphill, the way the critic's value rises]: "
            "[the deterministic policy gradient, from twenty fourteen.]",
            m.AnimationGroup(m.FadeIn(slope, run_time=0.6), m.Succession(m.Wait(0.8), climb(3.4))),
            m.AnimationGroup(m.FadeIn(dpg, shift=0.1 * m.UP), m.FadeIn(cards[0], shift=0.15 * m.UP)),
        )
        self.say(
            "[In twenty fifteen, DDPG added episode thirteen's tricks]: [a replay memory and slow target networks], "
            "[plus noise on the torque, to explore.]",
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(cards[1][0]), m.FadeIn(cards[1][1])),
            m.FadeIn(cards[1][2:], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(slope), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in noise], lag_ratio=0.2),
                             m.FadeIn(noise_lab)),
        )
        self.say(
            "[In twenty eighteen, TD3 tamed its overconfidence]: [two critics, trusting the smaller one], "
            "[the cure from episode fourteen.]",
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(cards[2][0]), m.FadeIn(cards[2][1])),
            m.AnimationGroup(m.FadeIn(qc2), m.Succession(m.Wait(0.5), m.Create(qmin)), m.FadeIn(cards[2][2])),
            m.AnimationGroup(m.FadeIn(cards[2][3]), m.Circumscribe(cards[2], color=ACTION, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · soft actor-critic

    def soft(self):
        self.next_section("4 · Soft actor-critic", notes=(
            "Haarnoja, Zhou, Abbeel & Levine, 'Soft Actor-Critic', ICML 2018 (arXiv 1801.01290): a stochastic (Gaussian) "
            "actor learned off-policy from a replay buffer, two Q-functions with the minimum. Eq. 1: J(π) = Σ_t "
            "E[r(s_t, a_t) + α H(π(·|s_t))]; α → 0 recovers the usual objective. Abstract: 'to succeed at the task while "
            "acting as randomly as possible'. Automatic temperature (Haarnoja et al. 2018b, 'Soft Actor-Critic Algorithms "
            "and Applications', arXiv 1812.05905): α is tuned to hold the entropy near a target −dim(A) (−1 for the "
            "pendulum); our run (deep_rl.json methods.sac.hp) used it: autotune_alpha = true, target_entropy = −1. The two "
            f"bells are an illustration (σ = {SD_WIDE:g} and {SD_NARROW:g}); their entropies ½ ln(2πeσ²) = {H_WIDE:.3f} and "
            f"{H_NARROW:.3f} nats are computed. The slider's motion is illustrative."))
        name = pill("soft actor-critic", ACTION, CORAL_FILL, 24)
        left_at(name, -6.4, 2.05)
        src = cite("Haarnoja, Zhou, Abbeel & Levine, 2018").next_to(name, m.RIGHT, buff=0.35)
        mb = mini_bell(0, 0, 0.6, width=0.9, height=0.45)
        t1 = text("keeps the bell curve", 22, INK)
        strip = cell_grid(1, 5, size=0.22, gap=0.05, color=STATE, fill=TEAL_FILL)
        t2 = text("off-policy, from replay", 22, INK)
        crit2 = m.VGroup(m.Dot(radius=0.1, color=VALUE), m.Dot(radius=0.1, color=VALUE)).arrange(m.RIGHT, buff=0.1)
        t3 = text("two critics, like TD3", 22, INK)
        pairs = [m.VGroup(a, b).arrange(m.RIGHT, buff=0.2) for a, b in ((mb, t1), (strip, t2), (crit2, t3))]
        m.VGroup(*pairs).arrange(m.RIGHT, buff=0.75).move_to([0, 1.2, 0])
        tags = m.VGroup(mb, t1, strip, t2, crit2, t3)

        parts = [r"J(\pi) = \sum_t \mathbb{E}\big[\,", r"r", r"\;+\;", r"\alpha", r"\,", r"\mathcal{H}\big(\pi(\cdot \mid s_t)\big)",
                 r"\,\big]"]
        goal = formula(parts, [INK, REWARD, INK, EXPLORE, INK, EXPLORE, INK], 48).move_to([-0.3, 0.0, 0])
        lab_r = code("reward", 20, REWARD, weight=m.BOLD).next_to(goal[1], m.DOWN, buff=0.3)
        lab_a = code("temperature", 20, EXPLORE, weight=m.BOLD).next_to(goal[3], m.UP, buff=0.3)
        lab_h = code("entropy", 20, EXPLORE, weight=m.BOLD).next_to(goal[5], m.DOWN, buff=0.3)

        wide = mini_bell(-4.7, -2.45, SD_WIDE)
        narrow = mini_bell(-1.4, -2.45, SD_NARROW)
        h_wide = colored([("entropy", MUTED), (f"{H_WIDE:.2f}", EXPLORE)], 20, weight=m.BOLD).move_to([-4.7, -2.85, 0])
        h_narrow = colored([("entropy", MUTED), (f"{H_NARROW:.2f}", EXPLORE)], 20, weight=m.BOLD).move_to([-1.4, -2.85, 0])
        bonus = m.VGroup(text("bigger bonus", 20, EXPLORE, weight=m.BOLD).move_to([-4.7, -3.25, 0]),
                         text("smaller bonus", 20, MUTED).move_to([-1.4, -3.25, 0]))

        quote = text("“Succeed at the task while acting as randomly as possible.”", 28, INK, weight=m.BOLD)
        quote.move_to([-0.1, 1.3, 0])
        q_src = right_at(cite("Haarnoja et al., 2018, abstract"), 6.4, 0.8)

        sx0, sx1, sy = 1.3, 6.1, -1.9
        track = m.Line([sx0, sy, 0], [sx1, sy, 0]).set_stroke(EDGE, 6)
        ends = m.VGroup(code("reward only", 20, REWARD).move_to([sx0 + 0.55, sy - 0.38, 0]),
                        code("pure randomness", 20, EXPLORE).move_to([sx1 - 0.85, sy - 0.38, 0]))
        knob = m.Dot([sx0 + 0.75 * (sx1 - sx0), sy, 0], radius=0.15, color=EXPLORE).set_z_index(3)
        s_title = left_at(code("temperature α", 22, EXPLORE, weight=m.BOLD), sx0, sy + 0.55)
        auto = left_at(text("tuned automatically to a target entropy", 20, MUTED), sx0 - 0.1, sy - 0.95)
        auto_src = left_at(cite("Haarnoja et al., 2018b · our run did this too"), sx0 - 0.1, sy - 1.32)

        def settle(_, a):
            x = sx0 + (0.75 - 0.4 * a + 0.12 * math.sin(3 * math.pi * a) * (1 - a)) * (sx1 - sx0)
            knob.move_to([x, sy, 0])

        self.say(
            "[Soft actor-critic, also from twenty eighteen, keeps the bell curve], [but learns off-policy, from a replay memory.]",
            m.AnimationGroup(self.new_header("04 · soft actor-critic", "Reward, plus a bonus for spread"),
                             m.FadeIn(name, shift=0.1 * m.RIGHT), m.FadeIn(src), m.FadeIn(mb), m.FadeIn(t1)),
            m.AnimationGroup(m.FadeIn(strip), m.FadeIn(t2), m.Succession(m.Wait(0.6), m.AnimationGroup(m.FadeIn(crit2),
                                                                                                    m.FadeIn(t3)))),
        )
        self.say(
            "[Its goal adds a bonus to every reward]: [the temperature, times the entropy of the bell curve.]",
            m.AnimationGroup(m.FadeIn(goal[:3], shift=0.1 * m.UP), m.FadeIn(goal[6], shift=0.1 * m.UP), m.FadeIn(lab_r)),
            m.AnimationGroup(m.FadeIn(goal[3:6], shift=0.1 * m.UP), m.FadeIn(lab_a, shift=0.1 * m.DOWN),
                             m.FadeIn(lab_h, shift=0.1 * m.UP)),
        )
        self.say(
            "[Entropy measures the spread]: [a wide curve earns a bigger bonus than a narrow one.]",
            m.AnimationGroup(m.FadeIn(wide), m.FadeIn(h_wide), m.Circumscribe(goal[5], color=EXPLORE, buff=0.08)),
            m.AnimationGroup(m.FadeIn(narrow), m.FadeIn(h_narrow), m.FadeIn(bonus, shift=0.1 * m.UP)),
        )
        self.say(
            "[In its authors' words]: [succeed at the task while acting as randomly as possible.]",
            m.AnimationGroup(m.FadeOut(tags), m.FadeIn(q_src)),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.say(
            "[The temperature sets the trade-off]; "
            "[a later version tunes it automatically, as our run does, to hold the entropy near a target.]",
            m.AnimationGroup(m.FadeIn(track), m.FadeIn(ends), m.FadeIn(s_title), m.FadeIn(knob, scale=0.5),
                             m.Circumscribe(goal[3], color=EXPLORE, buff=0.08)),
            m.AnimationGroup(m.UpdateFromAlphaFunc(knob, settle, run_time=3.0), m.FadeIn(auto), m.FadeIn(auto_src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · measured on this Mac

    def measured_runs(self):
        self.next_section("5 · The pendulum, learned on this Mac", notes=(
            "trajectories.json policies.sac_pendulum: run sac_pendulum_s10 (seed 10, final weights after 20,000 env steps), "
            f"reset seed 2026 (this start is {T_START:.0f}° from upright, partway up), deterministic torque = 2·tanh(mean), "
            f"200 steps, return {TRAJ['return']:.3f} (recomputed here with pendulum_step: {T_RET[-1]:.3f}); full torque for "
            f"the first 13 steps, upright (|θ| < 0.1 rad) from step {T_UP} = {T_UP * P['dt']:.2f} s. The raw means come from "
            "load_policy('sac_pendulum') (only the mean head was exported); training sampled u ~ N(mean, σ) and played "
            f"2·tanh(u). deep_rl.json, {SEEDS} seeds (10–14), evaluation = deterministic policy on 10 fixed start states "
            f"(mean over seeds, band = min–max): SAC {H_SAC['final_return_mean']} ± {H_SAC['final_return_std']} (seeds "
            f"{H_SAC['final_return_min']} to {H_SAC['final_return_max']}) after {SAC_STEPS:,} steps, {SAC_WALL:.1f} s per "
            f"run; first evaluation above −200 after {SAC_200:,.0f} steps ({SAC_200 / P['max_steps']:.0f} episodes; evals "
            f"every 1,000 steps). PPO (Gaussian policy, clip 0.2, γ = 0.9, first-choice settings): {PPO_RET} ± {PPO_SD} "
            f"after {PPO_STEPS:,} steps, {PPO_WALL:.1f} s per run; first above −200 after {H_PPO['steps_to_minus200_mean']:,.0f}"
            f" steps. Per step: {SAC_MS:.2f} ms vs {PPO_MS:.3f} ms (≈ {PER_STEP:.1f}×). Wall times: ten seeds ran at once "
            "on a shared machine. For comparison (not ours): RL-Zoo's tuned SAC −157.0 at 20k, PPO −172.2 at 100k "
            "(SB3 benchmark, notes §19)."))
        pc = np.array([-4.35, 0.55, 0])
        pend = PendulumView(center=pc, length=1.3)
        pend.set_angle(T_TH[0])
        mac = left_at(cite(f"trained on this Mac · seed 10 · {SAC_STEPS // 1000}k steps"), -6.4, 2.35)

        def step_text(n):
            return left_at(code(f"step {n:>3} of 200", 20, INK), -5.7, -1.2)

        def ret_text(n):
            return left_at(colored([("return", REWARD), (minus(f"{T_RET[n]:.1f}"), REWARD)], 22, weight=m.BOLD), -5.7, -1.62)

        counter, ret = step_text(0), ret_text(0)
        gauge = Gauge((-3.8, -2.5, 0), width=3.0)
        gauge.set(T_U[0])

        tplot = Plot((-4, 4), (-2.5, 2.5), width=5.0, height=3.0, center=(2.85, 0.1, 0), x_ticks=[-4, -2, 0, 2, 4],
                     y_ticks=[-2, 0, 2], x_label="network output: the mean", y_label="torque")
        xs = np.linspace(-4, 4, 161)
        t1 = tplot.line(xs, np.tanh(xs), MUTED, 3)
        t2 = tplot.line(xs, P["max_torque"] * np.tanh(xs), ACTION, 5)
        lims = m.VGroup(tplot.hline(2.0, FAINT, 2), tplot.hline(-2.0, FAINT, 2))
        eq = formula([r"\text{torque}", r"= 2\,", r"\tanh", r"(\text{mean})"], [ACTION, INK, INK, ACTION], 36)
        right_at(eq, 6.4, 2.3)
        sq = code("squash: −1 … +1", 20, MUTED).move_to(tplot.coords(2.2, -1.0))
        train = left_at(cite("in training: 2·tanh of a sample around the mean"), 0.4, -2.45)

        def tdot_at(n):
            u = float(np.clip(T_MEAN[min(n, 199)], -4, 4))
            return tplot.coords(u, P["max_torque"] * math.tanh(u))

        tdot = m.Dot(tdot_at(0), radius=0.12, color=ACTION).set_z_index(4)

        def replay_step(n):
            counter.become(step_text(n))
            ret.become(ret_text(n))
            gauge.set(T_U[min(n, 199)])
            tdot.move_to(tdot_at(n))

        # the measured comparison
        plot = Plot((1000, 600000), (-1600, 0), width=5.3, height=3.2, center=(-3.1, 0.05, 0), log_x=True,
                    x_ticks=[1000, 10000, 100000, 500000], y_ticks=[-1500, -1000, -500, 0],
                    x_label="environment steps (log scale)", y_label="return per episode")
        tick_names = {"1000": "1k", "10000": "10k", "100000": "100k", "500000": "500k"}
        for old in [t for t in plot.ticks if isinstance(t, m.Text) and t.text in tick_names]:
            new = code(tick_names[old.text], 20, MUTED).move_to(old.get_center())
            plot.ticks.remove(old)
            plot.ticks.add(new)
        top_note = right_at(cite(f"trained on this Mac · {SEEDS} seeds each · mean, band = min–max"), 6.5, 2.35)
        good = plot.hline(-200, GREEN, 2.5)
        good_lab = code("−200", 20, GREEN, weight=m.BOLD).next_to(plot.coords(1000, -200), m.LEFT, buff=0.16)
        sac_band = plot.band(SAC_X, SAC_Y["min"], SAC_Y["max"], SAC_C, 0.2)
        sac_line = plot.line(SAC_X, SAC_Y["mean"], SAC_C, 5)
        ppo_band = plot.band(PPO_X, PPO_Y["min"], PPO_Y["max"], PPO_C, 0.18)
        ppo_line = plot.line(PPO_X, PPO_Y["mean"], PPO_C, 4)
        sac_end = plot.dot(SAC_STEPS, H_SAC["final_return_mean"], SAC_C, 0.09)
        ppo_end = plot.dot(PPO_X[-1], PPO_Y["mean"][-1], PPO_C, 0.09)
        hit = plot.vline(SAC_200, SAC_C, 2)
        hit_dot = plot.dot(SAC_200, -200, SAC_C, 0.09)
        hit_lab = code(f"{SAC_200 / 1000:.1f}k steps", 20, SAC_C, weight=m.BOLD)
        left_at(hit_lab, plot.coords(SAC_200, 0)[0] + 0.12, plot.coords(1000, -950)[1])
        leg_x = plot.coords(40000, 0)[0]
        leg = m.VGroup(
            left_at(colored([("SAC", SAC_C), (minus(f"{H_SAC['final_return_mean']:.1f} ± {H_SAC['final_return_std']:.1f}"),
                                               SAC_C)], 20, weight=m.BOLD), leg_x, plot.coords(1000, -1180)[1]),
            left_at(colored([("PPO", PPO_C), (minus(f"{PPO_RET:.0f} ± {PPO_SD:.0f}"), PPO_C)], 20, weight=m.BOLD), leg_x,
                    plot.coords(1000, -1420)[1]))
        leg_sw = m.VGroup(m.Line([0, 0, 0], [0.35, 0, 0]).set_stroke(SAC_C, 5).next_to(leg[0], m.LEFT, buff=0.15),
                          m.Line([0, 0, 0], [0.35, 0, 0]).set_stroke(PPO_C, 4).next_to(leg[1], m.LEFT, buff=0.15))

        # the scoreboard
        bx = 1.75
        rows_y = (1.55, -0.05, -1.65)
        heads = [left_at(text(t, 22, INK, weight=m.BOLD), 0.75, y + 0.42) for t, y in
                 zip(("environment steps", "final return, 5 seeds", "wall time per run"), rows_y)]
        notes = [text("fewer is better", 20, MUTED).next_to(heads[0], m.RIGHT, buff=0.25).align_to(heads[0], m.DOWN),
                 text("closer to 0 is better", 20, MUTED).next_to(heads[1], m.RIGHT, buff=0.25).align_to(heads[1], m.DOWN),
                 text("on a shared Mac", 20, MUTED).next_to(heads[2], m.RIGHT, buff=0.25).align_to(heads[2], m.DOWN)]

        def bar_row(y, k, value, scale, label, color, tag):
            yy = y - 0.42 * k
            t = right_at(code(tag, 20, color), bx - 0.15, yy)
            b = hbar(bx, yy, scale * value, 0.28, color)
            v = left_at(code(label, 20, color, weight=m.BOLD), bx + scale * value + 0.15, yy)
            return m.VGroup(t, b, v)

        steps_sac = bar_row(rows_y[0], 0, SAC_STEPS, 3.4 / PPO_STEPS, f"{SAC_STEPS // 1000}k", SAC_C, "SAC")
        steps_ppo = bar_row(rows_y[0], 1, PPO_STEPS, 3.4 / PPO_STEPS, f"{PPO_STEPS / 1000:.0f}k", PPO_C, "PPO")
        ret_sac = m.VGroup(right_at(code("SAC", 20, SAC_C), bx - 0.15, rows_y[1]),
                           left_at(code(minus(f"{H_SAC['final_return_mean']:.1f} ± {H_SAC['final_return_std']:.1f}"), 22,
                                        SAC_C, weight=m.BOLD), bx, rows_y[1]))
        ret_ppo = m.VGroup(right_at(code("PPO", 20, PPO_C), bx - 0.15, rows_y[1] - 0.42),
                           left_at(code(minus(f"{PPO_RET:.0f} ± {PPO_SD:.0f}"), 22, PPO_C, weight=m.BOLD), bx,
                                   rows_y[1] - 0.42))
        wall_sac = bar_row(rows_y[2], 0, SAC_WALL, 3.4 / 60, f"{SAC_WALL:.0f} s", SAC_C, "SAC")
        wall_ppo = bar_row(rows_y[2], 1, PPO_WALL, 3.4 / 60, f"{PPO_WALL:.0f} s", PPO_C, "PPO")
        per = left_at(colored([("per step:", MUTED), (f"{SAC_MS:.1f} ms", SAC_C), ("vs", MUTED), (f"{PPO_MS:.2f} ms", PPO_C)],
                              20), 0.75, rows_y[2] - 0.85)
        off = pill("off-policy: reuses old steps, sample-efficient", SAC_C, CORAL_FILL, 20)
        on = pill("on-policy: simpler, cheap per step", PPO_C, CARD, 20)
        pills = m.VGroup(off, on).arrange(m.RIGHT, buff=0.35).move_to([0, -3.22, 0])

        phase_a = m.VGroup(pend, mac, counter, ret, gauge, tplot, t2, lims, eq, sq, train, tdot)
        self.say(
            "[Here's soft actor-critic on the pendulum], [trained on this Mac.]",
            m.AnimationGroup(self.new_header("05 · on this Mac", "The pendulum, learned"), m.FadeIn(pend),
                             m.FadeIn(counter), m.FadeIn(ret), m.FadeIn(gauge)),
            m.FadeIn(mac, shift=0.1 * m.DOWN),
        )
        self.say(
            "[The network's output can be any number], [so a squashing curve folds it into minus one to one], "
            "[and doubling gives the torque: always between minus two and two.]",
            m.AnimationGroup(m.FadeIn(tplot), m.FadeIn(eq[3], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(t1), m.FadeIn(sq), m.FadeIn(eq[2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.ReplacementTransform(t1, t2), m.FadeIn(lims), m.FadeIn(eq[:2], shift=0.1 * m.UP),
                             m.FadeIn(tdot, scale=0.5), m.FadeIn(train)),
        )
        self.say(
            "[One hard swing, and within a second and a half it holds the top]: [minus one hundred eighteen, for this start.]",
            m.Succession(swing(pend, T_TH[:T_UP + 1], 3.0, replay_step, rate=m.linear),
                         swing(pend, T_TH[T_UP:], 2.4, lambda n: replay_step(n + T_UP), rate=m.linear)),
            m.Circumscribe(ret, color=REWARD, buff=0.08),
        )
        self.say(
            "[Over five seeds, it scored minus one hundred forty-seven, give or take one point six], "
            "[after only twenty thousand steps.]",
            m.AnimationGroup(m.FadeOut(phase_a), m.FadeIn(plot), m.FadeIn(top_note), m.Create(good), m.FadeIn(good_lab),
                             m.Succession(m.Wait(0.6), m.AnimationGroup(m.FadeIn(sac_band), m.Create(sac_line, run_time=1.8))),
                             m.Succession(m.Wait(1.8), m.AnimationGroup(m.FadeIn(sac_end, scale=0.5), m.FadeIn(leg_sw[0]),
                                                                        m.FadeIn(leg[0]), m.FadeIn(heads[1]),
                                                                        m.FadeIn(notes[1]), m.FadeIn(ret_sac)))),
            m.AnimationGroup(m.FadeIn(heads[0]), m.FadeIn(notes[0]), m.FadeIn(steps_sac[0]),
                             m.GrowFromEdge(steps_sac[1], m.LEFT), m.FadeIn(steps_sac[2])),
        )
        self.say(
            "[It was already better than minus two hundred after about four thousand four hundred steps.]",
            m.AnimationGroup(m.Create(hit), m.FadeIn(hit_dot, scale=0.5), m.FadeIn(hit_lab, shift=0.1 * m.LEFT),
                             m.Circumscribe(good_lab, color=GREEN, buff=0.06)),
        )
        self.say(
            "[For contrast, PPO from episode eighteen took about half a million steps], "
            "[and ended lower, at minus two hundred twenty-two, give or take twenty-two.]",
            m.AnimationGroup(m.FadeIn(ppo_band), m.Create(ppo_line, run_time=2.0), m.FadeIn(steps_ppo[0]),
                             m.GrowFromEdge(steps_ppo[1], m.LEFT), m.FadeIn(steps_ppo[2])),
            m.AnimationGroup(m.FadeIn(ppo_end, scale=0.5), m.FadeIn(leg_sw[1]), m.FadeIn(leg[1]), m.FadeIn(ret_ppo)),
        )
        self.say(
            "[Yet each of its steps is far cheaper]: [its whole run took twenty-nine seconds, against fifty-six.]",
            m.AnimationGroup(m.FadeIn(heads[2]), m.FadeIn(notes[2]), m.FadeIn(per, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(wall_sac[0]), m.GrowFromEdge(wall_sac[1], m.LEFT), m.FadeIn(wall_sac[2]),
                             m.FadeIn(wall_ppo[0]), m.GrowFromEdge(wall_ppo[1], m.LEFT), m.FadeIn(wall_ppo[2])),
        )
        self.say(
            "[Reusing old experience pays when steps are expensive, like on a real robot]; "
            "[on-policy learning is simpler, and wins when steps are cheap.]",
            m.FadeIn(off, shift=0.1 * m.UP),
            m.FadeIn(on, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · sim to real

    def sim_to_real(self):
        self.next_section("6 · Sim to real", notes=(
            "Real robots are slow and fragile, so they learn in simulation; a policy tuned to one simulator can miss in the "
            "real world (the two arms are an illustration: the 'real' arm's links are 22% and 18% longer, and it "
            "plays the simulator's joint angles). Domain randomization: train over random masses, friction, lengths, "
            "textures/colours and delays — Tobin, Fong, Ray, Schneider, Zaremba & Abbeel, IROS 2017 (a real-world object "
            "detector, accurate to 1.5 cm, trained only on simulated images with random textures; quote verbatim). "
            "OpenAI (Akkaya et al.), 'Solving Rubik's Cube with a Robot Hand', Oct 2019: automatic domain randomization, "
            f"'roughly 13 thousand years' of cumulative experience; solved scrambles needing 15 face rotations {RUBIK_15}% "
            f"of the time and 26 rotations {RUBIK_26}%; the move sequence came from Kociemba's solver — RL learned the "
            "manipulation. OpenAI (Andrychowicz et al.), 'Learning Dexterous In-Hand Manipulation', 2018: about 3 years of "
            "simulated experience without randomization vs about 100 years fully randomized for the same performance. The "
            "ghost arms (default_rng(19), lengths ±10%) and the cube's colours are illustrative."))
        sc, rc = np.array([-3.35, 0.4, 0]), np.array([3.35, 0.4, 0])
        sim = card(6.1, 3.6, STATE, CARD).move_to(sc)
        real = card(6.1, 3.6, INK, CARD).move_to(rc)
        s_lab = left_at(text("simulator", 24, STATE, weight=m.BOLD), sc[0] - 2.8, sc[1] + 1.42)
        r_lab = left_at(text("real world", 24, INK, weight=m.BOLD), rc[0] - 2.8, rc[1] + 1.42)
        L1, L2 = 1.35, 1.1
        s_arm = Arm(sc + [-1.95, -1.35, 0], L1, L2)
        r_arm = Arm(rc + [-1.95, -1.35, 0], L1 * 1.22, L2 * 1.18)
        s_tgt_p, r_tgt_p = s_arm.base() + [1.75, 1.05, 0], r_arm.base() + [1.75, 1.05, 0]

        def target(p):
            return m.VGroup(m.Circle(radius=0.2).set_stroke(GOLD, 3), m.Dot(radius=0.08, color=GOLD)).move_to(p)

        s_tgt, r_tgt = target(s_tgt_p), target(r_tgt_p)
        q_sim = s_arm.solve(s_tgt_p)
        miss_arm = Arm(r_arm.base(), r_arm.l1, r_arm.l2).set_q(*q_sim)
        miss_p = miss_arm.tip()
        assert np.linalg.norm(miss_p - r_tgt_p) > 0.3          # the simulator's angles miss in the "real" arm
        miss = m.VGroup(m.Circle(radius=0.22).set_stroke(RED, 3).move_to(miss_p),
                        left_at(code("miss", 20, RED, weight=m.BOLD), miss_p[0] + 0.5, miss_p[1]))
        rng = np.random.default_rng(19)
        ghosts = []
        for _ in range(6):
            j1, j2 = rng.uniform(0.9, 1.1, 2)
            g = Arm(sc + [-1.95, -1.35, 0], L1 * j1, L2 * j2, opacity=0.35, stroke=7)
            ghosts.append((g, g.solve(s_tgt_p)))
        tints = [mix(CARD, c, 0.28) for c in (TEAL, VIOLET, GOLD, BLUE, CARD)]

        def recolor(_, a):
            k = min(int(a * len(tints)), len(tints) - 1)          # one pass through the tints, ending plain
            sim.set_fill(tints[k], 1)

        chips = m.VGroup(*[chip(w, STATE, TEAL_FILL, width=1.3, height=0.44, size=22)
                           for w in ("mass", "friction", "length", "colour", "delay")]).arrange(m.DOWN, buff=0.07)
        chips.move_to([sc[0] + 2.2, sc[1] - 0.25, 0])
        dice = icon(DICE, 0.45).move_to([sc[0] + 2.2, sc[1] + 1.4, 0])
        row = m.VGroup(dice, chips)
        quote = m.VGroup(text("“With enough variability in the simulator, the real world", 22, INK),
                         text("may appear to the model as just another variation.”", 22, INK),
                         cite("Tobin et al., 2017")).arrange(m.DOWN, buff=0.12)
        quote.move_to([0, -2.45, 0])
        check = m.VGroup(m.Circle(radius=0.22).set_stroke(GREEN, 3).move_to(r_tgt_p),
                         left_at(code("reached", 20, GREEN, weight=m.BOLD), r_tgt_p[0] + 0.5, r_tgt_p[1]))

        # the Rubik's cube
        cb = cube(-3.9, -0.35, s=0.62)
        hand = icon("robot-arm", 1.3).move_to([-5.85, -1.3, 0])
        c_src = cite("OpenAI (Akkaya et al.), 2019").move_to([-3.9, -2.75, 0])
        xr = 0.3
        yrs = m.VGroup(title_text(f"{RUBIK_YEARS:,} years", 52, STATE), text("of simulated practice", 22, MUTED))
        yrs.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(yrs, xr, 1.75)
        r15 = m.VGroup(title_text(f"{RUBIK_15}%", 40, ACTION), text("of scrambles needing 15 turns", 22, INK)).arrange(m.RIGHT, buff=0.3)
        r26 = m.VGroup(title_text(f"{RUBIK_26}%", 40, ACTION), text("of the hardest, 26 turns", 22, INK)).arrange(m.RIGHT, buff=0.3)
        left_at(r15, xr, 0.5)
        left_at(r26, xr, -0.15)
        r26[1].align_to(r15[1], m.LEFT)
        solver = left_at(text("moves: Kociemba's solver · RL: the fingers", 20, MUTED), xr, -0.8)
        d_head = left_at(text("the price of randomizing", 22, INK, weight=m.BOLD), xr, -1.6)
        d_rows = m.VGroup()
        for k, (lab, yv, col) in enumerate((("plain", DACTYL_PLAIN, MUTED), ("randomized", DACTYL_RAND, STATE))):
            y = -2.1 - 0.45 * k
            d_rows.add(m.VGroup(right_at(code(lab, 20, col), xr + 1.75, y), hbar(xr + 1.9, y, 0.025 * yv, 0.28, col),
                                left_at(code(f"{yv} years", 20, col, weight=m.BOLD), xr + 1.9 + 0.025 * yv + 0.15, y)))
        d_src = left_at(cite("OpenAI's earlier hand (Andrychowicz et al.), 2018"), xr, -3.1)

        self.say(
            "[Real robots are slow and fragile], [so they practise in simulation.]",
            m.AnimationGroup(self.new_header("06 · sim to real", "Noise in the simulator"), m.FadeIn(sim), m.FadeIn(s_lab),
                             m.FadeIn(s_arm), m.FadeIn(s_tgt, scale=0.5)),
            s_arm.move(q_sim, 2.0),
        )
        self.say(
            "[But no simulator matches the real world exactly], [and a policy tuned to one world can miss in another.]",
            m.AnimationGroup(m.FadeIn(real), m.FadeIn(r_lab), m.FadeIn(r_arm), m.FadeIn(r_tgt, scale=0.5)),
            m.AnimationGroup(r_arm.move(q_sim, 2.0), m.Succession(m.Wait(1.8), m.FadeIn(miss, scale=0.6))),
        )
        self.say(
            "[So randomize what you can't model]: [masses, friction, lengths, colours, delays.]",
            m.AnimationGroup(m.FadeIn(dice, scale=0.6), m.UpdateFromAlphaFunc(sim, recolor, run_time=2.6, rate_func=m.linear)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in chips], lag_ratio=0.25),
                             m.LaggedStart(*[m.FadeIn(g) for g, _ in ghosts], lag_ratio=0.1),
                             m.Succession(m.Wait(1.7), m.AnimationGroup(*[g.move(q, 2.0) for g, q in ghosts]))),
        )
        self.say(
            "[Tobin and colleagues, twenty seventeen]: [with enough variety, the real world may look like just another variation.]",
            m.FadeIn(quote, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(miss), r_arm.move(r_arm.solve(r_tgt_p), 2.0),
                             m.Succession(m.Wait(1.9), m.FadeIn(check, scale=0.6))),
        )
        self.say(
            "[In twenty nineteen, OpenAI's robot hand learned to turn a Rubik's cube this way]: "
            "[about thirteen thousand years of simulated practice.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(sim, real, s_lab, r_lab, s_arm, r_arm, s_tgt, r_tgt, check, row, quote,
                                                *[g for g, _ in ghosts])),
                             m.Succession(m.Wait(0.5), m.AnimationGroup(m.FadeIn(cb, scale=0.85), m.FadeIn(hand),
                                                                        m.FadeIn(c_src)))),
            m.FadeIn(yrs, shift=0.1 * m.UP),
        )
        self.say(
            "[It solved medium scrambles sixty percent of the time, and the hardest twenty percent]; "
            "[the moves came from a classic solver, and the learning was in the fingers.]",
            m.AnimationGroup(m.FadeIn(r15, shift=0.1 * m.UP), m.Succession(m.Wait(1.2), m.FadeIn(r26, shift=0.1 * m.UP))),
            m.FadeIn(solver, shift=0.1 * m.UP),
        )
        self.say(
            "[And randomizing isn't free]: "
            "[OpenAI's earlier hand needed about a hundred years of simulated practice with it, against three without.]",
            m.FadeIn(d_head, shift=0.1 * m.UP),
            m.AnimationGroup(*[m.AnimationGroup(m.FadeIn(r[0]), m.GrowFromEdge(r[1], m.LEFT), m.FadeIn(r[2])) for r in d_rows],
                             m.FadeIn(d_src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · recap

    def recap(self):
        self.next_section("7 · Recap", notes=(
            "Spread (a Gaussian policy samples torques; its spread is exploration, and SAC pays a bonus for it), slope (a "
            "deterministic actor follows the critic's slope: DPG, DDPG, TD3), randomize (vary the simulator so the real "
            f"world is one more variation). Pendulum, measured here: SAC {H_SAC['final_return_mean']} ± "
            f"{H_SAC['final_return_std']} after {SAC_STEPS:,} steps; PPO {PPO_RET} ± {PPO_SD} after {PPO_STEPS:,} ({SEEDS} "
            "seeds). This closes Module 5; Episode 20 (Module 6): the model is the policy."))
        halo = glow(EXPLORE, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Spread", 58, EXPLORE), title_text("Slope", 58, SIGNAL),
                         title_text("Randomize", 58, STATE))
        words.arrange(m.RIGHT, buff=1.1).move_to([0, 1.05, 0])
        on_baseline(words)
        subs = m.VGroup(text("a bell curve explores", 24, MUTED), text("follow the critic uphill", 24, MUTED),
                        text("noise in the simulator", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), words[0].get_bottom()[1] - 0.5, 0])
        on_baseline(subs)
        assert all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        proof = colored([("Pendulum, this Mac:", MUTED), (f"SAC {SAC_STEPS // 1000}k steps,", SAC_C),
                         (minus(f"{H_SAC['final_return_mean']:.1f}"), SAC_C), ("·", MUTED),
                         (f"PPO {PPO_STEPS / 1000:.0f}k steps,", PPO_C), (minus(f"{PPO_RET:.0f}"), PPO_C)], 24)
        proof.move_to([0, -1.25, 0])
        bot = rover(0.6).move_to([-3.3, -2.6, 0])
        nxt = pill("episode 20 · the model is the policy", PINK, PINK_FILL, 22).next_to(bot, m.RIGHT, buff=0.45)

        self.say(
            "[To sum up]: [a bell curve's spread explores]; [an actor can follow its critic uphill]; "
            "[and a randomized simulator reaches the real world.]",
            m.AnimationGroup(self.new_header("07 · recap", "Spread, slope, randomize"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[On this Mac, soft actor-critic swung the pendulum up in twenty thousand steps.]",
            m.FadeIn(proof, shift=0.1 * m.UP),
        )
        self.say(
            "[That closes module five.] [Next, the policy becomes a language model, and its actions are tokens.]",
            m.FadeIn(bot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()
