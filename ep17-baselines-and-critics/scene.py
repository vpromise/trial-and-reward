"""Trial and Reward · Episode 17 · Baselines and Critics (Module 5 · Learning the Policy)

Episode 16 ended on noise: REINFORCE pushes up every move in proportion to its return, and on CartPole every return
is positive. This episode subtracts a baseline. The running example stays CartPole: the very same eight episodes of
the untrained policy that Episode 16 drew (reset seeds 3000–3007, actions from default_rng(16), re-simulated here),
whose returns 45, 51, 33, 39, 12, 15, 11, 9 all push up; minus the average of 100 such episodes (21.9 ≈ 22) they
become advantages scattered around zero, and the bad episodes push their moves down. Why it adds no bias (Sutton &
Barto §13.4, eq. 13.10), shown on Episode 16's step-33 state (the trained policy pushes left 94%): the two moves'
pushes on left's score, weighted by their chances, cancel exactly. The natural baseline is V(s): A = G − V(s). An
actor (CORAL Rover) and a critic (BLUE Rover) trading signals: the critic's TD error δ = r + γV(s′) − V(s) (Episode 9)
is a one-step advantage (PINK). GAE (Schulman et al., 2016) on the notes' worked example (the last three steps before
the pole falls, V = 10, 9, 8, computed with rl_reference.gae). A3C/A2C in one line, our A2C's eight CartPoles (the
seed-10 A2C policy, rolled out here for 16 steps each). Measured on this Mac (deep_rl.json): the gradient-variance
study (one frozen policy, 300 episodes, three estimators: 45× and 101× at the 'learning' snapshot; 2.7×/5.9×
untrained, 22×/121× good) and the learning curves of all three learners (5 seeds each, thin per-seed curves behind
the mean). Honest: A2C needed more steps here than REINFORCE + baseline, but finished steadiest and ran fastest per
step (it was NOT fastest to solve: 1.9 s vs 1.1 s); CartPole is easy.

Every number on screen is computed below from rl_reference.py and asserted, or loaded with measured(...), or cited.

Sources (research/notes.md §16, §17, Appendix C row 17; facts dated October 2026):
- Williams (1992), Machine Learning 8:229–256: Δw = α (r − b) e — the baseline is in REINFORCE's first paper.
- Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), §13.4: eq. 13.10 (Σ_a b(s) ∇π(a|s) =
  b(s) ∇ Σ_a π(a|s) = b(s) ∇1 = 0), eq. 13.11 (REINFORCE with baseline); §13.5 eqs. 13.12–13.14 (one-step
  actor–critic with the TD error δ). "In general, the baseline leaves the expected value of the update unchanged,
  but it can have a large effect on its variance."
- Schulman, Moritz, Levine, Jordan & Abbeel (2016), "High-Dimensional Continuous Control Using Generalized Advantage
  Estimation", ICLR 2016 (all at UC Berkeley then): Â_t = Σ_l (γλ)^l δ_t+l; λ = 0 one-step (low variance, biased if V
  is wrong), λ = 1 return minus V (unbiased, high variance). λ = 0.95 with γ = 0.99: PPO's defaults (Schulman et al.
  2017, Table 3) and OpenAI Five (Berner et al. 2019).
- Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver & Kavukcuoglu (2016), "Asynchronous Methods for Deep
  Reinforcement Learning", ICML 2016 (DeepMind): A3C, 16 actor-learner threads on one machine, no GPU, asynchronous
  updates. OpenAI blog "OpenAI Baselines: ACKTR & A2C" (18 Aug 2017): "A2C is a synchronous, deterministic variant of
  Asynchronous Advantage Actor Critic (A3C) which we've found gives equal performance."
- research/experiments/deep_rl.md + deep_rl.json (this Mac Studio, M4 Max; seeds 10–14, mean ± sample std):
  headline.reinforce / reinforce_baseline / a2c_gae, methods.*.hp and per_seed, curves.*.avg100_per_seed, gradvar.

Illustrative (said "for example" or labelled): the critic's guesses V = 10, 9, 8 in the GAE section (notes §17
worked example); the cart's small push in the actor–critic section; "push right" as the actor's move there.

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

PREFIX = "ep17"
DEEP = "deep_rl"
LEFT, RIGHT = 0, 1
PLAIN_C, BASE_C, GAE_C = REWARD, VALUE, SIGNAL     # no baseline (raw returns) · a learned baseline V(s) · GAE advantages

# ---------------------------------------------------------------------------- the numbers (all computed here)


def probs(policy, s):
    """A policy's two move probabilities in state s (gymnasium hands the agent float32)."""
    with np.errstate(all="ignore"):                 # Accelerate's matmul raises spurious warnings
        return softmax(policy(cartpole_obs(s)))


def play(policy, reset_seed, rng):
    """One sampled episode: (undiscounted return, first move) — Episode 16's simulation, unchanged."""
    s = cartpole_reset(np.random.default_rng(reset_seed))
    G, first = 0.0, None
    for _ in range(CARTPOLE["max_steps"]):
        a = int(rng.choice(2, p=probs(policy, s)))
        first = a if first is None else first
        s, r, done = cartpole_step(s, a)
        G += r
        if done:
            break
    return G, first


# --- sections 1–2: Episode 16's eight untrained episodes, and the baseline (the average of 100)
PI0 = load_policy("random_cartpole")
_rng = np.random.default_rng(16)
RUNS = [play(PI0, 3000 + k, _rng) for k in range(100)]
RETS = np.array([g for g, _ in RUNS])
B = float(RETS.mean())
G8 = [g for g, _ in RUNS[:8]]
A8 = [g - B for g in G8]
assert G8 == [45, 51, 33, 39, 12, 15, 11, 9]                   # the same eight as Episode 16's section 7
assert round(B) == 22 and abs(B - measured("trajectories", "sources", "random_cartpole", "eval", "mean_return")) < 1
assert min(G8) > 0 and sum(a > 0 for a in A8) == 4 and sum(a < 0 for a in A8) == 4   # "scattered around zero"
assert [f"{a:+.0f}" for a in A8] == ["+23", "+29", "+11", "+17", "-10", "-7", "-11", "-13"]
RF = measured(DEEP, "headline", "reinforce")
assert RF["solved_seeds"] == "3/5"                              # "only three of five runs solved it"

# --- section 3: why the baseline adds no bias, on Episode 16's step-33 state
PI = load_policy("reinforce_cartpole")
TRAJ = measured("trajectories", "policies", "reinforce_cartpole")
STATES16 = np.array(TRAJ["states"], dtype=float)          # section 1 replays its first 150 steps (as Episode 16 did)
REPLAY_N = 150
assert TRAJ["length"] == 500 and len(STATES16) > REPLAY_N
S_EX = np.array(TRAJ["states"][33], dtype=float)
P_EX = probs(PI, S_EX)
PL, PR = float(P_EX[LEFT]), float(P_EX[RIGHT])
PUSH_L, PUSH_R = 1.0 - PL, -PL                                  # ∂ log π(move) / ∂ (left's score) = one-hot − π
W_L, W_R = PL * PUSH_L, PR * PUSH_R                              # weighted by how often each move happens
assert round(PL, 2) == 0.94 and round(PR, 2) == 0.06 and round(math.degrees(S_EX[2]), 1) == -4.5
assert f"{PUSH_L:+.2f}" == "+0.06" and f"{PUSH_R:+.2f}" == "-0.94"
assert f"{W_L:+.2f}" == "+0.06" and f"{W_R:+.2f}" == "-0.06" and abs(W_L + W_R) < 1e-12   # "they cancel exactly"

# --- section 6: GAE on the notes' worked example (the last three steps before the pole falls; V illustrative)
GAMMA, LAM = 0.99, 0.95
R3, V3 = [1.0, 1.0, 1.0], [10.0, 9.0, 8.0]                      # V(s3) = 0: the pole has fallen (terminal)
DELTA = gae(R3, V3, gamma=GAMMA, lam=0.0)                      # λ = 0: the one-step TD errors
A095 = gae(R3, V3, gamma=GAMMA, lam=LAM)
A1 = gae(R3, V3, gamma=GAMMA, lam=1.0)
assert [f"{d:.2f}" for d in DELTA] == ["-0.09", "-0.08", "-7.00"]
assert [f"{a:.2f}" for a in A095] == ["-6.36", "-6.66", "-7.00"] and round(A095[0], 3) == -6.357   # notes: −6.357
assert f"{A1[0]:.2f}" == "-7.03" and abs(A1[0] - (sum(GAMMA ** k for k in range(3)) - V3[0])) < 1e-12  # return − V
assert abs(A095[0] - sum((GAMMA * LAM) ** k * DELTA[k] for k in range(3))) < 1e-12   # Σ (γλ)^l δ_t+l
HORIZON = 1 / (1 - GAMMA * LAM)
assert round(HORIZON) == 17 and round(GAMMA * LAM, 2) == 0.94                       # "about seventeen steps deep"

# --- section 7: our A2C — eight CartPoles, 16 steps each per update (the seed-10 A2C policy, rolled out here)
HP_AC = measured(DEEP, "methods", "a2c_gae", "hp")
N_ENVS, N_STEPS = HP_AC["n_envs"], HP_AC["n_steps"]
assert N_ENVS == 8 and N_STEPS == 16 and HP_AC["gae_lambda"] == LAM and HP_AC["gamma"] == GAMMA


def run_params(name):
    """A network saved by the training runs (weights/runs/<name>.npz, layout x @ W + b)."""
    with np.load(WEIGHTS / "runs" / f"{name}.npz") as z:
        n = sum(k.startswith("W") for k in z.files)
        return [(z[f"W{i}"].astype(np.float64), z[f"b{i}"].astype(np.float64)) for i in range(n)]


AC_PI = run_params("pg_a2c_gae_s10_policy")
assert [W.shape for W, _ in AC_PI] == [(4, 64), (64, 64), (64, 2)]
_rng7 = np.random.default_rng(17)
ROLLOUTS = []
for k in range(N_ENVS):
    s = cartpole_reset(np.random.default_rng(2026 + k))
    rows = [s]
    for _ in range(N_STEPS):
        with np.errstate(all="ignore"):
            p = softmax(mlp_forward(AC_PI, cartpole_obs(s)))
        s, _, done = cartpole_step(s, int(_rng7.choice(2, p=p)))
        rows.append(s)
        assert not done
    ROLLOUTS.append(np.array(rows))

# --- section 8: how much noise a baseline removes (deep_rl.json gradvar; same frozen policy, same 300 episodes)
GV = measured(DEEP, "gradvar")
STAGES = {s["policy"]: s for s in GV["stages"]}
HG = measured(DEEP, "headline", "gradvar")
LRN = STAGES["learning"]
EST = LRN["estimators"]
assert LRN["episodes"] == 300 == HG["estimates_per_estimator"] and HG["policy_stage"] == "learning"
assert LRN["snapshot_episode"] == 400 and round(LRN["training_avg100_at_snapshot"]) == 177 and GV["run"] == "pg_reinforce_s10"
assert round(HG["variance_ratio_no_baseline_over_baseline"]) == 45 and round(HG["variance_ratio_no_baseline_over_gae"]) == 101
assert LRN["n_params"] == 4610 and round(EST["baseline"]["cosine_with_no_baseline_mean"], 2) == 0.97
RATIOS = {k: (STAGES[k]["variance_ratio_no_baseline_over_baseline"], STAGES[k]["variance_ratio_no_baseline_over_gae"])
          for k in ("untrained", "learning", "good")}
assert [f"{a:.1f}" for a in RATIOS["untrained"]] == ["2.7", "5.9"]               # "under three-fold"
assert [round(a) for a in RATIOS["learning"]] == [45, 101] and [round(a) for a in RATIOS["good"]] == [22, 121]
LENS = {k: STAGES[k]["mean_episode_length"] for k in RATIOS}
assert [round(LENS[k]) for k in RATIOS] == [23, 163, 368]
WHISK = {k: (EST[k]["mean_bias_push_right"], math.sqrt(EST[k]["var_bias_push_right"])) for k in ("no_baseline", "baseline", "gae")}
assert [round(WHISK[k][1]) for k in WHISK] == [344, 50, 32] and all(40 < WHISK[k][0] < 50 for k in WHISK)

# --- section 9: the three learners on this Mac (5 seeds × 500,000 steps each)
HB = measured(DEEP, "headline", "reinforce_baseline")
HA = measured(DEEP, "headline", "a2c_gae")
CURVES = {k: measured(DEEP, "curves", k) for k in ("reinforce", "reinforce_baseline", "a2c_gae")}
X_K = np.array(CURVES["reinforce"]["x_steps"]) / 1000
assert X_K[-1] == 500 and all(np.array_equal(np.array(CURVES[k]["x_steps"]) / 1000, X_K) for k in CURVES)
assert round(RF["steps_to_solve_mean"] / 1000) == 260 and round(RF["steps_to_solve_std"] / 1000) == 53
assert HB["solved_seeds"] == "5/5" and round(HB["steps_to_solve_mean"] / 1000) == 83 and round(HB["steps_to_solve_std"] / 1000) == 20
assert round(HB["final_avg100_mean"]) == 498 and round(HB["final_avg100_std"]) == 3
assert HA["solved_seeds"] == "5/5" and round(HA["steps_to_solve_mean"] / 1000) == 213 and round(HA["steps_to_solve_std"] / 1000) == 53
assert HA["final_avg100_mean"] == 500 and HA["final_avg100_std"] == 0                          # "every seed ended at 500"
assert all(c[-1] == 500 for c in CURVES["a2c_gae"]["avg100_per_seed"])
assert 0.28 < HB["steps_to_solve_mean"] / RF["steps_to_solve_mean"] < 0.36                      # "a third as many"
WALL = {k: measured(DEEP, "headline", k, "wall_s_total_mean") for k in ("reinforce", "reinforce_baseline", "a2c_gae")}
assert WALL["a2c_gae"] < min(WALL["reinforce"], WALL["reinforce_baseline"]) and WALL["a2c_gae"] == 4.5   # fastest per step
assert HA["wall_s_to_solve_mean"] > HB["wall_s_to_solve_mean"]                                  # …but not fastest to solve
THRESH = 475

# ---------------------------------------------------------------------------- look


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


def fit(mob, width):
    if mob.width > width:
        mob.scale(width / mob.width)
    return mob


def prob_bars(values, center, width=2.4, height=2.0, size=22):
    return Bars(list(values), labels=["left", "right"], top=1.0, bottom=0.0, width=width, height=height,
                color=ACTION, gap=0.35, fmt="{:.2f}", size=size, center=center)


def up_arrow(x, y0, y1, color=SIGNAL, width=6):
    return m.Arrow([x, y0, 0], [x, y1, 0], buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.4)


def swatch(color, label, size=20):
    return m.VGroup(m.Line([0, 0, 0], [0.4, 0, 0]).set_stroke(color, 5), code(label, size, color, weight=m.BOLD)).arrange(m.RIGHT, buff=0.15)


# ---------------------------------------------------------------------------- the episode


class Ep17BaselinesAndCritics(Episode):
    number = 17

    def lesson(self):
        self.noise_and_baseline()
        self.no_bias()
        self.advantage()
        self.actor_critic()
        self.gae_dial()
        self.many_actors()
        self.measured_noise()
        self.on_this_mac()
        self.recap()

    # ------------------------------------------------------------------ 1–2 · every return is positive → subtract a baseline

    def noise_and_baseline(self):
        self.next_section("1 · Every return is positive", notes=(
            "Episode 16: REINFORCE (Williams 1992), θ ← θ + α G_t ∇ log π_θ(a_t|s_t). The bars are the same eight "
            "episodes of the untrained policy (weights/random_cartpole) that Episode 16's section 7 drew, re-simulated "
            "here with rl_reference.cartpole_step (reset seeds 3000–3007, actions sampled with default_rng(16)): returns "
            f"{', '.join(f'{g:.0f}' for g in G8)}. CartPole pays +1 per step, so every return is positive and every move "
            "made is pushed up; the arrows show only the direction of the push. Measured (deep_rl.json "
            f"headline.reinforce): solved in {RF['solved_seeds']} seeds."))
        cx, cy, cw, ch = -3.6, -0.45, 5.6, 3.3
        bars = Bars(G8, labels=[str(i + 1) for i in range(8)], top=60, bottom=0, width=cw, height=ch, color=PLAIN_C,
                    gap=0.22, fmt="{:.0f}", size=20, center=(cx, cy, 0))
        adv = Bars(A8, labels=[str(i + 1) for i in range(8)], top=60 - B, bottom=-B, width=cw, height=ch, color=GAE_C,
                   gap=0.22, fmt="{:+.0f}", size=20, center=(cx, cy, 0))
        assert abs(adv.level(0) - bars.level(B)) < 1e-9          # the advantages' zero sits on the baseline
        cap_g = left_at(text("return of each episode", 22, PLAIN_C), -6.45, 2.2)
        cap_a = left_at(text("advantage: return minus baseline", 22, GAE_C), -6.45, 2.2)
        foot = cite("8 untrained episodes, simulated here as in episode 16").move_to([cx, -2.95, 0])
        pushes = m.VGroup(*[up_arrow(x, 1.3, 1.78) for x in bars.xs])
        flips = [i for i, a in enumerate(A8) if a < 0]
        replay = CartPoleView(center=(cx, -0.6, 0), units_per_m=0.8, pole_units=2.0)
        replay.set_state(STATES16[0][0], STATES16[0][2])
        r_mascot = rover(0.5).move_to([cx - 2.1, 1.35, 0])
        r_lab = cite("episode 16's final network, one episode").move_to([cx, -1.35, 0])
        r_group = m.VGroup(replay, r_mascot, r_lab)

        rx = 1.15
        f1 = formula([r"\theta", r"\leftarrow", r"\theta", "+", r"\alpha", r"\,G_t", r"\,\nabla_\theta",
                      r"\log \pi_\theta(a_t \mid s_t)"], [ACTION, INK, ACTION, INK, INK, PLAIN_C, SIGNAL, ACTION], 40)
        left_at(fit(f1, 5.4), rx, 1.85)
        f1_cite = left_at(cite("REINFORCE, from episode 16"), rx, 1.25)
        f2 = formula([r"\theta", r"\leftarrow", r"\theta", "+", r"\alpha", r"\,\big(G_t", "-", r"b(s_t)\big)", r"\,\nabla_\theta",
                      r"\log \pi_\theta(a_t \mid s_t)"], [ACTION, INK, ACTION, INK, INK, PLAIN_C, INK, BASE_C, SIGNAL, ACTION], 40)
        left_at(fit(f2, 5.4), rx, 1.85)
        f2_cite = left_at(cite("with a baseline · Williams, 1992"), rx, 1.25)
        solved = left_at(code(f"solved in {RF['n_solved']} of 5 runs", 24, INK, weight=m.BOLD), rx, 0.35)
        noisy = pill("noisy", RED, RED_FILL, 22).next_to(solved, m.RIGHT, buff=0.3)

        y_b = bars.level(B)
        b_line = m.DashedLine([cx - cw / 2 - 0.15, y_b, 0], [cx + cw / 2 + 0.15, y_b, 0], dash_length=0.12)
        b_line.set_stroke(BASE_C, 3.5).set_z_index(2)
        b_lab = left_at(code(f"b ≈ {B:.0f}", 22, BASE_C, weight=m.BOLD), cx + cw / 2 + 0.25, y_b)
        b_txt = m.VGroup(text("baseline: the average return", 24, BASE_C, weight=m.BOLD),
                         cite(f"of 100 untrained episodes: {B:.1f}"))
        b_txt.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(b_txt, rx, 0.3)
        legend = m.VGroup(
            m.VGroup(up_arrow(0, -0.22, 0.22), text("better than typical: push its moves up", 22, INK)).arrange(m.RIGHT, buff=0.2),
            m.VGroup(up_arrow(0, 0.22, -0.22), text("worse than typical: push them down", 22, INK)).arrange(m.RIGHT, buff=0.2),
        ).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
        left_at(legend, rx, -1.35)

        self.say(
            "[In episode sixteen, REINFORCE learned CartPole]: [after each episode, it pushed up every move it made], "
            "[in proportion to the return.]",
            m.AnimationGroup(self.new_header("01 · the noise", "Every return is positive"), m.FadeIn(f1, shift=0.1 * m.UP),
                             m.FadeIn(r_group, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(f1_cite), m.Circumscribe(f1[6:], color=SIGNAL, buff=0.06),
                             replay.replay(STATES16[:REPLAY_N], 3.2)),
            m.Circumscribe(f1[5], color=PLAIN_C, buff=0.06),
        )
        self.say(
            "[Here are eight episodes of the untrained policy]: [in CartPole, every return is positive.]",
            m.AnimationGroup(m.FadeOut(r_group), m.FadeIn(bars.zero), m.FadeIn(bars.labels), m.FadeIn(cap_g), m.FadeIn(foot),
                             m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(b_, m.DOWN), m.FadeIn(n_))
                                             for b_, n_ in zip(bars.bars, bars.numbers)], lag_ratio=0.2, run_time=2.4)),
            m.Circumscribe(bars.bars, color=PLAIN_C, buff=0.08),
        )
        self.say(
            "[So every move gets pushed up], [even in the short, bad episodes.]",
            m.LaggedStart(*[m.GrowArrow(a_) for a_ in pushes], lag_ratio=0.12, run_time=1.6),
            m.Circumscribe(m.VGroup(*[bars.bars[i] for i in flips]), color=RED, buff=0.08),
        )
        self.say(
            "[It's right on average, but noisy]: [only three of five runs solved it.]",
            m.FadeIn(noisy, scale=0.8),
            m.FadeIn(solved, shift=0.1 * m.UP),
        )

        self.next_section("2 · Subtract a baseline", notes=(
            f"The baseline b = the average return of 100 untrained episodes (reset seeds 3000–3099, default_rng(16)): "
            f"{B:.2f} (the export's own evaluation of this network over the same reset seeds: 22.4). For the first move, "
            "this is what a value network would learn as V(s0), since every start state is nearly the same. Advantages "
            f"G − b: {', '.join(f'{a:+.1f}' for a in A8)}. Each bar now covers exactly the stretch between its return and "
            "the baseline. Williams (1992) already wrote the update as α (r − b) e: the baseline is part of REINFORCE "
            "as published (S&B eq. 13.11 for the per-step form)."))
        self.say(
            "[The fix: compare each return with what's typical], [a baseline.] "
            "[Here, the average of a hundred episodes: about twenty-two.]",
            m.AnimationGroup(self.new_header("02 · baseline", "Subtract what's typical"),
                             m.FadeOut(m.VGroup(solved, noisy), shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.Create(b_line), m.FadeIn(b_lab, shift=0.1 * m.LEFT)),
            m.FadeIn(b_txt, shift=0.1 * m.UP),
        )
        self.say(
            "[Subtract it], [and the returns become advantages], [scattered around zero.]",
            m.AnimationGroup(m.ReplacementTransform(bars, adv), m.FadeOut(b_line), run_time=1.4),
            m.AnimationGroup(m.FadeOut(cap_g, shift=0.15 * m.UP), m.FadeIn(cap_a, shift=0.15 * m.UP)),
            m.Circumscribe(adv.zero, color=BASE_C, buff=0.08),
        )
        self.say(
            "[Now the bad episodes push their moves down], [and the good ones still push theirs up.]",
            m.AnimationGroup(*[pushes[i].animate.rotate(math.pi) for i in flips], m.FadeIn(legend[1], shift=0.1 * m.UP),
                             m.Circumscribe(m.VGroup(*[adv.bars[i] for i in flips]), color=GAE_C, buff=0.08)),
            m.AnimationGroup(m.FadeIn(legend[0], shift=0.1 * m.UP),
                             m.Circumscribe(m.VGroup(*[adv.bars[i] for i in range(8) if i not in flips]), color=GAE_C, buff=0.08)),
        )
        self.say(
            "[In the update, the return becomes the return minus the baseline.] [Williams already wrote it that way, in nineteen ninety-two.]",
            m.AnimationGroup(m.FadeOut(f1, shift=0.15 * m.UP), m.FadeOut(f1_cite), m.FadeIn(f2, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(f2_cite), m.Circumscribe(f2[5:8], color=GAE_C, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · no bias

    def no_bias(self):
        self.next_section("3 · No bias: it cancels", notes=(
            "Sutton & Barto 2018 §13.4, eq. 13.10: Σ_a b(s) ∇π(a|s, θ) = b(s) ∇ Σ_a π(a|s, θ) = b(s) ∇1 = 0, so "
            "E[b(s) ∇ log π(a|s)] = 0 for any b that does not depend on the action. Shown on Episode 16's example state "
            "(step 33 of the reinforce_cartpole replay, trajectories.json; pole −4.5°, falling left): the trained network "
            f"(load_policy + softmax) gives push left {PL:.4f}, push right {PR:.4f}. For a softmax with one score per move "
            "the push on left's score is one-hot − π: pushing left gives 1 − π(left) = "
            f"{PUSH_L:+.4f}, pushing right gives −π(left) = {PUSH_R:+.4f}. Weighted by their chances: {W_L:+.5f} and "
            f"{W_R:+.5f}, which sum to exactly 0 (π_L π_R − π_R π_L). A baseline that depends on the action would bias the "
            "gradient (notes §17, misconceptions)."))
        view = CartPoleView(center=(-5.3, 0.05, 0), units_per_m=0.45, pole_units=1.6)
        view.set_state(S_EX[0], S_EX[2])
        v_lab = text("the state from episode 16", 22, STATE).move_to([-5.3, -0.65, 0])
        pb = prob_bars(P_EX, center=(-1.75, 0.4, 0), width=2.3, height=2.0)
        pb_lab = text("the policy here", 22, ACTION).move_to([-1.75, 1.95, 0])

        cols = [0.9, 2.25, 3.6, 5.3]
        heads = m.VGroup(*[code(t_, 22, c_).move_to([x, 1.6, 0]) for t_, c_, x in
                           zip(("move", "chance", "push", "× chance"), (MUTED, MUTED, SIGNAL, MUTED), cols)])
        push_cap = cite("push on left's score: move made − probability").move_to([3.3, 2.15, 0])

        def row(y, move, chance, push, prod):
            return m.VGroup(code(move, 24, ACTION, weight=m.BOLD).move_to([cols[0], y, 0]),
                            code(chance, 24, INK).move_to([cols[1], y, 0]),
                            code(minus(push), 24, SIGNAL, weight=m.BOLD).move_to([cols[2], y, 0]),
                            code(minus(prod), 24, INK).move_to([cols[3], y, 0]))

        r_left = row(0.95, "left", f"{PL:.2f}", f"{PUSH_L:+.2f}", f"{W_L:+.2f}")
        r_right = row(0.3, "right", f"{PR:.2f}", f"{PUSH_R:+.2f}", f"{W_R:+.2f}")
        rule = m.Line([4.3, -0.12, 0], [6.3, -0.12, 0]).set_stroke(EDGE, 2.5)
        total = m.VGroup(code("average", 20, MUTED), title_text("0", 44, GREEN)).arrange(m.RIGHT, buff=0.3)
        right_at(total, cols[3] + 0.45, -0.6)

        ident = formula([r"\sum_a \pi(a \mid s)\; b\; \nabla_\theta \log \pi(a \mid s)", "=",
                         r"b\, \nabla_\theta \sum_a \pi(a \mid s)", "=", r"b\, \nabla_\theta\, 1", "=", "0"],
                        [INK, INK, INK, INK, INK, INK, GREEN], 38)
        fit(ident.move_to([0, -1.75, 0]), 12.0)
        id_cite = cite("Sutton & Barto, 2018, §13.4").next_to(ident, m.DOWN, buff=0.18)
        ok = pill("no bias, as long as b ignores the move", GREEN, GREEN_FILL, 22).move_to([0, -3.1, 0])

        self.say(
            "[Doesn't that change the gradient? Not on average.] [Take the state from episode sixteen]: "
            "[the policy pushes left ninety-four percent of the time.]",
            self.new_header("03 · no bias", "On average, it cancels"),
            m.AnimationGroup(m.FadeIn(view, shift=0.15 * m.UP), m.FadeIn(v_lab)),
            m.AnimationGroup(m.FadeIn(pb, shift=0.15 * m.UP), m.FadeIn(pb_lab)),
        )
        self.say(
            "[Pushing left is common], [so it nudges left's score only a little]: [plus zero point zero six.]",
            m.AnimationGroup(m.FadeIn(heads), m.FadeIn(push_cap), m.FadeIn(r_left[:2], shift=0.1 * m.UP)),
            m.FadeIn(r_left[2], shift=0.1 * m.UP),
            m.Circumscribe(r_left[2], color=SIGNAL, buff=0.08),
        )
        self.say(
            "[Pushing right is rare, so it nudges left's score a lot]: [minus zero point nine four.]",
            m.FadeIn(r_right[:3], shift=0.1 * m.UP),
            m.Circumscribe(r_right[2], color=SIGNAL, buff=0.08),
        )
        self.say(
            "[Weigh each by how often it happens], [and they cancel exactly.]",
            m.LaggedStart(m.FadeIn(r_left[3], shift=0.1 * m.LEFT), m.FadeIn(r_right[3], shift=0.1 * m.LEFT), lag_ratio=0.4),
            m.AnimationGroup(m.Create(rule), m.FadeIn(total, scale=0.8)),
        )
        self.say(
            "[So a baseline's push averages to zero]: [no bias, only less noise], "
            "[as long as the baseline ignores which move was made.]",
            m.AnimationGroup(m.Write(ident), m.FadeIn(id_cite)),
            m.Circumscribe(ident[-1], color=GREEN, buff=0.08),
            m.FadeIn(ok, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · advantage

    def advantage(self):
        self.next_section("4 · The advantage", notes=(
            "The natural baseline is the state's value V(s), the expected return from s; then G_t − V(s_t) estimates the "
            "advantage A(s, a) = Q(s, a) − V(s): how much better than usual this action was (notes §17). Positive pushes "
            "the move up, negative pushes it down, zero gives no signal. Colours: returns GOLD (as in Episode 16), values "
            "BLUE, the advantage — the learning signal — PINK."))
        f = formula([r"A_t", "=", r"G_t", "-", r"V(s_t)"], [SIGNAL, INK, PLAIN_C, INK, BASE_C], 76)
        f.move_to([0, 1.0, 0])
        b_v = m.Brace(f[4], m.DOWN, buff=0.15, color=BASE_C)
        t_v = text("the return you'd expect from here", 22, BASE_C).next_to(b_v, m.DOWN, buff=0.1)
        b_g = m.Brace(f[2], m.DOWN, buff=0.15, color=PLAIN_C)
        t_g = text("what happened", 22, PLAIN_C).next_to(b_g, m.DOWN, buff=0.1)
        if t_g.get_right()[0] > t_v.get_left()[0] - 0.3:
            t_g.align_to(b_g, m.RIGHT)
            t_v.align_to(b_v, m.LEFT)
        b_a = m.Brace(f[0], m.UP, buff=0.15, color=SIGNAL)
        t_a = text("better than expected?", 22, SIGNAL).next_to(b_a, m.UP, buff=0.1)
        formal = m.VGroup(tex(r"A(s,a) = Q(s,a) - V(s)", 32, MUTED), cite("the advantage of a move"))
        formal.arrange(m.RIGHT, buff=0.3).move_to([0, -0.85, 0])

        cases = [("A > 0", "push the move up", GREEN, 1), ("A < 0", "push the move down", RED, -1),
                 ("A = 0", "nothing to learn", MUTED, 0)]
        cards = m.VGroup()
        for x, (lab, desc, col, d) in zip((-4.4, 0.0, 4.4), cases):
            box = card(3.9, 1.35, col, CARD, radius=0.18, stroke=2.5).move_to([x, -2.45, 0])
            sym = (up_arrow(0, -0.3 * d, 0.3 * d, color=col, width=7) if d else
                   m.Line([-0.25, 0, 0], [0.25, 0, 0]).set_stroke(col, 7))
            words = m.VGroup(code(lab, 26, SIGNAL, weight=m.BOLD), text(desc, 22, INK)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            body = m.VGroup(sym, words).arrange(m.RIGHT, buff=0.35).move_to(box)
            cards.add(m.VGroup(box, body))

        self.say(
            "[The best baseline depends on the state]: [the value of the state, the return you'd expect from here.]",
            m.AnimationGroup(self.new_header("04 · advantage", "Better than expected?"), m.FadeIn(f[2:], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromCenter(b_v), m.FadeIn(t_v), m.Circumscribe(f[4], color=BASE_C, buff=0.08)),
        )
        self.say(
            "[Return minus value is the advantage]: [how much better than expected this move turned out.]",
            m.AnimationGroup(m.FadeIn(f[:2], shift=0.1 * m.RIGHT), m.GrowFromCenter(b_g), m.FadeIn(t_g)),
            m.AnimationGroup(m.GrowFromCenter(b_a), m.FadeIn(t_a), m.FadeIn(formal, shift=0.1 * m.UP)),
        )
        self.say(
            "[Positive, push the move up.] [Negative, push it down.] [Zero, nothing to learn.]",
            *[m.FadeIn(c_, shift=0.15 * m.UP) for c_ in cards],
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · actor and critic

    def actor_critic(self):
        self.next_section("5 · An actor and a critic", notes=(
            "One-step actor–critic (Sutton & Barto §13.5, eqs. 13.12–13.14): the critic learns V(s) by TD (Episode 9); "
            "its TD error δ = r + γ V(s′) − V(s) is itself an estimate of the advantage of the move just made. The actor "
            "updates θ ← θ + α δ ∇ log π(a|s); the critic w ← w + α_w δ ∇V(s). The actor no longer waits for the episode "
            "to end, and the variance falls; the price is bias whenever the critic is wrong (notes §17). CORAL actor, BLUE "
            "critic, PINK signal. The push and the cart's motion are illustrative."))
        actor = rover(0.85, CORAL, CORAL_FILL).move_to([-5.1, 0.95, 0])
        a_lab = m.VGroup(text("actor", 26, CORAL, weight=m.BOLD), tex(r"\pi_\theta(a \mid s)", 30, CORAL)).arrange(m.DOWN, buff=0.1)
        a_lab.next_to(actor, m.DOWN, buff=0.25)
        critic = rover(0.85, BLUE, BLUE_FILL).move_to([5.1, 0.95, 0])
        c_lab = m.VGroup(text("critic", 26, BLUE, weight=m.BOLD), tex(r"V(s)", 30, BLUE)).arrange(m.DOWN, buff=0.1)
        c_lab.next_to(critic, m.DOWN, buff=0.25)
        view = CartPoleView(center=(0, 0.45, 0), units_per_m=0.45, pole_units=1.35)
        view.set_state(0.0, 0.06)

        def link(x0, x1, y, color, label, above=True):
            arr = m.Arrow([x0, y, 0], [x1, y, 0], buff=0, color=color, stroke_width=6, max_tip_length_to_length_ratio=0.12)
            words = code(label, 22, color, weight=m.BOLD).next_to(arr, m.UP if above else m.DOWN, buff=0.12)
            return m.VGroup(arr, words)

        act = link(-4.25, -1.75, 1.3, ACTION, "push right")
        new_s = link(1.75, 4.25, 1.3, STATE, "new state s′")
        rew = link(1.75, 4.25, 0.6, REWARD, "reward +1", above=False)

        td = formula([r"\delta", "=", "r", "+", r"\gamma\, V(s')", "-", r"V(s)"], [SIGNAL, INK, REWARD, INK, BASE_C, INK, BASE_C], 52)
        td.move_to([0, -1.25, 0])
        td_lab = cite("the TD error, from episode 9").next_to(td, m.UP, buff=0.18)
        y_loop = -2.5
        path = m.VMobject().set_points_as_corners([np.array([5.1, -1.25, 0]), np.array([5.1, y_loop, 0]),
                                                   np.array([-5.1, y_loop, 0]), np.array([-5.1, -1.3, 0])])
        path.set_stroke(SIGNAL, 6)
        tip = m.Arrow([-5.1, -1.6, 0], [-5.1, -1.15, 0], buff=0, color=SIGNAL, stroke_width=6, max_tip_length_to_length_ratio=0.6)
        sig = m.VGroup(path, tip)
        sig_lab = code("δ: a one-step advantage", 22, SIGNAL, weight=m.BOLD).move_to([0, y_loop + 0.33, 0])
        sig_lab.set_z_index(2)
        back = m.Arrow(td.get_right() + [0.15, 0.1, 0], [4.45, -0.55, 0], buff=0, color=SIGNAL, stroke_width=5,
                       max_tip_length_to_length_ratio=0.25)
        pros = m.VGroup(pill("no waiting for the end · less noise", GREEN, GREEN_FILL, 22),
                        pill("biased when the critic is wrong", RED, RED_FILL, 22)).arrange(m.RIGHT, buff=0.5)
        pros.move_to([0, -3.2, 0])

        def push(_, alpha):
            view.set_state(0.22 * alpha, 0.06 - 0.05 * alpha)

        self.say(
            "[Now let a second network learn that value]: [the critic.] [The policy becomes the actor.]",
            m.AnimationGroup(self.new_header("05 · actor-critic", "An actor and a critic"), m.FadeIn(view, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(critic, shift=0.15 * m.UP), m.FadeIn(c_lab)),
            m.AnimationGroup(m.FadeIn(actor, shift=0.15 * m.UP), m.FadeIn(a_lab)),
        )
        self.say(
            "[The actor pushes the cart.] [The world answers with a reward and a new state.]",
            m.AnimationGroup(m.GrowArrow(act[0]), m.FadeIn(act[1]), m.UpdateFromAlphaFunc(view, push, run_time=1.2)),
            m.AnimationGroup(m.GrowArrow(new_s[0]), m.FadeIn(new_s[1]), m.GrowArrow(rew[0]), m.FadeIn(rew[1])),
        )
        self.say(
            "[The critic compares the reward plus its guess for the new state], [with its guess for the old one.]",
            m.FadeIn(td[:5], shift=0.1 * m.UP),
            m.FadeIn(td[5:], shift=0.1 * m.UP),
        )
        self.say(
            "[That difference is the TD error from episode nine]: [a one-step advantage.]",
            m.AnimationGroup(m.FadeIn(td_lab), m.Circumscribe(td[0], color=SIGNAL, buff=0.08)),
            m.Flash(td[0].get_center(), color=SIGNAL, flash_radius=0.45, line_length=0.25),
        )
        self.say(
            "[It flows to the actor as the learning signal], [and back to the critic, to sharpen its guesses.]",
            m.AnimationGroup(m.Create(path, run_time=1.4), m.FadeIn(tip), m.FadeIn(sig_lab)),
            m.AnimationGroup(m.GrowArrow(back), m.Indicate(critic, color=BLUE, scale_factor=1.08)),
        )
        self.say(
            "[No waiting for the episode to end, and less noise.] [The price: when the critic is wrong, the signal is biased.]",
            m.FadeIn(pros[0], shift=0.1 * m.UP),
            m.FadeIn(pros[1], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · GAE

    def gae_dial(self):
        self.next_section("6 · Generalised advantage estimation", notes=(
            "Schulman, Moritz, Levine, Jordan & Abbeel (2016), ICLR (UC Berkeley): Â_t = Σ_l (γλ)^l δ_t+l, computed "
            "backwards as Â_t = δ_t + γλ Â_t+1 (rl_reference.gae; the code shown is its loop, verbatim). Worked example "
            "from notes §17 (illustrative critic): the last three steps before the pole falls, rewards (1, 1, 1), the "
            "critic's guesses V(s0..s2) = (10, 9, 8), V(s3) = 0 (terminal), γ = 0.99. TD errors "
            f"{', '.join(f'{d:+.2f}' for d in DELTA)}. Advantage of the first move: λ = 0 → {DELTA[0]:+.2f} (only its own "
            f"step: the fall is invisible); λ = 0.95 → {A095[0]:+.3f} (Â1 = {A095[1]:+.4f}, Â2 = {A095[2]:+.2f}); λ = 1 → "
            f"{A1[0]:+.4f} = (1 + 0.99 + 0.99²) − 10, the Monte Carlo return minus V. Effective horizon 1/(1 − γλ) = "
            f"{HORIZON:.1f} steps. λ = 0.95, γ = 0.99 are PPO's defaults (Table 3) and OpenAI Five's; Schulman et al. found "
            "intermediate λ in [0.9, 0.99] best. GAE(γ, 1) is unbiased (γ-just) whatever V; GAE(γ, 0) only if V is exact."))
        xs = [-5.6, -3.5, -1.4, 0.7]
        y_n = 1.4
        nodes = m.VGroup()
        for i, x in enumerate(xs):
            col, fill = (RED, RED_FILL) if i == 3 else (STATE, TEAL_FILL)
            box = card(0.95, 0.7, col, fill, radius=0.14, stroke=2.5).move_to([x, y_n, 0])
            nodes.add(m.VGroup(box, tex(f"s_{i}", 32, INK).move_to(box)))
        links = m.VGroup(*[arrow(nodes[i].get_right(), nodes[i + 1].get_left(), MUTED, buff=0.08) for i in range(3)])
        chips = m.VGroup(*[reward_chip(1, 20).move_to([(xs[i] + xs[i + 1]) / 2, y_n + 0.5, 0]) for i in range(3)])
        falls = text("pole falls", 22, RED)
        vals = m.VGroup(*[code(f"V = {v:g}", 20, BASE_C).move_to([x, y_n - 0.7, 0]) for x, v in zip(xs, V3 + [0.0])])
        falls.next_to(vals[3], m.DOWN, buff=0.12)
        guess = cite("the critic's guesses, for example").next_to(vals, m.DOWN, buff=0.15).align_to([xs[0] - 0.5, 0, 0], m.LEFT)
        y_d = y_n - 1.75
        deltas = m.VGroup(*[code(minus(f"δ = {d:.2f}"), 20, SIGNAL).move_to([(xs[i] + xs[i + 1]) / 2, y_d, 0])
                            for i, d in enumerate(DELTA)])
        y_a = y_d - 0.6
        hats = [code(minus(f"Â = {a:.2f}"), 20, SIGNAL, weight=m.BOLD).move_to([xs[i], y_a, 0]) for i, a in enumerate(A095)]
        spark = m.Dot(radius=0.12, color=SIGNAL).set_z_index(4).move_to(deltas[2].get_top() + [0, 0.21, 0])

        g_rec = formula([r"\hat A_t", "=", r"\delta_t", "+", r"\gamma\lambda", r"\,\hat A_{t+1}"], [SIGNAL, INK, SIGNAL, INK, INK, SIGNAL], 44)
        g_sum = formula(["=", r"\sum_{l \ge 0}", r"(\gamma\lambda)^l", r"\,\delta_{t+l}"], [INK, INK, INK, SIGNAL], 40)
        gx = 2.05
        left_at(g_rec, gx, 1.85)
        g_sum.next_to(g_rec, m.DOWN, buff=0.3).align_to(g_rec[1], m.LEFT)
        g_cite = m.VGroup(cite("Schulman et al., 2016"), cite("computed with rl_reference.gae")).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        g_cite.next_to(g_sum, m.DOWN, buff=0.25).align_to([gx, 0, 0], m.LEFT)
        blk = m.VGroup(g_rec, g_sum, g_cite)
        gl = code(f"γλ = {GAMMA} × {LAM} ≈ {GAMMA * LAM:.2f}", 22, INK).next_to(g_cite, m.DOWN, buff=0.3).align_to([gx, 0, 0], m.LEFT)

        rows_y = [-1.95, -2.6, -3.25]
        lam_rows = []
        for y, lam, a, desc in zip(rows_y, ("0", "0.95", "1"), (DELTA[0], A095[0], A1[0]),
                                   ("only its own step: the fall is invisible", f"the usual choice: about {HORIZON:.0f} steps deep",
                                    "the whole return minus the value: noisy")):
            lab = left_at(code(f"λ = {lam}", 22, INK, weight=m.BOLD), -6.45, y)
            ln = max(0.62 * abs(a), 0.04)
            bar = m.Rectangle(width=ln, height=0.34).set_stroke(SIGNAL, 2).set_fill(PINK_FILL, 1)
            left_at(bar, -4.7, y)
            val = left_at(code(minus(f"{a:.2f}"), 22, SIGNAL, weight=m.BOLD), -4.7 + ln + 0.15, y)
            words = left_at(text(desc, 22, MUTED), 0.9, y)
            lam_rows.append(m.VGroup(lab, bar, val, words))
        a_cap = left_at(cite("the first move's advantage"), -6.45, -1.45)

        def travel(_, alpha):
            k = alpha * 2
            j = min(int(k), 1)
            f_ = k - j
            pts = [deltas[2].get_top() + [0, 0.21, 0], deltas[1].get_top() + [0, 0.21, 0], deltas[0].get_top() + [0, 0.21, 0]]
            spark.move_to((1 - f_) * pts[j] + f_ * pts[j + 1])
            spark.scale(0.24 * (GAMMA * LAM) ** k / spark.width)

        self.say(
            "[Between one step and the whole episode, there's a dial]: [generalised advantage estimation], "
            "[from Berkeley, in twenty sixteen.]",
            self.new_header("06 · GAE", "A dial between short and long"),
            m.AnimationGroup(m.FadeIn(g_rec, shift=0.1 * m.LEFT), m.FadeIn(g_sum, shift=0.1 * m.LEFT)),
            m.FadeIn(g_cite),
        )
        self.say(
            "[For example, the last three steps before the pole falls.] [The critic expects ten, then nine, then eight.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(n_, shift=0.1 * m.RIGHT) for n_ in nodes], lag_ratio=0.25),
                             m.FadeIn(links), m.FadeIn(chips), m.FadeIn(falls)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(v_) for v_ in vals], lag_ratio=0.3), m.FadeIn(guess)),
        )
        self.say(
            "[The one-step errors are tiny], [until the fall]: [minus seven.]",
            m.FadeIn(deltas[:2], shift=0.1 * m.DOWN),
            m.FadeIn(deltas[2], shift=0.1 * m.DOWN),
            m.AnimationGroup(m.Circumscribe(deltas[2], color=SIGNAL, buff=0.08), m.Flash(deltas[2].get_center(), color=SIGNAL, flash_radius=0.75, line_length=0.25)),
        )
        self.say(
            "[The estimator adds them up backwards], [shrinking by gamma times lambda at each step.]",
            m.Succession(m.FadeIn(spark, scale=0.5, run_time=0.3), m.FadeIn(hats[2], run_time=0.4),
                         m.AnimationGroup(m.UpdateFromAlphaFunc(spark, travel, run_time=2.2, rate_func=m.linear),
                                          m.Succession(m.Wait(1.0), m.FadeIn(hats[1], run_time=0.3), m.Wait(0.8),
                                                       m.FadeIn(hats[0], run_time=0.3)))),
            m.AnimationGroup(m.FadeIn(gl, shift=0.1 * m.UP), m.Circumscribe(g_rec[4], color=SIGNAL, buff=0.06)),
        )
        self.say(
            "[With lambda at zero, the first move sees only its own step]: [the fall is invisible.]",
            m.AnimationGroup(m.FadeOut(spark), m.FadeIn(a_cap), m.FadeIn(lam_rows[0][:3], shift=0.1 * m.RIGHT)),
            m.FadeIn(lam_rows[0][3]),
        )
        self.say(
            "[At one, it's the whole return minus the value]: [unbiased, but noisy.]",
            m.AnimationGroup(m.FadeIn(lam_rows[2][0]), m.GrowFromEdge(lam_rows[2][1], m.LEFT), m.FadeIn(lam_rows[2][2])),
            m.FadeIn(lam_rows[2][3]),
        )
        self.say(
            "[The usual choice is zero point nine five]: [it sees the fall coming], [about seventeen steps deep.]",
            m.AnimationGroup(m.FadeIn(lam_rows[1][0]), m.GrowFromEdge(lam_rows[1][1], m.LEFT), m.FadeIn(lam_rows[1][2])),
            m.Circumscribe(m.VGroup(hats[0], lam_rows[1][2]), color=SIGNAL, buff=0.08),
            m.FadeIn(lam_rows[1][3]),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · many actors: A3C / A2C

    def many_actors(self):
        self.next_section("7 · Many actors at once", notes=(
            "A3C: Mnih et al. (2016), ICML, DeepMind — 16 actor-learner threads on one machine with no GPU, each with "
            "its own copy of the environment, applying n-step advantage gradients (t_max = 5) asynchronously to shared "
            "weights; no replay buffer. A2C: the synchronous version — wait for every actor, then average (OpenAI "
            "Baselines blog, 18 Aug 2017: 'equal performance'; 'we have not seen any evidence that the noise introduced "
            "by asynchrony provides any performance benefit'). Our A2C (deep_rl.json methods.a2c_gae.hp): "
            f"{N_ENVS} CartPoles × {N_STEPS} steps per update, GAE λ = {LAM}, γ = {GAMMA}. The eight carts replay 16 steps "
            "of the seed-10 A2C policy's final weights (weights/runs/pg_a2c_gae_s10_policy.npz), reset seeds 2026–2033, "
            "actions sampled with default_rng(17)."))
        xs, ys = [-5.4, -3.8, -2.2, -0.6], [1.3, -0.75]
        views, strips = [], []
        for k in range(N_ENVS):
            x, y = xs[k % 4], ys[k // 4]
            v = CartPoleView(center=(0, 0, 0), units_per_m=0.45, pole_units=1.4)
            v.scale(0.42).move_to([x, y, 0])
            v.set_state(ROLLOUTS[k][0][0], ROLLOUTS[k][0][2])
            views.append(v)
            strip = m.VGroup(*[m.Square(0.062).set_stroke(EDGE, 1).set_fill(CARD, 1) for _ in range(N_STEPS)])
            strip.arrange(m.RIGHT, buff=0.014).move_to([x, y - 0.55, 0])
            strips.append(strip)
        grid = m.VGroup(*views, *strips)
        cap = cite(f"{N_ENVS} CartPoles × {N_STEPS} steps each").move_to([-3.0, -1.85, 0])
        box = card(3.9, 2.9, SIGNAL, CARD, radius=0.2, stroke=2.5).move_to([4.15, 0.3, 0])
        pair = m.VGroup(rover(0.6, CORAL, CORAL_FILL), rover(0.6, BLUE, BLUE_FILL)).arrange(m.RIGHT, buff=0.5)
        pair.move_to(box).shift(0.4 * m.UP)
        upd = code(f"one update: {N_ENVS * N_STEPS} steps", 22, SIGNAL, weight=m.BOLD).move_to(box).shift(0.75 * m.DOWN)
        into = arrow([0.35, 0.3, 0], [2.1, 0.3, 0], MUTED)
        src = m.VGroup(cite("A3C · Mnih et al., 2016 · 16 CPU threads, updating asynchronously"),
                       cite("A2C · OpenAI, 2017 · synchronous: wait for all, then average")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(src, -6.45, -2.85)

        def roll(_, alpha):
            i = alpha * N_STEPS
            j = min(int(i), N_STEPS - 1)
            f_ = i - j
            for k, v in enumerate(views):
                st = (1 - f_) * ROLLOUTS[k][j] + f_ * ROLLOUTS[k][j + 1]
                v.set_state(st[0], st[2])
                for c_, sq in enumerate(strips[k]):
                    sq.set_fill(TEAL_FILL if c_ < int(i + 1e-9) else CARD, 1).set_stroke(STATE if c_ < int(i + 1e-9) else EDGE, 1)

        live = m.UpdateFromAlphaFunc(grid, roll, run_time=2.6, rate_func=m.linear)

        self.say(
            "[Actor-critics also run many copies of the world at once]: [A3C, from DeepMind in twenty sixteen], "
            "[and its synchronous twin, A2C.]",
            m.AnimationGroup(self.new_header("07 · A2C", "Many actors at once"),
                             m.LaggedStart(*[m.FadeIn(m.VGroup(v, s_), shift=0.1 * m.UP) for v, s_ in zip(views, strips)], lag_ratio=0.1)),
            m.FadeIn(src[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(src[1], shift=0.1 * m.UP), m.FadeIn(box), m.FadeIn(pair, shift=0.1 * m.UP)),
        )
        self.say(
            "[Ours ran eight copies of CartPole side by side], [and updated after every sixteen steps of each.]",
            m.AnimationGroup(live, m.FadeIn(cap)),
            m.AnimationGroup(m.GrowArrow(into), m.FadeIn(upd, shift=0.1 * m.UP), pulse(into, SIGNAL)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · measured: the noise

    def measured_noise(self):
        nb, bl, ga = WHISK["no_baseline"], WHISK["baseline"], WHISK["gae"]
        self.next_section("8 · Measured: the noise", notes=(
            "deep_rl.md / deep_rl.json gradvar (scripts/deep_gradvar.py): one frozen policy — the REINFORCE run of seed 10 "
            f"at episode {LRN['snapshot_episode']} (step {LRN['snapshot_step']:,}; its last 100 training episodes averaged "
            f"{LRN['training_avg100_at_snapshot']}), 'learning' — and {LRN['episodes']} fresh episodes (mean length "
            f"{LRN['mean_episode_length']:.0f}). On the very same episodes, the textbook per-episode estimate "
            "Σ_t A_t ∇ log π(a_t|s_t) three ways: A = G_t; A = G_t − V(s_t) with V fit to the returns of 100 separate "
            f"episodes (R² {LRN['value_fit_r2']:.2f}); A = GAE(λ = 0.95). Variance across episodes, summed over all "
            f"{LRN['n_params']:,} weights: no baseline ÷ baseline = {HG['variance_ratio_no_baseline_over_baseline']:.2f}, "
            f"÷ GAE = {HG['variance_ratio_no_baseline_over_gae']:.2f}. One weight (the push-right bias), mean ± std: "
            f"{nb[0]:.1f} ± {nb[1]:.1f}, {bl[0]:.1f} ± {bl[1]:.1f}, {ga[0]:.1f} ± {ga[1]:.1f}. Cosine between the mean "
            f"gradients (baseline vs none): {EST['baseline']['cosine_with_no_baseline_mean']:.3f}. Signal-to-noise per "
            f"episode {EST['no_baseline']['snr']:.3f} → {EST['baseline']['snr']:.2f} → {EST['gae']['snr']:.2f}. By stage "
            f"(baseline / GAE): untrained {RATIOS['untrained'][0]:.2f} / {RATIOS['untrained'][1]:.2f} (23-step episodes), "
            f"learning {RATIOS['learning'][0]:.1f} / {RATIOS['learning'][1]:.1f}, good {RATIOS['good'][0]:.1f} / "
            f"{RATIOS['good'][1]:.1f} (368 steps; the baseline's fit is weaker there, R² "
            f"{STAGES['good']['value_fit_r2']:.2f}, since MC returns cut at 500 steps depend on time, which the state does "
            "not show)."))
        top = m.VGroup(cite("measured on this Mac · one frozen policy: REINFORCE mid-training (average return 177)"),
                       cite("the same 300 fresh episodes, three ways"))
        top.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(top, -6.45, 2.25)

        x0, x1, v0, v1 = -6.0, 0.6, -350.0, 450.0

        def X(v):
            return x0 + (v - v0) * (x1 - x0) / (v1 - v0)

        y_ax = -1.85
        axis = m.Line([x0, y_ax, 0], [x1, y_ax, 0]).set_stroke(FAINT, 2.5)
        ticks = m.VGroup()
        for t_ in (-300, 0, 300):
            ticks.add(m.Line([X(t_), y_ax, 0], [X(t_), y_ax - 0.09, 0]).set_stroke(FAINT, 2),
                      code(minus(f"{t_:+d}") if t_ else "0", 20, MUTED).move_to([X(t_), y_ax - 0.33, 0]))
        ax_cap = left_at(cite("one weight's gradient (push-right bias): mean ± 1 std"), x0, -2.65)
        rows = []
        for y, (mu, sd), col, name in ((0.95, nb, PLAIN_C, "no baseline"), (-0.1, bl, BASE_C, "learned baseline"),
                                       (-1.15, ga, GAE_C, "GAE")):
            line = m.Line([X(mu - sd), y, 0], [X(mu + sd), y, 0]).set_stroke(col, 7)
            caps = m.VGroup(*[m.Line([X(v), y - 0.17, 0], [X(v), y + 0.17, 0]).set_stroke(col, 5) for v in (mu - sd, mu + sd)])
            dot = m.Dot([X(mu), y, 0], radius=0.11, color=INK).set_z_index(3)
            lab = left_at(text(name, 22, col, weight=m.BOLD), x0, y + 0.42)
            rows.append(m.VGroup(lab, line, caps, dot))
        mean_x = X(float(np.mean([nb[0], bl[0], ga[0]])))
        mean_line = m.DashedLine([mean_x, y_ax, 0], [mean_x, 1.35, 0], dash_length=0.1).set_stroke(MUTED, 2).set_z_index(1)
        mean_lab = code(f"average ≈ {np.mean([nb[0], bl[0], ga[0]]):.0f}", 20, MUTED).move_to([mean_x + 0.9, 1.6, 0])

        rx = 1.75
        s1 = m.VGroup(title_text(f"{HG['variance_ratio_no_baseline_over_baseline']:.0f}×", 64, BASE_C),
                      text("less variance: learned baseline", 22, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(s1, rx, 0.9)
        s2 = m.VGroup(title_text(f"{HG['variance_ratio_no_baseline_over_gae']:.0f}×", 64, GAE_C),
                      text("less variance: GAE", 22, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(s2, rx, -0.75)
        s3 = m.VGroup(cite(f"all {LRN['n_params']:,} weights · same direction:"),
                      cite(f"cosine {EST['baseline']['cosine_with_no_baseline_mean']:.2f} between the averages"))
        s3.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        left_at(s3, rx, -2.0)

        # the stage chart (linear scale)
        base_y, k_h = -2.0, 3.2 / 125
        groups, labels_g = m.VGroup(), m.VGroup()
        for xc, key in zip((-5.0, -2.95, -0.9), ("untrained", "learning", "good")):
            pair = m.VGroup()
            for dx, val, col in ((-0.36, RATIOS[key][0], BASE_C), (0.36, RATIOS[key][1], GAE_C)):
                h = max(val * k_h, 0.02)
                rect = m.Rectangle(width=0.62, height=h).set_stroke(col, 2).set_fill(FILLS[col], 1)
                rect.move_to([xc + dx, base_y + h / 2, 0])
                num = code(f"{val:.1f}×" if val < 10 else f"{val:.0f}×", 20, col).move_to([xc + dx, base_y + h + 0.22, 0])
                pair.add(m.VGroup(rect, num))
            groups.add(pair)
            labels_g.add(m.VGroup(code(key, 20, INK, weight=m.BOLD).move_to([xc, base_y - 0.3, 0]),
                                  cite(f"≈ {LENS[key]:.0f} steps").move_to([xc, base_y - 0.65, 0])))
        g_axis = m.Line([-6.2, base_y, 0], [0.4, base_y, 0]).set_stroke(FAINT, 2.5)
        g_leg = m.VGroup(swatch(BASE_C, "baseline"), swatch(GAE_C, "GAE")).arrange(m.RIGHT, buff=0.5)
        g_leg.move_to([-2.9, -3.2, 0])
        g_cap = left_at(cite("variance cut, by stage of training"), -6.45, 1.45)

        self.say(
            "[How much noise do they remove? We measured it on this Mac]: [one frozen policy, mid-training], "
            "[the same three hundred episodes, three ways.]",
            m.AnimationGroup(self.new_header("08 · measured", "Same policy, less noise"), m.FadeIn(top[0], shift=0.1 * m.DOWN)),
            m.Circumscribe(top[0], color=MUTED, buff=0.06),
            m.FadeIn(top[1], shift=0.1 * m.DOWN),
        )
        self.say(
            "[Each episode gives one gradient estimate.] [Without a baseline, one weight's estimates swing wildly.]",
            m.AnimationGroup(m.Create(axis), m.FadeIn(ticks), m.FadeIn(ax_cap)),
            m.AnimationGroup(m.FadeIn(rows[0][0]), m.GrowFromCenter(rows[0][1]), m.FadeIn(rows[0][2]), m.FadeIn(rows[0][3], scale=0.5)),
        )
        self.say(
            "[Subtract a learned baseline]: [same average, a fraction of the spread]: [forty-five times less variance.]",
            m.AnimationGroup(m.FadeIn(rows[1][0]), m.GrowFromCenter(rows[1][1]), m.FadeIn(rows[1][2]), m.FadeIn(rows[1][3], scale=0.5)),
            m.AnimationGroup(m.Create(mean_line), m.FadeIn(mean_lab)),
            m.FadeIn(s1, shift=0.1 * m.UP),
        )
        self.say(
            "[With GAE, a hundred and one times less], [and the average still points the same way.]",
            m.AnimationGroup(m.FadeIn(rows[2][0]), m.GrowFromCenter(rows[2][1]), m.FadeIn(rows[2][2]), m.FadeIn(rows[2][3], scale=0.5),
                             m.FadeIn(s2, shift=0.1 * m.UP)),
            m.FadeIn(s3, shift=0.1 * m.UP),
        )
        self.say(
            "[The cut depends on the stage.] [Untrained, episodes are short], [and the baseline cuts under three-fold.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(axis, ticks, ax_cap, *rows, mean_line, mean_lab)),
                             m.FadeIn(g_axis), m.FadeIn(labels_g), m.FadeIn(g_leg), m.FadeIn(g_cap)),
            m.AnimationGroup(*[m.GrowFromEdge(p_[0], m.DOWN) for p_ in groups[0]], *[m.FadeIn(p_[1]) for p_ in groups[0]]),
            m.Circumscribe(groups[0][0], color=BASE_C, buff=0.1),
        )
        self.say(
            "[While learning, forty-five]; [once the policy is good, twenty-two], [while GAE still cuts a hundred and twenty-one.]",
            m.AnimationGroup(*[m.GrowFromEdge(p_[0], m.DOWN) for p_ in groups[1]], *[m.FadeIn(p_[1]) for p_ in groups[1]]),
            m.AnimationGroup(m.GrowFromEdge(groups[2][0][0], m.DOWN), m.FadeIn(groups[2][0][1])),
            m.AnimationGroup(m.GrowFromEdge(groups[2][1][0], m.DOWN), m.FadeIn(groups[2][1][1])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 9 · learning curves on this Mac

    def on_this_mac(self):
        self.next_section("9 · Three learners on this Mac", notes=(
            "deep_rl.md / deep_rl.json (this Mac Studio, M4 Max; 5 seeds (10–14) × 500,000 env steps each; numpy MLPs "
            "4-64-64, Adam 1e-3, γ = 0.99, no advantage normalisation). Score = mean return of the last 100 training "
            "episodes (sampled actions); solved at ≥ 475. Thin lines: each seed (curves.*.avg100_per_seed, every 2,500 "
            f"steps); thick: their mean. REINFORCE: {RF['solved_seeds']}, {RF['steps_to_solve_mean']:,.0f} ± "
            f"{RF['steps_to_solve_std']:,.0f} steps for the 3 that solved, final {RF['final_avg100_mean']} ± "
            f"{RF['final_avg100_std']}. + learned baseline (V fit to the returns, value_scale 100): {HB['solved_seeds']}, "
            f"{HB['steps_to_solve_mean']:,.0f} ± {HB['steps_to_solve_std']:,.0f} (min {HB['steps_to_solve_min']:,.0f}, max "
            f"{HB['steps_to_solve_max']:,.0f}), {HB['episodes_to_solve_mean']:.0f} episodes, final {HB['final_avg100_mean']} ± "
            f"{HB['final_avg100_std']}. A2C + GAE ({N_ENVS} envs × {N_STEPS} steps, λ = {LAM}): {HA['solved_seeds']}, "
            f"{HA['steps_to_solve_mean']:,.0f} ± {HA['steps_to_solve_std']:,.0f}, final {HA['final_avg100_mean']} ± "
            f"{HA['final_avg100_std']} (every seed stayed ≥ 475 after solving). Wall time per 500k-step run: "
            f"{WALL['reinforce']} s, {WALL['reinforce_baseline']} s, {WALL['a2c_gae']} s (A2C fastest per step); time to "
            f"solve: {RF['wall_s_to_solve_mean']} s, {HB['wall_s_to_solve_mean']} s, {HA['wall_s_to_solve_mean']} s — so A2C "
            "was NOT the fastest to solve (its small 128-sample updates need more steps). Runs shared the machine. CartPole is "
            "easy: the robust findings are the large effects (deep_rl.md caveats)."))
        plot = Plot((0, 525), (0, 525), width=5.9, height=3.9, center=(-2.9, -0.45, 0),
                    x_ticks=[0, 100, 200, 300, 400, 500], y_ticks=[0, 100, 200, 300, 400, 500],
                    x_label="thousand environment steps", y_label="score: average of the last 100 episodes")
        th_line = plot.hline(THRESH, MUTED, 2.5).set_z_index(0)
        th_lab = code("≥ 475 solved", 20, INK).next_to(plot.coords(525, THRESH), m.RIGHT, buff=0.12)
        sets = {}
        for key, col in (("reinforce", PLAIN_C), ("reinforce_baseline", BASE_C), ("a2c_gae", GAE_C)):
            thin = m.VGroup(*[plot.line(X_K, c, col, 2).set_stroke(opacity=0.35) for c in CURVES[key]["avg100_per_seed"]])
            thick = plot.line(X_K, CURVES[key]["avg100"]["mean"], col, 5)
            sets[key] = (thin.set_z_index(1), thick.set_z_index(2))
        meas = cite("trained on this Mac · 5 seeds each · thin: seeds · thick: mean").move_to([-2.9, -3.4, 0])

        rx = 1.3

        def item(col, name, lines, y):
            head = swatch(col, name, 22)
            subs = [code(minus(s_), 20, INK) for s_ in lines]
            g = m.VGroup(head, *subs).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            for s_ in subs:
                s_.align_to(head[1], m.LEFT)
            return left_at(g, rx, y)

        k = lambda v: f"{v / 1000:.0f}k"  # noqa: E731
        it1 = item(PLAIN_C, "REINFORCE", [f"{RF['n_solved']} of 5 solved · {k(RF['steps_to_solve_mean'])} ± {k(RF['steps_to_solve_std'])} steps"], 1.8)
        it2 = item(BASE_C, "+ learned baseline", [f"5 of 5 solved · {k(HB['steps_to_solve_mean'])} ± {k(HB['steps_to_solve_std'])} steps",
                                                   f"final {HB['final_avg100_mean']:.0f} ± {HB['final_avg100_std']:.0f}"], 0.3)
        it3 = item(GAE_C, "actor-critic: A2C + GAE", [f"5 of 5 solved · {k(HA['steps_to_solve_mean'])} ± {k(HA['steps_to_solve_std'])} steps",
                                                       f"final {HA['final_avg100_mean']:.0f} ± {HA['final_avg100_std']:.0f}"], -1.15)
        wall = m.VGroup(code(f"seconds per run: {WALL['reinforce']} · {WALL['reinforce_baseline']} · {WALL['a2c_gae']}", 20, INK,
                             t2c={re.escape(f"{WALL['reinforce']} "): PLAIN_C, re.escape(f"{WALL['reinforce_baseline']} "): BASE_C,
                                  re.escape(f"{WALL['a2c_gae']}") + "$": GAE_C}),
                        cite(f"to solve: {HB['wall_s_to_solve_mean']} s with a baseline · {HA['wall_s_to_solve_mean']} s A2C"))
        wall.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(wall, rx, -2.35)
        easy = pill("CartPole is easy: step counts say little", MUTED, CARD, 20)
        left_at(easy, rx, -3.15)
        end_dot = plot.dot(500, 500, GAE_C, 0.11)

        self.say(
            "[We trained all three on this Mac]: [five seeds each, half a million steps per run.]",
            m.AnimationGroup(self.new_header("09 · on this Mac", "Three learners, five seeds each"), m.FadeIn(plot)),
            m.AnimationGroup(m.FadeIn(meas, shift=0.1 * m.DOWN), m.Create(th_line), m.FadeIn(th_lab)),
        )
        self.say(
            "[Plain REINFORCE, from episode sixteen]: [three of five seeds solved it.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(l_) for l_ in sets["reinforce"][0]], lag_ratio=0.1, run_time=2.2),
                             m.Create(sets["reinforce"][1], run_time=2.2), m.FadeIn(it1[0])),
            m.FadeIn(it1[1:], shift=0.1 * m.UP),
        )
        self.say(
            "[With a learned baseline], [all five solved it], [in about eighty-three thousand steps, a third as many.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(l_) for l_ in sets["reinforce_baseline"][0]], lag_ratio=0.1, run_time=2.2),
                             m.Create(sets["reinforce_baseline"][1], run_time=2.2), m.FadeIn(it2[0])),
            m.FadeIn(it2[1], shift=0.1 * m.UP),
            m.FadeIn(it2[2], shift=0.1 * m.UP),
        )
        self.say(
            "[The actor-critic with GAE solved all five too]: [slower in steps here], [but every seed ended at five hundred.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(l_) for l_ in sets["a2c_gae"][0]], lag_ratio=0.1, run_time=2.2),
                             m.Create(sets["a2c_gae"][1], run_time=2.2), m.FadeIn(it3[0])),
            m.FadeIn(it3[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(it3[2], shift=0.1 * m.UP), m.FadeIn(end_dot, scale=0.5)),
        )
        self.say(
            "[It also got through its steps fastest]: [four and a half seconds per run.]",
            m.FadeIn(wall[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(wall[1]), m.Circumscribe(wall[0], color=GAE_C, buff=0.06)),
        )
        self.say(
            "[CartPole is easy, so step counts say little.] [The robust result is the noise]: "
            "[once episodes get long, twenty to over a hundred times less.]",
            m.FadeIn(easy, shift=0.1 * m.UP),
            m.Circumscribe(m.VGroup(*sets["reinforce"]), color=PLAIN_C, buff=0.05),
            m.Circumscribe(m.VGroup(*sets["reinforce_baseline"], *sets["a2c_gae"]), color=GAE_C, buff=0.05),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 10 · recap

    def recap(self):
        self.next_section("10 · Recap", notes=(
            "Subtract (a baseline b(s): the gradient stays unbiased, its variance drops — measured here 45× with a learned "
            "baseline, 101× with GAE, on one frozen policy mid-training), critique (a critic learns V(s); the advantage "
            "G − V(s), or the TD error δ, tells the actor how much better than expected a move was), blend (GAE's λ mixes "
            "one-step and long returns; λ = 0.95 typical). REINFORCE solved CartPole in 3 of 5 seeds; with a baseline 5 of 5. "
            "Next: Episode 18, PPO — reuse each batch without stepping too far."))
        halo = glow(SIGNAL, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Subtract", 54, SIGNAL), title_text("Critique", 54, BASE_C), title_text("Blend", 54, PLAIN_C))
        words.arrange(m.RIGHT, buff=1.25).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text("a baseline: less noise", 22, MUTED), text("a critic judges each move", 22, MUTED),
                        text("short and long, with λ", 22, MUTED))
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), words[1].get_bottom()[1] - 0.55, 0])
        on_baseline(subs)
        if subs[0].get_right()[0] > subs[1].get_left()[0] - 0.25 or subs[1].get_right()[0] > subs[2].get_left()[0] - 0.25:
            raise AssertionError("recap captions overlap")
        res = formula([r"\theta", r"\leftarrow", r"\theta", "+", r"\alpha", r"\,\hat A_t", r"\,\nabla_\theta",
                       r"\log \pi_\theta(a_t \mid s_t)"], [ACTION, INK, ACTION, INK, INK, SIGNAL, SIGNAL, ACTION], 42)
        res.move_to([0, -1.2, 0])
        proof = code(f"on this Mac: variance ÷ {HG['variance_ratio_no_baseline_over_baseline']:.0f} · solved 5 of 5, not 3 of 5",
                     22, INK, t2c={r"÷ \d+": BASE_C, r"5 of 5": GREEN, r"3 of 5": RED})
        proof.move_to([0, -2.25, 0])
        pair = m.VGroup(rover(0.5, CORAL, CORAL_FILL), rover(0.5, BLUE, BLUE_FILL)).arrange(m.RIGHT, buff=0.35)
        pair.move_to([0, -3.15, 0])

        self.say(
            "[To sum up]: [subtract], [critique], [blend.]",
            m.AnimationGroup(self.new_header("10 · recap", "Subtract, critique, blend"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Subtract a baseline to cut the noise], [let a critic judge each move], "
            "[and blend short and long returns with lambda.]",
            m.AnimationGroup(m.FadeIn(res, shift=0.1 * m.UP), m.Circumscribe(words[0], color=SIGNAL, buff=0.12)),
            m.AnimationGroup(m.Circumscribe(words[1], color=BASE_C, buff=0.12), m.FadeIn(pair, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Circumscribe(words[2], color=PLAIN_C, buff=0.12), m.FadeIn(proof, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()
