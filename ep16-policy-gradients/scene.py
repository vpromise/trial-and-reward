"""Trial and Reward · Episode 16 · Policy Gradients (Module 5 · Learning the Policy)

Module 5 opens by dropping the values. Episodes 10–14 learned how good each move is and took the argmax; now a
network outputs the policy itself, a probability for every move (softmax over two scores), and Rover acts by rolling
those dice. The running example stays CartPole (Episodes 12–13): the state of the REINFORCE replay at step 33 (pole
leaning 4.5° left and falling left), where the network trained on this Mac pushes left 94% of the time and the
untrained one 50/50. The objective J(θ) = expected return (the untrained policy: about 22, simulated here over 100
episodes; the most possible is 500). The log-derivative trick in three lines, then the physics dropping out of
∇ log p_θ(τ): ∇J = E[Σ_t G_t ∇ log π_θ(a_t|s_t)]. REINFORCE (Williams 1992): after each episode, push up the
log-probability of every move in proportion to the return that followed; for a softmax the push is one-hot − π.
Two episodes through one state (illustrative): a good one (pushed left, G = 40) and a bad one (pushed right, G = 8).
Every CartPole return is positive, so plain REINFORCE pushes every move up and only the size differs — said
honestly; the baseline that turns the bad push into a downward one is previewed for Episode 17. Why learn a policy
(Sutton & Barto §13.1): stochastic optima (the short corridor, computed here), continuous actions (Episode 19),
smooth changes. Measured on this Mac (deep_rl.json, 5 seeds × 500k steps): REINFORCE without a baseline solved
CartPole in 3 of 5 seeds (260k ± 53k steps for those), 2 of those 3 fell back below 300, final average 229 ± 147.
One run's final network replayed (trajectories.json): 500 steps, sampling every move. The catch: noise.

Every number on screen is computed below from rl_reference.py and asserted, or loaded with measured(...), or cited.

Sources (research/notes.md §16, §17 worked example C, Appendix C row 16; facts dated October 2026):
- Williams, R. J. (1992), "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement
  Learning", Machine Learning 8:229–256: REINFORCE, Δw = α (r − b) e with e = ∂ ln g/∂w; Theorem 1: the expected
  update is α times the gradient of the expected reinforcement (unbiased in expectation).
- Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), ch. 13: §13.1 (why parameterise the policy:
  stochastic optimal policies, continuous actions, smooth changes; Example 13.1 the short corridor with switched
  actions: best probability of right about 0.59, value about −11.6; ε-greedy with ε = 0.1 below −44 and −82),
  eq. 13.8 (REINFORCE: θ ← θ + α G_t ∇ ln π(A_t|S_t, θ)), eq. 13.9 (softmax: ∇ ln π = x(s,a) − Σ_b π(b|s) x(s,b),
  i.e. one-hot − π with one logit per action), §13.3 ("as a Monte Carlo method REINFORCE may be of high variance").
- The log-derivative trick (notes §16 key formula 2): ∇E[f] = Σ f ∇p = Σ f p ∇ln p = E[f ∇ln p]; ln p_θ(τ) =
  ln p(s0) + Σ ln π_θ(a_t|s_t) + Σ ln P(s_t+1|s_t,a_t), and only the middle sum depends on θ. Reward-to-go G_t
  instead of G_0: earlier rewards cannot depend on a_t.
- research/experiments/deep_rl.md + deep_rl.json: headline.reinforce (solved 3/5, steps to solve 260,143 ± 52,667 over
  the 3 that solved, final avg100 228.7 ± 147.0 over 5 seeds, 6.3 s per 500k-step run), methods.reinforce.hp
  (4-64-64-2 tanh, Adam 1e-3, γ = 0.99, one episode per update, loss = mean over the episode of −G_t log π, no
  advantage normalisation, Monte Carlo returns), per_seed (solve points; min avg100 after solving), curves.reinforce
  (avg100 per seed every 2,500 steps), seeds_dropping_below_300_after_solving = 2. Seeds 10–14, mean ± sample std.
  "Solved" = mean return of the last 100 training episodes ≥ 475 (CartPole-v1's registered threshold).
- research/experiments/trajectories.json: reinforce_cartpole (run pg_reinforce_s10, final weights, reset seed 2026,
  actions sampled, 500 steps, truncated; p_push_right stored per step and recomputed here with load_policy +
  mlp_forward + softmax). That seed did not meet the training criterion (its last 100 training episodes averaged
  157.5), yet its final weights averaged 490 over 100 fresh sampled episodes — the noise of this episode, in one run.
  random_cartpole: the untrained network (logits ≈ 0, 50/50).

Illustrative (said "for example" or labelled on screen): the Q-value bars in section 1 (no numbers); the J(θ) hill in
section 2; the two episodes through one state in section 4 (one state's two scores as the parameters, starting at
0/0 like the untrained network, α = 0.01, returns 40 and 8 — the notes' worked example B mirrored); the baseline
b = 24 (the mean of 40 and 8) in the Episode 17 preview. The 12 untrained returns (section 2) and the eight first
moves (section 7) are simulated here (reset seeds 3000–3011, actions from default_rng(16)); the ten sampled moves in
section 1 come from default_rng(6).

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

PREFIX = "ep16"
DEEP = "deep_rl"
LEFT, RIGHT = 0, 1

# ---------------------------------------------------------------------------- the numbers (all computed here)

# --- the trained policy and its replay (one run's final network, trained on this Mac)
PI = load_policy("reinforce_cartpole")
PI0 = load_policy("random_cartpole")
TRAJ = measured("trajectories", "policies", "reinforce_cartpole")
SRC = measured("trajectories", "sources", "reinforce_cartpole")
STATES = np.array(TRAJ["states"], dtype=float)
ACTS = list(TRAJ["actions"])
P_RIGHT = np.array(TRAJ["p_push_right"], dtype=float)
assert PI.output == "logits" and PI.meta["layers"] == [4, 64, 64, 2] and PI.meta["actions"] == ["push left", "push right"]
assert TRAJ["length"] == 500 == len(ACTS) and TRAJ["return"] == 500.0 and TRAJ["truncated"] and TRAJ["action_mode"] == "sampled"
assert SRC["run"] == "pg_reinforce_s10" and SRC["weights"] == "final"


def probs(policy, s):
    """The policy's two move probabilities in state s (gymnasium hands the agent float32)."""
    with np.errstate(all="ignore"):                 # Accelerate's matmul raises spurious warnings
        return softmax(policy(cartpole_obs(s)))


with np.errstate(all="ignore"):
    _p_all = softmax(PI(cartpole_obs(STATES[:-1])))[:, 1]
assert np.abs(_p_all - P_RIGHT).max() < 3e-4                     # the stored probabilities are this network's

T_S = 33                                                          # the example state: step 33 of the replay
S_EX = STATES[T_S]
P_EX = probs(PI, S_EX)
P0_EX = probs(PI0, S_EX)
LEAN = math.degrees(S_EX[2])
assert round(LEAN, 1) == -4.5 and S_EX[3] < 0                     # "leans left, and it's falling left"
assert round(P_EX[LEFT], 2) == 0.94                               # "pushes left ninety-four percent of the time"
assert abs(P0_EX[LEFT] - 0.5) < 0.005                              # untrained: "fifty-fifty"
assert np.abs(PI0(cartpole_obs(S_EX))).max() < 0.01                # its two scores are about 0
MOVES = np.random.default_rng(6).choice(2, size=10, p=P_EX)
assert MOVES.sum() == 1                                            # "mostly left, sometimes right"

# the first four steps of the replay, one by one
SLOW = 4
assert ACTS[:SLOW] == [RIGHT, LEFT, RIGHT, LEFT] and P_RIGHT[0] > 0.9 and P_RIGHT[1] < 0.1   # "push right, push left, …"


# --- J(θ): the untrained policy's returns (simulated here: reset seeds 3000–3099, actions from default_rng(16))
def play(policy, reset_seed, rng):
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


_rng = np.random.default_rng(16)
RUNS = [play(PI0, 3000 + k, _rng) for k in range(100)]
RETS = [g for g, _ in RUNS]
J0 = float(np.mean(RETS))
J0_EVAL = measured("trajectories", "sources", "random_cartpole", "eval", "mean_return")
assert round(J0) == 22 and abs(J0 - J0_EVAL) < 2.5                 # "about twenty-two" (the export's own eval: 22.4)
SHOW = RETS[:12]
assert max(SHOW) <= 60 and min(SHOW) >= 5
J_MAX = CARTPOLE["max_steps"]
assert J_MAX == 500                                                 # "a perfect policy scores five hundred"
FIRST8 = RUNS[:8]                                                   # section 7: eight first moves and their returns


# --- REINFORCE on one state (illustrative): the state's two scores are the parameters, starting at 0 / 0
def score(a, p):
    """∇ log π(a) for a softmax over one score per move: one-hot(a) − π."""
    return np.eye(2)[a] - p


ALPHA, G_GOOD, G_BAD = 0.01, 40.0, 8.0
Z0 = np.zeros(2)
P0 = softmax(Z0)
SC0 = score(LEFT, P0)
Z1 = Z0 + ALPHA * G_GOOD * SC0                                      # a good episode pushed left
P1 = softmax(Z1)
SC1 = score(RIGHT, P1)
Z2 = Z1 + ALPHA * G_BAD * SC1                                       # a bad episode pushed right
P2 = softmax(Z2)
B = (G_GOOD + G_BAD) / 2                                            # Episode 17 preview: a baseline
Z2B = Z1 + ALPHA * (G_BAD - B) * SC1
P2B = softmax(Z2B)
assert list(SC0) == [0.5, -0.5] and np.allclose(Z1, [0.2, -0.2])                 # "[+0.5, −0.5]", "±0.2"
assert round(P1[LEFT], 2) == 0.60 and round(P1[RIGHT], 2) == 0.40                # "left climbs to sixty percent"
assert round(P2[RIGHT], 2) == 0.42 and P2[RIGHT] > P1[RIGHT]                     # "right goes up too, a little"
assert round(P2B[RIGHT], 2) == 0.36 and B == 24                                  # with a baseline: down
PUSH_GOOD, PUSH_BAD = ALPHA * G_GOOD * SC0[LEFT], ALPHA * G_BAD * SC1[RIGHT]
assert PUSH_GOOD > 4 * PUSH_BAD > 0

# the code shown in section 4 runs as written (on the untrained policy's replay)
CODE = """G = discounted_returns(rewards, gamma=0.99)
logp = log_softmax(policy(states))
taken = logp[np.arange(T), actions]
loss = -(G * taken).mean()"""
_rand = measured("trajectories", "policies", "random_cartpole")
_ns = dict(np=np, discounted_returns=discounted_returns, log_softmax=log_softmax, policy=PI0,
           states=np.array(_rand["states"][:-1], dtype=np.float32), actions=np.array(_rand["actions"]),
           rewards=np.ones(_rand["length"]), T=_rand["length"])
with np.errstate(all="ignore"):
    exec(CODE, _ns)
assert np.isfinite(_ns["loss"]) and _ns["loss"] > 0 and _ns["G"][0] > _ns["G"][-1] == 1.0
HP = measured(DEEP, "methods", "reinforce", "hp")
assert HP["gamma"] == 0.99 and "-G_t log pi" in HP["loss"] and HP["variant"] == "reinforce"
assert HP["hidden"] == [64, 64] and HP["lr_policy"] == 1e-3 and HP["total_steps"] == 500_000
assert HP["advantage_normalisation"] is False and HP["episodes_per_update"] == 1


# --- the short corridor with switched actions (Sutton & Barto Example 13.1), computed
def corridor_value(p):
    """Start-state value when every state picks right with probability p: −1 per step; the 2nd state swaps the moves."""
    A = np.array([[p, -p, 0.0], [-p, 1.0, -(1 - p)], [0.0, -(1 - p), 1.0]])
    return float(np.linalg.solve(A, -np.ones(3))[0])


_ps = np.linspace(0.01, 0.99, 9801)
_vs = np.array([corridor_value(p) for p in _ps])
P_BEST, V_BEST = float(_ps[_vs.argmax()]), float(_vs.max())
assert round(P_BEST, 2) == 0.59 and abs(V_BEST + 11.6) < 0.1                    # S&B: "about 0.59", "about −11.6"
assert corridor_value(0.95) < -44 and corridor_value(0.05) < -82                # ε-greedy: "less than −44 and −82"

# --- measured on this Mac (5 seeds × 500,000 steps)
RF = measured(DEEP, "headline", "reinforce")
SEEDS = measured(DEEP, "methods", "reinforce", "per_seed")
CURVES = measured(DEEP, "curves", "reinforce")
FELL = measured(DEEP, "methods", "reinforce", "seeds_dropping_below_300_after_solving")
assert RF["solved_seeds"] == "3/5" and RF["n_solved"] == 3 and len(SEEDS) == 5                  # "three of the five"
assert round(RF["steps_to_solve_mean"] / 1000) == 260 and round(RF["steps_to_solve_std"] / 1000) == 53
assert round(RF["final_avg100_mean"]) == 229 and round(RF["final_avg100_std"]) == 147         # "229, give or take 147"
assert FELL == 2                                                                              # "two of the three fell back"
assert sum(1 for s in SEEDS if s["solved_at"] is None) == 2                                    # "two never got there"
assert round(RF["wall_s_total_mean"]) == 6
X_K = np.array(CURVES["x_steps"]) / 1000
PER_SEED = [np.array(c, dtype=float) for c in CURVES["avg100_per_seed"]]
assert len(PER_SEED) == 5 and X_K[-1] == 500 and len(X_K) == len(PER_SEED[0])
SOLVE_PTS = [(s["solved_at"]["env_steps"] / 1000, i) for i, s in enumerate(SEEDS) if s["solved_at"]]
for _k, _i in SOLVE_PTS:                       # each solve point lies on its own curve (sampled every 2,500 steps)
    _j = int(np.searchsorted(X_K, _k))
    assert PER_SEED[_i][min(_j, len(X_K) - 1)] >= 470 or PER_SEED[_i][max(_j - 1, 0)] >= 470
for _i, s in enumerate(SEEDS):
    assert abs(PER_SEED[_i][-1] - s["final_avg100"]) < 15
THRESH = 475

# ---------------------------------------------------------------------------- look

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


def prob_bars(values, center, width=2.6, height=2.4, size=22):
    return Bars(list(values), labels=["left", "right"], top=1.0, bottom=0.0, width=width, height=height,
                color=ACTION, gap=0.35, fmt="{:.2f}", size=size, center=center)


def set_bars(bars, values):
    """Jump a Bars to new values (inside an updater)."""
    for i, v in enumerate(values):
        bars.bars[i].become(bars._rect(i, v))
        bars.numbers[i].become(bars._number(i, v))


def push_arrow(view, action, size=0.75):
    """A CORAL arrow pushing the cart left or right."""
    c = view.cart.get_center()
    w = view.cart.width / 2
    if action == RIGHT:
        return m.Arrow(c + [-w - size - 0.08, 0, 0], c + [-w - 0.08, 0, 0], buff=0, color=ACTION, stroke_width=6,
                       max_tip_length_to_length_ratio=0.35)
    return m.Arrow(c + [w + size + 0.08, 0, 0], c + [w + 0.08, 0, 0], buff=0, color=ACTION, stroke_width=6,
                   max_tip_length_to_length_ratio=0.35)


def nudge(bars, i, dz, v, scale=4.0, color=SIGNAL):
    """A PINK arrow beside bar i (whose new value is v), from the bar's top, up or down by the change in its score."""
    side = -1 if i == 0 else 1
    x = bars.xs[i] + side * (bars.bar_w / 2 + 0.22)
    y = bars.level(v) - (scale * dz if dz > 0 else 0.0)
    y = max(y, bars.level(0) + 0.05)
    return m.Arrow([x, y, 0], [x, y + scale * dz, 0], buff=0, color=color, stroke_width=6,
                   max_tip_length_to_length_ratio=0.35 if abs(scale * dz) > 0.4 else 0.6)


def hill(t):
    return 0.14 + 0.74 * math.exp(-((t - 1.0) ** 2) / 3.0)


def hill_slope(t):
    return (hill(t) - 0.14) * (-2 * (t - 1.0) / 3.0)


HILL_STEPS = [-1.6]                       # gradient ascent on the illustrative hill: θ ← θ + 2.5 J′(θ)
for _ in range(5):
    HILL_STEPS.append(HILL_STEPS[-1] + 2.5 * hill_slope(HILL_STEPS[-1]))
assert all(hill(b) > hill(a) for a, b in zip(HILL_STEPS, HILL_STEPS[1:])) and abs(HILL_STEPS[-1] - 1.0) < 0.05


# ---------------------------------------------------------------------------- the episode


class Ep16PolicyGradients(Episode):
    number = 16

    def lesson(self):
        self.the_policy()
        self.objective()
        self.trick()
        self.reinforce()
        self.why_policies()
        self.on_this_mac()
        self.the_catch()
        self.recap()

    # ------------------------------------------------------------------ 1 · the policy itself

    def the_policy(self):
        self.next_section("1 · Learn the policy itself", notes=(
            "Episodes 10–14 learned action values and acted greedily (the Q bars here are an illustration, no numbers). "
            "Now the network outputs the policy: two scores (logits) → softmax → probabilities. The state is step "
            f"{T_S} of the replay of weights/reinforce_cartpole (trajectories.json, reset seed 2026): pole "
            f"{LEAN:.1f}° (leaning left), turning speed {S_EX[3]:.3f} rad/s (falling further left). The trained network "
            f"(REINFORCE, seed 10, final weights; 4-64-64-2 tanh) gives push left {P_EX[0]:.3f}, push right {P_EX[1]:.3f} "
            f"(load_policy + mlp_forward + softmax; matches the stored p_push_right to 3e-4). Ten sampled moves: "
            f"default_rng(6) with those probabilities → {''.join('LR'[a] for a in MOVES)}."))
        Y = -0.55                                   # the row's centre line: state → network → probabilities
        view = CartPoleView(center=(-4.6, Y - 0.35, 0), units_per_m=0.6, pole_units=1.9)
        view.set_state(S_EX[0], S_EX[2])
        top = view.pole.get_end()
        lean = code(f"{abs(LEAN):.1f}° left, falling left", 20, STATE).move_to(top + [0.0, 0.36, 0])
        st_lab = text("the state: four numbers", 22, STATE).move_to([-4.6, Y - 1.05, 0])

        # values (Episodes 10–14): a value network, two BLUE bars (no numbers), the argmax
        qnet = NeuralNet(layers=(4, 6, 6, 2), width=2.3, height=2.1, center=(-0.35, Y, 0), color=VALUE, radius=0.1)
        qbars = Bars([0.74, 0.58], labels=["left", "right"], top=1.0, bottom=0.0, width=2.6, height=2.4, color=VALUE,
                     gap=0.35, show_values=False, size=22, center=(4.0, Y - 0.2, 0))
        q_lab = tex(r"Q(s,a)", 38, VALUE).move_to([4.0, Y + 1.5, 0])
        argmax = pill("argmax → push left", VALUE, BLUE_FILL, 22).move_to([4.0, Y + 2.25, 0])
        q_cite = cite("episodes 10 to 14").next_to(qnet, m.UP, buff=0.3)

        # the policy network
        net = NeuralNet(layers=(4, 6, 6, 2), width=2.3, height=2.1, center=(-0.35, Y, 0), color=ACTION, radius=0.1)
        mascot = rover(0.45).move_to([-1.15, Y + 1.75, 0])
        pol_lab = m.VGroup(text("Rover's policy", 22, ACTION, weight=m.BOLD), cite("trained on this Mac"))
        pol_lab.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT).next_to(mascot, m.RIGHT, buff=0.25)
        a_in = arrow([-2.55, Y, 0], [-1.7, Y, 0], color=STATE)
        outs = net.columns[-1]
        z_labs = m.VGroup(text("score", 26, ACTION).next_to(outs[0], m.RIGHT, buff=0.15),
                          text("score", 26, ACTION).next_to(outs[1], m.RIGHT, buff=0.15))
        a_out = arrow([1.75, Y, 0], [2.45, Y, 0], color=ACTION)
        bars = prob_bars(P_EX, center=(4.0, Y - 0.2, 0))
        soft = formula([r"\pi_\theta(a \mid s)", "=", r"\frac{e^{z_a}}{\sum_b e^{z_b}}"], [ACTION, INK, INK], 36)
        soft.move_to([4.0, Y + 2.25, 0])
        soft_lab = cite("softmax: scores → probabilities").next_to(soft, m.DOWN, buff=0.12)

        # the dice: ten moves sampled from those probabilities
        row = m.VGroup()
        for a in MOVES:
            box = card(0.56, 0.56, ACTION, CORAL_FILL, radius=0.1, stroke=2)
            d = 0.17 if a == RIGHT else -0.17
            row.add(m.VGroup(box, m.Arrow(box.get_center() - [d, 0, 0], box.get_center() + [d, 0, 0], buff=0, color=INK,
                                          stroke_width=5, max_tip_length_to_length_ratio=0.5)))
        row.arrange(m.RIGHT, buff=0.12).move_to([-0.1, -3.05, 0])
        dice = icon(DICE, 0.62).next_to(row, m.LEFT, buff=0.45)
        stoch = pill("a stochastic policy", ACTION, CORAL_FILL, 22).next_to(row, m.RIGHT, buff=0.45)

        self.say(
            "[Until now, Rover learned values]: [how good each move is], [then took the best one.]",
            m.AnimationGroup(self.new_header("01 · the policy itself", "Learn the policy itself"),
                             m.FadeIn(view, shift=0.15 * m.UP), m.FadeIn(st_lab)),
            m.AnimationGroup(m.FadeIn(qnet), m.FadeIn(q_cite), m.FadeIn(qbars, shift=0.15 * m.UP), m.FadeIn(q_lab)),
            m.FadeIn(argmax, shift=0.1 * m.DOWN),
        )
        self.say(
            "[Now we skip the values], [and learn the policy itself]: [a probability for every move.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(qbars, q_lab, argmax, q_cite), shift=0.2 * m.UP), m.FadeOut(qnet)),
            m.AnimationGroup(m.FadeIn(net), m.FadeIn(mascot, shift=0.15 * m.UP), m.FadeIn(pol_lab)),
            m.FadeIn(bars, shift=0.15 * m.UP),
        )
        self.say(
            "[A network reads CartPole's four numbers], [gives each move a score], "
            "[and a softmax turns the scores into probabilities.]",
            m.AnimationGroup(m.GrowArrow(a_in), net.forward(ACTION, run_time=1.4)),
            m.AnimationGroup(m.FadeIn(z_labs, shift=0.1 * m.RIGHT), m.GrowArrow(a_out)),
            m.AnimationGroup(m.Write(soft), m.FadeIn(soft_lab)),
        )
        self.say(
            "[Here the pole leans left, and it's falling left], "
            "[so the trained policy pushes left ninety-four percent of the time.]",
            m.AnimationGroup(m.FadeIn(lean, shift=0.1 * m.UP), m.Circumscribe(view.pole, color=STATE, buff=0.1)),
            m.Circumscribe(m.VGroup(bars.bars[0], bars.numbers[0]), color=ACTION, buff=0.1),
        )
        self.say(
            "[To act, it rolls the dice]: [mostly left, sometimes right.] [That's a stochastic policy.]",
            m.Succession(m.FadeIn(dice, scale=0.6, run_time=0.4), m.Rotate(dice, angle=2 * math.pi, run_time=0.9)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.12 * m.DOWN) for c in row], lag_ratio=0.18, run_time=1.8),
            m.FadeIn(stoch, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the objective

    def objective(self):
        self.next_section("2 · The objective J(θ)", notes=(
            "J(θ) = E_τ∼π_θ[G(τ)], the expected return of the policy (notes §16; Sutton et al. 2000 write ρ). The "
            f"bars are 12 episodes of the untrained network (weights/random_cartpole), simulated here with "
            f"rl_reference.cartpole_step from reset seeds 3000–3011, actions sampled with default_rng(16): "
            f"{', '.join(f'{g:.0f}' for g in SHOW)}. The dashed line is the mean of 100 such episodes (seeds "
            f"3000–3099): {J0:.2f} (the export's own evaluation over the same reset seeds: {J0_EVAL}). The cap is "
            "500 steps (CartPole-v1 truncates), so J ≤ 500. The hill is an illustration of gradient ascent, not data. "
            "θ reaches J only through sampled actions (no derivative through a dice roll) and through the environment "
            "(the agent has no equations for it)."))
        ebars = Bars(SHOW, labels=[str(i + 1) for i in range(len(SHOW))], top=60, bottom=0, width=5.9, height=3.0,
                     color=REWARD, gap=0.12, fmt="{:.0f}", size=20, center=(-3.45, -0.55, 0))
        e_lab = text("12 episodes of the untrained policy", 22, MUTED).move_to([-3.45, -2.75, 0])
        g_lab = left_at(text("return of each episode", 22, REWARD), -6.4, 1.45)
        # J: the average over all episodes it could play (estimated from 100), as one more bar, set apart
        jx, jw = 0.35, ebars.bar_w * 1.25
        j_rect = m.Rectangle(width=jw, height=ebars.level(J0) - ebars.level(0)).set_stroke(REWARD, 3).set_fill(REWARD, 0.55)
        j_rect.move_to([jx, (ebars.level(J0) + ebars.level(0)) / 2, 0])
        j_base = m.Line([jx - jw / 2 - 0.15, ebars.level(0), 0], [jx + jw / 2 + 0.15, ebars.level(0), 0]).set_stroke(FAINT, 2)
        j_num = code(f"{J0:.0f}", 22, REWARD, weight=m.BOLD).next_to(j_rect, m.UP, buff=0.12)
        j_name = code("J", 24, REWARD, weight=m.BOLD).move_to([jx, ebars.labels[0].get_y(), 0])
        j_line = m.VGroup(j_base, j_rect)
        j_val = m.VGroup(j_num, j_name)

        jf = formula([r"J(\theta)", "=", r"\mathbb{E}_{\tau \sim \pi_\theta}", r"\big[\,", r"G(\tau)", r"\,\big]"],
                     [REWARD, INK, ACTION, INK, REWARD, INK], 48)
        jf.move_to([4.1, 1.45, 0])
        jf_lab = text("the expected return", 24, INK).next_to(jf, m.DOWN, buff=0.25)
        th_lab = m.VGroup(tex(r"\theta", 34, ACTION), text(": the network's weights", 22, MUTED)).arrange(m.RIGHT, buff=0.08)
        th_lab.next_to(jf_lab, m.DOWN, buff=0.22)
        avg_lab = m.VGroup(cite("the average of"), cite("100 episodes,"), cite("simulated here")).arrange(m.DOWN, buff=0.06)
        avg_lab.next_to(j_num, m.UP, buff=0.2)
        j_stats = m.VGroup(
            m.VGroup(text("untrained", 22, MUTED), code(f"J ≈ {J0:.0f}", 28, REWARD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25),
            m.VGroup(text("a perfect policy", 22, MUTED), code(f"J = {J_MAX}", 28, REWARD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25),
        ).arrange(m.DOWN, buff=0.25, aligned_edge=m.RIGHT).move_to([4.4, -1.45, 0])

        # the hill: gradient ascent (illustration)
        plot = Plot((-3, 3), (0, 1), width=5.0, height=3.0, center=(-3.4, -0.35, 0), x_label="θ", y_label="J(θ)")
        xs = np.linspace(-3, 3, 121)
        curve = plot.line(xs, [hill(t) for t in xs], REWARD, 5)
        ill = cite("illustration").next_to(plot, m.DOWN, buff=0.1).align_to(plot, m.RIGHT)
        steps = HILL_STEPS
        dot = m.Dot(plot.coords(steps[0], hill(steps[0])), radius=0.11, color=INK).set_z_index(4)

        def tangent(t, length=1.0):
            p = plot.coords(t, hill(t))
            d = plot.coords(t + 0.05, hill(t + 0.05)) - p
            d = d / np.linalg.norm(d)
            return m.Arrow(p, p + length * d, buff=0.12, color=SIGNAL, stroke_width=6, max_tip_length_to_length_ratio=0.3).set_z_index(3)

        grad = tangent(steps[0])
        climb = m.VGroup(dot, grad)

        def go(_, alpha):
            k = alpha * (len(steps) - 1)
            j = min(int(k), len(steps) - 2)
            t = steps[j] + (k - j) * (steps[j + 1] - steps[j])
            dot.move_to(plot.coords(t, hill(t)))
            grad.become(tangent(t, 1.0 * max(0.25, min(1.0, 1.4 * hill_slope(t) / hill_slope(steps[0])))))

        upd = formula([r"\theta", r"\leftarrow", r"\theta", "+", r"\alpha", r"\,\nabla_\theta", r"J(\theta)"],
                      [ACTION, INK, ACTION, INK, INK, SIGNAL, REWARD], 46)
        upd.move_to([3.4, 1.7, 0])
        upd_lab = text("step uphill: gradient ascent", 22, MUTED).next_to(upd, m.DOWN, buff=0.2)

        # the obstacle: θ → dice → physics → return
        chain_y = -1.05
        th_node = pill("θ", ACTION, CORAL_FILL, 30).move_to([0.75, chain_y, 0])
        dice = icon(DICE, 0.75).move_to([2.45, chain_y, 0])
        mini = CartPoleView(center=(0, 0, 0), units_per_m=0.5, pole_units=1.1)
        mini.set_state(0.0, 0.15)
        mini.scale(0.8).move_to([4.4, chain_y + 0.1, 0])
        g_node = pill("G", REWARD, GOLD_FILL, 30).move_to([6.15, chain_y, 0])
        links = m.VGroup(arrow(th_node.get_right(), dice.get_left(), MUTED, buff=0.15),
                         arrow(dice.get_right(), mini.get_left(), MUTED, buff=0.15),
                         arrow(mini.get_right(), g_node.get_left(), MUTED, buff=0.15))
        names = m.VGroup(text("dice rolls", 20, ACTION).next_to(dice, m.DOWN, buff=0.22),
                         text("the physics", 20, STATE).next_to(mini, m.DOWN, buff=0.22))
        names[1].align_to(names[0], m.DOWN)
        nope_pill = pill("no derivative through either", RED, RED_FILL, 22)
        nope_pill.move_to([(dice.get_x() + mini.get_x()) / 2, names.get_bottom()[1] - 0.45, 0])
        nope_br = m.VGroup(*[m.Line(n.get_bottom() + [0, -0.06, 0], [n.get_x(), nope_pill.get_top()[1] + 0.04, 0]).set_stroke(RED, 2.5)
                             for n in names])
        nope = m.VGroup(nope_br, nope_pill)

        self.say(
            "[How good is a policy?] [Let it play]: [every episode ends with a return.]",
            m.AnimationGroup(self.new_header("02 · the objective", "How good is a policy?"), m.FadeIn(ebars.zero)),
            m.AnimationGroup(m.FadeIn(e_lab), m.FadeIn(g_lab), m.FadeIn(ebars.labels)),
            m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(b, m.DOWN), m.FadeIn(n_)) for b, n_ in zip(ebars.bars, ebars.numbers)],
                          lag_ratio=0.2, run_time=2.4),
        )
        self.say(
            "[Its score is the expected return]: [the average over all the episodes it could play.] "
            "[We call it J of theta.]",
            m.AnimationGroup(m.Write(jf), m.FadeIn(jf_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(j_base), m.GrowFromEdge(j_rect, m.DOWN), m.FadeIn(j_val), m.FadeIn(avg_lab)),
            m.AnimationGroup(m.Circumscribe(jf[0], color=REWARD, buff=0.08), m.FadeIn(th_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[Untrained, it's about twenty-two]; [a perfect policy scores five hundred.]",
            m.AnimationGroup(m.FadeIn(j_stats[0], shift=0.1 * m.UP), m.Circumscribe(j_line, color=REWARD, buff=0.08)),
            m.FadeIn(j_stats[1], shift=0.1 * m.UP),
        )
        self.say(
            "[To improve, climb]: [nudge theta uphill, along the gradient.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(ebars, e_lab, g_lab, j_line, j_val, jf, jf_lab, th_lab, avg_lab, j_stats)),
                             m.FadeIn(plot), m.Create(curve), m.FadeIn(ill), m.FadeIn(climb), m.FadeIn(upd, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(upd_lab), m.UpdateFromAlphaFunc(climb, go, run_time=2.6, rate_func=m.smooth)),
        )
        self.say(
            "[But theta changes the return only through dice rolls], [and through CartPole's physics]: "
            "[neither gives us a derivative.]",
            m.AnimationGroup(m.FadeIn(th_node), m.GrowArrow(links[0]), m.FadeIn(dice, scale=0.7), m.FadeIn(names[0])),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(mini), m.FadeIn(names[1]), m.GrowArrow(links[2]), m.FadeIn(g_node)),
            m.AnimationGroup(m.Create(nope_br), m.FadeIn(nope_pill, scale=0.8)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · the log-derivative trick

    def trick(self):
        self.next_section("3 · The log-derivative trick", notes=(
            "notes §16 key formula 2 [standard derivation]: ∇E_p[f] = Σ f ∇p = Σ f p ∇ln p = E[f ∇ln p]. For a "
            "trajectory, ln p_θ(τ) = ln p(s0) + Σ_t ln π_θ(a_t|s_t) + Σ_t ln P(s_t+1|s_t,a_t); only the middle sum depends "
            "on θ, so ∇J = E[Σ_t G_t ∇ln π_θ(a_t|s_t)], with the reward-to-go G_t allowed because rewards before t cannot "
            "depend on a_t. No model of the environment and no derivative of the reward are needed (the reward is only a "
            "multiplier) — which is why a 0/1 answer checker can train a language model later (Episode 24). The policy "
            "gradient theorem: Sutton, McAllester, Singh & Mansour (2000)."))
        L0 = formula([r"J(\theta)", "=", r"\sum_{\tau}", r"p_\theta(\tau)", r"\,G(\tau)"],
                     [REWARD, INK, INK, ACTION, REWARD])
        L1 = formula([r"\nabla_\theta", r"J(\theta)", "=", r"\sum_{\tau}", r"G(\tau)", r"\,\nabla_\theta", r"p_\theta(\tau)"],
                     [SIGNAL, REWARD, INK, INK, REWARD, SIGNAL, ACTION])
        L2 = formula(["=", r"\sum_{\tau}", r"G(\tau)", r"\,p_\theta(\tau)", r"\,\nabla_\theta", r"\log p_\theta(\tau)"],
                     [INK, INK, REWARD, ACTION, SIGNAL, ACTION])
        L3 = formula(["=", r"\mathbb{E}_{\tau \sim \pi_\theta}", r"\Big[", r"G(\tau)", r"\,\nabla_\theta",
                      r"\log p_\theta(\tau)", r"\Big]"], [INK, INK, INK, REWARD, SIGNAL, ACTION, INK])
        eq_x = -3.4
        for f_, y, k in ((L0, 1.95, 1), (L1, 0.85, 2), (L2, -0.25, 0), (L3, -1.4, 0)):
            f_.move_to([0, y, 0])
            f_.shift((eq_x - f_[k].get_x()) * m.RIGHT)
        nums = m.VGroup(*[code(f"{i}", 20, FAINT).move_to([-6.3, f_.get_y(), 0]) for i, f_ in ((1, L1), (2, L2), (3, L3))])
        ident = formula([r"\nabla p", "=", r"p\,\nabla \log p"], [SIGNAL, INK, INK], 36)
        ident_lab = cite("because ∇ log p = ∇p ÷ p")
        ident_in = m.VGroup(ident, ident_lab).arrange(m.DOWN, buff=0.18)
        ident_box = card(ident_in.width + 0.6, ident_in.height + 0.5, SIGNAL, PINK_FILL, radius=0.18, stroke=2.5)
        ident_box.move_to(ident_in)
        ident_g = m.VGroup(ident_box, ident_in)
        ident_g.move_to([0, L2.get_y(), 0]).align_to([L2.get_right()[0] + 0.55, 0, 0], m.LEFT)
        assert ident_g.get_right()[0] < 6.6
        avg_lab = text("an average over episodes we can play", 22, INK).next_to(L3, m.DOWN, buff=0.3).align_to(L3[1], m.LEFT)

        # part two: the physics drops out
        L4 = formula([r"\log p_\theta(\tau)", "=", r"\sum_t \log \pi_\theta(a_t \mid s_t)", "+",
                      r"\sum_t \log P(s_{t+1} \mid s_t, a_t) + \log p(s_0)"], [ACTION, INK, ACTION, INK, STATE], 36)
        L4.move_to([0, 0.75, 0])
        L3b = formula([r"\nabla_\theta", r"J(\theta)", "=", r"\mathbb{E}_{\tau \sim \pi_\theta}", r"\Big[", r"G(\tau)", r"\,\nabla_\theta",
                       r"\log p_\theta(\tau)", r"\Big]"], [SIGNAL, REWARD, INK, INK, INK, REWARD, SIGNAL, ACTION, INK])
        L3b.move_to([0, 2.05, 0])
        b_pol = m.Brace(L4[2], m.DOWN, buff=0.12, color=ACTION)
        t_pol = text("the policy's choices", 22, ACTION).next_to(b_pol, m.DOWN, buff=0.1)
        b_phy = m.Brace(L4[4], m.DOWN, buff=0.12, color=STATE)
        t_phy = text("the physics", 22, STATE).next_to(b_phy, m.DOWN, buff=0.1)
        zero = text("no θ in it: gradient zero", 22, RED).move_to(t_phy)
        res = formula([r"\nabla_\theta", r"J(\theta)", "=", r"\mathbb{E}", r"\Big[\sum_t", r"G_t", r"\,\nabla_\theta",
                       r"\log \pi_\theta(a_t \mid s_t)", r"\Big]"],
                      [SIGNAL, REWARD, INK, INK, INK, REWARD, SIGNAL, ACTION, INK], 50)
        res.move_to([0, -2.1, 0])
        res_box = m.SurroundingRectangle(res, buff=0.25, corner_radius=0.16).set_stroke(SIGNAL, 3)
        res_lab = code("the policy gradient", 22, SIGNAL, weight=m.BOLD).next_to(res_box, m.UP, buff=0.12).align_to(res_box, m.LEFT)
        gt_lab = cite("G_t: the return from step t on · policy gradient theorem: Sutton et al., 2000").next_to(res_box, m.DOWN, buff=0.15)
        frees = m.VGroup(pill("no model", GOOD, GOOD_FILL, 20), pill("no reward derivative", GOOD, GOOD_FILL, 20))
        frees.arrange(m.RIGHT, buff=0.25).next_to(res_box, m.UP, buff=0.12).align_to(res_box, m.RIGHT)

        self.say(
            "[Here's the trick.] [The expected return is a sum over every possible episode]: [its probability, times its return.]",
            self.new_header("03 · the trick", "The log-derivative trick"),
            m.FadeIn(L0[:3], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(L0[3:], shift=0.1 * m.UP), m.Circumscribe(L0[3], color=ACTION, buff=0.06),
                             m.Circumscribe(L0[4], color=REWARD, buff=0.06)),
        )
        self.say(
            "[Take the gradient]: [it lands on the probabilities alone], [since the returns are just numbers.]",
            m.AnimationGroup(m.FadeIn(L1[:3], shift=0.1 * m.DOWN), m.FadeIn(nums[0])),
            m.AnimationGroup(m.FadeIn(L1[3:], shift=0.1 * m.DOWN), m.Circumscribe(L1[5:], color=SIGNAL, buff=0.06)),
            m.Circumscribe(L1[4], color=REWARD, buff=0.06),
        )
        self.say(
            "[Now one identity]: [the gradient of a probability is the probability, times the gradient of its log.]",
            m.FadeIn(ident_g, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeIn(L2, shift=0.1 * m.DOWN), m.FadeIn(nums[1])),
        )
        self.say(
            "[And a sum weighted by probabilities is an average]: [over episodes we can simply play.]",
            m.AnimationGroup(m.FadeIn(L3, shift=0.1 * m.DOWN), m.FadeIn(nums[2])),
            m.AnimationGroup(m.FadeIn(avg_lab, shift=0.1 * m.UP), m.Circumscribe(L3[1], color=INK, buff=0.06)),
        )
        self.say(
            "[An episode's log-probability splits in two]: [the policy's choices], [plus the physics.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(L0, L1, L2, nums, ident_g, avg_lab), shift=0.3 * m.UP),
                             m.FadeOut(L3, target_position=L3b), m.FadeIn(L3b, target_position=L3), m.FadeIn(L4, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowFromCenter(b_pol), m.FadeIn(t_pol)),
            m.AnimationGroup(m.GrowFromCenter(b_phy), m.FadeIn(t_phy)),
        )
        self.say(
            "[The physics doesn't depend on theta], [so its gradient is zero.]",
            m.Circumscribe(L4[4], color=STATE, buff=0.06),
            m.AnimationGroup(L4[4].animate.set_opacity(0.3), b_phy.animate.set_color(RED), m.FadeOut(t_phy), m.FadeIn(zero)),
        )
        self.say(
            "[What's left needs no model and no derivative of the reward]: "
            "[just the return, times the gradient of each move's log-probability.]",
            m.AnimationGroup(m.FadeIn(res, shift=0.15 * m.UP), m.Create(res_box), m.FadeIn(res_lab), m.FadeIn(frees, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(res[5], color=REWARD, buff=0.06), m.Circumscribe(res[6:8], color=SIGNAL, buff=0.06),
                             m.FadeIn(gt_lab)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · REINFORCE

    def reinforce(self):
        self.next_section("4 · REINFORCE", notes=(
            "Williams (1992), Machine Learning 8:229–256; Sutton & Barto eq. 13.8: θ ← θ + α G_t ∇ ln π(A_t|S_t, θ) "
            "(the boxed pseudocode carries γ^t, which deep-RL code drops). Softmax with one score per move: ∇ ln π(a) = "
            "one-hot(a) − π (S&B eq. 13.9). Illustration (notes §16 worked example B, mirrored): one state (the same "
            f"step-{T_S} state), its two scores z = (0, 0) as the parameters (the untrained network's are "
            f"{PI0(cartpole_obs(S_EX))[0]:+.4f} / {PI0(cartpole_obs(S_EX))[1]:+.4f}), α = {ALPHA}. A good episode pushed "
            f"left and scored G = {G_GOOD:.0f}: Δz = α·G·(one-hot − π) = ({Z1[0]:+.2f}, {Z1[1]:+.2f}) → π = "
            f"({P1[0]:.3f}, {P1[1]:.3f}). A bad episode pushed right and scored G = {G_BAD:.0f}: Δz = "
            f"({Z2[0] - Z1[0]:+.3f}, {Z2[1] - Z1[1]:+.3f}) → π = ({P2[0]:.3f}, {P2[1]:.3f}): right goes UP, by less. "
            "Every CartPole return is positive (+1 per step), so plain REINFORCE pushes every sampled move up; only the "
            "size differs (notes §17 worked example C). The code is an excerpt in rl_reference's names; it runs as "
            "written in this file (on the untrained policy's replay) and matches the measured runs' loss (deep_rl.json "
            "methods.reinforce.hp: mean over the episode of −G_t log π, γ = 0.99, one episode per update)."))
        upd = formula([r"\theta", r"\leftarrow", r"\theta", "+", r"\alpha", r"\,G_t", r"\,\nabla_\theta",
                       r"\log \pi_\theta(a_t \mid s_t)"], [ACTION, INK, ACTION, INK, INK, REWARD, SIGNAL, ACTION], 46)
        upd.move_to([-0.9, 1.95, 0])
        w_cite = m.VGroup(code("REINFORCE", 22, ACTION, weight=m.BOLD), cite("Williams, 1992"))
        w_cite.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(w_cite, upd.get_right()[0] + 0.5, 1.95)
        b_g = m.Brace(upd[5], m.DOWN, buff=0.1, color=REWARD)
        t_g = text("the return that followed", 22, REWARD).next_to(b_g, m.DOWN, buff=0.08)
        b_l = m.Brace(upd[6:], m.DOWN, buff=0.1, color=SIGNAL)
        t_l = text("push up the move it made", 22, SIGNAL).next_to(b_l, m.DOWN, buff=0.08)
        if t_g.get_right()[0] > t_l.get_left()[0] - 0.3:
            t_g.align_to(b_g, m.RIGHT)
            t_l.align_to(b_l, m.LEFT)

        view = CartPoleView(center=(-4.85, -1.95, 0), units_per_m=0.5, pole_units=1.6)
        view.set_state(S_EX[0], S_EX[2])
        v_lab = text("one state", 22, STATE).move_to([-4.85, -2.75, 0])
        bars = prob_bars(P0, center=(-1.25, -1.45, 0), width=2.5, height=2.1)
        b_cap = cite("its two probabilities").next_to(bars.labels, m.DOWN, buff=0.15)

        px = 1.0
        sc_lab = left_at(text("the push for left: move made − probabilities", 22, SIGNAL), px, 0.0)
        SC_KEY = minus(f"[+{SC0[0]:.1f}, {SC0[1]:.1f}]")
        sc_t = code(minus(f"[1, 0] - [{P0[0]:.1f}, {P0[1]:.1f}] = ") + SC_KEY, 22, INK, t2c={re.escape(SC_KEY): SIGNAL})
        left_at(sc_t, px, -0.55)
        ep_good = m.VGroup(pill(f"G = {G_GOOD:.0f}", REWARD, GOLD_FILL, 22, mono=True), text("good episode: pushed left", 22, INK))
        ep_good.arrange(m.RIGHT, buff=0.25)
        left_at(ep_good, px, -1.35)
        ep_bad = m.VGroup(pill(f"G = {G_BAD:.0f}", REWARD, GOLD_FILL, 22, mono=True), text("bad episode: pushed right", 22, INK))
        ep_bad.arrange(m.RIGHT, buff=0.25)
        left_at(ep_bad, px, -2.1)
        every = left_at(text("no negative returns: every move goes up", 22, RED), px, -2.95)
        ill4 = left_at(cite("illustration · α = 0.01 · its two scores start at 0"), px, 0.5)
        a_good = nudge(bars, LEFT, PUSH_GOOD, P1[LEFT], scale=4.0)
        a_bad = nudge(bars, RIGHT, PUSH_BAD, P2[RIGHT], scale=4.0)
        a_good.set_z_index(3)
        a_bad.set_z_index(3)
        arr_good = push_arrow(view, LEFT, 0.6)
        arr_bad = push_arrow(view, RIGHT, 0.6)

        blk = code_block(CODE, size=22, title="reinforce.py · one update, after each episode", accent=ACTION)
        blk.move_to([-1.75, -1.25, 0])
        rows = blk[1]
        line_ys = []
        lines_src = CODE.splitlines()
        k = 0
        for ln in lines_src:
            n_ = len(ln.replace(" ", ""))
            line_ys.append(float(np.mean([g.get_center()[1] for g in rows[k:k + n_]])))
            k += n_
        notes_x = blk.get_right()[0] + 0.35
        ann = m.VGroup(
            left_at(text("the return from every step", 20, REWARD), notes_x, line_ys[0]),
            left_at(text("log-probabilities of both moves", 20, ACTION), notes_x, line_ys[1]),
            left_at(text("…of the move it made", 20, ACTION), notes_x, line_ys[2]),
            left_at(text("descend it, and J climbs", 20, SIGNAL), notes_x, line_ys[3]),
        )

        self.say(
            "[Put it to work, and you get REINFORCE]: [Ronald Williams, nineteen ninety-two.]",
            m.AnimationGroup(self.new_header("04 · REINFORCE", "Push up, in proportion to the return"),
                             m.FadeIn(upd, shift=0.1 * m.UP)),
            m.FadeIn(w_cite, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Play a whole episode.] [Then, at every step, push up the log-probability of the move it made], "
            "[in proportion to the return that followed.]",
            m.Circumscribe(upd, color=MUTED, buff=0.1),
            m.AnimationGroup(m.GrowFromCenter(b_l), m.FadeIn(t_l)),
            m.AnimationGroup(m.GrowFromCenter(b_g), m.FadeIn(t_g)),
        )
        self.say(
            "[Take the state from before.] [Untrained, the policy says fifty-fifty.]",
            m.AnimationGroup(m.FadeIn(view, shift=0.15 * m.UP), m.FadeIn(v_lab)),
            m.AnimationGroup(m.FadeIn(bars, shift=0.15 * m.UP), m.FadeIn(b_cap)),
        )
        self.say(
            "[For a softmax, the push is simple]: [the move made, minus the probabilities.] "
            "[For left, that's plus a half, and minus a half for right.]",
            m.FadeIn(sc_lab, shift=0.1 * m.UP),
            m.FadeIn(sc_t, shift=0.1 * m.UP),
            m.Circumscribe(glyphs(sc_t, SC_KEY), color=SIGNAL, buff=0.06),
        )
        self.say(
            "[Say a good episode pushed left here, and scored forty]: [left climbs to sixty percent.]",
            m.AnimationGroup(m.FadeIn(ep_good, shift=0.1 * m.UP), m.GrowArrow(arr_good), m.FadeIn(ill4)),
            m.AnimationGroup(m.GrowArrow(a_good), bars.set(P1, run_time=1.0)),
        )
        self.say(
            "[A bad episode pushed right, and scored only eight]: [right goes up too, but only a little.]",
            m.AnimationGroup(m.FadeIn(ep_bad, shift=0.1 * m.UP), m.FadeOut(arr_good), m.GrowArrow(arr_bad)),
            m.AnimationGroup(m.GrowArrow(a_bad), bars.set(P2, run_time=1.0)),
        )
        self.say(
            "[In CartPole, a return is never negative], [so every move gets pushed up]; "
            "[good episodes just push harder, and win on average.]",
            m.FadeIn(every, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(a_good, color=SIGNAL, buff=0.06), m.Circumscribe(a_bad, color=SIGNAL, buff=0.06)),
            m.Circumscribe(m.VGroup(bars.bars[0], bars.numbers[0]), color=ACTION, buff=0.08),
        )
        self.say(
            "[In code, it's four lines]: [the returns], [the log-probabilities of the moves made], "
            "[and a loss that weighs one by the other.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(view, v_lab, bars, b_cap, sc_lab, sc_t, ep_good, ep_bad,
                                                every, a_good, a_bad, arr_bad, b_g, t_g, b_l, t_l, ill4)),
                             m.FadeIn(blk[0]), m.FadeIn(blk[2])),
            m.AnimationGroup(m.FadeIn(rows[:len(lines_src[0].replace(' ', ''))]), m.FadeIn(ann[0])),
            m.AnimationGroup(*[m.FadeIn(rows[sum(len(l.replace(' ', '')) for l in lines_src[:i]):
                                             sum(len(l.replace(' ', '')) for l in lines_src[:i + 1])]) for i in (1, 2)],
                             m.FadeIn(ann[1]), m.FadeIn(ann[2])),
            m.AnimationGroup(m.FadeIn(rows[sum(len(l.replace(' ', '')) for l in lines_src[:3]):]), m.FadeIn(ann[3])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · why learn a policy

    def why_policies(self):
        self.next_section("5 · Why learn a policy", notes=(
            "Sutton & Barto 2018 §13.1. (1) Stochastic optimal policies — Example 13.1, the short corridor with switched "
            "actions: −1 per step, three non-terminal states that look identical, the second one swaps left and right. "
            f"Computed here (a 3 × 3 linear system per probability): best P(right) = {P_BEST:.3f}, value {V_BEST:.2f} "
            "(the book: 'about 0.59', 'about −11.6'); ε-greedy with ε = 0.1 can only choose right or left with "
            f"probability 0.95: {corridor_value(0.95):.1f} and {corridor_value(0.05):.1f} (the book: less than −44 and −82). "
            "(2) Continuous actions: a Gaussian policy (S&B §13.7) — Episode 19, the pendulum's torque in [−2, 2]. "
            "(3) Smooth changes: a small change in θ changes the probabilities smoothly; an argmax over Q can flip "
            "(the curves in card 3 are an illustration)."))
        xs_c = [-4.4, 0.0, 4.4]
        cards = []
        titles = [("1 · Random can be best", "Sutton & Barto, 2018, Ex. 13.1"),
                  ("2 · Actions as numbers", "Sutton & Barto, 2018, §13.7"),
                  ("3 · Smooth changes", "Sutton & Barto, 2018, §13.2")]
        for x, (t_, sub) in zip(xs_c, titles):
            box = card(4.0, 4.5, ACTION, CARD, radius=0.2, stroke=2.5).move_to([x, -0.6, 0])
            head = text(t_, 24, ACTION, weight=m.BOLD).move_to([x, 1.15, 0])
            assert head.width < 3.6
            foot = cite(sub).move_to([x, -2.45, 0])
            cards.append(m.VGroup(box, head, foot))

        # card 1: the corridor
        cx = xs_c[0]
        sq = 0.62
        cells = m.VGroup(*[m.Square(sq).set_stroke(EDGE, 2).set_fill(CARD_HI, 1) for _ in range(4)]).arrange(m.RIGHT, buff=0.1)
        cells.move_to([cx, 0.35, 0])
        cells[3].set_stroke(GOLD, 2.5).set_fill(GOLD_FILL, 1)
        cells[1].set_stroke(RED, 2.5)
        c_labs = m.VGroup(code("S", 20, INK).move_to(cells[0]), code("⇄", 22, RED).move_to(cells[1]), code("G", 20, GOLD).move_to(cells[3]))
        swap = cite("swapped moves").next_to(cells[1], m.DOWN, buff=0.12)
        best = m.VGroup(text("best: right", 22, INK), code(f"{P_BEST:.0%}", 26, ACTION, weight=m.BOLD)).arrange(m.RIGHT, buff=0.18)
        best.move_to([cx, -0.75, 0])
        vals = m.VGroup(code("value ≈ −11.6", 20, GOOD),
                        code("near-certain: ≤ −44", 20, RED)).arrange(m.DOWN, buff=0.1)
        vals.move_to([cx, -1.6, 0])

        # card 2: a Gaussian over torque
        gx = xs_c[1]
        g_ax = m.Line([gx - 1.5, -0.65, 0], [gx + 1.5, -0.65, 0]).set_stroke(FAINT, 2.5)
        tq = np.linspace(-2, 2, 81)
        mu, sd = 0.6, 0.45
        pts = [[gx + 0.75 * t, -0.65 + 1.5 * math.exp(-((t - mu) ** 2) / (2 * sd * sd)), 0] for t in tq]
        bell = m.VMobject().set_points_smoothly([np.array(p) for p in pts]).set_stroke(ACTION, 4)
        bell_fill = m.Polygon(*[np.array(p) for p in pts], [gx + 1.5, -0.65, 0], [gx - 1.5, -0.65, 0], stroke_width=0,
                              fill_color=ACTION, fill_opacity=0.18)
        t_labs = m.VGroup(code("−2", 20, MUTED).move_to([gx - 1.5, -0.95, 0]), code("+2", 20, MUTED).move_to([gx + 1.5, -0.95, 0]),
                          text("torque", 20, MUTED).move_to([gx, -0.95, 0]))
        sample = m.Dot([gx + 0.75 * 0.85, -0.65, 0], radius=0.09, color=INK).set_z_index(3)
        ep19 = pill("episode 19", ACTION, CORAL_FILL, 20, mono=True).move_to([gx, -1.75, 0])

        # card 3: smooth vs argmax (illustration)
        sx = xs_c[2]
        ax3 = m.VGroup(m.Line([sx - 1.5, -1.3, 0], [sx + 1.55, -1.3, 0]), m.Line([sx - 1.5, -1.3, 0], [sx - 1.5, 0.45, 0]))
        ax3.set_stroke(FAINT, 2.5)
        th = np.linspace(-1.5, 1.5, 61)
        smooth = m.VMobject().set_points_smoothly([np.array([sx + t, -1.3 + 1.6 / (1 + math.exp(-2.2 * t)), 0]) for t in th])
        smooth.set_stroke(ACTION, 5)
        step = m.VMobject().set_points_as_corners([np.array([sx - 1.5, -1.22, 0]), np.array([sx, -1.22, 0]),
                                                   np.array([sx, 0.3, 0]), np.array([sx + 1.5, 0.3, 0])]).set_stroke(VALUE, 4)
        step.set_stroke(opacity=0.9)
        leg = m.VGroup(code("probability", 20, ACTION), code("argmax", 20, VALUE)).arrange(m.RIGHT, buff=0.35)
        leg.move_to([sx, -1.75, 0])
        th_lab = tex(r"\theta", 30, MUTED).next_to(ax3[0], m.RIGHT, buff=0.08)

        self.say(
            "[Why learn a policy at all?] [Here are three reasons.]",
            self.new_header("05 · why policies", "Three reasons to learn a policy"),
            m.LaggedStart(*[m.FadeIn(c[0], shift=0.15 * m.UP) for c in cards], lag_ratio=0.2),
        )
        self.say(
            "[First, the best policy can be random]: [in Sutton and Barto's short corridor, going right fifty-nine percent of the time] "
            "[beats any near-certain choice.]",
            m.AnimationGroup(m.FadeIn(cards[0][1:]), m.FadeIn(cells), m.FadeIn(c_labs), m.FadeIn(swap)),
            m.FadeIn(best, shift=0.1 * m.UP),
            m.FadeIn(vals, shift=0.1 * m.UP),
        )
        self.say(
            "[Second, actions can be numbers, like a torque]: [the policy is a bell curve to sample from], "
            "[in episode nineteen.]",
            m.AnimationGroup(m.FadeIn(cards[1][1:]), m.Create(g_ax), m.FadeIn(t_labs)),
            m.AnimationGroup(m.Create(bell), m.FadeIn(bell_fill), m.FadeIn(sample, scale=0.5)),
            m.FadeIn(ep19, shift=0.1 * m.UP),
        )
        self.say(
            "[Third, a small step in theta changes each probability smoothly]; [picking the highest value can flip all at once.]",
            m.AnimationGroup(m.FadeIn(cards[2][1:]), m.Create(ax3), m.FadeIn(th_lab),
                             m.Create(smooth), m.FadeIn(leg[0])),
            m.AnimationGroup(m.Create(step), m.FadeIn(leg[1])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · trained on this Mac

    def on_this_mac(self):
        self.next_section("6 · Trained on this Mac", notes=(
            "research/experiments/deep_rl.md + deep_rl.json (measured on this Mac Studio, M4 Max, 5 seeds × 500,000 env "
            "steps; numpy MLP with hand-written backprop): REINFORCE without a baseline, policy 4-64-64-2 tanh, Adam "
            "1e-3, γ = 0.99, one episode per update, Monte Carlo returns. Solved (avg of the last 100 training episodes "
            f"≥ 475) in {RF['solved_seeds']} seeds after {RF['steps_to_solve_mean']:,.0f} ± {RF['steps_to_solve_std']:,.0f} "
            f"steps (the 3 that solved; {RF['episodes_to_solve_mean']:,.0f} episodes); {FELL} of those 3 later fell below "
            f"300 (min after solving: 112.1 and 132.4); final avg100 {RF['final_avg100_mean']} ± {RF['final_avg100_std']} "
            f"over all 5 (per seed: {', '.join(str(s['final_avg100']) for s in SEEDS)}). {RF['wall_s_total_mean']} s per "
            "run. Curves: avg100 per seed every 2,500 steps. The replay: trajectories.json reinforce_cartpole — seed 10's "
            "final weights (that seed never met the training criterion: last-100 average 157.5; yet these weights "
            f"averaged {SRC['eval']['mean_return']:.0f} over 100 fresh sampled episodes), reset seed 2026, actions sampled, "
            f"500 steps (truncated). First four steps: P(right) = {', '.join(f'{p:.2f}' for p in P_RIGHT[:SLOW])}, moves "
            "R, L, R, L. Comparison not on screen: Episode 13's DQN solved 5/5 (252k ± 118k steps, 70 s per run)."))
        plot = Plot((0, 500), (0, 500), width=6.3, height=3.9, center=(-2.85, -0.45, 0),
                    x_ticks=[0, 100, 200, 300, 400, 500], y_ticks=[0, 100, 200, 300, 400],
                    x_label="thousand environment steps", y_label="score: average of the last 100 episodes")
        lines = m.VGroup(*[plot.line(X_K, c, REWARD, 3).set_stroke(opacity=0.85) for c in PER_SEED]).set_z_index(1)
        th_line = plot.hline(THRESH, MUTED, 2.5).set_z_index(0)
        th_lab = code("475", 20, GREEN, weight=m.BOLD).next_to(plot.coords(0, THRESH), m.LEFT, buff=0.16)
        th_lab.set_z_index(3)
        solve_dots = m.VGroup(*[plot.dot(k, THRESH, GREEN, 0.1) for k, _ in SOLVE_PTS])
        end_bad = m.VGroup(*[plot.dot(500, SEEDS[i]["final_avg100"], RED, 0.09) for i, s in enumerate(SEEDS) if s["solved_at"] is None])
        fell = [i for i, s in enumerate(SEEDS) if s["solved_at"] and s.get("min_avg100_after_solve", 500) < 300]
        assert len(fell) == FELL
        fell_dots = m.VGroup(*[plot.dot(500, SEEDS[i]["final_avg100"], RED, 0.09) for i in fell])

        rx = 1.75
        src = m.VGroup(cite("trained on this Mac"), cite("5 seeds · 500k steps each"),
                       cite("network 4-64-64-2 · no baseline"))
        src.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(src, rx, 1.95)
        s1 = m.VGroup(code("solved (475+): 3 of 5", 24, GREEN, weight=m.BOLD),
                      code(f"{RF['steps_to_solve_mean'] / 1000:.0f}k ± {RF['steps_to_solve_std'] / 1000:.0f}k steps", 22, INK),
                      cite("for the three that solved"))
        s1.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(s1, rx, 0.45)
        s2 = m.VGroup(code("never solved: 2", 22, RED, weight=m.BOLD), code(f"fell back below 300: {FELL} of 3", 22, RED))
        s2.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(s2, rx, -0.95)
        s3 = m.VGroup(code(f"final: {RF['final_avg100_mean']:.0f} ± {RF['final_avg100_std']:.0f}", 26, REWARD, weight=m.BOLD),
                      cite("last 100 episodes · mean ± std, 5 seeds"))
        s3.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(s3, rx, -2.05)
        noisy = pill("noisy", RED, RED_FILL, 24).move_to([0, -3.0, 0]).align_to([rx, 0, 0], m.LEFT)

        # the replay
        view = CartPoleView(center=(-2.4, -1.25, 0), units_per_m=0.95, pole_units=2.0)
        view.set_state(STATES[0][0], STATES[0][2])
        rbars = prob_bars([1 - P_RIGHT[0], P_RIGHT[0]], center=(4.15, -1.05, 0), width=2.5, height=2.3)
        rb_lab = text("its dice, this step", 22, ACTION).next_to(rbars, m.UP, buff=0.35)
        r_src = m.VGroup(cite("one run's final network (seed 10) · one episode, every move sampled"),
                         cite(f"these weights averaged {SRC['eval']['mean_return']:.0f} over 100 fresh episodes"))
        r_src.arrange(m.DOWN, buff=0.08).move_to([-2.4, -2.75, 0])
        step_t = [code(f"step {t}", 24, INK) for t in range(TRAJ["length"] + 1)]
        ret_t = [code(f"return {t}", 24, REWARD) for t in range(TRAJ["length"] + 1)]
        for a_, b_ in zip(step_t, ret_t):
            right_at(a_, -2.65, 1.55)
            left_at(b_, -2.15, 1.55)
        step_slot, ret_slot = step_t[0].copy(), ret_t[0].copy()
        push = push_arrow(view, ACTS[0], 0.7)
        dice = icon(DICE, 0.55).next_to(rb_lab, m.LEFT, buff=0.2)
        done = pill("500 steps · the limit", GREEN, GREEN_FILL, 22).move_to([-2.4, 2.2, 0])

        slow_live = m.VGroup(view, rbars.bars, rbars.numbers, push, step_slot, ret_slot)

        def slow(_, alpha):
            i = alpha * SLOW
            j = min(int(math.floor(i)), SLOW - 1)
            f = i - j
            x = (1 - f) * STATES[j][0] + f * STATES[j + 1][0]
            th_ = (1 - f) * STATES[j][2] + f * STATES[j + 1][2]
            view.set_state(x, th_)
            set_bars(rbars, [1 - P_RIGHT[j], P_RIGHT[j]])
            push.become(push_arrow(view, ACTS[j], 0.7))
            k = min(int(math.floor(i + 1e-9)), SLOW)
            step_slot.become(step_t[k])
            ret_slot.become(ret_t[k])

        fast_live = m.VGroup(view, step_slot, ret_slot)
        slow_play = m.UpdateFromAlphaFunc(slow_live, slow, run_time=3.2, rate_func=m.linear)
        slow_play.introducer = True          # its parts fade in on earlier cues of the same line

        def fast(_, alpha):
            n = TRAJ["length"] - SLOW
            i = SLOW + alpha * n
            j = min(int(math.floor(i)), TRAJ["length"] - 1)
            f = i - j
            view.set_state((1 - f) * STATES[j][0] + f * STATES[j + 1][0], (1 - f) * STATES[j][2] + f * STATES[j + 1][2])
            k = min(int(math.floor(i + 1e-9)), TRAJ["length"])
            step_slot.become(step_t[k])
            ret_slot.become(ret_t[k])

        self.say(
            "[We trained REINFORCE on this Mac]: [the same small network], [five seeds, half a million steps each.]",
            m.AnimationGroup(self.new_header("06 · on this Mac", "Trained on this Mac"), m.FadeIn(plot)),
            m.FadeIn(src, shift=0.1 * m.DOWN),
            m.LaggedStart(*[m.Create(l_, run_time=2.4) for l_ in lines], lag_ratio=0.12),
        )
        self.say(
            "[Three of the five seeds solved it]: [a hundred-episode average of four hundred seventy-five], "
            "[after about two hundred sixty thousand steps.]",
            m.FadeIn(s1[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(th_line), m.FadeIn(th_lab), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in solve_dots], lag_ratio=0.25)),
            m.FadeIn(s1[1:], shift=0.1 * m.UP),
        )
        self.say(
            "[The other two never got there], [and two of the three that did fell back.]",
            m.AnimationGroup(m.FadeIn(end_bad, scale=0.5), m.FadeIn(s2[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(fell_dots, scale=0.5), m.FadeIn(s2[1], shift=0.1 * m.UP)),
        )
        self.say(
            "[At the end, the last hundred episodes averaged two hundred twenty-nine], "
            "[give or take a hundred forty-seven]: [noisy.]",
            m.FadeIn(s3[0], shift=0.1 * m.UP),
            m.FadeIn(s3[1], shift=0.1 * m.UP),
            m.FadeIn(noisy, scale=0.8),
        )
        self.say(
            "[Here's one run's final network, playing one episode.] [Fifty times a second, it rolls the dice]: "
            "[push right, push left, push right, push left.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(plot, lines, th_line, th_lab, solve_dots, end_bad, fell_dots, src, s1, s2, s3, noisy)),
                             m.FadeIn(view, shift=0.15 * m.UP), m.FadeIn(r_src), m.FadeIn(step_slot), m.FadeIn(ret_slot)),
            m.AnimationGroup(m.FadeIn(rbars, shift=0.15 * m.UP), m.FadeIn(rb_lab), m.FadeIn(dice, scale=0.6), m.FadeIn(push)),
            slow_play,
        )
        self.say(
            "[It keeps rolling, and the pole stays up], [for all five hundred steps.]",
            m.Succession(m.AnimationGroup(m.FadeOut(m.VGroup(rbars, rb_lab, dice, push)), run_time=0.4),
                         m.UpdateFromAlphaFunc(fast_live, fast, run_time=5.0, rate_func=m.linear)),
            m.FadeIn(done, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · the catch

    def the_catch(self):
        self.next_section("7 · The catch: noise", notes=(
            "Each push is G × (one-hot − π): its size is a whole episode's return, one draw from a wide distribution. The "
            "eight arrows are the first moves of eight episodes of the untrained policy (simulated here: reset seeds "
            f"3000–3007, default_rng(16)); returns {', '.join(f'{g:.0f}' for g, _ in FIRST8)}, first moves "
            f"{''.join('LR'[a] for _, a in FIRST8)}; arrow = G × 0.5, up for a push toward left, down toward right. "
            "Williams (1992), Theorem 1: the expected update is the gradient (unbiased); S&B §13.3: 'as a Monte Carlo "
            "method REINFORCE may be of high variance and thus produce slow learning.' Episode 17 preview (illustration, "
            f"the section-4 state after the good episode, π = ({P1[0]:.2f}, {P1[1]:.2f})): the bad episode (pushed right, "
            f"G = {G_BAD:.0f}) without a baseline raises right to {P2[1]:.3f}; with a baseline b = {B:.0f} (the mean of the "
            f"two returns, e.g.) the push is (G − b) = {G_BAD - B:.0f}, and right falls to {P2B[1]:.3f}."))
        base_y = -0.4
        xs = [-4.95 + 0.62 * i for i in range(8)]
        scale = 0.035
        arrows = m.VGroup()
        chips = m.VGroup()
        for x, (g, a) in zip(xs, FIRST8):
            d = (1 if a == LEFT else -1) * scale * g
            arrows.add(m.Arrow([x, base_y, 0], [x, base_y + d, 0], buff=0, color=SIGNAL, stroke_width=6,
                               max_tip_length_to_length_ratio=0.25))
            chips.add(code(f"{g:.0f}", 20, REWARD).move_to([x, -2.55, 0]))
        axis = m.Line([-5.3, base_y, 0], [-0.3, base_y, 0]).set_stroke(FAINT, 2)
        up_lab = left_at(text("toward left", 20, MUTED), -6.55, base_y + 0.9)
        dn_lab = left_at(text("toward right", 20, MUTED), -6.55, base_y - 0.9)
        g_lab = left_at(text("return", 20, REWARD), -6.55, -2.55)
        cap = cite("first moves of 8 untrained episodes · simulated here").move_to([-3.4, -3.25, 0])
        ew = m.VGroup(tex(r"\mathbb{E}[\,\mathrm{push}\,] = \nabla_\theta J(\theta)", 36, INK), cite("Williams, 1992, Theorem 1"))
        ew.arrange(m.DOWN, buff=0.12).move_to([-2.8, 2.0, 0])

        # Episode 17 preview: the bad episode with and without a baseline
        px_ = [2.15, 5.0]
        start = [prob_bars(P1, center=(x, -1.0, 0), width=2.1, height=2.0, size=20) for x in px_]
        heads = m.VGroup(text("plain", 22, INK, weight=m.BOLD).move_to([px_[0], 1.15, 0]),
                         text("with a baseline", 22, SIGNAL, weight=m.BOLD).move_to([px_[1], 1.15, 0]))
        subs = m.VGroup(code(f"G = {G_BAD:.0f}", 20, REWARD).move_to([px_[0], 0.7, 0]),
                        code(minus(f"G - b = {G_BAD:.0f} - {B:.0f}"), 20, REWARD).move_to([px_[1], 0.7, 0]))
        bad = cite("the bad episode again: pushed right").move_to([3.6, 1.75, 0])
        ill7 = cite(f"illustration · b = {B:.0f}, the mean of {G_GOOD:.0f} and {G_BAD:.0f}").move_to([3.6, -2.8, 0])
        up = nudge(start[0], RIGHT, PUSH_BAD, P2[RIGHT], scale=6.0).set_z_index(3)
        dn = nudge(start[1], RIGHT, ALPHA * (G_BAD - B) * SC1[RIGHT], P1[RIGHT], scale=6.0).set_z_index(3)
        ep17 = pill("episode 17", SIGNAL, PINK_FILL, 20, mono=True).move_to([3.6, -3.25, 0])
        divider = m.DashedLine([0.75, 2.0, 0], [0.75, -3.3, 0], dash_length=0.1).set_stroke(EDGE, 2)

        self.say(
            "[Why so noisy?] [Each push is scaled by a whole episode's return]: [one roll of the dice.]",
            m.AnimationGroup(self.new_header("07 · the catch", "Noisy, but right on average"), m.Create(axis),
                             m.FadeIn(up_lab), m.FadeIn(dn_lab), m.FadeIn(g_lab), m.FadeIn(cap)),
            m.LaggedStart(*[m.AnimationGroup(m.GrowArrow(a_), m.FadeIn(c_)) for a_, c_ in zip(arrows, chips)], lag_ratio=0.25,
                          run_time=2.6),
            m.Circumscribe(m.VGroup(arrows[0], arrows[1]), color=SIGNAL, buff=0.08),
        )
        self.say(
            "[On average, the pushes point the right way: Williams proved it.] [But each one alone is mostly noise.]",
            m.FadeIn(ew, shift=0.1 * m.DOWN),
            m.AnimationGroup(*[m.Indicate(a_, color=SIGNAL, scale_factor=1.08) for a_ in arrows]),
        )
        self.say(
            "[The fix comes next: measure each return against what's typical], "
            "[so a bad episode pushes its move down instead], [and the noise drops.]",
            m.AnimationGroup(m.Create(divider), m.FadeIn(bad), m.FadeIn(heads), m.FadeIn(subs), *[m.FadeIn(b_) for b_ in start]),
            m.AnimationGroup(m.GrowArrow(up), start[0].set(P2, run_time=1.0), m.GrowArrow(dn), start[1].set(P2B, run_time=1.0)),
            m.AnimationGroup(m.FadeIn(ep17, shift=0.1 * m.UP), m.FadeIn(ill7)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · recap

    def recap(self):
        self.next_section("8 · Recap", notes=(
            "Sample (act by drawing moves from π_θ, a softmax over the network's scores), score (each episode's return; "
            "J(θ) = the expected return), nudge (θ += α G_t ∇ log π_θ(a_t|s_t), the log-derivative trick — Williams "
            "1992). Measured here: REINFORCE without a baseline solved CartPole in 3 of 5 seeds; noisy → Episode 17."))
        halo = glow(ACTION, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Sample", 54, ACTION), title_text("Score", 54, REWARD), title_text("Nudge", 54, SIGNAL))
        words.arrange(m.RIGHT, buff=1.5).move_to([0, 1.1, 0])
        on_baseline(words)
        subs = m.VGroup(*[m.VGroup(text(a, 22, MUTED), text(b, 22, MUTED)).arrange(m.DOWN, buff=0.08)
                          for a, b in (("moves from", "the policy"), ("each episode", "by its return"),
                                       ("log-probabilities,", "in proportion"))])
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), 0, 0]).align_to([0, words[1].get_bottom()[1] - 0.3, 0], m.UP)
        assert all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        res = formula([r"\nabla_\theta", r"J(\theta)", "=", r"\mathbb{E}", r"\Big[\sum_t", r"G_t", r"\,\nabla_\theta",
                       r"\log \pi_\theta(a_t \mid s_t)", r"\Big]"],
                      [SIGNAL, REWARD, INK, INK, INK, REWARD, SIGNAL, ACTION, INK], 40)
        res.move_to([0, -1.45, 0])
        proof = code("CartPole on this Mac: solved in 3 of 5 seeds · noisy → episode 17", 22, INK,
                     t2c={r"3 of 5": GREEN, r"noisy": RED, r"episode 17": SIGNAL})
        proof.move_to([0, -2.45, 0])
        mascot = rover(0.6).move_to([0, -3.25, 0])

        self.say(
            "[To sum up]: [sample], [score], [nudge.]",
            m.AnimationGroup(self.new_header("08 · recap", "Sample, score, nudge"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Sample moves from the policy], [score each episode by its return], "
            "[and nudge each move's log-probability in proportion.]",
            m.AnimationGroup(m.FadeIn(res, shift=0.1 * m.UP), m.Circumscribe(words[0], color=ACTION, buff=0.12)),
            m.Circumscribe(words[1], color=REWARD, buff=0.12),
            m.AnimationGroup(m.Circumscribe(words[2], color=SIGNAL, buff=0.12), m.FadeIn(proof, shift=0.1 * m.UP),
                             m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()
