"""Trial and Reward · Episode 13 · Deep Q-Networks (Module 4 · Deep Value Learning)

A neural network can learn Q-values if you stabilise the targets. The running example is CartPole, learned by a DQN
on this Mac: a random pusher drops the pole after 30 steps, the trained network balances it for all 500. Episode 10's
Q-learning with a network instead of a table (four numbers in, one Q-value per action out; the target r + γ max Q);
why it breaks done naively (back-to-back steps are near copies; the target moves with the guess: Episode 12's deadly
triad); the two stabilisers — experience replay (a memory filling and sampled at random) and a target network (a frozen
copy refreshed every 1,000 steps); ε decay; the whole loop. Then the runs measured here (5 seeds × 500k steps): DQN
solved CartPole 5/5, without replay 0/5, without a target network 0/5, the Q-values (≈ 113, ≈ 202, ≈ 4 × 10⁹ against a
ceiling of 100) and the honest wobble (2 of 5 seeds fell back below 300). Mnih et al. 2015: 49 Atari games, ≥ 75% of
the human score on 29, the human = one professional tester after ~2 h of practice per game, Montezuma's Revenge 0,
the paper's own replay/target ablation on Breakout. Recap: store, sample, freeze. Episode 14: overestimation.

Every number on screen is computed below from rl_reference.py and asserted, loaded with measured(...), or cited.

Sources (facts dated October 2026):
- research/experiments/deep_rl.md + deep_rl.json (measured("deep_rl", ...)): headline.dqn / dqn_no_replay /
  dqn_no_target, methods.dqn.hp (4-256-256-2 ReLU, γ 0.99, Adam 2.5e-4, Huber, buffer 100k, batch 128, one gradient
  step every 4 env steps, target copied every 1,000 steps, ε 1 → 0.01 over 20k steps, 500k steps), methods.dqn.per_seed,
  curves.*.avg100 (mean of 5 seeds) / avg100_per_seed / q_batch_mean, overestimation.value_ceiling (100 = 1/(1 − γ)).
  Seeds 10–14, mean ± sample std. "Solved" = mean return of the last 100 training episodes ≥ 475 (CartPole-v1's
  registered threshold; notes Appendix A row 12).
- research/experiments/trajectories.json (measured("trajectories", "policies", ...)): dqn_cartpole (greedy, reset seed
  2026, 500 steps, truncated) and random_cartpole (an untrained network sampling ~50/50, 30 steps, terminated);
  weights/dqn_cartpole.npz via load_policy (the exported network: seed 10, best snapshot after 410k steps).
- research/notes.md §13 (Mnih et al. 2015, Nature 518:529–533): replay of the last 1,000,000 transitions, minibatch 32,
  target network refreshed every C = 10,000 parameter updates, ε 1 → 0.1 over the first million frames, 84×84×4 input,
  conv 32-64-64 + FC 512, 4–18 actions, a separate network per game with the same architecture and hyperparameters;
  ≥ 75% of the human score on 29 of 49 games ("more than half of the games (29 games…)"), ≥ 100% on 23 (notes'
  recount of Extended Data Table 2); the human = one professional tester, ~20 episodes of ≤ 5 min after ~2 h of
  practice per game; Video Pinball 2,539.4% (best), Breakout 1,327.2%, Montezuma's Revenge 0% (worst); Extended Data
  Table 3 (Breakout: 316.8 with replay + target, 240.7 replay only, 10.2 target only, 3.2 neither). The three causes of
  instability and the quotes ("known to be unstable or even to diverge…", "which motivates the choice of Q-learning").
  Experience replay: Lin (1992). Appendix C row 13 checked (1M, C = 10,000, 29 of 49, one tester ~2 h).

Illustrative (said "for example" or labelled "illustration" on screen): the two chase sketches (a target that moves
with every update; a target frozen between copies) use made-up numbers to show the shape, not data; which buffer slots
the random minibatch picks (default_rng(13)). The Q-table in section 2 is Rover's world from Episodes 7 and 10, its
values computed with value_iteration + q_from_v. The 7 × 7 Atari grid shows counts only (sorted), not per-game rows;
only its first square (Video Pinball, best) and last (Montezuma's Revenge, worst) are named.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep13"
DEEP = "deep_rl"

# ---------------------------------------------------------------------------- measured on this Mac

HL = measured(DEEP, "headline", "dqn")
NR = measured(DEEP, "headline", "dqn_no_replay")
NT = measured(DEEP, "headline", "dqn_no_target")
HP = measured(DEEP, "methods", "dqn", "hp")
PER_SEED = measured(DEEP, "methods", "dqn", "per_seed")

assert HP["hidden"] == [256, 256] and HP["activation"] == "relu" and HP["gamma"] == 0.99
assert HP["buffer_size"] == 100_000 and HP["batch_size"] == 128 and HP["train_every"] == 4
assert HP["target_update_every"] == 1_000 and HP["total_steps"] == 500_000
assert (HP["eps_start"], HP["eps_end"], HP["eps_decay_steps"]) == (1.0, 0.01, 20_000)
SEEDS = 5
assert HL["solved_seeds"] == "5/5" and NR["solved_seeds"] == "0/5" and NT["solved_seeds"] == "0/5" and len(PER_SEED) == SEEDS

STEPS_K, STEPS_SD_K = round(HL["steps_to_solve_mean"] / 1000), round(HL["steps_to_solve_std"] / 1000)
assert (STEPS_K, STEPS_SD_K) == (252, 118)                       # said: "about two hundred fifty thousand"
EPISODES_TO_SOLVE, WALL_S = HL["episodes_to_solve_mean"], HL["wall_s_to_solve_mean"]
assert (EPISODES_TO_SOLVE, WALL_S) == (1376.0, 35.0)              # said: "about fourteen hundred", "thirty-five seconds"
FINAL, FINAL_SD = round(HL["final_avg100_mean"]), round(HL["final_avg100_std"])
assert (FINAL, FINAL_SD) == (283, 184)
DROPPED = [s for s in PER_SEED if s["min_avg100_after_solve"] < 300]
assert HL["seeds_dropping_below_300_after_solving"] == len(DROPPED) == 2      # said: "two of the five"
NR_BEST, NR_Q = round(NR["best_avg100_mean"]), round(NR["q_batch_mean_final_mean"])
assert (NR_BEST, NR_Q) == (157, 202)                               # said: "a hundred fifty-seven", "about two hundred"
NT_LEN, NT_Q = round(NT["final_avg100_mean"]), NT["q_batch_mean_final_mean"]
assert NT_LEN == 9 and round(NT_Q / 1e9) == 4                      # said: "about nine steps", "about four billion"
DQN_Q = round(HL["q_batch_mean_final_mean"])
assert DQN_Q == 113
CEIL = measured(DEEP, "headline", "overestimation", "value_ceiling")
assert CEIL == 100.0 and abs(1 / (1 - HP["gamma"]) - CEIL) < 1e-9 and DQN_Q > CEIL and NR_Q / CEIL > 2
USES = HP["batch_size"] // HP["train_every"]                       # expected replays of one memory (derived)
assert USES == 32                                                  # said: "about thirty-two times"
SOLVED = 475

CURVES = {k: measured(DEEP, "curves", k) for k in ("dqn", "dqn_no_replay", "dqn_no_target")}
X_K = [x / 1000 for x in CURVES["dqn"]["x_steps"]]
assert all(CURVES[k]["x_steps"] == CURVES["dqn"]["x_steps"] for k in CURVES) and X_K[-1] == 500
SEED_CURVES = CURVES["dqn"]["avg100_per_seed"]
assert [round(c[-1], 1) for c in SEED_CURVES] == [round(s["final_avg100"], 1) for s in PER_SEED]
assert all(max(c) >= SOLVED for c in SEED_CURVES)                                  # every seed got there
assert max(max(c) for c in CURVES["dqn_no_replay"]["avg100_per_seed"]) < SOLVED
assert max(max(c) for c in CURVES["dqn_no_target"]["avg100_per_seed"]) < SOLVED
DROP_IDX = [i for i, s in enumerate(PER_SEED) if s in DROPPED]
Q_CURVES = {k: CURVES[k]["q_batch_mean"] for k in CURVES}
assert Q_CURVES["dqn"]["x_steps"] == Q_CURVES["dqn_no_target"]["x_steps"]
Q_X_K = [x / 1000 for x in Q_CURVES["dqn"]["x_steps"]]

# ---------------------------------------------------------------------------- the two pole balancers (replayed, not simulated)

POL = measured("trajectories", "policies")
GOOD = np.array(POL["dqn_cartpole"]["states"])
BAD = np.array(POL["random_cartpole"]["states"])
assert POL["dqn_cartpole"]["length"] == 500 and POL["dqn_cartpole"]["truncated"] and len(GOOD) == 501
assert POL["random_cartpole"]["length"] == 30 and POL["random_cartpole"]["terminated"] and len(BAD) == 31
assert abs(BAD[-1][2]) > CARTPOLE["theta_threshold"] and CARTPOLE["max_steps"] == 500   # fell past 12 degrees
assert np.abs(cartpole_reset(np.random.default_rng(2026)) - GOOD[0]).max() < 1e-4        # both start from reset seed 2026
assert np.abs(GOOD[0] - BAD[0]).max() < 1e-9

NET = load_policy("dqn_cartpole")
with np.errstate(all="ignore"):
    Q_ALL = np.array([NET(cartpole_obs(s)) for s in GOOD[:-1]])
assert np.isfinite(Q_ALL).all()
assert np.mean(np.argmax(Q_ALL, axis=1) == np.array(POL["dqn_cartpole"]["actions"])) > 0.99   # the replay is this network
K_LEAN = 18                                                       # the pole leans right, about 9 degrees
S_LEAN = GOOD[K_LEAN]
Q_LEFT, Q_RIGHT = (float(v) for v in Q_ALL[K_LEAN])
assert 0.15 < S_LEAN[2] < 0.17 and Q_RIGHT > Q_LEFT and Q_RIGHT < CEIL
assert POL["dqn_cartpole"]["actions"][K_LEAN] == 1 and (round(Q_LEFT, 2), round(Q_RIGHT, 2)) == (99.61, 99.89)
N_PARAMS = sum(W.size + b.size for W, b in NET.params)
assert [W.shape for W, _ in NET.params] == [(4, 256), (256, 256), (256, 2)] and N_PARAMS == 67_586
STRIP = list(range(40, 46))                                       # six steps in a row, 0.1 s of play
assert abs(CARTPOLE["tau"] * (len(STRIP) - 1) - 0.1) < 1e-12

# Rover's world (Episodes 5–10): a Q-table, computed
V_STAR, _ = value_iteration(ROVER_WORLD)
Q_TAB = q_from_v(ROVER_WORLD, V_STAR)
TAB_CELLS = [(1, 1), (2, 1), (3, 1), (1, 2)]
assert ROVER_WORLD.actions == ("N", "E", "S", "W") and round(Q_TAB[(1, 1)]["N"], 2) == 0.71

# ---------------------------------------------------------------------------- Mnih et al. 2015 (notes §13, cited)

ATARI, AT_75, AT_100 = 49, 29, 23
assert AT_100 <= AT_75 < ATARI and AT_75 > ATARI / 2                # "more than half of the games (29 games)"
REPLAY_2015, BATCH_2015, C_2015 = 1_000_000, 32, 10_000
EPS_2015 = (1.0, 0.1, 1_000_000)
BREAKOUT_PCT = 100 * (401.2 - 1.7) / (31.8 - 1.7)                  # Extended Data Table 2 (human 31.8, random 1.7)
assert round(BREAKOUT_PCT, 1) == 1327.2
PINBALL_PCT, MONTEZUMA_PCT = 2539.4, 0.0
ABLATION = [("replay + target", 316.8), ("replay only", 240.7), ("target only", 10.2), ("neither", 3.2)]
assert round(ABLATION[0][1] / ABLATION[-1][1]) == 99
CONV = [(32, 8 * 8 * 4), (64, 4 * 4 * 32), (64, 3 * 3 * 64), (512, 7 * 7 * 64), (18, 512)]
ATARI_PARAMS = sum(out * (inp + 1) for out, inp in CONV)
assert ATARI_PARAMS == 1_693_362                                    # notes §13 worked example (18 actions)

# ---------------------------------------------------------------------------- look

LOCK = recolored("lock", VIOLET, VIOLET_FILL, prefix=PREFIX)
DICE = recolored("dice", EXPLORE, VIOLET_FILL, prefix=PREFIX)
PAD = recolored("gamepad", VIOLET, VIOLET_FILL, prefix=PREFIX)
FULL_C, NR_C, NT_C = GREEN, MUTED, RED        # the three runs: both tricks (good), no replay, no target (diverged)


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def left_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.RIGHT)


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


def backward(net, color=SIGNAL, run_time=1.2):
    """A gradient step: a PINK flash sweeping from the outputs back to the inputs."""
    r = net.columns[0][0].width / 2
    flashes = []
    for a, b in reversed(list(zip(net.columns[:-1], net.columns[1:]))):
        lines = m.VGroup(*[m.Line(q.get_center(), p.get_center(), buff=r).set_stroke(color, 3) for p in a for q in b])
        flashes.append(m.ShowPassingFlash(lines, time_width=0.6))
    return m.LaggedStart(*flashes, lag_ratio=0.7, run_time=run_time)


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


def mini_pole(state, w=0.8, h=0.95):
    """A thumbnail of one CartPole state (cart x, pole angle), true angle."""
    box = card(w, h, EDGE, CARD, radius=0.08, stroke=1.5)
    y = box.get_bottom()[1] + 0.2
    track = m.Line([-w / 2 + 0.08, y, 0], [w / 2 - 0.08, y, 0]).set_stroke(FAINT, 2)
    cx = 0.25 * state[0]
    cart = m.Rectangle(width=0.26, height=0.11).set_stroke(STATE, 2).set_fill(TEAL_FILL, 1).move_to([cx, y + 0.07, 0])
    base = np.array([cx, y + 0.12, 0])
    pole = m.Line(base, base + 0.58 * np.array([np.sin(state[2]), np.cos(state[2]), 0])).set_stroke(ACTION, 4)
    return m.VGroup(box, track, cart, pole)


class Stage:
    """One card with a CartPole, its title and a step counter."""

    def __init__(self, cx, title, sub, color, cy=0.2, w=6.3, h=4.1):
        self.box = card(w, h, color, CARD, radius=0.2, stroke=2.5).move_to([cx, cy, 0])
        top, left = cy + h / 2, cx - w / 2 + 0.35
        self.title = left_at(text(title, 26, INK, weight=m.BOLD), left, top - 0.42)
        self.sub = left_at(text(sub, 20, MUTED), left, top - 0.85)
        self.view = CartPoleView(center=(cx, cy - 1.15, 0), units_per_m=0.55, pole_units=1.45)
        self.left, self.right_x, self.count_y = left, cx + w / 2 - 0.35, cy - h / 2 + 0.42
        self.counter = self.count(0)

    def count(self, n):
        return left_at(code(f"step {n}", 22, INK), self.left, self.count_y)

    def everything(self):
        return m.VGroup(self.box, self.title, self.sub, self.view, self.counter)


def chase_plot(center, width=3.8, height=1.5, top=10):
    return Plot((0, 8), (0, top), width=width, height=height, center=center, x_label="updates", y_label="value", size=20)


def moving_target():
    """Illustration: the target is computed by the network being trained, so each update lifts it too."""
    g, t = [1.0], [3.0]
    for _ in range(8):
        g.append(g[-1] + 0.5 * (t[-1] - g[-1]))
        t.append(t[-1] + 1.0 * (g[-1] - g[-2]) + 0.25)
    return g, t


def frozen_target():
    """Illustration: the target stays put between copies; at each copy it is recomputed from the caught-up guess."""
    g, t = [1.0], [4.0]
    for k in range(1, 9):
        if k % 3 == 0:
            t.append(3.0 + 0.5 * g[-1])
        else:
            t.append(t[-1])
        g.append(g[-1] + 0.62 * (t[-2] - g[-1]))
    return g, t


G_MOVE, T_MOVE = moving_target()
G_FROZ, T_FROZ = frozen_target()
MOVE_TOP, FROZ_TOP = 17, 7
assert T_MOVE[-1] - G_MOVE[-1] > T_MOVE[0] - G_MOVE[0] and max(T_MOVE) < MOVE_TOP   # the gap never closes
assert abs(T_FROZ[-1] - G_FROZ[-1]) < 0.5 and max(T_FROZ) < FROZ_TOP


# ---------------------------------------------------------------------------- the episode


class Ep13DeepQNetworks(Episode):
    number = 13

    def lesson(self):
        self.two_carts()
        self.network_for_q()
        self.why_it_breaks()
        self.experience_replay()
        self.target_net()
        self.the_loop()
        self.on_this_mac()
        self.runaway_values()
        self.atari()
        self.recap()

    # ------------------------------------------------------------------ 1 · two carts

    def two_carts(self):
        self.next_section("1 · Thirty steps or five hundred", notes=(
            "Two CartPoles from the same start (reset seed 2026; trajectories.json, replayed, not simulated). Left: "
            "random_cartpole, an untrained network pushing left or right about 50/50: the pole passes 12° after 30 "
            "steps (terminated). Right: dqn_cartpole, the DQN trained on this Mac (seed 10, best snapshot after 410k "
            "steps; greedy): all 500 steps, CartPole-v1's time limit (truncated). Playback is not real time (the random "
            "episode is slowed down, the trained one sped up a little). DQN = Q-learning (Episode 10) + experience replay "
            "+ a target network (Mnih et al. 2015)."))
        left = Stage(-3.45, "pushes at random", "an untrained network, 50/50", EDGE)
        right = Stage(3.45, "a trained network", "deep Q-network · trained on this Mac", VIOLET)
        left.view.set_state(BAD[0][0], BAD[0][2])
        right.view.set_state(GOOD[0][0], GOOD[0][2])
        bot = rover(0.42).move_to([right.box.get_right()[0] - 0.65, right.box.get_top()[1] - 0.6, 0])
        fell = right_at(pill(f"fell · {len(BAD) - 1} steps", RED, RED_FILL, 20), left.right_x, left.count_y)
        held = right_at(pill(f"{len(GOOD) - 1} steps · the limit", GREEN, GREEN_FILL, 20), right.right_x, right.count_y)

        p_q = pill("Q-learning · ep 10", VALUE, BLUE_FILL, 21)
        p_rep = pill("experience replay", STATE, TEAL_FILL, 21)
        p_tgt = pill("target network", VIOLET, VIOLET_FILL, 21)
        p_dqn = pill("deep Q-network", VIOLET, VIOLET_FILL, 22)
        signs = [code(s, 38, MUTED, weight=m.BOLD) for s in ("+", "+", "=")]
        eq = m.VGroup(p_q, signs[0], p_rep, signs[1], p_tgt, signs[2], p_dqn).arrange(m.RIGHT, buff=0.18)
        eq.move_to([0, -2.85, 0])
        p_dqn[0].set_stroke(VIOLET, 3.5)

        self.say(
            "[Two carts, two poles.] [The left one pushes at random]: [its pole falls after thirty steps.]",
            m.AnimationGroup(self.new_header("01 · two carts", "Thirty steps or five hundred"),
                             m.FadeIn(left.everything(), shift=0.1 * m.UP), m.FadeIn(right.everything(), shift=0.1 * m.UP)),
            m.Circumscribe(left.title, color=MUTED, buff=0.08),
            m.Succession(play(left.view, BAD, 1.6, left.counter, left.count),
                         topple(left.view, BAD[-1][0], BAD[-1][2]),
                         m.FadeIn(fell, shift=0.1 * m.LEFT, run_time=0.4)),
        )
        self.say(
            "[The right one is a neural network, trained on this Mac. "
            "It balances for all five hundred steps, the most CartPole allows.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.15 * m.DOWN, run_time=0.6),
                             m.Succession(play(right.view, GOOD, 6.0, right.counter, right.count),
                                          m.FadeIn(held, shift=0.1 * m.LEFT, run_time=0.4))),
        )
        self.say(
            "[It learned with Q-learning from episode ten], [plus two tricks that keep it steady]: [a deep Q-network.]",
            m.FadeIn(p_q, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(signs[0]), m.FadeIn(p_rep, shift=0.1 * m.UP), m.FadeIn(signs[1]),
                             m.FadeIn(p_tgt, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(signs[2]), m.FadeIn(p_dqn, scale=0.8)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · a network for Q

    def network_for_q(self):
        self.next_section("2 · A network for Q", notes=(
            "Episode 10's Q-table, shown on Rover's world (value_iteration + q_from_v, γ = 1; four of its cells). "
            "CartPole's state (cartpole_obs): cart position (m), cart velocity, pole angle (rad), pole angular velocity — "
            f"step {K_LEAN} of the trained episode: " + ", ".join(f"{v:+.3f}" for v in S_LEAN) + ". The exported DQN "
            f"(load_policy('dqn_cartpole'), 4-256-256-2 ReLU, {N_PARAMS:,} parameters) gives Q(s, left) = {Q_LEFT:.2f}, "
            f"Q(s, right) = {Q_RIGHT:.2f}; it pushes right (the recorded action). Q-values are discounted returns with "
            "γ = 0.99. Target y = r + γ max_a′ Q(s′, a′); TD error δ = y − Q(s, a); the loss is a Huber loss of δ "
            "(squared for small errors), minimised by a gradient step (Mnih et al. 2015: L = (y − Q)², error clipped)."))
        # the Q-table of episode 10 (Rover's world)
        tx, ty = 1.0, 0.9
        cols = [tx - 1.55 + 0.95 * k for k in range(5)]
        head = m.VGroup(code("state", 20, MUTED).move_to([cols[0], ty + 0.85, 0]),
                        *[code(a, 22, ACTION, weight=m.BOLD).move_to([cols[k + 1], ty + 0.85, 0])
                          for k, a in enumerate(ROVER_WORLD.actions)])
        rows = m.VGroup()
        for r, s in enumerate(TAB_CELLS):
            y = ty + 0.3 - 0.5 * r
            rows.add(m.VGroup(code(f"({s[0]}, {s[1]})", 20, STATE).move_to([cols[0], y, 0]),
                              *[code(f"{Q_TAB[s][a]:.2f}", 20, VALUE).move_to([cols[k + 1], y, 0])
                                for k, a in enumerate(ROVER_WORLD.actions)]))
        dots = code("⋮", 24, MUTED).move_to([cols[0], ty + 0.3 - 0.5 * len(TAB_CELLS) + 0.05, 0])
        frame = m.SurroundingRectangle(m.VGroup(head, rows, dots), buff=0.25, corner_radius=0.14).set_stroke(EDGE, 2)
        frame.set_fill(CARD, 0)
        tab_title = cite("episode 10 · Rover's world · a Q-table").next_to(frame, m.UP, buff=0.15)
        table = m.VGroup(frame, head, rows, dots, tab_title)
        row_hl = m.SurroundingRectangle(rows[0], buff=0.06, corner_radius=0.08).set_stroke(STATE, 3)
        col_hl = m.SurroundingRectangle(m.VGroup(head[1], *[rw[1] for rw in rows]), buff=0.06, corner_radius=0.08).set_stroke(ACTION, 3)
        infinite = text("one row per state? CartPole has endless states", 22, RED).next_to(frame, m.DOWN, buff=0.22)
        infinite.shift(0.35 * m.RIGHT)

        # CartPole and its four numbers (left column)
        view = CartPoleView(center=(-4.4, 1.25, 0), units_per_m=0.45, pole_units=1.1)
        view.set_state(S_LEAN[0], S_LEAN[2])
        names = ("position", "speed", "angle", "spin")
        net = NeuralNet(layers=(4, 256, 256, 2), width=2.7, height=2.5, center=(0.9, -0.55, 0), color=VIOLET,
                        radius=0.11, max_nodes=7)
        in_ys = [c.get_center()[1] for c in net.columns[0]]
        state_rows = m.VGroup()
        for name, v, y in zip(names, S_LEAN, in_ys):
            state_rows.add(m.VGroup(right_at(code(name, 20, MUTED), -3.0, y),
                                    left_at(code(minus(f"{v:+.3f}"), 22, STATE, weight=m.BOLD), -2.75, y)))
        feeds = m.VGroup(*[m.Arrow([-1.4, y, 0], net.columns[0][k].get_left() + [-0.05, 0, 0], buff=0.05, color=STATE,
                                   stroke_width=3, max_tip_length_to_length_ratio=0.25) for k, y in enumerate(in_ys)])
        hid = cite("256 · 256").next_to(net, m.DOWN, buff=0.18)
        size_lab = cite(f"{N_PARAMS:,} parameters").next_to(net, m.UP, buff=0.18)
        out_ys = [c.get_center()[1] for c in net.columns[-1]]
        outs = m.VGroup()
        for (act, q), y in zip((("push left", Q_LEFT), ("push right", Q_RIGHT)), out_ys):
            outs.add(m.VGroup(left_at(code(act, 22, ACTION), 2.75, y + 0.2),
                              left_at(code(f"Q = {q:.2f}", 24, VALUE, weight=m.BOLD), 2.75, y - 0.2)))
        best = m.SurroundingRectangle(outs[1], buff=0.12, corner_radius=0.1).set_stroke(ACTION, 3)
        push = m.Arrow([0, 0, 0], [0.75, 0, 0], buff=0, color=ACTION, stroke_width=6, max_tip_length_to_length_ratio=0.35)
        push.next_to(view.cart, m.RIGHT, buff=0.12)

        # the target and the TD error (bottom row)
        f1 = m.MathTex("y", "=", "r", "+", r"\gamma", r"\max_{a'} Q(s', a')", font_size=40, color=INK)
        f1[2].set_color(REWARD)
        f1[5].set_color(VALUE)
        f1.move_to([-3.2, -2.85, 0])
        f1_lab = code("target", 20, MUTED).next_to(f1, m.LEFT, buff=0.3)
        f2 = m.MathTex(r"\delta", "=", "y", "-", "Q(s, a)", font_size=40, color=INK)
        f2[0].set_color(SIGNAL)
        f2[4].set_color(VALUE)
        f2.move_to([3.4, -2.85, 0])
        f2_lab = code("TD error", 20, SIGNAL).next_to(f2, m.LEFT, buff=0.3)
        shrink = cite("a gradient step makes δ² smaller").next_to(f2, m.DOWN, buff=0.15)

        self.say(
            "[In episode ten, Q-values lived in a table]: [one row per state], [one column per action.]",
            m.AnimationGroup(self.new_header("02 · a network for Q", "Swap the table for a network"),
                             m.FadeIn(table, shift=0.1 * m.UP)),
            m.Create(row_hl),
            m.AnimationGroup(m.FadeOut(row_hl), m.Create(col_hl)),
        )
        self.say(
            "[But CartPole's state is four real numbers]: [cart position and speed, pole angle and spin.] "
            "[No table has a row for every one.]",
            m.AnimationGroup(m.FadeOut(col_hl), m.FadeIn(view, shift=0.1 * m.RIGHT)),
            m.LaggedStart(*[m.FadeIn(r_, shift=0.1 * m.RIGHT) for r_ in state_rows], lag_ratio=0.25),
            m.AnimationGroup(table.animate.set_opacity(0.3), m.FadeIn(infinite, shift=0.1 * m.UP)),
        )
        self.say(
            "[So a network takes the four numbers in], [and puts out one Q-value per action], [both in a single pass.]",
            m.AnimationGroup(m.FadeOut(table), m.FadeOut(infinite), m.FadeIn(net), m.FadeIn(hid), m.FadeIn(size_lab),
                             m.LaggedStart(*[m.GrowArrow(a) for a in feeds], lag_ratio=0.15)),
            m.LaggedStart(*[m.FadeIn(o, shift=0.1 * m.LEFT) for o in outs], lag_ratio=0.3),
            net.forward(VIOLET, run_time=1.4),
        )
        self.say(
            "[Here the pole leans right, so pushing right scores a little higher], [and that's the move.]",
            m.Circumscribe(outs[1], color=ACTION, buff=0.1),
            m.AnimationGroup(m.Create(best), m.GrowArrow(push)),
        )
        self.say(
            "[To learn, it aims at episode ten's target]: [the reward, plus gamma times the best Q-value at the next state.]",
            m.AnimationGroup(m.FadeIn(f1_lab), m.FadeIn(f1[:2])),
            m.LaggedStart(m.FadeIn(f1[2:5], shift=0.1 * m.UP), m.FadeIn(f1[5], shift=0.1 * m.UP), lag_ratio=0.5),
        )
        self.say(
            "[The gap is the TD error], [and a gradient step shrinks it, nudging every weight a little.]",
            m.AnimationGroup(m.FadeIn(f2_lab), m.FadeIn(f2, shift=0.1 * m.UP)),
            m.AnimationGroup(backward(net, SIGNAL, run_time=1.6), m.FadeIn(shrink)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · why it breaks

    def why_it_breaks(self):
        self.next_section("3 · Why it breaks", notes=(
            "Mnih et al. (2015) list three causes of instability: consecutive observations are correlated; small changes "
            "to Q change the policy and so the data; the target uses the network being updated (notes §13). Quote: "
            "'Reinforcement learning is known to be unstable or even to diverge when a nonlinear function approximator "
            "such as a neural network is used to represent the action-value (also known as Q) function.' Left panel: "
            f"steps {STRIP[0]}–{STRIP[-1]} of the trained episode (0.02 s apart), drawn at their true angles. Right panel: "
            "an ILLUSTRATION with made-up numbers (moving_target()): each update moves the guess 60% of the way to the "
            "target, and the target, computed by the same network, rises with it. The deadly triad (function "
            "approximation + bootstrapping + off-policy data): Sutton & Barto ch. 11, Episode 12."))
        quote = m.VGroup(
            text("“Reinforcement learning is known to be unstable or even to diverge", 22, INK),
            text("when a nonlinear function approximator such as a neural network is used …”", 22, INK),
        ).arrange(m.DOWN, buff=0.1)
        quote.move_to([0, 2.2, 0])
        q_src = cite("Mnih et al., 2015").next_to(quote, m.DOWN, buff=0.12).align_to(quote, m.RIGHT)

        pw, ph, py = 6.2, 3.55, -0.8
        p1 = card(pw, ph, EDGE, CARD, radius=0.18, stroke=2).move_to([-3.4, py, 0])
        p2 = card(pw, ph, EDGE, CARD, radius=0.18, stroke=2).move_to([3.4, py, 0])
        t1 = left_at(text("1 · steps in a row look alike", 24, INK, weight=m.BOLD), -6.2, py + ph / 2 - 0.38)
        t2 = left_at(text("2 · the target moves with the guess", 24, INK, weight=m.BOLD), 0.6, py + ph / 2 - 0.38)

        frames = m.VGroup(*[mini_pole(GOOD[k]) for k in STRIP]).arrange(m.RIGHT, buff=0.12).move_to([-3.4, py + 0.35, 0])
        f_lab = cite(f"steps {STRIP[0]}–{STRIP[-1]} · a tenth of a second").next_to(frames, m.DOWN, buff=0.18)
        fit = m.VGroup(m.Brace(frames, m.DOWN, buff=0.55, color=RED))
        fit_lab = text("learned in order: the net fits this moment", 22, RED).next_to(fit, m.DOWN, buff=0.1)
        forget = text("… and forgets the rest", 22, RED).next_to(fit_lab, m.DOWN, buff=0.08)

        plot = chase_plot((3.2, py - 0.3, 0), height=1.2, top=MOVE_TOP)
        ks = list(range(len(G_MOVE)))
        g_line = plot.line(ks, G_MOVE, VALUE, 4)
        g_dots = m.VGroup(*[plot.dot(k, v, VALUE, 0.06) for k, v in zip(ks, G_MOVE)])
        t_pts = [plot.coords(k, v) for k, v in zip(ks, T_MOVE)]
        t_line = m.VMobject().set_points_as_corners(t_pts).set_stroke(SIGNAL, 3)
        t_dash = m.DashedVMobject(t_line, num_dashes=40).set_stroke(SIGNAL, 3)
        g_lab = code("guess", 20, VALUE, weight=m.BOLD).next_to(plot.coords(8, G_MOVE[-1]), m.RIGHT, buff=0.12)
        t_lab = code("target", 20, SIGNAL, weight=m.BOLD).next_to(plot.coords(8, T_MOVE[-1]), m.RIGHT, buff=0.12)
        t_start = m.Circle(radius=0.08).set_stroke(SIGNAL, 3).move_to(plot.coords(0, T_MOVE[0]))
        ill = cite("illustration").move_to([0, py - ph / 2 + 0.28, 0]).align_to([6.3, 0, 0], m.RIGHT)

        triad = m.VGroup(pill("function approximation", VIOLET, VIOLET_FILL, 20),
                         pill("bootstrapping", SIGNAL, PINK_FILL, 20),
                         pill("off-policy data", ACTION, CORAL_FILL, 20)).arrange(m.RIGHT, buff=0.25)
        ep12 = text("= episode 12's deadly triad", 22, RED, weight=m.BOLD)
        tri_row = m.VGroup(triad, ep12).arrange(m.RIGHT, buff=0.35).move_to([0, -3.15, 0])

        self.say(
            "[Trained naively, this falls apart.] [The DQN paper says so itself.]",
            m.AnimationGroup(self.new_header("03 · why it breaks", "Two ways to fall apart"),
                             m.FadeIn(p1), m.FadeIn(p2), m.FadeIn(t1), m.FadeIn(t2)),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.DOWN), m.FadeIn(q_src)),
        )
        self.say(
            "[First, back-to-back steps are near copies.] [Learn from them in order, and the network fits this moment], "
            "[and forgets the rest.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.RIGHT) for f in frames], lag_ratio=0.2),
                             m.FadeIn(f_lab)),
            m.AnimationGroup(m.GrowFromCenter(fit), m.FadeIn(fit_lab, shift=0.1 * m.UP)),
            m.FadeIn(forget, shift=0.1 * m.UP),
        )
        self.say(
            "[Second, the target comes from the very network being trained.] [Every update chases the target], "
            "[and the target runs away.]",
            m.AnimationGroup(m.FadeIn(plot), m.FadeIn(ill), m.FadeIn(g_dots[0], scale=0.5),
                             m.FadeIn(t_start, scale=0.5)),
            m.AnimationGroup(m.Create(g_line, run_time=2.4), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in g_dots[1:]],
                                                                              lag_ratio=0.5, run_time=2.4)),
            m.AnimationGroup(m.Create(t_dash, run_time=2.0), m.FadeIn(g_lab), m.FadeIn(t_lab)),
        )
        self.say(
            "[A function approximator, bootstrapping and off-policy data]: [episode twelve's deadly triad, all at once.]",
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in triad], lag_ratio=0.35),
            m.FadeIn(ep12, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · experience replay

    def experience_replay(self):
        self.next_section("4 · Experience replay", notes=(
            f"On this Mac: the last {HP['buffer_size']:,} transitions (s, a, r, s′, done), minibatches of "
            f"{HP['batch_size']} drawn uniformly at random, one gradient step every {HP['train_every']} environment "
            f"steps. Each stored transition is therefore replayed about batch ÷ train_every = {USES} times on average "
            f"while it stays in the buffer (DERIVED from the settings: {HP['buffer_size'] // HP['train_every']:,} "
            f"updates × {HP['batch_size']} / {HP['buffer_size']:,}). The paper (Mnih et al. 2015, Extended Data Table 1): "
            f"replay memory {REPLAY_2015:,}, minibatch {BATCH_2015}, also one update every 4 actions (≈ 8 uses each). "
            "Experience replay is Lin (1992). Off-policy: 'when learning by experience replay, it is necessary to learn "
            "off-policy (because our current parameters are different to those used to generate the sample), which "
            "motivates the choice of Q-learning' (Mnih et al. 2015, Methods). The buffer drawing has 70 slots and the "
            "minibatch 8 squares (a picture of 100,000 and 128); the slots picked use default_rng(13)."))
        rows_n, cols_n = 5, 14
        buf = cell_grid(rows_n, cols_n, size=0.3, gap=0.07, color=EDGE, fill=CARD).move_to([-0.25, 0.95, 0])
        cells = [buf[r][c] for r in range(rows_n) for c in range(cols_n)]
        buf_title = left_at(text("replay memory", 24, STATE, weight=m.BOLD), buf.get_left()[0], 2.25)
        order = right_at(cite("oldest → newest"), buf.get_right()[0], 2.25)

        view = CartPoleView(center=(-5.2, 0.15, 0), units_per_m=0.38, pole_units=1.2)
        view.set_state(GOOD[0][0], GOOD[0][2])
        bot = rover(0.36).move_to([-5.2, 2.2, 0])

        net = NeuralNet(layers=(4, 256, 256, 2), width=1.9, height=1.6, center=(4.95, 1.05, 0), color=VIOLET,
                        radius=0.09, max_nodes=6)
        net_lab = cite("network").next_to(net, m.UP, buff=0.15)

        # one transition, colour-coded
        parts = [("s", "state", STATE), ("a", "action", ACTION), ("r", "reward", REWARD), ("s′", "next state", STATE)]
        chips = m.VGroup(*[chip(sym, col, FILLS[col], width=0.8, height=0.66, size=28) for sym, _, col in parts])
        chips.arrange(m.RIGHT, buff=0.85)
        words = m.VGroup(*[code(w, 20, col) for _, w, col in parts])
        for w, c in zip(words, chips):
            w.next_to(c, m.DOWN, buff=0.12)
        on_baseline(words)
        trans = m.VGroup(chips, words).move_to([-3.4, -1.85, 0])
        trans_lab = cite("one memory = one step").next_to(trans, m.UP, buff=0.18)
        token = m.Square(0.3).set_stroke(STATE, 2).set_fill(mix(CARD, TEAL, 0.55), 1).move_to(cells[0])

        fill_c = mix(CARD, TEAL, 0.45)
        n_fill = len(cells)

        def filling(_, alpha):
            i = alpha * (n_fill - 1)
            for k, cl in enumerate(cells):
                if k <= i:
                    cl.set_fill(fill_c, 1).set_stroke(STATE, 1.5)
            j = int(alpha * 69)
            view.set_state(GOOD[j][0], GOOD[j][2])

        fill_anim = m.UpdateFromAlphaFunc(buf, filling, run_time=2.2, rate_func=m.linear)
        oldest = [buf[r][c] for r in range(1) for c in range(5)]
        new_c = mix(CARD, TEAL, 0.8)
        old_lab = cite("oldest out, newest in").next_to(buf, m.DOWN, buff=0.18).align_to(buf, m.LEFT)

        rng = np.random.default_rng(13)
        picks = sorted(rng.choice(len(cells), size=8, replace=False).tolist())
        assert len({p // cols_n for p in picks}) >= 4                  # drawn from all over the memory
        rings = m.VGroup(*[m.SurroundingRectangle(cells[p], buff=0.03, corner_radius=0.04).set_stroke(INK, 3) for p in picks])
        batch = m.VGroup(*[m.Square(0.22).set_stroke(STATE, 2).set_fill(fill_c, 1) for _ in picks]).arrange(m.RIGHT, buff=0.06)
        batch.move_to([4.95, -0.25, 0])
        batch_lab = cite("random minibatch").next_to(batch, m.DOWN, buff=0.12)

        stat_rows = m.VGroup(
            colored([("each memory replayed", MUTED), (f"≈ {USES} times", STATE)], 22, weight=m.BOLD),
            cite(f"{HP['batch_size']} per update ÷ one update every {HP['train_every']} steps"),
            colored([("this Mac:", MUTED), (f"{HP['buffer_size']:,}", STATE), ("· batch", MUTED),
                     (f"{HP['batch_size']}", INK)], 20),
            colored([("the paper:", MUTED), (f"{REPLAY_2015:,}", STATE), ("· batch", MUTED), (f"{BATCH_2015}", INK)], 20),
        ).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        left_at(stat_rows, 0.55, -2.0)
        src = cite("Mnih et al., 2015 · replay: Lin, 1992").move_to([0, -3.35, 0]).align_to([6.5, 0, 0], m.RIGHT)
        offp = m.VGroup(text("old memories come from older policies", 22, INK),
                        code("→ the learner must be off-policy", 22, ACTION, weight=m.BOLD),
                        code("→ Q-learning", 22, VALUE, weight=m.BOLD)).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        left_at(offp, -6.3, -1.95)
        offp_src = left_at(cite("“… which motivates the choice of Q-learning” · Mnih et al., 2015"), -6.3, -3.3)

        self.say(
            "[The first fix is a memory.] [After every step, Rover stores what happened]: "
            "[the state, the action, the reward and the next state.]",
            m.AnimationGroup(self.new_header("04 · experience replay", "Remember, then shuffle"),
                             m.FadeIn(buf), m.FadeIn(buf_title), m.FadeIn(order)),
            m.AnimationGroup(m.FadeIn(view), m.FadeIn(bot, shift=0.1 * m.DOWN), m.FadeIn(net), m.FadeIn(net_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(m.VGroup(c, w), shift=0.1 * m.UP) for c, w in zip(chips, words)],
                                           lag_ratio=0.25), m.FadeIn(trans_lab),
                             m.Succession(m.Wait(1.0), m.FadeIn(token, target_position=chips, scale=0.4))),
        )
        self.say(
            "[The memory fills], [and once it's full, the oldest steps make room for new ones.]",
            m.AnimationGroup(fill_anim, m.FadeOut(token, run_time=0.6)),
            m.AnimationGroup(m.FadeIn(old_lab),
                             m.Succession(*[cl.animate(run_time=0.25).set_fill(CARD, 1) for cl in oldest],
                                          m.AnimationGroup(*[cl.animate.set_fill(new_c, 1).set_stroke(STATE, 2) for cl in oldest],
                                                           run_time=0.5))),
        )
        self.say(
            "[To learn, Rover draws a random handful]: [moments from many different episodes], [and trains on them.]",
            m.LaggedStart(*[m.Create(r_) for r_ in rings], lag_ratio=0.15),
            m.LaggedStart(*[m.FadeIn(b, target_position=cells[p]) for b, p in zip(batch, picks)], lag_ratio=0.1,
                          run_time=1.2),
            m.AnimationGroup(m.FadeIn(batch_lab), m.Succession(net.forward(VIOLET, 1.0), backward(net, SIGNAL, 1.0))),
        )
        self.say(
            "[That breaks the chain of near copies], [and every memory gets used again and again]: "
            "[about thirty-two times here.]",
            m.Circumscribe(batch, color=STATE, buff=0.08),
            m.FadeIn(stat_rows[2:], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(stat_rows[:2], shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[Old memories come from older versions of the policy], [so the learner must be off-policy]: "
            "[that's why it's Q-learning.]",
            m.AnimationGroup(m.FadeOut(trans), m.FadeOut(trans_lab), m.FadeIn(offp[0], shift=0.1 * m.UP)),
            m.FadeIn(offp[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(offp[2], shift=0.1 * m.UP), m.FadeIn(offp_src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · target network

    def target_net(self):
        self.next_section("5 · Target network", notes=(
            "Targets come from a frozen copy Q̂ with parameters θ⁻: y = r + γ max_a′ Q̂(s′, a′; θ⁻); the copy is refreshed "
            f"from the learning network every {HP['target_update_every']:,} environment steps on this Mac, every "
            f"C = {C_2015:,} parameter updates in the paper (= 40,000 agent steps; notes §13: not 10,000 frames). "
            "'Generating the targets using an older set of parameters adds a delay between the time an update to Q is "
            "made and the time the update affects the targets yj, making divergence or oscillations much more unlikely' "
            "(Mnih et al. 2015, Methods). The chase sketch is an ILLUSTRATION with made-up numbers (frozen_target(): "
            "the target is held for three updates, then recomputed from the caught-up guess)."))
        f = m.MathTex("y", "=", "r", "+", r"\gamma", r"\max_{a'} Q(s', a';", r"\theta^-", ")", font_size=42, color=INK)
        f[2].set_color(REWARD)
        f[5].set_color(VALUE)
        f[7].set_color(VALUE)
        f[6].set_color(VIOLET)
        f.move_to([-1.0, 2.2, 0])
        theta_lab = m.VGroup(tex(r"\theta^-", 34, VIOLET), text("= the frozen copy's weights", 22, VIOLET)).arrange(m.RIGHT, buff=0.15)
        theta_lab.next_to(f, m.RIGHT, buff=0.45)

        on = NeuralNet(layers=(4, 256, 256, 2), width=2.6, height=2.0, center=(-3.5, -0.35, 0), color=VIOLET, max_nodes=6)
        tg = NeuralNet(layers=(4, 256, 256, 2), width=2.6, height=2.0, center=(3.5, -0.35, 0), color=VIOLET, max_nodes=6)
        tg.edges.set_stroke(opacity=0.55)
        on_lab = m.VGroup(text("learning network", 24, INK, weight=m.BOLD), cite("updated every 4 steps"))
        on_lab.arrange(m.DOWN, buff=0.08).next_to(on, m.UP, buff=0.3)
        lock = icon(LOCK, 0.42)
        tg_name = text("target network", 24, INK, weight=m.BOLD)
        tg_lab = m.VGroup(m.VGroup(lock, tg_name).arrange(m.RIGHT, buff=0.15), cite("frozen"))
        tg_lab.arrange(m.DOWN, buff=0.08).next_to(tg, m.UP, buff=0.3)
        y_arrow = m.Arrow([1.85, 0.15, 0], [-1.85, 0.15, 0], buff=0, color=SIGNAL, stroke_width=5, max_tip_length_to_length_ratio=0.08)
        y_lab = code("targets y", 22, SIGNAL, weight=m.BOLD).next_to(y_arrow, m.UP, buff=0.1)
        copy_arrow = m.CurvedArrow(np.array([-2.5, -1.55, 0]), np.array([2.5, -1.55, 0]), angle=0.55, color=VIOLET,
                                   stroke_width=5, tip_length=0.22)
        copy_lab = code("copy weights", 22, VIOLET, weight=m.BOLD).next_to(copy_arrow, m.DOWN, buff=0.08)

        def ticks(n):
            return code(f"{n:,}", 30, INK, weight=m.BOLD).move_to([0, -0.5, 0])

        counter = ticks(0)
        since = cite("steps since the last copy").move_to([0, -0.98, 0])
        every = colored([("refresh: this Mac every", MUTED), (f"{HP['target_update_every']:,} steps", VIOLET), ("·", MUTED),
                         ("the paper every", MUTED), (f"{C_2015:,} updates", VIOLET)], 22).move_to([0, -3.2, 0])
        paper = cite("Mnih et al., 2015").next_to(every, m.UP, buff=0.12)

        def count_to(n0, n1, seconds):
            def step(_, alpha):
                counter.become(ticks(int(round(n0 + (n1 - n0) * alpha))))
            return m.UpdateFromAlphaFunc(counter, step, run_time=seconds, rate_func=m.linear)

        plot = chase_plot((-0.3, -0.75, 0), width=7.0, height=2.6, top=FROZ_TOP)
        ks = list(range(len(G_FROZ)))
        g_line = plot.line(ks, G_FROZ, VALUE, 4)
        g_dots = m.VGroup(*[plot.dot(k, v, VALUE, 0.06) for k, v in zip(ks, G_FROZ)])
        stair = []
        for k in range(len(T_FROZ) - 1):
            stair += [plot.coords(k, T_FROZ[k]), plot.coords(k + 1, T_FROZ[k])]
        t_line = m.DashedVMobject(m.VMobject().set_points_as_corners(stair), num_dashes=60).set_stroke(SIGNAL, 3)
        copies = m.VGroup(*[m.VGroup(plot.vline(k, VIOLET, 2).set_stroke(opacity=0.6),
                                     code("copy", 20, VIOLET).next_to(plot.coords(k, FROZ_TOP), m.UP, buff=0.08))
                            for k in (3, 6)])
        g_end = code("guess", 20, VALUE, weight=m.BOLD).next_to(plot.coords(8, G_FROZ[-1]), m.RIGHT, buff=0.12).shift(0.15 * m.DOWN)
        t_end = code("target", 20, SIGNAL, weight=m.BOLD).next_to(plot.coords(8, T_FROZ[-1]), m.RIGHT, buff=0.12).shift(0.15 * m.UP)
        ill = cite("illustration").next_to(plot, m.DOWN, buff=0.1).align_to(plot, m.RIGHT)

        self.say(
            "[The second fix is a frozen copy of the network.]",
            m.AnimationGroup(self.new_header("05 · target network", "A target that holds still"),
                             m.FadeIn(on), m.FadeIn(on_lab), m.FadeIn(tg, target_position=on), m.FadeIn(tg_lab)),
        )
        self.say(
            "[The copy computes the targets], [while the learning network keeps updating.]",
            m.AnimationGroup(m.FadeIn(f, shift=0.1 * m.DOWN), m.FadeIn(theta_lab), tg.forward(VIOLET, 1.0),
                             m.GrowArrow(y_arrow), m.FadeIn(y_lab)),
            m.AnimationGroup(m.FadeIn(counter), m.FadeIn(since), m.Succession(backward(on, SIGNAL, 0.9), backward(on, SIGNAL, 0.9)),
                             m.Succession(m.Wait(0.2), count_to(0, 8, 1.6))),
        )
        self.say(
            "[Every thousand steps, the copy is refreshed]: [the learner's weights are copied over.] "
            "[The paper waited ten thousand updates.]",
            count_to(8, HP["target_update_every"], 2.0),
            m.Succession(m.AnimationGroup(m.Create(copy_arrow), m.FadeIn(copy_lab)),
                         m.AnimationGroup(tg.forward(VIOLET, 0.9), m.Circumscribe(lock, color=VIOLET, buff=0.06)),
                         count_to(0, 0, 0.1)),
            m.AnimationGroup(m.FadeIn(every, shift=0.1 * m.UP), m.FadeIn(paper)),
        )
        self.say(
            "[Between refreshes, the target sits still], [so the learner can actually catch it.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(on, on_lab, tg, tg_lab, y_arrow, y_lab, copy_arrow, copy_lab, counter,
                                                since, every, paper)),
                             m.FadeIn(plot), m.FadeIn(ill), m.Create(t_line, run_time=1.6), m.FadeIn(copies), m.FadeIn(t_end)),
            m.AnimationGroup(m.Create(g_line, run_time=2.4),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in g_dots], lag_ratio=0.4, run_time=2.4),
                             m.FadeIn(g_end)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the loop

    def the_loop(self):
        self.next_section("6 · The whole loop", notes=(
            f"ε-greedy behaviour with ε decayed linearly from {HP['eps_start']:g} to {HP['eps_end']:g} over the first "
            f"{HP['eps_decay_steps']:,} steps, then held (this Mac, methods.dqn.hp). The paper: 1 → 0.1 over the first "
            f"million frames, ε = 0.05 for evaluation (Mnih et al. 2015, Extended Data Table 1). The loop: act (ε-greedy), "
            f"store (every step), sample + learn (every {HP['train_every']} steps, a minibatch of {HP['batch_size']}), copy "
            f"(every {HP['target_update_every']:,} steps)."))
        plot = Plot((0, 50), (0, 1), width=5.0, height=2.9, center=(-3.4, -0.25, 0), x_ticks=[0, 10, 20, 30, 40, 50],
                    y_ticks=[0, 0.5, 1], x_label="thousand steps", y_label="epsilon: chance of a random move")
        d0, d1 = HP["eps_decay_steps"] / 1000, HP["eps_end"]
        eps = plot.line([0, d0, 50], [HP["eps_start"], d1, d1], EXPLORE, 5)
        dice = icon(DICE, 0.5).next_to(plot.coords(0, 1), m.RIGHT, buff=0.35)
        end_lab = code(f"{d1:g} from {d0:g}k on", 20, EXPLORE, weight=m.BOLD).next_to(plot.coords(35, d1), m.UP, buff=0.15)
        paper = cite("the paper: 1 → 0.1 over a million frames").next_to(plot, m.DOWN, buff=0.25)

        ring = loop_ring(["act", "store", "sample", "learn", "copy"], radius=1.75,
                         colors=[ACTION, STATE, MUTED, SIGNAL, VIOLET], size=24, center=(3.6, -0.1, 0))
        nodes, arcs = ring
        when = cite(f"act, store: every step · sample, learn: every {HP['train_every']} steps · "
                    f"copy: every {HP['target_update_every']:,} steps").move_to([0, -3.3, 0])
        bot = rover(0.42).move_to([3.6, -0.1, 0])

        self.say(
            "[One more knob: epsilon.] [Rover starts out acting completely at random], "
            "[then lowers epsilon to one percent over the first twenty thousand steps.]",
            m.AnimationGroup(self.new_header("06 · the loop", "The whole recipe"), m.FadeIn(plot)),
            m.AnimationGroup(m.FadeIn(dice, scale=0.6), m.FadeIn(plot.dot(0, 1, EXPLORE, 0.08), scale=0.5)),
            m.AnimationGroup(m.Create(eps, run_time=2.0), m.FadeIn(end_lab), m.FadeIn(paper)),
        )
        self.say(
            "[That's the whole recipe]: [act, store, sample, learn], [and every so often, copy.]",
            m.AnimationGroup(m.FadeIn(bot, scale=0.8)),
            m.Succession(*[m.AnimationGroup(m.FadeIn(nodes[k], scale=0.85), m.Create(arcs[k - 1]) if k else m.Wait(0.01),
                                            run_time=0.5) for k in range(4)]),
            m.AnimationGroup(m.FadeIn(nodes[4], scale=0.85), m.Create(arcs[3]), m.Create(arcs[4]), m.FadeIn(when)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · on this Mac

    def results_plot(self, y_range, y_ticks, y_label, y_fmt="{:g}", center=(-2.25, -0.55, 0), width=7.3):
        return Plot((0, 525), y_range, width=width, height=4.2, center=center,
                    x_ticks=[0, 100, 200, 300, 400, 500], y_ticks=y_ticks, x_label="thousand environment steps",
                    y_label=y_label, y_fmt=y_fmt)

    def run_legend(self, rows, y0=1.75):
        """Coloured run names with a line of numbers under each, in the right column."""
        group = m.VGroup()
        for k, (name, color, line) in enumerate(rows):
            sw = m.Line([0, 0, 0], [0.4, 0, 0]).set_stroke(color, 5)
            nm = text(name, 22, color, weight=m.BOLD)
            head = m.VGroup(sw, nm).arrange(m.RIGHT, buff=0.15)
            sub = code(line, 20, INK)
            item = m.VGroup(head, sub).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            left_at(item, 2.75, y0 - 1.2 * k)
            sub.align_to(nm, m.LEFT)
            group.add(item)
        return group

    def on_this_mac(self):
        self.next_section("7 · Trained on this Mac", notes=(
            f"deep_rl.md / deep_rl.json, CartPole-v1, {SEEDS} seeds (10–14) per variant, {HP['total_steps']:,} steps each. "
            f"Score = mean return of the last 100 training episodes (ε-greedy); solved at ≥ {SOLVED}. Thick lines: the "
            "mean over seeds (curves.*.avg100.mean); thin: DQN's five seeds. DQN: 5/5 solved after "
            f"{HL['steps_to_solve_mean']:,.0f} ± {HL['steps_to_solve_std']:,.0f} steps (min {HL['steps_to_solve_min']:,.0f}, "
            f"max {HL['steps_to_solve_max']:,.0f}), {EPISODES_TO_SOLVE:,.0f} episodes, {WALL_S:.0f} s of wall time. "
            f"No replay (update on the newest transition only): 0/5, best 100-episode average {NR['best_avg100_mean']}. "
            f"No target network: 0/5, episodes collapse to {NT['final_avg100_mean']} steps. After solving, "
            f"{len(DROPPED)} of 5 seeds fell below 300 (seeds {', '.join(str(s['seed']) for s in DROPPED)}); final average "
            f"{HL['final_avg100_mean']} ± {HL['final_avg100_std']} (mean ± sample std). The mean curve itself peaks at "
            f"{max(CURVES['dqn']['avg100']['mean']):.1f}: the seeds solve at different times."))
        plot = self.results_plot((0, 525), [0, 100, 200, 300, 400], "score: average of the last 100 episodes")
        solved = plot.hline(SOLVED, INK, 2)
        p475 = plot.coords(0, SOLVED)
        solved_lab = m.VGroup(m.Line(p475, p475 + [-0.09, 0, 0]).set_stroke(INK, 2),
                              code(f"{SOLVED}", 20, INK, weight=m.BOLD).next_to(p475, m.LEFT, buff=0.16),
                              code("solved", 20, INK).next_to(plot.coords(8, SOLVED), m.DOWN, buff=0.1)
                              .align_to(plot.coords(8, 0), m.LEFT))
        meas = cite(f"trained on this Mac · {SEEDS} seeds each · mean").move_to([0, 2.3, 0]).align_to([6.5, 0, 0], m.RIGHT)

        full = plot.line(X_K, CURVES["dqn"]["avg100"]["mean"], FULL_C, 5)
        seeds = m.VGroup(*[plot.line(X_K, c, FULL_C, 2).set_stroke(opacity=0.35) for c in SEED_CURVES])
        nrep = plot.line(X_K, CURVES["dqn_no_replay"]["avg100"]["mean"], NR_C, 4)
        ntgt = plot.line(X_K, CURVES["dqn_no_target"]["avg100"]["mean"], NT_C, 4)

        leg = self.run_legend([
            ("replay + target network", FULL_C, f"{HL['n_solved']} of {SEEDS} solved"),
            ("no replay", NR_C, f"0 of {SEEDS} · best ≈ {NR_BEST}"),
            ("no target network", NT_C, f"0 of {SEEDS} · ≈ {NT_LEN} steps"),
        ])
        full_more = m.VGroup(code(f"{STEPS_K}k ± {STEPS_SD_K}k steps", 20, INK),
                             code(f"{EPISODES_TO_SOLVE:,.0f} episodes · {WALL_S:.0f} s", 20, INK))
        full_more.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).next_to(leg[0][1], m.DOWN, buff=0.08).align_to(leg[0][1], m.LEFT)
        leg[1].shift(0.7 * m.DOWN)
        leg[2].shift(0.7 * m.DOWN)

        dropped = m.VGroup(*[plot.line(X_K, SEED_CURVES[i], FULL_C, 3) for i in DROP_IDX])
        line300 = plot.hline(300, RED, 2)
        ends = m.VGroup(*[plot.dot(500, SEED_CURVES[i][-1], RED, 0.08) for i in DROP_IDX])
        wobble = m.VGroup(code(f"{len(DROPPED)} of {SEEDS} fell below 300", 20, RED, weight=m.BOLD),
                          code(f"final {FINAL} ± {FINAL_SD}", 20, RED))
        wobble.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).move_to([0, -2.9, 0]).align_to([2.75, 0, 0], m.LEFT)

        self.say(
            "[Here it is, trained on this Mac]: [five seeds, half a million steps each.]",
            m.AnimationGroup(self.new_header("07 · on this Mac", "The score climbs"), m.FadeIn(plot)),
            m.FadeIn(meas, shift=0.1 * m.DOWN),
        )
        self.say(
            "[The score is the average of the last hundred episodes]; [four hundred seventy-five counts as solved.]",
            m.Circumscribe(plot.y_label, color=MUTED, buff=0.06),
            m.AnimationGroup(m.Create(solved), m.FadeIn(solved_lab)),
        )
        self.say(
            "[With both tricks, every seed solved it]: [on average after about two hundred fifty thousand steps], "
            "[or fourteen hundred episodes and thirty-five seconds.]",
            m.AnimationGroup(m.Create(full, run_time=2.6), m.LaggedStart(*[m.Create(s) for s in seeds], lag_ratio=0.1,
                                                                           run_time=2.6), m.FadeIn(leg[0])),
            m.FadeIn(full_more[0], shift=0.1 * m.UP),
            m.FadeIn(full_more[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Without replay, no seed got there]: [its best average was about a hundred fifty-seven.]",
            m.AnimationGroup(m.Create(nrep, run_time=2.0)),
            m.FadeIn(leg[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Without the target network], [it collapsed to episodes of about nine steps.]",
            m.Create(ntgt, run_time=1.6),
            m.FadeIn(leg[2], shift=0.1 * m.UP),
        )
        self.say(
            "[And even with both, it wobbles]: [two of the five runs later fell back below three hundred.]",
            m.AnimationGroup(seeds.animate.set_stroke(opacity=0.15), m.Create(dropped, run_time=1.6)),
            m.AnimationGroup(m.Create(line300), m.FadeIn(ends, scale=0.5), m.FadeIn(wobble, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · runaway values

    def runaway_values(self):
        self.next_section("8 · Runaway values", notes=(
            "Mean Q-value on training batches (curves.*.q_batch_mean.mean, every 5,000 steps, mean of 5 seeds), on a "
            "log scale (values below 1 drawn at 1). CartPole pays +1 per step, γ = 0.99, so no policy can be worth more "
            f"than 1/(1 − γ) = {CEIL:g} (headline.overestimation.value_ceiling; a full 500-step episode collects 99.34). Final "
            f"(headline q_batch_mean_final_mean): DQN {HL['q_batch_mean_final_mean']} (a little above the ceiling: "
            f"Episode 14), no replay {NR['q_batch_mean_final_mean']} (≈ 2×), no target network {NT_Q:.4g} — the "
            "deadly triad of Episode 12, measured."))
        plot = self.results_plot((0, 10.5), None, "mean Q-value (log scale)", center=(-1.6, -0.55, 0), width=6.8)
        ticks = m.VGroup()
        for e, lab in ((0, "1"), (2, "100"), (4, r"10^{4}"), (6, r"10^{6}"), (8, r"10^{8}"), (10, r"10^{10}")):
            p = plot.coords(0, e)
            ticks.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2))
            mark = code(lab, 20, MUTED) if e <= 2 else tex(lab, 26, MUTED)
            ticks.add(mark.next_to(p, m.LEFT, buff=0.16))
        plot.add(ticks)

        def logq(key):
            return [float(np.log10(max(v, 1.0))) for v in Q_CURVES[key]["mean"]]

        ceil = plot.hline(np.log10(CEIL), INK, 2)
        ceil_sw = m.DashedLine([0, 0, 0], [0.4, 0, 0], dash_length=0.08).set_stroke(INK, 2.5)
        ceil_lab = m.VGroup(ceil_sw, text("ceiling: 1/(1 − γ) = 100", 20, INK)).arrange(m.RIGHT, buff=0.15)
        left_at(ceil_lab, 2.75, 1.85)
        full = plot.line(Q_X_K, logq("dqn"), FULL_C, 5)
        nrep = plot.line(Q_X_K, logq("dqn_no_replay"), NR_C, 4)
        ntgt = plot.line(Q_X_K, logq("dqn_no_target"), NT_C, 4)
        meas = cite(f"trained on this Mac · {SEEDS} seeds each · mean").move_to([0, 2.3, 0]).align_to([6.5, 0, 0], m.RIGHT)
        leg = self.run_legend([
            ("replay + target network", FULL_C, f"Q ≈ {DQN_Q}"),
            ("no replay", NR_C, f"Q ≈ {NR_Q} · ×2"),
            ("no target network", NT_C, f"Q ≈ {NT_Q / 1e9:.0f} × 10⁹"),
        ], y0=1.2)
        ep14 = pill("a little high: episode 14", VALUE, BLUE_FILL, 20).next_to(leg[0], m.DOWN, buff=0.15).align_to(leg[0], m.LEFT)
        leg[1].shift(0.4 * m.DOWN)
        leg[2].shift(0.4 * m.DOWN)
        triad = text("the deadly triad, for real", 22, RED, weight=m.BOLD).next_to(leg[2], m.DOWN, buff=0.3).align_to(leg[2], m.LEFT)

        self.say(
            "[Now the Q-values themselves.] [With gamma at zero point nine nine], "
            "[no CartPole policy can be worth more than a hundred.]",
            m.AnimationGroup(self.new_header("08 · runaway values", "Values with no brakes"), m.FadeIn(plot),
                             m.FadeIn(meas)),
            m.Circumscribe(plot.y_label, color=MUTED, buff=0.06),
            m.AnimationGroup(m.Create(ceil), m.FadeIn(ceil_lab)),
        )
        self.say(
            "[With both tricks, the values settle near that ceiling], [a little above it: more on that next time.]",
            m.AnimationGroup(m.Create(full, run_time=2.0), m.FadeIn(leg[0])),
            m.FadeIn(ep14, shift=0.1 * m.UP),
        )
        self.say(
            "[Without replay, they drifted to about two hundred]: [twice what's possible.]",
            m.Create(nrep, run_time=2.0),
            m.FadeIn(leg[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Without the target network, they exploded]: [about four billion.] [The deadly triad, for real.]",
            m.Create(ntgt, run_time=2.0),
            m.FadeIn(leg[2], shift=0.1 * m.UP),
            m.FadeIn(triad, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · Atari

    def atari(self):
        self.next_section("9 · Atari", notes=(
            "Mnih et al. (2015), Nature 518:529–533 (notes §13): 49 Atari 2600 games; input the last 4 frames, 84×84 "
            "grey; conv 32 8×8/4 → 64 4×4/2 → 64 3×3/1 → 512 → one Q-value per action (4–18); "
            f"{ATARI_PARAMS:,} parameters for 18 actions (derived, notes §13); the same architecture and settings for all "
            "games, a separate network trained per game; inputs: pixels, the score (rewards clipped to ±1), the number "
            f"of actions and the life counter. ≥ 75% of the human score on {AT_75} of {ATARI} games (the paper); ≥ 100% "
            f"on {AT_100} (notes' recount of Extended Data Table 2). The 7×7 grid shows counts only, sorted: its first "
            f"square is Video Pinball ({PINBALL_PCT:,}%, the best), its last Montezuma's Revenge (0%, the worst); "
            f"Breakout {BREAKOUT_PCT:,.1f}% = 100 × (401.2 − 1.7)/(31.8 − 1.7). The human: one professional games tester, "
            "~20 episodes of up to 5 min per game after ~2 h of practice, no pausing or saving. Extended Data Table 3 "
            "(10M frames): Breakout 316.8 with replay + target, 240.7 replay only, 10.2 target only, 3.2 neither (≈ 99×)."))
        # pixels → network → Q-values
        fx, fy = -5.3, 0.95
        frames = m.VGroup(*[card(1.1, 1.1, EDGE, CARD, radius=0.06, stroke=2).move_to([fx + 0.14 * k, fy - 0.14 * k, 0])
                            for k in range(4)])
        px = m.VGroup()
        rng = np.random.default_rng(5)
        top = frames[-1]
        for i in range(5):
            for j in range(5):
                if rng.random() < 0.35:
                    px.add(m.Square(0.16).set_stroke(width=0).set_fill(mix(CARD, INK, 0.45), 1)
                           .move_to(top.get_center() + [(j - 2) * 0.19, (i - 2) * 0.19, 0]))
        fr_lab = cite("4 frames · 84 × 84").next_to(frames, m.DOWN, buff=0.15)
        convs = m.VGroup(*[m.Polygon([0, -h, 0], [0.32, -h + 0.18, 0], [0.32, h - 0.18, 0], [0, h, 0])
                           .set_stroke(VIOLET, 2.5).set_fill(VIOLET_FILL, 1) for h in (0.75, 0.6, 0.48)])
        convs.arrange(m.RIGHT, buff=0.18).move_to([-3.2, fy, 0])
        fc = m.VGroup(*[m.Circle(0.07).set_stroke(VIOLET, 2).set_fill(BG, 1) for _ in range(6)]).arrange(m.DOWN, buff=0.1)
        fc.move_to([-2.2, fy, 0])
        qs = m.VGroup(*[m.Rectangle(width=0.42 + 0.1 * (k % 3), height=0.14).set_stroke(VALUE, 2).set_fill(BLUE_FILL, 1)
                        for k in range(6)]).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).move_to([-1.25, fy, 0])
        q_lab = cite("Q per action").next_to(qs, m.DOWN, buff=0.15)
        conv_lab = cite("3 conv + 512").next_to(m.VGroup(convs, fc), m.DOWN, buff=0.15)
        a1 = arrow(frames.get_right() + [0.05, 0.2, 0], convs.get_left() + [-0.05, 0, 0], MUTED)
        a2 = arrow(fc.get_right() + [0.02, 0, 0], qs.get_left() + [-0.02, 0, 0], MUTED)
        pad = icon(PAD, 0.6).move_to([-6.0, 2.15, 0])
        params = cite(f"≈ {ATARI_PARAMS / 1e6:.1f} million parameters").next_to(pad, m.RIGHT, buff=0.3)
        pipe = m.VGroup(frames, px, fr_lab, convs, fc, qs, q_lab, conv_lab, a1, a2)
        per_game = m.VGroup(text("same design and settings for every game", 22, INK),
                            text("a separate network trained for each", 22, INK)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(per_game, -6.4, -1.2)

        # the 49 games, sorted
        sq, gap = 0.32, 0.08
        grid = cell_grid(7, 7, size=sq, gap=gap, color=EDGE, fill=CARD).move_to([1.75, 0.85, 0])
        games = [grid[r][c] for r in range(7) for c in range(7)]
        hi, mid, lo = mix(CARD, GREEN, 0.85), mix(CARD, GREEN, 0.4), mix(CARD, MUTED, 0.12)
        colors = [hi] * AT_100 + [mid] * (AT_75 - AT_100) + [lo] * (ATARI - AT_75)
        assert len(colors) == len(games) == ATARI
        legend = m.VGroup()
        for col, lab in ((hi, f"≥ 100% · {AT_100}"), (mid, f"75–100% · {AT_75 - AT_100}"), (lo, f"< 75% · {ATARI - AT_75}")):
            sw = m.Square(0.26).set_stroke(EDGE, 1.5).set_fill(col, 1)
            legend.add(m.VGroup(sw, code(lab, 20, INK)).arrange(m.RIGHT, buff=0.15))
        legend.arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).next_to(grid, m.RIGHT, buff=0.4).align_to(grid, m.UP)
        g_title = text(f"{ATARI} games, % of the human score", 22, INK, weight=m.BOLD).next_to(grid, m.UP, buff=0.25)
        g_title.align_to(grid, m.LEFT)
        gx = grid.get_left()[0]
        best = left_at(text(f"best: Video Pinball {PINBALL_PCT:,.0f}%", 20, GREEN), gx, -0.9)
        brk = text(f"· Breakout {BREAKOUT_PCT:,.0f}%", 20, GREEN).next_to(best, m.RIGHT, buff=0.15)
        worst = left_at(text("worst: Montezuma's Revenge 0%", 20, RED), gx, -1.35)
        w_ring = m.SurroundingRectangle(games[-1], buff=0.04, corner_radius=0.04).set_stroke(RED, 3)
        b_ring = m.SurroundingRectangle(games[0], buff=0.04, corner_radius=0.04).set_stroke(GREEN, 3)
        human = m.VGroup(text("the human: one professional games tester", 22, INK),
                         text("about 2 hours of practice per game", 22, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(human, grid.get_left()[0], -2.62)
        src = cite("Mnih et al., 2015 · Nature").move_to([0, -3.35, 0]).align_to([6.5, 0, 0], m.RIGHT)
        ep11 = pill("sparse rewards · ep 11", REWARD, GOLD_FILL, 20).next_to(worst, m.DOWN, buff=0.15).align_to(worst, m.LEFT)

        # the paper's own ablation, Breakout
        bx0, bw = -6.3, 4.4
        abl = m.VGroup()
        for k, (lab, v) in enumerate(ABLATION):
            y = 0.95 - 0.72 * k
            col = FULL_C if k == 0 else (NT_C if k == len(ABLATION) - 1 else NR_C)
            w = max(bw * v / ABLATION[0][1], 0.04)
            bar = m.Rectangle(width=w, height=0.26).set_stroke(col, 2).set_fill(FILLS.get(col, CARD), 1)
            bar.move_to([bx0 + w / 2, y - 0.17, 0])
            name = left_at(code(lab, 20, MUTED), bx0, y + 0.2)
            num = code(f"{v:g}", 20, col, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.12)
            abl.add(m.VGroup(bar, name, num))
        abl_title = left_at(text("Breakout, the paper's own test", 22, INK, weight=m.BOLD), bx0, 1.75)
        abl_src = left_at(cite("Mnih et al., 2015 · Extended Data Table 3 · 10M frames"), bx0, -2.3)

        self.say(
            "[In twenty fifteen, DeepMind's DQN took on forty-nine Atari games], "
            "[learning from the screen's pixels and the score.]",
            m.AnimationGroup(self.new_header("09 · Atari", "Forty-nine games"), m.FadeIn(pad, scale=0.7), m.FadeIn(src),
                             m.FadeIn(grid), m.FadeIn(g_title)),
            m.AnimationGroup(m.FadeIn(frames, shift=0.1 * m.RIGHT), m.FadeIn(px), m.FadeIn(fr_lab), m.GrowArrow(a1),
                             m.FadeIn(convs), m.FadeIn(fc), m.FadeIn(conv_lab), m.GrowArrow(a2), m.FadeIn(qs),
                             m.FadeIn(q_lab), m.FadeIn(params)),
        )
        self.say(
            "[One design and one set of settings for every game], [but a separate network trained for each.]",
            m.FadeIn(per_game[0], shift=0.1 * m.UP),
            m.FadeIn(per_game[1], shift=0.1 * m.UP),
        )
        self.say(
            "[It reached at least seventy-five percent of the human score on twenty-nine of the forty-nine games.]",
            m.AnimationGroup(m.LaggedStart(*[g.animate.set_fill(c, 1) for g, c in zip(games, colors)], lag_ratio=0.04,
                                           run_time=2.4), m.FadeIn(legend)),
        )
        self.say(
            "[The human was one professional tester], [after about two hours of practice per game.]",
            m.FadeIn(human[0], shift=0.1 * m.UP),
            m.FadeIn(human[1], shift=0.1 * m.UP),
        )
        self.say(
            "[It crushed Breakout], [and scored zero on Montezuma's Revenge], [the sparse-reward game from episode eleven.]",
            m.AnimationGroup(m.Create(b_ring), m.FadeIn(best), m.FadeIn(brk)),
            m.AnimationGroup(m.Create(w_ring), m.FadeIn(worst)),
            m.FadeIn(ep11, shift=0.1 * m.UP),
        )
        self.say(
            "[And in the paper's own test], [Breakout scored over three hundred with both tricks], [and three with neither.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(pipe, per_game, pad, params)), m.FadeIn(abl_title), m.FadeIn(abl_src)),
            m.AnimationGroup(m.FadeIn(abl[0][1:]), m.GrowFromEdge(abl[0][0], m.LEFT),
                             m.FadeIn(abl[1][1:]), m.GrowFromEdge(abl[1][0], m.LEFT),
                             m.FadeIn(abl[2][1:]), m.GrowFromEdge(abl[2][0], m.LEFT)),
            m.AnimationGroup(m.FadeIn(abl[3][1:]), m.GrowFromEdge(abl[3][0], m.LEFT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 10 · recap

    def recap(self):
        self.next_section("10 · Recap", notes=(
            "Store (every step goes into the replay memory), sample (random minibatches break correlations and reuse "
            "data), freeze (targets from a copy refreshed every 1,000 steps here, every 10,000 updates in the paper). "
            f"On this Mac: with both, {HL['n_solved']}/5 seeds solved CartPole; without replay 0/5, without a target "
            f"network 0/5; even with both, {len(DROPPED)} of 5 later fell below 300 and the Q-values ended above the "
            "ceiling. Episode 14: overestimation and Double DQN."))
        halo = glow(VIOLET, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Store", 70, STATE), title_text("Sample", 70, SIGNAL), title_text("Freeze", 70, VIOLET))
        words.arrange(m.RIGHT, buff=1.3).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text("every step in a memory", 24, MUTED), text("random minibatches", 24, MUTED),
                        text("a target that holds still", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        base = max(s_.get_center()[1] for s_ in subs)
        for s_ in subs:
            s_.set_y(base)
        proof = m.VGroup(colored([("CartPole, this Mac:", MUTED), (f"both tricks {HL['n_solved']}/{SEEDS}", FULL_C)], 24),
                         colored([("·", MUTED), (f"no replay 0/{SEEDS}", NR_C), ("·", MUTED), (f"no target 0/{SEEDS}", NT_C)], 24))
        proof.arrange(m.RIGHT, buff=0.3).move_to([0, -1.2, 0])
        bot = rover(0.6).move_to([-2.4, -2.6, 0])
        nxt = pill("episode 14 · values that run too high", VALUE, BLUE_FILL, 22).next_to(bot, m.RIGHT, buff=0.45)

        self.say(
            "[To sum up]: [store every step], [sample at random], [and freeze the target.]",
            m.AnimationGroup(self.new_header("10 · recap", "Store, sample, freeze"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[On this Mac, that balanced the pole in five runs out of five]; [take away either trick, and none did.]",
            m.FadeIn(proof[0], shift=0.1 * m.UP),
            m.FadeIn(proof[1], shift=0.1 * m.UP),
        )
        self.say(
            "[But the runs still wobbled, and the values ran a little high.] [Next time: why, and a fix.]",
            m.FadeIn(bot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()
