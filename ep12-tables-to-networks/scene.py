"""Trial and Reward · Episode 12 · From Tables to Networks (Module 4 · Deep Value Learning)

Module 4 opens by retiring the table. Every value so far lived in one (Rover's world: 11 states). CartPole's state is
four real numbers (cart position and velocity, pole angle and turning speed): pushing at random, the replayed episode
(research/experiments/trajectories.json, random_cartpole, reset seed 2026) falls after 30 steps, and a table of 10
bins per number already needs 10,000 cells (100 bins: 10^8), each learning alone. On a map of two of the four numbers
(pole angle × turning speed) a 10 × 10 table dissolves into a smooth value surface: the values of the network trained
on this Mac (weights/dqn_cartpole, Episode 13's DQN, 4-256-256-2, 67,586 parameters), sliced with the cart centred
and still. V(s) ≈ f(s; θ). Features and generalisation with tile coding (S&B §9.5.4): four offset 4 × 4 grids, 64
features with exactly four ones; one update nudges four weights and moves every state that shares a tile; the next
state of a real transition (steps 18 → 19 of the replay) shares three of them, so the target moves too. Semi-gradient
TD in one line (S&B eq. 9.7 with the TD target; §9.3), Tsitsiklis & Van Roy (1997) in one line. The deadly triad (S&B
§11.3): function approximation + bootstrapping (Episode 9) + off-policy data (Episode 10's Q-learning). Baird's
counterexample computed (bairds_counterexample: w8 = 316.7 after 1,000 steps; |w| = 5.1e11 after 10,000; α = 0.001
still climbs to 405). Then for real: the DQN without a target network (deep_rl.json dqn_no_target, 5 seeds on this
Mac): mean Q-values 4.1e9, 0 of 5 seeds solved, against ≈ 110 with a target network — Episode 13.

Every number on screen is computed below from rl_reference.py and asserted, or loaded with measured(...), or cited.

Sources (research/notes.md §12, Appendix A rows 12, Appendix C row 12; facts dated October 2026):
- Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018): §9.3 (semi-gradient: bootstrapping
  methods "take into account the effect of changing the weight vector on the estimate, but ignore its effect on the
  target"), eq. 9.7/9.9 (gradient and linear TD updates), §9.5.4 tile coding (four 4 × 4 tilings → 64 components, all
  0 except four; asymmetric offsets), §11.2 Baird's counterexample (Figs. 11.1–11.2: seven states, eight weights,
  rewards 0, γ = 0.99, α = 0.01, w0 = (1,1,1,1,1,1,10,1); "The instability occurs for any positive step size, no
  matter how small"), §11.3 the deadly triad ("If any two elements of the deadly triad are present, but not all
  three, then instability can be avoided"). Baird is S&B §11.2, not Example 11.1 (Appendix A).
- Tsitsiklis & Van Roy (1997), IEEE Trans. Automatic Control 42(5): linear TD(λ) on trajectories of the Markov chain
  (on-policy) converges with probability one.
- Gymnasium CartPole-v1 (Barto, Sutton & Anderson, 1983): state = (x, ẋ, θ, θ̇), ±12° and ±2.4 end the episode, +1 a
  step; rl_reference.cartpole_step is an exact port (classic.json gym_ports).
- research/experiments/classic.md §8 + classic.json baird (headline baird_td_w8_after_1000 = 316.7,
  td_norm_after_10000 = 5.089e11) — recomputed here and asserted.
- research/experiments/deep_rl.md (Episode 13 section) + deep_rl.json headline dqn / dqn_no_target and curves
  q_batch_mean (5 seeds, mean and min–max, every 5,000 env steps up to 500,000).
- weights/dqn_cartpole.json: "Q-values are discounted returns with gamma = 0.99 (ceiling 1/(1-gamma) = 100)".

Illustrative (computed, but chosen): the 2-D slice (x = ẋ = 0) and its colour scale; the 10 × 10 table's values are
the same network at the bin centres; the tile coding (four 4 × 4 tilings stretched over the slice, offsets (1, 3)
in quarter tiles); the state and next state of the tile example (steps 18 and 19 of the random replay); the "one
cell learns, its neighbours don't" strip in section 0 (no values shown).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from rl_reference import *  # noqa: E402,F403

PREFIX = "ep12"
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------- the numbers (all computed here)

# --- Rover's world: a table of eleven values (Episode 7)
RW = ROVER_WORLD
V_PLAN, _SWEEPS = value_iteration(RW)
assert len(RW.states) == 11 and all(round(V_PLAN[s], 3) == RN_UTILITIES[s] for s in RN_UTILITIES)

# --- CartPole: the random policy's replay (trajectories.json, "policies")
TRAJ = json.loads((EXPERIMENTS / "trajectories.json").read_text())["policies"]["random_cartpole"]
STATES = np.array(TRAJ["states"], dtype=float)
ACTS = list(TRAJ["actions"])
N_STEPS = TRAJ["length"]
assert N_STEPS == 30 == len(ACTS) == len(STATES) - 1 and TRAJ["return"] == 30.0          # "after thirty steps"
assert TRAJ["terminated"] and not TRAJ["truncated"] and TRAJ["env"] == "CartPole-v1"
_s, _ends = STATES[0].copy(), []
for _t, _a in enumerate(ACTS):                       # the exact gymnasium port replays the recorded states
    _s, _r, _done = cartpole_step(_s, _a)
    assert np.abs(_s - STATES[_t + 1]).max() < 1e-3 and _r == 1.0
    _ends.append(_done)
assert _ends == [False] * 29 + [True]
TH_MAX = CARTPOLE["theta_threshold"]                 # 12 degrees in radians
assert abs(math.degrees(TH_MAX) - 12) < 1e-9
assert abs(STATES[-1][2]) > TH_MAX and abs(STATES[-1][0]) < CARTPOLE["x_threshold"]      # it fell by its angle
FALL_DEG = math.degrees(STATES[-1][2])
BINS_10, BINS_100 = 10 ** 4, 100 ** 4
assert (BINS_10, BINS_100) == (10_000, 100_000_000)                                       # "ten thousand", "a hundred million"

# --- the network trained on this Mac (Episode 13's DQN) and its value surface on a 2-D slice
NET = load_policy("dqn_cartpole")
N_PARAMS = sum(W.size + b.size for W, b in NET.params)
assert NET.meta["layers"] == [4, 256, 256, 2] and NET.output == "q" and N_PARAMS == 67_586   # "about sixty-eight thousand"
OM_MAX = 2.5                                         # the map's turning-speed range, rad/s (illustrative window)
MAP_TH = math.radians(13.5)                          # the map's angle range: a little past the ±12° edge


def net_value(theta, omega):
    """The network's value of states with the cart centred and still: the larger of its two Q-values."""
    th, om = np.broadcast_arrays(np.asarray(theta, float), np.asarray(omega, float))
    obs = np.stack([np.zeros(th.size), np.zeros(th.size), th.ravel(), om.ravel()], axis=1)
    with np.errstate(all="ignore"):              # Accelerate's matmul raises spurious warnings
        return NET(obs).max(axis=1).reshape(th.shape)


_GX, _GY = 162, 116                                  # network evaluations; the image is a bilinear upsampling
_th = np.linspace(-MAP_TH, MAP_TH, _GX)
_om = np.linspace(OM_MAX, -OM_MAX, _GY)                # row 0 = the top of the map
V_GRID = net_value(*np.meshgrid(_th, _om))
_inside = np.abs(_th) <= TH_MAX
V_LO, V_HI = float(V_GRID[:, _inside].min()), float(V_GRID[:, _inside].max())   # colour scale from states inside ±12°
assert np.isfinite(V_GRID).all() and V_HI > V_LO
HIGH_PT, LOW_PTS = (math.radians(6.0), -1.4), [(math.radians(8.5), 1.9), (math.radians(-8.5), -1.9)]
V_HIGH = float(net_value(*HIGH_PT))
V_LOWS = [float(net_value(*p)) for p in LOW_PTS]
assert V_HIGH > 100 and all(v < 70 for v in V_LOWS)            # "high where swinging back, low where falling"
assert float(net_value(0.0, 0.0)) > 95                         # upright and still: high too
STRENGTH = 0.85


def surface_colour(v):
    f = float(np.clip((v - V_LO) / (V_HI - V_LO), 0, 1))
    return mix(CARD, BLUE, STRENGTH * f)


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    out = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    if not path.exists() or path.read_bytes() != out:
        path.write_bytes(out)


MAP_W, MAP_H = 6.0, 4.3
MAP_C = np.array([-3.05, -0.3, 0.0])
_PX_W, _PX_H = 810, round(810 * MAP_H / MAP_W)


def _surface_png():
    xi, yi = np.linspace(0, _GX - 1, _PX_W), np.linspace(0, _GY - 1, _PX_H)
    x0, y0 = np.minimum(xi.astype(int), _GX - 2), np.minimum(yi.astype(int), _GY - 2)
    fx, fy = (xi - x0)[None, :], (yi - y0)[:, None]
    V = ((1 - fy) * ((1 - fx) * V_GRID[y0][:, x0] + fx * V_GRID[y0][:, x0 + 1])
         + fy * ((1 - fx) * V_GRID[y0 + 1][:, x0] + fx * V_GRID[y0 + 1][:, x0 + 1]))
    f = np.clip((V - V_LO) / (V_HI - V_LO), 0, 1)[..., None] * STRENGTH
    lo = np.array([int(CARD[i:i + 2], 16) for i in (1, 3, 5)], float)
    hi = np.array([int(BLUE[i:i + 2], 16) for i in (1, 3, 5)], float)
    rgb = np.round(lo + (hi - lo) * f).astype(np.uint8)
    path = HERE / "ep12-surface.png"
    write_png(path, rgb)
    return path


SURFACE_PNG = _surface_png()
TABLE_N = 10                                          # 10 × 10 bins over the map
_tc = (np.arange(TABLE_N) + 0.5) / TABLE_N
TABLE_V = net_value(*np.meshgrid(-MAP_TH + 2 * MAP_TH * _tc, OM_MAX - 2 * OM_MAX * _tc))  # [row from top][col]

# --- tile coding over the same slice (Sutton & Barto §9.5.4): four 4 × 4 tilings, offsets (1, 3) quarter tiles
N_TILINGS, TILES = 4, 4
TILE_TH, TILE_OM = 2 * MAP_TH / 3.25, 2 * OM_MAX / 3.25        # 4 tiles cover the map plus the largest offset
OFFSETS = [((k * 1) % 4 / 4, (k * 3) % 4 / 4) for k in range(N_TILINGS)]


def tiling_origin(k):
    ox, oy = OFFSETS[k]
    return -MAP_TH - ox * TILE_TH, -OM_MAX - oy * TILE_OM


def active_tiles(theta, omega):
    """(i, j) of the tile the state lands in, for each tiling (i along the angle, j along the turning speed)."""
    out = []
    for k in range(N_TILINGS):
        x0, y0 = tiling_origin(k)
        i, j = int(math.floor((theta - x0) / TILE_TH)), int(math.floor((omega - y0) / TILE_OM))
        assert 0 <= i < TILES and 0 <= j < TILES
        out.append((i, j))
    return out


def features(theta, omega):
    x = np.zeros(N_TILINGS * TILES * TILES)
    for k, (i, j) in enumerate(active_tiles(theta, omega)):
        x[k * 16 + j * 4 + i] = 1.0
    return x


T_S = 18                                               # the example transition: steps 18 → 19 of the replay
S_TH, S_OM = STATES[T_S][2], STATES[T_S][3]
S2_TH, S2_OM = STATES[T_S + 1][2], STATES[T_S + 1][3]
X_S = features(S_TH, S_OM)
assert X_S.size == 64 and X_S.sum() == 4                                            # "sixty-four numbers, four of them ones"
SHARED = int(X_S @ features(S2_TH, S2_OM))
assert SHARED == 3                                                                  # "it shares three tiles"
TILES_S, TILES_S2 = active_tiles(S_TH, S_OM), active_tiles(S2_TH, S2_OM)
_bin = lambda v, lo, hi: int(np.clip(math.floor((v - lo) / (hi - lo) * TABLE_N), 0, TABLE_N - 1))
S_CELL = (_bin(-S_OM, -OM_MAX, OM_MAX), _bin(S_TH, -MAP_TH, MAP_TH))                 # (row from top, col)

# --- Baird's counterexample (S&B §11.2), semi-gradient off-policy TD(0)
B_STEPS = 10_000
HB = bairds_counterexample(steps=B_STEPS)                         # alpha 0.01, gamma 0.99, seed 0
HB_SLOW = bairds_counterexample(steps=B_STEPS, alpha=0.001)
NORM, NORM_SLOW = np.linalg.norm(HB, axis=1), np.linalg.norm(HB_SLOW, axis=1)
BAIRD = measured("classic", "baird")
assert BAIRD["alpha"] == 0.01 and BAIRD["gamma"] == 0.99 and BAIRD["w0"] == list(HB[0])
assert round(HB[1000][7], 1) == measured("classic", "headline", "baird_td_w8_after_1000") == 316.7   # "about three hundred and seventeen"
assert abs(NORM[B_STEPS] / BAIRD["td_norm_after_10000"] - 1) < 1e-3 and 5.0e11 < NORM[B_STEPS] < 5.2e11  # "about five hundred billion"
assert round(NORM[1000], 1) == BAIRD["methods"]["td"]["norm"]["1000"]
assert np.all(HB[:1001] > 0)                                       # every weight stays positive: bars from zero
assert NORM_SLOW[B_STEPS] > NORM_SLOW[5000] > NORM_SLOW[1000] > NORM_SLOW[0] and round(NORM_SLOW[B_STEPS]) == 405
B_TOP = 330.0

# --- the deep Q-network without a target network (measured on this Mac, 5 seeds)
NT = measured("deep_rl", "headline", "dqn_no_target")
DQ = measured("deep_rl", "headline", "dqn")
assert NT["solved_seeds"] == "0/5" and NT["n_solved"] == 0 and DQ["solved_seeds"] == "5/5"   # "not one seed"
Q_NT, Q_DQ = NT["q_batch_mean_final_mean"], DQ["q_batch_mean_final_mean"]
assert 4.0e9 < Q_NT < 4.2e9 and 90 < Q_DQ < 140                   # "past four billion", "near a hundred"
C_NT = measured("deep_rl", "curves", "dqn_no_target", "q_batch_mean")
C_DQ = measured("deep_rl", "curves", "dqn", "q_batch_mean")
assert C_NT["x_steps"][-1] == C_DQ["x_steps"][-1] == 496_000 and abs(C_NT["mean"][-1] / Q_NT - 1) < 0.03   # curves sampled every 5k
Q_CEIL = 1 / (1 - 0.99)
assert round(Q_CEIL) == 100 and "1/(1-gamma) = 100" in NET.meta["notes"]                # "no state is worth more than a hundred"

# ---------------------------------------------------------------------------- look

ALERT = recolored("alert", RED, RED_FILL, prefix=PREFIX)
PARAM = VIOLET                                         # θ, the weights: the function's parameters (the network's colour)
XL = 0.85                                              # the right panel's left edge (beside the map)


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def left_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([0, y, 0]).align_to([x, 0, 0], m.RIGHT)


def mono_baseline(words):
    flat = [g.get_bottom()[1] for g, c in zip(words, words.text.replace(" ", "")) if c not in "gjpqy,;()[]{}/"]
    return float(np.median(flat)) if len(words) == len(words.text.replace(" ", "")) and flat else float(
        np.median([g.get_bottom()[1] for g in words]))


def sit_with(words, ref):
    """Put the baseline of `words` on the baseline of `ref` (Menlo and Avenir digits/letters without descenders)."""
    return words.shift((mono_baseline(ref) - mono_baseline(words)) * m.UP)


def at(theta, omega):
    """Scene point of a state on the map (pole angle → right, turning speed → up)."""
    return MAP_C + np.array([theta / MAP_TH * MAP_W / 2, omega / OM_MAX * MAP_H / 2, 0.0])


def map_rect(th0, th1, om0, om1, clip=True):
    if clip:
        th0, th1 = max(th0, -MAP_TH), min(th1, MAP_TH)
        om0, om1 = max(om0, -OM_MAX), min(om1, OM_MAX)
    a, b = at(th0, om0), at(th1, om1)
    return m.Rectangle(width=b[0] - a[0], height=b[1] - a[1]).move_to((a + b) / 2)


def tile_rect(k, i, j):
    x0, y0 = tiling_origin(k)
    return map_rect(x0 + i * TILE_TH, x0 + (i + 1) * TILE_TH, y0 + j * TILE_OM, y0 + (j + 1) * TILE_OM)


def tiling_lines(k):
    x0, y0 = tiling_origin(k)
    lines = m.VGroup()
    for i in range(TILES + 1):
        th = x0 + i * TILE_TH
        if -MAP_TH + 1e-6 < th < MAP_TH - 1e-6:
            lines.add(m.Line(at(th, -OM_MAX), at(th, OM_MAX)))
    for j in range(TILES + 1):
        om = y0 + j * TILE_OM
        if -OM_MAX + 1e-6 < om < OM_MAX - 1e-6:
            lines.add(m.Line(at(-MAP_TH, om), at(MAP_TH, om)))
    return lines.set_stroke(STATE, 1.6, opacity=0.6)


def state_map():
    """The map's frame (the ±12° sides dashed RED: the episode ends there), ticks and axis names."""
    frame = m.Rectangle(width=MAP_W, height=MAP_H).move_to(MAP_C).set_stroke(EDGE, 2).set_fill(CARD, 1).set_z_index(-3)
    edges = m.VGroup(*[m.DashedLine(at(s * TH_MAX, -OM_MAX), at(s * TH_MAX, OM_MAX), dash_length=0.12).set_stroke(RED, 3)
                       for s in (-1, 1)]).set_z_index(2)
    beyond = m.VGroup(*[map_rect(s * TH_MAX, s * MAP_TH, -OM_MAX, OM_MAX, clip=False) if s > 0 else
                        map_rect(-MAP_TH, -TH_MAX, -OM_MAX, OM_MAX, clip=False) for s in (-1, 1)])
    beyond.set_stroke(width=0).set_fill(BG, 0.6).set_z_index(-1.5)
    ticks = m.VGroup()
    for deg in (-12, -6, 0, 6, 12):
        p = at(math.radians(deg), -OM_MAX)
        ticks.add(m.Line(p, p + [0, -0.09, 0]).set_stroke(FAINT, 2),
                  code(minus(f"{deg:+d}°" if deg else "0°"), 20, RED if abs(deg) == 12 else MUTED).move_to(p + [0, -0.32, 0]))
    for om in (-2, 0, 2):
        p = at(-MAP_TH, om)
        ticks.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2),
                  right_at(code(minus(f"{om:d}"), 20, MUTED), p[0] - 0.17, p[1]))
    x_name = text("pole angle", 22, MUTED).move_to([MAP_C[0], MAP_C[1] - MAP_H / 2 - 0.8, 0])
    y_name = left_at(text("turning speed (rad/s)", 22, MUTED), MAP_C[0] - MAP_W / 2, MAP_C[1] + MAP_H / 2 + 0.3)
    return m.VGroup(frame, beyond, edges, ticks, x_name, y_name)


def surface():
    image = m.ImageMobject(str(SURFACE_PNG))
    image.height = MAP_H
    image.move_to(MAP_C)
    image.set_z_index(-2)
    return image


def table_cells():
    cw, ch = MAP_W / TABLE_N, MAP_H / TABLE_N
    left_top = MAP_C + np.array([-MAP_W / 2, MAP_H / 2, 0])
    cells = m.VGroup()
    for r in range(TABLE_N):
        for c in range(TABLE_N):
            sq = m.Rectangle(width=cw, height=ch).move_to(left_top + [(c + 0.5) * cw, -(r + 0.5) * ch, 0])
            cells.add(sq.set_stroke(EDGE, 1.5).set_fill(surface_colour(TABLE_V[r][c]), 1))
    return cells.set_z_index(-1)


def path_points(states):
    return [at(s[2], s[3]) for s in states]


def legend_bar(width=3.0):
    n = 30
    seg = width / n
    bar = m.VGroup(*[m.Rectangle(width=seg + 0.005, height=0.22).set_stroke(width=0)
                     .set_fill(mix(CARD, BLUE, STRENGTH * i / (n - 1)), 1).move_to([i * seg, 0, 0]) for i in range(n)])
    box = m.Rectangle(width=width + seg, height=0.22).move_to(bar).set_stroke(EDGE, 1.5).set_fill(opacity=0)
    lo, hi = code("low", 20, MUTED), code("high", 20, VALUE)
    lo.next_to(box, m.LEFT, buff=0.18)
    hi.next_to(box, m.RIGHT, buff=0.18)
    return m.VGroup(bar, box, lo, hi)


def log_ticks(plot, decades, size=30):
    group = m.VGroup()
    for k in decades:
        p = plot.coords(plot.x0, k)
        group.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2),
                  tex("1" if k == 0 else f"10^{{{k}}}", size, MUTED).next_to(p, m.LEFT, buff=0.16))
    return group


def lg(v):
    return float(np.log10(max(float(v), 1.0)))


def sci(v, digits=1):
    mant, exp = f"{v:.{digits}e}".split("e")
    return tex(rf"{mant} \times 10^{{{int(exp)}}}", 34)


def mini_net(center, width=2.4, height=1.2):
    return NeuralNet(layers=(4, 7, 7, 2), width=width, height=height, center=center, color=PARAM, radius=0.075)


# ---------------------------------------------------------------------------- the episode


class Ep12FromTablesToNetworks(Episode):
    number = 12

    def lesson(self):
        self.too_many()
        self.a_function()
        self.generalise()
        self.semi_gradient()
        self.triad()
        self.baird()
        self.for_real()
        self.recap()

    # ------------------------------------------------------------------ 0 · four numbers, no table

    def too_many(self):
        self.next_section("0 · Four numbers, no table", notes=(
            f"Rover's world (R&N 4x3, Episodes 5–8) has {len(RW.states)} states, so its values fit a table (V from "
            "value_iteration, Episode 7). CartPole-v1 (Gymnasium; Barto, Sutton & Anderson 1983): the state is four real "
            "numbers (cart position m, cart velocity m/s, pole angle rad, pole angular velocity rad/s); push left or right; "
            "+1 every step; the episode ends past ±12° or ±2.4 m. The replay is research/experiments/trajectories.json "
            f"random_cartpole (reset seed 2026, actions sampled 50/50): {N_STEPS} steps, return {TRAJ['return']:.0f}, ended "
            f"by the angle ({FALL_DEG:.1f}°); re-simulated here with rl_reference.cartpole_step (agrees to 1e-3). Bins: "
            f"10 per number → 10^4 = {BINS_10:,} cells; 100 per number → 10^8 (notes §12 worked example 4). The strip "
            "is illustrative (no values)."))
        grid = GridView.from_world(RW, cell=1.05, center=(-2.6, -0.35, 0))
        stat_ = m.VGroup(text("Rover's world", 26, MUTED), title_text("11 states", 64, STATE),
                         text("a table: one value for each", 22, MUTED)).arrange(m.DOWN, buff=0.22).move_to([3.4, 0.45, 0])
        mascot = rover(0.5).move_to([3.4, -1.75, 0])

        # CartPole, its counters and the four numbers
        view = CartPoleView(center=(-2.4, -0.75, 0), units_per_m=1.0)
        demo = m.VGroup(
            m.Arrow(view.cart.get_left() + [-0.15, 0, 0], view.cart.get_left() + [-1.0, 0, 0], buff=0, color=ACTION,
                    stroke_width=6, max_tip_length_to_length_ratio=0.3),
            m.Arrow(view.cart.get_right() + [0.15, 0, 0], view.cart.get_right() + [1.0, 0, 0], buff=0, color=ACTION,
                    stroke_width=6, max_tip_length_to_length_ratio=0.3))
        demo_labels = m.VGroup(code("left", 20, ACTION).next_to(demo[0], m.UP, buff=0.12),
                               code("right", 20, ACTION).next_to(demo[1], m.UP, buff=0.12))
        rew = reward_chip(1, 22)
        rew_lab = text("every step it stays up", 22, MUTED)
        rew_row = m.VGroup(rew, rew_lab).arrange(m.RIGHT, buff=0.2).move_to([-2.4, -2.05, 0])
        step_t = [code(f"step {t}", 24, INK) for t in range(N_STEPS + 1)]
        ret_t = [code(f"return {t}", 24, REWARD) for t in range(N_STEPS + 1)]
        for a_, b_ in zip(step_t, ret_t):
            right_at(a_, -2.65, 2.05)
            left_at(b_, -2.15, 2.05)
        step_slot, ret_slot = step_t[0].copy(), ret_t[0].copy()
        cite_cp = cite("CartPole-v1 · Barto, Sutton & Anderson, 1983").move_to([-2.4, -3.3, 0])

        names = ["cart position", "cart velocity", "pole angle", "turning speed"]
        units = ["m", "m/s", "rad", "rad/s"]
        box = card(5.2, 3.15, STATE, CARD, radius=0.2, stroke=2.5).move_to([4.05, 0.15, 0])
        title_ = left_at(text("the state: four numbers", 24, STATE, weight=m.BOLD), 1.75, 1.3)
        ys = [0.65, 0.05, -0.55, -1.15]
        row_names = [left_at(text(n_, 22, MUTED), 1.75, y) for n_, y in zip(names, ys)]
        nums = [[right_at(code(minus(f"{STATES[t][i]:+.3f}"), 26, STATE), 5.45, ys[i]) for i in range(4)]
                for t in range(N_STEPS + 1)]
        for t in range(N_STEPS + 1):
            for i in range(4):
                sit_with(nums[t][i], code("0", 26).move_to([0, ys[i], 0]))
        unit_t = [left_at(code(u, 22, MUTED), 5.6, y) for u, y in zip(units, ys)]
        for u, n_ in zip(unit_t, nums[0]):
            sit_with(u, n_)
        for rn, n_ in zip(row_names, nums[0]):
            rn.shift((n_.get_bottom()[1] - rn.get_bottom()[1]) * m.UP)
        slots = [nums[0][i].copy() for i in range(4)]
        readout = m.VGroup(box, title_, *row_names, *slots, *unit_t)

        push = m.Arrow(np.zeros(3), np.array([1.0, 0, 0]), buff=0, color=ACTION, stroke_width=6)
        fell = text(f"fell: past 12°", 24, RED, weight=m.BOLD)

        def push_arrow(action):
            c = view.cart.get_center()
            if action == 1:
                return m.Arrow(c + [-1.25, 0, 0], c + [-0.5, 0, 0], buff=0, color=ACTION, stroke_width=6,
                               max_tip_length_to_length_ratio=0.3)
            return m.Arrow(c + [1.25, 0, 0], c + [0.5, 0, 0], buff=0, color=ACTION, stroke_width=6,
                           max_tip_length_to_length_ratio=0.3)

        push.become(push_arrow(ACTS[0])).set_opacity(0)
        live = m.VGroup(view, *slots, step_slot, ret_slot, push)

        def run(_, alpha):
            i = alpha * N_STEPS
            j = min(int(math.floor(i)), N_STEPS - 1)
            f = i - j
            x = (1 - f) * STATES[j][0] + f * STATES[j + 1][0]
            th = (1 - f) * STATES[j][2] + f * STATES[j + 1][2]
            view.set_state(x, th)
            k = min(int(math.floor(i + 1e-9)), N_STEPS)
            for r in range(4):
                slots[r].become(nums[k][r])
            step_slot.become(step_t[k])
            ret_slot.become(ret_t[k])
            push.become(push_arrow(ACTS[j]))
            if alpha >= 1.0:
                push.set_opacity(0)

        fell.move_to([-0.95, 1.25, 0])

        # bins
        hx1, hx2 = 2.55, 6.45
        bin_head = m.VGroup(code("bins per number", 20, FAINT).move_to([hx1, -1.95, 0]),
                            right_at(code("table cells", 20, FAINT), hx2, -1.95))
        bin_rows = m.VGroup()
        for k, (b, n_) in enumerate(((10, BINS_10), (100, BINS_100))):
            y = -2.45 - 0.48 * k
            bin_rows.add(m.VGroup(code(f"{b}", 26, INK).move_to([hx1, y, 0]),
                                  right_at(code(f"{n_:,}", 26, INK if k == 0 else RED, weight=m.BOLD), hx2, y)))
        # every cell learns alone
        strip = m.VGroup(*[m.Square(0.46).set_stroke(EDGE, 2).set_fill(CARD, 1) for _ in range(10)]).arrange(m.RIGHT, buff=0.06)
        strip.move_to([-2.4, -2.55, 0])
        qs = m.VGroup(*[code("?", 22, MUTED, weight=m.BOLD).move_to(sq) for sq in strip])
        strip_lab = cite("the pole's angle in ten bins").move_to([-2.4, -3.2, 0])
        lit = 6
        learned = m.Square(0.46).move_to(strip[lit]).set_stroke(VALUE, 3).set_fill(BLUE_FILL, 1)
        learned_v = code("V", 22, VALUE, weight=m.BOLD).move_to(strip[lit])
        nbrs = m.VGroup(strip[lit - 1], strip[lit + 1])

        self.say(
            "[Until now, every value lived in a table], [one entry for each state]: [Rover's world needed just eleven.]",
            m.AnimationGroup(self.new_header("00 · too many states", "Four numbers, no table"), m.FadeIn(grid)),
            grid.world_values(V_PLAN, fmt="{:.2f}", size=20, run_time=1.0),
            m.AnimationGroup(m.FadeIn(stat_, shift=0.15 * m.UP), m.FadeIn(mascot, shift=0.15 * m.UP)),
        )
        self.say(
            "[Now meet CartPole]: [push a cart left or right to keep a pole upright], "
            "[and earn plus one for every step it stays up.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(grid, stat_, mascot, *grid.numbers.values()), shift=0.3 * m.LEFT),
                             m.FadeIn(view, shift=0.2 * m.UP), m.FadeIn(cite_cp)),
            m.AnimationGroup(m.GrowArrow(demo[0]), m.GrowArrow(demo[1]), m.FadeIn(demo_labels),
                             m.FadeIn(readout, shift=0.15 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(rew_row, shift=0.1 * m.UP), m.FadeIn(step_slot), m.FadeIn(ret_slot)),
        )
        self.say(
            "[Pushing at random, the pole falls after thirty steps.]",
            m.Succession(m.FadeOut(m.VGroup(demo, demo_labels), run_time=0.3),
                         m.UpdateFromAlphaFunc(live, run, run_time=3.6, rate_func=m.linear),
                         m.AnimationGroup(m.Flash(view.pole.get_end(), color=RED, flash_radius=0.35, line_length=0.18),
                                          m.FadeIn(fell, shift=0.1 * m.UP), run_time=0.7)),
        )
        self.say(
            "[Its state is four real numbers]: [the cart's position and speed], [the pole's angle, and how fast it turns.]",
            m.Circumscribe(m.VGroup(*slots, *unit_t), color=STATE, buff=0.1),
            m.Circumscribe(m.VGroup(*row_names[:2], *slots[:2], *unit_t[:2]), color=STATE, buff=0.08),
            m.Circumscribe(m.VGroup(*row_names[2:], *slots[2:], *unit_t[2:]), color=STATE, buff=0.08),
        )
        self.say(
            "[Cut each number into ten bins], [and a table needs ten thousand cells]; [a hundred bins, a hundred million.]",
            m.AnimationGroup(m.FadeIn(bin_head), m.FadeIn(bin_rows[0][0], shift=0.1 * m.UP)),
            m.FadeIn(bin_rows[0][1], shift=0.1 * m.UP),
            m.FadeIn(bin_rows[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Worse, every cell learns alone]: [what one cell learns] [teaches its neighbour nothing.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(rew_row, cite_cp)), m.FadeIn(strip), m.FadeIn(qs), m.FadeIn(strip_lab)),
            m.AnimationGroup(m.FadeOut(qs[lit]), m.FadeIn(learned), m.FadeIn(learned_v, scale=0.6),
                             m.Flash(strip[lit].get_center(), color=SIGNAL, flash_radius=0.4, line_length=0.15)),
            m.AnimationGroup(*[m.Circumscribe(n_, color=RED, buff=0.05) for n_ in nbrs]),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · a value for every point

    def a_function(self):
        self.next_section("1 · A value for every point", notes=(
            "The map: pole angle (±13.5°; the dashed RED lines at ±12° end the episode, the strips beyond are dimmed) × "
            "turning speed (±2.5 rad/s), two of CartPole's four numbers; the cart is held centred and still (x = ẋ = 0) "
            f"for the colours. The path is the random replay's 31 states; its last one lies past −12° ({FALL_DEG:.1f}°). Values: weights/dqn_cartpole (Episode 13's DQN with "
            "replay and a target network, seed 10, best weights at 410k steps, greedy 500/500 on 100 evaluation "
            f"episodes), value = the larger of its two Q-values; colour from {V_LO:.0f} (dark) to {V_HI:.0f} (bright). "
            f"Its values slightly exceed the true ceiling of 100 (γ = 0.99): overestimation, Episode 14. "
            f"High: e.g. +6°, −1.4 rad/s → {V_HIGH:.0f} (leaning right, turning back); low: ±8.5°, ±1.9 rad/s → "
            f"{V_LOWS[0]:.0f} / {V_LOWS[1]:.0f} (falling). The 10 × 10 table holds the same network's values at the bin "
            f"centres; colour scale from the states inside ±12°. V(s) ≈ f(s; θ); the network has {N_PARAMS:,} parameters "
            "(4·256 + 256 + 256·256 + 256 + 256·2 + 2)."))
        smap = state_map()
        image = surface()
        cells = table_cells()
        pts = path_points(STATES)
        trail = m.VMobject().set_points_as_corners([pts[0], pts[0] + [1e-4, 0, 0]]).set_stroke(STATE, 4).set_z_index(3)
        dot = m.Dot(pts[0], radius=0.09, color=STATE).set_z_index(4)
        dots = m.VGroup(*[m.Dot(p, radius=0.045, color=STATE) for p in pts[:-1]]).set_z_index(3)
        cross = m.VGroup(m.Line([-0.1, -0.1, 0], [0.1, 0.1, 0]), m.Line([-0.1, 0.1, 0], [0.1, -0.1, 0]))
        cross.set_stroke(RED, 5).move_to(pts[-1]).set_z_index(4)

        mini = CartPoleView(center=(0, 0, 0)).scale(0.55).move_to([3.75, 1.5, 0])
        chips = m.VGroup(pill("position", MUTED, CARD, 20), pill("velocity", MUTED, CARD, 20),
                         pill("angle", STATE, TEAL_FILL, 20), pill("turning speed", STATE, TEAL_FILL, 20))
        chips.arrange_in_grid(rows=2, cols=2, buff=(0.3, 0.25)).move_to([3.75, -0.1, 0])
        trail.set_stroke(opacity=0)
        dot.set_opacity(0)
        live = m.VGroup(mini, trail, dot)

        def run(_, alpha):
            i = alpha * N_STEPS
            j = min(int(math.floor(i)), N_STEPS - 1)
            f = i - j
            mini.set_state((1 - f) * STATES[j][0] + f * STATES[j + 1][0], (1 - f) * STATES[j][2] + f * STATES[j + 1][2])
            p = (1 - f) * pts[j] + f * pts[j + 1]
            trail.set_points_as_corners(pts[:j + 1] + [p + [1e-4, 0, 0]]).set_stroke(STATE, 4, opacity=1)
            dot.move_to(p).set_opacity(1)

        # right panel
        tbl_t = m.VGroup(title_text("A table", 40, INK), text("100 cells here, one value each", 22, MUTED))
        tbl_t.arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT)
        left_at(tbl_t, XL + 0.25, 1.55)
        fn_t = m.VGroup(title_text("A function", 40, VALUE), text("a value for every point", 22, MUTED))
        fn_t.arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT)
        left_at(fn_t, XL + 0.25, 1.55)
        legend = legend_bar(2.8)
        left_at(legend, XL + 0.25, 0.45)
        src = m.VGroup(text("the network trained on this Mac (episode 13)", 20, INK),
                       cite("larger of its two Q-values · cart centred and still"))
        src.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        left_at(src, XL + 0.25, -0.25)
        hi_pill = pill("upright, or swinging back", VALUE, BLUE_FILL, 20).move_to(at(*HIGH_PT) + [0.15, 0.1, 0]).set_z_index(5)
        lo_pills = m.VGroup(*[pill("falling", RED, RED_FILL, 20).move_to(at(*p)).set_z_index(5) for p in LOW_PTS])
        formula = m.MathTex(r"V(s)", r"\approx", r"f(s;", r"\theta", r")", font_size=56, color=INK)
        formula[0].set_color(VALUE)
        formula[3].set_color(PARAM)
        left_at(formula, XL + 0.25, -1.3)
        theta_lab = left_at(text("θ: the parameters we learn", 22, PARAM), XL + 0.25, -1.95)
        net = mini_net((XL + 1.45, -2.8, 0), width=2.3, height=0.8)
        n_lab = m.VGroup(code(f"{N_PARAMS:,} parameters", 24, PARAM, weight=m.BOLD), text("for infinitely many states", 20, MUTED))
        n_lab.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(n_lab, XL + 2.95, -2.8)

        order = list(range(len(cells)))
        np.random.default_rng(12).shuffle(order)

        self.say(
            "[Take two of the four numbers], [the pole's angle and its turning speed], "
            "[and every state becomes a point on this map.]",
            m.AnimationGroup(self.new_header("01 · a function", "A value for every point"), m.FadeIn(mini, shift=0.15 * m.UP),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in chips], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(smap), m.Circumscribe(chips[2], color=STATE, buff=0.06),
                             m.Circumscribe(chips[3], color=STATE, buff=0.06)),
            m.Succession(m.UpdateFromAlphaFunc(live, run, run_time=3.0, rate_func=m.linear),
                         m.AnimationGroup(m.FadeIn(dots), m.FadeIn(cross, scale=0.5), run_time=0.5)),
        )
        self.say(
            "[A table chops the map into cells], [each holding its own value.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(mini, chips), shift=0.2 * m.RIGHT),
                             m.LaggedStart(*[m.FadeIn(cells[i]) for i in order], lag_ratio=0.01, run_time=1.4),
                             trail.animate.set_stroke(opacity=0.5), dots.animate.set_opacity(0.5), dot.animate.set_opacity(0)),
            m.FadeIn(tbl_t, shift=0.1 * m.UP),
        )
        self.say(
            "[A function gives every point a value]: [the table dissolves into a smooth surface.]",
            m.AnimationGroup(m.FadeOut(tbl_t, shift=0.1 * m.UP), m.FadeIn(fn_t, shift=0.1 * m.UP), m.FadeIn(image)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeOut(cells[i], scale=0.6) for i in order], lag_ratio=0.012, run_time=2.2),
                             m.FadeIn(legend, run_time=1.0)),
        )
        self.say(
            "[These values are real]: [they come from the network we train on this Mac in episode thirteen.]",
            m.Circumscribe(m.VGroup(fn_t[0]), color=VALUE, buff=0.1),
            m.FadeIn(src, shift=0.1 * m.UP),
        )
        self.say(
            "[High where the pole is upright or swinging back], [low where it's already falling.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(trail, dots, cross)), m.FadeIn(hi_pill, scale=0.8)),
            m.LaggedStart(*[m.FadeIn(p, scale=0.8) for p in lo_pills], lag_ratio=0.3),
        )
        self.say(
            "[So the value of a state becomes a function], [with parameters theta: the numbers we learn.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(hi_pill, lo_pills)), m.Write(formula)),
            m.FadeIn(theta_lab, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Our network has about sixty-eight thousand parameters], [for infinitely many states.]",
            m.AnimationGroup(m.FadeIn(net), m.FadeIn(n_lab[0], shift=0.1 * m.UP), net.forward(PARAM, run_time=1.4)),
            m.FadeIn(n_lab[1], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.map_keep = (smap, image)
        self.clear(smap, image)

    # ------------------------------------------------------------------ 2 · one update, many states

    def generalise(self):
        self.next_section("2 · One update, many states", notes=(
            "Linear function approximation, V(s; θ) = θᵀx(s) (S&B §9.4). Tile coding (S&B §9.5.4): four 4 × 4 tilings "
            "offset by (1, 3) quarter tiles, stretched over this map (each tile 1/3.25 of the map so every tiling covers "
            "it): 64 binary features, exactly four ones, so V(s) is the sum of four weights. A semi-gradient update adds "
            "αδ to those four weights (∇V = x(s)); every state's value changes by αδ × (tiles it shares with s): the PINK "
            f"overlap. The example is the random replay's step {T_S} → {T_S + 1}: s = ({math.degrees(S_TH):.1f}°, "
            f"{S_OM:.2f} rad/s), s′ = ({math.degrees(S2_TH):.1f}°, {S2_OM:.2f} rad/s); they share {SHARED} of 4 tiles, so "
            "the update also raises the target r + γV(s′). A table changes one cell (the 10 × 10 bin of section 1)."))
        smap, image = self.map_keep
        dim = m.Rectangle(width=MAP_W, height=MAP_H).move_to(MAP_C).set_stroke(width=0).set_fill(BG, 0.62).set_z_index(-1)
        grids = m.VGroup(*[tiling_lines(k) for k in range(N_TILINGS)]).set_z_index(1)
        tiles = m.VGroup(*[tile_rect(k, i, j).set_stroke(STATE, 3.5).set_fill(STATE, 0.0) for k, (i, j) in enumerate(TILES_S)])
        tiles.set_z_index(2)
        pinks = m.VGroup(*[tile_rect(k, i, j).set_stroke(width=0).set_fill(SIGNAL, 0.27) for k, (i, j) in enumerate(TILES_S)])
        pinks.set_z_index(1)
        s_dot = m.VGroup(m.Circle(radius=0.11).set_stroke(INK, 3).set_fill(BG, 1), m.Dot(radius=0.06, color=INK))
        s_dot.move_to(at(S_TH, S_OM)).set_z_index(5)
        s_lab = pill("s", INK, BG, 24, mono=True).next_to(s_dot, m.UL, buff=0.02).set_z_index(6)
        s2_dot = m.VGroup(m.Circle(radius=0.09).set_stroke(STATE, 3).set_fill(BG, 1), m.Dot(radius=0.045, color=STATE))
        s2_dot.move_to(at(S2_TH, S2_OM)).set_z_index(5)
        s2_lab = pill("s′", STATE, BG, 24, mono=True).next_to(s2_dot, m.DR, buff=0.0).set_z_index(6)
        s2_tiles = m.VGroup(*[m.DashedVMobject(tile_rect(k, i, j), num_dashes=40).set_stroke(STATE, 2.5)
                              for k, (i, j) in enumerate(TILES_S2) if (i, j) != TILES_S[k]]).set_z_index(2)
        cw, ch = MAP_W / TABLE_N, MAP_H / TABLE_N
        r_, c_ = S_CELL
        one_cell = m.Rectangle(width=cw, height=ch).move_to(
            MAP_C + np.array([-MAP_W / 2 + (c_ + 0.5) * cw, MAP_H / 2 - (r_ + 0.5) * ch, 0])).set_stroke(SIGNAL, 5).set_z_index(4)
        assert np.linalg.norm(one_cell.get_center() - at(S_TH, S_OM)) < 0.5

        # right panel: V = θᵀx, the feature strip
        lin = m.MathTex(r"V(s;", r"\theta", r")", "=", r"\theta^{\top}", r"x(s)", font_size=50, color=INK)
        lin[0].set_color(VALUE)
        lin[2].set_color(VALUE)
        lin[1].set_color(PARAM)
        lin[4].set_color(PARAM)
        lin[5].set_color(STATE)
        left_at(lin, XL + 0.25, 1.95)
        lin_sub = left_at(text("features x(s), weighed by θ", 22, MUTED), XL + 0.25, 1.4)
        tc_cite = left_at(cite("tile coding · Sutton & Barto, 2018, §9.5.4"), XL + 0.25, 0.95)
        sq_, gap_ = 0.25, 0.05
        strip = m.VGroup()
        for k in range(N_TILINGS):
            row = m.VGroup(*[m.Square(sq_).set_stroke(EDGE, 1.5).set_fill(CARD, 1) for _ in range(16)]).arrange(m.RIGHT, buff=gap_)
            strip.add(row)
        strip.arrange(m.DOWN, buff=gap_)
        left_at(strip, XL + 0.95, -0.05)
        row_labs = m.VGroup(*[code(f"{k + 1}", 20, FAINT).next_to(strip[k], m.LEFT, buff=0.2) for k in range(N_TILINGS)])
        lit = m.VGroup()
        for k, (i, j) in enumerate(TILES_S):
            idx = j * 4 + i
            assert X_S[k * 16 + idx] == 1
            lit.add(m.Square(sq_).move_to(strip[k][idx]).set_stroke(STATE, 2).set_fill(STATE, 1))
        strip_cap = left_at(text("64 features, four of them 1", 20, MUTED), XL + 0.95, -0.82)
        sum_t = left_at(text("V(s) = the sum of four weights", 22, INK, t2c={"V(s)": VALUE}), XL + 0.25, -1.3)
        upd_lab = left_at(text("one update: +αδ on those four weights", 22, SIGNAL), XL + 0.25, -1.8)
        table_lab = left_at(text("a table: just one cell", 22, MUTED), XL + 0.25, -2.3)
        gen = left_at(title_text("generalisation", 34, SIGNAL), XL + 0.25, -2.95)
        net = mini_net((XL + 4.85, -2.9, 0), width=1.4, height=0.75)
        sh_lab = left_at(text(f"s′ shares {SHARED} of its 4 tiles: the target moves too", 22, STATE), XL + 0.25, -2.3)

        self.say(
            "[The simplest function is linear]: [describe a state by features], [then weigh them.]",
            m.AnimationGroup(self.new_header("02 · generalise", "One update, many states"), m.FadeIn(dim),
                             m.FadeIn(s_dot, scale=0.6), m.FadeIn(s_lab)),
            m.Write(lin),
            m.FadeIn(lin_sub, shift=0.1 * m.UP),
        )
        self.say(
            "[Tile coding, for example], [lays four offset grids over the map]; [a state lands in one tile of each.]",
            m.FadeIn(tc_cite),
            m.LaggedStart(*[m.Create(g) for g in grids], lag_ratio=0.35, run_time=2.0),
            m.AnimationGroup(grids.animate.set_stroke(opacity=0.18), m.LaggedStart(*[m.Create(t) for t in tiles], lag_ratio=0.25)),
        )
        self.say(
            "[So its features are sixty-four numbers, four of them ones], [and its value is the sum of those four weights.]",
            m.AnimationGroup(m.FadeIn(strip), m.FadeIn(row_labs), m.FadeIn(strip_cap),
                             m.LaggedStart(*[m.FadeIn(l_, target_position=t_) for l_, t_ in zip(lit, tiles)], lag_ratio=0.25)),
            m.FadeIn(sum_t, shift=0.1 * m.UP),
        )
        self.say(
            "[Now one update at this state]: [it nudges those four weights], [and every state sharing a tile moves too.]",
            m.Flash(s_dot.get_center(), color=SIGNAL, flash_radius=0.4, line_length=0.2, num_lines=12),
            m.AnimationGroup(*[m.Indicate(l_, color=SIGNAL, scale_factor=1.3) for l_ in lit], m.FadeIn(upd_lab, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(p) for p in pinks], lag_ratio=0.25, run_time=1.6),
        )
        self.say(
            "[The closer a state, the more tiles it shares, and the more it moves.] [A table would have moved one cell.]",
            m.Circumscribe(s_dot, color=SIGNAL, buff=0.08, shape=m.Circle),
            m.AnimationGroup(m.Create(one_cell), m.FadeIn(table_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[That's generalisation]: [learn here, and the neighbours learn too.] "
            "[A network does the same, with features it learns itself.]",
            m.FadeIn(gen, shift=0.1 * m.UP),
            m.AnimationGroup(pinks.animate(rate_func=m.there_and_back, run_time=1.2).set_fill(SIGNAL, 0.4)),
            m.AnimationGroup(m.FadeIn(net), net.forward(PARAM, run_time=1.4)),
        )
        self.say(
            "[But one neighbour matters most]: [the next state, whose value is the target.] "
            "[It shares three tiles, so the target moved too.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(one_cell, table_lab, gen, net)), m.Circumscribe(s_dot, color=INK, buff=0.06, shape=m.Circle)),
            m.AnimationGroup(m.FadeIn(s2_dot, scale=0.6), m.FadeIn(s2_lab), m.Create(s2_tiles)),
            m.AnimationGroup(m.FadeIn(sh_lab, shift=0.1 * m.UP), m.Flash(s2_dot.get_center(), color=SIGNAL, flash_radius=0.35, line_length=0.16)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · semi-gradient TD

    def semi_gradient(self):
        self.next_section("3 · Semi-gradient TD", notes=(
            "Semi-gradient TD(0) (S&B eq. 9.7 with the TD target; notes §12): θ ← θ + α[r + γV(s′; θ) − V(s; θ)] ∇V(s; θ). "
            "The bracket is Episode 9's TD error δ. The target depends on θ but is not differentiated: 'They take into "
            "account the effect of changing the weight vector on the estimate, but ignore its effect on the target' "
            "(S&B §9.3), so it is not true gradient descent: semi-gradient. Linear: ∇V(s; θ) = x(s) (eq. 9.9), the four "
            "tiles. Tsitsiklis & Van Roy (1997), Theorem 1: linear TD(λ) on the chain's own trajectories (on-policy) "
            "converges with probability one (to the TD fixed point, not always the best fit; step sizes Σα = ∞, Σα² < ∞)."))
        f = m.MathTex(r"\theta", r"\leftarrow", r"\theta", "+", r"\alpha", "[", "r", "+", r"\gamma", r"V(s';\theta)", "-",
                      r"V(s;\theta)", "]", r"\nabla V(s;\theta)", font_size=56, color=INK)
        for i in (0, 2):
            f[i].set_color(PARAM)
        for i in (5, 12):
            f[i].set_color(SIGNAL)
        f[6].set_color(REWARD)
        f[9].set_color(VALUE)
        f[11].set_color(VALUE)
        f[13].set_color(SIGNAL)
        f.move_to([0, 0.75, 0])
        d_brace = m.Brace(f[5:13], m.DOWN, buff=0.15, color=SIGNAL)
        d_lab = text("the TD error δ, from episode 9", 22, SIGNAL).next_to(d_brace, m.DOWN, buff=0.12)
        t_brace = m.Brace(f[6:10], m.UP, buff=0.15, color=MUTED)
        t_lab = text("the target: treated as fixed", 22, INK).next_to(t_brace, m.UP, buff=0.12)
        g_brace = m.Brace(f[13], m.DOWN, buff=0.15, color=SIGNAL)
        g_lab = text("only this side's gradient", 22, SIGNAL).next_to(g_brace, m.DOWN, buff=0.12)
        if g_lab.get_left()[0] < d_lab.get_right()[0] + 0.3:
            g_lab.align_to(g_brace, m.RIGHT)
        if g_lab.get_left()[0] < d_lab.get_right()[0] + 0.3:
            d_lab.align_to(d_brace, m.LEFT)
        semi = m.VGroup(title_text("semi-gradient", 34, INK), cite("ignores how the target moves · Sutton & Barto, 2018, §9.3"))
        semi.arrange(m.DOWN, buff=0.12).move_to([0, -1.55, 0])
        lin = m.VGroup(text("linear:", 22, MUTED), m.MathTex(r"\nabla V(s;\theta) = x(s)", font_size=40, color=INK),
                       text("the four tiles", 22, STATE)).arrange(m.RIGHT, buff=0.25)
        lin.move_to([-3.2, -2.85, 0])
        tvr = m.VGroup(pill("linear + on-policy: converges", GOOD, GOOD_FILL, 22), cite("Tsitsiklis & Van Roy, 1997"))
        tvr.arrange(m.DOWN, buff=0.12).move_to([3.4, -2.85, 0])

        self.say(
            "[Here's the update.] [It starts from the TD error of episode nine]: "
            "[reward, plus gamma times the next value, minus this value.]",
            m.AnimationGroup(self.new_header("03 · semi-gradient", "Chasing a moving target"), m.FadeIn(f[:5], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(f[5:13], shift=0.1 * m.UP), m.GrowFromCenter(d_brace)),
            m.FadeIn(d_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[Then theta moves along the gradient of this value], [scaled by alpha and the error.]",
            m.AnimationGroup(m.FadeIn(f[13], shift=0.1 * m.LEFT), m.GrowFromCenter(g_brace), m.FadeIn(g_lab)),
            m.AnimationGroup(m.Circumscribe(f[4], color=INK, buff=0.08), m.Circumscribe(f[5:13], color=SIGNAL, buff=0.08)),
        )
        self.say(
            "[The target depends on theta too], [but the update treats it as fixed]: [that's why it's called semi-gradient.]",
            m.AnimationGroup(m.GrowFromCenter(t_brace), m.Indicate(f[9], color=VALUE)),
            m.FadeIn(t_lab, shift=0.1 * m.DOWN),
            m.FadeIn(semi, shift=0.1 * m.UP),
        )
        self.say(
            "[With linear features and on-policy data, it provably converges]: [a result from nineteen ninety-seven.]",
            m.FadeIn(lin, shift=0.1 * m.UP),
            m.FadeIn(tvr, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · the deadly triad

    def triad(self):
        self.next_section("4 · The deadly triad", notes=(
            "Sutton & Barto 2018 §11.3 (p. 264): 'the danger of instability and divergence arises whenever we combine all "
            "of the following three elements, making up what we call the deadly triad': function approximation, "
            "bootstrapping (targets that include existing estimates, as in TD — Episode 9), off-policy training (learning "
            "from a distribution other than the target policy's — Episode 10's Q-learning learns the greedy policy from "
            "ε-greedy moves). 'If any two elements of the deadly triad are present, but not all three, then instability "
            "can be avoided.' Identified by Sutton (1995), analysed by Tsitsiklis & Van Roy (1997), named by Sutton (2015)."))
        specs = [("Function approximation", "a function instead of a table", "this episode", PARAM, (0, 1.4)),
                 ("Bootstrapping", "a guess from a guess", "episode 9 · TD", VALUE, (-3.75, -1.15)),
                 ("Off-policy data", "moves from a different policy", "episode 10 · Q-learning", ACTION, (3.75, -1.15))]
        cards = []
        for name, sub, tag, col, (x, y) in specs:
            box = card(4.7, 1.6, col, CARD, radius=0.2, stroke=3)
            body = m.VGroup(text(name, 28, col, weight=m.BOLD), text(sub, 20, MUTED), pill(tag, col, FILLS[col], 20, mono=True))
            body.arrange(m.DOWN, buff=0.1).move_to(box)
            cards.append(m.VGroup(box, body).move_to([x, y, 0]))
        cards = m.VGroup(*cards).set_z_index(1)
        ctr = [np.array([x, y, 0.0]) for *_, (x, y) in specs]
        edges = m.VGroup(*[m.Line(ctr[a], ctr[b]).set_stroke(RED, 5) for a, b in ((0, 1), (1, 2), (2, 0))]).set_z_index(0)
        safe = m.Line(ctr[0], ctr[1]).set_stroke(GREEN, 5).set_z_index(0)
        check = icon("check", 0.5).move_to((ctr[0] + ctr[1]) / 2 + np.array([0.15, 0.1, 0])).set_z_index(2)
        alert = icon(ALERT, 0.8).move_to([0, -0.05, 0]).set_z_index(2)
        quote = m.VGroup(text("“If any two elements of the deadly triad are present, but not all three,", 22, INK),
                         text("then instability can be avoided.”", 22, INK),
                         cite("Sutton & Barto, 2018, §11.3")).arrange(m.DOWN, buff=0.1).move_to([0, -2.95, 0])
        warn = text("it can blow up", 26, RED, weight=m.BOLD).move_to([0, -0.05, 0])

        self.say(
            "[Change those conditions, and it can blow up.] [Three ingredients do it together.]",
            m.AnimationGroup(self.new_header("04 · the danger", "The deadly triad"), m.FadeIn(warn, scale=0.8)),
            m.FadeOut(warn, scale=1.2),
        )
        self.say("[One: function approximation, this episode's idea.]", m.FadeIn(cards[0], shift=0.15 * m.DOWN))
        self.say("[Two: bootstrapping, learning a guess from a guess, like TD in episode nine.]",
                 m.FadeIn(cards[1], shift=0.15 * m.RIGHT))
        self.say(
            "[Three: off-policy data, learning about one policy from another's moves], [like Q-learning in episode ten.]",
            m.FadeIn(cards[2], shift=0.15 * m.LEFT),
            m.Circumscribe(cards[2][1][2], color=ACTION, buff=0.06),
        )
        self.say(
            "[Sutton and Barto call it the deadly triad]: [any two are safe], [but all three can diverge.]",
            m.FadeIn(quote, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(safe), m.FadeIn(check, scale=0.6)),
            m.AnimationGroup(m.FadeOut(m.VGroup(safe, check)), m.Create(edges), m.FadeIn(alert, scale=0.6)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · Baird's counterexample

    def baird(self):
        self.next_section("5 · Baird's counterexample", notes=(
            "Sutton & Barto 2018 §11.2, Figs. 11.1–11.2 (Baird 1995, slightly modified): seven states; upper state i has "
            "value 2w_i + w8, the lower one w7 + 2w8; every reward 0, γ = 0.99, so w = 0 is exact. Behaviour: dashed "
            "(to one of the six upper states) 6/7, solid (to the lower state) 1/7; target policy: always solid. "
            "rl_reference.bairds_counterexample (semi-gradient off-policy TD(0), importance ratio 7 or 0, α = 0.01, "
            f"w0 = (1,1,1,1,1,1,10,1), seed 0): after 1,000 steps w = {tuple(int(round(v)) for v in HB[1000])} "
            f"(w8 = {HB[1000][7]:.1f}; classic.json 316.7); |w| {NORM[0]:.1f} → {NORM[1000]:.0f} → {NORM[B_STEPS]:.3g} "
            f"after 10,000 (classic.json td_norm_after_10000 = 5.089e11). With α = 0.001: |w| = {NORM_SLOW[B_STEPS]:.0f} "
            "after 10,000 steps, still climbing. 'The instability occurs for any positive step size, no matter how small' "
            "(S&B §11.2). w7 drifts down from 10 while the rest grow, as in Fig. 11.2."))
        xs = [-5.2 + 1.5 * i for i in range(6)]
        up_y, low = 1.7, np.array([-1.45, 0.0, 0.0])
        uppers = m.VGroup()
        for i, x in enumerate(xs):
            box = card(1.3, 0.62, STATE, TEAL_FILL, radius=0.14, stroke=2.5).move_to([x, up_y, 0])
            lab = code(f"2w{i + 1}+w8", 20, INK, t2c={r"w\d": PARAM}).move_to(box)
            uppers.add(m.VGroup(box, lab))
        lower_box = card(1.5, 0.62, STATE, TEAL_FILL, radius=0.14, stroke=2.5).move_to(low)
        lower = m.VGroup(lower_box, code("w7+2w8", 20, INK, t2c={r"w\d": PARAM}).move_to(lower_box))
        solid = m.VGroup(*[m.Arrow(u[0].get_bottom(), lower_box.get_top() + [0.09 * (i - 2.5), 0, 0], buff=0.06, color=ACTION,
                                   stroke_width=3.5, max_tip_length_to_length_ratio=0.12) for i, u in enumerate(uppers)])
        loop = m.Arc(radius=0.26, start_angle=2.2, angle=-4.4, arc_center=lower_box.get_right() + [0.2, 0, 0])
        loop.set_stroke(ACTION, 3.5).add_tip(tip_length=0.14)
        loop.set_color(ACTION)
        frame_up = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.2, width=xs[-1] - xs[0] + 1.75, height=0.98)
                                    .move_to([(xs[0] + xs[-1]) / 2, up_y, 0]), num_dashes=70).set_stroke(ACTION, 2.5)
        d_arc = m.ArcBetweenPoints(lower_box.get_left() + [-0.05, 0, 0], np.array([xs[0] - 0.85, up_y - 0.3, 0]), angle=1.1)
        dashed = m.VGroup(m.DashedVMobject(d_arc, num_dashes=22).set_stroke(ACTION, 3.5),
                          m.Arrow(d_arc.point_from_proportion(0.9), d_arc.point_from_proportion(1.0), buff=0, color=ACTION,
                                  stroke_width=3.5, max_tip_length_to_length_ratio=0.9))
        dash_lab = m.VGroup(code("dashed", 22, ACTION, weight=m.BOLD), text("to one of the six · 6 in 7 moves", 20, MUTED))
        dash_lab.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(dash_lab, -6.5, -0.6)
        solid_lab = m.VGroup(code("solid", 22, ACTION, weight=m.BOLD), text("to the lower state · 1 in 7 moves", 20, MUTED))
        solid_lab.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(solid_lab, 0.35, -0.05)
        zero = m.VGroup(m.VGroup(text("every reward", 22, MUTED), code("0", 28, REWARD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2),
                        code("γ = 0.99", 20, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        left_at(zero, 3.85, 1.85)
        truth = m.VGroup(text("true values: all 0", 22, INK), text("w = 0 would be perfect", 22, GOOD))
        truth.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        left_at(truth, 3.85, 0.75)
        target_p = pill("target policy: always solid", ACTION, CORAL_FILL, 20).move_to([0, 0, 0])
        right_at(target_p, 6.5, -0.7)
        sb = cite("Sutton & Barto, 2018, §11.2").move_to([0, -3.35, 0]).align_to([6.6, 0, 0], m.RIGHT)

        bars = Bars(list(HB[0]), labels=[f"w{i}" for i in range(1, 9)], top=B_TOP, bottom=0.0, width=8.0, height=1.6,
                    center=(-1.9, -2.2, 0), color=PARAM, gap=0.35, fmt="{:.0f}", size=20)
        for lab_ in bars.labels:
            lab_.set_color(PARAM)
        step_t = [right_at(code(f"step {k:,}", 24, INK), 6.5, -2.0) for k in range(0, 1001)]
        step_slot = step_t[0].copy()
        method = right_at(cite("off-policy TD · α = 0.01"), 6.5, -2.55)
        live = m.VGroup(bars.bars, bars.numbers, step_slot)

        def grow(_, alpha):
            k = int(round(alpha * 1000))
            for i in range(8):
                bars.bars[i].become(bars._rect(i, HB[k][i]))
                bars.numbers[i].become(bars._number(i, HB[k][i]))
            step_slot.become(step_t[k])

        w8_ring = m.SurroundingRectangle(m.VGroup(bars.bars[7], bars._number(7, HB[1000][7])), buff=0.08, corner_radius=0.08)
        w8_ring.set_stroke(RED, 3)

        # the log plot
        plot = Plot((0, 10_500), (0, 12.6), width=8.6, height=4.0, center=(-1.25, -0.6, 0), x_ticks=[0, 5000, 10000],
                    x_fmt="{:,}", y_ticks=[], x_label="steps", y_label="size of the weights, |w| (log scale)")
        lt = log_ticks(plot, [0, 3, 6, 9, 12])
        ks = np.unique(np.concatenate([np.arange(0, 1000, 5), np.arange(1000, B_STEPS + 1, 25)]))
        fast = plot.line(ks, [lg(NORM[k]) for k in ks], RED, 5).set_z_index(2)
        slow = plot.line(ks, [lg(NORM_SLOW[k]) for k in ks], RED, 3).set_stroke(opacity=0.55).set_z_index(2)
        end_f = plot.dot(B_STEPS, lg(NORM[B_STEPS]), RED)
        end_s = plot.dot(B_STEPS, lg(NORM_SLOW[B_STEPS]), mix(RED, CARD, 0.4))
        lab_f = m.VGroup(code("α = 0.01", 20, RED), sci(NORM[B_STEPS], 1).set_color(RED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        lab_f.next_to(end_f, m.RIGHT, buff=0.2)
        lab_s = m.VGroup(code("α = 0.001", 20, mix(RED, INK, 0.3)), code(f"{NORM_SLOW[B_STEPS]:.0f}, still climbing", 20, MUTED))
        lab_s.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).next_to(end_s, m.RIGHT, buff=0.2)
        quote = m.VGroup(text("“The instability occurs for any positive step size, no matter how small.”", 22, INK),
                         cite("Sutton & Barto, 2018, §11.2")).arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT)
        quote.move_to([0, 2.35, 0]).align_to([6.6, 0, 0], m.RIGHT)

        self.say(
            "[The classic demonstration is Baird's counterexample]: [seven states, eight weights], [and every reward is zero.]",
            m.AnimationGroup(self.new_header("05 · counterexample", "Zero reward, endless growth"), m.FadeIn(sb)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(u, shift=0.1 * m.DOWN) for u in uppers], lag_ratio=0.12),
                             m.FadeIn(lower, shift=0.1 * m.UP)),
            m.FadeIn(zero, shift=0.1 * m.LEFT),
        )
        self.say(
            "[So every true value is zero], [and all-zero weights would be perfect.]",
            m.FadeIn(truth[0], shift=0.1 * m.UP),
            m.FadeIn(truth[1], shift=0.1 * m.UP),
        )
        self.say(
            "[The data takes the dashed move six times in seven]; [we learn about a policy that always goes solid.]",
            m.AnimationGroup(m.Create(frame_up), m.Create(dashed), m.FadeIn(dash_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in solid], lag_ratio=0.1), m.Create(loop),
                             m.FadeIn(solid_lab, shift=0.1 * m.UP), m.FadeIn(target_p, shift=0.1 * m.UP)),
        )
        self.say(
            "[Run semi-gradient TD]: [after a thousand steps], [the eighth weight has climbed to about three hundred and seventeen.]",
            m.AnimationGroup(m.FadeIn(bars.zero), m.FadeIn(bars.labels), m.FadeIn(bars.bars), m.FadeIn(bars.numbers),
                             m.FadeIn(step_slot), m.FadeIn(method)),
            m.UpdateFromAlphaFunc(live, grow, run_time=3.2, rate_func=m.linear),
            m.Create(w8_ring),
        )
        self.say(
            "[After ten thousand steps, the weights have grown to about five hundred billion.]",
            m.Succession(
                m.AnimationGroup(m.FadeOut(m.VGroup(uppers, lower, solid, loop, frame_up, dashed, dash_lab, solid_lab, zero,
                                                    truth, target_p, bars, step_slot, method, w8_ring)), run_time=0.6),
                m.AnimationGroup(m.FadeIn(plot), m.FadeIn(lt), run_time=0.6),
                m.Create(fast, run_time=2.4),
                m.AnimationGroup(m.FadeIn(end_f, scale=0.5), m.FadeIn(lab_f, shift=0.1 * m.LEFT), run_time=0.6)),
        )
        self.say(
            "[A ten times smaller step only slows it down]: [it diverges for any positive step size.]",
            m.AnimationGroup(m.Create(slow, run_time=2.0), m.FadeIn(end_s, scale=0.5), m.FadeIn(lab_s, shift=0.1 * m.LEFT)),
            m.FadeIn(quote, shift=0.1 * m.DOWN),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · it happens for real

    def for_real(self):
        self.next_section("6 · It happens for real", notes=(
            "research/experiments/deep_rl.md + deep_rl.json (measured on this Mac, 5 seeds × 500,000 env steps, network "
            "4-256-256-2, Adam 2.5e-4, Huber, γ = 0.99, replay 100k): DQN with a target network copied every 1,000 steps: "
            f"5/5 solved, mean Q over a batch at the end {Q_DQ:.1f}; without the target network (bootstrapping from the "
            f"network being trained): 0/5 solved, final mean Q {Q_NT:.3g} (max seed {NT['q_batch_mean_final_max']:.3g}), "
            f"episodes of ~{NT['final_avg100_mean']:.0f} steps at the end. Curves: q_batch_mean, mean over seeds with the "
            "min–max band, every 5,000 steps. Ceiling: rewards of 1 per step discounted by γ = 0.99 sum to at most "
            "1/(1−γ) = 100 (weights/dqn_cartpole.json notes). DQN keeps all three triad ingredients (notes §12, van "
            "Hasselt et al. 2018); its replay buffer and target network tame them — Episode 13."))
        xs_nt, xs_dq = np.array(C_NT["x_steps"]), np.array(C_DQ["x_steps"])
        plot = Plot((0, 540_000), (0, 10.6), width=8.4, height=4.0, center=(-1.35, -0.55, 0), x_ticks=[0, 250_000, 500_000],
                    x_fmt="{:,}", y_ticks=[], x_label="environment steps", y_label="average Q-value (log scale)")
        lt = log_ticks(plot, [0, 2, 4, 6, 8, 10])
        ceil = m.DashedLine(plot.coords(0, lg(Q_CEIL)), plot.coords(480_000, lg(Q_CEIL)), dash_length=0.1)
        ceil.set_stroke(GOLD, 2.5).set_z_index(1)
        ceil_lab = text("the most any state is worth: 100", 20, REWARD).next_to(plot.coords(0, lg(Q_CEIL)), m.UR, buff=0.1)
        ceil_lab.shift(0.15 * m.RIGHT)
        band = plot.band(xs_nt, [lg(v) for v in C_NT["min"]], [lg(v) for v in C_NT["max"]], RED, 0.25)
        nt = plot.line(xs_nt, [lg(v) for v in C_NT["mean"]], RED, 5).set_z_index(2)
        dq = plot.line(xs_dq, [lg(v) for v in C_DQ["mean"]], VALUE, 5).set_z_index(2)
        e_nt = plot.dot(500_000, lg(Q_NT), RED)            # the final measurement (headline, at 500,000 steps)
        e_dq = plot.dot(500_000, lg(Q_DQ), VALUE)
        l_nt = m.VGroup(text("no target network", 20, RED), sci(Q_NT, 1).set_color(RED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        l_nt.next_to(e_nt, m.RIGHT, buff=0.2)
        l_dq = m.VGroup(text("with a target network", 20, VALUE), code(f"{Q_DQ:.0f}", 24, VALUE, weight=m.BOLD),
                        cite("a little too high: episode 14"))
        l_dq.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(e_dq, m.RIGHT, buff=0.2)
        src = m.VGroup(cite("deep Q-network on CartPole · trained on this Mac"), cite("5 seeds · line = mean, band = min to max"))
        src.arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT)
        solved = m.VGroup(pill("0 of 5 seeds learned to balance", RED, RED_FILL, 20),
                          cite(f"final episodes: about {NT['final_avg100_mean']:.0f} steps"))
        solved.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        solved.move_to(plot.coords(300_000, 5.6))
        src.move_to([0, 2.3, 0]).align_to([6.6, 0, 0], m.RIGHT)

        self.say(
            "[And it happens for real]: [we trained a deep Q-network on CartPole without its target network], "
            "[five times, on this Mac.]",
            m.AnimationGroup(self.new_header("06 · for real", "It happens to real networks"), m.FadeIn(plot), m.FadeIn(lt)),
            m.FadeIn(src, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.FadeIn(band), m.Create(nt, run_time=2.4)),
        )
        self.say(
            "[No CartPole state is worth more than a hundred]; [its Q-values climbed past four billion], "
            "[and not one of the five learned to balance.]",
            m.AnimationGroup(m.Create(ceil), m.FadeIn(ceil_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(e_nt, scale=0.5), m.FadeIn(l_nt, shift=0.1 * m.LEFT)),
            m.FadeIn(solved, shift=0.1 * m.UP),
        )
        self.say(
            "[With a target network, the same setup stayed near a hundred]: [episode thirteen shows how.]",
            m.AnimationGroup(m.Create(dq, run_time=2.0), m.FadeIn(e_dq, scale=0.5), m.FadeIn(l_dq, shift=0.1 * m.LEFT)),
            m.Circumscribe(m.VGroup(l_dq), color=VALUE, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · recap

    def recap(self):
        self.next_section("7 · Recap", notes=(
            "Approximate (V(s) ≈ f(s; θ): a function where no table fits — CartPole's four real numbers), generalise "
            "(one update moves every state that shares features; the target too), beware (function approximation + "
            "bootstrapping + off-policy data = the deadly triad, S&B §11.3: Baird's weights to 5.1e11, our DQN without a "
            "target network to 4.1e9)."))
        halo = glow(PARAM, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Approximate", 52, PARAM), title_text("Generalise", 52, SIGNAL), title_text("Beware", 52, RED))
        words.arrange(m.RIGHT, buff=0.75).move_to([0, 1.0, 0])
        on_baseline(words)
        subs = m.VGroup(text("a function, not a table", 24, MUTED), text("one update moves many states", 24, MUTED),
                        text("the deadly triad", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        for s_, w in zip(subs, words):
            s_.move_to([w.get_x(), words[1].get_bottom()[1] - 0.55, 0])
        on_baseline(subs)
        proof = code(f"CartPole: 4 numbers · {N_PARAMS:,} parameters · Baird: |w| → 5.1e11 with zero reward", 22, INK,
                     t2c={r"67,586": PARAM, r"5\.1e11": RED})
        proof.move_to([0, -1.3, 0])
        mascot = rover(0.7).move_to([0, -2.65, 0])

        self.say(
            "[To sum up]: [approximate], [generalise], [beware.]",
            m.AnimationGroup(self.new_header("07 · recap", "Approximate, generalise, beware"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[When states are too many to list, approximate the values with a function]; "
            "[each update generalises to similar states]; [and beware the triad.]",
            m.AnimationGroup(m.FadeIn(proof, shift=0.1 * m.UP), m.Circumscribe(words[0], color=PARAM, buff=0.12)),
            m.Circumscribe(words[1], color=SIGNAL, buff=0.12),
            m.AnimationGroup(m.Circumscribe(words[2], color=RED, buff=0.12), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()
