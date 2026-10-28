"""Trial and Reward · Episode 15 · Search and Self-Play (Module 4 · Deep Value Learning)

Planning with a learned evaluation beats either alone. Episodes 12–14 learned Q-values from experience; in a game Rover
knows the rules, so it can look ahead. Tic-tac-toe is small (5,478 legal positions, 765 up to the board's 8 symmetries,
perfect play draws: enumerated and solved here); Go is not (≈ 250 moves a turn for ≈ 150 turns: 250^150 ≈ 10^360).
The running example is one tic-tac-toe position, MIDGAME = "X...O...X" (X holds opposite corners, Rover's O the
centre, O to move: every edge draws, a corner loses to a fork). Rover searches it live with rl_reference's mcts (UCT,
c = √2, random rollouts, most-visited move): the first seven simulations, the eighth in its four steps (select the right
edge, expand X's reply, a random roll-out X wins, back up), then the tree after 20, 50, 100, 300 and 1,000 simulations
(328 positions; visits 233 / 47 / 245 / 251 / 36 / 188, most visited: the right edge). Why the eighth went right: UCT
scores (average + √(2 ln N / n)) after seven simulations, computed with uct_score. Measured on this Mac: 10 vs 1,000
simulations pick a drawing edge 60% vs 100% of 50 seeds (recomputed here); MCTS(1,000) draws 197 of 200 games against
minimax and wins 191 of 200 against random. Then the learned guess (value and policy heads, PUCT = puct_score; AlphaGo
Zero's raw network 3,055 Elo vs 5,185 with search), self-play (two copies of the same Rover; visit counts → policy
target π, result → value target z; policy iteration as in Episode 7), the four milestones (AlphaGo 2016, AlphaGo Zero
2017, AlphaZero 2018, MuZero 2020), the recap (search, count, learn) and Module 4's close: value-based methods pick
actions through Q; Module 5 learns the policy directly.

Every number on screen is computed below from rl_reference.py (TicTacToe, MIDGAME, minimax_value, best_moves, mcts,
MCTSNode, uct_score, puct_score) and asserted, loaded with measured("classic", ...), or cited.

Sources (research/notes.md §15, Appendix A rows 15, Appendix C row 15; papers as published; facts dated October 2026):
- research/experiments/classic.md §9 + classic.json mcts (UCT c = √2, uniformly random rollouts, final move = most
  visited child; MIDGAME seed-0 visits for 10 / 100 / 1,000 simulations; "picks a perfect move" over seeds 0–49:
  60% / 84% / 100%; matches: 100 games as X and 100 as O, MCTS(1,000) vs minimax 0/100/0 as X and 0/97/3 as O,
  vs random 100/0/0 and 91/9/0; headline keys mcts_midgame_perfect_move_pct_10_sims / _1000_sims,
  mcts1000_vs_minimax_draws_of_200, mcts1000_vs_random_wins_of_200).
- Tic-tac-toe counts: 5,478 legal positions incl. the empty board, 765 up to symmetry, value 0 (notes §15 table,
  recomputed here by enumeration). Go b ≈ 250, d ≈ 150 (Silver et al. 2016, Nature 529:484).
- UCT: Kocsis & Szepesvári (2006); the course uses UCB1's √(2 ln N / n) (notes Appendix A row 15), as in Episode 3.
- AlphaGo (Silver et al. 2016): SL policy network on 30 million KGS positions, then RL self-play; Fan Hui 5–0
  (5–9 October 2015); Lee Sedol 4–1 in Seoul, March 2016 (DeepMind AlphaGo page); 494 of 495 games won against other
  Go programs (the strongest MCTS programs of the day: Crazy Stone, Zen, Pachi, Fuego).
- AlphaGo Zero (Silver et al. 2017, Nature 550:354): tabula rasa; 3-day run, 4.9 million self-play games, 100–0 against
  AlphaGo Lee; PUCT U = c P √ΣN / (1 + N); the loss (z − v)² − πᵀ log p; "MCTS may therefore be viewed as a powerful
  policy improvement operator"; Elo: AlphaGo Zero 5,185, raw network (no search) 3,055.
- AlphaZero (Silver et al. 2018, Science 362:1140; the Science numbers, per notes Appendix A row 15): chess, shogi and
  Go with one algorithm; training ≈ 9 h chess, 12 h shogi, 13 days Go; 1,000 games vs Stockfish 8: 155 wins, 6 losses
  (839 draws, derived); 60,000 positions/s vs Stockfish's 60 million.
- MuZero (Schrittwieser et al., arXiv 2019; Nature 588:604, 2020): no knowledge of the rules; a learned model
  (representation, dynamics, prediction) used for planning; matched AlphaZero at Go, chess and shogi; Atari: beat R2D2
  in 42 of 57 games (median 2,041% human-normalised vs 1,921%).

Illustrative (said "for example" or labelled on screen): the Go board drawings and their stones; the value (+0.3) and
the prior bars of the network sketch in section 6; the Q-value and policy bars in section 10. The self-play game in
section 7 is real but chosen: mcts(…, 50, seed=i) for both sides from the empty board (X wins in 7 moves); its first
search's visit counts are shown. The tree draws positions up to three moves deep (deeper ones are counted, not drawn);
dot sizes and edge widths grow with log(visits).

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

PREFIX = "ep15"
T = TicTacToe
C_UCT = math.sqrt(2)
MC = measured("classic", "mcts")
HL = measured("classic", "headline")

# ---------------------------------------------------------------------------- the numbers (all computed here)

# --- tic-tac-toe is small: every legal position, and the same up to the board's 8 symmetries
_IDX = np.arange(9).reshape(3, 3)
SYMS = [np.rot90(_IDX, k).flatten() for k in range(4)] + [np.rot90(_IDX.T, k).flatten() for k in range(4)]
assert len({tuple(s) for s in SYMS}) == 8


def all_positions():
    seen, frontier = {T.EMPTY}, [T.EMPTY]
    while frontier:
        board = frontier.pop()
        if T.winner(board) is None:
            for mv in T.legal_moves(board):
                nxt = T.play(board, mv)
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    return seen


POSITIONS = all_positions()
SYMMETRIC = {min("".join(b[i] for i in s) for s in SYMS) for b in POSITIONS}
assert len(POSITIONS) == 5478 and len(SYMMETRIC) == 765 and minimax_value(T.EMPTY) == 0
GO_DIGITS = 150 * math.log10(250)
assert round(GO_DIGITS) == 360                                                          # 250^150 ≈ 10^360

# --- the mid-game position: O (Rover) to move; edges draw, corners lose
assert MIDGAME == "X...O...X" and T.to_move(MIDGAME) == "O"
MOVES = T.legal_moves(MIDGAME)
assert MOVES == [1, 2, 3, 5, 6, 7]
TRUTH = {mv: -minimax_value(T.play(MIDGAME, mv)) for mv in MOVES}                      # from O's side
assert TRUTH == {1: 0, 2: -1, 3: 0, 5: 0, 6: -1, 7: 0} and sorted(best_moves(MIDGAME)) == [1, 3, 5, 7]
assert {int(k): v for k, v in MC["positions"]["midgame"]["perfect_play_value_per_move"].items()} == TRUTH
EDGES, CORNERS = [1, 3, 5, 7], [2, 6]
FORK = [T.play(MIDGAME, 2)]
FORK.append(T.play(FORK[0], 6))
assert FORK == ["X.O.O...X", "X.O.O.X.X"] and minimax_value(FORK[1]) == 1              # X blocks, then forks
FORK_LINES = [(0, 3, 6), (6, 7, 8)]                                                     # two threats at once
assert all(sum(FORK[1][i] == "X" for i in ln) == 2 and "." in [FORK[1][i] for i in ln] for ln in FORK_LINES)


# --- the search, traced: the same loop as rl_reference.mcts, keeping each simulation's path and roll-out
def traced_mcts(board, n_sims, c=C_UCT, seed=0):
    rng = np.random.default_rng(seed)
    root = MCTSNode(board, None, None, rng)
    trace = []
    for _ in range(n_sims):
        node, path = root, [root]
        while not node.untried and node.children:                          # 1. selection
            node = max(node.children, key=lambda ch: uct_score(ch, node.N, c))
            path.append(node)
        expanded = None
        if node.untried:                                                   # 2. expansion
            move = node.untried.pop()
            child = MCTSNode(T.play(node.board, move), move, T.to_move(node.board), rng)
            node.children.append(child)
            path.append(child)
            node, expanded = child, move
        boards = [node.board]                                              # 3. random roll-out
        while T.winner(boards[-1]) is None:
            moves = T.legal_moves(boards[-1])
            boards.append(T.play(boards[-1], moves[int(rng.random() * len(moves))]))
        result = T.winner(boards[-1])
        for n in path:                                                     # 4. backup
            n.N += 1
            if n.player is not None and result != "draw":
                n.W += 1.0 if result == n.player else -1.0
        trace.append(dict(path=[p.move for p in path[1:]], expanded=expanded, rollout=boards, result=result))
    return root, trace


def tree_state(root, depth=3):
    """path of moves → (visits, total result for the player who made the last move), up to `depth` moves deep."""
    out, frontier = {}, [((), root)]
    while frontier:
        path, node = frontier.pop()
        for ch in node.children:
            p = path + (ch.move,)
            out[p] = (ch.N, ch.W)
            if len(p) < depth:
                frontier.append((p, ch))
    return out


def tree_size(root):
    return 1 + sum(tree_size(ch) for ch in root.children)


_TR_ROOT, TRACE = traced_mcts(MIDGAME, 1000)
_ref = mcts(MIDGAME, 1000, seed=0)
assert {ch.move: ch.N for ch in _TR_ROOT.children} == _ref.visits and tree_size(_TR_ROOT) == _ref.tree_size == 328
assert {ch.move: ch.W / ch.N for ch in _TR_ROOT.children} == _ref.values and _ref.best_move == 5
SNAP_AT = (1, 2, 3, 4, 5, 6, 7, 8, 20, 50, 100, 300, 1000)
SNAP = {k: mcts(MIDGAME, k, seed=0) for k in SNAP_AT}
TREES = {k: tree_state(r.root, depth=6) for k, r in SNAP.items()}
assert tree_state(_TR_ROOT, depth=6) == TREES[1000]
for k in (10, 100, 1000):
    seed0 = MC["positions"]["midgame"]["by_sims"][str(k)]["seed0"]
    r = mcts(MIDGAME, k, seed=0)
    assert {int(a): v for a, v in seed0["visits"].items()} == r.visits and seed0["tree_size"] == r.tree_size
VIS1000 = SNAP[1000].visits
assert VIS1000 == {1: 233, 2: 47, 3: 245, 5: 251, 6: 36, 7: 188}
assert max(VIS1000, key=VIS1000.get) == 5 and sorted(VIS1000, key=VIS1000.get)[:2] == [6, 2]  # corners fewest

# the first seven simulations try each free square (the seventh goes one deeper); the eighth, in four steps
assert [t["path"] for t in TRACE[:7]] == [[2], [1], [6], [7], [3], [5], [3, 2]]
FIRST_RESULTS = {t["path"][0]: t["result"] for t in TRACE[:6]}
assert FIRST_RESULTS == {2: "X", 1: "X", 6: "X", 7: "X", 3: "O", 5: "O"}
SIM8 = TRACE[7]
assert SIM8["path"] == [5, 7] and SIM8["expanded"] == 7 and SIM8["result"] == "X"
ROLL = SIM8["rollout"]
assert ROLL[0] == "X...OO.XX" and len(ROLL) - 1 == 4 and T.winner(ROLL[-1]) == "X"
ROLL_MOVES = [next(i for i in range(9) if a[i] != b[i]) for a, b in zip(ROLL, ROLL[1:])]
assert ROLL_MOVES == [2, 3, 1, 6]
WIN_LINE8 = next(ln for ln in LINES if all(ROLL[-1][i] == "X" for i in ln))
assert WIN_LINE8 == (6, 7, 8)
assert TREES[8][(5,)] == (2, 0.0) and TREES[7][(5,)] == (1, 1.0)                         # +1 → 0 after the loss

# --- why the eighth went right: UCT after seven simulations
R7 = SNAP[7].root
UCT7 = {ch.move: (ch.W / ch.N, C_UCT * math.sqrt(math.log(R7.N) / ch.N), uct_score(ch, R7.N, C_UCT)) for ch in R7.children}
assert all(abs(q + b - s) < 1e-12 for q, b, s in UCT7.values())
assert max(UCT7, key=lambda mv: UCT7[mv][2]) == 5 == SIM8["path"][0]
assert UCT7[3][0] == UCT7[5][0] == 1.0 and SNAP[7].visits[3] == 2 and SNAP[7].visits[5] == 1
assert (round(UCT7[3][2], 2), round(UCT7[5][2], 2), round(UCT7[1][2], 2)) == (2.39, 2.97, 0.97)

# --- measured: 10 vs 1,000 simulations (seeds 0–49), and whole games (classic.json)
PERFECT = {k: sum(mcts(MIDGAME, k, seed=s).best_move in EDGES for s in range(50)) for k in (10, 1000)}
assert PERFECT == {10: 30, 1000: 50}
assert 2 * PERFECT[10] == HL["mcts_midgame_perfect_move_pct_10_sims"] == 60.0
assert 2 * PERFECT[1000] == HL["mcts_midgame_perfect_move_pct_1000_sims"] == 100.0
VIS10 = SNAP_10 = mcts(MIDGAME, 10, seed=0).visits
assert VIS10 == {1: 1, 2: 1, 3: 4, 5: 2, 6: 1, 7: 1}
_mm = MC["matches"]["results"]["mcts1000_vs_minimax"]
_rd = MC["matches"]["results"]["mcts1000_vs_random"]
DRAWS_MM = _mm["as_X"]["draw"] + _mm["as_O"]["draw"]
LOSS_MM = _mm["as_X"]["loss"] + _mm["as_O"]["loss"]
WINS_RD = _rd["as_X"]["win"] + _rd["as_O"]["win"]
LOSS_RD = _rd["as_X"]["loss"] + _rd["as_O"]["loss"]
GAMES = 2 * MC["matches"]["games_per_side"]
assert (DRAWS_MM, LOSS_MM, _mm["as_O"]["loss"], GAMES) == (197, 3, 3, 200)
assert DRAWS_MM == HL["mcts1000_vs_minimax_draws_of_200"] and WINS_RD == HL["mcts1000_vs_random_wins_of_200"] == 191
assert LOSS_RD == 0

# --- the learned guess: PUCT, and what search adds (cited)
assert abs(puct_score(0.5, 0.25, 16, 3, 2.0) - 1.0) < 1e-12                             # Q + c·P·√N/(1+n)
ELO_NET, ELO_SEARCH = 3055, 5185                     # AlphaGo Zero, raw network vs with MCTS (Silver et al., 2017)
assert ELO_SEARCH - ELO_NET > 2000                                                       # "over two thousand Elo"
GO_PROGRAMS = (494, 495)                             # AlphaGo vs other Go programs (Silver et al., 2016)

# --- self-play: one real game, MCTS(50) for both sides (seed = move number)
SP_SIMS = 50
SP_BOARDS, SP_MOVES = [T.EMPTY], []
while T.winner(SP_BOARDS[-1]) is None:
    SP_MOVES.append(mcts(SP_BOARDS[-1], SP_SIMS, seed=len(SP_MOVES)).best_move)
    SP_BOARDS.append(T.play(SP_BOARDS[-1], SP_MOVES[-1]))
assert SP_MOVES == [2, 6, 0, 1, 8, 4, 5] and T.winner(SP_BOARDS[-1]) == "X"
SP_WIN = next(ln for ln in LINES if all(SP_BOARDS[-1][i] == "X" for i in ln))
assert SP_WIN == (2, 5, 8)
SP_VISITS = mcts(T.EMPTY, SP_SIMS, seed=0).visits
assert max(SP_VISITS, key=SP_VISITS.get) == SP_MOVES[0] and sum(SP_VISITS.values()) == SP_SIMS
SP_Z = [1 if T.to_move(b) == "X" else -1 for b in SP_BOARDS[:-1]]                      # X won: +1 where X moved
assert SP_Z == [1, -1, 1, -1, 1, -1, 1]

# --- the milestones (research/notes.md §15)
AZ_W, AZ_L, AZ_GAMES = 155, 6, 1000
AZ_D = AZ_GAMES - AZ_W - AZ_L
assert AZ_D == 839 and 60_000_000 // 60_000 == 1000                                      # "a thousand times fewer"

# ---------------------------------------------------------------------------- look

ROOT_C = np.array([-1.33, 2.12, 0])
CX = {1: -4.63, 2: -3.31, 3: -1.99, 5: -0.67, 6: 0.65, 7: 1.97}
Y1, Y2, Y3 = 0.75, -0.25, -0.92
YD = {4: -1.12, 5: -1.25, 6: -1.38}  # positions four to six moves deep: rows of small dots, counted, not linked
NODE = 0.86
ROW_Y = {"visits": -1.8, "value": -2.25, "perfect": -2.7, "bonus": -2.7, "score": -3.15}
ROW_X = -5.3                        # the table's row labels end here
PANEL_X = 4.75                      # the right panel's centre
SIDE = {1: "edge", 2: "corner", 3: "edge", 5: "edge", 6: "corner", 7: "edge"}


def cite(s, size=20):
    return text(s, size, MUTED)


def minus(s):
    return s.replace("-", "−")


def left_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.LEFT)


def right_at(mob, x, y):
    return mob.move_to([x, y, 0]).align_to([x, 0, 0], m.RIGHT)


def signed(v):
    return "0.00" if abs(v) < 0.005 else minus(f"{v:+.2f}")


class Board(m.VGroup):
    """A tic-tac-toe board: a TEAL grid (the state), X marks in INK, Rover's O marks in CORAL. `.at(i)` is the centre
    of square i (0-8, row by row from the top left); `.mark(i, who)` a new mark there (not added); `.tint(i, colour)`
    a filled square to put behind a mark. Build boards in place: positions are kept from construction."""

    def __init__(self, board=T.EMPTY, cell=0.5, center=(0, 0, 0), grid_color=STATE, width=None):
        super().__init__()
        self.cell, self.c = cell, np.array(center, dtype=float)
        self.w = width or (1.6 + 3.2 * cell)
        h = 1.5 * cell
        lines = []
        for k in (-0.5, 0.5):
            lines.append(m.Line(self.c + [k * cell, -h, 0], self.c + [k * cell, h, 0]))
            lines.append(m.Line(self.c + [-h, k * cell, 0], self.c + [h, k * cell, 0]))
        self.grid = m.VGroup(*lines).set_stroke(grid_color, self.w)
        self.marks = {}
        self.add(self.grid)
        for i, ch in enumerate(board):
            if ch != ".":
                self.marks[i] = self.mark(i, ch)
                self.add(self.marks[i])

    def at(self, i):
        r, col = divmod(i, 3)
        return self.c + np.array([(col - 1) * self.cell, (1 - r) * self.cell, 0])

    def mark(self, i, who, color=None):
        p, s, w = self.at(i), 0.27 * self.cell, self.w * 1.3
        if who == "X":
            g = m.VGroup(m.Line(p + [-s, -s, 0], p + [s, s, 0]), m.Line(p + [-s, s, 0], p + [s, -s, 0]))
            return g.set_stroke(color or INK, w)
        return m.Circle(radius=0.29 * self.cell).move_to(p).set_stroke(color or ACTION, w).set_fill(opacity=0)

    def tint(self, i, color, size=0.84):
        return m.Square(self.cell * size).move_to(self.at(i)).set_stroke(width=0).set_fill(color, 1)

    def line_through(self, line, color=GOLD, width=6):
        a, b = self.at(line[0]), self.at(line[2])
        d = (b - a) / np.linalg.norm(b - a) * 0.32 * self.cell
        return m.Line(a - d, b + d).set_stroke(color, width)


def node_card(board, move=None, size=NODE, center=(0, 0, 0), stroke=EDGE):
    """A tree node: a small card holding the position; the move that led here sits on a CORAL tint."""
    box = card(size, size, stroke, CARD, radius=0.12, stroke=2)
    box.move_to(center)
    b = Board(board, cell=0.24, center=center, width=1.9)
    parts = [box]
    if move is not None:
        parts.append(b.tint(move, CORAL_FILL))
    parts.append(b)
    g = m.VGroup(*parts)
    g.box, g.board = box, b
    return g


def node_xy(path):
    """Where a node of the tree sits: children by square, deeper nodes by their move among the legal ones."""
    x, y = CX[path[0]], Y1
    b = T.play(MIDGAME, path[0])
    if len(path) >= 2:
        x += (T.legal_moves(b).index(path[1]) - 2) * 0.25
        y = Y2
        b = T.play(b, path[1])
    if len(path) >= 3:
        x += (T.legal_moves(b).index(path[2]) - 1.5) * 0.062
        y = Y3
    return np.array([x, y, 0])


def dot_radius(path, n):
    lg = math.log2(1 + n)
    return 0.03 + 0.011 * lg if len(path) == 2 else min(0.03, 0.013 + 0.0045 * lg)


def edge_width(path, n):
    lg = math.log2(1 + n)
    return (1.5 + 0.45 * lg, 1.2 + 0.35 * lg, 0.8 + 0.2 * lg)[len(path) - 1]


class SearchTree(m.VGroup):
    """The search tree of MIDGAME as it grows: the root card, six child cards (one per free square), positions two
    and three moves deep as dots (X's replies MUTED, Rover's moves CORAL). `.to(state)` animates to a tree state
    (from tree_state): new nodes fade in, others resize with their visits, missing ones fade out."""

    def __init__(self):
        super().__init__()
        self.root = node_card(MIDGAME, None, center=ROOT_C)
        self.cards = {mv: node_card(T.play(MIDGAME, mv), mv, center=[CX[mv], Y1, 0]) for mv in MOVES}
        self.edges, self.dots = {}, {}

    def end_of(self, path):
        if len(path) == 1:
            return np.array([CX[path[0]], Y1 + NODE / 2, 0])
        return node_xy(path)

    def start_of(self, path):
        if len(path) == 1:
            return ROOT_C - [0, NODE / 2, 0]
        if len(path) == 2:
            return np.array([CX[path[0]], Y1 - NODE / 2, 0])
        return node_xy(path[:-1])

    def make_edge(self, path, n):
        a, b = self.start_of(path), self.end_of(path)
        if len(path) >= 2:
            d = (b - a) / np.linalg.norm(b - a)
            b = b - d * dot_radius(path, n)
            if len(path) == 3:
                a = a + d * dot_radius(path[:-1], 1)
        return m.Line(a, b).set_stroke(FAINT, edge_width(path, n)).set_z_index(-2)

    def make_dot(self, path, n, xy=None):
        color = MUTED if len(path) % 2 == 0 else ACTION             # X's replies MUTED, Rover's moves CORAL
        radius = dot_radius(path, n) if len(path) <= 3 else 0.017
        return m.Dot(node_xy(path) if xy is None else xy, radius=radius, color=color).set_z_index(1)

    def to(self, state, run_time=0.8, cards=True):
        anims = []
        deep = {}
        for path in state:
            if len(path) >= 4:
                deep.setdefault((path[0], len(path)), []).append(path)
        deep_xy = {}
        for (mv, d), paths in deep.items():
            paths.sort()
            for i, p in enumerate(paths):
                deep_xy[p] = np.array([CX[mv] + (i - (len(paths) - 1) / 2) * 0.045, YD[d], 0])
        for path, (n, _) in sorted(state.items(), key=lambda kv: (len(kv[0]), kv[0])):
            if len(path) >= 4:
                dot = self.make_dot(path, n, deep_xy[path])
                if path in self.dots:
                    anims.append(m.Transform(self.dots[path], dot))
                else:
                    self.dots[path] = dot
                    anims.append(m.FadeIn(dot, scale=0.5))
                continue
            edge = self.make_edge(path, n)
            if path in self.edges:
                anims.append(m.Transform(self.edges[path], edge))
            else:
                self.edges[path] = edge
                anims.append(m.Create(edge) if len(path) > 1 else m.GrowFromPoint(edge, edge.get_start()))
            if len(path) == 1:
                if cards and path[0] not in getattr(self, "_shown", set()):
                    self._shown = getattr(self, "_shown", set()) | {path[0]}
                    anims.append(m.FadeIn(self.cards[path[0]], shift=0.15 * m.DOWN))
                continue
            dot = self.make_dot(path, n)
            if path in self.dots:
                anims.append(m.Transform(self.dots[path], dot))
            else:
                self.dots[path] = dot
                anims.append(m.FadeIn(dot, scale=0.5))
        for path in [p for p in list(self.edges) if p not in state]:
            anims.append(m.FadeOut(self.edges.pop(path)))
        for path in [p for p in list(self.dots) if p not in state]:
            anims.append(m.FadeOut(self.dots.pop(path)))
        return m.AnimationGroup(*anims, run_time=run_time)

    def everything(self):
        return m.VGroup(self.root, *self.cards.values(), *self.edges.values(), *self.dots.values())


def step_chip(i, label, active=False, width=3.3, height=0.54):
    color, fill, ink = (VIOLET, VIOLET_FILL, INK) if active else (EDGE, CARD, MUTED)
    box = card(width, height, color, fill, radius=0.12, stroke=2)
    num = code(str(i), 22, VIOLET if active else FAINT, weight=m.BOLD)
    words = text(label, 24, ink, weight=m.BOLD if active else m.NORMAL)
    num.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    words.next_to(num, m.RIGHT, buff=0.3)
    words.shift((num[0].get_bottom()[1] - words[0].get_bottom()[1]) * m.UP)
    return m.VGroup(box, num, words)


def go_board(center, size=2.0, n=9, stones=None, grid=STATE, width=1.6):
    """A drawing of a Go board (n lines, a few stones): illustration only."""
    c = np.array(center, dtype=float)
    step = size / (n - 1)
    lines = m.VGroup()
    for k in range(n):
        off = -size / 2 + k * step
        lines.add(m.Line(c + [off, -size / 2, 0], c + [off, size / 2, 0]), m.Line(c + [-size / 2, off, 0], c + [size / 2, off, 0]))
    lines.set_stroke(grid, width)
    rim = m.Square(size + step).move_to(c).set_stroke(EDGE, 2).set_fill(TEAL_FILL, 0.35).set_z_index(-1)
    group = m.VGroup(rim, lines)
    for (i, j), white in (stones or {}).items():
        p = c + [-size / 2 + i * step, -size / 2 + j * step, 0]
        s = m.Circle(radius=0.42 * step).move_to(p)
        s.set_fill(INK if white else "#0C0912", 1).set_stroke(INK if white else MUTED, 1.5)
        group.add(s)
    return group


GO_STONES = {(2, 6): 0, (6, 6): 1, (6, 2): 0, (2, 2): 1, (3, 6): 1, (5, 5): 0, (4, 4): 0, (3, 3): 1, (6, 4): 1,
             (2, 4): 0, (5, 7): 1, (4, 2): 0}


def hbar(x0, y, width, height, color):
    return m.Rectangle(width=max(width, 0.02), height=height).set_stroke(color, 2).set_fill(FILLS.get(color, CARD), 1).move_to(
        [x0 + width / 2, y, 0])


def visit_chart(visits, cx, top, title, base_y=-0.05, height=1.75, width=5.6):
    n = len(MOVES)
    gap = 0.4
    bw = (width - gap * (n - 1)) / n
    xs = [cx - width / 2 + bw / 2 + i * (bw + gap) for i in range(n)]
    axis = m.Line([cx - width / 2 - 0.12, base_y, 0], [cx + width / 2 + 0.12, base_y, 0]).set_stroke(FAINT, 2)
    bars, nums, kinds = m.VGroup(), m.VGroup(), m.VGroup()
    for x, mv in zip(xs, MOVES):
        h = max(height * visits[mv] / top, 0.02)
        bars.add(m.Rectangle(width=bw, height=h).set_stroke(ACTION, 2).set_fill(CORAL_FILL, 1).move_to([x, base_y + h / 2, 0]))
        nums.add(code(str(visits[mv]), 20, ACTION).move_to([x, base_y + h + 0.2, 0]))
        kinds.add(text(SIDE[mv], 24, GOOD if TRUTH[mv] == 0 else BAD).move_to([x, base_y - 0.3, 0]))
    head = m.VGroup(text(title, 24, INK, weight=m.BOLD), cite("· seed 0")).arrange(m.RIGHT, buff=0.15).move_to([cx, 2.3, 0])
    head[1].align_to(head[0], m.DOWN)
    return m.VGroup(head, axis, bars, nums, kinds)


def milestone(x, year, name, drawing, facts, source, y_top=1.7, width=3.05, height=4.9):
    box = card(width, height, EDGE, CARD, radius=0.18, stroke=2).move_to([x, y_top - height / 2, 0])
    yr = title_text(year, 34, EXPLORE).move_to([x, 2.28, 0])
    nm = title_text(name, 28, INK).move_to([x, y_top - 0.42, 0])
    drawing.move_to([x, y_top - 1.25, 0])
    lines = m.VGroup()
    for i, (s, hi) in enumerate(facts):
        t = text(s, 20, INK, t2c=hi or {})
        lines.add(left_at(t, x - width / 2 + 0.22, y_top - 2.1 - 0.4 * i))
    src = cite(source).move_to([x, y_top - height + 0.32, 0])
    return m.VGroup(box, nm, drawing, lines, src), yr


# ---------------------------------------------------------------------------- the episode


class Ep15SearchAndSelfPlay(Episode):
    number = 15

    def lesson(self):
        self.look_ahead()
        self.position()
        self.tree_search()
        self.uct()
        self.measured_here()
        self.learned_guess()
        self.self_play()
        self.milestones()
        self.recap()
        self.module_close()

    def fade(self, *mobs, run_time=0.6):
        mobs = [mb for mb in mobs if mb is not None]
        if mobs:
            self.play(*[m.FadeOut(mb) for mb in mobs], run_time=run_time)

    # ------------------------------------------------------------------ 1 · look ahead

    def look_ahead(self):
        self.next_section("1 · Think before you move", notes=(
            "Episodes 12–14 learned Q-values from experience (networks, DQN, Double DQN). In a game Rover knows the "
            "rules, so it can look ahead. Tic-tac-toe: 5,478 legal positions reachable in play (empty board "
            f"included), {len(SYMMETRIC)} up to the board's 8 symmetries; minimax value 0: perfect play draws (all "
            "enumerated and solved here; notes §15 table). Go: b ≈ 250 legal moves, d ≈ 150 moves per game (Silver "
            f"et al. 2016), so b^d = 250^150 ≈ 10^{GO_DIGITS:.1f}. The Go board and the imagined boards are drawings."))
        chips_ = m.VGroup(*[chip(lab, VIOLET, VIOLET_FILL, width=3.2, height=0.66, size=22)
                            for lab in ("12 · networks", "13 · DQN", "14 · double DQN")])
        chips_.arrange(m.RIGHT, buff=0.35).move_to([0, 1.55, 0])
        learned = text("values learned from experience, one move at a time", 24, VALUE).move_to([0, 0.55, 0])

        bd = Board("X...O...X", cell=0.55, center=[-5.0, -1.0, 0])
        bot = rover(0.6).move_to([-3.0, -1.75, 0])
        dots_ = m.VGroup(*[m.Circle(radius=r).set_stroke(MUTED, 2).move_to(p) for r, p in
                           ((0.07, [-2.95, -0.95, 0]), (0.1, [-3.1, -0.6, 0]))])
        cloud = card(3.2, 1.2, EDGE, CARD, radius=0.4, stroke=2).move_to([-3.55, 0.65, 0])
        futures = m.VGroup(*[Board(b, cell=0.2, center=[-4.6 + 1.05 * k, 0.65, 0], width=1.6)
                             for k, b in enumerate(("XO..O...X", "X...OO..X", "X..OO...X"))])
        bubble_ = m.VGroup(dots_, cloud, futures)

        t_head = left_at(text("tic-tac-toe", 26, STATE, weight=m.BOLD), 0.0, 2.05)
        def centred_stat(value, label, color, x):
            num = title_text(value, 50, color).move_to([x, 1.2, 0])
            return m.VGroup(num, text(label, 22, MUTED).move_to([x, 0.6, 0]))

        s1 = centred_stat(f"{len(POSITIONS):,}", "legal positions", INK, 0.75)
        s2 = centred_stat(f"{len(SYMMETRIC)}", "up to symmetry", INK, 3.1)
        s3 = centred_stat("draw", "with perfect play", GOOD, 5.45)
        on_baseline(m.VGroup(s1[0], s2[0], s3[0]))

        go = go_board([1.2, -1.85, 0], size=1.9, stones=GO_STONES)
        g_head = left_at(text("Go", 26, STATE, weight=m.BOLD), 2.65, -0.8)
        g1 = m.VGroup(left_at(text("≈ 250 moves a turn", 22, INK), 2.65, -1.25),
                      left_at(text("for ≈ 150 turns", 22, INK), 2.65, -1.62))
        g2 = left_at(tex(r"250^{150} \approx 10^{360}", 40, INK), 2.65, -2.3)
        g3 = m.VGroup(left_at(cite("move sequences, roughly"), 2.65, -2.9), left_at(cite("Silver et al., 2016"), 2.65, -3.28))

        self.say(
            "[For three episodes, Rover learned values from experience], [one move at a time.]",
            m.AnimationGroup(self.new_header("01 · look ahead", "Think before you move"),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips_], lag_ratio=0.3)),
            m.FadeIn(learned, shift=0.1 * m.UP),
        )
        self.say(
            "[But in a game, Rover knows the rules]: [so it can try moves in its head before it makes one.]",
            m.AnimationGroup(m.FadeOut(chips_), m.FadeOut(learned), m.FadeIn(bd), m.FadeIn(bot, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(dots_), m.FadeIn(cloud, scale=0.9),
                             m.LaggedStart(*[m.FadeIn(f, scale=0.8) for f in futures], lag_ratio=0.35)),
        )
        self.say(
            "[Tic-tac-toe is small]: [five thousand four hundred and seventy-eight legal positions], "
            "[and with perfect play, every game is a draw.]",
            m.FadeIn(t_head, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(s1, shift=0.1 * m.UP), m.FadeIn(s2, shift=0.1 * m.UP)),
            m.FadeIn(s3, shift=0.1 * m.UP),
        )
        self.say(
            "[Go has about two hundred and fifty moves a turn, for about a hundred and fifty turns]: "
            "[far too many futures to search them all.]",
            m.AnimationGroup(m.FadeIn(go), m.FadeIn(g_head), m.FadeIn(g1, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(g2, shift=0.1 * m.UP), m.FadeIn(g3)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the position

    def position(self):
        self.next_section("2 · Rover's move", notes=(
            "MIDGAME = 'X...O...X' (rl_reference): X took two opposite corners (squares 0 and 8), Rover's O the centre; O "
            "to move. Minimax from O's side: squares 1, 3, 5, 7 (edges) draw, 2 and 6 (corners) lose (best_moves, "
            "asserted; = classic.json mcts.positions.midgame). Why a corner loses: O takes 2, threatening 2-4-6; X "
            "must block at 6 and then threatens 0-3-6 and 6-7-8 at once (minimax value +1 for X, asserted)."))
        bd = Board(T.EMPTY, cell=1.0, center=[-3.75, -0.1, 0])
        xs = m.VGroup(bd.mark(0, "X"), bd.mark(8, "X"))
        o4 = bd.mark(4, "O")
        leg_x = m.VGroup(Board(T.EMPTY, cell=0.3).mark(4, "X"), text("the opponent", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        leg_o = m.VGroup(rover(0.36), text("Rover plays O", 22, ACTION)).arrange(m.RIGHT, buff=0.2)
        left_at(leg_x, -5.6, -2.3)
        left_at(leg_o, -3.2, -2.3)
        to_move = pill("O to move", ACTION, CORAL_FILL, 22).move_to([-3.75, 1.95, 0])

        verdict = {mv: text("draw" if TRUTH[mv] == 0 else "loses", 22, GOOD if TRUTH[mv] == 0 else BAD, weight=m.BOLD)
                   .move_to(bd.at(mv)) for mv in MOVES}
        tints = {mv: bd.tint(mv, GREEN_FILL if TRUTH[mv] == 0 else RED_FILL, 0.9).set_z_index(-1) for mv in MOVES}

        f_head = text("why a corner loses", 24, INK, weight=m.BOLD).move_to([3.05, 1.75, 0])
        fx = [0.6, 3.05, 5.5]
        fb = [Board(b, cell=0.44, center=[x, 0.15, 0]) for x, b in zip(fx, ("X...O...X", FORK[0], FORK[1]))]
        f_tint = fb[1].tint(2, RED_FILL).set_z_index(-1)
        threats = m.VGroup(*[fb[2].line_through(ln, RED, 5) for ln in FORK_LINES])
        f_arrows = m.VGroup(*[arrow([fx[k] + 0.78, 0.15, 0], [fx[k + 1] - 0.78, 0.15, 0], MUTED) for k in range(2)])
        f_caps = m.VGroup(*[cite(s).move_to([x, -0.9, 0]) for x, s in
                            zip(fx, ("the position", "O takes a corner", "X blocks: a fork"))])
        f_note = cite("two threats at once: O can only stop one").move_to([3.05, -1.55, 0])

        self.say(
            "[Here is the position Rover will search]: [X holds two opposite corners], "
            "[and Rover's O holds the centre. Rover to move.]",
            m.AnimationGroup(self.new_header("02 · the position", "Rover's move"), m.FadeIn(bd.grid)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(x) for x in xs], lag_ratio=0.4), m.FadeIn(leg_x)),
            m.AnimationGroup(m.Create(o4), m.FadeIn(leg_o), m.FadeIn(to_move, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[Full search says every edge draws], [but a corner loses]: [X blocks, then threatens two lines at once.]",
            m.AnimationGroup(*[m.FadeIn(tints[mv]) for mv in EDGES], *[m.FadeIn(verdict[mv]) for mv in EDGES]),
            m.AnimationGroup(*[m.FadeIn(tints[mv]) for mv in CORNERS], *[m.FadeIn(verdict[mv]) for mv in CORNERS]),
            m.Succession(
                m.AnimationGroup(m.FadeIn(f_head), m.FadeIn(fb[0]), m.FadeIn(f_caps[0])),
                m.AnimationGroup(m.GrowArrow(f_arrows[0]), m.FadeIn(f_tint), m.FadeIn(fb[1]), m.FadeIn(f_caps[1])),
                m.AnimationGroup(m.GrowArrow(f_arrows[1]), m.FadeIn(fb[2]), m.FadeIn(f_caps[2])),
                m.AnimationGroup(m.LaggedStart(*[m.Create(t) for t in threats], lag_ratio=0.4), m.FadeIn(f_note)),
            ),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · tree search

    def table_entries(self, state, rows=("visits", "value")):
        """Numbers under each child: visits (CORAL) and Rover's average result (BLUE)."""
        out = {}
        for mv in MOVES:
            if (mv,) not in state:
                continue
            n, w = state[(mv,)]
            if "visits" in rows:
                out[("visits", mv)] = code(f"{n:,}", 22, ACTION, weight=m.BOLD).move_to([CX[mv], ROW_Y["visits"], 0])
            if "value" in rows:
                out[("value", mv)] = code(signed(w / n), 22, VALUE).move_to([CX[mv], ROW_Y["value"], 0])
        return out

    def table_to(self, state, run_time=0.8):
        anims = []
        for key, mob in self.table_entries(state).items():
            if key in self.cells:
                anims.append(m.Transform(self.cells[key], mob))
            else:
                self.cells[key] = mob
                anims.append(m.FadeIn(mob))
        return m.AnimationGroup(*anims, run_time=run_time)

    def counter(self, s):
        return code(s, 22, MUTED).move_to([PANEL_X, -3.15, 0])

    def tree_search(self):
        self.next_section("3 · Grow a tree of futures", notes=(
            "rl_reference.mcts(MIDGAME, n, seed=0): UCT with c = √2, uniformly random roll-outs, results +1 win / 0 draw "
            "/ −1 loss for the player who made each node's move, final move = most visited child. Because the search "
            "is seeded and sequential, mcts(MIDGAME, k, seed=0) is exactly the first k simulations of the 1,000-run, "
            "so the tree is drawn from those snapshots (k = 1…8, 20, 50, 100, 300, 1,000); a traced copy of the loop "
            "(asserted identical to mcts) supplies the eighth simulation's path and roll-out. Simulations 1–6 expand "
            "squares 2, 1, 6, 7, 3, 5 in turn (random roll-outs: X, X, X, X, O, O win); the seventh selects the left "
            "edge and adds X's reply. The eighth: select the right edge (square 5), expand X's reply at 7, roll out "
            "O 2, X 3, O 1, X 6 (X wins with 6-7-8), back up: the right edge goes from 1 visit / +1.00 to 2 / 0.00. "
            "After 1,000 simulations: visits 1: 233, 2: 47, 3: 245, 5: 251, 6: 36, 7: 188 (= classic.json seed0), "
            f"averages {', '.join(f'{mv}: {SNAP[1000].values[mv]:+.3f}' for mv in MOVES)}; the tree holds 328 "
            "positions (drawn up to three moves deep). Most visited: square 5, the right edge (a draw with perfect "
            "play); the corners got the fewest visits."))
        tree = self.tree = SearchTree()
        self.cells = {}
        now = left_at(cite("the position now"), ROOT_C[0] + NODE / 2 + 0.2, ROOT_C[1])
        steps = [step_chip(i + 1, s) for i, s in enumerate(("select", "expand", "roll out", "back up"))]
        for k, st in enumerate(steps):
            st.move_to([PANEL_X, 2.15 - 0.62 * k, 0])
        actives = [step_chip(i + 1, s, True).move_to(steps[i]) for i, s in enumerate(("select", "expand", "roll out", "back up"))]
        self.steps = steps

        def activate(i):
            anims = []
            for k in range(4):
                target = (actives[k] if k == i else step_chip(k + 1, ("select", "expand", "roll out", "back up")[k]).move_to(steps[k]))
                anims.append(m.Transform(steps[k], target))
            return m.AnimationGroup(*anims, run_time=0.5)

        labels = {r: right_at(text(s, 20, MUTED), ROW_X, ROW_Y[r]) for r, s in
                  (("visits", "visits"), ("value", "value"), ("perfect", "full search"))}
        val_note = left_at(cite("value: Rover's average result · +1 win, 0 draw, −1 loss"), -6.2, -3.15)
        sim_count = self.counter("simulation 7")

        # the eighth simulation
        p_root = ROOT_C - [0, NODE / 2, 0]
        hl1 = m.Line(p_root, [CX[5], Y1 + NODE / 2, 0]).set_stroke(ACTION, 7).set_z_index(2)
        new_xy = node_xy((5, 7))
        hl2 = m.Line([CX[5], Y1 - NODE / 2, 0], new_xy + [0, 0.06, 0]).set_stroke(ACTION, 6).set_z_index(2)
        new_dot = m.Dot(new_xy, radius=0.07, color=ACTION).set_z_index(3)
        r_cap = cite("a random finish").move_to([PANEL_X, -0.42, 0])
        rb = Board(ROLL[0], cell=0.4, center=[PANEL_X, -1.22, 0])
        r_tint = m.VGroup(rb.tint(5, CORAL_FILL), rb.tint(7, "#2A2238")).set_z_index(-1)
        r_marks = [rb.mark(mv, T.to_move(b)) for mv, b in zip(ROLL_MOVES, ROLL)]
        r_line = rb.line_through(WIN_LINE8, RED, 5)
        result = pill("X wins: −1 for Rover", RED, RED_FILL, 20).move_to([PANEL_X, -2.2, 0])
        path_pts = [new_xy, [CX[5], Y1 - NODE / 2, 0], [CX[5], Y1 + NODE / 2, 0], p_root, ROOT_C]
        back = m.VMobject().set_points_as_corners([np.array(p, dtype=float) for p in path_pts])
        flash = m.ShowPassingFlash(back.set_stroke(GOLD, 9), time_width=0.6, run_time=1.2)

        # the first seven simulations
        firsts = []
        for k in range(1, 8):
            firsts.append(m.AnimationGroup(tree.to(TREES[k], run_time=0.55)))

        self.say(
            "[Rover's tool is Monte Carlo tree search]: [it grows a tree of possible futures, one simulation at a time.]",
            m.AnimationGroup(self.new_header("03 · tree search", "Grow a tree of futures"),
                             m.FadeIn(tree.root, shift=0.15 * m.DOWN), m.FadeIn(now)),
            m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.LEFT) for s in steps], lag_ratio=0.25),
        )
        self.say(
            "[The first simulations try each free square], [each finished with random moves to see who wins.]",
            m.Succession(*firsts),
            m.AnimationGroup(m.FadeIn(labels["visits"]), m.FadeIn(labels["value"]), self.table_to(TREES[7]),
                             m.FadeIn(val_note), m.FadeIn(sim_count)),
        )
        self.say(
            "[Every simulation has four steps; watch the eighth. First, select]: [walk down the tree, picking a promising move.]",
            m.AnimationGroup(activate(0), m.Transform(sim_count, self.counter("simulation 8"))),
            m.AnimationGroup(m.Create(hl1), tree.cards[5].box.animate.set_stroke(ACTION, 3)),
        )
        self.say(
            "[Second, expand]: [add one new position: here, X's reply.]",
            activate(1),
            m.AnimationGroup(m.Create(hl2), m.FadeIn(new_dot, scale=0.4)),
        )
        self.say(
            "[Third, roll out]: [finish the game with random moves.] [X wins this one.]",
            m.AnimationGroup(activate(2), m.FadeIn(r_cap), m.FadeIn(r_tint), m.FadeIn(rb)),
            m.Succession(*[m.FadeIn(mk, scale=0.6, run_time=0.45) for mk in r_marks]),
            m.AnimationGroup(m.Create(r_line), m.FadeIn(result, shift=0.1 * m.UP)),
        )
        self.say(
            "[Fourth, back up]: [every position on the path counts one more visit, and adds the result.]",
            activate(3),
            m.AnimationGroup(flash, m.FadeOut(new_dot), tree.to(TREES[8], run_time=1.0), self.table_to(TREES[8], run_time=1.0)),
        )
        grow_a = m.Succession(*[m.AnimationGroup(tree.to(TREES[k], run_time=0.8), self.table_to(TREES[k], run_time=0.8),
                                                 m.Transform(sim_count, self.counter(f"simulations: {k:,}"), run_time=0.8))
                                for k in (20, 50, 100)])
        rest = m.AnimationGroup(m.FadeOut(hl1), m.FadeOut(hl2), m.FadeOut(r_cap), m.FadeOut(rb), m.FadeOut(r_tint),
                                *[m.FadeOut(mk) for mk in r_marks], m.FadeOut(r_line), m.FadeOut(result),
                                activate(-1), tree.cards[5].box.animate.set_stroke(EDGE, 2), run_time=0.6)
        grow_b = m.Succession(*[m.AnimationGroup(tree.to(TREES[k], run_time=0.9), self.table_to(TREES[k], run_time=0.9),
                                                 m.Transform(sim_count, self.counter(f"simulations: {k:,}"), run_time=0.9))
                                for k in (300, 1000)])
        size_note = cite(f"{SNAP[1000].tree_size} positions · 6 moves deep").move_to([PANEL_X, -2.7, 0])
        self.say(
            "[Then repeat, a thousand times]: [the tree grows deepest where the moves look best.]",
            m.AnimationGroup(rest, grow_a),
            m.AnimationGroup(grow_b, m.FadeIn(size_note)),
        )
        truth = {mv: text("draw" if TRUTH[mv] == 0 else "lose", 20, GOOD if TRUTH[mv] == 0 else BAD, weight=m.BOLD)
                 .move_to([CX[mv], ROW_Y["perfect"], 0]) for mv in MOVES}
        best_box = m.SurroundingRectangle(m.VGroup(tree.cards[5], self.cells[("visits", 5)]), color=ACTION, buff=0.12,
                                          corner_radius=0.12, stroke_width=3)
        self.say(
            "[Finally, Rover plays the move with the most visits]: [the right-hand edge.] [The corners got the fewest.]",
            m.Create(best_box),
            m.AnimationGroup(m.FadeIn(labels["perfect"]), *[m.FadeIn(truth[mv]) for mv in EDGES]),
            m.AnimationGroup(*[m.FadeIn(truth[mv]) for mv in CORNERS],
                             *[m.Circumscribe(self.cells[("visits", mv)], color=BAD, buff=0.08) for mv in CORNERS]),
        )
        self.wait(0.3)
        self.fade(best_box, labels["perfect"], *truth.values(), size_note, val_note, *steps, now)
        self.row_labels = labels
        self.sim_count = sim_count

    # ------------------------------------------------------------------ 4 · UCT

    def uct(self):
        self.next_section("4 · A bandit at every node", notes=(
            "Selection treats each position as a multi-armed bandit (Episode 3): UCT (Kocsis & Szepesvári, 2006) "
            "picks the child with the highest average result + √(2 ln N / n), N = the position's visits, n = the "
            "move's (UCB1's constant c = √2, as rl_reference.uct_score and Episode 3; notes Appendix A row 15). After "
            "seven simulations (N = 7): " + ", ".join(f"square {mv}: {q:+.2f} + {b:.2f} = {s:.2f}" for mv, (q, b, s)
                                                       in UCT7.items()) +
            ". Squares 3 and 5 both averaged +1, but 5 had one visit to 3's two, so its bonus (1.97 vs 1.39) wins: "
            "the eighth simulation goes to square 5 (asserted against the traced search)."))
        tree = self.tree
        q_label = right_at(text("bonus", 20, MUTED), ROW_X, ROW_Y["bonus"])
        s_label = right_at(text("UCT score", 20, MUTED), ROW_X, ROW_Y["score"])
        bonus = {mv: code(f"{b:.2f}", 22, EXPLORE).move_to([CX[mv], ROW_Y["bonus"], 0]) for mv, (q, b, s) in UCT7.items()}
        score = {mv: code(f"{s:.2f}", 22, INK, weight=m.BOLD).move_to([CX[mv], ROW_Y["score"], 0]) for mv, (q, b, s) in UCT7.items()}

        machines = m.VGroup(*[SlotMachine(lab, size=0.48) for lab in ("1", "2", "3")]).arrange(m.RIGHT, buff=0.55)
        machines.move_to([PANEL_X - 0.15, 1.35, 0])
        m_cap = cite("each move is an arm · episode 3").move_to([PANEL_X, 0.25, 0])

        f = m.MathTex(r"\mathrm{score}", "=", r"\bar{Q}", "+", r"\sqrt{\frac{2\ln N}{n}}", font_size=36, color=INK)
        f[2].set_color(VALUE)
        f[4].set_color(EXPLORE)
        f.move_to([PANEL_X - 0.05, 1.75, 0])
        f_l1 = left_at(text("average result so far", 20, VALUE), 2.95, 0.75)
        f_l2 = left_at(text("big when tried rarely", 20, EXPLORE), 2.95, 0.35)
        f_l3 = left_at(cite("N: visits here, n: to the move"), 2.95, -0.05)
        f_src = left_at(cite("Kocsis & Szepesvári, 2006"), 2.95, -0.45)
        count7 = self.counter("after 7 simulations")
        col5 = m.SurroundingRectangle(m.VGroup(self.cells[("value", 5)], score[5]), color=ACTION, buff=0.1,
                                      corner_radius=0.1, stroke_width=3)
        win_note = left_at(text("same average, fewer visits:", 20, INK), 2.95, -1.25)
        win_note2 = left_at(text("a bigger bonus wins the turn", 20, ACTION, weight=m.BOLD), 2.95, -1.62)

        self.say(
            "[Why did the eighth simulation go right? Each position is a row of slot machines], [like Rover's in episode three.]",
            m.AnimationGroup(self.new_header("04 · UCT", "A bandit at every node"), tree.to(TREES[7], run_time=1.0, cards=False),
                             self.table_to(TREES[7], run_time=1.0), m.Transform(self.sim_count, count7)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(mc, shift=0.1 * m.UP) for mc in machines], lag_ratio=0.25),
                             m.FadeIn(m_cap)),
        )
        self.say(
            "[The rule, called UCT, scores each move by its average result], [plus a bonus that is big when the move has been tried rarely.]",
            m.AnimationGroup(m.FadeOut(machines), m.FadeOut(m_cap), m.FadeIn(f[0:3], shift=0.1 * m.UP), m.FadeIn(f_l1)),
            m.AnimationGroup(m.FadeIn(f[3:], shift=0.1 * m.UP), m.FadeIn(f_l2), m.FadeIn(f_l3), m.FadeIn(f_src),
                             m.FadeIn(q_label), m.FadeIn(s_label),
                             m.LaggedStart(*[m.FadeIn(bonus[mv]) for mv in MOVES], lag_ratio=0.1),
                             m.LaggedStart(*[m.FadeIn(score[mv]) for mv in MOVES], lag_ratio=0.1)),
        )
        self.say(
            "[Both side edges had won every game so far], [but the right edge had fewer visits, so its bonus, and its score, came out on top.]",
            m.AnimationGroup(*[m.Circumscribe(self.cells[("value", mv)], color=VALUE, buff=0.08) for mv in (3, 5)]),
            m.AnimationGroup(m.Create(col5), m.FadeIn(win_note), m.FadeIn(win_note2)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · measured here

    def measured_here(self):
        self.next_section("5 · Ten simulations or a thousand", notes=(
            "Measured on this Mac (research/experiments/classic.md §9, classic.json mcts), recomputed here: on MIDGAME "
            f"the most-visited move is a drawing edge for {PERFECT[10]} of 50 seeds (0–49) with 10 simulations and "
            f"{PERFECT[1000]} of 50 with 1,000 (headline: 60% and 100%; 100 simulations: 84%). Bars: seed 0's root "
            f"visits ({VIS10} and {VIS1000}). Whole games, 100 as X and 100 as O, 1,000 simulations per move: vs "
            f"minimax (perfect play) {DRAWS_MM} draws and {LOSS_MM} losses (all as O); vs uniformly random moves "
            f"{WINS_RD} wins, {GAMES - WINS_RD} draws, {LOSS_RD} losses. No values are learned: each search starts from "
            "scratch with the rules, random roll-outs and visit counts."))
        left = visit_chart(VIS10, -3.35, 4.0, "10 simulations")
        right = visit_chart(VIS1000, 3.35, 251, "1,000 simulations")
        st10 = title_text(f"{2 * PERFECT[10]}%", 44, INK).move_to([-3.35, -1.02, 0])
        st1k = title_text(f"{2 * PERFECT[1000]}%", 44, GOOD).move_to([3.35, -1.02, 0])
        lab10 = cite("pick a drawing edge · 50 seeds · measured here").next_to(st10, m.DOWN, buff=0.14)
        lab1k = cite("pick a drawing edge · 50 seeds · measured here").next_to(st1k, m.DOWN, buff=0.14)

        def match_card(x, big, color, line1, line2):
            box = card(5.9, 0.98, EDGE, CARD, radius=0.14, stroke=2).move_to([x, -2.45, 0])
            num = title_text(big, 30, color)
            words = m.VGroup(text(line1, 20, INK), cite(line2)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
            row = m.VGroup(num, words).arrange(m.RIGHT, buff=0.3).move_to(box)
            return m.VGroup(box, row)

        mm = match_card(-3.2, f"{DRAWS_MM} of {GAMES}", INK, "drawn against perfect play", f"{LOSS_MM} lost, all as O")
        rd = match_card(3.2, f"{WINS_RD} of {GAMES}", GOOD, "won against random moves", "none lost")
        nothing = pill("no learning: the rules, random games, counting", INK, CARD, 20).move_to([0, -3.3, 0])

        self.say(
            "[With only ten simulations, the counts are mostly noise]: "
            "[over fifty seeds, Rover picks a drawing edge only sixty percent of the time.]",
            m.AnimationGroup(self.new_header("05 · measured", "Ten simulations or a thousand"), m.FadeIn(left)),
            m.AnimationGroup(m.FadeIn(st10, shift=0.1 * m.UP), m.FadeIn(lab10)),
        )
        self.say(
            "[With a thousand simulations, it picks one every time.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(st1k, shift=0.1 * m.UP), m.FadeIn(lab1k)),
        )
        self.say(
            "[Playing whole games with a thousand per move], [it drew one hundred and ninety-seven of two hundred against perfect play], "
            "[and won one hundred and ninety-one of two hundred against random moves.]",
            m.Wait(0.1),
            m.FadeIn(mm, shift=0.1 * m.UP),
            m.FadeIn(rd, shift=0.1 * m.UP),
        )
        self.say(
            "[And nothing was learned]: [just the rules, random games and counting.]",
            m.FadeIn(nothing, shift=0.1 * m.UP),
            m.Circumscribe(nothing, color=STATE, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · a learned guess

    def learned_guess(self):
        self.next_section("6 · Search with a network", notes=(
            "On Go, random roll-outs are a weak judge. AlphaGo (2016) mixed roll-outs with a value network; AlphaGo Zero "
            "(2017) and later drop roll-outs: one network with two heads, a value v (who is winning) and a policy p "
            "(a prior P over moves). PUCT (rl_reference.puct_score; Silver et al. 2017): Q + c·P·√N / (1 + n): the "
            "prior steers the search, the bonus fades with visits. What search adds: AlphaGo Zero's raw network "
            f"(no search) {ELO_NET:,} Elo vs {ELO_SEARCH:,} with MCTS (Silver et al. 2017); AlphaGo won "
            f"{GO_PROGRAMS[0]} of {GO_PROGRAMS[1]} games against the other Go programs of the day (Crazy Stone, Zen, Pachi, "
            "Fuego, GnuGo: tree search with hand-made or shallow learned patterns, no deep networks; Silver et al. 2016). "
            "AlphaGo (2016) mixed roll-outs with its value network (λ = 0.5); AlphaGo Zero dropped roll-outs. "
            "Illustrative: the Go board, the value +0.3 and the prior bars."))
        go = go_board([-5.35, 0.45, 0], size=1.5, stones=GO_STONES)
        dice = icon(recolored("dice", EXPLORE, VIOLET_FILL, PREFIX), 0.95).move_to([-3.1, 0.75, 0])
        dice_cap = text("random finishes", 22, EXPLORE).move_to([-3.1, -0.15, 0])
        cross = m.VGroup(m.Line([-3.75, 0.1, 0], [-2.45, 1.4, 0]), m.Line([-3.75, 1.4, 0], [-2.45, 0.1, 0])).set_stroke(RED, 6)
        net = NeuralNet(layers=(5, 7, 7, 4), width=1.9, height=1.7, center=[-3.1, 0.45, 0])
        a_in = arrow([-4.45, 0.45, 0], [-4.15, 0.45, 0], MUTED, buff=0.0)
        v_head = m.VGroup(code("value", 20, VALUE, weight=m.BOLD), code("+0.3", 26, VALUE, weight=m.BOLD)).arrange(m.DOWN, buff=0.08)
        left_at(v_head, -1.3, 1.5)
        v_cap = left_at(cite("who's winning?"), -1.3, 0.98)
        pri = [0.08, 0.35, 0.12, 0.3, 0.05, 0.1]
        p_bars = m.VGroup(*[m.Rectangle(width=0.2, height=0.05 + 1.6 * p).set_stroke(ACTION, 1.5).set_fill(CORAL_FILL, 1)
                            for p in pri]).arrange(m.RIGHT, buff=0.07, aligned_edge=m.DOWN)
        p_bars.move_to([-0.95, -0.45, 0]).align_to([0, -0.85, 0], m.DOWN)
        p_cap = code("prior P", 20, ACTION, weight=m.BOLD).next_to(p_bars, m.DOWN, buff=0.12)
        heads = m.VGroup(arrow([-2.0, 0.85, 0], [-1.45, 1.45, 0], MUTED, buff=0.0), arrow([-2.0, 0.1, 0], [-1.5, -0.3, 0], MUTED, buff=0.0))
        eg = cite("for example").move_to([-3.1, -1.55, 0])

        f = m.MathTex("Q", "+", "c", r"\cdot", "P", r"\cdot", r"\frac{\sqrt{N}}{1+n}", font_size=46, color=INK)
        f[0].set_color(VALUE)
        f[4].set_color(ACTION)
        f[6].set_color(EXPLORE)
        f.move_to([3.6, 1.75, 0])
        f_name = left_at(code("PUCT", 22, MUTED, weight=m.BOLD), 1.0, 1.75)
        f_l = m.VGroup(text("Q: the average result", 20, VALUE), text("P: the network's prior", 20, ACTION),
                       text("the bonus fades with visits", 20, EXPLORE))
        for i, t in enumerate(f_l):
            left_at(t, 1.85, 0.98 - 0.36 * i)
        f_src = left_at(cite("AlphaGo Zero · Silver et al., 2017"), 1.85, -0.12)

        x0, full = 2.75, 2.9
        e_rows = []
        for i, (lab, v, col) in enumerate((("network alone", ELO_NET, MUTED), ("network + search", ELO_SEARCH, GOOD))):
            y = -0.95 - 0.6 * i
            bar = hbar(x0, y, full * v / 5500, 0.38, col)
            e_rows.append(m.VGroup(right_at(text(lab, 22, INK), x0 - 0.15, y), bar,
                                   code(f"{v:,}", 22, col, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.12)))
        e_src = left_at(cite("Elo ratings · AlphaGo Zero · Silver et al., 2017"), 0.55, -2.2)
        e_old = m.VGroup(cite("the other Go programs of 2015 (tree search, no deep networks)"),
                         cite(f"won {GO_PROGRAMS[1] - GO_PROGRAMS[0]} of {GO_PROGRAMS[1]} games against AlphaGo · Silver et al., 2016"))
        e_old.arrange(m.DOWN, buff=0.12).move_to([0, -3.0, 0])

        self.say(
            "[On a Go board, random finishes are a weak judge of who is winning], [so the AlphaGo programs brought in a network.]",
            m.AnimationGroup(self.new_header("06 · a learned guess", "Search with a network"), m.FadeIn(go), m.FadeIn(dice),
                             m.FadeIn(dice_cap), m.Create(cross, run_time=1.2)),
            m.AnimationGroup(m.FadeOut(dice), m.FadeOut(dice_cap), m.FadeOut(cross), m.FadeIn(net), m.GrowArrow(a_in)),
        )
        self.say(
            "[Its value head guesses who's winning from the position alone], [and its policy head gives each move a prior.]",
            m.AnimationGroup(net.forward(VIOLET), m.GrowArrow(heads[0]), m.FadeIn(v_head, shift=0.1 * m.RIGHT),
                             m.FadeIn(v_cap), m.FadeIn(eg)),
            m.AnimationGroup(m.GrowArrow(heads[1]), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in p_bars], lag_ratio=0.12),
                             m.FadeIn(p_cap)),
        )
        self.say(
            "[AlphaGo's version of the rule adds that prior to the bonus]: [moves the network likes get searched first], "
            "[and the bonus still fades with visits.]",
            m.AnimationGroup(m.FadeIn(f_name), m.FadeIn(f[0:2], shift=0.1 * m.UP), m.FadeIn(f_l[0]), m.FadeIn(f_src)),
            m.AnimationGroup(m.FadeIn(f[2:6], shift=0.1 * m.UP), m.FadeIn(f_l[1])),
            m.AnimationGroup(m.FadeIn(f[6], shift=0.1 * m.UP), m.FadeIn(f_l[2])),
        )
        self.say(
            "[Planning with a learned guess beats either one alone]: [AlphaGo Zero's network by itself rated about three thousand Elo]; "
            "[with search, over five thousand.]",
            m.FadeIn(e_old, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(e_rows[0][0]), m.GrowFromEdge(e_rows[0][1], m.LEFT), m.FadeIn(e_rows[0][2]), m.FadeIn(e_src)),
            m.AnimationGroup(m.FadeIn(e_rows[1][0]), m.GrowFromEdge(e_rows[1][1], m.LEFT), m.FadeIn(e_rows[1][2])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · self-play

    def self_play(self):
        self.next_section("7 · Two copies of one Rover", notes=(
            "Self-play: the same network plays both sides, so its opponent improves exactly as fast as it does. The game "
            f"shown is real: mcts(board, {SP_SIMS}, seed=move number) for both sides from the empty board, moves "
            f"{SP_MOVES} (X wins with {SP_WIN}). The visit counts are the first search's ({SP_VISITS}). In AlphaGo Zero "
            "(Silver et al. 2017) the search's visit distribution π is the policy head's target (cross-entropy) and the "
            "final result z is the value head's target for every position of the game: loss (z − v)² − πᵀ log p + "
            "c‖θ‖². 'MCTS may therefore be viewed as a powerful policy improvement operator': search improves the "
            "policy, playing to the end evaluates it, which is policy iteration (Episode 7)."))
        bd = Board(T.EMPTY, cell=0.55, center=[-3.25, -0.55, 0])
        rx, ro = rover(0.72).move_to([-5.55, -0.65, 0]), rover(0.72).move_to([-0.95, -0.65, 0])
        lx = text("Rover as X", 22, INK).move_to([-5.55, -1.6, 0])
        lo = text("Rover as O", 22, ACTION).move_to([-0.95, -1.6, 0])
        net = NeuralNet(layers=(4, 6, 6, 3), width=1.5, height=0.85, center=[-3.25, 1.75, 0], radius=0.07)
        net_lab = cite("one network").move_to([-3.25, 1.0, 0])
        links = m.VGroup(m.DashedLine([-4.1, 1.6, 0], rx.get_top() + [0.1, 0.12, 0], dash_length=0.08),
                         m.DashedLine([-2.4, 1.6, 0], ro.get_top() + [-0.1, 0.12, 0], dash_length=0.08)).set_stroke(MUTED, 2)

        vb = Board(T.EMPTY, cell=0.5, center=[2.0, 1.15, 0])
        top_n = max(SP_VISITS.values())
        v_tints = m.VGroup(*[vb.tint(i, mix(CARD, CORAL, 0.15 + 0.7 * SP_VISITS.get(i, 0) / top_n)).set_z_index(-1) for i in range(9)])
        v_nums = m.VGroup(*[code(str(SP_VISITS.get(i, 0)), 20, INK, weight=m.BOLD).move_to(vb.at(i)) for i in range(9)])
        v_cap = cite(f"visit counts of the first search ({SP_SIMS})").move_to([2.0, 2.2, 0])
        v_arrow = arrow([2.9, 1.15, 0], [3.6, 1.15, 0], ACTION)
        pi_t = pill("policy target π", ACTION, CORAL_FILL, 22, mono=True).move_to([5.02, 1.15, 0])

        marks = [bd.mark(mv, T.to_move(b)) for mv, b in zip(SP_MOVES, SP_BOARDS)]
        win = bd.line_through(SP_WIN, GOLD, 6)
        movers = [rx if T.to_move(b) == "X" else ro for b in SP_BOARDS[:-1]]

        def move_anim(k, rt=0.5):
            return m.AnimationGroup(m.FadeIn(marks[k], scale=0.6),
                                    movers[k].animate(rate_func=m.there_and_back).shift(0.12 * m.UP), run_time=rt)

        z_cap = cite(f"X won: one result per position, from the mover's side").move_to([3.3, -0.35, 0])
        z_chips = m.VGroup(*[reward_chip(z, 20) for z in SP_Z]).arrange(m.RIGHT, buff=0.12).move_to([3.3, -0.95, 0])
        z_arrow = arrow([3.3, -1.3, 0], [3.3, -1.85, 0], VALUE)
        z_t = pill("value target z", VALUE, BLUE_FILL, 22, mono=True).move_to([3.3, -2.2, 0])

        ring = loop_ring(["search plays", "train", "better network"], radius=1.85, colors=[ACTION, SIGNAL, VALUE],
                         size=22, center=(3.55, -0.05, 0))
        r_cap = cite("policy iteration, as in episode 7").move_to([3.55, -2.15, 0])
        r_src = cite("search improves the policy, games evaluate it · Silver et al., 2017").move_to([0, -3.15, 0])

        self.say(
            "[Where do its training games come from? Self-play]: [two copies of the same Rover, sharing one network, take turns.]",
            m.AnimationGroup(self.new_header("07 · self-play", "Two copies of one Rover"), m.FadeIn(bd)),
            m.AnimationGroup(m.FadeIn(rx, shift=0.15 * m.RIGHT), m.FadeIn(ro, shift=0.15 * m.LEFT), m.FadeIn(lx), m.FadeIn(lo),
                             m.FadeIn(net), m.FadeIn(net_lab), m.Create(links)),
        )
        self.say(
            "[Every move is chosen by search], [and the search's visit counts become the policy's new target.]",
            m.AnimationGroup(m.FadeIn(v_cap), m.FadeIn(v_tints), m.FadeIn(vb), m.FadeIn(v_nums), net.forward(VIOLET), move_anim(0, 1.0)),
            m.AnimationGroup(m.GrowArrow(v_arrow), m.FadeIn(pi_t, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[When the game ends], [its result becomes the value target for every position in it.]",
            m.AnimationGroup(m.Succession(*[move_anim(k, 0.42) for k in range(1, len(marks))]), m.Create(win, run_time=3.2,
                             rate_func=lambda t: max(0.0, (t - 0.85) / 0.15))),
            m.AnimationGroup(m.FadeIn(z_cap), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in z_chips], lag_ratio=0.12),
                             m.GrowArrow(z_arrow), m.FadeIn(z_t, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[A better network makes a sharper search, and a sharper search makes better games]: [policy iteration, just like episode seven.]",
            m.AnimationGroup(m.FadeOut(v_cap), m.FadeOut(v_tints), m.FadeOut(vb), m.FadeOut(v_nums), m.FadeOut(v_arrow), m.FadeOut(pi_t),
                             m.FadeOut(z_cap), m.FadeOut(z_chips), m.FadeOut(z_arrow), m.FadeOut(z_t),
                             m.FadeIn(ring[0]), m.Create(ring[1], run_time=1.6)),
            m.AnimationGroup(m.FadeIn(r_cap), m.FadeIn(r_src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · four milestones

    def milestones(self):
        self.next_section("8 · From AlphaGo to MuZero", notes=(
            "AlphaGo (Silver et al., Nature 2016): an SL policy network on 30 million positions from KGS games, then RL "
            "by self-play, MCTS with roll-outs + a value network; beat Fan Hui 5–0 (5–9 October 2015) and Lee Sedol 4–1 "
            "(Seoul, March 2016). AlphaGo Zero (Silver et al., Nature 2017): tabula rasa, one network with two heads, "
            "3-day run of 4.9 million self-play games, beat AlphaGo Lee 100–0. AlphaZero (Silver et al., Science 2018; "
            "the Science numbers, notes Appendix A): chess, shogi and Go with one algorithm; ≈ 9 h of training for chess "
            f"(12 h shogi, 13 days Go); vs Stockfish 8, 1,000 games: {AZ_W} wins, {AZ_L} losses, {AZ_D} draws (derived); "
            "60,000 positions/s vs 60 million. MuZero (Schrittwieser et al., arXiv 2019, Nature 2020): never told the "
            "rules, learns a model (representation, dynamics, prediction) and plans inside it; matched AlphaZero at Go, "
            "chess and shogi; Atari: beat R2D2 in 42 of 57 games. The board drawings are illustrations."))
        xs = [-4.95, -1.65, 1.65, 4.95]
        az = m.VGroup()
        chess = m.VGroup(*[m.Square(0.13).set_stroke(width=0).set_fill(STATE if (i + j) % 2 else TEAL_FILL, 1)
                           .move_to([(i - 2.5) * 0.13, (j - 2.5) * 0.13, 0]) for i in range(6) for j in range(6)])
        shogi = Board(T.EMPTY, cell=0.26).grid.copy()
        shogi_rim = m.Square(0.78).set_stroke(STATE, 1.6)
        az.add(chess.move_to([-0.95, 0, 0]), m.VGroup(shogi, shogi_rim).move_to([0, 0, 0]),
               go_board([0.95, 0, 0], size=0.66, n=7, stones={(2, 4): 0, (4, 2): 1, (3, 3): 0, (4, 4): 1}, width=1.2))
        mu = m.VGroup(m.DashedVMobject(m.Square(0.85).set_stroke(EXPLORE, 2.5), num_dashes=24),
                      title_text("?", 40, EXPLORE))
        mu[1].move_to(mu[0])
        cards_ = [
            milestone(xs[0], "2016", "AlphaGo", go_board([0, 0, 0], size=0.82, n=7, stones={(1, 5): 0, (4, 4): 1, (2, 2): 0, (5, 1): 1}, width=1.2),
                      [("30M human positions,", None), ("then self-play", None), ("Fan Hui 5–0 · 2015", {"5–0": GOLD}),
                       ("Lee Sedol 4–1 · 2016", {"4–1": GOLD})], "Silver et al., 2016"),
            milestone(xs[1], "2017", "AlphaGo Zero", go_board([0, 0, 0], size=0.82, n=7, stones={(3, 3): 0, (2, 4): 1, (4, 2): 0, (1, 1): 1}, width=1.2),
                      [("no human games", None), ("3 days of self-play", None), ("4.9M games", None),
                       ("beat AlphaGo Lee", None), ("100 games to 0", {"100": GOLD})], "Silver et al., 2017"),
            milestone(xs[2], "2018", "AlphaZero", az,
                      [("chess, shogi and Go", None), ("chess: 9 h of training", None), ("vs Stockfish:", None),
                       (f"{AZ_W} wins, {AZ_L} losses", {f"{AZ_W} wins": GOLD}), (f"in {AZ_GAMES:,} games", None),
                       ("60k vs 60M positions/s", None)], "Silver et al., 2018"),
            milestone(xs[3], "2020", "MuZero", mu,
                      [("never told the rules", None), ("learns its own model", None), ("and plans inside it", None),
                       ("matched AlphaZero", {"matched": GOLD}), ("Atari: beat R2D2", None), ("in 42 of 57 games", None)],
                      "Schrittwieser et al., 2020"),
        ]
        on_baseline(m.VGroup(*[body[1] for body, _ in cards_]))
        years = [yr for _, yr in cards_]
        links = m.VGroup(*[m.Arrow(years[k].get_right() + [0.15, 0, 0], years[k + 1].get_left() - [0.15, 0, 0], buff=0,
                                   color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.08) for k in range(3)])

        def show(k):
            body, yr = cards_[k]
            parts = [m.FadeIn(yr, shift=0.1 * m.DOWN), m.FadeIn(body[0]), m.FadeIn(body[1]), m.FadeIn(body[2], scale=0.9),
                     m.FadeIn(body[4])]
            if k:
                parts.append(m.GrowArrow(links[k - 1]))
            return m.AnimationGroup(*parts)

        def facts(k):
            return m.LaggedStart(*[m.FadeIn(t, shift=0.08 * m.UP) for t in cards_[k][0][3]], lag_ratio=0.3)

        self.say(
            "[First, AlphaGo]: [it learned from thirty million human positions, then from self-play], "
            "[and beat Fan Hui five games to nothing in twenty fifteen, and Lee Sedol four to one in twenty sixteen.]",
            m.AnimationGroup(self.new_header("08 · four milestones", "From AlphaGo to MuZero"), show(0)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.08 * m.UP) for t in cards_[0][0][3][:2]], lag_ratio=0.3),
            m.LaggedStart(*[m.FadeIn(t, shift=0.08 * m.UP) for t in cards_[0][0][3][2:]], lag_ratio=0.4),
        )
        self.say(
            "[In twenty seventeen, AlphaGo Zero started from random play, with no human games]: "
            "[after three days, it beat the version that beat Lee Sedol, one hundred to nothing.]",
            m.AnimationGroup(show(1), m.LaggedStart(*[m.FadeIn(t, shift=0.08 * m.UP) for t in cards_[1][0][3][:1]])),
            m.LaggedStart(*[m.FadeIn(t, shift=0.08 * m.UP) for t in cards_[1][0][3][1:]], lag_ratio=0.35),
        )
        self.say(
            "[In twenty eighteen, AlphaZero used one algorithm for chess, shogi and Go]: "
            "[after nine hours of chess training, it beat Stockfish], [while searching a thousand times fewer positions per second.]",
            m.AnimationGroup(show(2), m.FadeIn(cards_[2][0][3][0], shift=0.08 * m.UP)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.08 * m.UP) for t in cards_[2][0][3][1:5]], lag_ratio=0.3),
            m.FadeIn(cards_[2][0][3][5], shift=0.08 * m.UP),
        )
        self.say(
            "[And in twenty twenty, MuZero was never told the rules]: "
            "[it learned its own model of the game, planned inside it, and matched AlphaZero.]",
            m.AnimationGroup(show(3), m.FadeIn(cards_[3][0][3][0], shift=0.08 * m.UP)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.08 * m.UP) for t in cards_[3][0][3][1:]], lag_ratio=0.3),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            "Search (simulate futures with the rules: Monte Carlo tree search), count (visits decide; UCT's bonus "
            "explores), learn (a network's value and prior sharpen the search; self-play makes the training games). "
            f"Tic-tac-toe, measured here: a drawing edge {2 * PERFECT[10]}% → {2 * PERFECT[1000]}% of 50 seeds with 10 → "
            f"1,000 simulations. Go: AlphaGo Zero's network {ELO_NET:,} Elo alone, {ELO_SEARCH:,} with search "
            "(Silver et al. 2017)."))
        halo = glow(VIOLET, radius=3.4).move_to([0, 0.8, 0])
        words = m.VGroup(title_text("Search", 70, STATE), title_text("Count", 70, ACTION), title_text("Learn", 70, VALUE))
        words.arrange(m.RIGHT, buff=1.5).move_to([0, 1.05, 0])
        on_baseline(words)
        subs = m.VGroup(text("simulate with the rules", 24, MUTED), text("the most-visited move", 24, MUTED),
                        text("a guess that sharpens it", 24, MUTED))
        for s_, w in zip(subs, words):
            s_.next_to(w, m.DOWN, buff=0.4)
        on_baseline(subs)
        p1 = code(f"tic-tac-toe · a drawing edge {2 * PERFECT[10]}% → {2 * PERFECT[1000]}% · 10 → 1,000 simulations", 22, INK,
                  t2c={rf"{2 * PERFECT[1000]}%": GOOD})
        p2 = code(f"Go · network {ELO_NET:,} Elo → with search {ELO_SEARCH:,}", 22, INK, t2c={rf"{ELO_SEARCH:,}": GOOD})
        proof = m.VGroup(p1, p2).arrange(m.DOWN, buff=0.22).move_to([0, -1.25, 0])
        src = cite("measured on this Mac, 50 seeds · Silver et al., 2017").next_to(proof, m.DOWN, buff=0.2)
        bot = rover(0.6).move_to([0, -3.0, 0])

        self.say(
            "[To sum up]: [search the future using the rules], [count which moves earn the visits], "
            "[and learn a guess that sharpens the search.]",
            m.AnimationGroup(self.new_header("09 · recap", "Search, count, learn"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[On tic-tac-toe, a thousand simulations found the drawing move every time]; "
            "[on Go, search added over two thousand Elo to the network.]",
            m.AnimationGroup(m.FadeIn(p1, shift=0.1 * m.UP), m.FadeIn(bot, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(p2, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 10 · module 4 closes

    def module_close(self):
        self.next_section("10 · Values, then policies", notes=(
            "Module 4 (Episodes 12–15: function approximation, DQN, Double DQN and friends, search and self-play) picked "
            "actions through values: estimate Q(s, a), take the argmax (search, too, ranks moves by values and visits). "
            "Module 5 (Episodes 16–19) learns the policy π(a | s) directly and nudges each move's probability up or down "
            "by reward (Episode 16: policy gradients, REINFORCE). The bars are illustrative (for example)."))
        chips_ = m.VGroup(*[chip(lab, VIOLET, VIOLET_FILL, width=2.85, height=0.62, size=22)
                            for lab in ("12 · networks", "13 · DQN", "14 · double DQN", "15 · search")])
        chips_.arrange(m.RIGHT, buff=0.28).move_to([0, 2.0, 0])
        acts = ("N", "E", "S", "W")
        qv = (0.2, 0.7, 0.1, 0.4)
        pv = (0.15, 0.6, 0.05, 0.2)

        def bars(vals, cx, color, top):
            g = m.VGroup()
            for i, (a, v) in enumerate(zip(acts, vals)):
                x = cx - 1.2 + 0.8 * i
                h = 1.5 * v / top
                g.add(m.VGroup(m.Rectangle(width=0.5, height=h).set_stroke(color, 2).set_fill(FILLS[color], 1).move_to([x, -1.55 + h / 2, 0]),
                               code(a, 20, MUTED).move_to([x, -1.85, 0]),
                               code(f"{v:.2f}", 20, color).move_to([x, -1.55 + h + 0.2, 0])))
            return g

        lt = text("value-based · module 4", 24, VALUE, weight=m.BOLD).move_to([-3.4, 0.95, 0])
        qb = bars(qv, -3.4, VALUE, 0.75)
        q_lab = code("Q(s, a)", 22, VALUE).move_to([-5.6, -0.5, 0])
        pick = m.SurroundingRectangle(qb[1], color=ACTION, buff=0.08, stroke_width=3)
        then = left_at(text("estimate Q, then take the best:", 22, INK), -5.6, -2.55)
        act = chip("E", ACTION, CORAL_FILL, width=0.62, height=0.52, size=22, mono=True).next_to(then, m.RIGHT, buff=0.2)
        rt = text("the policy itself · module 5", 24, ACTION, weight=m.BOLD).move_to([3.4, 0.95, 0])
        pb = bars(pv, 3.4, ACTION, 0.75)
        p_lab = code("π(a | s)", 22, ACTION).move_to([5.6, -0.5, 0])
        nudges = m.VGroup(*[arrow(pb[i][0].get_top() + [0.42, -0.1 if d < 0 else 0.05, 0],
                                  pb[i][0].get_top() + [0.42, 0.35 * d, 0], SIGNAL, buff=0) for i, d in ((1, 1), (3, -1))])
        p_cap = left_at(text("each move's chance, nudged by reward", 22, INK), 1.2, -2.55)
        eg = cite("for example").move_to([0, -3.15, 0])

        self.say(
            "[That closes module four]: [a network to learn the values, steadier targets, honest maximums, and now search.]",
            self.new_header("10 · module 4", "Values, then policies"),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips_], lag_ratio=0.4),
        )
        self.say(
            "[Every method in it picked actions through values]: [estimate a Q-value for each move, then take the best.]",
            m.AnimationGroup(m.FadeIn(lt), m.FadeIn(q_lab), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in qb], lag_ratio=0.15),
                             m.FadeIn(eg)),
            m.AnimationGroup(m.Create(pick), m.FadeIn(then), m.FadeIn(act, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[Module five learns the policy directly]: [the chance of each move, nudged up or down by reward.]",
            m.AnimationGroup(m.FadeIn(rt), m.FadeIn(p_lab), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in pb], lag_ratio=0.15)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in nudges], lag_ratio=0.4), m.FadeIn(p_cap)),
        )
        self.wait(0.6)
        self.clear()
