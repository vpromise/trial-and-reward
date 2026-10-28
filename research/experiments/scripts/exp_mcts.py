"""Experiment 9: MCTS (UCT, random rollouts) on tic-tac-toe - visit counts from the empty board and a mid-game
position, and MCTS(n) against a random player and a perfect (minimax) player.
Run from the series folder: research/experiments/.venv-classic/bin/python research/experiments/scripts/exp_mcts.py"""

import time

import numpy as np

from common import parallel, rl, save

T = rl.TicTacToe
SIMS = (10, 100, 1000)


def search_stats(job):
    board, n, seed = job
    r = rl.mcts(board, n, seed=seed)
    return board, n, seed, r.visits, r.values, r.tree_size, r.best_move


def positions():
    out = {}
    jobs = [(b, n, s) for b in (T.EMPTY, rl.MIDGAME) for n in SIMS for s in range(50)]
    rows = parallel(search_stats, jobs, chunksize=5)
    for board, name in ((T.EMPTY, "empty"), (rl.MIDGAME, "midgame")):
        sign = 1 if T.to_move(board) == "X" else -1
        perfect = {str(m): sign * rl.minimax_value(T.play(board, m)) for m in T.legal_moves(board)}
        entry = dict(board=board, to_move=T.to_move(board), perfect_play_value_per_move=perfect,
                     best_moves=rl.best_moves(board), by_sims={})
        for n in SIMS:
            mine = [r for r in rows if r[0] == board and r[1] == n]
            seed0 = next(r for r in mine if r[2] == 0)
            moves = T.legal_moves(board)
            share = {str(m): np.mean([r[3].get(m, 0) / n for r in mine]) for m in moves}
            picked = [r[6] for r in mine]
            entry["by_sims"][str(n)] = dict(
                seed0=dict(visits={str(m): v for m, v in sorted(seed0[3].items())},
                           mean_value={str(m): v for m, v in sorted(seed0[4].items())},
                           tree_size=seed0[5], best_move=seed0[6]),
                over_50_seeds=dict(visit_share=share, best_move_counts={str(m): picked.count(m) for m in moves},
                                   picks_a_perfect_move=np.mean([m in entry["best_moves"] for m in picked]),
                                   tree_size_mean=np.mean([r[5] for r in mine])))
        out[name] = entry
    return out


def play_game(job):
    """One game: MCTS with n simulations against `opponent`, MCTS playing X or O. Returns MCTS's result."""
    n, opponent, mcts_plays, game = job
    rng = np.random.default_rng(10_000 + game)
    board, move_no = T.EMPTY, 0
    while T.winner(board) is None:
        if T.to_move(board) == mcts_plays:
            move = rl.mcts(board, n, seed=1_000_000 * n + 1000 * game + move_no).best_move
        elif opponent == "random":
            moves = T.legal_moves(board)
            move = moves[int(rng.integers(len(moves)))]
        else:
            moves = rl.best_moves(board)
            move = moves[int(rng.integers(len(moves)))]
        board = T.play(board, move)
        move_no += 1
    w = T.winner(board)
    return job, "draw" if w == "draw" else ("win" if w == mcts_plays else "loss")


def matches(games=100):
    jobs = [(n, opp, side, g) for n in SIMS for opp in ("random", "minimax") for side in "XO" for g in range(games)]
    rows = parallel(play_game, jobs, chunksize=4)
    out = {}
    for n in SIMS:
        for opp in ("random", "minimax"):
            for side in "XO":
                results = [r for job, r in rows if job[:3] == (n, opp, side)]
                out.setdefault(f"mcts{n}_vs_{opp}", {})[f"as_{side}"] = {k: results.count(k) for k in ("win", "draw", "loss")}
    return dict(games_per_side=games, results=out)


if __name__ == "__main__":
    started = time.time()
    t = time.time(); pos = positions(); pos_wall = time.time() - t
    t = time.time(); m = matches(); m_wall = time.time() - t
    save("mcts", dict(c="sqrt(2)", rollouts="uniformly random moves", final_move="most visited child",
                      positions=pos, matches=m, wall_s_positions=pos_wall, wall_s_matches=m_wall), started)
    for name, e in pos.items():
        for n, v in e["by_sims"].items():
            print(name, n, v["seed0"]["visits"], "tree", v["seed0"]["tree_size"], "perfect-move rate", v["over_50_seeds"]["picks_a_perfect_move"])
    for k, v in m["results"].items():
        print(k, v)
    print("walls", round(pos_wall, 1), round(m_wall, 1))
