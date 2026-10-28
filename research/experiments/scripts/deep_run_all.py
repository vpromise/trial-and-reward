"""Run every reported deep-RL job (fresh seeds 10-14; pilots used seeds 0-5), 10 processes at a time, one core each.

  cd research/experiments && nohup .venv-deep/bin/python scripts/deep_run_all.py > logs/deep_run_all.out 2>&1 &
then:  scripts/deep_gradvar.py, deep_export.py, deep_aggregate.py, deep_report.py
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
PY = str(EXP / ".venv-deep" / "bin" / "python")
SEEDS = [10, 11, 12, 13, 14]
ENV = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def jobs():
    J = []
    for v in ["dqn", "double", "no_target", "no_replay"]:                       # slowest first
        J += [["scripts/deep_dqn.py", "--variant", v, "--seed", str(s)] for s in SEEDS]
    J += [["scripts/deep_sac.py", "--seed", str(s)] for s in SEEDS]
    J += [["scripts/deep_ppo.py", "--variant", "pendulum_clip", "--seed", str(s)] for s in SEEDS]
    for v in ["cartpole_clip", "cartpole_noclip", "cartpole_clip_aggressive", "cartpole_noclip_aggressive"]:
        J += [["scripts/deep_ppo.py", "--variant", v, "--seed", str(s)] for s in SEEDS]
    for v in ["reinforce", "reinforce_baseline", "a2c_gae"]:
        J += [["scripts/deep_pg.py", "--variant", v, "--seed", str(s)] + (["--snapshots"] if s == SEEDS[0] else [])
              for s in SEEDS]
    return J


def run(cmd):
    t = time.perf_counter()
    p = subprocess.run([PY] + cmd, cwd=EXP, env=ENV, capture_output=True, text=True)
    line = (p.stdout.strip().splitlines() or ["(no output)"])[-1]
    msg = f"[{time.perf_counter() - t:6.1f}s rc={p.returncode}] {' '.join(cmd[1:])} :: {line}"
    if p.returncode:
        msg += "\n" + p.stderr[-600:]
    print(msg, flush=True)
    return p.returncode


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None     # e.g. "deep_ppo,reinforce_baseline"
    todo = [j for j in jobs() if only is None or any(o in " ".join(j) for o in only)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(run, todo))
    print(f"ALL DONE in {time.perf_counter() - t0:.0f}s, failures: {sum(1 for r in rcs if r)}", flush=True)
