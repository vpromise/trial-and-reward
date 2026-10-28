"""Shared helpers for the classic experiments: import rl_reference, downsample curves, save results, run in parallel.
Each exp_*.py writes scripts/out/<name>.json; build_classic.py merges them into classic.json and classic.md."""

import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SERIES = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERIES))
import rl_reference as rl  # noqa: E402,F401

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
WORKERS = max(1, (os.cpu_count() or 2) - 2)


def rounded(x, digits=4):
    """Round floats (and nested lists / dicts of them) to `digits` significant digits for compact JSON."""
    if isinstance(x, dict):
        return {str(k): rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, digits) for v in x]
    if isinstance(x, np.ndarray):
        return rounded(x.tolist(), digits)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if x == 0 or not np.isfinite(x) else float(f"{x:.{digits}g}")
    return x


def curve(y, max_points=200, dense_start=30, x0=1):
    """A curve as {"x": [...], "y": [...]} with at most max_points points: every point of the first `dense_start`
    steps (where the spikes are), then evenly spaced points (each the mean of its bin) for the rest."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= max_points:
        return {"x": list(range(x0, x0 + n)), "y": rounded(y)}
    xs, ys = list(range(x0, x0 + dense_start)), list(y[:dense_start])
    edges = np.linspace(dense_start, n, max_points - dense_start + 1).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs.append(int(x0 + (lo + hi - 1) // 2))
        ys.append(float(y[lo:hi].mean()))
    return {"x": xs, "y": rounded(ys)}


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0)
    return dict(chip=chip, memory_gb=round(mem / 2 ** 30), os=platform.platform(), python=sys.version.split()[0],
                numpy=np.__version__, cores=os.cpu_count(), workers=WORKERS)


def parallel(fn, jobs, chunksize=1):
    """fn over jobs on all cores but two (results in job order)."""
    with ProcessPoolExecutor(WORKERS) as pool:
        return list(pool.map(fn, jobs, chunksize=chunksize))


def save(name, data, started):
    data = dict(data, wall_s=round(time.time() - started, 1), machine=machine())
    (OUT / f"{name}.json").write_text(json.dumps(rounded(data), indent=1))
    print(f"wrote scripts/out/{name}.json in {data['wall_s']} s")
