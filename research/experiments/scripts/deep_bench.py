"""Framework micro-benchmark for tiny RL networks on this Mac (numpy vs PyTorch CPU vs MLX).

Times (a) one forward pass on a single observation (acting) and (b) one full training step
(forward + backward + Adam) of an MLP on a minibatch, at the two sizes this course uses:
  DQN-size   4-64-64-2,   batch 64
  SAC-size   3-256-256-1, batch 256
Run:  .venv-deep/bin/python scripts/deep_bench.py  -> logs/deep_bench.json
"""
import json, time, platform, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent


def timeit(fn, n=2000, warm=200):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6  # microseconds


# ----------------------------------------------------------------- numpy (manual backprop)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep_common import MLP, Adam  # noqa: E402


def bench_numpy(sizes, batch):
    rng = np.random.default_rng(0)
    net = MLP(sizes, "relu", rng)
    opt = Adam(net.params(), lr=1e-3)
    x1 = rng.standard_normal((1, sizes[0])).astype(np.float32)
    xb = rng.standard_normal((batch, sizes[0])).astype(np.float32)
    yb = rng.standard_normal((batch, sizes[-1])).astype(np.float32)

    def act():
        net.forward(x1)

    def train():
        out, cache = net.forward_cache(xb)
        g = 2 * (out - yb) / out.size
        grads, _ = net.backward(cache, g)
        opt.step(grads)
    return timeit(act), timeit(train)


def bench_torch(sizes, batch):
    import torch
    torch.set_num_threads(1)
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x1 = torch.randn(1, sizes[0])
    xb = torch.randn(batch, sizes[0]); yb = torch.randn(batch, sizes[-1])

    def act():
        with torch.no_grad():
            net(x1)

    def train():
        loss = ((net(xb) - yb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return timeit(act), timeit(train)


def bench_mlx(sizes, batch, device):
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    mx.set_default_device(device)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

        def __call__(self, x):
            for l in self.layers[:-1]:
                x = nn.relu(l(x))
            return self.layers[-1](x)
    net = Net(); mx.eval(net.parameters())
    opt = optim.Adam(learning_rate=1e-3)
    x1 = mx.random.normal((1, sizes[0]))
    xb = mx.random.normal((batch, sizes[0])); yb = mx.random.normal((batch, sizes[-1]))
    loss_and_grad = nn.value_and_grad(net, lambda m, x, y: ((m(x) - y) ** 2).mean())

    def act():
        out = net(x1); mx.eval(out); np.array(out)  # acting needs the value back on the host

    def train():
        loss, grads = loss_and_grad(net, xb, yb)
        opt.update(net, grads); mx.eval(net.parameters(), opt.state)
    return timeit(act, n=1000, warm=100), timeit(train, n=1000, warm=100)


if __name__ == "__main__":
    import torch, mlx.core as mx
    res = {"machine": "Apple M4 Max, 64 GB", "python": platform.python_version(),
           "numpy": np.__version__, "torch": torch.__version__, "mlx": mx.__version__,
           "unit": "microseconds per call (mean)", "results": {}}
    for name, sizes, batch in [("dqn_4-64-64-2_b64", [4, 64, 64, 2], 64),
                               ("sac_3-256-256-1_b256", [3, 256, 256, 1], 256)]:
        r = {}
        r["numpy"] = bench_numpy(sizes, batch)
        r["torch_cpu_1thread"] = bench_torch(sizes, batch)
        r["mlx_cpu"] = bench_mlx(sizes, batch, mx.cpu)
        r["mlx_gpu"] = bench_mlx(sizes, batch, mx.gpu)
        res["results"][name] = {k: {"act_us": round(v[0], 1), "train_step_us": round(v[1], 1)} for k, v in r.items()}
        print(name, json.dumps(res["results"][name]))
    (HERE / "logs" / "deep_bench.json").write_text(json.dumps(res, indent=1))
