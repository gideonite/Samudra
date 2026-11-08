"""Data generation, splits, caching and the FIXED evaluation function for the PCGB toy study.

Frozen after setup (see notes.md, "Interface contract between prepare.py and train.py").

    python prepare.py --all | --dataset ks1d|kolmo2d|grayscott2d|denoise [--force]

Datasets (CPU profile)
    ks1d         Kuramoto-Sivashinsky, L=22*pi, 128 pts, ETDRK4 dt=0.25, model step 2.0 time units.
    kolmo2d      2D Kolmogorov flow (vorticity, k_f=4, drag), simulated on 64^2, spectrally truncated to 32^2.
    grayscott2d  Gray-Scott reaction-diffusion, 2 channels, 32^2, spot regime (non-chaotic control).
    denoise      MNIST (or procedural fallback) noisy->clean, T=2 "trajectories", not periodic.
"""

import argparse
import functools
import math
import os
import time

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DATASETS = ["ks1d", "kolmo2d", "grayscott2d", "denoise"]
SPLITS = ["train", "val", "test", "calib"]
N_TRAJ = {"train": 64, "val": 16, "test": 16, "calib": 16}  # physics: trajectories per split
N_SNAP = 65  # snapshots per physics trajectory -> 64 pairs per trajectory
ROLLOUT = 10
N_IC = 64  # fixed rollout ICs per split (physics)
SPLIT_SEED = {"train": 1, "val": 2, "test": 3, "calib": 4}


# ----------------------------------------------------------------------------- Kuramoto-Sivashinsky
def _etdrk4_coeffs(Lop, h, M=64):
    """Kassam & Trefethen contour-integral ETDRK4 coefficients (Lop: complex [K])."""
    r = torch.exp(1j * math.pi * (torch.arange(1, M + 1, dtype=torch.float64) - 0.5) / M)
    LR = h * Lop[:, None].to(torch.complex128) + r[None, :]
    E, E2 = torch.exp(h * Lop.to(torch.complex128)), torch.exp(h * Lop.to(torch.complex128) / 2)
    Q = h * ((torch.exp(LR / 2) - 1) / LR).mean(1).real
    f1 = h * ((-4 - LR + torch.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3).mean(1).real
    f2 = h * ((2 + LR + torch.exp(LR) * (-2 + LR)) / LR**3).mean(1).real
    f3 = h * ((-4 - 3 * LR - LR**2 + torch.exp(LR) * (4 - LR)) / LR**3).mean(1).real
    return E, E2, Q, f1, f2, f3


def ks_simulate(u0, n_steps, sample_every, dt=0.25, L=22 * math.pi):
    """u0: [B,N] float64. Returns snapshots [B, n_steps//sample_every + 1, N] (incl. initial state)."""
    B, N = u0.shape
    k = 2 * math.pi / L * torch.fft.rfftfreq(N, 1.0 / N).to(torch.float64)
    Lop = k**2 - k**4
    E, E2, Q, f1, f2, f3 = [c.to(torch.complex128) for c in _etdrk4_coeffs(Lop, dt)]
    g = -0.5j * k  # nonlinear term: -(1/2) d/dx (u^2)
    g[-1] = 0  # no derivative of the (real-valued) Nyquist mode

    def Nl(v):
        return g * torch.fft.rfft(torch.fft.irfft(v, n=N) ** 2)

    v = torch.fft.rfft(u0)
    out = [u0.clone()]
    for s in range(1, n_steps + 1):
        Nv = Nl(v)
        a = E2 * v + Q * Nv
        Na = Nl(a)
        b = E2 * v + Q * Na
        Nb = Nl(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = Nl(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if s % sample_every == 0:
            out.append(torch.fft.irfft(v, n=N))
    return torch.stack(out, 1)


def gen_ks1d(n_traj, seed, burn_time=600.0, dt=0.25, step_dt=2.0, n_snap=N_SNAP):
    g = torch.Generator().manual_seed(seed)
    N = 128
    x = torch.arange(N, dtype=torch.float64) * (22 * math.pi / N)
    u0 = torch.zeros(n_traj, N, dtype=torch.float64)
    for m in range(1, 6):  # random smooth low modes, small amplitude
        a = 0.1 * torch.randn(n_traj, 2, generator=g, dtype=torch.float64) / m
        u0 += a[:, :1] * torch.cos(m * x / 11) + a[:, 1:] * torch.sin(m * x / 11)
    burn = int(burn_time / dt)
    u = ks_simulate(u0, burn, burn)[:, -1]
    per = int(round(step_dt / dt))
    return ks_simulate(u, per * (n_snap - 1), per).float()[:, :, None, :]  # [B,T,1,128]


# ----------------------------------------------------------------------------- 2D Kolmogorov flow
KOLMO = dict(nu=1.0 / 1500, alpha=0.1, kf=4, F0=1.0, nsim=64, ngrid=32, dt=0.01, snap_dt=0.15, burn_time=100.0)


def _kolmo_setup(n, dtype=torch.float32):
    k = torch.fft.fftfreq(n, 1.0 / n).to(dtype)
    kr = torch.fft.rfftfreq(n, 1.0 / n).to(dtype)
    KX, KY = kr[None, :].expand(n, -1), k[:, None].expand(-1, len(kr))
    K2 = KX**2 + KY**2
    inv = torch.where(K2 > 0, 1.0 / K2.clamp(min=1), torch.zeros_like(K2))
    mask = ((KX.abs() < n / 3) & (KY.abs() < n / 3)).to(dtype)
    return KX, KY, K2, inv, mask


def kolmo_simulate(w0, n_steps, sample_every, p=KOLMO):
    """Pseudo-spectral vorticity eq with integrating-factor RK4 (float32). w0: [B,n,n] (physical)."""
    n = w0.shape[-1]
    w0 = w0.float()
    KX, KY, K2, inv, mask = _kolmo_setup(n)
    y = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    # forcing: u-forcing f=(F0 sin(kf y),0) -> vorticity forcing  -d/dy f = -F0 kf cos(kf y)
    Fh = torch.fft.rfft2((-p["F0"] * p["kf"] * torch.cos(p["kf"] * y))[:, None].expand(n, n).contiguous())
    lin = -(p["nu"] * K2 + p["alpha"])
    dt = p["dt"]
    Eh, E = torch.exp(lin * dt / 2), torch.exp(lin * dt)

    def rhs(wh):  # nonlinear term + forcing, dealiased
        psi = wh * inv  # laplacian psi = -w -> psi_hat = w_hat / k^2
        u = torch.fft.irfft2(1j * KY * psi, s=(n, n))
        v = torch.fft.irfft2(-1j * KX * psi, s=(n, n))
        wx = torch.fft.irfft2(1j * KX * wh, s=(n, n))
        wy = torch.fft.irfft2(1j * KY * wh, s=(n, n))
        return -torch.fft.rfft2(u * wx + v * wy) * mask + Fh

    wh = torch.fft.rfft2(w0) * mask
    out = [w0.clone()]
    for s in range(1, n_steps + 1):
        k1 = rhs(wh)
        k2 = rhs(Eh * (wh + dt / 2 * k1))
        k3 = rhs(Eh * wh + dt / 2 * k2)
        k4 = rhs(E * wh + dt * Eh * k3)
        wh = E * wh + dt / 6 * (E * k1 + 2 * Eh * (k2 + k3) + k4)
        if s % sample_every == 0:
            out.append(torch.fft.irfft2(wh, s=(n, n)))
    return torch.stack(out, 1)


def spectral_downsample(w, m):
    """w: [..., n, n] real -> [..., m, m] by spectral truncation (keeps |k| < m/2)."""
    n = w.shape[-1]
    wh = torch.fft.fft2(w)
    h = m // 2
    idx = torch.cat([torch.arange(0, h), torch.arange(n - h, n)])
    return torch.fft.ifft2(wh[..., idx[:, None], idx[None, :]]).real * (m / n) ** 2


def gen_kolmo2d(n_traj, seed, n_snap=N_SNAP, p=KOLMO):
    g = torch.Generator().manual_seed(seed)
    n = p["nsim"]
    # random smooth IC: band-limited noise, then burn in to the statistical steady state
    wh = torch.randn(n_traj, n, n // 2 + 1, dtype=torch.complex64, generator=g)
    KX, KY, K2, _, _ = _kolmo_setup(n)
    wh = wh * torch.exp(-K2 / 18.0)
    w0 = torch.fft.irfft2(wh, s=(n, n))
    w0 = w0 / w0.std((-1, -2), keepdim=True)
    per = int(round(p["snap_dt"] / p["dt"]))
    w = kolmo_simulate(w0, int(p["burn_time"] / p["dt"]), int(p["burn_time"] / p["dt"]), p)[:, -1]
    traj = kolmo_simulate(w, per * (n_snap - 1), per, p)
    return spectral_downsample(traj, p["ngrid"]).float()[:, :, None]  # [B,T,1,32,32]


# ----------------------------------------------------------------------------- Gray-Scott
GS = dict(F=0.04, k=0.06, Du=0.16, Dv=0.08, dt=1.0, burn=2000, snap=1000, n=32)


def gen_grayscott2d(n_traj, seed, n_snap=N_SNAP, p=GS):
    g = torch.Generator().manual_seed(seed)
    n = p["n"]
    u = torch.ones(n_traj, n, n, dtype=torch.float64)
    v = torch.zeros(n_traj, n, n, dtype=torch.float64)
    for b in range(n_traj):  # random square seeds + noise
        for _ in range(int(torch.randint(6, 13, (1,), generator=g))):
            i, j = torch.randint(0, n, (2,), generator=g).tolist()
            s = int(torch.randint(3, 7, (1,), generator=g))
            ii, jj = (torch.arange(i, i + s) % n)[:, None], (torch.arange(j, j + s) % n)[None, :]
            u[b, ii, jj], v[b, ii, jj] = 0.5, 0.25
    u += 0.02 * torch.randn(u.shape, generator=g, dtype=torch.float64)
    v += 0.02 * torch.randn(v.shape, generator=g, dtype=torch.float64)

    def lap(a):
        return torch.roll(a, 1, -1) + torch.roll(a, -1, -1) + torch.roll(a, 1, -2) + torch.roll(a, -1, -2) - 4 * a

    out = []
    for s in range(1, p["burn"] + p["snap"] * (n_snap - 1) + 1):
        uvv = u * v * v
        u, v = (u + p["dt"] * (p["Du"] * lap(u) - uvv + p["F"] * (1 - u)),
                v + p["dt"] * (p["Dv"] * lap(v) + uvv - (p["F"] + p["k"]) * v))
        if s >= p["burn"] and (s - p["burn"]) % p["snap"] == 0:
            out.append(torch.stack([u, v], 1))
    out = torch.stack(out, 1).float()  # [B,T,2,32,32]
    assert out[..., 1, :, :].std((-1, -2)).min() > 0.02, "Gray-Scott pattern died out in some trajectory"
    return out


# ----------------------------------------------------------------------------- denoise
DENOISE = dict(sigma=0.2, n_train=8192, n_eval=512, size=32)


def _procedural_images(n, seed, size=32):
    """Deterministic fallback: random superpositions of Gabor patches, bars and discs, in [0,1]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    imgs = np.zeros((n, size, size), np.float32)
    for i in range(n):
        im = np.zeros((size, size))
        for _ in range(rng.integers(2, 6)):
            cx, cy = rng.uniform(-0.7, 0.7, 2)
            th, amp = rng.uniform(0, math.pi), rng.uniform(0.4, 1.0)
            kind = rng.integers(0, 3)
            xr, yr = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th), -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
            if kind == 0:  # Gabor
                s, f = rng.uniform(0.15, 0.4), rng.uniform(3, 8)
                im += amp * np.exp(-(xr**2 + yr**2) / (2 * s**2)) * (0.5 + 0.5 * np.cos(f * xr + rng.uniform(0, 6.28)))
            elif kind == 1:  # oriented bar
                im += amp * ((np.abs(yr) < rng.uniform(0.04, 0.12)) & (np.abs(xr) < rng.uniform(0.2, 0.6)))
            else:  # soft disc
                im += amp * 1 / (1 + np.exp((np.sqrt(xr**2 + yr**2) - rng.uniform(0.1, 0.35)) / 0.03))
        imgs[i] = np.clip(im, 0, 1)
    return torch.from_numpy(imgs)


def _mnist_images():
    """Try MNIST via torchvision; return (train_imgs, test_imgs) float [0,1] 28x28 -> padded to 32, or None."""
    try:
        import torchvision

        root = os.path.join(DATA_DIR, "mnist_raw")
        tr = torchvision.datasets.MNIST(root, train=True, download=True)
        te = torchvision.datasets.MNIST(root, train=False, download=True)
        pad = lambda d: torch.nn.functional.pad(d.data.float() / 255.0, (2, 2, 2, 2))  # noqa: E731
        return pad(tr), pad(te)
    except Exception as e:  # noqa: BLE001 - fallback is documented and reported loudly
        print(f"[denoise] MNIST unavailable ({type(e).__name__}: {str(e)[:100]}); using procedural fallback")
        return None


def gen_denoise():
    d = DENOISE
    m = _mnist_images()
    if m is not None:
        tr, te = m
        g = torch.Generator().manual_seed(0)
        p = torch.randperm(len(tr), generator=g)
        n_ev = d["n_eval"]
        parts = {"train": tr[p[: d["n_train"]]], "val": tr[p[d["n_train"] : d["n_train"] + n_ev]],
                 "calib": tr[p[d["n_train"] + n_ev : d["n_train"] + 2 * n_ev]], "test": te[:n_ev]}
        src = "MNIST (32x32, zero-padded)"
    else:
        n = {"train": d["n_train"], "val": d["n_eval"], "test": d["n_eval"], "calib": d["n_eval"]}
        parts = {s: _procedural_images(n[s], 1000 + SPLIT_SEED[s], d["size"]) for s in SPLITS}
        src = "procedural Gabor/bar/disc images (32x32)"
    out = {"source": src}
    for s in SPLITS:
        clean = parts[s][:, None]  # [N,1,32,32]
        g = torch.Generator().manual_seed(5000 + SPLIT_SEED[s])  # FIXED noise per sample
        noisy = clean + d["sigma"] * torch.randn(clean.shape, generator=g)
        out[s] = torch.stack([noisy, clean], 1).float()  # [N,2,1,32,32]
    return out


# ----------------------------------------------------------------------------- caching / loading
def _normalize(raw):
    """raw: dict split -> [N,T,C,*sp] physical. Stats from train only."""
    tr = raw["train"]
    dims = [0, 1] + list(range(3, tr.ndim))
    mean = tr.mean(dims, keepdim=True)[0, 0]  # [C,1..]
    std = tr.std(dims, keepdim=True)[0, 0]
    out = {s: ((raw[s] - mean) / std).float().contiguous() for s in SPLITS}
    out["mean"], out["std"] = mean.flatten().clone(), std.flatten().clone()
    return out


def _meta(name, data):
    x = data["train"]
    return dict(periodic=name != "denoise", ndim=x.ndim - 3, channels=x.shape[2], spatial=tuple(x.shape[3:]),
                rollout_steps=1 if name == "denoise" else ROLLOUT, mean=data["mean"], std=data["std"])


def _fixed_ics(data, name, split):
    """Deterministic (traj, t0) rollout starts; returns (x0 [n,C,*], targets [n,R,C,*])."""
    x = data[split]
    R = 1 if name == "denoise" else ROLLOUT
    if name == "denoise":
        n = min(len(x), 512)
        return x[:n, 0], x[:n, 1:2]
    rng = np.random.default_rng(100 + SPLIT_SEED[split])
    pool = [(i, t) for i in range(x.shape[0]) for t in range(x.shape[1] - R)]
    sel = rng.choice(len(pool), N_IC, replace=False)
    idx = [pool[j] for j in sel]
    x0 = torch.stack([x[i, t] for i, t in idx])
    tg = torch.stack([x[i, t + 1 : t + 1 + R] for i, t in idx])
    return x0, tg


def _path(name):
    return os.path.join(DATA_DIR, f"{name}.pt")


def generate(name, force=False):
    torch.set_num_threads(2)  # 4 threads is slower than 2 for these small FFTs; train.py sets its own
    if os.path.exists(_path(name)) and not force:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    if name == "denoise":
        raw = gen_denoise()
        source = raw.pop("source")
    else:
        gen = {"ks1d": gen_ks1d, "kolmo2d": gen_kolmo2d, "grayscott2d": gen_grayscott2d}[name]
        raw = {s: gen(N_TRAJ[s], seed=SPLIT_SEED[s]) for s in SPLITS}
        source = "simulated"
    data = _normalize(raw)
    data["meta"] = _meta(name, data)
    data["source"], data["gen_seconds"] = source, time.time() - t0
    for s in SPLITS:
        data[f"ic_{s}"] = _fixed_ics(data, name, s)
    torch.save(data, _path(name))


@functools.lru_cache(maxsize=None)
def _load(name):
    if not os.path.exists(_path(name)):
        raise FileNotFoundError(f"{_path(name)} missing: run `python prepare.py --dataset {name}`")
    return torch.load(_path(name), weights_only=False)


def load(name):
    return _load(name)


def _pairs(x, idx):
    T = x.shape[1]
    return x[idx // (T - 1), idx % (T - 1)], x[idx // (T - 1), idx % (T - 1) + 1]


def get_train_pairs(name, n_train, seed):
    """n_train (x_t, x_{t+1}) pairs sampled uniformly without replacement from the train trajectories."""
    x = load(name)["train"]
    total = x.shape[0] * (x.shape[1] - 1)
    assert n_train <= total, f"n_train={n_train} > available pairs {total}"
    idx = torch.from_numpy(np.random.default_rng(seed).permutation(total)[:n_train])
    return _pairs(x, idx)


def get_calib_pairs(name, n=256, seed=0):
    x = load(name)["calib"]
    total = x.shape[0] * (x.shape[1] - 1)
    assert n <= total
    idx = torch.from_numpy(np.random.default_rng(seed).permutation(total)[:n])
    return _pairs(x, idx)


# ----------------------------------------------------------------------------- evaluation (FIXED)
@torch.no_grad()
def evaluate(model_fn, name, split="val", batch_size=64):
    """Autoregressive rollout from the fixed ICs. All errors in normalized units (= RMSE / std)."""
    d = load(name)
    x0, tg = d[f"ic_{split}"]
    R = tg.shape[1]
    sq = torch.zeros(R, tg.shape[2], dtype=torch.float64)  # sum of squared errors per step & channel
    finite = True
    for i in range(0, len(x0), batch_size):
        x = x0[i : i + batch_size]
        for r in range(R):
            x = model_fn(x)
            if not torch.isfinite(x).all():
                finite = False
            e = (x.double() - tg[i : i + batch_size, r].double()) ** 2
            sq[r] += e.sum(dim=[0] + list(range(2, e.ndim)))
    n_el = len(x0) * int(np.prod(tg.shape[3:]))
    mse = sq / n_el  # [R,C]
    per_step = mse.sqrt().mean(1)  # mean over channels
    ok = finite and bool(torch.isfinite(per_step).all())
    metric = float(per_step.mean()) if ok else float("inf")
    return {"metric": metric, "onestep_mse": float(mse[0].mean()), "per_step_nrmse": per_step.tolist(),
            "finite": ok}


# ----------------------------------------------------------------------------- summary / CLI
def _lag_corr(x):
    a, b = x[:, :-1].flatten(), x[:, 1:].flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _divergence_check(name):
    """Two nearby ICs on the attractor: report log10 of the RMS separation over time."""
    if name == "ks1d":
        u = gen_ks1d(2, 77, burn_time=600.0, n_snap=2)[:, -1, 0].double()
        v = u + 1e-8 * torch.randn_like(u)
        tr = ks_simulate(torch.stack([u[0], v[0]]), 4 * 8 * 40, 32)  # 40 model steps of 2.0... sampled every 8 time units
        sep = (tr[0] - tr[1]).pow(2).mean(-1).sqrt()
        return [f"t={8 * i}:{sep[i]:.1e}" for i in range(0, len(sep), 5)]
    if name == "kolmo2d":
        p = KOLMO
        g = torch.Generator().manual_seed(9)
        n = p["nsim"]
        w0 = torch.fft.irfft2(torch.randn(1, n, n // 2 + 1, dtype=torch.complex64, generator=g)
                              * torch.exp(-_kolmo_setup(n)[2] / 18.0), s=(n, n))
        nb = int(p["burn_time"] / p["dt"])
        w = kolmo_simulate(w0 / w0.std(), nb, nb, p)[:, -1]
        w2 = w + 1e-4 * torch.randn(w.shape, generator=g)  # float32 sim: perturbation must exceed eps
        per = int(round(1.0 / p["dt"]))  # sample every 1.0 time unit
        tr = kolmo_simulate(torch.cat([w, w2]), per * 15, per, p)
        sep = (tr[0] - tr[1]).pow(2).mean((-1, -2)).sqrt()
        return [f"t={i}:{sep[i]:.1e}" for i in range(0, len(sep), 2)] + [f"(rms w={tr[0].pow(2).mean().sqrt():.2f})"]
    return None


def summarize(name, diverge=True):
    d = load(name)
    m = d["meta"]
    print(f"== {name}  source={d['source']}  gen_time={d['gen_seconds']:.0f}s  file={os.path.getsize(_path(name)) / 1e6:.1f}MB")
    print(f"   meta: periodic={m['periodic']} ndim={m['ndim']} C={m['channels']} spatial={m['spatial']} "
          f"rollout={m['rollout_steps']} mean={m['mean'].tolist()} std={m['std'].tolist()}")
    print("   shapes: " + " ".join(f"{s}={tuple(d[s].shape)}" for s in SPLITS))
    print(f"   lag-1 corr (normalized, all splits train): {_lag_corr(d['train']):.3f}")
    ident = evaluate(lambda x: x, name, "val")
    ident_t = evaluate(lambda x: x, name, "test")
    x0, tg = d["ic_val"]
    state = {"r": 0}

    def oracle(x):  # true next step given the *true* previous state (matches by looking up in IC targets)
        r = state["r"]
        state["r"] = (r + 1) % tg.shape[1]
        return tg[: len(x), r] if len(x) == len(tg) else tg[: len(x), r]

    orc = evaluate(oracle, name, "val", batch_size=len(x0))
    print(f"   persistence val: metric={ident['metric']:.4f} onestep_mse={ident['onestep_mse']:.4f} "
          f"test metric={ident_t['metric']:.4f}")
    print(f"   oracle val: metric={orc['metric']:.2e} onestep_mse={orc['onestep_mse']:.2e}")
    if diverge:
        div = _divergence_check(name)
        if div:
            print("   divergence of tiny perturbation (ks 1e-8, kolmo 1e-4) (RMS separation, physical units): " + " ".join(div))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-diverge", action="store_true")
    a = ap.parse_args()
    names = DATASETS if a.all else [a.dataset]
    assert names != [None], "pass --all or --dataset"
    for n in names:
        generate(n, a.force)
        summarize(n, diverge=not a.no_diverge)


if __name__ == "__main__":
    main()
