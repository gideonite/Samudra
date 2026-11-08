"""PCGB toy study: ConvNeXt U-Net (generic 1D/2D) + five training arms under a wall-clock budget.

Arms: baseline | stochdepth | mask_only | reweight_only | pcgb  (see program.md for the method).

Conventions
- Skip index i = 0..3 is the skip from down level i (0 = highest resolution). Up level j consumes skip 3-j.
- Block index 0-3 = down blocks, 4 = mid, 5-8 = up blocks (5 is the deepest), 9 = final block.
- Skip-mask id m in 0..15: bit i of m (m >> i & 1) == 1 means skip i is KEPT. m=15 all-keep, m=0 all-drop,
  m=0b0001 keeps only the highest-resolution skip.
- Rounds are wall-clock slices: round t ends when elapsed >= (t+1) * budget / n_rounds, so every arm
  (including baseline/stochdepth, which just log) shares the same slicing; K (steps/round) is whatever fits.
  Scoring + reweighting time is spent inside the slice, i.e. it is charged to the budget.
"""
import argparse, fcntl, json, math, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare

HERE = Path(__file__).resolve().parent
ARMS = ["baseline", "stochdepth", "mask_only", "reweight_only", "pcgb"]
N_SKIPS, N_BLOCKS = 4, 10


# ----------------------------------------------------------------------------- model
DROP_FNS = {  # (skip tensor, scale in [0,1] or 1/(1-p)) -> tensor fed to the decoder. Add new modes here.
    "zero": lambda s, k: s * k,
    "mean": lambda s, k: s * k + (1 - k) * s.mean(dim=tuple(range(2, s.ndim)), keepdim=True),
}


class Block(nn.Module):
    """ConvNeXt block: dwconv -> GroupNorm(1) -> 1x1 expand x4 -> GELU -> 1x1 -> residual add."""

    def __init__(self, c, nd, k, periodic):
        super().__init__()
        conv = (nn.Conv1d, nn.Conv2d)[nd - 1]
        self.dw = conv(c, c, k, padding=k // 2, groups=c, padding_mode="circular" if periodic else "zeros")
        self.norm = nn.GroupNorm(1, c)
        self.pw1, self.pw2 = conv(c, 4 * c, 1), conv(4 * c, c, 1)

    def forward(self, x, scale=1.0):
        if isinstance(scale, float) and scale == 0.0:
            return x  # dropped: residual branch contributes 0
        return x + scale * self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))


class UNet(nn.Module):
    def __init__(self, meta, width=10, drop_mode="zero"):
        super().__init__()
        nd, cin, per = meta["ndim"], meta["channels"], meta["periodic"]
        k = 7 if nd == 1 else 5
        conv = (nn.Conv1d, nn.Conv2d)[nd - 1]
        ch = [width * m for m in (1, 2, 4, 8, 8)]
        self.drop = DROP_FNS[drop_mode]
        self.stem = conv(cin, ch[0], 3, padding=1, padding_mode="circular" if per else "zeros")
        self.blocks = nn.ModuleList([Block(c, nd, k, per) for c in ch[:4] + [ch[4]] + ch[3::-1] + [ch[0]]])
        self.downs = nn.ModuleList([conv(ch[i], ch[i + 1], 2, stride=2) for i in range(4)])
        self.projs = nn.ModuleList([conv(ch[i + 1] + ch[i], ch[i], 1) for i in (3, 2, 1, 0)])
        self.head = conv(ch[0], cin, 1)
        with torch.no_grad():
            self.head.weight.mul_(0.1), self.head.bias.zero_()
        self.mode = "nearest"
        self.skip_gnorm = [0.0] * N_SKIPS  # filled by hooks when track_grads is set
        self.track_grads = False

    @staticmethod
    def _scales(m, n):
        return [1.0] * n if m is None else [float(v) for v in m]

    def encode(self, x, bm):
        h, skips = self.stem(x), []
        for i in range(4):
            h = self.blocks[i](h, bm[i]); skips.append(h); h = self.downs[i](h)
        return self.blocks[4](h, bm[4]), skips

    def decode(self, x, h, skips, sm, bm):
        for j in range(4):
            i = 3 - j
            s = skips[i] if sm[i] == 1.0 else self.drop(skips[i], sm[i])
            if self.track_grads and s.requires_grad:  # grad flowing into the tensor consumed by the decoder
                s.register_hook(lambda g, i=i: self.skip_gnorm.__setitem__(i, float(g.norm())))
            h = self.projs[j](torch.cat([F.interpolate(h, scale_factor=2, mode=self.mode), s], 1))
            h = self.blocks[5 + j](h, bm[5 + j])
        return x + self.head(self.blocks[9](h, bm[9]))  # residual prediction

    def forward(self, x, skip_mask=None, block_mask=None, enc=None):
        """enc = cached self.encode(x, bm) output (valid while block_mask/encoder blocks are unchanged)."""
        sm, bm = self._scales(skip_mask, N_SKIPS), self._scales(block_mask, N_BLOCKS)
        h, skips = enc if enc is not None else self.encode(x, bm)
        return self.decode(x, h, skips, sm, bm)


def mask_bools(m):
    return [bool((m >> i) & 1) for i in range(N_SKIPS)]


# ----------------------------------------------------------------------------- PCGB pieces
def weight_stats(D):
    p = D / D.sum()
    ent = float(-(p * (p + 1e-30).log()).sum() / math.log(len(D))) if len(D) > 1 else 1.0
    return ent, float(D.max() / D.min())


def reweight(D, idx, r, lam=0.3, ratio=5.0):
    """AdaBoost.R2 variant on the scored subset `idx` (residual norms r); D elsewhere stays stale.
    D' is rescaled to the subset's D-mass so that 0.7*D + 0.3*D' mixes consistently. Then clamp + mean-1."""
    d = D[idx]
    L = r / r.max().clamp_min(1e-12)
    Lbar = float(((d / d.sum()) * L).sum())
    beta = min(Lbar / max(1 - Lbar, 1e-12), 1.0)
    dp = d * beta ** (1 - L)
    dp = dp * d.sum() / dp.sum()
    D = D.clone()
    D[idx] = (1 - lam) * d + lam * dp
    med = D.median()  # clamp symmetrically (in log space) around the median so max/min <= ratio exactly
    D = D.clamp(med / math.sqrt(ratio), med * math.sqrt(ratio))
    return D / D.mean(), Lbar, beta


@torch.no_grad()
def score_round(model, args, X, Y, D, rnd, calib):
    """Returns (idx, r, S[16] or None). Scoring subset = strided slice of the TRAIN set (<=256 samples),
    offset by round so all samples get visited. r = per-sample RMS residual of the unmasked net.
    S[m] = mean_i D_i * mse_i(masked net m) on the same subset (--score_on train, D lives on train samples),
    or the unweighted mean over calibration pairs (--score_on calib)."""
    n = len(X)
    stride = math.ceil(n / args.score_n)
    idx = torch.arange(rnd % stride, n, stride)[: args.score_n]
    x, y = X[idx], Y[idx]
    model.eval()
    def err(xx, yy, m, enc=None): return ((model(xx, skip_mask=mask_bools(m), enc=enc) - yy) ** 2).flatten(1).mean(1)
    enc = model.encode(x, [1.0] * N_BLOCKS)  # encoder is mask-independent: compute once for all 16 masks
    e_full = err(x, y, 15, enc)
    S = None
    if args.arm in ("mask_only", "pcgb"):
        if args.score_on == "train":
            S = torch.stack([(D[idx] * (e_full if m == 15 else err(x, y, m, enc))).mean() for m in range(16)])
        else:
            xc, yc = calib[0][: args.score_n], calib[1][: args.score_n]
            encc = model.encode(xc, [1.0] * N_BLOCKS)
            S = torch.stack([err(xc, yc, m, encc).mean() for m in range(16)])
    model.train()
    return idx, e_full.sqrt(), S


def select_mask(S, rule, rng, rnd, temp):
    if rule == "argmax": return int(S.argmax())
    if rule == "argmin": return int(S.argmin())
    if rule == "random": return rng.randrange(16)
    if rule == "roundrobin": return rnd % 16
    if rule == "softmax":
        p = torch.softmax((S - S.mean()) / (S.std() + 1e-12) / temp, 0).tolist()
        return rng.choices(range(16), weights=p)[0]
    raise ValueError(rule)


# ----------------------------------------------------------------------------- helpers
def git_commit():
    def run(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    c = run("rev-parse", "--short", "HEAD") or "nogit"
    # dirty = tracked code files changed (results.tsv/runs are written concurrently, so excluded)
    return c + ("-dirty" if run("status", "--porcelain", "-uno", "--", "train.py", "prepare.py") else "")


def run_eval(model, name, split):
    model.eval()
    with torch.no_grad():
        out = prepare.evaluate(lambda x: model(x), name, split=split)
    model.train()
    return out


def lr_at(frac, base, warm=0.02):
    if frac < warm: return base * max(frac / warm, 0.01)
    return base * 0.5 * (1 + math.cos(math.pi * min((frac - warm) / (1 - warm), 1.0)))


class Sampler:  # epoch-based, without replacement, seeded
    def __init__(self, n, bs, seed):
        self.n, self.bs, self.g, self.perm, self.pos = n, bs, torch.Generator().manual_seed(seed), None, 0
        self.perm = torch.randperm(n, generator=self.g)

    def next(self):
        if self.pos + self.bs > self.n:
            self.perm, self.pos = torch.randperm(self.n, generator=self.g), 0
        b = self.perm[self.pos:self.pos + self.bs]; self.pos += self.bs
        return b


class Log:
    def __init__(self, path): self.f = open(path, "w")
    def __call__(self, **kw): self.f.write(json.dumps(kw) + "\n"); self.f.flush()


def append_row(path, row):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            f.write("commit\tdataset\tarm\tn_train\tseed\tmetric\tonestep_mse\twallclock_s\tnotes\n")
        f.write("\t".join(str(v) for v in row) + "\n"); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def build_model(meta, width=10, drop_mode="zero"):
    return UNet(meta, width, drop_mode)


# ----------------------------------------------------------------------------- main
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ks1d"); p.add_argument("--arm", choices=ARMS, default="baseline")
    p.add_argument("--n_train", type=int, default=512); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget_s", type=float, default=720); p.add_argument("--tag", default=None)
    p.add_argument("--out_dir", default=str(HERE), help="where runs/ and results.tsv live")
    p.add_argument("--eval_test", action="store_true"); p.add_argument("--n_evals", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--width", type=int, default=10); p.add_argument("--drop_mode", default="zero", choices=list(DROP_FNS))
    p.add_argument("--n_rounds", type=int, default=30)
    p.add_argument("--score_n", type=int, default=128); p.add_argument("--score_on", choices=["train", "calib"], default="train")
    p.add_argument("--lam", type=float, default=0.3); p.add_argument("--clamp_ratio", type=float, default=5.0)
    p.add_argument("--mask_rule", default="argmax", choices=["argmax", "argmin", "softmax", "random", "roundrobin"])
    p.add_argument("--mask_temp", type=float, default=1.0)
    p.add_argument("--reset_adam", action="store_true", help="re-create AdamW at every round (default: keep state)")
    p.add_argument("--sd_skip_p", type=float, default=0.2, help="stochdepth: skip drop prob")
    p.add_argument("--sd_block_p", type=float, default=0.1, help="stochdepth: block p_l = sd_block_p*(l+1)/10")
    p.add_argument("--log_grad_norms", type=int, default=0, help="every N steps log block/skip grad norms (0=off)")
    return p.parse_args(argv)


def main(args):
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    commit = git_commit()
    name, arm = args.dataset, args.arm
    tag = args.tag or f"{name}_{arm}_n{args.n_train}_s{args.seed}_{commit}"
    out = Path(args.out_dir); run_dir = out / "runs" / tag; run_dir.mkdir(parents=True, exist_ok=True)
    log = Log(run_dir / "log.jsonl")
    meta = prepare.load(name)["meta"]
    X, Y = prepare.get_train_pairs(name, args.n_train, args.seed)
    calib = prepare.get_calib_pairs(name, 256, 0)
    n, bs = len(X), min(args.bs, len(X))
    model = build_model(meta, args.width, args.drop_mode).train()
    n_params = sum(p.numel() for p in model.parameters())
    mk_opt = lambda: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    opt, sampler = mk_opt(), Sampler(n, bs, args.seed)
    rng = random.Random(args.seed); gen = torch.Generator().manual_seed(args.seed + 1)
    (run_dir / "args.json").write_text(json.dumps({**vars(args), "n_params": n_params, "meta_ndim": meta["ndim"]}))
    log(type="config", **vars(args), n_params=n_params, commit=commit)

    use_mask, use_rw = arm in ("mask_only", "pcgb"), arm in ("reweight_only", "pcgb")
    D, mask = torch.ones(n), 15
    slice_s, eval_s = args.budget_s / args.n_rounds, args.budget_s / args.n_evals
    next_round, next_eval = slice_s, eval_s
    step = rnd = r_steps = 0; r_loss = 0.0
    masks, ents, diverged, curve = [], [], False, []
    t0 = time.perf_counter()
    while True:
        el = time.perf_counter() - t0
        if el >= args.budget_s: break
        if el >= next_eval and len(curve) < args.n_evals - 1:  # intermediate val evals, counted in the budget
            ev = run_eval(model, name, "val"); curve.append(ev["metric"])
            log(type="eval", step=step, elapsed=time.perf_counter() - t0, metric=ev["metric"], onestep_mse=ev["onestep_mse"])
            next_eval += eval_s; continue
        if el >= next_round:  # ---- end of round
            entry = dict(type="round", round=rnd, steps=step, K=r_steps, elapsed=el, mask=mask,
                         train_loss=r_loss / max(r_steps, 1))
            if use_mask or use_rw:
                idx, r, S = score_round(model, args, X, Y, D, rnd, calib)
                if use_rw:
                    D, Lbar, beta = reweight(D, idx, r, args.lam, args.clamp_ratio)
                    entry.update(Lbar=Lbar, beta=beta)
                if use_mask:
                    mask = select_mask(S, args.mask_rule, rng, rnd, args.mask_temp)
                    entry["S"] = S.tolist()
                masks.append(mask)
                entry["next_mask"] = mask; entry["score_s"] = time.perf_counter() - t0 - el
            ent, ratio = weight_stats(D); ents.append(ent)
            entry.update(entropy=ent, ratio=ratio); log(**entry)
            if args.reset_adam: opt = mk_opt()
            rnd += 1; r_steps, r_loss = 0, 0.0
            next_round = (int((time.perf_counter() - t0) / slice_s) + 1) * slice_s
            continue
        # ---- one training step
        for g in opt.param_groups: g["lr"] = lr_at(el / args.budget_s, args.lr)
        b = sampler.next(); xb, yb = X[b], Y[b]
        sk, bk = mask_bools(mask), None
        if arm == "stochdepth":
            ps = torch.tensor([args.sd_skip_p] * N_SKIPS)
            pb = torch.tensor([args.sd_block_p * (l + 1) / N_BLOCKS for l in range(N_BLOCKS)])
            sk = ((torch.rand(N_SKIPS, generator=gen) >= ps).float() / (1 - ps)).tolist()
            bk = ((torch.rand(N_BLOCKS, generator=gen) >= pb).float() / (1 - pb)).tolist()
        track = args.log_grad_norms and step % args.log_grad_norms == 0
        model.track_grads = bool(track)
        mse = ((model(xb, skip_mask=sk, block_mask=bk) - yb) ** 2).flatten(1).mean(1)
        loss = (D[b] * mse).mean()
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward()
        if track:
            gn = [math.sqrt(sum(float(p.grad.norm()) ** 2 for p in blk.parameters())) for blk in model.blocks]
            log(type="gradnorm", step=step, block=gn, skip=list(model.skip_gnorm), mask=mask)
        model.track_grads = False
        if args.clip > 0: nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        step += 1; r_steps += 1; r_loss += loss.item()
    wall = time.perf_counter() - t0

    # ---- final evals (outside budget; unmasked final weights)
    fin = run_eval(model, name, "val")
    log(type="final", split="val", steps=step, wall=wall, diverged=diverged, **{k: fin[k] for k in ("metric", "onestep_mse", "per_step_nrmse")})
    test = run_eval(model, name, "test") if args.eval_test else None
    if test: log(type="final", split="test", **{k: test[k] for k in ("metric", "onestep_mse", "per_step_nrmse")})
    torch.save(model.state_dict(), run_dir / "model.pt")

    notes = [f"rounds={rnd}", f"steps={step}", f"steps_per_s={step / max(wall, 1e-9):.1f}", f"params={n_params}"]
    if masks:
        top = max(set(masks), key=masks.count)
        if use_mask and masks.count(top) / len(masks) > 0.7: notes.append(f"MASK_COLLAPSE(m={top},{masks.count(top)}/{len(masks)})")
    if use_rw:
        notes.append(f"minH={min(ents):.3f}")
        if ents[-1] < 0.8: notes.append(f"ENTROPY_WARN(final={ents[-1]:.3f})")
    if diverged: notes.append("DIVERGED")
    if not math.isfinite(fin["metric"]): notes.append("NONFINITE_ROLLOUT")
    if test: notes.append(f"test_metric={test['metric']:.5g};test_onestep={test['onestep_mse']:.5g}")
    extra = [f"{k}={getattr(args, k)}" for k, d in (("mask_rule", "argmax"), ("score_on", "train"), ("lam", 0.3), ("clamp_ratio", 5.0),
             ("drop_mode", "zero"), ("lr", 1e-3), ("width", 10), ("reset_adam", False)) if getattr(args, k) != d]
    append_row(out / "results.tsv", [commit, name, arm, args.n_train, args.seed, f"{fin['metric']:.6g}", f"{fin['onestep_mse']:.6g}",
                                    f"{wall:.1f}", " ".join(notes + extra)])
    print(f"{tag}: metric={fin['metric']:.5g} onestep={fin['onestep_mse']:.5g} steps={step} rounds={rnd} {' '.join(notes)}")
    if diverged: sys.exit(1)


if __name__ == "__main__":
    main(get_args())
