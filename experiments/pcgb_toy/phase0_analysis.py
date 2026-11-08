"""Phase 0 analysis: lesion study (+random-k block removal), gradient-norm concentration, effective path length.

Usage: python phase0_analysis.py <run_dir> <label> [--out DIR] [--skip-lesion]
Single thread; writes <out>/<label>.json. Plots are made by phase0_plots.py from these jsons.
Unit convention: skip i bit set = kept (i=0 highest res); blocks 0-3 down, 4 mid, 5-8 up (5 deepest), 9 final.
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch

import prepare
import train

BL = train.N_BLOCKS


def gini(v):
    v = np.sort(np.asarray(v, float)); n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def lesion(model, name, split, skip=None, block=None):
    with torch.no_grad():
        ev = prepare.evaluate(lambda x: model(x, skip_mask=skip, block_mask=block), name, split=split)
    return dict(metric=ev["metric"], onestep=ev["onestep_mse"])


def lesion_study(model, name, split, n_sub=10, seed=0):
    res = {"full": lesion(model, name, split), "skip": {}, "block": {}, "randk": {}}
    for m in range(16):
        res["skip"][m] = lesion(model, name, split, skip=train.mask_bools(m))
    for b in range(BL):
        res["block"][b] = lesion(model, name, split, block=[i != b for i in range(BL)])
    rng = random.Random(seed)
    for k in range(1, 6):
        rs = []
        for _ in range(n_sub):
            drop = set(rng.sample(range(BL), k))
            rs.append(lesion(model, name, split, block=[i not in drop for i in range(BL)]))
        res["randk"][k] = dict(metric=[r["metric"] for r in rs], onestep=[r["onestep"] for r in rs])
    return res


def grad_analysis(run, model):
    rows = [json.loads(l) for l in open(run / "log.jsonl")]
    g = [r for r in rows if r["type"] == "gradnorm"]
    B = np.array([r["block"] for r in g]); S = np.array([r["skip"] for r in g]); steps = [r["step"] for r in g]
    npb = np.array([sum(p.numel() for p in b.parameters()) for b in model.blocks])
    # skip tensor numel per sample: ch[i] * L / 2^i   (infer from a dummy forward)
    meta_x = None
    out = dict(steps=steps, block=B.tolist(), skip=S.tolist(), block_params=npb.tolist())
    n = len(g); third = max(n // 3, 1)
    phases = {"early": slice(0, third), "mid": slice(third, 2 * third), "late": slice(2 * third, n)}
    summ = {}
    for ph, sl in phases.items():
        b, s = B[sl], S[sl]
        bsq = (b ** 2).mean(0); ssq = (s ** 2).mean(0)  # mean squared norm -> energy share
        bmean, smean = b.mean(0), s.mean(0)
        pp = bmean / np.sqrt(npb)
        summ[ph] = dict(
            block_norm=bmean.tolist(), block_share=(bsq / bsq.sum()).tolist(), block_perparam_rms=pp.tolist(),
            block_perparam_share=(pp ** 2 / (pp ** 2).sum()).tolist(),
            skip_norm=smean.tolist(), skip_share=(ssq / ssq.sum()).tolist(),
            gini_block=gini(bmean), gini_block_perparam=gini(pp), gini_skip=gini(smean),
            top2_block=float(np.sort(bsq)[-2:].sum() / bsq.sum()), top2_skip=float(np.sort(ssq)[-2:].sum() / ssq.sum()),
            deepest_skip_over_max=float(smean[3] / smean.max()), deepest_blocks_over_max=float(max(bmean[4], bmean[5]) / bmean.max()))
    out["phases"] = summ
    return out


def path_length(model, X, Y, n_per_k=30, seed=0):
    """Veit-style: 10 blocks + 4 skips are 14 binary units; sample random subsets with k units kept.
    For each: loss (one-step MSE on a train batch), ||d loss/d theta||, and ||(J-I)^T v|| (input-gradient of a fixed random
    cotangent through the masked net -- path-structure quantity independent of the residual error; the outer identity x+head(.) is subtracted since it would dominate)."""
    rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    x = X[:64].clone().requires_grad_(True); y = Y[:64]
    with torch.no_grad(): v = torch.randn(model(x.detach()).shape, generator=gen)
    out = {}
    model.eval()
    for k in range(0, 15):
        L, GP, GX = [], [], []
        for _ in range(n_per_k):
            keep = set(rng.sample(range(14), k))
            bm = [float(i in keep) for i in range(BL)]; sm = [i + BL in keep for i in range(4)]
            model.zero_grad(set_to_none=True)
            yh = model(x, skip_mask=sm, block_mask=bm)
            loss = ((yh - y) ** 2).mean()
            gp = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True, allow_unused=True)
            gx = torch.autograd.grad((yh * v).sum() / v.numel() ** 0.5, x)[0]
            L.append(float(loss.detach())); GP.append(float(torch.sqrt(sum((g ** 2).sum() for g in gp if g is not None)))); GX.append(float((gx - v / v.numel() ** 0.5).norm()))
        out[k] = dict(loss=L, gparam=GP, gx=GX)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir"); p.add_argument("label"); p.add_argument("--split", default="val")
    p.add_argument("--out", default="/tmp/claude-0/-home-user-Samudra/09c6e2b1-2a64-5045-8ba8-dbdc1b668dd9/scratchpad/phase0")
    p.add_argument("--n_sub", type=int, default=10); p.add_argument("--no_lesion", action="store_true"); p.add_argument("--no_path", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    run = Path(a.run_dir); cfg = json.loads((run / "args.json").read_text()); name = cfg["dataset"]
    meta = prepare.load(name)["meta"]
    model = train.build_model(meta, cfg["width"], cfg["drop_mode"]); model.load_state_dict(torch.load(run / "model.pt")); model.eval()
    res = dict(run=str(run), dataset=name, n_train=cfg["n_train"], arm=cfg["arm"])
    if not a.no_lesion: res["lesion"] = lesion_study(model, name, a.split, a.n_sub)
    res["grad"] = grad_analysis(run, model)
    if not a.no_path:
        X, Y = prepare.get_train_pairs(name, cfg["n_train"], cfg["seed"])
        res["path"] = path_length(model, X, Y)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out, f"{a.label}.json")
    if a.no_lesion and f.exists(): res = {**json.loads(f.read_text()), **{k: v for k, v in res.items() if k in ("grad", "path")}}  # merge into existing
    f.write_text(json.dumps(res))
    print("done", a.label)


if __name__ == "__main__":
    main()
