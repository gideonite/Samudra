"""Rollout-horizon analysis (50 steps) on the VAL split only. Usage: python horizon.py [--check]
Loads runs/<tag>/model.pt, rolls out from ICs (every val traj x t0 in {0,5,10}), writes figs/horizon_<ds>_n<n>.png and
runs/_horizon.json. Exploded = non-finite or |x|>20 anywhere in the field; the flag is sticky once raised.
Error is computed on all ICs (non-finite predictions are mapped to +-1e6 so exploded rollouts show as huge nRMSE, not NaN;
the state of non-finite ICs is reset to 0 so they cannot contaminate other ICs)."""
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import prepare, train

HERE = Path(__file__).parent
H, T0S, EXPL = 50, (0, 5, 10), 20.0
DATASETS, NS, SEEDS = ("ks1d", "kolmo2d"), (512, 2048), (0, 1, 2)
ARMS = ("baseline", "stochdepth", "pcgb", "mask_only", "reweight_only")
PHASE1_COMMITS = {"f5d18a7", "61a7986", "1a864ce", "f06f258", "87d1bc7", "52aac42", "a8895b9"}


def find_run(ds, arm, n, s):
    """Phase-1 run dir: regular tag (7-hex commit suffix, not a Phase-2 commit whose tags were overwritten); calib_* as fallback."""
    cands = []
    for d in glob.glob(str(HERE / "runs" / f"{ds}_{arm}_n{n}_s{s}_*")):
        suf = os.path.basename(d).split("_")[-1]
        if re.fullmatch(r"[0-9a-f]{7}", suf) and suf in PHASE1_COMMITS and (Path(d) / "model.pt").exists():
            cands.append(d)
    if cands:
        assert len(cands) == 1, cands
        return cands[0]
    c = HERE / "runs" / f"calib_{ds}_{arm}"
    if c.exists() and n == 512 and s == 0:
        a = json.loads((c / "args.json").read_text())
        if (a["dataset"], a["arm"], a["n_train"], a["seed"]) == (ds, arm, n, s):
            return str(c)
    return None


def make_ics(val):
    x0, tg = [], []
    for t0 in T0S:
        assert t0 + H <= val.shape[1] - 1
        x0.append(val[:, t0]); tg.append(val[:, t0 + 1:t0 + 1 + H])
    return torch.cat(x0), torch.cat(tg)  # [M,C,*S], [M,H,C,*S]


def rollout(step_fn, x0, tg, bs=64):
    M, C = x0.shape[0], x0.shape[1]
    sp = tuple(range(2, x0.ndim))
    sq = torch.zeros(H, C, dtype=torch.float64); psq = torch.zeros(H, C, dtype=torch.float64)
    tsq = torch.zeros(H, C, dtype=torch.float64); bad = torch.zeros(H, dtype=torch.float64)
    n_el = int(np.prod(x0.shape[2:]))
    with torch.no_grad():
        for i in range(0, M, bs):
            x = x0[i:i + bs]; flag = torch.zeros(len(x), dtype=torch.bool)
            for r in range(H):
                x = step_fn(x)
                nonfin = ~torch.isfinite(x).flatten(1).all(1)
                flag |= nonfin | (x.abs().flatten(1).amax(1) > EXPL)
                x = torch.where(nonfin.view(-1, *[1] * (x.ndim - 1)), torch.zeros_like(x), x)
                xe = torch.nan_to_num(x, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6).double()
                t = tg[i:i + bs, r].double()
                sq[r] += ((xe - t) ** 2).sum(dim=[0] + list(sp))
                psq[r] += (xe ** 2).sum(dim=[0] + list(sp)); tsq[r] += (t ** 2).sum(dim=[0] + list(sp))
                bad[r] += flag.sum()
    n = M * n_el
    return dict(nrmse=(sq / n).sqrt().mean(1).tolist(),
                std_ratio=((psq / n).sqrt() / (tsq / n).sqrt()).mean(1).tolist(),  # fields are ~zero-mean (normalized)
                exploded=(bad / M).tolist())


def stats(curves, key):
    return np.array([c[key] for c in curves])


def first_cross(c):
    idx = np.nonzero(np.asarray(c) > 1.0)[0]
    return int(idx[0]) + 1 if len(idx) else H + 1  # H+1 = never within 50 steps (censored)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true", help="also verify prepare.evaluate vs logged val metric")
    a = ap.parse_args(); torch.set_num_threads(1)
    (HERE / "figs").mkdir(exist_ok=True)
    out, missing, checks = {}, [], []
    for ds in DATASETS:
        d = prepare.load(ds); x0, tg = make_ics(d["val"]); meta = d["meta"]
        refs = {"persistence": rollout(lambda x: x, x0, tg), "climatology": rollout(torch.zeros_like, x0, tg)}
        for n in NS:
            res = {"reference": refs, "arms": {}}
            for arm in ARMS:
                runs = []
                for s in SEEDS:
                    rd = find_run(ds, arm, n, s)
                    if rd is None: missing.append((ds, arm, n, s)); continue
                    cfg = json.loads((Path(rd) / "args.json").read_text())
                    m = train.build_model(meta, cfg["width"], cfg["drop_mode"])
                    m.load_state_dict(torch.load(Path(rd) / "model.pt")); m.eval()
                    r = rollout(m, x0, tg); r.update(seed=s, run=os.path.basename(rd)); runs.append(r)
                    if a.check:
                        with torch.no_grad(): ev = prepare.evaluate(m, ds, "val")
                        fin = [json.loads(l) for l in open(Path(rd) / "log.jsonl")][-1]
                        checks.append((os.path.basename(rd), ev["metric"], fin.get("metric")))
                if runs: res["arms"][arm] = runs
            out[f"{ds}_n{n}"] = res
    # summaries
    table = {}
    for key, res in out.items():
        table[key] = {}
        for arm, runs in res["arms"].items():
            nr = stats(runs, "nrmse"); fc = np.array([first_cross(c) for c in nr])
            row = {f"step{k}": [float(nr[:, k - 1].mean()), float(nr[:, k - 1].std())] for k in (1, 10, 25, 50)}
            row["first_gt1"] = [float(fc.mean()), float(fc.std()), int((fc > H).sum())]  # 51 = never (censored)
            row["exploded_frac_step50"] = [float(stats(runs, "exploded")[:, -1].mean()), float(stats(runs, "exploded")[:, -1].max())]
            row["std_ratio"] = {f"step{k}": float(stats(runs, "std_ratio")[:, k - 1].mean()) for k in (1, 10, 25, 50)}
            row["n_seeds"] = len(runs); row["mean10"] = float(nr[:, :10].mean(axis=1).mean())
            table[key][arm] = row
    json.dump({"config": dict(H=H, t0s=T0S, explode_thresh=EXPL), "missing": missing, "summary": table, "curves": out},
              open(HERE / "runs" / "_horizon.json", "w"), indent=1)
    # figures
    cols = dict(baseline="#1f77b4", stochdepth="#2ca02c", pcgb="#d62728", mask_only="#ff7f0e", reweight_only="#9467bd")
    for key, res in out.items():
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4)); x = np.arange(1, H + 1)
        for arm, runs in res["arms"].items():
            for j, (k, ylog) in enumerate((("nrmse", True), ("std_ratio", False), ("exploded", False))):
                v = stats(runs, k); v = np.clip(v, 1e-4, 1e4) if k == "nrmse" else v
                ax[j].plot(x, v.mean(0), color=cols[arm], label=f"{arm} (n={len(runs)})", ls="--" if arm in ("reweight_only", "mask_only") else "-")
                ax[j].fill_between(x, v.min(0), v.max(0), color=cols[arm], alpha=0.18, lw=0)
        for nm, ls in (("persistence", ":"), ("climatology", "-.")):
            ax[0].plot(x, res["reference"][nm]["nrmse"], color="k", ls=ls, lw=1, label=nm)
            ax[1].plot(x, res["reference"][nm]["std_ratio"], color="k", ls=ls, lw=1)
        ax[0].axhline(1, color="gray", lw=0.5); ax[0].set_yscale("log"); ax[0].set_title(f"{key}: per-step nRMSE (mean, min/max over seeds)")
        ax[1].axhline(1, color="gray", lw=0.5); ax[1].set_title("predicted/true field std"); ax[2].set_title(f"fraction exploded (|x|>{EXPL:g} or non-finite)")
        for A in ax: A.set_xlabel("rollout step"); A.grid(alpha=0.25)
        ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(HERE / "figs" / f"horizon_{key}.png", dpi=110); plt.close(fig)
    # table
    for key, t in table.items():
        print(f"\n== {key}  (persistence@1/10/50: " + "/".join(f"{out[key]['reference']['persistence']['nrmse'][k-1]:.2f}" for k in (1, 10, 50)) + ")")
        print(f"{'arm':14s} {'s1':>14s} {'s10':>14s} {'s25':>14s} {'s50':>14s} {'first>1':>12s} {'expl@50':>8s} {'stdratio@10/50':>15s}")
        for arm, r in t.items():
            f = lambda v: f"{v[0]:.3g}±{v[1]:.2g}"
            fc = r["first_gt1"]; fcs = (">50" if fc[2] == r["n_seeds"] else f"{fc[0]:.1f}±{fc[1]:.1f}" + (f"({fc[2]}cens)" if fc[2] else ""))
            print(f"{arm:14s} {f(r['step1']):>14s} {f(r['step10']):>14s} {f(r['step25']):>14s} {f(r['step50']):>14s} {fcs:>12s} {r['exploded_frac_step50'][0]:8.2f} {r['std_ratio']['step10']:7.2f}/{r['std_ratio']['step50']:.2f}")
    print("\nmissing:", missing)
    for c in checks: print("check", c)


if __name__ == "__main__":
    main()
