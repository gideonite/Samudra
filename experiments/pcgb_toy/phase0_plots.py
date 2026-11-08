"""Tables + figures from phase0 jsons. Usage: python phase0_plots.py [label ...]  (reads/writes the scratchpad phase0 dir)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("/tmp/claude-0/-home-user-Samudra/09c6e2b1-2a64-5045-8ba8-dbdc1b668dd9/scratchpad/phase0")
labels = sys.argv[1:] or ["ks1d_base", "kolmo_base", "denoise_base"]
R = {l: json.loads((D / f"{l}.json").read_text()) for l in labels if (D / f"{l}.json").exists()}
BN = ["d0", "d1", "d2", "d3", "mid", "u5(deep)", "u6", "u7", "u8", "fin"]
for l, r in R.items():
    L = r["lesion"]; f = L["full"]; print(f"\n== {l} full metric={f['metric']:.4g} onestep={f['onestep']:.3g}")
    print("skip masks (rel metric / rel onestep):")
    print(" ".join(f"{m}:{L['skip'][str(m)]['metric']/f['metric']:.2f}/{L['skip'][str(m)]['onestep']/f['onestep']:.2f}" for m in range(16)))
    print("blocks removed (rel metric / rel onestep):")
    print(" ".join(f"{BN[b]}:{L['block'][str(b)]['metric']/f['metric']:.2f}/{L['block'][str(b)]['onestep']/f['onestep']:.2f}" for b in range(10)))
    print("rand-k blocks (median [min,max] rel metric; rel onestep median):")
    for k, v in L["randk"].items():
        m = np.array(v["metric"]) / f["metric"]; o = np.array(v["onestep"]) / f["onestep"]
        print(f" k={k}: {np.median(m):.2f} [{m.min():.2f},{m.max():.2f}]  onestep {np.median(o):.2f}")
    g = r["grad"]["phases"]
    for ph in ("early", "late"):
        s = g[ph]; print(f"grad {ph}: block_share={np.round(s['block_share'],3).tolist()}\n   perparam_share={np.round(s['block_perparam_share'],3).tolist()}"
                         f"\n   skip_share={np.round(s['skip_share'],3).tolist()} gini_blk={s['gini_block']:.2f} gini_blk_pp={s['gini_block_perparam']:.2f} top2_blk={s['top2_block']:.2f} gini_skip={s['gini_skip']:.2f} deepskip/max={s['deepest_skip_over_max']:.2f} deepblk/max={s['deepest_blocks_over_max']:.2f}")
    if "path" in r:
        print("path-length (k kept of 14): median loss / ||dL/dtheta|| / ||(J-I)^T v||")
        for k, v in r["path"].items(): print(f" k={k}: {np.median(v['loss']):.3g} / {np.median(v['gparam']):.3g} / {np.median(v['gx']):.3g}")

# ---- fig 1: lesion
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for l, r in R.items():
    L = r["lesion"]; f = L["full"]["metric"]
    ax[0].plot(range(16), [L["skip"][str(m)]["metric"] / f for m in range(16)], "o-", label=l)
    ax[1].plot(range(10), [L["block"][str(b)]["metric"] / f for b in range(10)], "o-", label=l)
    ks = list(range(1, 6)); med = [np.median(L["randk"][str(k)]["metric"]) / f for k in ks]
    lo = [np.min(L["randk"][str(k)]["metric"]) / f for k in ks]; hi = [np.max(L["randk"][str(k)]["metric"]) / f for k in ks]
    ax[2].plot(ks, med, "o-", label=l); ax[2].fill_between(ks, lo, hi, alpha=0.15)
ax[0].set_title("keep-mask id (bit i = skip i kept; 15=all)"); ax[1].set_title("remove one block (0-3 down, 4 mid, 5-8 up, 9 final)")
ax[2].set_title("remove k random blocks (median, min-max)")
for a in ax: a.set_yscale("log"); a.axhline(1, c="k", lw=.5); a.set_ylabel("val rollout metric / unmasked"); a.legend()
plt.tight_layout(); plt.savefig(D / "fig_lesion.png", dpi=110); plt.close()
# ---- fig 2: gradient shares over training
fig, ax = plt.subplots(2, len(R), figsize=(5 * len(R), 7), squeeze=False)
for j, (l, r) in enumerate(R.items()):
    g = r["grad"]; B = np.array(g["block"]); S = np.array(g["skip"]); st = g["steps"]; npb = np.array(g["block_params"])
    pp = B / np.sqrt(npb); ax[0, j].stackplot(st, (pp ** 2 / (pp ** 2).sum(1, keepdims=True)).T, labels=BN)
    ax[0, j].set_title(f"{l}: per-param grad energy share by block"); ax[0, j].set_xlabel("step")
    for i in range(4): ax[1, j].plot(st, S[:, i], label=f"skip{i}")
    ax[1, j].set_yscale("log"); ax[1, j].set_title("grad norm into skip tensor"); ax[1, j].legend(); ax[1, j].set_xlabel("step")
ax[0, 0].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(D / "fig_grad.png", dpi=110); plt.close()
# ---- fig 3: path length
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for l, r in R.items():
    if "path" not in r: continue
    ks = sorted(int(k) for k in r["path"])
    for a, key in zip(ax, ("loss", "gparam", "gx")):
        a.plot(ks, [np.median(r["path"][str(k)][key]) for k in ks], "o-", label=l); a.set_yscale("log"); a.set_xlabel("# units kept (of 10 blocks+4 skips)"); a.set_title(key)
ax[0].legend(); plt.tight_layout(); plt.savefig(D / "fig_path.png", dpi=110); plt.close()
