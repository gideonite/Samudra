"""Tables + figures for the ResNet study (reads resnet/results.tsv, resnet/runs/_final_test.json, resnet/phase0/*.json).
Usage: python resnet_report.py {noise|phase1|gb|screen|ab|figs} ; prints markdown, figures go to resnet/figs/."""
import csv, json, sys, statistics as st, collections
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; R = HERE / "resnet"; FIG = R / "figs"; FIG.mkdir(parents=True, exist_ok=True)
SKIP = ("arch", "rounds", "steps", "steps_per_s", "params", "minH", "test_metric", "test_onestep", "ens_metric", "ens_masks", "stage_loss",
        "stage_tvar", "stage_val", "stages", "stage_width", "D_ratio_max", "D_ratio_final")
COLS = [("ks1d", 128), ("ks1d", 512), ("ks1d", 2048), ("kolmo2d", 128), ("kolmo2d", 512), ("kolmo2d", 2048), ("denoise", 512), ("grayscott2d", 512)]
PERSIST = dict(ks1d=1.14, kolmo2d=1.03, denoise=0.65, grayscott2d=0.79)


def load(path):
    out = []
    for r in csv.DictReader(open(path), delimiter="\t"):
        if float(r["wallclock_s"]) > 3600: continue  # 10x-budget U-Net runs
        toks = r["notes"].split()
        flags = " ".join(sorted(t for t in toks if "=" in t and t.split("=")[0] not in SKIP))
        d = {t.split("=")[0]: t.split("=")[1] for t in toks if "=" in t}
        out.append(dict(ds=r["dataset"], arm=r["arm"], n=int(r["n_train"]), seed=int(r["seed"]), m=float(r["metric"]), flags=flags,
                        sps=float(d["steps_per_s"]) if "steps_per_s" in d else None, notes=r["notes"]))
    return out


def get(rows, ds, n, arm, flags=""):
    v = {r["seed"]: r["m"] for r in rows if (r["ds"], r["n"], r["arm"], r["flags"]) == (ds, n, arm, flags)}
    return [v[s] for s in sorted(v)]


def ms(v): return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), float("nan"))
def fmt(v): m, s = ms(v); return "–" if not v else f"{m:.3f} ± {s:.3f}" + (f" [k={len(v)}]" if len(v) != 3 else "")
def win(ref, new):  # keep rule: delta > 2x the larger std of the two arms
    (a, sa), (b, sb) = ms(ref), ms(new); return (a - b) > 2 * max(sa, sb)


def noise(rows, urows):
    print("| dataset | n=512 baseline val | steps/s ResNet | steps/s U-Net | ratio | U-Net val | persistence |\n|---|---|---|---|---|---|---|")
    for ds in ("ks1d", "kolmo2d", "denoise", "grayscott2d"):
        sp = [r["sps"] for r in rows if (r["ds"], r["n"], r["arm"], r["flags"]) == (ds, 512, "baseline", "") and r["sps"]]
        su = [r["sps"] for r in urows if (r["ds"], r["n"], r["arm"], r["flags"]) == (ds, 512, "baseline", "") and r["sps"] and r["notes"].count("steps=") and "BUDGET" not in r["flags"]]
        su = [x for x in su if x]
        print(f"| {ds} | {fmt(get(rows, ds, 512, 'baseline'))} | {st.mean(sp):.2f} | {st.mean(su):.1f} | {st.mean(sp)/st.mean(su):.2f} | {fmt(get(urows, ds, 512, 'baseline'))} | {PERSIST[ds]} |")


def phase1(rows):
    arms = [("baseline", ""), ("stochdepth", ""), ("mask_only", ""), ("reweight_only", ""), ("pcgb", ""), ("pcgb+warm0.5", "warm_frac=0.5")]
    print("| arm | " + " | ".join(f"{d} {n}" for d, n in COLS) + " |\n|---|" + "---|" * len(COLS))
    for nm, fl in arms:
        arm = nm.split("+")[0]
        print(f"| {nm} | " + " | ".join(fmt(get(rows, d, n, arm, fl)) for d, n in COLS) + " |")
    print("\nKeep rule (Δ mean > 2x larger seed std) vs baseline, per column (+ = arm beats baseline, - = significantly worse, 0 = tie):")
    for nm, fl in arms[1:]:
        arm = nm.split("+")[0]; o = []
        for d, n in COLS:
            b, a = get(rows, d, n, "baseline"), get(rows, d, n, arm, fl)
            o.append("." if not a else "+" if win(b, a) else "-" if win(a, b) else "0")
        print(f"- {nm}: " + " ".join(f"{d}{n}:{x}" for (d, n), x in zip(COLS, o)))


def gb(rows):
    ds3 = ("ks1d", "kolmo2d", "grayscott2d")
    cfg = [("baseline", "baseline", ""), ("stochdepth", "stochdepth", ""), ("reweight_only (adaboost)", "reweight_only", ""), ("reweight_only (gb)", "reweight_only", "rw_kind=gb"),
           ("pcgb (adaboost)", "pcgb", ""), ("pcgb (gb)", "pcgb", "rw_kind=gb"), ("gboost K=2", "gboost", "gb_stages=2"), ("gboost K=4", "gboost", "")]
    print("| arm | " + " | ".join(ds3) + " |\n|---|---|---|---|")
    for nm, arm, fl in cfg: print(f"| {nm} | " + " | ".join(fmt(get(rows, d, 512, arm, fl)) for d in ds3) + " |")
    print("\nWins (Δ>2σ) vs baseline / stochdepth, new configs:")
    for nm, arm, fl in cfg[3:4] + cfg[5:]:
        for d in ds3:
            a = get(rows, d, 512, arm, fl); b = get(rows, d, 512, "baseline"); s = get(rows, d, 512, "stochdepth")
            print(f"- {nm} {d}: vs baseline {'WIN' if win(b, a) else 'worse' if win(a, b) else 'tie'}; vs stochdepth {'WIN' if win(s, a) else 'worse' if win(a, s) else 'tie'}")


def screen(rows):
    vs = [("pcgb v1 (incumbent)", "pcgb", ""), ("mask_rule=random", "pcgb", "mask_rule=random"), ("norepeat=2", "pcgb", "norepeat=2"),
          ("mask_target=blocks", "pcgb", "mask_target=blocks"), ("mask_exclude_all_drop", "pcgb", "mask_exclude_all_drop=True"),
          ("pcgb + warm_frac=0.5 (stage b)", "pcgb", "warm_frac=0.5"), ("stochdepth p_L=0.2 (default)", "stochdepth", ""), ("stochdepth p_L=0.5", "stochdepth", "sd_block_p_res=0.5"),
          ("baseline", "baseline", "")]
    print("| config | ks1d 512 | kolmo2d 512 | verdict vs pcgb v1 (ks1d / kolmo2d) | verdict vs baseline |\n|---|---|---|---|---|")
    for nm, arm, fl in vs:
        row = [get(rows, d, 512, arm, fl) for d in ("ks1d", "kolmo2d")]
        inc = [get(rows, d, 512, "pcgb") for d in ("ks1d", "kolmo2d")]; bl = [get(rows, d, 512, "baseline") for d in ("ks1d", "kolmo2d")]
        v = lambda ref, a: "." if not a or not ref else "keep" if win(ref, a) else "worse" if win(a, ref) else "tie"
        print(f"| {nm} | {fmt(row[0])} | {fmt(row[1])} | {v(inc[0], row[0])} / {v(inc[1], row[1])} | {v(bl[0], row[0])} / {v(bl[1], row[1])} |")


def test_cells(path):
    runs = json.load(open(path))["runs"]; c = collections.defaultdict(list)
    for r in runs: c[(r["dataset"], r["n_train"], r["arm"])].append(r["test_metric"])
    return c


def ab():
    ur, rr = test_cells(HERE / "runs" / "_final_test.json"), test_cells(R / "runs" / "_final_test.json")
    print("| arm | " + " | ".join(f"{d} {n}" for d, n in COLS) + " |\n|---|" + "---|" * len(COLS))
    for arm in ("baseline", "stochdepth", "mask_only", "reweight_only", "pcgb", "pcgb_warm"):
        for nm, c in (("U-Net", ur), ("ResNet", rr)):
            cells = []
            for d, n in COLS:
                v = c.get((d, n, arm), [])
                cells.append("–" if not v else f"{ms(v)[0]:.3f}±{ms(v)[1]:.3f}")
            print(f"| {arm} ({nm}) | " + " | ".join(cells) + " |")
    print("\nResNet/U-Net test ratio, seed means:")
    for arm in ("baseline", "stochdepth", "pcgb"):
        print(f"- {arm}: " + " ".join(f"{d}{n}:{ms(rr[(d,n,arm)])[0]/ms(ur[(d,n,arm)])[0]:.2f}" for d, n in COLS if (d, n, arm) in rr and (d, n, arm) in ur))
    print("\nBest arm per column (test mean): " + "; ".join(f"{d}{n}: U={min((ms(v)[0], a) for (dd, nn, a), v in ur.items() if (dd, nn) == (d, n))[1]} R={min((ms(v)[0], a) for (dd, nn, a), v in rr.items() if (dd, nn) == (d, n))[1]}" for d, n in COLS))


def figs(rows):
    P0 = R / "phase0"; lab = dict(ks1d="ks1d", kolmo2d="kolmo2d", denoise="denoise", grayscott2d="grayscott2d")
    data = {d: json.load(open(P0 / f"{d}.json")) for d in lab if (P0 / f"{d}.json").exists()}
    if data:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        w = 0.8 / max(len(data), 1)
        for j, (d, r) in enumerate(data.items()):
            ph = r["grad"]["phases"]["late"]; ax[0].bar(np.arange(12) + j * w, ph["block_perparam_share"], w, label=d)
            ax[1].bar(np.arange(4) + j * w, ph["group_share"], w, label=d)
            g = r["grad"]; B = np.array(g["block"]); npb = np.array(g["block_params"]); pp = (B / np.sqrt(npb)) ** 2
            ax[2].plot(g["steps"], pp[:, :3].sum(1) / pp.sum(1), label=f"{d} grp0 (input side)"); ax[2].plot(g["steps"], pp[:, 9:].sum(1) / pp.sum(1), "--", label=f"{d} grp3 (head side)")
        ax[0].set_title("per-param grad-energy share per block (late phase)"); ax[0].set_xlabel("block 0..11")
        ax[1].set_title("grad-energy share per group (late phase)"); ax[1].set_xlabel("group 0..3 (0 nearest the stem)")
        ax[2].set_title("group 0 vs group 3 share over training"); ax[2].set_xlabel("step")
        for a in ax: a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "phase0_grad.png", dpi=110); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for d, r in data.items():
            L = r["lesion"]; f = L["full"]["metric"]
            ax[0].plot(range(16), [L["skip"][str(m)]["metric"] / f for m in range(16)], "o-", label=d)
            ax[1].plot(range(12), [L["block"][str(b)]["metric"] / f for b in range(12)], "o-", label=d)
            ks = sorted(int(k) for k in L["randk"]); ax[2].plot(ks, [np.median(L["randk"][str(k)]["metric"]) / f for k in ks], "o-", label=d)
        ax[0].set_title("group-mask id (bit g = group g kept; 15 = full)"); ax[1].set_title("remove one block"); ax[2].set_title("remove k random blocks (median)")
        for a in ax: a.set_yscale("log"); a.axhline(1, c="k", lw=.5); a.set_ylabel("val metric / unmasked"); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "phase0_lesion.png", dpi=110); plt.close(fig)
    ftj = R / "runs" / "_final_test.json"
    if ftj.exists():
        c = test_cells(ftj); arms = ["baseline", "stochdepth", "mask_only", "reweight_only", "pcgb", "pcgb_warm"]
        fig, ax = plt.subplots(2, 4, figsize=(17, 7))
        for a, (d, n) in zip(ax.flat, COLS):
            for i, arm in enumerate(arms):
                v = c.get((d, n, arm), [])
                if v: a.bar(i, ms(v)[0], yerr=ms(v)[1], color=f"C{i}"); a.scatter([i] * len(v), v, c="k", s=8, zorder=3)
            a.axhline(PERSIST[d], c="gray", ls=":", lw=1); a.set_title(f"{d} n={n} (test)"); a.set_xticks(range(len(arms))); a.set_xticklabels(arms, rotation=60, fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "main_table_bars.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cfg = [("baseline", "baseline", ""), ("stochdepth", "stochdepth", ""), ("rw adaboost", "reweight_only", ""), ("rw gb", "reweight_only", "rw_kind=gb"),
           ("pcgb adaboost", "pcgb", ""), ("pcgb gb", "pcgb", "rw_kind=gb"), ("gboost K=2", "gboost", "gb_stages=2"), ("gboost K=4", "gboost", "")]
    for a, d in zip(ax, ("ks1d", "kolmo2d", "grayscott2d")):
        for i, (nm, arm, fl) in enumerate(cfg):
            v = get(rows, d, 512, arm, fl)
            if v: a.bar(i, ms(v)[0], yerr=ms(v)[1], color=f"C{i}"); a.scatter([i] * len(v), v, c="k", s=8, zorder=3)
        a.set_xticks(range(len(cfg))); a.set_xticklabels([c[0] for c in cfg], rotation=60, fontsize=7); a.set_title(f"{d} n=512 (val)"); a.axhline(PERSIST[d], c="gray", ls=":")
    fig.tight_layout(); fig.savefig(FIG / "gb_variants.png", dpi=110); plt.close(fig)
    print("figs written to", FIG)


if __name__ == "__main__":
    rows, urows = load(R / "results.tsv"), load(HERE / "results.tsv")
    {"noise": lambda: noise(rows, urows), "phase1": lambda: phase1(rows), "gb": lambda: gb(rows), "screen": lambda: screen(rows), "ab": ab, "figs": lambda: figs(rows)}[sys.argv[1]]()
